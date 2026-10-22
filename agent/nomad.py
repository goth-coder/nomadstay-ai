"""
search_stays: the NomadStay search behind the agent's tool call.

    filters (code) -> precomputed evidence -> ranking -> shortlist
    -> paraphrase evidence (embeddings + online judge) -> re-rank -> top results

The output keeps three kinds of evidence in separate fields: what the listing
claims, what guests report (verbatim quotes with review id and date) and what
the map shows. The LLM writes the answer only from this output.
"""

import datetime as dt
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from openai import OpenAI

from constants import ARTIFACT_VERSION, CITY, EMBEDDING_MODEL, refinedDir
from domain.aspects import ASPECTS
from domain.ranking import RankedListing, RankingConfig, rankListings
from domain.schemas import Listing
from ingest.insideairbnb import readListings
from retrieval.catalog import DEFAULT_GYM_RADIUS_M, Catalog, normalizeCity
from retrieval.chunk_index import LISTINGS_FILE, MANIFEST, ChunkIndex, loadChunkIndex
from retrieval.evidence import (
    EVIDENCE_FILE,
    SERVING_FILES,
    EvidenceStore,
    judgeParaphrases,
    paraphraseCandidates,
)

logger = logging.getLogger(__name__)

SHORTLIST = 15
RESULTS = 5
QUOTES_PER_STANCE = 1
MAX_QUOTE_CHARS = 300
CLAIMABLE = [a for a, spec in ASPECTS.items() if spec.amenity is not None]
ATTRIBUTION = "Listings and reviews: Inside Airbnb (CC BY 4.0). Gyms: © OpenStreetMap contributors (ODbL)."


@dataclass
class NomadStore:
    catalog: Catalog
    evidence: EvidenceStore
    index: ChunkIndex | None = None  # needed only for the paraphrase pass


def loadNomadStore(directory, model_key: str, evidence_path=None, with_index: bool = True) -> NomadStore:
    directory = Path(directory)
    index = loadChunkIndex(directory, model_key) if with_index else None
    evidence = pd.read_parquet(Path(evidence_path) if evidence_path else directory / EVIDENCE_FILE)
    listings = readListings(directory / LISTINGS_FILE)
    # snapshot, city and currency come from the data, with or without the index
    first = listings.iloc[0]
    catalog = Catalog(listings, first["city"], first["currency"], str(first["snapshot_date"]))
    return NomadStore(catalog, EvidenceStore(evidence), index)


def loadStore(city: str = CITY, version: int = ARTIFACT_VERSION,
              model_key: str = EMBEDDING_MODEL) -> NomadStore:
    """
    The serving set from data/refined/v{version}/{city}; downloaded from S3
    first when it is not there (the manifest is the last file written, so a
    partial download is fetched again). The loader checks every file hash.
    """
    directory = refinedDir(city, version)
    if not (directory / MANIFEST).exists():
        from storage.s3 import downloadArtifacts
        logger.info("serving set v%s/%s not found locally, downloading from S3", version, city)
        downloadArtifacts(version, city, directory, SERVING_FILES)
    return loadNomadStore(directory, model_key)


def toListing(row: dict) -> Listing:
    clean = {k: (None if not isinstance(v, (list, dict)) and pd.isna(v) else v)
             for k, v in row.items() if k in Listing.model_fields}
    return Listing(**clean)


def relaxSuggestions(counts: dict, max_price, must_have, max_gym_distance_m) -> list[str]:
    suggestions = []
    if max_price is not None and counts["without_budget_limit"]:
        suggestions.append(f"raise the budget ({counts['without_budget_limit']} stays fit the other filters)")
    if must_have and counts["without_must_have"]:
        suggestions.append(f"drop the must-haves {must_have} ({counts['without_must_have']} stays)")
    if max_gym_distance_m is not None and counts["without_gym_distance"]:
        suggestions.append(f"accept a gym farther away ({counts['without_gym_distance']} stays)")
    return suggestions


def aspectSummary(verdict, listing: Listing) -> dict:
    guest = [e for e in verdict.evidence if e.source == "review"]
    quotes = []
    for stance in ("contradicts", "supports"):  # complaints first: they matter most
        picked = [e for e in guest if e.stance == stance and e.text][:QUOTES_PER_STANCE]
        quotes += [{"stance": e.stance, "quote": e.text[:MAX_QUOTE_CHARS], "review_id": e.ref,
                    "date": e.date.isoformat() if e.date else None} for e in picked]
    summary = {
        "verdict": verdict.verdict,
        "listing_claim": verdict.claim,
        "guest_reports": {"supports": sum(e.stance == "supports" for e in guest),
                          "contradicts": sum(e.stance == "contradicts" for e in guest)},
        "guest_quotes": quotes,
        "map": next((e.text for e in verdict.evidence if e.source == "map"), None),
        "flags": verdict.flags,
    }
    if verdict.aspect == "wifi" and listing.wifi_mbps:
        summary["listed_speed_test_mbps"] = listing.wifi_mbps
    return summary


def formatResult(ranked: RankedListing, listing: Listing) -> dict:
    return {
        "listing_id": listing.listing_id,
        "name": listing.name,
        "url": listing.url,
        "neighbourhood": listing.neighbourhood,
        "price_per_night": listing.price,
        "price_source": listing.price_status,
        "rating": listing.rating,
        "number_of_reviews": listing.number_of_reviews,
        "score": ranked.score,
        "aspects": {a: aspectSummary(v, listing) for a, v in ranked.aspects.items()},
    }


def withParaphrases(store, client, shortlist, listings, aspects, evidence, today, config):
    """Adds paraphrase evidence and re-ranks the shortlist. On any failure the
    precomputed ranking stands: this pass may improve the answer, never break it."""
    ids = [r.listing_id for r in shortlist]  # list, not set: stable order on ties
    try:
        chunks = paraphraseCandidates(store.index, store.evidence, set(ids), aspects)
        extra, usage = judgeParaphrases(client, chunks) if chunks else ([], {})
    except Exception:
        logger.warning("paraphrase pass failed, using precomputed evidence only", exc_info=True)
        return shortlist
    logger.info("paraphrase pass: %s chunks judged, %s evidence, usage %s",
                len(chunks), len(extra), usage)
    own = [e for e in evidence if e.listing_id in set(ids)]
    return rankListings([listings[i] for i in ids], aspects, own + extra, today, config)


def search_stays(
    city: str,
    max_price: float | None,
    currency: str | None,
    must_have: list[str],
    preferences: list[str],
    max_gym_distance_m: float | None,
    store: NomadStore,
    client: OpenAI | None = None,
    limit: int = RESULTS,
    paraphrases: bool = True,
) -> dict:
    catalog = store.catalog
    if normalizeCity(city) != catalog.city:
        return {"status": "unsupported_city", "requested_city": city,
                "supported_cities": [catalog.city],
                "message": "No data for this city. Recommend nothing; say which cities are covered."}
    if max_price is not None and currency and currency.upper() != catalog.currency:
        return {"status": "unsupported_currency", "requested_currency": currency,
                "dataset_currency": catalog.currency,
                "message": f"Prices are in {catalog.currency}; ask for a budget in {catalog.currency}. "
                           "Do not convert currencies."}

    # quiet/transport can't be claimed by a listing, so they can't be hard filters
    must_have = [a for a in dict.fromkeys(must_have) if a in CLAIMABLE]
    aspects = list(dict.fromkeys([*preferences, *must_have]))
    if max_gym_distance_m is not None and "gym_nearby" not in aspects:
        aspects.append("gym_nearby")  # the filter must be explainable in the results
    radius = DEFAULT_GYM_RADIUS_M if max_gym_distance_m is None else max_gym_distance_m
    config = RankingConfig(gym_radius_m=radius)
    candidates, counts = catalog.filter(max_price, must_have, max_gym_distance_m)
    base = {
        "city": catalog.city, "snapshot_date": catalog.snapshot_date, "currency": catalog.currency,
        "filters": {"max_price": max_price, "must_have": must_have,
                    "max_gym_distance_m": max_gym_distance_m},
        "counts": counts,
        "notes": [
            (f"Prices are indicative nightly prices in {catalog.currency} from the "
             f"{catalog.snapshot_date} snapshot, not live quotes."),
            "Availability is unknown.",
            "Gym distances are straight-line and approximate (listing locations are shifted up to ~150 m).",
        ],
        "attribution": ATTRIBUTION,
    }
    if max_price is not None and not currency:
        base["notes"].append(f"The budget was read as {catalog.currency}.")
    if candidates.empty:
        return {**base, "status": "no_results",
                "relax": relaxSuggestions(counts, max_price, must_have, max_gym_distance_m)}

    listings = {row["listing_id"]: toListing(row) for row in candidates.to_dict("records")}
    today = dt.date.fromisoformat(catalog.snapshot_date) if catalog.snapshot_date else dt.date.today()

    evidence = store.evidence.precomputed(listings.keys(), aspects)
    ranked = rankListings(list(listings.values()), aspects, evidence, today, config)
    shortlist = ranked[:max(SHORTLIST, limit)]

    if paraphrases and aspects and client is not None and store.index is not None:
        shortlist = withParaphrases(store, client, shortlist, listings, aspects, evidence, today, config)

    return {**base, "status": "ok", "aspects": aspects,
            "results": [formatResult(r, listings[r.listing_id]) for r in shortlist[:limit]]}


def get_stay_details(listing_id: str, aspects: list[str], store: NomadStore) -> dict:
    """
    One stay found by search_stays: host claims (amenities) and the evidence
    for the requested aspects. Precomputed evidence only: no LLM call.
    """
    rows = store.catalog.listings[store.catalog.listings["listing_id"] == listing_id]
    if rows.empty:
        return {"status": "not_found", "listing_id": listing_id,
                "message": "Unknown listing_id: use one returned by search_stays."}
    listing = toListing(rows.iloc[0].to_dict())
    aspects = [a for a in dict.fromkeys(aspects) if a in ASPECTS]
    today = dt.date.fromisoformat(store.catalog.snapshot_date)
    evidence = store.evidence.precomputed([listing_id], aspects)
    ranked = rankListings([listing], aspects, evidence, today)[0]
    return {
        "status": "ok",
        **formatResult(ranked, listing),
        "property_type": listing.property_type,
        "room_type": listing.room_type,
        "accommodates": listing.accommodates,
        "host_amenities": listing.amenities,  # host claims, not verified
        "snapshot_date": store.catalog.snapshot_date,
        "notes": [(f"Prices are indicative nightly prices in {store.catalog.currency} from the "
                   f"{store.catalog.snapshot_date} snapshot."),
                  "host_amenities are what the host lists, not verified by guests."],
        "attribution": ATTRIBUTION,
    }


QUOTED = re.compile(r"[\"“”«»„]([^\"“”«»„]{8,})[\"“”«»„]")
TYPOGRAPHY = str.maketrans({"’": "'", "‘": "'", "`": "'", "–": "-", "—": "-", "…": "..."})


def normalizeQuote(text: str) -> str:
    return " ".join(text.translate(TYPOGRAPHY).lower().split()).strip(" .")


def unverifiedQuotes(answer: str, tool_outputs: list[dict]) -> list[str]:
    """
    Quoted passages (8+ chars, straight or typographic double quotes) in the
    final answer that are neither a returned guest quote nor a listing claim.
    Anything here is a citation the model made up or edited. Known gap: it does
    not check that a real quote is attributed to the right listing.
    """
    known = []
    for out in tool_outputs:
        for r in out.get("results", []):
            for a in r["aspects"].values():
                known += [normalizeQuote(q["quote"]) for q in a["guest_quotes"]]
                if a.get("listing_claim"):
                    known.append(normalizeQuote(a["listing_claim"]))
    return [m for m in QUOTED.findall(answer)
            if not any(normalizeQuote(m) in k for k in known)]
