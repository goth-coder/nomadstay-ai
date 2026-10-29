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
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from openai import OpenAI

from agent.judge import judgeQuestion
from constants import ARTIFACT_VERSION, CITY, EMBEDDING_MODEL, refinedDir
from domain.aspects import ASPECTS
from domain.ranking import RankedListing, RankingConfig, prescore, rankingFrame, rankListings
from domain.schemas import Listing
from ingest.insideairbnb import readListings
from retrieval.catalog import DEFAULT_GYM_RADIUS_M, Catalog, normalizeCity
from retrieval.chunk_index import LISTINGS_FILE, MANIFEST, ChunkIndex, loadChunkIndex
from retrieval.evidence import (
    EVIDENCE_FILE,
    QUESTION_CHUNKS_PER_LISTING,
    SERVING_FILES,
    EvidenceStore,
    judgeParaphrases,
    paraphraseCandidates,
    questionCandidates,
)

logger = logging.getLogger(__name__)

SHORTLIST = 15
RESULTS = 5
RANKING_POOL = 100  # vectorized score picks these; rankListings ranks them (margin over SHORTLIST)
QUOTES_PER_STANCE = 1
MAX_QUOTE_CHARS = 300
CLAIMABLE = [a for a, spec in ASPECTS.items() if spec.amenity is not None]
MAX_REVIEW_LISTINGS = 5  # search_reviews: bounds cost to <= 20 excerpts, <= 2 judge calls
QUOTES_PER_LISTING = 3
ATTRIBUTION = "Listings and reviews: Inside Airbnb (CC BY 4.0). Gyms: © OpenStreetMap contributors (ODbL)."


@dataclass
class NomadStore:
    catalog: Catalog
    evidence: EvidenceStore
    index: ChunkIndex | None = None  # needed for the paraphrase pass and quote context
    review_rows: dict | None = None  # review_id -> chunk rows, built with the index
    listings_by_id: dict[str, Listing] = field(init=False, repr=False)
    ranking_frame: pd.DataFrame = field(init=False, repr=False)

    def __post_init__(self):
        # Built once at load: turning ~11k rows into Listing objects on every
        # search was most of a search's CPU time (and Python runs one thread at a time)
        self.listings_by_id = {row["listing_id"]: toListing(row)
                               for row in self.catalog.listings.to_dict("records")}
        self.ranking_frame = rankingFrame(self.listings_by_id.values())

    def reviewText(self, review_id) -> str | None:
        """The full review, rebuilt from its chunks (split at sentence ends, no overlap)."""
        if self.index is None or self.review_rows is None:
            return None
        rows = self.review_rows.get(review_id)
        if rows is None:
            return None
        return "\n".join(self.index.chunks["text"].iloc[rows])


def loadNomadStore(directory, model_key: str, evidence_path=None, with_index: bool = True) -> NomadStore:
    directory = Path(directory)
    index = loadChunkIndex(directory, model_key) if with_index else None
    evidence = pd.read_parquet(Path(evidence_path) if evidence_path else directory / EVIDENCE_FILE)
    listings = readListings(directory / LISTINGS_FILE)
    # snapshot, city and currency come from the data, with or without the index
    first = listings.iloc[0]
    catalog = Catalog(listings, first["city"], first["currency"], str(first["snapshot_date"]))
    review_rows = index.chunks.groupby("review_id").indices if index is not None else None
    return NomadStore(catalog, EvidenceStore(evidence), index, review_rows)


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


# Ready-made wording, so the answer can't call a host's claim (or speed test) verified
VERDICT_MEANING = {
    "verified": "listed by the host and confirmed by guests",
    "claim_only": "listed by the host only; no guest confirmed it",
    "evidence_only": "reported by guests or map data; not listed by the host",
    "mixed": "guests disagree",
    "contradicted": "guests contradict it",
    "unknown": "no evidence",
}
SENTENCE_ENDS = ".!?\n"


def quoteInContext(text: str | None, quote: str, max_chars: int = MAX_QUOTE_CHARS) -> str:
    """
    The whole sentence around a verified quote, verbatim from the review, so
    a short span ("Wlan und Espressomaschine") keeps its meaning. The quote
    itself when the review is unknown, the quote can't be placed or the
    sentence is too long.
    """
    start = text.find(quote) if text else -1
    if start < 0:
        return quote[:max_chars]
    end = start + len(quote)
    left = max(text.rfind(c, 0, start) for c in SENTENCE_ENDS) + 1
    # from the quote's last character: a quote that ends a sentence stops there
    ends = [i for i in (text.find(c, max(end - 1, start)) for c in SENTENCE_ENDS) if i >= 0]
    right = min(ends) + 1 if ends else len(text)
    sentence = text[left:right].strip()
    return sentence if len(sentence) <= max_chars else quote[:max_chars]


def aspectSummary(verdict, listing: Listing, store: NomadStore | None = None) -> dict:
    guest = [e for e in verdict.evidence if e.source == "review"]
    quotes = []
    for stance in ("contradicts", "supports"):  # complaints first: they matter most
        picked = [e for e in guest if e.stance == stance and e.text][:QUOTES_PER_STANCE]
        quotes += [{"stance": e.stance,
                    "quote": quoteInContext(store.reviewText(e.ref) if store else None, e.text),
                    "review_id": e.ref,
                    "date": e.date.isoformat() if e.date else None} for e in picked]
    summary = {
        "verdict": verdict.verdict,
        "meaning": VERDICT_MEANING[verdict.verdict],
        "listing_claim": verdict.claim,
        "guest_reports": {"supports": sum(e.stance == "supports" for e in guest),
                          "contradicts": sum(e.stance == "contradicts" for e in guest)},
        "guest_quotes": quotes,
        "map": next((e.text for e in verdict.evidence if e.source == "map"), None),
        "flags": verdict.flags,
    }
    if verdict.aspect == "wifi" and listing.wifi_mbps:
        summary["listed_speed_test_mbps"] = listing.wifi_mbps
        summary["speed_note"] = (f"{listing.wifi_mbps:g} Mbps is the host's own speed test; "
                                 "guests did not measure it.")
    return summary


def priceLabel(price, currency: str, snapshot_date: str) -> str | None:
    """The price as the answer must state it: indicative, with the snapshot date."""
    if price is None or pd.isna(price):
        return None
    return f"{price:.2f} {currency}/night, indicative on {snapshot_date} (not a live quote)"


def formatResult(ranked: RankedListing, listing: Listing, store: NomadStore | None = None) -> dict:
    catalog = store.catalog if store else None
    return {
        "listing_id": listing.listing_id,
        "name": listing.name,
        "url": listing.url,
        "neighbourhood": listing.neighbourhood,
        "price_per_night": listing.price,
        "price": priceLabel(listing.price, catalog.currency, catalog.snapshot_date) if catalog else None,
        "price_source": listing.price_status,
        "rating": listing.rating,
        "number_of_reviews": listing.number_of_reviews,
        "score": ranked.score,
        "aspects": {a: aspectSummary(v, listing, store) for a, v in ranked.aspects.items()},
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


def rankCandidates(store: NomadStore, candidate_ids: list[str], aspects: list[str], today: dt.date,
                   config: RankingConfig, pool: int = RANKING_POOL):
    """
    Two steps, one result: a vectorized score of every candidate picks the
    best `pool`, then rankListings ranks and explains only those. Ranking
    all ~11k listings one at a time in Python was the API's CPU bottleneck.
    Returns (ranked, listings by id, precomputed evidence) of the pool.
    """
    scores = prescore(store.ranking_frame.loc[candidate_ids], aspects, store.evidence.scorable, today, config)
    ids = list(scores.index[:pool])
    listings = {i: store.listings_by_id[i] for i in ids}
    evidence = store.evidence.precomputed(ids, aspects)
    return rankListings(list(listings.values()), aspects, evidence, today, config), listings, evidence


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

    today = dt.date.fromisoformat(catalog.snapshot_date) if catalog.snapshot_date else dt.date.today()
    ranked, listings, evidence = rankCandidates(store, list(candidates["listing_id"]), aspects, today, config)
    shortlist = ranked[:max(SHORTLIST, limit)]

    if paraphrases and aspects and client is not None and store.index is not None:
        shortlist = withParaphrases(store, client, shortlist, listings, aspects, evidence, today, config)

    return {**base, "status": "ok", "aspects": aspects,
            "results": [formatResult(r, listings[r.listing_id], store) for r in shortlist[:limit]]}


def get_stay_details(listing_id: str, aspects: list[str], store: NomadStore) -> dict:
    """
    One stay found by search_stays: host claims (amenities) and the evidence
    for the requested aspects. Precomputed evidence only: no LLM call.
    """
    listing = store.listings_by_id.get(listing_id)
    if listing is None:
        return {"status": "not_found", "listing_id": listing_id,
                "message": "Unknown listing_id: use one returned by search_stays."}
    aspects = [a for a in dict.fromkeys(aspects) if a in ASPECTS]
    today = dt.date.fromisoformat(store.catalog.snapshot_date)
    evidence = store.evidence.precomputed([listing_id], aspects)
    ranked = rankListings([listing], aspects, evidence, today)[0]
    return {
        "status": "ok",
        **formatResult(ranked, listing, store),
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


def search_reviews(question: str, listing_ids: list[str], store: NomadStore,
                   client: OpenAI | None = None) -> dict:
    """
    Open RAG over guest reviews: what guests say about any question (host,
    kids, kitchen, stairs...) at the given stays. Hybrid retrieval inside each
    listing, then an online judge with the question as the criterion; only
    quotes verified against the review text are returned.
    """
    if store.index is None or client is None:
        return {"status": "unavailable", "message": "Review search is not available right now."}
    known = set(store.catalog.listings["listing_id"])
    ids = [i for i in dict.fromkeys(listing_ids) if i in known][:MAX_REVIEW_LISTINGS]
    if not ids:
        return {"status": "not_found", "message": "Use listing_ids returned by search_stays."}

    excerpts = questionCandidates(store.index, ids, question)
    labels, usage = judgeQuestion(client, question,
                                  [{"id": n, "text": e["text"]} for n, e in enumerate(excerpts)])
    logger.info("search_reviews: %s listings, %s excerpts, usage %s", len(ids), len(excerpts), usage)

    names = store.catalog.listings.set_index("listing_id")["name"]
    listings = []
    for listing_id in ids:
        found = [(e, l) for e, l in zip(excerpts, labels) if e["listing_id"] == listing_id
                 and l["stance"] in ("supports", "contradicts") and l["quote_valid"] and l["quote"]]
        found.sort(key=lambda pair: pair[1]["stance"] != "contradicts")  # complaints first
        listings.append({
            "listing_id": listing_id,
            "name": names[listing_id],
            "guest_reports": {"supports": sum(l["stance"] == "supports" for _, l in found),
                              "contradicts": sum(l["stance"] == "contradicts" for _, l in found)},
            "guest_quotes": [
                {"stance": l["stance"],
                 "quote": quoteInContext(store.reviewText(e["review_id"]) or e["text"], l["quote"]),
                 "review_id": e["review_id"],
                 "date": e["date"].isoformat() if e["date"] else None}
                for e, l in found[:QUOTES_PER_LISTING]
            ],
        })
    return {
        "status": "ok",
        "question": question,
        "listings": listings,
        "notes": [(f"Searched the {QUESTION_CHUNKS_PER_LISTING} most relevant review excerpts per stay. "
                   "No quotes means no guest addressed it there, not that the answer is no.")],
        "attribution": ATTRIBUTION,
    }


# No line breaks inside, so a stray quote mark can't pair with one paragraphs away
QUOTED = re.compile(r"[\"“”«»„]([^\"“”«»„\n]{8,400})[\"“”«»„]")
TYPOGRAPHY = str.maketrans({"’": "'", "‘": "'", "`": "'", "–": "-", "—": "-", "…": "..."})


def normalizeQuote(text: str) -> str:
    return " ".join(text.translate(TYPOGRAPHY).lower().split()).strip(" .,;:!?")


def quotableTexts(value) -> list[str]:
    """Every text a tool output lets the model quote: guest quotes, listing
    claims, host amenities and listing names, wherever they sit in the output."""
    texts = []
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "guest_quotes":
                texts += [q["quote"] for q in item]
            elif key in ("listing_claim", "name") and item:
                texts.append(item)
            elif key == "host_amenities":
                texts += list(item)
            else:
                texts += quotableTexts(item)
    elif isinstance(value, list):
        for item in value:
            texts += quotableTexts(item)
    return texts


def quotedFrom(quote: str, source: str) -> bool:
    """
    The quote is verbatim in the source. An elision ("..." between parts) is
    allowed: every part verbatim, in order, from this one source, so words
    can be cut but never changed, translated or taken from another quote.
    """
    position = 0
    for part in normalizeQuote(quote).split("..."):
        part = part.strip(" .,;:!?")
        if not part:
            continue
        found = source.find(part, position)
        if found < 0:
            return False
        position = found + len(part)
    return True


def unverifiedQuotes(answer: str, tool_outputs: list[dict]) -> list[str]:
    """
    Quoted passages (8+ chars, straight or typographic double quotes) in the
    final answer that no tool output contains. Anything here is a citation
    the model made up or edited. Known gap: it does not check that a real
    quote is attributed to the right listing.
    """
    known = [normalizeQuote(t) for t in quotableTexts(tool_outputs)]
    return [m for m in QUOTED.findall(answer)
            if not any(quotedFrom(m, k) for k in known)]
