"""
Evidence-based ranking (pure: no I/O, no LLM, no clock).

For each aspect the user asked for, three kinds of evidence are kept apart:
- claim: what the host lists as an amenity (Listing.claims)
- review: guest comments already judged as supports / contradicts
- map:   objective data, e.g. distance to the nearest mapped gym (OSM)

Reviews are weighted by recency relative to the snapshot date, so the
same data always gives the same ranking.
"""

import datetime as dt
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

from domain.aspects import ASPECTS
from domain.schemas import Listing

Source = Literal["review", "map"]
Stance = Literal["supports", "contradicts", "irrelevant"]
Verdict = Literal["verified", "claim_only", "evidence_only", "mixed", "contradicted", "unknown"]


@dataclass(frozen=True)
class Evidence:
    listing_id: str
    aspect: str
    source: Source
    stance: Stance
    ref: str  # review_id, or osm id for map evidence
    text: str  # verbatim quote (review) or description (map)
    date: dt.date | None = None
    distance_m: float | None = None


@dataclass(frozen=True)
class RankingConfig:
    # Calibrated in T8 (eval); defaults are a starting point
    half_life_days: int = 540  # a review loses half its weight every ~18 months
    stale_after_days: int = 1095  # all evidence older than 3 years → flag
    min_weight: float = 0.15  # below this, decayed evidence no longer counts
    contradiction_penalty: float = 1.5  # complaints weigh more than praise
    dominance: float = 2.0  # s+ >= 2·s- (or the reverse) decides; else mixed
    claim_weight: float = 0.2  # amenity lists are near-universal (93% list Wifi)
    fast_wifi_mbps: int = 100  # listed speed test counts as a full claim from here
    plain_claim: float = 0.6  # strength of a claim without a speed test
    gym_radius_m: float = 500
    map_weight: float = 1.0
    rating_weight: float = 0.1


DEFAULT_CONFIG = RankingConfig()


@dataclass
class AspectVerdict:
    aspect: str
    verdict: Verdict
    score: float  # 0..1
    claim: str | None
    support: float  # recency-weighted
    contradiction: float
    evidence: list[Evidence] = field(default_factory=list)  # strongest first
    flags: list[str] = field(default_factory=list)


@dataclass
class RankedListing:
    listing_id: str
    score: float
    aspects: dict[str, AspectVerdict]
    flags: list[str]


def recencyWeight(day: dt.date | None, today: dt.date, config: RankingConfig) -> float:
    if day is None:  # map data has no review date: full weight
        return 1.0
    age_days = max((today - day).days, 0)
    return 0.5 ** (age_days / config.half_life_days)


def mapEvidence(listing: Listing, aspect: str, config: RankingConfig) -> list[Evidence]:
    """
    Objective evidence from the map. A missing or far gym is not a
    contradiction: OpenStreetMap may simply not have it.
    """
    if aspect != "gym_nearby" or listing.nearest_gym_m is None:
        return []
    if listing.nearest_gym_m > config.gym_radius_m:
        return []
    name = listing.nearest_gym_name or "a gym"
    return [
        Evidence(
            listing_id=listing.listing_id, aspect=aspect, source="map", stance="supports", ref="osm",
            text=f"{name}, ~{round(listing.nearest_gym_m)} m in a straight line (OpenStreetMap)",
            distance_m=listing.nearest_gym_m,
        )
    ]


def claimStrength(listing: Listing, aspect: str, config: RankingConfig) -> float:
    if aspect not in listing.claims:
        return 0.0
    if aspect == "wifi" and (listing.wifi_mbps or 0) < config.fast_wifi_mbps:
        return config.plain_claim
    return 1.0


def judgeAspect(
    listing: Listing,
    aspect: str,
    evidence: list[Evidence],
    today: dt.date,
    config: RankingConfig = DEFAULT_CONFIG,
) -> AspectVerdict:
    claim = listing.claims.get(aspect)
    items = [
        e for e in evidence
        if e.listing_id == listing.listing_id and e.aspect == aspect and e.stance != "irrelevant"
    ]
    items += mapEvidence(listing, aspect, config)

    weighted = []
    for item in items:
        weight = recencyWeight(item.date, today, config)
        if item.source == "map":
            weight *= config.map_weight
        weighted.append((weight, item))
    weighted.sort(key=lambda pair: pair[0], reverse=True)

    support = sum(w for w, e in weighted if e.stance == "supports")
    contradiction = sum(w for w, e in weighted if e.stance == "contradicts")

    flags = []
    if weighted and all(
        e.date is not None and (today - e.date).days > config.stale_after_days
        for _, e in weighted
    ):
        flags.append("stale_evidence")

    # Decayed evidence below min_weight no longer decides the verdict
    has_support = support >= config.min_weight
    has_contradiction = contradiction >= config.min_weight

    if has_contradiction and (not has_support or contradiction >= config.dominance * support):
        verdict = "contradicted"
    elif has_support and (not has_contradiction or support >= config.dominance * contradiction):
        verdict = "verified" if claim else "evidence_only"
    elif has_support and has_contradiction:
        verdict = "mixed"
    else:
        verdict = "claim_only" if claim else "unknown"

    if claim and verdict in ("contradicted", "mixed"):
        flags.append("claim_contradicted")

    # Evidence balance in [-1, 1] → [0, 1]; no evidence = 0.5 (neutral)
    balance = (support - config.contradiction_penalty * contradiction) / (
        support + contradiction + 1
    )
    evidence_score = (max(-1.0, min(1.0, balance)) + 1) / 2

    if ASPECTS[aspect].amenity is None:  # quiet, transport: nothing to claim
        score = evidence_score
    else:
        strength = claimStrength(listing, aspect, config)
        score = config.claim_weight * strength + (1 - config.claim_weight) * evidence_score

    return AspectVerdict(
        aspect=aspect,
        verdict=verdict,
        score=round(score, 4),
        claim=claim,
        support=round(support, 4),
        contradiction=round(contradiction, 4),
        evidence=[e for _, e in weighted],
        flags=flags,
    )


def scoreListing(
    listing: Listing,
    aspects: list[str],
    evidence: list[Evidence],
    today: dt.date,
    config: RankingConfig = DEFAULT_CONFIG,
) -> RankedListing:
    verdicts = {a: judgeAspect(listing, a, evidence, today, config) for a in aspects}

    aspect_score = sum(v.score for v in verdicts.values()) / len(verdicts) if verdicts else 0.5
    # Missing rating is neutral, not bad (new listings)
    rating = listing.rating / 5 if listing.rating is not None else 0.5
    score = (1 - config.rating_weight) * aspect_score + config.rating_weight * rating

    flags = sorted({f"{a}:{flag}" for a, v in verdicts.items() for flag in v.flags})
    return RankedListing(listing.listing_id, round(score, 4), verdicts, flags)


def rankingFrame(listings) -> pd.DataFrame:
    """The listing fields scoreListing reads, as columns (one row per listing, index listing_id)."""
    rows = [{"listing_id": l.listing_id, "price": l.price, "rating": l.rating,
             "wifi_mbps": l.wifi_mbps, "nearest_gym_m": l.nearest_gym_m,
             **{f"claims_{a}": a in l.claims for a, spec in ASPECTS.items() if spec.amenity is not None}}
            for l in listings]
    return pd.DataFrame(rows).set_index("listing_id").astype({"price": float, "rating": float,
                                                               "wifi_mbps": float, "nearest_gym_m": float})


def prescore(
    listings: pd.DataFrame,
    aspects: list[str],
    reviews: pd.DataFrame,
    today: dt.date,
    config: RankingConfig = DEFAULT_CONFIG,
) -> pd.DataFrame:
    """
    scoreListing for many listings at once: the same formula on columns
    (numpy) instead of one listing at a time in Python, without the details
    (verdicts, quotes). It picks the pool that rankListings then ranks and
    explains; scores differ from scoreListing only by its 4-decimal rounding.

    listings: rankingFrame rows. reviews: listing_id, aspect, stance, date
    (datetime64) of judged reviews with a verified quote.
    Returns `score` and `evidence` per listing, best first (rankListings' order).
    """
    index = listings.index
    total = pd.Series(0.0, index=index)
    evidence = pd.Series(0.0, index=index)
    for aspect in aspects:
        rows = reviews[reviews["aspect"] == aspect]
        age_days = ((pd.Timestamp(today) - rows["date"]).dt.days).clip(lower=0)
        weight = (0.5 ** (age_days / config.half_life_days)).fillna(1.0)  # no date: full weight

        def totalFor(stance):
            picked = rows["stance"] == stance
            return weight[picked].groupby(rows.loc[picked, "listing_id"]).sum().reindex(index, fill_value=0.0)

        support, contradiction = totalFor("supports"), totalFor("contradicts")
        if aspect == "gym_nearby":  # a mapped gym within the radius counts as support
            gym = listings["nearest_gym_m"]
            support = support + (gym.notna() & (gym <= config.gym_radius_m)) * config.map_weight

        balance = (support - config.contradiction_penalty * contradiction) / (support + contradiction + 1)
        score = (balance.clip(-1.0, 1.0) + 1) / 2
        if ASPECTS[aspect].amenity is not None:
            slow_wifi = (aspect == "wifi") & (listings["wifi_mbps"].fillna(0) < config.fast_wifi_mbps)
            strength = listings[f"claims_{aspect}"] * np.where(slow_wifi, config.plain_claim, 1.0)
            score = config.claim_weight * strength + (1 - config.claim_weight) * score
        total += score
        evidence += support + contradiction

    aspect_score = total / len(aspects) if aspects else pd.Series(0.5, index=index)
    rating = (listings["rating"] / 5).fillna(0.5)  # missing rating is neutral
    out = pd.DataFrame({"score": (1 - config.rating_weight) * aspect_score + config.rating_weight * rating,
                        "evidence": evidence, "price": listings["price"].fillna(np.inf)})
    # stable sort, like rankListings: ties keep the candidates' order
    return out.sort_values(["score", "evidence", "price"], ascending=[False, False, True], kind="stable")


def rankListings(
    listings: list[Listing],
    aspects: list[str],
    evidence: list[Evidence],
    today: dt.date,
    config: RankingConfig = DEFAULT_CONFIG,
) -> list[RankedListing]:
    """
    Ranks listings that already passed the deterministic filters.
    `evidence` may mix listings; each item is matched by its listing_id.
    Ties: more weighted evidence first, then cheaper.
    """
    by_listing: dict[str, list[Evidence]] = {}
    for item in evidence:
        by_listing.setdefault(item.listing_id, []).append(item)

    ranked = [
        (listing, scoreListing(listing, aspects, by_listing.get(listing.listing_id, []), today, config))
        for listing in listings
    ]
    ranked.sort(
        key=lambda pair: (
            -pair[1].score,
            -sum(v.support + v.contradiction for v in pair[1].aspects.values()),
            pair[0].price if pair[0].price is not None else float("inf"),
        )
    )
    return [result for _, result in ranked]
