"""
T4: verdicts and scores of domain/ranking.py (pure, no I/O).
Scenarios mirror the mini_porto fixture.
"""

from datetime import date, timedelta

import pytest

from domain.aspects import ASPECTS
from domain.ranking import (
    Evidence,
    RankingConfig,
    judgeAspect,
    rankListings,
    scoreListing,
)
from domain.schemas import Listing

TODAY = date(2026, 9, 19)  # snapshot date, never the real clock


def listing(listing_id="1", claims=None, **overrides) -> Listing:
    base = {
        "listing_id": listing_id, "name": f"Listing {listing_id}",
        "url": f"https://www.airbnb.com/rooms/{listing_id}", "city": "porto",
        "latitude": 41.14, "longitude": -8.62, "price": 80.0, "currency": "EUR",
        "price_status": "listed", "snapshot_date": TODAY, "rating": 4.5,
        "claims": claims or {},
    }
    return Listing(**(base | overrides))


def review(stance, aspect="wifi", days_ago=30, listing_id="1", ref="r1") -> Evidence:
    return Evidence(
        listing_id=listing_id, aspect=aspect, source="review", stance=stance,
        ref=ref, text=f"{stance} {aspect}", date=TODAY - timedelta(days=days_ago),
    )


WIFI = {"wifi": "Wifi"}


# verdicts


def test_claim_confirmed_by_recent_reviews_is_verified():
    result = judgeAspect(listing(claims=WIFI), "wifi", [review("supports"), review("supports", ref="r2")], TODAY)

    assert result.verdict == "verified"
    assert result.flags == []
    assert result.claim == "Wifi"


def test_claim_without_evidence_is_claim_only():
    assert judgeAspect(listing(claims=WIFI), "wifi", [], TODAY).verdict == "claim_only"


def test_evidence_without_claim_is_evidence_only():
    # "quiet" can never be claimed: only guests can tell
    result = judgeAspect(listing(), "quiet", [review("supports", aspect="quiet")], TODAY)

    assert result.verdict == "evidence_only"


def test_nothing_known_is_unknown():
    assert judgeAspect(listing(), "quiet", [], TODAY).verdict == "unknown"


def test_claim_contradicted_by_reviews_is_flagged():
    evidence = [review("contradicts"), review("contradicts", ref="r2")]

    result = judgeAspect(listing(claims=WIFI), "wifi", evidence, TODAY)

    assert result.verdict == "contradicted"
    assert "claim_contradicted" in result.flags


def test_balanced_reviews_are_mixed():
    evidence = [review("supports"), review("contradicts", ref="r2")]

    result = judgeAspect(listing(claims=WIFI), "wifi", evidence, TODAY)

    assert result.verdict == "mixed"
    assert "claim_contradicted" in result.flags


def test_clear_majority_wins_over_a_single_complaint():
    evidence = [review("supports", ref=f"r{i}") for i in range(3)] + [review("contradicts", ref="r9")]

    assert judgeAspect(listing(claims=WIFI), "wifi", evidence, TODAY).verdict == "verified"


def test_old_praise_no_longer_counts_and_is_flagged_stale():
    # Baixa Classic Flat: wifi praise only ~9 years old
    evidence = [review("supports", days_ago=3300), review("supports", days_ago=3250, ref="r2")]

    result = judgeAspect(listing(claims=WIFI), "wifi", evidence, TODAY)

    assert result.verdict == "claim_only"
    assert "stale_evidence" in result.flags


def test_irrelevant_evidence_is_ignored():
    # Sé Studio: a prompt injection that mentions wifi is not evidence
    injection = review("irrelevant")

    result = judgeAspect(listing(claims=WIFI), "wifi", [injection], TODAY)

    assert result.verdict == "claim_only"
    assert result.evidence == []


def test_evidence_from_other_listings_is_ignored():
    other = review("contradicts", listing_id="2")

    assert judgeAspect(listing(claims=WIFI), "wifi", [other], TODAY).verdict == "claim_only"


def test_evidence_is_sorted_strongest_first():
    old, recent = review("supports", days_ago=700, ref="old"), review("supports", days_ago=10, ref="new")

    result = judgeAspect(listing(claims=WIFI), "wifi", [old, recent], TODAY)

    assert [e.ref for e in result.evidence] == ["new", "old"]


# gym nearby: claim (amenity), map (OSM) and reviews


def test_mapped_gym_within_radius_is_evidence_with_distance():
    stay = listing(nearest_gym_m=312.4, nearest_gym_name="Fitness Hut")

    result = judgeAspect(stay, "gym_nearby", [], TODAY)

    assert result.verdict == "evidence_only"
    assert result.evidence[0].source == "map"
    assert result.evidence[0].text == "Fitness Hut, ~312 m in a straight line (OpenStreetMap)"


def test_gym_claim_plus_map_is_verified():
    stay = listing(claims={"gym_nearby": "Shared gym nearby"}, nearest_gym_m=200)

    assert judgeAspect(stay, "gym_nearby", [], TODAY).verdict == "verified"


def test_far_or_unmapped_gym_is_not_a_contradiction():
    for stay in (listing(nearest_gym_m=900), listing(nearest_gym_m=None)):
        result = judgeAspect(stay, "gym_nearby", [], TODAY)

        assert result.verdict == "unknown"
        assert result.contradiction == 0


def test_mapped_metro_station_within_radius_is_transport_evidence():
    stay = listing(nearest_metro_m=287.6, nearest_metro_name="Trindade")

    result = judgeAspect(stay, "transport", [], TODAY)

    assert result.verdict == "evidence_only"  # nothing to claim: transport has no amenity
    assert result.evidence[0].source == "map"
    assert result.evidence[0].text == "Trindade, ~288 m in a straight line (OpenStreetMap)"


def test_far_or_unmapped_station_is_not_a_contradiction():
    for stay in (listing(nearest_metro_m=900), listing(nearest_metro_m=None)):
        result = judgeAspect(stay, "transport", [], TODAY)

        assert result.verdict == "unknown"
        assert result.contradiction == 0


def test_map_evidence_reads_only_its_own_distance():
    # a station but no gym, and a gym but no station: each aspect sees only its kind
    station_only = listing(nearest_metro_m=200, nearest_gym_m=None)
    gym_only = listing(nearest_gym_m=200, nearest_metro_m=None)

    assert judgeAspect(station_only, "transport", [], TODAY).verdict == "evidence_only"
    assert judgeAspect(station_only, "gym_nearby", [], TODAY).verdict == "unknown"
    assert judgeAspect(gym_only, "transport", [], TODAY).verdict == "unknown"
    assert judgeAspect(gym_only, "gym_nearby", [], TODAY).verdict == "evidence_only"


def test_unnamed_station_and_its_radius():
    stay = listing(nearest_metro_m=450)

    assert judgeAspect(stay, "transport", [], TODAY).evidence[0].text.startswith("a metro station, ~450 m")
    assert judgeAspect(stay, "transport", [], TODAY, RankingConfig(metro_radius_m=400)).evidence == []


def test_guest_mention_of_nearby_gym_counts():
    # Trindade Metro Studio: "There is a gym around the corner"
    result = judgeAspect(listing(), "gym_nearby", [review("supports", aspect="gym_nearby")], TODAY)

    assert result.verdict == "evidence_only"


# scores


def test_listed_speed_test_is_a_stronger_claim_than_plain_wifi():
    fast = listing(claims={"wifi": "Fast wifi – 300 Mbps"}, wifi_mbps=300)
    plain = listing(claims=WIFI)

    assert judgeAspect(fast, "wifi", [], TODAY).score > judgeAspect(plain, "wifi", [], TODAY).score


def test_complaints_weigh_more_than_praise():
    result = judgeAspect(listing(), "quiet", [
        review("supports", aspect="quiet"), review("contradicts", aspect="quiet", ref="r2"),
    ], TODAY)

    assert result.score < 0.5  # 0.5 = no evidence


def test_scores_depend_on_snapshot_date_not_on_the_clock():
    evidence = [review("supports", days_ago=0)]
    now = judgeAspect(listing(claims=WIFI), "wifi", evidence, TODAY)
    later = judgeAspect(listing(claims=WIFI), "wifi", evidence, TODAY + timedelta(days=540))

    assert later.support == pytest.approx(now.support / 2)


@pytest.mark.parametrize("aspect", list(ASPECTS))
def test_every_aspect_scores_between_0_and_1(aspect):
    evidence = [review("supports", aspect=aspect), review("contradicts", aspect=aspect, ref="r2")]

    result = judgeAspect(listing(), aspect, evidence, TODAY)

    assert 0 <= result.score <= 1


def test_score_listing_prefixes_flags_with_the_aspect():
    result = scoreListing(listing(claims=WIFI), ["wifi"], [review("contradicts")], TODAY)

    assert result.flags == ["wifi:claim_contradicted"]


def test_missing_rating_is_neutral():
    unrated = scoreListing(listing(rating=None), ["quiet"], [], TODAY)
    average = scoreListing(listing(rating=2.5), ["quiet"], [], TODAY)

    assert unrated.score == average.score


# ranking


def test_ranking_prefers_verified_then_claimed_then_contradicted():
    ribeira = listing("ribeira", claims={"wifi": "Fast wifi – 300 Mbps"}, wifi_mbps=300)
    foz = listing("foz", claims=WIFI, rating=4.95)
    bolhao = listing("bolhao", claims=WIFI)
    evidence = [
        review("supports", listing_id="ribeira"),
        review("supports", listing_id="ribeira", ref="r2"),
        review("contradicts", listing_id="bolhao", ref="r3"),
        review("contradicts", listing_id="bolhao", ref="r4"),
    ]

    ranked = rankListings([bolhao, foz, ribeira], ["wifi"], evidence, TODAY)

    assert [r.listing_id for r in ranked] == ["ribeira", "foz", "bolhao"]


def test_ties_prefer_more_evidence_then_cheaper():
    cheap, pricey = listing("cheap", price=50.0), listing("pricey", price=90.0)

    ranked = rankListings([pricey, cheap], ["quiet"], [], TODAY)

    assert [r.listing_id for r in ranked] == ["cheap", "pricey"]


def test_multi_aspect_score_averages_the_requested_aspects():
    stay = listing(claims=WIFI)
    evidence = [review("supports"), review("contradicts", aspect="quiet", ref="r2")]

    result = scoreListing(stay, ["wifi", "quiet"], evidence, TODAY)

    assert set(result.aspects) == {"wifi", "quiet"}
    assert result.aspects["wifi"].verdict == "verified"
    assert result.aspects["quiet"].verdict == "contradicted"
