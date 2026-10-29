"""
Vectorized prescore (domain.ranking.prescore) vs the reference ranking
(rankListings on every candidate): same scores and, through rankCandidates,
the same stays in the same order, for every combination of aspects.
"""

import datetime as dt
from itertools import combinations

import pytest
from conftest import MODEL

from agent.nomad import loadNomadStore, rankCandidates
from domain.ranking import RankingConfig, prescore, rankListings

ASPECTS = ["wifi", "workspace", "quiet", "gym_nearby", "transport"]
COMBOS = [list(c) for n in range(0, 4) for c in combinations(ASPECTS, n)]


@pytest.fixture(scope="module")
def store(serving):
    return loadNomadStore(serving, MODEL)


def reference(store, ids, aspects, today, config):
    """The ranking before prescore: every candidate through rankListings."""
    listings = [store.listings_by_id[i] for i in ids]
    return rankListings(listings, aspects, store.evidence.precomputed(ids, aspects), today, config)


@pytest.mark.parametrize("aspects", COMBOS, ids=lambda a: "+".join(a) or "none")
def test_prescore_matches_score_listing(store, aspects):
    today = dt.date.fromisoformat(store.catalog.snapshot_date)
    config = RankingConfig()
    ids = list(store.ranking_frame.index)

    fast = prescore(store.ranking_frame.loc[ids], aspects, store.evidence.scorable, today, config)
    exact = {r.listing_id: r.score for r in reference(store, ids, aspects, today, config)}

    for listing_id, row in fast.iterrows():
        assert row["score"] == pytest.approx(exact[listing_id], abs=1e-3), listing_id


@pytest.mark.parametrize("aspects", COMBOS, ids=lambda a: "+".join(a) or "none")
@pytest.mark.parametrize("radius", [500, 1500])
def test_small_pool_gives_the_reference_top(store, aspects, radius):
    today = dt.date.fromisoformat(store.catalog.snapshot_date)
    config = RankingConfig(gym_radius_m=radius)
    ids = list(store.ranking_frame.index)

    ranked, _, _ = rankCandidates(store, ids, aspects, today, config, pool=5)
    exact = reference(store, ids, aspects, today, config)

    top = 3  # a pool barely above the top: prescore alone must find it
    assert [(r.listing_id, r.score) for r in ranked[:top]] == [(r.listing_id, r.score) for r in exact[:top]]
