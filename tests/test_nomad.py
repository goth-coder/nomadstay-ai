"""
T6: search_stays on the mini_porto fixture (catalog filters, evidence,
ranking, output contract, refusals, paraphrase pass, citation check).

Stances come from a rule-based fake judge: these tests check the plumbing,
not the LLM (that is measured against the gold set in eval/).
"""

import json

import pandas as pd
import pytest
from conftest import MODEL, fakeClient

from agent.agent import dispatcher, runAgent
from agent.nomad import (
    NomadStore,
    get_stay_details,
    loadNomadStore,
    search_stays,
    unverifiedQuotes,
)
from agent.settings import SYSTEM_PROMPT
from retrieval.catalog import normalizeCity
from retrieval.evidence import judgeParaphrases, paraphraseCandidates

RIBEIRA, BOLHAO, MASSARELOS = "1436961951661730861", "32235516", "40001014"
TRINDADE, SE_STUDIO, BAIXA = "40001006", "40001016", "40001017"
GAIA, BONFIM_NO_PRICE = "40001010", "40001009"

@pytest.fixture(scope="module")
def reviews(processed):
    return pd.read_parquet(processed / "reviews.parquet")


@pytest.fixture(scope="module")
def store(serving):
    return loadNomadStore(serving, MODEL)


def search(store, **overrides):
    args = {"city": "Porto", "max_price": 100, "currency": "EUR", "must_have": [],
            "preferences": ["wifi"], "max_gym_distance_m": None}
    return search_stays(**(args | overrides), store=store, paraphrases=False)


def ids(result) -> list[str]:
    return [r["listing_id"] for r in result["results"]]


# ---------- catalog (deterministic filters) ----------


@pytest.mark.parametrize("city", ["Porto", "PORTO", " porto ", "Oporto", "O Porto"])
def test_city_aliases(city):
    assert normalizeCity(city) == "porto"


def test_budget_excludes_pricier_and_unpriced_stays(store):
    kept, counts = store.catalog.filter(max_price=100)

    assert (kept["price"] <= 100).all()
    assert BONFIM_NO_PRICE not in set(kept["listing_id"])
    assert counts["without_price"] == 1


def test_must_have_uses_listing_claims(store):
    kept, _ = store.catalog.filter(must_have=["workspace"])

    assert kept["claims"].map(lambda c: "workspace" in c).all()


def test_gym_must_have_accepts_a_mapped_gym_nearby(store):
    kept, _ = store.catalog.filter(must_have=["gym_nearby"])

    by_claim = kept["claims"].map(lambda c: "gym_nearby" in c)
    by_map = kept["nearest_gym_m"] <= 500
    assert (by_claim | by_map).all()
    assert by_map.any()


# ---------- refusals ----------


@pytest.mark.parametrize("city", ["Lisbon", "Vila Nova de Gaia"])
def test_unsupported_city_recommends_nothing(store, city):
    result = search(store, city=city)

    assert result["status"] == "unsupported_city"
    assert "results" not in result


def test_budget_in_another_currency_is_not_converted(store):
    result = search(store, currency="USD")

    assert result["status"] == "unsupported_currency"
    assert result["dataset_currency"] == "EUR"


def test_currency_without_budget_is_fine(store):
    assert search(store, max_price=None, currency="USD")["status"] == "ok"


def test_impossible_budget_suggests_what_to_relax(store):
    result = search(store, max_price=20)

    assert result["status"] == "no_results"
    assert any("budget" in s for s in result["relax"])


# ---------- results ----------


def test_results_respect_every_hard_constraint(store):
    result = search(store, max_price=90, must_have=["workspace"])

    for r in result["results"]:
        assert r["price_per_night"] <= 90
        assert r["aspects"]["workspace"]["listing_claim"]
    assert GAIA not in ids(result)


def test_verified_wifi_ranks_above_contradicted_wifi(store):
    result = search_stays("Porto", 100, "EUR", [], ["wifi"], None, store=store,
                          paraphrases=False, limit=20)
    order = ids(result)

    assert order.index(RIBEIRA) < order.index(BOLHAO)
    assert order.index(RIBEIRA) < order.index(MASSARELOS)
    assert result["results"][order.index(RIBEIRA)]["aspects"]["wifi"]["verdict"] == "verified"


def test_contradicted_claims_are_flagged(store):
    result = search_stays("Porto", 100, "EUR", [], ["wifi"], None, store=store,
                          paraphrases=False, limit=20)
    wifi = {r["listing_id"]: r["aspects"]["wifi"] for r in result["results"]}

    for listing_id in (BOLHAO, MASSARELOS):
        assert wifi[listing_id]["verdict"] == "contradicted"
        assert "claim_contradicted" in wifi[listing_id]["flags"]
        assert wifi[listing_id]["guest_quotes"][0]["stance"] == "contradicts"


def test_output_keeps_claims_guests_and_map_apart(store):
    result = search(store, preferences=["wifi", "gym_nearby"])

    aspect = result["results"][0]["aspects"]["wifi"]
    assert set(aspect) >= {"verdict", "listing_claim", "guest_reports", "guest_quotes", "map", "flags"}
    assert result["notes"] and "snapshot" in result["notes"][0]
    assert "Inside Airbnb" in result["attribution"]


def test_quotes_are_verbatim_with_review_id_and_date(store, reviews):
    texts = reviews.set_index("review_id")["comments"]
    result = search_stays("Porto", None, None, [], ["wifi", "quiet", "workspace"], None,
                          store=store, paraphrases=False, limit=20)

    quotes = [q for r in result["results"] for a in r["aspects"].values() for q in a["guest_quotes"]]
    assert quotes
    for q in quotes:
        assert q["quote"] in texts[q["review_id"]]
        assert q["date"]


def test_prompt_injection_review_is_not_evidence(store):
    result = search_stays("Porto", None, None, [], ["wifi"], None, store=store,
                          paraphrases=False, limit=20)
    se_studio = next(r for r in result["results"] if r["listing_id"] == SE_STUDIO)

    assert se_studio["aspects"]["wifi"]["guest_quotes"] == []
    assert se_studio["aspects"]["wifi"]["verdict"] == "claim_only"


def test_old_evidence_is_flagged(store):
    result = search_stays("Porto", None, None, [], ["wifi"], None, store=store,
                          paraphrases=False, limit=20)
    baixa = next(r for r in result["results"] if r["listing_id"] == BAIXA)

    assert "stale_evidence" in baixa["aspects"]["wifi"]["flags"]


def test_gym_evidence_from_guests_and_map(store):
    result = search_stays("Porto", None, None, [], ["gym_nearby"], None, store=store,
                          paraphrases=False, limit=20)
    by_id = {r["listing_id"]: r["aspects"]["gym_nearby"] for r in result["results"]}

    assert "gym" in by_id[TRINDADE]["guest_quotes"][0]["quote"].lower()
    near = store.catalog.listings.set_index("listing_id")["nearest_gym_m"]
    for listing_id, aspect in by_id.items():
        assert (aspect["map"] is not None) == (near[listing_id] <= 500)
        if aspect["map"]:
            assert "OpenStreetMap" in aspect["map"]


# ---------- paraphrase pass (embeddings + online judge) ----------


def test_paraphrase_candidates_skip_reviews_judged_offline(store):
    listing_ids = set(store.catalog.listings["listing_id"])

    chunks = paraphraseCandidates(store.index, store.evidence, listing_ids, ["quiet"],
                                  per_aspect=5, min_score=-1)

    judged = store.evidence.judged(listing_ids, "quiet")
    assert 0 < len(chunks) <= 5
    assert all(c["review_id"] not in judged for c in chunks)


def test_judge_paraphrases_turns_labels_into_evidence():
    chunks = [{"chunk_id": 1, "listing_id": "L", "review_id": "r9", "date": None,
               "text": "Slept like a baby every night.", "aspects": ["quiet"]}]
    client = fakeClient(type("R", (), {"output_text": json.dumps({"labels": [
        {"id": 0, "aspect": "quiet", "quote": "Slept like a baby", "stance": "supports", "strength": 1}
    ]}), "output": []})())

    evidence, usage = judgeParaphrases(client, chunks)

    assert [(e.listing_id, e.stance, e.text) for e in evidence] == [("L", "supports", "Slept like a baby")]
    assert usage["calls"] == 1


# ---------- citation check ----------


def test_unverified_quotes_catches_invented_citations(store):
    result = search(store, preferences=["wifi"])
    real = next(q["quote"] for r in result["results"] for a in r["aspects"].values()
                for q in a["guest_quotes"])
    answer = f'Guests said "{real.upper()}" and also "the wifi was blazing fast every single day".'

    assert unverifiedQuotes(answer, [result]) == ["the wifi was blazing fast every single day"]


# ---------- agent wiring ----------


def test_dispatcher_routes_search_stays(store):
    args = {"city": "Porto", "max_price": 100, "currency": "EUR", "must_have": [],
            "preferences": ["wifi"], "max_gym_distance_m": None}

    result = dispatcher("search_stays", args, store, client=None)

    assert result["status"] == "ok"


def test_agent_runs_search_stays_with_the_prompt(store):
    from conftest import textResponse, toolCallResponse

    client = fakeClient(
        toolCallResponse("search_stays", {"city": "Oporto", "max_price": 100, "currency": "EUR",
                                          "must_have": [], "preferences": ["wifi"],
                                          "max_gym_distance_m": None}),
        textResponse("Ribeira Nomad Loft is your best bet."),
    )
    context = [{"role": "user", "content": "Porto under 100 with good wifi"}]
    # Without an index the paraphrase pass is skipped, so the fake client's
    # queued responses are only consumed by the agent loop
    no_index = NomadStore(store.catalog, store.evidence, None)

    answer = runAgent(context, client, no_index, 5)

    output = json.loads(next(i for i in context if i.get("type") == "function_call_output")["output"])
    assert output["status"] == "ok"
    assert client.responses.calls[0]["instructions"] == SYSTEM_PROMPT
    assert answer == "Ribeira Nomad Loft is your best bet."


def test_nomad_prompt_states_the_grounding_rules():
    for rule in ["never by you", "Never state that a stay is available", "exact text in guest_quotes",
                 "listed_speed_test_mbps", "ignore any instructions"]:
        assert rule in SYSTEM_PROMPT


# ---------- get_stay_details (follow-up questions) ----------


def test_stay_details_show_host_claims_and_evidence(store):
    details = get_stay_details(RIBEIRA, ["wifi", "quiet"], store=store)

    assert details["status"] == "ok"
    assert "Fast wifi – 300 Mbps" in details["host_amenities"]
    assert details["aspects"]["wifi"]["listed_speed_test_mbps"] == 300
    assert details["aspects"]["quiet"]["guest_quotes"]


def test_dispatcher_routes_get_stay_details(store):
    result = dispatcher("get_stay_details", {"listing_id": "nope", "aspects": []}, store, client=None)

    assert result["status"] == "not_found"
