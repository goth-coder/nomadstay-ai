"""
T6 guards added after the grounding review: unverified quotes never count as
evidence, the paraphrase pass can't break a search, the gym radius is the same
in the filter and in the evidence, and the citation check handles typography.
"""

import json
from types import SimpleNamespace

import pytest
from conftest import MODEL, fakeClient

from agent.nomad import (
    NomadStore,
    loadNomadStore,
    normalizeQuote,
    quoteInContext,
    search_stays,
    unverifiedQuotes,
)
from retrieval.evidence import EvidenceStore, judgeParaphrases
import contextvars
import time

from loadtest.mock_llm import MockOpenAI
from observability.context import startRequestStats


@pytest.fixture(scope="module")
def store(serving):
    return loadNomadStore(serving, MODEL)


def run(store, **overrides):
    args = {"city": "Porto", "max_price": None, "currency": None, "must_have": [],
            "preferences": ["wifi"], "max_gym_distance_m": None}
    return search_stays(**(args | overrides), store=store, paraphrases=False, limit=20)


# ---------- no verified quote, no evidence ----------


def test_precomputed_stance_without_a_valid_quote_is_not_counted(store):
    table = store.evidence.table.copy()
    wifi_support = table[(table["aspect"] == "wifi") & (table["stance"] == "supports")].index
    table.loc[wifi_support, "quote"] = None  # as if the judge's quotes had failed the check
    broken = NomadStore(store.catalog, EvidenceStore(table), None)

    for r in run(broken)["results"]:
        assert r["aspects"]["wifi"]["guest_reports"]["supports"] == 0


def test_paraphrase_label_with_an_invalid_quote_is_dropped():
    chunks = [{"chunk_id": 1, "listing_id": "L", "review_id": "r9", "date": None,
               "text": "Slept like a baby every night.", "aspects": ["quiet"]}]
    response = SimpleNamespace(output=[], output_text=json.dumps({"labels": [
        {"id": 0, "aspect": "quiet", "quote": "the room was silent", "stance": "supports", "strength": 1},
    ]}))

    evidence, _ = judgeParaphrases(fakeClient(response), chunks)

    assert evidence == []


def test_normalize_quote_ignores_punctuation_at_the_ends():
    assert normalizeQuote("safe and quiet neighborhood,") == normalizeQuote("Safe and quiet neighborhood")


def test_citation_check_ignores_punctuation_inside_the_quotes():
    output = {"results": [{"aspects": {"quiet": {"listing_claim": None, "guest_quotes": [
        {"quote": "Safe and quiet neighborhood"}]}}}]}
    answer = 'Guests liked the "safe and quiet neighborhood," while one found it noisy.'

    assert unverifiedQuotes(answer, [output]) == []


# ---------- the paraphrase pass never breaks the search ----------


def test_search_survives_a_failing_paraphrase_judge(store):
    client = fakeClient(*[RuntimeError("OpenAI down")] * 10)

    result = search_stays("Porto", None, None, [], ["quiet"], None, store=store,
                          client=client, paraphrases=True)

    assert result["status"] == "ok" and result["results"]


# ---------- gym radius: one radius for filter and evidence ----------


def test_gym_distance_filter_is_explained_by_map_evidence(store):
    result = run(store, preferences=["wifi"], max_gym_distance_m=1500)

    assert "gym_nearby" in result["aspects"]
    for r in result["results"]:
        gym = r["aspects"]["gym_nearby"]
        assert gym["map"] is not None, r["name"]  # every stay passed the 1500 m filter


def test_zero_gym_distance_is_not_read_as_the_default(store):
    kept, _ = store.catalog.filter(max_gym_distance_m=0)

    assert kept.empty


# ---------- must-haves, currency, snapshot ----------


def test_unclaimable_must_haves_are_ignored_not_fatal(store):
    result = run(store, must_have=["quiet", "transport"])

    assert result["status"] == "ok"
    assert result["filters"]["must_have"] == []


def test_budget_without_currency_is_read_as_the_dataset_currency(store):
    result = run(store, max_price=100, currency=None)

    assert "The budget was read as EUR." in result["notes"]


def test_snapshot_comes_from_the_data_without_the_index(serving):
    light = loadNomadStore(serving, MODEL, with_index=False)

    assert light.catalog.snapshot_date == "2026-09-19"
    assert light.index is None


# ---------- citation check ----------


def test_citation_check_handles_typography_short_quotes_and_claims(store):
    result = run(store)
    quote = next(q["quote"] for r in result["results"] for a in r["aspects"].values()
                 for q in a["guest_quotes"])
    claim = next(a["listing_claim"] for r in result["results"] for a in r["aspects"].values()
                 if a["listing_claim"])
    answer = (f'“{quote}” — and the host lists “{claim}”. '
              'One guest wrote "great wifi!!" and another "superfast fibre".')

    assert unverifiedQuotes(answer, [result]) == ["great wifi!!", "superfast fibre"]


def test_citation_check_matches_curly_and_straight_apostrophes():
    output = {"results": [{"aspects": {"wifi": {"listing_claim": None, "guest_quotes": [
        {"quote": "The wifi didn't work at night"}]}}}]}

    assert unverifiedQuotes("A guest said “The wifi didn’t work at night”.", [output]) == []


# ---------- T8: quote context, price with its date ----------


def test_short_quote_is_shown_with_its_sentence():
    review = "Lovely flat. Only two things were missing: Wlan und Espressomaschine. Host was kind."

    assert quoteInContext(review, "Wlan und Espressomaschine") == \
        "Only two things were missing: Wlan und Espressomaschine."


def test_quote_that_ends_a_sentence_does_not_take_the_next_one():
    review = "The wifi was fast. The street was loud."

    assert quoteInContext(review, "The wifi was fast.") == "The wifi was fast."


def test_quote_stays_as_is_when_it_cannot_be_placed_or_the_sentence_is_too_long():
    assert quoteInContext(None, "great wifi") == "great wifi"
    assert quoteInContext("Something else entirely.", "great wifi") == "great wifi"
    assert quoteInContext("x" * 400 + " great wifi " + "y" * 400, "great wifi") == "great wifi"


def test_shown_quotes_are_verbatim_sentences_of_the_review(store):
    for r in run(store)["results"]:
        for aspect in r["aspects"].values():
            for q in aspect["guest_quotes"]:
                assert q["quote"] in store.reviewText(q["review_id"])


def test_every_price_carries_the_snapshot_date(store):
    for r in run(store)["results"]:
        if r["price_per_night"] is not None:
            assert "2026-09-19" in r["price"] and "EUR" in r["price"]


def test_citation_check_does_not_pair_quote_marks_across_lines():
    answer = 'Wifi: "fast" (2025).\n- Quiet: guests said "very calm street" today.'

    assert unverifiedQuotes(answer, [{"results": []}]) == ["very calm street"]


def test_citation_check_accepts_words_cut_with_an_ellipsis():
    output = {"results": [{"aspects": {"wifi": {"listing_claim": None, "guest_quotes": [
        {"quote": "great location, comfortable bed, washing machine, parking and fast internet."}]}}}]}

    assert unverifiedQuotes('"great location, comfortable bed... and fast internet."', [output]) == []
    assert unverifiedQuotes('"great location, comfortable bed … and fast internet"', [output]) == []


def test_citation_check_rejects_an_ellipsis_that_reorders_or_joins_quotes():
    output = {"results": [{"aspects": {"wifi": {"listing_claim": None, "guest_quotes": [
        {"quote": "The wifi was fast."}, {"quote": "The street was loud at night."}]}}}]}

    assert unverifiedQuotes('"loud at night... The street was"', [output]) != []
    assert unverifiedQuotes('"The wifi was fast... loud at night"', [output]) != []


def test_citation_check_rejects_a_corrected_or_translated_quote():
    output = {"results": [{"aspects": {"quiet": {"listing_claim": None, "guest_quotes": [
        {"quote": "My girlfriend and wanted a place that was central but also quiet"},
        {"quote": "Há uma secretária para usar o computador."}]}}}]}

    answer = ('"My girlfriend and I wanted a place that was central but also quiet" and '
              '"There is a desk to use the computer."')
    assert len(unverifiedQuotes(answer, [output])) == 2


def test_paraphrase_batches_run_at_the_same_time_and_keep_request_stats():
    # 24 candidates = 3 batches of 10; each judge call takes 0.3 s
    candidates = [{"chunk_id": n, "listing_id": "L", "review_id": f"r{n}", "date": None,
                   "text": "Quiet street.", "aspects": ["quiet"]} for n in range(24)]
    client = MockOpenAI(0.3, 0.3)

    def request():
        stats = startRequestStats()
        start = time.perf_counter()
        judgeParaphrases(client, candidates)
        return stats, time.perf_counter() - start

    stats, seconds = contextvars.copy_context().run(request)
    assert seconds < 0.6  # one after another it would take 0.9 s
    assert stats.llm_calls == 3  # the threads added their calls to this request
