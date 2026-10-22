"""
T5: evaluation metrics (eval/metrics.py), scenario file (eval/queries.yaml)
and the stratified sampling of labeling candidates.
"""

from pathlib import Path

import pandas as pd
import pytest
import yaml

from domain.aspects import ASPECTS
from eval.build_gold import goldRow, validateLabel
from eval.metrics import (
    citationValidity,
    constraintViolations,
    ndcgAtK,
    percentile,
    recallAtK,
    reciprocalRank,
    stanceReport,
)
from eval.sample_for_labeling import sampleCandidates

QUERIES = Path(__file__).parents[1] / "eval" / "queries.yaml"


# retrieval


def test_recall_at_k_counts_relevant_in_the_first_k():
    assert recallAtK(["a", "b", "c", "d"], {"b", "d", "z"}, k=2) == pytest.approx(1 / 3)
    assert recallAtK(["a"], set(), k=5) == 0.0


def test_ndcg_is_1_for_the_ideal_order_and_lower_otherwise():
    grades = {"a": 2, "b": 1}

    assert ndcgAtK(["a", "b", "c"], grades, k=3) == pytest.approx(1.0)
    assert ndcgAtK(["c", "b", "a"], grades, k=3) < 1.0
    assert ndcgAtK(["x"], {}, k=3) == 0.0


def test_reciprocal_rank_uses_the_first_hit():
    assert reciprocalRank(["x", "a", "b"], {"a", "b"}) == 0.5
    assert reciprocalRank(["x"], {"a"}) == 0.0


# stance


def test_stance_report_per_class_and_contradiction_recall():
    gold = ["supports", "contradicts", "contradicts", "irrelevant"]
    pred = ["supports", "contradicts", "supports", "irrelevant"]

    report = stanceReport(gold, pred)

    assert report["contradiction_recall"] == 0.5
    assert report["per_class"]["supports"]["precision"] == 0.5
    assert report["accuracy"] == 0.75
    assert 0 < report["macro_f1"] < 1


def test_stance_report_rejects_misaligned_inputs():
    with pytest.raises(ValueError):
        stanceReport(["supports"], [])


# recommendation


def test_constraint_violations_catch_city_price_and_must_have():
    results = [
        {"listing_id": "ok", "city": "porto", "price": 80, "claims": {"wifi": "Wifi"}},
        {"listing_id": "pricey", "city": "porto", "price": 120, "claims": {"wifi": "Wifi"}},
        {"listing_id": "noprice", "city": "porto", "price": None, "claims": {"wifi": "Wifi"}},
        {"listing_id": "gaia", "city": "gaia", "price": 70, "claims": {"wifi": "Wifi"}},
        {"listing_id": "nowifi", "city": "porto", "price": 70, "claims": {}},
    ]

    violations = constraintViolations(
        results, {"city": "porto", "max_price": 100, "must_have": ["wifi"]}
    )

    assert {v.split(":")[0] for v in violations} == {"pricey", "noprice", "gaia", "nowifi"}


def test_constraint_violations_without_constraints_is_empty():
    assert constraintViolations([{"listing_id": "x", "price": None}], {}) == []


def test_citation_validity_requires_verbatim_quotes_from_the_cited_review():
    sources = {"r1": "The wifi was super fast, I had video calls all week."}
    citations = [
        {"ref": "r1", "quote": "The wifi was super fast"},  # verbatim
        {"ref": "r1", "quote": "The wifi was very fast"},  # paraphrase
        {"ref": "r9", "quote": "The wifi was super fast"},  # wrong review
    ]

    assert citationValidity(citations, sources) == pytest.approx(1 / 3)
    assert citationValidity([], sources) == 1.0


def test_percentile_nearest_rank():
    values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    assert percentile(values, 50) == 5
    assert percentile(values, 95) == 10
    assert percentile([], 95) == 0.0


# scenarios


@pytest.fixture(scope="module")
def scenarios() -> list[dict]:
    return yaml.safe_load(QUERIES.read_text(encoding="utf-8"))["scenarios"]


def test_scenarios_are_well_formed(scenarios):
    expects = {"results", "no_results", "unsupported_city", "unsupported_currency"}

    assert len(scenarios) >= 25
    assert len({s["id"] for s in scenarios}) == len(scenarios)
    for s in scenarios:
        intent = s["intent"]
        assert s["expect"] in expects, s["id"]
        assert set(intent["must_have"]) <= set(ASPECTS), s["id"]
        assert set(intent["preferences"]) <= set(ASPECTS), s["id"]
        assert intent["max_price"] is None or intent["max_price"] > 0, s["id"]
        assert s["must_not"], s["id"]


def test_scenarios_cover_the_refusal_cases(scenarios):
    expects = {s["expect"] for s in scenarios}

    assert {"no_results", "unsupported_city", "unsupported_currency"} <= expects


# labeling candidates


def test_sample_candidates_is_stratified_deterministic_and_unique():
    reviews = pd.DataFrame({
        "review_id": [str(i) for i in range(40)],
        "listing_id": ["L"] * 40,
        "aspects": [["wifi"]] * 20 + [["wifi", "quiet"]] * 10 + [["quiet"]] * 10,
    })

    first = sampleCandidates(reviews, {"wifi": 15, "quiet": 15}, seed=1)
    second = sampleCandidates(reviews, {"wifi": 15, "quiet": 15}, seed=1)

    assert first == second
    assert sum(r["aspect"] == "wifi" for r in first) == 15
    assert len({r["review_id"] for r in first}) == len(first)  # one aspect per review
    assert all(set(r) == {"review_id", "listing_id", "aspect"} for r in first)  # ids only


# gold set


def label(stance="supports", strength=2, quote="fast wifi"):
    return {"item": 0, "review_id": "r1", "aspect": "wifi", "stance": stance,
            "strength": strength, "quote": quote, "note": ""}


TEXT = "Great host. The fast wifi made calls easy."


def test_validate_label_accepts_a_verbatim_quote():
    assert validateLabel(label(), TEXT) == []
    assert validateLabel(label("irrelevant", 0, ""), TEXT) == []


@pytest.mark.parametrize(
    "bad",
    [
        label(quote="very fast wifi"),  # paraphrase
        label(quote=""),  # supports without a quote
        label("irrelevant", 0, "fast wifi"),  # irrelevant with a quote
        label(strength=0),  # supports needs strength 1-2
        label("maybe", 1),  # unknown stance
    ],
)
def test_validate_label_rejects_inconsistent_labels(bad):
    assert validateLabel(bad, TEXT)


def test_gold_row_stores_offsets_instead_of_text():
    candidate = {"review_id": "r1", "listing_id": "L1", "aspect": "wifi"}

    row = goldRow(label(), candidate, TEXT, "test")

    assert "quote" not in row and "text" not in row
    assert TEXT[row["quote_start"]:row["quote_end"]] == "fast wifi"
