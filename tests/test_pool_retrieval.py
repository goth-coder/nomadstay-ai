"""
T5b: pooling for the retrieval evaluation (eval/pool_retrieval.py).
Embeddings are replaced by a deterministic fake: no model, no network.
"""

import numpy as np
import pandas as pd
import pytest

from eval import pool_retrieval
from eval.pool_retrieval import buildPool, reviewWindows, selectListings


def reviews(listing_id: str, n: int, wifi_mentions: int) -> list[dict]:
    return [
        {"listing_id": listing_id, "review_id": f"{listing_id}-{i}",
         "comments": f"Review {i} of {listing_id}. Lovely place.",
         "aspects": ["wifi"] if i < wifi_mentions else []}
        for i in range(n)
    ]


def test_select_listings_needs_enough_reviews_and_a_wifi_mention():
    frame = pd.DataFrame(reviews("A", 20, 1) + reviews("B", 20, 0) + reviews("C", 5, 3))

    assert selectListings(frame, n=10) == ["A"]


def test_select_listings_is_deterministic():
    frame = pd.DataFrame([r for i in range(40) for r in reviews(f"L{i}", 15, 1)])

    assert selectListings(frame, n=5, seed=1) == selectListings(frame, n=5, seed=1)


def test_review_windows_key_by_review_and_window():
    frame = pd.DataFrame([{
        "listing_id": "A", "review_id": "r1",
        "comments": "Fast wifi for calls. " * 60,  # longer than one window
    }])

    windows = reviewWindows(frame)

    assert list(windows["window"]) == list(range(len(windows)))
    assert len(windows) > 1
    assert set(windows["review_id"]) == {"r1"}


@pytest.fixture
def fakeEmbeddings(monkeypatch):
    """
    Each model 'prefers' a different chunk: model m scores chunk text 'cM'
    highest, so the pool must be the union of every model's top-k.
    """
    def embed(texts, key, kind="passage"):
        vectors = np.zeros((len(texts), 3), dtype=np.float32)
        for i, text in enumerate(texts):
            if kind == "query":
                vectors[i, ["e5-small", "minilm", "multilingual-minilm"].index(key)] = 1
            else:
                if text.startswith("c"):
                    vectors[i, int(text[1])] = 1
        return vectors

    monkeypatch.setattr(pool_retrieval, "embedTexts", embed)


def test_pool_is_the_union_of_each_models_top_k(fakeEmbeddings):
    windows = pd.DataFrame({
        "listing_id": ["A"] * 4,
        "review_id": ["r0", "r1", "r2", "r3"],
        "window": [0, 0, 0, 0],
        "text": ["c0 text", "c1 text", "c2 text", "x other"],
    })

    entries = buildPool(windows, ["e5-small", "minilm", "multilingual-minilm"], ["wifi"], depth=1)

    assert {e["review_id"]: e["pooled_by"] for e in entries} == {
        "r0": ["e5-small"], "r1": ["minilm"], "r2": ["multilingual-minilm"],
    }


def test_depth_zero_pools_every_chunk(fakeEmbeddings):
    windows = pd.DataFrame({
        "listing_id": ["A", "A", "B"],
        "review_id": ["r0", "r1", "r2"],
        "window": [0, 0, 0],
        "text": ["c0 text", "x other", "c1 text"],
    })

    entries = buildPool(windows, ["minilm"], ["wifi", "quiet"], depth=0)

    assert len(entries) == 3 * 2  # every chunk x every aspect


def test_evaluate_scores_each_method_on_listings_with_evidence():
    from eval.run_retrieval_eval import evaluate

    windows = pd.DataFrame({
        "listing_id": ["A", "A", "A", "B"],
        "review_id": ["r0", "r1", "r2", "r3"],
        "window": [0, 0, 0, 0],
        "text": ["x", "y", "z", "w"],
    })
    gold = pd.DataFrame([
        {"listing_id": "A", "aspect": "wifi", "review_id": "r2", "window": 0, "grade": 2},
        {"listing_id": "A", "aspect": "wifi", "review_id": "r0", "window": 0, "grade": 0},
        {"listing_id": "B", "aspect": "wifi", "review_id": "r3", "window": 0, "grade": 0},
    ])
    methods = {
        "perfect": {"wifi": np.array([0.0, 0.1, 0.9, 0.0])},
        "worst": {"wifi": np.array([0.9, 0.5, 0.1, 0.0])},
    }

    results = evaluate(windows, gold, methods).set_index("method")

    assert set(results["listing_id"]) == {"A"}  # B has no evidence: skipped
    assert results.loc["perfect", "mrr"] == 1.0
    assert results.loc["worst", "mrr"] == pytest.approx(1 / 3)
    assert results.loc["worst", "recall@4"] == 1.0  # only 3 chunks
