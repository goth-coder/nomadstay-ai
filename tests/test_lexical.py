"""
BM25 and Reciprocal Rank Fusion (retrieval/lexical.py).
"""

import numpy as np

from retrieval.lexical import BM25, ranks, reciprocalRankFusion, tokenize

CORPUS = [
    "The wifi was fast and stable",
    "Lovely host and great location",
    "Great location near the metro",
    "Noisy street at night, the wifi dropped",
    "Great host, great breakfast",
]


def test_tokenize_folds_case_and_accents():
    assert tokenize("Ruído À Noite, wi-fi OK!") == ["ruido", "noite", "wi", "fi", "ok"]


def test_bm25_ranks_documents_with_the_query_terms_first():
    bm25 = BM25().fit(CORPUS)

    scores = bm25.score("wifi", bm25.transform(CORPUS))

    assert set(np.flatnonzero(scores > 0)) == {0, 3}


def test_rare_terms_weigh_more_than_common_ones():
    bm25 = BM25().fit(CORPUS)
    docs = bm25.transform(["great metro"])
    vocab = bm25.vectorizer.vocabulary_

    assert bm25.idf[vocab["metro"]] > bm25.idf[vocab["great"]]
    assert bm25.score("metro", docs)[0] > bm25.score("great", docs)[0]


def test_unknown_query_terms_score_zero():
    bm25 = BM25().fit(CORPUS)

    assert not bm25.score("piscina", bm25.transform(CORPUS)).any()


def test_ranks_are_one_based_and_stable_on_ties():
    assert list(ranks(np.array([0.2, 0.9, 0.2]))) == [2, 1, 3]


def test_rrf_rewards_items_ranked_well_by_both_methods():
    # item 0: 1st lexical, last semantic; item 1: the reverse; item 2: 2nd in both
    lexical = np.array([5.0, 0.0, 4.0, 3.0, 2.0])
    semantic = np.array([0.0, 0.9, 0.8, 0.5, 0.4])

    fused = reciprocalRankFusion(lexical, semantic)

    assert np.argmax(fused) == 2  # consensus beats being first in only one
