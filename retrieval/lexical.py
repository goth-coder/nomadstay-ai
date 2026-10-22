"""
Lexical retrieval: BM25 over chunk texts, and Reciprocal Rank Fusion (RRF)
to combine it with the embedding ranking.

BM25 matches words, so a query only finds texts in its own language; the
curated multilingual lexicon (domain.aspects) covers the known aspects.
IDF should come from the whole city corpus, not only the chunks being ranked.
"""

import re
import unicodedata

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer

TOKEN = re.compile(r"\w\w+")


def tokenize(text: str) -> list[str]:
    # lowercase, accents removed ("ruído" == "ruido"), words of 2+ chars
    folded = unicodedata.normalize("NFKD", text.lower())
    folded = "".join(c for c in folded if not unicodedata.combining(c))
    return TOKEN.findall(folded)


class BM25:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.vectorizer = CountVectorizer(tokenizer=tokenize, lowercase=False, token_pattern=None)

    def fit(self, corpus: list[str]) -> "BM25":
        counts = self.vectorizer.fit_transform(corpus)
        n_docs = counts.shape[0]
        df = np.bincount(counts.indices, minlength=counts.shape[1])
        self.idf = np.log((n_docs - df + 0.5) / (df + 0.5) + 1)
        self.avgdl = counts.sum() / max(n_docs, 1)
        return self

    def transform(self, docs: list[str]) -> sparse.csr_matrix:
        """Term counts of the documents to rank (vocabulary from fit)."""
        return self.vectorizer.transform(docs).tocsr()

    def score(self, query: str, docs: sparse.csr_matrix) -> np.ndarray:
        terms = [self.vectorizer.vocabulary_[t] for t in set(tokenize(query))
                 if t in self.vectorizer.vocabulary_]
        if not terms:
            return np.zeros(docs.shape[0])
        doc_len = np.asarray(docs.sum(axis=1)).ravel()
        norm = self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
        tf = docs[:, terms].toarray()
        return (self.idf[terms] * tf * (self.k1 + 1) / (tf + norm[:, None])).sum(axis=1)


def ranks(scores: np.ndarray) -> np.ndarray:
    """1-based rank of each item (1 = best); ties keep their order."""
    order = np.argsort(-scores, kind="stable")
    result = np.empty(len(scores), dtype=int)
    result[order] = np.arange(1, len(scores) + 1)
    return result


def reciprocalRankFusion(*score_lists: np.ndarray, k: int = 60) -> np.ndarray:
    """RRF: sum of 1 / (k + rank) over the rankings; uses ranks, not raw scores,
    so BM25 (unbounded) and cosine (-1..1) can be combined."""
    return sum(1 / (k + ranks(scores)) for scores in score_lists)

