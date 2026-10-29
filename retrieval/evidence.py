"""
Guest evidence for the search, from two sources:

- precomputed: every review that mentions an aspect keyword was judged
  offline (ingest/stance.py) -> evidence.parquet. Most evidence lives here.
- paraphrases: reviews that talk about an aspect without its keywords
  ("slept like a baby", "thin walls") are found by the embedding search,
  only among the shortlisted listings, and judged online (capped).

Together this is the hybrid retrieval that won the T5 evaluation:
lexicon first, embeddings for what the lexicon misses.
"""
import contextvars
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
import pandas as pd
from openai import OpenAI

from agent.judge import judgeReviews
from domain.aspects import ASPECTS
from domain.ranking import Evidence
from retrieval.chunk_index import (
    CHUNKS_FILE,
    INDEX_FILE,
    LISTINGS_FILE,
    MANIFEST,
    POIS_FILE,
    ChunkIndex,
    search,
)
from retrieval.lexical import BM25, reciprocalRankFusion

EVIDENCE_FILE = "evidence.parquet"
# What the API serves: the index set plus the evidence table; manifest last,
# so a partial download is never mistaken for a complete set
SERVING_FILES = [INDEX_FILE, CHUNKS_FILE, LISTINGS_FILE, POIS_FILE, EVIDENCE_FILE, MANIFEST]

PARAPHRASES_PER_ASPECT = 8
MAX_PARAPHRASES = 24  # per search, across aspects: at most 3 judge batches
MIN_PARAPHRASE_SCORE = 0.5  # cosine, multilingual-minilm; calibrated in T8
JUDGE_BATCH = 10
JUDGE_WORKERS = 3  # MAX_PARAPHRASES / JUDGE_BATCH: every batch at once


@dataclass
class EvidenceStore:
    table: pd.DataFrame  # listing_id, review_id, aspect, stance, strength, date, quote

    def __post_init__(self):
        self.rows_by_listing = self.table.groupby("listing_id").indices
        # The rows precomputed() turns into evidence, as columns for domain.ranking.prescore
        judged = self.table[self.table["stance"].isin(["supports", "contradicts"])
                            & self.table["quote"].notna() & (self.table["quote"] != "")]
        self.scorable = judged[["listing_id", "aspect", "stance"]].assign(date=pd.to_datetime(judged["date"]))

    def rows(self, listing_ids) -> pd.DataFrame:
        positions = [p for i in listing_ids for p in self.rows_by_listing.get(i, [])]
        return self.table.iloc[sorted(positions)]

    def precomputed(self, listing_ids, aspects: list[str]) -> list[Evidence]:
        rows = self.rows(listing_ids)
        rows = rows[rows["aspect"].isin(aspects) & rows["stance"].isin(["supports", "contradicts"])]
        # A stance whose quote could not be verified is not shown nor counted:
        # "no quote means irrelevant", enforced in code
        rows = rows[rows["quote"].notna() & (rows["quote"] != "")]
        return [
            Evidence(listing_id=r.listing_id, aspect=r.aspect, source="review", stance=r.stance,
                     ref=r.review_id, text=r.quote or "", date=r.date)
            for r in rows.itertuples()
        ]

    def judged(self, listing_ids, aspect: str) -> set[str]:
        """Review ids already judged for this aspect (any stance)."""
        rows = self.rows(listing_ids)
        return set(rows.loc[rows["aspect"] == aspect, "review_id"])


def paraphraseCandidates(index: ChunkIndex, evidence: EvidenceStore, listing_ids: set[str],
                         aspects: list[str], per_aspect: int = PARAPHRASES_PER_ASPECT,
                         min_score: float = MIN_PARAPHRASE_SCORE) -> list[dict]:
    """
    Review chunks of the given listings that the embedding search ranks high
    for an aspect but that were not judged offline for it.
    Returns {chunk_id, listing_id, review_id, date, text, aspects}.
    """
    by_chunk: dict[int, dict] = {}
    for aspect in aspects:
        judged = evidence.judged(listing_ids, aspect)
        hits = {}
        for query in ASPECTS[aspect].queries:
            for hit in search(index, query, k=per_aspect * 5, allowed_listing_ids=listing_ids):
                if (hit["source"] == "review" and hit["review_id"] not in judged
                        and hit["score"] >= min_score):
                    best = hits.get(hit["chunk_id"])
                    if best is None or hit["score"] > best["score"]:
                        hits[hit["chunk_id"]] = hit
        for hit in sorted(hits.values(), key=lambda h: -h["score"])[:per_aspect]:
            entry = by_chunk.setdefault(hit["chunk_id"], {
                "chunk_id": hit["chunk_id"], "listing_id": hit["listing_id"],
                "review_id": hit["review_id"], "date": hit["date"], "text": hit["text"],
                "aspects": [], "score": hit["score"],
            })
            entry["score"] = max(entry["score"], hit["score"])
            entry["aspects"].append(aspect)
    ranked = sorted(by_chunk.values(), key=lambda c: -c["score"])
    return [{k: v for k, v in c.items() if k != "score"} for c in ranked[:MAX_PARAPHRASES]]


def judgeBatch(client: OpenAI, batch: list[dict]) -> tuple[list[dict], dict]:
    """One judge call for one batch of paraphrase candidates."""
    local = [{"id": i, "text": c["text"], "aspects": c["aspects"]} for i, c in enumerate(batch)]
    return judgeReviews(client, local)


def judgeParaphrases(client: OpenAI, candidates: list[dict]) -> tuple[list[Evidence], dict]:
    """Online judge for the paraphrase candidates: the batches run at the same time."""
    batches = [candidates[start:start + JUDGE_BATCH] for start in range(0, len(candidates), JUDGE_BATCH)]

    with ThreadPoolExecutor(max_workers=JUDGE_WORKERS) as pool:
        # each batch in its own copy of the request context: request_id and cost stats follow it
        futures = [pool.submit(contextvars.copy_context().run, judgeBatch, client, batch)
                   for batch in batches]
        results = [future.result() for future in futures]

    evidence, usage = [], {"input_tokens": 0, "output_tokens": 0, "calls": 0}
    for batch, (labels, used) in zip(batches, results):
        for k in usage:
            usage[k] += used[k]
        for label in labels:
            if label["stance"] in ("supports", "contradicts") and label["quote_valid"] and label["quote"]:
                c = batch[label["id"]]
                evidence.append(Evidence(
                    listing_id=c["listing_id"], aspect=label["aspect"], source="review",
                    stance=label["stance"], ref=c["review_id"], text=label["quote"], date=c["date"],
                ))
    return evidence, usage


QUESTION_CHUNKS_PER_LISTING = 4


def questionCandidates(index: ChunkIndex, listing_ids: list[str], question: str,
                       per_listing: int = QUESTION_CHUNKS_PER_LISTING) -> list[dict]:
    """
    Review chunks of each listing that best match a free-text question.
    Hybrid ranking inside each listing: embeddings (meaning, any language)
    fused with BM25 (exact words: names, rare terms) by reciprocal rank fusion.
    Returns {chunk_id, listing_id, review_id, date, text}, best first per listing.
    """
    allowed = [i for i in listing_ids if i in index.positions_by_listing]
    if not allowed:
        return []
    total = sum(len(index.positions_by_listing[i]) for i in allowed)
    hits = [h for h in search(index, question, k=total, allowed_listing_ids=set(allowed))
            if h["source"] == "review"]

    picked = []
    for listing_id in allowed:
        own = [h for h in hits if h["listing_id"] == listing_id]
        if not own:
            continue
        texts = [h["text"] for h in own]
        lexical = BM25().fit(texts).score(question, BM25().fit(texts).transform(texts)) \
            if len(texts) > 1 else np.zeros(len(texts))
        semantic = np.array([h["score"] for h in own])
        order = np.argsort(-reciprocalRankFusion(lexical, semantic), kind="stable")
        picked += [{k: own[i][k] for k in ("chunk_id", "listing_id", "review_id", "date", "text")}
                   for i in order[:per_listing]]
    return picked
