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

from dataclasses import dataclass

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

EVIDENCE_FILE = "evidence.parquet"
# What the API serves: the index set plus the evidence table; manifest last,
# so a partial download is never mistaken for a complete set
SERVING_FILES = [INDEX_FILE, CHUNKS_FILE, LISTINGS_FILE, POIS_FILE, EVIDENCE_FILE, MANIFEST]

PARAPHRASES_PER_ASPECT = 8
MAX_PARAPHRASES = 24  # per search, across aspects: at most 3 judge batches
MIN_PARAPHRASE_SCORE = 0.5  # cosine, multilingual-minilm; calibrated in T8
JUDGE_BATCH = 10


@dataclass
class EvidenceStore:
    table: pd.DataFrame  # listing_id, review_id, aspect, stance, strength, date, quote

    def __post_init__(self):
        self.rows_by_listing = self.table.groupby("listing_id").indices

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


def judgeParaphrases(client: OpenAI, candidates: list[dict]) -> tuple[list[Evidence], dict]:
    """Online judge for the paraphrase candidates, in small batches."""
    evidence, usage = [], {"input_tokens": 0, "output_tokens": 0, "calls": 0}
    for start in range(0, len(candidates), JUDGE_BATCH):
        batch = candidates[start:start + JUDGE_BATCH]
        local = [{"id": i, "text": c["text"], "aspects": c["aspects"]} for i, c in enumerate(batch)]
        labels, used = judgeReviews(client, local)
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
