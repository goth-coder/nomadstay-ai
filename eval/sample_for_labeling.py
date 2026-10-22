"""
Samples review/aspect pairs to label with a stance (T5).

    python -m eval.sample_for_labeling --city porto

Stratified by aspect (keyword mentions), fixed seed. Writes ids only to
eval/labels/stance_candidates.jsonl: review text stays in the dataset and is
joined at labeling time, so the repo doesn't redistribute guest comments.
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from constants import processedDir

LABELS_DIR = Path("eval/labels")
PER_ASPECT = {"wifi": 100, "workspace": 60, "quiet": 60, "gym_nearby": 40, "transport": 40}
SEED = 20260919


def sampleCandidates(reviews: pd.DataFrame, per_aspect: dict[str, int], seed: int = SEED) -> list[dict]:
    """One row per (review, aspect). A review is used for one aspect only."""
    used: set[str] = set()
    rows = []
    for aspect, n in per_aspect.items():
        mentions = reviews[reviews["aspects"].map(lambda names, a=aspect: a in names)]
        mentions = mentions[~mentions["review_id"].isin(used)]
        picked = mentions.sample(n=min(n, len(mentions)), random_state=seed)
        used.update(picked["review_id"])
        rows += [
            {"review_id": r.review_id, "listing_id": r.listing_id, "aspect": aspect}
            for r in picked.itertuples()
        ]
    return rows


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Sample review/aspect pairs to label")
    parser.add_argument("--city", default="porto")
    args = parser.parse_args(argv)

    reviews = pd.read_parquet(processedDir(args.city) / "reviews.parquet")
    rows = sampleCandidates(reviews, PER_ASPECT)

    LABELS_DIR.mkdir(parents=True, exist_ok=True)
    out = LABELS_DIR / "stance_candidates.jsonl"
    out.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    print(f"{len(rows)} candidates -> {out}")


if __name__ == "__main__":
    main()
