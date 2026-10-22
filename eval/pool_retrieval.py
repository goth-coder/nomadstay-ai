"""
Pooling for the retrieval evaluation (T5b).

    python -m eval.pool_retrieval pool  --city porto --out <dir>
    python -m eval.pool_retrieval merge --city porto --labels <dir> --labeler "..."

pool: picks listings, splits their reviews into chunks (same windows as the
index), ranks each listing's chunks per aspect with every embedding model and
keeps the union of each model's top-k. Only the pool gets labeled: a relevant
chunk that no model ranks in its top-k cannot change the comparison between
models (TREC-style pooling). Recall measured on the pool is slightly optimistic;
compare models, don't read it as an absolute number.

Only review chunks are pooled: the ranking uses guest reviews as evidence,
never the host's description.

With --depth 0 every chunk is labeled (exhaustive): exact recall, and new
models can be evaluated later without labeling again.

merge: validates the labels (stance guidelines, grade = strength) and writes
eval/labels/retrieval_gold.jsonl with ids only.
"""

import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from constants import processedDir
from domain.aspects import ASPECTS
from eval.build_gold import validateLabel
from ingest.chunks import splitText
from retrieval.embeddings import EMBEDDING_MODELS, embedTexts

LABELS_DIR = Path("eval/labels")
POOL_ASPECTS = ["wifi", "quiet", "workspace"]
N_LISTINGS = 30
MIN_REVIEWS = 15
DEPTH = 10
SEED = 20260919


def selectListings(reviews: pd.DataFrame, n: int = N_LISTINGS, seed: int = SEED) -> list[str]:
    """
    Listings with enough reviews and at least one wifi mention, so the pool has
    evidence to find. This favors well-reviewed listings: stated as a limit.
    """
    per_listing = reviews.groupby("listing_id").agg(
        reviews=("review_id", "size"),
        wifi=("aspects", lambda col: sum("wifi" in a for a in col)),
    )
    eligible = per_listing[(per_listing["reviews"] >= MIN_REVIEWS) & (per_listing["wifi"] > 0)]
    return sorted(eligible.sample(n=min(n, len(eligible)), random_state=seed).index)


def reviewWindows(reviews: pd.DataFrame) -> pd.DataFrame:
    """One row per review window; (review_id, window) is the stable chunk key."""
    rows = [
        {"listing_id": r.listing_id, "review_id": r.review_id, "window": i, "text": piece}
        for r in reviews.itertuples()
        for i, piece in enumerate(splitText(r.comments))
    ]
    return pd.DataFrame(rows)


def rankByAspect(windows: pd.DataFrame, model_key: str, aspects: list[str], depth: int) -> dict:
    """(listing_id, aspect) -> top-`depth` window row positions, best first.
    A chunk's score for an aspect is its best similarity over the aspect's queries."""
    passages = embedTexts(windows["text"].tolist(), model_key, kind="passage")
    ranked = {}
    for aspect in aspects:
        queries = embedTexts(ASPECTS[aspect].queries, model_key, kind="query")
        scores = (passages @ queries.T).max(axis=1)
        for listing_id, rows in windows.groupby("listing_id").indices.items():
            order = rows[np.argsort(-scores[rows], kind="stable")]
            ranked[(listing_id, aspect)] = (order[:depth] if depth > 0 else order).tolist()
    return ranked


def buildPool(windows: pd.DataFrame, models: list[str], aspects: list[str], depth: int) -> list[dict]:
    pooled: dict[tuple, set] = {}
    for model_key in models:
        for (listing_id, aspect), rows in rankByAspect(windows, model_key, aspects, depth).items():
            for row in rows:
                pooled.setdefault((listing_id, aspect, row), set()).add(model_key)

    return [
        {
            "listing_id": listing_id, "aspect": aspect,
            "review_id": windows.at[row, "review_id"], "window": int(windows.at[row, "window"]),
            "pooled_by": sorted(by),
        }
        for (listing_id, aspect, row), by in sorted(pooled.items())
    ]


def pool(args) -> None:
    reviews = pd.read_parquet(processedDir(args.city) / "reviews.parquet")
    listing_ids = selectListings(reviews)
    windows = reviewWindows(reviews[reviews["listing_id"].isin(listing_ids)]).reset_index(drop=True)

    entries = buildPool(windows, sorted(EMBEDDING_MODELS), POOL_ASPECTS, args.depth)

    LABELS_DIR.mkdir(parents=True, exist_ok=True)
    (LABELS_DIR / "retrieval_pool.jsonl").write_text(
        "".join(json.dumps(e) + "\n" for e in entries), encoding="utf-8")

    # Labeling batches with text go outside the repo
    text = windows.set_index(["review_id", "window"])["text"]
    items = [{"item": i, **e, "text": text[(e["review_id"], e["window"])]} for i, e in enumerate(entries)]
    args.out.mkdir(parents=True, exist_ok=True)
    for k in range(args.batches):
        batch = items[k::args.batches]
        (args.out / f"pool_batch_{k}.jsonl").write_text(
            "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in batch), encoding="utf-8")

    judged = len(entries)
    exhaustive = len(windows) * len(POOL_ASPECTS)
    print(f"{len(listing_ids)} listings, {len(windows)} review chunks")
    print(f"pool: {judged} (chunk, aspect) pairs to label vs {exhaustive} exhaustive "
          f"({judged / exhaustive:.0%})")
    print(pd.Series([len(e["pooled_by"]) for e in entries]).value_counts().sort_index()
          .rename("pairs found by N models").to_string())


def merge(args) -> None:
    entries = [json.loads(line) for line in
               (LABELS_DIR / "retrieval_pool.jsonl").read_text(encoding="utf-8").splitlines()]
    labels = {}
    texts = {}
    for path in sorted(args.labels.glob("pool_labels_*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            label = json.loads(line)
            labels[label["item"]] = label
    for path in sorted(args.labels.glob("pool_batch_*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            item = json.loads(line)
            texts[item["item"]] = item["text"]

    gold, problems = [], []
    for item, entry in enumerate(entries):
        label = labels.get(item)
        if label is None:
            problems.append(f"item {item}: not labeled")
            continue
        if (label["review_id"], label["aspect"]) != (entry["review_id"], entry["aspect"]):
            problems.append(f"item {item}: review/aspect mismatch")
            continue
        if errors := validateLabel(label, texts[item]):
            problems.append(f"item {item}: {', '.join(errors)}")
            continue
        gold.append({
            "listing_id": entry["listing_id"], "aspect": entry["aspect"],
            "review_id": entry["review_id"], "window": entry["window"],
            "stance": label["stance"],
            "grade": label["strength"],  # 2 clear, 1 weak, 0 irrelevant (for nDCG)
            "labeler": args.labeler, "labeled_on": date.today().isoformat(),
            "guidelines_version": 1.1,
        })
    if problems:
        raise SystemExit("Invalid labels:\n" + "\n".join(problems[:20]))

    out = LABELS_DIR / "retrieval_gold.jsonl"
    out.write_text("".join(json.dumps(g) + "\n" for g in gold), encoding="utf-8")
    frame = pd.DataFrame(gold)
    print(f"{len(gold)} labels -> {out}")
    print(frame.pivot_table(index="aspect", columns="grade", values="review_id",
                            aggfunc="count", fill_value=0))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Pooling for the retrieval evaluation")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("pool")
    p.add_argument("--city", default="porto")
    p.add_argument("--out", type=Path, required=True, help="where labeling batches (with text) go")
    p.add_argument("--batches", type=int, default=4)
    p.add_argument("--depth", type=int, default=DEPTH,
                   help="top-k per model to pool; 0 = every chunk (exhaustive labels)")
    m = sub.add_parser("merge")
    m.add_argument("--city", default="porto")
    m.add_argument("--labels", type=Path, required=True)
    m.add_argument("--labeler", required=True)
    args = parser.parse_args(argv)
    pool(args) if args.command == "pool" else merge(args)


if __name__ == "__main__":
    main()
