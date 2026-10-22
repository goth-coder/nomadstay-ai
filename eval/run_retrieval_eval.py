"""
Retrieval evaluation on the exhaustive gold set (T5b / T8).

    python -m eval.run_retrieval_eval --city porto

For each (listing, aspect) with at least one relevant chunk, every method ranks
that listing's review chunks; relevant = gold grade >= 1 (supports or
contradicts: the judge decides the stance later, retrieval only has to find
the evidence). Methods:
  <model>          cosine similarity, best over the aspect's queries
  keywords         chunks that mention the aspect first (domain.aspects), then text order
  hybrid:<model>   keyword mentions first, the model's similarity breaks ties
Writes eval/reports/retrieval_<date>.json.
"""

import argparse
import json
import subprocess
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from constants import processedDir
from domain.aspects import ASPECTS
from eval.metrics import ndcgAtK, recallAtK, reciprocalRank
from eval.pool_retrieval import reviewWindows
from retrieval.embeddings import EMBEDDING_MODELS, embedTexts

GOLD = Path("eval/labels/retrieval_gold.jsonl")
REPORTS = Path("eval/reports")


def methodScores(windows: pd.DataFrame, aspects: list[str]) -> dict[str, dict[str, np.ndarray]]:
    """method -> aspect -> score per window (higher = earlier)."""
    mentions = {a: windows["text"].map(lambda t, a=a: bool(ASPECTS[a].keywords.search(t))).to_numpy(float)
                for a in aspects}
    methods = {"keywords": {a: mentions[a] for a in aspects}}
    for key in sorted(EMBEDDING_MODELS):
        passages = embedTexts(windows["text"].tolist(), key, kind="passage")
        semantic = {}
        for aspect in aspects:
            queries = embedTexts(ASPECTS[aspect].queries, key, kind="query")
            semantic[aspect] = (passages @ queries.T).max(axis=1)
        methods[key] = semantic
        # similarity is in [-1, 1], so +2 puts every mention ahead
        methods[f"hybrid:{key}"] = {a: semantic[a] + 2 * mentions[a] for a in aspects}
    return methods


def evaluate(windows: pd.DataFrame, gold: pd.DataFrame, methods: dict) -> pd.DataFrame:
    grades = {(g.review_id, g.window, g.aspect): g.grade for g in gold.itertuples()}
    keys = list(zip(windows["review_id"], windows["window"]))
    rows = []
    for (listing_id, aspect), group in gold.groupby(["listing_id", "aspect"]):
        positions = np.flatnonzero(windows["listing_id"].to_numpy() == listing_id)
        graded = {f"{keys[p][0]}#{keys[p][1]}": grades.get((*keys[p], aspect), 0) for p in positions}
        relevant = {k for k, g in graded.items() if g >= 1}
        if not relevant:
            continue
        for method, scores in methods.items():
            order = positions[np.argsort(-scores[aspect][positions], kind="stable")]
            ranked = [f"{keys[p][0]}#{keys[p][1]}" for p in order]
            rows.append({
                "method": method, "aspect": aspect, "listing_id": listing_id,
                "relevant": len(relevant), "chunks": len(ranked),
                "recall@4": recallAtK(ranked, relevant, 4),
                "recall@10": recallAtK(ranked, relevant, 10),
                "mrr": reciprocalRank(ranked, relevant),
                "ndcg@10": ndcgAtK(ranked, graded, 10),
            })
    return pd.DataFrame(rows)


def gitCommit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Retrieval evaluation on the gold set")
    parser.add_argument("--city", default="porto")
    args = parser.parse_args(argv)

    gold = pd.DataFrame([json.loads(line) for line in GOLD.read_text(encoding="utf-8").splitlines()])
    reviews = pd.read_parquet(processedDir(args.city) / "reviews.parquet")
    windows = reviewWindows(reviews[reviews["listing_id"].isin(set(gold["listing_id"]))])
    windows = windows.reset_index(drop=True)

    aspects = sorted(gold["aspect"].unique())
    results = evaluate(windows, gold, methodScores(windows, aspects))

    metrics = ["recall@4", "recall@10", "mrr", "ndcg@10"]
    overall = results.groupby("method")[metrics].mean().sort_values("recall@4", ascending=False)
    by_aspect = results.groupby(["aspect", "method"])[metrics].mean()
    queries = results[results["method"] == "keywords"].groupby("aspect").size()

    pd.set_option("display.width", 120)
    print(f"queries (listing x aspect with evidence): {queries.to_dict()}\n")
    print(overall.round(3).to_string(), "\n")
    print(by_aspect.round(3).to_string())

    REPORTS.mkdir(parents=True, exist_ok=True)
    out = REPORTS / f"retrieval_{date.today().isoformat()}.json"
    out.write_text(json.dumps({
        "date": date.today().isoformat(), "commit": gitCommit(), "city": args.city,
        "gold": str(GOLD), "gold_labels": len(gold), "queries": queries.to_dict(),
        "overall": overall.round(4).to_dict(orient="index"),
        "by_aspect": {f"{a}|{m}": v for (a, m), v in by_aspect.round(4).to_dict(orient="index").items()},
    }, indent=2), encoding="utf-8")
    print(f"\nreport -> {out}")


if __name__ == "__main__":
    main()
