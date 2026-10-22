"""
Stance judge evaluation against the Opus gold set (T5.5 / T8). Costs money.

    python -m eval.run_stance_eval --city porto [--model gpt-4o-mini] [--split dev|test|all]

Prompt changes are tuned on the dev half only; the test half is the number
to report. The split is a stable hash of review_id.

The judge is called exactly as in production: each review with every aspect
it mentions, 10 reviews per call; only the gold aspect is scored.
Report -> eval/reports/stance_<model>_<date>.json (metrics only);
disagreements with text -> data/eval/ (gitignored).
"""

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

from agent.judge import JUDGE_MODEL, judgeReviews
from constants import processedDir
from domain.aspects import mentionedAspects
from eval.metrics import stanceReport
from eval.run_retrieval_eval import gitCommit

GOLD = Path("eval/labels/stance_gold.jsonl")
REPORTS = Path("eval/reports")
AUDIT_DIR = Path("data/eval")
BATCH = 10
WORKERS = 6
# List prices per 1M tokens (verify on the provider's pricing page)
PRICES = {"gpt-4o-mini": (0.15, 0.60)}


def splitOf(review_id: str) -> str:
    return "dev" if int(hashlib.sha1(review_id.encode()).hexdigest(), 16) % 2 == 0 else "test"


def batches(items: list, size: int) -> list[list]:
    return [items[i:i + size] for i in range(0, len(items), size)]


def judgeAll(client: OpenAI, reviews: list[dict], model: str) -> tuple[dict, dict]:
    """reviews: {review_id, text, aspects} -> ({(review_id, aspect): label}, usage)."""
    def run(batch):
        local = [{"id": i, "text": r["text"], "aspects": r["aspects"]} for i, r in enumerate(batch)]
        labels, usage = judgeReviews(client, local, model)
        return {(batch[l["id"]]["review_id"], l["aspect"]): l for l in labels}, usage

    results, total = {}, {"input_tokens": 0, "output_tokens": 0, "calls": 0}
    with ThreadPoolExecutor(WORKERS) as pool:
        for labels, usage in pool.map(run, batches(reviews, BATCH)):
            results.update(labels)
            for k in total:
                total[k] += usage[k]
    return results, total


def cost(usage: dict, model: str) -> float | None:
    if model not in PRICES:
        return None
    price_in, price_out = PRICES[model]
    return usage["input_tokens"] / 1e6 * price_in + usage["output_tokens"] / 1e6 * price_out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate the stance judge on the gold set")
    parser.add_argument("--city", default="porto")
    parser.add_argument("--model", default=JUDGE_MODEL)
    parser.add_argument("--split", default="dev", choices=["dev", "test", "all"])
    args = parser.parse_args(argv)
    load_dotenv()

    gold = pd.DataFrame([json.loads(line) for line in GOLD.read_text(encoding="utf-8").splitlines()])
    if args.split != "all":
        gold = gold[gold["review_id"].map(splitOf) == args.split].reset_index(drop=True)
    reviews = pd.read_parquet(processedDir(args.city) / "reviews.parquet").set_index("review_id")
    items = [
        {"review_id": rid, "text": reviews.at[rid, "comments"],
         # current lexicon, plus the gold aspect so every gold pair is judged
         "aspects": sorted(mentionedAspects(reviews.at[rid, "comments"])
                           | set(gold.loc[gold["review_id"] == rid, "aspect"]))}
        for rid in gold["review_id"].unique()
    ]

    labels, usage = judgeAll(OpenAI(), items, args.model)
    gold["predicted"] = [labels[(g.review_id, g.aspect)]["stance"] or "missing" for g in gold.itertuples()]
    gold["quote_valid"] = [labels[(g.review_id, g.aspect)]["quote_valid"] for g in gold.itertuples()]

    overall = stanceReport(list(gold["stance"]), list(gold["predicted"]))
    by_aspect = {a: stanceReport(list(g["stance"]), list(g["predicted"])) for a, g in gold.groupby("aspect")}
    cited = gold[gold["predicted"].isin(["supports", "contradicts"])]
    quote_validity = float(cited["quote_valid"].mean()) if len(cited) else 1.0
    print(f"model {args.model}: accuracy {overall['accuracy']:.3f}, macro F1 {overall['macro_f1']:.3f}, "
          f"contradiction recall {overall['contradiction_recall']:.3f}, quote validity {quote_validity:.3f}")
    for stance, m in overall["per_class"].items():
        print(f"  {stance:12s} P {m['precision']:.3f} R {m['recall']:.3f} F1 {m['f1']:.3f} (n={m['support']})")
    for aspect, m in by_aspect.items():
        print(f"  {aspect:11s} acc {m['accuracy']:.3f} macro F1 {m['macro_f1']:.3f} "
              f"contradiction recall {m['contradiction_recall']:.3f}")
    print(pd.crosstab(gold["stance"], gold["predicted"], rownames=["gold"], colnames=["judge"]))
    spent = cost(usage, args.model)
    print(f"usage {usage}, cost ${spent:.4f}" if spent is not None else f"usage {usage}")

    REPORTS.mkdir(parents=True, exist_ok=True)
    out = REPORTS / f"stance_{args.model}_{args.split}_{date.today().isoformat()}.json"
    out.write_text(json.dumps({
        "date": date.today().isoformat(), "commit": gitCommit(), "model": args.model,
        "gold": str(GOLD), "split": args.split, "n": len(gold), "overall": overall, "by_aspect": by_aspect,
        "quote_validity": quote_validity, "usage": usage, "cost_usd": spent,
        "confusion": pd.crosstab(gold["stance"], gold["predicted"]).to_dict(),
    }, indent=2), encoding="utf-8")

    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    wrong = gold[gold["stance"] != gold["predicted"]].assign(text=lambda f: f["review_id"].map(reviews["comments"]))
    wrong[["review_id", "aspect", "stance", "predicted", "note", "text"]].to_csv(
        AUDIT_DIR / f"stance_disagreements_{args.model}_{args.split}.csv", index=False, encoding="utf-8-sig")
    print(f"report -> {out}; {len(wrong)} disagreements -> {AUDIT_DIR}")


if __name__ == "__main__":
    main()
