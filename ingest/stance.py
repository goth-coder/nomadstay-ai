"""
Precomputed stance per (review, aspect): T5.5. Runs once per snapshot, so the
online search needs no judge call for reviews that mention an aspect.

    python -m ingest.stance submit  --city porto    # Batch API (50% cheaper, async, <24h)
    python -m ingest.stance collect --city porto    # when the batch is done
    python -m ingest.stance run     --city porto    # direct calls (small cities, tests)

Pairs = every sampled review x every aspect its text mentions (current lexicon).
Writes stances.parquet next to reviews.parquet: review_id, listing_id, aspect,
stance, strength, quote_start, quote_end, quote_valid, judge_model,
guidelines_version. Pairs the judge skipped in the batch are retried directly.
"""

import argparse
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import yaml
from dotenv import load_dotenv
from openai import OpenAI

from agent.judge import (
    GUIDELINES,
    JUDGE_MODEL,
    collectLabels,
    judgeReviews,
    missingReviews,
    requestParams,
    unlabeled,
)
from constants import processedDir
from domain.aspects import mentionedAspects
from ingest.insideairbnb import updateReport

logger = logging.getLogger(__name__)

GROUP_SIZE = 10
WORKERS = 6
STATE_FILE = "stance_batch.json"
OUT_FILE = "stances.parquet"
# List prices per 1M tokens (verify); the Batch API charges half
PRICES = {"gpt-4o-mini": (0.15, 0.60)}


def reviewGroups(reviews: pd.DataFrame, size: int = GROUP_SIZE) -> list[list[dict]]:
    """Reviews that mention an aspect, in groups of `size`, local ids 0..size-1."""
    items = []
    for r in reviews.sort_values(["listing_id", "review_id"]).itertuples():
        aspects = sorted(mentionedAspects(r.comments))
        if aspects:
            items.append({"review_id": r.review_id, "listing_id": r.listing_id,
                          "text": r.comments, "aspects": aspects})
    return [
        [{**item, "id": i} for i, item in enumerate(items[start:start + size])]
        for start in range(0, len(items), size)
    ]


def batchLines(groups: list[list[dict]], model: str) -> list[str]:
    return [
        json.dumps({"custom_id": f"g{n}", "method": "POST", "url": "/v1/responses",
                    "body": requestParams(group, model)}, ensure_ascii=False)
        for n, group in enumerate(groups)
    ]


def responseText(body: dict) -> str:
    """Output text of a Responses API body as returned by the Batch API."""
    return "".join(
        part["text"]
        for item in body.get("output", []) if item.get("type") == "message"
        for part in item.get("content", []) if part.get("type") == "output_text"
    )


def stanceRows(group: list[dict], labels: dict, model: str, version) -> list[dict]:
    rows = []
    for r in group:
        for aspect in r["aspects"]:
            label = labels.get((r["id"], aspect)) or unlabeled(r["id"], aspect)
            start = label["quote_start"]
            rows.append({
                "review_id": r["review_id"], "listing_id": r["listing_id"], "aspect": aspect,
                "stance": label["stance"], "strength": label["strength"],
                "quote_start": start,
                "quote_end": start + len(label["quote"]) if start is not None else None,
                "quote_valid": label["quote_valid"],
                "judge_model": model, "guidelines_version": version,
            })
    return rows


def guidelinesVersion():
    return yaml.safe_load(GUIDELINES.read_text(encoding="utf-8"))["version"]


def writeResults(out_dir: Path, rows: list[dict], usage: dict, model: str, mode: str) -> None:
    frame = pd.DataFrame(rows)
    frame["quote_start"] = frame["quote_start"].astype("Int64")
    frame["quote_end"] = frame["quote_end"].astype("Int64")
    frame["strength"] = frame["strength"].astype("Int64")
    frame.to_parquet(out_dir / OUT_FILE, index=False)

    price_in, price_out = PRICES.get(model, (None, None))
    discount = 0.5 if mode == "batch" else 1.0
    cost = (usage["input_tokens"] / 1e6 * price_in + usage["output_tokens"] / 1e6 * price_out) * discount \
        if price_in is not None else None
    report = {
        "model": model, "mode": mode, "guidelines_version": guidelinesVersion(),
        "pairs": len(frame), "reviews": int(frame["review_id"].nunique()),
        "by_aspect_stance": {f"{a}|{s}": int(n) for (a, s), n in
                             frame.groupby(["aspect", frame["stance"].fillna("missing")]).size().items()},
        "missing": int(frame["stance"].isna().sum()),
        "quote_validity": float(frame.loc[frame["stance"].isin(["supports", "contradicts"]),
                                          "quote_valid"].mean()),
        "usage": usage, "cost_usd_estimate": round(cost, 4) if cost is not None else None,
        "finished_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    updateReport(out_dir / "ingest_report.json", "stance", report)
    print(json.dumps(report, indent=2))


def retrySkipped(client: OpenAI, groups: list[list[dict]], labels_by_group: dict,
                 model: str, usage: dict) -> None:
    """Direct calls for pairs the batch skipped or failed."""
    for n, group in enumerate(groups):
        labels = labels_by_group.setdefault(n, {})
        pending = missingReviews(group, labels)
        if not pending:
            continue
        retried, extra = judgeReviews(client, pending, model)
        for label in retried:
            if label["stance"] is not None:
                labels[(label["id"], label["aspect"])] = label
        for k in ("input_tokens", "output_tokens"):
            usage[k] += extra[k]


def submit(args, client: OpenAI) -> None:
    out_dir = processedDir(args.city)
    groups = reviewGroups(pd.read_parquet(out_dir / "reviews.parquet"))
    requests = out_dir / "stance_requests.jsonl"
    requests.write_text("\n".join(batchLines(groups, args.model)) + "\n", encoding="utf-8")

    with open(requests, "rb") as f:
        uploaded = client.files.create(file=f, purpose="batch")
    batch = client.batches.create(input_file_id=uploaded.id, endpoint="/v1/responses",
                                  completion_window="24h")
    state = {"batch_id": batch.id, "input_file_id": uploaded.id, "model": args.model,
             "groups": [[{k: r[k] for k in ("id", "review_id", "listing_id", "aspects")} for r in g]
                        for g in groups],
             "submitted_at": datetime.now(UTC).isoformat(timespec="seconds")}
    (out_dir / STATE_FILE).write_text(json.dumps(state), encoding="utf-8")
    print(f"batch {batch.id}: {len(groups)} requests, "
          f"{sum(len(r['aspects']) for g in groups for r in g)} pairs")


def collect(args, client: OpenAI) -> None:
    out_dir = processedDir(args.city)
    state = json.loads((out_dir / STATE_FILE).read_text(encoding="utf-8"))
    batch = client.batches.retrieve(state["batch_id"])
    if batch.status != "completed":
        print(f"batch {batch.id}: {batch.status} {batch.request_counts}")
        return

    texts = pd.read_parquet(out_dir / "reviews.parquet").set_index("review_id")["comments"]
    groups = [[{**r, "text": texts[r["review_id"]]} for r in g] for g in state["groups"]]
    labels_by_group, usage = {}, {"input_tokens": 0, "output_tokens": 0, "calls": 0}
    for line in client.files.content(batch.output_file_id).text.splitlines():
        result = json.loads(line)
        n = int(result["custom_id"][1:])
        response = result.get("response") or {}
        if response.get("status_code") != 200:
            continue  # retried below
        body = response["body"]
        usage["calls"] += 1
        usage["input_tokens"] += body.get("usage", {}).get("input_tokens", 0)
        usage["output_tokens"] += body.get("usage", {}).get("output_tokens", 0)
        try:
            collectLabels(responseText(body), groups[n], labels_by_group.setdefault(n, {}))
        except (json.JSONDecodeError, KeyError):
            logger.warning("unreadable output for group %s, retrying directly", n)

    retrySkipped(client, groups, labels_by_group, state["model"], usage)
    version = guidelinesVersion()
    rows = [row for n, g in enumerate(groups)
            for row in stanceRows(g, labels_by_group.get(n, {}), state["model"], version)]
    writeResults(out_dir, rows, usage, state["model"], "batch")


def run(args, client: OpenAI) -> None:
    out_dir = processedDir(args.city)
    groups = reviewGroups(pd.read_parquet(out_dir / "reviews.parquet"))
    version = guidelinesVersion()

    def judge(group):
        labels, usage = judgeReviews(client, group, args.model)
        return {(l["id"], l["aspect"]): l for l in labels}, usage

    rows, total = [], {"input_tokens": 0, "output_tokens": 0, "calls": 0}
    with ThreadPoolExecutor(WORKERS) as pool:
        for group, (labels, usage) in zip(groups, pool.map(judge, groups)):
            rows += stanceRows(group, labels, args.model, version)
            for k in total:
                total[k] += usage[k]
    writeResults(out_dir, rows, total, args.model, "direct")


def evidenceTable(stances: pd.DataFrame, reviews: pd.DataFrame) -> pd.DataFrame:
    """
    Serving table for the search: one row per judged (review, aspect), with the
    review date and the quote text cut from the review by its offsets (so it
    is verbatim by construction). Irrelevant rows are kept without a quote:
    they tell the search which reviews were already judged.
    """
    texts = reviews.set_index("review_id")
    out = stances[["listing_id", "review_id", "aspect", "stance", "strength"]].copy()
    out["date"] = out["review_id"].map(texts["date"])
    quotes = []
    for row in stances.itertuples():
        valid = row.quote_valid and not pd.isna(row.quote_start)
        quotes.append(texts.at[row.review_id, "comments"][int(row.quote_start):int(row.quote_end)]
                      if valid else None)
    out["quote"] = quotes
    return out.reset_index(drop=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Precompute review stances per aspect")
    parser.add_argument("command", choices=["submit", "collect", "run"])
    parser.add_argument("--city", default="porto")
    parser.add_argument("--model", default=JUDGE_MODEL)
    args = parser.parse_args(argv)
    load_dotenv()
    client = OpenAI(max_retries=5)
    {"submit": submit, "collect": collect, "run": run}[args.command](args, client)


if __name__ == "__main__":
    main()
