"""
Merges labeled batches into the stance gold set (T5).

    python -m eval.build_gold --labels <dir with labels_*.jsonl> --labeler "claude-opus-5-5 (Claude Code session)"

Every quote is checked again against the review text in reviews.parquet.
The gold file keeps quote offsets, not the quote text, so the repo does not
redistribute guest comments. A sample for human audit (with text) goes to
data/eval/, which is gitignored.
"""

import argparse
import json
from datetime import date
from pathlib import Path

import pandas as pd

from constants import processedDir
from eval.metrics import STANCES

LABELS_DIR = Path("eval/labels")
AUDIT_DIR = Path("data/eval")
AUDIT_SIZE = 100
SEED = 20260919


def validateLabel(label: dict, text: str) -> list[str]:
    errors = []
    if label["stance"] not in STANCES:
        errors.append(f"stance {label['stance']!r}")
    expected_strength = {0} if label["stance"] == "irrelevant" else {1, 2}
    if label["strength"] not in expected_strength:
        errors.append(f"strength {label['strength']} for {label['stance']}")
    quote = label["quote"]
    if label["stance"] == "irrelevant" and quote:
        errors.append("irrelevant with a quote")
    if label["stance"] != "irrelevant" and not quote:
        errors.append("missing quote")
    if quote and quote not in text:
        errors.append("quote is not verbatim")
    return errors


def goldRow(label: dict, candidate: dict, text: str, labeler: str) -> dict:
    start = text.find(label["quote"]) if label["quote"] else -1
    return {
        "review_id": candidate["review_id"],
        "listing_id": candidate["listing_id"],
        "aspect": candidate["aspect"],
        "stance": label["stance"],
        "strength": label["strength"],
        "quote_start": start if start >= 0 else None,
        "quote_end": start + len(label["quote"]) if start >= 0 else None,
        "note": label.get("note", ""),
        "labeler": labeler,
        "labeled_on": date.today().isoformat(),
        "guidelines_version": 1.1,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build the stance gold set")
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--labeler", required=True)
    parser.add_argument("--city", default="porto")
    args = parser.parse_args(argv)

    lines = (LABELS_DIR / "stance_candidates.jsonl").read_text(encoding="utf-8").splitlines()
    candidates = [json.loads(line) for line in lines]
    labels = {}
    for path in sorted(args.labels.glob("labels_*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            label = json.loads(line)
            labels[label["item"]] = label

    texts = pd.read_parquet(processedDir(args.city) / "reviews.parquet").set_index("review_id")["comments"]

    gold, problems = [], []
    for item, candidate in enumerate(candidates):
        label = labels.get(item)
        if label is None:
            problems.append(f"item {item}: not labeled")
            continue
        if (label["review_id"], label["aspect"]) != (candidate["review_id"], candidate["aspect"]):
            problems.append(f"item {item}: review/aspect mismatch")
            continue
        text = texts[candidate["review_id"]]
        if errors := validateLabel(label, text):
            problems.append(f"item {item}: {', '.join(errors)}")
            continue
        gold.append(goldRow(label, candidate, text, args.labeler))

    if problems:
        raise SystemExit("Invalid labels:\n" + "\n".join(problems[:20]))

    out = LABELS_DIR / "stance_gold.jsonl"
    out.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in gold), encoding="utf-8")

    # Human audit sample, stratified by aspect, with text (gitignored)
    frame = pd.DataFrame(gold)
    frame["text"] = frame["review_id"].map(texts)
    audit = frame.groupby("aspect").sample(frac=AUDIT_SIZE / len(frame), random_state=SEED)
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    audit_path = AUDIT_DIR / "stance_audit.csv"
    audit[["review_id", "aspect", "stance", "strength", "note", "text"]].assign(
        your_stance="", comment=""
    ).to_csv(audit_path, index=False, encoding="utf-8-sig")

    print(f"{len(gold)} gold labels -> {out}")
    print(f"{len(audit)} rows for human audit -> {audit_path}")
    print(frame.pivot_table(index="aspect", columns="stance", values="review_id", aggfunc="count", fill_value=0))


if __name__ == "__main__":
    main()
