"""
Evaluation metrics (pure functions, no I/O).

Retrieval:      recallAtK, ndcgAtK, reciprocalRank
Stance (judge): stanceReport (per-class P/R/F1, macro F1, contradiction recall)
Recommendation: constraintViolations, citationValidity, percentile
"""

import math
from collections.abc import Sequence

STANCES = ("supports", "contradicts", "irrelevant")


# ---------- retrieval ----------


def recallAtK(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    return len(set(retrieved[:k]) & relevant) / len(relevant)


def ndcgAtK(retrieved: Sequence[str], grades: dict[str, float], k: int) -> float:
    """grades: id -> relevance (e.g. 2 = clear evidence, 1 = partial, 0 = none)."""
    dcg = sum(grades.get(doc, 0) / math.log2(i + 2) for i, doc in enumerate(retrieved[:k]))
    ideal = sorted(grades.values(), reverse=True)[:k]
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def reciprocalRank(retrieved: Sequence[str], relevant: set[str]) -> float:
    for i, doc in enumerate(retrieved):
        if doc in relevant:
            return 1 / (i + 1)
    return 0.0


# ---------- stance ----------


def stanceReport(gold: Sequence[str], predicted: Sequence[str]) -> dict:
    """
    Per-class precision/recall/F1 and macro F1 over the stances.
    contradiction_recall is reported apart: missing a complaint is the
    costliest error (we would recommend a place guests say fails).
    """
    if len(gold) != len(predicted):
        raise ValueError("gold and predicted must have the same length")

    per_class = {}
    for stance in STANCES:
        tp = sum(g == p == stance for g, p in zip(gold, predicted))
        predicted_n = sum(p == stance for p in predicted)
        gold_n = sum(g == stance for g in gold)
        precision = tp / predicted_n if predicted_n else 0.0
        recall = tp / gold_n if gold_n else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[stance] = {"precision": precision, "recall": recall, "f1": f1, "support": gold_n}

    present = [s for s in STANCES if per_class[s]["support"]]
    return {
        "per_class": per_class,
        "macro_f1": sum(per_class[s]["f1"] for s in present) / len(present) if present else 0.0,
        "contradiction_recall": per_class["contradicts"]["recall"],
        "accuracy": sum(g == p for g, p in zip(gold, predicted)) / len(gold) if gold else 0.0,
    }


# ---------- recommendation ----------


def constraintViolations(results: Sequence[dict], constraints: dict) -> list[str]:
    """
    Hard constraints that recommended listings break. Must always be empty.
    results: dicts with city, price, claims (aspect -> amenity).
    constraints: city, max_price, must_have (list of aspects), all optional.
    """
    violations = []
    for r in results:
        rid = r.get("listing_id", "?")
        if constraints.get("city") and r.get("city") != constraints["city"]:
            violations.append(f"{rid}: city {r.get('city')!r}")
        max_price = constraints.get("max_price")
        if max_price is not None and (r.get("price") is None or r["price"] > max_price):
            violations.append(f"{rid}: price {r.get('price')} > {max_price}")
        for aspect in constraints.get("must_have") or []:
            if aspect not in (r.get("claims") or {}):
                violations.append(f"{rid}: missing must-have {aspect}")
    return violations


def citationValidity(citations: Sequence[dict], sources: dict[str, str]) -> float:
    """
    Share of citations whose quote appears verbatim in the cited source.
    citations: {"ref": review_id, "quote": str}; sources: review_id -> text.
    An unknown ref or a paraphrased quote counts as invalid.
    """
    if not citations:
        return 1.0
    valid = sum(
        1 for c in citations
        if c.get("quote") and c["quote"].strip() in sources.get(c.get("ref"), "")
    )
    return valid / len(citations)


def percentile(values: Sequence[float], q: float) -> float:
    """Nearest-rank percentile, q in [0, 100]."""
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(math.ceil(q / 100 * len(ordered)), 1)
    return ordered[rank - 1]
