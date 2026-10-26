"""
End-to-end agent evaluation over the scenarios in eval/queries.yaml (T8). Costs money.

    python -m eval.run_agent_eval --label baseline [--only id1,id2]

Each scenario runs through the real agent loop (gpt-4o-mini + tools) on the
local serving set. Checks done in code, per scenario:
  intent       search_stays arguments vs the expected intent
  status       search_stays status vs `expect`
  quotes       quoted passages not found in any tool output
  price_date   a returned stay's price in the answer, with no snapshot date in it
  speed        a Wi-Fi speed called verified (it is the host's own speed test)
  attribution  the data attribution line is present
`must_not` needs judgment: answers are saved for review, not scored here.

Report -> eval/reports/agent_<label>_<date>.json (metrics + checks);
answers -> data/eval/agent_answers_<label>.jsonl (gitignored).
"""

import argparse
import contextvars
import json
import re
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

import yaml
from dotenv import load_dotenv
from openai import OpenAI

from agent.agent import runAgent
from agent.nomad import loadStore, unverifiedQuotes
from eval.run_retrieval_eval import gitCommit
from observability.context import startRequestStats

SCENARIOS = Path("eval/queries.yaml")
REPORTS = Path("eval/reports")
ANSWERS_DIR = Path("data/eval")
WORKERS = 5
STATUS = {"results": "ok"}

NUMBER = re.compile(r"\d+(?:[.,]\d+)?")
# "verified"/"verificado"/"vérifié" within a few characters of a speed, on one line
SPEED_VERIFIED = re.compile(r"v[ée]rifi\w*[^\n]{0,30}\d+\s*mbps|\d+\s*mbps[^\n]{0,30}v[ée]rifi",
                            re.IGNORECASE)
# ...unless the same line says the speed is the host's test ("Verified (216 Mbps per host's speed test)")
HOST_TEST = re.compile(r"host|anfitri|speed test|prueba|teste de velocidade", re.IGNORECASE)
MONTHS = r"(?:sep(?:t(?:ember)?)?|set(?:embro)?|sept(?:embre)?|septiembre)"


def snapshotPattern(snapshot: str) -> re.Pattern:
    """The snapshot date as people write it: 2026-09-19, 19/09/2026, 19 Sep 2026, September 19..."""
    year, month, day = snapshot.split("-")
    day = str(int(day))
    return re.compile(rf"({snapshot}|{day}/{month}/{year}|{day}\.{month}\.{year}|"
                      rf"{day}(?:th)?\s+(?:de\s+)?{MONTHS}|{MONTHS}\.?\s+{day}\b)", re.IGNORECASE)


def toolCalls(context: list) -> list[tuple[str, dict, dict | None]]:
    """(name, arguments, output) of every tool call in the conversation."""
    outputs = {item["call_id"]: json.loads(item["output"]) for item in context
               if item.get("type") == "function_call_output"}
    return [(item["name"], json.loads(item["arguments"]), outputs.get(item["call_id"]))
            for item in context if item.get("type") == "function_call"]


def intentChecks(args: dict | None, intent: dict, expect: str) -> dict:
    if args is None:
        return {"called": False}
    # must-have vs preference is a judgment call ("with a gym"): what counts is
    # that every expected need reached the search, as either
    expected = set(intent["must_have"]) | set(intent["preferences"])
    asked = set(args["must_have"]) | set(args["preferences"])
    checks = {
        "called": True,
        "city": (args["city"] or "").strip().lower() in {intent["city"], "oporto"}
                if intent["city"] == "porto" else (args["city"] or "").strip().lower() == intent["city"],
        "must_have_exact": set(args["must_have"]) == set(intent["must_have"]),
        "needs_recall": len(expected & asked) / len(expected) if expected else 1.0,
        "needs_added": sorted(asked - expected),  # needs the user never mentioned
    }
    if expect == "unsupported_currency":
        checks["currency"] = (args["currency"] or "").upper() not in {"", "EUR"}
    else:
        checks["max_price"] = args["max_price"] == intent["max_price"]
    return checks


def statesListingPrice(answer: str, outputs: list[dict]) -> bool:
    """A price of a returned stay appears in the answer (the user's own budget doesn't count)."""
    prices = {round(float(r["price_per_night"]), 2) for out in outputs for r in out.get("results", [])
              if r.get("price_per_night") is not None}
    stated = {round(float(n.replace(",", ".")), 2) for n in NUMBER.findall(answer)}
    return bool(prices & stated)


def answerChecks(answer: str, outputs: list[dict], snapshot: str, status: str | None) -> dict:
    states_price = statesListingPrice(answer, outputs)
    return {
        "unverified_quotes": unverifiedQuotes(answer, outputs),
        "price_without_date": states_price and not snapshotPattern(snapshot).search(answer),
        "speed_called_verified": any(SPEED_VERIFIED.search(line) and not HOST_TEST.search(line)
                                     for line in answer.splitlines()),
        # owed when stays are shown; a refusal shows no data
        "attribution": "Inside Airbnb" in answer if status == "ok" else None,
    }


def runScenario(scenario: dict, client: OpenAI, store, snapshot: str) -> dict:
    stats = startRequestStats()  # runs in its own context copy (see main)
    context = deque([{"role": "user", "content": scenario["request"]}])
    try:
        answer = runAgent(context, client, store, 5)
        error = None
    except Exception as exc:  # one failed scenario is a result, not the end of the run
        answer, error = "", f"{type(exc).__name__}: {exc}"

    calls = toolCalls(list(context))
    search = next(((args, out) for name, args, out in calls if name == "search_stays"), (None, None))
    outputs = [out for _, _, out in calls if out]
    status = search[1]["status"] if search[1] else None
    expect = scenario["expect"]
    return {
        "id": scenario["id"], "request": scenario["request"], "expect": expect,
        "status": status, "status_ok": status == STATUS.get(expect, expect),
        "intent": intentChecks(search[0], scenario["intent"], expect),
        "search_args": search[0],
        **answerChecks(answer, outputs, snapshot, status),
        "tools": [name for name, _, _ in calls],
        "llm_calls": stats.llm_calls, "cost_usd": round(stats.cost_usd, 6),
        "error": error, "must_not": scenario.get("must_not", []), "notes": scenario.get("notes"),
        "answer": answer, "tool_outputs": outputs,
    }


def summary(rows: list[dict]) -> dict:
    n = len(rows)
    called = [r["intent"] for r in rows if r["intent"]["called"]]

    def share(values):
        values = list(values)
        return round(sum(values) / len(values), 3) if values else None

    return {
        "scenarios": n,
        "errors": sum(r["error"] is not None for r in rows),
        "search_called": share(r["intent"]["called"] for r in rows),
        "status_accuracy": share(r["status_ok"] for r in rows),
        "intent_city": share(c["city"] for c in called),
        "intent_max_price": share(c["max_price"] for c in called if "max_price" in c),
        "intent_needs_recall": share(c["needs_recall"] for c in called),
        "intent_must_have_exact": share(c["must_have_exact"] for c in called),
        "searches_with_added_needs": sum(bool(c["needs_added"]) for c in called),
        "answers_with_unverified_quotes": sum(bool(r["unverified_quotes"]) for r in rows),
        "answers_with_price_without_date": sum(r["price_without_date"] for r in rows),
        "answers_with_speed_called_verified": sum(r["speed_called_verified"] for r in rows),
        "attribution": share(r["attribution"] for r in rows if r["attribution"] is not None),
        "cost_usd": round(sum(r["cost_usd"] for r in rows), 4),
        "llm_calls": sum(r["llm_calls"] for r in rows),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True, help="Name of this run (e.g. baseline, prompt-v2).")
    parser.add_argument("--only", help="Comma-separated scenario ids.")
    args = parser.parse_args()

    load_dotenv(".env")
    spec = yaml.safe_load(SCENARIOS.read_text(encoding="utf-8"))
    scenarios = spec["scenarios"]
    if args.only:
        wanted = set(args.only.split(","))
        scenarios = [s for s in scenarios if s["id"] in wanted]

    store = loadStore()
    client = OpenAI()
    snapshot = store.catalog.snapshot_date

    def run(scenario):  # each scenario gets its own context, so its own request stats
        return contextvars.copy_context().run(runScenario, scenario, client, store, snapshot)

    with ThreadPoolExecutor(WORKERS) as pool:
        rows = list(pool.map(run, scenarios))

    ANSWERS_DIR.mkdir(parents=True, exist_ok=True)
    answers = ANSWERS_DIR / f"agent_answers_{args.label}.jsonl"
    answers.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")

    report = {
        "label": args.label, "date": date.today().isoformat(), "commit": gitCommit(),
        "model": "gpt-4o-mini", "snapshot": snapshot, "summary": summary(rows),
        "checks": [{k: v for k, v in r.items() if k not in ("answer", "tool_outputs", "request", "must_not", "notes")}
                   for r in rows],
    }
    REPORTS.mkdir(parents=True, exist_ok=True)
    out = REPORTS / f"agent_{args.label}_{date.today().isoformat()}.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    print(f"report -> {out}\nanswers -> {answers}")


if __name__ == "__main__":
    main()
