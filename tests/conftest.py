import json
import re
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest


class FakeResponses:
    """
    Stands in for client.responses: returns the queued responses in order
    and records every call, so tests never hit the OpenAI API.
    """

    def __init__(self, outputs: list):
        self.outputs = list(outputs)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        output = self.outputs.pop(0)
        if isinstance(output, Exception):
            raise output
        return output


def fakeClient(*outputs) -> SimpleNamespace:
    return SimpleNamespace(responses=FakeResponses(outputs), close=lambda: None)


def textResponse(text: str) -> SimpleNamespace:
    # Final answer: a single "message" item, no tool calls
    return SimpleNamespace(output_text=text, output=[SimpleNamespace(type="message")])


class FakeFunctionCall(SimpleNamespace):
    def model_dump(self, exclude_none=True):
        return {
            "type": "function_call",
            "name": self.name,
            "arguments": self.arguments,
            "call_id": self.call_id,
        }


def toolCallResponse(name: str, args: dict, call_id: str = "call_1") -> SimpleNamespace:
    item = FakeFunctionCall(
        type="function_call", name=name, arguments=json.dumps(args), call_id=call_id
    )
    return SimpleNamespace(output_text="", output=[item])


@pytest.fixture(autouse=True)
def localServicesOnly(monkeypatch):
    # A local .env may turn the load-test mock on or point sessions to the real
    # DynamoDB table; tests opt in explicitly (test_mock_llm.py, a FakeTable)
    monkeypatch.delenv("MOCK_LLM", raising=False)
    monkeypatch.delenv("SESSION_TABLE", raising=False)


# ---------- artifact set built from the mini_porto fixture ----------

FIXTURES = Path(__file__).parent / "fixtures"
MINI = FIXTURES / "mini_porto"
MODEL = "multilingual-minilm"


@pytest.fixture(scope="session")
def processed(tmp_path_factory):
    # What T2 leaves behind: ingest + OSM step on the mini_porto fixture
    from ingest import osm
    from ingest.insideairbnb import ingest, readListings, updateReport, writeListings

    snapshot = json.loads((MINI / "expected.json").read_text(encoding="utf-8"))["snapshot_date"]
    out = tmp_path_factory.mktemp("processed")
    ingest(MINI / "listings.csv", MINI / "reviews.csv", out, "porto", snapshot)
    pois = pd.concat([
        osm.parsePois(json.loads((FIXTURES / f"overpass_{kind}s.json").read_text(encoding="utf-8")), kind)
        for kind in ["gym", "metro"]], ignore_index=True)
    pois.to_parquet(out / "pois.parquet", index=False)
    listings = readListings(out / "listings.parquet")
    for kind in ["gym", "metro"]:
        listings = osm.addNearestPoi(listings, pois, kind)
    writeListings(listings, out / "listings.parquet")
    updateReport(out / "ingest_report.json", "osm",
                 {"attribution": osm.ATTRIBUTION, "fetched_date": "2026-10-22"})
    return out


@pytest.fixture(scope="session")
def built(processed, tmp_path_factory):
    from ingest.build import buildArtifacts

    out = tmp_path_factory.mktemp("refined")
    manifest = buildArtifacts(processed, out, "porto", MODEL, 2)
    return out, manifest


# ---------- serving set (index set + evidence) ----------

NEGATIVE = re.compile(r"dropped|slow|unreliable|no funcionaba|noisy|loud|ruidoso|no desk|tiny shelf",
                      re.IGNORECASE)


def fakeStances(reviews: pd.DataFrame) -> pd.DataFrame:
    """
    Rule-based stand-in for the LLM judge (ingest/stance.py output format):
    every aspect mention judged by simple rules, the quote is the first sentence.
    Tests check the plumbing; the real judge is measured in eval/.
    """
    from domain.aspects import mentionedAspects

    rows = []
    for r in reviews.itertuples():
        text = r.comments
        quote = re.split(r"(?<=[.!?])\s", text)[0]
        for aspect in sorted(mentionedAspects(text)):
            if "ignore all previous" in text.lower():
                stance, strength, start = "irrelevant", 0, None
            else:
                stance = "contradicts" if NEGATIVE.search(text) else "supports"
                strength, start = 2, text.find(quote)
            rows.append({"review_id": r.review_id, "listing_id": r.listing_id, "aspect": aspect,
                         "stance": stance, "strength": strength, "quote_start": start,
                         "quote_end": start + len(quote) if start is not None else None,
                         "quote_valid": start is not None})
    return pd.DataFrame(rows)


@pytest.fixture(scope="session")
def serving(built, processed, tmp_path_factory):
    """The set the API loads, packaged like ingest/package.py does it."""
    from ingest.package import packageServingSet

    fakeStances(pd.read_parquet(processed / "reviews.parquet")).to_parquet(
        processed / "stances.parquet", index=False)
    out = tmp_path_factory.mktemp("serving")
    packageServingSet(built[0], processed, out, 3)
    return out
