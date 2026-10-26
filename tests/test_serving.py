"""
Serving: the packaged set (ingest/package.py), loading it locally or from S3
(agent/nomad.loadStore) and the API on top of it, with a fake OpenAI client.
"""

import json
import shutil
from types import SimpleNamespace

import pytest
from conftest import MODEL, fakeClient, textResponse, toolCallResponse
from fastapi.testclient import TestClient
from test_chunk_index import FakeS3

import api.app as app_module
from agent import nomad
from agent.nomad import loadNomadStore, loadStore
from retrieval.chunk_index import MANIFEST
from retrieval.evidence import EVIDENCE_FILE, SERVING_FILES
from storage import s3

RIBEIRA = "1436961951661730861"
SEARCH = {"city": "Porto", "max_price": 100, "currency": "EUR", "must_have": [],
          "preferences": ["wifi"], "max_gym_distance_m": None}


# ---------- package ----------


def test_package_adds_evidence_to_the_index_set(serving, built):
    manifest = json.loads((serving / MANIFEST).read_text(encoding="utf-8"))

    assert manifest["version"] == 3
    assert manifest["based_on"]["version"] == 2
    assert set(manifest["files"]) == set(SERVING_FILES) - {MANIFEST}
    assert manifest["evidence"]["pairs"] > 0
    # the index is reused, not rebuilt
    assert manifest["files"]["chunks.index"] == built[1]["files"]["chunks.index"]


def test_loader_refuses_a_tampered_evidence_file(serving, tmp_path):
    copy = tmp_path / "set"
    shutil.copytree(serving, copy)
    (copy / EVIDENCE_FILE).write_bytes(b"tampered")

    with pytest.raises(ValueError, match="evidence.parquet"):
        loadNomadStore(copy, MODEL)


# ---------- loadStore: local copy or S3 ----------


@pytest.fixture
def fakeS3(monkeypatch):
    fake = FakeS3()
    monkeypatch.setattr(s3, "session", SimpleNamespace(client=lambda name: fake))
    monkeypatch.setattr(s3, "BUCKET", "test-bucket")
    return fake


def test_load_store_downloads_the_set_when_missing(serving, fakeS3, tmp_path, monkeypatch):
    s3.uploadArtifacts(serving, 3, "porto", SERVING_FILES)
    monkeypatch.setattr(nomad, "refinedDir", lambda city, version: tmp_path / f"v{version}" / city)

    store = loadStore("porto", 3, MODEL)

    assert (tmp_path / "v3" / "porto" / MANIFEST).exists()
    assert store.catalog.city == "porto" and store.index.index.ntotal > 0


def test_load_store_uses_the_local_set_without_s3(serving, fakeS3, tmp_path, monkeypatch):
    local = tmp_path / "v3" / "porto"
    shutil.copytree(serving, local)
    monkeypatch.setattr(nomad, "refinedDir", lambda city, version: local)

    loadStore("porto", 3, MODEL)

    assert fakeS3.objects == {}  # nothing downloaded


# ---------- API ----------


@pytest.fixture
def api(serving, monkeypatch):
    """The FastAPI app over the mini_porto serving set, OpenAI replaced by a queue."""
    monkeypatch.setenv("API_KEY", "test-key")
    store = loadNomadStore(serving, MODEL)
    store.index = None  # no paraphrase pass: the queue only feeds the agent loop
    monkeypatch.setattr(app_module, "loadStore", lambda: store)
    openai = fakeClient()
    queue = openai.responses.outputs  # tests append the responses each call will get
    monkeypatch.setattr(app_module, "OpenAI", lambda **kwargs: openai)
    with TestClient(app_module.app) as client:
        yield client, queue, store


def ribeiraQuote(store) -> str:
    rows = store.evidence.table
    return rows[(rows["listing_id"] == RIBEIRA) & (rows["aspect"] == "wifi")]["quote"].dropna().iloc[0]


def test_health_reports_the_dataset(api):
    client, _, _ = api

    health = client.get("/health").json()

    assert health["city"] == "porto" and health["listings"] > 0 and health["snapshot_date"]


def test_run_requires_the_api_key(api):
    client, _, _ = api

    assert client.post("/agent/run", json={"message": "hi"}).status_code == 401


def test_run_searches_and_answers_with_a_verified_quote(api):
    client, queue, store = api
    quote = ribeiraQuote(store)
    queue += [toolCallResponse("search_stays", SEARCH),
              textResponse(f'Ribeira Nomad Loft: a guest wrote "{quote}".')]

    body = client.post("/agent/run", headers={"X-API-Key": "test-key"},
                       json={"message": "Porto under 100 with good wifi"}).json()

    assert body["answer"].startswith("Ribeira Nomad Loft")
    assert "warnings" not in body


def test_run_flags_an_invented_quote(api):
    client, queue, _ = api
    queue += [toolCallResponse("search_stays", SEARCH),
              textResponse('Guests say "the wifi reached 900 Mbps every day".')]

    body = client.post("/agent/run", headers={"X-API-Key": "test-key"},
                       json={"message": "Porto under 100 with good wifi"}).json()

    assert body["warnings"]["unverified_quotes"] == ["the wifi reached 900 Mbps every day"]


def test_follow_up_uses_stay_details_in_the_same_session(api):
    client, queue, store = api
    quote = ribeiraQuote(store)
    queue += [
        toolCallResponse("search_stays", SEARCH),
        textResponse("Ribeira Nomad Loft fits."),
        toolCallResponse("get_stay_details", {"listing_id": RIBEIRA, "aspects": ["wifi"]}, "call_2"),
        textResponse(f'A guest wrote "{quote}".'),
    ]
    headers = {"X-API-Key": "test-key"}

    first = client.post("/agent/run", headers=headers, json={"message": "Porto under 100, wifi"}).json()
    second = client.post("/agent/run", headers=headers,
                         json={"message": "Tell me more about it", "session_id": first["session_id"]}).json()

    assert "warnings" not in second  # the quote came from get_stay_details
    history = client.get(f"/agent/sessions/{first['session_id']}", headers=headers).json()["history"]
    assert [i["name"] for i in history if i.get("type") == "function_call"] == [
        "search_stays", "get_stay_details"]
