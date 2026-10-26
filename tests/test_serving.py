"""
Serving: the packaged set (ingest/package.py), loading it locally or from S3
(agent/nomad.loadStore) and the API on top of it, with a fake OpenAI client.
"""

import json
import logging
import shutil
from types import SimpleNamespace

import pytest
from conftest import MODEL, fakeClient, textResponse, toolCallResponse
from fastapi.testclient import TestClient
from test_chunk_index import FakeS3

import api.app as app_module
from agent import nomad
from agent.nomad import loadNomadStore, loadStore
from observability.context import currentRequestId
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


class RequestIdRecorder(logging.Handler):
    """Reads the request id when the line is logged, as the JSON formatter does."""

    def __init__(self):
        super().__init__()
        self.ids = []

    def emit(self, record):
        self.ids.append(currentRequestId())


def test_logs_inside_the_agent_carry_the_request_id_of_the_response(api):
    client, queue, store = api
    recorder = RequestIdRecorder()
    logger = logging.getLogger("agent.agent")
    level = logger.level
    logger.addHandler(recorder)
    logger.setLevel(logging.INFO)
    queue += [toolCallResponse("search_stays", SEARCH), textResponse("Ribeira Nomad Loft.")]

    try:
        response = client.post("/agent/run", headers={"X-API-Key": "test-key"},
                               json={"message": "Porto under 100 with good wifi"})
    finally:
        logger.removeHandler(recorder)
        logger.setLevel(level)

    assert recorder.ids  # the tool_call line
    assert set(recorder.ids) == {response.headers["X-Request-ID"]}


def test_every_model_call_of_a_run_is_logged(api, caplog):
    client, queue, _ = api
    caplog.set_level(logging.INFO, logger="observability.llm")
    queue += [toolCallResponse("search_stays", SEARCH), textResponse("Ribeira Nomad Loft.")]

    client.post("/agent/run", headers={"X-API-Key": "test-key"},
                json={"message": "Porto under 100 with good wifi"})

    events = [r.fields for r in caplog.records if r.getMessage() == "llm_call"]
    assert [e["purpose"] for e in events] == ["agent", "agent"]  # tool call, then the answer
    assert all(e["status"] == "ok" for e in events)


# ---------- one "request" line per request (RED) ----------


@pytest.fixture
def requestLines(caplog):
    caplog.set_level(logging.DEBUG, logger="observability.access_log")
    return lambda: [(r.levelno, r.fields) for r in caplog.records if r.getMessage() == "request"]


def test_run_logs_one_request_line_with_its_model_usage(api, requestLines):
    client, queue, _ = api
    queue += [toolCallResponse("search_stays", SEARCH), textResponse("Ribeira Nomad Loft.")]

    client.post("/agent/run", headers={"X-API-Key": "test-key"},
                json={"message": "Porto under 100 with good wifi"})

    [(level, line)] = requestLines()
    assert level == logging.INFO
    assert (line["method"], line["route"], line["status"]) == ("POST", "/agent/run", 200)
    assert line["llm_calls"] == 2 and line["duration_ms"] >= line["llm_ms"] >= 0


def test_rejected_request_is_logged_with_its_status(api, requestLines):
    client, _, _ = api

    client.post("/agent/run", json={"message": "hi"})

    assert requestLines()[0][1]["status"] == 401


def test_route_is_the_template_not_the_real_path(api, requestLines):
    client, _, _ = api

    client.get("/agent/sessions/abc123", headers={"X-API-Key": "test-key"})
    client.get("/wp-login.php")

    routes = [line["route"] for _, line in requestLines()]
    assert routes == ["/agent/sessions/{session_id}", "unmatched"]


def test_health_checks_are_logged_at_debug(api, requestLines):
    client, _, _ = api

    client.get("/health")

    assert requestLines()[0][0] == logging.DEBUG


def test_unhandled_error_is_logged_as_500(api, requestLines, monkeypatch):
    client, _, _ = api

    def explode(session_id):
        raise RuntimeError("store down")

    monkeypatch.setattr(client.app.state.sessions, "get", explode)
    with pytest.raises(RuntimeError):
        client.get("/agent/sessions/abc123", headers={"X-API-Key": "test-key"})

    assert requestLines()[0][1]["status"] == 500
