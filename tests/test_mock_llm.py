"""
Load-test mock (loadtest/mock_llm.py): with MOCK_LLM=1 the API answers end to
end through the real tools and filters, with no OpenAI call and no cost.
"""

import pytest
from conftest import MODEL
from fastapi.testclient import TestClient

import api.app as app_module
from agent.nomad import loadNomadStore


@pytest.fixture
def mockApi(serving, monkeypatch):
    monkeypatch.setenv("API_KEY", "test-key")
    monkeypatch.setenv("MOCK_LLM", "1")
    monkeypatch.setenv("MOCK_LLM_MIN_S", "0")
    monkeypatch.setenv("MOCK_LLM_MAX_S", "0")
    store = loadNomadStore(serving, MODEL)
    monkeypatch.setattr(app_module, "loadStore", lambda: store)
    monkeypatch.setattr(app_module, "OpenAI", lambda **kwargs: pytest.fail("OpenAI must not be created"))
    with TestClient(app_module.app) as client:
        yield client


def test_health_says_the_llm_is_a_mock(mockApi):
    assert mockApi.get("/health").json()["llm"] == "mock"


def test_question_and_follow_up_run_through_the_real_tools(mockApi):
    headers = {"X-API-Key": "test-key"}

    first = mockApi.post("/agent/run", headers=headers, json={"message": "Porto, quiet"}).json()
    second = mockApi.post("/agent/run", headers=headers,
                          json={"message": "and the second one?", "session_id": first["session_id"]}).json()
    history = mockApi.get(f"/agent/sessions/{first['session_id']}", headers=headers).json()["history"]

    assert "mock answer" in first["answer"] and "mock answer" in second["answer"]
    calls = [item["name"] for item in history if item.get("type") == "function_call"]
    assert calls == ["search_stays", "search_stays"]  # each turn calls the tool, then answers


def test_mock_judge_labels_every_pair_so_nothing_is_retried():
    from agent.judge import judgeReviews
    from loadtest.mock_llm import MockOpenAI

    client = MockOpenAI(0, 0)
    reviews = [{"id": 0, "text": "Fast wifi.", "aspects": ["wifi", "quiet"]},
               {"id": 1, "text": "Calm street.", "aspects": ["quiet"]}]

    labels, usage = judgeReviews(client, reviews)

    assert usage["calls"] == 1  # a retry would mean the mock looks like a judge skipping pairs
    assert [label["stance"] for label in labels] == ["irrelevant"] * 3
