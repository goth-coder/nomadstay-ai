"""
Characterization tests: freeze how the v1 tools (data.HOTELS) behave today,
so the NomadStay changes can't silently break the existing contracts.
"""

import json

import pytest

from agent import dispatcher, runAgent
from conftest import fakeClient, judgeResponse, textResponse, toolCallResponse
from data import HOTELS
from rag import similaritySearch
from tools import filter_hotels, get_hotel_info, retriever


# get_hotel_info


def test_get_hotel_info_is_case_insensitive_and_hides_review():
    hotel = get_hotel_info("Atlas Hub")

    assert hotel == {"city": "Prague", "price_per_night": 85, "rating": 4.3}


def test_get_hotel_info_does_not_mutate_the_knowledge_base():
    get_hotel_info("atlas hub")

    assert "review" in HOTELS["atlas hub"]


def test_get_hotel_info_unknown_hotel():
    assert get_hotel_info("Hotel California") == {"error": "Hotel not in the database"}


# filter_hotels


def names(result: list[dict]) -> set[str]:
    return {name for item in result for name in item}


def test_filter_hotels_by_max_price_is_inclusive():
    assert names(filter_hotels(max_price=95)) == {"nordic loft", "velvet residence", "atlas hub"}


def test_filter_hotels_by_city():
    assert names(filter_hotels(city="Prague")) == {"aurora suites", "atlas hub"}


def test_filter_hotels_by_city_and_price():
    assert names(filter_hotels(max_price=100, city="Prague")) == {"atlas hub"}


def test_filter_hotels_without_filters_returns_all_without_reviews():
    result = filter_hotels()

    assert names(result) == set(HOTELS)
    assert all("review" not in hotel for item in result for hotel in item.values())


def test_filter_hotels_city_match_is_case_sensitive_today():
    # Current behavior, kept on purpose: the v2 catalog will normalize city names
    assert filter_hotels(city="prague") == []


# similaritySearch / retriever


def test_similarity_search_returns_docs_from_the_index(v1_store):
    docs, ids = similaritySearch("reliable wifi for remote work", v1_store, top_k=3)

    assert len(docs) == len(ids) <= 3
    assert all(v1_store.docs[i] == doc for i, doc in zip(ids, docs))


def test_similarity_search_drops_results_below_threshold(v1_store):
    docs, ids = similaritySearch("reliable wifi", v1_store, top_k=3, threshold=1.01)

    assert docs == [] and ids == []


def test_retriever_keeps_only_chunks_the_judge_accepts(v1_store):
    client = fakeClient(judgeResponse([True, False, False]))

    chunks = retriever("reliable wifi for remote work", v1_store, client, top_k=3)

    assert len(chunks) == 1
    assert "Hotel name:" in chunks[0]


@pytest.mark.parametrize(
    "judge_output",
    [judgeResponse([True]), RuntimeError("API down")],
    ids=["wrong-number-of-flags", "api-error"],
)
def test_retriever_falls_back_to_all_candidates_when_judge_fails(v1_store, judge_output):
    client = fakeClient(judge_output)

    chunks = retriever("reliable wifi for remote work", v1_store, client, top_k=3)

    assert len(chunks) == 3


# agent loop


def test_dispatcher_rejects_unknown_tools():
    with pytest.raises(ValueError):
        dispatcher("book_hotel", {}, v_store=None, client=None)


def test_run_agent_answers_without_tools():
    client = fakeClient(textResponse("Hello!"))
    context = [{"role": "user", "content": "hi"}]

    answer = runAgent(context, client, v_store=None, steps_number=5)

    assert answer == "Hello!"
    assert context[-1] == {"role": "assistant", "content": "Hello!"}


def test_run_agent_runs_tool_and_feeds_result_back():
    client = fakeClient(
        toolCallResponse("filter_hotels", {"max_price": 100, "city": "Prague"}),
        textResponse("Atlas Hub fits your budget."),
    )
    context = [{"role": "user", "content": "Hotels in Prague under 100?"}]

    answer = runAgent(context, client, v_store=None, steps_number=5)

    tool_output = next(item for item in context if item.get("type") == "function_call_output")
    assert json.loads(tool_output["output"]) == filter_hotels(max_price=100, city="Prague")
    assert answer == "Atlas Hub fits your budget."
    # The second model call sees the tool result
    assert tool_output in client.responses.calls[1]["input"]
