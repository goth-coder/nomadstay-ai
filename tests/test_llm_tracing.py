"""
Module 3: every model call is timed, priced and logged as "llm_call", and
inside a request it adds to that request's totals, even from another thread.
"""

import contextvars
import logging
import threading
from types import SimpleNamespace

import pytest
from conftest import fakeClient

from observability.context import currentRequestStats, startRequestStats
from observability.llm import costUsd, tracedCall

PARAMS = {"model": "gpt-4o-mini", "input": "hi"}


def withUsage(input_tokens, output_tokens, cached_tokens=0):
    usage = SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens,
                            input_tokens_details=SimpleNamespace(cached_tokens=cached_tokens))
    return SimpleNamespace(output_text="ok", output=[], usage=usage)


def inFreshContext(function):
    """Runs a test body in its own context, so request stats don't leak to other tests."""
    return contextvars.copy_context().run(function)


@pytest.fixture
def calls(caplog):
    caplog.set_level(logging.INFO, logger="observability.llm")
    return lambda: [r.fields for r in caplog.records if r.getMessage() == "llm_call"]


# ---------- cost ----------


def test_cost_uses_input_and_output_prices():
    assert costUsd("gpt-4o-mini", 1_000_000, 1_000_000) == pytest.approx(0.15 + 0.60)


def test_cached_input_tokens_cost_less():
    assert costUsd("gpt-4o-mini", 1_000_000, 0, cached_tokens=1_000_000) == pytest.approx(0.075)


def test_unknown_model_has_unknown_cost_not_zero():
    assert costUsd("some-new-model", 1000, 1000) is None


# ---------- the log line ----------


def test_call_is_logged_with_latency_tokens_and_cost(calls):
    response = tracedCall(fakeClient(withUsage(2000, 100)), "agent", **PARAMS)

    [event] = calls()
    assert response.output_text == "ok"
    assert event["purpose"] == "agent" and event["model"] == "gpt-4o-mini"
    assert event["status"] == "ok" and event["latency_ms"] >= 0
    assert (event["input_tokens"], event["output_tokens"]) == (2000, 100)
    assert event["cost_usd"] == pytest.approx(costUsd("gpt-4o-mini", 2000, 100))


def test_failed_call_is_logged_and_still_raises(calls):
    with pytest.raises(RuntimeError):
        tracedCall(fakeClient(RuntimeError("OpenAI down")), "judge_question", **PARAMS)

    [event] = calls()
    assert event["status"] == "error:RuntimeError" and event["input_tokens"] == 0


def test_response_without_usage_counts_zero_tokens(calls):
    tracedCall(fakeClient(SimpleNamespace(output_text="", output=[])), "agent", **PARAMS)

    assert calls()[0]["input_tokens"] == calls()[0]["output_tokens"] == 0


# ---------- request totals ----------


def test_calls_add_up_in_the_request_totals():
    def body():
        stats = startRequestStats()
        client = fakeClient(withUsage(1000, 100), withUsage(3000, 200))
        tracedCall(client, "agent", **PARAMS)
        tracedCall(client, "agent", **PARAMS)
        return stats

    stats = inFreshContext(body)

    assert stats.llm_calls == 2
    assert (stats.input_tokens, stats.output_tokens) == (4000, 300)
    assert stats.cost_usd == pytest.approx(costUsd("gpt-4o-mini", 4000, 300))


def test_totals_made_in_another_thread_reach_the_request():
    # FastAPI runs the route in a thread with a copy of the context, like this
    def body():
        stats = startRequestStats()
        route = contextvars.copy_context()
        thread = threading.Thread(target=route.run, args=(tracedCall, fakeClient(withUsage(500, 50)), "agent"),
                                  kwargs=PARAMS)
        thread.start()
        thread.join()
        return stats

    assert inFreshContext(body).input_tokens == 500


def test_calls_outside_a_request_are_logged_not_summed(calls):
    def body():
        tracedCall(fakeClient(withUsage(10, 1)), "judge_stance", **PARAMS)
        return currentRequestStats()

    assert inFreshContext(body) is None
    assert len(calls()) == 1
