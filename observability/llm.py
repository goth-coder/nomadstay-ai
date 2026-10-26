"""
Every call to the model goes through tracedCall: one "llm_call" log line with
latency, tokens and cost, added to the request's totals when inside a request.
"""

import logging
import time

from observability.context import currentRequestStats

logger = logging.getLogger(__name__)

# USD per million tokens. Check https://openai.com/api/pricing when a model or price changes.
PRICES_PER_MILLION = {
    "gpt-4o-mini": {"input": 0.15, "cached_input": 0.075, "output": 0.60},
}


def costUsd(model: str, input_tokens: int, output_tokens: int, cached_tokens: int = 0) -> float | None:
    """None for a model without a price: unknown is not zero."""
    price = PRICES_PER_MILLION.get(model)
    if price is None:
        return None
    return ((input_tokens - cached_tokens) * price["input"]
            + cached_tokens * price["cached_input"]
            + output_tokens * price["output"]) / 1_000_000


def usageCounts(response) -> tuple[int, int, int]:
    """(input, output, cached) tokens; zeros when the response has no usage."""
    usage = getattr(response, "usage", None)
    if usage is None:
        return 0, 0, 0
    details = getattr(usage, "input_tokens_details", None)
    cached = getattr(details, "cached_tokens", 0) or 0
    return usage.input_tokens, usage.output_tokens, cached


def tracedCall(client, purpose: str, **params):
    """client.responses.create(**params), timed, priced and logged. Errors are logged and re-raised."""
    model = params.get("model")
    start = time.perf_counter()
    try:
        response = client.responses.create(**params)
    except Exception as error:
        logCall(purpose, model, start, None, f"error:{type(error).__name__}")
        raise
    logCall(purpose, model, start, response, "ok")
    return response


def logCall(purpose: str, model: str | None, start: float, response, status: str) -> None:
    latency_ms = round((time.perf_counter() - start) * 1000, 1)
    input_tokens, output_tokens, cached_tokens = usageCounts(response)
    cost = costUsd(model, input_tokens, output_tokens, cached_tokens)

    stats = currentRequestStats()
    if stats is not None:
        stats.add(latency_ms, input_tokens, output_tokens, cost)

    logger.info("llm_call", extra={"fields": {
        "purpose": purpose, "model": model, "status": status, "latency_ms": latency_ms,
        "input_tokens": input_tokens, "cached_tokens": cached_tokens,
        "output_tokens": output_tokens,
        "cost_usd": round(cost, 6) if cost is not None else None,
    }})
