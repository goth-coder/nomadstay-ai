"""
One "request" log line per HTTP request: the RED data (rate, errors,
duration) plus the model usage the request accumulated (observability/llm.py).
Percentiles are computed from these lines in CloudWatch Logs Insights.
"""

import logging

from observability.context import RequestStats

logger = logging.getLogger(__name__)

# Called every 30 s by the ECS health check: logged at DEBUG so it doesn't count as traffic
QUIET_ROUTES = {"/health"}


def routeTemplate(scope: dict) -> str:
    """
    "/agent/sessions/{session_id}", not the real path: one value per route
    keeps the field groupable (low cardinality). Unknown paths share one value.
    """
    route = scope.get("route")
    return getattr(route, "path", "unmatched")


def logRequest(method: str, route: str, status: int, duration_ms: float,
               stats: RequestStats) -> None:
    level = logging.DEBUG if route in QUIET_ROUTES else logging.INFO
    logger.log(level, "request", extra={"fields": {
        "method": method, "route": route, "status": status,
        "duration_ms": round(duration_ms, 1),
        "llm_calls": stats.llm_calls, "llm_ms": round(stats.llm_ms, 1),
        "input_tokens": stats.input_tokens, "output_tokens": stats.output_tokens,
        "cost_usd": round(stats.cost_usd, 6),
    }})
