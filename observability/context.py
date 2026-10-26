import threading
import uuid
from contextvars import ContextVar
from dataclasses import dataclass, field

# One value per request: concurrent requests (and the threads FastAPI runs
# sync routes in) each see their own id. "-" means outside a request (startup).
_request_id: ContextVar[str] = ContextVar("request_id", default="-")


def newRequestId() -> str:
    request_id = uuid.uuid4().hex[:12]
    _request_id.set(request_id)
    return request_id


def currentRequestId() -> str:
    return _request_id.get()


@dataclass
class RequestStats:
    """
    Model usage of one request. Kept as an object, not numbers, because the
    route runs in another thread with a copy of the context: the copy points
    to this same object, so what the route adds the middleware sees.
    """

    llm_calls: int = 0
    llm_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add(self, latency_ms: float, input_tokens: int, output_tokens: int, cost_usd: float | None):
        with self._lock:
            self.llm_calls += 1
            self.llm_ms += latency_ms
            self.input_tokens += input_tokens
            self.output_tokens += output_tokens
            self.cost_usd += cost_usd or 0.0


_request_stats: ContextVar[RequestStats | None] = ContextVar("request_stats", default=None)


def startRequestStats() -> RequestStats:
    stats = RequestStats()
    _request_stats.set(stats)
    return stats


def currentRequestStats() -> RequestStats | None:
    """None outside a request (CLI, ingestion): calls are logged, not summed."""
    return _request_stats.get()
