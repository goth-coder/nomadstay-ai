import uuid
from contextvars import ContextVar

# One value per request: concurrent requests (and the threads FastAPI runs
# sync routes in) each see their own id. "-" means outside a request (startup).
_request_id: ContextVar[str] = ContextVar("request_id", default="-")


def newRequestId() -> str:
    request_id = uuid.uuid4().hex[:12]
    _request_id.set(request_id)
    return request_id


def currentRequestId() -> str:
    return _request_id.get()
