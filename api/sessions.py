import json
import logging
import os
import threading
import time
from collections import OrderedDict, deque
from uuid import uuid4

logger = logging.getLogger(__name__)


class DynamoSessionStore:
    """
    Sessions in a DynamoDB table, shared by every task behind the load balancer.
    Item: session_id (partition key), context (JSON string), expires_at (epoch s).
    DynamoDB TTL deletes expired items, but up to days later: get() checks
    expires_at itself. Two turns of one session at the same time: the last save wins.
    """

    def __init__(self, table, ttl_seconds: int = 1800, clock=time.time):
        self.table = table
        self.ttl_seconds = ttl_seconds
        self.clock = clock

    def create(self) -> tuple[str, deque]:
        # Saved only after the first answer: a failed turn leaves nothing behind
        return str(uuid4()), deque()

    def get(self, session_id: str) -> deque | None:
        item = self.table.get_item(Key={"session_id": session_id}).get("Item")
        if item is None or int(item["expires_at"]) <= self.clock():
            return None
        return deque(json.loads(item["context"]))

    def save(self, session_id: str, context: deque):
        self.table.put_item(Item={
            "session_id": session_id,
            "context": json.dumps(list(context), ensure_ascii=False),
            "expires_at": int(self.clock()) + self.ttl_seconds,  # every turn renews the TTL
        })


class SessionStore:
    """
    In-memory sessions with limits:
      - ttl_seconds: sessions idle longer than this are removed
      - max_sessions: when full, the least recently used session is removed
    The OrderedDict is kept in LRU order (oldest first).
    """

    def __init__(self, ttl_seconds: int = 1800, max_sessions: int = 500):
        self.ttl_seconds = ttl_seconds
        self.max_sessions = max_sessions
        self._sessions: OrderedDict[str, tuple[float, deque]] = OrderedDict()
        # FastAPI runs sync endpoints in a thread pool
        self._lock = threading.Lock()

    def create(self) -> tuple[str, deque]:
        with self._lock:
            self._evictExpired()
            while len(self._sessions) >= self.max_sessions:
                old_id, _ = self._sessions.popitem(last=False)
                logger.info("Session evicted (LRU) session_id=%s", old_id)

            session_id = str(uuid4())
            context = deque()
            self._sessions[session_id] = (time.monotonic(), context)
            return session_id, context

    def get(self, session_id: str) -> deque | None:
        with self._lock:
            self._evictExpired()
            entry = self._sessions.get(session_id)
            if entry is None:
                return None

            # Touch: renews the TTL and moves it to the end (most recent)
            self._sessions[session_id] = (time.monotonic(), entry[1])
            self._sessions.move_to_end(session_id)
            return entry[1]

    def save(self, session_id: str, context: deque):
        """The context is this store's own deque, already changed in place: only renews the TTL."""
        self.get(session_id)

    def _evictExpired(self):
        now = time.monotonic()
        while self._sessions:
            session_id, (last_access, _) = next(iter(self._sessions.items()))
            if now - last_access < self.ttl_seconds:
                break
            self._sessions.popitem(last=False)
            logger.info("Session expired session_id=%s", session_id)


def trimToLastTurns(context: deque, max_turns: int):
    """
    Keeps only the last `max_turns` user turns.
    Cuts at a user message, so a function_call is never
    separated from its function_call_output.
    """
    user_positions = [i for i, item in enumerate(context) if item.get("role") == "user"]
    if len(user_positions) <= max_turns:
        return

    for _ in range(user_positions[-max_turns]):
        context.popleft()


def sessionStoreFromEnv():
    """DynamoDB when SESSION_TABLE is set (more than one task); else in memory (local, tests)."""
    ttl_seconds = int(os.environ.get("SESSION_TTL_SECONDS", "1800"))
    table_name = os.environ.get("SESSION_TABLE")
    if table_name:
        # Same profile and region as the artifacts
        from storage.s3 import session as aws

        return DynamoSessionStore(aws.resource("dynamodb").Table(table_name), ttl_seconds)
    return SessionStore(ttl_seconds, int(os.environ.get("MAX_SESSIONS", "500")))
