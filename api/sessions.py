import logging
import threading
import time
from collections import OrderedDict, deque
from uuid import uuid4

logger = logging.getLogger(__name__)


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
