"""
Module 2: every request gets an id that reaches every log line it produces
and comes back to the client in the X-Request-ID header.
"""

import asyncio
import json
import logging

from fastapi.testclient import TestClient

from observability.context import currentRequestId, newRequestId
from observability.logging_config import JsonFormatter


def test_current_request_id_is_the_one_just_created():
    request_id = newRequestId()

    assert currentRequestId() == request_id


def test_log_lines_carry_the_request_id():
    request_id = newRequestId()
    record = logging.LogRecord("api", logging.INFO, __file__, 1, "hello", None, None)

    event = json.loads(JsonFormatter().format(record))

    assert event["request_id"] == request_id


def test_concurrent_requests_keep_their_own_id():
    async def handle():
        request_id = newRequestId()
        await asyncio.sleep(0.01)  # the other request runs meanwhile
        return request_id, currentRequestId()

    async def both():
        return await asyncio.gather(handle(), handle())

    for created, seen in asyncio.run(both()):
        assert created == seen


def test_every_response_gets_its_own_request_id_header(monkeypatch):
    from api.app import app

    monkeypatch.setenv("API_KEY", "secret")
    client = TestClient(app)  # no "with": the lifespan (data loading) is not needed for a 401

    first = client.post("/agent/run", json={"message": "hi"})
    second = client.post("/agent/run", json={"message": "hi"})

    assert first.status_code == second.status_code == 401
    assert first.headers["X-Request-ID"] != second.headers["X-Request-ID"]
