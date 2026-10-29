"""
Smoke tests for the package layout: the API module imports and keeps its
routes, and project loggers still get LOG_LEVEL after the move to packages.
"""

import logging

from observability.logging_config import setupLogging


def test_api_imports_and_keeps_its_routes():
    from api.app import app

    routes = {(route.path, method) for route in app.routes for method in route.methods}
    assert {
        ("/health", "GET"),
        ("/agent/run", "POST"),
        ("/agent/sessions/{session_id}", "GET"),
    } <= routes


def test_submodule_loggers_inherit_the_project_level(monkeypatch):
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    setupLogging()

    for name in ("agent.nomad", "retrieval.catalog", "storage.s3", "api.app"):
        assert logging.getLogger(name).getEffectiveLevel() == logging.DEBUG, name
    assert logging.getLogger("botocore").getEffectiveLevel() == logging.WARNING


def test_health_runs_on_the_event_loop_not_in_the_thread_pool():
    # A sync route waits for a free worker thread: under load the ECS health
    # check (5 s timeout) would fail and the task would be killed when busiest
    import inspect

    from api.app import healthcheck

    assert inspect.iscoroutinefunction(healthcheck)
