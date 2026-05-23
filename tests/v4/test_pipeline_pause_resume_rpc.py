"""V7-H06: pipeline.pause / pipeline.resume RPC dispatcher round-trips.

These are the dispatcher-side handlers. The end-to-end pause-during-
train flow is covered by the e2e Playwright test (84_pause_resume).
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from cppmega_v4.jsonrpc import create_app
from cppmega_v4.runtime import job_control


@pytest.fixture(autouse=True)
def _clean_state():
    job_control.reset()
    yield
    job_control.reset()


@pytest.fixture
def client():
    return TestClient(create_app(cache_capacity=2))


def test_pipeline_pause_marks_token_paused(client):
    payload = {"jsonrpc": "2.0", "id": "p1", "method": "pipeline.pause",
               "params": {"run_id": "run-abc"}}
    r = client.post("/rpc", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body["result"]["run_id"] == "run-abc"
    assert job_control.is_paused("run-abc") is True


def test_pipeline_resume_clears_pause(client):
    job_control.pause("run-xyz")
    assert job_control.is_paused("run-xyz") is True
    payload = {"jsonrpc": "2.0", "id": "r1", "method": "pipeline.resume",
               "params": {"run_id": "run-xyz"}}
    r = client.post("/rpc", json=payload)
    assert r.status_code == 200
    assert r.json()["result"]["run_id"] == "run-xyz"
    assert job_control.is_paused("run-xyz") is False


def test_pause_resume_methods_appear_in_registry(client):
    r = client.get("/schema/methods")
    methods = set(r.json()["methods"])
    assert "pipeline.pause" in methods
    assert "pipeline.resume" in methods
