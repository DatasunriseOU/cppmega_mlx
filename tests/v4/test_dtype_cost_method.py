"""V7-D06: dtype.cost_estimate RPC round-trip."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from cppmega_v4.jsonrpc import create_app


@pytest.fixture
def client():
    return TestClient(create_app(cache_capacity=2))


def test_dtype_cost_estimate_returns_three_rows(client):
    payload = {
        "jsonrpc": "2.0", "id": "d1", "method": "dtype.cost_estimate",
        "params": {"n_iter": 2, "hidden": 32, "batch": 1, "seq": 4},
    }
    r = client.post("/rpc", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert "error" not in body, body
    res = body["result"]
    assert {row["dtype"] for row in res["rows"]} == {"fp32", "bf16", "fp16"}
    for row in res["rows"]:
        if row["supported"]:
            assert row["fwd_ms"] is not None and row["fwd_ms"] >= 0
            assert row["fwd_ms_per_token"] is not None
            assert row["fwd_ms_per_token"] >= 0
            assert row["fwdbwd_ms"] is not None
        else:
            # Even when unsupported, the row still appears and carries
            # an error string the UI can show.
            assert row["error"] is not None


def test_dtype_cost_estimate_rejects_huge_params(client):
    payload = {
        "jsonrpc": "2.0", "id": "d2", "method": "dtype.cost_estimate",
        "params": {"n_iter": 10000},
    }
    r = client.post("/rpc", json=payload)
    body = r.json()
    assert "error" in body
    assert body["error"]["code"] == -32602
