"""VBGui F-A FastAPI server tests — HTTP + WebSocket transports."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from cppmega_v4.jsonrpc import SCHEMA_VERSION, create_app


_GRAPH = {
    "nodes": [{"id": "a", "kind": "mlp"}, {"id": "b", "kind": "mlp"}],
    "edges": [{"src": "a", "dst": "b"}],
}
_DIM_ENV = {"B": 1, "S": 4, "H": 64, "nh": 2, "nkv": 1, "head_dim": 32,
            "num_experts": 8, "top_k": 2}
_LOSS = {"kind": "cross_entropy", "head_outputs": ["b"]}
_OPTIM = {"kind": "adamw",
          "groups": [{"matcher": "all", "lr": 3e-4,
                      "weight_decay": 0.01, "betas": [0.9, 0.95]}]}


@pytest.fixture
def client():
    return TestClient(create_app(cache_capacity=4))


# ---------------------------------------------------------------------------
# HTTP transport.
# ---------------------------------------------------------------------------


def test_health_reports_schema_version(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "schema_version": SCHEMA_VERSION}


def test_methods_lists_documented_endpoints(client):
    r = client.get("/schema/methods")
    assert r.status_code == 200
    methods = r.json()["methods"]
    assert "verify" in methods
    assert "probe.run" in methods
    assert "pipeline.abort" in methods


def test_rpc_verify_round_trip(client):
    payload = {
        "jsonrpc": "2.0", "id": "v_1", "method": "verify",
        "params": {"graph": _GRAPH, "dim_env": _DIM_ENV,
                   "loss": _LOSS, "optim": _OPTIM},
    }
    r = client.post("/rpc", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body["id"] == "v_1"
    assert body["jsonrpc"] == "2.0"
    assert "result" in body
    assert "resolved" in body["result"]


def test_rpc_invalid_method_returns_error_envelope(client):
    r = client.post("/rpc", json={
        "jsonrpc": "2.0", "id": 1, "method": "no_such", "params": {},
    })
    assert r.status_code == 200
    body = r.json()
    assert "error" in body
    assert body["error"]["code"] == -32601


def test_rpc_cache_stats_reflect_calls(client):
    payload = {
        "jsonrpc": "2.0", "id": "c1", "method": "verify",
        "params": {"graph": _GRAPH, "dim_env": _DIM_ENV,
                   "loss": _LOSS, "optim": _OPTIM},
    }
    client.post("/rpc", json=payload)
    client.post("/rpc", json={**payload, "id": "c2"})  # same canonical params
    r = client.get("/cache/stats")
    stats = r.json()
    assert stats["hits"] == 1
    assert stats["misses"] == 1


def test_cache_clear_resets(client):
    payload = {
        "jsonrpc": "2.0", "id": "c", "method": "verify",
        "params": {"graph": _GRAPH, "dim_env": _DIM_ENV,
                   "loss": _LOSS, "optim": _OPTIM},
    }
    client.post("/rpc", json=payload)
    client.post("/cache/clear")
    assert client.get("/cache/stats").json()["size"] == 0


# ---------------------------------------------------------------------------
# WebSocket transport.
# ---------------------------------------------------------------------------


def test_ws_verify_round_trip(client):
    payload = {
        "jsonrpc": "2.0", "id": "ws_1", "method": "backend.status",
    }
    with client.websocket_connect("/ws") as ws:
        ws.send_text(json.dumps(payload))
        msg = ws.receive_json()
    assert msg["id"] == "ws_1"
    assert msg["result"] == {"status": "ok"}


def test_ws_parse_error_on_bad_json(client):
    with client.websocket_connect("/ws") as ws:
        ws.send_text("not json {")
        msg = ws.receive_json()
    assert msg["error"]["code"] == -32700


def test_ws_invalid_method_returns_error_envelope(client):
    payload = {"jsonrpc": "2.0", "id": "wsx", "method": "no_such"}
    with client.websocket_connect("/ws") as ws:
        ws.send_text(json.dumps(payload))
        msg = ws.receive_json()
    assert msg["error"]["code"] == -32601
