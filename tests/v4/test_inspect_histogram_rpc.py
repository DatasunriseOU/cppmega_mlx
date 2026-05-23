"""V7-H08: inspect.histogram RPC round-trip + registry membership.

Honest-closure: cppmega_v4/jsonrpc/histogram_method.py existed but
was not registered in _ROUTES, so UI calls got method_not_found.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from cppmega_v4.jsonrpc import create_app


@pytest.fixture
def client():
    return TestClient(create_app(cache_capacity=2))


def _spec_payload(brick_id: str = "mlp") -> dict:
    return {
        "graph": {
            "nodes": [
                {"id": "attn", "kind": "attention", "params": {}},
                {"id": brick_id, "kind": "mlp",
                 "params": {"intermediate_size": 64, "activation": "swiglu"}},
            ],
            "edges": [{"src": "attn", "dst": brick_id}],
        },
        "dim_env": {"B": 1, "S": 8, "H": 32, "nh": 2, "nkv": 1,
                    "head_dim": 16},
        "loss": {"kind": "cross_entropy", "head_outputs": [brick_id]},
        "optim": {"kind": "adamw",
                  "groups": [{"matcher": "all", "lr": 1e-3,
                              "weight_decay": 0.01, "betas": [0.9, 0.95]}]},
    }


def test_inspect_histogram_in_registry(client):
    r = client.get("/schema/methods")
    methods = set(r.json()["methods"])
    assert "inspect.histogram" in methods


def test_inspect_histogram_returns_bins_and_counts(client):
    payload = {
        "jsonrpc": "2.0", "id": "h1", "method": "inspect.histogram",
        "params": {"spec": _spec_payload(),
                    "brick_id": "mlp",
                    "kind": "weight",
                    "buckets": 16},
    }
    r = client.post("/rpc", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert "error" not in body, body
    res = body["result"]
    assert res["brick_id"] == "mlp"
    assert res["buckets"] == 16
    assert len(res["counts"]) == 16
    assert len(res["bins"]) == 17       # buckets + 1 bin edges
    assert res["n_values"] > 0
    assert res["min"] <= res["max"]


def test_inspect_histogram_rejects_unknown_brick_id(client):
    payload = {
        "jsonrpc": "2.0", "id": "h2", "method": "inspect.histogram",
        "params": {"spec": _spec_payload(),
                    "brick_id": "does_not_exist"},
    }
    r = client.post("/rpc", json=payload)
    body = r.json()
    assert "error" in body
