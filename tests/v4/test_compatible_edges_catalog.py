"""E-AUDIT-02 backend: catalog.list_options('compatible_edges')."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from cppmega_v4.jsonrpc import create_app
from cppmega_v4.spec.shape_contract import (
    compatible_edges, registered_kinds,
)


@pytest.fixture
def client():
    return TestClient(create_app(cache_capacity=2))


def test_compatible_edges_helper_nonempty_and_well_typed():
    pairs = compatible_edges()
    assert len(pairs) > 0
    kinds = set(registered_kinds())
    for src, dst in pairs:
        assert src in kinds
        assert dst in kinds


def test_compatible_edges_includes_attention_to_mlp():
    """attention.outputs and mlp.inputs share the canonical hidden
    state channel — the most common edge in any transformer block."""
    pairs = set(compatible_edges())
    assert ("attention", "mlp") in pairs


def test_catalog_endpoint_returns_pair_encoded_options(client):
    payload = {"jsonrpc": "2.0", "id": "e1",
               "method": "catalog.list_options",
               "params": {"category": "compatible_edges"}}
    r = client.post("/rpc", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert "error" not in body, body
    opts = body["result"]["options"]
    assert len(opts) > 0
    for o in opts[:5]:
        assert "->" in o["name"]
        src, dst = o["name"].split("->")
        assert o["paper_ref"] == src
        assert o["summary"] == dst
