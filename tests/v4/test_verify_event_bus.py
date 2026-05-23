"""V7-L45: verify_event_bus + verify() publish progress events."""

from __future__ import annotations

import queue

import pytest
from fastapi.testclient import TestClient

from cppmega_v4.jsonrpc import create_app
from cppmega_v4.jsonrpc.methods import verify
from cppmega_v4.jsonrpc.schema import VerifyParams
from cppmega_v4.runtime import verify_event_bus as bus


@pytest.fixture(autouse=True)
def _clean():
    bus.reset()
    yield
    bus.reset()


def _spec_payload() -> dict:
    return {
        "graph": {
            "nodes": [
                {"id": "attn", "kind": "attention", "params": {}},
                {"id": "mlp", "kind": "mlp",
                 "params": {"intermediate_size": 64, "activation": "swiglu"}},
            ],
            "edges": [{"src": "attn", "dst": "mlp"}],
        },
        "dim_env": {"B": 1, "S": 8, "H": 32, "nh": 2, "nkv": 1,
                    "head_dim": 16},
        "loss": {"kind": "cross_entropy", "head_outputs": ["mlp"]},
        "optim": {"kind": "adamw",
                  "groups": [{"matcher": "all", "lr": 1e-3,
                              "weight_decay": 0.01, "betas": [0.9, 0.95]}]},
    }


def test_spec_hash_stable_across_equivalent_dicts():
    h1 = bus.spec_hash(_spec_payload())
    h2 = bus.spec_hash(_spec_payload())
    assert h1 == h2
    assert len(h1) == 64


def test_verify_publishes_phase_events_to_subscribers():
    params = VerifyParams.model_validate(_spec_payload())
    h = bus.spec_hash(params)
    q = bus.subscribe(h)
    verify(params)
    phases: list[str] = []
    while True:
        try:
            ev = q.get_nowait()
        except queue.Empty:
            break
        if ev is None:
            phases.append("__finish__")
            break
        phases.append(ev["phase"])
    # The handler emits at least start → graph_built → resolve_shapes
    # → memory_estimated → done, plus the sentinel.
    for required in ("start", "graph_built", "memory_estimated",
                      "done", "__finish__"):
        assert required in phases, phases


def test_verify_no_subscriber_is_a_noop():
    params = VerifyParams.model_validate(_spec_payload())
    out = verify(params)
    assert out.elapsed_ms >= 0


def test_ws_verify_endpoint_appears_in_routes():
    app = create_app(cache_capacity=2)
    paths = {r.path for r in app.routes
             if hasattr(r, "path") and getattr(r, "path", "")}
    assert "/ws/verify/{spec_hash}" in paths
