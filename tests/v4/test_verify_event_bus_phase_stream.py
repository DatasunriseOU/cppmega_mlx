"""V7-H37: verify() publishes phase events to verify_event_bus so the
UI subscriber on /ws/verify/{spec_hash} sees progress."""

from __future__ import annotations

import threading
import time

import pytest

from cppmega_v4.jsonrpc.methods import verify
from cppmega_v4.jsonrpc.schema import VerifyParams
from cppmega_v4.runtime import verify_event_bus as vb


@pytest.fixture(autouse=True)
def _reset():
    vb.reset()
    yield
    vb.reset()


def _spec_params() -> VerifyParams:
    return VerifyParams.model_validate({
        "graph": {
            "nodes": [
                {"id": "attn", "kind": "attention",
                 "params": {"num_heads": 4, "head_dim": 64}},
                {"id": "mlp", "kind": "mlp", "params": {}},
            ],
            "edges": [{"src": "attn", "dst": "mlp"}],
        },
        "dim_env": {"B": 1, "S": 8, "H": 128,
                    "nh": 2, "nkv": 1, "head_dim": 64},
        "loss": {"kind": "cross_entropy", "head_outputs": ["mlp"]},
        "optim": {"kind": "adamw",
                  "groups": [{"matcher": "all", "lr": 1e-3,
                              "weight_decay": 0.01,
                              "betas": [0.9, 0.95]}]},
    })


def test_v7_h37_spec_hash_is_stable_for_equivalent_payloads():
    p1 = _spec_params()
    p2 = _spec_params()
    assert vb.spec_hash(p1) == vb.spec_hash(p2)
    assert len(vb.spec_hash(p1)) == 64  # sha256 hex


def test_v7_h37_verify_publishes_phase_events_then_finish():
    params = _spec_params()
    h = vb.spec_hash(params)

    collected: list[dict | None] = []

    def _collect():
        q = vb.subscribe(h)
        while True:
            try:
                ev = q.get(timeout=3.0)
            except Exception:
                break
            collected.append(ev)
            if ev is None:
                break

    t = threading.Thread(target=_collect, daemon=True)
    t.start()
    time.sleep(0.05)  # let subscribe land before publish

    verify(params)
    # The verify handler doesn't emit a sentinel — the WS endpoint does
    # on disconnect. So we wait until at least the start + resolve
    # frames land, then drain.
    deadline = time.time() + 1.0
    while time.time() < deadline and len(collected) < 2:
        time.sleep(0.02)
    vb.publish(h, None)  # close the subscriber loop
    t.join(timeout=1.0)

    phases = [e["phase"] for e in collected if isinstance(e, dict)]
    assert "start" in phases, phases
    assert "resolve_shapes" in phases, phases
