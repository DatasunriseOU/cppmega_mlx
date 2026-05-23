"""V7-L42..L44: per-step WS event payload includes grad_norms,
mem_mb, expert_load, ts."""

from __future__ import annotations

import queue

import pytest

from cppmega_v4.jsonrpc.schema import VerifyParams
from cppmega_v4.runner import Pipeline, run_pipeline
from cppmega_v4.runtime import train_event_bus as bus


@pytest.fixture(autouse=True)
def _clean():
    bus.reset()
    yield
    bus.reset()


def _spec() -> VerifyParams:
    return VerifyParams.model_validate({
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
    })


def _drain(q: queue.Queue) -> list[dict]:
    out: list[dict] = []
    while True:
        try:
            ev = q.get(timeout=0.1)
        except queue.Empty:
            break
        if ev is None:
            break
        out.append(ev)
    return out


def test_per_step_event_payload_includes_extended_fields():
    q = bus.subscribe("run-l42")
    run_pipeline(_spec(), Pipeline.from_dict({
        "stages": ["parse", "verify_build_spec", "build_model", "train"],
        "stage_options": {"train": {"num_steps": 2, "run_id": "run-l42"}},
    }))
    events = _drain(q)
    assert len(events) == 2
    for ev in events:
        # L42: grad_norms — dict keyed by brick path with float values.
        assert "grad_norms" in ev
        assert isinstance(ev["grad_norms"], dict)
        # L43: mem_mb — float or None (None when mx.metal unavailable).
        assert "mem_mb" in ev
        assert ev["mem_mb"] is None or isinstance(ev["mem_mb"], float)
        # L44: expert_load — None for non-MoE graphs.
        assert "expert_load" in ev
        # ts is a wall-clock float so the UI's dead-man-switch works.
        assert "ts" in ev
        assert isinstance(ev["ts"], float)
