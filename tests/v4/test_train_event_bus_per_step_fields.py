"""V7-H34/H35/H36: per-step train_event_bus payload carries grad_norms,
mem_mb, expert_load fields.

Previously only {step, loss, lr, overflow} were published, so the UI
LiveTrainPanel could only show loss/lr live; everything else had to
wait until extras landed at pipeline.run resolve."""

from __future__ import annotations

import threading
import time

import pytest

from cppmega_v4.jsonrpc.schema import VerifyParams
from cppmega_v4.runner import Pipeline, run_pipeline
from cppmega_v4.runtime import train_event_bus as bus


@pytest.fixture(autouse=True)
def _reset():
    bus.reset()
    yield
    bus.reset()


def _spec() -> VerifyParams:
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


def _collect_events(run_id: str, out: list, sentinel_seen: list) -> None:
    q = bus.subscribe(run_id)
    while True:
        try:
            ev = q.get(timeout=3.0)
        except Exception:
            break
        if ev is None:
            sentinel_seen.append(True)
            break
        out.append(ev)


def test_v7_h34_h35_h36_per_step_payload_carries_grad_norms_mem_expert():
    events: list[dict] = []
    sentinel: list[bool] = []
    t = threading.Thread(target=_collect_events,
                         args=("rid-grad", events, sentinel),
                         daemon=True)
    t.start()
    # Give subscriber a chance to register before train publishes.
    time.sleep(0.05)
    rep = run_pipeline(_spec(), Pipeline.from_dict({
        "stages": ["parse", "verify_build_spec", "build_model", "train"],
        "stage_options": {"train": {
            "num_steps": 2, "run_id": "rid-grad",
            "abort_token": "rid-grad",
        }},
    }))
    t.join(timeout=4.0)
    tr = next(s for s in rep.stages if s.name == "train")
    assert tr.status == "ok", tr.error
    assert sentinel == [True]
    assert len(events) == 2, f"expected 2 step events, got {events}"

    e0 = events[0]
    # Backwards-compat fields still present.
    assert e0["step"] == 0
    assert isinstance(e0["loss"], float)
    # V7-H34: grad_norms dict, per-brick keyed.
    assert "grad_norms" in e0
    assert isinstance(e0["grad_norms"], dict)
    assert len(e0["grad_norms"]) >= 1
    for k, v in e0["grad_norms"].items():
        assert isinstance(k, str)
        assert isinstance(v, (int, float))
        assert v >= 0
    # V7-H35: mem_mb float-or-None per step (Apple-only; None on Linux).
    assert "mem_mb" in e0
    assert e0["mem_mb"] is None or isinstance(e0["mem_mb"], (int, float))
    # V7-H36: expert_load list or None depending on whether MoE present.
    assert "expert_load" in e0
    assert e0["expert_load"] is None \
        or isinstance(e0["expert_load"], list)
    # V7-H40 precursor — wall-clock timestamp per step.
    assert "ts" in e0
    assert isinstance(e0["ts"], float)
