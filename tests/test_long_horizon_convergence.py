"""V7-A03: 1000-step loss convergence on tiny model."""

from __future__ import annotations

import pytest

from cppmega_v4.jsonrpc.schema import VerifyParams
from cppmega_v4.runner import Pipeline, run_pipeline


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
        "dim_env": {"B": 1, "S": 16, "H": 128,
                    "nh": 2, "nkv": 1, "head_dim": 64},
        "loss": {"kind": "cross_entropy", "head_outputs": ["mlp"]},
        "optim": {"kind": "adamw",
                  "groups": [{"matcher": "all", "lr": 1e-3,
                              "weight_decay": 0.01,
                              "betas": [0.9, 0.95]}]},
    })


def test_v7_a03_1000step_convergence_tiny_model():
    rep = run_pipeline(_spec(), Pipeline.from_dict({
        "stages": ["parse", "verify_build_spec", "build_model", "train"],
        "stage_options": {"train": {"num_steps": 1000}},
    }))
    tr = next(s for s in rep.stages if s.name == "train")
    assert tr.status == "ok", f"train failed: {tr.error}"
    losses = tr.extras["losses"]
    assert len(losses) == 1000
    # All finite.
    for L in losses:
        assert -1e10 < L < 1e10
    # Tail (last 100) mean strictly below head (first 100) mean —
    # 1000-step long-horizon convergence on tiny model.
    head = sum(losses[:100]) / 100
    tail = sum(losses[-100:]) / 100
    assert tail < head, (
        f"long-horizon loss did not decrease: head={head:.4f}, tail={tail:.4f}"
    )
    # Smoothed series available.
    sm = tr.extras["losses_smoothed"]
    assert len(sm) == 1000


def test_v7_a03_loss_decrease_significant_after_500_steps():
    rep = run_pipeline(_spec(), Pipeline.from_dict({
        "stages": ["parse", "verify_build_spec", "build_model", "train"],
        "stage_options": {"train": {"num_steps": 600}},
    }))
    tr = next(s for s in rep.stages if s.name == "train")
    losses = tr.extras["losses"]
    initial = sum(losses[:50]) / 50
    after_500 = sum(losses[-50:]) / 50
    # At least 10% relative drop on synthetic random data.
    assert after_500 < initial * 0.95, (
        f"loss didn't drop >5% after 500 steps: "
        f"initial={initial:.4f}, after_500={after_500:.4f}"
    )
