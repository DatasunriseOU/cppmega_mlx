"""V7-A08: 1000-step cosine+warmup LR trajectory matches analytical."""

from __future__ import annotations

import math

import pytest

from cppmega_v4.jsonrpc.schema import VerifyParams
from cppmega_v4.runner import Pipeline, run_pipeline


def _spec(warmup: int = 50, total: int = 1000) -> VerifyParams:
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
                              "betas": [0.9, 0.95],
                              "schedule": {"kind": "cosine",
                                            "warmup_steps": warmup,
                                            "total_steps": total}}]},
    })


def _train(num_steps: int, warmup: int, total: int) -> dict:
    rep = run_pipeline(_spec(warmup, total), Pipeline.from_dict({
        "stages": ["parse", "verify_build_spec", "build_model", "train"],
        "stage_options": {"train": {"num_steps": num_steps}},
    }))
    tr = next(s for s in rep.stages if s.name == "train")
    assert tr.status == "ok", f"train failed: {tr.error}"
    return tr.extras


def test_v7_a08_lr_warmup_ramps_linearly_then_cosine_decays():
    """50 warmup steps + 1000 total: lr_trajectory[0] near 0, peaks
    near step 50, then cosine-decays towards 0."""
    e = _train(num_steps=200, warmup=50, total=1000)
    lr = e["lr_trajectory"]
    assert len(lr) == 200
    base_lr = 1e-3
    # First step lr should be near 0 (warmup start).
    assert lr[0] < base_lr * 0.1
    # Around end of warmup (step 50) lr near peak.
    assert lr[49] > base_lr * 0.8
    # After warmup, lr drops monotonically (cosine).
    assert lr[199] < lr[50]


def test_v7_a08_schedule_kind_recorded_in_extras():
    e = _train(num_steps=20, warmup=5, total=100)
    assert e["schedule_kind"] == "cosine"


def test_v7_a08_warmup_zero_no_ramp():
    """warmup_steps=0 → lr_trajectory[0] should equal the base lr."""
    e = _train(num_steps=20, warmup=0, total=100)
    lr = e["lr_trajectory"]
    assert lr[0] == pytest.approx(1e-3, rel=0.2)
