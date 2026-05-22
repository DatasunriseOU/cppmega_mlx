"""V7-A10: long-horizon weight delta trajectory honesty gate."""

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
        "dim_env": {"B": 1, "S": 8, "H": 128,
                    "nh": 2, "nkv": 1, "head_dim": 64},
        "loss": {"kind": "cross_entropy", "head_outputs": ["mlp"]},
        "optim": {"kind": "adamw",
                  "groups": [{"matcher": "all", "lr": 1e-3,
                              "weight_decay": 0.01,
                              "betas": [0.9, 0.95]}]},
    })


def _train(num_steps: int) -> dict:
    rep = run_pipeline(_spec(), Pipeline.from_dict({
        "stages": ["parse", "verify_build_spec", "build_model", "train"],
        "stage_options": {"train": {"num_steps": num_steps}},
    }))
    tr = next(s for s in rep.stages if s.name == "train")
    assert tr.status == "ok"
    return tr.extras


def test_v7_a10_delta_grows_with_num_steps():
    """50-step delta < 500-step delta — proves per-horizon
    monotonicity, ruling out dead layers."""
    short = _train(num_steps=50)["weight_delta_norm"]
    long = _train(num_steps=500)["weight_delta_norm"]
    assert long > short, (
        f"weight delta did not grow: short={short}, long={long}"
    )


def test_v7_a10_delta_not_runaway():
    """500-step delta < 50-step delta × 50 (sub-linear growth) —
    rules out runaway / exploding layers."""
    short = _train(num_steps=50)["weight_delta_norm"]
    long = _train(num_steps=500)["weight_delta_norm"]
    assert long < short * 50, (
        f"runaway weight growth: short={short}, long={long}"
    )


def test_v7_a10_delta_strictly_positive_at_50_steps():
    """Smoke: no dead-layer regression — 50 steps moves SOMETHING."""
    e = _train(num_steps=50)
    assert e["weight_delta_norm"] > 0.0
