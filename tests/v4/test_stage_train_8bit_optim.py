"""V7-D05: 8-bit quantised optimisers end-to-end through stage_train."""

from __future__ import annotations

import pytest

from cppmega_v4.jsonrpc.schema import VerifyParams
from cppmega_v4.runner import Pipeline, run_pipeline


def _spec(optim_kind: str = "adamw") -> VerifyParams:
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
        "optim": {"kind": optim_kind,
                  "groups": [{"matcher": "all", "lr": 1e-3,
                              "weight_decay": 0.01,
                              "betas": [0.9, 0.95]}]},
    })


def _train(spec, num_steps: int = 4) -> dict:
    rep = run_pipeline(spec, Pipeline.from_dict({
        "stages": ["parse", "verify_build_spec", "build_model", "train"],
        "stage_options": {"train": {"num_steps": num_steps}},
    }))
    tr = next(s for s in rep.stages if s.name == "train")
    assert tr.status == "ok", f"train failed: {tr.error}"
    return tr.extras


@pytest.mark.parametrize("kind", ["lion8bit", "adam8bit"])
def test_v7_d05_8bit_optim_runs_through_stage_train(kind):
    """Quantised optimiser kinds reach extras.optimizer_kind and
    produce a finite losses array."""
    extras = _train(_spec(optim_kind=kind))
    assert extras["optimizer_kind"] == kind
    assert len(extras["losses"]) == 4
    for L in extras["losses"]:
        assert -1e10 < L < 1e10


def test_v7_d05_lion8bit_weight_delta_positive():
    """Lion8bit must actually move weights — not silently fall back
    to zero update."""
    extras = _train(_spec(optim_kind="lion8bit"), num_steps=4)
    assert extras["weight_delta_norm"] > 0.0
