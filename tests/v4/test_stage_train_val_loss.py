"""V7-A04: validation loss surfaced alongside train loss."""

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


def _train(**opts) -> dict:
    rep = run_pipeline(_spec(), Pipeline.from_dict({
        "stages": ["parse", "verify_build_spec", "build_model", "train"],
        "stage_options": {"train": opts},
    }))
    tr = next(s for s in rep.stages if s.name == "train")
    assert tr.status == "ok", f"train failed: {tr.error}"
    return tr.extras


def test_v7_a04_val_every_zero_emits_no_val_losses():
    e = _train(num_steps=10)
    assert e["val_every"] == 0
    assert e["val_losses"] == []


def test_v7_a04_val_every_5_emits_val_losses_at_cadence():
    e = _train(num_steps=20, val_every=5)
    assert e["val_every"] == 5
    # Cadence: step 5, 10, 15, 20 → 4 entries.
    assert len(e["val_losses"]) == 4
    for v in e["val_losses"]:
        assert -1e10 < v < 1e10


def test_v7_a04_val_losses_finite():
    e = _train(num_steps=15, val_every=3)
    assert len(e["val_losses"]) == 5
    for v in e["val_losses"]:
        assert v == v  # not NaN
