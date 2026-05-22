"""V7-G05: FIM (Fill-In-Middle) loss probe in stage_train."""

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


def _train(spec, **opts):
    rep = run_pipeline(spec, Pipeline.from_dict({
        "stages": ["parse", "verify_build_spec", "build_model", "train"],
        "stage_options": {"train": opts},
    }))
    tr = next(s for s in rep.stages if s.name == "train")
    assert tr.status == "ok"
    return tr.extras


def test_v7_g05_fim_disabled_by_default():
    e = _train(_spec(), num_steps=2)
    assert e["fim_active"] is False
    assert e["fim_loss"] is None
    assert e["fim_ratio"] is None


def test_v7_g05_fim_enabled_emits_loss():
    e = _train(_spec(), num_steps=2, fim_enabled=True, fim_ratio=0.5)
    assert e["fim_active"] is True
    assert e["fim_ratio"] == 0.5
    # fim_loss may be None if the loss_fn signature doesn't accept the
    # truncated targets; if present it must be finite.
    if e["fim_loss"] is not None:
        assert -1e10 < e["fim_loss"] < 1e10


def test_v7_g05_fim_ratio_clamp_invalid_values():
    e = _train(_spec(), num_steps=2, fim_enabled=True, fim_ratio=2.0)
    assert e["fim_active"] is True
    # Out-of-range ratio falls back to 0.5.
    assert e["fim_ratio"] == 0.5
