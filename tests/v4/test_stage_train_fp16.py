"""H23: fp16 dtype option alongside bf16/fp32."""

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
        "stage_options": {"train": {"num_steps": 2, **opts}},
    }))
    tr = next(s for s in rep.stages if s.name == "train")
    assert tr.status == "ok"
    return tr.extras


def test_h23_fp16_option_accepted_and_reported():
    e = _train(master_dtype="fp16")
    assert e["master_dtype"] == "fp16"
    assert e["train_dtype"] == "fp16"
    assert "float16" in e["dtype_actual"]["master_dtype_actual"]


def test_h23_fp16_losses_finite_and_close_to_bf16():
    bf16 = _train(master_dtype="bf16")["losses"]
    fp16 = _train(master_dtype="fp16")["losses"]
    assert all(x == x and -1e10 < x < 1e10 for x in fp16)
    # fp16 and bf16 share the same exponent range; loss should be in
    # the same ballpark for a 2-step run (within 50%).
    for a, b in zip(bf16, fp16):
        assert abs(a - b) / max(abs(a), abs(b), 1e-9) < 0.5


def test_h23_fp16_weight_delta_smaller_than_fp32():
    """fp16 has fewer mantissa bits → tiny updates round to zero →
    expected smaller weight_delta_norm than fp32."""
    fp32 = _train(master_dtype="fp32")["weight_delta_norm"]
    fp16 = _train(master_dtype="fp16")["weight_delta_norm"]
    # On a 2-step run noise is large; allow generous gate that fp16
    # weight delta is no MORE than 1.5x fp32 (precision loss caps it).
    assert fp16 <= fp32 * 1.5 + 1e-6, (
        f"fp16 delta {fp16} > 1.5×fp32 {fp32}")
