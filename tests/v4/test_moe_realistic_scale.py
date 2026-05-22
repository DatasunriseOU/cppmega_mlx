"""V7-A02: MoE at H>=512, num_experts>=8 through stage_train."""

from __future__ import annotations

import pytest

from cppmega_v4.jsonrpc.schema import VerifyParams
from cppmega_v4.runner import Pipeline, run_pipeline


def _moe_spec(H: int = 512, num_experts: int = 8,
              top_k: int = 2) -> VerifyParams:
    return VerifyParams.model_validate({
        "graph": {
            "nodes": [
                {"id": "attn", "kind": "attention",
                 "params": {"num_heads": 8, "head_dim": 64}},
                {"id": "moe", "kind": "moe",
                 "params": {"num_experts": num_experts,
                            "top_k": top_k}},
            ],
            "edges": [{"src": "attn", "dst": "moe"}],
        },
        "dim_env": {"B": 1, "S": 8, "H": H,
                    "nh": 8, "nkv": 4, "head_dim": 64,
                    "num_experts": num_experts, "top_k": top_k},
        "loss": {"kind": "cross_entropy", "head_outputs": ["moe"]},
        "optim": {"kind": "adamw",
                  "groups": [{"matcher": "all", "lr": 1e-3,
                              "weight_decay": 0.01,
                              "betas": [0.9, 0.95]}]},
    })


def _train(num_steps: int, H: int, num_experts: int, top_k: int) -> dict:
    rep = run_pipeline(
        _moe_spec(H, num_experts, top_k),
        Pipeline.from_dict({
            "stages": ["parse", "verify_build_spec", "build_model",
                       "train"],
            "stage_options": {"train": {"num_steps": num_steps}},
        }),
    )
    tr = next(s for s in rep.stages if s.name == "train")
    assert tr.status == "ok", f"train failed: {tr.error}"
    return tr.extras


def test_v7_a02_moe_h512_8experts_topk2_runs():
    e = _train(num_steps=2, H=512, num_experts=8, top_k=2)
    moe = e["moe"]
    assert moe["num_experts"] == 8
    assert moe["top_k"] == 2
    assert moe["routing_entropy"] is not None
    assert moe["routing_entropy"] > 0


def test_v7_a02_moe_top_k_4_grad_path():
    """top_k>2 still produces finite losses + non-zero weight delta."""
    e = _train(num_steps=2, H=256, num_experts=8, top_k=4)
    assert e["moe"]["top_k"] == 4
    for L in e["losses"]:
        assert -1e10 < L < 1e10
    assert e["weight_delta_norm"] > 0.0
