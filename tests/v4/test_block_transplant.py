"""V7-F54: cross-preset block transplant.

Proves brick modularity: take MoE (from a mixtral-style preset),
graft it into a llama-style attention→mlp chain, build, train.
"""

from __future__ import annotations

import pytest

from cppmega_v4.jsonrpc.schema import VerifyParams
from cppmega_v4.runner import Pipeline, run_pipeline


def _hybrid_spec() -> VerifyParams:
    """attention → moe (transplanted) → mlp."""
    return VerifyParams.model_validate({
        "graph": {
            "nodes": [
                {"id": "attn", "kind": "attention",
                 "params": {"num_heads": 4, "head_dim": 64}},
                {"id": "moe_xplant", "kind": "moe",
                 "params": {"num_experts": 4, "top_k": 2}},
                {"id": "mlp", "kind": "mlp", "params": {}},
            ],
            "edges": [
                {"src": "attn", "dst": "moe_xplant"},
                {"src": "moe_xplant", "dst": "mlp"},
            ],
        },
        "dim_env": {"B": 1, "S": 8, "H": 128,
                    "nh": 2, "nkv": 1, "head_dim": 64,
                    "num_experts": 4, "top_k": 2},
        "loss": {"kind": "cross_entropy", "head_outputs": ["mlp"]},
        "optim": {"kind": "adamw",
                  "groups": [{"matcher": "all", "lr": 1e-3,
                              "weight_decay": 0.01,
                              "betas": [0.9, 0.95]}]},
    })


def test_v7_f54_transplanted_moe_trains_in_llama_chain():
    rep = run_pipeline(_hybrid_spec(), Pipeline.from_dict({
        "stages": ["parse", "verify_build_spec", "build_model",
                   "train"],
        "stage_options": {"train": {"num_steps": 4}},
    }))
    tr = next(s for s in rep.stages if s.name == "train")
    assert tr.status == "ok", (
        f"transplanted graph failed: {tr.error}"
    )
    for L in tr.extras["losses"]:
        assert -1e10 < L < 1e10
    assert tr.extras["weight_delta_norm"] > 0


def test_v7_f54_moe_extras_surface_through_transplant():
    """The transplanted MoE still surfaces extras.moe — its routing
    metrics aren't lost just because it's in a foreign chain."""
    rep = run_pipeline(_hybrid_spec(), Pipeline.from_dict({
        "stages": ["parse", "verify_build_spec", "build_model",
                   "train"],
        "stage_options": {"train": {"num_steps": 2}},
    }))
    tr = next(s for s in rep.stages if s.name == "train")
    moe = tr.extras["moe"]
    assert moe is not None
    assert moe["num_experts"] == 4
    assert moe["top_k"] == 2
