"""V7-B06: per-device memory profile under FSDP shard."""

from __future__ import annotations

import pytest

from cppmega_v4.jsonrpc.schema import VerifyParams
from cppmega_v4.runner import Pipeline, run_pipeline


def _spec_with_shard(degree: int) -> VerifyParams:
    return VerifyParams.model_validate({
        "graph": {
            "nodes": [
                {"id": "attn", "kind": "attention",
                 "params": {"num_heads": 4, "head_dim": 64}},
                {"id": "mlp", "kind": "mlp", "params": {}},
            ],
            "edges": [{"src": "attn", "dst": "mlp"}],
        },
        "dim_env": {"B": 1, "S": 8, "H": 256,
                    "nh": 4, "nkv": 2, "head_dim": 64},
        "loss": {"kind": "cross_entropy", "head_outputs": ["mlp"]},
        "optim": {"kind": "adamw",
                  "groups": [{"matcher": "all", "lr": 1e-3,
                              "weight_decay": 0.01,
                              "betas": [0.9, 0.95]}]},
        "sharding": {
            "topology": {"factory": "h100_8x", "kwargs": {}},
            "axis_assignments": [
                {"axis_name": "dp", "kind": "fsdp2",
                 "degree": degree},
            ],
            "compile_mode": "regional",
            "fp8_enabled": False,
        },
    })


def _train(spec, num_steps: int = 2) -> dict:
    rep = run_pipeline(spec, Pipeline.from_dict({
        "stages": ["parse", "verify_build_spec", "build_model",
                   "train"],
        "stage_options": {"train": {"num_steps": num_steps}},
    }))
    tr = next(s for s in rep.stages if s.name == "train")
    assert tr.status == "ok", f"train failed: {tr.error}"
    return tr.extras


@pytest.mark.parametrize("degree", [1, 2, 4, 8])
def test_v7_b06_per_rank_param_bytes_falls_linearly_with_degree(degree):
    """8-way FSDP shards params into 1/8th per rank, 4-way 1/4th, etc."""
    extras = _train(_spec_with_shard(degree))
    sa = extras["sharding_applied"]
    assert sa is not None
    assert sa["shard_dim"] == degree
    assert sa["per_rank_param_bytes"] == (
        sa["total_param_bytes"] // degree)


def test_v7_b06_8_way_vs_2_way_per_rank_ratio_4x():
    e2 = _train(_spec_with_shard(2))["sharding_applied"]
    e8 = _train(_spec_with_shard(8))["sharding_applied"]
    assert e2["per_rank_param_bytes"] // 4 == e8["per_rank_param_bytes"]


def test_v7_b06_per_rank_activation_bytes_falls_too():
    e1 = _train(_spec_with_shard(1))["sharding_applied"]
    e8 = _train(_spec_with_shard(8))["sharding_applied"]
    assert e1["per_rank_activation_bytes"] > e8["per_rank_activation_bytes"]
    # 1-way / 8-way ratio ≈ 8
    assert (e1["per_rank_activation_bytes"]
            // e8["per_rank_activation_bytes"]) == 8
