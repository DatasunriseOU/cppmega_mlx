"""H15: per-rank shard simulation in stage_train.

Asserts that:
  * sharding_applied.per_rank_param_bytes is populated.
  * an 8-way shard halves (~÷8) the per-rank param bytes vs unsharded.
  * The loss trajectory matches bit-identical (≤ 1e-6 abs diff) between
    sharded and unsharded — sharding only changes the byte accounting,
    not the math.
"""

from __future__ import annotations

import pytest

from cppmega_v4.jsonrpc.schema import VerifyParams
from cppmega_v4.runner import Pipeline, run_pipeline


def _spec(*, axis_assignments: list[dict] | None = None) -> VerifyParams:
    d = {
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
    }
    if axis_assignments is not None:
        d["sharding"] = {
            "topology": {"factory": "h100_8x", "kwargs": {}},
            "axis_assignments": axis_assignments,
            "compile_mode": "regional",
            "fp8_enabled": False,
        }
    return VerifyParams.model_validate(d)


def _train(spec: VerifyParams) -> dict:
    rep = run_pipeline(spec, Pipeline.from_dict({
        "stages": ["parse", "verify_build_spec", "build_model", "train"],
        "stage_options": {"train": {"num_steps": 2}},
    }))
    tr = next(s for s in rep.stages if s.name == "train")
    assert tr.status == "ok", f"train failed: {tr.error}"
    return tr.extras


def test_h15_sharding_applied_carries_per_rank_param_bytes():
    """An FSDP-axis assignment populates per_rank_param_bytes."""
    spec = _spec(axis_assignments=[
        {"axis_name": "dp", "kind": "fsdp2", "degree": 8},
    ])
    extras = _train(spec)
    sa = extras["sharding_applied"]
    assert sa is not None
    assert sa["shard_dim"] == 8
    assert sa["per_rank_param_bytes"] > 0
    assert sa["per_rank_activation_bytes"] > 0
    assert sa["total_param_bytes"] > 0
    # 8-way shard divides ~/8.
    assert sa["per_rank_param_bytes"] == sa["total_param_bytes"] // 8


def test_h15_8way_shard_halves_per_rank_vs_2way():
    """Doubling shard_dim halves per_rank bytes (linear)."""
    spec_2 = _spec(axis_assignments=[
        {"axis_name": "dp", "kind": "fsdp2", "degree": 2},
    ])
    spec_8 = _spec(axis_assignments=[
        {"axis_name": "dp", "kind": "fsdp2", "degree": 8},
    ])
    sa2 = _train(spec_2)["sharding_applied"]
    sa8 = _train(spec_8)["sharding_applied"]
    # 8/2 = 4× smaller per-rank.
    assert sa2["per_rank_param_bytes"] // 4 == sa8["per_rank_param_bytes"]


def test_h15_loss_bit_identical_sharded_vs_unsharded():
    """Sharding is bookkeeping only — losses must match bit-identical
    (synthetic data + same seeds). Tolerance ≤ 1e-6."""
    losses_unsharded = _train(_spec())["losses"]
    losses_sharded = _train(_spec(axis_assignments=[
        {"axis_name": "dp", "kind": "fsdp2", "degree": 8},
    ]))["losses"]
    assert len(losses_unsharded) == len(losses_sharded)
    for u, s in zip(losses_unsharded, losses_sharded):
        assert abs(u - s) < 1e-6, (
            f"sharding shifted loss: unsharded={u}, sharded={s}")
