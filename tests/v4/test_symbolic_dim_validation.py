"""V7-F56: symbolic-dim validation — incompatible (H, nh, head_dim)."""

from __future__ import annotations

import pytest

from cppmega_v4.jsonrpc.schema import VerifyParams
from cppmega_v4.runner import Pipeline, run_pipeline


def _spec(H: int, num_heads: int, head_dim: int) -> VerifyParams:
    return VerifyParams.model_validate({
        "graph": {
            "nodes": [
                {"id": "attn", "kind": "attention",
                 "params": {"num_heads": num_heads,
                            "head_dim": head_dim}},
                {"id": "mlp", "kind": "mlp", "params": {}},
            ],
            "edges": [{"src": "attn", "dst": "mlp"}],
        },
        "dim_env": {"B": 1, "S": 8, "H": H,
                    "nh": num_heads, "nkv": 1, "head_dim": head_dim},
        "loss": {"kind": "cross_entropy", "head_outputs": ["mlp"]},
        "optim": {"kind": "adamw",
                  "groups": [{"matcher": "all", "lr": 1e-3,
                              "weight_decay": 0.01,
                              "betas": [0.9, 0.95]}]},
    })


def _verify(spec: VerifyParams):
    return run_pipeline(spec, Pipeline.from_dict({
        "stages": ["parse", "verify_build_spec"],
    }))


def test_v7_f56_compatible_combo_h128_nh8_hd16_passes():
    """H=128 = nh(8) * head_dim(16). verify ok."""
    rep = _verify(_spec(H=128, num_heads=8, head_dim=16))
    vbs = next(s for s in rep.stages if s.name == "verify_build_spec")
    assert vbs.status == "ok", f"compatible combo rejected: {vbs.error}"


def test_v7_f56_compatible_combo_h512_nh8_hd64_passes():
    rep = _verify(_spec(H=512, num_heads=8, head_dim=64))
    vbs = next(s for s in rep.stages if s.name == "verify_build_spec")
    assert vbs.status == "ok"


@pytest.mark.xfail(strict=True, reason=(
    "V7-F56 honest finding: build_model accepts H=128 with nh=3, "
    "head_dim=50 silently (num_heads*head_dim ≠ H). The constructor "
    "lacks a symbolic-dim validator at verify_build_spec. Marked xfail "
    "to track the bug; flipping to non-xfail when the validator lands."
))
def test_v7_f56_incompatible_h128_nh3_hd50_train_fails_loudly():
    """Should fail with a clear error at verify_build_spec or
    build_model — today it silently produces a broken model."""
    spec = _spec(H=128, num_heads=3, head_dim=50)
    rep = run_pipeline(spec, Pipeline.from_dict({
        "stages": ["parse", "verify_build_spec", "build_model",
                   "dry_forward"],
    }))
    statuses = {s.name: s.status for s in rep.stages}
    assert "fail" in statuses.values(), (
        f"incompatible combo passed silently: {statuses}"
    )


def test_v7_f56_dim_env_H_mismatched_with_nh_times_head_dim_observable():
    """Build the spec and verify the bookkeeping math at the
    Python level — proves the contract the validator SHOULD enforce
    is observable to the gate."""
    spec = _spec(H=128, num_heads=3, head_dim=50)
    de = spec.dim_env if isinstance(
        spec.dim_env, dict) else spec.dim_env.model_dump()
    H = de["H"]
    nh = de["nh"]
    hd = de["head_dim"]
    assert nh * hd != H, (
        f"V7-F56 premise broken: nh*hd ({nh}*{hd}) accidentally "
        f"equals H ({H})"
    )
