"""H16: real mlx dtype switching in stage_train.

Asserts that:
  * master_dtype="fp32" actually casts module params to float32.
  * master_dtype="bf16" casts to bfloat16.
  * fp8_active=True attempts fp8 and records honest fp8_fallback_reason
    when the platform lacks fp8 support.
"""

from __future__ import annotations

import pytest

from cppmega_v4.jsonrpc.schema import VerifyParams
from cppmega_v4.runner import Pipeline, run_pipeline


def _spec(*, mixed_precision: bool = True,
           fp8_enabled: bool = False) -> VerifyParams:
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
                  "mixed_precision": mixed_precision,
                  "groups": [{"matcher": "all", "lr": 1e-3,
                              "weight_decay": 0.01,
                              "betas": [0.9, 0.95]}]},
        "sharding": {
            "topology": {"factory": "h100_8x", "kwargs": {}},
            "axis_assignments": [
                {"axis_name": "dp", "kind": "fsdp2", "degree": 1}
            ],
            "compile_mode": "regional",
            "fp8_enabled": fp8_enabled,
        },
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


def test_h16_dtype_actual_populated_for_fp32_master():
    """master_dtype='fp32' (mixed_precision=True) → dtype_actual
    shows the post-cast param dtype is float32."""
    extras = _train(_spec(mixed_precision=True))
    da = extras["dtype_actual"]
    assert da["master_dtype_requested"] == "fp32"
    assert da.get("master_dtype_actual", "").endswith("float32")
    assert da["fp8_attempted"] is False
    assert da["fp8_fallback_reason"] is None
    assert da["train_dtype_actual"] == "bf16"


def test_h16_dtype_actual_populated_for_bf16_master():
    """master_dtype='bf16' (mixed_precision=False) → params cast to bf16."""
    extras = _train(_spec(mixed_precision=False))
    da = extras["dtype_actual"]
    assert da["master_dtype_requested"] == "bf16"
    actual = da.get("master_dtype_actual", "")
    assert actual.endswith("bfloat16") or actual.endswith("bf16"), (
        f"expected bf16 cast, got {actual!r}"
    )


def test_h16_fp8_attempt_records_honest_fallback_on_unsupported():
    """fp8_enabled=True → fp8_attempted true; if mx has no fp8 backend
    on this build, fp8_fallback_reason is populated and train_dtype
    drops back to bf16."""
    extras = _train(_spec(fp8_enabled=True))
    da = extras["dtype_actual"]
    assert da["fp8_attempted"] is True
    # On platforms without fp8, fallback is recorded.
    reason = da.get("fp8_fallback_reason")
    if reason is not None:
        assert isinstance(reason, str) and len(reason) > 0
        assert da["train_dtype_actual"] == "bf16"


def test_h16_fp32_master_vs_bf16_master_loss_differs_or_changes_param_dtype():
    """Switching master dtype must actually change SOMETHING the
    downstream observer can latch onto — at minimum the actual cast
    dtype string. (Loss may or may not differ depending on whether the
    op set rounds intermediate accumulations.)"""
    fp32 = _train(_spec(mixed_precision=True))["dtype_actual"]
    bf16 = _train(_spec(mixed_precision=False))["dtype_actual"]
    assert fp32.get("master_dtype_actual") != bf16.get("master_dtype_actual")
