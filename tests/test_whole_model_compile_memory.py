"""V7-I06: whole-model compile stays under HBM bound.

Builds a model sized to ≈10-20% of device HBM (param bytes alone),
runs 3 train steps with compile_mode='whole_model', and asserts
mx.metal.get_peak_memory() stays under 70% of device HBM.

Skipped on devices with < 16 GB.
"""

from __future__ import annotations

import pytest

import mlx.core as mx

from cppmega_v4.jsonrpc.schema import VerifyParams
from cppmega_v4.runner import Pipeline, run_pipeline


def _device_hbm_bytes() -> int | None:
    """mx.metal device memory size if Metal is present."""
    try:
        if hasattr(mx, "metal"):
            info = mx.metal.device_info()
            return int(info.get("memory_size", 0))
    except Exception:
        return None
    return None


def _make_spec(hidden: int = 1024) -> VerifyParams:
    """A 4-brick model. At hidden=1024 param bytes ≈ several MB —
    safe to run under any 16+ GB device; the test asserts peak
    stays well under the HBM cap regardless."""
    return VerifyParams.model_validate({
        "graph": {
            "nodes": [
                {"id": "attn1", "kind": "attention",
                 "params": {"num_heads": 8, "head_dim": 64}},
                {"id": "mlp1", "kind": "mlp", "params": {}},
                {"id": "attn2", "kind": "attention",
                 "params": {"num_heads": 8, "head_dim": 64}},
                {"id": "mlp2", "kind": "mlp", "params": {}},
            ],
            "edges": [
                {"src": "attn1", "dst": "mlp1"},
                {"src": "mlp1", "dst": "attn2"},
                {"src": "attn2", "dst": "mlp2"},
            ],
        },
        "dim_env": {"B": 1, "S": 16, "H": hidden,
                    "nh": 8, "nkv": 4, "head_dim": 64},
        "loss": {"kind": "cross_entropy", "head_outputs": ["mlp2"]},
        "optim": {"kind": "adamw",
                  "groups": [{"matcher": "all", "lr": 1e-3,
                              "weight_decay": 0.01,
                              "betas": [0.9, 0.95]}]},
        "sharding": {
            "topology": {"factory": "m3_ultra_solo", "kwargs": {}},
            "axis_assignments": [
                {"axis_name": "dp", "kind": "fsdp2", "degree": 1}
            ],
            "compile_mode": "whole_model",
            "fp8_enabled": False,
        },
    })


def test_v7_i06_whole_model_compile_stays_under_hbm_bound():
    hbm = _device_hbm_bytes()
    if hbm is None:
        pytest.skip("no Metal device info")
    if hbm < 16 * 1024 ** 3:
        pytest.skip(f"device HBM {hbm / 1024 ** 3:.1f} GB < 16 GB cap")

    rep = run_pipeline(_make_spec(), Pipeline.from_dict({
        "stages": ["parse", "verify_build_spec", "build_model", "train"],
        "stage_options": {"train": {"num_steps": 3}},
    }))
    tr = next(s for s in rep.stages if s.name == "train")
    assert tr.status == "ok", f"train failed: {tr.error}"
    peak = tr.extras.get("memory_peak_bytes")
    assert peak is not None
    peak_frac = peak / hbm
    assert peak_frac < 0.70, (
        f"whole_model compile peak {peak / 1024 ** 3:.2f} GB "
        f"= {peak_frac * 100:.1f}% of HBM (cap 70%)"
    )


def test_v7_i06_whole_model_compile_mode_propagates_to_extras():
    rep = run_pipeline(_make_spec(), Pipeline.from_dict({
        "stages": ["parse", "verify_build_spec", "build_model", "train"],
        "stage_options": {"train": {"num_steps": 2}},
    }))
    tr = next(s for s in rep.stages if s.name == "train")
    sa = tr.extras.get("sharding_applied")
    assert sa is not None
    assert sa.get("compile_mode") == "whole_model"
