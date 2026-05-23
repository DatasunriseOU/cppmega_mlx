"""V7-I01 / item 52: stage_train actually wraps loss_and_grad with
mx.compile when sharding.compile_mode='whole_model' — not just a
metadata echo.

Direct (no-subprocess) gate that loads the pipeline, runs N steps,
and asserts extras.sharding_applied.compile_engaged==True plus a
measurable first-vs-warm gap.
"""

from __future__ import annotations



from cppmega_v4.jsonrpc.schema import VerifyParams
from cppmega_v4.runner import Pipeline, run_pipeline


def _spec(compile_mode: str = "whole_model") -> VerifyParams:
    return VerifyParams.model_validate({
        "graph": {
            "nodes": [
                {"id": "attn", "kind": "attention",
                 "params": {"num_heads": 4, "head_dim": 64}},
                {"id": "mlp", "kind": "mlp", "params": {}},
            ],
            "edges": [{"src": "attn", "dst": "mlp"}],
        },
        "dim_env": {"B": 1, "S": 32, "H": 128,
                    "nh": 2, "nkv": 1, "head_dim": 64},
        "loss": {"kind": "cross_entropy", "head_outputs": ["mlp"]},
        "optim": {"kind": "adamw",
                  "groups": [{"matcher": "all", "lr": 1e-3,
                              "weight_decay": 0.01,
                              "betas": [0.9, 0.95]}]},
        "sharding": {
            "topology": {"factory": "m3_ultra_solo", "kwargs": {}},
            "axis_assignments": [
                {"axis_name": "dp", "kind": "fsdp2", "degree": 1}
            ],
            "compile_mode": compile_mode,
            "fp8_enabled": False,
        },
    })


def _run(num_steps: int, compile_mode: str) -> dict:
    rep = run_pipeline(_spec(compile_mode), Pipeline.from_dict({
        "stages": ["parse", "verify_build_spec", "build_model", "train"],
        "stage_options": {"train": {"num_steps": num_steps}},
    }))
    tr = next(s for s in rep.stages if s.name == "train")
    assert tr.status == "ok", f"train failed: {tr.error}"
    return tr.extras


def test_item52_whole_model_compile_engagement_proven_by_extras():
    """compile_mode='whole_model' must set extras.sharding_applied
    .compile_engaged=True and surface real per-step timing."""
    extras = _run(num_steps=6, compile_mode="whole_model")
    sa = extras["sharding_applied"]
    assert sa["compile_mode"] == "whole_model"
    assert sa["compile_engaged"] is True
    assert sa["compile_status"] == "engaged"
    assert sa["compile_error"] is None
    per_step = sa["per_step_ms"]
    assert len(per_step) == 6
    first = float(per_step[0])
    warm_mean = sum(per_step[1:]) / 5.0
    # mx.compile takes the build cost on the first step. We assert
    # first is >=2x slower than the warm mean — empirically the gap
    # is 10-200x on Metal but we keep the bound conservative so the
    # test is robust to bench noise on tiny shapes.
    assert first / max(1e-9, warm_mean) >= 2.0, (
        f"first={first}ms warm_mean={warm_mean}ms per_step={per_step}")


def test_item52_off_mode_records_no_compile_engagement():
    """compile_mode='off' must keep compile_engaged=False."""
    extras = _run(num_steps=3, compile_mode="off")
    sa = extras["sharding_applied"]
    assert sa["compile_engaged"] is False
    assert sa["compile_status"] == "off"


def test_item52_regional_mode_records_regional_status():
    """compile_mode='regional' is delegated to per-brick logic; the
    train stage records status='regional' and does NOT wrap the
    whole graph (compile_engaged stays False at this layer)."""
    extras = _run(num_steps=3, compile_mode="regional")
    sa = extras["sharding_applied"]
    assert sa["compile_engaged"] is False
    assert sa["compile_status"] == "regional"
