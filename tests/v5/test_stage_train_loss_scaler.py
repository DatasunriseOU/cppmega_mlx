"""V7-D03: stage_train integrates LossScaler when master_dtype='fp16'.

The honest-closure gap: LossScaler class lived in
cppmega_v4/runtime/loss_scaler.py but stage_train never touched it.
After D03 wiring, opts.master_dtype='fp16' must surface
extras.loss_scaler = {mode, scale, overflow_count, overflow_steps,...}.
Other dtype settings (bf16/fp32) must leave it as None.
"""

from __future__ import annotations

from cppmega_v4.jsonrpc.schema import VerifyParams
from cppmega_v4.runner import Pipeline, run_pipeline


def _spec() -> VerifyParams:
    return VerifyParams.model_validate({
        "graph": {
            "nodes": [
                {"id": "attn", "kind": "attention", "params": {}},
                {"id": "mlp", "kind": "mlp",
                 "params": {"intermediate_size": 64, "activation": "swiglu"}},
            ],
            "edges": [{"src": "attn", "dst": "mlp"}],
        },
        "dim_env": {"B": 1, "S": 8, "H": 32, "nh": 2, "nkv": 1,
                    "head_dim": 16},
        "loss": {"kind": "cross_entropy", "head_outputs": ["mlp"]},
        "optim": {"kind": "adamw",
                  "groups": [{"matcher": "all", "lr": 1e-3,
                              "weight_decay": 0.01, "betas": [0.9, 0.95]}]},
    })


def _run(master_dtype: str, num_steps: int = 2,
         loss_scaler_init: float | None = None) -> dict:
    train_opts: dict = {"num_steps": num_steps,
                         "master_dtype": master_dtype}
    if loss_scaler_init is not None:
        train_opts["loss_scaler_init"] = loss_scaler_init
    report = run_pipeline(_spec(), Pipeline.from_dict({
        "stages": ["parse", "verify_build_spec", "build_model", "train"],
        "stage_options": {"train": train_opts},
    }))
    train = next(s for s in report.stages if s.name == "train")
    assert train.status == "ok", f"stage_train failed: {train.error}"
    return train.extras


def test_fp32_master_dtype_no_loss_scaler():
    extras = _run("fp32")
    assert extras["loss_scaler"] is None


def test_bf16_master_dtype_no_loss_scaler():
    extras = _run("bf16")
    assert extras["loss_scaler"] is None


def test_fp16_master_dtype_engages_loss_scaler():
    # init_scale=1.0 keeps the math identical to no-scaling, so weights
    # still move while we verify the scaler engaged and reports.
    extras = _run("fp16", num_steps=3, loss_scaler_init=1.0)
    scaler = extras["loss_scaler"]
    assert scaler is not None
    assert scaler["mode"] == "dynamic"
    assert scaler["scale"] > 0
    assert isinstance(scaler["overflow_count"], int)
    assert scaler["overflow_count"] >= 0
    assert isinstance(scaler["overflow_steps"], list)
    for s in scaler["overflow_steps"]:
        assert isinstance(s, int) and 0 <= s < 3
    # snapshot fields surface for the UI overlay.
    assert "clean_steps_since_overflow" in scaler


def test_fp16_loss_scaler_overflow_mode_static():
    """Static mode does not back off the scale on overflow but still
    increments overflow_count — the UI overlay's main signal."""
    train_opts = {"num_steps": 2, "master_dtype": "fp16",
                  "loss_scaler_init": 1.0,
                  "loss_scaler_mode": "static"}
    report = run_pipeline(_spec(), Pipeline.from_dict({
        "stages": ["parse", "verify_build_spec", "build_model", "train"],
        "stage_options": {"train": train_opts},
    }))
    train = next(s for s in report.stages if s.name == "train")
    assert train.status == "ok", f"stage_train failed: {train.error}"
    scaler = train.extras["loss_scaler"]
    assert scaler["mode"] == "static"
