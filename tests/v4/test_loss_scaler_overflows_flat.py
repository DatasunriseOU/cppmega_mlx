"""V7-Q06.2: pin the flat extras.loss_scaler_overflows key.

The nested `loss_scaler.overflow_steps` was preserved for back-compat,
but UI reads the flat key — assert both shapes coexist when a scaler
is configured, and the flat key is an empty list when no scaler.
"""

from __future__ import annotations

from cppmega_v4.jsonrpc.schema import VerifyParams
from cppmega_v4.runner import Pipeline, run_pipeline


def _spec(loss_scaler: bool) -> VerifyParams:
    optim = {
        "kind": "adamw",
        "groups": [{"matcher": "all", "lr": 1e-3,
                    "weight_decay": 0.01, "betas": [0.9, 0.95]}],
    }
    return VerifyParams.model_validate({
        "graph": {
            "nodes": [
                {"id": "attn", "kind": "attention",
                 "params": {"num_heads": 2, "head_dim": 64}},
                {"id": "mlp", "kind": "mlp", "params": {}},
            ],
            "edges": [{"src": "attn", "dst": "mlp"}],
        },
        "dim_env": {"B": 1, "S": 8, "H": 128,
                    "nh": 2, "nkv": 1, "head_dim": 64},
        "loss": {"kind": "cross_entropy", "head_outputs": ["mlp"]},
        "optim": optim,
    })


def _run(opts: dict) -> dict:
    rep = run_pipeline(_spec(loss_scaler=True), Pipeline.from_dict({
        "stages": ["parse", "verify_build_spec", "build_model", "train"],
        "stage_options": {"train": opts},
    }))
    tr = next(s for s in rep.stages if s.name == "train")
    assert tr.status == "ok", tr.error
    return dict(tr.extras or {})


def test_no_scaler_flat_key_empty() -> None:
    """No loss_scaler configured -> extras.loss_scaler is None +
    extras.loss_scaler_overflows is []."""
    extras = _run({"num_steps": 2, "seed": 7})
    assert extras.get("loss_scaler") is None
    # Flat key always present (empty list when no scaler).
    assert extras.get("loss_scaler_overflows") == []


def test_scaler_active_both_keys_coexist() -> None:
    """When fp16 master_dtype triggers loss_scaler, both nested and
    flat keys hold the same overflow_steps list."""
    extras = _run({
        "num_steps": 3, "seed": 7,
        "master_dtype": "fp16",
        "loss_scaler_mode": "dynamic",
        "loss_scaler_init": 2.0,
    })
    nested = extras.get("loss_scaler")
    flat = extras.get("loss_scaler_overflows")
    assert nested is not None, (
        f"expected loss_scaler dict when fp16 active; got extras keys "
        f"{sorted(extras)}"
    )
    # Nested carries the overflow_steps list; flat must mirror it.
    assert nested.get("overflow_steps") == flat
    assert isinstance(flat, list)
