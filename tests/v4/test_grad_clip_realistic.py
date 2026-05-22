"""V7-A09: grad clip activates under realistic gradient norms."""

from __future__ import annotations

import pytest

from cppmega_v4.jsonrpc.schema import VerifyParams
from cppmega_v4.runner import Pipeline, run_pipeline


def _spec(grad_clip: float = 0.01) -> VerifyParams:
    """Tiny clip threshold + large H to force clip activations."""
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
                  "gradient_clip_norm": grad_clip,
                  "groups": [{"matcher": "all", "lr": 1e-3,
                              "weight_decay": 0.01,
                              "betas": [0.9, 0.95]}]},
    })


def _train(num_steps: int = 4, grad_clip: float = 0.01) -> dict:
    rep = run_pipeline(_spec(grad_clip), Pipeline.from_dict({
        "stages": ["parse", "verify_build_spec", "build_model", "train"],
        "stage_options": {"train": {"num_steps": num_steps}},
    }))
    tr = next(s for s in rep.stages if s.name == "train")
    assert tr.status == "ok", f"train failed: {tr.error}"
    return tr.extras


def test_v7_a09_clip_activates_at_low_threshold():
    e = _train(num_steps=4, grad_clip=0.01)
    clip = e.get("gradient_clip")
    assert clip is not None
    # max grad norm seen exceeds the tiny threshold → clips happen.
    assert clip["max_grad_norm_seen"] > 0.01
    assert clip["num_clips"] >= 1


def test_v7_a09_clip_inactive_at_high_threshold():
    e = _train(num_steps=4, grad_clip=1e9)
    clip = e["gradient_clip"]
    assert clip["num_clips"] == 0
