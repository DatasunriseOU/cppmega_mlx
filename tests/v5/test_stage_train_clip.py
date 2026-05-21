"""G23: gradient_clip_norm activation observable in extras.gradient_clip.

V4-1..V4 had OptimSpec.gradient_clip_norm field but stage_train never
applied it. extras.gradient_clip = {threshold, max_grad_norm_seen,
num_clips} now reports observed clipping behaviour.
"""

from __future__ import annotations

import pytest

from cppmega_v4.jsonrpc.schema import VerifyParams
from cppmega_v4.runner import Pipeline, run_pipeline


def _spec_with_optim(clip: float | None) -> VerifyParams:
    return VerifyParams.model_validate({
        "graph": {
            "nodes": [
                {"id": "attn", "kind": "attention", "params": {}},
                {"id": "mlp", "kind": "mlp",
                 "params": {"intermediate_size": 64, "activation": "swiglu"}},
            ],
            "edges": [{"src": "attn", "dst": "mlp"}],
        },
        "dim_env": {"B": 1, "S": 8, "H": 32, "nh": 2, "nkv": 1, "head_dim": 16},
        "loss": {"kind": "cross_entropy", "head_outputs": ["mlp"]},
        "optim": {"kind": "adamw",
                  "gradient_clip_norm": clip,
                  "groups": [{"matcher": "all", "lr": 1e-3,
                              "weight_decay": 0.01, "betas": [0.9, 0.95]}]},
    })


def _run(clip: float | None, num_steps: int = 4) -> dict:
    spec = _spec_with_optim(clip)
    report = run_pipeline(spec, Pipeline.from_dict({
        "stages": ["parse", "verify_build_spec", "build_model", "train"],
        "stage_options": {"train": {"num_steps": num_steps}},
    }))
    train = next(s for s in report.stages if s.name == "train")
    assert train.status == "ok", f"stage_train failed: {train.error}"
    return train.extras


def test_clip_extras_populated_with_threshold_1():
    extras = _run(clip=1.0)
    gc = extras["gradient_clip"]
    assert gc["threshold"] == 1.0
    assert gc["max_grad_norm_seen"] > 0


def test_clip_none_disables():
    """clip=None disables — max_grad_norm_seen stays 0 (loop skipped)."""
    extras = _run(clip=None)
    gc = extras["gradient_clip"]
    assert gc["threshold"] is None
    assert gc["max_grad_norm_seen"] == 0.0
    assert gc["num_clips"] == 0


def test_clip_tight_threshold_triggers_clipping():
    """clip=0.001 is far below typical grad norm → num_clips > 0."""
    extras = _run(clip=0.001, num_steps=4)
    gc = extras["gradient_clip"]
    assert gc["num_clips"] > 0
    assert gc["max_grad_norm_seen"] > 0.001


def test_clip_loose_threshold_no_clips():
    """clip=1000 is far above typical grad → num_clips == 0."""
    extras = _run(clip=1000.0, num_steps=4)
    gc = extras["gradient_clip"]
    assert gc["num_clips"] == 0
    # max_grad_norm_seen still populated (clipping disabled, only
    # observation skipped when threshold None — here we DO observe)
    assert gc["max_grad_norm_seen"] > 0
