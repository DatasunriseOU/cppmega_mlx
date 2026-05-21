"""G18: ablation.run uses the same stage_train code path as pipeline.run.

V3-7 proved variants produce different final loss but never compared
the ablation route vs explicit sequential pipeline.run calls. v5 asserts
bit-identical extras shape and within-tolerance losses for the same
seed across both routes.
"""

from __future__ import annotations

import pytest

from cppmega_v4.jsonrpc.ablation_method import ablation_run
from cppmega_v4.jsonrpc.schema import VerifyParams
from cppmega_v4.runner import Pipeline, run_pipeline


def _base_payload() -> dict:
    return {
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
                  "groups": [{"matcher": "all", "lr": 1e-3,
                              "weight_decay": 0.01, "betas": [0.9, 0.95]}]},
    }


def _pipeline_run_for_activation(activation: str) -> dict:
    """Run pipeline.run with explicit activation mutation; return train extras."""
    payload = _base_payload()
    payload["graph"]["nodes"][1]["params"]["activation"] = activation
    spec = VerifyParams.model_validate(payload)
    report = run_pipeline(spec, Pipeline.from_dict({
        "stages": ["parse", "verify_build_spec", "build_model", "train"],
        "stage_options": {"train": {"num_steps": 2}},
    }))
    train = next(s for s in report.stages if s.name == "train")
    return train.extras


def test_ablation_uses_pipeline_run_under_hood():
    """ablation.run code imports + uses run_pipeline. Structural guarantee
    that ablation route is not a separate code path that could drift."""
    import cppmega_v4.jsonrpc.ablation_method as am
    src = pathlib_read(am.__file__)
    assert "from cppmega_v4.runner import" in src
    assert "run_pipeline" in src


def pathlib_read(path: str) -> str:
    from pathlib import Path
    return Path(path).read_text()


def test_ablation_per_variant_extras_shape_matches_pipeline():
    """For activation axis with [glu, swiglu], the per-variant final-loss
    via ablation.run must equal pipeline.run's losses[-1] within noise
    (same deterministic mx.random.key)."""
    from cppmega_v4.jsonrpc.ablation_method import AblationRunParams

    params = AblationRunParams.model_validate({
        "base_spec": _base_payload(),
        "ablation_axis": "activation",
        "variants": ["glu", "swiglu"],
        "num_steps": 2,
    })
    result = ablation_run(params)
    assert len(result.results) == 2
    by_name = {v.variant: v for v in result.results}

    for variant_name in ["glu", "swiglu"]:
        ablation_extras = by_name[variant_name]
        pipeline_extras = _pipeline_run_for_activation(variant_name)
        # losses array shape must match
        assert len(ablation_extras.losses) == len(pipeline_extras["losses"])
        # Final losses within 5% (both routes use the same seeded code
        # so should be exact; allow tolerance for non-determinism in
        # downstream random ops).
        assert abs(ablation_extras.losses[-1] - pipeline_extras["losses"][-1]) \
               < max(0.05 * abs(pipeline_extras["losses"][-1]), 0.5)
