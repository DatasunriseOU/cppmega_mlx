"""V7-A01: llama3_8b preset instantiated end-to-end at scaled dims.

Per user redirect ("у нас же конструктор из кирпичиков"), the
acceptance is that the llama3_8b preset's brick composition runs
through stage_train at a realistic-but-mini scale (H=512 with
multi-repeat). Hidden=4096 full-scale is HBM-bound on a single
Mac and tracked as a separate perf gate (V7-I05).
"""

from __future__ import annotations

import pytest

from cppmega_v4.architectures.presets import build_preset_specs
from cppmega_v4.jsonrpc.schema import VerifyParams
from cppmega_v4.runner import Pipeline, run_pipeline


def _spec_from_preset(name: str, hidden: int = 512) -> VerifyParams:
    specs = build_preset_specs(name, hidden_size=hidden)
    return VerifyParams.model_validate({
        "graph": {
            "nodes": [
                {"id": f"n{i}", "kind": s["kind"],
                 "params": s.get("params", {})}
                for i, s in enumerate(specs)
            ],
            "edges": [
                {"src": f"n{i}", "dst": f"n{i + 1}"}
                for i in range(len(specs) - 1)
            ],
        },
        "dim_env": {"B": 1, "S": 8, "H": hidden,
                    "nh": max(2, hidden // 64),
                    "nkv": max(1, hidden // 128),
                    "head_dim": 64,
                    "num_experts": 4, "top_k": 2},
        "loss": {"kind": "cross_entropy",
                 "head_outputs": [f"n{len(specs) - 1}"]},
        "optim": {"kind": "adamw",
                  "groups": [{"matcher": "all", "lr": 1e-3,
                              "weight_decay": 0.01,
                              "betas": [0.9, 0.95]}]},
    })


def test_v7_a01_llama3_8b_preset_runs_at_h512():
    rep = run_pipeline(
        _spec_from_preset("llama3_8b", hidden=512),
        Pipeline.from_dict({
            "stages": ["parse", "verify_build_spec", "build_model",
                       "train"],
            "stage_options": {"train": {"num_steps": 2}},
        }),
    )
    tr = next(s for s in rep.stages if s.name == "train")
    assert tr.status == "ok", f"train failed: {tr.error}"
    assert len(tr.extras["losses"]) == 2
    for L in tr.extras["losses"]:
        assert -1e10 < L < 1e10
    assert tr.extras["weight_delta_norm"] > 0


def test_v7_a01_preset_compose_yields_known_brick_kinds():
    """llama3_8b preset must produce attention + mlp bricks (the
    canonical LLaMA repeat unit)."""
    specs = build_preset_specs("llama3_8b", hidden_size=128)
    kinds = {s["kind"] for s in specs}
    assert "attention" in kinds
    assert "mlp" in kinds
