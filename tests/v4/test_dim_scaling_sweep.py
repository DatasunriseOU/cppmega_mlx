"""V7-F53: dimension-scaling sweep on llama3_8b preset.

Proves the constructor flow the user described:
  pick preset → instantiate at small H → run through system →
  expand H → still works.

Pytest parametrises H in {64, 128, 256, 512} and asserts:
  (a) all four configurations land train.status='ok' with finite
      losses.
  (b) per-H weight_delta_norm > 0 (training really happened).
  (c) param footprint scales monotonically with H.
"""

from __future__ import annotations

import pytest

from cppmega_v4.architectures.presets import build_preset_specs
from cppmega_v4.jsonrpc.schema import VerifyParams
from cppmega_v4.runner import Pipeline, run_pipeline


def _spec_at(H: int) -> VerifyParams:
    specs = build_preset_specs("llama3_8b", hidden_size=H)
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
        "dim_env": {"B": 1, "S": 8, "H": H,
                    "nh": max(2, H // 64), "nkv": max(1, H // 128),
                    "head_dim": 64, "num_experts": 4, "top_k": 2},
        "loss": {"kind": "cross_entropy",
                 "head_outputs": [f"n{len(specs) - 1}"]},
        "optim": {"kind": "adamw",
                  "groups": [{"matcher": "all", "lr": 1e-3,
                              "weight_decay": 0.01,
                              "betas": [0.9, 0.95]}]},
    })


def _train_at(H: int, num_steps: int = 4) -> dict:
    rep = run_pipeline(_spec_at(H), Pipeline.from_dict({
        "stages": ["parse", "verify_build_spec", "build_model",
                   "estimate_memory", "train"],
        "stage_options": {"train": {"num_steps": num_steps}},
    }))
    return {
        s.name: s for s in rep.stages
    }


@pytest.mark.parametrize("H", [64, 128, 256, 512])
def test_v7_f53_each_dim_trains_with_finite_loss(H):
    stages = _train_at(H, num_steps=4)
    tr = stages["train"]
    assert tr.status == "ok", f"H={H} train failed: {tr.error}"
    for L in tr.extras["losses"]:
        assert -1e10 < L < 1e10, f"H={H} loss non-finite: {L}"
    # Weights moved.
    assert tr.extras["weight_delta_norm"] > 0, (
        f"H={H} weight_delta_norm zero — model didn't train"
    )


def test_v7_f53_param_count_monotone_with_H():
    """estimate_memory.params_bytes should grow with hidden size."""
    sizes: dict[int, int] = {}
    for H in (64, 128, 256, 512):
        stages = _train_at(H, num_steps=2)
        sizes[H] = int(stages["estimate_memory"].extras["params_bytes"])
    # Monotone increase across H.
    for a, b in zip(sorted(sizes.keys()), sorted(sizes.keys())[1:]):
        assert sizes[a] < sizes[b], (
            f"params_bytes not monotone: H={a}→{sizes[a]} vs "
            f"H={b}→{sizes[b]}"
        )
