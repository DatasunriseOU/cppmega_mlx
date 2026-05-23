"""V7-E01/E02: V4MoE with capacity_factor < 1 actually drops/reroutes
tokens and stage_train surfaces dropped_token_ratio + rerouted_token_ratio.

Honest-closure: the helper compute_drop_reroute_stats existed
(cppmega_v4/nn/moe_capacity.py) but V4MoE.__call__ never invoked it.
After E01/E02 wiring, capacity_factor in the moe brick params engages
the helper inside the forward pass and stage_train.moe_extras carries
the real numbers — not the hardcoded 0.0 placeholder.
"""

from __future__ import annotations

from cppmega_v4.jsonrpc.schema import VerifyParams
from cppmega_v4.runner import Pipeline, run_pipeline


def _spec(capacity_factor: float | None) -> VerifyParams:
    params = {"num_experts": 8, "top_k": 2,
              "expert_hidden_size": 32, "aux_loss_free": True}
    if capacity_factor is not None:
        params["capacity_factor"] = capacity_factor
    return VerifyParams.model_validate({
        "graph": {
            "nodes": [
                {"id": "attn", "kind": "attention", "params": {}},
                {"id": "moe", "kind": "moe", "params": params},
            ],
            "edges": [{"src": "attn", "dst": "moe"}],
        },
        "dim_env": {"B": 1, "S": 32, "H": 32, "nh": 2, "nkv": 1,
                    "head_dim": 16, "num_experts": 8, "top_k": 2},
        "loss": {"kind": "cross_entropy", "head_outputs": ["moe"]},
        "optim": {"kind": "adamw",
                  "groups": [{"matcher": "all", "lr": 1e-3,
                              "weight_decay": 0.01, "betas": [0.9, 0.95]}]},
    })


def _run(capacity_factor: float | None) -> dict:
    report = run_pipeline(_spec(capacity_factor), Pipeline.from_dict({
        "stages": ["parse", "verify_build_spec", "build_model", "train"],
        "stage_options": {"train": {"num_steps": 2}},
    }))
    train = next(s for s in report.stages if s.name == "train")
    assert train.status == "ok", f"stage_train failed: {train.error}"
    return train.extras["moe"]


def test_no_capacity_factor_zero_drop():
    moe = _run(None)
    assert moe["dropped_token_ratio"] == 0.0
    # No capacity bounding → no rerouted ratio metric, default 0.
    assert moe.get("rerouted_token_ratio", 0.0) == 0.0
    # capacity_factor absent in extras (None or missing key).
    assert moe.get("capacity_factor") in (None, 0.0)


def test_capacity_factor_tight_triggers_drop_or_reroute():
    # With C=0.25 and num_experts=8, top_k=2 over S=32 tokens →
    # 64 dispatch slots; cap_per_expert = ceil(0.25*64/8) = 2. With
    # uniform-ish routing on a tiny untrained model many experts will
    # overflow → either drop or reroute fires.
    moe = _run(0.25)
    assert "rerouted_token_ratio" in moe
    assert "capacity_per_expert" in moe
    assert "overflow_ratio" in moe
    assert moe["capacity_factor"] == 0.25
    # capacity_per_expert is at least 1.
    assert moe["capacity_per_expert"] >= 1
    # Total overflow (drop + reroute) is in [0, 1].
    assert 0.0 <= moe["overflow_ratio"] <= 1.0
    # With 8 experts and tight cap, at least *some* overflow must fire.
    assert moe["overflow_ratio"] > 0.0
