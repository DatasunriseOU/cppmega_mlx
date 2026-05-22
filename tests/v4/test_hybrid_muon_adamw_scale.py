"""V7-A07: Muon+AdamW hybrid at larger scale (4-layer H=256)."""

from __future__ import annotations

import pytest

from cppmega_v4.jsonrpc.schema import VerifyParams
from cppmega_v4.runner import Pipeline, run_pipeline


def _spec(H: int = 256) -> VerifyParams:
    return VerifyParams.model_validate({
        "graph": {
            "nodes": [
                {"id": "attn1", "kind": "attention",
                 "params": {"num_heads": 4, "head_dim": 64}},
                {"id": "mlp1", "kind": "mlp", "params": {}},
                {"id": "attn2", "kind": "attention",
                 "params": {"num_heads": 4, "head_dim": 64}},
                {"id": "mlp2", "kind": "mlp", "params": {}},
            ],
            "edges": [
                {"src": "attn1", "dst": "mlp1"},
                {"src": "mlp1", "dst": "attn2"},
                {"src": "attn2", "dst": "mlp2"},
            ],
        },
        "dim_env": {"B": 1, "S": 8, "H": H,
                    "nh": 4, "nkv": 2, "head_dim": 64},
        "loss": {"kind": "cross_entropy", "head_outputs": ["mlp2"]},
        "optim": {"kind": "muon_adamw_hybrid",
                  "groups": [{"matcher": "all", "lr": 1e-3,
                              "weight_decay": 0.01,
                              "betas": [0.9, 0.95]}]},
    })


def _train(num_steps: int = 4, H: int = 256) -> dict:
    rep = run_pipeline(_spec(H), Pipeline.from_dict({
        "stages": ["parse", "verify_build_spec", "build_model", "train"],
        "stage_options": {"train": {"num_steps": num_steps}},
    }))
    tr = next(s for s in rep.stages if s.name == "train")
    assert tr.status == "ok", f"train failed: {tr.error}"
    return tr.extras


def test_v7_a07_hybrid_at_h256_4layer_runs_and_routes():
    """Muon should grab the 2-D matrices, AdamW the 1-D / 3-D rest.
    Hybrid extras carry per-bucket group sizes that diverge."""
    e = _train(num_steps=4, H=256)
    assert e["optimizer_kind"] == "muon_adamw_hybrid"
    assert e["muon_group_size"] is not None
    assert e["adamw_group_size"] is not None
    assert e["muon_group_size"] > 0
    assert e["adamw_group_size"] > 0
    # hybrid_deltas dict should be populated (Muon vs AdamW updates
    # diverge in per-bucket weight delta).
    assert e["hybrid_deltas"] is not None


def test_v7_a07_hybrid_finite_losses_4layer():
    e = _train(num_steps=4, H=256)
    for L in e["losses"]:
        assert -1e10 < L < 1e10
