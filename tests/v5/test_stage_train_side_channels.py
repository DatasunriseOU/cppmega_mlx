"""G17: side_channels reach stage_train forward and change math.

V4-10 was observation-only. G17 routes doc_ids into same-document attention
masks plus doc conditioning and routes token_ids into conditional embeddings.
"""

from __future__ import annotations

import mlx.core as mx

from cppmega_v4.jsonrpc.schema import VerifyParams
from cppmega_v4.runner import Pipeline, run_pipeline


def _spec() -> VerifyParams:
    return VerifyParams.model_validate({
        "graph": {"nodes": [
            {"id": "attn", "kind": "attention", "params": {}},
            {"id": "mlp", "kind": "mlp",
             "params": {"intermediate_size": 64, "activation": "swiglu"}},
        ], "edges": [{"src": "attn", "dst": "mlp"}]},
        "dim_env": {"B": 1, "S": 8, "H": 32, "nh": 2, "nkv": 1, "head_dim": 16},
        "loss": {"kind": "cross_entropy", "head_outputs": ["mlp"]},
        "optim": {"kind": "adamw",
                  "groups": [{"matcher": "all", "lr": 1e-3,
                              "weight_decay": 0.01, "betas": [0.9, 0.95]}]},
    })


def _run(opts: dict) -> dict:
    mx.random.seed(17)
    report = run_pipeline(_spec(), Pipeline.from_dict({
        "stages": ["parse", "verify_build_spec", "build_model", "train"],
        "stage_options": {"train": opts},
    }))
    train = next(s for s in report.stages if s.name == "train")
    assert train.status == "ok", f"stage_train failed: {train.error}"
    return train.extras


def test_no_side_channels_forward_effect_none():
    extras = _run({"num_steps": 2})
    assert extras["side_channels_forward_effect"] is None


def test_doc_ids_populates_mask_density():
    extras = _run({"num_steps": 2,
                   "side_channels": {"doc_ids": [0, 0, 0, 1, 1, 1, 2, 2]}})
    fwd = extras["side_channels_forward_effect"]
    assert fwd is not None
    # 3 distinct docs → significant cross-doc fraction
    assert fwd["doc_ids_mask_density"] > 0.1
    assert fwd["token_ids_added_norm"] == 0.0


def test_doc_ids_change_loss_vs_disabled_same_seed():
    base = _run({"num_steps": 3})["losses"]
    doc = _run({"num_steps": 3,
                "side_channels": {"doc_ids": [0, 0, 0, 1, 1, 1, 2, 2]}})[
        "losses"
    ]
    assert doc != base


def test_token_ids_populates_added_norm():
    extras = _run({"num_steps": 2,
                   "side_channels": {"token_ids": [1, 2, 3, 4, 5, 6, 7, 8]}})
    fwd = extras["side_channels_forward_effect"]
    assert fwd is not None
    assert fwd["token_ids_added_norm"] > 0
    assert fwd["doc_ids_mask_density"] == 0.0


def test_token_ids_change_loss_vs_disabled_same_seed():
    base = _run({"num_steps": 3})["losses"]
    token = _run({"num_steps": 3,
                  "side_channels": {"token_ids": [1, 2, 3, 4, 5, 6, 7, 8]}})[
        "losses"
    ]
    assert token != base


def test_both_channels_both_metrics_populated():
    extras = _run({"num_steps": 2, "side_channels": {
        "doc_ids": [0, 0, 1, 1, 2, 2, 3, 3],
        "token_ids": [10, 20, 30, 40, 50, 60, 70, 80],
    }})
    fwd = extras["side_channels_forward_effect"]
    assert fwd is not None
    assert fwd["doc_ids_mask_density"] > 0
    assert fwd["token_ids_added_norm"] > 0
