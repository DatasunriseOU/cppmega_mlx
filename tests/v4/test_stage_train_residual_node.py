"""Regression coverage for parameter-free residual nodes in stage_train."""

from __future__ import annotations

import mlx.core as mx

from cppmega_v4.jsonrpc.schema import VerifyParams
from cppmega_v4.runner import Pipeline, run_pipeline


def _residual_spec() -> VerifyParams:
    return VerifyParams.model_validate({
        "graph": {
            "nodes": [
                {"id": "mlp_in", "kind": "mlp", "params": {}},
                {"id": "residual_add", "kind": "residual", "params": {}},
                {"id": "mlp_out", "kind": "mlp", "params": {}},
            ],
            "edges": [
                {"src": "mlp_in", "dst": "residual_add"},
                {"src": "residual_add", "dst": "mlp_out"},
            ],
        },
        "dim_env": {
            "B": 1,
            "S": 8,
            "H": 64,
            "nh": 1,
            "nkv": 1,
            "head_dim": 64,
            "num_experts": 4,
            "top_k": 1,
        },
        "loss": {
            "kind": "cross_entropy",
            "head_outputs": ["mlp_out"],
        },
        "optim": {
            "kind": "adamw",
            "groups": [{
                "matcher": "all",
                "lr": 1e-3,
                "weight_decay": 0.01,
                "betas": [0.9, 0.95],
            }],
        },
    })


def _zaya_shaped_fan_in_spec(
    *,
    include_bypass: bool = True,
) -> VerifyParams:
    """Small version of the nightly ZAYA graph with an explicit bypass."""
    edges = [
        {"src": "input_embedder", "dst": "cca"},
        {"src": "cca", "dst": "mlp"},
        {"src": "mlp", "dst": "moe"},
        {"src": "moe", "dst": "residual_add"},
        {"src": "residual_add", "dst": "output_deembedder"},
    ]
    if include_bypass:
        edges.insert(
            -1,
            {"src": "mlp", "dst": "residual_add"},
        )
    return VerifyParams.model_validate({
        "graph": {
            "nodes": [
                {
                    "id": "input_embedder",
                    "kind": "embedding_table",
                    "params": {"vocab_size": 256},
                },
                {
                    "id": "cca",
                    "kind": "cca_attention",
                    "params": {
                        "num_attention_heads": 1,
                        "num_key_value_heads": 1,
                        "head_dim": 64,
                        "fine_window": 8,
                        "coarse_block_size": 4,
                    },
                },
                {"id": "mlp", "kind": "mlp", "params": {}},
                {
                    "id": "moe",
                    "kind": "moe",
                    "params": {"num_experts": 2, "top_k": 1},
                },
                {"id": "residual_add", "kind": "residual", "params": {}},
                {
                    "id": "output_deembedder",
                    "kind": "embedding_table",
                    "params": {"vocab_size": 256},
                },
            ],
            "edges": edges,
        },
        "dim_env": {
            "B": 1,
            "S": 8,
            "H": 64,
            "nh": 1,
            "nkv": 1,
            "head_dim": 64,
            "num_experts": 2,
            "top_k": 1,
        },
        "loss": {
            "kind": "cross_entropy",
            "head_outputs": ["output_deembedder"],
        },
        "optim": {
            "kind": "adamw",
            "groups": [{
                "matcher": "all",
                "lr": 1e-3,
                "weight_decay": 0.01,
                "betas": [0.9, 0.95],
            }],
        },
    })


def _train(spec: VerifyParams):
    report = run_pipeline(spec, Pipeline.from_dict({
        "stages": [
            "parse",
            "verify_build_spec",
            "build_model",
            "train",
        ],
        "stage_options": {"train": {"num_steps": 2}},
    }))
    return next(stage for stage in report.stages if stage.name == "train")


def test_parameter_free_residual_is_instantiated_and_trains() -> None:
    train = _train(_residual_spec())
    assert train.status == "ok", train.error
    assert train.extras["weight_delta_norm"] > 0


def test_explicit_residual_fan_in_preserves_bypass_during_train() -> None:
    mx.random.seed(23)
    fan_in_train = _train(_zaya_shaped_fan_in_spec())
    mx.random.seed(23)
    single_path_train = _train(
        _zaya_shaped_fan_in_spec(include_bypass=False)
    )

    assert fan_in_train.status == "ok", fan_in_train.error
    assert fan_in_train.extras["weight_delta_norm"] > 0
    assert single_path_train.status == "fail"
    assert single_path_train.error["type"] == "WeightsUnchanged"
