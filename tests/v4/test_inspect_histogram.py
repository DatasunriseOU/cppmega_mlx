"""V7-H08: inspect.histogram backend RPC tests."""

from __future__ import annotations

import pytest

from cppmega_v4.jsonrpc.histogram_method import (
    HistogramParams, inspect_histogram,
)
from cppmega_v4.jsonrpc.schema import VerifyParams


def _spec() -> VerifyParams:
    return VerifyParams.model_validate({
        "graph": {
            "nodes": [
                {"id": "attn", "kind": "attention",
                 "params": {"num_heads": 4, "head_dim": 64}},
                {"id": "mlp", "kind": "mlp", "params": {}},
            ],
            "edges": [{"src": "attn", "dst": "mlp"}],
        },
        "dim_env": {"B": 1, "S": 8, "H": 128,
                    "nh": 2, "nkv": 1, "head_dim": 64},
        "loss": {"kind": "cross_entropy", "head_outputs": ["mlp"]},
        "optim": {"kind": "adamw",
                  "groups": [{"matcher": "all", "lr": 1e-3,
                              "weight_decay": 0.01,
                              "betas": [0.9, 0.95]}]},
    })


def test_v7_h08_histogram_shape_and_bin_count():
    r = inspect_histogram(HistogramParams(
        spec=_spec(), brick_id="attn", buckets=32, num_steps=2))
    assert r.buckets == 32
    assert len(r.bins) == 33  # buckets + 1 edge
    assert len(r.counts) == 32
    assert sum(r.counts) == r.n_values
    assert r.n_values > 0
    assert r.min <= r.mean <= r.max


def test_v7_h08_unknown_brick_raises():
    with pytest.raises(ValueError):
        inspect_histogram(HistogramParams(
            spec=_spec(), brick_id="does_not_exist",
            buckets=16, num_steps=1))


def test_v7_h08_bucket_clip_64():
    r = inspect_histogram(HistogramParams(
        spec=_spec(), brick_id="mlp", buckets=64, num_steps=1))
    assert r.buckets == 64
    assert len(r.counts) == 64
    # Bins are monotonic.
    for a, b in zip(r.bins, r.bins[1:]):
        assert a < b


def test_histogram_handles_rmsnorm_node():
    """fhxg: Inspect weight histogram on a `rmsnorm` canvas node must
    succeed — the kind is a norm primitive (not in BLOCK_BUILDERS) and
    used to surface 'Invalid params' before the fallback was added.

    RMSNorm has `dim` channels initialised to 1.0 (gamma), so the
    histogram is a single peak at 1.0.
    """
    from cppmega_v4.jsonrpc.histogram_method import (
        HistogramParams, inspect_histogram,
    )
    from cppmega_v4.jsonrpc.schema import VerifyParams
    spec = VerifyParams.model_validate({
        "graph": {
            "nodes": [
                {"id": "a", "kind": "attention", "params": {}},
                {"id": "n", "kind": "rmsnorm", "params": {}},
            ],
            "edges": [{"src": "a", "dst": "n"}],
        },
        "dim_env": {"H": 128},
        "loss": {"kind": "cross_entropy", "head_outputs": ["n"]},
        "optim": {"kind": "adamw", "groups": [
            {"matcher": "all", "lr": 1e-3, "weight_decay": 0.01,
             "betas": [0.9, 0.95]}]},
        "sharding": None,
        "training": True,
    })
    r = inspect_histogram(HistogramParams(
        spec=spec, brick_id="n", kind="weight", buckets=32))
    assert r.n_values == 128
    assert abs(r.mean - 1.0) < 1e-6


def test_histogram_handles_layernorm_node():
    from cppmega_v4.jsonrpc.histogram_method import (
        HistogramParams, inspect_histogram,
    )
    from cppmega_v4.jsonrpc.schema import VerifyParams
    spec = VerifyParams.model_validate({
        "graph": {
            "nodes": [
                {"id": "a", "kind": "attention", "params": {}},
                {"id": "n", "kind": "layernorm", "params": {}},
            ],
            "edges": [{"src": "a", "dst": "n"}],
        },
        "dim_env": {"H": 64},
        "loss": {"kind": "cross_entropy", "head_outputs": ["n"]},
        "optim": {"kind": "adamw", "groups": [
            {"matcher": "all", "lr": 1e-3, "weight_decay": 0.01,
             "betas": [0.9, 0.95]}]},
        "sharding": None,
        "training": True,
    })
    r = inspect_histogram(HistogramParams(
        spec=spec, brick_id="n", kind="weight", buckets=16))
    # LayerNorm has gamma + beta = 2 * H channels.
    assert r.n_values == 2 * 64


def test_histogram_residual_kind_rejected_cleanly():
    import pytest
    from cppmega_v4.jsonrpc.histogram_method import (
        HistogramParams, inspect_histogram,
    )
    from cppmega_v4.jsonrpc.schema import VerifyParams
    spec = VerifyParams.model_validate({
        "graph": {
            "nodes": [
                {"id": "a", "kind": "attention", "params": {}},
                {"id": "r", "kind": "residual", "params": {}},
            ],
            "edges": [{"src": "a", "dst": "r"}],
        },
        "dim_env": {"H": 64},
        "loss": {"kind": "cross_entropy", "head_outputs": ["r"]},
        "optim": {"kind": "adamw", "groups": [
            {"matcher": "all", "lr": 1e-3, "weight_decay": 0.01,
             "betas": [0.9, 0.95]}]},
        "sharding": None,
        "training": True,
    })
    with pytest.raises(ValueError, match="no trainable weights"):
        inspect_histogram(HistogramParams(
            spec=spec, brick_id="r", kind="weight", buckets=8))


def test_histogram_handles_linear_bridge_node():
    from cppmega_v4.jsonrpc.histogram_method import (
        HistogramParams, inspect_histogram,
    )
    from cppmega_v4.jsonrpc.schema import VerifyParams
    spec = VerifyParams.model_validate({
        "graph": {
            "nodes": [
                {"id": "a", "kind": "attention", "params": {}},
                {"id": "b", "kind": "linear_bridge", "params": {"H_in": 64, "H_out": 128}},
            ],
            "edges": [{"src": "a", "dst": "b"}],
        },
        "dim_env": {"H": 64},
        "loss": {"kind": "cross_entropy", "head_outputs": ["b"]},
        "optim": {"kind": "adamw", "groups": [
            {"matcher": "all", "lr": 1e-3, "weight_decay": 0.01,
             "betas": [0.9, 0.95]}]},
        "sharding": None,
        "training": True,
    })
    r = inspect_histogram(HistogramParams(
        spec=spec, brick_id="b", kind="weight", buckets=16))
    assert r.n_values == 64 * 128


def test_histogram_handles_adapter_linear_bridge_node():
    from cppmega_v4.jsonrpc.histogram_method import (
        HistogramParams, inspect_histogram,
    )
    from cppmega_v4.jsonrpc.schema import VerifyParams
    spec = VerifyParams.model_validate({
        "graph": {
            "nodes": [
                {"id": "a", "kind": "attention", "params": {}},
                {"id": "b", "kind": "adapter_linear_bridge", "params": {}},
            ],
            "edges": [{"src": "a", "dst": "b"}],
        },
        "dim_env": {"H": 64},
        "loss": {"kind": "cross_entropy", "head_outputs": ["b"]},
        "optim": {"kind": "adamw", "groups": [
            {"matcher": "all", "lr": 1e-3, "weight_decay": 0.01,
             "betas": [0.9, 0.95]}]},
        "sharding": None,
        "training": True,
    })
    r = inspect_histogram(HistogramParams(
        spec=spec, brick_id="b", kind="weight", buckets=16))
    # Defaults to H x H = 64 * 64 = 4096
    assert r.n_values == 64 * 64


def test_histogram_plumbing_kinds_rejected_cleanly():
    import pytest
    from cppmega_v4.jsonrpc.histogram_method import (
        HistogramParams, inspect_histogram,
    )
    from cppmega_v4.jsonrpc.schema import VerifyParams
    for pk in ["merge_heads", "adapter_split_heads", "transpose_bnsd"]:
        spec = VerifyParams.model_validate({
            "graph": {
                "nodes": [
                    {"id": "a", "kind": "attention", "params": {}},
                    {"id": "p", "kind": pk, "params": {}},
                ],
                "edges": [{"src": "a", "dst": "p"}],
            },
            "dim_env": {"H": 64},
            "loss": {"kind": "cross_entropy", "head_outputs": ["p"]},
            "optim": {"kind": "adamw", "groups": [
                {"matcher": "all", "lr": 1e-3, "weight_decay": 0.01,
                 "betas": [0.9, 0.95]}]},
            "sharding": None,
            "training": True,
        })
        with pytest.raises(ValueError, match="no trainable weights"):
            inspect_histogram(HistogramParams(
                spec=spec, brick_id="p", kind="weight", buckets=8))
