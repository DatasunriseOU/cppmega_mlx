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
