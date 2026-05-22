"""V7-B07: single-process collective proxies — correctness + bench."""

from __future__ import annotations

import mlx.core as mx
import pytest

from cppmega_v4.runtime.collective_proxy import (
    all_gather, measure_overhead_ms, reduce_scatter,
)


def test_v7_b07_all_gather_4x_replicates():
    shard = mx.array([1.0, 2.0, 3.0])
    full = all_gather(shard, 4)
    assert full.shape == (12,)
    assert mx.allclose(full[:3], shard, atol=0.0)
    assert mx.allclose(full[9:], shard, atol=0.0)


def test_v7_b07_reduce_scatter_inverse_of_all_gather():
    """reduce_scatter(all_gather(shard, W), W) == shard."""
    shard = mx.array([1.0, 2.0, 3.0, 4.0])
    for W in (1, 2, 4):
        ag = all_gather(shard, W)
        rs = reduce_scatter(ag, W)
        assert mx.allclose(rs, shard, atol=1e-6), W


def test_v7_b07_reduce_scatter_requires_divisible_shape():
    full = mx.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        reduce_scatter(full, world_size=2)  # 3 % 2 != 0


def test_v7_b07_measure_overhead_returns_positive_ms():
    r = measure_overhead_ms(
        world_size=4, shard_size=1024, n_iter=5)
    assert r["all_gather_ms_per_iter"] > 0
    assert r["reduce_scatter_ms_per_iter"] > 0
    assert r["world_size"] == 4.0


def test_v7_b07_world_size_validation():
    with pytest.raises(ValueError):
        all_gather(mx.array([1.0]), 0)
    with pytest.raises(ValueError):
        reduce_scatter(mx.array([1.0]), 0)
