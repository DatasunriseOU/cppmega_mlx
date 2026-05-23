"""V7-B14: collective_proxy.reduce_scatter bit-exact vs naive mean."""

from __future__ import annotations

import mlx.core as mx

from cppmega_v4.runtime.collective_proxy import (
    all_gather, reduce_scatter,
)


def test_reduce_scatter_world_size_4_mean_reduces_chunks():
    """Build a deliberately-replicated arr of shape (4*N,) where each
    chunk of N elements represents a per-rank value; reduce_scatter
    must mean-reduce them into a single chunk identical to arr.mean(0)
    when reshaped to (4, N)."""
    N = 8
    world = 4
    chunks = [
        mx.full((N,), float(r + 1), dtype=mx.float32)
        for r in range(world)
    ]
    arr = mx.concatenate(chunks, axis=0)
    out = reduce_scatter(arr, world)
    # Mean across the 4 chunks = (1+2+3+4)/4 = 2.5 broadcast over N.
    expected = mx.full((N,), 2.5, dtype=mx.float32)
    assert mx.array_equal(out, expected).item()


def test_reduce_scatter_world_size_1_is_identity():
    arr = mx.arange(8.0)
    out = reduce_scatter(arr, 1)
    assert mx.array_equal(out, arr).item()


def test_all_gather_round_trips_with_reduce_scatter():
    """all_gather(shard, W) followed by reduce_scatter(full, W) returns
    the original shard — V7-B contract."""
    for world in (1, 2, 4, 8):
        shard = mx.arange(4.0)
        full = all_gather(shard, world)
        back = reduce_scatter(full, world)
        assert mx.array_equal(back, shard).item(), (
            f"world={world} all_gather→reduce_scatter broke identity")


def test_reduce_scatter_rejects_misaligned_shape():
    import pytest
    arr = mx.arange(7.0)   # 7 % 4 != 0
    with pytest.raises(ValueError):
        reduce_scatter(arr, 4)
