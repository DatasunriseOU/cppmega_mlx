"""V7-B07: single-process all-gather / reduce-scatter proxies.

Single-device emulations of the two FSDP collectives. Pure mlx
operations — multi-GPU NCCL paths land under V7-B01.

  all_gather(shard, world_size)   → concatenated full tensor
  reduce_scatter(full, world_size)→ mean-reduced shard chunk

Pair: reduce_scatter(all_gather(shard, W), W) == shard for any W.
Also exposes measure_overhead_ms helper for the bench harness.
"""

from __future__ import annotations

import time

import mlx.core as mx


def all_gather(shard: mx.array, world_size: int) -> mx.array:
    """Simulate all-gather by replicating shard W times along axis 0."""
    if world_size < 1:
        raise ValueError("world_size must be >= 1")
    return mx.concatenate([shard for _ in range(world_size)], axis=0)


def reduce_scatter(full: mx.array, world_size: int) -> mx.array:
    """Simulate reduce-scatter: split full into W chunks along axis 0,
    mean-reduce them into a single chunk."""
    if world_size < 1:
        raise ValueError("world_size must be >= 1")
    if full.shape[0] % world_size != 0:
        raise ValueError(
            f"full[0]={full.shape[0]} not divisible by "
            f"world_size={world_size}")
    chunk = full.shape[0] // world_size
    pieces = [full[i * chunk:(i + 1) * chunk]
              for i in range(world_size)]
    acc = pieces[0]
    for p in pieces[1:]:
        acc = acc + p
    return acc / float(world_size)


def measure_overhead_ms(*, world_size: int, shard_size: int,
                         n_iter: int = 50) -> dict[str, float]:
    """Run all-gather + reduce-scatter `n_iter` times, return mean
    per-iteration ms for each."""
    shard = mx.random.normal(shape=(shard_size,),
                              key=mx.random.key(0))
    mx.eval(shard)
    # Warm.
    _ = reduce_scatter(all_gather(shard, world_size), world_size)
    mx.eval(_)
    t0 = time.perf_counter()
    for _ in range(n_iter):
        ag = all_gather(shard, world_size)
        mx.eval(ag)
    ag_ms = (time.perf_counter() - t0) * 1000.0 / n_iter
    full = all_gather(shard, world_size); mx.eval(full)
    t0 = time.perf_counter()
    for _ in range(n_iter):
        rs = reduce_scatter(full, world_size)
        mx.eval(rs)
    rs_ms = (time.perf_counter() - t0) * 1000.0 / n_iter
    return {
        "world_size": float(world_size),
        "shard_size": float(shard_size),
        "all_gather_ms_per_iter": round(ag_ms, 4),
        "reduce_scatter_ms_per_iter": round(rs_ms, 4),
    }


__all__ = ["all_gather", "reduce_scatter", "measure_overhead_ms"]
