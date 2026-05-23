"""V7-B-real: thin wrapper over mlx.core.distributed.

Honest-closure for the B-block audit:
  * Items 11-18 — every collective was 'proxy' (replay one forward
    multiple times, name-only files, no process groups). MLX ships
    a real distributed module (mx.distributed.all_sum / all_gather /
    sum_scatter / send / recv / Group / init), but cppmega_v4 never
    called it.

This module is the single integration point. Other distributed
runtime files (tp_proxy, pp_proxy, collective_proxy) delegate here
when world_size > 1, falling back to their single-process emulation
when world_size == 1 (so existing tests keep working).

  init()              → world_size, rank, group
  is_distributed()    → True when world_size > 1
  all_reduce(x, op)   → real or single-rank passthrough
  all_gather(x)       → real or single-rank concat replication
  reduce_scatter(x)   → real (sum_scatter / world_size) or split-mean
  send/recv(x, peer)  → MPI-style point-to-point
  barrier()           → synchronization point
  measure_collective_overhead_ms(...) → real wallclock bench

`mlx.launch` (mpirun-based) spawns peer processes; each calls
`init()` on import. Without a launcher world_size stays 1.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Literal

import mlx.core as mx


@dataclass(frozen=True)
class WorldInfo:
    world_size: int
    rank: int
    backend: str
    real: bool        # True when mx.distributed actually engaged.


_WORLD: WorldInfo | None = None


def _detect_backend() -> str:
    # mpirun / mlx.launch sets OMPI_* env vars; bare mx.distributed.init()
    # falls back to a single-process Group when neither MPI nor a
    # registered transport is available.
    if any(k.startswith("OMPI_") for k in os.environ):
        return "mpi"
    if "MLX_DIST_BACKEND" in os.environ:
        return os.environ["MLX_DIST_BACKEND"]
    return "ring"


def init(force_single: bool = False) -> WorldInfo:
    """Initialise the distributed runtime. Idempotent.

    `force_single=True` makes this a no-op single-process world even
    when mx.distributed.is_available() returns True — useful for tests
    that don't want to leak collective state across processes.
    """
    global _WORLD
    if _WORLD is not None:
        return _WORLD
    backend = _detect_backend()
    if force_single or not mx.distributed.is_available():
        _WORLD = WorldInfo(world_size=1, rank=0, backend="single",
                            real=False)
        return _WORLD
    try:
        group = mx.distributed.init()
        ws = int(group.size())
        rk = int(group.rank())
        _WORLD = WorldInfo(world_size=ws, rank=rk,
                            backend=backend, real=ws > 1)
    except Exception:
        _WORLD = WorldInfo(world_size=1, rank=0, backend="single",
                            real=False)
    return _WORLD


def world() -> WorldInfo:
    return init()


def is_distributed() -> bool:
    return world().real


def reset_for_test() -> None:
    """Test helper — drop the cached WorldInfo so init() re-runs."""
    global _WORLD
    _WORLD = None


def all_reduce(x: mx.array,
                op: Literal["sum", "mean", "max", "min"] = "sum",
                ) -> mx.array:
    """Element-wise reduction across all ranks.

    Single-process (world_size==1): returns x unchanged for sum/max/min
    and x/1 for mean (still identity). Multi-process: real
    mx.distributed call. mean is sum / world_size since mx.distributed
    has no all_mean primitive.
    """
    w = world()
    if not w.real:
        return x
    if op == "sum":
        return mx.distributed.all_sum(x)
    if op == "max":
        return mx.distributed.all_max(x)
    if op == "min":
        return mx.distributed.all_min(x)
    # mean.
    s = mx.distributed.all_sum(x)
    return s / float(w.world_size)


def all_gather(x: mx.array, axis: int = 0) -> mx.array:
    """Concat shard from every rank along `axis`.

    Single-process: replicates x along axis (matches the proxy contract).
    Multi-process: real mx.distributed.all_gather (which concats along
    axis=0 by default; we transpose into place if axis!=0).
    """
    w = world()
    if not w.real:
        # Match the legacy proxy: replicate W times along axis.
        return mx.concatenate([x] * max(1, w.world_size), axis=axis)
    if axis == 0:
        return mx.distributed.all_gather(x)
    # Move target axis to 0, gather, move back.
    perm = list(range(x.ndim))
    perm[0], perm[axis] = perm[axis], perm[0]
    return mx.transpose(
        mx.distributed.all_gather(mx.transpose(x, perm)), perm)


def reduce_scatter(x: mx.array, axis: int = 0,
                    op: Literal["sum", "mean"] = "mean") -> mx.array:
    """Sum across ranks, then keep only this rank's chunk along `axis`.

    mx.distributed.sum_scatter implements sum+split; we divide by
    world_size to get the mean variant.
    """
    w = world()
    if not w.real:
        # Single-process: split into world_size chunks along axis,
        # mean-reduce them. With world_size==1 returns x.
        if w.world_size == 1:
            return x
        chunk = x.shape[axis] // w.world_size
        pieces = [
            mx.take(x, mx.arange(i * chunk, (i + 1) * chunk), axis=axis)
            for i in range(w.world_size)
        ]
        acc = pieces[0]
        for p in pieces[1:]:
            acc = acc + p
        return acc / float(w.world_size) if op == "mean" else acc
    out = mx.distributed.sum_scatter(x)
    return out / float(w.world_size) if op == "mean" else out


def send(x: mx.array, dst: int) -> None:
    if not world().real:
        return
    mx.distributed.send(x, dst)


def recv_like(template: mx.array, src: int) -> mx.array:
    if not world().real:
        return template
    return mx.distributed.recv_like(template, src)


def barrier() -> None:
    """Synchronization point. Implemented as an all_sum of a tiny
    scalar — mx.distributed has no explicit barrier primitive."""
    if not world().real:
        return
    sentinel = mx.zeros((1,), dtype=mx.float32)
    mx.eval(mx.distributed.all_sum(sentinel))


def measure_collective_overhead_ms(
    *, shard_size: int, n_iter: int = 50,
) -> dict[str, float]:
    """Wallclock benchmark of the four collectives at the active
    world_size. Pairs with the V7-B05 NCCL-bench script."""
    w = world()
    shard = mx.random.normal(shape=(shard_size,), key=mx.random.key(0))
    mx.eval(shard)

    # Warm.
    _ = all_reduce(shard); mx.eval(_)
    _ = all_gather(shard); mx.eval(_)
    if shard.shape[0] % max(1, w.world_size) == 0:
        _ = reduce_scatter(shard); mx.eval(_)

    def _bench(fn) -> float:
        t0 = time.perf_counter()
        for _ in range(n_iter):
            y = fn(shard)
            mx.eval(y)
        return (time.perf_counter() - t0) * 1000.0 / n_iter

    ar_ms = _bench(lambda s: all_reduce(s, op="sum"))
    ag_ms = _bench(lambda s: all_gather(s))
    rs_ms = (_bench(lambda s: reduce_scatter(s))
             if shard.shape[0] % max(1, w.world_size) == 0
             else float("nan"))
    return {
        "world_size": float(w.world_size),
        "rank": float(w.rank),
        "shard_size": float(shard_size),
        "backend": w.backend,
        "real": float(w.real),
        "all_reduce_ms_per_iter": round(ar_ms, 4),
        "all_gather_ms_per_iter": round(ag_ms, 4),
        "reduce_scatter_ms_per_iter": (
            round(rs_ms, 4) if rs_ms == rs_ms else None),
    }


__all__ = [
    "WorldInfo", "init", "world", "is_distributed", "reset_for_test",
    "all_reduce", "all_gather", "reduce_scatter",
    "send", "recv_like", "barrier",
    "measure_collective_overhead_ms",
]
