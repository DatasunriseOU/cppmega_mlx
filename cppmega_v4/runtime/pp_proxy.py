"""V7-B04: pipeline-parallelism microbatch staging proxy.

Splits a (B, S, H) batch into N microbatches, runs each through a
chain of stage callables sequentially, and yields per-stage outputs
in order. The mathematical equivalent of PP forward without the
multi-process scheduler (which lives under V7-B01).
"""

from __future__ import annotations

from typing import Callable, Iterator

import mlx.core as mx


def split_microbatches(x: mx.array, num_microbatches: int
                        ) -> list[mx.array]:
    if num_microbatches < 1:
        raise ValueError("num_microbatches must be >= 1")
    if x.shape[0] % num_microbatches != 0:
        raise ValueError(
            f"batch {x.shape[0]} not divisible by {num_microbatches}")
    chunk = x.shape[0] // num_microbatches
    return [x[i * chunk:(i + 1) * chunk]
            for i in range(num_microbatches)]


def pipeline_forward(
    x: mx.array,
    stages: list[Callable[[mx.array], mx.array]],
    *,
    num_microbatches: int,
) -> mx.array:
    """Run x through stages with microbatching; concat results."""
    if not stages:
        return x
    mbs = split_microbatches(x, num_microbatches)
    out_mbs: list[mx.array] = []
    for mb in mbs:
        cur = mb
        for stage in stages:
            cur = stage(cur)
        out_mbs.append(cur)
    return mx.concatenate(out_mbs, axis=0)


def pipeline_schedule(num_microbatches: int, num_stages: int
                       ) -> Iterator[tuple[int, int]]:
    """Yield (microbatch_idx, stage_idx) pairs for a naive 1F1B-style
    PP schedule — for diagnostic / scheduler-test purposes."""
    for mb in range(num_microbatches):
        for st in range(num_stages):
            yield (mb, st)


__all__ = ["split_microbatches", "pipeline_forward",
           "pipeline_schedule"]
