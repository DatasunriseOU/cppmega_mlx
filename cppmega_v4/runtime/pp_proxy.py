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
    """Yield (microbatch_idx, stage_idx) pairs for a naive synchronous
    PP schedule — for diagnostic / scheduler-test purposes. For real
    1F1B / GPipe order, use gpipe_schedule / one_f_one_b_schedule."""
    for mb in range(num_microbatches):
        for st in range(num_stages):
            yield (mb, st)


# V7-B04 (real): explicit GPipe and 1F1B (synchronous) schedules.
# Each schedule yields ('F', stage, mb) and ('B', stage, mb) ops in
# the order a real PP scheduler would dispatch them. Send/recv between
# stages is delegated to cppmega_v4.runtime.distributed.send/recv_like
# so a launcher with world_size==num_stages produces real cross-rank
# activation passing.


def gpipe_schedule(num_microbatches: int, num_stages: int
                    ) -> Iterator[tuple[str, int, int]]:
    """GPipe: all forwards first across microbatches, then all backwards.

    Memory cost: O(num_microbatches) activations stashed concurrently —
    the price of full pipeline fill before any backward.
    """
    for mb in range(num_microbatches):
        for st in range(num_stages):
            yield ("F", st, mb)
    for mb in reversed(range(num_microbatches)):
        for st in reversed(range(num_stages)):
            yield ("B", st, mb)


def one_f_one_b_schedule(num_microbatches: int, num_stages: int
                          ) -> Iterator[tuple[str, int, int]]:
    """1F1B (synchronous): interleave forward + backward to bound the
    activation memory at O(num_stages).

    Phases:
      * warmup        — first num_stages-1 forwards on each stage.
      * steady (1F1B) — pairs of (F next_mb, B oldest_mb) across the
                        remaining microbatches.
      * cooldown      — drain the remaining backwards.

    Output is a flat op stream the test/scheduler consumes. The
    contract matches Megatron-LM's PipelineParallel1F1B order.
    """
    # Warmup: send a forward through each stage; the first stage sees
    # microbatch 0 first, then 1, etc. up to num_stages-1.
    pending_b: list[int] = []
    next_f = 0
    # The number of warmup forwards equals num_stages - 1 (last stage
    # immediately starts its backward in the steady state).
    warmup_count = min(num_stages - 1, num_microbatches)
    for _ in range(warmup_count):
        for st in range(num_stages):
            yield ("F", st, next_f)
        pending_b.append(next_f)
        next_f += 1
    # Steady: pair each new forward with the oldest pending backward.
    while next_f < num_microbatches:
        for st in range(num_stages):
            yield ("F", st, next_f)
        b_mb = pending_b.pop(0)
        for st in reversed(range(num_stages)):
            yield ("B", st, b_mb)
        pending_b.append(next_f)
        next_f += 1
    # Cooldown: drain remaining backwards.
    while pending_b:
        b_mb = pending_b.pop(0)
        for st in reversed(range(num_stages)):
            yield ("B", st, b_mb)


def pipeline_forward_real(
    x: mx.array,
    stages: list[Callable[[mx.array], mx.array]],
    *,
    num_microbatches: int,
    schedule: str = "1f1b",
) -> mx.array:
    """Multi-process-capable PP forward. When world_size>1 and matches
    len(stages), each rank owns one stage and activations flow via
    distributed.send/recv_like along the pipeline. When world_size==1
    the same code runs sequentially on one process — identical math.

    Returns the concatenated final-stage outputs in microbatch order.
    """
    from cppmega_v4.runtime import distributed as _d
    if not stages:
        return x
    mbs = split_microbatches(x, num_microbatches)
    w = _d.world()
    n_stages = len(stages)
    use_real = w.real and w.world_size == n_stages
    # Activations indexed by (stage, mb) — only kept until the next
    # stage consumes them so memory stays O(pipeline_depth).
    cache: dict[tuple[int, int], mx.array] = {}
    final_out: dict[int, mx.array] = {}
    ops = (one_f_one_b_schedule(num_microbatches, n_stages)
           if schedule == "1f1b"
           else gpipe_schedule(num_microbatches, n_stages))
    for phase, stage_idx, mb_idx in ops:
        if phase != "F":
            # Backward ops are no-ops in this forward-only proxy.
            continue
        if use_real and stage_idx != w.rank:
            continue
        # Input for this stage/microbatch.
        if stage_idx == 0:
            inp = mbs[mb_idx]
        else:
            key = (stage_idx - 1, mb_idx)
            if use_real:
                template = mbs[mb_idx]
                inp = _d.recv_like(template, stage_idx - 1)
            else:
                inp = cache.pop(key)
        out = stages[stage_idx](inp)
        if stage_idx == n_stages - 1:
            final_out[mb_idx] = out
        else:
            if use_real:
                _d.send(out, stage_idx + 1)
            else:
                cache[(stage_idx, mb_idx)] = out
    # Collect final outputs in microbatch order.
    if use_real and w.rank != n_stages - 1:
        # Non-last ranks have nothing to return; caller decides.
        return mx.zeros((0,))
    return mx.concatenate([final_out[mb] for mb in range(num_microbatches)],
                           axis=0)


__all__ = ["split_microbatches", "pipeline_forward",
           "pipeline_schedule", "gpipe_schedule",
           "one_f_one_b_schedule", "pipeline_forward_real"]
