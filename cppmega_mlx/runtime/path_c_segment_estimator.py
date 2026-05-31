"""Static per-segment resource estimator (Module 2 of the auto-split scheme).

All four estimates are computed at PLAN time from data already present (the
generated kernel source attached to the prim_func, the buffer_map, and the
``PathCModelShapeEnv``), with zero runtime measurement.  The estimator is called
at the same greedy accept/reject point where the buffer count is checked today,
so it adds no new compilation cost.

See ``docs/HW-AWARE-AUTOSPLIT-DESIGN.md`` §2.  The throughput / per-op-per-row
coefficients are NOT hardcoded here -- they come from the device-capability
record (``DeviceCaps``), which fills them from the preset table or TIER-3
calibration.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
import math
from typing import Any

from cppmega_mlx.runtime.path_c_device_caps import DeviceCaps
from cppmega_mlx.runtime.path_c_fusion import PathCModelShapeEnv


@dataclass(frozen=True)
class SegmentEstimate:
    """Static plan-time resource estimate for one candidate fused segment."""

    logical_shared_bytes: int
    physical_shared_bytes: int
    buffer_arg_count: int
    est_gpu_time_s: float
    is_recurrent: bool
    msl_source_bytes: int
    per_row_time_s: float  # est_gpu_time_s / S (independent ops only)


# ---------------------------------------------------------------------------
# 2.1 -- threadgroup-memory bytes (reuse the existing alloc_shared machinery).
# ---------------------------------------------------------------------------


def _coerce_shape(shape_text: str) -> tuple[int, ...]:
    value = ast.literal_eval(shape_text)
    if isinstance(value, int):
        return (int(value),)
    return tuple(int(dim) for dim in value)


def shared_bytes_from_source(
    source: str,
    caps: DeviceCaps,
    *,
    alloc_shared_re: Any,
    dtype_nbytes: dict[str, int],
    flattened_extent: Any,
) -> tuple[int, int]:
    """Return (logical_shared_bytes, physical_shared_bytes) from generated source.

    ``physical = ceil(logical * caps.logical_to_physical_shared_margin)`` -- the
    PHYSICAL value is what must be compared against ``caps.threadgroup_mem_bytes``
    (TileLang coalesces+pads the residual alloc_shared into one buf_dyn_shmem
    array running a few x the logical; comparing raw logical vs the cap is wrong).
    """

    logical_total = 0
    for line in source.splitlines():
        match = alloc_shared_re.match(line)
        if match is None:
            continue
        shape = _coerce_shape(match.group("shape"))
        logical_total += flattened_extent(shape) * dtype_nbytes[match.group("dtype")]
    physical = int(math.ceil(logical_total * caps.logical_to_physical_shared_margin))
    return logical_total, physical


# ---------------------------------------------------------------------------
# 2.3 -- GPU-time estimate (closed-form roofline + direct per-op-per-row coeffs).
# ---------------------------------------------------------------------------


def _op_flops(op_name: str, env: PathCModelShapeEnv) -> float:
    """Closed-form FLOP count per op-class, authored from the descriptor stages.

    These back the general roofline case when no direct per-op-per-row coefficient
    is calibrated; the watchdog-relevant ops use the direct coefficient instead.
    """

    s = float(env.sequence_length)
    hidden = float(env.hidden_size)
    base = op_name[: -len("_bwd")] if op_name.endswith("_bwd") else op_name
    if base in ("attention_qkv_projection",):
        # Q/KV linear projection: ~2 * S * hidden * (q_dim + kv_dim)
        q_dim = float(env.attention_num_q_heads * env.attention_head_dim)
        kv_dim = float(env.attention_num_kv_heads * env.attention_head_dim)
        return 2.0 * s * hidden * (q_dim + kv_dim)
    if base in ("sparse_mla_fp8_apply",):
        # sparse top-k attention: ~ S * topk * head_dim * num_q_heads * 2 (QK + AV)
        return (
            2.0
            * s
            * float(env.attention_sparse_topk)
            * float(env.attention_head_dim)
            * float(env.attention_num_q_heads)
        )
    if base in ("mamba3_mimo",):
        # in-projection dominates: ~2 * S * hidden * in_proj_dim
        return 2.0 * s * hidden * float(env.mamba_in_proj_dim)
    if base in ("m2rnn",):
        return 2.0 * s * hidden * hidden
    if base in ("residual_rmsnorm", "entry_rmsnorm"):
        return s * hidden * 8.0  # sum-of-squares + scale
    return s * hidden  # conservative default


def _op_bytes(op_name: str, env: PathCModelShapeEnv) -> float:
    s = float(env.sequence_length)
    hidden = float(env.hidden_size)
    bytes_per = float({"float32": 4}.get(env.model_value_dtype, 2))
    base = op_name[: -len("_bwd")] if op_name.endswith("_bwd") else op_name
    if base in ("attention_qkv_projection", "sparse_mla_fp8_apply", "mamba3_mimo", "m2rnn"):
        return 3.0 * s * hidden * bytes_per  # read input + weights touch + write
    return 2.0 * s * hidden * bytes_per


def est_op_gpu_time_s(op_name: str, env: PathCModelShapeEnv, caps: DeviceCaps) -> float:
    """Per-op GPU-time estimate (seconds).

    Preferred: a directly-calibrated per-op-per-row coefficient (preset/TIER-3).
    Fallback within the calibration domain: roofline from FLOPs/bytes.
    """

    coeff = caps.per_op_time_per_row_s.get(op_name)
    if coeff is not None:
        return coeff * float(env.sequence_length)
    if caps.effective_flop_s <= 0 or caps.effective_bytes_s <= 0:
        # No throughput model available and no direct coefficient: cannot estimate.
        # Return 0.0 so the time predicate never fires from a guessed number; the
        # watchdog-relevant ops always carry a direct coefficient in the preset.
        return 0.0
    flops = _op_flops(op_name, env)
    bytes_ = _op_bytes(op_name, env)
    return max(flops / caps.effective_flop_s, bytes_ / caps.effective_bytes_s)


# ---------------------------------------------------------------------------
# Assembly.
# ---------------------------------------------------------------------------


def estimate_segment_from_source(
    *,
    source: str,
    op_names: tuple[str, ...],
    buffer_arg_count: int,
    env: PathCModelShapeEnv | None,
    caps: DeviceCaps,
    is_recurrent: bool,
    alloc_shared_re: Any,
    dtype_nbytes: dict[str, int],
    flattened_extent: Any,
) -> SegmentEstimate:
    """Build a :class:`SegmentEstimate` from already-generated artifacts.

    ``source`` is ``prim_func._cppmega_path_c_generated_source`` (free -- attached
    before metallib compilation).  ``buffer_arg_count`` is the
    ``_kernel_parameter_count_for_target`` result (the buffer_map-only count).
    """

    logical_shared, physical_shared = shared_bytes_from_source(
        source,
        caps,
        alloc_shared_re=alloc_shared_re,
        dtype_nbytes=dtype_nbytes,
        flattened_extent=flattened_extent,
    )
    msl_source_bytes = len(source)

    est_gpu_time_s = 0.0
    per_row_time_s = 0.0
    if env is not None:
        est_gpu_time_s = sum(
            est_op_gpu_time_s(op_name, env, caps) for op_name in op_names
        )
        seq = max(1, int(env.sequence_length))
        per_row_time_s = est_gpu_time_s / seq

    return SegmentEstimate(
        logical_shared_bytes=logical_shared,
        physical_shared_bytes=physical_shared,
        buffer_arg_count=int(buffer_arg_count),
        est_gpu_time_s=est_gpu_time_s,
        is_recurrent=is_recurrent,
        msl_source_bytes=msl_source_bytes,
        per_row_time_s=per_row_time_s,
    )


# ---------------------------------------------------------------------------
# 4.7 -- derived chunking parameters (replace the hardcoded 64 / time-chunk).
# ---------------------------------------------------------------------------


def watchdog_safe_max_rows_per_launch(
    est: SegmentEstimate, caps: DeviceCaps
) -> int | None:
    """The largest row window whose per-launch GPU time fits the watchdog budget.

    ``floor(watchdog_window_s * safety / per_row_time_s)`` -- the watchdog-SAFE
    upper bound.  Returns ``None`` when there is no watchdog window or no per-row
    time to derive from (caller should then keep the descriptor default).
    """

    if caps.watchdog_window_s is None or est.per_row_time_s <= 0:
        return None
    budget_s = caps.watchdog_window_s * caps.safety_margin
    return int(math.floor(budget_s / est.per_row_time_s))


def derived_max_rows_per_launch(
    est: SegmentEstimate,
    caps: DeviceCaps,
    *,
    descriptor_default: int = 64,
    row_granularity: int = 1,
) -> int:
    """Device-derived watchdog-safe row window, capped by the descriptor default.

    Replaces the hardcoded ``DESCRIPTOR_DEFAULT_MAX_ROWS_PER_LAUNCH = 64`` with
    ``min(watchdog_safe_bound, descriptor_default)``:

      * The watchdog-safe bound is the hardware-derived ceiling
        (``floor(window * safety / per_row_time)``); on a device with a TIGHTER
        watchdog this shrinks the window below the descriptor default, and the
        planner RAISES PathCSplitInfeasible if even one row cannot fit.
      * The descriptor default (64) is the conservative hand-tuned row granularity
        the launcher emits per command buffer.  At the M4 Max / local_gb10_quarter
        scale the watchdog-safe bound (~1024) comfortably exceeds 64, so the
        derived value lands on the hand-tuned 64 (design §4.7 / §7.1).

    When no watchdog window / per-row time is available the descriptor default is
    returned unchanged (behaviour preserved at the calibration scale).
    """

    safe = watchdog_safe_max_rows_per_launch(est, caps)
    if safe is None:
        rows = descriptor_default
    else:
        rows = min(safe, descriptor_default)
    if rows <= 0:
        return max(1, row_granularity)
    if row_granularity > 1:
        rows = (rows // row_granularity) * row_granularity
    return max(row_granularity, rows)


def derived_time_chunk_count(est: SegmentEstimate, caps: DeviceCaps) -> int:
    """ceil(est_gpu_time_s / (watchdog_window_s * safety)) for recurrent ops."""

    if caps.watchdog_window_s is None or est.est_gpu_time_s <= 0:
        return 1
    budget_s = caps.watchdog_window_s * caps.safety_margin
    return max(1, int(math.ceil(est.est_gpu_time_s / budget_s)))
