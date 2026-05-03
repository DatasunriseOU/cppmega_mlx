"""Path C Sparse-MLA forward/backward via TileLang DSL ``@T.prim_func`` lowering.

This module is the first TileLang-DSL counterpart to the Path B direct-MSL
Sparse-MLA kernels in :mod:`cppmega_mlx.nn._tilelang.sparse_mla`.

The upstream TileLang Sparse-MLA backward examples are CUDA-oriented: they use
``T.gemm`` for the attention matmuls and atomics for dKV scatter. Both are
still the wrong first step on Apple Metal. This Path C kernel instead mirrors
Path B's partial-output contract:

* compute ``dq`` directly;
* emit ``dkv_partial[B, S, H, topk, D]`` without atomics;
* reuse Path B's host-side ``_reduce_dkv_partial`` scatter/reduction.

The kernel keeps the TOPK softmax state in static threadgroup buffers and uses
power-of-two tree reductions for the max/sum/rowsum phases. That mirrors Path
B's direct-MSL contract while keeping the source in TileLang DSL.
"""

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, cast

import mlx.core as mx

from cppmega_mlx.nn._tilelang import _msl_transform
from cppmega_mlx.nn._tilelang._msl_transform import (
    MSLDispatchUnsupported,
    can_run_metal,
    lower_tilelang_to_msl_inline,
)
from cppmega_mlx.nn._tilelang.sparse_mla import _reduce_dkv_partial
from cppmega_mlx.nn.sparse_mla import _resolve_shapes


@dataclass(frozen=True)
class SparseMLAPathCStatus:
    """Runtime status for the Path C TileLang DSL Sparse-MLA backward kernel."""

    available: bool
    reason: str
    fp32_carrier: bool = True


def _tilelang_available() -> tuple[bool, str]:
    try:
        import tilelang  # noqa: F401
        from tilelang import tvm as _tvm  # noqa: F401
        from tilelang.engine.lower import lower as _lower  # noqa: F401
        import tilelang.language as _T  # noqa: F401
    except Exception as exc:  # pragma: no cover - macOS without tilelang
        return False, f"tilelang import failed: {exc}"
    return True, "tilelang importable"


def sparse_mla_path_c_status() -> SparseMLAPathCStatus:
    """Return whether the Path C TileLang DSL kernel can dispatch on this host."""

    if not can_run_metal():
        return SparseMLAPathCStatus(
            available=False,
            reason="MLX Metal backend is not available on the default GPU device",
        )
    ok, reason = _tilelang_available()
    if not ok:
        return SparseMLAPathCStatus(available=False, reason=reason)
    return SparseMLAPathCStatus(
        available=True,
        reason="Sparse-MLA Path C forward/backward TileLang DSL ready",
    )


def _threadgroup_size(topk: int) -> int:
    """Match Path B's power-of-two threadgroup sizing for TOPK reductions."""

    threads = min(64, max(1, topk))
    power = 1
    while (power << 1) <= threads:
        power <<= 1
    return power


@lru_cache(maxsize=128)
def _fwd_kernel_for(
    BATCH: int,
    SEQ_LEN: int,
    HEADS: int,
    QK_DIM: int,
    KV_GROUP: int,
    HEAD_KV: int,
    TOPK: int,
    SEQ_LEN_KV: int,
    D_V: int,
    THREADS: int,
) -> tuple[Any, _msl_transform.TileLangMSLLowering]:
    """Build and cache a shape-specialized threadgroup Sparse-MLA fwd kernel."""

    import tilelang.language as T

    LANES = BATCH * SEQ_LEN * HEADS
    LOG_THREADS = THREADS.bit_length() - 1

    @T.prim_func
    def sparse_mla_fwd(
        q: T.Tensor((BATCH, SEQ_LEN, HEADS, QK_DIM), "float32"),
        kv: T.Tensor((BATCH, SEQ_LEN_KV, KV_GROUP, QK_DIM), "float32"),
        indices: T.Tensor((BATCH, SEQ_LEN, KV_GROUP, TOPK), "int32"),
        sm_scale_buf: T.Tensor((1,), "float32"),
        out: T.Tensor((BATCH, SEQ_LEN, HEADS, D_V), "float32"),
        lse: T.Tensor((BATCH, SEQ_LEN, HEADS), "float32"),
    ):
        with T.Kernel(LANES, threads=THREADS) as bx:
            lane = T.get_thread_binding()
            scores = T.alloc_shared((TOPK,), "float32", scope="shared")
            reduce_buf = T.alloc_shared((THREADS,), "float32", scope="shared")
            acc = T.alloc_local((1,), "float32")
            local = T.alloc_local((1,), "float32")
            inv_sum = T.alloc_local((1,), "float32")
            stride = T.alloc_local((1,), "int32")
            gather_idx = T.alloc_local((1,), "int32")

            h = bx % HEADS
            s = (bx // HEADS) % SEQ_LEN
            b = bx // (HEADS * SEQ_LEN)
            g = h // HEAD_KV
            sm_scale = sm_scale_buf[0]

            for k in T.serial(lane, TOPK, step=THREADS):
                gather_idx[0] = indices[b, s, g, k]
                if gather_idx[0] < 0:
                    scores[k] = -T.infinity("float32")
                else:
                    acc[0] = 0.0
                    for d in T.serial(QK_DIM):
                        acc[0] = acc[0] + q[b, s, h, d] * kv[b, gather_idx[0], g, d]
                    scores[k] = acc[0] * sm_scale
            T.sync_threads()

            local[0] = -T.infinity("float32")
            for k in T.serial(lane, TOPK, step=THREADS):
                if scores[k] > local[0]:
                    local[0] = scores[k]
            reduce_buf[lane] = local[0]
            T.sync_threads()
            for round_id in T.serial(LOG_THREADS):
                stride[0] = T.shift_right(THREADS, round_id + 1)
                if lane < stride[0]:
                    if reduce_buf[lane + stride[0]] > reduce_buf[lane]:
                        reduce_buf[lane] = reduce_buf[lane + stride[0]]
                T.sync_threads()
            row_max = reduce_buf[0]

            for k in T.serial(lane, TOPK, step=THREADS):
                if scores[k] == -T.infinity("float32"):
                    scores[k] = 0.0
                else:
                    scores[k] = T.exp(scores[k] - row_max)
            T.sync_threads()

            local[0] = 0.0
            for k in T.serial(lane, TOPK, step=THREADS):
                local[0] = local[0] + scores[k]
            reduce_buf[lane] = local[0]
            T.sync_threads()
            for round_id in T.serial(LOG_THREADS):
                stride[0] = T.shift_right(THREADS, round_id + 1)
                if lane < stride[0]:
                    reduce_buf[lane] = reduce_buf[lane] + reduce_buf[lane + stride[0]]
                T.sync_threads()
            sumexp = reduce_buf[0]

            inv_sum[0] = 0.0
            if sumexp > 0.0:
                inv_sum[0] = 1.0 / sumexp

            for d in T.serial(lane, D_V, step=THREADS):
                acc[0] = 0.0
                for k in T.serial(TOPK):
                    gather_idx[0] = indices[b, s, g, k]
                    if gather_idx[0] >= 0:
                        acc[0] = acc[0] + scores[k] * kv[b, gather_idx[0], g, d]
                out[b, s, h, d] = acc[0] * inv_sum[0]

            if lane == 0:
                if sumexp > 0.0:
                    lse[b, s, h] = row_max + T.log(sumexp)
                else:
                    lse[b, s, h] = 0.0

    lowering = lower_tilelang_to_msl_inline(sparse_mla_fwd)
    kernel = mx.fast.metal_kernel(
        name=(
            "cppmega_sparse_mla_path_c_fwd_"
            f"{BATCH}_{SEQ_LEN}_{HEADS}_{QK_DIM}_{KV_GROUP}_{TOPK}_{SEQ_LEN_KV}_{D_V}_{THREADS}"
        ),
        input_names=["indices", "kv", "q", "sm_scale_buf"],
        output_names=["lse", "out"],
        source=lowering.body,
        header=lowering.header,
        ensure_row_contiguous=True,
    )
    return kernel, lowering


@lru_cache(maxsize=128)
def _bwd_kernel_for(
    BATCH: int,
    SEQ_LEN: int,
    HEADS: int,
    QK_DIM: int,
    KV_GROUP: int,
    HEAD_KV: int,
    TOPK: int,
    SEQ_LEN_KV: int,
    D_V: int,
    THREADS: int,
) -> tuple[Any, _msl_transform.TileLangMSLLowering]:
    """Build and cache a shape-specialized threadgroup Sparse-MLA bwd kernel."""

    import tilelang.language as T

    LANES = BATCH * SEQ_LEN * HEADS
    LOG_THREADS = THREADS.bit_length() - 1

    @T.prim_func
    def sparse_mla_bwd(
        q: T.Tensor((BATCH, SEQ_LEN, HEADS, QK_DIM), "float32"),
        kv: T.Tensor((BATCH, SEQ_LEN_KV, KV_GROUP, QK_DIM), "float32"),
        d_out: T.Tensor((BATCH, SEQ_LEN, HEADS, D_V), "float32"),
        indices: T.Tensor((BATCH, SEQ_LEN, KV_GROUP, TOPK), "int32"),
        sm_scale_buf: T.Tensor((1,), "float32"),
        dq: T.Tensor((BATCH, SEQ_LEN, HEADS, QK_DIM), "float32"),
        dkv_partial: T.Tensor((BATCH, SEQ_LEN, HEADS, TOPK, QK_DIM), "float32"),
    ):
        with T.Kernel(LANES, threads=THREADS) as bx:
            lane = T.get_thread_binding()
            scores = T.alloc_shared((TOPK,), "float32", scope="shared")
            p = T.alloc_shared((TOPK,), "float32", scope="shared")
            dp = T.alloc_shared((TOPK,), "float32", scope="shared")
            ds = T.alloc_shared((TOPK,), "float32", scope="shared")
            reduce_buf = T.alloc_shared((THREADS,), "float32", scope="shared")
            acc = T.alloc_local((1,), "float32")
            local = T.alloc_local((1,), "float32")
            inv_sum = T.alloc_local((1,), "float32")
            stride = T.alloc_local((1,), "int32")
            gather_idx = T.alloc_local((1,), "int32")

            h = bx % HEADS
            s = (bx // HEADS) % SEQ_LEN
            b = bx // (HEADS * SEQ_LEN)
            g = h // HEAD_KV
            sm_scale = sm_scale_buf[0]

            for k in T.serial(lane, TOPK, step=THREADS):
                gather_idx[0] = indices[b, s, g, k]
                if gather_idx[0] < 0:
                    scores[k] = -T.infinity("float32")
                else:
                    acc[0] = 0.0
                    for d in T.serial(QK_DIM):
                        acc[0] = acc[0] + q[b, s, h, d] * kv[b, gather_idx[0], g, d]
                    scores[k] = acc[0] * sm_scale
            T.sync_threads()

            local[0] = -T.infinity("float32")
            for k in T.serial(lane, TOPK, step=THREADS):
                if scores[k] > local[0]:
                    local[0] = scores[k]
            reduce_buf[lane] = local[0]
            T.sync_threads()
            for round_id in T.serial(LOG_THREADS):
                stride[0] = T.shift_right(THREADS, round_id + 1)
                if lane < stride[0]:
                    if reduce_buf[lane + stride[0]] > reduce_buf[lane]:
                        reduce_buf[lane] = reduce_buf[lane + stride[0]]
                T.sync_threads()
            row_max = reduce_buf[0]

            for k in T.serial(lane, TOPK, step=THREADS):
                gather_idx[0] = indices[b, s, g, k]
                if gather_idx[0] < 0:
                    p[k] = 0.0
                else:
                    p[k] = T.exp(scores[k] - row_max)
            T.sync_threads()

            local[0] = 0.0
            for k in T.serial(lane, TOPK, step=THREADS):
                local[0] = local[0] + p[k]
            reduce_buf[lane] = local[0]
            T.sync_threads()
            for round_id in T.serial(LOG_THREADS):
                stride[0] = T.shift_right(THREADS, round_id + 1)
                if lane < stride[0]:
                    reduce_buf[lane] = reduce_buf[lane] + reduce_buf[lane + stride[0]]
                T.sync_threads()
            sumexp = reduce_buf[0]
            inv_sum[0] = 0.0
            if sumexp > 0.0:
                inv_sum[0] = 1.0 / sumexp

            for k in T.serial(lane, TOPK, step=THREADS):
                p[k] = p[k] * inv_sum[0]
            T.sync_threads()

            for k in T.serial(lane, TOPK, step=THREADS):
                gather_idx[0] = indices[b, s, g, k]
                if gather_idx[0] < 0:
                    dp[k] = 0.0
                else:
                    acc[0] = 0.0
                    for d in T.serial(D_V):
                        acc[0] = acc[0] + kv[b, gather_idx[0], g, d] * d_out[b, s, h, d]
                    dp[k] = acc[0]
            T.sync_threads()

            local[0] = 0.0
            for k in T.serial(lane, TOPK, step=THREADS):
                local[0] = local[0] + p[k] * dp[k]
            reduce_buf[lane] = local[0]
            T.sync_threads()
            for round_id in T.serial(LOG_THREADS):
                stride[0] = T.shift_right(THREADS, round_id + 1)
                if lane < stride[0]:
                    reduce_buf[lane] = reduce_buf[lane] + reduce_buf[lane + stride[0]]
                T.sync_threads()
            rowsum = reduce_buf[0]

            for k in T.serial(lane, TOPK, step=THREADS):
                ds[k] = p[k] * (dp[k] - rowsum)
            T.sync_threads()

            for d in T.serial(lane, QK_DIM, step=THREADS):
                acc[0] = 0.0
                for k in T.serial(TOPK):
                    gather_idx[0] = indices[b, s, g, k]
                    if gather_idx[0] >= 0:
                        acc[0] = acc[0] + ds[k] * kv[b, gather_idx[0], g, d]
                dq[b, s, h, d] = acc[0] * sm_scale

            for kd in T.serial(lane, TOPK * QK_DIM, step=THREADS):
                k = kd // QK_DIM
                d = kd % QK_DIM
                gather_idx[0] = indices[b, s, g, k]
                if gather_idx[0] < 0:
                    dkv_partial[b, s, h, k, d] = 0.0
                else:
                    acc[0] = sm_scale * ds[k] * q[b, s, h, d]
                    if d < D_V:
                        dkv_partial[b, s, h, k, d] = p[k] * d_out[b, s, h, d] + acc[0]
                    else:
                        dkv_partial[b, s, h, k, d] = acc[0]

    lowering = lower_tilelang_to_msl_inline(sparse_mla_bwd)
    kernel = mx.fast.metal_kernel(
        name=(
            "cppmega_sparse_mla_path_c_bwd_"
            f"{BATCH}_{SEQ_LEN}_{HEADS}_{QK_DIM}_{KV_GROUP}_{TOPK}_{SEQ_LEN_KV}_{D_V}_{THREADS}"
        ),
        input_names=["d_out", "indices", "kv", "q", "sm_scale_buf"],
        output_names=["dkv_partial", "dq"],
        source=lowering.body,
        header=lowering.header,
        ensure_row_contiguous=True,
    )
    return kernel, lowering


def sparse_mla_fwd_path_c(
    q: mx.array,
    kv: mx.array,
    indices: mx.array,
    *,
    sm_scale: float | None = None,
    d_v: int | None = None,
) -> tuple[mx.array, mx.array] | None:
    """TileLang DSL Path C Sparse-MLA forward.

    Returns ``(out, lse)`` or ``None`` if the Metal/TileLang path cannot be
    built. The kernel uses fp32 carrier/accumulators and casts ``out`` back to
    the input dtype so BF16 tests exercise the BF16 public contract.
    """

    status = sparse_mla_path_c_status()
    if not status.available:
        return None

    shapes = _resolve_shapes(q, kv, indices, d_v=d_v)
    if sm_scale is None:
        sm_scale_value = shapes.qk_dim ** -0.5
    else:
        sm_scale_value = sm_scale
    threads = _threadgroup_size(shapes.topk)

    out_dtype = q.dtype
    q32 = q.astype(mx.float32)
    kv32 = kv.astype(mx.float32)
    indices_i32 = indices.astype(mx.int32)
    sm_scale_buf = mx.array([float(sm_scale_value)], dtype=mx.float32)

    try:
        kernel, lowering = _fwd_kernel_for(
            shapes.batch,
            shapes.seq_len,
            shapes.heads,
            shapes.qk_dim,
            shapes.kv_group,
            shapes.head_kv,
            shapes.topk,
            shapes.seq_len_kv,
            shapes.d_v,
            threads,
        )
    except (MSLDispatchUnsupported, RuntimeError, ValueError):
        return None

    grid = (
        lowering.grid[0] * lowering.threadgroup[0],
        lowering.grid[1] * lowering.threadgroup[1],
        lowering.grid[2] * lowering.threadgroup[2],
    )

    try:
        outputs = kernel(
            inputs=[indices_i32, kv32, q32, sm_scale_buf],
            output_shapes=[
                (shapes.batch, shapes.seq_len, shapes.heads),
                (shapes.batch, shapes.seq_len, shapes.heads, shapes.d_v),
            ],
            output_dtypes=[mx.float32, mx.float32],
            grid=grid,
            threadgroup=lowering.threadgroup,
            stream=mx.gpu,
        )
    except Exception:
        return None

    lse, out = outputs
    return cast(mx.array, out.astype(out_dtype)), cast(mx.array, lse)


def sparse_mla_bwd_path_c(
    q: mx.array,
    kv: mx.array,
    d_out: mx.array,
    indices: mx.array,
    *,
    sm_scale: float | None = None,
    d_v: int | None = None,
) -> tuple[mx.array, mx.array] | None:
    """TileLang DSL Path C Sparse-MLA backward.

    Returns ``(dq, dkv)`` or ``None`` if the Metal/TileLang path cannot be
    built. The kernel emits fp32 partials; the host reduction returns fp32 dKV.
    """

    status = sparse_mla_path_c_status()
    if not status.available:
        return None

    shapes = _resolve_shapes(q, kv, indices, d_v=d_v)
    if sm_scale is None:
        sm_scale_value = shapes.qk_dim ** -0.5
    else:
        sm_scale_value = sm_scale
    threads = _threadgroup_size(shapes.topk)

    q32 = q.astype(mx.float32)
    kv32 = kv.astype(mx.float32)
    d_out32 = d_out.astype(mx.float32)
    indices_i32 = indices.astype(mx.int32)
    sm_scale_buf = mx.array([float(sm_scale_value)], dtype=mx.float32)

    try:
        kernel, lowering = _bwd_kernel_for(
            shapes.batch,
            shapes.seq_len,
            shapes.heads,
            shapes.qk_dim,
            shapes.kv_group,
            shapes.head_kv,
            shapes.topk,
            shapes.seq_len_kv,
            shapes.d_v,
            threads,
        )
    except (MSLDispatchUnsupported, RuntimeError, ValueError):
        return None

    grid = (
        lowering.grid[0] * lowering.threadgroup[0],
        lowering.grid[1] * lowering.threadgroup[1],
        lowering.grid[2] * lowering.threadgroup[2],
    )

    try:
        outputs = kernel(
            inputs=[d_out32, indices_i32, kv32, q32, sm_scale_buf],
            output_shapes=[
                (
                    shapes.batch,
                    shapes.seq_len,
                    shapes.heads,
                    shapes.topk,
                    shapes.qk_dim,
                ),
                (shapes.batch, shapes.seq_len, shapes.heads, shapes.qk_dim),
            ],
            output_dtypes=[mx.float32, mx.float32],
            grid=grid,
            threadgroup=lowering.threadgroup,
            stream=mx.gpu,
        )
    except Exception:
        return None

    dkv_partial, dq = outputs
    dkv = _reduce_dkv_partial(dkv_partial, indices_i32, shapes)
    return cast(mx.array, dq), cast(mx.array, dkv)


def dump_lowered_fwd_msl(
    *,
    batch: int,
    seq_len: int,
    heads: int,
    qk_dim: int,
    kv_group: int,
    topk: int,
    seq_len_kv: int,
    d_v: int | None = None,
) -> str:
    """Return raw lowered forward MSL for inspection/benchmark artifacts."""

    if d_v is None:
        d_v = qk_dim
    head_kv = heads // kv_group
    threads = _threadgroup_size(topk)
    _kernel, lowering = _fwd_kernel_for(
        batch,
        seq_len,
        heads,
        qk_dim,
        kv_group,
        head_kv,
        topk,
        seq_len_kv,
        d_v,
        threads,
    )
    return cast(str, lowering.msl_text)


def dump_lowered_bwd_msl(
    *,
    batch: int,
    seq_len: int,
    heads: int,
    qk_dim: int,
    kv_group: int,
    topk: int,
    seq_len_kv: int,
    d_v: int | None = None,
) -> str:
    """Return raw lowered MSL for inspection/benchmark artifacts."""

    if d_v is None:
        d_v = qk_dim
    head_kv = heads // kv_group
    threads = _threadgroup_size(topk)
    _kernel, lowering = _bwd_kernel_for(
        batch,
        seq_len,
        heads,
        qk_dim,
        kv_group,
        head_kv,
        topk,
        seq_len_kv,
        d_v,
        threads,
    )
    return cast(str, lowering.msl_text)


__all__ = [
    "SparseMLAPathCStatus",
    "dump_lowered_bwd_msl",
    "dump_lowered_fwd_msl",
    "sparse_mla_bwd_path_c",
    "sparse_mla_fwd_path_c",
    "sparse_mla_path_c_status",
]
