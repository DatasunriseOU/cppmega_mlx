"""Path C Sparse-MLA backward via TileLang DSL ``@T.prim_func`` lowering.

This module is the first TileLang-DSL counterpart to the Path B direct-MSL
Sparse-MLA backward kernel in :mod:`cppmega_mlx.nn._tilelang.sparse_mla`.

The upstream TileLang Sparse-MLA backward examples are CUDA-oriented: they use
``T.gemm`` for the attention matmuls and atomics for dKV scatter. Both are
still the wrong first step on Apple Metal. This Path C kernel instead mirrors
Path B's partial-output contract:

* compute ``dq`` directly;
* emit ``dkv_partial[B, S, H, topk, D]`` without atomics;
* reuse Path B's host-side ``_reduce_dkv_partial`` scatter/reduction.

It is intentionally scalar and conservative. The purpose of this lane is to
prove the chunked backward math and MLX dispatch path through TileLang's Metal
lowering before introducing shared-memory tiling or simdgroup reductions.
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
        reason="Sparse-MLA Path C backward TileLang DSL ready",
    )


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
) -> tuple[Any, _msl_transform.TileLangMSLLowering]:
    """Build and cache a shape-specialized scalar Sparse-MLA bwd kernel."""

    import tilelang.language as T

    LANES = BATCH * SEQ_LEN * HEADS

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
        with T.Kernel(LANES, threads=1) as bx:
            scores = T.alloc_local((TOPK,), "float32")
            probs = T.alloc_local((TOPK,), "float32")
            dp = T.alloc_local((TOPK,), "float32")
            ds = T.alloc_local((TOPK,), "float32")
            max_score = T.alloc_local((1,), "float32")
            sumexp = T.alloc_local((1,), "float32")
            rowsum = T.alloc_local((1,), "float32")
            inv_sum = T.alloc_local((1,), "float32")

            lane = bx
            h = lane % HEADS
            s = (lane // HEADS) % SEQ_LEN
            b = lane // (HEADS * SEQ_LEN)
            g = h // HEAD_KV
            sm_scale = sm_scale_buf[0]

            max_score[0] = -3.4028234663852886e38
            for k in T.serial(TOPK):
                gather_idx = indices[b, s, g, k]
                if gather_idx < 0:
                    scores[k] = -3.4028234663852886e38
                else:
                    acc = T.alloc_local((1,), "float32")
                    acc[0] = 0.0
                    for d in T.serial(QK_DIM):
                        acc[0] = acc[0] + q[b, s, h, d] * kv[b, gather_idx, g, d]
                    scores[k] = acc[0] * sm_scale
                    if scores[k] > max_score[0]:
                        max_score[0] = scores[k]

            sumexp[0] = 0.0
            for k in T.serial(TOPK):
                gather_idx = indices[b, s, g, k]
                if gather_idx < 0:
                    probs[k] = 0.0
                else:
                    probs[k] = T.exp(scores[k] - max_score[0])
                    sumexp[0] = sumexp[0] + probs[k]

            inv_sum[0] = 0.0
            if sumexp[0] > 0.0:
                inv_sum[0] = 1.0 / sumexp[0]

            for k in T.serial(TOPK):
                probs[k] = probs[k] * inv_sum[0]

            for k in T.serial(TOPK):
                gather_idx = indices[b, s, g, k]
                if gather_idx < 0:
                    dp[k] = 0.0
                else:
                    acc = T.alloc_local((1,), "float32")
                    acc[0] = 0.0
                    for d in T.serial(D_V):
                        acc[0] = acc[0] + kv[b, gather_idx, g, d] * d_out[b, s, h, d]
                    dp[k] = acc[0]

            rowsum[0] = 0.0
            for k in T.serial(TOPK):
                rowsum[0] = rowsum[0] + probs[k] * dp[k]

            for k in T.serial(TOPK):
                ds[k] = probs[k] * (dp[k] - rowsum[0])

            for d in T.serial(QK_DIM):
                acc = T.alloc_local((1,), "float32")
                acc[0] = 0.0
                for k in T.serial(TOPK):
                    gather_idx = indices[b, s, g, k]
                    if gather_idx >= 0:
                        acc[0] = acc[0] + ds[k] * kv[b, gather_idx, g, d]
                dq[b, s, h, d] = acc[0] * sm_scale

            for k in T.serial(TOPK):
                gather_idx = indices[b, s, g, k]
                for d in T.serial(QK_DIM):
                    if gather_idx < 0:
                        dkv_partial[b, s, h, k, d] = 0.0
                    else:
                        k_grad = sm_scale * ds[k] * q[b, s, h, d]
                        if d < D_V:
                            dkv_partial[b, s, h, k, d] = (
                                probs[k] * d_out[b, s, h, d] + k_grad
                            )
                        else:
                            dkv_partial[b, s, h, k, d] = k_grad

    lowering = lower_tilelang_to_msl_inline(sparse_mla_bwd)
    kernel = mx.fast.metal_kernel(
        name=(
            "cppmega_sparse_mla_path_c_bwd_"
            f"{BATCH}_{SEQ_LEN}_{HEADS}_{QK_DIM}_{KV_GROUP}_{TOPK}_{SEQ_LEN_KV}_{D_V}"
        ),
        input_names=["d_out", "indices", "kv", "q", "sm_scale_buf"],
        output_names=["dkv_partial", "dq"],
        source=lowering.body,
        header=lowering.header,
        ensure_row_contiguous=True,
    )
    return kernel, lowering


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
        sm_scale = shapes.qk_dim ** -0.5

    q32 = q.astype(mx.float32)
    kv32 = kv.astype(mx.float32)
    d_out32 = d_out.astype(mx.float32)
    indices_i32 = indices.astype(mx.int32)
    sm_scale_buf = mx.array([float(sm_scale)], dtype=mx.float32)

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
    )
    return cast(str, lowering.msl_text)


__all__ = [
    "SparseMLAPathCStatus",
    "dump_lowered_bwd_msl",
    "sparse_mla_bwd_path_c",
    "sparse_mla_path_c_status",
]
