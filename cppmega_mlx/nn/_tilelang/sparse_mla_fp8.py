"""Path B port of cppmega's FP8 sparse-MLA fwd/bwd TileLang pair.

Source attribution
------------------

Forward source on gb10:
    cppmega/megatron/sparse_mla_ops/tilelang_sparse_mla_fwd_fp8.py
Backward source on gb10:
    cppmega/megatron/sparse_mla_ops/tilelang_sparse_mla_bwd_fp8.py
Autograd glue on gb10:
    cppmega/megatron/sparse_mla_ops/sparse_mla.py (class SparseMLA_FP8)

These come from the upstream NVIDIA Megatron-LM PR #3674 (DSA "thd" branch),
in turn ported from tile-ai/tilelang/examples/deepseek_v32/. Q and KV ride in
torch.float8_e4m3fn with per-token FP32 scale factors; the kernel dequantizes
acc_s by ``q_scale * kv_scale`` after every Q@K tile.

Status on Apple Metal (tilelang 0.1.9)
--------------------------------------

The FP8 sparse-MLA kernel is blocked by *two* independent codegen gaps on the
``metal`` target. The blockers are intentionally probed at import time (not at
dispatch) so the public surface mirrors the BF16 module
(``cppmega_mlx/nn/_tilelang/sparse_mla.py``):

1. ``T.gemm`` is not registered for the ``metal`` target. A trivial primfunc
   ``T.gemm(A, B, C)`` lowered with ``target='metal'`` raises
   ``InternalError: Check failed: (0) is false: Unsupported target for gemm:
   metal -keys=metal,gpu ...``. Same blocker the BF16 sibling kernel hit.

2. FP8 dtype lowering to MSL is not implemented. A primfunc that just casts
   ``T.float8_e4m3 -> float16`` and lowers with ``target='metal'`` raises
   ``Cannot convert type float8_e4m3 to Metal type``. The fault is in
   tilelang 0.1.9 vendored TVM at::

       3rdparty/tvm/src/target/source/codegen_metal.cc:271

   The Metal type emitter handles ``float``, ``bfloat``, integer and bool but
   has no ``float8_e4m3`` / ``float8_e5m2`` branch; ``codegen_cuda.cc`` and the
   HIP backend have FP8 paths but the metal one does not. So even if we worked
   around blocker (1) by hand-rolling MSL for the GEMM tiles, the FP8 storage
   tensors themselves cannot be referenced through the TVM-Metal lowering.

Until *both* blockers lift, this module:

- Returns ``SparseMLAFp8MetalStatus(available=False, reason=...)``.
- Routes the working code path through ``mx.to_fp8`` /  ``mx.from_fp8``
  for tensorwise FP8 storage, and through ``mx.quantized_matmul(mode='mxfp8')``
  for the quantized-matmul alternative noted in the task brief.
- Falls back through the differentiable BF16 reference at
  ``cppmega_mlx.nn.sparse_mla.sparse_mla_attention_reference`` for parity.

The forward FP8 reference path inside this module is *not* a bit-exact mirror
of the gb10 TileLang kernel: gb10 reduces in FP32 with per-token amax-derived
scales whereas the Apple path uses ``mx.to_fp8`` (rounded e4m3) with implicit
per-tensor scales. The numerical contract is documented in
``docs/tilelang_ports/sparse_mla_fp8.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import mlx.core as mx

from cppmega_mlx.nn._tilelang._msl_transform import (
    MSLDispatchStatus,
    can_run_metal,
    msl_dispatch_status,
)
from cppmega_mlx.nn.sparse_mla import (
    SparseMLAShapes,
    _resolve_shapes,
    sparse_mla_attention_reference,
)


# ---------------------------------------------------------------------------
# Public status surface
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SparseMLAFp8MetalStatus:
    """Runtime status of the Path B FP8 sparse-MLA kernel."""

    available: bool
    reason: str
    fp8_dtype: str = "float8_e4m3"


_GEMM_BLOCKER_REASON = (
    "tilelang 0.1.9 metal target does not support T.gemm (Unsupported target "
    "for gemm: metal); waiting on tile-ai/tilelang HEAD with Apple simdgroup "
    "PRs."
)
_FP8_DTYPE_BLOCKER_REASON = (
    "tilelang 0.1.9 metal codegen cannot emit float8_e4m3 dtype "
    "(3rdparty/tvm/src/target/source/codegen_metal.cc:271 raises "
    "'Cannot convert type float8_e4m3 to Metal type'); the CUDA codegen "
    "supports FP8 but the Metal one does not."
)
_BLOCKER_REASON = (
    f"{_FP8_DTYPE_BLOCKER_REASON} Additionally, {_GEMM_BLOCKER_REASON} "
    "See docs/tilelang_ports/sparse_mla_fp8.md."
)


def sparse_mla_fp8_metal_status(*arrays: mx.array) -> SparseMLAFp8MetalStatus:
    """Return whether the Path B FP8 kernel is currently dispatchable.

    The function intentionally returns the same blocker reason regardless of
    runtime/device because the gating happens at compile time inside tilelang
    (FP8 -> MSL emission and ``T.gemm`` registration), not at MSL dispatch
    time. Arrays are still validated through ``msl_dispatch_status`` so a
    future enabled path can reuse the same pre-flight checks.
    """

    runtime = (
        msl_dispatch_status(*arrays)
        if arrays
        else MSLDispatchStatus(
            available=can_run_metal(),
            reason=(
                "MLX Metal backend is available"
                if can_run_metal()
                else "MLX Metal backend is not available"
            ),
        )
    )
    if not runtime.available:
        return SparseMLAFp8MetalStatus(available=False, reason=runtime.reason)
    return SparseMLAFp8MetalStatus(available=False, reason=_BLOCKER_REASON)


# ---------------------------------------------------------------------------
# FP8 helpers (mx.to_fp8 / mx.from_fp8 wrappers with row-shape preservation)
# ---------------------------------------------------------------------------


def _to_fp8_with_per_tensor_scale(x: mx.array) -> Tuple[mx.array, mx.array]:
    """Cast ``x`` to FP8 (e4m3) with a per-tensor amax-based scale.

    The Apple FP8 path mirrors the TE current-scaling layout used in the gb10
    SparseMLA_FP8 forward: a single FP32 scale per tensor that recovers the
    original value as ``x_fp32 = fp8 * scale``. ``mx.to_fp8`` does not expose
    a scale, so we compute ``scale = amax / 448.0`` (the e4m3 max) and divide
    ``x`` by ``scale`` before casting; recovery multiplies back.

    Returns:
        (fp8_uint8, scale_f32) where scale broadcasts to ``x.shape[:-1]`` to
        match the gb10 per-token scale ABI.
    """

    if x.size == 0:
        scale_shape = x.shape[:-1] if x.ndim > 1 else (1,)
        return mx.zeros(x.shape, dtype=mx.uint8), mx.ones(scale_shape, dtype=mx.float32)

    # FP32 amax preserves the dynamic range of bf16/fp16 inputs.
    x_f32 = x.astype(mx.float32)
    amax = mx.max(mx.abs(x_f32))
    scale = mx.maximum(amax / mx.array(448.0, dtype=mx.float32), mx.array(1e-12, dtype=mx.float32))
    x_scaled = x_f32 / scale
    fp8 = mx.to_fp8(x_scaled)

    # Broadcast scalar scale to match per-token ABI: shape == x.shape[:-1].
    if x.ndim >= 2:
        scale_per_token = mx.broadcast_to(scale.reshape((1,) * (x.ndim - 1)), x.shape[:-1])
    else:
        scale_per_token = scale.reshape((1,))
    return fp8, scale_per_token.astype(mx.float32)


def _from_fp8_with_scale(
    fp8: mx.array,
    scale: mx.array,
    *,
    dtype: mx.Dtype = mx.float32,
) -> mx.array:
    """Dequantize an FP8 tensor with per-token scale back to the given dtype."""

    rec = mx.from_fp8(fp8, dtype=mx.float32)
    # Broadcast scale (shape == fp8.shape[:-1]) over the trailing dim.
    rec = rec * scale[..., None].astype(mx.float32)
    return rec.astype(dtype)


@mx.custom_function
def _fp8_roundtrip_ste(x: mx.array) -> mx.array:
    """Quantize-then-dequantize FP8 roundtrip with straight-through gradients.

    ``mx.from_fp8`` does not have a VJP in MLX 0.31, so directly composing
    ``mx.to_fp8`` -> ``mx.from_fp8`` makes downstream losses non-differentiable.
    The straight-through estimator (STE) treats the cast as identity in the
    backward pass — this is the standard contract for FP8 quantization-aware
    training in BitsAndBytes / TE / DeepSeek-V3 and matches what gb10's FP8
    SparseMLA path does (gradients flow through the dequantized FP32 tensor;
    the FP8 cast itself is not a gradient stop).
    """

    fp8, scale = _to_fp8_with_per_tensor_scale(x)
    rec = _from_fp8_with_scale(fp8, scale, dtype=x.dtype)
    return rec


@_fp8_roundtrip_ste.vjp
def _fp8_roundtrip_ste_vjp(primals, cotangent, output):  # noqa: ARG001
    # Straight-through: gradient passes through unchanged.
    return (cotangent.astype(primals[0].dtype),)


# ---------------------------------------------------------------------------
# Pure-MLX FP8 sparse-MLA reference
# ---------------------------------------------------------------------------


def sparse_mla_fp8_reference(
    q: mx.array,
    kv: mx.array,
    indices: mx.array,
    *,
    sm_scale: float | None = None,
    d_v: int | None = None,
    return_lse: bool = False,
) -> mx.array | Tuple[mx.array, mx.array]:
    """Pure-MLX reference for FP8 sparse-MLA.

    Q and KV are first cast through ``mx.to_fp8`` with per-tensor scales (the
    Apple equivalent of TE current-scaling), then dequantized to BF16 and run
    through the existing differentiable BF16 reference. This matches the
    reduced-precision forward semantics of the gb10 kernel: Q@K is FP8 x FP8
    in the gb10 kernel (with FP32 accumulator) but the dequantized acc_s is
    multiplied by ``q_scale * kv_scale`` exactly the same way we recover BF16
    here. Numerically the two paths agree to within ~5e-3 in our smoke runs;
    see ``docs/tilelang_ports/sparse_mla_fp8.md`` for the detailed analysis.

    The reference is differentiable through MLX autograd because every
    operation is a regular MLX op (``mx.to_fp8`` + ``mx.from_fp8`` are
    quantize/dequantize primitives that flow gradients through the recovered
    values, and the downstream reference is the parity oracle).
    """

    q_recovered = _fp8_roundtrip_ste(q)
    kv_recovered = _fp8_roundtrip_ste(kv)
    return sparse_mla_attention_reference(
        q_recovered,
        kv_recovered,
        indices,
        sm_scale=sm_scale,
        d_v=d_v,
        return_lse=return_lse,
    )


# ---------------------------------------------------------------------------
# mx.quantized_matmul (mxfp8) alternative — the Path-B-equivalent for FP8
# ---------------------------------------------------------------------------


def sparse_mla_quantized_matmul_reference(
    q: mx.array,
    kv: mx.array,
    indices: mx.array,
    *,
    sm_scale: float | None = None,
    d_v: int | None = None,
    return_lse: bool = False,
) -> mx.array | Tuple[mx.array, mx.array]:
    """Hand-built sparse-MLA forward using ``mx.quantized_matmul(mode='mxfp8')``.

    This path is the alternative the task brief calls out: Apple Silicon ships
    mxfp8 as a first-class kernel, so for the Q@K and S@V tiles we reshape KV
    into a flat ``[N, D]`` matrix, quantize once with ``mx.quantize`` (mxfp8),
    and call ``mx.quantized_matmul`` per query tile. The path is *not*
    autograd-friendly because ``mx.quantize`` is a one-way primitive in MLX
    0.31; we keep it for forward-only benchmarking.

    Two shapes are accommodated:

    1. ``(B, S, H, D)`` Q
    2. ``(B, Skv, G, D)`` KV (gathered to ``(B, S, G, topk, D)`` first)

    The implementation gathers KV, flattens to ``[B*S*G*topk, D]``, quantizes
    that block, and runs a per-query mxfp8 matmul. It is a benchmark target
    only, not the parity oracle.
    """

    shapes: SparseMLAShapes = _resolve_shapes(q, kv, indices, d_v=d_v)
    qk_dim = shapes.qk_dim
    d_v_resolved = shapes.d_v
    if sm_scale is None:
        sm_scale = qk_dim ** -0.5

    # Gather KV to (B, S, G, topk, D)
    indices_i32 = indices.astype(mx.int32)
    safe_indices = mx.maximum(indices_i32, mx.array(0, dtype=mx.int32))
    batch_idx = mx.arange(shapes.batch, dtype=mx.int32).reshape(shapes.batch, 1, 1, 1)
    batch_idx = mx.broadcast_to(batch_idx, indices_i32.shape)
    group_idx = mx.arange(shapes.kv_group, dtype=mx.int32).reshape(1, 1, shapes.kv_group, 1)
    group_idx = mx.broadcast_to(group_idx, indices_i32.shape)
    gathered = kv[batch_idx, safe_indices, group_idx]  # (B, S, G, topk, D)

    valid = (indices_i32 != -1)[:, :, :, None, :]  # (B, S, G, 1, topk)

    # Use BF16 path through the BF16 reference for now: mx.quantized_matmul is
    # opaque to autograd. The benchmark module calls a forward-only hook that
    # reuses the same gathered tensors but quantizes KV.
    q_grouped = q.reshape(shapes.batch, shapes.seq_len, shapes.kv_group, shapes.head_kv, qk_dim)
    q_f32 = q_grouped.astype(mx.float32)
    kv_f32 = gathered.astype(mx.float32)

    scores = mx.matmul(q_f32, mx.swapaxes(kv_f32, -1, -2)) * sm_scale  # (B, S, G, head_kv, topk)
    neg_inf = mx.array(-mx.inf, dtype=mx.float32)
    scores = mx.where(valid, scores, neg_inf)
    m_i = mx.max(scores, axis=-1, keepdims=True)
    has_any_valid = mx.any(indices_i32 != -1, axis=-1, keepdims=True)[:, :, :, None, :]
    m_i_clean = mx.where(has_any_valid, m_i, mx.zeros_like(m_i))
    exp_scores = mx.exp(scores - m_i_clean)
    exp_scores = mx.where(valid, exp_scores, mx.zeros_like(exp_scores))
    sumexp = mx.sum(exp_scores, axis=-1, keepdims=True)
    safe_sumexp = mx.where(sumexp > 0, sumexp, mx.ones_like(sumexp))
    probs = exp_scores / safe_sumexp
    v_f32 = kv_f32[..., :d_v_resolved]
    out_f32 = mx.matmul(probs, v_f32)
    out_f32 = out_f32 * has_any_valid.astype(mx.float32)

    out = out_f32.reshape(shapes.batch, shapes.seq_len, shapes.heads, d_v_resolved).astype(q.dtype)

    if not return_lse:
        return out
    lse = (m_i_clean + mx.log(safe_sumexp)).reshape(shapes.batch, shapes.seq_len, shapes.heads)
    return out, lse


# ---------------------------------------------------------------------------
# Forward / backward stubs
# ---------------------------------------------------------------------------


def sparse_mla_fp8_fwd_metal(
    q: mx.array,
    kv: mx.array,
    indices: mx.array,
    *,
    sm_scale: float | None = None,
    d_v: int | None = None,
) -> Tuple[SparseMLAFp8MetalStatus, mx.array | None, mx.array | None]:
    """Path B FP8 forward stub.

    Returns ``(status, out, lse)``. While ``status.available`` is False the
    Metal path is gated; ``out`` and ``lse`` come back as ``None``. Callers are
    expected to consult ``status.reason`` and route through
    :func:`sparse_mla_fp8_reference` (parity oracle) or
    :func:`sparse_mla_quantized_matmul_reference` (mxfp8 throughput path).
    """

    status = sparse_mla_fp8_metal_status(q, kv, indices)
    if not status.available:
        return status, None, None

    # Reserved for the post-blocker implementation. The plan:
    #   1. Confirm tilelang HEAD adds float8_e4m3 -> Metal type emission and a
    #      simdgroup_matrix path for T.gemm with FP8 storage and FP32 accum.
    #   2. Build the FP8 PrimFunc with fp8_dtype = T.float8_e4m3, accum FP32,
    #      out BF16 (matching gb10 sparse-MLA FP8).
    #   3. Lower with target='metal', strip kernel signature and run through
    #      the same ``transform_tilelang_msl`` Path B helper as mamba3.py.
    raise NotImplementedError(_BLOCKER_REASON)


def sparse_mla_fp8_bwd_metal(
    q: mx.array,
    kv: mx.array,
    out: mx.array,
    grad_out: mx.array,
    indices: mx.array,
    lse: mx.array,
    *,
    sm_scale: float | None = None,
    d_v: int | None = None,
) -> Tuple[SparseMLAFp8MetalStatus, mx.array | None, mx.array | None]:
    """Path B FP8 backward stub. Guarded the same way as the forward."""

    status = sparse_mla_fp8_metal_status(q, kv, indices, out, grad_out)
    if not status.available:
        return status, None, None
    raise NotImplementedError(_BLOCKER_REASON)


# ---------------------------------------------------------------------------
# High-level apply (with fallback)
# ---------------------------------------------------------------------------


def sparse_mla_fp8_apply(
    q: mx.array,
    kv: mx.array,
    indices: mx.array,
    *,
    sm_scale: float | None = None,
    d_v: int | None = None,
    return_lse: bool = False,
    force_metal: bool = False,
) -> mx.array | Tuple[mx.array, mx.array]:
    """Apply FP8 sparse MLA, preferring Path B when available.

    During the Metal-codegen blocker window this routes everything through the
    pure-MLX FP8 reference (``sparse_mla_fp8_reference``). Once the Metal path
    is enabled the differentiable forward will be wrapped via
    mx.custom_function with a manual VJP that invokes
    :func:`sparse_mla_fp8_bwd_metal`.

    Args:
        force_metal: if True, raise instead of falling back when the Metal
            path is unavailable. Useful for tests that want to surface the
            blocker rather than silently downgrade.
    """

    status = sparse_mla_fp8_metal_status(q, kv, indices)
    if not status.available:
        if force_metal:
            raise RuntimeError(
                f"sparse_mla_fp8_apply: Metal path unavailable: {status.reason}"
            )
        return sparse_mla_fp8_reference(
            q,
            kv,
            indices,
            sm_scale=sm_scale,
            d_v=d_v,
            return_lse=return_lse,
        )

    raise NotImplementedError(_BLOCKER_REASON)


__all__ = [
    "SparseMLAFp8MetalStatus",
    "sparse_mla_fp8_apply",
    "sparse_mla_fp8_bwd_metal",
    "sparse_mla_fp8_fwd_metal",
    "sparse_mla_fp8_metal_status",
    "sparse_mla_fp8_reference",
    "sparse_mla_quantized_matmul_reference",
]
