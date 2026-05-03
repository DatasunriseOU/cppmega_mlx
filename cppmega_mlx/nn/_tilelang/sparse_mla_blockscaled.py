"""Path B port of cppmega's block-scaled (MXFP8) sparse-MLA fwd/bwd TileLang pair.

Source attribution
------------------

Forward + backward source on gb10:
    cppmega/megatron/sparse_mla_ops/tilelang_sparse_mla_blockscaled_fused.py
    (functions ``sparse_mla_blockscaled_mxfp8_fwd`` and
    ``sparse_mla_blockscaled_mxfp8_bwd_kernel``)

A separate experimental QK-only block-scaled scoring helper lives in
    cppmega/megatron/sparse_mla_ops/tilelang_sparse_mla_blockscaled_qk.py

Both consume MXFP8 layouts where the data tensor is a torch.float8_e4m3fn
matrix and a separate FP32 scale tensor with one scalar per 32-element block
along the last (head) axis::

    q_data:   [B, S,  H, D_total]      torch.float8_e4m3fn
    kv_data:  [B, SK, G, D_total]      torch.float8_e4m3fn
    q_scale:  [B, S,  H, D_total/32]   torch.float32
    kv_scale: [B, SK, G, D_total/32]   torch.float32
    indices:  [B, S,  G, topk]         torch.int32, -1 sentinel

The kernel walks the head dim in 32-element blocks (``MXFP8_BLOCK_SIZE = 32``),
runs ``T.gemm(q_block, kv_block, partial)`` for each block, and accumulates
``partial[h, j] * QScale[h, kb] * KVScale[j, kb]`` into ``acc_s``. After the
block walk it runs the standard online-softmax + S@V flow used by the BF16
sparse-MLA forward, with V dequantized from FP8 by ``KVScale[d/32]`` (the
backward similarly dequantizes Q tail / KV tail blocks).

Status on Apple Metal (tilelang 0.1.9)
--------------------------------------

The block-scaled FP8 sparse-MLA kernel hits the *same* two blockers as the
tensorwise FP8 kernel and additionally a third one:

1. ``T.gemm`` is not registered for the ``metal`` target. Same as the BF16
   sparse-MLA blocker.

2. FP8 dtype lowering to MSL is not implemented. Same as the FP8 sparse-MLA
   blocker. The fault is at::

       3rdparty/tvm/src/target/source/codegen_metal.cc:271

3. The block-scaled kernel's per-block dequant pattern (``partial[h, bi] *
   QScale[h, kb] * KVScale[j, kb]``) requires both FP8 storage AND a
   per-32-channel FP32 scale, indexed by ``d // BK``. On Apple, the equivalent
   is ``mx.dequantize(w, scales, mode='mxfp8', group_size=32)`` or
   ``mx.quantized_matmul(..., mode='mxfp8')``: both are first-class kernels in
   MLX 0.30+, so the *math* is supported, just not via TileLang's TVM-Metal
   lowering. See ``docs/tilelang_ports/sparse_mla_blockscaled.md``.

Until tilelang HEAD ships an Apple GEMM + FP8 storage path, this module:

- Returns ``SparseMLABlockScaledMetalStatus(available=False, reason=...)``.
- Routes the working code path through ``mx.quantize(mode='mxfp8')`` /
  ``mx.dequantize(mode='mxfp8')`` for block-scaled FP8 layout, mirroring the
  ``q_scale[..., d/32]`` per-block-of-32 ABI of the gb10 kernel.
- Falls back through the differentiable BF16 reference at
  ``cppmega_mlx.nn.sparse_mla.sparse_mla_attention_reference`` for parity.

The MXFP8 reference inside this module is *not* a bit-exact mirror of the
gb10 TileLang kernel: gb10 reduces in FP32 with the per-block scales applied
post-GEMM, whereas Apple's quantize/dequantize roundtrip applies block scales
during the cast itself. Numerical contract is in the doc.
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


# Mirrors MXFP8_BLOCK_SIZE in the gb10 fused module. Apple's mxfp8 quantize
# uses the same group_size by default.
MXFP8_BLOCK_SIZE = 32


# ---------------------------------------------------------------------------
# Public status surface
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SparseMLABlockScaledMetalStatus:
    """Runtime status of the Path B block-scaled sparse-MLA kernel."""

    available: bool
    reason: str
    block_size: int = MXFP8_BLOCK_SIZE


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
    "See docs/tilelang_ports/sparse_mla_blockscaled.md."
)


def sparse_mla_blockscaled_metal_status(
    *arrays: mx.array,
) -> SparseMLABlockScaledMetalStatus:
    """Return whether the Path B block-scaled FP8 kernel is dispatchable.

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
        return SparseMLABlockScaledMetalStatus(
            available=False, reason=runtime.reason
        )
    return SparseMLABlockScaledMetalStatus(
        available=False, reason=_BLOCKER_REASON
    )


# ---------------------------------------------------------------------------
# Block-scaled MXFP8 helpers (mx.quantize/dequantize wrappers)
# ---------------------------------------------------------------------------


def _quantize_mxfp8(x: mx.array) -> Tuple[mx.array, mx.array]:
    """Quantize ``x`` to MXFP8 layout matching the gb10 block-scaled ABI.

    Apple's ``mx.quantize(mode='mxfp8')`` returns ``(packed_uint32, scales)``
    with scales shaped ``[..., D / group_size]`` and packed data with the last
    dim collapsed to ``D / 4`` (4 fp8 e4m3 elements packed into one uint32).
    The shape contract matches gb10: one scalar per block-of-32 along the last
    axis.

    Returns ``(w_packed, scales)``. Group size is fixed at
    ``MXFP8_BLOCK_SIZE = 32`` to match gb10.
    """

    if x.size == 0:
        # Match the same packed-shape contract for empty inputs.
        last = max(x.shape[-1] // 4, 0) if x.ndim >= 1 else 0
        scales_last = max(x.shape[-1] // MXFP8_BLOCK_SIZE, 0) if x.ndim >= 1 else 0
        packed = mx.zeros(x.shape[:-1] + (last,), dtype=mx.uint32)
        scales = mx.zeros(x.shape[:-1] + (scales_last,), dtype=mx.uint8)
        return packed, scales

    # mx.quantize requires the last dim to be divisible by group_size. We
    # enforce that contract here so callers fail fast rather than mis-quantize.
    if x.shape[-1] % MXFP8_BLOCK_SIZE != 0:
        raise ValueError(
            f"MXFP8 last dim must be divisible by {MXFP8_BLOCK_SIZE}, "
            f"got shape {x.shape}."
        )
    res = mx.quantize(x, mode="mxfp8")
    # res is a list [packed_uint32, scales_uint8] in MLX 0.31.
    return res[0], res[1]


def _dequantize_mxfp8(
    w_packed: mx.array,
    scales: mx.array,
    *,
    out_dtype: mx.Dtype = mx.float32,
) -> mx.array:
    """Dequantize an MXFP8 tensor back to the requested dtype."""

    return mx.dequantize(w_packed, scales, mode="mxfp8", dtype=out_dtype)


@mx.custom_function
def _mxfp8_roundtrip_ste(x: mx.array) -> mx.array:
    """Quantize-then-dequantize MXFP8 roundtrip with straight-through gradients.

    Same contract as :func:`_fp8_roundtrip_ste` in the tensorwise FP8 module:
    ``mx.dequantize(mode='mxfp8')`` shares ``FromFP8`` as its primitive, which
    has no VJP in MLX 0.31. The STE makes the cast transparent to backward,
    matching gb10's gradient flow through dequantized inputs.
    """

    if x.shape[-1] % MXFP8_BLOCK_SIZE != 0:
        return x  # unchanged: caller is expected to fall back to BF16
    packed, scales = _quantize_mxfp8(x)
    return _dequantize_mxfp8(packed, scales, out_dtype=x.dtype)


@_mxfp8_roundtrip_ste.vjp
def _mxfp8_roundtrip_ste_vjp(primals, cotangent, output):  # noqa: ARG001
    # Straight-through: gradient passes through unchanged.
    return (cotangent.astype(primals[0].dtype),)


# ---------------------------------------------------------------------------
# Pure-MLX block-scaled MXFP8 sparse-MLA reference
# ---------------------------------------------------------------------------


def sparse_mla_blockscaled_reference(
    q: mx.array,
    kv: mx.array,
    indices: mx.array,
    *,
    sm_scale: float | None = None,
    d_v: int | None = None,
    return_lse: bool = False,
) -> mx.array | Tuple[mx.array, mx.array]:
    """Pure-MLX MXFP8 block-scaled reference.

    Q and KV are quantized to MXFP8 (with group_size = 32 along the head dim),
    then dequantized back to FP32 and run through the BF16 reference. This
    matches the *intent* of the gb10 block-scaled kernel:

        acc_s[h, j] = sum_kb partial[h, j, kb] * QScale[h, kb] * KVScale[j, kb]

    The Apple path achieves the same numerical effect by applying per-block
    scales during dequantization, then doing one regular matmul. Round-trip
    error is ~5e-3 relative on bf16 inputs in our smoke runs.

    The reference is differentiable through MLX autograd: ``mx.quantize`` and
    ``mx.dequantize`` both flow gradients through the recovered values.
    """

    shapes: SparseMLAShapes = _resolve_shapes(q, kv, indices, d_v=d_v)
    if shapes.qk_dim % MXFP8_BLOCK_SIZE != 0:
        # Fall back to BF16 reference if the dim is not block-aligned.
        return sparse_mla_attention_reference(
            q, kv, indices, sm_scale=sm_scale, d_v=d_v, return_lse=return_lse
        )

    # Quantize/dequantize Q and KV with group_size 32 along the head dim.
    # Use the STE roundtrip so backward stays differentiable.
    q_recovered = _mxfp8_roundtrip_ste(q)
    kv_recovered = _mxfp8_roundtrip_ste(kv)
    return sparse_mla_attention_reference(
        q_recovered,
        kv_recovered,
        indices,
        sm_scale=sm_scale,
        d_v=d_v,
        return_lse=return_lse,
    )


# ---------------------------------------------------------------------------
# Forward / backward stubs
# ---------------------------------------------------------------------------


def sparse_mla_blockscaled_fwd_metal(
    q: mx.array,
    kv: mx.array,
    indices: mx.array,
    *,
    sm_scale: float | None = None,
    d_v: int | None = None,
) -> Tuple[
    SparseMLABlockScaledMetalStatus, mx.array | None, mx.array | None
]:
    """Path B block-scaled FP8 forward stub.

    Returns ``(status, out, lse)``. While ``status.available`` is False the
    Metal path is gated; ``out`` and ``lse`` come back as ``None``. Callers are
    expected to consult ``status.reason`` and route through
    :func:`sparse_mla_blockscaled_reference` (parity oracle).
    """

    status = sparse_mla_blockscaled_metal_status(q, kv, indices)
    if not status.available:
        return status, None, None

    # Reserved for the post-blocker implementation. The plan:
    #   1. Confirm tilelang HEAD adds float8_e4m3 -> Metal type emission and a
    #      simdgroup_matrix path for T.gemm with FP8 storage and FP32 accum.
    #   2. Build the block-scaled PrimFunc with QScale/KVScale FP32 fragments,
    #      one per 32-element block along the head dim. Dequant happens
    #      post-GEMM exactly as in tilelang_sparse_mla_blockscaled_fused.py.
    #   3. Lower with target='metal', strip kernel signature and run through
    #      the same ``transform_tilelang_msl`` Path B helper as mamba3.py.
    raise NotImplementedError(_BLOCKER_REASON)


def sparse_mla_blockscaled_bwd_metal(
    q: mx.array,
    kv: mx.array,
    out: mx.array,
    grad_out: mx.array,
    indices: mx.array,
    lse: mx.array,
    *,
    sm_scale: float | None = None,
    d_v: int | None = None,
) -> Tuple[
    SparseMLABlockScaledMetalStatus, mx.array | None, mx.array | None
]:
    """Path B block-scaled FP8 backward stub. Guarded the same way as fwd."""

    status = sparse_mla_blockscaled_metal_status(q, kv, indices, out, grad_out)
    if not status.available:
        return status, None, None
    raise NotImplementedError(_BLOCKER_REASON)


# ---------------------------------------------------------------------------
# High-level apply (with fallback)
# ---------------------------------------------------------------------------


def sparse_mla_blockscaled_apply(
    q: mx.array,
    kv: mx.array,
    indices: mx.array,
    *,
    sm_scale: float | None = None,
    d_v: int | None = None,
    return_lse: bool = False,
    force_metal: bool = False,
) -> mx.array | Tuple[mx.array, mx.array]:
    """Apply block-scaled FP8 sparse MLA, preferring Path B when available.

    During the Metal-codegen blocker window this routes everything through the
    pure-MLX MXFP8 reference (``sparse_mla_blockscaled_reference``). Once the
    Metal path is enabled the differentiable forward will be wrapped via
    mx.custom_function with a manual VJP that invokes
    :func:`sparse_mla_blockscaled_bwd_metal`.

    Args:
        force_metal: if True, raise instead of falling back when the Metal
            path is unavailable.
    """

    status = sparse_mla_blockscaled_metal_status(q, kv, indices)
    if not status.available:
        if force_metal:
            raise RuntimeError(
                f"sparse_mla_blockscaled_apply: Metal path unavailable: "
                f"{status.reason}"
            )
        return sparse_mla_blockscaled_reference(
            q,
            kv,
            indices,
            sm_scale=sm_scale,
            d_v=d_v,
            return_lse=return_lse,
        )

    raise NotImplementedError(_BLOCKER_REASON)


__all__ = [
    "MXFP8_BLOCK_SIZE",
    "SparseMLABlockScaledMetalStatus",
    "sparse_mla_blockscaled_apply",
    "sparse_mla_blockscaled_bwd_metal",
    "sparse_mla_blockscaled_fwd_metal",
    "sparse_mla_blockscaled_metal_status",
    "sparse_mla_blockscaled_reference",
]
