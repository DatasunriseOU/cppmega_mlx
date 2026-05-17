"""Block-scaled sparse-MLA compatibility surface after direct-MSL retirement.

The historical Path B implementation in this module owned MXFP8 quantization,
unpacked the packed MLX MXFP8 bytes, and launched hand-written MSL through
``_msl_transform.dispatch``. P2 cleanup retires that raw direct-MSL runtime
surface. High-level float-carrier calls now use the pure-MLX MXFP8 reference,
while callers that already own prepared FP8 byte/scales buffers should use the
Path C prepared-buffer entry points in
:mod:`cppmega_mlx.nn._tilelang.sparse_mla_blockscaled_path_c`.

Public names remain for import compatibility. The retired ``*_fwd_metal`` and
``*_bwd_metal`` helpers still validate the shape contract and then fail closed
with ``None``. ``force_metal=True`` preserves its historical meaning, so it now
raises instead of silently selecting the reference or Path C.
"""

from __future__ import annotations

# pyright: reportFunctionMemberAccess=false

from dataclasses import dataclass
from typing import Tuple, cast

import mlx.core as mx

from cppmega_mlx.nn._tilelang.sparse_mla_blockscaled_path_c import (
    SparseMLABlockScaledQKReducePathCStatus,
    blockscaled_sparse_mla_qk_path_c_status,
    blockscaled_sparse_mla_qk_reduce_path_c,
    blockscaled_sparse_mla_qk_reduce_path_c_status,
)
from cppmega_mlx.nn.sparse_mla import (
    SparseMLAShapes,
    _resolve_shapes,
    sparse_mla_attention_reference,
)


MXFP8_BLOCK_SIZE = 32


@dataclass(frozen=True)
class SparseMLABlockScaledMetalStatus:
    """Runtime status of the retired Path B block-scaled sparse-MLA surface."""

    available: bool
    reason: str
    block_size: int = MXFP8_BLOCK_SIZE


_DIRECT_MSL_RETIRED_REASON = (
    "sparse_mla_blockscaled direct-MSL Path B is retired for production "
    "cleanup: raw _msl_transform.dispatch/mx.fast.metal_kernel callsites are "
    "no longer constructed. Use sparse_mla_blockscaled_path_c for prepared "
    "FP8 byte/scales owner-output routes, or the pure-MLX MXFP8 reference "
    "fallback for float-carrier inputs."
)


def _quantize_mxfp8(x: mx.array) -> Tuple[mx.array, mx.array]:
    """Quantize ``x`` to MXFP8 layout; returns ``(packed_uint32, scales_uint8)``."""

    if x.size == 0:
        last = max(x.shape[-1] // 4, 0) if x.ndim >= 1 else 0
        scales_last = max(x.shape[-1] // MXFP8_BLOCK_SIZE, 0) if x.ndim >= 1 else 0
        packed = mx.zeros(x.shape[:-1] + (last,), dtype=mx.uint32)
        scales = mx.zeros(x.shape[:-1] + (scales_last,), dtype=mx.uint8)
        return packed, scales

    if x.shape[-1] % MXFP8_BLOCK_SIZE != 0:
        raise ValueError(
            f"MXFP8 last dim must be divisible by {MXFP8_BLOCK_SIZE}, "
            f"got shape {x.shape}."
        )
    res = mx.quantize(x, mode="mxfp8")
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
    """Quantize-then-dequantize MXFP8 roundtrip with straight-through gradients."""

    if x.shape[-1] % MXFP8_BLOCK_SIZE != 0:
        return x
    packed, scales = _quantize_mxfp8(x)
    return _dequantize_mxfp8(packed, scales, out_dtype=x.dtype)


@_mxfp8_roundtrip_ste.vjp
def _mxfp8_roundtrip_ste_vjp(primals, cotangent, output):  # noqa: ARG001
    return (cotangent.astype(primals[0].dtype),)


def _unpack_mxfp8_to_uint8(packed: mx.array, last_dim: int) -> mx.array:
    """Re-expand the packed uint32 form into a flat uint8 e4m3 byte array.

    ``mx.quantize(mode='mxfp8')`` packs 4 e4m3 bytes per uint32 in the order
    ``[byte0 | byte1 | byte2 | byte3]``. The prepared-buffer Path C tests still
    use this helper when they create explicit FP8 byte/scales inputs.
    """

    bytes_view = packed.view(mx.uint8)
    expected_last = packed.shape[-1] * 4
    if expected_last != last_dim:
        raise ValueError(
            f"_unpack_mxfp8_to_uint8: expected unpacked last_dim={expected_last} "
            f"to match {last_dim}"
        )
    return bytes_view


def sparse_mla_blockscaled_metal_status(
    *arrays: mx.array,
) -> SparseMLABlockScaledMetalStatus:
    """Return whether the retired Path B kernel is currently dispatchable."""

    if len(arrays) == 3:
        _resolve_shapes(arrays[0], arrays[1], arrays[2], d_v=None)
    if not mx.metal.is_available():
        return SparseMLABlockScaledMetalStatus(
            available=False,
            reason="MLX Metal backend is not available on the default GPU device",
        )
    return SparseMLABlockScaledMetalStatus(
        available=False,
        reason=_DIRECT_MSL_RETIRED_REASON,
    )


def sparse_mla_blockscaled_fwd_metal(
    q: mx.array,
    kv: mx.array,
    indices: mx.array,
    *,
    sm_scale: float | None = None,
    d_v: int | None = None,
) -> Tuple[mx.array, mx.array] | None:
    """Retired direct-MSL Path B forward surface.

    Shape validation is preserved so callers still get useful contract errors,
    but no raw MSL kernel is constructed or dispatched.
    """

    shapes = _resolve_shapes(q, kv, indices, d_v=d_v)
    if sm_scale is None:
        sm_scale = shapes.qk_dim ** -0.5
    del sm_scale, shapes
    return None


def sparse_mla_blockscaled_bwd_metal(
    q: mx.array,
    kv: mx.array,
    d_out: mx.array,
    indices: mx.array,
    *,
    sm_scale: float | None = None,
    d_v: int | None = None,
) -> Tuple[mx.array, mx.array] | None:
    """Retired direct-MSL Path B backward surface."""

    shapes = _resolve_shapes(q, kv, indices, d_v=d_v)
    expected_d_out_shape = (shapes.batch, shapes.seq_len, shapes.heads, shapes.d_v)
    if tuple(d_out.shape) != expected_d_out_shape:
        raise ValueError(
            "sparse_mla_blockscaled_bwd_metal expected d_out shape "
            f"{expected_d_out_shape}, got {tuple(d_out.shape)}"
        )
    if sm_scale is None:
        sm_scale = shapes.qk_dim ** -0.5
    del sm_scale, shapes
    return None


def sparse_mla_blockscaled_reference(
    q: mx.array,
    kv: mx.array,
    indices: mx.array,
    *,
    sm_scale: float | None = None,
    d_v: int | None = None,
    return_lse: bool = False,
) -> mx.array | Tuple[mx.array, mx.array]:
    """Pure-MLX MXFP8 block-scaled reference."""

    shapes: SparseMLAShapes = _resolve_shapes(q, kv, indices, d_v=d_v)
    if shapes.qk_dim % MXFP8_BLOCK_SIZE != 0:
        return sparse_mla_attention_reference(
            q, kv, indices, sm_scale=sm_scale, d_v=d_v, return_lse=return_lse
        )

    q_recovered = cast(mx.array, _mxfp8_roundtrip_ste(q))
    kv_recovered = cast(mx.array, _mxfp8_roundtrip_ste(kv))
    return sparse_mla_attention_reference(
        q_recovered,
        kv_recovered,
        indices,
        sm_scale=sm_scale,
        d_v=d_v,
        return_lse=return_lse,
    )


@mx.custom_function
def sparse_mla_blockscaled_metal_apply(
    q: mx.array,
    kv: mx.array,
    indices: mx.array,
) -> mx.array:
    """Compatibility wrapper for the retired direct-MSL default forward."""

    return cast(mx.array, sparse_mla_blockscaled_reference(q, kv, indices))


@sparse_mla_blockscaled_metal_apply.vjp
def _sparse_mla_blockscaled_metal_apply_vjp(primals, cotangent, output):
    del output
    q, kv, indices = primals

    def _reference_apply(q_: mx.array, kv_: mx.array) -> mx.array:
        return cast(mx.array, sparse_mla_blockscaled_reference(q_, kv_, indices))

    _, vjps = mx.vjp(_reference_apply, (q, kv), (cotangent,))
    return (vjps[0], vjps[1], mx.zeros_like(indices))


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
    """Apply block-scaled FP8 sparse MLA through the reference fallback.

    ``force_metal=True`` keeps its historical meaning: require the old
    direct-MSL Path B surface. Since that surface is retired, forced calls raise
    instead of silently proxying to Path C, whose high-level float-carrier
    wrapper would need hidden quantization/unpacking staging.
    """

    shapes = _resolve_shapes(q, kv, indices, d_v=d_v)
    if sm_scale is None:
        sm_scale = shapes.qk_dim ** -0.5

    status = sparse_mla_blockscaled_metal_status(q, kv, indices)
    if force_metal:
        raise RuntimeError(
            f"sparse_mla_blockscaled_apply: Metal path unavailable: {status.reason}"
        )

    return sparse_mla_blockscaled_reference(
        q,
        kv,
        indices,
        sm_scale=sm_scale,
        d_v=d_v,
        return_lse=return_lse,
    )


__all__ = [
    "MXFP8_BLOCK_SIZE",
    "SparseMLABlockScaledMetalStatus",
    "SparseMLABlockScaledQKReducePathCStatus",
    "blockscaled_sparse_mla_qk_path_c_status",
    "blockscaled_sparse_mla_qk_reduce_path_c",
    "blockscaled_sparse_mla_qk_reduce_path_c_status",
    "sparse_mla_blockscaled_apply",
    "sparse_mla_blockscaled_bwd_metal",
    "sparse_mla_blockscaled_fwd_metal",
    "sparse_mla_blockscaled_metal_apply",
    "sparse_mla_blockscaled_metal_status",
    "sparse_mla_blockscaled_reference",
]
