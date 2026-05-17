"""Regular sparse-MLA compatibility surface after direct-MSL retirement.

The historical Path B forward in this module built a hand-written
``mx.fast.metal_kernel`` through ``_msl_transform.dispatch``. P2 cleanup retires
that raw direct-MSL runtime surface: accelerated production calls should route
through :mod:`cppmega_mlx.nn._tilelang.sparse_mla_path_c`, and unsupported or
unreceipted calls fall back to the pure-MLX reference.

The public names in this module remain for import compatibility. Status helpers
report the retirement explicitly, ``*_fwd_metal`` fails closed with ``None``,
and ``force_metal=True`` raises instead of silently selecting another backend.
The backward compatibility shim still delegates to the Path C owner-output
route because that surface no longer exposes public partial-output buffers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, cast

import mlx.core as mx

from cppmega_mlx.nn._tilelang._msl_transform import can_run_metal
from cppmega_mlx.nn.sparse_mla import (
    _resolve_shapes,
    sparse_mla_attention_reference,
)


@dataclass(frozen=True)
class SparseMLAMetalStatus:
    """Runtime status of the retired Path B sparse-MLA direct-MSL surface."""

    available: bool
    reason: str
    fp16_carrier: bool = True


_DIRECT_MSL_RETIRED_REASON = (
    "sparse_mla direct-MSL Path B is retired for production cleanup: raw "
    "_msl_transform.dispatch/mx.fast.metal_kernel callsites are no longer "
    "constructed. Use sparse_mla_path_c for the TileLang/tvm-ffi owner-output "
    "route, or the pure-MLX reference fallback for unsupported shapes."
)


def sparse_mla_metal_status(*arrays: mx.array) -> SparseMLAMetalStatus:
    """Return whether the retired Path B kernel is currently dispatchable."""

    del arrays
    if not can_run_metal():
        return SparseMLAMetalStatus(
            available=False,
            reason="MLX Metal backend is not available on the default GPU device",
        )
    return SparseMLAMetalStatus(available=False, reason=_DIRECT_MSL_RETIRED_REASON)


def sparse_mla_fwd_metal(
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


def sparse_mla_bwd_metal(
    q: mx.array,
    kv: mx.array,
    d_out: mx.array,
    indices: mx.array,
    *,
    sm_scale: float | None = None,
    d_v: int | None = None,
) -> Tuple[mx.array, mx.array] | None:
    """Backward compatibility shim backed by the Path C owner-output route."""

    from cppmega_mlx.nn._tilelang.sparse_mla_path_c import sparse_mla_bwd_path_c

    return sparse_mla_bwd_path_c(
        q,
        kv,
        d_out,
        indices,
        sm_scale=sm_scale,
        d_v=d_v,
    )


@mx.custom_function
def sparse_mla_metal_apply(
    q: mx.array,
    kv: mx.array,
    indices: mx.array,
) -> mx.array:
    """Compatibility wrapper for the retired direct-MSL default forward."""

    return cast(mx.array, sparse_mla_attention_reference(q, kv, indices))


@sparse_mla_metal_apply.vjp
def _sparse_mla_metal_apply_vjp(primals, cotangent, output):
    del output
    q, kv, indices = primals

    def _reference_apply(q_: mx.array, kv_: mx.array) -> mx.array:
        return cast(mx.array, sparse_mla_attention_reference(q_, kv_, indices))

    _, vjps = mx.vjp(_reference_apply, (q, kv), (cotangent,))
    return (vjps[0], vjps[1], mx.zeros_like(indices))


def sparse_mla_apply(
    q: mx.array,
    kv: mx.array,
    indices: mx.array,
    *,
    sm_scale: float | None = None,
    d_v: int | None = None,
    return_lse: bool = False,
    force_metal: bool = False,
) -> mx.array | Tuple[mx.array, mx.array]:
    """Apply sparse MLA through the retired Path B compatibility wrapper.

    ``force_metal=True`` keeps its historical meaning: require the old
    direct-MSL Path B surface. Since that surface is retired, forced calls raise
    instead of silently proxying to Path C.
    """

    shapes = _resolve_shapes(q, kv, indices, d_v=d_v)
    if sm_scale is None:
        sm_scale = shapes.qk_dim ** -0.5

    status = sparse_mla_metal_status(q, kv, indices)
    if force_metal:
        raise RuntimeError(f"sparse_mla_apply: Metal path unavailable: {status.reason}")

    return sparse_mla_attention_reference(
        q,
        kv,
        indices,
        sm_scale=sm_scale,
        d_v=d_v,
        return_lse=return_lse,
    )


__all__ = [
    "SparseMLAMetalStatus",
    "sparse_mla_apply",
    "sparse_mla_bwd_metal",
    "sparse_mla_fwd_metal",
    "sparse_mla_metal_apply",
    "sparse_mla_metal_status",
]
