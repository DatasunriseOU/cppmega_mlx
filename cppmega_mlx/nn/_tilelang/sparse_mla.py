"""Path B port of cppmega's sparse-MLA fwd/bwd TileLang pair.

Source attribution
------------------

Forward source on gb10:
    cppmega/megatron/sparse_mla_ops/tilelang_sparse_mla_fwd.py
Backward source on gb10:
    cppmega/megatron/sparse_mla_ops/tilelang_sparse_mla_bwd.py
Autograd glue:
    cppmega/megatron/sparse_mla_ops/sparse_mla.py (class SparseMLA)

These come from the upstream NVIDIA Megatron-LM PR #3674 (DSA "thd" branch),
in turn ported from tile-ai/tilelang/examples/deepseek_v32/.

Status on Apple Metal (tilelang 0.1.9)
--------------------------------------

The TileLang sparse-MLA kernels rely on ``T.gemm`` for every matmul tile (3 in
forward, 5 in backward). On the ``metal`` target tilelang 0.1.9 raises::

    InternalError: Check failed: (0) is false: Unsupported target for gemm:
    metal -keys=metal,gpu -max_function_args=31 ...

This is the same ``T.gemm`` blocker noted in the porting plan: tilelang's metal
codegen has no ``GemmInst`` registered. Until tile-ai/tilelang HEAD's Apple PRs
land that wire up the simdgroup_matrix path (a parallel agent is rebuilding
tilelang from HEAD), this module cannot lower the upstream sparse-MLA primfuncs
through Path B.

Until that blocker lifts, ``sparse_mla_fwd_metal`` and ``sparse_mla_bwd_metal``
return ``MetalSparseMLAStatus(available=False, reason=...)`` and
``sparse_mla_apply`` falls back to the pure-MLX reference at
``cppmega_mlx.nn.sparse_mla.sparse_mla_attention_reference``. That reference is
the parity oracle — it is differentiable through MLX autograd, so callers that
need gradients during the blocker window get correct values via MLX, just at
slower throughput than the planned Metal kernel.

bf16 vs fp16 carrier note
-------------------------

tilelang 0.1.9 has documented bf16 simdgroup MSL bugs (cubecl#1202). The Path B
contract is to force fp16 carrier for the Apple Metal port, downcast inputs at
the boundary, and document the dtype delta from gb10 (which uses bf16). The
fp16 forced-cast lives in ``_promote_to_fp16_carrier`` below. When the GEMM
blocker lifts, the Metal kernel will accept fp16 directly and skip the bf16
path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import mlx.core as mx

from cppmega_mlx.nn._tilelang._msl_transform import (
    can_run_metal,
    msl_dispatch_status,
)
from cppmega_mlx.nn.sparse_mla import sparse_mla_attention_reference


# ---------------------------------------------------------------------------
# Public status surface
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SparseMLAMetalStatus:
    """Runtime status of the Path B sparse-MLA kernel."""

    available: bool
    reason: str
    fp16_carrier: bool = True


_BLOCKER_REASON = (
    "tilelang 0.1.9 metal target does not support T.gemm (Unsupported target "
    "for gemm: metal); waiting on tile-ai/tilelang HEAD with Apple simdgroup "
    "PRs. See cppmega_mlx/nn/_tilelang/sparse_mla.py module docstring."
)


def sparse_mla_metal_status(*arrays: mx.array) -> SparseMLAMetalStatus:
    """Return whether the Path B kernel is currently dispatchable.

    The function intentionally returns the same blocker reason regardless of
    runtime/device because the gating happens at compile time inside tilelang,
    not at MSL dispatch time. Float-dtype arrays are still validated through
    ``msl_dispatch_status`` so that a future enabled path can reuse the same
    pre-flight checks; integer index buffers are skipped (the reference dtype
    set in ``_msl_transform`` is fp16/fp32/bf16 only).
    """

    if not can_run_metal():
        return SparseMLAMetalStatus(
            available=False,
            reason="MLX Metal backend is not available on the default GPU device",
        )
    float_arrays = [a for a in arrays if a.dtype in (mx.float16, mx.float32, mx.bfloat16)]
    if float_arrays:
        runtime = msl_dispatch_status(*float_arrays)
        if not runtime.available:
            return SparseMLAMetalStatus(available=False, reason=runtime.reason)
    return SparseMLAMetalStatus(available=False, reason=_BLOCKER_REASON)


# ---------------------------------------------------------------------------
# fp16 carrier helper
# ---------------------------------------------------------------------------


def _promote_to_fp16_carrier(x: mx.array) -> mx.array:
    """Force fp16 dtype for tensors entering the Path B Metal carrier."""

    if x.dtype == mx.float16:
        return x
    if x.dtype == mx.bfloat16:
        # Round-trip through fp32 to avoid bf16 mantissa loss on a direct cast.
        return x.astype(mx.float32).astype(mx.float16)
    return x.astype(mx.float16)


# ---------------------------------------------------------------------------
# Forward / backward stubs
# ---------------------------------------------------------------------------


def sparse_mla_fwd_metal(
    q: mx.array,
    kv: mx.array,
    indices: mx.array,
    *,
    sm_scale: float | None = None,
    d_v: int | None = None,
) -> Tuple[SparseMLAMetalStatus, mx.array | None, mx.array | None]:
    """Path B forward stub.

    Returns ``(status, out, lse)``. While ``status.available`` is False the
    Metal path is gated; ``out`` and ``lse`` come back as ``None``. Callers are
    expected to consult ``status.reason`` and route through the pure-MLX
    reference.
    """

    status = sparse_mla_metal_status(q, kv, indices)
    if not status.available:
        return status, None, None

    # Reserved for the post-blocker implementation. The plan:
    #   1. Build the TileLang PrimFunc with fp16 carrier (bf16 left for after PR
    #      tile-ai/tilelang#NNNN ships an MSL simdgroup_matrix bf16 fix).
    #   2. Lower with target='metal', strip ``kernel void`` signature, mark Q,
    #      KV, Indices as ``const device``, leave Output and Lse as ``device``.
    #   3. Buffer params come back alphabetic — see make_mlx_body in the Path B
    #      reference at /tmp/path_b_msl_mlx/bench_msl_path_b.py.
    raise NotImplementedError(_BLOCKER_REASON)


def sparse_mla_bwd_metal(
    q: mx.array,
    kv: mx.array,
    out: mx.array,
    grad_out: mx.array,
    indices: mx.array,
    lse: mx.array,
    *,
    sm_scale: float | None = None,
    d_v: int | None = None,
) -> Tuple[SparseMLAMetalStatus, mx.array | None, mx.array | None]:
    """Path B backward stub.

    Returns ``(status, dq, dkv)``. Guarded the same way as the forward.
    """

    status = sparse_mla_metal_status(q, kv, indices, out, grad_out)
    if not status.available:
        return status, None, None

    raise NotImplementedError(_BLOCKER_REASON)


# ---------------------------------------------------------------------------
# High-level apply (with fallback)
# ---------------------------------------------------------------------------


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
    """Apply sparse MLA, preferring Path B when available.

    During the GEMM blocker window this routes everything through the
    pure-MLX reference. Once the Metal path is enabled the differentiable
    forward will be wrapped via mx.custom_function with a manual VJP that
    invokes ``sparse_mla_bwd_metal``.

    Args:
        force_metal: if True, raise instead of falling back when the Metal
            path is unavailable. Useful for tests that want to surface the
            blocker rather than silently downgrade.
    """

    status = sparse_mla_metal_status(q, kv, indices)
    if not status.available:
        if force_metal:
            raise RuntimeError(
                f"sparse_mla_apply: Metal path unavailable: {status.reason}"
            )
        # Fallback through the pure-MLX reference (autograd-compatible).
        result = sparse_mla_attention_reference(
            q,
            kv,
            indices,
            sm_scale=sm_scale,
            d_v=d_v,
            return_lse=return_lse,
        )
        return result

    # Path B with carrier promotion + custom VJP wiring will go here when the
    # blocker lifts. For now we do not silently dispatch a half-built kernel.
    raise NotImplementedError(_BLOCKER_REASON)


__all__ = [
    "SparseMLAMetalStatus",
    "sparse_mla_apply",
    "sparse_mla_bwd_metal",
    "sparse_mla_fwd_metal",
    "sparse_mla_metal_status",
]
