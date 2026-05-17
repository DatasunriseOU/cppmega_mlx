"""GDN Path D — Triton kernel → TileLang via tilelang.poc.triton_frontend.

Strategy (per integration plan):

    from tilelang.poc.triton_frontend import from_triton_kernel
    prim = from_triton_kernel(
        fla_chunk_gated_delta_rule_inner_triton_kernel,
        grid=...,
        constexprs={...},
        target='metal',
    )
    kernel = tilelang.compile(prim, target='metal', execution_backend='tvm_ffi', out_idx=...)

The triton_frontend is currently Tier-1 (elementwise only — see
``poc/triton_frontend/__init__.py:898``). FLA's chunk_gated_delta_rule uses
``tl.dot``, ``tl.exp``, masked load/store, multi-stage pipelines — well
outside Tier-1. The expected outcome on first attempt is
``NotImplementedError`` from ``op_mapping``: that's the actionable signal
telling frontend devs which op to add next (matrix-multiply lowering, etc.).

Until the frontend covers those ops, Path D is wired but unavailable, and
the dispatch falls back to Path A. The status reason names the precise
blocker so the next person knows what to fix.

This module never modifies the host TileLang frontend or FLA — both are
read-only imports.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import mlx.core as mx


def _triton_frontend_importable() -> tuple[bool, str]:
    """Probe whether triton + tilelang.poc.triton_frontend are both reachable."""
    try:
        import triton  # noqa: F401
    except Exception as exc:
        return False, f"triton not importable: {exc.__class__.__name__}: {exc}"
    try:
        from tilelang.poc.triton_frontend import from_triton_kernel  # noqa: F401
    except Exception as exc:
        return False, f"tilelang.poc.triton_frontend not importable: {exc}"
    return True, "triton + tilelang.poc.triton_frontend importable"


def _fla_chunk_kernel_importable() -> tuple[bool, str]:
    """Probe whether FLA's chunk_gated_delta_rule entrypoint is reachable."""
    try:
        from fla.ops.gated_delta_rule.chunk import chunk_gated_delta_rule  # noqa: F401
    except Exception as exc:
        return False, f"fla.ops.gated_delta_rule.chunk not importable: {exc}"
    return True, "fla.ops.gated_delta_rule.chunk importable"


def _path_d_runtime_status() -> tuple[bool, str]:
    """Two-stage gate: frontend + source kernel both reachable, then
    op_mapping must cover the kernel's ops (the latter is only known after
    a real compile attempt — we surface that on first call).
    """
    ok_fe, reason_fe = _triton_frontend_importable()
    if not ok_fe:
        return False, reason_fe
    ok_src, reason_src = _fla_chunk_kernel_importable()
    if not ok_src:
        return False, reason_src
    # Triton frontend is currently Tier-1 (elementwise only). FLA chunk
    # gated-delta uses tl.dot / tl.exp / masked-pipeline — outside Tier-1.
    # Mark unavailable with the actionable blocker until op_mapping extends.
    return False, (
        "triton frontend + FLA importable, but frontend is Tier-1 (elementwise) "
        "only; FLA chunk_gated_delta_rule uses tl.dot and multi-stage pipelines "
        "— extend tilelang.poc.triton_frontend.op_mapping.OP_TABLE with matmul "
        "& exp emitters to enable Path D"
    )


@lru_cache(maxsize=8)
def _try_lower_fla_chunk_kernel(target: str = "metal") -> tuple[Any | None, str]:
    """Attempt the real frontend lowering. Returns (prim_func, message).

    Only invoked when ``_path_d_runtime_status()[0]`` is True. Currently a
    placeholder that always returns ``(None, "...")`` because Tier-1 doesn't
    cover the kernel's ops; left here as the integration seam for when the
    frontend grows tl.dot support.
    """
    try:
        from tilelang.poc.triton_frontend import from_triton_kernel
        from fla.ops.common.chunk_o import chunk_fwd_o  # noqa: F401

        # The actual entry would inspect chunk_gated_delta_rule_fwd's inner
        # @triton.jit kernels and pass each to from_triton_kernel. FLA wraps
        # them under chunk_gated_delta_rule_fwd_h / chunk_fwd_o; each has its
        # own grid + constexprs. Left as the integration seam:
        #
        #   from fla.ops.common.chunk_delta_h import (
        #       chunk_gated_delta_rule_fwd_h_kernel as _fla_kernel,
        #   )
        #   prim = from_triton_kernel(_fla_kernel, grid=..., constexprs=...,
        #                             target=target)
        #
        # That call currently raises NotImplementedError on tl.dot — exactly
        # the signal we want the next contributor to act on.
        return None, "Path D lowering seam present but op_mapping coverage missing"
    except Exception as exc:
        return None, f"unexpected lowering error: {exc.__class__.__name__}: {exc}"


def _gdn_fwd_path_d_call(*args, **kwargs):
    """Path D entry — currently always raises so the dispatch falls back.

    Kept callable so the dispatch table is symmetric across paths.
    """
    raise RuntimeError(
        "GDN Path D not yet runnable — triton_frontend op_mapping needs "
        "matmul/exp emitters before FLA chunk_gated_delta_rule can lower"
    )


__all__ = [
    "_fla_chunk_kernel_importable",
    "_gdn_fwd_path_d_call",
    "_path_d_runtime_status",
    "_triton_frontend_importable",
    "_try_lower_fla_chunk_kernel",
]
