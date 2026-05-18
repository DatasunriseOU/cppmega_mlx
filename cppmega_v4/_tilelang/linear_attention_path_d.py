"""GDN Path D — Triton kernel -> TileLang via ``poc.triton_frontend``.

Strategy (per integration plan):

    from poc.triton_frontend import from_triton_kernel
    prim = from_triton_kernel(
        fla_chunk_gated_delta_rule_inner_triton_kernel,
        grid=...,
        constexprs={...},
        target='metal',
    )
    kernel = tilelang.compile(prim, target='metal', execution_backend='tvm_ffi', out_idx=...)

The frontend now routes the real FLA chunk-delta-h TTIR op set through
OP_TABLE and produces a TileLang PrimFunc for the real captured chunk-h
kernel. Path D is still not a runnable cppmega_v4 backend until a compiled
runtime adapter maps that PrimFunc back to the GDN recurrent call
signature. The dispatch therefore keeps falling back to Path A, but the
status reason reports the real remaining blocker rather than stale
``tl.dot`` / ``tl.exp`` coverage.

This module never modifies the host TileLang frontend or FLA — both are
read-only imports.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from cppmega_v4._tilelang._path_d_deps import (
    ensure_fla_root,
    ensure_triton_frontend_root,
)


def _triton_frontend_importable() -> tuple[bool, str]:
    """Probe whether triton + ``poc.triton_frontend`` are both reachable."""
    try:
        import triton  # noqa: F401
    except Exception as exc:
        return False, f"triton not importable: {exc.__class__.__name__}: {exc}"
    ensure_triton_frontend_root()
    try:
        from poc.triton_frontend import from_triton_kernel  # noqa: F401
    except Exception as exc:
        return False, f"poc.triton_frontend not importable: {exc}"
    return True, "triton + poc.triton_frontend importable"


def _fla_chunk_kernel_importable() -> tuple[bool, str]:
    """Probe whether FLA's concrete chunk-delta-h kernel is reachable."""
    ensure_fla_root()
    try:
        from fla.ops.common.chunk_delta_h import (  # noqa: F401
            chunk_gated_delta_rule_fwd_kernel_h_blockdim64,
        )
    except Exception as exc:
        return False, f"FLA chunk_delta_h kernel not importable: {exc}"
    return True, "FLA chunk_delta_h kernel importable"


def _path_d_runtime_status() -> tuple[bool, str]:
    """Return runtime availability and the current concrete blocker."""

    ok_fe, reason_fe = _triton_frontend_importable()
    if not ok_fe:
        return False, reason_fe
    ok_src, reason_src = _fla_chunk_kernel_importable()
    if not ok_src:
        return False, reason_src

    from cppmega_v4._tilelang.linear_attention_path_d_real import lower_fla_chunk_h

    res = lower_fla_chunk_h()
    if res.status == "FAILED_OPS":
        return False, (
            "Triton frontend reached real FLA chunk_delta_h TTIR but OP_TABLE "
            f"is missing ops: {res.missing_ops!r}"
        )
    if res.status == "LOWERED_DEGRADED":
        return False, (
            "Triton frontend OP_TABLE covers real FLA chunk_delta_h TTIR "
            f"({len(set(res.visited_ops))} unique ops, missing=0), but this "
            "host is using the degraded text walker; no runnable PrimFunc / "
            "TileLang compile artifact is produced yet"
        )
    if res.status == "LOWERED_FULL":
        from cppmega_v4._tilelang.path_d_runtime_adapter import (
            gdn_runtime_adapter_status,
        )

        return gdn_runtime_adapter_status()
    return False, (
        f"Triton frontend could not lower FLA chunk_delta_h: status={res.status}; "
        f"error={res.error_type}: {res.error_message}"
    )


@lru_cache(maxsize=8)
def _try_lower_fla_chunk_kernel(target: str = "metal") -> tuple[Any | None, str]:
    """Attempt the real frontend lowering. Returns ``(prim_func, message)``."""

    del target  # lowering currently stops before backend-specific compilation
    ok_fe, reason_fe = _triton_frontend_importable()
    if not ok_fe:
        return None, reason_fe
    ok_src, reason_src = _fla_chunk_kernel_importable()
    if not ok_src:
        return None, reason_src

    from cppmega_v4._tilelang.linear_attention_path_d_real import lower_fla_chunk_h

    res = lower_fla_chunk_h()
    msg = (
        f"status={res.status}; visited={len(set(res.visited_ops))} unique ops; "
        f"missing={res.missing_ops!r}; error={res.error_type}: {res.error_message}"
    )
    return res.prim_func, msg


def _gdn_fwd_path_d_call(*args, **kwargs):
    """Path D entry — adapter hook raises so the dispatch falls back.

    Kept callable so the dispatch table is symmetric across paths.
    """
    from cppmega_v4._tilelang.path_d_runtime_adapter import gdn_fwd_runtime_call

    return gdn_fwd_runtime_call(*args, **kwargs)


__all__ = [
    "_fla_chunk_kernel_importable",
    "_gdn_fwd_path_d_call",
    "_path_d_runtime_status",
    "_triton_frontend_importable",
    "_try_lower_fla_chunk_kernel",
]
