"""KDA Path D — Triton kernel -> TileLang via ``poc.triton_frontend``.

Mirrors ``linear_attention_path_d.py`` but the KDA forward path fans out
through several FLA kernels: token-parallel intra, safe-gate intra,
inter/solve, common chunk_delta_h, and common chunk_o. Path D probes that
forward TTIR op surface through OP_TABLE and uses cppmega's runtime adapter
for the fixed prefill slice that has a non-degraded PrimFunc plus Metal
compile/launch coverage. The runtime adapter now covers the FLA forward
multi-kernel path for shape-specialized fp16 q/k/v inputs, fp32 gates,
custom scale, initial/final recurrent state, and packed varlen metadata.

Status therefore distinguishes "frontend ops are covered" from "backend
is runnable for the current signature". Host TileLang frontend and FLA are
read-only imports; cppmega owns compile/cache/launch and recurrent
signature binding.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from cppmega_v4._tilelang._path_d_deps import (
    ensure_fla_root,
    ensure_triton_frontend_root,
    import_triton_with_local_symbols,
    unsafe_fla_import_disabled_reason,
    unsafe_triton_frontend_import_disabled_reason,
    unsafe_triton_frontend_import_enabled,
)


def _triton_frontend_importable() -> tuple[bool, str]:
    root = ensure_triton_frontend_root()
    if not unsafe_triton_frontend_import_enabled():
        return False, unsafe_triton_frontend_import_disabled_reason(root)
    if root is None:
        return False, "poc.triton_frontend not importable: no local checkout found"
    try:
        import_triton_with_local_symbols()
    except Exception as exc:
        return False, f"triton not importable: {exc.__class__.__name__}: {exc}"
    try:
        from poc.triton_frontend import from_triton_kernel  # noqa: F401
    except Exception as exc:
        return False, f"poc.triton_frontend not importable: {exc}"
    return True, "triton + poc.triton_frontend importable"


def _fla_kda_chunk_importable() -> tuple[bool, str]:
    root = ensure_fla_root()
    if not unsafe_triton_frontend_import_enabled():
        return False, unsafe_fla_import_disabled_reason(root)
    if root is None:
        return False, "fla.ops.kda.chunk not importable: no local checkout found"
    try:
        from fla.ops.kda.chunk import chunk_kda  # noqa: F401
    except Exception as exc:
        return False, f"fla.ops.kda.chunk not importable: {exc}"
    return True, "fla.ops.kda.chunk importable"


def _path_d_runtime_status() -> tuple[bool, str]:
    ok_fe, reason_fe = _triton_frontend_importable()
    if not ok_fe:
        return False, reason_fe
    ok_src, reason_src = _fla_kda_chunk_importable()
    if not ok_src:
        return False, reason_src

    ok_cov, reason_cov = _kda_forward_op_coverage()
    if not ok_cov:
        return False, reason_cov
    from cppmega_v4._tilelang.path_d_runtime_adapter import (
        kda_runtime_adapter_status,
    )

    return kda_runtime_adapter_status(reason_cov)


def _try_lower_fla_kda_kernel(target: str = "metal") -> tuple[Any | None, str]:
    """Probe the real FLA KDA forward TTIR op surface.

    KDA Path D is a multi-kernel forward. This function does not compile or
    run it; it captures TTIR for representative forward kernels and confirms
    every visited op routes through OP_TABLE.
    """

    del target  # lowering currently stops before backend-specific compilation
    ok_fe, reason_fe = _triton_frontend_importable()
    if not ok_fe:
        return None, reason_fe
    ok_src, reason_src = _fla_kda_chunk_importable()
    if not ok_src:
        return None, reason_src
    ok_cov, reason_cov = _kda_forward_op_coverage()
    return None, reason_cov


@lru_cache(maxsize=1)
def _kda_forward_op_coverage() -> tuple[bool, str]:
    """Capture representative KDA forward TTIR and check OP_TABLE coverage."""

    try:
        from cppmega_v4._tilelang.linear_attention_path_d_real import (
            lower_fla_chunk_h,
            lower_fla_kda_chunk_o,
            lower_fla_kda_inter_solve,
            lower_fla_kda_intra_token_parallel,
            lower_fla_kda_recompute_w_u,
        )
    except Exception as exc:
        return False, f"KDA Path D coverage probe imports failed: {exc}"

    cases = (
        (
            "kda_intra_token_parallel",
            lambda: lower_fla_kda_intra_token_parallel(grid=(64, 1)),
        ),
        (
            "kda_inter_solve",
            lambda: lower_fla_kda_inter_solve(grid=(2, 1)),
        ),
        (
            "kda_recompute_w_u",
            lambda: lower_fla_kda_recompute_w_u(grid=(2, 1)),
        ),
        (
            "kda_chunk_delta_h",
            lambda: lower_fla_chunk_h(
                {
                    "BT": 32,
                    "BV": 16,
                    "USE_G": False,
                    "USE_GK": True,
                    "STORE_FINAL_STATE": True,
                    "SAVE_NEW_VALUE": True,
                },
                grid=(1, 1),
            ),
        ),
        (
            "kda_chunk_o_gk",
            lambda: lower_fla_kda_chunk_o(grid=(2, 2, 1)),
        ),
    )

    unique_ops = set()
    for name, lower_case in cases:
        res = lower_case()
        if res.status == "FAILED_OPS":
            return False, (
                f"KDA Path D coverage missing ops in {name}: "
                f"{res.missing_ops!r}"
            )
        if res.status != "LOWERED_FULL" or res.prim_func is None:
            return False, (
                f"KDA Path D runtime adapter installed; lowering failed in "
                f"{name}: status={res.status}; error={res.error_type}: "
                f"{res.error_message}. Run shim-dependent lowering in a "
                "subprocess (pytest-forked) or fresh interpreter."
            )
        if "DEGRADED" in res.prim_func.script():
            return False, (
                f"KDA Path D lowering produced DEGRADED markers in {name}"
            )
        unique_ops.update(res.visited_ops)

    return True, (
        "KDA Path D forward lowering coverage complete "
        f"({len(cases)} kernels, {len(unique_ops)} unique ops, missing=0)"
    )


def _kda_fwd_path_d_call(*args, **kwargs):
    ok_cov, reason_cov = _kda_forward_op_coverage()
    if not ok_cov:
        reason_cov = f"KDA Path D coverage failed: {reason_cov}"
    from cppmega_v4._tilelang.path_d_runtime_adapter import kda_fwd_runtime_call

    return kda_fwd_runtime_call(*args, coverage_reason=reason_cov, **kwargs)


__all__ = [
    "_fla_kda_chunk_importable",
    "_kda_fwd_path_d_call",
    "_kda_forward_op_coverage",
    "_path_d_runtime_status",
    "_triton_frontend_importable",
    "_try_lower_fla_kda_kernel",
]
