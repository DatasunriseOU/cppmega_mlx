"""KDA Path D — Triton kernel -> TileLang via ``poc.triton_frontend``.

Mirrors ``linear_attention_path_d.py`` but the KDA forward path fans out
through several FLA kernels: token-parallel intra, safe-gate intra,
inter/solve, common chunk_delta_h, and common chunk_o. Path D probes that
forward TTIR op surface through OP_TABLE, but remains unavailable as a
runtime backend until the frontend produces a runnable PrimFunc and
cppmega_v4 wires the compiled artifact to the KDA recurrent signature.

Status therefore distinguishes "frontend ops are covered" from "backend
is runnable". Dispatch falls back to Path A. Host TileLang frontend and
FLA are read-only imports.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from cppmega_v4._tilelang._path_d_deps import (
    ensure_fla_root,
    ensure_triton_frontend_root,
)


def _triton_frontend_importable() -> tuple[bool, str]:
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


def _fla_kda_chunk_importable() -> tuple[bool, str]:
    ensure_fla_root()
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
        from poc.triton_frontend import _walk_text_ttir
        from cppmega_v4._tilelang.linear_attention_path_d_real import (
            _capture_ttir_with_explicit_signature,
            _unwrap_to_jit_function,
            lower_fla_chunk_h,
        )
        from fla.ops.common.chunk_o import chunk_fwd_kernel_o
        from fla.ops.kda.chunk_intra import (
            chunk_kda_fwd_kernel_inter_solve_fused,
            chunk_kda_fwd_kernel_intra_sub_chunk,
        )
        from fla.ops.kda.chunk_intra_token_parallel import (
            chunk_kda_fwd_kernel_intra_token_parallel,
        )
    except Exception as exc:
        return False, f"KDA Path D coverage probe imports failed: {exc}"

    gdn_res = lower_fla_chunk_h()
    if gdn_res.status == "FAILED_OPS":
        return False, (
            "KDA Path D common chunk_delta_h coverage missing ops: "
            f"{gdn_res.missing_ops!r}"
        )
    if gdn_res.status not in {"LOWERED_DEGRADED", "LOWERED_FULL"}:
        return False, (
            "KDA Path D common chunk_delta_h probe failed: "
            f"status={gdn_res.status}; error={gdn_res.error_type}: "
            f"{gdn_res.error_message}"
        )

    cases = (
        (
            "kda_intra_token_parallel",
            chunk_kda_fwd_kernel_intra_token_parallel,
            {
                "H": 1,
                "HV": 1,
                "K": 64,
                "BT": 64,
                "BC": 16,
                "BH": 1,
                "IS_VARLEN": False,
            },
            {
                "q": "*fp16",
                "k": "*fp16",
                "g": "*fp32",
                "beta": "*fp32",
                "Aqk": "*fp32",
                "Akk": "*fp32",
                "scale": "fp32",
                "cu_seqlens": "*i64",
                "N": "i32",
                "T": "i32",
            },
        ),
        (
            "kda_intra_sub_chunk",
            chunk_kda_fwd_kernel_intra_sub_chunk,
            {
                "H": 1,
                "HV": 1,
                "K": 64,
                "BT": 64,
                "BC": 16,
                "BK": 64,
                "IS_VARLEN": False,
                "USE_GATHER": False,
            },
            {
                "q": "*fp16",
                "k": "*fp16",
                "g": "*fp32",
                "beta": "*fp32",
                "Aqk": "*fp32",
                "Akk": "*fp32",
                "scale": "fp32",
                "cu_seqlens": "*i64",
                "chunk_indices": "*i64",
                "T": "i32",
            },
        ),
        (
            "kda_inter_solve",
            chunk_kda_fwd_kernel_inter_solve_fused,
            {
                "H": 1,
                "HV": 1,
                "K": 64,
                "BT": 64,
                "BC": 16,
                "NC": 4,
                "BK": 64,
                "IS_VARLEN": False,
                "USE_SAFE_GATE": False,
            },
            {
                "q": "*fp16",
                "k": "*fp16",
                "g": "*fp32",
                "beta": "*fp32",
                "Aqk": "*fp32",
                "Akkd": "*fp32",
                "Akk": "*fp32",
                "scale": "fp32",
                "cu_seqlens": "*i64",
                "chunk_indices": "*i64",
                "T": "i32",
            },
        ),
        (
            "chunk_o",
            chunk_fwd_kernel_o,
            {
                "H": 1,
                "HV": 1,
                "K": 64,
                "V": 32,
                "BT": 64,
                "BK": 64,
                "BV": 32,
                "USE_G": False,
                "USE_G_GAMMA": False,
                "TRANSPOSE_STATE": False,
                "IS_VARLEN": False,
            },
            {
                "q": "*fp16",
                "k": "*fp16",
                "v": "*fp16",
                "h": "*fp16",
                "g": "*fp32",
                "g_gamma": "*fp32",
                "o": "*fp32",
                "cu_seqlens": "*i64",
                "chunk_indices": "*i64",
                "scale": "fp32",
                "T": "i32",
            },
        ),
    )

    unique_ops = set(gdn_res.visited_ops)
    for name, fn, constexprs, signature in cases:
        try:
            inner = _unwrap_to_jit_function(fn)
            ttir = _capture_ttir_with_explicit_signature(
                inner, constexprs, signature,
            )
            unique_ops.update(_walk_text_ttir(ttir))
        except NotImplementedError as exc:
            return False, f"KDA Path D OP_TABLE coverage gap in {name}: {exc}"
        except Exception as exc:
            return False, (
                f"KDA Path D TTIR capture failed in {name}: "
                f"{exc.__class__.__name__}: {exc}"
            )

    return True, (
        "KDA Path D forward TTIR OP_TABLE coverage complete "
        f"({len(cases) + 1} kernels, {len(unique_ops)} unique ops, missing=0)"
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
