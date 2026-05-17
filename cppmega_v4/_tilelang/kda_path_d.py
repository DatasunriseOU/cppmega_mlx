"""KDA Path D — Triton kernel → TileLang via tilelang.poc.triton_frontend.

Mirrors ``linear_attention_path_d.py``. FLA's KDA chunk kernels are even
more elaborate than GDN (11 prim funcs across ~3340 LoC including
``chunk_delta_h_fwd``, ``chunk_delta_bwd``, ``chunk_o``, ``chunk_bwd_intra``,
``chunk_inter_solve_fused``, ``chunk_bwd_dqkwg``, ``chunk_bwd_dv``,
``chunk_bwd_gla_dA``, ``chunk_intra_token_parallel``, ``wy_fast``,
``wy_fast_bwd``). All use ``tl.dot`` and multi-stage pipelines — outside
the current Tier-1 (elementwise) op coverage of the frontend.

Status reports unavailable with the precise blocker. Dispatch falls back
to Path A. Host TileLang frontend and FLA are read-only imports.
"""

from __future__ import annotations

from typing import Any


def _triton_frontend_importable() -> tuple[bool, str]:
    try:
        import triton  # noqa: F401
    except Exception as exc:
        return False, f"triton not importable: {exc.__class__.__name__}: {exc}"
    try:
        from tilelang.poc.triton_frontend import from_triton_kernel  # noqa: F401
    except Exception as exc:
        return False, f"tilelang.poc.triton_frontend not importable: {exc}"
    return True, "triton + tilelang.poc.triton_frontend importable"


def _fla_kda_chunk_importable() -> tuple[bool, str]:
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
    return False, (
        "triton frontend + FLA KDA importable, but frontend is Tier-1 only; "
        "FLA KDA chunk uses tl.dot + multi-stage pipelines across 11 prims "
        "— extend tilelang.poc.triton_frontend.op_mapping.OP_TABLE with "
        "matmul, exp, and masked-load emitters to enable Path D"
    )


def _try_lower_fla_kda_kernel(target: str = "metal") -> tuple[Any | None, str]:
    """Integration seam for when op_mapping covers KDA's ops."""
    try:
        from tilelang.poc.triton_frontend import from_triton_kernel  # noqa: F401
        return None, "Path D lowering seam present but op_mapping coverage missing"
    except Exception as exc:
        return None, f"unexpected lowering error: {exc.__class__.__name__}: {exc}"


def _kda_fwd_path_d_call(*args, **kwargs):
    raise RuntimeError(
        "KDA Path D not yet runnable — triton_frontend op_mapping needs "
        "matmul/exp/masked-load emitters before FLA KDA chunk can lower"
    )


__all__ = [
    "_fla_kda_chunk_importable",
    "_kda_fwd_path_d_call",
    "_path_d_runtime_status",
    "_triton_frontend_importable",
    "_try_lower_fla_kda_kernel",
]
