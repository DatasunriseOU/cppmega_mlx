"""KDA multi-path scaffolding (Paths B/C/D + auto-mode).

Same shape as ``linear_attention_paths.py`` but for the KDA backend. KDA has
no mlx-lm equivalent op (no Path E).

Backend status (May 2026):
    - Path A: pure-MLX naive recurrent KDA (golden) — always available.
    - Path B: hand-MSL KDA kernel — scaffold; pending Metal kernel.
    - Path C: TileLang DSL — scaffold; should lift ``tilelang/examples/kda/*.py``
      (11 prims, ~3340 LoC) through ``tilelang.compile(target='metal',
      execution_backend='tvm_ffi')`` mirroring ``mamba3_path_c.py`` skeleton.
    - Path D: Triton frontend — scaffold; should lift FLA KDA Triton kernels.

Env override: ``CPPMEGA_V4_KERNEL_PATH__KDA``.
"""

from __future__ import annotations

import importlib

from cppmega_v4._tilelang._dispatch import PathName, PathStatus, auto_pick, env_override
from cppmega_v4.nn._external.fla_naive_kda import naive_recurrent_kda

ENV_VAR = "CPPMEGA_V4_KERNEL_PATH__KDA"


def _path_a_status() -> PathStatus:
    return PathStatus(path="path_a", available=True, reason="pure-MLX KDA reference")


def _path_b_status() -> PathStatus:
    return PathStatus(
        path="path_b",
        available=False,
        reason=(
            "hand-MSL KDA kernel pending — DPLR transport A = αI + β u v^T plus "
            "gated-LA branch needs Metal kernel work beyond GDN's recurrence"
        ),
    )


def _path_c_status() -> PathStatus:
    try:
        importlib.import_module("tilelang")
    except Exception as exc:
        return PathStatus(
            path="path_c",
            available=False,
            reason=f"tilelang not importable: {exc}",
        )
    return PathStatus(
        path="path_c",
        available=False,
        reason=(
            "TileLang importable but KDA Path C wiring pending: lift "
            "tilelang/examples/kda/{chunk_delta_h_fwd,chunk_delta_bwd,chunk_o,"
            "chunk_bwd_intra,chunk_inter_solve_fused,chunk_bwd_dqkwg,"
            "chunk_bwd_dv,chunk_bwd_gla_dA,chunk_intra_token_parallel,"
            "wy_fast,wy_fast_bwd}.py via tilelang.compile(target='metal')"
        ),
    )


def _path_d_status() -> PathStatus:
    try:
        importlib.import_module("triton")
    except Exception:
        return PathStatus(
            path="path_d",
            available=False,
            reason="triton not importable",
        )
    return PathStatus(
        path="path_d",
        available=False,
        reason="Triton frontend KDA wiring pending",
    )


def _fallback(*args, **kwargs):
    return naive_recurrent_kda(*args, **kwargs)


def kda_path_statuses() -> dict[PathName, PathStatus]:
    return {
        "path_a": _path_a_status(),
        "path_b": _path_b_status(),
        "path_c": _path_c_status(),
        "path_d": _path_d_status(),
    }


def kda_auto_mode_for_inputs(*, env_var: str = ENV_VAR) -> PathName:
    forced = env_override(env_var)
    if forced is not None:
        if forced == "path_e":
            raise ValueError("KDA has no Path E; use path_a, path_b, path_c, path_d, or auto")
        return forced  # type: ignore[return-value]
    return auto_pick(
        kda_path_statuses(),
        preference=("path_c", "path_b", "path_d", "path_a"),
    )


def kda_recurrent_dispatch(*args, **kwargs):
    """Call the auto-selected KDA backend, falling back to Path A."""
    path = kda_auto_mode_for_inputs()
    # All B/C/D currently delegate to Path A.
    del path
    return _fallback(*args, **kwargs)


__all__ = [
    "ENV_VAR",
    "kda_auto_mode_for_inputs",
    "kda_path_statuses",
    "kda_recurrent_dispatch",
]
