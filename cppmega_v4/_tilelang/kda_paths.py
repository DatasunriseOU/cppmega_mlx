"""KDA multi-path dispatch (Paths B/C/D + auto-mode).

Same shape as ``linear_attention_paths.py`` but for the KDA backend. KDA
has no mlx-lm equivalent op (no Path E).

Backend status (May 2026):
    - Path A: pure-MLX naive recurrent KDA (golden) — always available.
    - Path B: hand-MSL KDA forward via mx.fast.metal_kernel (fwd only;
      bwd falls back to Path A). Supports any (B, T, H, HV, K, V) with
      HV % H == 0.
    - Path C: TileLang DSL @T.prim_func → tilelang.compile(target='metal',
      execution_backend='tvm_ffi'). Per-lane recurrent scan modeled on
      mamba3_path_c.py. Available iff tilelang + host MSL infra reachable.
    - Path D: ``poc.triton_frontend.from_triton_kernel`` over FLA KDA
      chunk kernels. Frontend op coverage is probed; runtime adapter is
      still pending.

Env override: ``CPPMEGA_V4_KERNEL_PATH__KDA``.
"""

from __future__ import annotations

import importlib

from cppmega_v4._tilelang._dispatch import PathName, PathStatus, auto_pick, env_override
from cppmega_v4.nn._external.fla_naive_kda import naive_recurrent_kda

ENV_VAR = "CPPMEGA_V4_KERNEL_PATH__KDA"


# ----- Path A -----


def _path_a_status() -> PathStatus:
    return PathStatus(path="path_a", available=True, reason="pure-MLX KDA reference")


def _path_a_call(*args, **kwargs):
    return naive_recurrent_kda(*args, **kwargs)


# ----- Path B (hand-MSL) -----


def _path_b_status() -> PathStatus:
    try:
        importlib.import_module("cppmega_v4._tilelang.kda_path_b")
        import mlx.core as mx
        if not hasattr(mx, "fast") or not hasattr(mx.fast, "metal_kernel"):
            return PathStatus(
                path="path_b", available=False,
                reason="mx.fast.metal_kernel not available on this build",
            )
        return PathStatus(
            path="path_b", available=True,
            reason=(
                "hand-MSL KDA forward via mx.fast.metal_kernel (fwd only; "
                "bwd falls back to Path A)"
            ),
        )
    except Exception as exc:
        return PathStatus(
            path="path_b", available=False,
            reason=f"path_b module not importable: {exc}",
        )


def _path_b_call(*args, **kwargs):
    if not _path_b_status().available:
        return _path_a_call(*args, **kwargs)
    mod = importlib.import_module("cppmega_v4._tilelang.kda_path_b")
    return mod.kda_forward_path_b(*args, **kwargs)


# ----- Path C (TileLang DSL) -----


def _path_c_status() -> PathStatus:
    try:
        from cppmega_v4._tilelang.kda_path_c import _path_c_runtime_status
    except Exception as exc:
        return PathStatus(
            path="path_c", available=False,
            reason=f"path_c module not importable: {exc}",
        )
    ok, reason = _path_c_runtime_status()
    return PathStatus(
        path="path_c", available=ok,
        reason=(
            f"KDA Path C: TileLang DSL @T.prim_func → tilelang.compile("
            f"target='metal', execution_backend='tvm_ffi'). {reason}"
        ),
    )


def _path_c_call(*args, **kwargs):
    if not _path_c_status().available:
        return _path_a_call(*args, **kwargs)
    try:
        from cppmega_v4._tilelang.kda_path_c import _kda_fwd_path_c_call
        return _kda_fwd_path_c_call(*args, **kwargs)
    except Exception:
        return _path_a_call(*args, **kwargs)


# ----- Path D (Triton frontend) -----


def _path_d_status() -> PathStatus:
    try:
        from cppmega_v4._tilelang.kda_path_d import _path_d_runtime_status
    except Exception as exc:
        return PathStatus(
            path="path_d", available=False,
            reason=f"path_d module not importable: {exc}",
        )
    ok, reason = _path_d_runtime_status()
    return PathStatus(
        path="path_d", available=ok,
        reason=(
            "KDA Path D: Triton kernel -> poc.triton_frontend."
            f"from_triton_kernel → tilelang.compile. {reason}"
        ),
    )


def _path_d_call(*args, **kwargs):
    if not _path_d_status().available:
        return _path_a_call(*args, **kwargs)
    try:
        from cppmega_v4._tilelang.kda_path_d import _kda_fwd_path_d_call
        return _kda_fwd_path_d_call(*args, **kwargs)
    except Exception:
        return _path_a_call(*args, **kwargs)


# ----- Public dispatch -----


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
            raise ValueError(
                "KDA has no Path E; use path_a, path_b, path_c, path_d, or auto"
            )
        return forced  # type: ignore[return-value]
    return auto_pick(
        kda_path_statuses(),
        preference=("path_c", "path_b", "path_d", "path_a"),
    )


def kda_recurrent_dispatch(*args, **kwargs):
    """Call the auto-selected KDA backend, falling back to Path A."""
    path = kda_auto_mode_for_inputs()
    fn = {
        "path_a": _path_a_call,
        "path_b": _path_b_call,
        "path_c": _path_c_call,
        "path_d": _path_d_call,
    }[path]
    return fn(*args, **kwargs)


__all__ = [
    "ENV_VAR",
    "kda_auto_mode_for_inputs",
    "kda_path_statuses",
    "kda_recurrent_dispatch",
]
