"""KDA multi-path dispatch (Paths B/C/D/E + auto-mode).

Same shape as ``linear_attention_paths.py`` but for the KDA backend. Path E
reuses the SAME mlx-lm gated_delta Metal kernel as GDN — the kernel
auto-selects the vectorised-gate variant when ``g.ndim == 4``
(``mlx_lm/models/gated_delta.py::gated_delta_kernel``), which is exactly
KDA's per-K log-decay shape. Mirrors how
``mlx_lm/models/kimi_linear.py::KimiDeltaAttention`` drives the same
upstream kernel from KDA-shaped inputs.

Backend status (May 2026):
    - Path A: pure-MLX naive recurrent KDA (golden) — always available.
    - Path B: hand-MSL KDA fwd + real Metal bwd via mx.fast.metal_kernel.
      Forward supports initial_state, custom scale, any (B, T, H, HV, K, V)
      with HV % H == 0. Backward runs the snapshot-based recurrent
      scan with multi-simdgroup shared-memory reductions for V > 32,
      capped at V <= 256.
    - Path C: TileLang DSL @T.prim_func → tilelang.compile(target='metal',
      execution_backend='tvm_ffi'). Per-lane recurrent scan modeled on
      mamba3_path_c.py. Available iff tilelang + host MSL infra reachable.
    - Path D: ``poc.triton_frontend.from_triton_kernel`` over FLA KDA
      chunk kernels. cppmega owns the runtime adapter for FLA's forward
      multi-kernel launch, including packed varlen metadata.
    - Path E: vendored mlx-lm gated_delta vectorised-gate Metal kernel
      (the same kernel that powers GDN Path E; KDA just hits the
      ``g.ndim == 4`` branch). Requires Dk % 32 == 0 and Dv % 4 == 0;
      smaller dims drop to the pure-ops reference path.

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
                "hand-MSL KDA forward via mx.fast.metal_kernel; the "
                "autograd-aware wrapper kda_apply_path_b in kda_path_b_bwd.py "
                "also provides a real Metal backward (V <= 256)"
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


# ----- Path E (vendored mlx-lm gated_delta vectorised-gate kernel) -----


def _path_e_status() -> PathStatus:
    try:
        importlib.import_module(
            "cppmega_v4.nn._external.mlx_lm_kda_update"
        )
    except Exception as exc:
        return PathStatus(
            path="path_e", available=False,
            reason=(
                "mlx-lm gated_delta vectorised-gate adapter not importable: "
                f"{exc}"
            ),
        )
    return PathStatus(
        path="path_e", available=True,
        reason=(
            "vendored mlx-lm gated_delta vectorised-gate Metal kernel "
            "(Dk%32==0 & Dv%4==0; smaller dims fall back to the pure-ops "
            "reference). The same kernel as GDN Path E — KDA hits the "
            "g.ndim==4 branch in gated_delta_kernel."
        ),
    )


def _path_e_call(*args, **kwargs):
    if not _path_e_status().available:
        return _path_a_call(*args, **kwargs)
    try:
        from cppmega_v4.nn._external.mlx_lm_kda_update import kda_update
        return kda_update(*args, **kwargs)
    except Exception:
        return _path_a_call(*args, **kwargs)


# ----- Public dispatch -----


def kda_path_statuses() -> dict[PathName, PathStatus]:
    return {
        "path_a": _path_a_status(),
        "path_b": _path_b_status(),
        "path_c": _path_c_status(),
        "path_d": _path_d_status(),
        "path_e": _path_e_status(),
    }


def kda_auto_mode_for_inputs(*, env_var: str = ENV_VAR) -> PathName:
    forced = env_override(env_var)
    if forced is not None:
        return forced  # type: ignore[return-value]
    return auto_pick(
        kda_path_statuses(),
        preference=("path_c", "path_b", "path_e", "path_d", "path_a"),
    )


def kda_recurrent_dispatch(*args, **kwargs):
    """Call the auto-selected KDA backend, falling back to Path A."""
    path = kda_auto_mode_for_inputs()
    fn = {
        "path_a": _path_a_call,
        "path_b": _path_b_call,
        "path_c": _path_c_call,
        "path_d": _path_d_call,
        "path_e": _path_e_call,
    }[path]
    return fn(*args, **kwargs)


__all__ = [
    "ENV_VAR",
    "kda_auto_mode_for_inputs",
    "kda_path_statuses",
    "kda_recurrent_dispatch",
]
