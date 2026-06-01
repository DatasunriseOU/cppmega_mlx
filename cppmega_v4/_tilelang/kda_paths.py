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
from collections.abc import Mapping

from cppmega_v4._tilelang._dispatch import PathName, PathStatus, auto_pick, env_override
from cppmega_v4.nn._external.fla_naive_kda import naive_recurrent_kda

ENV_VAR = "CPPMEGA_V4_KERNEL_PATH__KDA"
_VALID_PATHS: tuple[PathName, ...] = (
    "path_a", "path_b", "path_c", "path_d", "path_e",
)


# ----- Path A -----


def _path_a_status() -> PathStatus:
    return PathStatus(path="path_a", available=True, reason="pure-MLX KDA reference")


def _path_a_call(*args, **kwargs):
    return naive_recurrent_kda(*args, **kwargs)


# ----- Path B (hand-MSL) -----


def _path_b_status() -> PathStatus:
    try:
        b_mod = importlib.import_module("cppmega_v4._tilelang.kda_path_b")
        import mlx.core as mx
        # Device-aware: Apple -> hand-MSL mx.fast.metal_kernel; CUDA host ->
        # host _cuda_eager bridge (see kda_forward_path_b._device_can_run_metal).
        if b_mod._device_can_run_metal():
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
        from cppmega_mlx.nn._tilelang._cuda_eager import cuda_eager_available
        cuda_ok, cuda_reason = cuda_eager_available()
        return PathStatus(
            path="path_b", available=cuda_ok,
            reason=(
                "KDA Path B forward via TileLang-CUDA EAGER bridge "
                "(kda_fwd_cuda_eager; Metal unavailable on this CUDA host). "
                f"{cuda_reason}"
            ),
        )
    except Exception as exc:
        return PathStatus(
            path="path_b", available=False,
            reason=f"path_b module not importable: {exc}",
        )


def _fallback_or_raise(
    path: PathName,
    reason: str,
    allow_fallback: bool,
    *args,
    **kwargs,
):
    if allow_fallback:
        return _path_a_call(*args, **kwargs)
    raise RuntimeError(f"KDA {path} unavailable and fallback disabled: {reason}")


def _dispatch_failure_or_fallback(
    path: PathName,
    exc: Exception,
    allow_fallback: bool,
    *args,
    **kwargs,
):
    # RULE #1 (no automated/silent fallbacks): a RUNTIME crash inside the
    # selected KDA kernel is a bug in that clear path. Silently switching to
    # Path A here would return a *different* (degraded) result and hide the
    # bug. Always RAISE so the failure surfaces and we fix the root cause.
    # (Explicit *unavailability* routing still goes through
    # ``_fallback_or_raise``; that is selection, not an on-failure fallback.)
    del allow_fallback, args, kwargs
    raise RuntimeError(
        f"KDA dispatch: selected {path} crashed at runtime "
        f"({type(exc).__name__}: {exc}). Refusing to silently fall back to "
        f"Path A (RULE #1) — this points at a real bug in {path}."
    ) from exc


def _path_b_call(*args, allow_fallback: bool = True, **kwargs):
    if not _path_b_status().available:
        return _fallback_or_raise(
            "path_b", _path_b_status().reason, allow_fallback, *args, **kwargs,
        )
    mod = importlib.import_module("cppmega_v4._tilelang.kda_path_b")
    return mod.kda_forward_path_b(*args, **kwargs)


# ----- Path C (TileLang DSL) -----


def _path_c_status() -> PathStatus:
    try:
        from cppmega_v4._tilelang.kda_path_c import (
            _path_c_runtime_status,
            _device_can_run_metal,
        )
    except Exception as exc:
        return PathStatus(
            path="path_c", available=False,
            reason=f"path_c module not importable: {exc}",
        )
    ok, reason = _path_c_runtime_status()
    # Device-aware (mirrors _path_b_status): Apple compiles target='metal';
    # CUDA host routes the same recurrence via the host _cuda_eager bridge.
    if _device_can_run_metal():
        detail = (
            "KDA Path C: TileLang DSL @T.prim_func → tilelang.compile("
            "target='metal', execution_backend='tvm_ffi')."
        )
    else:
        detail = (
            "KDA Path C via TileLang-CUDA EAGER bridge (kda_fwd_cuda_eager; "
            "target='cuda', Metal unavailable on this CUDA host)."
        )
    return PathStatus(
        path="path_c", available=ok,
        reason=f"{detail} {reason}",
    )


def _path_c_call(*args, allow_fallback: bool = True, **kwargs):
    if not _path_c_status().available:
        return _fallback_or_raise(
            "path_c", _path_c_status().reason, allow_fallback, *args, **kwargs,
        )
    try:
        from cppmega_v4._tilelang.kda_path_c import _kda_fwd_path_c_call
        return _kda_fwd_path_c_call(*args, **kwargs)
    except Exception as exc:
        return _dispatch_failure_or_fallback(
            "path_c", exc, allow_fallback, *args, **kwargs,
        )


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


def _path_d_call(*args, allow_fallback: bool = True, **kwargs):
    if not _path_d_status().available:
        return _fallback_or_raise(
            "path_d", _path_d_status().reason, allow_fallback, *args, **kwargs,
        )
    try:
        from cppmega_v4._tilelang.kda_path_d import _kda_fwd_path_d_call
        return _kda_fwd_path_d_call(*args, **kwargs)
    except Exception as exc:
        return _dispatch_failure_or_fallback(
            "path_d", exc, allow_fallback, *args, **kwargs,
        )


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
            "(fast kernel for Dk%32==0 & Dv%4==0; fails closed for smaller "
            "dims so the dispatcher falls back to Path B/A instead of the slow "
            "pure-ops reference). The same kernel as GDN Path E — KDA hits the "
            "g.ndim==4 branch in gated_delta_kernel. No gate-sign constraint "
            "(KDA passes g_decay=exp(g) directly)."
        ),
    )


def _path_e_status_for_inputs(*args, **kwargs) -> PathStatus:
    """Input-aware KDA Path E status: importability + shape eligibility.

    KDA has no gate-sign constraint, only the fast-kernel shape (Dk%32==0 &
    Dv%4==0). ``auto_pick`` consumes this so it SKIPS Path E for ineligible
    shapes rather than selecting the slow upstream ops fallback. Best-effort:
    unparseable inputs fall back to the static status.
    """
    base = _path_e_status()
    if not base.available:
        return base
    # KDA signature: (q, k, v, g, beta, ...). q.shape[-1]=Dk, v.shape[-1]=Dv.
    try:
        from cppmega_v4.nn._external._path_e_eligibility import kda_eligibility

        q = kwargs.get("q", args[0] if len(args) > 0 else None)
        v = kwargs.get("v", args[2] if len(args) > 2 else None)
        if q is None or v is None:
            return base
        elig = kda_eligibility(int(q.shape[-1]), int(v.shape[-1]))
    except Exception:
        return base
    if elig.eligible:
        return base
    return PathStatus(path="path_e", available=False, reason=elig.reason)


def _path_e_call(*args, allow_fallback: bool = True, **kwargs):
    if not _path_e_status().available:
        return _fallback_or_raise(
            "path_e", _path_e_status().reason, allow_fallback, *args, **kwargs,
        )
    try:
        from cppmega_v4.nn._external._path_e_eligibility import PathEUnavailable
        from cppmega_v4.nn._external.mlx_lm_kda_update import kda_update
        return kda_update(*args, **kwargs)
    except PathEUnavailable as exc:
        # AVAILABILITY/SELECTION signal (e.g. an ineligible shape), NOT a
        # runtime crash. Route it like static unavailability above so the
        # dispatcher falls back to Path B/A (or RAISES when allow_fallback is
        # False). A genuine kernel crash still hits the always-raise handler.
        return _fallback_or_raise(
            "path_e", str(exc), allow_fallback, *args, **kwargs,
        )
    except Exception as exc:
        return _dispatch_failure_or_fallback(
            "path_e", exc, allow_fallback, *args, **kwargs,
        )


# ----- Public dispatch -----


def kda_path_statuses(
    status_overrides: Mapping[PathName, PathStatus] | None = None,
) -> dict[PathName, PathStatus]:
    statuses = {
        "path_a": _path_a_status(),
        "path_b": _path_b_status(),
        "path_c": _path_c_status(),
        "path_d": _path_d_status(),
        "path_e": _path_e_status(),
    }
    if status_overrides:
        for path, status in status_overrides.items():
            if path not in _VALID_PATHS:
                raise ValueError(f"unsupported KDA path override: {path!r}")
            if status.path != path:
                raise ValueError(
                    f"KDA status override key {path!r} does not match "
                    f"status.path {status.path!r}"
                )
            statuses[path] = status
    return statuses


def kda_auto_mode_for_inputs(
    *,
    env_var: str = ENV_VAR,
    status_overrides: Mapping[PathName, PathStatus] | None = None,
) -> PathName:
    forced = env_override(env_var)
    if forced is not None:
        return forced  # type: ignore[return-value]
    return auto_pick(
        kda_path_statuses(status_overrides=status_overrides),
        preference=("path_c", "path_b", "path_e", "path_d", "path_a"),
    )


def kda_recurrent_dispatch(
    *args,
    path: PathName | None = None,
    allow_fallback: bool = True,
    status_overrides: Mapping[PathName, PathStatus] | None = None,
    **kwargs,
):
    """Call the auto-selected KDA backend, falling back to Path A.

    In auto-mode Path E availability is recomputed from the actual shape so
    ``auto_pick`` SKIPS Path E for ineligible shapes (which would otherwise
    drop to the slow upstream ops fallback).
    """
    effective_overrides = dict(status_overrides) if status_overrides else {}
    if path is None and "path_e" not in effective_overrides:
        effective_overrides["path_e"] = _path_e_status_for_inputs(*args, **kwargs)
    status_overrides = effective_overrides or None
    statuses = kda_path_statuses(status_overrides=status_overrides)
    if path is None:
        path = kda_auto_mode_for_inputs(status_overrides=status_overrides)
    if path not in _VALID_PATHS:
        raise ValueError(f"unsupported KDA path {path!r}")
    status = statuses[path]
    if path != "path_a" and not status.available:
        return _fallback_or_raise(
            path, status.reason, allow_fallback, *args, **kwargs,
        )
    fn = {
        "path_a": _path_a_call,
        "path_b": _path_b_call,
        "path_c": _path_c_call,
        "path_d": _path_d_call,
        "path_e": _path_e_call,
    }[path]
    if path == "path_a":
        return fn(*args, **kwargs)
    return fn(*args, allow_fallback=allow_fallback, **kwargs)


__all__ = [
    "ENV_VAR",
    "kda_auto_mode_for_inputs",
    "kda_path_statuses",
    "kda_recurrent_dispatch",
]
