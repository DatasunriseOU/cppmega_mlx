"""GDN linear-attention multi-path scaffolding (Paths B/C/D/E + auto-mode).

Each path exposes the same callable signature as Path A
(``cppmega_v4.nn._external.fla_naive_gated_delta_rule.naive_recurrent_gated_delta_rule``)
and falls back to Path A's reference when its backend is not yet wired up.
The intent is to let the rest of the v4 stack import a single dispatch entry
(``gated_delta_recurrent_dispatch``) and not care which kernel actually
runs — env override ``CPPMEGA_V4_KERNEL_PATH__LINEAR_ATTENTION`` forces
selection during benchmarking.

Backend status (May 2026):
    - Path A: pure-MLX naive recurrent (golden reference) — always available.
    - Path B: hand-MSL via ``mx.fast.metal_kernel`` — scaffold; awaits
      adaptation of ``mlx-recurrence/gla_scan.py`` to add the delta term.
    - Path C: TileLang DSL via ``tilelang.compile(target="metal",
      execution_backend="tvm_ffi")`` — scaffold; awaits lift of
      ``tilelang/examples/gdn/example_chunk_delta_h.py`` and friends into a
      Path-C wrapper mirroring ``cppmega_mlx/nn/_tilelang/mamba3_path_c.py``.
    - Path D: Triton frontend via ``poc.triton_frontend.from_triton_kernel``
      on FLA's ``chunk_gated_delta_rule`` — frontend op coverage is probed;
      runtime adapter is still pending.
    - Path E: vendored mlx-lm ``gated_delta_update`` op (PR #1217) —
      verbatim copy under
      ``cppmega_v4/nn/_external/_mlx_lm_gated_delta_vendored.py`` with adapter
      ``cppmega_v4/nn/_external/mlx_lm_gated_delta_update.py`` mapping our
      (q, k, v, beta, g) → upstream (q, k, v, a, b, A_log, dt_bias) by
      softplus_inverse(-g) for the gate and logit(beta) for the betas.
      Upstream Metal kernel needs Dk%32==0 & Dv%4==0; smaller dims fall back
      to the upstream ops path automatically.

When the user wants to validate against a specific backend they set
``CPPMEGA_V4_KERNEL_PATH__LINEAR_ATTENTION=path_c`` (or the path of choice);
``auto`` (default) picks the first available path per ``auto_pick``'s
preference order (C > B > E > D > A).
"""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Callable

import mlx.core as mx

from cppmega_v4._tilelang._dispatch import PathName, PathStatus, auto_pick, env_override
from cppmega_v4.nn._external.fla_naive_gated_delta_rule import (
    naive_recurrent_gated_delta_rule,
)

ENV_VAR = "CPPMEGA_V4_KERNEL_PATH__LINEAR_ATTENTION"
PathFn = Callable[..., tuple[mx.array, mx.array | None]]
_VALID_PATHS: tuple[PathName, ...] = (
    "path_a", "path_b", "path_c", "path_d", "path_e",
)


# --- Path A (always available) --------------------------------------------


def _path_a_status() -> PathStatus:
    return PathStatus(path="path_a", available=True, reason="pure-MLX reference")


def _path_a_call(*args, **kwargs):
    return naive_recurrent_gated_delta_rule(*args, **kwargs)


# --- Path B (hand-MSL via mx.fast.metal_kernel) ---------------------------


def _path_b_status() -> PathStatus:
    try:
        b_mod = importlib.import_module(
            "cppmega_v4._tilelang.linear_attention_path_b"
        )
        # Device-aware: on Apple the forward runs the hand-MSL
        # mx.fast.metal_kernel; on a CUDA host it routes the same GDN
        # recurrence through the host _cuda_eager bridge (see
        # gdn_forward_path_b._device_can_run_metal).
        if b_mod._device_can_run_metal():
            if not hasattr(mx, "fast") or not hasattr(mx.fast, "metal_kernel"):
                return PathStatus(
                    path="path_b", available=False,
                    reason="mx.fast.metal_kernel not available on this build",
                )
            return PathStatus(
                path="path_b", available=True,
                reason=(
                    "hand-MSL GDN forward via mx.fast.metal_kernel; the "
                    "autograd-aware wrapper gdn_apply_path_b in "
                    "linear_attention_path_b_bwd.py also provides a real Metal "
                    "backward (max(K, V) <= 256)"
                ),
            )
        # CUDA host: route through the TileLang-CUDA EAGER bridge.
        from cppmega_mlx.nn._tilelang._cuda_eager import cuda_eager_available
        cuda_ok, cuda_reason = cuda_eager_available()
        return PathStatus(
            path="path_b", available=cuda_ok,
            reason=(
                "GDN Path B forward via TileLang-CUDA EAGER bridge "
                "(gdn_fwd_cuda_eager; Metal unavailable on this CUDA host). "
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
    raise RuntimeError(f"GDN {path} unavailable and fallback disabled: {reason}")


def _dispatch_failure_or_fallback(
    path: PathName,
    exc: Exception,
    allow_fallback: bool,
    *args,
    **kwargs,
):
    # RULE #1 (no automated/silent fallbacks): a RUNTIME crash inside the
    # selected GDN kernel is a bug in that clear path. Silently switching to
    # Path A here would return a *different* (degraded) result and hide the
    # bug. Always RAISE so the failure surfaces and we fix the root cause.
    # (Explicit *unavailability* routing — when a path advertises it cannot
    # run — still goes through ``_fallback_or_raise``; that is selection, not
    # an on-failure fallback.)
    del allow_fallback, args, kwargs
    raise RuntimeError(
        f"GDN dispatch: selected {path} crashed at runtime "
        f"({type(exc).__name__}: {exc}). Refusing to silently fall back to "
        f"Path A (RULE #1) — this points at a real bug in {path}."
    ) from exc


def _path_b_call(*args, allow_fallback: bool = True, **kwargs):
    status = _path_b_status()
    if not status.available:
        return _fallback_or_raise(
            "path_b", status.reason, allow_fallback, *args, **kwargs,
        )
    mod = importlib.import_module("cppmega_v4._tilelang.linear_attention_path_b")
    return mod.gdn_forward_path_b(*args, **kwargs)


# --- Path C (TileLang DSL -> Metal) ---------------------------------------


def _path_c_status() -> PathStatus:
    try:
        from cppmega_v4._tilelang.linear_attention_path_c import (
            _path_c_runtime_status,
            _device_can_run_metal,
        )
    except Exception as exc:
        return PathStatus(
            path="path_c", available=False,
            reason=f"path_c module not importable: {exc}",
        )
    ok, reason = _path_c_runtime_status()
    # Device-aware (mirrors _path_b_status): on Apple Path C compiles the
    # TileLang DSL for target='metal' (tvm_ffi); on a CUDA host it routes the
    # same recurrence through the host _cuda_eager bridge (target='cuda').
    if _device_can_run_metal():
        detail = (
            "GDN Path C: TileLang DSL @T.prim_func → tilelang.compile("
            "target='metal', execution_backend='tvm_ffi')."
        )
    else:
        detail = (
            "GDN Path C via TileLang-CUDA EAGER bridge (gdn_fwd_cuda_eager; "
            "target='cuda', Metal unavailable on this CUDA host)."
        )
    return PathStatus(
        path="path_c", available=ok,
        reason=f"{detail} {reason}",
    )


def _path_c_call(*args, allow_fallback: bool = True, **kwargs):
    """Try Path C; fall back to Path A on any error (compile, runtime, missing infra)."""
    status = _path_c_status()
    if not status.available:
        return _fallback_or_raise(
            "path_c", status.reason, allow_fallback, *args, **kwargs,
        )
    try:
        from cppmega_v4._tilelang.linear_attention_path_c import (
            _gdn_fwd_path_c_call,
        )
        return _gdn_fwd_path_c_call(*args, **kwargs)
    except Exception as exc:
        return _dispatch_failure_or_fallback(
            "path_c", exc, allow_fallback, *args, **kwargs,
        )


# --- Path D (Triton frontend) ---------------------------------------------


def _path_d_status() -> PathStatus:
    try:
        from cppmega_v4._tilelang.linear_attention_path_d import (
            _path_d_runtime_status,
        )
    except Exception as exc:
        return PathStatus(
            path="path_d", available=False,
            reason=f"path_d module not importable: {exc}",
        )
    ok, reason = _path_d_runtime_status()
    return PathStatus(
        path="path_d", available=ok,
        reason=(
            "GDN Path D: Triton kernel -> poc.triton_frontend."
            f"from_triton_kernel → tilelang.compile. {reason}"
        ),
    )


def _path_d_call(*args, allow_fallback: bool = True, **kwargs):
    status = _path_d_status()
    if not status.available:
        return _fallback_or_raise(
            "path_d", status.reason, allow_fallback, *args, **kwargs,
        )
    try:
        from cppmega_v4._tilelang.linear_attention_path_d import (
            _gdn_fwd_path_d_call,
        )
        return _gdn_fwd_path_d_call(*args, **kwargs)
    except Exception as exc:
        return _dispatch_failure_or_fallback(
            "path_d", exc, allow_fallback, *args, **kwargs,
        )


# --- Path E (vendored mlx-lm PR #1217) ------------------------------------


def _path_e_status() -> PathStatus:
    try:
        importlib.import_module(
            "cppmega_v4.nn._external.mlx_lm_gated_delta_update"
        )
        return PathStatus(
            path="path_e", available=True,
            reason=(
                "vendored mlx-lm PR #1217 gated_delta_update (fast Metal kernel; "
                "forward runs for ANY Dk via the in-MSL remainder-mask and any "
                "Dv; gate must be g<=0 for GDN, otherwise fails closed so the "
                "dispatcher falls back to Path B/A)"
            ),
        )
    except Exception:
        return PathStatus(
            path="path_e",
            available=False,
            reason=(
                "mlx-lm gated_delta_update not vendored yet: fetch from "
                "https://github.com/ml-explore/mlx-lm/pull/1217 and place under "
                "cppmega_v4/nn/_external/mlx_lm_gated_delta_update.py"
            ),
        )


def _path_e_status_for_inputs(*args, **kwargs) -> PathStatus:
    """Input-aware Path E status: combine importability with eligibility.

    Path E is only *truly* available for a concrete call when (a) the adapter
    imports AND (b) the gate is representable (g<=0; no amplifying gate) AND
    (c) the shape hits the fast Metal kernel. ``auto_pick`` consumes this so it
    SKIPS Path E (falls through to D/A) rather than selecting a path that would
    silently clamp the gate or drop to the slow upstream ops fallback.

    Best-effort: if inputs cannot be parsed (e.g. unusual call shape) we fall
    back to the static status so behaviour is never worse than before.
    """
    base = _path_e_status()
    if not base.available:
        return base
    # Expected signature: (q, k, v, beta, g, ...). Probe g + dims defensively.
    try:
        from cppmega_v4.nn._external._path_e_eligibility import gdn_eligibility

        q = kwargs.get("q", args[0] if len(args) > 0 else None)
        v = kwargs.get("v", args[2] if len(args) > 2 else None)
        g = kwargs.get("g", args[4] if len(args) > 4 else None)
        if q is None or v is None or g is None:
            return base
        elig = gdn_eligibility(g, int(q.shape[-1]), int(v.shape[-1]))
    except Exception:
        return base
    if elig.eligible:
        return base
    return PathStatus(path="path_e", available=False, reason=elig.reason)


def _path_e_call(*args, allow_fallback: bool = True, **kwargs):
    from cppmega_v4.nn._external._path_e_eligibility import PathEUnavailable

    status = _path_e_status()
    if not status.available:
        return _fallback_or_raise(
            "path_e", status.reason, allow_fallback, *args, **kwargs,
        )
    try:
        op = importlib.import_module(
            "cppmega_v4.nn._external.mlx_lm_gated_delta_update"
        )
        return op.gated_delta_update(*args, **kwargs)
    except PathEUnavailable as exc:
        # AVAILABILITY/SELECTION signal, NOT a runtime crash. Path E raises
        # PathEUnavailable (mlx_lm_gated_delta_update.py) when the gate is
        # amplifying (g>0) or the shape is ineligible — it explicitly cannot
        # represent this input. That is semantically identical to the static
        # unavailability routed above, so route it the same way (fall back to
        # Path B/A, or RAISE when allow_fallback=False). RULE #1 stays intact:
        # a genuine kernel crash falls through to the always-raise handler.
        return _fallback_or_raise(
            "path_e", str(exc), allow_fallback, *args, **kwargs,
        )
    except Exception as exc:
        return _dispatch_failure_or_fallback(
            "path_e", exc, allow_fallback, *args, **kwargs,
        )


# --- Public dispatch ------------------------------------------------------


def linear_attention_path_statuses(
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
                raise ValueError(f"unsupported GDN path override: {path!r}")
            if status.path != path:
                raise ValueError(
                    f"GDN status override key {path!r} does not match "
                    f"status.path {status.path!r}"
                )
            statuses[path] = status
    return statuses


def linear_attention_auto_mode_for_inputs(
    *,
    env_var: str = ENV_VAR,
    status_overrides: Mapping[PathName, PathStatus] | None = None,
) -> PathName:
    forced = env_override(env_var)
    if forced is not None:
        return forced
    return auto_pick(linear_attention_path_statuses(status_overrides=status_overrides))


def gated_delta_recurrent_dispatch(
    *args,
    path: PathName | None = None,
    allow_fallback: bool = True,
    status_overrides: Mapping[PathName, PathStatus] | None = None,
    **kwargs,
):
    """Call the auto-selected GDN backend, falling back to Path A.

    Same callable signature as ``naive_recurrent_gated_delta_rule``.

    In auto-mode (``path is None``) Path E availability is recomputed from the
    *actual* inputs (gate sign + shape) so ``auto_pick`` SKIPS Path E for
    amplifying gates or ineligible shapes — it never selects a path that would
    silently clamp the gate or drop to the slow upstream ops fallback.
    """
    effective_overrides = dict(status_overrides) if status_overrides else {}
    if path is None and "path_e" not in effective_overrides:
        effective_overrides["path_e"] = _path_e_status_for_inputs(*args, **kwargs)
    status_overrides = effective_overrides or None
    statuses = linear_attention_path_statuses(status_overrides=status_overrides)
    if path is None:
        path = linear_attention_auto_mode_for_inputs(status_overrides=status_overrides)
    if path not in _VALID_PATHS:
        raise ValueError(f"unsupported GDN path {path!r}")
    status = statuses[path]
    if path != "path_a" and not status.available:
        return _fallback_or_raise(
            path, status.reason, allow_fallback, *args, **kwargs,
        )
    fn: PathFn = {
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
    "gated_delta_recurrent_dispatch",
    "linear_attention_auto_mode_for_inputs",
    "linear_attention_path_statuses",
]
