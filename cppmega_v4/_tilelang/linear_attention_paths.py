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
    - Path D: Triton frontend via ``tilelang.poc.triton_frontend.from_triton_kernel``
      on FLA's ``chunk_gated_delta_rule`` — scaffold; awaits frontend op
      coverage for the FLA kernel.
    - Path E: vendored mlx-lm ``gated_delta_update`` op (PR #1217) — scaffold;
      awaits cherry-pick + vendoring under
      ``cppmega_v4/nn/_external/mlx_lm_gated_delta_update.py``.

When the user wants to validate against a specific backend they set
``CPPMEGA_V4_KERNEL_PATH__LINEAR_ATTENTION=path_c`` (or the path of choice);
``auto`` (default) picks the first available path per ``auto_pick``'s
preference order (C > B > E > D > A).
"""

from __future__ import annotations

import importlib
import os
from typing import Callable

import mlx.core as mx

from cppmega_v4._tilelang._dispatch import PathName, PathStatus, auto_pick, env_override
from cppmega_v4.nn._external.fla_naive_gated_delta_rule import (
    naive_recurrent_gated_delta_rule,
)

ENV_VAR = "CPPMEGA_V4_KERNEL_PATH__LINEAR_ATTENTION"
PathFn = Callable[..., tuple[mx.array, mx.array | None]]


# --- Path A (always available) --------------------------------------------


def _path_a_status() -> PathStatus:
    return PathStatus(path="path_a", available=True, reason="pure-MLX reference")


def _path_a_call(*args, **kwargs):
    return naive_recurrent_gated_delta_rule(*args, **kwargs)


# --- Path B (hand-MSL via mx.fast.metal_kernel) ---------------------------


def _path_b_status() -> PathStatus:
    try:
        importlib.import_module("cppmega_v4._tilelang.linear_attention_path_b")
        # Confirm mx.fast.metal_kernel is callable (Metal available).
        if not hasattr(mx, "fast") or not hasattr(mx.fast, "metal_kernel"):
            return PathStatus(
                path="path_b", available=False,
                reason="mx.fast.metal_kernel not available on this build",
            )
        return PathStatus(
            path="path_b", available=True,
            reason="hand-MSL GDN forward via mx.fast.metal_kernel (fwd only; bwd falls back to Path A)",
        )
    except Exception as exc:
        return PathStatus(
            path="path_b", available=False,
            reason=f"path_b module not importable: {exc}",
        )


def _path_b_call(*args, **kwargs):
    if not _path_b_status().available:
        return _path_a_call(*args, **kwargs)
    mod = importlib.import_module("cppmega_v4._tilelang.linear_attention_path_b")
    return mod.gdn_forward_path_b(*args, **kwargs)


# --- Path C (TileLang DSL -> Metal) ---------------------------------------


def _path_c_status() -> PathStatus:
    try:
        importlib.import_module("tilelang")
    except Exception as exc:
        return PathStatus(
            path="path_c",
            available=False,
            reason=f"tilelang not importable: {exc}",
        )
    # tilelang importable, but the GDN PrimFunc -> Metal lowering isn't wired
    # through cppmega_v4 yet — mirror mamba3_path_c.py structure when we land it.
    return PathStatus(
        path="path_c",
        available=False,
        reason=(
            "TileLang importable but GDN Path C wiring pending: copy skeleton "
            "from cppmega_mlx/nn/_tilelang/mamba3_path_c.py, lift "
            "tilelang/examples/gdn/example_chunk_delta_h.py etc. via "
            "tilelang.compile(target='metal', execution_backend='tvm_ffi')"
        ),
    )


def _path_c_call(*args, **kwargs):
    return _path_a_call(*args, **kwargs)  # fallback


# --- Path D (Triton frontend) ---------------------------------------------


def _path_d_status() -> PathStatus:
    try:
        importlib.import_module("triton")
    except Exception:
        return PathStatus(
            path="path_d",
            available=False,
            reason="triton not importable (CPU/Apple Silicon)",
        )
    return PathStatus(
        path="path_d",
        available=False,
        reason=(
            "Triton frontend wiring pending: tilelang/poc/triton_frontend/"
            "from_triton_kernel on FLA chunk_gated_delta_rule"
        ),
    )


def _path_d_call(*args, **kwargs):
    return _path_a_call(*args, **kwargs)  # fallback


# --- Path E (vendored mlx-lm PR #1217) ------------------------------------


def _path_e_status() -> PathStatus:
    try:
        importlib.import_module(
            "cppmega_v4.nn._external.mlx_lm_gated_delta_update"
        )
        return PathStatus(path="path_e", available=True, reason="vendored mlx-lm op present")
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


def _path_e_call(*args, **kwargs):
    if not _path_e_status().available:
        return _path_a_call(*args, **kwargs)
    op = importlib.import_module("cppmega_v4.nn._external.mlx_lm_gated_delta_update")
    return op.gated_delta_update(*args, **kwargs)


# --- Public dispatch ------------------------------------------------------


def linear_attention_path_statuses() -> dict[PathName, PathStatus]:
    return {
        "path_a": _path_a_status(),
        "path_b": _path_b_status(),
        "path_c": _path_c_status(),
        "path_d": _path_d_status(),
        "path_e": _path_e_status(),
    }


def linear_attention_auto_mode_for_inputs(*, env_var: str = ENV_VAR) -> PathName:
    forced = env_override(env_var)
    if forced is not None:
        return forced  # type: ignore[return-value]
    return auto_pick(linear_attention_path_statuses())


def gated_delta_recurrent_dispatch(*args, **kwargs):
    """Call the auto-selected GDN backend, falling back to Path A.

    Same callable signature as ``naive_recurrent_gated_delta_rule``.
    """
    path = linear_attention_auto_mode_for_inputs()
    fn: PathFn = {
        "path_a": _path_a_call,
        "path_b": _path_b_call,
        "path_c": _path_c_call,
        "path_d": _path_d_call,
        "path_e": _path_e_call,
    }[path]
    return fn(*args, **kwargs)


__all__ = [
    "ENV_VAR",
    "gated_delta_recurrent_dispatch",
    "linear_attention_auto_mode_for_inputs",
    "linear_attention_path_statuses",
]
