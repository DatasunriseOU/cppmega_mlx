# pyright: reportInvalidTypeForm=false, reportMissingImports=false
"""Bridge between cppmega's CuTe DSL kernels and TileLang's CuTeDSL backend.

Background
----------
TileLang's upstream "CuTe DSL bridge" (PR apache/tilelang#1421, merged) is **not**
an importer for hand-written ``@cute.kernel`` Python. It is a *codegen target*:
``tilelang.compile(prim_func, target="cutedsl")`` lowers a TileLang
``T.prim_func`` IR module to NVIDIA CuTeDSL Python source via
``tilelang.jit.adapter.cutedsl.CuTeDSLKernelAdapter``. The relevant entry
points (read-only, do not modify) are:

* ``tilelang.compile(prim, target="cutedsl")`` — the public API.
* ``tilelang.jit.adapter.cutedsl.CuTeDSLKernelAdapter`` — the adapter class.
* ``tilelang.jit.adapter.cutedsl.checks.check_cutedsl_available`` — gate that
  requires ``nvidia-cutlass-dsl>=4.3.1`` (excluding 4.3.4) and the
  ``cutlass.cute`` Python package.
* ``tilelang.utils.target.normalize_cutedsl_target`` — accepts ``"cutedsl"``
  and ``"cuda"`` Targets carrying the ``"cutedsl"`` key.

What this module does
---------------------
The cppmega side (``/Volumes/external/sources/cppmega/cppmega/megatron/cute_dsl_mimo/``)
ships hand-written CuTeDSL Python kernels (``@cute.kernel`` /
``@cute.jit``-decorated ``cutlass.cute`` / ``quack`` code). These kernels are
already compiled and launched by ``cute.compile(...)``; they do not exist as
TileLang IR and TileLang's bridge does not consume external CuTeDSL Python.

There are two supportable paths:

1. **TileLang-IR → CuTeDSL emission** (the supported direction). Take a
   TileLang ``T.prim_func`` (e.g. an MMA written with ``T.gemm``) and route it
   through the unified dispatcher::

       from cppmega_mlx.nn._tilelang._engine_dispatch import dispatch_lower
       artifact = dispatch_lower(prim, target="cutedsl")  # CuTeDSL backend

   ``dispatch_lower`` already calls ``tilelang.compile(prim, target=...)`` so
   it works as-is for ``"cutedsl"`` once the cppmega-mlx environment has
   ``CPPMEGA_MLX_TILELANG_ENGINE=engine`` (auto mode also works on Linux+CUDA).

2. **External-CuTeDSL → TileLang import** (NOT supported by the upstream
   bridge). PR #1421 does not implement IR import from ``@cute.kernel``
   functions — it only emits CuTeDSL from TileLang IR. Wiring cppmega's
   ``SingleGemmWGMMA`` etc. as TileLang prim functions would require either:

       (a) re-expressing the kernel in TileLang ``T`` ops, or
       (b) extending TileLang with a CuTeDSL frontend (out of scope here).

This module exposes :func:`cute_dsl_to_tilelang_prim` as a thin shim that
documents the gap loudly. It also exposes :func:`compile_prim_to_cutedsl` for
the supported direction — the test harness exercises that path.

See ``tests/test_cute_to_tilelang_bridge.py`` for an end-to-end smoke.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "CUTE_BRIDGE_KNOWN_GAPS",
    "TILELANG_CUTEDSL_ENTRY",
    "CuteBridgeUnsupported",
    "compile_prim_to_cutedsl",
    "cute_dsl_to_tilelang_prim",
    "tilelang_cutedsl_available",
]


TILELANG_CUTEDSL_ENTRY = {
    "public_api": "tilelang.compile(prim, target='cutedsl')",
    "adapter_class": "tilelang.jit.adapter.cutedsl.CuTeDSLKernelAdapter",
    "availability_check": "tilelang.jit.adapter.cutedsl.checks.check_cutedsl_available",
    "target_normalizer": "tilelang.utils.target.normalize_cutedsl_target",
    "direction": "tilelang_ir_to_cutedsl_emission",
    "pr": "apache/tilelang#1421",
}


CUTE_BRIDGE_KNOWN_GAPS = (
    # Direction mismatch — the most important one.
    "TileLang's CuTeDSL bridge emits CuTeDSL Python from a TileLang T.prim_func; "
    "it does not consume hand-written @cute.kernel modules. cppmega's "
    "cute_dsl_mimo/* kernels (SingleGemmWGMMA, FA4 backward, fused 10-GEMM "
    "bwd_bwd_sm90_p4) are external CuTeDSL — they cannot be 'imported' as "
    "TileLang IR without re-expressing them in T-ops or building a new "
    "CuTeDSL frontend for TileLang.",
    # Hopper-specific feature gaps — even for the supported direction.
    "TileLang's CuTeDSL emitter (sm_90 path) does not expose all WGMMA / TMA "
    "knobs that cppmega's hand-written kernels rely on (e.g. quack.sm90_utils, "
    "warpgroup.OperandSource selection, custom SmemAllocator layouts, "
    "StMatrix epilogues). Re-expressing fused_bwd_bwd_sm90_p4 in TileLang IR "
    "would lose those optimizations.",
    # Environment gating.
    "tilelang.compile(target='cutedsl') requires nvidia-cutlass-dsl>=4.3.1 "
    "(excluding 4.3.4) AND a CUDA host. Mac/MLX hosts cannot exercise this "
    "path; tests must skip explicitly with a reason string.",
)


class CuteBridgeUnsupported(RuntimeError):
    """Raised when an external CuTeDSL kernel is asked to be 'imported' as
    a TileLang prim_func — the upstream bridge does not support that
    direction. See :data:`CUTE_BRIDGE_KNOWN_GAPS`.
    """


def tilelang_cutedsl_available() -> tuple[bool, str]:
    """Probe whether ``tilelang.compile(target='cutedsl')`` is reachable here.

    Returns ``(ok, reason)``. ``reason`` is empty on success; otherwise it
    describes the first failure (no tilelang, no cutlass.cute, version too
    old, libtilelang dylib load failure, etc.). Cheap — does not import
    anything heavier than ``tilelang.jit.adapter.cutedsl.checks``.
    """

    try:
        from tilelang.jit.adapter.cutedsl.checks import (  # noqa: F401
            check_cutedsl_available,
        )
    except Exception as exc:  # pragma: no cover - covered by skip in test
        return False, f"tilelang import failed: {exc.__class__.__name__}: {exc}"
    try:
        check_cutedsl_available()
    except Exception as exc:  # pragma: no cover - covered by skip in test
        return False, f"check_cutedsl_available failed: {exc}"
    return True, ""


def cute_dsl_to_tilelang_prim(cute_kernel: Any) -> Any:  # noqa: ARG001
    """Stub for the *unsupported* direction (external CuTeDSL → TileLang IR).

    Raises :class:`CuteBridgeUnsupported` with a precise reason. This exists
    so callers can ``try: cute_dsl_to_tilelang_prim(k); except
    CuteBridgeUnsupported: <fall back to direct cute.compile>`` rather than
    silently relying on a non-existent bridge.

    To dispatch a TileLang ``T.prim_func`` through the CuTeDSL backend (the
    direction PR #1421 *does* support), call
    :func:`compile_prim_to_cutedsl` or ``dispatch_lower(prim, target='cutedsl')``
    from ``cppmega_mlx.nn._tilelang._engine_dispatch``.
    """

    raise CuteBridgeUnsupported(
        "TileLang's CuTeDSL bridge (PR #1421) emits CuTeDSL from TileLang IR; "
        "it does not import @cute.kernel Python. To use cppmega's "
        "cute_dsl_mimo/* kernels, call them directly via cute.compile(...). "
        "To route a TileLang prim_func through the CuTeDSL backend instead, "
        "use cppmega_mlx.nn._cute_bridge.compile_prim_to_cutedsl(prim) or "
        "dispatch_lower(prim, target='cutedsl')."
    )


def compile_prim_to_cutedsl(prim_func: Any) -> Any:
    """Lower a TileLang ``T.prim_func`` through the CuTeDSL backend.

    Thin wrapper over the unified dispatcher
    (:func:`cppmega_mlx.nn._tilelang._engine_dispatch.dispatch_lower`) with
    ``target='cutedsl'``. The dispatcher honours ``CPPMEGA_MLX_TILELANG_ENGINE``
    so callers can force engine mode in CI even when the host's default is
    ``"auto"``. Returns a ``tilelang.compile`` artifact (a
    ``CuTeDSLKernelAdapter`` instance).
    """

    from cppmega_mlx.nn._tilelang._engine_dispatch import dispatch_lower

    return dispatch_lower(prim_func, target="cutedsl")
