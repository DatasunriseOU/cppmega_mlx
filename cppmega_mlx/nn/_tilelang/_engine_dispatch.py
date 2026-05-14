# pyright: reportInvalidTypeForm=false, reportMissingImports=false
"""Migration phase-1 dispatcher: engine vs MSL-shim lowering for Path-C kernels.

Path-C TileLang kernels (e.g. ``fp8_amax``, ``dsa_splitk_indexer_loss``) used
to call ``tilelang.compile(prim, target=...)`` directly. The unified
fused-kernel pipeline at ``/private/tmp/tl_poc_review`` keeps that as the
production path but adds a fallback to the legacy MSL-string-rewrite shim
(:func:`cppmega_mlx.nn._tilelang._msl_transform.lower_tilelang_to_msl_inline`)
for environments where ``tilelang.compile`` is unavailable.

Selection is driven by the ``CPPMEGA_MLX_TILELANG_ENGINE`` env var:

* ``"auto"`` (default): try the unified engine; fall back to the MSL shim on
  ``ImportError`` / ``ModuleNotFoundError`` with a one-shot ``UserWarning``.
* ``"engine"``: force the unified engine; surface engine errors as-is (no
  fallback).
* ``"shim"``: force the legacy MSL-string lowering; never call
  ``tilelang.compile``.

Engine results carry a ``_tilelang_engine_target`` attribute so callers (and
tests) can distinguish them from shim results, which are
:class:`TileLangMSLLowering` dataclasses with an ``msl_text`` field.
"""

from __future__ import annotations

import os
import warnings
from typing import Any, Sequence


_VALID_MODES = ("auto", "engine", "shim", "engine_with_msl_extraction")
_FLAG_ENV = "CPPMEGA_MLX_TILELANG_ENGINE"
_FALLBACK_WARNED = False
_MSL_EXTRACTION_FALLBACK_WARNED = False


def tilelang_engine_mode() -> str:
    """Return the current dispatcher mode from ``$CPPMEGA_MLX_TILELANG_ENGINE``.

    Unknown values fall back to ``"auto"`` with a ``UserWarning``.
    """

    raw = os.environ.get(_FLAG_ENV, "auto").strip().lower()
    if raw in _VALID_MODES:
        return raw
    warnings.warn(
        f"{_FLAG_ENV}={raw!r} is not one of {_VALID_MODES!r}; defaulting to 'auto'.",
        UserWarning,
        stacklevel=2,
    )
    return "auto"


def _with_pass_context(pass_configs: dict[str, Any] | None):
    if not pass_configs:
        return None
    from tilelang import tvm

    return tvm.transform.PassContext(opt_level=3, config=dict(pass_configs))


def _ensure_path_c_metal_intrinsics_registered() -> None:
    try:
        from cppmega_mlx.nn._tilelang._msl_transform import (
            _register_path_c_metal_fp8_intrinsics,
        )

        _register_path_c_metal_fp8_intrinsics()
    except Exception:
        pass


def _engine_compile(
    prim_func: Any,
    target: str,
    *,
    pass_configs: dict[str, Any] | None = None,
) -> Any:
    """Run ``tilelang.compile`` and stamp the result with the target tag.

    Normalizes legacy CLI-form metal targets (e.g.
    ``"metal -thread_warp_size=32"``) through ``_as_metal_target`` from
    ``_msl_transform`` so they bypass tilelang's ``determine_target``
    base-name allowlist (which rejects strings with ``-flag=value``
    suffixes post-#2143). Non-string targets (already-built
    ``tvm.target.Target`` objects) pass through unchanged.
    """

    import tilelang  # noqa: F401  - intentional eager import for ImportError surfacing
    _ensure_path_c_metal_intrinsics_registered()

    compile_target: Any = target
    if isinstance(target, str) and target.startswith("metal") and "-" in target:
        from cppmega_mlx.nn._tilelang._msl_transform import _as_metal_target

        compile_target = _as_metal_target(target)

    pass_context = _with_pass_context(pass_configs)
    if pass_context is None:
        artifact = tilelang.compile(prim_func, target=compile_target, out_idx=None)
    else:
        with pass_context:
            artifact = tilelang.compile(prim_func, target=compile_target, out_idx=None)
    try:
        setattr(artifact, "_tilelang_engine_target", target)
    except (AttributeError, TypeError):
        # Some builds wrap the artifact in a frozen / __slots__ object; preserve
        # the artifact unchanged if we cannot stamp it.
        pass
    return artifact


def _prim_func_param_count(prim_func: Any) -> int:
    params = getattr(prim_func, "params", None)
    if params is None:
        raise ValueError("PrimFunc-like object must expose a .params sequence")
    return len(params)


def _compile_target_for_native_tvm_ffi(target: Any) -> Any:
    if isinstance(target, str) and target.startswith("metal") and "-" in target:
        from cppmega_mlx.nn._tilelang._msl_transform import _as_metal_target

        return _as_metal_target(target)
    return target


def compile_native_tilelang_kernel(
    prim_func: Any,
    target: Any,
    *,
    out_idx: int | Sequence[int] | None,
    pass_configs: dict[str, Any] | None = None,
    allow_graph_outputs: bool = False,
) -> Any:
    """Compile ``prim_func`` for the native TileLang TVM-FFI MLX boundary.

    This is the replacement boundary for Path C production callers that should
    not consume ``TileLangMSLLowering.body`` or build ``mx.fast.metal_kernel``.
    It always requests ``execution_backend="tvm_ffi"`` from TileLang and wraps
    the artifact in :class:`NativeTileLangKernel`, whose dispatch contract
    requires caller-owned ``out=`` buffers by default.
    """

    import tilelang  # noqa: F401 - intentional eager import for ImportError surfacing
    from cppmega_mlx.nn._tilelang._mlx_runtime import (
        NativeTileLangKernel,
        normalize_out_idx,
    )
    from cppmega_mlx.nn._tilelang._msl_transform import _ensure_single_libtvm_ffi_image

    _ensure_path_c_metal_intrinsics_registered()
    _ensure_single_libtvm_ffi_image()

    num_params = _prim_func_param_count(prim_func)
    result_indices = normalize_out_idx(out_idx, num_params=num_params)
    compile_target = _compile_target_for_native_tvm_ffi(target)
    pass_context = _with_pass_context(pass_configs)
    if pass_context is None:
        artifact = tilelang.compile(
            prim_func,
            target=compile_target,
            execution_backend="tvm_ffi",
            out_idx=out_idx,
        )
    else:
        with pass_context:
            artifact = tilelang.compile(
                prim_func,
                target=compile_target,
                execution_backend="tvm_ffi",
                out_idx=out_idx,
            )
    try:
        setattr(artifact, "_tilelang_engine_target", target)
        setattr(artifact, "_tilelang_execution_backend", "tvm_ffi")
        setattr(artifact, "_tilelang_result_indices", result_indices)
    except (AttributeError, TypeError):
        pass
    return NativeTileLangKernel(
        artifact=artifact,
        result_indices=result_indices,
        num_params=num_params,
        target=target,
        allow_graph_outputs=allow_graph_outputs,
    )


def _engine_lower_for_msl_extraction(
    prim_func: Any,
    target: str,
    *,
    pass_configs: dict[str, Any] | None = None,
) -> Any:
    """Run TileLang lowering directly for MSL text plus launch metadata.

    ``tilelang.compile`` may return a disk-cached JITKernel whose source is
    intact but whose lowered ``device_mod`` is not retained. The MLX
    ``mx.fast.metal_kernel`` bridge needs both the MSL text and TileLang's
    launch extents, so MSL extraction uses ``tilelang.lower`` directly.
    """

    import tilelang  # noqa: F401  - intentional eager import for ImportError surfacing
    from tilelang.engine.lower import lower as tl_lower
    _ensure_path_c_metal_intrinsics_registered()

    lower_target: Any = target
    if isinstance(target, str) and target.startswith("metal") and "-" in target:
        from cppmega_mlx.nn._tilelang._msl_transform import _as_metal_target

        lower_target = _as_metal_target(target)

    pass_context = _with_pass_context(pass_configs)
    if pass_context is None:
        return tl_lower(prim_func, target=lower_target)
    with pass_context:
        return tl_lower(prim_func, target=lower_target)


def _shim_lower(
    prim_func: Any,
    target: str,
    *,
    pass_configs: dict[str, Any] | None = None,
) -> Any:
    """Lower via the legacy MSL-string shim. Always targets metal."""

    from cppmega_mlx.nn._tilelang._msl_transform import lower_tilelang_to_msl_inline

    if target != "metal":
        warnings.warn(
            f"_engine_dispatch: shim mode is metal-only; ignoring target={target!r}.",
            UserWarning,
            stacklevel=2,
        )
    return lower_tilelang_to_msl_inline(
        prim_func,
        target="metal",
        pass_configs=pass_configs,
    )


def dispatch_lower(
    prim_func: Any,
    target: str,
    *,
    return_msl: bool = False,
    pass_configs: dict[str, Any] | None = None,
) -> Any:
    """Lower ``prim_func`` for ``target`` per the active engine mode.

    Returns either a ``tilelang.compile`` artifact (engine path; carries
    ``_tilelang_engine_target``) or a :class:`TileLangMSLLowering` instance
    (shim path; carries ``msl_text``). Callers that always need the
    runtime-callable (CompiledArtifact) should set
    ``CPPMEGA_MLX_TILELANG_ENGINE=engine``.

    Phase-3 MSL bridging: if ``return_msl=True`` (or env mode is
    ``"engine_with_msl_extraction"``) the dispatcher routes through the
    engine but extracts a :class:`TileLangMSLLowering`-shaped result via
    :func:`cppmega_mlx.nn._tilelang._msl_extraction.extract_msl_from_engine_artifact`,
    so legacy ``mx.fast.metal_kernel(...)`` callers can adopt the engine
    path without code churn. If extraction fails (target is not metal, or
    the artifact has no ``kernel_source``), falls back to the legacy shim
    with a one-shot warning so callers don't silently lose MSL text.
    """

    mode = tilelang_engine_mode()
    msl_requested = return_msl or mode == "engine_with_msl_extraction"

    if mode == "shim":
        return _shim_lower(prim_func, target, pass_configs=pass_configs)

    if mode == "engine" and not msl_requested:
        return _engine_compile(prim_func, target, pass_configs=pass_configs)

    if msl_requested:
        # engine path with required MSL extraction. On any failure (engine
        # error, non-metal target, no kernel_source), fall back to the shim
        # exactly once with a UserWarning.
        return _engine_with_msl_extraction(
            prim_func,
            target,
            pass_configs=pass_configs,
        )

    # auto: prefer engine, fall back to shim on import failure with a
    # one-shot warning. Other engine errors propagate (see _engine_compile
    # docstring rationale: silently swallowing TVM AttributeErrors and
    # PassContext drift previously masked real bugs).
    try:
        return _engine_compile(prim_func, target, pass_configs=pass_configs)
    except (ImportError, ModuleNotFoundError) as exc:
        global _FALLBACK_WARNED
        if not _FALLBACK_WARNED:
            warnings.warn(
                "cppmega_mlx._tilelang: tilelang engine unavailable "
                f"({exc.__class__.__name__}: {exc}); falling back to MSL shim. "
                f"Set {_FLAG_ENV}=engine to surface engine errors instead, or "
                f"{_FLAG_ENV}=shim to silence this warning.",
                UserWarning,
                stacklevel=2,
            )
            _FALLBACK_WARNED = True
        return _shim_lower(prim_func, target, pass_configs=pass_configs)


def _engine_with_msl_extraction(
    prim_func: Any,
    target: str,
    *,
    pass_configs: dict[str, Any] | None = None,
) -> Any:
    """Engine path that extracts an MSL-shaped lowering from the artifact.

    On any failure (ImportError, non-metal target, no ``kernel_source``,
    parse failure) falls back to ``_shim_lower`` with a one-shot
    ``UserWarning`` so callers see *one* signal that the new path didn't
    work and they're back on the legacy shim.
    """

    global _MSL_EXTRACTION_FALLBACK_WARNED

    from cppmega_mlx.nn._tilelang._msl_extraction import (
        extract_msl_from_engine_artifact,
    )

    try:
        artifact = _engine_lower_for_msl_extraction(
            prim_func,
            target,
            pass_configs=pass_configs,
        )
    except (ImportError, ModuleNotFoundError) as exc:
        if not _MSL_EXTRACTION_FALLBACK_WARNED:
            warnings.warn(
                "cppmega_mlx._tilelang: engine_with_msl_extraction requested "
                f"but tilelang unavailable ({exc.__class__.__name__}: {exc}); "
                "falling back to MSL shim.",
                UserWarning,
                stacklevel=2,
            )
            _MSL_EXTRACTION_FALLBACK_WARNED = True
        return _shim_lower(prim_func, target, pass_configs=pass_configs)

    lowering = extract_msl_from_engine_artifact(artifact, target=target)
    if lowering is None:
        if not _MSL_EXTRACTION_FALLBACK_WARNED:
            warnings.warn(
                "cppmega_mlx._tilelang: engine_with_msl_extraction returned "
                "None (non-metal target or artifact had no kernel_source); "
                "falling back to MSL shim.",
                UserWarning,
                stacklevel=2,
            )
            _MSL_EXTRACTION_FALLBACK_WARNED = True
        return _shim_lower(prim_func, target, pass_configs=pass_configs)
    return lowering


def dispatch_lower_supports_msl_extraction() -> bool:
    """Return True iff the engine_with_msl_extraction path is reachable.

    Thin wrapper over :func:`_msl_extraction.supports_msl_extraction` —
    importable from caller modules without dragging in the whole
    ``_msl_extraction`` namespace.
    """

    try:
        from cppmega_mlx.nn._tilelang._msl_extraction import supports_msl_extraction
    except ImportError:
        return False
    return supports_msl_extraction()


def _reset_fallback_warning_for_tests() -> None:
    """Test hook: re-arm the one-shot fallback warnings (auto + msl-extraction)."""

    global _FALLBACK_WARNED, _MSL_EXTRACTION_FALLBACK_WARNED
    _FALLBACK_WARNED = False
    _MSL_EXTRACTION_FALLBACK_WARNED = False


def artifact_to_source(artifact: Any) -> str:
    """Return rendered kernel source from a ``tilelang.compile`` / engine artifact.

    Works for both engine artifacts (CUDA/HIP/Metal source via
    ``kernel_source`` or ``rt_mod.get_source()``) and shim
    :class:`TileLangMSLLowering` instances (returns ``msl_text``). Phase-3
    callers use this to extract a single source string from whichever artifact
    :func:`dispatch_lower` produced for the active engine mode.
    """

    if hasattr(artifact, "msl_text"):
        return str(artifact.msl_text)
    if hasattr(artifact, "kernel_source"):
        return str(artifact.kernel_source)
    rt_mod = getattr(artifact, "rt_mod", None)
    if rt_mod is not None and hasattr(rt_mod, "get_source"):
        return str(rt_mod.get_source())
    return str(artifact)


__all__ = [
    "compile_native_tilelang_kernel",
    "dispatch_lower",
    "dispatch_lower_supports_msl_extraction",
    "tilelang_engine_mode",
    "artifact_to_source",
    "_reset_fallback_warning_for_tests",
]
