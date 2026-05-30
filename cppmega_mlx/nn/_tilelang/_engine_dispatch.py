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
    # RULE #1 (no automated/silent fallbacks): if the Metal FP8 intrinsics are
    # not registered, a Path C FP8 kernel would compile/run against MISSING
    # intrinsics and silently produce a wrong result. The only acceptable
    # silent skip is the genuine "feature not present in this tilelang build"
    # case (the assert helper itself does not exist) — that is a
    # pre-existing-feature-not-available check, not an on-failure fallback.
    try:
        from tilelang.language.fp8_op import assert_metal_fp8_intrinsics_registered
    except (ImportError, ModuleNotFoundError):
        # Older tilelang without the FP8 intrinsics module — nothing to assert.
        return
    # A failure of the assert itself means the intrinsics are genuinely NOT
    # registered: surface it instead of swallowing and running a broken kernel.
    assert_metal_fp8_intrinsics_registered()


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
    try:
        if pass_context is None:
            artifact = tilelang.compile(prim_func, target=compile_target, out_idx=None)
        else:
            with pass_context:
                artifact = tilelang.compile(prim_func, target=compile_target, out_idx=None)
    except ValueError as exc:
        message = str(exc)
        if "Cannot find global function target.build.tilelang_" in message:
            raise RuntimeError(
                f"TileLang backend for target {target!r} is unavailable: {message}"
            ) from exc
        raise
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
        # RULE #1: the engine lowering SUCCEEDED (we are past the ImportError
        # guard) but MSL extraction returned None. That is NOT "engine absent" —
        # it is extraction silently failing (non-metal target, or a metal
        # artifact with no kernel_source). Falling back to the legacy MSL-string
        # shim here is exactly the forbidden "MSL-instead-of-tvm-ffi" silent
        # fallback: it papers over a real extraction bug with a different
        # (legacy) lowering. Raise with where+what; if the caller genuinely
        # wants the shim they must route explicitly via CPPMEGA_MLX_TILELANG_ENGINE=shim.
        raise RuntimeError(
            f"_engine_with_msl_extraction: TileLang engine lowering for "
            f"target={target!r} succeeded but "
            f"extract_msl_from_engine_artifact returned None (non-metal target, "
            f"or the engine artifact exposed no kernel_source). Refusing to "
            f"silently fall back to the legacy MSL shim (RULE #1) — this points "
            f"at a real MSL-extraction bug for this artifact. Set "
            f"{_FLAG_ENV}=shim to explicitly use the legacy MSL lowering."
        )
    return lowering


# ---------------------------------------------------------------------------
# V7-N01 / V7-N02: fusion-emitter registry.
#
# FX patterns like `gemm_softmax` and `qk_reduce_sm_scale` need a path from
# pattern-match → dedicated kernel factory. `dispatch_lower` is the right
# integration point because it already chooses between shim and engine.
# Callers register concrete emitters via :func:`register_fusion_emitter`;
# the FX backend in cppmega_mlx.runtime.torch_compile_backend looks them up
# before falling through to the generic lowering path.
# ---------------------------------------------------------------------------


_FUSION_EMITTERS: dict[str, Callable[..., Any]] = {}


def register_fusion_emitter(pattern_name: str,
                              emitter: Callable[..., Any]) -> None:
    """Register a kernel factory for a named fusion pattern.

    `emitter` is invoked with arbitrary keyword shape/dtype kwargs and
    must return a TileLang ``@T.prim_func``. The actual emit happens
    inside :func:`emit_fusion_kernel`.
    """
    _FUSION_EMITTERS[pattern_name] = emitter


def fusion_emitters_available() -> tuple[str, ...]:
    """Return the names of every registered fusion emitter."""
    return tuple(sorted(_FUSION_EMITTERS))


def emit_fusion_kernel(pattern_name: str, **kwargs: Any) -> Any:
    """Call the registered emitter for ``pattern_name`` and lower it.

    Returns whatever :func:`dispatch_lower` returns for the emitted
    PrimFunc (shim path: ``TileLangMSLLowering``; engine path:
    compile artifact). Raises ``KeyError`` if the pattern has no
    registered emitter — callers should fall through to the generic
    lowering in that case.
    """
    if pattern_name not in _FUSION_EMITTERS:
        raise KeyError(
            f"no fusion emitter registered for {pattern_name!r}; "
            f"available: {fusion_emitters_available()}")
    prim = _FUSION_EMITTERS[pattern_name](**kwargs)
    target = kwargs.pop("_target", "cuda")
    return dispatch_lower(prim, target)


def _register_default_fusion_emitters() -> None:
    """Wire the kernel factories that ship in-tree (N01 gemm_softmax,
    N02 sparse-MLA qk_reduce). Called eagerly on import so callers can
    rely on ``fusion_emitters_available()`` reflecting reality."""
    # N02: the path-C qk_reduce factory exists in-tree as
    # make_fp8_sparse_mla_indexed_qk_reduce_kernel. Wrap it so the
    # registry's call contract matches (kwargs → prim_func).
    try:
        from cppmega_mlx.nn._tilelang.sparse_mla_fp8_path_c import (
            make_fp8_sparse_mla_indexed_qk_reduce_kernel,
        )
        _FUSION_EMITTERS.setdefault(
            "qk_reduce_sm_scale",
            lambda **kw: make_fp8_sparse_mla_indexed_qk_reduce_kernel(**kw),
        )
    except Exception:
        # Importing the path-C module pulls heavy TileLang DSL; missing
        # it just means the emitter is unregistered. Callers fall back
        # to the generic lowering path.
        pass

    # N01: gemm_softmax — a minimal TileLang PrimFunc that fuses
    # `softmax(q @ k.T)` into one kernel. The matcher upstream already
    # detected the pattern; this entry exposes it as a callable factory.
    # We register a thin closure that builds the PrimFunc on demand so
    # importing this module stays cheap even when tilelang is absent.
    def _gemm_softmax_factory(*, M: int, N: int, K: int,
                                in_dtype: str = "float16",
                                out_dtype: str = "float16", **_):
        try:
            import tilelang.language as T
        except Exception as exc:
            raise NotImplementedError(
                f"gemm_softmax emitter requires tilelang; got "
                f"{exc.__class__.__name__}: {exc}") from exc

        @T.prim_func
        def gemm_softmax(Q: T.Tensor((M, K), in_dtype),
                          K_buf: T.Tensor((N, K), in_dtype),
                          Out: T.Tensor((M, N), out_dtype)):
            # Fused: Out = softmax(Q @ K^T) along axis=-1.
            with T.Kernel(M) as i:
                row = T.alloc_fragment((N,), "float32")
                for j in T.Parallel(N):
                    acc = T.cast(0.0, "float32")
                    for k in T.serial(K):
                        acc += T.cast(Q[i, k], "float32") * T.cast(
                            K_buf[j, k], "float32")
                    row[j] = acc
                # online softmax (max-stable).
                m = T.alloc_fragment((1,), "float32")
                m[0] = T.cast(-1e30, "float32")
                for j in T.serial(N):
                    m[0] = T.max(m[0], row[j])
                s = T.alloc_fragment((1,), "float32")
                s[0] = T.cast(0.0, "float32")
                for j in T.serial(N):
                    row[j] = T.exp(row[j] - m[0])
                    s[0] += row[j]
                for j in T.Parallel(N):
                    Out[i, j] = T.cast(row[j] / s[0], out_dtype)

        return gemm_softmax

    _FUSION_EMITTERS.setdefault("gemm_softmax", _gemm_softmax_factory)


_register_default_fusion_emitters()


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
