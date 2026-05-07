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
from typing import Any


_VALID_MODES = ("auto", "engine", "shim")
_FLAG_ENV = "CPPMEGA_MLX_TILELANG_ENGINE"
_FALLBACK_WARNED = False


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


def _engine_compile(prim_func: Any, target: str) -> Any:
    """Run ``tilelang.compile`` and stamp the result with the target tag."""

    import tilelang  # noqa: F401  - intentional eager import for ImportError surfacing

    artifact = tilelang.compile(prim_func, target=target, out_idx=None)
    try:
        setattr(artifact, "_tilelang_engine_target", target)
    except (AttributeError, TypeError):
        # Some builds wrap the artifact in a frozen / __slots__ object; preserve
        # the artifact unchanged if we cannot stamp it.
        pass
    return artifact


def _shim_lower(prim_func: Any, target: str) -> Any:
    """Lower via the legacy MSL-string shim. Always targets metal."""

    from cppmega_mlx.nn._tilelang._msl_transform import lower_tilelang_to_msl_inline

    if target != "metal":
        warnings.warn(
            f"_engine_dispatch: shim mode is metal-only; ignoring target={target!r}.",
            UserWarning,
            stacklevel=2,
        )
    return lower_tilelang_to_msl_inline(prim_func, target="metal")


def dispatch_lower(prim_func: Any, target: str) -> Any:
    """Lower ``prim_func`` for ``target`` per the active engine mode.

    Returns either a ``tilelang.compile`` artifact (engine path; carries
    ``_tilelang_engine_target``) or a :class:`TileLangMSLLowering` instance
    (shim path; carries ``msl_text``). Callers that always need the
    runtime-callable (CompiledArtifact) should set
    ``CPPMEGA_MLX_TILELANG_ENGINE=engine``.
    """

    mode = tilelang_engine_mode()
    if mode == "engine":
        return _engine_compile(prim_func, target)
    if mode == "shim":
        return _shim_lower(prim_func, target)
    # auto: prefer engine, fall back to shim on import failure with a
    # one-shot warning. Other engine errors propagate (see _engine_compile
    # docstring rationale: silently swallowing TVM AttributeErrors and
    # PassContext drift previously masked real bugs).
    try:
        return _engine_compile(prim_func, target)
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
        return _shim_lower(prim_func, target)


def _reset_fallback_warning_for_tests() -> None:
    """Test hook: re-arm the one-shot fallback warning."""

    global _FALLBACK_WARNED
    _FALLBACK_WARNED = False


__all__ = [
    "dispatch_lower",
    "tilelang_engine_mode",
    "_reset_fallback_warning_for_tests",
]
