# pyright: reportMissingImports=false
"""Phase-3 migration adapter: extract a :class:`TileLangMSLLowering`-shaped
object from a unified ``tilelang.compile`` engine artifact.

Background
----------
The legacy lowering shim
:func:`cppmega_mlx.nn._tilelang._msl_transform.lower_tilelang_to_msl_inline`
returns a :class:`TileLangMSLLowering` dataclass (text strings: ``header``,
``body``, ``msl_text``, ``buffer_param_names``, ``kernel_name``, plus
``grid``/``threadgroup`` ints) that is consumed by
``mx.fast.metal_kernel(...)`` in ~17 callers. Phase-1 added
:func:`cppmega_mlx.nn._tilelang._engine_dispatch.dispatch_lower` which
returns *runtime callable* artifacts from ``tilelang.compile``. The 17
callers want **MSL text**.

This module bridges them: it consumes a ``tilelang.compile`` artifact and
parses out the same shape that the legacy shim emitted, so callers that
flip to the engine path keep working unchanged. The extraction reuses the
``_msl_transform`` text helpers (``_split_kernel_msl``,
``_inline_tilelang_kernel_body``, ``_parse_buffer_param_names``,
``_KERNEL_DEF_RE``) so the inlining + parsing rules stay in one place.

Metal-only by design: for CUDA / HIP targets ``extract_msl_from_engine_artifact``
returns ``None`` (the artifact is the runtime callable, no MSL text is
emitted). Callers that *only* want MSL text should keep using ``target="metal"``.

This file is intentionally side-effect-free at import: it never preloads
libz3, never imports tilelang. All those happen lazily inside the engine
path (``_engine_dispatch._engine_compile``).
"""

from __future__ import annotations

import warnings
from typing import Any


def _is_metal_target(target: Any) -> bool:
    """Best-effort metal detection across str / tvm.target.Target / unknown."""

    if target is None:
        return False
    if isinstance(target, str):
        return "metal" in target.lower()
    # tvm.target.Target has a ``kind.name`` attribute. Stringify defensively.
    kind = getattr(target, "kind", None)
    name = getattr(kind, "name", None)
    if isinstance(name, str):
        return "metal" in name.lower()
    return "metal" in str(target).lower()


def _artifact_kernel_source(artifact: Any) -> str | None:
    """Return the rendered MSL text from an engine artifact, or None.

    Tries (in order) ``artifact.kernel_source`` (TileLang JIT artifact),
    ``artifact.rt_mod.get_source()``, ``artifact.rt_mod.get_source("metal")``.
    Returns None if none of these yields a non-empty string.
    """

    src = getattr(artifact, "kernel_source", None)
    if src:
        return str(src)
    rt_mod = getattr(artifact, "rt_mod", None)
    if rt_mod is not None:
        get_source = getattr(rt_mod, "get_source", None)
        if callable(get_source):
            try:
                src = get_source()
            except TypeError:
                # Some TVM builds require a fmt arg.
                try:
                    src = get_source("metal")
                except Exception:  # pragma: no cover - defensive
                    src = None
            except Exception:  # pragma: no cover - defensive
                src = None
            if src:
                return str(src)
    return None


def _artifact_grid_threadgroup(artifact: Any) -> tuple[
    tuple[int, int, int], tuple[int, int, int]
]:
    """Return ``(grid, threadgroup)`` from the engine artifact's device_mod.

    Mirrors the parsing in ``_msl_transform.lower_tilelang_to_msl_inline``;
    falls back to ``(1,1,1)`` axes that aren't annotated.
    """

    grid = [1, 1, 1]
    block = [1, 1, 1]
    device_mod = getattr(artifact, "device_mod", None)
    if device_mod is None:
        return (1, 1, 1), (1, 1, 1)
    try:
        functions = device_mod.functions
    except AttributeError:
        return (1, 1, 1), (1, 1, 1)
    for _, func in functions.items():
        thread_extent = func.attrs.get("thread_extent")
        if thread_extent is None:
            continue
        for tag, extent in thread_extent.items():
            tag_str = str(tag)
            if "threadIdx" in tag_str:
                idx = "xyz".index(tag_str[-1])
                block[idx] = int(extent)
            elif "blockIdx" in tag_str:
                idx = "xyz".index(tag_str[-1])
                grid[idx] = int(extent)
        break
    return (grid[0], grid[1], grid[2]), (block[0], block[1], block[2])


def extract_msl_from_engine_artifact(
    artifact: Any, *, target: Any = "metal"
) -> Any | None:
    """Return a :class:`TileLangMSLLowering` from an engine artifact, or None.

    For non-metal targets returns ``None`` (engine artifacts on CUDA/HIP do
    not have MSL text; callers should use the runtime-callable path
    directly). For metal artifacts returns a frozen
    :class:`TileLangMSLLowering` whose fields match the legacy shim
    (``header``, ``body``, ``msl_text``, ``grid``, ``threadgroup``,
    ``buffer_param_names``, ``kernel_name``).

    On extraction failure (e.g. artifact has no ``kernel_source`` and no
    ``rt_mod.get_source()``), returns ``None`` with a one-shot
    ``UserWarning`` so callers can fall back to the shim. We deliberately
    avoid raising — this is a graceful-degradation adapter.
    """

    if not _is_metal_target(target):
        return None
    msl_text = _artifact_kernel_source(artifact)
    if not msl_text:
        warnings.warn(
            "extract_msl_from_engine_artifact: artifact has no kernel_source "
            "and no rt_mod.get_source(); cannot produce MSL text. Caller "
            "should fall back to lower_tilelang_to_msl_inline.",
            UserWarning,
            stacklevel=2,
        )
        return None
    # Lazy import: keep this module side-effect-free at import time.
    from cppmega_mlx.nn._tilelang._msl_transform import (  # type: ignore[import-not-found]
        TileLangMSLLowering,
        _KERNEL_DEF_RE,
        _inline_tilelang_kernel_body,
        _parse_buffer_param_names,
        _split_kernel_msl,
    )

    try:
        prelude, sig_text, body_text = _split_kernel_msl(msl_text)
    except RuntimeError as exc:
        warnings.warn(
            f"extract_msl_from_engine_artifact: split failed ({exc}); "
            "caller should fall back to the shim.",
            UserWarning,
            stacklevel=2,
        )
        return None
    inner = body_text[1:-1]
    body = _inline_tilelang_kernel_body(inner)
    grid, threadgroup = _artifact_grid_threadgroup(artifact)
    name_match = _KERNEL_DEF_RE.search(msl_text)
    kernel_name = name_match.group("name") if name_match else "kernel_main"
    return TileLangMSLLowering(
        header=prelude,
        body=body,
        grid=grid,
        threadgroup=threadgroup,
        msl_text=msl_text,
        buffer_param_names=_parse_buffer_param_names(sig_text),
        kernel_name=kernel_name,
    )


def supports_msl_extraction() -> bool:
    """Return True if a tilelang artifact can be probed for MSL text on this host.

    Cheap probe: import-checks tilelang and that ``compile`` exists. Does NOT
    actually compile a kernel — just guarantees the engine path is reachable.
    Returns False (silently) on import failure so callers can pre-decide
    whether to flip to the engine-with-MSL path.
    """

    try:
        import tilelang  # noqa: F401
    except (ImportError, ModuleNotFoundError):
        return False
    return hasattr(tilelang, "compile")


__all__ = [
    "extract_msl_from_engine_artifact",
    "supports_msl_extraction",
]
