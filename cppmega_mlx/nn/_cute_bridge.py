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

2. **External-CuTeDSL → TileLang import** (pattern-recognized subset).
   PR #1421 does not implement general IR import from ``@cute.kernel``
   functions — it only emits CuTeDSL from TileLang IR.  For cppmega's
   smallest smoke kernels, ``SingleGemmWGMMA``, ``MaskedLKQApplyWGMMA``,
   one-chunk ``StateApplyConsumersWGMMA``, fixed-nchunk
   ``MultiChunkStateApplyConsumersWGMMA``, Phase-4 ``FusedBwdBwdP4``,
   and FA4's
   ``FA4PatternFused3Gemm`` / ``FA4PatternFused3GemmV2`` chains, we can
   recover the tile contract and
   re-express it as TileLang ``T`` ops.  More complex kernels still require
   either:

       (a) re-expressing the kernel in TileLang ``T`` ops, or
       (b) extending TileLang with a CuTeDSL frontend.

This module exposes :func:`cute_dsl_to_tilelang_prim` for that narrow import
case and fails loudly for everything else. It also exposes
:func:`compile_prim_to_cutedsl` for the supported emit direction — the test
harness exercises both paths.

See ``tests/test_cute_to_tilelang_bridge.py`` for an end-to-end smoke.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

__all__ = [
    "CUTE_BRIDGE_KNOWN_GAPS",
    "TILELANG_CUTEDSL_ENTRY",
    "CuteBridgeUnsupported",
    "compile_prim_to_cutedsl",
    "cute_dsl_source_to_tilelang_prim",
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
    "it does not generally consume hand-written @cute.kernel modules. cppmega's "
    "SingleGemmWGMMA, MaskedLKQApplyWGMMA, one-chunk "
    "StateApplyConsumersWGMMA, fixed-nchunk "
    "MultiChunkStateApplyConsumersWGMMA, Phase-4 FusedBwdBwdP4, and "
    "FA4PatternFused3Gemm / FA4PatternFused3GemmV2 can be "
    "pattern-imported by re-expressing the recovered contracts in TileLang "
    "T-ops; complex cute_dsl_mimo kernels outside this recovered contract "
    "set still need dedicated TileLang rewrites or a real CuTeDSL frontend.",
    # Hopper-specific feature gaps — even for the supported direction.
    "TileLang's CuTeDSL emitter (sm_90 path) does not expose all WGMMA / TMA "
    "knobs that cppmega's hand-written kernels rely on (e.g. quack.sm90_utils, "
    "warpgroup.OperandSource selection, custom SmemAllocator layouts, "
    "StMatrix epilogues). The FusedBwdBwdP4 importer preserves the public "
    "10-GEMM dataflow contract, but not those Hopper-specific scheduling "
    "optimizations.",
    # Environment gating.
    "tilelang.compile(target='cutedsl') requires nvidia-cutlass-dsl>=4.3.1 "
    "(excluding 4.3.4) AND a CUDA host. Mac/MLX hosts cannot exercise this "
    "path; tests must skip explicitly with a reason string.",
)


class CuteBridgeUnsupported(RuntimeError):
    """Raised when an external CuTeDSL kernel has no TileLang importer."""


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


def cute_dsl_to_tilelang_prim(cute_kernel: Any) -> Any:
    """Import a recognized external CuTeDSL kernel as TileLang IR.

    The supported reverse-direction subset is intentionally small.  cppmega's
    ``SingleGemmWGMMA`` is re-expressed as a single-CTA TileLang ``T.gemm``
    PrimFunc with the same public contract, ``MaskedLKQApplyWGMMA`` is
    re-expressed as two TileLang GEMMs plus an in-IR future-mask spill, and
    one-chunk ``StateApplyConsumersWGMMA`` is re-expressed as three GEMMs plus
    the DV/DMIMO_V consumers. The fixed-nchunk multi-chunk state/apply importer
    and FA4 3-GEMM chains are source-only because their useful contracts are
    easier to recover from source without importing ``cutlass`` / ``quack``.
    Unsupported kernels still raise
    :class:`CuteBridgeUnsupported` with a precise reason rather than returning
    ``None`` or silently falling back to a launch boundary.

    To dispatch an existing TileLang ``T.prim_func`` through the CuTeDSL
    backend (the direction PR #1421 natively supports), call
    :func:`compile_prim_to_cutedsl` or ``dispatch_lower(prim, target='cutedsl')``
    from ``cppmega_mlx.nn._tilelang._engine_dispatch``.
    """

    name = _cute_kernel_class_name(cute_kernel)
    if name == "SingleGemmWGMMA":
        m = _positive_int_attr(cute_kernel, "M")
        n = _positive_int_attr(cute_kernel, "N")
        k = _positive_int_attr(cute_kernel, "K")
        dtype = _normalize_cute_dtype(getattr(cute_kernel, "dtype", "bfloat16"))
        return _build_single_gemm_wgmma_tilelang_prim(m, n, k, dtype)
    if name == "MaskedLKQApplyWGMMA":
        dim = _positive_int_attr(cute_kernel, "dim")
        rank = _positive_int_attr(cute_kernel, "rank")
        dtype = _normalize_cute_dtype(getattr(cute_kernel, "dtype", "bfloat16"))
        return _build_masked_lkq_apply_tilelang_prim(dim, rank, dtype)
    if name == "StateApplyConsumersWGMMA":
        dim = _positive_int_attr(cute_kernel, "dim")
        rank = _positive_int_attr(cute_kernel, "rank")
        chunk_size = _positive_int_attr(cute_kernel, "chunk_size")
        dtype = _normalize_cute_dtype(getattr(cute_kernel, "dtype", "bfloat16"))
        return _build_state_apply_consumers_tilelang_prim(
            dim,
            rank,
            chunk_size,
            dtype,
        )

    raise CuteBridgeUnsupported(
        "TileLang's CuTeDSL bridge (PR #1421) emits CuTeDSL from TileLang IR "
        "and cppmega_mlx currently imports only the external SingleGemmWGMMA, "
        "MaskedLKQApplyWGMMA, one-chunk StateApplyConsumersWGMMA, and "
        "source-level fixed-nchunk MultiChunkStateApplyConsumersWGMMA / "
        "FA4PatternFused3Gemm / FA4PatternFused3GemmV2 patterns as TileLang "
        "T-ops. Unsupported "
        f"@cute.kernel object {name or type(cute_kernel).__name__!r} needs a "
        "dedicated TileLang T-op rewrite or a real CuTeDSL frontend. To route "
        "an existing TileLang prim_func through the CuTeDSL backend instead, "
        "use cppmega_mlx.nn._cute_bridge.compile_prim_to_cutedsl(prim) or "
        "dispatch_lower(prim, target='cutedsl')."
    )


def cute_dsl_source_to_tilelang_prim(
    source_path: str | Path,
    *,
    class_name: str | None = None,
    nchunks: int | None = None,
) -> Any:
    """Statically import a recognized external CuTeDSL source file.

    This path intentionally avoids importing the external Python module, so it
    works on Mac/MLX hosts where ``cutlass`` and ``quack`` are not installed.
    Supported source patterns are cppmega's ``SingleGemmWGMMA``,
    ``MaskedLKQApplyWGMMA``, one-chunk ``StateApplyConsumersWGMMA``,
    fixed-nchunk ``MultiChunkStateApplyConsumersWGMMA``, Phase-4
    ``FusedBwdBwdP4``, and
    ``FA4PatternFused3Gemm`` / ``FA4PatternFused3GemmV2`` classes with
    ``@cute.kernel`` methods and
    literal constructor defaults for their tile-contract dimensions.
    """

    path = Path(source_path)
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except OSError as exc:
        raise CuteBridgeUnsupported(
            f"cannot read CuTeDSL source file {path}: {exc}"
        ) from exc
    except SyntaxError as exc:
        raise CuteBridgeUnsupported(
            f"cannot parse CuTeDSL source file {path}: {exc}"
        ) from exc

    cls = _find_cute_kernel_class(tree, class_name=class_name)
    if cls.name == "SingleGemmWGMMA":
        m, n, k, dtype = _single_gemm_defaults_from_class(cls)
        return _build_single_gemm_wgmma_tilelang_prim(m, n, k, dtype)
    if cls.name == "MaskedLKQApplyWGMMA":
        dim, rank, dtype = _masked_lkq_defaults_from_class(cls)
        return _build_masked_lkq_apply_tilelang_prim(dim, rank, dtype)
    if cls.name == "StateApplyConsumersWGMMA":
        dim, rank, chunk_size, dtype = _state_apply_consumers_defaults_from_class(cls)
        return _build_state_apply_consumers_tilelang_prim(
            dim,
            rank,
            chunk_size,
            dtype,
        )
    if cls.name == "MultiChunkStateApplyConsumersWGMMA":
        if nchunks is None:
            raise CuteBridgeUnsupported(
                "MultiChunkStateApplyConsumersWGMMA source import requires "
                "explicit nchunks because the CuTe kernel specializes it from "
                "the runtime tensor shape."
            )
        nchunks_value = _positive_int_value(nchunks, "nchunks")
        if nchunks_value not in (2, 4, 8):
            raise CuteBridgeUnsupported(
                "MultiChunkStateApplyConsumersWGMMA source import supports "
                f"nchunks in (2, 4, 8); got {nchunks_value}."
            )
        base_cls = _find_class_by_name(tree, "StateApplyConsumersWGMMA")
        dim, rank, chunk_size, dtype = _state_apply_consumers_defaults_from_class(
            base_cls
        )
        return _build_multi_chunk_state_apply_consumers_tilelang_prim(
            dim,
            rank,
            chunk_size,
            dtype,
            nchunks_value,
        )
    if cls.name == "FusedBwdBwdP4":
        if nchunks is None:
            raise CuteBridgeUnsupported(
                "FusedBwdBwdP4 source import requires explicit nchunks "
                "because the CuTe kernel specializes it from the runtime "
                "tensor shape."
            )
        nchunks_value = _positive_int_value(nchunks, "nchunks")
        if nchunks_value not in (2, 4, 8):
            raise CuteBridgeUnsupported(
                "FusedBwdBwdP4 source import supports nchunks in (2, 4, 8); "
                f"got {nchunks_value}."
            )
        n, p, r, chunk_size, dtype = _fused_bwd_bwd_p4_defaults_from_class(cls)
        return _build_fused_bwd_bwd_p4_tilelang_prim(
            n,
            p,
            r,
            chunk_size,
            dtype,
            nchunks_value,
        )
    if cls.name == "FA4PatternFused3Gemm":
        dim, dtype = _fa4_fused3_gemm_defaults_from_class(cls)
        return _build_fa4_v1_fused3_gemm_tilelang_prim(dim, dtype)
    if cls.name == "FA4PatternFused3GemmV2":
        dim, dtype = _fa4_fused3_gemm_defaults_from_class(cls)
        return _build_fa4_v2_fused3_gemm_tilelang_prim(dim, dtype)
    raise CuteBridgeUnsupported(
        "source-level CuTeDSL import currently supports only SingleGemmWGMMA, "
        "MaskedLKQApplyWGMMA, one-chunk StateApplyConsumersWGMMA, and "
        "fixed-nchunk MultiChunkStateApplyConsumersWGMMA plus "
        "FusedBwdBwdP4 and FA4PatternFused3Gemm / FA4PatternFused3GemmV2; "
        f"got {cls.name!r} from {path}."
    )


def _cute_kernel_class_name(cute_kernel: Any) -> str:
    """Return the stable class/function name for a CuTeDSL kernel object."""

    direct = getattr(cute_kernel, "__name__", None)
    if isinstance(direct, str) and direct:
        return direct
    cls = getattr(cute_kernel, "__class__", None)
    name = getattr(cls, "__name__", "")
    return name if isinstance(name, str) else ""


def _find_cute_kernel_class(
    tree: ast.Module,
    *,
    class_name: str | None,
) -> ast.ClassDef:
    """Find the requested or sole CuTeDSL class in a source AST."""

    classes = [node for node in tree.body if isinstance(node, ast.ClassDef)]
    if class_name is not None:
        matches = [node for node in classes if node.name == class_name]
        if not matches:
            raise CuteBridgeUnsupported(
                f"CuTeDSL source does not define class {class_name!r}."
            )
        candidate = matches[0]
        if not _class_has_cute_entrypoint(candidate):
            raise CuteBridgeUnsupported(
                f"CuTeDSL source class {class_name!r} has no @cute.kernel or @cute.jit method."
            )
        return candidate

    named = [node for node in classes if node.name == "SingleGemmWGMMA"]
    if named and _class_has_cute_entrypoint(named[0]):
        return named[0]

    cute_classes = [node for node in classes if _class_has_cute_entrypoint(node)]
    if len(cute_classes) == 1:
        return cute_classes[0]
    if not cute_classes:
        raise CuteBridgeUnsupported(
            "CuTeDSL source does not define a class with a @cute.kernel or @cute.jit method."
        )
    raise CuteBridgeUnsupported(
        "CuTeDSL source defines multiple @cute.kernel/@cute.jit classes; pass class_name."
    )


def _find_class_by_name(tree: ast.Module, class_name: str) -> ast.ClassDef:
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return node
    raise CuteBridgeUnsupported(f"CuTeDSL source does not define class {class_name!r}.")


def _class_has_cute_entrypoint(cls: ast.ClassDef) -> bool:
    for item in cls.body:
        if not isinstance(item, ast.FunctionDef):
            continue
        if any(
            _is_cute_decorator(decorator, "kernel")
            or _is_cute_decorator(decorator, "jit")
            for decorator in item.decorator_list
        ):
            return True
    return False


def _is_cute_decorator(node: ast.expr, attr: str) -> bool:
    target = node.func if isinstance(node, ast.Call) else node
    return (
        isinstance(target, ast.Attribute)
        and isinstance(target.value, ast.Name)
        and target.value.id == "cute"
        and target.attr == attr
    )


def _single_gemm_defaults_from_class(cls: ast.ClassDef) -> tuple[int, int, int, str]:
    init = next(
        (
            item
            for item in cls.body
            if isinstance(item, ast.FunctionDef) and item.name == "__init__"
        ),
        None,
    )
    if init is None:
        raise CuteBridgeUnsupported("SingleGemmWGMMA source has no __init__ method.")

    defaults_by_name = _function_defaults_by_name(init)
    try:
        m = _literal_positive_int(defaults_by_name["M"], "M")
        n = _literal_positive_int(defaults_by_name["N"], "N")
        k = _literal_positive_int(defaults_by_name["K"], "K")
    except KeyError as exc:
        raise CuteBridgeUnsupported(
            "SingleGemmWGMMA source __init__ must default M, N, and K."
        ) from exc
    dtype_node = defaults_by_name.get("dtype")
    dtype = _normalize_cute_dtype(
        _ast_default_to_name(dtype_node or ast.Constant("bfloat16"))
    )
    return m, n, k, dtype


def _masked_lkq_defaults_from_class(cls: ast.ClassDef) -> tuple[int, int, str]:
    init = next(
        (
            item
            for item in cls.body
            if isinstance(item, ast.FunctionDef) and item.name == "__init__"
        ),
        None,
    )
    if init is None:
        raise CuteBridgeUnsupported("MaskedLKQApplyWGMMA source has no __init__ method.")

    defaults_by_name = _function_defaults_by_name(init)
    try:
        dim = _literal_positive_int(defaults_by_name["dim"], "dim")
        rank = _literal_positive_int(defaults_by_name["rank"], "rank")
    except KeyError as exc:
        raise CuteBridgeUnsupported(
            "MaskedLKQApplyWGMMA source __init__ must default dim and rank."
        ) from exc
    dtype_node = defaults_by_name.get("dtype")
    dtype = _normalize_cute_dtype(
        _ast_default_to_name(dtype_node or ast.Constant("bfloat16"))
    )
    return dim, rank, dtype


def _state_apply_consumers_defaults_from_class(
    cls: ast.ClassDef,
) -> tuple[int, int, int, str]:
    init = next(
        (
            item
            for item in cls.body
            if isinstance(item, ast.FunctionDef) and item.name == "__init__"
        ),
        None,
    )
    if init is None:
        raise CuteBridgeUnsupported(
            "StateApplyConsumersWGMMA source has no __init__ method."
        )

    defaults_by_name = _function_defaults_by_name(init)
    try:
        dim = _literal_positive_int(defaults_by_name["dim"], "dim")
        rank = _literal_positive_int(defaults_by_name["rank"], "rank")
        chunk_size = _literal_positive_int(
            defaults_by_name["chunk_size"],
            "chunk_size",
        )
    except KeyError as exc:
        raise CuteBridgeUnsupported(
            "StateApplyConsumersWGMMA source __init__ must default dim, "
            "rank, and chunk_size."
        ) from exc
    dtype_node = defaults_by_name.get("dtype")
    dtype = _normalize_cute_dtype(
        _ast_default_to_name(dtype_node or ast.Constant("bfloat16"))
    )
    return dim, rank, chunk_size, dtype


def _fa4_fused3_gemm_defaults_from_class(cls: ast.ClassDef) -> tuple[int, str]:
    init = next(
        (
            item
            for item in cls.body
            if isinstance(item, ast.FunctionDef) and item.name == "__init__"
        ),
        None,
    )
    if init is None:
        raise CuteBridgeUnsupported(
            f"{cls.name} source has no __init__ method."
        )

    defaults_by_name = _function_defaults_by_name(init)
    try:
        dim = _literal_positive_int(defaults_by_name["DIM"], "DIM")
    except KeyError as exc:
        raise CuteBridgeUnsupported(
            f"{cls.name} source __init__ must default DIM."
        ) from exc
    dtype_node = defaults_by_name.get("dtype")
    dtype = _normalize_cute_dtype(
        _ast_default_to_name(dtype_node or ast.Constant("bfloat16"))
    )
    return dim, dtype


def _fused_bwd_bwd_p4_defaults_from_class(
    cls: ast.ClassDef,
) -> tuple[int, int, int, int, str]:
    init = next(
        (
            item
            for item in cls.body
            if isinstance(item, ast.FunctionDef) and item.name == "__init__"
        ),
        None,
    )
    if init is None:
        raise CuteBridgeUnsupported("FusedBwdBwdP4 source has no __init__ method.")

    defaults_by_name = _function_defaults_by_name(init)
    try:
        n = _literal_positive_int(defaults_by_name["N"], "N")
        p = _literal_positive_int(defaults_by_name["P"], "P")
        r = _literal_positive_int(defaults_by_name["R"], "R")
        chunk_size = _literal_positive_int(
            defaults_by_name["chunk_size"],
            "chunk_size",
        )
    except KeyError as exc:
        raise CuteBridgeUnsupported(
            "FusedBwdBwdP4 source __init__ must default N, P, R, "
            "and chunk_size."
        ) from exc
    dtype_node = defaults_by_name.get("dtype")
    dtype = _normalize_cute_dtype(
        _ast_default_to_name(dtype_node or ast.Constant("bfloat16"))
    )
    return n, p, r, chunk_size, dtype


def _fa4_v2_defaults_from_class(cls: ast.ClassDef) -> tuple[int, str]:
    return _fa4_fused3_gemm_defaults_from_class(cls)


def _function_defaults_by_name(fn: ast.FunctionDef) -> dict[str, ast.expr]:
    args = fn.args.args
    defaults = fn.args.defaults
    if not defaults:
        return {}
    defaulted_args = args[-len(defaults):]
    return {arg.arg: default for arg, default in zip(defaulted_args, defaults)}


def _literal_positive_int(node: ast.expr, name: str) -> int:
    if not isinstance(node, ast.Constant) or isinstance(node.value, bool):
        raise CuteBridgeUnsupported(
            f"CuTeDSL source default {name!r} must be an integer literal."
        )
    try:
        value = int(node.value)
    except (TypeError, ValueError) as exc:
        raise CuteBridgeUnsupported(
            f"CuTeDSL source default {name!r} must be an integer literal."
        ) from exc
    if value <= 0:
        raise CuteBridgeUnsupported(
            f"CuTeDSL source default {name!r} must be positive."
        )
    return value


def _ast_default_to_name(node: ast.expr) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parts = [node.attr]
        base = node.value
        while isinstance(base, ast.Attribute):
            parts.append(base.attr)
            base = base.value
        if isinstance(base, ast.Name):
            parts.append(base.id)
        return ".".join(reversed(parts))
    if isinstance(node, ast.Constant):
        return str(node.value)
    raise CuteBridgeUnsupported(
        "CuTeDSL source dtype default must be a simple name or literal."
    )


def _positive_int_attr(cute_kernel: Any, attr: str) -> int:
    """Read a positive integer kernel dimension from an external kernel."""

    if not hasattr(cute_kernel, attr):
        raise CuteBridgeUnsupported(
            f"CuTeDSL import requires integer attribute {attr!r}."
        )
    return _positive_int_value(getattr(cute_kernel, attr), attr)


def _positive_int_value(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise CuteBridgeUnsupported(
            f"CuTeDSL value {name!r} must be a positive integer."
        )
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise CuteBridgeUnsupported(
            f"CuTeDSL value {name!r} must be a positive integer; "
            f"got {value!r}."
        ) from exc
    if result <= 0:
        raise CuteBridgeUnsupported(
            f"CuTeDSL value {name!r} must be positive; got {result}."
        )
    return result


def _normalize_cute_dtype(dtype: Any) -> str:
    """Map common CuTe/CUTLASS dtype spellings to TileLang dtype strings."""

    spellings = [
        getattr(dtype, "__name__", ""),
        getattr(dtype, "name", ""),
        type(dtype).__name__,
        str(dtype),
    ]
    joined = " ".join(str(item).lower() for item in spellings if item)
    if "bfloat16" in joined or "bf16" in joined:
        return "bfloat16"
    if "float16" in joined or "f16" in joined or "half" in joined:
        return "float16"
    raise CuteBridgeUnsupported(
        "CuTeDSL import supports bfloat16 and float16 inputs; "
        f"got dtype={dtype!r}."
    )


def _build_single_gemm_wgmma_tilelang_prim(
    m: int,
    n: int,
    k: int,
    dtype: str,
) -> Any:
    """Build TileLang IR for cppmega's ``SingleGemmWGMMA`` contract."""

    try:
        import tilelang.language as T
    except Exception as exc:  # pragma: no cover - host/environment dependent
        raise CuteBridgeUnsupported(
            "SingleGemmWGMMA import requires a working tilelang import; "
            f"got {exc.__class__.__name__}: {exc}."
        ) from exc

    threads = 128

    @T.prim_func
    def single_gemm_wgmma_imported(
        A: T.Tensor((m, k), dtype),
        B: T.Tensor((n, k), dtype),
        C: T.Tensor((m, n), dtype),
    ):
        with T.Kernel(1, 1, threads=threads) as (_bx, _by):
            A_sh = T.alloc_shared((m, k), dtype)
            B_sh = T.alloc_shared((n, k), dtype)
            C_loc = T.alloc_fragment((m, n), "float32")
            T.clear(C_loc)
            T.copy(A[0, 0], A_sh)
            T.copy(B[0, 0], B_sh)
            T.gemm(A_sh, B_sh, C_loc, False, True)
            T.copy(C_loc, C[0, 0])

    return single_gemm_wgmma_imported


def _build_masked_lkq_apply_tilelang_prim(
    dim: int,
    rank: int,
    dtype: str,
) -> Any:
    """Build TileLang IR for cppmega's ``MaskedLKQApplyWGMMA`` contract."""

    try:
        import tilelang.language as T
    except Exception as exc:  # pragma: no cover - host/environment dependent
        raise CuteBridgeUnsupported(
            "MaskedLKQApplyWGMMA import requires a working tilelang import; "
            f"got {exc.__class__.__name__}: {exc}."
        ) from exc

    threads = 128

    @T.prim_func
    def masked_lkq_apply_wgmma_imported(
        K: T.Tensor((dim, dim), dtype),
        Q: T.Tensor((dim, dim), dtype),
        DPhT: T.Tensor((dim, dim), dtype),
        Apply: T.Tensor((dim, dim), dtype),
    ):
        with T.Kernel(1, 1, threads=threads) as (_bx, _by):
            K_sh = T.alloc_shared((dim, dim), dtype)
            Q_sh = T.alloc_shared((dim, dim), dtype)
            DPhT_sh = T.alloc_shared((dim, dim), dtype)
            LKQ_sh = T.alloc_shared((dim, dim), dtype)
            LKQ_acc = T.alloc_fragment((dim, dim), "float32")
            Apply_acc = T.alloc_fragment((dim, dim), "float32")
            T.clear(LKQ_acc)
            T.clear(Apply_acc)
            T.copy(K[0, 0], K_sh)
            T.copy(Q[0, 0], Q_sh)
            T.copy(DPhT[0, 0], DPhT_sh)
            T.gemm(K_sh, Q_sh, LKQ_acc, False, True)
            for row, col in T.Parallel(dim, dim):
                LKQ_sh[row, col] = T.if_then_else(
                    (row // rank) < (col // rank),
                    T.cast(LKQ_acc[row, col], dtype),
                    T.cast(0, dtype),
                )
            T.gemm(LKQ_sh, DPhT_sh, Apply_acc, False, True)
            T.copy(Apply_acc, Apply[0, 0])

    return masked_lkq_apply_wgmma_imported


def _build_state_apply_consumers_tilelang_prim(
    dim: int,
    rank: int,
    chunk_size: int,
    dtype: str,
) -> Any:
    """Build TileLang IR for cppmega's one-chunk state/apply consumer tile."""

    try:
        import tilelang.language as T
    except Exception as exc:  # pragma: no cover - host/environment dependent
        raise CuteBridgeUnsupported(
            "StateApplyConsumersWGMMA import requires a working tilelang import; "
            f"got {exc.__class__.__name__}: {exc}."
        ) from exc

    threads = 128

    @T.prim_func
    def state_apply_consumers_wgmma_imported(
        K: T.Tensor((dim, dim), dtype),
        Q: T.Tensor((dim, dim), dtype),
        DstT: T.Tensor((dim, dim), dtype),
        DPhT: T.Tensor((dim, dim), dtype),
        V: T.Tensor((chunk_size, dim), dtype),
        MimoV: T.Tensor((rank, dim), dtype),
        DV: T.Tensor((chunk_size, dim), "float32"),
        DMimoV: T.Tensor((rank, dim), "float32"),
    ):
        with T.Kernel(1, 1, threads=threads) as (_bx, _by):
            K_sh = T.alloc_shared((dim, dim), dtype)
            Q_sh = T.alloc_shared((dim, dim), dtype)
            DstT_sh = T.alloc_shared((dim, dim), dtype)
            DPhT_sh = T.alloc_shared((dim, dim), dtype)
            State_sh = T.alloc_shared((dim, dim), dtype)
            LKQ_sh = T.alloc_shared((dim, dim), dtype)
            Apply_sh = T.alloc_shared((dim, dim), dtype)
            State_acc = T.alloc_fragment((dim, dim), "float32")
            LKQ_acc = T.alloc_fragment((dim, dim), "float32")
            Apply_acc = T.alloc_fragment((dim, dim), "float32")
            dv_acc = T.alloc_local((1,), "float32")
            dmimo_acc = T.alloc_local((1,), "float32")

            T.clear(State_acc)
            T.clear(LKQ_acc)
            T.clear(Apply_acc)
            T.copy(K[0, 0], K_sh)
            T.copy(Q[0, 0], Q_sh)
            T.copy(DstT[0, 0], DstT_sh)
            T.copy(DPhT[0, 0], DPhT_sh)

            T.gemm(K_sh, DstT_sh, State_acc, False, True)
            T.gemm(K_sh, Q_sh, LKQ_acc, False, True)
            for row, col in T.Parallel(dim, dim):
                State_sh[row, col] = T.cast(State_acc[row, col], dtype)
                LKQ_sh[row, col] = T.if_then_else(
                    (row // rank) < (col // rank),
                    T.cast(LKQ_acc[row, col], dtype),
                    T.cast(0, dtype),
                )

            T.gemm(LKQ_sh, DPhT_sh, Apply_acc, False, True)
            for row, col in T.Parallel(dim, dim):
                Apply_sh[row, col] = T.cast(Apply_acc[row, col], dtype)

            for t, p in T.Parallel(chunk_size, dim):
                dv_acc[0] = 0.0
                for r in T.serial(rank):
                    dv_acc[0] = dv_acc[0] + (
                        (
                            T.cast(State_sh[t * rank + r, p], "float32")
                            + T.cast(Apply_sh[t * rank + r, p], "float32")
                        )
                        * T.cast(MimoV[r, p], "float32")
                    )
                DV[t, p] = dv_acc[0]

            for r, p in T.Parallel(rank, dim):
                dmimo_acc[0] = 0.0
                for t in T.serial(chunk_size):
                    dmimo_acc[0] = dmimo_acc[0] + (
                        (
                            T.cast(State_sh[t * rank + r, p], "float32")
                            + T.cast(Apply_sh[t * rank + r, p], "float32")
                        )
                        * T.cast(V[t, p], "float32")
                    )
                DMimoV[r, p] = dmimo_acc[0]

    return state_apply_consumers_wgmma_imported


def _build_multi_chunk_state_apply_consumers_tilelang_prim(
    dim: int,
    rank: int,
    chunk_size: int,
    dtype: str,
    nchunks: int,
) -> Any:
    """Build TileLang IR for cppmega's fixed-nchunk reverse-scan consumer tile."""

    try:
        import tilelang.language as T
    except Exception as exc:  # pragma: no cover - host/environment dependent
        raise CuteBridgeUnsupported(
            "MultiChunkStateApplyConsumersWGMMA import requires a working "
            f"tilelang import; got {exc.__class__.__name__}: {exc}."
        ) from exc

    threads = 128

    @T.prim_func
    def multi_chunk_state_apply_consumers_wgmma_imported(
        K: T.Tensor((nchunks, dim, dim), dtype),
        Q: T.Tensor((nchunks, dim, dim), dtype),
        QT: T.Tensor((nchunks, dim, dim), dtype),
        DPhT: T.Tensor((nchunks, dim, dim), dtype),
        DACS: T.Tensor((nchunks, chunk_size), "float32"),
        DACSRev: T.Tensor((nchunks, chunk_size), "float32"),
        Segsum: T.Tensor((nchunks, chunk_size, chunk_size), "float32"),
        QKDot: T.Tensor((nchunks, chunk_size, rank, rank), dtype),
        Gamma: T.Tensor((nchunks, chunk_size), "float32"),
        D: T.Tensor((1,), "float32"),
        V: T.Tensor((nchunks, chunk_size, dim), dtype),
        MimoV: T.Tensor((rank, dim), dtype),
        DV: T.Tensor((nchunks, chunk_size, dim), "float32"),
        DMimoV: T.Tensor((rank, dim), "float32"),
    ):
        with T.Kernel(1, 1, threads=threads) as (_bx, _by):
            K_sh = T.alloc_shared((dim, dim), dtype)
            Q_sh = T.alloc_shared((dim, dim), dtype)
            QT_sh = T.alloc_shared((dim, dim), dtype)
            DPhT_sh = T.alloc_shared((dim, dim), dtype)
            DPhScaledT_sh = T.alloc_shared((dim, dim), dtype)
            CarryT_sh = T.alloc_shared((dim, dim), dtype)
            State_sh = T.alloc_shared((dim, dim), dtype)
            LKQ_sh = T.alloc_shared((dim, dim), dtype)
            Apply_sh = T.alloc_shared((dim, dim), dtype)
            Carry_acc = T.alloc_fragment((dim, dim), "float32")
            State_acc = T.alloc_fragment((dim, dim), "float32")
            LKQ_acc = T.alloc_fragment((dim, dim), "float32")
            Apply_acc = T.alloc_fragment((dim, dim), "float32")
            dv_acc = T.alloc_local((1,), "float32")
            dmimo_acc = T.alloc_local((1,), "float32")
            dpsi = T.alloc_local((1,), "float32")

            T.clear(Carry_acc)
            for r, p in T.Parallel(rank, dim):
                DMimoV[r, p] = 0.0

            for chunk_rev in T.serial(0, nchunks):
                chunk_idx = nchunks - 1 - chunk_rev
                T.copy(K[chunk_idx, 0, 0], K_sh)
                T.copy(Q[chunk_idx, 0, 0], Q_sh)
                T.copy(QT[chunk_idx, 0, 0], QT_sh)
                T.copy(DPhT[chunk_idx, 0, 0], DPhT_sh)

                for p, f in T.Parallel(dim, dim):
                    DPhScaledT_sh[p, f] = T.cast(
                        T.cast(DPhT[chunk_idx, p, f], "float32")
                        * T.exp(DACS[chunk_idx, f // rank]),
                        dtype,
                    )
                    CarryT_sh[p, f] = T.cast(Carry_acc[p, f], dtype)

                T.clear(State_acc)
                T.clear(LKQ_acc)
                T.clear(Apply_acc)

                T.gemm(K_sh, CarryT_sh, State_acc, False, True)
                for row, col in T.Parallel(dim, dim):
                    State_acc[row, col] = (
                        State_acc[row, col] * T.exp(DACSRev[chunk_idx, row // rank])
                    )

                T.gemm(K_sh, Q_sh, LKQ_acc, False, True)
                for row, col in T.Parallel(dim, dim):
                    State_sh[row, col] = T.cast(State_acc[row, col], dtype)
                    LKQ_sh[row, col] = T.if_then_else(
                        (row // rank) < (col // rank),
                        T.cast(
                            LKQ_acc[row, col]
                            * T.exp(Segsum[chunk_idx, col // rank, row // rank]),
                            dtype,
                        ),
                        T.cast(0, dtype),
                    )

                T.gemm(LKQ_sh, DPhT_sh, Apply_acc, False, True)
                for row, col in T.Parallel(dim, dim):
                    Apply_sh[row, col] = T.cast(Apply_acc[row, col], dtype)

                for t, p in T.Parallel(chunk_size, dim):
                    dv_acc[0] = 0.0
                    for r in T.serial(rank):
                        dpsi[0] = (
                            T.cast(State_sh[t * rank + r, p], "float32")
                            + T.cast(Apply_sh[t * rank + r, p], "float32")
                            + D[0] * T.cast(DPhT[chunk_idx, p, t * rank + r], "float32")
                        )
                        for r_out in T.serial(rank):
                            dpsi[0] = dpsi[0] + (
                                Gamma[chunk_idx, t]
                                * T.cast(QKDot[chunk_idx, t, r_out, r], "float32")
                                * T.cast(
                                    DPhT[chunk_idx, p, t * rank + r_out],
                                    "float32",
                                )
                            )
                        dpsi[0] = T.cast(T.cast(dpsi[0], dtype), "float32")
                        dv_acc[0] = dv_acc[0] + (
                            dpsi[0] * T.cast(MimoV[r, p], "float32")
                        )
                    DV[chunk_idx, t, p] = dv_acc[0]

                for r, p in T.Parallel(rank, dim):
                    dmimo_acc[0] = 0.0
                    for t in T.serial(chunk_size):
                        dpsi[0] = (
                            T.cast(State_sh[t * rank + r, p], "float32")
                            + T.cast(Apply_sh[t * rank + r, p], "float32")
                            + D[0] * T.cast(DPhT[chunk_idx, p, t * rank + r], "float32")
                        )
                        for r_out in T.serial(rank):
                            dpsi[0] = dpsi[0] + (
                                Gamma[chunk_idx, t]
                                * T.cast(QKDot[chunk_idx, t, r_out, r], "float32")
                                * T.cast(
                                    DPhT[chunk_idx, p, t * rank + r_out],
                                    "float32",
                                )
                            )
                        dpsi[0] = T.cast(T.cast(dpsi[0], dtype), "float32")
                        dmimo_acc[0] = dmimo_acc[0] + (
                            dpsi[0] * T.cast(V[chunk_idx, t, p], "float32")
                        )
                    DMimoV[r, p] = DMimoV[r, p] + dmimo_acc[0]

                for row, col in T.Parallel(dim, dim):
                    Carry_acc[row, col] = (
                        Carry_acc[row, col] * T.exp(DACS[chunk_idx, chunk_size - 1])
                    )
                T.gemm(DPhScaledT_sh, QT_sh, Carry_acc, True, True)

    return multi_chunk_state_apply_consumers_wgmma_imported


def _build_fused_bwd_bwd_p4_tilelang_prim(
    n: int,
    p: int,
    r: int,
    chunk_size: int,
    dtype: str,
    nchunks: int,
) -> Any:
    """Build TileLang IR for cppmega's Phase-4 10-GEMM CuTe tile contract."""

    try:
        import tilelang.language as T
    except Exception as exc:  # pragma: no cover - host/environment dependent
        raise CuteBridgeUnsupported(
            "FusedBwdBwdP4 import requires a working tilelang import; "
            f"got {exc.__class__.__name__}: {exc}."
        ) from exc

    fcs = chunk_size * r
    if not (n == p == fcs):
        raise CuteBridgeUnsupported(
            "FusedBwdBwdP4 source import currently supports the source's "
            "single tiled_mma contract N == P == chunk_size * R; got "
            f"N={n}, P={p}, chunk_size={chunk_size}, R={r}."
        )

    threads = 128

    @T.prim_func
    def fused_bwd_bwd_p4_imported(
        K: T.Tensor((nchunks, fcs, n), dtype),
        K_T: T.Tensor((nchunks, n, fcs), dtype),
        Q: T.Tensor((nchunks, fcs, n), dtype),
        Q_T: T.Tensor((nchunks, n, fcs), dtype),
        Dst: T.Tensor((nchunks, n, p), dtype),
        DPh: T.Tensor((nchunks, fcs, p), dtype),
        DPh_T: T.Tensor((nchunks, p, fcs), dtype),
        Psi: T.Tensor((nchunks, fcs, p), dtype),
        Sts: T.Tensor((nchunks, n, p), dtype),
        DPsiV: T.Tensor((nchunks, fcs, p), dtype),
        DK: T.Tensor((nchunks, fcs, n), dtype),
        DQ: T.Tensor((nchunks, fcs, n), dtype),
        Dqkd: T.Tensor((nchunks, fcs, fcs), dtype),
        DstatesOut: T.Tensor((n, p), dtype),
    ):
        with T.Kernel(1, 1, threads=threads) as (_bx, _by):
            K_sh = T.alloc_shared((fcs, n), dtype)
            K_T_sh = T.alloc_shared((n, fcs), dtype)
            Q_sh = T.alloc_shared((fcs, n), dtype)
            Q_T_sh = T.alloc_shared((n, fcs), dtype)
            Dst_sh = T.alloc_shared((n, p), dtype)
            DPh_sh = T.alloc_shared((fcs, p), dtype)
            DPh_T_sh = T.alloc_shared((p, fcs), dtype)
            Psi_sh = T.alloc_shared((fcs, p), dtype)
            Sts_sh = T.alloc_shared((n, p), dtype)
            LKQ_sh = T.alloc_shared((fcs, fcs), dtype)
            DKI_sh = T.alloc_shared((fcs, fcs), dtype)
            DKI_T_sh = T.alloc_shared((fcs, fcs), dtype)

            acc_dstates = T.alloc_fragment((n, p), "float32")
            acc_dPsiV = T.alloc_fragment((fcs, p), "float32")
            acc_lkq = T.alloc_fragment((fcs, fcs), "float32")
            acc_dqkd = T.alloc_fragment((fcs, fcs), "float32")
            acc_dk = T.alloc_fragment((fcs, n), "float32")
            acc_dki = T.alloc_fragment((fcs, fcs), "float32")
            acc_dki_T = T.alloc_fragment((fcs, fcs), "float32")
            acc_dq = T.alloc_fragment((fcs, n), "float32")

            T.clear(acc_dstates)
            for chunk_rev in T.serial(0, nchunks):
                chunk_idx = nchunks - 1 - chunk_rev
                T.copy(K[chunk_idx, 0, 0], K_sh)
                T.copy(K_T[chunk_idx, 0, 0], K_T_sh)
                T.copy(Q[chunk_idx, 0, 0], Q_sh)
                T.copy(Q_T[chunk_idx, 0, 0], Q_T_sh)
                T.copy(Dst[chunk_idx, 0, 0], Dst_sh)
                T.copy(DPh[chunk_idx, 0, 0], DPh_sh)
                T.copy(DPh_T[chunk_idx, 0, 0], DPh_T_sh)
                T.copy(Psi[chunk_idx, 0, 0], Psi_sh)
                T.copy(Sts[chunk_idx, 0, 0], Sts_sh)

                T.clear(acc_dPsiV)
                T.clear(acc_lkq)
                T.clear(acc_dqkd)
                T.clear(acc_dk)
                T.clear(acc_dki)
                T.clear(acc_dki_T)
                T.clear(acc_dq)

                # GEMM1: dPsiV = K @ Dst.T
                T.gemm(K_sh, Dst_sh, acc_dPsiV, False, True)
                # GEMM2: lkq = K @ Q.T
                T.gemm(K_sh, Q_sh, acc_lkq, False, True)
                # GEMM4: dqkd = DPh @ Psi.T
                T.gemm(DPh_sh, Psi_sh, acc_dqkd, False, True)
                # GEMM5: dk = Psi @ Dst.T
                T.gemm(Psi_sh, Dst_sh, acc_dk, False, True)
                # GEMM6: dki = Psi @ DPh.T
                T.gemm(Psi_sh, DPh_sh, acc_dki, False, True)
                # GEMM6': dki_T = DPh @ Psi.T
                T.gemm(DPh_sh, Psi_sh, acc_dki_T, False, True)
                # GEMM8: dq = DPh @ Sts.T
                T.gemm(DPh_sh, Sts_sh, acc_dq, False, True)
                # GEMM10: loop-carried dstates += Q.T @ DPh
                T.gemm(Q_T_sh, DPh_T_sh, acc_dstates, False, True)

                for i, j in T.Parallel(fcs, fcs):
                    LKQ_sh[i, j] = T.cast(acc_lkq[i, j], dtype)
                    DKI_sh[i, j] = T.cast(acc_dki[i, j], dtype)
                    DKI_T_sh[i, j] = T.cast(acc_dki_T[i, j], dtype)

                # GEMM3: dPsiV += lkq @ DPh.T
                T.gemm(LKQ_sh, DPh_sh, acc_dPsiV, False, True)
                # GEMM7: dk += dki @ Q
                T.gemm(DKI_sh, Q_T_sh, acc_dk, False, True)
                # GEMM9: dq += dki.T @ K
                T.gemm(DKI_T_sh, K_T_sh, acc_dq, False, True)

                T.copy(acc_dPsiV, DPsiV[chunk_idx, 0, 0])
                T.copy(acc_dk, DK[chunk_idx, 0, 0])
                T.copy(acc_dq, DQ[chunk_idx, 0, 0])
                T.copy(acc_dqkd, Dqkd[chunk_idx, 0, 0])

            T.copy(acc_dstates, DstatesOut[0, 0])

    return fused_bwd_bwd_p4_imported


def _build_fa4_v1_fused3_gemm_tilelang_prim(
    dim: int,
    dtype: str,
) -> Any:
    """Build TileLang IR for FA4 v1's 3-GEMM chain with LKQ output."""

    try:
        import tilelang.language as T
    except Exception as exc:  # pragma: no cover - host/environment dependent
        raise CuteBridgeUnsupported(
            "FA4PatternFused3Gemm import requires a working tilelang import; "
            f"got {exc.__class__.__name__}: {exc}."
        ) from exc

    threads = 128

    @T.prim_func
    def fa4_v1_fused3_gemm_imported(
        K: T.Tensor((dim, dim), dtype),
        Q: T.Tensor((dim, dim), dtype),
        DstT: T.Tensor((dim, dim), dtype),
        DPhT: T.Tensor((dim, dim), dtype),
        DPs: T.Tensor((dim, dim), dtype),
        LKQ: T.Tensor((dim, dim), dtype),
    ):
        with T.Kernel(1, 1, threads=threads) as (_bx, _by):
            K_sh = T.alloc_shared((dim, dim), dtype)
            Q_sh = T.alloc_shared((dim, dim), dtype)
            DstT_sh = T.alloc_shared((dim, dim), dtype)
            DPhT_sh = T.alloc_shared((dim, dim), dtype)
            LKQ_sh = T.alloc_shared((dim, dim), dtype)
            DPs_acc = T.alloc_fragment((dim, dim), "float32")
            LKQ_acc = T.alloc_fragment((dim, dim), "float32")

            T.clear(DPs_acc)
            T.clear(LKQ_acc)
            T.copy(K[0, 0], K_sh)
            T.copy(Q[0, 0], Q_sh)
            T.copy(DstT[0, 0], DstT_sh)
            T.copy(DPhT[0, 0], DPhT_sh)

            T.gemm(K_sh, DstT_sh, DPs_acc, False, True)
            T.gemm(K_sh, Q_sh, LKQ_acc, False, True)
            for row, col in T.Parallel(dim, dim):
                LKQ_sh[row, col] = T.cast(LKQ_acc[row, col], dtype)
            T.copy(LKQ_sh, LKQ[0, 0])
            T.gemm(LKQ_sh, DPhT_sh, DPs_acc, False, True)
            T.copy(DPs_acc, DPs[0, 0])

    return fa4_v1_fused3_gemm_imported


def _build_fa4_v2_fused3_gemm_tilelang_prim(
    dim: int,
    dtype: str,
) -> Any:
    """Build TileLang IR for FA4 v2's no-LKQ-output 3-GEMM chain."""

    try:
        import tilelang.language as T
    except Exception as exc:  # pragma: no cover - host/environment dependent
        raise CuteBridgeUnsupported(
            "FA4PatternFused3GemmV2 import requires a working tilelang import; "
            f"got {exc.__class__.__name__}: {exc}."
        ) from exc

    threads = 128

    @T.prim_func
    def fa4_v2_fused3_gemm_imported(
        K: T.Tensor((dim, dim), dtype),
        Q: T.Tensor((dim, dim), dtype),
        DstT: T.Tensor((dim, dim), dtype),
        DPhT: T.Tensor((dim, dim), dtype),
        DPs: T.Tensor((dim, dim), dtype),
    ):
        with T.Kernel(1, 1, threads=threads) as (_bx, _by):
            K_sh = T.alloc_shared((dim, dim), dtype)
            Q_sh = T.alloc_shared((dim, dim), dtype)
            DstT_sh = T.alloc_shared((dim, dim), dtype)
            DPhT_sh = T.alloc_shared((dim, dim), dtype)
            LKQ_sh = T.alloc_shared((dim, dim), dtype)
            DPs_acc = T.alloc_fragment((dim, dim), "float32")
            LKQ_acc = T.alloc_fragment((dim, dim), "float32")

            T.clear(DPs_acc)
            T.clear(LKQ_acc)
            T.copy(K[0, 0], K_sh)
            T.copy(Q[0, 0], Q_sh)
            T.copy(DstT[0, 0], DstT_sh)
            T.copy(DPhT[0, 0], DPhT_sh)

            T.gemm(K_sh, DstT_sh, DPs_acc, False, True)
            T.gemm(K_sh, Q_sh, LKQ_acc, False, True)
            for row, col in T.Parallel(dim, dim):
                LKQ_sh[row, col] = T.cast(LKQ_acc[row, col], dtype)
            T.gemm(LKQ_sh, DPhT_sh, DPs_acc, False, True)
            T.copy(DPs_acc, DPs[0, 0])

    return fa4_v2_fused3_gemm_imported


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
