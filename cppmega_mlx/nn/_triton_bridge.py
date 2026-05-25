# pyright: reportInvalidTypeForm=false, reportMissingImports=false
"""Triton -> TileLang bridge for the unified ``dispatch_lower`` pipeline.

This module wraps the (in-development) POC frontend at
``tl_poc_review/poc/triton_frontend`` so that callers in cppmega.mlx can
take a ``@triton.jit`` kernel from cppmega/megatron and route it through
the same ``dispatch_lower(prim, target=...)`` entrypoint already used by
the Path-C TileLang kernels (see :mod:`cppmega_mlx.nn._tilelang._engine_dispatch`).

The convergence design lives in
``/private/tmp/tl_apache_tvm_swap/RFC_unified_fused_kernel.md`` (sections 5
and 6). The frontend itself is at
``/private/tmp/tl_poc_review/poc/triton_frontend/`` and is *not* on the
default ``sys.path``; this module adds it on demand and re-exports a
single ergonomic helper, :func:`triton_to_tilelang_prim`.

Usage::

    from cppmega_mlx.nn._triton_bridge import triton_to_tilelang_prim
    from cppmega_mlx.nn._tilelang._engine_dispatch import dispatch_lower

    # `kernel` is a function decorated with @triton.jit
    prim = triton_to_tilelang_prim(kernel, constexprs={...})
    artifact = dispatch_lower(prim, target="cuda")

Known limitations (as of 2026-05-07)
-----------------------------------
1. The POC frontend's ``OP_TABLE`` covers a Tier-1 surface (load / store /
   make_range / program_id / dot / reduce / where / broadcast / splat /
   expand_dims / reshape / trans / atomic_rmw / async_copy / mbarrier /
   TMA / partial_barrier / print). Many emitters still raise
   ``NotImplementedError`` on operands they haven't seen — bugs surface as
   ``NotImplementedError("triton_frontend: ...")`` from the walker.
2. The ``PtrAnalysis`` C++ shim (microsoft/triton-shared) is vendored but
   not built on this host; the frontend silently falls back to a scalar
   "MVP" path that synthesises placeholder buffers. Multi-element tile
   loads degrade to per-element BufferLoad/Store. See
   ``poc/triton_frontend/ptr_analysis.py``.
3. Without ``mlir.ir`` Python bindings the frontend uses a regex-based
   text-TTIR walker that confirms op coverage but does NOT populate
   ``ctx.value_map`` / ``ctx.buffers``. The resulting PrimFunc is a stub
   shell — useful for end-to-end import smoke but not runnable.
4. Triton itself is an optional dependency; this bridge raises
   ``ModuleNotFoundError`` (not ``ImportError``) if triton is missing so
   pytest.importorskip can pick it up.

Per the codebase ``feedback_no_silent_delete`` rule we never silently
swallow lowering failures: any unexpected exception from the frontend
walker is wrapped in :class:`TritonBridgeError` with the original cause
attached so callers can decide whether to fall back, raise, or log.
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

__all__ = [
    "TritonBridgeError",
    "triton_to_tilelang_prim",
    "triton_to_tilelang_compile",
    "frontend_available",
    "rewrite_dot_trans_b_to_transpose",
    "TRITON_FRONTEND_PATH_ENV",
]


def rewrite_dot_trans_b_to_transpose(fn: Any) -> Any:
    """V7-N03 honest closure: rewrite ``tl.dot(a, b, trans_b=True)``
    source-level into ``tl.dot(a, tl.trans(b))`` so the POC frontend's
    OP_TABLE — which covers ``tl.trans`` but not the ``trans_b`` kwarg
    of ``tl.dot`` under MVP scalar lowering — can lower the kernel.

    Returns a wrapped triton.jit object with the same name / signature
    whose source has the AST substitution applied. The original is
    untouched (so the same kernel can be lowered through both paths).
    """
    import ast
    import inspect
    import textwrap

    try:
        src = textwrap.dedent(inspect.getsource(fn))
    except (TypeError, OSError) as exc:
        raise TritonBridgeError(
            f"rewrite_dot_trans_b_to_transpose: cannot read source "
            f"of {getattr(fn, '__name__', '?')!r}: {exc}") from exc

    tree = ast.parse(src)

    class _Rewriter(ast.NodeTransformer):
        def visit_Call(self, node: ast.Call) -> ast.AST:
            self.generic_visit(node)
            # Match `tl.dot(a, b, trans_b=True, ...)`
            is_tl_dot = (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "dot"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "tl"
            )
            if not is_tl_dot:
                return node
            kept_kw: list[ast.keyword] = []
            trans_b = False
            for kw in node.keywords:
                if (kw.arg == "trans_b"
                        and isinstance(kw.value, ast.Constant)
                        and kw.value.value is True):
                    trans_b = True
                    continue
                kept_kw.append(kw)
            if not trans_b or len(node.args) < 2:
                return node
            # Replace second positional arg b with tl.trans(b).
            new_b = ast.Call(
                func=ast.Attribute(value=ast.Name(id="tl",
                                                    ctx=ast.Load()),
                                    attr="trans", ctx=ast.Load()),
                args=[node.args[1]],
                keywords=[],
            )
            new_args = list(node.args)
            new_args[1] = new_b
            return ast.Call(func=node.func, args=new_args,
                             keywords=kept_kw)

    new_tree = ast.fix_missing_locations(_Rewriter().visit(tree))
    new_src = ast.unparse(new_tree)

    # Re-exec in the original function's globals so triton.jit resolves.
    g = dict(getattr(fn, "__globals__", {}))
    loc: dict[str, Any] = {}
    exec(compile(new_src, f"<rewritten {fn.__name__}>", "exec"), g, loc)
    rewritten = loc.get(fn.__name__, fn)
    return rewritten


#: Env var override: point at a different ``tl_poc_review`` checkout.
TRITON_FRONTEND_PATH_ENV = "CPPMEGA_MLX_TRITON_FRONTEND_PATH"

#: Historical location of the POC frontend on dev hosts. Newer checkouts keep
#: TileLang as a sibling of ``cppmega.mlx``.
_DEFAULT_FRONTEND_ROOT = Path("/private/tmp/tl_poc_review")


class TritonBridgeError(RuntimeError):
    """Raised when the Triton -> TileLang lowering fails inside the POC frontend.

    Carries the original exception via ``__cause__`` so callers can
    inspect / re-raise without losing the lower-level traceback. We
    deliberately do NOT subclass :class:`ImportError` — missing triton
    surfaces as ``ModuleNotFoundError`` directly so test files can use
    ``pytest.importorskip``.
    """


def _frontend_root() -> Path:
    """Return the directory that contains the ``poc/triton_frontend`` package.

    Override via ``$CPPMEGA_MLX_TRITON_FRONTEND_PATH``. Falls back to the
    standard dev path, then to a sibling ``../tilelang`` checkout. We do NOT
    raise if no candidate exists — callers go through :func:`frontend_available`
    first.
    """

    raw = os.environ.get(TRITON_FRONTEND_PATH_ENV)
    if raw:
        return Path(raw)
    candidates = (
        _DEFAULT_FRONTEND_ROOT,
        Path(__file__).resolve().parents[3] / "tilelang",
    )
    for candidate in candidates:
        if (candidate / "poc" / "triton_frontend" / "__init__.py").exists():
            return candidate
    return _DEFAULT_FRONTEND_ROOT


def frontend_available() -> bool:
    """Return True iff the POC ``triton_frontend`` package is importable.

    Side effect: prepends the frontend root to ``sys.path`` if needed.
    Idempotent — repeated calls do not duplicate the path entry.
    """

    root = _frontend_root()
    if not root.exists():
        return False
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    try:
        import poc.triton_frontend  # noqa: F401
    except (ImportError, OSError) as exc:
        warnings.warn(
            "cppmega_mlx._triton_bridge: poc.triton_frontend import failed "
            f"(root={root_str!r}, error={exc!r}). Bridge is unavailable.",
            UserWarning,
            stacklevel=2,
        )
        return False
    return True


def _require_frontend() -> Any:
    """Import ``poc.triton_frontend`` or raise a clear error.

    Returns the module so callers can grab ``from_triton_kernel`` /
    ``from_ttir`` directly without re-importing.
    """

    if not frontend_available():
        raise TritonBridgeError(
            "POC triton_frontend not importable from "
            f"{_frontend_root()!s}. Set "
            f"{TRITON_FRONTEND_PATH_ENV} to a checkout of "
            "tl_poc_review or clone it to /private/tmp/tl_poc_review."
        )
    import poc.triton_frontend as tf  # type: ignore[import-not-found]

    return tf


def _triton_to_tilelang_prim_via_subprocess_ttir(
    tf: Any,
    fn: Callable[..., Any],
    *,
    grid: Optional[Tuple[int, ...]],
    constexprs: Optional[Dict[str, Any]],
    target: Optional[str],
    name: Optional[str],
) -> Any:
    """Fallback for frontend paths that cannot co-load Triton and PtrAnalysis."""

    import inspect
    import textwrap

    from poc.triton_frontend._test_harness.jit_to_ttir import (  # type: ignore[import-not-found]
        triton_jit_to_ttir_subprocess_from_source,
    )

    underlying_fn = getattr(fn, "fn", fn)
    source = textwrap.dedent(inspect.getsource(underlying_fn))
    kernel_name = (
        getattr(underlying_fn, "__name__", None)
        or getattr(fn, "__name__", None)
        or "triton_kernel"
    )
    ttir_text = triton_jit_to_ttir_subprocess_from_source(
        source=source,
        kernel_name=kernel_name,
        constexprs=constexprs,
        target=target,
        extra_sys_path=[str(_frontend_root())],
    )
    return tf.from_ttir(
        ttir_text,
        target=target,
        name=name or kernel_name,
        grid=grid,
    )


def triton_to_tilelang_prim(
    fn: Callable[..., Any],
    *,
    grid: Optional[Tuple[int, ...]] = None,
    constexprs: Optional[Dict[str, Any]] = None,
    target: Optional[str] = None,
    name: Optional[str] = None,
) -> Any:
    """Lower a ``@triton.jit`` Python function to a TileLang ``PrimFunc``.

    Thin wrapper over ``poc.triton_frontend.from_triton_kernel`` that:

    * Validates that ``fn`` actually carries the ``triton.jit`` marker.
    * Re-raises POC-frontend ``NotImplementedError`` /
      ``RuntimeError`` as :class:`TritonBridgeError` so production
      callers can ``except TritonBridgeError`` once instead of guessing
      at the frontend's evolving error taxonomy.
    * Forwards ``grid`` / ``constexprs`` / ``target`` / ``name`` so the
      caller doesn't have to reach into the POC frontend's signature.

    Parameters
    ----------
    fn:
        A function decorated with ``@triton.jit``. We accept either the
        ``JITFunction`` wrapper or its underlying ``fn`` attribute.
    grid:
        Optional launch grid (lifted from kernel metadata when absent).
    constexprs:
        Triton ``constexpr`` bindings, e.g. ``{"BLOCK_M": 128}``.
    target:
        TileLang target string passed through to the frontend (does NOT
        compile — that's :func:`triton_to_tilelang_compile`'s job).
    name:
        Symbol name to assign to the resulting PrimFunc. Defaults to
        ``fn.__name__``.

    Returns
    -------
    tvm.tir.PrimFunc
        Ready to feed into :func:`dispatch_lower`.
    """

    tf = _require_frontend()

    # Triton's ``@triton.jit`` wraps the function in ``JITFunction`` whose
    # underlying callable lives at ``.fn``. Keep the wrapper for the frontend:
    # Triton's 3.6+ TTIR capture path needs ``JITFunction.params`` and other
    # compile metadata, while the underlying callable is only used here for
    # validation and name inference.
    underlying_fn = getattr(fn, "fn", fn)
    if not callable(underlying_fn):
        raise TritonBridgeError(
            f"triton_to_tilelang_prim: expected a callable, got {type(fn)!r}"
        )

    inferred_name = name or getattr(underlying_fn, "__name__", None) or "triton_kernel"

    try:
        prim = tf.from_triton_kernel(
            fn,
            grid=grid,
            constexprs=constexprs,
            target=target,
        )
    except ModuleNotFoundError:
        # Triton itself missing — surface the original exception so
        # ``pytest.importorskip("triton")`` works at the call site.
        raise
    except NotImplementedError as exc:
        # V7-N03 honest closure: when the POC frontend's OP_TABLE hits a
        # known coverage gap (today: tl.dot(trans_b=True) under the
        # MVP scalar lowering), surface the gap with an actionable
        # diagnostic that names the gap and points callers at the
        # AST rewrite helper :func:`rewrite_dot_trans_b_to_transpose`.
        # Other NotImplementedError paths still propagate as a generic
        # bridge error so we don't silently lose coverage signals.
        detail = str(exc)
        if "trans_b" in detail or "dot" in detail.lower():
            raise TritonBridgeError(
                f"Triton frontend OP_TABLE missing tl.dot trans_b "
                f"variant while lowering {inferred_name!r}: {exc}. "
                f"Workaround: wrap the kernel with "
                f"cppmega_mlx.nn._triton_bridge."
                f"rewrite_dot_trans_b_to_transpose(kernel) which "
                f"rewrites tl.dot(a, b, trans_b=True) → "
                f"tl.dot(a, tl.trans(b)) — covered by the existing "
                f"OP_TABLE entry for tl.trans."
            ) from exc
        raise TritonBridgeError(
            f"Triton frontend coverage gap while lowering {inferred_name!r}: {exc}"
        ) from exc
    except Exception as exc:  # pragma: no cover - defensive
        if (
            isinstance(exc, RuntimeWarning)
            and "PtrAnalysis C++ shim disabled" in str(exc)
        ):
            try:
                prim = _triton_to_tilelang_prim_via_subprocess_ttir(
                    tf,
                    fn,
                    grid=grid,
                    constexprs=constexprs,
                    target=target,
                    name=name,
                )
            except Exception as fallback_exc:
                raise TritonBridgeError(
                    f"Triton frontend failed for {inferred_name!r}: {exc!r}; "
                    f"fresh-process TTIR fallback also failed: {fallback_exc!r}"
                ) from fallback_exc
            else:
                if name and prim is not None and hasattr(prim, "with_attr"):
                    prim = prim.with_attr("global_symbol", name)
                return prim
        raise TritonBridgeError(
            f"Triton frontend failed for {inferred_name!r}: {exc!r}"
        ) from exc

    # The POC ``from_triton_kernel`` already names the PrimFunc using
    # ``fn.__name__``; honour the user override if supplied.
    if name and prim is not None and hasattr(prim, "with_attr"):
        prim = prim.with_attr("global_symbol", name)
    return prim


def triton_to_tilelang_compile(
    fn: Callable[..., Any],
    *,
    target: str = "cuda",
    grid: Optional[Tuple[int, ...]] = None,
    constexprs: Optional[Dict[str, Any]] = None,
    name: Optional[str] = None,
) -> Any:
    """End-to-end: ``@triton.jit`` -> TileLang PrimFunc -> dispatch_lower.

    Convenience wrapper that runs :func:`triton_to_tilelang_prim` and
    immediately hands the PrimFunc to
    :func:`cppmega_mlx.nn._tilelang._engine_dispatch.dispatch_lower`.
    Engine-vs-shim selection respects ``$CPPMEGA_MLX_TILELANG_ENGINE``.
    """

    prim = triton_to_tilelang_prim(
        fn,
        grid=grid,
        constexprs=constexprs,
        target=target,
        name=name,
    )
    # Local import so this module does not eagerly drag tilelang in when
    # callers only want the PrimFunc (e.g. unit tests for the lowering
    # surface).
    from cppmega_mlx.nn._tilelang._engine_dispatch import dispatch_lower

    return dispatch_lower(prim, target=target)
