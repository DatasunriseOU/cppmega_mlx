# pyright: reportInvalidTypeForm=false, reportMissingImports=false
"""End-to-end chain: ``@triton.jit`` Python -> CuTeDSL Python source.

There is no direct ``triton -> cute_dsl`` bridge (PR apache/tilelang#1421
ships only ``tilelang -> cute_dsl`` forward emit). This module composes
two existing one-way bridges to bridge the gap:

    @triton.jit fn
        |
        v   poc.triton_frontend.from_triton_kernel  (Wave A-L)
        |
    tvm.tir.PrimFunc
        |
        v   tilelang.compile(prim, target='cutedsl') (PR #1421)
        |
    CuTeDSLKernelAdapter -> .get_kernel_source() -> str

The first leg lives in :mod:`cppmega_mlx.nn._triton_bridge`
(:func:`triton_to_tilelang_prim`); the second leg lives in
:mod:`cppmega_mlx.nn._cute_bridge` (:func:`compile_prim_to_cutedsl`).
This module is a *thin* composition — no new lowering logic — so the
contract surface stays at one place per direction and the chain is
trivially debuggable: a failure here is always a failure in one of the
underlying bridges, which already raise precise typed exceptions.

Public API: :func:`compile_triton_to_cute_dsl`.
"""
from __future__ import annotations

from typing import Any, Callable, Mapping, Optional, Tuple

from ._cute_bridge import compile_prim_to_cutedsl
from ._triton_bridge import triton_to_tilelang_prim

__all__ = [
    "compile_triton_to_cute_dsl",
]


def compile_triton_to_cute_dsl(
    kernel: Callable[..., Any],
    *,
    signature: Optional[Mapping[str, str]] = None,
    constexprs: Optional[Mapping[str, Any]] = None,
    grid: Optional[Tuple[int, ...]] = None,
    name: Optional[str] = None,
    kernel_only: bool = True,
) -> str:
    """Compile a Triton kernel to CuTeDSL Python source via TileLang.

    Two-step chain that composes existing one-way bridges:

    1. ``triton -> tilelang``: lower the ``@triton.jit`` kernel through
       the POC ``triton_frontend`` (TTIR text -> ``tvm.tir.PrimFunc``).
       Implemented by :func:`triton_to_tilelang_prim`.
    2. ``tilelang -> cute_dsl``: emit CuTeDSL Python source from the
       PrimFunc using PR apache/tilelang#1421's CuTeDSL backend.
       Implemented by :func:`compile_prim_to_cutedsl`.

    The reverse direction (CuTeDSL -> Triton) does not exist anywhere in
    the toolchain and is not provided.

    Parameters
    ----------
    kernel:
        A function decorated with ``@triton.jit``. The bridge accepts
        either the ``JITFunction`` wrapper or its underlying ``fn``.
    signature:
        Optional Triton-style argument signature mapping (e.g.
        ``{"x_ptr": "*fp32", "n_elements": "i32"}``). Currently
        unused by the underlying frontend (which infers types from the
        TTIR), retained for API symmetry with :mod:`triton.compiler`
        so callers do not have to learn a different surface.
    constexprs:
        Triton ``tl.constexpr`` bindings, e.g. ``{"BLOCK_SIZE": 128}``.
        Forwarded to the triton frontend.
    grid:
        Optional launch grid (lifted from kernel metadata when absent).
    name:
        Symbol name override for the resulting PrimFunc / CuTeDSL
        entrypoint.
    kernel_only:
        Forwarded to ``CuTeDSLKernelAdapter.get_kernel_source``. When
        ``True`` (default) returns just the ``@cute.kernel`` /
        ``@cute.jit`` body; when ``False`` returns the full module
        including imports and the host launcher.

    Returns
    -------
    str
        Emitted CuTeDSL Python source as a string.

    Raises
    ------
    cppmega_mlx.nn._triton_bridge.TritonBridgeError
        Coverage gap in the POC triton frontend (op not yet wired in
        ``OP_TABLE``, ``PtrAnalysis`` failure, etc.).
    ModuleNotFoundError
        Triton itself is not installed.
    Exception
        Any error raised by ``tilelang.compile(target='cutedsl')`` —
        this includes missing ``nvidia-cutlass-dsl``, libtilelang built
        without the CuTeDSL codegen translation unit (typical on
        non-CUDA hosts), or a CuTeDSL adapter init failure.

    Notes
    -----
    The signature/constexprs split mirrors how Triton itself separates
    runtime arguments (``signature``) from compile-time constants
    (``constexprs``). The POC frontend currently consumes only the
    ``constexprs`` channel; the ``signature`` parameter is kept so this
    function's surface remains drop-in compatible with future versions
    that thread runtime types through.
    """

    _ = signature  # currently unused; see docstring "Notes" above.

    prim = triton_to_tilelang_prim(
        kernel,
        grid=grid,
        constexprs=dict(constexprs) if constexprs is not None else None,
        name=name,
    )
    artifact = compile_prim_to_cutedsl(prim)
    return artifact.get_kernel_source(kernel_only=kernel_only)
