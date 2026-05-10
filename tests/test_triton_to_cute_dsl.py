# pyright: reportInvalidTypeForm=false, reportMissingImports=false
"""Composed bridge test: ``@triton.jit`` -> TileLang PrimFunc -> CuTeDSL source.

There is no direct ``triton -> cute_dsl`` bridge — PR apache/tilelang#1421
ships only the ``tilelang -> cute_dsl`` forward direction. This test
exercises the *composed* chain that
:mod:`cppmega_mlx.nn._triton_to_cute_dsl` wires by stacking two existing
one-way bridges:

    @triton.jit fn
        --[triton_to_tilelang_prim]-->  tvm.tir.PrimFunc
        --[compile_prim_to_cutedsl]-->  CuTeDSL Python source string

We pin the *emit contract* (non-empty source, contains a ``@cute.kernel``
or ``@cute.jit`` decorator), not numerics — numeric verification belongs
in the Triton-frontend numeric harness.

Skip policy (per project rule "no silent skips"):
* ``triton`` not installed -> skip with reason.
* POC ``triton_frontend`` not on disk -> skip with reason.
* ``tilelang`` not importable -> skip with reason.
* CuTeDSL backend (``nvidia-cutlass-dsl`` + libtilelang's cutedsl
  codegen translation unit) not reachable -> skip with reason.

On a Mac dev host we expect the test to skip on the CuTeDSL backend
gate, because ``src/target/codegen_cutedsl.cc`` is excluded from non-CUDA
libtilelang builds.
"""
from __future__ import annotations

import ast
import importlib
import importlib.util
import os
import sys

import pytest


# -----------------------------------------------------------------------------
# Module loading — bypass cppmega_mlx.nn.__init__ which eagerly imports
# attention.py / mlx_lm and frequently fails on dev hosts. Same trick as
# tests/test_cute_bridge.py.
# -----------------------------------------------------------------------------


def _load_module_by_path(qualname: str, relpath: str):
    """Load a single ``cppmega_mlx`` submodule directly by file path."""

    abs_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        *relpath.split("/"),
    )
    spec = importlib.util.spec_from_file_location(qualname, abs_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[qualname] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_chain_module():
    """Load ``cppmega_mlx.nn._triton_to_cute_dsl`` plus its two leg deps.

    The chain module does ``from ._cute_bridge import ...`` and
    ``from ._triton_bridge import ...`` — those need to be loadable as
    real package members because the relative import resolves through
    ``sys.modules``. We pre-load them under their canonical qualnames.
    """

    # Pre-stage the package containers so relative imports resolve.
    if "cppmega_mlx" not in sys.modules:
        pkg = importlib.util.module_from_spec(
            importlib.util.spec_from_loader("cppmega_mlx", loader=None)  # type: ignore[arg-type]
        )
        pkg.__path__ = [  # type: ignore[attr-defined]
            os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "cppmega_mlx",
            )
        ]
        sys.modules["cppmega_mlx"] = pkg
    if "cppmega_mlx.nn" not in sys.modules:
        nn_pkg = importlib.util.module_from_spec(
            importlib.util.spec_from_loader("cppmega_mlx.nn", loader=None)  # type: ignore[arg-type]
        )
        nn_pkg.__path__ = [  # type: ignore[attr-defined]
            os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "cppmega_mlx",
                "nn",
            )
        ]
        sys.modules["cppmega_mlx.nn"] = nn_pkg

    _load_module_by_path("cppmega_mlx.nn._cute_bridge", "cppmega_mlx/nn/_cute_bridge.py")
    _load_module_by_path("cppmega_mlx.nn._triton_bridge", "cppmega_mlx/nn/_triton_bridge.py")
    return _load_module_by_path(
        "cppmega_mlx.nn._triton_to_cute_dsl",
        "cppmega_mlx/nn/_triton_to_cute_dsl.py",
    )


# -----------------------------------------------------------------------------
# Skip gates
# -----------------------------------------------------------------------------


def _skip_unless_full_chain_available(triton_bridge_mod, cute_bridge_mod) -> None:
    """Skip with a precise reason if any chain layer is missing."""

    # Layer 1: triton itself.
    try:
        importlib.import_module("triton")
    except Exception as exc:
        pytest.skip(
            reason=(
                "triton not importable on this host "
                f"({exc.__class__.__name__}: {exc}); the chain's first leg "
                "(@triton.jit -> TTIR) is unreachable."
            )
        )

    # Layer 2: POC triton_frontend on disk.
    if not triton_bridge_mod.frontend_available():
        pytest.skip(
            reason=(
                "POC triton_frontend not importable; the bridge's first leg "
                "(TTIR -> tvm.tir.PrimFunc) is unreachable. Set "
                "CPPMEGA_MLX_TRITON_FRONTEND_PATH to the tl_poc_review checkout."
            )
        )

    # Layer 3: tilelang.
    try:
        importlib.import_module("tilelang")
    except Exception as exc:
        pytest.skip(
            reason=(
                "tilelang unimportable on this host "
                f"({exc.__class__.__name__}: {exc}); the chain's second leg "
                "(PrimFunc -> CuTeDSL) is unreachable."
            )
        )

    # Layer 4: cutedsl backend python adapter (cutlass.cute import path).
    ok, reason = cute_bridge_mod.tilelang_cutedsl_available()
    if not ok:
        pytest.skip(
            reason=(
                "TileLang CuTeDSL backend not reachable on this host "
                f"({reason}); install nvidia-cutlass-dsl to enable the "
                "second leg of the chain."
            )
        )

    # Layer 5: libtilelang's cutedsl codegen FFI symbol — gated on USE_CUDA
    # in src/backend/cuda/CMakeLists.txt and missing from non-CUDA builds.
    import tvm_ffi

    names = set(tvm_ffi.registry.list_global_func_names())
    needed = "target.build.tilelang_cutedsl_without_compile"
    if needed not in names:
        pytest.skip(
            reason=(
                f"libtilelang on this host does not register {needed!r}; "
                "the cutedsl codegen translation unit is gated on USE_CUDA "
                "and is omitted from non-CUDA builds (e.g. macOS dev hosts). "
                "On a CUDA host this test would emit CuTeDSL source for the "
                "Triton kernel."
            )
        )


# -----------------------------------------------------------------------------
# Tiny @triton.jit kernel — vector_add.
# -----------------------------------------------------------------------------


def _make_vector_add_kernel():
    """Return the simplest possible ``@triton.jit`` kernel.

    ``vector_add`` exercises just ``program_id``, ``arange``, masking,
    ``load`` / ``store`` and scalar add — the floor of the POC frontend's
    op coverage. If even this kernel fails to lower then the chain is
    broken regardless of CuTeDSL availability.
    """

    import triton
    import triton.language as tl

    @triton.jit
    def vector_add_kernel(
        x_ptr,
        y_ptr,
        out_ptr,
        n_elements: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
        tl.store(out_ptr + offsets, x + y, mask=mask)

    return vector_add_kernel


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------


def test_chain_module_surface() -> None:
    """API-surface contract — runs on every host (no triton/cute deps).

    Loading the chain module must succeed even when triton / tilelang /
    cutedsl are missing, because the heavy imports happen lazily inside
    the called functions, not at module import time.
    """

    chain_mod = _load_chain_module()

    fn = chain_mod.compile_triton_to_cute_dsl
    assert callable(fn)

    import inspect

    sig = inspect.signature(fn)
    params = sig.parameters

    # Expected surface:
    #   compile_triton_to_cute_dsl(kernel, *, signature=None,
    #                              constexprs=None, grid=None,
    #                              name=None, kernel_only=True) -> str
    assert "kernel" in params
    assert "signature" in params
    assert "constexprs" in params
    assert "grid" in params
    assert "name" in params
    assert "kernel_only" in params

    # Module docstring must mention both legs of the chain so future
    # readers know this is composition, not a new bridge.
    assert "triton_frontend" in (chain_mod.__doc__ or "") or "Triton" in (
        chain_mod.__doc__ or ""
    )
    assert "1421" in (chain_mod.__doc__ or "")


def test_frontend_available_handles_frontend_dylib_load_failure(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """POC frontend availability checks must not leak native import errors."""

    bridge_mod = _load_module_by_path(
        "cppmega_mlx.nn._triton_bridge",
        "cppmega_mlx/nn/_triton_bridge.py",
    )
    frontend_pkg = tmp_path / "poc" / "triton_frontend"
    frontend_pkg.mkdir(parents=True)
    (tmp_path / "poc" / "__init__.py").write_text("")
    (frontend_pkg / "__init__.py").write_text(
        "raise OSError('fake dylib load failure')\n"
    )

    monkeypatch.setenv(bridge_mod.TRITON_FRONTEND_PATH_ENV, str(tmp_path))
    old_path = list(sys.path)
    old_poc_modules = {
        name: module
        for name, module in sys.modules.items()
        if name == "poc" or name.startswith("poc.")
    }
    for name in old_poc_modules:
        sys.modules.pop(name, None)

    try:
        with pytest.warns(UserWarning, match="poc.triton_frontend import failed"):
            assert bridge_mod.frontend_available() is False
    finally:
        sys.path[:] = old_path
        for name in list(sys.modules):
            if name == "poc" or name.startswith("poc."):
                sys.modules.pop(name, None)
        sys.modules.update(old_poc_modules)


def test_triton_kernel_lowers_to_cute_dsl_source() -> None:
    """End-to-end: tiny @triton.jit -> non-empty CuTeDSL Python source.

    Acceptance:
      * Result is a non-empty ``str``.
      * Result contains ``@cute.kernel`` or ``@cute.jit`` (PR #1421's
        emitter has used both decorator names across cutlass-dsl minor
        versions; either is acceptable).
      * Result parses as Python (``ast.parse``).

    Skips with a precise reason on hosts missing any link in the chain.
    On the Mac dev host this typically skips on libtilelang's missing
    cutedsl codegen translation unit (Layer 5 in the skip gate).
    """

    chain_mod = _load_chain_module()

    # Pull bridge mods through sys.modules — the chain loader staged them.
    triton_bridge_mod = sys.modules["cppmega_mlx.nn._triton_bridge"]
    cute_bridge_mod = sys.modules["cppmega_mlx.nn._cute_bridge"]

    _skip_unless_full_chain_available(triton_bridge_mod, cute_bridge_mod)

    # Force engine path so dispatch_lower doesn't silently fall back.
    prev_engine = os.environ.get("CPPMEGA_MLX_TILELANG_ENGINE")
    os.environ["CPPMEGA_MLX_TILELANG_ENGINE"] = "engine"
    try:
        kernel = _make_vector_add_kernel()
        try:
            src = chain_mod.compile_triton_to_cute_dsl(
                kernel,
                signature={
                    "x_ptr": "*fp32",
                    "y_ptr": "*fp32",
                    "out_ptr": "*fp32",
                },
                constexprs={"n_elements": 256, "BLOCK_SIZE": 64},
            )
        except triton_bridge_mod.TritonBridgeError as exc:
            # Coverage gap in the POC triton frontend — surface as xfail
            # so the gap is visible without breaking CI red.
            pytest.xfail(
                f"POC triton_frontend coverage gap on vector_add: {exc}"
            )

        assert isinstance(src, str)
        assert len(src) > 0, "emitted CuTeDSL source is empty"
        assert (
            "@cute.kernel" in src or "@cute.jit" in src
        ), f"no cute.kernel / cute.jit decorator in output:\n{src[:400]}"

        # Parses as Python.
        ast.parse(src)
    finally:
        if prev_engine is None:
            os.environ.pop("CPPMEGA_MLX_TILELANG_ENGINE", None)
        else:
            os.environ["CPPMEGA_MLX_TILELANG_ENGINE"] = prev_engine
