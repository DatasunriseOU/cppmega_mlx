# pyright: reportInvalidTypeForm=false, reportMissingImports=false
"""Smoke test for the cppmega CuTeDSL <-> TileLang CuTeDSL-target bridge.

Goal (per task brief):
1. A CuTe DSL kernel from cppmega can be loaded.
2. It is recognized by TileLang's CuTeDSL bridge (PR apache/tilelang#1421).
3. ``dispatch_lower(prim, target='cuda'/'cutedsl')`` produces a runnable
   artifact, or fails loudly with a clear reason.

Important reality check
-----------------------
TileLang's bridge emits CuTeDSL Python from TileLang IR; it does NOT import
hand-written ``@cute.kernel`` modules (see
:mod:`cppmega_mlx.nn._cute_bridge` docstring and ``CUTE_BRIDGE_KNOWN_GAPS``).
So this test:

* Verifies cppmega's smallest CuTeDSL kernel module
  (``single_gemm_test.py`` from ``cppmega/megatron/cute_dsl_mimo/``) is
  source-loadable as a Python module *if* cutlass is installed (load-only,
  no compile/launch — those need CUDA + Hopper).
* Confirms the upstream bridge entry points exist with their expected
  symbols. The PR #1421 surface is stable enough that this is a meaningful
  contract test even on Mac.
* Exercises the *supported* direction by building a tiny TileLang
  ``T.prim_func`` GEMM and calling
  ``compile_prim_to_cutedsl(prim)`` — when the host has CUDA + cutlass.
* Skips with an explicit reason on Mac / non-CUDA / missing-cutlass hosts
  (per memory rule: never silently skip).
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import sys

import pytest

# CuTeDSL kernels are CUDA-only; gate the entire module on torch+CUDA upfront.
torch = pytest.importorskip(
    "torch",
    reason="CuTeDSL bridge smoke test requires PyTorch (torch.cuda for runtime)",
)

CPPMEGA_CUTE_DIR = (
    "/Volumes/external/sources/cppmega/cppmega/megatron/cute_dsl_mimo"
)
CPPMEGA_SMALLEST_KERNEL = "single_gemm_test"  # 7.5K LOC, simplest entrypoint


def _ensure_cppmega_cute_on_path() -> bool:
    """Add cppmega's cute_dsl_mimo dir to sys.path (read-only). Returns True
    if the directory exists.
    """

    if not os.path.isdir(CPPMEGA_CUTE_DIR):
        return False
    if CPPMEGA_CUTE_DIR not in sys.path:
        sys.path.insert(0, CPPMEGA_CUTE_DIR)
    return True


def test_tilelang_cutedsl_bridge_symbols_exist() -> None:
    """PR apache/tilelang#1421 contract: the documented entry points are
    importable as named in the bridge wrapper.

    This test does NOT require CUDA or cutlass — it only checks that
    tilelang's adapter module exposes the symbols our wrapper claims it
    exposes. If tilelang itself cannot load (e.g. missing libz3 on macOS
    dev hosts), skip with a precise reason rather than fail.
    """

    try:
        adapter_mod = importlib.import_module("tilelang.jit.adapter.cutedsl")
    except Exception as exc:
        pytest.skip(
            reason=(
                "tilelang.jit.adapter.cutedsl unimportable on this host "
                f"({exc.__class__.__name__}: {exc}); typical on macOS dev "
                "hosts where libtilelang.dylib has unmet libz3 dep — bridge "
                "contract test requires a working tilelang import."
            )
        )

    # PR #1421 surface — the wrapper relies on these names.
    for name in (
        "CuTeDSLKernelAdapter",
        "TLCuTeDSLSourceWrapper",
        "CuTeDSLLibraryGenerator",
        "check_cutedsl_available",
    ):
        assert hasattr(adapter_mod, name), (
            f"tilelang.jit.adapter.cutedsl is missing {name!r} — PR #1421 "
            "surface drift; update _cute_bridge.TILELANG_CUTEDSL_ENTRY."
        )

    # Target normalizer must recognize the 'cutedsl' string-form target.
    target_mod = importlib.import_module("tilelang.utils.target")
    assert hasattr(target_mod, "normalize_cutedsl_target")


def test_cppmega_cute_kernel_module_is_loadable() -> None:
    """The smallest cppmega CuTeDSL kernel can be located and (when
    cutlass is installed) imported as a Python module.

    Read-only: we never modify ``cute_dsl_mimo/*``. This proves task
    bullet #1 ("a CuTe DSL kernel from cppmega can be loaded") in the
    weakest non-trivial sense — source exists and is parseable. Actually
    importing requires ``cutlass`` and ``quack``; if either is missing,
    skip with a reason rather than fail.
    """

    if not _ensure_cppmega_cute_on_path():
        pytest.skip(
            reason=(
                f"cppmega CuTeDSL reference dir not present at "
                f"{CPPMEGA_CUTE_DIR}; this checkout does not include the "
                "cppmega submodule — bridge wiring still verifiable via the "
                "tilelang-side contract test, but kernel-load smoke needs "
                "the read-only cppmega tree."
            )
        )

    src_path = os.path.join(CPPMEGA_CUTE_DIR, f"{CPPMEGA_SMALLEST_KERNEL}.py")
    assert os.path.isfile(src_path), src_path

    # Source-level sanity: the file uses @cute.kernel / @cute.jit.
    with open(src_path, encoding="utf-8") as fh:
        src = fh.read()
    assert "@cute.kernel" in src
    assert "import cutlass" in src

    if importlib.util.find_spec("cutlass") is None:
        pytest.skip(
            reason=(
                "nvidia-cutlass-dsl (cutlass.cute) not installed; cppmega's "
                "@cute.kernel module cannot be imported. Install via "
                "`pip install -U 'nvidia-cutlass-dsl>=4.3.1,!=4.3.4'`."
            )
        )

    # quack is a runtime dep of single_gemm_test; surface that too.
    if importlib.util.find_spec("quack") is None:
        pytest.skip(
            reason=(
                "cppmega/cute_dsl_mimo/single_gemm_test.py imports `quack` "
                "(NVIDIA quack helpers); not installed on this host."
            )
        )

    mod = importlib.import_module(CPPMEGA_SMALLEST_KERNEL)
    assert hasattr(mod, "SingleGemmWGMMA"), (
        f"{CPPMEGA_SMALLEST_KERNEL}.py no longer exposes SingleGemmWGMMA — "
        "smallest-entrypoint pick has drifted; update CPPMEGA_SMALLEST_KERNEL."
    )


def test_external_cute_kernel_is_explicitly_unsupported() -> None:
    """Loud-failure contract for the *unsupported* import direction.

    ``cute_dsl_to_tilelang_prim`` must raise ``CuteBridgeUnsupported`` with
    a reason mentioning PR #1421 — callers should never get a silent
    None / empty stub.
    """

    # Import the bridge module directly via importlib so we don't pay the
    # cost of the cppmega_mlx.nn package __init__ (which eagerly imports
    # attention.py → mlx_lm; on hosts with an mlx version newer than the
    # pinned mlx_lm, that cascade raises AttributeError unrelated to this
    # bridge). Loading the bridge by file path keeps the test hermetic.
    import importlib.util
    import os

    bridge_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "cppmega_mlx",
        "nn",
        "_cute_bridge.py",
    )
    spec = importlib.util.spec_from_file_location(
        "_cute_bridge_under_test", bridge_path
    )
    assert spec is not None and spec.loader is not None
    bridge_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bridge_mod)

    with pytest.raises(bridge_mod.CuteBridgeUnsupported) as excinfo:
        bridge_mod.cute_dsl_to_tilelang_prim(object())
    msg = str(excinfo.value)
    assert "PR #1421" in msg
    assert "compile_prim_to_cutedsl" in msg or "dispatch_lower" in msg


def test_dispatch_lower_to_cutedsl_end_to_end() -> None:
    """End-to-end smoke for the supported direction:
    TileLang T.prim_func → ``dispatch_lower(prim, target='cutedsl')`` →
    runnable ``CuTeDSLKernelAdapter`` artifact.

    Skips loudly with a reason on Mac/non-CUDA hosts and on hosts missing
    nvidia-cutlass-dsl. Per memory rule, the skip reason must be specific.
    """

    if not torch.cuda.is_available():
        pytest.skip(
            reason=(
                "torch.cuda.is_available() is False on this host (likely "
                "macOS / MLX dev box); CuTeDSL kernels need a CUDA device "
                "(Hopper sm_90 for full feature parity)."
            )
        )

    from cppmega_mlx.nn._cute_bridge import (
        compile_prim_to_cutedsl,
        tilelang_cutedsl_available,
    )

    ok, reason = tilelang_cutedsl_available()
    if not ok:
        pytest.skip(
            reason=(
                "TileLang CuTeDSL backend not reachable on this host: "
                f"{reason}"
            )
        )

    # Force the engine path so dispatch_lower doesn't silently fall back to
    # the metal MSL shim (which is not applicable for target='cutedsl').
    prev = os.environ.get("CPPMEGA_MLX_TILELANG_ENGINE")
    os.environ["CPPMEGA_MLX_TILELANG_ENGINE"] = "engine"
    try:
        import tilelang.language as T

        M, N, K = 128, 128, 64
        block_M, block_N, block_K = 64, 64, 32

        @T.prim_func
        def gemm(
            A: T.Tensor((M, K), "float16"),
            B: T.Tensor((K, N), "float16"),
            C: T.Tensor((M, N), "float16"),
        ):
            with T.Kernel(
                T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128
            ) as (bx, by):
                A_sh = T.alloc_shared((block_M, block_K), "float16")
                B_sh = T.alloc_shared((block_K, block_N), "float16")
                C_loc = T.alloc_fragment((block_M, block_N), "float32")
                T.clear(C_loc)
                for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=2):
                    T.copy(A[by * block_M, k * block_K], A_sh)
                    T.copy(B[k * block_K, bx * block_N], B_sh)
                    T.gemm(A_sh, B_sh, C_loc, False, False)
                T.copy(C_loc, C[by * block_M, bx * block_N])

        artifact = compile_prim_to_cutedsl(gemm)

        # Adapter contract: artifact must be the cutedsl adapter (not a CUDA
        # one) and expose a callable forward.
        from tilelang.jit.adapter.cutedsl import CuTeDSLKernelAdapter

        assert isinstance(artifact, CuTeDSLKernelAdapter), (
            f"Expected CuTeDSLKernelAdapter, got {type(artifact).__name__}; "
            "target='cutedsl' did not route through the CuTeDSL backend."
        )
        assert callable(artifact)

        # Functional check: run the kernel and compare to torch.matmul.
        a = torch.randn(M, K, dtype=torch.float16, device="cuda")
        b = torch.randn(K, N, dtype=torch.float16, device="cuda")
        c = torch.empty(M, N, dtype=torch.float16, device="cuda")
        artifact(a, b, c)
        ref = (a.float() @ b.float()).to(torch.float16)
        torch.testing.assert_close(c, ref, atol=1e-2, rtol=1e-2)
    finally:
        if prev is None:
            os.environ.pop("CPPMEGA_MLX_TILELANG_ENGINE", None)
        else:
            os.environ["CPPMEGA_MLX_TILELANG_ENGINE"] = prev
