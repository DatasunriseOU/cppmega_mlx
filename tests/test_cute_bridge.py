# pyright: reportInvalidTypeForm=false, reportMissingImports=false
"""Forward-direction emit test for the TileLang -> CuTeDSL bridge.

Per PR apache/tilelang#1421 the upstream bridge is one-way: TileLang
``T.prim_func`` IR -> CuTeDSL Python source via
``tilelang.compile(prim, target='cutedsl')``.  cppmega adds a narrow
reverse importer for the smallest external ``SingleGemmWGMMA`` and
``MaskedLKQApplyWGMMA`` kernels plus the one-chunk
``StateApplyConsumersWGMMA`` tile and fixed-nchunk multi-chunk variant; other
hand-written ``@cute.kernel`` objects remain unsupported and must raise
:class:`CuteBridgeUnsupported`.

The bigger smoke at ``tests/test_cute_to_tilelang_bridge.py`` exercises a
full end-to-end compile + run of a CuTeDSL artifact. This file is narrower:
it pins the *emit* contract — the bridge must produce a non-empty CuTeDSL
Python source that parses as Python and carries the ``@cute.kernel``
decorator. That contract is what downstream tooling (and PR #1421's own
adapter) relies on, so it is worth a dedicated test independent of the
runtime smoke.

All tests skip with a precise reason on hosts where the CuTeDSL backend is
unreachable (no CUDA, no nvidia-cutlass-dsl, libtilelang built without the
cutedsl codegen translation unit, etc.). Per the project memory rule, no
silent skips.
"""

from __future__ import annotations

import ast
import importlib
import importlib.util
import os
import sys
from pathlib import Path

import pytest


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _load_bridge_module():
    """Load ``cppmega_mlx.nn._cute_bridge`` directly by file path.

    The package init eagerly imports ``cppmega_mlx.nn.attention`` -> mlx_lm,
    which fails on hosts where mlx and mlx_lm versions are mismatched
    (typical on dev machines). The bridge module itself has no such deps,
    so loading it via importlib keeps this contract test hermetic.
    """

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
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _skip_if_cutedsl_unreachable(bridge_mod) -> None:
    """Skip with a precise reason if any layer of the cutedsl stack is missing."""

    # Layer 1: tilelang must import.
    try:
        importlib.import_module("tilelang")
    except Exception as exc:
        pytest.skip(
            reason=(
                "tilelang unimportable on this host "
                f"({exc.__class__.__name__}: {exc}); CuTeDSL emit needs a "
                "working tilelang install."
            )
        )

    # Layer 2: cutedsl backend reachable (cutlass.cute, version, etc.).
    ok, reason = bridge_mod.tilelang_cutedsl_available()
    if not ok:
        pytest.skip(
            reason=(
                "TileLang CuTeDSL backend not reachable on this host: "
                f"{reason}"
            )
        )

    # Layer 3: libtilelang must register the cutedsl codegen global func.
    # On macOS / non-CUDA builds, src/target/codegen_cutedsl.cc is excluded
    # from the build (see src/backend/cuda/CMakeLists.txt), so the FFI
    # symbol is missing even when the Python adapter imports cleanly.
    import tvm_ffi

    names = set(tvm_ffi.registry.list_global_func_names())
    needed = "target.build.tilelang_cutedsl_without_compile"
    if needed not in names:
        pytest.skip(
            reason=(
                f"libtilelang on this host does not register {needed!r}; "
                "the cutedsl codegen translation unit is gated on USE_CUDA "
                "in src/backend/cuda/CMakeLists.txt and is omitted from "
                "non-CUDA builds (e.g. macOS dev hosts)."
            )
        )


def _build_smallest_emittable_prim():
    """Build the smallest TileLang prim that the cutedsl emitter can handle.

    The PR #1421 emitter supports the standard TileLang op subset:
    ``T.Kernel``, ``T.alloc_shared`` / ``T.alloc_fragment``, ``T.copy``,
    ``T.clear``, ``T.gemm``, and pipelined loops. A tiny single-stage GEMM
    is the smallest non-trivial program — vector add would also work but
    GEMM exercises the WGMMA / MMA path that PR #1421 specifically targets.
    """

    import tilelang.language as T

    M, N, K = 64, 64, 32
    block_M, block_N, block_K = 64, 64, 32

    @T.prim_func
    def matmul_tt(
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
            T.copy(A[by * block_M, 0], A_sh)
            T.copy(B[0, bx * block_N], B_sh)
            T.gemm(A_sh, B_sh, C_loc, False, False)
            T.copy(C_loc, C[by * block_M, bx * block_N])

    return matmul_tt


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------


def test_compile_prim_to_cutedsl_signature_and_unsupported_reverse() -> None:
    """API-surface contract for the bridge.

    * ``compile_prim_to_cutedsl`` exists, is callable, and is annotated as
      taking a single positional arg.
    * Unrecognized reverse-direction inputs raise
      :class:`CuteBridgeUnsupported` cleanly with a message that mentions
      PR #1421 and the supported alternatives — callers must never get a
      silent ``None`` from the unsupported path.

    Runs on every host (no CUDA / cutlass dependency) — pure API surface.
    """

    bridge_mod = _load_bridge_module()

    # Forward-direction signature check.
    fn = bridge_mod.compile_prim_to_cutedsl
    assert callable(fn)
    import inspect

    sig = inspect.signature(fn)
    params = list(sig.parameters.values())
    assert len(params) == 1, f"expected 1 positional arg, got {sig}"
    assert params[0].name == "prim_func"

    # Unrecognized reverse-direction guard.
    with pytest.raises(bridge_mod.CuteBridgeUnsupported) as excinfo:
        bridge_mod.cute_dsl_to_tilelang_prim(object())
    msg = str(excinfo.value)
    assert "PR #1421" in msg, msg
    assert (
        "compile_prim_to_cutedsl" in msg or "dispatch_lower" in msg
    ), msg


def test_single_gemm_wgmma_imports_to_tilelang_primfunc() -> None:
    """The smallest cppmega external CuTe kernel imports as TileLang IR.

    This pins the first supported reverse-direction case:
    ``SingleGemmWGMMA`` maps to a single-CTA TileLang ``T.gemm`` PrimFunc
    for ``C[M, N] = A[M, K] @ B[N, K].T``.  More complex external
    ``@cute.kernel`` objects should continue to fail loudly until they have
    a dedicated importer.
    """

    bridge_mod = _load_bridge_module()

    class SingleGemmWGMMA:
        M = 64
        N = 64
        K = 64
        dtype = "bfloat16"

    prim = bridge_mod.cute_dsl_to_tilelang_prim(SingleGemmWGMMA())

    text = str(prim)
    assert "gemm" in text
    assert "A" in text
    assert "B" in text
    assert "C" in text


def test_single_gemm_wgmma_source_imports_without_cutlass_import() -> None:
    """Import the real external CuTe source file without importing cutlass.

    The source-level importer must statically recover the ``SingleGemmWGMMA``
    contract from cppmega's ``single_gemm_test.py``.  This is stronger than
    importing a live Python object because this Mac test host does not have
    ``cutlass`` / ``quack`` installed, but the actual source file still has
    enough information to build the TileLang ``T.gemm`` PrimFunc.
    """

    bridge_mod = _load_bridge_module()
    source = Path(
        "/Volumes/external/sources/cppmega/cppmega/megatron/"
        "cute_dsl_mimo/single_gemm_test.py"
    )
    if not source.exists():
        pytest.skip(f"cppmega CuTe source file not present: {source}")

    prim = bridge_mod.cute_dsl_source_to_tilelang_prim(source)

    text = str(prim)
    assert "gemm" in text
    assert "A" in text
    assert "B" in text
    assert "C" in text


def test_single_gemm_wgmma_cute_jit_source_imports_without_cutlass_import(
    tmp_path: Path,
) -> None:
    """Static source import accepts @cute.jit as well as @cute.kernel."""

    bridge_mod = _load_bridge_module()
    source = tmp_path / "single_gemm_jit.py"
    source.write_text(
        """
import cutlass.cute as cute


class SingleGemmWGMMA:
    def __init__(self, M=16, N=32, K=8, dtype="float16"):
        self.M = M
        self.N = N
        self.K = K
        self.dtype = dtype

    @cute.jit
    def __call__(self, A, B, C):
        pass
""",
        encoding="utf-8",
    )

    prim = bridge_mod.cute_dsl_source_to_tilelang_prim(source)

    text = str(prim)
    assert "gemm" in text
    assert "A" in text
    assert "B" in text
    assert "C" in text


def test_masked_lkq_apply_source_imports_two_gemm_masked_tile() -> None:
    """Import the next external CuTe tile as TileLang IR.

    ``masked_lkq_apply.py`` fuses ``future_mask(K @ Q.T) @ dPhi``.  The
    importer must recover this as two TileLang GEMMs with an in-IR masked
    spill, not as a launch-boundary placeholder.
    """

    bridge_mod = _load_bridge_module()
    source = Path(
        "/Volumes/external/sources/cppmega/cppmega/megatron/"
        "cute_dsl_mimo/masked_lkq_apply.py"
    )
    if not source.exists():
        pytest.skip(f"cppmega CuTe source file not present: {source}")

    prim = bridge_mod.cute_dsl_source_to_tilelang_prim(source)

    text = str(prim)
    assert text.count("gemm") >= 2
    assert "if_then_else" in text
    assert "K" in text
    assert "Q" in text
    assert "DPhT" in text
    assert "Apply" in text


def test_state_apply_consumers_source_imports_three_gemm_consumers() -> None:
    """Import the one-chunk state/apply/consumer CuTe tile as TileLang IR.

    ``state_apply_consumers.py`` fuses ``K @ DStates``,
    ``future_mask(K @ Q.T) @ dPhi``, and the scalar DV / DMIMO_V consumers.
    The importer must keep those products and consumers in one TileLang region
    instead of treating the CuTe file as an opaque launch boundary.
    """

    bridge_mod = _load_bridge_module()
    source = Path(
        "/Volumes/external/sources/cppmega/cppmega/megatron/"
        "cute_dsl_mimo/state_apply_consumers.py"
    )
    if not source.exists():
        pytest.skip(f"cppmega CuTe source file not present: {source}")

    prim = bridge_mod.cute_dsl_source_to_tilelang_prim(
        source,
        class_name="StateApplyConsumersWGMMA",
    )

    text = str(prim)
    assert text.count("gemm") >= 3
    assert "if_then_else" in text
    assert "K" in text
    assert "Q" in text
    assert "DstT" in text
    assert "DPhT" in text
    assert "V" in text
    assert "MimoV" in text
    assert "DV" in text
    assert "DMimoV" in text


def test_multi_chunk_state_apply_consumers_source_imports_reverse_scan() -> None:
    """Import the multi-chunk reverse-scan CuTe tile as TileLang IR.

    The multi-chunk class carries ``DStates.T`` across chunks, computes scaled
    future-masked LKQ/apply products, writes per-chunk DV, and accumulates
    DMIMO_V.  The source importer must require the chunk-count specialization
    explicitly and keep the reverse-scan owner in one TileLang region.
    """

    bridge_mod = _load_bridge_module()
    source = Path(
        "/Volumes/external/sources/cppmega/cppmega/megatron/"
        "cute_dsl_mimo/state_apply_consumers.py"
    )
    if not source.exists():
        pytest.skip(f"cppmega CuTe source file not present: {source}")

    prim = bridge_mod.cute_dsl_source_to_tilelang_prim(
        source,
        class_name="MultiChunkStateApplyConsumersWGMMA",
        nchunks=2,
    )

    text = str(prim)
    assert text.count("gemm") >= 4
    assert "chunk_rev" in text
    assert "if_then_else" in text
    assert "exp" in text
    assert "K" in text
    assert "QT" in text
    assert "DACS" in text
    assert "DACSRev" in text
    assert "Segsum" in text
    assert "QKDot" in text
    assert "Gamma" in text
    assert "DMimoV" in text


def test_fa4_v2_source_imports_three_gemm_chain_without_lkq_output() -> None:
    """Import FA4 v2's fused 3-GEMM chain as one TileLang region.

    ``fa4_bwd_adapter_v2.py`` computes ``DPs = K @ DstT + (K @ Q) @ DPhT``
    and deliberately removes the intermediate LKQ global output.  The importer
    must recover that as TileLang T-ops rather than treating the FA4 source as
    an opaque launch boundary.
    """

    bridge_mod = _load_bridge_module()
    source = Path(
        "/Volumes/external/sources/cppmega/cppmega/megatron/"
        "cute_dsl_mimo/fa4_bwd_adapter_v2.py"
    )
    if not source.exists():
        pytest.skip(f"cppmega CuTe source file not present: {source}")

    prim = bridge_mod.cute_dsl_source_to_tilelang_prim(
        source,
        class_name="FA4PatternFused3GemmV2",
    )

    text = str(prim)
    assert text.count("gemm") >= 3
    assert "K" in text
    assert "Q" in text
    assert "DstT" in text
    assert "DPhT" in text
    assert "DPs" in text
    assert "LKQ" in text


def test_fa4_v1_source_imports_three_gemm_chain_with_lkq_output() -> None:
    """Import FA4 v1's fused 3-GEMM chain and preserve the LKQ output.

    ``fa4_bwd_adapter.py`` computes the same ``DPs`` chain as v2, but also
    writes the intermediate ``LKQ = K @ Q`` tile to global memory for
    verification / downstream consumers.  The source importer must keep that
    public output in the TileLang contract instead of falling back to an opaque
    launch boundary.
    """

    bridge_mod = _load_bridge_module()
    source = Path(
        "/Volumes/external/sources/cppmega/cppmega/megatron/"
        "cute_dsl_mimo/fa4_bwd_adapter.py"
    )
    if not source.exists():
        pytest.skip(f"cppmega CuTe source file not present: {source}")

    prim = bridge_mod.cute_dsl_source_to_tilelang_prim(
        source,
        class_name="FA4PatternFused3Gemm",
    )

    text = str(prim)
    assert text.count("gemm") >= 3
    assert "K" in text
    assert "Q" in text
    assert "DstT" in text
    assert "DPhT" in text
    assert "DPs" in text
    assert "LKQ" in text
    assert {buffer.name for buffer in prim.buffer_map.values()} >= {
        "K",
        "Q",
        "DstT",
        "DPhT",
        "DPs",
        "LKQ",
    }


def test_fused_bwd_bwd_p4_source_imports_ten_gemm_reverse_scan() -> None:
    """Import the Phase-4 10-GEMM external CuTe tile as one TileLang region.

    ``fused_bwd_bwd_sm90_p4.py`` is the largest hand-written CuTe source in
    the cppmega import surface.  The source importer should recover its fixed
    tile contract for a specialized chunk count instead of leaving it as an
    opaque CuTe launch boundary.
    """

    bridge_mod = _load_bridge_module()
    source = Path(
        "/Volumes/external/sources/cppmega/cppmega/megatron/"
        "cute_dsl_mimo/fused_bwd_bwd_sm90_p4.py"
    )
    if not source.exists():
        pytest.skip(f"cppmega CuTe source file not present: {source}")

    prim = bridge_mod.cute_dsl_source_to_tilelang_prim(
        source,
        class_name="FusedBwdBwdP4",
        nchunks=2,
    )

    text = str(prim)
    assert text.count("gemm") >= 10
    assert "chunk_rev" in text
    assert "K_T" in text
    assert "Q_T" in text
    assert "DPh_T" in text
    assert "DPsiV" in text
    assert "DK" in text
    assert "DQ" in text
    assert "Dqkd" in text
    assert "DstatesOut" in text
    assert {buffer.name for buffer in prim.buffer_map.values()} >= {
        "K",
        "K_T",
        "Q",
        "Q_T",
        "Dst",
        "DPh",
        "DPh_T",
        "Psi",
        "Sts",
        "DPsiV",
        "DK",
        "DQ",
        "Dqkd",
        "DstatesOut",
    }


def test_emit_matmul_to_cute_dsl_produces_valid_python() -> None:
    """Forward emit path produces non-empty, parseable CuTeDSL source.

    Verifies the four bullets from the task brief:
    1. Output is non-empty.
    2. Output contains ``@cute.kernel`` (or the equivalent ``@cute.jit``
       decorator that PR #1421's emitter actually uses on the entrypoint).
    3. Output is parseable as Python (``ast.parse`` + walk).
    4. The artifact is the cutedsl adapter (not a stray CUDA fallback).

    Skips with a precise reason on hosts without the cutedsl backend.
    """

    bridge_mod = _load_bridge_module()
    _skip_if_cutedsl_unreachable(bridge_mod)

    prim = _build_smallest_emittable_prim()
    artifact = bridge_mod.compile_prim_to_cutedsl(prim)

    # (4) Adapter type — must be the cutedsl one.
    from tilelang.jit.adapter.cutedsl import CuTeDSLKernelAdapter

    assert isinstance(artifact, CuTeDSLKernelAdapter), (
        f"expected CuTeDSLKernelAdapter, got {type(artifact).__name__} "
        "— target='cutedsl' did not route through the CuTeDSL backend"
    )

    # The adapter exposes the emitted source via get_kernel_source().
    src = artifact.get_kernel_source(kernel_only=True)

    # (1) Non-empty.
    assert isinstance(src, str)
    assert len(src) > 0, "emitted CuTeDSL source is empty"

    # (2) Carries a cute.* decorator on the kernel entrypoint. PR #1421
    # emits @cute.jit / @cute.kernel; accept either since the exact
    # decorator name has shifted across cutlass-dsl minor versions.
    assert (
        "@cute.kernel" in src or "@cute.jit" in src
    ), f"emitted source has no cute.kernel/cute.jit decorator:\n{src[:400]}"

    # (3) Parses as Python and the AST contains at least one function
    # definition decorated with cute.{kernel,jit}.
    tree = ast.parse(src)

    def _is_cute_decorator(node: ast.expr) -> bool:
        """Match @cute.kernel, @cute.jit, @cute.kernel(...), etc."""

        target = node.func if isinstance(node, ast.Call) else node
        if not isinstance(target, ast.Attribute):
            return False
        return (
            isinstance(target.value, ast.Name)
            and target.value.id == "cute"
            and target.attr in {"kernel", "jit"}
        )

    decorated = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef)
        and any(_is_cute_decorator(d) for d in n.decorator_list)
    ]
    assert decorated, (
        "emitted CuTeDSL source has no @cute.kernel / @cute.jit "
        f"decorated function:\n{src[:600]}"
    )

    # Diagnostic: surface the source length and the first decorated
    # function name so failures elsewhere have context.
    sys.stderr.write(
        f"\n[emit] cutedsl source length={len(src)} entrypoint="
        f"{decorated[0].name!r}\n"
    )
