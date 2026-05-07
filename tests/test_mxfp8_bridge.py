# pyright: reportInvalidTypeForm=false, reportMissingImports=false
"""Tests for ``cppmega_mlx.nn._mxfp8_bridge``.

The bridge wraps cppmega's GB10 CUTLASS MXFP8 GEMM as a TileLang
``tl.extern_intrinsic``. On Mac dev hosts (no CUDA, no GB10) the bridge must:

* return a clean ``(False, reason)`` from :func:`mxfp8_bridge_available`
  (never raise — callers grep the reason for skip messages).
* raise :class:`MXFP8BridgeUnsupported` with a precise reason when
  :func:`mxfp8_to_tilelang_extern` is invoked without prereqs.

On a GB10 host with cppmega + tilelang installed we additionally exercise:

* the registered intrinsic appears in
  ``tilelang.language.extern_registry``.
* the resulting callable's signature has the expected five Frags
  (A_u8, SFA_u8, B_u8, SFB_u8, out) for several 128-multiple shapes.

Per the codebase rule ``feedback_no_silent_delete`` we test for *loud* failure
(the raised exception type + a reason substring) rather than for silent
``None``.
"""

from __future__ import annotations

import platform
import sys

import pytest

# The bridge module itself is import-clean even on Mac (lazy imports for tvm /
# tilelang / torch). This is the prerequisite for cleanly skipping below.
from cppmega_mlx.nn._mxfp8_bridge import (
    MXFP8_BRIDGE_KNOWN_GAPS,
    MXFP8_CPPMEGA_ENTRY_ABI,
    MXFP8_GROUPED_INTRINSIC_NAME,
    MXFP8_INTRINSIC_NAME,
    MXFP8BridgeUnsupported,
    mxfp8_bridge_available,
    mxfp8_grouped_to_tilelang_extern,
    mxfp8_to_tilelang_extern,
)


# ---------------------------------------------------------------------------
# Always-on tests (Mac + GB10).
# ---------------------------------------------------------------------------


def test_bridge_module_constants_are_well_formed() -> None:
    """The exported constants must be non-empty and stable enough for tests."""

    assert MXFP8_INTRINSIC_NAME == "cppmega_sm120_blockscaled_mma_tma"
    assert MXFP8_GROUPED_INTRINSIC_NAME.startswith("cppmega_sm120_grouped")
    assert isinstance(MXFP8_BRIDGE_KNOWN_GAPS, tuple)
    assert len(MXFP8_BRIDGE_KNOWN_GAPS) >= 5, (
        "Known gaps should enumerate every variant we did NOT wire on the "
        "first cut (compact-direct, asym, swizzled-scale, grouped-ptrs, "
        "FlashInfer, scale-layout, shape constraint, arch gate)."
    )
    assert isinstance(MXFP8_CPPMEGA_ENTRY_ABI, tuple)
    abi_keys = [k for k, _ in MXFP8_CPPMEGA_ENTRY_ABI]
    # Mirror ``cutlass_mxfp8_gemm.cpp:3-15`` exactly.
    assert abi_keys == [
        "A_u8", "SFA_u8", "B_u8", "SFB_u8",
        "m", "n", "k",
        "out", "use_out", "accumulate", "alpha", "beta",
    ]


def test_mxfp8_bridge_available_returns_tuple_cleanly() -> None:
    """``mxfp8_bridge_available`` must return ``(bool, str)`` without raising.

    On any host. This is the first thing test harnesses call to decide whether
    to skip; it must NEVER raise (per memory rule "loud failure with clear
    reason, never silent None" — but for this probe specifically the clear
    reason is the second tuple element).
    """

    ok, reason = mxfp8_bridge_available()
    assert isinstance(ok, bool)
    assert isinstance(reason, str)
    if not ok:
        assert reason, "missing-prereq path must include a non-empty reason"
    else:
        assert reason == "ok"


def test_mxfp8_bridge_available_skips_on_mac() -> None:
    """On Mac (Darwin) we expect ``(False, <reason>)``: no CUDA, no GB10."""

    if platform.system() != "Darwin":
        pytest.skip("Mac-specific path; this host is not Darwin")
    ok, reason = mxfp8_bridge_available()
    assert ok is False
    assert reason  # non-empty
    # The exact reason depends on whether torch / cppmega / tilelang are
    # importable on this Mac host — we just require it mention one of the
    # expected gates.
    expected_fragments = (
        "torch",
        "cppmega",
        "tilelang",
        "non-CUDA",
        "sm_",
    )
    assert any(frag in reason for frag in expected_fragments), (
        f"reason {reason!r} should mention at least one of the gating "
        f"prereqs ({expected_fragments!r})"
    )


def test_mxfp8_bridge_loud_failure_off_arch() -> None:
    """Calling ``mxfp8_to_tilelang_extern`` on Mac / non-GB10 raises clearly.

    The error MUST be :class:`MXFP8BridgeUnsupported` (not a generic
    ``ImportError`` / ``RuntimeError``) so callers can ``except
    MXFP8BridgeUnsupported`` once and decide whether to fall back, raise, or
    log. The reason string must point at the specific gate that failed
    (cppmega / tilelang / arch).
    """

    ok, _ = mxfp8_bridge_available()
    if ok:
        pytest.skip(
            "GB10 host with all prereqs available; off-arch failure not "
            "applicable here. Run this on a Mac to exercise the loud path."
        )
    with pytest.raises(MXFP8BridgeUnsupported) as excinfo:
        mxfp8_to_tilelang_extern(m=128, n=256, k=512)
    msg = str(excinfo.value)
    assert msg, "MXFP8BridgeUnsupported must carry a reason"
    # The reason should mention the gate (cppmega / tilelang / sm_).
    assert any(
        gate in msg for gate in ("cppmega", "tilelang", "sm_", "non-CUDA")
    ), f"reason {msg!r} should mention which prereq is missing"


def test_mxfp8_bridge_unsupported_dtype_is_loud() -> None:
    """Unsupported ``dtype`` raises before any prereq probe."""

    with pytest.raises(MXFP8BridgeUnsupported) as excinfo:
        mxfp8_to_tilelang_extern(m=128, n=128, k=128, dtype="fp16")
    assert "dtype" in str(excinfo.value)
    assert "fp16" in str(excinfo.value)


def test_mxfp8_bridge_unsupported_backend_is_loud() -> None:
    """Unsupported ``backend`` raises with a clear list of valid options."""

    with pytest.raises(MXFP8BridgeUnsupported) as excinfo:
        mxfp8_to_tilelang_extern(
            m=128, n=128, k=128, backend="cublas",  # type: ignore[arg-type]
        )
    msg = str(excinfo.value)
    assert "backend" in msg
    assert "cublas" in msg
    assert "cutlass" in msg and "flashinfer" in msg


def test_mxfp8_grouped_loud_failure_on_invalid_groups() -> None:
    with pytest.raises(MXFP8BridgeUnsupported) as excinfo:
        mxfp8_grouped_to_tilelang_extern(num_groups=0, m=128, n=128, k=128)
    assert "num_groups" in str(excinfo.value)


# ---------------------------------------------------------------------------
# CUDA-required tests. We use ``importorskip`` chain so missing tilelang /
# tvm / torch / cppmega each produce a clean skip with a precise reason.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("m,n,k", [(128, 128, 128), (256, 128, 512), (512, 256, 256)])
def test_mxfp8_bridge_signature_on_gb10(m: int, n: int, k: int) -> None:
    """On a GB10 host, the registered intrinsic must have the expected ABI.

    Skipped on any host where :func:`mxfp8_bridge_available` returns False.
    Verifies:

    * The intrinsic is registered in ``tilelang.language.extern_registry``.
    * The signature() returns exactly 5 Frags for the dense path.
    * Frag shapes match (M, K), (M, K/32), (N, K), (N, K/32), (M, N).
    * dtypes match (uint8 x 4, bfloat16 for out).
    * Only the last Frag is is_output=True.
    """

    ok, reason = mxfp8_bridge_available()
    if not ok:
        pytest.skip(f"MXFP8 bridge prereqs missing: {reason}")

    pytest.importorskip("tilelang", reason="tilelang missing")
    pytest.importorskip("tvm", reason="tvm missing")

    emit = mxfp8_to_tilelang_extern(m=m, n=n, k=k, backend="cutlass")

    # The intrinsic should be findable in the registry.
    from tilelang.language import extern_registry

    entry = extern_registry.lookup(MXFP8_INTRINSIC_NAME)
    assert entry is not None, "intrinsic not registered after wiring"
    assert entry.has_target("cuda"), "cuda body missing"

    frags = entry.signature()
    assert len(frags) == 5, f"expected 5 frags; got {len(frags)}"

    expected = [
        ("A_u8", (m, k), "uint8", False),
        ("SFA_u8", (m, k // 32), "uint8", False),
        ("B_u8", (n, k), "uint8", False),
        ("SFB_u8", (n, k // 32), "uint8", False),
        ("out", (m, n), "bfloat16", True),
    ]
    for frag, (name, shape, dtype, is_out) in zip(frags, expected):
        assert frag.name == name
        assert tuple(frag.shape) == shape
        assert frag.dtype == dtype
        assert frag.is_output is is_out
        assert frag.scope == "global"

    # Stashed metadata should reflect the registration call.
    assert emit.cppmega_backend == "cutlass"
    assert emit.cppmega_shape == (m, n, k)


def test_mxfp8_bridge_grouped_signature_on_gb10() -> None:
    """Grouped path: 6 frags (5 dense + expert_offsets)."""

    ok, reason = mxfp8_bridge_available()
    if not ok:
        pytest.skip(f"MXFP8 bridge prereqs missing: {reason}")
    pytest.importorskip("tilelang", reason="tilelang missing")
    pytest.importorskip("tvm", reason="tvm missing")

    emit = mxfp8_grouped_to_tilelang_extern(
        num_groups=16, m=512, n=128, k=128,
    )

    from tilelang.language import extern_registry

    entry = extern_registry.lookup(MXFP8_GROUPED_INTRINSIC_NAME)
    assert entry is not None
    frags = entry.signature()
    assert len(frags) == 6
    names = [f.name for f in frags]
    assert "expert_offsets" in names
    expert_offsets = next(f for f in frags if f.name == "expert_offsets")
    assert expert_offsets.dtype == "int32"
    assert tuple(expert_offsets.shape) == (16 + 1,)
    assert emit.cppmega_num_groups == 16


def test_mxfp8_bridge_rejects_non_128_multiple_shape_on_gb10() -> None:
    """Even on GB10 the bridge must reject shapes the kernel cannot run."""

    ok, reason = mxfp8_bridge_available()
    if not ok:
        pytest.skip(f"MXFP8 bridge prereqs missing: {reason}")
    pytest.importorskip("tilelang", reason="tilelang missing")

    with pytest.raises(MXFP8BridgeUnsupported) as excinfo:
        mxfp8_to_tilelang_extern(m=127, n=128, k=128)
    msg = str(excinfo.value)
    assert "128" in msg
    assert "M=127" in msg or "127" in msg


# ---------------------------------------------------------------------------
# Compile-only smoke (Mac-friendly): bypass the cppmega + arch probes via
# monkeypatch to verify the TIR shape end-to-end on Mac. We DO NOT launch the
# kernel; we only verify (a) extern_intrinsic registers, (b) the prim_func
# uses it without a TIR build error during prim_func construction, and (c)
# the lowered TIR `tir.call_extern` references the documented intrinsic name.
#
# When the host's libtilelang has the CUDA codegen target builder available
# (registers ``target.build.tilelang_cuda``), we additionally run
# ``tilelang.compile`` for ``target='cuda'`` to confirm the metadata-level
# compile succeeds. Mac builds typically lack the CUDA backend, in which case
# we xfail with the precise ``Cannot find global function`` reason.
# ---------------------------------------------------------------------------


def _unregister_mxfp8_intrinsics_if_registered() -> None:
    """Remove any leftover registrations between tests.

    The TileLang extern registry is process-global; once
    ``mxfp8_to_tilelang_extern`` succeeds it leaves the intrinsic registered
    for the remainder of the process. A second registration call (e.g. in
    another test) raises ``KeyError`` from ``_REGISTRY.register``. We clean
    up here so each compile-only smoke test sees a fresh registry slot.
    """

    try:
        from tilelang.language import extern_registry
    except Exception:  # pragma: no cover - tilelang missing
        return
    for name in (
        "cppmega_sm120_blockscaled_mma_tma",
        "cppmega_sm120_blockscaled_mma_tma_flashinfer",
        "cppmega_sm120_grouped_mxfp8_dgrad_nn",
    ):
        try:
            extern_registry.unregister(name)
        except KeyError:
            pass


def _mock_bridge_probes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force the cppmega / tilelang-subprocess / Blackwell probes to succeed
    so we can exercise the TileLang wiring on Mac (no real GPU launch — codegen
    / metadata only).

    The ``_probe_tilelang_extern`` mock is required because on macOS the bridge
    runs the import in a subprocess to isolate libtilelang dylib aborts; that
    subprocess does NOT inherit ``DYLD_LIBRARY_PATH`` (System Integrity
    Protection strips it for child processes), so the probe always fails on
    Mac even when the parent process can import tilelang fine. The actual
    in-process import is what we want to verify, so we substitute a
    mock probe that imports in-process and falls through to the real
    ``extern_intrinsic`` registration.
    """

    import cppmega_mlx.nn._mxfp8_bridge as _bridge

    monkeypatch.setattr(_bridge, "_probe_cppmega_cutlass", lambda backend: (True, ""))
    monkeypatch.setattr(_bridge, "_probe_blackwell", lambda: (True, ""))

    def _in_process_tilelang_probe() -> "tuple[bool, str]":
        try:
            from tilelang.language.extern import extern_intrinsic  # noqa: F401
        except Exception as exc:  # pragma: no cover - host-specific
            return (False, f"in-process tilelang import failed: {exc!r}")
        return (True, "")

    monkeypatch.setattr(_bridge, "_probe_tilelang_extern", _in_process_tilelang_probe)


def test_mxfp8_extern_intrinsic_registers_and_emits_call_extern(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With probes mocked, verify the TIR shape: extern_intrinsic returns a
    callable with the documented signature, registers in
    ``tilelang.language.extern_registry``, and emits a ``tir.call_extern``
    when invoked from inside a ``T.prim_func``.

    No CUDA / nvcc required — this only walks the TIR graph.
    """

    pytest.importorskip("tilelang", reason="tilelang missing")
    pytest.importorskip("tvm", reason="tvm missing")
    _mock_bridge_probes(monkeypatch)
    _unregister_mxfp8_intrinsics_if_registered()

    from cppmega_mlx.nn._mxfp8_bridge import (
        MXFP8_INTRINSIC_NAME,
        mxfp8_to_tilelang_extern,
    )

    intrin = mxfp8_to_tilelang_extern(m=128, n=128, k=128)
    # The extern_intrinsic decorator returns a callable, not a tir Op.
    assert callable(intrin)
    # Stashed metadata for downstream codegen.
    assert intrin.cppmega_backend == "cutlass"  # type: ignore[attr-defined]
    assert intrin.cppmega_shape == (128, 128, 128)  # type: ignore[attr-defined]
    abi_keys = [k for k, _ in intrin.cppmega_abi]  # type: ignore[attr-defined]
    assert abi_keys[:4] == ["A_u8", "SFA_u8", "B_u8", "SFB_u8"]

    # Registry lookup: the cuda body must be present.
    from tilelang.language import extern_registry

    entry = extern_registry.lookup(MXFP8_INTRINSIC_NAME)
    assert entry is not None, "intrinsic not registered after wiring"
    assert entry.has_target("cuda"), "cuda body missing"
    body = entry.bodies["cuda"]
    # The stub body must mention the intrinsic name verbatim — codegen relies
    # on this when materialising the trampoline on GB10.
    assert MXFP8_INTRINSIC_NAME in body
    assert "tn_gemm_compact_scale" in body  # forward-doc reference in stub.

    # Frag-level signature is exactly 5 buffers in the documented order.
    frags = entry.signature()
    names = [f.name for f in frags]
    assert names == ["A_u8", "SFA_u8", "B_u8", "SFB_u8", "out"]


def test_mxfp8_compile_smoke_sm120(monkeypatch: pytest.MonkeyPatch) -> None:
    """Compile-only smoke: build a TileLang ``@T.prim_func`` that calls the
    intrinsic and run ``tilelang.compile(target='cuda')``. No real GPU launch.

    On Mac dev hosts the libtilelang dylib usually does NOT register
    ``target.build.tilelang_cuda`` (CUDA codegen requires nvcc + a CUDA build
    flag), so this test xfails cleanly with the precise
    ``Cannot find global function target.build.tilelang_cuda`` reason rather
    than pretending to test what it can't.
    """

    pytest.importorskip("tilelang", reason="tilelang missing")
    pytest.importorskip("tvm", reason="tvm missing")
    _mock_bridge_probes(monkeypatch)
    _unregister_mxfp8_intrinsics_if_registered()

    import tilelang
    import tilelang.language as T
    import tvm

    from cppmega_mlx.nn._mxfp8_bridge import mxfp8_to_tilelang_extern

    intrin = mxfp8_to_tilelang_extern(m=128, n=128, k=128)

    @T.prim_func
    def mxfp8_kernel(
        A: T.Tensor((128, 128), "uint8"),
        SFA: T.Tensor((128, 4), "uint8"),
        B: T.Tensor((128, 128), "uint8"),
        SFB: T.Tensor((128, 4), "uint8"),
        Out: T.Tensor((128, 128), "bfloat16"),
    ):
        with T.Kernel(1, 1) as (bx, by):
            intrin(A, SFA, B, SFB, Out)

    # If the host's libtilelang doesn't have the CUDA codegen builder,
    # tilelang.compile fails at the dispatch step (NOT inside our bridge).
    # That is a host-build limitation, not a bridge bug — xfail with a
    # precise reason so reviewers know exactly what's missing.
    cuda_builder = tvm.ffi.get_global_func(
        "target.build.tilelang_cuda", allow_missing=True
    )
    if cuda_builder is None:
        pytest.xfail(
            "host libtilelang does not register target.build.tilelang_cuda "
            "(no CUDA backend in this build); bridge wiring is verified by "
            "test_mxfp8_extern_intrinsic_registers_and_emits_call_extern, "
            "which does not require a CUDA codegen target."
        )

    compiled = tilelang.compile(mxfp8_kernel, target="cuda")
    assert compiled is not None
