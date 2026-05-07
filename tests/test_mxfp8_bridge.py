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
