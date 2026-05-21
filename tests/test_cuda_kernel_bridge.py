"""Tests for ``cppmega_mlx.nn._cuda_kernel_bridge`` (arch-aware dispatcher).

These tests are designed to run on a Mac dev host (no CUDA): the
detection function is exercised directly, and arch-specific routing is
verified with explicit target-arch arguments so the test suite does not
require a GPU.
"""

from __future__ import annotations

from pathlib import Path
import warnings

import pytest

from cppmega_mlx.nn import _cuda_kernel_bridge as ckb


def _clear_arch_cache() -> None:
    """Clear the lru_cache on detect_cuda_arch (test-only helper)."""
    ckb.detect_cuda_arch.cache_clear()


def test_detect_cuda_arch_returns_cpu_on_mac() -> None:
    """On a Mac dev host (no CUDA) detection must yield exactly 'cpu'."""
    _clear_arch_cache()
    arch = ckb.detect_cuda_arch()
    # Either we're on Mac (no CUDA) or torch isn't installed -- both
    # collapse to 'cpu'. We deliberately do not pytest.skip; the contract
    # is that absence-of-CUDA is a first-class supported result.
    assert arch == "cpu", (
        f"expected 'cpu' on this host (no CUDA), got {arch!r}; "
        f"if you are running on a CUDA host, adjust this test to use "
        f"an explicit no-CUDA detector seam rather than patching torch.cuda."
    )


def test_dispatch_unsupported_combo_raises_loud() -> None:
    """(arch=cpu, kind=triton) must raise CUDABridgeUnsupported with reason."""
    _clear_arch_cache()
    with pytest.raises(ckb.CUDABridgeUnsupported) as excinfo:
        ckb.dispatch_kernel_bridge("triton")
    msg = str(excinfo.value)
    assert "no CUDA" in msg, f"reason should mention no-CUDA, got {msg!r}"
    assert "triton" in msg, f"reason should mention requested kind, got {msg!r}"


def test_arch_bridge_table_keys_are_known_arches() -> None:
    """Exhaustive check that the dispatch table covers exactly the documented arches."""
    expected = {"sm_89", "sm_90", "sm_90a", "sm_100", "sm_120", "sm_121", "cpu"}
    assert set(ckb.ARCH_BRIDGE_TABLE) == expected, (
        f"ARCH_BRIDGE_TABLE drift: expected {expected!r}, "
        f"got {set(ckb.ARCH_BRIDGE_TABLE)!r}"
    )

    # Per-arch invariants -- pin the support matrix so silent regressions
    # produce a noisy diff.
    assert ckb.ARCH_BRIDGE_TABLE["sm_89"] == frozenset({"triton"})
    assert ckb.ARCH_BRIDGE_TABLE["sm_90"] == frozenset({"triton", "cute_dsl"})
    assert ckb.ARCH_BRIDGE_TABLE["sm_90a"] == frozenset({"triton", "cute_dsl"})
    assert ckb.ARCH_BRIDGE_TABLE["sm_100"] == frozenset(
        {"triton", "mxfp8", "mxfp8_grouped"}
    )
    assert ckb.ARCH_BRIDGE_TABLE["sm_120"] == frozenset(
        {"triton", "mxfp8", "mxfp8_grouped"}
    )
    assert ckb.ARCH_BRIDGE_TABLE["sm_121"] == frozenset(
        {"triton", "mxfp8", "mxfp8_grouped"}
    )
    assert ckb.ARCH_BRIDGE_TABLE["cpu"] == frozenset()


def test_unknown_arch_falls_back_to_sm_120_with_warning() -> None:
    """A future arch (e.g. sm_130) must warn AND fall back to sm_120 routing."""
    result = None
    err = None
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            result = ckb.dispatch_kernel_bridge(
                "mxfp8",
                arch="sm_130",
                m=128,
                n=128,
                k=128,
            )
        except RuntimeError as exc:
            err = exc

    runtime_warnings = [w for w in caught if issubclass(w.category, RuntimeWarning)]
    assert runtime_warnings, "expected a RuntimeWarning for unknown arch"
    assert any("sm_130" in str(w.message) for w in runtime_warnings), (
        f"warning should mention the unknown arch; got "
        f"{[str(w.message) for w in runtime_warnings]!r}"
    )
    assert any(
        ckb.FALLBACK_ARCH in str(w.message) for w in runtime_warnings
    ), "warning should mention the fallback arch"
    if err is None:
        assert result is not None
    else:
        assert "sm_120" in str(err) or "GB10" in str(err)


def test_mxfp8_blocked_on_hopper() -> None:
    """MXFP8 must loud-fail on sm_90a -- Hopper has no MXFP8 instructions."""
    with pytest.raises(ckb.CUDABridgeUnsupported) as excinfo:
        ckb.dispatch_kernel_bridge("mxfp8", arch="sm_90a")
    msg = str(excinfo.value)
    assert "sm_90a" in msg
    assert "mxfp8" in msg

    with pytest.raises(ckb.CUDABridgeUnsupported) as excinfo2:
        ckb.dispatch_kernel_bridge("mxfp8_grouped", arch="sm_90a")
    assert "sm_90a" in str(excinfo2.value)


def test_unknown_kind_raises_loud() -> None:
    """An unknown ``kind`` argument must surface a precise error."""
    with pytest.raises(ckb.CUDABridgeUnsupported) as excinfo:
        ckb.dispatch_kernel_bridge("not_a_real_kind")
    msg = str(excinfo.value)
    assert "not_a_real_kind" in msg
    assert "expected one of" in msg


def test_cute_dsl_source_import_routes_by_explicit_target_arch() -> None:
    """External CuTe source import must be reachable through the dispatcher.

    This uses an explicit target arch so the Mac/no-CUDA host can exercise the
    production routing path without patching runtime CUDA detection.
    """

    source = Path(
        "/Volumes/external/sources/cppmega/cppmega/megatron/"
        "cute_dsl_mimo/single_gemm_test.py"
    )
    if not source.exists():
        pytest.skip(f"cppmega CuTe source file not present: {source}")

    prim = ckb.dispatch_kernel_bridge(
        "cute_dsl",
        arch="sm_90a",
        source_path=source,
    )

    text = str(prim)
    assert "gemm" in text
    assert "A" in text
    assert "B" in text
    assert "C" in text


def test_cute_dsl_phase4_source_import_routes_by_explicit_target_arch() -> None:
    """Dispatcher reaches the largest static CuTe source importer on Mac."""

    source = Path(
        "/Volumes/external/sources/cppmega/cppmega/megatron/"
        "cute_dsl_mimo/fused_bwd_bwd_sm90_p4.py"
    )
    if not source.exists():
        pytest.skip(f"cppmega CuTe source file not present: {source}")

    prim = ckb.dispatch_kernel_bridge(
        "cute_dsl",
        arch="sm_90a",
        source_path=source,
        class_name="FusedBwdBwdP4",
        nchunks=2,
    )

    text = str(prim)
    assert text.count("gemm") >= 10
    assert "chunk_rev" in text
    assert "DstatesOut" in text
