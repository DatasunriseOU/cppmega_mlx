"""Tests for ``cppmega_mlx.nn._cuda_kernel_bridge`` (arch-aware dispatcher).

These tests are designed to run on a Mac dev host (no CUDA): the
detection function is exercised directly, and arch-specific routing is
verified via monkeypatching so the test suite does not require a GPU.
"""

from __future__ import annotations

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
        f"if you are running on a CUDA host, this test should be "
        f"adjusted to monkeypatch torch.cuda."
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


def test_unknown_arch_falls_back_to_sm_120_with_warning(monkeypatch) -> None:
    """A future arch (e.g. sm_130) must warn AND fall back to sm_120 routing."""
    monkeypatch.setattr(ckb, "detect_cuda_arch", lambda: "sm_130")

    # Stub out the MXFP8 bridge so we can observe a successful sm_120
    # routing decision without a real CUDA host. We use a small fake module.
    class _FakeMxfp8Bridge:
        @staticmethod
        def mxfp8_to_tilelang_extern(*, marker: str = "fallback") -> str:
            return f"mxfp8::{marker}"

    import sys

    monkeypatch.setitem(
        sys.modules, "cppmega_mlx.nn._mxfp8_bridge", _FakeMxfp8Bridge
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = ckb.dispatch_kernel_bridge("mxfp8", marker="ok")

    assert result == "mxfp8::ok"
    runtime_warnings = [w for w in caught if issubclass(w.category, RuntimeWarning)]
    assert runtime_warnings, "expected a RuntimeWarning for unknown arch"
    assert any("sm_130" in str(w.message) for w in runtime_warnings), (
        f"warning should mention the unknown arch; got "
        f"{[str(w.message) for w in runtime_warnings]!r}"
    )
    assert any(
        ckb.FALLBACK_ARCH in str(w.message) for w in runtime_warnings
    ), "warning should mention the fallback arch"


def test_mxfp8_blocked_on_hopper(monkeypatch) -> None:
    """MXFP8 must loud-fail on sm_90a -- Hopper has no MXFP8 instructions."""
    monkeypatch.setattr(ckb, "detect_cuda_arch", lambda: "sm_90a")

    with pytest.raises(ckb.CUDABridgeUnsupported) as excinfo:
        ckb.dispatch_kernel_bridge("mxfp8")
    msg = str(excinfo.value)
    assert "sm_90a" in msg
    assert "mxfp8" in msg

    with pytest.raises(ckb.CUDABridgeUnsupported) as excinfo2:
        ckb.dispatch_kernel_bridge("mxfp8_grouped")
    assert "sm_90a" in str(excinfo2.value)


def test_unknown_kind_raises_loud() -> None:
    """An unknown ``kind`` argument must surface a precise error."""
    with pytest.raises(ckb.CUDABridgeUnsupported) as excinfo:
        ckb.dispatch_kernel_bridge("not_a_real_kind")
    msg = str(excinfo.value)
    assert "not_a_real_kind" in msg
    assert "expected one of" in msg
