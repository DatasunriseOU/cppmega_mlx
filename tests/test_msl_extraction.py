"""Phase-3 migration adapter tests: MSL extraction from engine artifacts.

These tests exercise the bridge that lets ``_msl_transform.lower_tilelang_to_msl_inline``
callers flip to the unified ``tilelang.engine.lower(target="metal")`` path
without losing access to the MSL text shape consumed by
``mx.fast.metal_kernel(...)``.

Coverage:
  * ``test_supports_msl_extraction_probe_does_not_raise`` — cheap import probe.
  * ``test_extract_returns_none_for_non_metal_target`` — CUDA/HIP yield None.
  * ``test_extract_returns_lowering_from_kernel_source`` — happy path with a
    minimal stub artifact (covers the no-Metal-host case via duck-typing).
  * ``test_extract_returns_none_when_kernel_source_missing`` — graceful
    degradation, emits UserWarning.
  * ``test_dispatch_lower_engine_with_msl_mode_routes_through_extraction`` —
    end-to-end env-driven path; falls back to shim on extraction failure
    with a single warning.
  * ``test_dispatch_lower_return_msl_kwarg_routes_through_extraction`` — same,
    but driven by the kwarg instead of the env mode.
"""

from __future__ import annotations

import warnings

import pytest


pytestmark = pytest.mark.filterwarnings("default::UserWarning")


# ---------------------------------------------------------------------------
# Probe / non-metal target
# ---------------------------------------------------------------------------


def test_supports_msl_extraction_probe_does_not_raise() -> None:
    """The probe is a cheap import-check; it must never raise.

    Returns True iff ``import tilelang`` succeeds; we assert only the
    no-raise contract here so the test passes on every dev host (with or
    without tilelang installed).
    """

    from cppmega_mlx.nn._tilelang._msl_extraction import supports_msl_extraction

    result = supports_msl_extraction()
    assert isinstance(result, bool)


def test_extract_returns_none_for_non_metal_target() -> None:
    """CUDA / HIP artifacts must yield ``None`` (no MSL text exists)."""

    from cppmega_mlx.nn._tilelang._msl_extraction import (
        extract_msl_from_engine_artifact,
    )

    class _FakeArtifact:
        kernel_source = "// some non-metal source"

    assert extract_msl_from_engine_artifact(_FakeArtifact(), target="cuda") is None
    assert extract_msl_from_engine_artifact(_FakeArtifact(), target="hip") is None


# ---------------------------------------------------------------------------
# Happy path: minimal stub artifact carrying realistic MSL text
# ---------------------------------------------------------------------------


# A trimmed-but-realistic MSL kernel taken from TileLang's metal codegen
# shape. The body has a single threadgroup statement so
# ``_inline_tilelang_kernel_body`` has something concrete to inline.
_STUB_MSL = """\
#include <metal_stdlib>
using namespace metal;

kernel void kernel_main_stub(
  device float* a [[buffer(0)]],
  device float* b [[buffer(1)]],
  device float* c [[buffer(2)]],
  uint3 blockIdx [[threadgroup_position_in_grid]],
  uint3 threadIdx [[thread_position_in_threadgroup]]
) {
  threadgroup float shared_buf[16];
  shared_buf[threadIdx.x] = a[threadIdx.x] + b[threadIdx.x];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  c[threadIdx.x] = shared_buf[threadIdx.x];
}
"""


class _StubArtifact:
    """Quacks like a tilelang.compile artifact for parsing-time tests."""

    def __init__(self, source: str = _STUB_MSL) -> None:
        self.kernel_source = source
        # device_mod is intentionally absent -> grid/threadgroup default to (1,1,1).


def test_extract_returns_lowering_from_kernel_source() -> None:
    """Happy path: stub artifact with kernel_source produces a TileLangMSLLowering."""

    from cppmega_mlx.nn._tilelang._msl_extraction import (
        extract_msl_from_engine_artifact,
    )
    from cppmega_mlx.nn._tilelang._msl_transform import TileLangMSLLowering

    lowering = extract_msl_from_engine_artifact(_StubArtifact(), target="metal")
    assert isinstance(lowering, TileLangMSLLowering)
    assert lowering.kernel_name == "kernel_main_stub"
    assert lowering.msl_text == _STUB_MSL
    # Header must contain the prelude (#include) but not the kernel signature.
    assert "metal_stdlib" in lowering.header
    assert "kernel void" not in lowering.header
    # Body has been inlined: no leading "{" / trailing "}" and contains the
    # threadgroup write.
    assert "shared_buf" in lowering.body
    # Buffer names must be picked up via the existing parser.
    assert lowering.buffer_param_names  # non-empty list


def test_extract_returns_none_when_kernel_source_missing() -> None:
    """Missing kernel_source yields None + a single UserWarning."""

    from cppmega_mlx.nn._tilelang._msl_extraction import (
        extract_msl_from_engine_artifact,
    )

    class _Empty:
        pass

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = extract_msl_from_engine_artifact(_Empty(), target="metal")
    assert result is None
    assert any(
        "kernel_source" in str(w.message) and issubclass(w.category, UserWarning)
        for w in caught
    )


def test_extract_metal_target_via_tvm_target_object_shape() -> None:
    """The is-metal probe must accept tvm.target.Target-like objects."""

    from cppmega_mlx.nn._tilelang._msl_extraction import (
        extract_msl_from_engine_artifact,
    )

    class _Kind:
        name = "metal"

    class _MetalTarget:
        kind = _Kind()

    lowering = extract_msl_from_engine_artifact(_StubArtifact(), target=_MetalTarget())
    assert lowering is not None
    assert lowering.kernel_name == "kernel_main_stub"


# ---------------------------------------------------------------------------
# End-to-end: dispatch_lower env / kwarg routing
# ---------------------------------------------------------------------------


def test_dispatch_lower_engine_with_msl_mode_falls_back_on_import_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``CPPMEGA_MLX_TILELANG_ENGINE=engine_with_msl_extraction`` falls back
    to the shim with a single UserWarning when tilelang is unimportable.

    We force the ImportError by patching the engine helper, which is the
    cheapest way to validate the fallback contract without requiring a
    Metal host.
    """

    from cppmega_mlx.nn._tilelang import _engine_dispatch
    from cppmega_mlx.nn._tilelang._msl_transform import TileLangMSLLowering

    monkeypatch.setenv("CPPMEGA_MLX_TILELANG_ENGINE", "engine_with_msl_extraction")
    _engine_dispatch._reset_fallback_warning_for_tests()

    def _raise(*_args, **_kwargs):
        raise ImportError("tilelang unavailable in this test")

    monkeypatch.setattr(_engine_dispatch, "_engine_compile", _raise)

    class _FakeMSLLowering(TileLangMSLLowering):
        pass

    fake_lowering = TileLangMSLLowering(
        header="// stub",
        body="{}",
        grid=(1, 1, 1),
        threadgroup=(1, 1, 1),
        msl_text=_STUB_MSL,
        buffer_param_names=["a", "b", "c"],
        kernel_name="stub",
    )

    monkeypatch.setattr(
        _engine_dispatch, "_shim_lower", lambda *_a, **_kw: fake_lowering
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = _engine_dispatch.dispatch_lower(object(), target="metal")

    assert result is fake_lowering
    fallback_warnings = [
        w for w in caught
        if "engine_with_msl_extraction" in str(w.message)
        and issubclass(w.category, UserWarning)
    ]
    assert fallback_warnings, "expected one-shot fallback warning"


def test_dispatch_lower_return_msl_kwarg_falls_back_on_extraction_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``return_msl=True`` falls back to shim if extraction returns None."""

    from cppmega_mlx.nn._tilelang import _engine_dispatch
    from cppmega_mlx.nn._tilelang._msl_transform import TileLangMSLLowering

    monkeypatch.setenv("CPPMEGA_MLX_TILELANG_ENGINE", "auto")
    _engine_dispatch._reset_fallback_warning_for_tests()

    class _NoSourceArtifact:
        pass

    monkeypatch.setattr(
        _engine_dispatch,
        "_engine_compile",
        lambda *_a, **_kw: _NoSourceArtifact(),
    )

    fake_lowering = TileLangMSLLowering(
        header="// shim-fallback",
        body="{}",
        grid=(1, 1, 1),
        threadgroup=(1, 1, 1),
        msl_text=_STUB_MSL,
        buffer_param_names=[],
        kernel_name="shim_fallback",
    )
    monkeypatch.setattr(
        _engine_dispatch, "_shim_lower", lambda *_a, **_kw: fake_lowering
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = _engine_dispatch.dispatch_lower(
            object(), target="metal", return_msl=True
        )

    assert result is fake_lowering
    assert any(
        "engine_with_msl_extraction" in str(w.message)
        and issubclass(w.category, UserWarning)
        for w in caught
    )


def test_dispatch_lower_return_msl_kwarg_returns_extracted_lowering(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``return_msl=True`` returns the extracted MSL lowering on the happy path."""

    from cppmega_mlx.nn._tilelang import _engine_dispatch

    monkeypatch.setenv("CPPMEGA_MLX_TILELANG_ENGINE", "auto")
    _engine_dispatch._reset_fallback_warning_for_tests()

    monkeypatch.setattr(
        _engine_dispatch,
        "_engine_compile",
        lambda *_a, **_kw: _StubArtifact(),
    )
    # Sentinel: shim must NOT be reached on the happy path.
    monkeypatch.setattr(
        _engine_dispatch,
        "_shim_lower",
        lambda *_a, **_kw: pytest.fail("shim should not be invoked"),
    )

    result = _engine_dispatch.dispatch_lower(
        object(), target="metal", return_msl=True
    )
    assert getattr(result, "kernel_name", None) == "kernel_main_stub"
    assert getattr(result, "msl_text", "") == _STUB_MSL
