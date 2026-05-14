"""Tests for the native TileLang TVM-FFI runtime boundary."""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest


class _RecordingArtifact:
    def __init__(self, result: Any | None = None) -> None:
        self.calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
        self.result = result

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.calls.append((args, dict(kwargs)))
        if self.result is not None:
            return self.result
        if "out" in kwargs:
            return kwargs["out"]
        return args[-1]


def test_native_kernel_requires_owner_outputs_by_default() -> None:
    from cppmega_mlx.nn._tilelang._mlx_runtime import (
        NativeTileLangKernel,
        NativeTileLangRuntimeError,
    )

    artifact = _RecordingArtifact()
    kernel = NativeTileLangKernel(
        artifact=artifact,
        result_indices=(1,),
        num_params=2,
        target="metal",
    )

    with pytest.raises(NativeTileLangRuntimeError, match="out="):
        kernel(object())

    assert artifact.calls == []


def test_native_kernel_dispatches_with_owner_output_and_checks_identity() -> None:
    from cppmega_mlx.nn._tilelang._mlx_runtime import NativeTileLangKernel

    artifact = _RecordingArtifact()
    kernel = NativeTileLangKernel(
        artifact=artifact,
        result_indices=(1,),
        num_params=2,
        target="metal",
    )
    source = object()
    out = object()

    returned = kernel(source, out=out)

    assert returned is out
    assert artifact.calls == [((source,), {"out": out})]


def test_native_kernel_rejects_wrong_owner_output_identity() -> None:
    from cppmega_mlx.nn._tilelang._mlx_runtime import (
        NativeTileLangKernel,
        NativeTileLangRuntimeError,
    )

    artifact = _RecordingArtifact(result=object())
    kernel = NativeTileLangKernel(
        artifact=artifact,
        result_indices=(1,),
        num_params=2,
        target="metal",
    )

    with pytest.raises(NativeTileLangRuntimeError, match="caller-owned output"):
        kernel(object(), out=object())


def test_native_kernel_accepts_explicit_full_abi_owner_output() -> None:
    from cppmega_mlx.nn._tilelang._mlx_runtime import NativeTileLangKernel

    artifact = _RecordingArtifact()
    kernel = NativeTileLangKernel(
        artifact=artifact,
        result_indices=(1,),
        num_params=2,
        target="metal",
    )
    source = object()
    out = object()

    returned = kernel(source, out)

    assert returned is out
    assert artifact.calls == [((source, out), {})]


def test_normalize_out_idx_rejects_duplicates_and_out_of_range() -> None:
    from cppmega_mlx.nn._tilelang._mlx_runtime import (
        NativeTileLangRuntimeError,
        normalize_out_idx,
    )

    assert normalize_out_idx(-1, num_params=4) == (3,)
    assert normalize_out_idx([1, -1], num_params=4) == (1, 3)

    with pytest.raises(NativeTileLangRuntimeError, match="outside"):
        normalize_out_idx(-5, num_params=4)
    with pytest.raises(NativeTileLangRuntimeError, match="duplicate"):
        normalize_out_idx([1, -3], num_params=4)


def test_compile_native_tilelang_kernel_uses_tvm_ffi_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cppmega_mlx.nn._tilelang import _engine_dispatch
    from cppmega_mlx.nn._tilelang._mlx_runtime import NativeTileLangKernel
    import cppmega_mlx.nn._tilelang._msl_transform as _msl_transform

    calls: list[dict[str, Any]] = []
    artifact = _RecordingArtifact()

    def fake_compile(*args: Any, **kwargs: Any) -> _RecordingArtifact:
        calls.append({"args": args, **kwargs})
        return artifact

    fake_tilelang = types.SimpleNamespace(compile=fake_compile)
    monkeypatch.setitem(sys.modules, "tilelang", fake_tilelang)
    monkeypatch.setattr(
        _engine_dispatch,
        "_ensure_path_c_metal_intrinsics_registered",
        lambda: None,
    )
    monkeypatch.setattr(
        _msl_transform,
        "_ensure_single_libtvm_ffi_image",
        lambda: None,
    )

    class _Prim:
        params = (object(), object())

    kernel = _engine_dispatch.compile_native_tilelang_kernel(
        _Prim(),
        target="metal",
        out_idx=-1,
    )

    assert isinstance(kernel, NativeTileLangKernel)
    assert kernel.result_indices == (1,)
    assert calls and calls[0]["execution_backend"] == "tvm_ffi"
    assert calls[0]["out_idx"] == -1
    assert calls[0]["target"] == "metal"
