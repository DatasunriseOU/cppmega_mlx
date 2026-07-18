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


class _ArrayHandle:
    def __init__(self, shape: tuple[int, ...], dtype: str) -> None:
        self.shape = shape
        self.dtype = dtype


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


def test_native_kernel_graph_output_route_is_explicit_and_skips_tvm_ffi() -> None:
    from cppmega_mlx.nn._tilelang._mlx_runtime import NativeTileLangKernel

    artifact = _RecordingArtifact()
    graph_calls: list[tuple[Any, ...]] = []

    def graph_runner(inputs: tuple[Any, ...]) -> list[str]:
        graph_calls.append(inputs)
        return ["graph-result"]

    kernel = NativeTileLangKernel(
        artifact=artifact,
        result_indices=(1,),
        num_params=2,
        target="metal",
        allow_graph_outputs=True,
        graph_runner=graph_runner,
    )
    source = object()

    assert kernel(source) == "graph-result"
    assert graph_calls == [(source,)]
    assert artifact.calls == []


def test_native_graph_launch_metadata_failure_is_not_reparsed_as_primfunc() -> None:
    from cppmega_mlx.nn._tilelang._mlx_runtime import _native_graph_launch_config

    class _Adapter:
        def _metal_launch_config(self):
            raise RuntimeError("corrupt launch receipt")

    artifact = types.SimpleNamespace(
        adapter=_Adapter(),
        prim_func=types.SimpleNamespace(
            script=lambda: 'T.launch_thread("blockIdx.x", 8)'
        ),
    )

    with pytest.raises(RuntimeError, match="corrupt launch receipt"):
        _native_graph_launch_config(artifact)


def test_native_graph_launch_metadata_is_validated_before_graph_launch() -> None:
    from cppmega_mlx.nn._tilelang._mlx_runtime import (
        NativeTileLangRuntimeError,
        _native_graph_launch_config,
    )

    class _Adapter:
        def _metal_launch_config(self):
            return (8, 1), (0, 1, 1)

    artifact = types.SimpleNamespace(adapter=_Adapter())
    with pytest.raises(NativeTileLangRuntimeError, match="positive extents"):
        _native_graph_launch_config(artifact)


def test_native_graph_launch_uses_primfunc_only_when_helper_declares_unavailable() -> None:
    from cppmega_mlx.nn._tilelang._mlx_runtime import _native_graph_launch_config

    class _Adapter:
        def _metal_launch_config(self):
            raise NotImplementedError

    artifact = types.SimpleNamespace(
        adapter=_Adapter(),
        prim_func=types.SimpleNamespace(
            script=lambda: (
                'T.launch_thread("blockIdx.x", 8)\n'
                'T.launch_thread("threadIdx.x", 32)'
            )
        ),
    )
    assert _native_graph_launch_config(artifact) == ((8, 1, 1), (32, 1, 1))


def test_native_graph_result_unwraps_single_output_and_validates_count() -> None:
    from cppmega_mlx.nn._tilelang._mlx_runtime import (
        NativeTileLangRuntimeError,
        _normalize_graph_result,
    )

    output = object()
    assert _normalize_graph_result([output], output_count=1) is output
    assert _normalize_graph_result(output, output_count=1) is output
    pair = [object(), object()]
    assert _normalize_graph_result(pair, output_count=2) is pair

    with pytest.raises(NativeTileLangRuntimeError, match="expected 1"):
        _normalize_graph_result(pair, output_count=1)
    with pytest.raises(NativeTileLangRuntimeError, match="expected 2"):
        _normalize_graph_result(output, output_count=2)


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


def test_native_kernel_returns_owner_output_when_tvm_ffi_returns_fresh_handle() -> None:
    from cppmega_mlx.nn._tilelang._mlx_runtime import NativeTileLangKernel

    out = _ArrayHandle((2, 3), "float32")
    fresh_handle = _ArrayHandle((2, 3), "float32")
    artifact = _RecordingArtifact(result=fresh_handle)
    kernel = NativeTileLangKernel(
        artifact=artifact,
        result_indices=(1,),
        num_params=2,
        target="metal",
    )

    returned = kernel(object(), out=out)

    assert returned is out


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
