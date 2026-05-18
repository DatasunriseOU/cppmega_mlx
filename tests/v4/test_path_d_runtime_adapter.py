"""Path D runtime adapter tests.

These tests lock the cppmega-side boundary: Triton frontend may produce a
PrimFunc, but cppmega owns grid specialization, artifact caching, launch
eligibility, and recurrent-signature orchestration.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import pytest

from cppmega_v4._tilelang.kda_path_d import _path_d_runtime_status as kda_status
from cppmega_v4._tilelang.linear_attention_path_d import (
    _path_d_runtime_status as gdn_status,
)


class _FakeVar:
    def __init__(self, name: str, dtype: str = ""):
        self.name = name
        self.dtype = dtype

    def __str__(self) -> str:
        return self.name


class _FakePrim:
    def __init__(self, text: str = ""):
        self.params = (
            _FakeVar("arg0"),
            _FakeVar("gridDim_0"),
            _FakeVar("gridDim_1"),
        )
        self.text = text
        self.specialized = None

    def script(self) -> str:
        return self.text

    def specialize(self, mapping):
        self.specialized = {str(k): v for k, v in mapping.items()}
        return self


class _FakePrimWithScalar:
    def __init__(self):
        self.arg = _FakeVar("arg0")
        self.scalar = _FakeVar("T")
        self.params = (self.arg, self.scalar)
        self.buffer_map = {self.arg: object()}
        self.specialized = None

    def script(self) -> str:
        return ""

    def specialize(self, mapping):
        self.specialized = {str(k): v for k, v in mapping.items()}
        return self


class _FakePrimWithScaleAndT:
    def __init__(self):
        self.arg = _FakeVar("arg0")
        self.scale = _FakeVar("scale", "float32")
        self.t = _FakeVar("T", "int32")
        self.params = (self.arg, self.scale, self.t)
        self.buffer_map = {self.arg: object()}
        self.specialized = None

    def script(self) -> str:
        return ""

    def specialize(self, mapping):
        self.specialized = {str(k): v for k, v in mapping.items()}
        return self


@dataclass
class _CompileRecorder:
    called: bool = False

    def __call__(self, *args, **kwargs):
        self.called = True
        return object()


def test_runtime_adapter_specializes_grid_params_before_compile():
    from cppmega_v4._tilelang.path_d_runtime_adapter import (
        PathDKernelPlan,
        compile_tilelang_primfunc,
    )

    prim = _FakePrim()
    recorder = _CompileRecorder()
    result = compile_tilelang_primfunc(
        prim,
        PathDKernelPlan(name="fake", out_idx=(0,), grid=(3, 5)),
        compile_fn=recorder,
    )

    assert result.available is True
    assert recorder.called is True
    assert prim.specialized == {"gridDim_0": 3, "gridDim_1": 5}


def test_runtime_adapter_specializes_static_scalar_params_before_compile():
    from cppmega_v4._tilelang.path_d_runtime_adapter import (
        PathDKernelPlan,
        compile_tilelang_primfunc,
    )

    prim = _FakePrimWithScalar()
    recorder = _CompileRecorder()
    result = compile_tilelang_primfunc(
        prim,
        PathDKernelPlan(
            name="fake",
            out_idx=(0,),
            grid=(1, 1),
            scalar_specializations=(64,),
        ),
        compile_fn=recorder,
    )

    assert result.available is True
    assert recorder.called is True
    assert prim.specialized == {"T": 64}


def test_runtime_adapter_specializes_static_t_not_scale():
    from cppmega_v4._tilelang.path_d_runtime_adapter import (
        PathDKernelPlan,
        compile_tilelang_primfunc,
    )

    prim = _FakePrimWithScaleAndT()
    recorder = _CompileRecorder()
    result = compile_tilelang_primfunc(
        prim,
        PathDKernelPlan(
            name="fake",
            out_idx=(0,),
            grid=(1, 1, 1),
            scalar_specializations=(64,),
        ),
        compile_fn=recorder,
    )

    assert result.available is True
    assert recorder.called is True
    assert prim.specialized == {"T": 64}


def test_runtime_adapter_specializes_scale_and_static_t():
    from cppmega_v4._tilelang.path_d_runtime_adapter import (
        PathDKernelPlan,
        compile_tilelang_primfunc,
    )

    prim = _FakePrimWithScaleAndT()
    recorder = _CompileRecorder()
    result = compile_tilelang_primfunc(
        prim,
        PathDKernelPlan(
            name="fake",
            out_idx=(0,),
            grid=(1, 1, 1),
            scalar_specializations=(0.125, 64),
        ),
        compile_fn=recorder,
    )

    assert result.available is True
    assert recorder.called is True
    assert prim.specialized == {"scale": 0.125, "T": 64}


def test_runtime_adapter_blocks_degraded_primfunc_before_compile():
    from cppmega_v4._tilelang.path_d_runtime_adapter import (
        PathDKernelPlan,
        compile_tilelang_primfunc,
    )

    prim = _FakePrim("# DEGRADED: tt.addptr without PtrAnalysis shim")
    recorder = _CompileRecorder()
    result = compile_tilelang_primfunc(
        prim,
        PathDKernelPlan(name="fake", out_idx=(0,), grid=(1, 1)),
        compile_fn=recorder,
    )

    assert result.available is False
    assert recorder.called is False
    assert "DEGRADED" in result.reason


def test_path_d_statuses_are_runtime_adapter_driven():
    ok_gdn, reason_gdn = gdn_status()
    ok_kda, reason_kda = kda_status()

    assert isinstance(ok_gdn, bool)
    assert ok_kda is False
    assert "runtime adapter" in reason_gdn
    assert "runtime adapter" in reason_kda


def test_gdn_chunk_o_metal_threadgroup_memory_fits_device_limit():
    pytest.importorskip("tilelang")
    from cppmega_v4._tilelang.path_d_runtime_adapter import (
        compile_gdn_chunk_o_artifact,
    )

    result = compile_gdn_chunk_o_artifact()
    if not result.available or result.artifact is None:
        pytest.skip(result.reason)
    source = result.artifact.kernel_source or ""
    match = re.search(r"threadgroup uchar buf_shmem\[(\d+)\]", source)

    assert match is not None
    assert int(match.group(1)) <= 32 * 1024


def test_gdn_kkt_lowering_is_non_degraded():
    pytest.importorskip("tilelang")
    from cppmega_v4._tilelang.linear_attention_path_d_real import (
        lower_fla_gdn_kkt_solve,
    )

    result = lower_fla_gdn_kkt_solve(grid=(1, 1))
    assert result.status == "LOWERED_FULL"
    assert result.prim_func is not None
    assert "DEGRADED" not in result.prim_func.script()


def test_gdn_runtime_adapter_launches_fixed_prefill_smoke():
    pytest.importorskip("tilelang")
    import mlx.core as mx

    from cppmega_v4._tilelang.path_d_runtime_adapter import (
        gdn_fwd_runtime_call,
        gdn_runtime_adapter_status,
    )

    ok, reason = gdn_runtime_adapter_status()
    if not ok:
        pytest.skip(reason)

    q = mx.zeros((1, 64, 1, 64), dtype=mx.float16)
    k = mx.zeros((1, 64, 1, 64), dtype=mx.float16)
    v = mx.zeros((1, 64, 1, 32), dtype=mx.float16)
    beta = mx.zeros((1, 64, 1), dtype=mx.float32)
    g = mx.zeros((1, 64, 1), dtype=mx.float32)

    y, final_state = gdn_fwd_runtime_call(
        q,
        k,
        v,
        beta,
        g,
        output_final_state=True,
    )
    mx.eval(y, final_state)

    assert y.shape == (1, 64, 1, 32)
    assert y.dtype == mx.float16
    assert final_state is not None
    assert final_state.shape == (1, 1, 64, 32)
    assert final_state.dtype == mx.float32
