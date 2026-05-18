"""Path D runtime adapter tests.

These tests lock the cppmega-side boundary: Triton frontend may produce a
PrimFunc, but cppmega owns grid specialization, artifact caching, launch
eligibility, and recurrent-signature orchestration.
"""

from __future__ import annotations

from dataclasses import dataclass

from cppmega_v4._tilelang.kda_path_d import _path_d_runtime_status as kda_status
from cppmega_v4._tilelang.linear_attention_path_d import (
    _path_d_runtime_status as gdn_status,
)


class _FakeVar:
    def __init__(self, name: str):
        self.name = name

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

    assert ok_gdn is False
    assert ok_kda is False
    assert "runtime adapter" in reason_gdn
    assert "runtime adapter" in reason_kda
