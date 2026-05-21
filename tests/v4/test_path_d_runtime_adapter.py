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


def _count_named_buffer_loads(prim_func, names: set[str]) -> int:
    from tvm.tir import stmt_functor

    count = 0

    def visit(node):
        nonlocal count
        buffer = getattr(node, "buffer", None)
        if buffer is not None and str(getattr(buffer, "name", "")) in names:
            count += 1

    stmt_functor.post_order_visit(prim_func.body, visit)
    return count


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


def test_kda_topology_policy_promotes_only_repeated_varlen_layout():
    from cppmega_v4._tilelang.path_d_runtime_adapter import (
        _kda_varlen_topology_key,
        _record_kda_topology_hit,
        _reset_kda_topology_cache_for_tests,
    )

    _reset_kda_topology_cache_for_tests()
    key = _kda_varlen_topology_key(
        cu_values=(0, 16, 64),
        chunk_indices=(0, 0, 1, 0, 1, 1),
        chunk_offsets=(0, 1, 3),
        total_tokens=64,
        h_heads=1,
        hv_heads=1,
        k_dim=64,
        v_dim=32,
        scale=0.125,
        use_initial_state=True,
        output_final_state=True,
    )

    first = _record_kda_topology_hit(key, threshold=2)
    second = _record_kda_topology_hit(key, threshold=2)
    other = _record_kda_topology_hit(
        _kda_varlen_topology_key(
            cu_values=(0, 32, 64),
            chunk_indices=(0, 0, 1, 0),
            chunk_offsets=(0, 1, 2),
            total_tokens=64,
            h_heads=1,
            hv_heads=1,
            k_dim=64,
            v_dim=32,
            scale=0.125,
            use_initial_state=True,
            output_final_state=True,
        ),
        threshold=2,
    )

    assert first.hits == 1
    assert first.use_specialized is False
    assert second.hits == 2
    assert second.use_specialized is True
    assert other.hits == 1
    assert other.use_specialized is False


def test_kda_topology_hit_cache_eviction_is_lru():
    from cppmega_v4._tilelang.path_d_runtime_adapter import (
        _kda_varlen_topology_key,
        _record_kda_topology_hit,
        _reset_kda_topology_cache_for_tests,
    )

    _reset_kda_topology_cache_for_tests()

    def make_key(cu_values, chunk_offsets):
        return _kda_varlen_topology_key(
            cu_values=cu_values,
            chunk_indices=(),
            chunk_offsets=chunk_offsets,
            total_tokens=64,
            h_heads=1,
            hv_heads=1,
            k_dim=64,
            v_dim=32,
            scale=None,
            use_initial_state=False,
            output_final_state=False,
        )

    key_a = make_key((0, 16, 64), (0, 1, 3))
    key_b = make_key((0, 32, 64), (0, 1, 2))
    key_c = make_key((0, 8, 64), (0, 1, 3))

    _record_kda_topology_hit(key_a, threshold=10, max_entries=2)
    _record_kda_topology_hit(key_b, threshold=10, max_entries=2)
    _record_kda_topology_hit(key_a, threshold=10, max_entries=2)
    _record_kda_topology_hit(key_c, threshold=10, max_entries=2)
    b_after_eviction = _record_kda_topology_hit(
        key_b, threshold=10, max_entries=2,
    )

    assert b_after_eviction.hits == 1


def test_kda_compact_topology_invariants_are_not_exact_arrays():
    from cppmega_v4._tilelang.path_d_runtime_adapter import (
        _kda_compact_topology_invariants,
    )

    invariants = _kda_compact_topology_invariants(
        cu_values=(0, 16, 64),
        chunk_offsets=(0, 1, 3),
        total_tokens=64,
        chunk_size=32,
    )

    assert invariants.bounds_valid is True
    assert invariants.cu_monotonic is True
    assert invariants.chunk_offsets_monotonic is True
    assert invariants.total_tokens == 64
    assert invariants.num_sequences == 2
    assert invariants.num_chunks == 3
    assert invariants.max_sequence_tokens == 48
    assert invariants.max_chunk_id == 2


def test_kda_topology_fingerprint_covers_specialization_contract():
    from cppmega_v4._tilelang.path_d_runtime_adapter import (
        _kda_varlen_topology_descriptor,
        _kda_varlen_topology_fingerprint,
    )

    descriptor = _kda_varlen_topology_descriptor(
        cu_values=(0, 16, 64),
        chunk_offsets=(0, 1, 3),
        total_tokens=64,
        h_heads=1,
        hv_heads=2,
        k_dim=64,
        v_dim=32,
        chunk_size=32,
        scale=0.125,
        use_initial_state=True,
        output_final_state=False,
    )
    fingerprint = _kda_varlen_topology_fingerprint(descriptor)
    changed = {
        **descriptor,
        "chunk_size": 16,
    }

    assert descriptor["total_tokens"] == 64
    assert descriptor["num_sequences"] == 2
    assert descriptor["h_heads"] == 1
    assert descriptor["hv_heads"] == 2
    assert descriptor["k_dim"] == 64
    assert descriptor["v_dim"] == 32
    assert descriptor["chunk_size"] == 32
    assert descriptor["lengths"] == (16, 48)
    assert descriptor["chunk_offsets"] == (0, 1, 3)
    assert descriptor["use_initial_state"] is True
    assert descriptor["output_final_state"] is False
    assert descriptor["scale"] == 0.125
    assert len(fingerprint) == 64
    assert _kda_varlen_topology_fingerprint(changed) != fingerprint


def test_kda_topology_disk_manifest_round_trips(tmp_path):
    from cppmega_v4._tilelang.path_d_runtime_adapter import (
        _read_kda_topology_manifest,
        _write_kda_topology_manifest,
    )

    descriptor = {
        "total_tokens": 64,
        "num_sequences": 2,
        "k_dim": 64,
        "v_dim": 32,
        "h_heads": 1,
        "hv_heads": 1,
        "chunk_size": 32,
        "lengths": (16, 48),
        "chunk_offsets": (0, 1, 3),
        "use_initial_state": True,
        "output_final_state": True,
        "scale": 0.125,
    }

    _write_kda_topology_manifest(
        "a" * 64,
        descriptor=descriptor,
        status="compiled",
        stages=("token", "inter"),
        cache_dir=str(tmp_path),
    )
    payload = _read_kda_topology_manifest("a" * 64, cache_dir=str(tmp_path))

    assert payload is not None
    assert payload["fingerprint"] == "a" * 64
    assert payload["status"] == "compiled"
    assert payload["descriptor"]["lengths"] == [16, 48]
    assert payload["stages"] == ["token", "inter"]


def test_kda_runtime_writes_manifest_for_promoted_topology(tmp_path):
    import mlx.core as mx

    from cppmega_v4._tilelang import path_d_runtime_adapter as adapter

    adapter._reset_kda_topology_cache_for_tests()
    compile_topology_constants = []

    def fake_compile_stages(**kwargs):
        compile_topology_constants.append(kwargs.get("topology_constants"))
        names = (
            "kda.intra_token_parallel",
            "kda.inter_solve",
            "kda.recompute_w_u",
            "gdn.chunk_delta_h",
            "kda.chunk_o_gk",
        )
        return tuple(
            adapter.PathDCompileResult(
                available=True,
                reason="fake",
                artifact=object(),
                plan=adapter.PathDKernelPlan(name=name, out_idx=(), grid=(1,)),
            )
            for name in names
        )

    q = mx.zeros((1, 32, 1, 8), dtype=mx.float16)
    k = mx.zeros((1, 32, 1, 8), dtype=mx.float16)
    v = mx.zeros((1, 32, 1, 8), dtype=mx.float16)
    g = mx.zeros((1, 32, 1, 8), dtype=mx.float32)
    beta = mx.zeros((1, 32, 1), dtype=mx.float32)
    cu_seqlens = mx.array([0, 16, 32], dtype=mx.int64)
    initial_state = mx.zeros((2, 1, 8, 8), dtype=mx.float32)

    y, final_state = adapter.kda_fwd_runtime_call(
        q,
        k,
        v,
        g,
        beta,
        cu_seqlens=cu_seqlens,
        initial_state=initial_state,
        output_final_state=True,
        topology_specialization_threshold=1,
        topology_cache_dir=str(tmp_path),
        compile_stages_fn=fake_compile_stages,
        launch_stage_fn=lambda _stage, *args: None,
    )
    manifests = list(tmp_path.glob("*.json"))

    assert y.shape == (1, 32, 1, 8)
    assert final_state is not None
    assert final_state.shape == (2, 1, 8, 8)
    assert compile_topology_constants[0] is None
    assert compile_topology_constants[1] is not None
    assert len(manifests) == 1


def test_runtime_adapter_specializes_topology_metadata_loads():
    pytest.importorskip("tvm")
    from tvm import tir

    from cppmega_v4._tilelang.path_d_runtime_adapter import (
        specialize_primfunc_for_topology_metadata,
    )

    cu_buffer = tir.decl_buffer((3,), "int64", name="cu_seqlens")
    idx = tir.Var("idx", "int64")
    func = tir.PrimFunc(
        [cu_buffer.data, idx],
        tir.Evaluate(tir.BufferLoad(cu_buffer, [idx])),
        buffer_map={cu_buffer.data: cu_buffer},
    )

    specialized = specialize_primfunc_for_topology_metadata(
        func,
        {"cu_seqlens": (0, 16, 64)},
    )

    assert _count_named_buffer_loads(func, {"cu_seqlens"}) == 1
    assert _count_named_buffer_loads(specialized, {"cu_seqlens"}) == 0


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
    assert isinstance(ok_kda, bool)
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


def test_kda_runtime_adapter_launches_fixed_prefill_smoke():
    pytest.importorskip("tilelang")
    import mlx.core as mx

    from cppmega_v4._tilelang.path_d_runtime_adapter import (
        kda_fwd_runtime_call,
        kda_runtime_adapter_status,
    )

    ok, reason = kda_runtime_adapter_status("test coverage complete")
    if not ok:
        pytest.skip(f"KDA runtime adapter not available on this host: {reason}")

    q = mx.zeros((1, 64, 1, 64), dtype=mx.float16)
    k = mx.zeros((1, 64, 1, 64), dtype=mx.float16)
    v = mx.zeros((1, 64, 1, 32), dtype=mx.float16)
    g = mx.zeros((1, 64, 1, 64), dtype=mx.float32)
    beta = mx.zeros((1, 64, 1), dtype=mx.float32)

    y, final_state = kda_fwd_runtime_call(
        q,
        k,
        v,
        g,
        beta,
        output_final_state=True,
    )
    mx.eval(y, final_state)

    assert y.shape == (1, 64, 1, 32)
    assert y.dtype == mx.float16
    assert final_state is not None
    assert final_state.shape == (1, 1, 64, 32)
    assert final_state.dtype == mx.float32


def test_kda_runtime_adapter_launches_dynamic_shape_custom_scale_smoke():
    pytest.importorskip("tilelang")
    import mlx.core as mx

    from cppmega_v4._tilelang.path_d_runtime_adapter import (
        kda_fwd_runtime_call,
        kda_runtime_adapter_status,
    )
    ok, reason = kda_runtime_adapter_status("test coverage complete")
    if not ok:
        pytest.skip(f"KDA runtime adapter not available on this host: {reason}")

    q = mx.zeros((1, 32, 1, 8), dtype=mx.float16)
    k = mx.zeros((1, 32, 1, 8), dtype=mx.float16)
    v = mx.zeros((1, 32, 1, 8), dtype=mx.float16)
    g = mx.zeros((1, 32, 1, 8), dtype=mx.float32)
    beta = mx.zeros((1, 32, 1), dtype=mx.float32)

    y, final_state = kda_fwd_runtime_call(
        q,
        k,
        v,
        g,
        beta,
        scale=0.25,
        output_final_state=True,
    )
    mx.eval(y, final_state)

    assert y.shape == (1, 32, 1, 8)
    assert y.dtype == mx.float16
    assert final_state is not None
    assert final_state.shape == (1, 1, 8, 8)
    assert final_state.dtype == mx.float32


def test_kda_runtime_adapter_launches_varlen_smoke():
    pytest.importorskip("tilelang")
    import mlx.core as mx

    from cppmega_v4._tilelang.path_d_runtime_adapter import (
        kda_fwd_runtime_call,
        kda_runtime_adapter_status,
    )
    ok, reason = kda_runtime_adapter_status("test coverage complete")
    if not ok:
        pytest.skip(f"KDA runtime adapter not available on this host: {reason}")

    q = mx.zeros((1, 64, 1, 64), dtype=mx.float16)
    k = mx.zeros((1, 64, 1, 64), dtype=mx.float16)
    v = mx.zeros((1, 64, 1, 32), dtype=mx.float16)
    g = mx.zeros((1, 64, 1, 64), dtype=mx.float32)
    beta = mx.zeros((1, 64, 1), dtype=mx.float32)
    cu_seqlens = mx.array([0, 16, 64], dtype=mx.int64)
    initial_state = mx.zeros((2, 1, 64, 32), dtype=mx.float32)

    y, final_state = kda_fwd_runtime_call(
        q,
        k,
        v,
        g,
        beta,
        cu_seqlens=cu_seqlens,
        initial_state=initial_state,
        output_final_state=True,
    )
    mx.eval(y, final_state)

    assert y.shape == (1, 64, 1, 32)
    assert y.dtype == mx.float16
    assert final_state is not None
    assert final_state.shape == (2, 1, 64, 32)
    assert final_state.dtype == mx.float32

    initial_state = mx.zeros((1, 1, 64, 32), dtype=mx.float32)
    y, final_state = kda_fwd_runtime_call(
        q,
        k,
        v,
        g,
        beta,
        initial_state=initial_state,
        output_final_state=True,
    )
    mx.eval(y, final_state)

    assert y.shape == (1, 64, 1, 32)
    assert y.dtype == mx.float16
    assert final_state is not None
    assert final_state.shape == (1, 1, 64, 32)
    assert final_state.dtype == mx.float32
