from __future__ import annotations

import tilelang.language as T
from tvm.tir import PrimFunc

from cppmega_mlx.nn._tilelang import (
    m2rnn_path_c,
    mamba3_path_c,
    sparse_mla_fp8_path_c,
)
from cppmega_mlx.runtime.path_c_fusion import (
    build_mamba3_fp8_train_tilelang_region_from_prim_funcs,
    compile_mamba3_fp8_train_tilelang_region_from_prim_funcs,
    mamba3_fp8_train_schedule_status_from_prim_funcs,
)


def _assert_raw_prim_func(candidate: object) -> PrimFunc:
    assert isinstance(candidate, PrimFunc)
    assert not isinstance(candidate, str)
    assert "code_block_source" not in candidate.attrs
    return candidate


@T.prim_func
def _toy_fused_train_block(
    hidden: T.Buffer((4,), "float32"),
    mamba_state: T.Buffer((4,), "float32"),
    scan_state: T.Buffer((4,), "float32"),
    q_fp8: T.Buffer((4,), "float32"),
    q_scale: T.Buffer((4,), "float32"),
    kv_fp8: T.Buffer((4,), "float32"),
    kv_scale: T.Buffer((4,), "float32"),
):
    with T.Kernel(1, threads=1):
        scan_y = T.alloc_local((4,), "float32")
        post_y = T.alloc_local((4,), "float32")
        scan_y[0] = hidden[0] + mamba_state[0]
        scan_state[0] = scan_y[0]
        post_y[0] = scan_y[0]
        q_fp8[0] = post_y[0]
        q_scale[0] = 1.0
        kv_fp8[0] = post_y[0]
        kv_scale[0] = 1.0


def test_mamba3_train_surfaces_are_available_as_raw_prim_funcs() -> None:
    _assert_raw_prim_func(
        mamba3_path_c._make_fwd_prim_func(  # pyright: ignore[reportPrivateUsage]
            1,
            2,
            2,
            4,
            2,
        )
    )
    _assert_raw_prim_func(
        mamba3_path_c._make_fwd_with_snapshots_prim_func(  # pyright: ignore[reportPrivateUsage]
            1,
            2,
            2,
            4,
            2,
        )
    )


def test_m2rnn_packed_post_surfaces_are_available_as_raw_prim_funcs() -> None:
    common = dict(
        batch=1,
        seq=2,
        total_heads=2,
        q_heads=2,
        k_heads=2,
        v_heads=2,
        g_heads=2,
        w_heads=2,
        f_heads=2,
        projected_dim=16,
        carrier_dtype="float32",
    )

    generic = _assert_raw_prim_func(
        m2rnn_path_c._make_mapped_packed_post_fwd_prim_func(  # pyright: ignore[reportPrivateUsage]
            **common,
            k_dim=2,
            v_dim=2,
        )
    )
    k_parallel = _assert_raw_prim_func(
        m2rnn_path_c._make_mapped_packed_post_fwd_k_parallel_prim_func(  # pyright: ignore[reportPrivateUsage]
            **common,
            k_dim=8,
            v_dim=4,
        )
    )

    assert "m2rnn_mapped_packed_post_fwd_" in str(generic.attrs["global_symbol"])
    assert "m2rnn_mapped_packed_post_fwd_kp_" in str(
        k_parallel.attrs["global_symbol"]
    )


def test_fp8_sparse_mla_prepared_surface_is_raw_prim_func() -> None:
    _assert_raw_prim_func(
        sparse_mla_fp8_path_c._make_fp8_sparse_mla_apply_kernel(  # pyright: ignore[reportPrivateUsage]
            batch=1,
            seq_len=2,
            heads=2,
            seq_len_kv=4,
            kv_group=2,
            head_kv=1,
            topk=4,
            K=8,
            d_v=8,
            threads=8,
            out_dtype="float32",
        )
    )


def test_mamba3_fp8_train_block_can_build_tilelang_region_from_raw_prim_funcs() -> None:
    from tilelang import tvm
    from tilelang.engine.fusion import compile_fusion_region
    from tilelang.transform import PassConfigKey

    mamba3_scan = mamba3_path_c._make_fwd_with_snapshots_prim_func(  # pyright: ignore[reportPrivateUsage]
        1,
        2,
        2,
        4,
        2,
    )
    m2rnn_packed_post = m2rnn_path_c._make_mapped_packed_post_fwd_prim_func(  # pyright: ignore[reportPrivateUsage]
        batch=1,
        seq=2,
        total_heads=2,
        q_heads=2,
        k_heads=2,
        v_heads=2,
        g_heads=2,
        w_heads=2,
        f_heads=2,
        k_dim=2,
        v_dim=2,
        projected_dim=16,
        carrier_dtype="float32",
    )
    sparse_mla_fp8_prepared = sparse_mla_fp8_path_c._make_fp8_sparse_mla_apply_kernel(  # pyright: ignore[reportPrivateUsage]
        batch=1,
        seq_len=2,
        heads=2,
        seq_len_kv=4,
        kv_group=2,
        head_kv=1,
        topk=4,
        K=8,
        d_v=8,
        threads=8,
        out_dtype="float32",
    )

    region = build_mamba3_fp8_train_tilelang_region_from_prim_funcs(
        mamba3_scan=mamba3_scan,
        m2rnn_packed_post=m2rnn_packed_post,
        sparse_mla_fp8_prepared=sparse_mla_fp8_prepared,
    )
    called = {}

    def fake_lowerer(func_or_mod, **kwargs):
        called["func_or_mod"] = func_or_mod
        called["pass_config"] = dict(tvm.transform.PassContext.current().config)
        return "compiled-train-block-region"

    result = compile_fusion_region(region, target="metal", lowerer=fake_lowerer)

    assert result.artifact == "compiled-train-block-region"
    assert result.plan.node_names == (
        "mamba3_scan",
        "m2rnn_packed_post",
        "sparse_mla_fp8_prepared",
    )
    assert result.plan.edges == ()
    assert bool(called["pass_config"][PassConfigKey.TL_Z3_PROOF_BARRIER_MINIMIZATION.value]) is True
    assert isinstance(called["func_or_mod"], tvm.IRModule)
    assert [global_var.name_hint for global_var in called["func_or_mod"].functions] == [
        "mamba3_fp8_train_block",
        "m2rnn_packed_post",
        "sparse_mla_fp8_prepared",
    ]
    module_script = called["func_or_mod"].script()
    assert "tl.fusion.region" in module_script
    assert "tl.fusion.node" in module_script
    assert "code_block_source" not in module_script


def test_mamba3_fp8_train_schedule_status_reports_real_raw_abi_blockers() -> None:
    mamba3_scan = mamba3_path_c._make_fwd_with_snapshots_prim_func(  # pyright: ignore[reportPrivateUsage]
        1,
        2,
        2,
        4,
        2,
    )
    m2rnn_packed_post = m2rnn_path_c._make_mapped_packed_post_fwd_prim_func(  # pyright: ignore[reportPrivateUsage]
        batch=1,
        seq=2,
        total_heads=2,
        q_heads=2,
        k_heads=2,
        v_heads=2,
        g_heads=2,
        w_heads=2,
        f_heads=2,
        k_dim=2,
        v_dim=2,
        projected_dim=16,
        carrier_dtype="float32",
    )
    sparse_mla_fp8_prepared = sparse_mla_fp8_path_c._make_fp8_sparse_mla_apply_kernel(  # pyright: ignore[reportPrivateUsage]
        batch=1,
        seq_len=2,
        heads=2,
        seq_len_kv=4,
        kv_group=2,
        head_kv=1,
        topk=4,
        K=8,
        d_v=8,
        threads=8,
        out_dtype="float32",
    )

    status = mamba3_fp8_train_schedule_status_from_prim_funcs(
        mamba3_scan=mamba3_scan,
        m2rnn_packed_post=m2rnn_packed_post,
        sparse_mla_fp8_prepared=sparse_mla_fp8_prepared,
    )

    assert status.status == "blocked_fp8_prepare_producer_missing"
    assert status.single_kernel_fused is False
    assert [edge.buffer for edge in status.blocked_edges] == [
        "scan_y",
        "post_y",
        "fp8_prepare",
    ]
    assert "y" in status.blocked_edges[0].producer_buffers
    assert "conv_input" in status.blocked_edges[0].consumer_buffers
    assert "post" in status.blocked_edges[1].producer_buffers
    assert "q_fp8" in status.blocked_edges[1].consumer_buffers
    assert status.blocked_edges[2].kind == (
        "prepared_apply_consumer_not_prepare_producer"
    )
    assert "out" in status.blocked_edges[2].consumer_buffers
    assert "lse" in status.blocked_edges[2].consumer_buffers
    assert "not the FP8 prepare producer" in status.blocked_edges[2].reason


def test_mamba3_fp8_train_block_can_use_explicit_single_entry_schedule() -> None:
    from tilelang import tvm
    from tilelang.engine.fusion import compile_fusion_region

    mamba3_scan = mamba3_path_c._make_fwd_with_snapshots_prim_func(  # pyright: ignore[reportPrivateUsage]
        1,
        2,
        2,
        4,
        2,
    )
    m2rnn_packed_post = m2rnn_path_c._make_mapped_packed_post_fwd_prim_func(  # pyright: ignore[reportPrivateUsage]
        batch=1,
        seq=2,
        total_heads=2,
        q_heads=2,
        k_heads=2,
        v_heads=2,
        g_heads=2,
        w_heads=2,
        f_heads=2,
        k_dim=2,
        v_dim=2,
        projected_dim=16,
        carrier_dtype="float32",
    )
    sparse_mla_fp8_prepared = sparse_mla_fp8_path_c._make_fp8_sparse_mla_apply_kernel(  # pyright: ignore[reportPrivateUsage]
        batch=1,
        seq_len=2,
        heads=2,
        seq_len_kv=4,
        kv_group=2,
        head_kv=1,
        topk=4,
        K=8,
        d_v=8,
        threads=8,
        out_dtype="float32",
    )

    region = build_mamba3_fp8_train_tilelang_region_from_prim_funcs(
        mamba3_scan=mamba3_scan,
        m2rnn_packed_post=m2rnn_packed_post,
        sparse_mla_fp8_prepared=sparse_mla_fp8_prepared,
        schedule_template=lambda _: _toy_fused_train_block,
    )
    result = compile_fusion_region(
        region,
        target="metal",
        lowerer=lambda *args, **kwargs: "compiled-explicit-schedule",
        require_single_kernel=True,
    )
    entry = result.lowered_module[region.entry_symbol]

    assert result.artifact == "compiled-explicit-schedule"
    assert result.plan.require_single_kernel is True
    assert isinstance(result.lowered_module, tvm.IRModule)
    assert [global_var.name_hint for global_var in result.lowered_module.functions] == [
        "mamba3_fp8_train_block",
    ]
    assert [buffer.name for buffer in entry.buffer_map.values()] == [
        "hidden",
        "mamba_state",
        "scan_state",
        "q_fp8",
        "q_scale",
        "kv_fp8",
        "kv_scale",
    ]


def test_mamba3_fp8_train_block_compile_requires_explicit_schedule() -> None:
    mamba3_scan = mamba3_path_c._make_fwd_with_snapshots_prim_func(  # pyright: ignore[reportPrivateUsage]
        1,
        2,
        2,
        4,
        2,
    )
    m2rnn_packed_post = m2rnn_path_c._make_mapped_packed_post_fwd_prim_func(  # pyright: ignore[reportPrivateUsage]
        batch=1,
        seq=2,
        total_heads=2,
        q_heads=2,
        k_heads=2,
        v_heads=2,
        g_heads=2,
        w_heads=2,
        f_heads=2,
        k_dim=2,
        v_dim=2,
        projected_dim=16,
        carrier_dtype="float32",
    )
    sparse_mla_fp8_prepared = sparse_mla_fp8_path_c._make_fp8_sparse_mla_apply_kernel(  # pyright: ignore[reportPrivateUsage]
        batch=1,
        seq_len=2,
        heads=2,
        seq_len_kv=4,
        kv_group=2,
        head_kv=1,
        topk=4,
        K=8,
        d_v=8,
        threads=8,
        out_dtype="float32",
    )

    try:
        compile_mamba3_fp8_train_tilelang_region_from_prim_funcs(
            mamba3_scan=mamba3_scan,
            m2rnn_packed_post=m2rnn_packed_post,
            sparse_mla_fp8_prepared=sparse_mla_fp8_prepared,
            lowerer=lambda *args, **kwargs: "must-not-compile",
        )
    except RuntimeError as exc:
        message = str(exc)
    else:
        raise AssertionError("compile unexpectedly accepted missing fused schedule")

    assert "blocked_fp8_prepare_producer_missing" in message
    assert "prepared_apply_consumer_not_prepare_producer" in message
    assert "scan_y" in message
    assert "post_y" in message
    assert "fp8_prepare" in message


def test_mamba3_fp8_train_block_compile_accepts_explicit_schedule() -> None:
    from tilelang import tvm

    mamba3_scan = mamba3_path_c._make_fwd_with_snapshots_prim_func(  # pyright: ignore[reportPrivateUsage]
        1,
        2,
        2,
        4,
        2,
    )
    m2rnn_packed_post = m2rnn_path_c._make_mapped_packed_post_fwd_prim_func(  # pyright: ignore[reportPrivateUsage]
        batch=1,
        seq=2,
        total_heads=2,
        q_heads=2,
        k_heads=2,
        v_heads=2,
        g_heads=2,
        w_heads=2,
        f_heads=2,
        k_dim=2,
        v_dim=2,
        projected_dim=16,
        carrier_dtype="float32",
    )
    sparse_mla_fp8_prepared = sparse_mla_fp8_path_c._make_fp8_sparse_mla_apply_kernel(  # pyright: ignore[reportPrivateUsage]
        batch=1,
        seq_len=2,
        heads=2,
        seq_len_kv=4,
        kv_group=2,
        head_kv=1,
        topk=4,
        K=8,
        d_v=8,
        threads=8,
        out_dtype="float32",
    )

    result = compile_mamba3_fp8_train_tilelang_region_from_prim_funcs(
        mamba3_scan=mamba3_scan,
        m2rnn_packed_post=m2rnn_packed_post,
        sparse_mla_fp8_prepared=sparse_mla_fp8_prepared,
        schedule_template=lambda _: _toy_fused_train_block,
        lowerer=lambda *args, **kwargs: "compiled-explicit-schedule",
    )

    assert result.artifact == "compiled-explicit-schedule"
    assert isinstance(result.lowered_module, tvm.IRModule)
    assert [global_var.name_hint for global_var in result.lowered_module.functions] == [
        "mamba3_fp8_train_block",
    ]
