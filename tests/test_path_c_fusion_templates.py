from __future__ import annotations

import tilelang.language as T
from tvm.tir import PrimFunc

from cppmega_mlx.nn._tilelang import (
    m2rnn_path_c,
    mamba3_path_c,
    sparse_mla_fp8_path_c,
)
from cppmega_mlx.runtime.path_c_fusion import (
    SparseMLAFp8PrepareSpec,
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
    hidden: T.Tensor((4,), "float32"),
    mamba_state: T.Tensor((4,), "float32"),
    scan_state: T.Tensor((4,), "float32"),
    attention_out: T.Tensor((4,), "float32"),
    lse: T.Tensor((4,), "float32"),
):
    with T.Kernel(1, threads=1):
        scan_y = T.alloc_local((4,), "float32")
        post_y = T.alloc_local((4,), "float32")
        q_fp8 = T.alloc_local((4,), "float32")
        q_scale = T.alloc_local((4,), "float32")
        kv_fp8 = T.alloc_local((4,), "float32")
        kv_scale = T.alloc_local((4,), "float32")
        scan_y[0] = hidden[0] + mamba_state[0]
        scan_state[0] = scan_y[0]
        post_y[0] = scan_y[0]
        q_fp8[0] = post_y[0]
        q_scale[0] = 1.0
        kv_fp8[0] = post_y[0]
        kv_scale[0] = 1.0
        attention_out[0] = q_fp8[0] + kv_fp8[0]
        lse[0] = q_scale[0] + kv_scale[0]


@T.prim_func
def _toy_fp8_prepare(
    post_y: T.Tensor((4,), "float32"),
    q_fp8: T.Tensor((4,), "float32"),
    q_scale: T.Tensor((4,), "float32"),
    kv_fp8: T.Tensor((4,), "float32"),
    kv_scale: T.Tensor((4,), "float32"),
):
    with T.Kernel(1, threads=1):
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


def test_fp8_sparse_mla_prepare_surface_is_raw_prim_func() -> None:
    prim = _assert_raw_prim_func(
        sparse_mla_fp8_path_c.make_fp8_sparse_mla_prepare_kernel(
            q_rows=4,
            kv_rows=2,
            K=8,
            in_dtype="float32",
        )
    )

    assert {buffer.name for buffer in prim.buffer_map.values()} == {
        "post_y",
        "q_fp8",
        "q_scale",
        "kv_fp8",
        "kv_scale",
    }
    buffer_dtypes = {
        str(buffer.name): str(buffer.dtype) for buffer in prim.buffer_map.values()
    }
    assert buffer_dtypes["q_fp8"] == "uint8"
    assert buffer_dtypes["kv_fp8"] == "uint8"


def test_fp8_sparse_mla_prepare_can_expose_legacy_float8_storage() -> None:
    prim = _assert_raw_prim_func(
        sparse_mla_fp8_path_c.make_fp8_sparse_mla_prepare_kernel(
            q_rows=4,
            kv_rows=2,
            K=8,
            in_dtype="float32",
            storage_dtype="float8_e4m3",
        )
    )

    buffer_dtypes = {
        str(buffer.name): str(buffer.dtype) for buffer in prim.buffer_map.values()
    }
    assert buffer_dtypes["q_fp8"] == "float8_e4m3"
    assert buffer_dtypes["kv_fp8"] == "float8_e4m3"


def test_fp8_sparse_mla_prepare_uint8_storage_lowers_with_software_encoder() -> None:
    import tilelang

    prim = sparse_mla_fp8_path_c.make_fp8_sparse_mla_prepare_kernel(
        q_rows=1,
        kv_rows=1,
        K=4,
        in_dtype="float32",
    )

    artifact = tilelang.lower(prim, target="metal")
    source = artifact.kernel_source or ""

    assert "log2" in source
    assert "exp2" in source
    assert "as_type<uchar>(((uchar)" not in source


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
        "sparse_mla_fp8_apply",
    )
    assert result.plan.edges == ()
    assert bool(called["pass_config"][PassConfigKey.TL_Z3_PROOF_BARRIER_MINIMIZATION.value]) is True
    assert isinstance(called["func_or_mod"], tvm.IRModule)
    assert [global_var.name_hint for global_var in called["func_or_mod"].functions] == [
        "mamba3_fp8_train_block",
        "m2rnn_packed_post",
        "sparse_mla_fp8_apply",
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
    assert status.blocked_edges[1].consumer == "fp8_prepare"
    assert status.blocked_edges[1].kind == "fp8_prepare_tilelang_prim_func_missing"
    assert status.blocked_edges[1].consumer_buffers == ()
    assert status.blocked_edges[2].kind == (
        "prepared_apply_consumer_not_prepare_producer"
    )
    assert status.blocked_edges[2].consumer == "sparse_mla_fp8_apply"
    assert "q_fp8" in status.blocked_edges[2].consumer_buffers
    assert "out" in status.blocked_edges[2].consumer_buffers
    assert "lse" in status.blocked_edges[2].consumer_buffers
    assert "not the FP8 prepare producer" in status.blocked_edges[2].reason


def test_mamba3_fp8_train_schedule_status_accepts_real_fp8_prepare_prim_func() -> None:
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
    fp8_prepare = sparse_mla_fp8_path_c.make_fp8_sparse_mla_prepare_kernel(
        q_rows=4,
        kv_rows=8,
        K=8,
        in_dtype="float32",
    )

    status = mamba3_fp8_train_schedule_status_from_prim_funcs(
        mamba3_scan=mamba3_scan,
        m2rnn_packed_post=m2rnn_packed_post,
        sparse_mla_fp8_prepared=sparse_mla_fp8_prepared,
        fp8_prepare=fp8_prepare,
    )

    assert status.status == "blocked_raw_abi_mismatch"
    assert "fp8_prepare_tilelang_prim_func_missing" not in {
        edge.kind for edge in status.blocked_edges
    }
    assert "prepared_apply_consumer_not_prepare_producer" not in {
        edge.kind for edge in status.blocked_edges
    }
    assert [edge.buffer for edge in status.blocked_edges] == ["scan_y", "post_y"]
    assert status.blocked_edges[0].kind == "raw_abi_signature_mismatch"
    assert status.blocked_edges[0].producer_signature == "y:1x2x2x4:float32"
    assert status.blocked_edges[0].consumer_signature == "conv_input:1x2x12:float32"
    assert "layout transform" in status.blocked_edges[0].reason
    assert status.blocked_edges[1].producer == "m2rnn_packed_post"
    assert status.blocked_edges[1].consumer == "fp8_prepare"
    assert status.blocked_edges[1].kind == "raw_abi_signature_mismatch"
    assert "post" in status.blocked_edges[1].producer_buffers
    assert "post_y" in status.blocked_edges[1].consumer_buffers
    assert status.blocked_edges[1].producer_signature == "post:1x2x4:float32"
    assert status.blocked_edges[1].consumer_signature == "post_y:96:float32"


def test_mamba3_fp8_train_schedule_status_builds_fp8_prepare_from_spec() -> None:
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
        fp8_prepare_spec=SparseMLAFp8PrepareSpec(
            q_rows=4,
            kv_rows=8,
            K=8,
            in_dtype="float32",
        ),
    )

    assert status.status == "blocked_raw_abi_mismatch"
    assert "fp8_prepare_tilelang_prim_func_missing" not in {
        edge.kind for edge in status.blocked_edges
    }
    assert "prepared_apply_consumer_not_prepare_producer" not in {
        edge.kind for edge in status.blocked_edges
    }
    assert [edge.buffer for edge in status.blocked_edges] == ["scan_y", "post_y"]
    assert status.blocked_edges[0].kind == "raw_abi_signature_mismatch"
    assert status.blocked_edges[0].producer_signature == "y:1x2x2x4:float32"
    assert status.blocked_edges[0].consumer_signature == "conv_input:1x2x12:float32"
    assert "layout transform" in status.blocked_edges[0].reason
    assert status.blocked_edges[1].producer == "m2rnn_packed_post"
    assert status.blocked_edges[1].consumer == "fp8_prepare"
    assert status.blocked_edges[1].kind == "raw_abi_signature_mismatch"
    assert "post" in status.blocked_edges[1].producer_buffers
    assert "post_y" in status.blocked_edges[1].consumer_buffers
    assert status.blocked_edges[1].producer_signature == "post:1x2x4:float32"
    assert status.blocked_edges[1].consumer_signature == "post_y:96:float32"


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
    assert result.plan.node_names == (
        "mamba3_scan",
        "m2rnn_packed_post",
        "fp8_prepare",
        "sparse_mla_fp8_apply",
    )
    assert [global_var.name_hint for global_var in result.lowered_module.functions] == [
        "mamba3_fp8_train_block",
    ]
    assert [buffer.name for buffer in entry.buffer_map.values()] == [
        "hidden",
        "mamba_state",
        "scan_state",
        "attention_out",
        "lse",
    ]


def test_mamba3_fp8_train_block_can_use_real_fp8_prepare_prim_func_in_region() -> None:
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
        fp8_prepare=_toy_fp8_prepare,
        sparse_mla_fp8_prepared=sparse_mla_fp8_prepared,
        schedule_template=lambda _: _toy_fused_train_block,
    )

    assert region.nodes[2].name == "fp8_prepare"
    assert region.nodes[2].prim_func is _toy_fp8_prepare
    assert [(edge.producer, edge.consumer, edge.buffer) for edge in region.edges] == [
        ("mamba3_scan", "m2rnn_packed_post", "scan_y"),
        ("m2rnn_packed_post", "fp8_prepare", "post_y"),
        ("fp8_prepare", "sparse_mla_fp8_apply", "q_fp8"),
        ("fp8_prepare", "sparse_mla_fp8_apply", "q_scale"),
        ("fp8_prepare", "sparse_mla_fp8_apply", "kv_fp8"),
        ("fp8_prepare", "sparse_mla_fp8_apply", "kv_scale"),
    ]


def test_mamba3_fp8_train_block_can_build_fp8_prepare_from_spec_in_region() -> None:
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
        fp8_prepare_spec=SparseMLAFp8PrepareSpec(
            q_rows=4,
            kv_rows=8,
            K=8,
            in_dtype="float32",
        ),
        sparse_mla_fp8_prepared=sparse_mla_fp8_prepared,
        schedule_template=lambda _: _toy_fused_train_block,
    )

    assert region.nodes[2].name == "fp8_prepare"
    assert region.nodes[2].prim_func is not None
    assert {buffer.name for buffer in region.nodes[2].prim_func.buffer_map.values()} == {
        "post_y",
        "q_fp8",
        "q_scale",
        "kv_fp8",
        "kv_scale",
    }


def test_mamba3_fp8_prepare_spec_changes_fusion_cache_key_material() -> None:
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

    def plan_for(spec: SparseMLAFp8PrepareSpec) -> tuple[str, ...]:
        region = build_mamba3_fp8_train_tilelang_region_from_prim_funcs(
            mamba3_scan=mamba3_scan,
            m2rnn_packed_post=m2rnn_packed_post,
            fp8_prepare_spec=spec,
            sparse_mla_fp8_prepared=sparse_mla_fp8_prepared,
            schedule_template=lambda _: _toy_fused_train_block,
        )
        result = compile_fusion_region(
            region,
            target="metal",
            lowerer=lambda *args, **kwargs: "compiled",
            require_single_kernel=True,
        )
        return tuple(result.plan.cache_key_material)

    plan_a = plan_for(SparseMLAFp8PrepareSpec(q_rows=4, kv_rows=8, K=8, in_dtype="float32"))
    plan_b = plan_for(SparseMLAFp8PrepareSpec(q_rows=2, kv_rows=4, K=8, in_dtype="float32"))

    assert plan_a != plan_b
    assert any(part.startswith("prim_funcs:") and "fp8_prepare" in part for part in plan_a)
    assert any(part.startswith("prim_funcs:") and "fp8_prepare" in part for part in plan_b)


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


def test_mamba3_fp8_train_block_compile_with_prepare_spec_reports_next_raw_abi_blocker() -> None:
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
            fp8_prepare_spec=SparseMLAFp8PrepareSpec(
                q_rows=4,
                kv_rows=8,
                K=8,
                in_dtype="float32",
            ),
            sparse_mla_fp8_prepared=sparse_mla_fp8_prepared,
            lowerer=lambda *args, **kwargs: "must-not-compile",
        )
    except RuntimeError as exc:
        message = str(exc)
    else:
        raise AssertionError("compile unexpectedly accepted missing fused schedule")

    assert "blocked_raw_abi_mismatch" in message
    assert "scan_y" in message
    assert "post_y" in message
    assert "float8_e4m3" not in message
    assert "fp8_prepare_tilelang_prim_func_missing" not in message
    assert "prepared_apply_consumer_not_prepare_producer" not in message


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
