from __future__ import annotations

from tvm.tir import PrimFunc

from cppmega_mlx.nn._tilelang import (
    m2rnn_path_c,
    mamba3_path_c,
    sparse_mla_fp8_path_c,
)


def _assert_raw_prim_func(candidate: object) -> PrimFunc:
    assert isinstance(candidate, PrimFunc)
    assert not isinstance(candidate, str)
    assert "code_block_source" not in candidate.attrs
    return candidate


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
