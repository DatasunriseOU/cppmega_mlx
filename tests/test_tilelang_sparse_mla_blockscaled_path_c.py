"""Coverage for the Path C TileLang DSL Sparse-MLA blockscaled (E8M0) port."""

# pyright: reportMissingImports=false

from __future__ import annotations

import pytest
import numpy as np

import mlx.core as mx

from cppmega_mlx.nn._tilelang.sparse_mla_blockscaled_path_c import (
    E8M0_BLOCK_SIZE,
    E8M0_LAYOUT,
    E8M0_SCALE_FORMAT,
    SparseMLABlockScaledPathCStatus,
    SparseMLABlockScaledQKReducePathCStatus,
    blockscaled_sparse_mla_qk_msl_features,
    blockscaled_sparse_mla_qk_path_c_status,
    blockscaled_sparse_mla_qk_reduce_path_c,
    blockscaled_sparse_mla_qk_reduce_path_c_status,
    blockscaled_sparse_mla_qk_scaled_matmul_probe_status,
    lower_blockscaled_sparse_mla_qk_msl,
    lower_blockscaled_sparse_mla_qk_reduce_msl,
    sparse_mla_blockscaled_path_c_apply,
)
from cppmega_mlx.nn._tilelang.sparse_mla_blockscaled import (
    _quantize_mxfp8,
    _unpack_mxfp8_to_uint8,
    sparse_mla_blockscaled_fwd_metal,
)


# ---------------------------------------------------------------------------
# Import-smoke: pin the constants the routing layer depends on.
# ---------------------------------------------------------------------------


def test_e8m0_format_constants_are_stable() -> None:
    """E8M0 format identity must not silently drift; these are wire-format."""

    assert E8M0_BLOCK_SIZE == 32
    # Layout/format strings are read by the AUTO router; case/value matters.
    assert isinstance(E8M0_LAYOUT, str) and E8M0_LAYOUT
    assert isinstance(E8M0_SCALE_FORMAT, str) and E8M0_SCALE_FORMAT


# ---------------------------------------------------------------------------
# E8M0 QK probe -- the only thing the path_c module actually exports for
# Sparse-MLA blockscaled today (per Meta agent F).
# ---------------------------------------------------------------------------


def test_qk_scaled_matmul_probe_returns_status_with_reason() -> None:
    """The probe returns a status dataclass with an actionable reason string,
    even when the simdgroup MMA tile is not eligible (e.g. M=1 Sparse-MLA tile)."""

    status = blockscaled_sparse_mla_qk_scaled_matmul_probe_status()
    assert isinstance(status, SparseMLABlockScaledPathCStatus)
    assert isinstance(status.available, bool)
    assert isinstance(status.reason, str) and status.reason
    assert status.transpose_B is True
    assert status.m == 1
    # When unavailable, the reason must list at least one concrete blocker so
    # the routing doc can stay in sync.
    if not status.available:
        assert ":" in status.reason or ";" in status.reason


def test_qk_path_c_status_is_available_or_explains_why() -> None:
    status = blockscaled_sparse_mla_qk_path_c_status()
    assert isinstance(status, SparseMLABlockScaledPathCStatus)
    assert isinstance(status.available, bool)
    assert isinstance(status.reason, str) and status.reason


def test_qk_reduce_path_c_status_reports_shape() -> None:
    """The reducer probe records the (N, K) tile it just probed."""

    status = blockscaled_sparse_mla_qk_reduce_path_c_status(N=16, K=64)
    assert isinstance(status, SparseMLABlockScaledQKReducePathCStatus)
    assert status.n == 16
    assert status.k == 64
    assert isinstance(status.reason, str) and status.reason


# ---------------------------------------------------------------------------
# MSL feature inspection: the QK lowering must expose recognisable E8M0
# decode markers so the AUTO router (and the bench harness) can introspect
# what TileLang emitted without re-lowering.
# ---------------------------------------------------------------------------


def test_lower_blockscaled_qk_msl_emits_inspectable_kernel() -> None:
    try:
        import tilelang  # noqa: F401
    except (ImportError, OSError) as exc:
        pytest.skip(f"tilelang unavailable on this host: {exc}")
    try:
        msl = lower_blockscaled_sparse_mla_qk_msl(M=1, N=16, K=64)
    except (ImportError, OSError) as exc:
        pytest.skip(f"tilelang dylib failed to load: {exc}")
    assert isinstance(msl, str) and msl
    assert "kernel void" in msl
    features = blockscaled_sparse_mla_qk_msl_features(msl)
    assert isinstance(features, dict)
    # Must surface at least one recognised E8M0 marker -- otherwise the
    # AUTO router's "looks like FP8 dot4 / E8M0 decode" heuristic is broken.
    assert any(
        bool(features.get(key))
        for key in (
            "metal_fp8_dot4_helper",
            "e8m0_exp2",
            "e8m0_bias_subtract_127",
            "simdgroup_multiply_accumulate",
        )
    ), f"no E8M0/FP8 markers in QK MSL features: {features}"


def test_lower_blockscaled_qk_reduce_msl_emits_kernel() -> None:
    try:
        import tilelang  # noqa: F401
    except (ImportError, OSError) as exc:
        pytest.skip(f"tilelang unavailable on this host: {exc}")
    try:
        msl = lower_blockscaled_sparse_mla_qk_reduce_msl(N=16, K=64)
    except (ImportError, OSError) as exc:
        pytest.skip(f"tilelang dylib failed to load: {exc}")
    assert isinstance(msl, str) and msl
    assert "kernel void" in msl


# ---------------------------------------------------------------------------
# Reducer dispatch smoke: only meaningful on a Metal-capable host. Off-Metal
# the function returns None (documented contract); it must NOT raise.
# ---------------------------------------------------------------------------


def _metal_available() -> bool:
    metal = getattr(mx, "metal", None)
    return mx.default_device() == mx.gpu and metal is not None and metal.is_available()


def test_qk_reduce_path_c_returns_none_when_metal_missing_or_correct_shape() -> None:
    """Reducer must either dispatch and produce (1, N) fp32, or return None."""

    n, k = 16, 64
    A_fp8 = mx.zeros((1, k), dtype=mx.uint8)
    # E8M0 scale layout: one byte per K/32 block per row; for M=1 / K=64 -> 2.
    a_scale = mx.zeros((1, k // E8M0_BLOCK_SIZE), dtype=mx.uint8)
    B_fp8 = mx.zeros((n, k), dtype=mx.uint8)
    b_scale = mx.zeros((n, k // E8M0_BLOCK_SIZE), dtype=mx.uint8)

    out = blockscaled_sparse_mla_qk_reduce_path_c(A_fp8, a_scale, B_fp8, b_scale)
    if out is None:
        # Allowed: TileLang/Metal unavailable on this host.
        return
    mx.eval(out)
    assert tuple(out.shape) == (1, n)
    assert out.dtype == mx.float32


def test_qk_reduce_path_c_fail_closes_without_serial_metal_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A lowering failure must not build a hand-written mx.fast fallback."""

    import cppmega_mlx.nn._tilelang.sparse_mla_blockscaled_path_c as path_c_module

    path_c_module._qk_reduce_kernel_for.cache_clear()

    def fail_lower(*args, **kwargs):
        del args, kwargs
        raise RuntimeError("synthetic TileLang lowering failure")

    def fail_metal_kernel(*args, **kwargs):
        del args, kwargs
        raise AssertionError("lowering failure must not build mx.fast.metal_kernel")

    monkeypatch.setattr(path_c_module, "can_run_metal", lambda: True)
    monkeypatch.setattr(path_c_module, "dispatch_lower", fail_lower)
    monkeypatch.setattr(path_c_module.mx.fast, "metal_kernel", fail_metal_kernel)

    status = path_c_module.blockscaled_sparse_mla_qk_reduce_path_c_status(N=16, K=64)
    assert status.available is False
    assert "synthetic TileLang lowering failure" in status.reason
    assert status.features == {}

    n, k = 16, 64
    A_fp8 = mx.zeros((1, k), dtype=mx.uint8)
    A_scale = mx.zeros((k // E8M0_BLOCK_SIZE,), dtype=mx.uint8)
    B_fp8 = mx.zeros((n, k), dtype=mx.uint8)
    B_scale = mx.zeros((n, k // E8M0_BLOCK_SIZE), dtype=mx.uint8)
    assert (
        path_c_module.blockscaled_sparse_mla_qk_reduce_path_c(
            A_fp8,
            A_scale,
            B_fp8,
            B_scale,
        )
        is None
    )


def test_qk_reduce_path_c_rejects_non_e8m0_scale_dtype(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import cppmega_mlx.nn._tilelang.sparse_mla_blockscaled_path_c as path_c_module

    monkeypatch.setattr(path_c_module, "can_run_metal", lambda: True)
    n, k = 16, 64
    A_fp8 = mx.zeros((1, k), dtype=mx.uint8)
    A_scale = mx.zeros((k // E8M0_BLOCK_SIZE,), dtype=mx.float32)
    B_fp8 = mx.zeros((n, k), dtype=mx.uint8)
    B_scale = mx.zeros((n, k // E8M0_BLOCK_SIZE), dtype=mx.uint8)

    with pytest.raises(TypeError, match="E8M0 storage"):
        path_c_module.blockscaled_sparse_mla_qk_reduce_path_c(
            A_fp8,
            A_scale,
            B_fp8,
            B_scale,
        )


def test_blockscaled_path_c_forward_uses_tvm_ffi_owner_outputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import cppmega_mlx.nn._tilelang.sparse_mla_blockscaled_path_c as path_c_module

    batch, seq, heads, kv_group, seq_kv, topk, dim = 1, 2, 2, 1, 8, 4, 64
    scale_blocks = dim // E8M0_BLOCK_SIZE
    q_fp8 = mx.zeros((batch, seq, heads, dim), dtype=mx.uint8)
    q_scale = mx.zeros((batch, seq, heads, scale_blocks), dtype=mx.uint8)
    kv_fp8 = mx.zeros((batch, seq_kv, kv_group, dim), dtype=mx.uint8)
    kv_scale = mx.zeros((batch, seq_kv, kv_group, scale_blocks), dtype=mx.uint8)
    indices = mx.zeros((batch, seq, kv_group, topk), dtype=mx.int32)
    out = mx.zeros((batch, seq, heads, dim), dtype=mx.float16)
    lse = mx.zeros((batch, seq, heads), dtype=mx.float32)
    calls: list[tuple[tuple[object, ...], tuple[mx.array, mx.array]]] = []

    def fake_tvm_ffi_kernel_for(*args: object, **kwargs: object):
        del args, kwargs

        def fake_kernel(*kernel_args: object, out: tuple[mx.array, mx.array]):
            calls.append((kernel_args, out))
            return out

        return fake_kernel

    def fail_legacy_kernel(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError("owner-output forward must not build mx.fast wrapper")

    monkeypatch.setattr(path_c_module, "can_run_metal", lambda: True)
    monkeypatch.setattr(
        path_c_module,
        "_blockscaled_apply_tvm_ffi_kernel_for",
        fake_tvm_ffi_kernel_for,
    )
    monkeypatch.setattr(path_c_module, "_blockscaled_apply_kernel_for", fail_legacy_kernel)
    monkeypatch.setattr(path_c_module.mx.fast, "metal_kernel", fail_legacy_kernel)

    result = path_c_module.sparse_mla_blockscaled_path_c_apply(
        q_fp8,
        q_scale,
        kv_fp8,
        kv_scale,
        indices,
        sm_scale=dim ** -0.5,
        d_v=dim,
        return_lse=True,
        force_path_c=True,
        out=out,
        lse=lse,
    )

    assert result == (out, lse)
    assert len(calls) == 1
    kernel_args, owner_outputs = calls[0]
    assert kernel_args[0] is q_fp8
    assert kernel_args[1] is q_scale
    assert kernel_args[2] is kv_fp8
    assert kernel_args[3] is kv_scale
    assert kernel_args[4] is indices
    assert owner_outputs == (out, lse)


def test_blockscaled_path_c_forward_owner_output_abi_is_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import cppmega_mlx.nn._tilelang.sparse_mla_blockscaled_path_c as path_c_module

    batch, seq, heads, kv_group, seq_kv, topk, dim = 1, 2, 2, 1, 8, 4, 64
    scale_blocks = dim // E8M0_BLOCK_SIZE
    q_fp8 = mx.zeros((batch, seq, heads, dim), dtype=mx.uint8)
    q_scale = mx.zeros((batch, seq, heads, scale_blocks), dtype=mx.uint8)
    kv_fp8 = mx.zeros((batch, seq_kv, kv_group, dim), dtype=mx.uint8)
    kv_scale = mx.zeros((batch, seq_kv, kv_group, scale_blocks), dtype=mx.uint8)
    indices = mx.zeros((batch, seq, kv_group, topk), dtype=mx.int32)
    out = mx.zeros((batch, seq, heads, dim), dtype=mx.float32)
    lse = mx.zeros((batch, seq, heads), dtype=mx.float32)

    monkeypatch.setattr(path_c_module, "can_run_metal", lambda: True)
    with pytest.raises(ValueError, match="requires both out and lse"):
        path_c_module.sparse_mla_blockscaled_path_c_apply(
            q_fp8,
            q_scale,
            kv_fp8,
            kv_scale,
            indices,
            sm_scale=dim ** -0.5,
            d_v=dim,
            out=out,
        )
    with pytest.raises(TypeError, match="out must be mx.float16"):
        path_c_module.sparse_mla_blockscaled_path_c_apply(
            q_fp8,
            q_scale,
            kv_fp8,
            kv_scale,
            indices,
            sm_scale=dim ** -0.5,
            d_v=dim,
            out=out,
            lse=lse,
        )
    with pytest.raises(TypeError, match="q_scale/kv_scale must be uint8 E8M0"):
        bad_q_scale = mx.zeros((batch, seq, heads, scale_blocks), dtype=mx.float32)
        path_c_module.sparse_mla_blockscaled_path_c_apply(
            q_fp8,
            bad_q_scale,
            kv_fp8,
            kv_scale,
            indices,
            sm_scale=dim ** -0.5,
            d_v=dim,
            out=mx.zeros((batch, seq, heads, dim), dtype=mx.float16),
            lse=lse,
        )


def test_blockscaled_path_c_apply_matches_path_b() -> None:
    if not _metal_available():
        pytest.skip("Metal backend not available on this host")

    rng = np.random.RandomState(12)
    batch, seq, heads, kv_group, seq_kv, topk, dim = 1, 2, 2, 1, 8, 4, 64
    q = mx.array((rng.standard_normal((batch, seq, heads, dim)) * 0.1).astype(np.float16))
    kv = mx.array((rng.standard_normal((batch, seq_kv, kv_group, dim)) * 0.1).astype(np.float16))
    indices = mx.array(rng.randint(0, seq_kv, size=(batch, seq, kv_group, topk)).astype(np.int32))
    sm_scale = dim ** -0.5

    q_packed, q_scales = _quantize_mxfp8(q)
    kv_packed, kv_scales = _quantize_mxfp8(kv)
    q_fp8 = _unpack_mxfp8_to_uint8(q_packed, dim)
    kv_fp8 = _unpack_mxfp8_to_uint8(kv_packed, dim)

    path_b = sparse_mla_blockscaled_fwd_metal(q, kv, indices, sm_scale=sm_scale, d_v=dim)
    if path_b is None:
        pytest.skip("Path B blockscaled Metal unavailable on this host")
    out_b, _lse_b = path_b
    out_c = sparse_mla_blockscaled_path_c_apply(
        q_fp8,
        q_scales,
        kv_fp8,
        kv_scales,
        indices,
        sm_scale=sm_scale,
        d_v=dim,
        force_path_c=True,
    )
    assert out_c is not None
    np.testing.assert_allclose(
        np.asarray(out_c.astype(mx.float32)),
        np.asarray(out_b.astype(mx.float32)),
        atol=2e-2,
        rtol=5e-2,
    )
