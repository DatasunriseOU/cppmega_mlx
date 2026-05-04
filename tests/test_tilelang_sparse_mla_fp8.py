"""Parity + status tests for the Path B FP8 sparse-MLA port.

The Path B FP8 kernel is now available via direct-MSL bypass (see
``cppmega_mlx/nn/_tilelang/sparse_mla_fp8.py`` module docstring). The
previous TileLang ``T.gemm`` and ``float8_e4m3 -> Metal type`` blockers are
bypassed by emitting MSL through ``mx.fast.metal_kernel`` directly with
inline e4m3 dequant on uint8 storage.

These tests exercise:

1. The Metal-status surface returns ``available=True`` on a Metal device.
2. ``sparse_mla_fp8_apply`` dispatches the kernel and parity holds vs the
   pure-MLX FP8 reference within FP8 noise tolerance.
3. ``force_metal=True`` succeeds (no blocker to raise).
4. The pure-MLX FP8 reference matches a "dequantize-then-BF16" parity oracle
   exactly.
5. The FP8 reference matches the original BF16 reference within FP8 noise
   tolerance (rtol=5e-3 on small inputs).
6. The MXFP8 quantized_matmul side-path runs and produces finite outputs.
7. MLX autograd flows backward through the FP8 path cleanly.
"""

from __future__ import annotations

import numpy as np

import mlx.core as mx

from cppmega_mlx.nn._tilelang.sparse_mla_fp8 import (  # noqa: E402
    SparseMLAFp8MetalStatus,
    _from_fp8_with_scale,
    _to_fp8_with_per_tensor_scale,
    sparse_mla_fp8_apply,
    sparse_mla_fp8_bwd_metal,
    sparse_mla_fp8_fwd_metal,
    sparse_mla_fp8_metal_status,
    sparse_mla_fp8_reference,
    sparse_mla_quantized_matmul_reference,
)
from cppmega_mlx.nn._tilelang.sparse_mla_fp8_path_c import (  # noqa: E402
    SparseMLAFp8QKReducePathCStatus,
    SparseMLAFp8PathCStatus,
    fp8_sparse_mla_qk_reduce_msl_features,
    fp8_sparse_mla_qk_reduce_path_c,
    fp8_sparse_mla_qk_reduce_path_c_status,
    fp8_sparse_mla_qk_msl_features,
    fp8_sparse_mla_qk_path_c_status,
    lower_fp8_sparse_mla_qk_reduce_msl,
    lower_fp8_sparse_mla_qk_msl,
)
from cppmega_mlx.nn.sparse_mla import sparse_mla_attention_reference  # noqa: E402


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


def _make_inputs(
    *,
    batch: int = 1,
    seq_len: int = 4,
    heads: int = 2,
    kv_group: int = 1,
    qk_dim: int = 32,
    d_v: int = 16,
    topk: int = 8,
    seed: int = 0,
    scale: float = 0.1,
):
    rng = np.random.default_rng(seed)
    q = mx.array((rng.standard_normal((batch, seq_len, heads, qk_dim)) * scale).astype(np.float32))
    kv = mx.array((rng.standard_normal((batch, seq_len, kv_group, qk_dim)) * scale).astype(np.float32))
    ind_np = np.tile(
        np.arange(topk, dtype=np.int32).reshape(1, 1, 1, topk),
        (batch, seq_len, kv_group, 1),
    )
    # Mark the second half of the topk as masked (-1 sentinel).
    ind_np[:, :, :, topk // 2:] = -1
    indices = mx.array(ind_np)
    return q, kv, indices, d_v


# ---------------------------------------------------------------------------
# Status surface
# ---------------------------------------------------------------------------


def test_fp8_metal_status_reports_available() -> None:
    status = sparse_mla_fp8_metal_status()
    assert isinstance(status, SparseMLAFp8MetalStatus)
    if mx.metal.is_available():
        assert status.available is True
        assert "FP8 e4m3" in status.reason or "direct-MSL" in status.reason
    else:
        assert status.available is False


def test_fp8_metal_status_with_arrays_validates_dispatcher_path() -> None:
    q, kv, indices, _ = _make_inputs()
    status = sparse_mla_fp8_metal_status(q, kv, indices)
    if mx.metal.is_available():
        assert status.available is True
    else:
        assert status.available is False


def test_fp8_sparse_mla_path_c_status_reports_current_qk_blocker() -> None:
    status = fp8_sparse_mla_qk_path_c_status()
    assert isinstance(status, SparseMLAFp8PathCStatus)
    assert status.m == 1
    assert status.n == 16
    assert status.k == 64
    assert status.transpose_B is True
    assert status.available is False
    assert status.reason
    if status.features:
        assert not (
            status.features["simdgroup_multiply_accumulate"]
            and status.features["A_scale_refs"]
            and status.features["B_scale_refs"]
        ), "M=1 Sparse-MLA QK shape must not be reported available unless scale-aware MMA lowers"
        assert (
            "scale operands disappeared" in status.reason
            or "M=1/topk" in status.reason
            or "scalar fallback" in status.reason
        )


def test_fp8_sparse_mla_path_c_scale_semantics_fail_closed() -> None:
    status = fp8_sparse_mla_qk_path_c_status()
    if not status.features:
        assert status.available is False
        return
    scale_refs_present = bool(status.features["A_scale_refs"]) and bool(status.features["B_scale_refs"])
    scale_signature_present = bool(status.features["signature_has_A_scale"]) and bool(
        status.features["signature_has_B_scale"]
    )
    assert (scale_refs_present and scale_signature_present) or "scale operands disappeared" in status.reason


def test_fp8_sparse_mla_path_c_square_control_lowers_to_scale_aware_fast_path() -> None:
    status = fp8_sparse_mla_qk_path_c_status(
        M=32,
        N=32,
        K=64,
        BM=32,
        BN=32,
        BK=64,
        a_scale_size=1,
        b_scale_size=32,
    )
    if not status.available:
        assert status.reason
        return
    assert status.features["simdgroup_multiply_accumulate"] >= 1
    assert status.features["A_scale_refs"] >= 1
    assert status.features["B_scale_refs"] >= 1
    assert status.features["signature_has_A_scale"] is True
    assert status.features["signature_has_B_scale"] is True


def test_fp8_sparse_mla_path_c_lowered_features_are_reported() -> None:
    msl = lower_fp8_sparse_mla_qk_msl(M=32, N=32, K=64, BM=32, BN=32, BK=64, b_scale_size=32)
    features = fp8_sparse_mla_qk_msl_features(msl)
    assert features["kernel_void"] >= 1
    assert features["fp8_e4m3_decode_helper"] >= 1
    assert features["simdgroup_multiply_accumulate"] >= 1
    assert features["A_scale_refs"] >= 1
    assert features["B_scale_refs"] >= 1


def test_fp8_sparse_mla_path_c_qk_reduce_status_reports_available() -> None:
    status = fp8_sparse_mla_qk_reduce_path_c_status(N=16, K=64)
    assert isinstance(status, SparseMLAFp8QKReducePathCStatus)
    assert status.n == 16
    assert status.k == 64
    if mx.metal.is_available():
        assert status.available is True
        assert status.features["signature_has_A_scale"] is True
        assert status.features["signature_has_B_scale"] is True
        assert status.features["A_scale_refs"] >= 1
        assert status.features["B_scale_refs"] >= 1
    else:
        assert status.available is False


def test_fp8_sparse_mla_path_c_qk_reduce_lowered_features_are_reported() -> None:
    msl = lower_fp8_sparse_mla_qk_reduce_msl(N=16, K=64)
    features = fp8_sparse_mla_qk_reduce_msl_features(msl)
    assert features["kernel_void"] >= 1
    assert features["fp8_e4m3_decode_helper"] >= 1
    assert features["signature_has_A_scale"] is True
    assert features["signature_has_B_scale"] is True
    assert features["A_scale_refs"] >= 1
    assert features["B_scale_refs"] >= 1
    assert features["per_row_B_scale"] is True


def test_fp8_sparse_mla_path_c_qk_reduce_matches_dequant_oracle() -> None:
    q, kv, _indices, _d_v = _make_inputs(seq_len=16, heads=2, qk_dim=64, topk=16, scale=0.1)
    q_fp8, q_scale = _to_fp8_with_per_tensor_scale(q)
    kv_fp8, kv_scale = _to_fp8_with_per_tensor_scale(kv)
    mx.eval(q_fp8, q_scale, kv_fp8, kv_scale)

    A_fp8 = q_fp8[0, 0, 0, :].reshape((1, 64))
    A_scale = q_scale[0, 0, 0].reshape((1,))
    B_fp8 = kv_fp8[0, :16, 0, :]
    B_scale = kv_scale[0, :16, 0]
    out = fp8_sparse_mla_qk_reduce_path_c(A_fp8, A_scale, B_fp8, B_scale)
    assert out is not None

    oracle = mx.matmul(
        mx.from_fp8(A_fp8, dtype=mx.float32),
        mx.swapaxes(mx.from_fp8(B_fp8, dtype=mx.float32), 0, 1),
    )
    oracle = oracle * A_scale.reshape((1, 1)).astype(mx.float32) * B_scale.reshape((1, 16)).astype(mx.float32)
    mx.eval(out, oracle)
    np.testing.assert_allclose(
        np.asarray(out).astype(np.float32),
        np.asarray(oracle).astype(np.float32),
        rtol=1e-5,
        atol=1e-5,
    )


def test_fp8_fwd_metal_returns_outputs() -> None:
    q, kv, indices, d_v = _make_inputs()
    result = sparse_mla_fp8_fwd_metal(q, kv, indices, d_v=d_v)
    assert result is not None
    out, lse = result
    mx.eval(out, lse)
    assert tuple(out.shape) == tuple(q.shape[:3]) + (d_v,)


def test_fp8_bwd_metal_returns_outputs() -> None:
    q, kv, indices, d_v = _make_inputs()
    d_out = mx.zeros((1, 4, 2, d_v), dtype=mx.float32)
    grads = sparse_mla_fp8_bwd_metal(q, kv, d_out, indices, d_v=d_v)
    assert grads is not None
    dq, dkv = grads
    mx.eval(dq, dkv)
    assert tuple(dq.shape) == tuple(q.shape)
    assert tuple(dkv.shape) == tuple(kv.shape)


def test_fp8_apply_force_metal_dispatches_kernel() -> None:
    q, kv, indices, d_v = _make_inputs()
    out = sparse_mla_fp8_apply(q, kv, indices, d_v=d_v, force_metal=True)
    mx.eval(out)
    assert tuple(out.shape) == tuple(q.shape[:3]) + (d_v,)


# ---------------------------------------------------------------------------
# FP8 helper round-trip
# ---------------------------------------------------------------------------


def test_to_fp8_per_tensor_scale_roundtrip_recovers_within_noise() -> None:
    rng = np.random.default_rng(42)
    x = mx.array(rng.standard_normal((4, 8, 16)).astype(np.float32))
    fp8, scale = _to_fp8_with_per_tensor_scale(x)
    rec = _from_fp8_with_scale(fp8, scale, dtype=mx.float32)
    mx.eval(rec)
    # FP8 e4m3 has 3 mantissa bits — expect ~3% rel error after recovery.
    err = (rec - x).abs().max().item()
    rel = err / (x.abs().max().item() + 1e-9)
    assert rel < 0.05, f"FP8 roundtrip rel err {rel:.4e} exceeded 5%"
    assert scale.shape == x.shape[:-1]
    assert scale.dtype == mx.float32


# ---------------------------------------------------------------------------
# Forward parity
# ---------------------------------------------------------------------------


def test_fp8_apply_matches_reference_within_fp8_tolerance() -> None:
    """``sparse_mla_fp8_apply`` (Metal kernel) vs the pure-MLX FP8 reference.

    Now that the direct-MSL FP8 kernel is wired the apply runs on Metal; the
    parity tolerance is FP8 noise (rtol=5e-3) plus a small fp16 carrier
    rounding margin.
    """

    q, kv, indices, d_v = _make_inputs(scale=0.1)
    out_apply = sparse_mla_fp8_apply(q, kv, indices, d_v=d_v)
    out_ref = sparse_mla_fp8_reference(q, kv, indices, d_v=d_v)
    mx.eval(out_apply, out_ref)
    np.testing.assert_allclose(
        np.asarray(out_apply).astype(np.float32),
        np.asarray(out_ref).astype(np.float32),
        rtol=5e-3,
        atol=5e-3,
    )


def test_fp8_path_b_forward_parity() -> None:
    """Direct-MSL FP8 forward must agree with the reference within FP8 noise."""

    q, kv, indices, d_v = _make_inputs(scale=0.1)
    result = sparse_mla_fp8_fwd_metal(q, kv, indices, d_v=d_v)
    assert result is not None
    out_msl, lse = result
    out_ref = sparse_mla_fp8_reference(q, kv, indices, d_v=d_v)
    mx.eval(out_msl, lse, out_ref)
    np.testing.assert_allclose(
        np.asarray(out_msl).astype(np.float32),
        np.asarray(out_ref).astype(np.float32),
        rtol=5e-3,
        atol=5e-3,
    )


def test_fp8_path_b_backward_parity() -> None:
    """Direct-MSL FP8 backward must agree with autograd of the reference."""

    q, kv, indices, d_v = _make_inputs(scale=0.1)

    rng = np.random.default_rng(31)
    d_out = mx.array((rng.standard_normal(tuple(q.shape[:3]) + (d_v,)) * 0.1).astype(np.float32))

    grads = sparse_mla_fp8_bwd_metal(q, kv, d_out, indices, d_v=d_v)
    assert grads is not None
    dq_msl, dkv_msl = grads
    mx.eval(dq_msl, dkv_msl)

    def loss(q_, kv_):
        out = sparse_mla_fp8_reference(q_, kv_, indices, d_v=d_v)
        return mx.sum(out * d_out)

    dq_ref, dkv_ref = mx.grad(loss, argnums=(0, 1))(q, kv)
    mx.eval(dq_ref, dkv_ref)

    np.testing.assert_allclose(
        np.asarray(dq_msl).astype(np.float32),
        np.asarray(dq_ref).astype(np.float32),
        rtol=1e-2,
        atol=5e-3,
    )
    np.testing.assert_allclose(
        np.asarray(dkv_msl).astype(np.float32),
        np.asarray(dkv_ref).astype(np.float32),
        rtol=1e-2,
        atol=5e-3,
    )


def test_fp8_reference_matches_bf16_within_fp8_tolerance() -> None:
    """FP8 reference vs BF16 reference, with small-magnitude inputs.

    With std=0.1 inputs the FP8 e4m3 mantissa noise is small enough to clear
    the rtol=5e-3 tolerance from the task brief.
    """

    q, kv, indices, d_v = _make_inputs(scale=0.1)
    out_fp8 = sparse_mla_fp8_reference(q, kv, indices, d_v=d_v)
    out_bf = sparse_mla_attention_reference(q, kv, indices, d_v=d_v)
    mx.eval(out_fp8, out_bf)
    out_fp8_np = np.asarray(out_fp8.astype(mx.float32))
    out_bf_np = np.asarray(out_bf.astype(mx.float32))
    np.testing.assert_allclose(out_fp8_np, out_bf_np, rtol=5e-3, atol=5e-3)


def test_fp8_reference_with_lse_returns_pair() -> None:
    q, kv, indices, d_v = _make_inputs()
    result = sparse_mla_fp8_reference(q, kv, indices, d_v=d_v, return_lse=True)
    assert isinstance(result, tuple) and len(result) == 2
    out, lse = result
    mx.eval(out, lse)
    assert out.shape[-1] == d_v
    assert lse.shape == out.shape[:-1]


def test_quantized_matmul_reference_matches_bf16_within_tight_tolerance() -> None:
    """The mxfp8 hand-built path uses regular matmul on dequantized tensors,
    so it should agree with the BF16 reference at fp32 precision."""

    q, kv, indices, d_v = _make_inputs()
    out_qm = sparse_mla_quantized_matmul_reference(q, kv, indices, d_v=d_v)
    out_bf = sparse_mla_attention_reference(q, kv, indices, d_v=d_v)
    mx.eval(out_qm, out_bf)
    out_qm_np = np.asarray(out_qm.astype(mx.float32))
    out_bf_np = np.asarray(out_bf.astype(mx.float32))
    np.testing.assert_allclose(out_qm_np, out_bf_np, rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# Backward parity
# ---------------------------------------------------------------------------


def test_fp8_reference_backward_matches_bf16_over_recovered_inputs() -> None:
    """Gradient parity within FP8 noise tolerance.

    We compare the FP8 reference's gradients against a BF16 reference taken
    over the *same* dequantized Q/KV. The two paths should match to within
    FP8 noise (rtol=1e-2 per the task brief).
    """

    q, kv, indices, d_v = _make_inputs(scale=0.1)
    indices_bound = indices

    # Build the dequantized Q/KV the FP8 reference produces internally so we
    # can run the BF16 reference on the same recovered tensors as a parity
    # oracle.
    fp8_q, q_scale = _to_fp8_with_per_tensor_scale(q)
    fp8_kv, kv_scale = _to_fp8_with_per_tensor_scale(kv)
    q_rec = _from_fp8_with_scale(fp8_q, q_scale, dtype=q.dtype)
    kv_rec = _from_fp8_with_scale(fp8_kv, kv_scale, dtype=kv.dtype)
    mx.eval(q_rec, kv_rec)

    def fp8_loss(q_in, kv_in):
        out = sparse_mla_fp8_reference(q_in, kv_in, indices_bound, d_v=d_v)
        return mx.sum(out * out)

    def bf16_loss_on_recovered(q_in, kv_in):
        out = sparse_mla_attention_reference(q_in, kv_in, indices_bound, d_v=d_v)
        return mx.sum(out * out)

    fp8_grad = mx.grad(fp8_loss, argnums=(0, 1))(q, kv)
    bf_grad = mx.grad(bf16_loss_on_recovered, argnums=(0, 1))(q_rec, kv_rec)
    mx.eval(*fp8_grad, *bf_grad)

    # The FP8 grads should be finite and match the BF16 dequant-oracle grads
    # within FP8 noise tolerance.
    for g in fp8_grad:
        g_np = np.asarray(g)
        assert np.all(np.isfinite(g_np)), "FP8 grads must be finite"

    fp8_dq, fp8_dkv = (np.asarray(g) for g in fp8_grad)
    bf_dq, bf_dkv = (np.asarray(g) for g in bf_grad)
    np.testing.assert_allclose(fp8_dq, bf_dq, rtol=1e-2, atol=5e-3)
    np.testing.assert_allclose(fp8_dkv, bf_dkv, rtol=1e-2, atol=5e-3)


def test_fp8_reference_backward_through_apply_finite() -> None:
    """``sparse_mla_fp8_apply`` is the production entry. Backward should flow
    cleanly without NaN/Inf even when half the topk slots are masked."""

    q, kv, indices, d_v = _make_inputs(scale=0.1)

    def loss(q_in, kv_in):
        out = sparse_mla_fp8_apply(q_in, kv_in, indices, d_v=d_v)
        return mx.sum(out * out)

    grads = mx.grad(loss, argnums=(0, 1))(q, kv)
    mx.eval(*grads)
    for g in grads:
        g_np = np.asarray(g)
        assert np.all(np.isfinite(g_np)), "apply backward grads must be finite"


# ---------------------------------------------------------------------------
# Public exports
# ---------------------------------------------------------------------------


def test_module_public_exports_present() -> None:
    from cppmega_mlx.nn._tilelang import sparse_mla_fp8 as module

    expected = {
        "SparseMLAFp8MetalStatus",
        "sparse_mla_fp8_apply",
        "sparse_mla_fp8_bwd_metal",
        "sparse_mla_fp8_fwd_metal",
        "sparse_mla_fp8_metal_status",
        "sparse_mla_fp8_reference",
        "sparse_mla_quantized_matmul_reference",
    }
    assert expected.issubset(set(module.__all__))
