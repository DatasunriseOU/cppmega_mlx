"""Parity + status tests for the Path B FP8 sparse-MLA port.

The FP8 path through tilelang's TVM-Metal lowering is blocked by both
``T.gemm`` registration and ``float8_e4m3 -> Metal type`` codegen (see
``cppmega_mlx/nn/_tilelang/sparse_mla_fp8.py`` module docstring). These tests
exercise:

1. The Metal-status surface returns ``available=False`` with the documented
   reason while the codegen blockers are in place.
2. ``sparse_mla_fp8_apply`` falls back to the FP8 reference and
   ``force_metal=True`` raises with the blocker message.
3. The pure-MLX FP8 reference matches a "dequantize-then-BF16" parity oracle
   exactly (because both paths consume the same FP8-recovered Q/KV).
4. The FP8 reference matches the original BF16 reference within FP8 noise
   tolerance (rtol=5e-3 on small inputs, where the e4m3 mantissa preserves
   enough precision for attention-scale outputs).
5. The MXFP8 quantized_matmul side-path runs and produces finite outputs.
6. MLX autograd flows backward through the FP8 reference cleanly, with grads
   that match a BF16 reference taken over the same recovered tensors within
   FP8 noise tolerance (rtol=1e-2).
"""

from __future__ import annotations

import numpy as np
import pytest

import mlx.core as mx

from cppmega_mlx.nn._tilelang.sparse_mla_fp8 import (
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
from cppmega_mlx.nn.sparse_mla import sparse_mla_attention_reference


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


def test_fp8_metal_status_returns_blocker_reason() -> None:
    status = sparse_mla_fp8_metal_status()
    assert isinstance(status, SparseMLAFp8MetalStatus)
    assert status.available is False
    assert "float8_e4m3" in status.reason
    assert "codegen_metal.cc:271" in status.reason


def test_fp8_metal_status_with_arrays_validates_dispatcher_path() -> None:
    q, kv, indices, _ = _make_inputs()
    status = sparse_mla_fp8_metal_status(q, kv, indices)
    assert status.available is False
    # Reason is either the dispatcher reason or the blocker reason (whichever
    # fires first); both branches are valid contract behavior.
    assert status.reason


def test_fp8_fwd_metal_returns_status_and_none_outputs() -> None:
    q, kv, indices, d_v = _make_inputs()
    status, out, lse = sparse_mla_fp8_fwd_metal(q, kv, indices, d_v=d_v)
    assert status.available is False
    assert out is None and lse is None


def test_fp8_bwd_metal_returns_status_and_none_outputs() -> None:
    q, kv, indices, d_v = _make_inputs()
    out_dummy = mx.zeros((1, 4, 2, d_v), dtype=mx.float32)
    grad_dummy = mx.zeros_like(out_dummy)
    lse_dummy = mx.zeros((1, 4, 2), dtype=mx.float32)
    status, dq, dkv = sparse_mla_fp8_bwd_metal(
        q, kv, out_dummy, grad_dummy, indices, lse_dummy, d_v=d_v
    )
    assert status.available is False
    assert dq is None and dkv is None


def test_fp8_apply_force_metal_raises_with_blocker_reason() -> None:
    q, kv, indices, d_v = _make_inputs()
    with pytest.raises(RuntimeError) as exc:
        sparse_mla_fp8_apply(q, kv, indices, d_v=d_v, force_metal=True)
    assert "Metal path unavailable" in str(exc.value)


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


def test_fp8_apply_falls_back_to_reference() -> None:
    q, kv, indices, d_v = _make_inputs()
    out_apply = sparse_mla_fp8_apply(q, kv, indices, d_v=d_v)
    out_ref = sparse_mla_fp8_reference(q, kv, indices, d_v=d_v)
    mx.eval(out_apply, out_ref)
    np.testing.assert_array_equal(np.asarray(out_apply), np.asarray(out_ref))


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
