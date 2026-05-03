"""Tests for the Path B sparse-MLA port + pure-MLX reference parity oracle.

The Path B Metal kernel is currently blocked by tilelang 0.1.9's missing
``T.gemm`` lowering for the ``metal`` target. While that blocker is in place
the tests verify:

1. The pure-MLX reference matches a hand-rolled NumPy reference (forward
   parity oracle).
2. The reference is differentiable via mx.value_and_grad and gradient norms
   are finite.
3. The Path B status helpers report the blocker reason and the apply helper
   falls back to the reference rather than dispatching a half-built kernel.
4. The Metal kernel test surface is collected (skipped when tilelang is not
   importable or the GEMM blocker is active).

Tolerances: rtol=1e-3, atol=1e-3 for fp16 (plus generous fp32 hand checks).
"""

from __future__ import annotations

import importlib
import os

import numpy as np
import pytest

import mlx.core as mx

from cppmega_mlx.nn._tilelang.sparse_mla import (
    SparseMLAMetalStatus,
    sparse_mla_apply,
    sparse_mla_metal_status,
)
from cppmega_mlx.nn.sparse_mla import (
    sparse_mla_attention,
    sparse_mla_attention_reference,
)


# ---------------------------------------------------------------------------
# Hand-rolled NumPy reference (correctness oracle)
# ---------------------------------------------------------------------------


def _np_sparse_mla(
    q: np.ndarray,
    kv: np.ndarray,
    indices: np.ndarray,
    *,
    sm_scale: float,
    d_v: int,
) -> np.ndarray:
    """Per-token loop reference for sparse-MLA in float32."""

    B, S, H, qk_dim = q.shape
    _, Skv, G, _ = kv.shape
    head_kv = H // G
    out = np.zeros((B, S, H, d_v), dtype=np.float32)
    q32 = q.astype(np.float32)
    kv32 = kv.astype(np.float32)
    for b in range(B):
        for s in range(S):
            for g in range(G):
                k_indices = indices[b, s, g, :]
                valid = k_indices != -1
                gathered = kv32[b, np.maximum(k_indices, 0), g]
                for h_off in range(head_kv):
                    h = g * head_kv + h_off
                    qrow = q32[b, s, h, :]
                    scores = (qrow @ gathered.T) * sm_scale
                    scores = np.where(valid, scores, -np.inf)
                    if not valid.any():
                        out[b, s, h, :] = 0
                        continue
                    m = scores.max()
                    exp = np.exp(scores - m)
                    exp = np.where(valid, exp, 0.0)
                    probs = exp / exp.sum()
                    out[b, s, h, :] = probs @ gathered[:, :d_v]
    return out


# ---------------------------------------------------------------------------
# Shape grid for the parity oracle
# ---------------------------------------------------------------------------


SMOKE_SHAPES = [
    pytest.param(
        dict(B=2, S=128, H=8, D=64, G=1, topk=16, Skv=128),
        id="B2_S128_H8_D64",
    ),
    pytest.param(
        dict(B=4, S=512, H=8, D=64, G=1, topk=32, Skv=512),
        id="B4_S512_H8_D64",
    ),
    pytest.param(
        dict(B=1, S=64, H=8, D=64, G=2, topk=16, Skv=128),
        id="B1_S64_H8_D64_G2",
    ),
    pytest.param(
        dict(B=2, S=64, H=4, D=48, G=1, topk=16, Skv=96, d_v=32),
        id="tail_dim16",
    ),
]


# ---------------------------------------------------------------------------
# Forward parity (reference vs hand NumPy)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cfg", SMOKE_SHAPES)
def test_reference_matches_numpy_oracle(cfg) -> None:
    rng = np.random.default_rng(0)
    B, S, H, D = cfg["B"], cfg["S"], cfg["H"], cfg["D"]
    G = cfg["G"]
    topk = cfg["topk"]
    Skv = cfg["Skv"]
    d_v = cfg.get("d_v", D)

    q_np = rng.standard_normal((B, S, H, D)).astype(np.float16)
    kv_np = rng.standard_normal((B, Skv, G, D)).astype(np.float16)
    indices_np = rng.integers(0, Skv, size=(B, S, G, topk)).astype(np.int32)

    sm_scale = D ** -0.5

    out_mlx = sparse_mla_attention_reference(
        mx.array(q_np), mx.array(kv_np), mx.array(indices_np), sm_scale=sm_scale, d_v=d_v
    )
    mx.eval(out_mlx)
    out_np = np.array(out_mlx).astype(np.float32)

    ref = _np_sparse_mla(q_np, kv_np, indices_np, sm_scale=sm_scale, d_v=d_v)
    np.testing.assert_allclose(out_np, ref, atol=1e-3, rtol=1e-3)


# ---------------------------------------------------------------------------
# Mask handling: -1 sentinel zeros that token's output, isn't NaN
# ---------------------------------------------------------------------------


def test_invalid_indices_zero_output() -> None:
    rng = np.random.default_rng(1)
    B, S, H, D = 2, 8, 4, 32
    G = 1
    topk = 4
    Skv = 16

    q_np = rng.standard_normal((B, S, H, D)).astype(np.float16)
    kv_np = rng.standard_normal((B, Skv, G, D)).astype(np.float16)
    indices_np = rng.integers(0, Skv, size=(B, S, G, topk)).astype(np.int32)
    indices_np[0, 0, 0, :] = -1  # all invalid for first token

    out = sparse_mla_attention_reference(
        mx.array(q_np), mx.array(kv_np), mx.array(indices_np)
    )
    mx.eval(out)
    out_np = np.array(out)
    assert not np.isnan(out_np).any()
    np.testing.assert_array_equal(out_np[0, 0, 0], np.zeros(D, dtype=out_np.dtype))


def test_partial_invalid_indices_match_oracle() -> None:
    rng = np.random.default_rng(2)
    B, S, H, D = 1, 4, 4, 16
    G = 1
    topk = 6
    Skv = 8
    sm_scale = D ** -0.5

    q_np = rng.standard_normal((B, S, H, D)).astype(np.float16)
    kv_np = rng.standard_normal((B, Skv, G, D)).astype(np.float16)
    indices_np = rng.integers(0, Skv, size=(B, S, G, topk)).astype(np.int32)
    # Mask half the indices for token (0,1)
    indices_np[0, 1, 0, ::2] = -1

    out_mlx = sparse_mla_attention_reference(
        mx.array(q_np), mx.array(kv_np), mx.array(indices_np), sm_scale=sm_scale
    )
    mx.eval(out_mlx)
    ref = _np_sparse_mla(q_np, kv_np, indices_np, sm_scale=sm_scale, d_v=D)
    np.testing.assert_allclose(np.array(out_mlx).astype(np.float32), ref, atol=1e-3, rtol=1e-3)


# ---------------------------------------------------------------------------
# Backward parity: gradient norms should be finite and match between two
# autograd traces of the reference (anchors backward correctness through MLX).
# ---------------------------------------------------------------------------


def test_reference_backward_finite() -> None:
    rng = np.random.default_rng(3)
    B, S, H, D = 2, 16, 4, 32
    G = 1
    topk = 8
    Skv = 32

    q_np = rng.standard_normal((B, S, H, D)).astype(np.float32)
    kv_np = rng.standard_normal((B, Skv, G, D)).astype(np.float32)
    indices_np = rng.integers(0, Skv, size=(B, S, G, topk)).astype(np.int32)

    q = mx.array(q_np)
    kv = mx.array(kv_np)
    indices = mx.array(indices_np)

    def loss(q_in: mx.array, kv_in: mx.array) -> mx.array:
        out = sparse_mla_attention(q_in, kv_in, indices, sm_scale=D ** -0.5)
        return mx.mean(out * out)

    grads = mx.grad(loss, argnums=(0, 1))(q, kv)
    dq, dkv = grads
    mx.eval(dq, dkv)
    assert dq.shape == q.shape
    assert dkv.shape == kv.shape
    dq_np = np.array(dq)
    dkv_np = np.array(dkv)
    assert np.isfinite(dq_np).all()
    assert np.isfinite(dkv_np).all()
    # Gradients should be non-zero somewhere
    assert np.linalg.norm(dq_np) > 0
    assert np.linalg.norm(dkv_np) > 0


def test_reference_backward_against_finite_difference() -> None:
    """Spot-check a single q entry's gradient via central finite differences."""

    rng = np.random.default_rng(4)
    B, S, H, D = 1, 4, 2, 8
    G = 1
    topk = 3
    Skv = 6
    sm_scale = D ** -0.5

    q_np = rng.standard_normal((B, S, H, D)).astype(np.float32)
    kv_np = rng.standard_normal((B, Skv, G, D)).astype(np.float32)
    indices_np = rng.integers(0, Skv, size=(B, S, G, topk)).astype(np.int32)

    q = mx.array(q_np)
    kv = mx.array(kv_np)
    indices = mx.array(indices_np)

    def scalar_loss(q_in: mx.array) -> mx.array:
        out = sparse_mla_attention(q_in, kv, indices, sm_scale=sm_scale)
        return mx.sum(out)

    grad_q = mx.grad(scalar_loss)(q)
    mx.eval(grad_q)
    grad_q_np = np.array(grad_q)

    eps = 1e-3
    # Probe a handful of entries in q
    probes = [(0, 0, 0, 0), (0, 1, 1, 3), (0, 2, 0, 5)]
    for idx in probes:
        q_plus = q_np.copy()
        q_plus[idx] += eps
        q_minus = q_np.copy()
        q_minus[idx] -= eps
        loss_plus = float(np.array(scalar_loss(mx.array(q_plus))))
        loss_minus = float(np.array(scalar_loss(mx.array(q_minus))))
        fd = (loss_plus - loss_minus) / (2 * eps)
        analytic = float(grad_q_np[idx])
        np.testing.assert_allclose(analytic, fd, atol=5e-3, rtol=5e-3)


# ---------------------------------------------------------------------------
# Path B status surface
# ---------------------------------------------------------------------------


def test_metal_status_blocker_reported() -> None:
    """Until the GEMM blocker lifts the Metal path must not appear available."""

    status = sparse_mla_metal_status()
    assert isinstance(status, SparseMLAMetalStatus)
    assert status.available is False
    assert "tilelang" in status.reason.lower() or "metal" in status.reason.lower()


def test_apply_falls_back_to_reference() -> None:
    rng = np.random.default_rng(5)
    B, S, H, D = 2, 8, 4, 32
    G = 1
    topk = 4
    Skv = 16

    q_np = rng.standard_normal((B, S, H, D)).astype(np.float16)
    kv_np = rng.standard_normal((B, Skv, G, D)).astype(np.float16)
    indices_np = rng.integers(0, Skv, size=(B, S, G, topk)).astype(np.int32)

    out_apply = sparse_mla_apply(
        mx.array(q_np), mx.array(kv_np), mx.array(indices_np), sm_scale=D ** -0.5
    )
    out_ref = sparse_mla_attention_reference(
        mx.array(q_np), mx.array(kv_np), mx.array(indices_np), sm_scale=D ** -0.5
    )
    mx.eval(out_apply, out_ref)
    np.testing.assert_array_equal(np.array(out_apply), np.array(out_ref))


def test_apply_force_metal_raises_during_blocker() -> None:
    rng = np.random.default_rng(6)
    q = mx.array(rng.standard_normal((1, 4, 2, 8)).astype(np.float16))
    kv = mx.array(rng.standard_normal((1, 8, 1, 8)).astype(np.float16))
    indices = mx.array(rng.integers(0, 8, size=(1, 4, 1, 3)).astype(np.int32))
    with pytest.raises(RuntimeError):
        sparse_mla_apply(q, kv, indices, force_metal=True)


# ---------------------------------------------------------------------------
# Path B kernel suite (skipped while T.gemm blocker is active)
# ---------------------------------------------------------------------------


def _tilelang_metal_gemm_blocked() -> bool:
    try:
        importlib.import_module("tilelang")
    except Exception:
        return True
    # Until the blocker lifts the status helper reports unavailable.
    return not sparse_mla_metal_status().available


@pytest.mark.skipif(
    _tilelang_metal_gemm_blocked(),
    reason="Path B sparse-MLA kernel blocked by tilelang 0.1.9 metal T.gemm gap",
)
def test_path_b_forward_parity() -> None:
    # Placeholder: the moment the blocker lifts, replace with a real parity
    # check between sparse_mla_fwd_metal and the reference.
    pytest.skip("sparse_mla_fwd_metal is not yet implemented")


@pytest.mark.skipif(
    _tilelang_metal_gemm_blocked(),
    reason="Path B sparse-MLA kernel blocked by tilelang 0.1.9 metal T.gemm gap",
)
def test_path_b_backward_parity() -> None:
    pytest.skip("sparse_mla_bwd_metal is not yet implemented")
