"""Tests for the Path B sparse-MLA port + pure-MLX reference parity oracle.

The Path B Metal kernel is now available via direct-MSL bypass (see
``cppmega_mlx/nn/_tilelang/sparse_mla.py`` module docstring): we emit MSL
through ``mx.fast.metal_kernel`` directly, skipping TileLang's TVM-Metal
lowering entirely. The previous T.gemm blocker is bypassed.

The tests verify:

1. The pure-MLX reference matches a hand-rolled NumPy reference (forward
   parity oracle).
2. The reference is differentiable via mx.value_and_grad and gradient norms
   are finite.
3. The direct-MSL Path B kernel matches the pure-MLX reference within fp16
   tolerance (forward) and within autograd-grad tolerance (backward).
4. The metal status helper reports availability and ``sparse_mla_apply``
   exercises the Metal kernel with a fallback to the reference if needed.

Tolerances: rtol=1e-3, atol=1e-3 for fp16 (plus generous fp32 hand checks).
"""

from __future__ import annotations

import numpy as np
import pytest

import mlx.core as mx

from cppmega_mlx.nn._tilelang.sparse_mla import (  # noqa: E402
    SparseMLAMetalStatus,
    sparse_mla_apply,
    sparse_mla_bwd_metal,
    sparse_mla_fwd_metal,
    sparse_mla_metal_status,
)
from cppmega_mlx.nn.sparse_mla import (  # noqa: E402
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
    """Spot-check a single q entry's gradient via central finite differences.

    Targets the pure-MLX reference explicitly (independent of the production
    dispatcher) so the FD comparison stays at fp32 precision.
    """

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
        out = sparse_mla_attention_reference(q_in, kv, indices, sm_scale=sm_scale)
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


def test_metal_status_reports_available() -> None:
    """The direct-MSL bypass should report available on a Metal device."""

    status = sparse_mla_metal_status()
    assert isinstance(status, SparseMLAMetalStatus)
    # On a Metal-capable host the kernel must be available.
    if mx.metal.is_available():
        assert status.available is True
        assert status.fp16_carrier is True
    else:
        assert status.available is False


def test_apply_matches_reference_within_fp16_tolerance() -> None:
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
    # fp16 tolerance: the MSL kernel uses fp16 carrier with fp32 accumulators.
    np.testing.assert_allclose(
        np.array(out_apply).astype(np.float32),
        np.array(out_ref).astype(np.float32),
        rtol=1e-3,
        atol=2e-3,
    )


def test_apply_force_metal_dispatches_kernel() -> None:
    """force_metal=True must succeed now that the direct-MSL kernel exists."""

    rng = np.random.default_rng(6)
    q = mx.array(rng.standard_normal((1, 4, 2, 32)).astype(np.float16))
    kv = mx.array(rng.standard_normal((1, 8, 1, 32)).astype(np.float16))
    indices = mx.array(rng.integers(0, 8, size=(1, 4, 1, 4)).astype(np.int32))
    out = sparse_mla_apply(q, kv, indices, force_metal=True)
    mx.eval(out)
    assert tuple(out.shape) == (1, 4, 2, 32)


# ---------------------------------------------------------------------------
# Path B kernel parity (replaces the previous "blocked" placeholders).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "cfg",
    [
        dict(B=1, S=4, H=2, D=16, G=1, topk=4, Skv=8),
        dict(B=2, S=16, H=4, D=32, G=1, topk=8, Skv=32),
        dict(B=1, S=8, H=4, D=32, G=2, topk=8, Skv=16),
        dict(B=2, S=8, H=4, D=48, G=1, topk=8, Skv=16, d_v=32),
    ],
    ids=["small", "medium", "multigroup", "tail_dim"],
)
def test_path_b_forward_parity(cfg) -> None:
    rng = np.random.default_rng(13)
    B, S, H, D = cfg["B"], cfg["S"], cfg["H"], cfg["D"]
    G = cfg["G"]
    topk = cfg["topk"]
    Skv = cfg["Skv"]
    d_v = cfg.get("d_v")

    q = mx.array(rng.standard_normal((B, S, H, D)).astype(np.float16))
    kv = mx.array(rng.standard_normal((B, Skv, G, D)).astype(np.float16))
    indices = mx.array(rng.integers(0, Skv, size=(B, S, G, topk)).astype(np.int32))

    result = sparse_mla_fwd_metal(q, kv, indices, d_v=d_v)
    assert result is not None, "direct-MSL Path B kernel must dispatch"
    out_msl, lse = result
    mx.eval(out_msl, lse)

    out_ref = sparse_mla_attention_reference(q, kv, indices, d_v=d_v)
    mx.eval(out_ref)

    np.testing.assert_allclose(
        np.array(out_msl).astype(np.float32),
        np.array(out_ref).astype(np.float32),
        rtol=1e-3,
        atol=2e-3,
    )


def test_path_b_forward_parity_with_invalid_indices() -> None:
    """Sentinel handling: -1 indices should produce zero output for fully-masked tokens."""

    rng = np.random.default_rng(17)
    B, S, H, D = 2, 4, 2, 32
    G = 1
    topk = 4
    Skv = 8
    q = mx.array(rng.standard_normal((B, S, H, D)).astype(np.float16))
    kv = mx.array(rng.standard_normal((B, Skv, G, D)).astype(np.float16))
    ind_np = rng.integers(0, Skv, size=(B, S, G, topk)).astype(np.int32)
    ind_np[0, 0, 0, :] = -1  # all invalid for first token
    indices = mx.array(ind_np)

    result = sparse_mla_fwd_metal(q, kv, indices)
    assert result is not None
    out, _ = result
    mx.eval(out)
    out_np = np.array(out)
    assert not np.isnan(out_np).any()
    np.testing.assert_array_equal(out_np[0, 0, 0], np.zeros(D, dtype=out_np.dtype))


def test_path_b_backward_parity() -> None:
    rng = np.random.default_rng(23)
    B, S, H, D = 2, 8, 4, 32
    G = 1
    topk = 8
    Skv = 16

    q = mx.array(rng.standard_normal((B, S, H, D)).astype(np.float32))
    kv = mx.array(rng.standard_normal((B, Skv, G, D)).astype(np.float32))
    indices = mx.array(rng.integers(0, Skv, size=(B, S, G, topk)).astype(np.int32))
    d_out = mx.array(rng.standard_normal((B, S, H, D)).astype(np.float32))

    grads = sparse_mla_bwd_metal(q, kv, d_out, indices)
    assert grads is not None
    dq_msl, dkv_msl = grads
    mx.eval(dq_msl, dkv_msl)

    # Reference: autograd of pure-MLX path.
    def loss(q_, kv_):
        out = sparse_mla_attention_reference(q_, kv_, indices)
        return mx.sum(out * d_out)

    dq_ref, dkv_ref = mx.grad(loss, argnums=(0, 1))(q, kv)
    mx.eval(dq_ref, dkv_ref)

    # fp16 carrier means slightly looser tolerance than fp32.
    np.testing.assert_allclose(
        np.array(dq_msl).astype(np.float32),
        np.array(dq_ref).astype(np.float32),
        rtol=5e-3,
        atol=5e-3,
    )
    np.testing.assert_allclose(
        np.array(dkv_msl).astype(np.float32),
        np.array(dkv_ref).astype(np.float32),
        rtol=5e-3,
        atol=5e-3,
    )


def test_apply_backward_through_custom_vjp() -> None:
    """``sparse_mla_apply`` must propagate gradients via the custom VJP."""

    rng = np.random.default_rng(29)
    B, S, H, D = 1, 4, 2, 16
    G = 1
    topk = 4
    Skv = 8
    q = mx.array(rng.standard_normal((B, S, H, D)).astype(np.float32))
    kv = mx.array(rng.standard_normal((B, Skv, G, D)).astype(np.float32))
    indices = mx.array(rng.integers(0, Skv, size=(B, S, G, topk)).astype(np.int32))

    def loss(q_, kv_):
        out = sparse_mla_apply(q_, kv_, indices)
        return mx.sum(out * out)

    dq, dkv = mx.grad(loss, argnums=(0, 1))(q, kv)
    mx.eval(dq, dkv)
    assert np.all(np.isfinite(np.array(dq)))
    assert np.all(np.isfinite(np.array(dkv)))
    assert np.linalg.norm(np.array(dq)) > 0
    assert np.linalg.norm(np.array(dkv)) > 0
