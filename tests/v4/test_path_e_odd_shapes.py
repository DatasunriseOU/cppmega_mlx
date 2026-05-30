"""Path E odd-shape parity (Metal): the Dk remainder-mask fix.

The vendored fast Metal forward kernel used to require ``Dk % 32 == 0`` because
``n_per_t = Dk / 32`` (integer floor) silently dropped the trailing ``Dk % 32``
keys -> wrong answer. The in-MSL remainder-mask (ceil tiling + per-i
``s_idx < Dk`` guards; vectorized gate decay=1.0 on the tail) makes the forward
kernel correct for ANY Dk.

These tests pin parity of the fast kernel against the pure-MLX
``gated_delta_ops`` reference (no shape limit) for non-multiple-of-32 Dk, the
vectorized KDA gate (the decisive tail-masking case), asymmetric Dk!=Dv, group
expansion Hv>Hk, and state round-trip. They also assert the training/VJP path
still fail-closes for Dv % 4 != 0.

All require Metal; skipped otherwise.
"""

from __future__ import annotations

import mlx.core as mx
import pytest

from cppmega_v4.nn._external._mlx_lm_gated_delta_vendored import (
    gated_delta_kernel,
    gated_delta_ops,
    gated_delta_update,
)
from cppmega_v4.nn._external._path_e_eligibility import (
    shape_uses_fast_kernel,
    shape_uses_fast_kernel_backward,
)

pytestmark = pytest.mark.skipif(
    not mx.metal.is_available(), reason="Path E fast kernel requires Metal"
)


def _rel(a: mx.array, b: mx.array) -> float:
    return float(mx.max(mx.abs(a - b)) / (mx.max(mx.abs(b)) + 1e-6))


def _make_inputs(B, T, Hk, Hv, Dk, Dv, *, vectorized, seed=0):
    mx.random.seed(seed)
    q = mx.random.normal((B, T, Hk, Dk))
    k = mx.random.normal((B, T, Hk, Dk))
    v = mx.random.normal((B, T, Hv, Dv))
    beta = mx.sigmoid(mx.random.normal((B, T, Hv)))
    if vectorized:
        g = mx.exp(-mx.abs(mx.random.normal((B, T, Hv, Dk)) * 0.1))
    else:
        g = mx.exp(-mx.abs(mx.random.normal((B, T, Hv)) * 0.1))
    state = mx.zeros((B, Hv, Dv, Dk), dtype=mx.float32)
    return q, k, v, g, beta, state


# Dk %32 != 0 cases are the regression targets: 40, 96, 100, 192.
_DKS = [32, 40, 64, 96, 100, 128, 192]
_DVS = [4, 5, 6, 16, 17, 128]


@pytest.mark.parametrize("Dk", _DKS)
@pytest.mark.parametrize("Dv", _DVS)
def test_scalar_forward_parity_odd_shapes(Dk, Dv):
    """Scalar-gate (GDN) forward parity vs the ops reference for any Dk/Dv."""
    args = _make_inputs(1, 6, 2, 2, Dk, Dv, vectorized=False, seed=Dk * 7 + Dv)
    yk, sk = gated_delta_kernel(*args)
    yo, so = gated_delta_ops(*args)
    mx.eval(yk, sk, yo, so)
    assert _rel(yk, yo) < 1e-4, f"y mismatch Dk={Dk} Dv={Dv}"
    assert _rel(sk, so) < 1e-4, f"state mismatch Dk={Dk} Dv={Dv}"


@pytest.mark.parametrize("Dk", [40, 96, 100, 192])
@pytest.mark.parametrize("Dv", [4, 16, 128])
def test_vectorized_gate_forward_parity_odd_dk(Dk, Dv):
    """KDA per-Dk gate with Dk %32 != 0 — the decisive tail-masking case.

    If the masked tail lane read g OOB (or used a non-identity decay), the
    pre-slice ``simd_sum(kv_mem)`` would be poisoned and parity would break.
    """
    args = _make_inputs(1, 6, 2, 2, Dk, Dv, vectorized=True, seed=Dk * 11 + Dv)
    yk, sk = gated_delta_kernel(*args)
    yo, so = gated_delta_ops(*args)
    mx.eval(yk, sk, yo, so)
    assert _rel(yk, yo) < 1e-4, f"vec y mismatch Dk={Dk} Dv={Dv}"
    assert _rel(sk, so) < 1e-4, f"vec state mismatch Dk={Dk} Dv={Dv}"


@pytest.mark.parametrize("Dk,Dv", [(192, 128), (96, 64), (40, 17)])
def test_asymmetric_dk_ne_dv(Dk, Dv):
    args = _make_inputs(1, 5, 2, 2, Dk, Dv, vectorized=False, seed=Dk + Dv)
    yk, sk = gated_delta_kernel(*args)
    yo, so = gated_delta_ops(*args)
    mx.eval(yk, sk, yo, so)
    assert _rel(yk, yo) < 1e-4
    assert _rel(sk, so) < 1e-4


@pytest.mark.parametrize("Hk,Hv,Dk,Dv", [(2, 4, 40, 16), (2, 8, 96, 64), (1, 4, 100, 17)])
def test_group_expansion_hv_gt_hk_odd_dk(Hk, Hv, Dk, Dv):
    args = _make_inputs(1, 6, Hk, Hv, Dk, Dv, vectorized=False, seed=Hv * 13 + Dk)
    yk, sk = gated_delta_kernel(*args)
    yo, so = gated_delta_ops(*args)
    mx.eval(yk, sk, yo, so)
    assert _rel(yk, yo) < 1e-4
    assert _rel(sk, so) < 1e-4


@pytest.mark.parametrize("Dk,Dv,vec", [(100, 17, False), (40, 16, True), (192, 128, False)])
def test_state_round_trip_odd_dk(Dk, Dv, vec):
    """Splitting T and feeding the final state back must equal the full run."""
    q, k, v, g, beta, st0 = _make_inputs(1, 8, 2, 2, Dk, Dv, vectorized=vec, seed=5)
    yf, sf = gated_delta_kernel(q, k, v, g, beta, st0)
    gA = g[:, :4]
    gB = g[:, 4:]
    y1, s1 = gated_delta_kernel(q[:, :4], k[:, :4], v[:, :4], gA, beta[:, :4], st0)
    y2, s2 = gated_delta_kernel(q[:, 4:], k[:, 4:], v[:, 4:], gB, beta[:, 4:], s1)
    ycat = mx.concatenate([y1, y2], axis=1)
    mx.eval(yf, sf, ycat, s2)
    assert _rel(ycat, yf) < 1e-5
    assert _rel(s2, sf) < 1e-5


def test_eligibility_forward_relaxed_backward_keeps_dv4():
    # Forward: any Dk/Dv is fast now.
    assert shape_uses_fast_kernel(40, 16)
    assert shape_uses_fast_kernel(100, 17)
    assert shape_uses_fast_kernel(192, 128)
    # Backward: Dk lifted, Dv % 4 still required.
    assert shape_uses_fast_kernel_backward(40, 16)
    assert shape_uses_fast_kernel_backward(100, 128)
    assert not shape_uses_fast_kernel_backward(40, 17)
    assert not shape_uses_fast_kernel_backward(64, 6)


def test_training_fail_closes_for_dv_not_mult4():
    """Training with Dv % 4 != 0 must NOT hit the Metal VJP (no crash); it
    silently falls back to the Python-ops VJP reference. We assert it runs and
    produces finite output (the fast backward is gated off for ragged Dv)."""
    B, T, Hk, Hv, Dk, Dv = 1, 6, 2, 2, 40, 17  # Dv % 4 != 0
    q = mx.random.normal((B, T, Hk, Dk))
    k = mx.random.normal((B, T, Hk, Dk))
    v = mx.random.normal((B, T, Hv, Dv))
    a = mx.random.normal((B, T, Hv))
    b = mx.random.normal((B, T, Hv))
    A_log = mx.zeros((Hv,))
    dt_bias = mx.zeros((Hv,))
    # training=True with Dv%4 != 0: can_use_metal is False -> Python VJP ref.
    y, _ = gated_delta_update(
        q, k, v, a, b, A_log, dt_bias, training=True
    )
    mx.eval(y)
    assert y.shape == (B, T, Hv, Dv)
    assert not bool(mx.any(mx.isnan(y)).item())
