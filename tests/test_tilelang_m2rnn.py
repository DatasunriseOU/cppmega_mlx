"""Tests for the retired M2RNN direct-MSL compatibility surface.

Coverage:
  - :func:`m2rnn_metal_status` reports the retired fail-closed reason.
  - Forward parity vs :func:`m2rnn_scan` reference (FP32 atol=1e-4 rtol=1e-3,
    FP16 atol=2e-3 rtol=5e-3).
  - VJP through :func:`m2rnn_apply` matches autograd through the reference.
  - :func:`m2rnn_apply_with_state` returns ``(y, h_last)`` with shapes /
    values matching the reference scan.
  - Dispatch-friendly behavior: empty seq, broadcast heads, and reuse via
    the bwd kernel.
"""

from __future__ import annotations

import importlib

from typing import cast

import numpy as np
import pytest

import mlx.core as mx

from cppmega_mlx.nn._tilelang import (
    M2RNNMetalStatus,
    m2rnn_apply,
    m2rnn_apply_with_state,
    m2rnn_bwd_metal,
    m2rnn_fwd_metal,
    m2rnn_metal_status,
    m2rnn_reference,
)


def _np(x: mx.array) -> np.ndarray:
    if x.dtype == mx.bfloat16:
        x = x.astype(mx.float32)
    mx.eval(x)
    return np.asarray(x)


def _make_inputs(
    *,
    batch: int,
    seq: int,
    heads: int,
    k_dim: int,
    v_dim: int,
    dtype: mx.Dtype,
    seed: int = 17,
) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array, mx.array]:
    mx.random.seed(seed)
    q = (mx.random.normal((batch, seq, heads, k_dim)) * 0.1).astype(dtype)
    k = (mx.random.normal((batch, seq, heads, k_dim)) * 0.1).astype(dtype)
    v = (mx.random.normal((batch, seq, heads, v_dim)) * 0.1).astype(dtype)
    eye = mx.broadcast_to(mx.eye(v_dim)[None], (heads, v_dim, v_dim))
    W = (eye + mx.random.normal((heads, v_dim, v_dim)) * 0.05).astype(dtype)
    xf = mx.random.uniform(0.1, 0.9, (batch, seq, heads)).astype(dtype)
    h0 = mx.zeros((batch, heads, k_dim, v_dim), dtype=dtype)
    mx.eval(q, k, v, W, xf, h0)
    return q, k, v, W, xf, h0


# ---------------------------------------------------------------------------
# Status probe
# ---------------------------------------------------------------------------


def test_status_is_available_or_explains_why() -> None:
    status = m2rnn_metal_status()
    assert isinstance(status, M2RNNMetalStatus)
    assert status.available is False
    assert "direct-MSL Path B is retired" in status.reason


def test_legacy_m2rnn_blocker_points_to_native_tvm_ffi_route() -> None:
    mod = importlib.import_module("cppmega_mlx.nn._tilelang.m2rnn")
    assert mod.__doc__ is not None
    assert "Retired direct-MSL compatibility surface" in mod.__doc__
    assert "m2rnn_path_c.py" in mod.__doc__
    assert not hasattr(mod, "_FWD_KERNEL_SOURCE")
    assert not hasattr(mod, "_BWD_KERNEL_SOURCE")


# ---------------------------------------------------------------------------
# Forward parity
# ---------------------------------------------------------------------------


def test_fwd_metal_matches_reference_at_fp32() -> None:
    inputs = _make_inputs(batch=2, seq=8, heads=2, k_dim=4, v_dim=4, dtype=mx.float32)
    q, k, v, W, xf, h0 = inputs

    y_ref, h_ref = m2rnn_reference(q, k, v, W, xf, h0=h0)
    y_met, h_met, _tcache = m2rnn_fwd_metal(q, k, v, W, xf, h0)
    np.testing.assert_allclose(_np(y_met), _np(y_ref), atol=1e-4, rtol=1e-3)
    np.testing.assert_allclose(_np(h_met), _np(h_ref), atol=1e-4, rtol=1e-3)


def test_fwd_metal_matches_reference_at_fp16() -> None:
    inputs = _make_inputs(batch=2, seq=32, heads=4, k_dim=8, v_dim=4, dtype=mx.float16)
    q, k, v, W, xf, h0 = inputs

    y_ref, h_ref = m2rnn_reference(q, k, v, W, xf, h0=h0)
    y_met, h_met, _tcache = m2rnn_fwd_metal(q, k, v, W, xf, h0)
    y_ref_f = _np(y_ref.astype(mx.float32))
    y_met_f = _np(y_met.astype(mx.float32))
    norm = float(np.sqrt((y_ref_f**2).sum()))
    diff = float(np.max(np.abs(y_ref_f - y_met_f)))
    assert diff <= max(5e-3 * norm + 2e-3, 5e-3), f"max_abs={diff}, norm={norm}"
    np.testing.assert_allclose(
        _np(h_met.astype(mx.float32)),
        _np(h_ref.astype(mx.float32)),
        atol=2e-3,
        rtol=5e-3,
    )


def test_fwd_metal_handles_seq_zero() -> None:
    q, k, v, W, xf, h0 = _make_inputs(
        batch=1, seq=0, heads=1, k_dim=2, v_dim=2, dtype=mx.float32
    )
    y_met, h_met, tcache = m2rnn_fwd_metal(q, k, v, W, xf, h0)
    assert y_met.shape == (1, 0, 1, 2)
    assert h_met.shape == h0.shape
    assert tcache.shape[1] == 0


def test_fwd_metal_supports_broadcast_heads() -> None:
    """Heads-broadcast contract: q/k/W with 1 head, v/xf with 4 heads."""

    mx.random.seed(13)
    batch, seq, k_dim, v_dim = 2, 8, 4, 4
    n_q, n_k, n_v, n_w, n_f = 1, 1, 4, 1, 4
    q = (mx.random.normal((batch, seq, n_q, k_dim)) * 0.1).astype(mx.float32)
    k = (mx.random.normal((batch, seq, n_k, k_dim)) * 0.1).astype(mx.float32)
    v = (mx.random.normal((batch, seq, n_v, v_dim)) * 0.1).astype(mx.float32)
    eye = mx.broadcast_to(mx.eye(v_dim)[None], (n_w, v_dim, v_dim))
    W = (eye + mx.random.normal((n_w, v_dim, v_dim)) * 0.05).astype(mx.float32)
    xf = mx.random.uniform(0.1, 0.9, (batch, seq, n_f)).astype(mx.float32)

    y_ref, h_ref = m2rnn_reference(q, k, v, W, xf)
    y_met, h_met, _ = m2rnn_fwd_metal(q, k, v, W, xf)
    np.testing.assert_allclose(_np(y_met), _np(y_ref), atol=1e-4, rtol=1e-3)
    np.testing.assert_allclose(_np(h_met), _np(h_ref), atol=1e-4, rtol=1e-3)


def test_fwd_metal_with_nonzero_h0() -> None:
    inputs = _make_inputs(batch=1, seq=4, heads=2, k_dim=4, v_dim=4, dtype=mx.float32)
    q, k, v, W, xf, _ = inputs
    h0_real = mx.random.normal((1, 2, 4, 4), dtype=mx.float32) * 0.1
    mx.eval(h0_real)
    y_ref, h_ref = m2rnn_reference(q, k, v, W, xf, h0=h0_real)
    y_met, h_met, _ = m2rnn_fwd_metal(q, k, v, W, xf, h0_real)
    np.testing.assert_allclose(_np(y_met), _np(y_ref), atol=1e-4, rtol=1e-3)
    np.testing.assert_allclose(_np(h_met), _np(h_ref), atol=1e-4, rtol=1e-3)


# ---------------------------------------------------------------------------
# Backward parity
# ---------------------------------------------------------------------------


def test_bwd_metal_matches_autograd_through_reference_fp32() -> None:
    q, k, v, W, xf, h0 = _make_inputs(
        batch=1, seq=6, heads=2, k_dim=4, v_dim=4, dtype=mx.float32
    )

    def ref_loss(q: mx.array, k: mx.array, v: mx.array, W: mx.array, xf: mx.array) -> mx.array:
        y, _ = m2rnn_reference(q, k, v, W, xf)
        return mx.sum(y * y) * 0.5

    def metal_loss(
        q: mx.array, k: mx.array, v: mx.array, W: mx.array, xf: mx.array, h0: mx.array
    ) -> mx.array:
        y = cast(mx.array, m2rnn_apply(q, k, v, W, xf, h0))
        return mx.sum(y * y) * 0.5

    g_ref = mx.grad(ref_loss, argnums=(0, 1, 2, 3, 4))(q, k, v, W, xf)
    g_met = mx.grad(metal_loss, argnums=(0, 1, 2, 3, 4))(q, k, v, W, xf, h0)
    mx.eval(*g_ref, *g_met)

    names = ["q", "k", "v", "W", "xf"]
    for name, gr, gm in zip(names, g_ref, g_met):
        gr_np = _np(gr)
        gm_np = _np(gm)
        norm = float(np.sqrt((gr_np**2).sum() + 1e-12))
        diff = float(np.max(np.abs(gm_np - gr_np)))
        rel = diff / (norm + 1e-12)
        assert rel < 5e-3, (
            f"{name}: rel diff {rel} (norm_ref={norm}, max_abs={diff})"
        )
        np.testing.assert_allclose(gm_np, gr_np, atol=1e-4, rtol=1e-3)


def test_bwd_metal_returns_correct_zero_seq_shapes() -> None:
    q, k, v, W, xf, h0 = _make_inputs(
        batch=1, seq=0, heads=1, k_dim=2, v_dim=2, dtype=mx.float32
    )
    dy = mx.zeros((1, 0, 1, 2), dtype=q.dtype)
    tcache = mx.zeros((1, 0, 1, 2, 2), dtype=q.dtype)
    grads = m2rnn_bwd_metal(dy, q, k, v, W, xf, tcache, h0)
    assert grads[0].shape == q.shape
    assert grads[1].shape == k.shape
    assert grads[2].shape == v.shape
    assert grads[3].shape == W.shape
    assert grads[4].shape == xf.shape
    assert grads[5].shape == h0.shape


def test_apply_with_state_returns_h_last_with_correct_shape() -> None:
    inputs = _make_inputs(batch=2, seq=8, heads=2, k_dim=4, v_dim=4, dtype=mx.float32)
    q, k, v, W, xf, h0 = inputs
    y, h_last = m2rnn_apply_with_state(q, k, v, W, xf, h0)
    mx.eval(y, h_last)
    y_ref, h_ref = m2rnn_reference(q, k, v, W, xf, h0=h0)
    assert y.shape == y_ref.shape
    assert h_last.shape == h_ref.shape
    np.testing.assert_allclose(_np(y), _np(y_ref), atol=1e-4, rtol=1e-3)
    np.testing.assert_allclose(_np(h_last), _np(h_ref), atol=1e-4, rtol=1e-3)


def test_apply_with_state_grad_flows_through_y_only() -> None:
    """Cotangent on h_last is treated as zero — gradient must match m2rnn_apply."""

    q, k, v, W, xf, h0 = _make_inputs(
        batch=1, seq=4, heads=2, k_dim=4, v_dim=4, dtype=mx.float32
    )

    def loss_only_y(
        q: mx.array, k: mx.array, v: mx.array, W: mx.array, xf: mx.array, h0: mx.array
    ) -> mx.array:
        y, _ = m2rnn_apply_with_state(q, k, v, W, xf, h0)
        return mx.sum(y * y) * 0.5

    def loss_apply(
        q: mx.array, k: mx.array, v: mx.array, W: mx.array, xf: mx.array, h0: mx.array
    ) -> mx.array:
        y = cast(mx.array, m2rnn_apply(q, k, v, W, xf, h0))
        return mx.sum(y * y) * 0.5

    g_state = mx.grad(loss_only_y, argnums=(0, 1, 2, 3, 4))(q, k, v, W, xf, h0)
    g_apply = mx.grad(loss_apply, argnums=(0, 1, 2, 3, 4))(q, k, v, W, xf, h0)
    mx.eval(*g_state, *g_apply)

    for gs, ga in zip(g_state, g_apply):
        np.testing.assert_allclose(_np(gs), _np(ga), atol=1e-5, rtol=1e-4)


def test_m2rnn_reference_calls_parent_module_unmodified() -> None:
    """The reference function delegates to the parent module's m2rnn_scan.

    Sanity check that the import surface still resolves and produces the
    same output as the parent module directly.
    """

    from cppmega_mlx.nn.m2rnn import m2rnn_scan

    inputs = _make_inputs(batch=1, seq=4, heads=1, k_dim=2, v_dim=2, dtype=mx.float32)
    q, k, v, W, xf, h0 = inputs
    y_a, h_a = m2rnn_reference(q, k, v, W, xf, h0=h0)
    y_b, h_b = m2rnn_scan(q, k, v, W, xf, h0=h0)
    np.testing.assert_allclose(_np(y_a), _np(y_b), atol=0.0, rtol=0.0)
    np.testing.assert_allclose(_np(h_a), _np(h_b), atol=0.0, rtol=0.0)


# ---------------------------------------------------------------------------
# End-to-end mini-config parity (closer to production shape)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [mx.float32, mx.float16])
def test_metal_kernel_block_parity_at_mini_shape(dtype: mx.Dtype) -> None:
    """Compatibility forward matches the chunked-scan reference at a mini shape."""

    inputs = _make_inputs(batch=2, seq=64, heads=4, k_dim=8, v_dim=4, dtype=dtype)
    q, k, v, W, xf, h0 = inputs
    y_ref, h_ref = m2rnn_reference(q, k, v, W, xf, h0=h0)
    y_met, h_met, _ = m2rnn_fwd_metal(q, k, v, W, xf, h0)
    if dtype == mx.float32:
        atol = 1e-4
        rtol = 1e-3
    else:  # fp16
        atol = 2e-3
        rtol = 5e-3
    np.testing.assert_allclose(
        _np(y_met.astype(mx.float32)),
        _np(y_ref.astype(mx.float32)),
        atol=atol,
        rtol=rtol,
    )
    np.testing.assert_allclose(
        _np(h_met.astype(mx.float32)),
        _np(h_ref.astype(mx.float32)),
        atol=atol,
        rtol=rtol,
    )
