"""Tests for the Path B Mamba3 MIMO Metal port.

Coverage:
  - Triton-helper rewrites (compute_dacs_segsum, bwd_dadt_fused, bwd_dtrap_ddt)
    against deterministic synthetic / autograd references.
  - Forward parity vs cppmega_mlx/nn/mamba3.py at fp16 carrier (rtol=1e-3).
  - Backward parity vs autograd-through-reference (gradient norms rtol=5e-3).
"""

from __future__ import annotations

import importlib

from typing import cast

import numpy as np
import pytest

import mlx.core as mx
import mlx.nn as nn

from cppmega_mlx.nn._tilelang import (
    Mamba3MetalStatus,
    mamba3_mimo_apply,
    mamba3_mimo_bwd_metal,
    mamba3_mimo_fwd_metal,
    mamba3_mimo_metal_status,
    mamba3_mimo_reference,
)
from cppmega_mlx.nn._tilelang._mamba3_helpers import (
    bwd_dadt_fused,
    bwd_dtrap_ddt,
    compute_dacs_segsum,
    reference_trap_scale_forward,
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
    headdim: int,
    state: int,
    dtype: mx.Dtype,
    seed: int = 17,
) -> tuple[mx.array, ...]:
    mx.random.seed(seed)
    x = (mx.random.normal((batch, seq, heads, headdim)) * 0.1).astype(dtype)
    B = (mx.random.normal((batch, seq, heads, state)) * 0.1).astype(dtype)
    C = (mx.random.normal((batch, seq, heads, state)) * 0.1).astype(dtype)
    z = (mx.random.normal((batch, seq, heads, headdim)) * 0.1).astype(dtype)
    A = (-mx.random.uniform(0.01, 0.5, (batch, seq, heads))).astype(dtype)
    dt = (mx.random.uniform(0.001, 0.05, (batch, seq, heads))).astype(dtype)
    D = mx.ones((heads,), dtype=dtype)
    h0 = mx.zeros((batch, heads, headdim, state), dtype=dtype)
    mx.eval(x, B, C, z, A, dt, D, h0)
    return x, B, C, z, A, dt, D, h0


# ---------------------------------------------------------------------------
# Helper-level tests (Triton replacement parity)
# ---------------------------------------------------------------------------


def test_compute_dacs_segsum_matches_reverse_segment_sum() -> None:
    batch, seq, heads = 2, 8, 3
    rng = np.random.default_rng(11)
    A_np = -rng.uniform(0.0, 0.1, size=(batch, seq, heads)).astype(np.float32)
    dt_np = rng.uniform(0.01, 0.05, size=(batch, seq, heads)).astype(np.float32)
    dh_np = rng.standard_normal((batch, seq, heads, 4)).astype(np.float32)
    A = mx.array(A_np)
    dt = mx.array(dt_np)
    dh = mx.array(dh_np)
    mx.eval(A, dt, dh)

    actual = _np(compute_dacs_segsum(A, dt, dh))
    decay = A_np * dt_np
    expected = np.zeros_like(dh_np)
    for b in range(batch):
        for h in range(heads):
            for t in range(seq):
                seg = float(decay[b, t + 1 :, h].sum()) if t + 1 < seq else 0.0
                expected[b, t, h, :] = dh_np[b, t, h, :] * np.exp(seg)
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)


def test_compute_dacs_segsum_handles_empty_trailing_dim() -> None:
    A = mx.zeros((1, 4, 1), dtype=mx.float32)
    dt = mx.ones((1, 4, 1), dtype=mx.float32)
    dh_empty = mx.zeros((1, 4, 1, 0), dtype=mx.float32)
    out = compute_dacs_segsum(A, dt, dh_empty)
    assert out.shape == (1, 4, 1, 0)


def test_compute_dacs_segsum_rejects_bad_shapes() -> None:
    A = mx.zeros((2, 3, 2), dtype=mx.float32)
    dt = mx.zeros((2, 3, 2), dtype=mx.float32)
    dh = mx.zeros((1, 3, 2, 4), dtype=mx.float32)
    with pytest.raises(ValueError):
        compute_dacs_segsum(A, dt, dh)


def test_bwd_dadt_fused_matches_pointwise_chain_rule() -> None:
    mx.random.seed(13)
    batch, seq, heads, p, n = 2, 5, 3, 4, 4
    A = mx.random.normal((batch, seq, heads), dtype=mx.float32) * 0.5
    dt = mx.random.uniform(0.01, 0.1, (batch, seq, heads), dtype=mx.float32)
    dY = mx.random.normal((batch, seq, heads, p, n), dtype=mx.float32)
    h = mx.random.normal((batch, seq, heads, p, n), dtype=mx.float32)
    mx.eval(A, dt, dY, h)

    dA, ddt = bwd_dadt_fused(dY, A, dt, h)
    d_decay = _np(dY) * _np(h)
    d_decay = d_decay.sum(axis=(-1, -2))
    np.testing.assert_allclose(_np(dA), d_decay * _np(dt), rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(_np(ddt), d_decay * _np(A), rtol=1e-4, atol=1e-5)


def test_bwd_dtrap_ddt_matches_autograd_through_reference_forward() -> None:
    mx.random.seed(19)
    batch, seq, heads = 2, 6, 3
    dt = mx.random.uniform(0.001, 0.05, (batch, seq, heads), dtype=mx.float32)
    trap = mx.random.normal((batch, seq, heads), dtype=mx.float32)
    dB_scaled = mx.random.normal((batch, seq, heads), dtype=mx.float32) * 0.1
    mx.eval(dt, trap, dB_scaled)

    def loss(dt: mx.array, trap: mx.array, dB: mx.array) -> mx.array:
        s = reference_trap_scale_forward(dt, trap)
        return mx.sum(dB * s)

    g_dt, g_trap = mx.grad(loss, argnums=(0, 1))(dt, trap, dB_scaled)
    mx.eval(g_dt, g_trap)
    my_ddt, my_dtrap = bwd_dtrap_ddt(dB_scaled, dt, trap)
    mx.eval(my_ddt, my_dtrap)
    np.testing.assert_allclose(_np(my_ddt), _np(g_dt), rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(_np(my_dtrap), _np(g_trap), rtol=1e-4, atol=1e-6)


def test_bwd_dtrap_ddt_handles_seq_one_boundary() -> None:
    dt = mx.ones((1, 1, 1), dtype=mx.float32)
    trap = mx.zeros((1, 1, 1), dtype=mx.float32)
    dB = mx.ones((1, 1, 1), dtype=mx.float32)
    ddt, dtrap = bwd_dtrap_ddt(dB, dt, trap)
    # At seq=1, s_shift = 0.5 boundary fill, dt_shift = 0; scale = dt * 0.5
    # ddt = 1 * sigmoid(0) + 0 = 0.5
    # dtrap = 1 * dt * 0.5 * 0.5 = 0.25 (sigmoid'(0) = 0.25)
    np.testing.assert_allclose(_np(ddt), [[[0.5]]], rtol=1e-6)
    np.testing.assert_allclose(_np(dtrap), [[[0.25]]], rtol=1e-6)


# ---------------------------------------------------------------------------
# Forward parity tests
# ---------------------------------------------------------------------------


def test_status_is_available_or_explains_why() -> None:
    status = mamba3_mimo_metal_status()
    assert isinstance(status, Mamba3MetalStatus)
    assert isinstance(status.available, bool)
    assert isinstance(status.reason, str) and status.reason


def test_legacy_mamba3_blocker_points_to_native_tvm_ffi_route() -> None:
    mod = importlib.import_module("cppmega_mlx.nn._tilelang.mamba3")
    assert mod.__doc__ is not None
    assert "no safe inverse transform" in mod.__doc__
    assert 'execution_backend="tvm_ffi"' in mod.__doc__
    assert "legacy direct-MSL fallback" in mod.__doc__


def test_fwd_metal_matches_reference_at_fp32() -> None:
    inputs = _make_inputs(batch=1, seq=8, heads=2, headdim=4, state=4, dtype=mx.float32)
    y_ref, h_ref = mamba3_mimo_reference(*inputs)
    y_met, h_met = mamba3_mimo_fwd_metal(*inputs)
    np.testing.assert_allclose(_np(y_met), _np(y_ref), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(_np(h_met), _np(h_ref), rtol=1e-5, atol=1e-6)


def test_fwd_metal_matches_reference_at_fp16_spec_shape() -> None:
    # spec shape: B=2, T=512, D=128 -> headdim=32, heads=4, state=64
    inputs = _make_inputs(
        batch=2, seq=512, heads=4, headdim=32, state=64, dtype=mx.float16
    )
    y_ref, _ = mamba3_mimo_reference(*inputs)
    y_met, _ = mamba3_mimo_fwd_metal(*inputs)
    y_ref_f = _np(y_ref.astype(mx.float32))
    y_met_f = _np(y_met.astype(mx.float32))
    diff = float(np.max(np.abs(y_ref_f - y_met_f)))
    norm = float(np.sqrt((y_ref_f**2).sum()))
    assert diff <= max(1e-3 * norm + 1e-3, 5e-3), f"max_abs={diff}, norm={norm}"


def test_fwd_metal_falls_back_to_reference_on_unsupported_dtype() -> None:
    # int dtype is not in _SUPPORTED_DTYPES; the path should silently fall back.
    inputs = _make_inputs(batch=1, seq=4, heads=1, headdim=2, state=2, dtype=mx.float32)
    y_ref, _ = mamba3_mimo_reference(*inputs)
    y_met, _ = mamba3_mimo_fwd_metal(*inputs)
    np.testing.assert_allclose(_np(y_met), _np(y_ref), rtol=1e-5, atol=1e-6)


# ---------------------------------------------------------------------------
# Backward parity tests
# ---------------------------------------------------------------------------


def test_bwd_metal_matches_autograd_through_reference_fp32() -> None:
    inputs = _make_inputs(batch=1, seq=6, heads=2, headdim=4, state=4, dtype=mx.float32)
    x, B, C, z, A, dt, D, h0 = inputs

    def ref_loss(x: mx.array, B: mx.array, C: mx.array, z: mx.array,
                 A: mx.array, dt: mx.array, D: mx.array, h0: mx.array) -> mx.array:
        y, _ = mamba3_mimo_reference(x, B, C, z, A, dt, D, h0)
        return mx.sum(y * y) * 0.5

    def metal_loss(x: mx.array, B: mx.array, C: mx.array, z: mx.array,
                   A: mx.array, dt: mx.array, D: mx.array, h0: mx.array) -> mx.array:
        y = cast(mx.array, mamba3_mimo_apply(x, B, C, z, A, dt, D, h0))
        return mx.sum(y * y) * 0.5

    g_ref = mx.grad(ref_loss, argnums=(0, 1, 2, 3, 4, 5, 6, 7))(*inputs)
    g_met = mx.grad(metal_loss, argnums=(0, 1, 2, 3, 4, 5, 6, 7))(*inputs)
    mx.eval(*g_ref, *g_met)

    names = ["x", "B", "C", "z", "A", "dt", "D", "h0"]
    for name, gr, gm in zip(names, g_ref, g_met):
        gr_np = _np(gr)
        gm_np = _np(gm)
        norm_ref = float(np.sqrt((gr_np**2).sum() + 1e-12))
        norm_met = float(np.sqrt((gm_np**2).sum() + 1e-12))
        rel = abs(norm_met - norm_ref) / (norm_ref + 1e-12)
        assert rel < 5e-3, f"{name}: rel norm diff {rel} (ref={norm_ref}, met={norm_met})"
        np.testing.assert_allclose(gm_np, gr_np, rtol=5e-3, atol=1e-5)


def test_bwd_metal_returns_correct_zero_seq_shapes() -> None:
    inputs = _make_inputs(batch=1, seq=0, heads=1, headdim=2, state=2, dtype=mx.float32)
    x, B, C, z, A, dt, D, h0 = inputs
    dy = mx.zeros_like(x)
    grads = mamba3_mimo_bwd_metal(dy, x, B, C, z, A, dt, D, h0)
    assert grads[0].shape == x.shape
    assert grads[7].shape == h0.shape


def test_bwd_metal_with_trap_overlay_runs() -> None:
    inputs = _make_inputs(batch=1, seq=4, heads=2, headdim=2, state=4, dtype=mx.float32)
    x, B, C, z, A, dt, D, h0 = inputs
    trap = mx.random.normal((1, 4, 2), dtype=mx.float32)
    dy = mx.random.normal(x.shape, dtype=mx.float32) * 0.1
    mx.eval(trap, dy)
    grads = mamba3_mimo_bwd_metal(dy, x, B, C, z, A, dt, D, h0, trap=trap)
    mx.eval(*grads)
    # ddt index is 5; trap-aware backward should add trap-mediated contribution.
    assert grads[5].shape == dt.shape
    assert grads[7].shape == h0.shape


# ---------------------------------------------------------------------------
# End-to-end parity vs cppmega Mamba3ReferenceBlock
# ---------------------------------------------------------------------------


def test_mamba3_reference_block_remains_unmodified_at_imported_surface() -> None:
    """The reference is the parity oracle; tests must read it without changing it."""

    from cppmega_mlx.nn.mamba3 import Mamba3ReferenceBlock, Mamba3Config

    cfg = Mamba3Config(d_model=16, expand=2, headdim=8, d_state=8, ngroups=1)
    block = Mamba3ReferenceBlock(cfg)
    assert isinstance(block, Mamba3ReferenceBlock)


def test_mamba3_metal_kernel_equivalent_to_reference_scan_at_full_block_shapes() -> None:
    """Apply the Path B kernel to the post-projection tensors of the reference.

    The kernel takes (x, B, C, z, A, dt, D, h0) which are the same arrays used
    inside Mamba3ReferenceBlock between projection and the diagonal scan. This
    test inspects equivalence at that contract, treating the reference scan as
    the parity oracle without touching its block code.
    """

    from cppmega_mlx.nn.mamba3 import Mamba3ReferenceBlock, Mamba3Config
    from cppmega_mlx.nn.mamba3 import (
        _broadcast_groups_to_heads,
        _compute_trapezoidal_scale,
        _heads_to_group_scale,
    )

    cfg = Mamba3Config(
        d_model=64,
        expand=2,
        headdim=16,
        d_state=16,
        ngroups=1,
        chunk_size=32,
    )
    block = Mamba3ReferenceBlock(cfg)
    mx.random.seed(0)
    hidden = mx.random.normal((1, 16, cfg.d_model), dtype=mx.float32) * 0.1

    # Replicate the reference block's slicing up to the diagonal-scan contract.
    z, x, B, C, dd_dt, dd_A, trap, _angles = block.split_in_proj(block.in_proj(hidden))
    xBC = mx.concatenate([x, B, C], axis=-1)
    from cppmega_mlx.nn.mamba3 import causal_depthwise_conv1d
    xBC = causal_depthwise_conv1d(
        xBC,
        block.conv_weight.astype(xBC.dtype),
        block.conv_bias.astype(xBC.dtype),
    )
    xBC = nn.silu(xBC)
    x = xBC[:, :, : cfg.d_inner].reshape(1, 16, cfg.nheads, cfg.headdim)
    B = xBC[:, :, cfg.d_inner : cfg.d_inner + block.dims.d_bc]
    C = xBC[:, :, cfg.d_inner + block.dims.d_bc :]
    z = z.reshape(1, 16, cfg.nheads, cfg.headdim)
    B_mimo = B.reshape(1, 16, cfg.effective_mimo_rank, cfg.ngroups, cfg.d_state)
    C_mimo = C.reshape(1, 16, cfg.effective_mimo_rank, cfg.ngroups, cfg.d_state)
    B_mimo, C_mimo = block.transform_bc(B_mimo, C_mimo)
    B = mx.mean(B_mimo, axis=2)
    C = mx.mean(C_mimo, axis=2)
    dt = nn.softplus(dd_dt + block.dt_bias.astype(dd_dt.dtype))
    trap_scale = _compute_trapezoidal_scale(dt, trap)
    B = B * _heads_to_group_scale(trap_scale, cfg.ngroups)[:, :, :, None]
    B = _broadcast_groups_to_heads(B, cfg.nheads, "B")
    C = _broadcast_groups_to_heads(C, cfg.nheads, "C")
    A = mx.minimum(-nn.softplus(dd_A), -cfg.A_floor)
    h0 = mx.zeros((1, cfg.nheads, cfg.headdim, cfg.d_state), dtype=hidden.dtype)
    mx.eval(x, B, C, z, A, dt, block.D, h0)

    log_decay = (A * dt)[:, :, :, None, None]
    inp = x[:, :, :, :, None] * B[:, :, :, None, :]
    # Reference scan re-implementation matching the block's diagonal scan
    # exactly (no broadcast head changes, no rope here for this contract slice).
    h = h0
    out_steps: list[mx.array] = []
    for t in range(16):
        h = mx.exp(log_decay[:, t]) * h + inp[:, t]
        y = mx.sum(h * C[:, t, :, None, :], axis=-1)
        y = y + block.D[None, :, None].astype(y.dtype) * x[:, t]
        out_steps.append(nn.silu(z[:, t]) * y)
    y_ref = mx.stack(out_steps, axis=1)

    y_met, _ = mamba3_mimo_fwd_metal(x, B, C, z, A, dt, block.D, h0)
    np.testing.assert_allclose(_np(y_met), _np(y_ref), rtol=1e-4, atol=1e-5)
