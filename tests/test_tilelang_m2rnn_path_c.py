"""Coverage for the Path C TileLang DSL m2rnn forward/backward surface."""

# pyright: reportMissingImports=false

from __future__ import annotations

import importlib

import numpy as np
import pytest

import mlx.core as mx
import mlx.nn as nn

import cppmega_mlx.nn._tilelang.m2rnn_path_c as m2rnn_path_c
from cppmega_mlx.nn._tilelang import _msl_transform
from cppmega_mlx.nn._tilelang.m2rnn import (
    m2rnn_apply,
    m2rnn_bwd_metal,
    m2rnn_fwd_metal,
    m2rnn_metal_status,
    m2rnn_reference,
)
from cppmega_mlx.nn._tilelang.m2rnn_path_c import (
    M2RNNPathCStatus,
    m2rnn_apply_mapped_packed_post_with_state_path_c,
    m2rnn_apply_mapped_packed_with_state_path_c,
    m2rnn_apply_post_residual_gate_path_c,
    m2rnn_apply_packed_with_state_path_c,
    m2rnn_apply_path_c,
    m2rnn_apply_with_state_path_c,
    m2rnn_apply_with_state_path_c_or_fallback,
    m2rnn_bwd_path_c,
    m2rnn_fwd_with_state_path_c,
    m2rnn_mapped_packed_path_c_status,
    m2rnn_mapped_packed_post_path_c_status,
    m2rnn_packed_bwd_path_c,
    m2rnn_packed_path_c_status,
    m2rnn_post_residual_gate_path_c_status,
    m2rnn_path_c_status,
)


def _np(x: mx.array) -> np.ndarray:
    if x.dtype == mx.bfloat16:
        x = x.astype(mx.float32)
    mx.eval(x)
    return np.array(x, copy=True)


def _make_m2rnn_inputs(
    *,
    batch: int = 1,
    seq: int = 4,
    heads: int = 2,
    k_dim: int = 4,
    v_dim: int = 4,
    dtype: mx.Dtype = mx.float32,
    seed: int = 7,
) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array, mx.array]:
    mx.random.seed(seed)
    q = (mx.random.normal((batch, seq, heads, k_dim)) * 0.1).astype(dtype)
    k = (mx.random.normal((batch, seq, heads, k_dim)) * 0.1).astype(dtype)
    v = (mx.random.normal((batch, seq, heads, v_dim)) * 0.1).astype(dtype)
    W = (mx.random.normal((heads, v_dim, v_dim)) * 0.1).astype(dtype)
    xf = (mx.random.uniform(0.001, 0.05, (batch, seq, heads))).astype(dtype)
    h0 = mx.zeros((batch, heads, k_dim, v_dim), dtype=dtype)
    mx.eval(q, k, v, W, xf, h0)
    return q, k, v, W, xf, h0


def test_m2rnn_path_c_module_imports() -> None:
    module = importlib.import_module("cppmega_mlx.nn._tilelang.m2rnn_path_c")
    assert hasattr(module, "m2rnn_apply_path_c")
    assert hasattr(module, "m2rnn_bwd_path_c")


def test_m2rnn_path_c_status_surface_includes_backward() -> None:
    status = m2rnn_path_c_status()
    assert isinstance(status, M2RNNPathCStatus)
    assert status.reason


def test_m2rnn_path_c_launch_geometry_comes_from_tilelang_lowering() -> None:
    _require_m2rnn_path_c()

    _fwd_kernel, fwd_lowering = m2rnn_path_c._fwd_kernel_for(
        1, 4, 2, 4, 4, "float32"
    )
    assert fwd_lowering.grid == (1, 1, 1)
    assert fwd_lowering.threadgroup == (8, 1, 1)
    assert _msl_transform.metal_grid_for_lowering(fwd_lowering) == (8, 1, 1)

    _bwd_kernel, bwd_lowering = m2rnn_path_c._bwd_kernel_for(
        1, 4, 2, 4, 4, "float32"
    )
    assert bwd_lowering.grid == (1, 1, 1)
    assert bwd_lowering.threadgroup == (2, 1, 1)
    assert _msl_transform.metal_grid_for_lowering(bwd_lowering) == (2, 1, 1)


def test_m2rnn_packed_path_c_large_k_uses_k_parallel_lowering() -> None:
    _require_m2rnn_path_c()

    _fwd_kernel, fwd_lowering = m2rnn_path_c._packed_fwd_kernel_for(
        1, 4, 2, 16, 4, "float32"
    )
    assert fwd_lowering.grid == (2, 1, 1)
    assert fwd_lowering.threadgroup == (16, 1, 1)
    assert _msl_transform.metal_grid_for_lowering(fwd_lowering) == (32, 1, 1)

    _bwd_kernel, bwd_lowering = m2rnn_path_c._packed_bwd_kernel_for(
        1, 4, 2, 16, 4, "float32"
    )
    assert bwd_lowering.grid == (2, 1, 1)
    assert bwd_lowering.threadgroup == (16, 1, 1)
    assert _msl_transform.metal_grid_for_lowering(bwd_lowering) == (32, 1, 1)


def test_m2rnn_packed_path_c_small_k_uses_k_parallel_lowering() -> None:
    _require_m2rnn_path_c()

    _fwd_kernel, fwd_lowering = m2rnn_path_c._packed_fwd_kernel_for(
        1, 4, 2, 4, 4, "float32"
    )
    assert fwd_lowering.grid == (2, 1, 1)
    assert fwd_lowering.threadgroup == (4, 1, 1)
    assert _msl_transform.metal_grid_for_lowering(fwd_lowering) == (8, 1, 1)

    _bwd_kernel, bwd_lowering = m2rnn_path_c._packed_bwd_kernel_for(
        1, 4, 2, 4, 4, "float32"
    )
    assert bwd_lowering.grid == (2, 1, 1)
    assert bwd_lowering.threadgroup == (4, 1, 1)
    assert _msl_transform.metal_grid_for_lowering(bwd_lowering) == (8, 1, 1)


def test_m2rnn_apply_path_c_fails_closed_instead_of_path_b(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(m2rnn_path_c, "_path_c_inputs_eligible", lambda *_args: False)
    monkeypatch.setattr(
        m2rnn_path_c,
        "m2rnn_path_c_status",
        lambda: M2RNNPathCStatus(False, "forced unavailable"),
    )
    inputs = _make_m2rnn_inputs(dtype=mx.float32)
    with pytest.raises(RuntimeError, match="forced unavailable"):
        m2rnn_apply_path_c(*inputs)


def test_m2rnn_path_c_or_fallback_fails_closed_instead_of_path_b(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(m2rnn_path_c, "_path_c_inputs_eligible", lambda *_args: False)
    monkeypatch.setattr(
        m2rnn_path_c,
        "m2rnn_path_c_status",
        lambda: M2RNNPathCStatus(False, "forced unavailable"),
    )

    def fail_path_b(*_args: object, **_kwargs: object) -> tuple[mx.array, mx.array]:
        raise AssertionError("M2RNN Path C helper silently fell back to Path B")

    monkeypatch.setattr(m2rnn_path_c, "m2rnn_apply_with_state", fail_path_b, raising=False)
    inputs = _make_m2rnn_inputs(dtype=mx.float32)
    with pytest.raises(RuntimeError, match="forced unavailable"):
        m2rnn_apply_with_state_path_c_or_fallback(*inputs)


def test_m2rnn_bwd_path_c_fails_closed_instead_of_path_b(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(m2rnn_path_c, "_m2rnn_bwd_path_c_kernel", lambda *_args: None)
    monkeypatch.setattr(
        m2rnn_path_c,
        "m2rnn_path_c_status",
        lambda: M2RNNPathCStatus(False, "forced bwd unavailable"),
    )
    inputs = _make_m2rnn_inputs(dtype=mx.float32)
    q, k, v, W, xf, h0 = inputs
    dy = mx.zeros((1, 4, 2, 4), dtype=mx.float32)
    tanh_cache = mx.zeros((1, 4, 2, 4, 4), dtype=mx.float32)
    with pytest.raises(RuntimeError, match="forced bwd unavailable"):
        m2rnn_bwd_path_c(dy, q, k, v, W, xf, tanh_cache, h0)


def _try_import_m2rnn_path_c():  # type: ignore[no-untyped-def]
    try:
        return importlib.import_module("cppmega_mlx.nn._tilelang.m2rnn_path_c")
    except Exception:
        return None


def _require_m2rnn_path_c() -> None:
    status = m2rnn_path_c_status()
    if not status.available:
        pytest.skip(f"m2rnn Path C unavailable on this host: {status.reason}")


def test_m2rnn_path_c_forward_matches_path_b_when_available() -> None:
    """When ``m2rnn_path_c.m2rnn_apply_path_c`` exists, it must match Path B
    within fp32 tolerance on a small canonical shape."""

    module = _try_import_m2rnn_path_c()
    if module is None:
        pytest.xfail("m2rnn_path_c module not implemented yet")

    apply_path_c = getattr(module, "m2rnn_apply_path_c", None)
    if apply_path_c is None:
        pytest.xfail(
            "m2rnn_path_c module exists but does not expose m2rnn_apply_path_c yet"
        )

    _require_m2rnn_path_c()

    inputs = _make_m2rnn_inputs(dtype=mx.float32)
    y_pc = apply_path_c(*inputs, force_path_c=True)
    y_pb = m2rnn_apply(*inputs)
    np.testing.assert_allclose(_np(y_pc), _np(y_pb), rtol=1e-3, atol=1e-4)


def test_m2rnn_path_c_with_state_matches_raw_forward() -> None:
    _require_m2rnn_path_c()
    inputs = _make_m2rnn_inputs(dtype=mx.float32)
    y_state, h_state = m2rnn_apply_with_state_path_c(*inputs)
    raw = m2rnn_fwd_with_state_path_c(*inputs)
    assert raw is not None
    y_raw, h_raw = raw
    mx.eval(y_state, h_state, y_raw, h_raw)
    np.testing.assert_allclose(_np(y_state), _np(y_raw), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(_np(h_state), _np(h_raw), rtol=1e-6, atol=1e-6)


def test_m2rnn_packed_path_c_matches_unpacked_forward_and_grad() -> None:
    _require_m2rnn_path_c()
    inputs = _make_m2rnn_inputs(dtype=mx.float32)
    q, k, v, W, xf, h0 = inputs
    conv_input = mx.concatenate(
        [
            q.reshape(q.shape[0], q.shape[1], -1),
            k.reshape(k.shape[0], k.shape[1], -1),
            v.reshape(v.shape[0], v.shape[1], -1),
        ],
        axis=-1,
    )
    y_packed, h_packed = m2rnn_apply_packed_with_state_path_c(conv_input, W, xf, h0)
    y_raw, h_raw = m2rnn_apply_with_state_path_c(q, k, v, W, xf, h0)
    mx.eval(y_packed, h_packed, y_raw, h_raw)
    np.testing.assert_allclose(_np(y_packed), _np(y_raw), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(_np(h_packed), _np(h_raw), rtol=1e-6, atol=1e-6)

    def packed_loss(conv_input_, W_, xf_, h0_):  # type: ignore[no-untyped-def]
        y, _h = m2rnn_apply_packed_with_state_path_c(conv_input_, W_, xf_, h0_)
        return mx.sum(y * y) * 0.5

    def unpacked_loss(q_, k_, v_, W_, xf_, h0_):  # type: ignore[no-untyped-def]
        y = m2rnn_apply_path_c(q_, k_, v_, W_, xf_, h0_, force_path_c=True)
        return mx.sum(y * y) * 0.5

    dconv, dW_p, dxf_p, dh0_p = mx.grad(packed_loss, argnums=(0, 1, 2, 3))(
        conv_input,
        W,
        xf,
        h0,
    )
    dq, dk, dv, dW_u, dxf_u, dh0_u = mx.grad(
        unpacked_loss,
        argnums=tuple(range(6)),
    )(*inputs)
    dconv_expected = mx.concatenate(
        [
            dq.reshape(q.shape[0], q.shape[1], -1),
            dk.reshape(k.shape[0], k.shape[1], -1),
            dv.reshape(v.shape[0], v.shape[1], -1),
        ],
        axis=-1,
    )
    mx.eval(dconv, dW_p, dxf_p, dh0_p, dconv_expected, dW_u, dxf_u, dh0_u)
    np.testing.assert_allclose(_np(dconv), _np(dconv_expected), rtol=2e-3, atol=2e-4)
    np.testing.assert_allclose(_np(dW_p), _np(dW_u), rtol=2e-3, atol=2e-4)
    np.testing.assert_allclose(_np(dxf_p), _np(dxf_u), rtol=2e-3, atol=2e-4)
    np.testing.assert_allclose(_np(dh0_p), _np(dh0_u), rtol=2e-3, atol=2e-4)


def test_m2rnn_mapped_packed_path_c_matches_grouped_reference_and_grad() -> None:
    _require_m2rnn_path_c()
    batch, seq, k_dim, v_dim = 1, 4, 4, 3
    q_heads, k_heads, v_heads = 1, 1, 2
    total_heads, w_heads, f_heads = 4, 1, 2
    mx.random.seed(23)
    q = (mx.random.normal((batch, seq, q_heads, k_dim)) * 0.1).astype(mx.float32)
    k = (mx.random.normal((batch, seq, k_heads, k_dim)) * 0.1).astype(mx.float32)
    v = (mx.random.normal((batch, seq, v_heads, v_dim)) * 0.1).astype(mx.float32)
    W = (mx.random.normal((w_heads, v_dim, v_dim)) * 0.1).astype(mx.float32)
    xf = (mx.random.uniform(0.001, 0.05, (batch, seq, f_heads))).astype(mx.float32)
    h0 = mx.zeros((batch, total_heads, k_dim, v_dim), dtype=mx.float32)
    conv_input = mx.concatenate(
        [
            q.reshape(batch, seq, -1),
            k.reshape(batch, seq, -1),
            v.reshape(batch, seq, -1),
        ],
        axis=-1,
    )

    status = m2rnn_mapped_packed_path_c_status(
        conv_input,
        W,
        xf,
        h0,
        q_heads=q_heads,
        k_heads=k_heads,
        v_heads=v_heads,
    )
    if not status.available:
        pytest.skip(f"mapped packed m2rnn Path C unavailable: {status.reason}")

    y_mapped, h_mapped = m2rnn_apply_mapped_packed_with_state_path_c(
        conv_input,
        W,
        xf,
        h0,
        q_heads=q_heads,
        k_heads=k_heads,
        v_heads=v_heads,
    )

    def expand_heads(x: mx.array, axis: int) -> mx.array:
        heads = x.shape[axis]
        if heads == total_heads:
            return x
        if heads == 1:
            target_shape = list(x.shape)
            target_shape[axis] = total_heads
            return mx.broadcast_to(x, tuple(target_shape))
        return mx.repeat(x, repeats=total_heads // heads, axis=axis)

    y_ref, h_ref = m2rnn_reference(
        expand_heads(q, -2),
        expand_heads(k, -2),
        expand_heads(v, -2),
        expand_heads(W, 0),
        expand_heads(xf, -1),
        h0=h0,
    )
    mx.eval(y_mapped, h_mapped, y_ref, h_ref)
    np.testing.assert_allclose(_np(y_mapped), _np(y_ref), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(_np(h_mapped), _np(h_ref), rtol=1e-6, atol=1e-6)

    q_stop = q_heads * k_dim
    k_stop = q_stop + k_heads * k_dim

    def mapped_loss(conv_input_, W_, xf_, h0_):  # type: ignore[no-untyped-def]
        y, _h = m2rnn_apply_mapped_packed_with_state_path_c(
            conv_input_,
            W_,
            xf_,
            h0_,
            q_heads=q_heads,
            k_heads=k_heads,
            v_heads=v_heads,
        )
        return mx.sum(y * y) * 0.5

    def reference_loss(conv_input_, W_, xf_, h0_):  # type: ignore[no-untyped-def]
        q_ = conv_input_[:, :, :q_stop].reshape(batch, seq, q_heads, k_dim)
        k_ = conv_input_[:, :, q_stop:k_stop].reshape(batch, seq, k_heads, k_dim)
        v_ = conv_input_[:, :, k_stop:].reshape(batch, seq, v_heads, v_dim)
        y, _h = m2rnn_reference(
            expand_heads(q_, -2),
            expand_heads(k_, -2),
            expand_heads(v_, -2),
            expand_heads(W_, 0),
            expand_heads(xf_, -1),
            h0=h0_,
        )
        return mx.sum(y * y) * 0.5

    mapped_grads = mx.grad(mapped_loss, argnums=(0, 1, 2, 3))(
        conv_input,
        W,
        xf,
        h0,
    )
    ref_grads = mx.grad(reference_loss, argnums=(0, 1, 2, 3))(
        conv_input,
        W,
        xf,
        h0,
    )
    mx.eval(*mapped_grads, *ref_grads)
    for got, expected in zip(mapped_grads, ref_grads, strict=True):
        np.testing.assert_allclose(_np(got), _np(expected), rtol=2e-3, atol=2e-4)


def test_m2rnn_mapped_packed_inline_post_path_c_matches_reference_and_grad() -> None:
    _require_m2rnn_path_c()
    batch, seq, k_dim, v_dim = 1, 4, 4, 3
    q_heads, k_heads, v_heads, g_heads = 1, 1, 2, 2
    total_heads, w_heads, f_heads = 4, 1, 2
    conv_dim = q_heads * k_dim + k_heads * k_dim + v_heads * v_dim
    projected_dim = conv_dim + f_heads + g_heads * v_dim
    mx.random.seed(43)
    q = (mx.random.normal((batch, seq, q_heads, k_dim)) * 0.1).astype(mx.float32)
    k = (mx.random.normal((batch, seq, k_heads, k_dim)) * 0.1).astype(mx.float32)
    v = (mx.random.normal((batch, seq, v_heads, v_dim)) * 0.1).astype(mx.float32)
    W = (mx.random.normal((w_heads, v_dim, v_dim)) * 0.1).astype(mx.float32)
    xf = (mx.random.uniform(0.001, 0.05, (batch, seq, f_heads))).astype(mx.float32)
    h0 = mx.zeros((batch, total_heads, k_dim, v_dim), dtype=mx.float32)
    D = (mx.random.normal((total_heads, v_dim)) * 0.1).astype(mx.float32)
    projected = (mx.random.normal((batch, seq, projected_dim)) * 0.1).astype(mx.float32)
    conv_input = mx.concatenate(
        [
            q.reshape(batch, seq, -1),
            k.reshape(batch, seq, -1),
            v.reshape(batch, seq, -1),
        ],
        axis=-1,
    )

    status = m2rnn_mapped_packed_post_path_c_status(
        conv_input,
        W,
        xf,
        h0,
        D,
        projected,
        q_heads=q_heads,
        k_heads=k_heads,
        v_heads=v_heads,
        g_heads=g_heads,
    )
    if not status.available:
        pytest.skip(f"mapped packed inline post m2rnn Path C unavailable: {status.reason}")

    def expand_heads(x: mx.array, axis: int) -> mx.array:
        heads = x.shape[axis]
        if heads == total_heads:
            return x
        if heads == 1:
            target_shape = list(x.shape)
            target_shape[axis] = total_heads
            return mx.broadcast_to(x, tuple(target_shape))
        return mx.repeat(x, repeats=total_heads // heads, axis=axis)

    v_offset = q_heads * k_dim + k_heads * k_dim
    g_offset = projected_dim - g_heads * v_dim

    def reference(conv_input_, W_, xf_, h0_, D_, projected_):  # type: ignore[no-untyped-def]
        q_ = conv_input_[:, :, : q_heads * k_dim].reshape(batch, seq, q_heads, k_dim)
        k_stop = q_heads * k_dim + k_heads * k_dim
        k_ = conv_input_[:, :, q_heads * k_dim : k_stop].reshape(
            batch,
            seq,
            k_heads,
            k_dim,
        )
        v_ = conv_input_[:, :, k_stop:].reshape(batch, seq, v_heads, v_dim)
        y, h = m2rnn_reference(
            expand_heads(q_, -2),
            expand_heads(k_, -2),
            expand_heads(v_, -2),
            expand_heads(W_, 0),
            expand_heads(xf_, -1),
            h0=h0_,
        )
        v_source = conv_input_[:, :, v_offset:].reshape(batch, seq, v_heads, v_dim)
        v_broadcast = mx.repeat(v_source, repeats=total_heads // v_heads, axis=-2)
        skipped = y + v_broadcast * D_.astype(y.dtype)
        g_flat = projected_[:, :, g_offset:]
        g_repeat = mx.repeat(g_flat, repeats=total_heads // g_heads, axis=-1)
        return skipped.reshape(batch, seq, total_heads * v_dim) * nn.silu(g_repeat), h

    out_path_c, h_path_c = m2rnn_apply_mapped_packed_post_with_state_path_c(
        conv_input,
        W,
        xf,
        h0,
        D,
        projected,
        q_heads=q_heads,
        k_heads=k_heads,
        v_heads=v_heads,
        g_heads=g_heads,
    )
    out_ref, h_ref = reference(conv_input, W, xf, h0, D, projected)
    mx.eval(out_path_c, h_path_c, out_ref, h_ref)
    np.testing.assert_allclose(_np(out_path_c), _np(out_ref), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(_np(h_path_c), _np(h_ref), rtol=1e-6, atol=1e-6)

    def path_c_loss(conv_input_, W_, xf_, h0_, D_, projected_):  # type: ignore[no-untyped-def]
        out, _h = m2rnn_apply_mapped_packed_post_with_state_path_c(
            conv_input_,
            W_,
            xf_,
            h0_,
            D_,
            projected_,
            q_heads=q_heads,
            k_heads=k_heads,
            v_heads=v_heads,
            g_heads=g_heads,
        )
        return mx.sum(out * out) * 0.5

    def reference_loss(conv_input_, W_, xf_, h0_, D_, projected_):  # type: ignore[no-untyped-def]
        out, _h = reference(conv_input_, W_, xf_, h0_, D_, projected_)
        return mx.sum(out * out) * 0.5

    path_c_grads = mx.grad(path_c_loss, argnums=tuple(range(6)))(
        conv_input,
        W,
        xf,
        h0,
        D,
        projected,
    )
    ref_grads = mx.grad(reference_loss, argnums=tuple(range(6)))(
        conv_input,
        W,
        xf,
        h0,
        D,
        projected,
    )
    mx.eval(*path_c_grads, *ref_grads)
    for got, expected in zip(path_c_grads, ref_grads, strict=True):
        np.testing.assert_allclose(_np(got), _np(expected), rtol=3e-3, atol=3e-4)


def test_m2rnn_mapped_packed_inline_post_path_c_bfloat16_grad_runs() -> None:
    _require_m2rnn_path_c()
    batch, seq, k_dim, v_dim = 1, 2, 8, 3
    q_heads, k_heads, v_heads, g_heads = 1, 1, 2, 2
    total_heads, w_heads, f_heads = 4, 1, 2
    conv_dim = q_heads * k_dim + k_heads * k_dim + v_heads * v_dim
    projected_dim = conv_dim + f_heads + g_heads * v_dim
    mx.random.seed(44)
    q = (mx.random.normal((batch, seq, q_heads, k_dim)) * 0.1).astype(mx.bfloat16)
    k = (mx.random.normal((batch, seq, k_heads, k_dim)) * 0.1).astype(mx.bfloat16)
    v = (mx.random.normal((batch, seq, v_heads, v_dim)) * 0.1).astype(mx.bfloat16)
    W = (mx.random.normal((w_heads, v_dim, v_dim)) * 0.1).astype(mx.bfloat16)
    xf = (mx.random.uniform(0.001, 0.05, (batch, seq, f_heads))).astype(mx.bfloat16)
    h0 = mx.zeros((batch, total_heads, k_dim, v_dim), dtype=mx.bfloat16)
    D = (mx.random.normal((total_heads, v_dim)) * 0.1).astype(mx.bfloat16)
    projected = (mx.random.normal((batch, seq, projected_dim)) * 0.1).astype(
        mx.bfloat16
    )
    conv_input = mx.concatenate(
        [
            q.reshape(batch, seq, -1),
            k.reshape(batch, seq, -1),
            v.reshape(batch, seq, -1),
        ],
        axis=-1,
    )

    status = m2rnn_mapped_packed_post_path_c_status(
        conv_input,
        W,
        xf,
        h0,
        D,
        projected,
        q_heads=q_heads,
        k_heads=k_heads,
        v_heads=v_heads,
        g_heads=g_heads,
    )
    if not status.available:
        pytest.skip(f"mapped packed inline post m2rnn Path C unavailable: {status.reason}")

    def path_c_loss(conv_input_, W_, xf_, h0_, D_, projected_):  # type: ignore[no-untyped-def]
        out, _h = m2rnn_apply_mapped_packed_post_with_state_path_c(
            conv_input_,
            W_,
            xf_,
            h0_,
            D_,
            projected_,
            q_heads=q_heads,
            k_heads=k_heads,
            v_heads=v_heads,
            g_heads=g_heads,
        )
        out_f32 = out.astype(mx.float32)
        return mx.sum(out_f32 * out_f32) * 0.5

    grads = mx.grad(path_c_loss, argnums=tuple(range(6)))(
        conv_input,
        W,
        xf,
        h0,
        D,
        projected,
    )
    mx.eval(*grads)
    for grad, primal in zip(grads, (conv_input, W, xf, h0, D, projected), strict=True):
        assert grad.dtype == primal.dtype
        assert bool(mx.all(mx.isfinite(grad)).item())


def test_m2rnn_post_residual_gate_path_c_matches_grouped_mlx_and_grad() -> None:
    _require_m2rnn_path_c()
    batch, seq, total_heads, k_dim, v_dim = 1, 4, 4, 4, 3
    q_heads, k_heads, v_heads, g_heads = 1, 1, 2, 2
    conv_dim = q_heads * k_dim + k_heads * k_dim + v_heads * v_dim
    projected_dim = conv_dim + total_heads + g_heads * v_dim
    mx.random.seed(41)
    y = (mx.random.normal((batch, seq, total_heads, v_dim)) * 0.1).astype(mx.float32)
    conv_input = (mx.random.normal((batch, seq, conv_dim)) * 0.1).astype(mx.float32)
    D = (mx.random.normal((total_heads, v_dim)) * 0.1).astype(mx.float32)
    projected = (mx.random.normal((batch, seq, projected_dim)) * 0.1).astype(mx.float32)

    status = m2rnn_post_residual_gate_path_c_status(
        y,
        conv_input,
        D,
        projected,
        q_heads=q_heads,
        k_heads=k_heads,
        v_heads=v_heads,
        g_heads=g_heads,
    )
    if not status.available:
        pytest.skip(f"m2rnn post residual/gate Path C unavailable: {status.reason}")

    v_offset = q_heads * k_dim + k_heads * k_dim
    g_offset = projected_dim - g_heads * v_dim

    def reference(y_, conv_input_, D_, projected_):  # type: ignore[no-untyped-def]
        v = conv_input_[:, :, v_offset:].reshape(batch, seq, v_heads, v_dim)
        v_broadcast = mx.repeat(v, repeats=total_heads // v_heads, axis=-2)
        skipped = y_ + v_broadcast * D_.astype(y_.dtype)
        flat = skipped.reshape(batch, seq, total_heads * v_dim)
        g_flat = projected_[:, :, g_offset:]
        g_repeat = mx.repeat(g_flat, repeats=total_heads // g_heads, axis=-1)
        return flat * nn.silu(g_repeat).astype(flat.dtype)

    path_c = m2rnn_apply_post_residual_gate_path_c(
        y,
        conv_input,
        D,
        projected,
        q_heads=q_heads,
        k_heads=k_heads,
        v_heads=v_heads,
        g_heads=g_heads,
    )
    ref = reference(y, conv_input, D, projected)
    mx.eval(path_c, ref)
    np.testing.assert_allclose(_np(path_c), _np(ref), rtol=1e-6, atol=1e-6)

    def path_c_loss(y_, conv_input_, D_, projected_):  # type: ignore[no-untyped-def]
        out = m2rnn_apply_post_residual_gate_path_c(
            y_,
            conv_input_,
            D_,
            projected_,
            q_heads=q_heads,
            k_heads=k_heads,
            v_heads=v_heads,
            g_heads=g_heads,
        )
        return mx.sum(out * out) * 0.5

    def reference_loss(y_, conv_input_, D_, projected_):  # type: ignore[no-untyped-def]
        out = reference(y_, conv_input_, D_, projected_)
        return mx.sum(out * out) * 0.5

    path_c_grads = mx.grad(path_c_loss, argnums=(0, 1, 2, 3))(
        y,
        conv_input,
        D,
        projected,
    )
    ref_grads = mx.grad(reference_loss, argnums=(0, 1, 2, 3))(
        y,
        conv_input,
        D,
        projected,
    )
    mx.eval(*path_c_grads, *ref_grads)
    for got, expected in zip(path_c_grads, ref_grads, strict=True):
        np.testing.assert_allclose(_np(got), _np(expected), rtol=2e-3, atol=2e-4)


def test_m2rnn_packed_path_c_k_parallel_matches_path_b_backward() -> None:
    _require_m2rnn_path_c()
    if not m2rnn_metal_status().available:
        pytest.skip("m2rnn Metal Path B is not available on this host")

    inputs = _make_m2rnn_inputs(seq=4, k_dim=16, v_dim=4, dtype=mx.float32)
    q, k, v, W, xf, h0 = inputs
    conv_input = mx.concatenate(
        [
            q.reshape(q.shape[0], q.shape[1], -1),
            k.reshape(k.shape[0], k.shape[1], -1),
            v.reshape(v.shape[0], v.shape[1], -1),
        ],
        axis=-1,
    )
    mx.random.seed(13)
    dy = (mx.random.normal((1, 4, 2, 4)) * 0.1).astype(mx.float32)

    packed_full = m2rnn_path_c._m2rnn_packed_fwd_path_c_full(
        conv_input,
        W,
        xf,
        h0,
    )
    assert packed_full is not None
    y_packed, h_packed, tanh_packed = packed_full
    y_path_b, h_path_b, tanh_path_b = m2rnn_fwd_metal(*inputs)
    grads_packed = m2rnn_packed_bwd_path_c(
        dy,
        conv_input,
        W,
        xf,
        tanh_packed,
        h0,
    )
    grads_path_b = m2rnn_bwd_metal(dy, q, k, v, W, xf, tanh_path_b, h0)
    mx.eval(y_packed, h_packed, y_path_b, h_path_b, *grads_packed, *grads_path_b)

    dconv, dW_packed, dxf_packed, dh0_packed = grads_packed
    dq_packed = dconv[:, :, : q.shape[2] * q.shape[3]].reshape(q.shape)
    dk_start = q.shape[2] * q.shape[3]
    dk_stop = dk_start + k.shape[2] * k.shape[3]
    dk_packed = dconv[:, :, dk_start:dk_stop].reshape(k.shape)
    dv_packed = dconv[:, :, dk_stop:].reshape(v.shape)
    packed_parts = (
        dq_packed,
        dk_packed,
        dv_packed,
        dW_packed,
        dxf_packed,
        dh0_packed,
    )

    np.testing.assert_allclose(_np(y_packed), _np(y_path_b), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(_np(h_packed), _np(h_path_b), rtol=1e-6, atol=1e-6)
    for got, expected in zip(packed_parts, grads_path_b, strict=True):
        np.testing.assert_allclose(_np(got), _np(expected), rtol=2e-3, atol=2e-4)


def test_m2rnn_packed_path_c_k16_bfloat16_status_matches_callable() -> None:
    inputs = _make_m2rnn_inputs(seq=4, k_dim=16, v_dim=4, dtype=mx.bfloat16)
    q, k, v, W, xf, h0 = inputs
    conv_input = mx.concatenate(
        [
            q.reshape(q.shape[0], q.shape[1], -1),
            k.reshape(k.shape[0], k.shape[1], -1),
            v.reshape(v.shape[0], v.shape[1], -1),
        ],
        axis=-1,
    )
    status = m2rnn_packed_path_c_status(
        conv_input,
        W,
        xf,
        h0,
        require_backward=False,
    )
    assert (
        m2rnn_path_c._packed_path_c_inputs_eligible(conv_input, W, xf, h0)
        is status.available
    )

    if not status.available:
        assert (
            m2rnn_path_c._m2rnn_packed_fwd_path_c_full(conv_input, W, xf, h0)
            is None
        )
        with pytest.raises(
            RuntimeError,
            match="m2rnn_apply_packed_with_state_path_c unavailable",
        ):
            m2rnn_apply_packed_with_state_path_c(conv_input, W, xf, h0)
        return

    full = m2rnn_path_c._m2rnn_packed_fwd_path_c_full(conv_input, W, xf, h0)
    assert full is not None
    y_full, h_full, _tanh_cache = full
    y_call, h_call = m2rnn_apply_packed_with_state_path_c(conv_input, W, xf, h0)
    y_ref, h_ref = m2rnn_reference(q, k, v, W, xf, h0=h0)
    mx.eval(y_full, h_full, y_call, h_call, y_ref, h_ref)
    assert y_call.dtype == mx.bfloat16
    assert h_call.dtype == mx.bfloat16
    np.testing.assert_allclose(_np(y_full), _np(y_call), rtol=0.0, atol=0.0)
    np.testing.assert_allclose(_np(h_full), _np(h_call), rtol=0.0, atol=0.0)
    np.testing.assert_allclose(_np(y_call), _np(y_ref), rtol=3e-2, atol=3e-2)
    np.testing.assert_allclose(_np(h_call), _np(h_ref), rtol=3e-2, atol=3e-2)


def test_m2rnn_path_c_forward_backward_no_owner_outputs_are_mlx_compile_safe() -> None:
    _require_m2rnn_path_c()
    m2rnn_path_c._fwd_kernel_for.cache_clear()
    m2rnn_path_c._bwd_kernel_for.cache_clear()

    inputs = _make_m2rnn_inputs(dtype=mx.float32)

    def compiled_fwd(
        q: mx.array,
        k: mx.array,
        v: mx.array,
        W: mx.array,
        xf: mx.array,
        h0: mx.array,
    ) -> tuple[mx.array, mx.array, mx.array]:
        full = m2rnn_path_c._m2rnn_fwd_path_c_full(q, k, v, W, xf, h0)
        assert full is not None
        return full

    full = mx.compile(compiled_fwd)(*inputs)
    assert full is not None
    y, h_last, tanh_cache = full

    q, k, v, W, xf, h0 = inputs
    dy = mx.ones(y.shape, dtype=mx.float32)

    def compiled_bwd(
        dy: mx.array,
        q: mx.array,
        k: mx.array,
        v: mx.array,
        W: mx.array,
        xf: mx.array,
        tanh_cache: mx.array,
        h0: mx.array,
    ) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array, mx.array]:
        return m2rnn_bwd_path_c(
            dy,
            q,
            k,
            v,
            W,
            xf,
            tanh_cache,
            h0,
            force_path_c=True,
        )

    grads = mx.compile(compiled_bwd)(dy, q, k, v, W, xf, tanh_cache, h0)
    mx.eval(y, h_last, tanh_cache, *grads)
    assert y.shape == (1, 4, 2, 4)
    assert h_last.shape == (1, 2, 4, 4)
    assert [grad.shape for grad in grads] == [
        q.shape,
        k.shape,
        v.shape,
        W.shape,
        xf.shape,
        h0.shape,
    ]


def test_m2rnn_fwd_path_c_owner_outputs_avoid_hidden_zero_alloc(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _require_m2rnn_path_c()
    inputs = _make_m2rnn_inputs(dtype=mx.float32)
    y_out = mx.zeros((1, 4, 2, 4), dtype=mx.float32)
    h_out = mx.zeros((1, 2, 4, 4), dtype=mx.float32)
    tanh_out = mx.zeros((1, 4, 2, 4, 4), dtype=mx.float32)
    mx.eval(y_out, h_out, tanh_out)

    def fail_zero_alloc(*_args: object, **_kwargs: object) -> mx.array:
        raise AssertionError("owner-output fwd route must not allocate mx.zeros")

    monkeypatch.setattr(m2rnn_path_c.mx, "zeros", fail_zero_alloc)

    full = m2rnn_path_c._m2rnn_fwd_path_c_full(
        *inputs,
        out=(y_out, h_out, tanh_out),
    )
    assert full is not None
    y, h_last, tanh_cache = full
    mx.eval(y, h_last, tanh_cache)
    assert y is y_out
    assert h_last is h_out
    assert tanh_cache is tanh_out


def test_m2rnn_bwd_path_c_rejects_public_backward_owner_outputs() -> None:
    _require_m2rnn_path_c()
    inputs = _make_m2rnn_inputs(dtype=mx.float32)
    full = m2rnn_path_c._m2rnn_fwd_path_c_full(*inputs)
    assert full is not None
    y, _h_last, tanh_cache = full
    q, k, v, W, xf, h0 = inputs
    dy = mx.ones(y.shape, dtype=mx.float32)
    owner_outputs = (
        mx.zeros(q.shape, dtype=mx.float32),
        mx.zeros(k.shape, dtype=mx.float32),
        mx.zeros(v.shape, dtype=mx.float32),
        mx.zeros(W.shape, dtype=mx.float32),
        mx.zeros(xf.shape, dtype=mx.float32),
        mx.zeros(h0.shape, dtype=mx.float32),
        mx.zeros((1, 2, 4, 4, 4), dtype=mx.float32),
    )
    mx.eval(dy, tanh_cache, *owner_outputs)

    with pytest.raises(RuntimeError, match="does not expose backward owner-output"):
        m2rnn_bwd_path_c(
            dy,
            q,
            k,
            v,
            W,
            xf,
            tanh_cache,
            h0,
            force_path_c=True,
            out=owner_outputs,
        )


def test_m2rnn_packed_bwd_path_c_rejects_public_backward_owner_outputs() -> None:
    inputs = _make_m2rnn_inputs(dtype=mx.float32)
    q, k, v, W, xf, h0 = inputs
    conv_input = mx.concatenate(
        [
            q.reshape(q.shape[0], q.shape[1], -1),
            k.reshape(k.shape[0], k.shape[1], -1),
            v.reshape(v.shape[0], v.shape[1], -1),
        ],
        axis=-1,
    )
    dy = mx.ones((1, 4, 2, 4), dtype=mx.float32)
    tanh_cache = mx.zeros((1, 4, 2, 4, 4), dtype=mx.float32)
    owner_outputs = (
        mx.zeros(conv_input.shape, dtype=mx.float32),
        mx.zeros(W.shape, dtype=mx.float32),
        mx.zeros(xf.shape, dtype=mx.float32),
        mx.zeros(h0.shape, dtype=mx.float32),
        mx.zeros((1, 2, 4, 4, 4), dtype=mx.float32),
    )
    mx.eval(dy, tanh_cache, *owner_outputs)

    with pytest.raises(RuntimeError, match="does not expose backward owner-output"):
        m2rnn_packed_bwd_path_c(
            dy,
            conv_input,
            W,
            xf,
            tanh_cache,
            h0,
            out=owner_outputs,
        )


def test_m2rnn_mapped_packed_bwd_path_c_rejects_public_backward_owner_outputs() -> None:
    batch, seq, k_dim, v_dim = 1, 4, 4, 3
    q_heads, k_heads, v_heads = 1, 1, 2
    total_heads, w_heads, f_heads = 4, 1, 2
    conv_dim = q_heads * k_dim + k_heads * k_dim + v_heads * v_dim
    conv_input = mx.zeros((batch, seq, conv_dim), dtype=mx.float32)
    W = mx.zeros((w_heads, v_dim, v_dim), dtype=mx.float32)
    xf = mx.zeros((batch, seq, f_heads), dtype=mx.float32)
    h0 = mx.zeros((batch, total_heads, k_dim, v_dim), dtype=mx.float32)
    dy = mx.zeros((batch, seq, total_heads, v_dim), dtype=mx.float32)
    tanh_cache = mx.zeros((batch, seq, total_heads, k_dim, v_dim), dtype=mx.float32)
    owner_outputs = (
        mx.zeros(conv_input.shape, dtype=mx.float32),
        mx.zeros((w_heads, v_dim, v_dim), dtype=mx.float32),
        mx.zeros(xf.shape, dtype=mx.float32),
        mx.zeros(h0.shape, dtype=mx.float32),
        mx.zeros((batch, total_heads, seq, k_dim, v_dim), dtype=mx.float32),
    )
    mx.eval(dy, tanh_cache, *owner_outputs)

    with pytest.raises(RuntimeError, match="does not expose backward owner-output"):
        m2rnn_path_c.m2rnn_mapped_packed_bwd_path_c(
            dy,
            conv_input,
            W,
            xf,
            tanh_cache,
            h0,
            q_heads=q_heads,
            k_heads=k_heads,
            v_heads=v_heads,
            out=owner_outputs,
        )


def test_m2rnn_path_c_mixed_dtype_fails_closed_without_hidden_casts() -> None:
    inputs = _make_m2rnn_inputs(dtype=mx.float32)
    q, k, v, W, xf, h0 = inputs
    assert m2rnn_path_c._m2rnn_fwd_path_c_full(
        q,
        k,
        v,
        W,
        xf,
        h0.astype(mx.float16),
    ) is None


def test_m2rnn_path_c_backward_matches_path_b_when_available() -> None:
    _require_m2rnn_path_c()
    if not m2rnn_metal_status().available:
        pytest.skip("m2rnn Metal Path B is not available on this host")

    inputs = _make_m2rnn_inputs(dtype=mx.float32)
    q, k, v, W, xf, h0 = inputs
    mx.random.seed(11)
    dy = (mx.random.normal((1, 4, 2, 4)) * 0.1).astype(mx.float32)
    _y, _h, tanh_cache = m2rnn_fwd_metal(*inputs)
    grads_pc = m2rnn_bwd_path_c(
        dy,
        q,
        k,
        v,
        W,
        xf,
        tanh_cache,
        h0,
        force_path_c=True,
    )
    grads_pb = m2rnn_bwd_metal(dy, q, k, v, W, xf, tanh_cache, h0)
    mx.eval(*grads_pc, *grads_pb)
    for got, expected in zip(grads_pc, grads_pb):
        np.testing.assert_allclose(_np(got), _np(expected), rtol=2e-3, atol=2e-4)


def test_m2rnn_path_c_vjp_runs_under_mlx_graph_transform() -> None:
    _require_m2rnn_path_c()
    if not m2rnn_metal_status().available:
        pytest.skip("m2rnn Metal Path B is not available on this host")

    inputs = _make_m2rnn_inputs(dtype=mx.float32)

    def path_c_loss(q, k, v, W, xf, h0):  # type: ignore[no-untyped-def]
        y = m2rnn_apply_path_c(q, k, v, W, xf, h0, force_path_c=True)
        return mx.sum(y * y) * 0.5

    def path_b_loss(q, k, v, W, xf, h0):  # type: ignore[no-untyped-def]
        y = m2rnn_apply(q, k, v, W, xf, h0)
        return mx.sum(y * y) * 0.5

    grads_pc = mx.grad(path_c_loss, argnums=tuple(range(6)))(*inputs)
    grads_pb = mx.grad(path_b_loss, argnums=tuple(range(6)))(*inputs)
    mx.eval(*grads_pc, *grads_pb)
    for got, expected in zip(grads_pc, grads_pb, strict=True):
        got_np = _np(got)
        assert np.isfinite(got_np).all()
        np.testing.assert_allclose(got_np, _np(expected), rtol=2e-3, atol=2e-4)
