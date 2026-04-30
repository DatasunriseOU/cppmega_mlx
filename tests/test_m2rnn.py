from __future__ import annotations

import math
from functools import partial

import numpy as np
import pytest

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten

from cppmega_mlx.nn.m2rnn import (
    DEFAULT_CHUNK_SIZE,
    M2RNNConfig,
    M2RNNMixer,
    broadcast_m2rnn_heads,
    chunked_m2rnn_scan,
    m2rnn_softplus_decay_gate,
    m2rnn_scan,
)


def _rand(shape: tuple[int, ...], rng: np.random.Generator) -> mx.array:
    return mx.array(rng.standard_normal(shape, dtype=np.float32))


def _inputs(
    *,
    batch: int = 2,
    seq: int = 32,
    n_q: int = 2,
    n_k: int | None = None,
    n_v: int | None = None,
    n_w: int | None = None,
    n_f: int | None = None,
    k_dim: int = 8,
    v_dim: int = 4,
    seed: int = 0,
) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array]:
    rng = np.random.default_rng(seed)
    n_k = n_q if n_k is None else n_k
    n_v = n_q if n_v is None else n_v
    n_w = n_q if n_w is None else n_w
    n_f = n_q if n_f is None else n_f

    q = _rand((batch, seq, n_q, k_dim), rng)
    k = _rand((batch, seq, n_k, k_dim), rng)
    v = _rand((batch, seq, n_v, v_dim), rng)
    W = mx.array(np.broadcast_to(np.eye(v_dim, dtype=np.float32), (n_w, v_dim, v_dim)).copy())
    W = W + 0.01 * _rand((n_w, v_dim, v_dim), rng)
    xf = mx.array(rng.random((batch, seq, n_f), dtype=np.float32))
    return q, k, v, W, xf


def _assert_close(actual: mx.array, expected: mx.array, *, atol: float = 1e-5) -> None:
    np.testing.assert_allclose(np.array(actual), np.array(expected), atol=atol, rtol=atol)


def _sum_repeated_head_groups(grad: mx.array, orig_heads: int, *, axis: int) -> mx.array:
    """Collapse repeat-interleaved head gradients back to their source heads."""

    if grad.shape[axis] == orig_heads:
        return grad
    if grad.shape[axis] % orig_heads != 0:
        raise ValueError(f"cannot collapse {grad.shape[axis]} heads to {orig_heads}")
    repeats = grad.shape[axis] // orig_heads
    normalized_axis = axis if axis >= 0 else grad.ndim + axis
    shape = list(grad.shape)
    shape[normalized_axis : normalized_axis + 1] = [orig_heads, repeats]
    return grad.reshape(shape).sum(axis=normalized_axis + 1)


def _numpy_megatron_m2rnn_reference(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    W: mx.array,
    xf: mx.array,
    *,
    h0: mx.array | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    q_np = np.array(q)
    k_np = np.array(k)
    v_np = np.array(v)
    W_np = np.array(W)
    xf_np = np.array(xf)

    batch, seq, n_q, k_dim = q_np.shape
    n_k = k_np.shape[-2]
    n_v = v_np.shape[-2]
    n_w = W_np.shape[0]
    n_f = xf_np.shape[-1]
    v_dim = v_np.shape[-1]
    heads = max(n_q, n_k, n_v, n_w, n_f)

    if n_q != heads:
        q_np = np.repeat(q_np, heads // n_q, axis=-2)
    if n_k != heads:
        k_np = np.repeat(k_np, heads // n_k, axis=-2)
    if n_v != heads:
        v_np = np.repeat(v_np, heads // n_v, axis=-2)
    if n_w != heads:
        W_np = np.repeat(W_np, heads // n_w, axis=0)
    if n_f != heads:
        xf_np = np.repeat(xf_np, heads // n_f, axis=-1)

    h = (
        np.zeros((batch, heads, k_dim, v_dim), dtype=q_np.dtype)
        if h0 is None
        else np.array(h0).copy()
    )
    x = k_np[..., None] * v_np[..., None, :]
    W_expanded = W_np[None, ...]
    y = np.empty((batch, seq, heads, k_dim, v_dim), dtype=q_np.dtype)
    for s in range(seq):
        f = xf_np[:, s, :, None, None]
        h_new = np.tanh(np.matmul(h, W_expanded) + x[:, s])
        h = f * h + (1.0 - f) * h_new
        y[:, s] = h
    out = np.einsum("bshk,bshkv->bshv", q_np, y)
    return out, h


def _flat_tree(tree) -> dict[str, np.ndarray]:
    mx.eval(tree)
    return {name: np.array(value) for name, value in tree_flatten(tree)}


def _mixer_config(
    *,
    d_model: int = 16,
    k_head_dim: int = 4,
    v_head_dim: int = 3,
    num_q_heads: int = 1,
    num_k_heads: int = 1,
    num_v_heads: int = 2,
    num_f_heads: int = 2,
    num_g_heads: int = 2,
    num_weight_heads: int = 1,
    chunk_size: int = 3,
    use_residual: bool = True,
    A_init_min: float = 0.0,
    A_init_max: float = 16.0,
    dt_init_min: float = 1e-3,
    dt_init_max: float = 0.1,
    dt_init_floor: float = 1e-4,
) -> M2RNNConfig:
    return M2RNNConfig(
        d_model=d_model,
        k_head_dim=k_head_dim,
        v_head_dim=v_head_dim,
        num_q_heads=num_q_heads,
        num_k_heads=num_k_heads,
        num_v_heads=num_v_heads,
        num_f_heads=num_f_heads,
        num_g_heads=num_g_heads,
        num_weight_heads=num_weight_heads,
        chunk_size=chunk_size,
        use_residual=use_residual,
        A_init_min=A_init_min,
        A_init_max=A_init_max,
        dt_init_min=dt_init_min,
        dt_init_max=dt_init_max,
        dt_init_floor=dt_init_floor,
    )


@pytest.mark.parametrize("chunk_size", [1, 2, 7, 16, 64, 128])
def test_chunk_sizes_match_sequential_scan(chunk_size: int) -> None:
    q, k, v, W, xf = _inputs(seq=33, seed=chunk_size)

    out_seq, h_seq = m2rnn_scan(q, k, v, W, xf)
    out_chunk, h_chunk = chunked_m2rnn_scan(q, k, v, W, xf, chunk_size=chunk_size)
    mx.eval(out_seq, h_seq, out_chunk, h_chunk)

    _assert_close(out_chunk, out_seq)
    _assert_close(h_chunk, h_seq)


def test_non_divisible_sequence_length_matches_sequential_scan() -> None:
    q, k, v, W, xf = _inputs(seq=37, seed=37)

    out_seq, h_seq = m2rnn_scan(q, k, v, W, xf)
    out_chunk, h_chunk = chunked_m2rnn_scan(q, k, v, W, xf, chunk_size=8)
    mx.eval(out_seq, h_seq, out_chunk, h_chunk)

    _assert_close(out_chunk, out_seq)
    _assert_close(h_chunk, h_seq)


def test_compiled_chunked_scan_matches_eager_scan() -> None:
    q, k, v, W, xf = _inputs(batch=1, seq=8, n_q=2, k_dim=4, v_dim=3, seed=38)

    @mx.compile
    def compiled_chunked_scan(q, k, v, W, xf):
        return chunked_m2rnn_scan(q, k, v, W, xf, chunk_size=3)

    out_eager, h_eager = chunked_m2rnn_scan(q, k, v, W, xf, chunk_size=3)
    out_compiled, h_compiled = compiled_chunked_scan(q, k, v, W, xf)
    mx.eval(out_eager, h_eager, out_compiled, h_compiled)

    _assert_close(out_compiled, out_eager)
    _assert_close(h_compiled, h_eager)


def test_chunked_scan_matches_megatron_reference_formula_with_broadcast_h0() -> None:
    q, k, v, W, xf = _inputs(
        batch=2,
        seq=7,
        n_q=1,
        n_k=2,
        n_v=4,
        n_w=1,
        n_f=2,
        k_dim=5,
        v_dim=3,
        seed=39,
    )
    h0 = _rand((2, 4, 5, 3), np.random.default_rng(40))

    expected_out, expected_h = _numpy_megatron_m2rnn_reference(q, k, v, W, xf, h0=h0)
    out, h = chunked_m2rnn_scan(q, k, v, W, xf, h0=h0, chunk_size=3)
    mx.eval(out, h)

    np.testing.assert_allclose(np.array(out), expected_out, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(np.array(h), expected_h, atol=1e-5, rtol=1e-5)


def test_seq_len_one_matches_sequential_scan() -> None:
    q, k, v, W, xf = _inputs(seq=1, seed=1)

    out_seq, h_seq = m2rnn_scan(q, k, v, W, xf)
    out_chunk, h_chunk = chunked_m2rnn_scan(q, k, v, W, xf, chunk_size=128)
    mx.eval(out_seq, h_seq, out_chunk, h_chunk)

    assert out_chunk.shape == (2, 1, 2, 4)
    assert h_chunk.shape == (2, 2, 8, 4)
    _assert_close(out_chunk, out_seq)
    _assert_close(h_chunk, h_seq)


def test_h0_none_equals_explicit_zeros() -> None:
    q, k, v, W, xf = _inputs(batch=1, seq=11, n_q=3, k_dim=6, v_dim=5, seed=11)
    h0 = mx.zeros((1, 3, 6, 5), dtype=q.dtype)

    out_none, h_none = chunked_m2rnn_scan(q, k, v, W, xf, h0=None, chunk_size=4)
    out_zero, h_zero = chunked_m2rnn_scan(q, k, v, W, xf, h0=h0, chunk_size=4)
    mx.eval(out_none, h_none, out_zero, h_zero)

    _assert_close(out_none, out_zero)
    _assert_close(h_none, h_zero)


def test_h0_continues_recurrence_across_chunked_segments() -> None:
    q, k, v, W, xf = _inputs(seq=10, seed=44)

    full_out, full_h = chunked_m2rnn_scan(q, k, v, W, xf, chunk_size=4)
    prefix_out, prefix_h = chunked_m2rnn_scan(
        q[:, :4],
        k[:, :4],
        v[:, :4],
        W,
        xf[:, :4],
        chunk_size=3,
    )
    suffix_out, suffix_h = chunked_m2rnn_scan(
        q[:, 4:],
        k[:, 4:],
        v[:, 4:],
        W,
        xf[:, 4:],
        h0=prefix_h,
        chunk_size=2,
    )
    stitched_out = mx.concatenate([prefix_out, suffix_out], axis=1)
    mx.eval(full_out, full_h, stitched_out, suffix_h)

    _assert_close(stitched_out, full_out)
    _assert_close(suffix_h, full_h)


def test_chunked_scan_supports_direct_mlx_gradients() -> None:
    q, k, v, W, xf = _inputs(batch=1, seq=4, n_q=1, k_dim=3, v_dim=2, seed=57)

    def loss_fn(
        q: mx.array,
        k: mx.array,
        v: mx.array,
        W: mx.array,
        xf: mx.array,
    ) -> mx.array:
        out, h = chunked_m2rnn_scan(q, k, v, W, xf, chunk_size=2)
        return mx.mean(mx.square(out)) + 0.01 * mx.mean(mx.square(h))

    loss, grads = mx.value_and_grad(loss_fn, argnums=[0, 1, 2, 3, 4])(q, k, v, W, xf)
    mx.eval(loss, grads)

    assert math.isfinite(float(loss.item()))
    expected = (("q", q), ("k", k), ("v", v), ("W", W), ("xf", xf))
    for grad, (name, input_array) in zip(grads, expected, strict=True):
        grad_np = np.array(grad)
        assert grad.shape == input_array.shape
        assert np.isfinite(grad_np).all(), name
        assert float(np.max(np.abs(grad_np))) > 0.0, name


def test_empty_sequence_preserves_initial_state() -> None:
    q, k, v, W, xf = _inputs(seq=0, seed=55)
    h0 = _rand((2, 2, 8, 4), np.random.default_rng(56))

    out, h = chunked_m2rnn_scan(q, k, v, W, xf, h0=h0, chunk_size=5)
    mx.eval(out, h)

    assert out.shape == (2, 0, 2, 4)
    assert h.shape == h0.shape
    _assert_close(h, h0)


def test_broadcast_nq1_nk1_nv4_nw1() -> None:
    q, k, v, W, xf = _inputs(
        batch=1,
        seq=13,
        n_q=1,
        n_k=1,
        n_v=4,
        n_w=1,
        n_f=4,
        k_dim=6,
        v_dim=3,
        seed=123,
    )

    bq, bk, bv, bW, bxf = broadcast_m2rnn_heads(q, k, v, W, xf)
    assert bq.shape == (1, 13, 4, 6)
    assert bk.shape == (1, 13, 4, 6)
    assert bv.shape == (1, 13, 4, 3)
    assert bW.shape == (4, 3, 3)
    assert bxf.shape == (1, 13, 4)

    out_seq, h_seq = m2rnn_scan(q, k, v, W, xf)
    out_chunk, h_chunk = chunked_m2rnn_scan(q, k, v, W, xf, chunk_size=5)
    mx.eval(out_seq, h_seq, out_chunk, h_chunk)

    assert out_chunk.shape == (1, 13, 4, 3)
    assert h_chunk.shape == (1, 4, 6, 3)
    _assert_close(out_chunk, out_seq)
    _assert_close(h_chunk, h_seq)


def test_broadcast_nq1_nk1_nv4_nw1_nf1_matches_chunked_scan() -> None:
    q, k, v, W, xf = _inputs(
        batch=1,
        seq=9,
        n_q=1,
        n_k=1,
        n_v=4,
        n_w=1,
        n_f=1,
        k_dim=5,
        v_dim=3,
        seed=124,
    )

    out_seq, h_seq = m2rnn_scan(q, k, v, W, xf)
    out_chunk, h_chunk = chunked_m2rnn_scan(q, k, v, W, xf, chunk_size=4)
    mx.eval(out_seq, h_seq, out_chunk, h_chunk)

    assert out_chunk.shape == (1, 9, 4, 3)
    assert h_chunk.shape == (1, 4, 5, 3)
    _assert_close(out_chunk, out_seq)
    _assert_close(h_chunk, h_seq)


def test_production_style_broadcast_gradients_match_explicit_head_reduction() -> None:
    q, k, v, W, xf = _inputs(
        batch=2,
        seq=11,
        n_q=1,
        n_k=1,
        n_v=8,
        n_w=1,
        n_f=1,
        k_dim=4,
        v_dim=3,
        seed=125,
    )
    rng = np.random.default_rng(126)
    h0 = _rand((2, 8, 4, 3), rng)
    out_probe = _rand((2, 11, 8, 3), rng)
    h_probe = _rand((2, 8, 4, 3), rng)

    def loss_fn(
        q_arg: mx.array,
        k_arg: mx.array,
        v_arg: mx.array,
        W_arg: mx.array,
        xf_arg: mx.array,
        h0_arg: mx.array,
    ) -> mx.array:
        out, h = chunked_m2rnn_scan(
            q_arg,
            k_arg,
            v_arg,
            W_arg,
            xf_arg,
            h0=h0_arg,
            chunk_size=5,
        )
        return mx.mean(out * out_probe) + 0.07 * mx.mean(h * h_probe)

    grad_fn = mx.value_and_grad(loss_fn, argnums=[0, 1, 2, 3, 4, 5])
    implicit_loss, implicit_grads = grad_fn(q, k, v, W, xf, h0)
    explicit_q, explicit_k, explicit_v, explicit_W, explicit_xf = broadcast_m2rnn_heads(
        q, k, v, W, xf
    )
    explicit_loss, explicit_grads = grad_fn(
        explicit_q,
        explicit_k,
        explicit_v,
        explicit_W,
        explicit_xf,
        h0,
    )
    mx.eval(implicit_loss, implicit_grads, explicit_loss, explicit_grads)

    _assert_close(implicit_loss, explicit_loss, atol=2e-5)
    expected_grads = (
        _sum_repeated_head_groups(explicit_grads[0], q.shape[-2], axis=-2),
        _sum_repeated_head_groups(explicit_grads[1], k.shape[-2], axis=-2),
        explicit_grads[2],
        _sum_repeated_head_groups(explicit_grads[3], W.shape[0], axis=0),
        _sum_repeated_head_groups(explicit_grads[4], xf.shape[-1], axis=-1),
        explicit_grads[5],
    )
    for name, implicit_grad, expected_grad in zip(
        ("q", "k", "v", "W", "xf", "h0"),
        implicit_grads,
        expected_grads,
        strict=True,
    ):
        grad_np = np.array(implicit_grad)
        assert implicit_grad.shape == expected_grad.shape
        assert np.isfinite(grad_np).all(), name
        assert float(np.max(np.abs(grad_np))) > 0.0, name
        _assert_close(implicit_grad, expected_grad, atol=3e-5)


def test_compiled_production_style_broadcast_gradients_match_eager() -> None:
    q, k, v, W, xf = _inputs(
        batch=1,
        seq=7,
        n_q=1,
        n_k=1,
        n_v=4,
        n_w=1,
        n_f=1,
        k_dim=5,
        v_dim=3,
        seed=127,
    )
    rng = np.random.default_rng(128)
    h0 = _rand((1, 4, 5, 3), rng)
    out_probe = _rand((1, 7, 4, 3), rng)
    h_probe = _rand((1, 4, 5, 3), rng)

    def loss_fn(
        q_arg: mx.array,
        k_arg: mx.array,
        v_arg: mx.array,
        W_arg: mx.array,
        xf_arg: mx.array,
        h0_arg: mx.array,
    ) -> mx.array:
        out, h = chunked_m2rnn_scan(
            q_arg,
            k_arg,
            v_arg,
            W_arg,
            xf_arg,
            h0=h0_arg,
            chunk_size=3,
        )
        return mx.mean(mx.square(out - out_probe)) + 0.03 * mx.mean(mx.square(h - h_probe))

    grad_fn = mx.value_and_grad(loss_fn, argnums=[0, 1, 2, 3, 4, 5])

    @mx.compile
    def compiled_grad_fn(
        q_arg: mx.array,
        k_arg: mx.array,
        v_arg: mx.array,
        W_arg: mx.array,
        xf_arg: mx.array,
        h0_arg: mx.array,
    ):
        return grad_fn(q_arg, k_arg, v_arg, W_arg, xf_arg, h0_arg)

    eager_loss, eager_grads = grad_fn(q, k, v, W, xf, h0)
    compiled_loss, compiled_grads = compiled_grad_fn(q, k, v, W, xf, h0)
    mx.eval(eager_loss, eager_grads, compiled_loss, compiled_grads)

    _assert_close(compiled_loss, eager_loss, atol=2e-5)
    for name, compiled_grad, eager_grad in zip(
        ("q", "k", "v", "W", "xf", "h0"),
        compiled_grads,
        eager_grads,
        strict=True,
    ):
        grad_np = np.array(compiled_grad)
        assert compiled_grad.shape == eager_grad.shape
        assert np.isfinite(grad_np).all(), name
        assert float(np.max(np.abs(grad_np))) > 0.0, name
        _assert_close(compiled_grad, eager_grad, atol=3e-5)


def test_suffix_loss_backpropagates_through_explicit_h0() -> None:
    q, k, v, W, xf = _inputs(
        batch=2,
        seq=10,
        n_q=1,
        n_k=1,
        n_v=4,
        n_w=1,
        n_f=1,
        k_dim=5,
        v_dim=3,
        seed=129,
    )
    split = 4
    rng = np.random.default_rng(130)
    h0 = _rand((2, 4, 5, 3), rng)
    out_probe = _rand((2, 6, 4, 3), rng)
    h_probe = _rand((2, 4, 5, 3), rng)

    def suffix_loss(h0_arg: mx.array) -> mx.array:
        out, h = chunked_m2rnn_scan(
            q[:, split:],
            k[:, split:],
            v[:, split:],
            W,
            xf[:, split:],
            h0=h0_arg,
            chunk_size=4,
        )
        return mx.mean(out * out_probe) + 0.05 * mx.mean(h * h_probe)

    loss, grad_h0 = mx.value_and_grad(suffix_loss)(h0)
    mx.eval(loss, grad_h0)

    grad_np = np.array(grad_h0)
    assert math.isfinite(float(loss.item()))
    assert grad_h0.shape == h0.shape
    assert np.isfinite(grad_np).all()
    assert float(np.max(np.abs(grad_np))) > 1e-4

    max_index = np.unravel_index(int(np.argmax(np.abs(grad_np))), grad_np.shape)
    mask_np = np.zeros(h0.shape, dtype=np.float32)
    mask_np[max_index] = 1.0
    mask = mx.array(mask_np)
    eps = 1e-2
    plus_loss = suffix_loss(h0 + eps * mask)
    minus_loss = suffix_loss(h0 - eps * mask)
    mx.eval(plus_loss, minus_loss)
    finite_difference = (float(plus_loss.item()) - float(minus_loss.item())) / (2.0 * eps)

    np.testing.assert_allclose(grad_np[max_index], finite_difference, atol=5e-3, rtol=5e-2)


def test_non_divisible_direct_head_broadcast_rejected() -> None:
    q, k, v, W, xf = _inputs(n_q=2, n_k=3, n_v=2, n_w=1, n_f=1, seed=132)

    with pytest.raises(ValueError, match="head count .* must divide"):
        broadcast_m2rnn_heads(q, k, v, W, xf)


def test_scan_rejects_mixed_input_dtypes() -> None:
    q, k, v, W, xf = _inputs(seed=133)

    with pytest.raises(TypeError, match="k dtype"):
        m2rnn_scan(q, k.astype(mx.float16), v, W, xf)


def test_scan_rejects_integer_forget_gate() -> None:
    q, k, v, W, xf = _inputs(seed=134)

    with pytest.raises(TypeError, match="xf must use a floating dtype"):
        m2rnn_scan(q, k, v, W, xf.astype(mx.int32))


def test_scan_rejects_h0_dtype_mismatch() -> None:
    q, k, v, W, xf = _inputs(batch=1, seq=3, n_q=2, k_dim=4, v_dim=3, seed=135)
    h0 = mx.zeros((1, 2, 4, 3), dtype=mx.float16)

    with pytest.raises(TypeError, match="h0 dtype"):
        chunked_m2rnn_scan(q, k, v, W, xf, h0=h0, chunk_size=2)


def test_default_chunk_size_constant() -> None:
    assert DEFAULT_CHUNK_SIZE == 128


def test_lightweight_mixer_forward_shape_dtype_and_finiteness() -> None:
    mx.random.seed(5)
    cfg = _mixer_config()
    mixer = M2RNNMixer(cfg)
    hidden = _rand((2, 5, 16), np.random.default_rng(5))

    out, h = mixer(hidden)
    mx.eval(out, h)

    assert out.shape == hidden.shape
    assert out.dtype == hidden.dtype
    assert h.shape == (2, 2, 4, 3)
    assert h.dtype == hidden.dtype
    assert np.isfinite(np.array(out)).all()
    assert np.isfinite(np.array(h)).all()


def test_lightweight_mixer_ports_megatron_output_gate_and_norm_shapes() -> None:
    mx.random.seed(6)
    cfg = _mixer_config(num_q_heads=1, num_k_heads=1, num_v_heads=2, num_f_heads=2, num_g_heads=1)
    mixer = M2RNNMixer(cfg)
    hidden = _rand((2, 5, cfg.d_model), np.random.default_rng(6))

    out, h = mixer(hidden)
    mx.eval(out, h, mixer.parameters())

    assert mixer.g_dim == cfg.num_g_heads * cfg.v_head_dim
    assert mixer.in_proj.weight.shape == (
        mixer.q_dim + mixer.k_dim + mixer.v_dim + mixer.f_dim + mixer.g_dim,
        cfg.d_model,
    )
    assert mixer.g_norm.weight.shape == (cfg.num_heads * cfg.v_head_dim,)
    assert mixer.out_proj.weight.shape == (cfg.d_model, cfg.num_heads * cfg.v_head_dim)
    assert out.shape == hidden.shape
    assert h.shape == (2, cfg.num_heads, cfg.k_head_dim, cfg.v_head_dim)


def test_softplus_decay_gate_range_shape_and_parameter_sensitivity() -> None:
    f_input = mx.array(
        [
            [[-1.0, 0.0], [0.5, 1.0], [2.0, -2.0]],
            [[0.25, -0.25], [1.5, -1.5], [0.0, 0.75]],
        ],
        dtype=mx.float32,
    )
    A_log = mx.log(mx.array([0.5, 2.0], dtype=mx.float32))
    dt_bias = mx.array([-0.5, 0.25], dtype=mx.float32)

    gate = m2rnn_softplus_decay_gate(f_input, A_log, dt_bias)
    faster_decay_gate = m2rnn_softplus_decay_gate(f_input, A_log + 0.5, dt_bias)
    shifted_dt_gate = m2rnn_softplus_decay_gate(f_input, A_log, dt_bias + 0.5)
    mx.eval(gate, faster_decay_gate, shifted_dt_gate)

    gate_np = np.array(gate)
    assert gate.shape == f_input.shape
    assert gate.dtype == f_input.dtype
    assert np.isfinite(gate_np).all()
    assert np.all(gate_np > 0.0)
    assert np.all(gate_np <= 1.0)
    assert np.max(np.abs(np.array(faster_decay_gate) - gate_np)) > 1e-4
    assert np.max(np.abs(np.array(shifted_dt_gate) - gate_np)) > 1e-4


def test_softplus_decay_gate_broadcasts_forget_heads_to_state_heads() -> None:
    f_input = mx.zeros((2, 3, 1), dtype=mx.float32)
    A_log = mx.log(mx.array([0.5, 1.0, 2.0, 4.0], dtype=mx.float32))
    dt_bias = mx.zeros((4,), dtype=mx.float32)

    gate = m2rnn_softplus_decay_gate(f_input, A_log, dt_bias)
    mx.eval(gate)

    assert gate.shape == (2, 3, 4)
    assert np.isfinite(np.array(gate)).all()


def test_lightweight_mixer_initializes_megatron_style_decay_parameters() -> None:
    mx.random.seed(13)
    cfg = _mixer_config(A_init_min=0.25, A_init_max=1.0, dt_init_min=0.01, dt_init_max=0.02)
    mixer = M2RNNMixer(cfg)
    mx.eval(mixer.A_log, mixer.dt_bias)

    assert mixer.A_log.shape == (cfg.num_heads,)
    assert mixer.dt_bias.shape == (cfg.num_heads,)
    assert mixer.A_log.dtype == mx.float32
    assert mixer.dt_bias.dtype == mx.float32

    A = np.exp(np.array(mixer.A_log))
    dt = np.array(nn.softplus(mixer.dt_bias))
    assert np.isfinite(A).all()
    assert np.isfinite(dt).all()
    assert np.all(A >= 0.25 - 1e-6)
    assert np.all(A <= 1.0 + 1e-6)
    assert np.all(dt >= 0.01 - 1e-6)
    assert np.all(dt <= 0.02 + 1e-6)


def test_lightweight_mixer_h0_continues_recurrence_across_segments() -> None:
    mx.random.seed(15)
    cfg = _mixer_config(chunk_size=4)
    mixer = M2RNNMixer(cfg)
    hidden = _rand((2, 9, cfg.d_model), np.random.default_rng(15))

    full_out, full_h = mixer(hidden, chunk_size=4)
    prefix_out, prefix_h = mixer(hidden[:, :4], chunk_size=3)
    suffix_out, suffix_h = mixer(hidden[:, 4:], h0=prefix_h, chunk_size=2)
    stitched_out = mx.concatenate([prefix_out, suffix_out], axis=1)
    mx.eval(full_out, full_h, stitched_out, suffix_h)

    _assert_close(stitched_out, full_out)
    _assert_close(suffix_h, full_h)


def test_lightweight_mixer_h0_rejects_wrong_state_shape_after_head_broadcast() -> None:
    mx.random.seed(16)
    cfg = _mixer_config(
        num_q_heads=1,
        num_k_heads=1,
        num_v_heads=4,
        num_f_heads=2,
        num_g_heads=2,
        num_weight_heads=1,
    )
    mixer = M2RNNMixer(cfg)
    hidden = _rand((2, 3, cfg.d_model), np.random.default_rng(16))
    h0 = mx.zeros((2, cfg.num_heads - 1, cfg.k_head_dim, cfg.v_head_dim), dtype=hidden.dtype)

    with pytest.raises(ValueError, match="h0 must have shape"):
        mixer(hidden, h0=h0)


def test_lightweight_mixer_broadcast_head_runtime_has_finite_state_and_gradients() -> None:
    mx.random.seed(21)
    cfg = _mixer_config(
        num_q_heads=1,
        num_k_heads=1,
        num_v_heads=4,
        num_f_heads=2,
        num_g_heads=2,
        num_weight_heads=1,
        chunk_size=2,
    )
    mixer = M2RNNMixer(cfg)
    hidden = _rand((2, 5, cfg.d_model), np.random.default_rng(21))
    target = _rand((2, 5, cfg.d_model), np.random.default_rng(22))

    def loss_fn(model: M2RNNMixer, x: mx.array, y: mx.array) -> mx.array:
        pred, final_state = model(x)
        return mx.mean(mx.square(pred - y)) + 0.01 * mx.mean(mx.square(final_state))

    loss_and_grad = nn.value_and_grad(mixer, loss_fn)
    loss, grads = loss_and_grad(mixer, hidden, target)
    out, final_state = mixer(hidden)
    mx.eval(loss, grads, out, final_state)

    assert math.isfinite(float(loss.item()))
    assert out.shape == hidden.shape
    assert final_state.shape == (2, cfg.num_heads, cfg.k_head_dim, cfg.v_head_dim)

    flat_grads = _flat_tree(grads)
    for name in ("in_proj.weight", "state_weight", "A_log", "dt_bias", "D"):
        assert name in flat_grads
        assert np.isfinite(flat_grads[name]).all(), name
        assert float(np.max(np.abs(flat_grads[name]))) > 0.0, name


def test_lightweight_mixer_finite_loss_gradients_and_optimizer_step() -> None:
    mx.random.seed(17)
    cfg = _mixer_config()
    mixer = M2RNNMixer(cfg)
    optimizer = optim.AdamW(learning_rate=1e-2, weight_decay=0.0)
    hidden = _rand((2, 6, cfg.d_model), np.random.default_rng(18))
    target = _rand((2, 6, cfg.d_model), np.random.default_rng(19))
    key_params = (
        "in_proj.weight",
        "g_norm.weight",
        "out_proj.weight",
        "state_weight",
        "A_log",
        "dt_bias",
        "D",
    )
    before = _flat_tree(mixer.parameters())

    def loss_fn(model: M2RNNMixer, x: mx.array, y: mx.array) -> mx.array:
        pred, _ = model(x)
        return mx.mean(mx.square(pred - y))

    loss_and_grad = nn.value_and_grad(mixer, loss_fn)
    loss, grads = loss_and_grad(mixer, hidden, target)
    mx.eval(loss, grads)

    assert math.isfinite(float(loss.item()))
    assert float(loss.item()) > 0

    flat_grads = _flat_tree(grads)
    for name in key_params:
        assert name in flat_grads
        assert np.isfinite(flat_grads[name]).all(), name
        assert float(np.max(np.abs(flat_grads[name]))) > 0.0, name

    optimizer.update(mixer, grads)
    mx.eval(mixer.parameters(), optimizer.state)
    after = _flat_tree(mixer.parameters())

    for name in key_params:
        assert float(np.max(np.abs(after[name] - before[name]))) > 0.0, name


@pytest.mark.parametrize(("seq", "chunk_size"), [(2, 4), (5, 2), (7, 3), (9, 4)])
def test_lightweight_mixer_train_step_updates_recurrence_across_lengths(
    seq: int,
    chunk_size: int,
) -> None:
    mx.random.seed(37 + seq)
    cfg = _mixer_config(chunk_size=chunk_size)
    mixer = M2RNNMixer(cfg)
    optimizer = optim.AdamW(learning_rate=1e-2, weight_decay=0.0)
    hidden = _rand((2, seq, cfg.d_model), np.random.default_rng(137 + seq))
    target = _rand((2, seq, cfg.d_model), np.random.default_rng(237 + seq))
    recurrence_params = ("state_weight", "A_log", "dt_bias", "D")
    before = _flat_tree(mixer.parameters())

    def loss_fn(model: M2RNNMixer, x: mx.array, y: mx.array) -> mx.array:
        pred, final_state = model(x)
        return mx.mean(mx.square(pred - y)) + 0.01 * mx.mean(mx.square(final_state))

    loss_and_grad = nn.value_and_grad(mixer, loss_fn)
    loss, grads = loss_and_grad(mixer, hidden, target)
    optimizer.update(mixer, grads)
    mx.eval(loss, grads, mixer.parameters(), optimizer.state)
    after = _flat_tree(mixer.parameters())

    assert math.isfinite(float(loss.item()))
    assert float(loss.item()) > 0.0

    flat_grads = _flat_tree(grads)
    for name in recurrence_params:
        assert name in flat_grads
        assert np.isfinite(flat_grads[name]).all(), name
        assert float(np.max(np.abs(flat_grads[name]))) > 0.0, name
        assert float(np.max(np.abs(after[name] - before[name]))) > 0.0, name


def test_lightweight_mixer_compiled_step_updates_state_weight_and_residual() -> None:
    mx.random.seed(29)
    cfg = _mixer_config()
    mixer = M2RNNMixer(cfg)
    optimizer = optim.AdamW(learning_rate=1e-2, weight_decay=0.0)
    hidden = _rand((2, 6, cfg.d_model), np.random.default_rng(30))
    target = _rand((2, 6, cfg.d_model), np.random.default_rng(31))
    key_params = ("g_norm.weight", "state_weight", "A_log", "dt_bias", "D")
    before = _flat_tree(mixer.parameters())

    def loss_fn(model: M2RNNMixer, x: mx.array, y: mx.array) -> mx.array:
        pred, _ = model(x)
        return mx.mean(mx.square(pred - y))

    loss_and_grad = nn.value_and_grad(mixer, loss_fn)
    captured_state = [mixer.state, optimizer.state, mx.random.state]

    @partial(mx.compile, inputs=captured_state, outputs=captured_state)
    def step(x: mx.array, y: mx.array):
        loss, grads = loss_and_grad(mixer, x, y)
        optimizer.update(mixer, grads)
        return loss, grads

    loss, grads = step(hidden, target)
    mx.eval(captured_state, loss, grads)
    after = _flat_tree(mixer.parameters())

    assert math.isfinite(float(loss.item()))
    assert float(loss.item()) > 0

    flat_grads = _flat_tree(grads)
    for name in key_params:
        assert name in flat_grads
        assert np.isfinite(flat_grads[name]).all(), name
        assert float(np.max(np.abs(flat_grads[name]))) > 0.0, name
        assert float(np.max(np.abs(after[name] - before[name]))) > 0.0, name


def test_invalid_mixer_config_values_fail_fast() -> None:
    invalid_cases = (
        ("d_model", lambda: _mixer_config(d_model=0)),
        ("k_head_dim", lambda: _mixer_config(k_head_dim=0)),
        ("v_head_dim", lambda: _mixer_config(v_head_dim=0)),
        ("num_q_heads", lambda: _mixer_config(num_q_heads=0)),
        ("num_k_heads", lambda: _mixer_config(num_k_heads=0)),
        ("num_v_heads", lambda: _mixer_config(num_v_heads=0)),
        ("num_f_heads", lambda: _mixer_config(num_f_heads=0)),
        ("num_g_heads", lambda: _mixer_config(num_g_heads=0)),
        ("num_weight_heads", lambda: _mixer_config(num_weight_heads=0)),
        ("chunk_size", lambda: _mixer_config(chunk_size=0)),
        ("A_init_min", lambda: _mixer_config(A_init_min=-1e-3)),
        ("A_init_max", lambda: _mixer_config(A_init_min=1.0, A_init_max=1.0)),
        ("dt_init_min", lambda: _mixer_config(dt_init_min=0.0)),
        ("dt_init_max", lambda: _mixer_config(dt_init_min=0.1, dt_init_max=0.01)),
        ("dt_init_floor", lambda: _mixer_config(dt_init_floor=0.0)),
        ("dt_init_max", lambda: _mixer_config(dt_init_floor=0.2, dt_init_max=0.1)),
    )

    for field, make_config in invalid_cases:
        with pytest.raises(ValueError, match=field):
            make_config()


def test_invalid_broadcast_head_config_fails_fast() -> None:
    with pytest.raises(ValueError, match="divide broadcast head count"):
        _mixer_config(num_q_heads=2, num_k_heads=3, num_v_heads=2, num_f_heads=2)


def test_invalid_output_gate_head_config_fails_fast() -> None:
    with pytest.raises(ValueError, match="num_g_heads"):
        _mixer_config(num_q_heads=4, num_k_heads=1, num_v_heads=4, num_f_heads=4, num_g_heads=3)


def test_mixer_rejects_explicit_zero_chunk_override() -> None:
    mx.random.seed(23)
    mixer = M2RNNMixer(_mixer_config())
    hidden = _rand((1, 2, 16), np.random.default_rng(23))

    with pytest.raises(ValueError, match="chunk_size"):
        mixer(hidden, chunk_size=0)
