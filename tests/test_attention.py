from __future__ import annotations

import numpy as np
import pytest

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from cppmega_mlx.nn.attention import AttentionConfig, CausalSelfAttention


def _rand(shape: tuple[int, ...], seed: int = 0) -> mx.array:
    rng = np.random.default_rng(seed)
    return mx.array(rng.standard_normal(shape, dtype=np.float32))


def _tree_arrays(tree):
    if isinstance(tree, dict):
        for value in tree.values():
            yield from _tree_arrays(value)
    elif isinstance(tree, (list, tuple)):
        for value in tree:
            yield from _tree_arrays(value)
    else:
        yield tree


def test_attention_config_preserves_mla_and_dsa_modes() -> None:
    mla = AttentionConfig(d_model=16, num_q_heads=4, num_kv_heads=2, mode="mla")
    dsa = AttentionConfig(d_model=16, num_q_heads=4, num_kv_heads=2, mode="dsa")

    assert mla.mode == "mla"
    assert dsa.mode == "dsa"
    assert mla.is_gqa
    assert dsa.q_head_dim == 4


def test_attention_config_validation_fails_closed() -> None:
    with pytest.raises(ValueError, match="mode"):
        AttentionConfig(d_model=16, num_q_heads=4, mode="dense")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="divisible"):
        AttentionConfig(d_model=16, num_q_heads=4, num_kv_heads=3)


@pytest.mark.parametrize("mode", ["mla", "dsa"])
def test_causal_attention_output_shape_and_route_marker(mode: str) -> None:
    cfg = AttentionConfig(d_model=16, num_q_heads=4, num_kv_heads=2, mode=mode)  # type: ignore[arg-type]
    attn = CausalSelfAttention(cfg)
    x = _rand((2, 5, 16), seed=5)

    out = attn(x)
    mx.eval(out)

    assert out.shape == x.shape
    assert attn.route_info.mode == mode
    assert attn.route_info.backend == "mlx.fast.sdpa"
    assert not attn.route_info.sparse_reference


def test_causal_prefix_invariance_with_gqa() -> None:
    cfg = AttentionConfig(d_model=16, num_q_heads=4, num_kv_heads=2, mode="mla")
    attn = CausalSelfAttention(cfg)
    prefix = _rand((1, 4, 16), seed=1)
    suffix_a = _rand((1, 3, 16), seed=2)
    suffix_b = _rand((1, 3, 16), seed=3)

    out_a = attn(mx.concatenate([prefix, suffix_a], axis=1))
    out_b = attn(mx.concatenate([prefix, suffix_b], axis=1))
    mx.eval(out_a, out_b)

    np.testing.assert_allclose(
        np.array(out_a[:, : prefix.shape[1], :]),
        np.array(out_b[:, : prefix.shape[1], :]),
        atol=1e-5,
        rtol=1e-5,
    )


def test_attention_train_step_has_finite_loss_and_gradients() -> None:
    cfg = AttentionConfig(d_model=16, num_q_heads=4, num_kv_heads=2, mode="dsa")
    attn = CausalSelfAttention(cfg)
    opt = optim.Adam(learning_rate=1e-3)
    x = _rand((2, 6, 16), seed=6)

    def loss_fn(model: CausalSelfAttention, hidden: mx.array) -> mx.array:
        return mx.mean(model(hidden) ** 2)

    loss, grads = nn.value_and_grad(attn, loss_fn)(attn, x)
    opt.update(attn, grads)
    mx.eval(loss, grads, attn.parameters())

    assert np.isfinite(np.array(loss)).all()
    grad_arrays = list(_tree_arrays(grads))
    assert grad_arrays
    for grad in grad_arrays:
        assert np.isfinite(np.array(grad)).all()
