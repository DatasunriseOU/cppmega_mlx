from __future__ import annotations

import numpy as np
import pytest

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from cppmega_mlx.inference.engine import ContiguousKVCacheConfig, make_contiguous_kv_cache
from cppmega_mlx.nn.attention import AttentionConfig, CausalSelfAttention, causal_sdpa_mask


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
    with pytest.raises(ValueError, match="sliding_window"):
        AttentionConfig(d_model=16, num_q_heads=4, sliding_window=0)


def test_causal_sdpa_mask_is_boolean_and_sliding_windowed() -> None:
    mask = causal_sdpa_mask(5, sliding_window=2, expand_heads=True)
    mx.eval(mask)

    assert mask.shape == (1, 1, 5, 5)
    assert mask.dtype == mx.bool_
    np.testing.assert_array_equal(
        np.array(mask[0, 0]),
        np.array(
            [
                [True, False, False, False, False],
                [True, True, False, False, False],
                [False, True, True, False, False],
                [False, False, True, True, False],
                [False, False, False, True, True],
            ],
            dtype=np.bool_,
        ),
    )


def test_causal_sdpa_mask_rejects_invalid_lengths() -> None:
    with pytest.raises(ValueError, match="seq_length"):
        causal_sdpa_mask(0)
    with pytest.raises(ValueError, match="key_length"):
        causal_sdpa_mask(4, key_length=0)
    with pytest.raises(ValueError, match="query_offset"):
        causal_sdpa_mask(4, query_offset=-1)
    with pytest.raises(ValueError, match="sliding_window"):
        causal_sdpa_mask(4, sliding_window=-1)


def test_causal_sdpa_mask_supports_cached_decode_offsets() -> None:
    mask = causal_sdpa_mask(
        2,
        query_offset=3,
        key_length=5,
        sliding_window=3,
        expand_heads=True,
    )
    mx.eval(mask)

    assert mask.shape == (1, 1, 2, 5)
    np.testing.assert_array_equal(
        np.array(mask[0, 0]),
        np.array(
            [
                [False, True, True, True, False],
                [False, False, True, True, True],
            ],
            dtype=np.bool_,
        ),
    )


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


def test_causal_mask_sentinel_matches_default_mask() -> None:
    cfg = AttentionConfig(d_model=16, num_q_heads=4, num_kv_heads=2, mode="mla")
    attn = CausalSelfAttention(cfg)
    x = _rand((1, 5, 16), seed=12)

    default_out = attn(x)
    sentinel_out = attn(x, mask="causal")
    mx.eval(default_out, sentinel_out)

    np.testing.assert_allclose(
        np.array(sentinel_out),
        np.array(default_out),
        atol=1e-6,
        rtol=1e-6,
    )


def test_explicit_boolean_mask_is_forwarded_without_string_comparison() -> None:
    cfg = AttentionConfig(d_model=16, num_q_heads=4, num_kv_heads=2, mode="mla")
    attn = CausalSelfAttention(cfg)
    x = _rand((1, 5, 16), seed=13)
    explicit_mask = causal_sdpa_mask(5)

    explicit_out = attn(x, mask=explicit_mask)
    default_out = attn(x)
    mx.eval(explicit_out, default_out)

    np.testing.assert_allclose(
        np.array(explicit_out),
        np.array(default_out),
        atol=1e-6,
        rtol=1e-6,
    )


def test_sliding_window_attention_does_not_see_old_prefix_tokens() -> None:
    cfg = AttentionConfig(
        d_model=16,
        num_q_heads=4,
        num_kv_heads=2,
        mode="mla",
        sliding_window=3,
    )
    attn = CausalSelfAttention(cfg)
    prefix_a = _rand((1, 2, 16), seed=7)
    prefix_b = _rand((1, 2, 16), seed=8)
    local_tail = _rand((1, 3, 16), seed=9)

    out_a = attn(mx.concatenate([prefix_a, local_tail], axis=1))
    out_b = attn(mx.concatenate([prefix_b, local_tail], axis=1))
    mx.eval(out_a, out_b)

    np.testing.assert_allclose(
        np.array(out_a[:, -1, :]),
        np.array(out_b[:, -1, :]),
        atol=1e-5,
        rtol=1e-5,
    )


def test_causal_attention_contiguous_kv_cache_matches_full_prefix_last_token() -> None:
    cfg = AttentionConfig(d_model=16, num_q_heads=4, num_kv_heads=2, mode="mla")
    attn = CausalSelfAttention(cfg)
    prefix = _rand((1, 4, 16), seed=21)
    next_token = _rand((1, 1, 16), seed=22)
    full = attn(mx.concatenate([prefix, next_token], axis=1))

    cache = make_contiguous_kv_cache(
        ContiguousKVCacheConfig(
            num_layers=1,
            batch_size=1,
            num_kv_heads=2,
            head_dim=4,
            max_seq_len=8,
        )
    )
    _ = attn(prefix, kv_cache=cache, layer_idx=0)
    cached_next = attn(next_token, kv_cache=cache, layer_idx=0)
    mx.eval(full, cached_next)

    assert cache.position() == 5
    np.testing.assert_allclose(
        np.array(cached_next[:, -1, :]),
        np.array(full[:, -1, :]),
        atol=1e-5,
        rtol=1e-5,
    )


def test_causal_attention_kv_cache_requires_layer_idx() -> None:
    attn = CausalSelfAttention(AttentionConfig(d_model=16, num_q_heads=4, num_kv_heads=2))
    cache = make_contiguous_kv_cache(
        num_layers=1,
        batch_size=1,
        num_kv_heads=2,
        head_dim=4,
    )

    with pytest.raises(ValueError, match="layer_idx"):
        attn(_rand((1, 2, 16), seed=23), kv_cache=cache)


def test_causal_attention_rejects_quantized_kv_cache_until_sdpa_path_lands() -> None:
    attn = CausalSelfAttention(AttentionConfig(d_model=16, num_q_heads=4, num_kv_heads=2))
    cache = make_contiguous_kv_cache(
        num_layers=1,
        batch_size=1,
        num_kv_heads=2,
        head_dim=64,
        quantized=True,
        kv_group_size=64,
    )

    with pytest.raises(NotImplementedError, match="quantized KV cache"):
        attn(_rand((1, 2, 16), seed=24), kv_cache=cache, layer_idx=0)


def test_attention_sinks_are_forwarded_to_fast_sdpa() -> None:
    cfg = AttentionConfig(d_model=16, num_q_heads=4, num_kv_heads=2, mode="mla")
    attn = CausalSelfAttention(cfg)
    x = _rand((1, 4, 16), seed=10)
    sinks = mx.array([0.0, 0.25, -0.125, 0.5], dtype=mx.float32)

    out = attn(x, sinks=sinks)
    mx.eval(out)

    assert out.shape == x.shape
    assert np.isfinite(np.array(out)).all()


def test_attention_sink_shape_validation_fails_closed() -> None:
    cfg = AttentionConfig(d_model=16, num_q_heads=4, num_kv_heads=2, mode="mla")
    attn = CausalSelfAttention(cfg)
    x = _rand((1, 4, 16), seed=11)

    with pytest.raises(ValueError, match="one value per query head"):
        attn(x, sinks=mx.zeros((2,), dtype=mx.float32))
    with pytest.raises(ValueError, match="1D"):
        attn(x, sinks=mx.zeros((1, 4), dtype=mx.float32))


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
