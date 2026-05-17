"""ROI 5 — FlashMLA absorb trick parity tests."""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from cppmega_v4.nn.mla_absorb import absorb_weights, absorbed_mla_decode, standard_mla_decode


def _rand(*shape, seed=0):
    rng = np.random.default_rng(seed)
    return mx.array(rng.standard_normal(shape).astype(np.float32))


def test_absorb_weights_shapes():
    h, d_kv, d_k, d_v, d_model = 2, 8, 4, 6, 12
    w_uk = _rand(h, d_kv, d_k, seed=1)
    w_uv = _rand(h, d_kv, d_v, seed=2)
    w_o = _rand(h * d_v, d_model, seed=3)
    w_uk_abs, w_uv_w_o = absorb_weights(w_uk, w_uv, w_o)
    assert w_uk_abs.shape == (h, d_k, d_kv)
    assert w_uv_w_o.shape == (h, d_kv, d_model)


def test_absorb_weights_rejects_bad_shapes():
    with pytest.raises(ValueError):
        absorb_weights(mx.zeros((2, 8)), mx.zeros((2, 8, 6)), mx.zeros((12, 16)))
    with pytest.raises(ValueError):
        absorb_weights(mx.zeros((2, 8, 4)), mx.zeros((3, 8, 6)), mx.zeros((12, 16)))
    with pytest.raises(ValueError):
        absorb_weights(mx.zeros((2, 8, 4)), mx.zeros((2, 8, 6)), mx.zeros((10, 16)))


def test_absorbed_equals_standard_decode():
    """Numerical parity: absorbed and standard MLA decode produce identical output."""
    b, t, h, d_k, d_kv, d_v, d_model, t_kv = 1, 2, 2, 4, 8, 6, 12, 5
    q = _rand(b, t, h, d_k, seed=10)
    c_kv = _rand(b, t_kv, d_kv, seed=11)
    w_uk = _rand(h, d_kv, d_k, seed=12)
    w_uv = _rand(h, d_kv, d_v, seed=13)
    w_o = _rand(h * d_v, d_model, seed=14)

    w_uk_abs, w_uv_w_o = absorb_weights(w_uk, w_uv, w_o)
    out_abs = absorbed_mla_decode(q, c_kv, w_uk_abs, w_uv_w_o)
    out_std = standard_mla_decode(q, c_kv, w_uk, w_uv, w_o)
    np.testing.assert_allclose(np.array(out_abs), np.array(out_std), atol=1e-4, rtol=1e-4)


def test_absorbed_decode_with_mask():
    b, t, h, d_k, d_kv, d_v, d_model, t_kv = 1, 1, 2, 4, 6, 4, 8, 4
    q = _rand(b, t, h, d_k, seed=20)
    c_kv = _rand(b, t_kv, d_kv, seed=21)
    w_uk = _rand(h, d_kv, d_k, seed=22)
    w_uv = _rand(h, d_kv, d_v, seed=23)
    w_o = _rand(h * d_v, d_model, seed=24)
    w_uk_abs, w_uv_w_o = absorb_weights(w_uk, w_uv, w_o)
    # Mask out last KV position.
    mask = mx.zeros((b, t, h, t_kv))
    mask = mx.concatenate([mask[..., :-1], mx.full((b, t, h, 1), -1e9)], axis=-1)
    out = absorbed_mla_decode(q, c_kv, w_uk_abs, w_uv_w_o, mask=mask)
    assert out.shape == (b, t, d_model)
    assert not bool(mx.any(mx.isnan(out)).item())


def test_absorbed_decode_rejects_bad_shapes():
    h, d_k, d_kv, d_model = 2, 4, 8, 12
    q = _rand(1, 1, h, d_k)
    c_kv = _rand(1, 3, d_kv)
    w_uk_abs = _rand(h, d_k, d_kv)
    w_uv_w_o = _rand(h, d_kv, d_model)
    # Bad q rank.
    with pytest.raises(ValueError):
        absorbed_mla_decode(mx.zeros((1, 1, d_k)), c_kv, w_uk_abs, w_uv_w_o)
    # Bad w_uk_abs shape.
    with pytest.raises(ValueError):
        absorbed_mla_decode(q, c_kv, _rand(h, d_k + 1, d_kv), w_uv_w_o)
