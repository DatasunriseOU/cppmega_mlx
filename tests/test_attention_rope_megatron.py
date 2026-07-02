from __future__ import annotations

import mlx.core as mx

from cppmega_mlx.nn.attention import apply_rotary_emb


def test_apply_rotary_emb_matches_megatron_non_interleaved_sign():
    # x = [x1, x2], Megatron rotate_half(x) = [-x2, x1].
    x = mx.array([[[[1.0, 2.0, 10.0, 20.0]]]], dtype=mx.float32)
    cos = mx.array([[[[0.5, 0.25]]]], dtype=mx.float32)
    sin = mx.array([[[[0.1, 0.2]]]], dtype=mx.float32)

    y = apply_rotary_emb(x, cos, sin)
    expected = mx.array(
        [[[[1.0 * 0.5 - 10.0 * 0.1, 2.0 * 0.25 - 20.0 * 0.2,
            10.0 * 0.5 + 1.0 * 0.1, 20.0 * 0.25 + 2.0 * 0.2]]]],
        dtype=mx.float32,
    )

    assert mx.allclose(y, expected)
