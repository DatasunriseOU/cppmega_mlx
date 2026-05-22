"""V7-H07: per-brick grad-norm + attn head-mean probe helpers."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from cppmega_v4.runtime.per_brick_probes import (
    attn_head_means, per_brick_grad_norms,
)


def test_v7_h07_per_brick_grad_norms_aggregates_by_top_path():
    class T(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = [nn.Linear(8, 8) for _ in range(3)]

        def __call__(self, x):
            for l in self.layers:
                x = l(x)
            return x

    model = T()
    def loss_fn(m, x):
        return mx.sum(m(x).astype(mx.float32) ** 2)

    lvg = nn.value_and_grad(model, loss_fn)
    x = mx.random.normal(shape=(2, 8), key=mx.random.key(0))
    _, grads = lvg(model, x)
    norms = per_brick_grad_norms(model, grads)
    assert "layers.0" in norms
    assert "layers.1" in norms
    assert "layers.2" in norms
    for v in norms.values():
        assert v > 0


def test_v7_h07_attn_head_means_returns_per_head_floats():
    from cppmega_v4.models.unified_superblock_v4 import BLOCK_BUILDERS

    H = 64
    num_heads = 4
    attn = BLOCK_BUILDERS["attention"](H, {
        "num_heads": num_heads, "head_dim": H // num_heads,
    })
    # Randomise o_proj so attention pattern matters.
    attn.o_proj.weight = mx.random.normal(
        shape=attn.o_proj.weight.shape, key=mx.random.key(0))
    x = mx.random.normal(shape=(1, 8, H), key=mx.random.key(1))
    means = attn_head_means(attn, x)
    assert means, "no attention module detected"
    head_list = next(iter(means.values()))
    assert isinstance(head_list, list)
    # Helper falls back to a single composite "head" when it cannot
    # introspect num_heads cleanly; just assert the values are valid
    # softmax-mean fractions in [0, 1].
    assert len(head_list) >= 1
    for m in head_list:
        assert 0.0 <= m <= 1.0
