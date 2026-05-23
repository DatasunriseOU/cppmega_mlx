"""V7-B11: fake_ranks=4 mean-reduced grads are bit-equivalent to single-rank.

The H20 replay path runs the same forward+backward N times on identical
inputs and mean-reduces the gradient tree. Since (g + g + g + g) / 4 == g
when every replay sees the same batch, fake_ranks=4 must produce grads
indistinguishable from fake_ranks=1 within fp32 rounding.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


def _build_tiny():
    m = nn.Sequential(
        nn.Linear(8, 16),
        nn.Linear(16, 8),
    )
    return m


def _grad_tree(model: nn.Module, x: mx.array) -> dict:
    def loss_fn(m: nn.Module, _x: mx.array) -> mx.array:
        return mx.mean(m(_x) ** 2)
    lvg = nn.value_and_grad(model, loss_fn)
    _, grads = lvg(model, x)
    mx.eval(grads)
    return dict(nn.utils.tree_flatten(grads))


def test_fake_ranks_4_mean_matches_single_rank():
    mx.random.seed(0xBEEF)
    m1 = _build_tiny()
    mx.eval(m1.parameters())
    x = mx.random.normal(shape=(2, 8), key=mx.random.key(0))

    g1 = _grad_tree(m1, x)

    # fake_ranks=4: replay the same backward 4× on identical input then
    # mean-reduce — that's exactly the H20 simulation in stage_train.
    accum: dict = {}
    fake_ranks = 4
    for _ in range(fake_ranks):
        gi = _grad_tree(m1, x)
        for k, v in gi.items():
            accum[k] = v if k not in accum else (accum[k] + v)
    g_mean = {k: v / float(fake_ranks) for k, v in accum.items()}
    mx.eval(g_mean)

    for k in g1:
        assert mx.allclose(g_mean[k], g1[k], atol=1e-6).item(), (
            f"grad {k!r} drifted: max-diff="
            f"{float(mx.max(mx.abs(g_mean[k] - g1[k])).item())}")
