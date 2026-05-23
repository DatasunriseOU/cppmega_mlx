"""V7-N05: reference KL parity for the DSA split-K indexer-loss kernel."""

from __future__ import annotations

import math

import mlx.core as mx

from cppmega_mlx.nn.dsa_indexer_loss_reference import (
    reference_indexer_kl_loss, EPS,
)


def test_kl_zero_when_q_equals_p():
    B, Sq, AH, AD, Sk = 1, 4, 2, 8, 4
    Q = mx.random.normal(shape=(B, Sq, AH, AD), key=mx.random.key(0))
    K = mx.random.normal(shape=(B, Sk, AH, AD), key=mx.random.key(1))
    # Construct IndexScores such that softmax matches the attention
    # softmax: we re-use the attention logits sum-over-heads / AH as the
    # IndexScores. KL should then collapse to 0.
    softmax_scale = 1.0 / math.sqrt(AD)
    logits = mx.einsum("bshd,bthd->bhst",
                       Q.astype(mx.float32),
                       K.astype(mx.float32)) * softmax_scale
    # mean over heads in log-space ≠ mean of softmaxes, so we go the
    # other way: derive IndexScores so that softmax(IndexScores) ==
    # mean over heads softmax(logits).
    head_sm = mx.softmax(logits, axis=-1)
    mean_p = mx.mean(head_sm, axis=1)  # (B, Sq, Sk)
    # log gives back the logits up to a constant per (b, sq).
    IndexScores = mx.log(mean_p + EPS)

    _, scalar = reference_indexer_kl_loss(
        Q, K, IndexScores,
        softmax_scale=softmax_scale, loss_coeff=1.0, causal=False,
    )
    assert float(scalar.item()) < 1e-4


def test_kl_positive_for_random_index_scores():
    B, Sq, AH, AD, Sk = 1, 4, 2, 8, 4
    Q = mx.random.normal(shape=(B, Sq, AH, AD), key=mx.random.key(7))
    K = mx.random.normal(shape=(B, Sk, AH, AD), key=mx.random.key(8))
    IndexScores = mx.random.normal(shape=(B, Sq, Sk),
                                     key=mx.random.key(9))
    per_pos, scalar = reference_indexer_kl_loss(
        Q, K, IndexScores,
        softmax_scale=1.0 / math.sqrt(AD), loss_coeff=2.5,
        causal=True,
    )
    assert per_pos.shape == (B, Sq)
    s = float(scalar.item())
    # KL is non-negative; with random scores it must be > 0.
    assert s > 0.0
    # loss_coeff scales the scalar — without it the same call returns
    # s / 2.5.
    _, scalar1 = reference_indexer_kl_loss(
        Q, K, IndexScores,
        softmax_scale=1.0 / math.sqrt(AD), loss_coeff=1.0, causal=True,
    )
    assert abs(s - float(scalar1.item()) * 2.5) < 1e-4


def test_kl_loss_coeff_zero_returns_zero():
    B, Sq, AH, AD, Sk = 1, 2, 2, 4, 2
    Q = mx.random.normal(shape=(B, Sq, AH, AD), key=mx.random.key(0))
    K = mx.random.normal(shape=(B, Sk, AH, AD), key=mx.random.key(1))
    IndexScores = mx.random.normal(shape=(B, Sq, Sk),
                                     key=mx.random.key(2))
    _, scalar = reference_indexer_kl_loss(
        Q, K, IndexScores,
        softmax_scale=1.0, loss_coeff=0.0, causal=False,
    )
    assert float(scalar.item()) == 0.0
