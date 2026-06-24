"""Tests for opt-in gradient checkpointing + chunked CE in DenseCppLM.

These guard the two activation-memory features that make 4x4096 bf16 feasible
without changing numerics (project RULE #1: the default path must be unchanged
and the opt-in paths must give the SAME loss/grad within fp tolerance):

  1. Per-block gradient checkpointing (``grad_checkpoint=True``) must yield a
     loss that is bit-identical and parameter gradients that match the
     non-checkpointed path (the whole point of checkpointing is recompute, not
     a different result).
  2. Chunked / streaming cross-entropy (``chunked_ce=True``) must give the same
     loss (within fp tolerance) and matching grads as the dense full-logits CE.

Everything runs on a tiny config so it is fast and CPU-friendly.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx.utils import tree_flatten

from cppmega_mlx.models.dense_cpp_lm import DenseCppLM, DenseCppLMConfig


def _tiny_cfg(**overrides) -> DenseCppLMConfig:
    base = dict(
        vocab_size=512,
        hidden_size=64,
        depth=3,
        ffn_hidden_size=128,
        max_seq_length=32,
        num_query_heads=4,
        num_kv_heads=2,
        head_dim=16,
        ngram_hash_enabled=False,
        structure_residual_scale=0.0,
        platform_residual_scale=0.0,
        ngram_residual_scale=0.0,
    )
    base.update(overrides)
    return DenseCppLMConfig(**base)


def _fresh_model(cfg: DenseCppLMConfig, seed: int = 0) -> DenseCppLM:
    mx.random.seed(seed)
    model = DenseCppLM(cfg)
    mx.eval(model.parameters())
    return model


def _batch(batch: int, seq: int, vocab: int, seed: int = 7):
    mx.random.seed(seed)
    input_ids = mx.random.randint(0, vocab, (batch, seq)).astype(mx.int32)
    targets = mx.random.randint(0, vocab, (batch, seq)).astype(mx.int32)
    loss_mask = mx.ones((batch, seq), dtype=mx.float32)
    return input_ids, targets, loss_mask


def _loss_and_grads(model, input_ids, targets, loss_mask):
    def lf(m):
        _, loss = m(input_ids, targets=targets, loss_mask=loss_mask)
        return loss

    return nn.value_and_grad(model, lf)(model)


def _max_grad_diff(g_a, g_b) -> float:
    fa = dict(tree_flatten(g_a))
    fb = dict(tree_flatten(g_b))
    assert set(fa) == set(fb), (set(fa) ^ set(fb))
    worst = 0.0
    for k in fa:
        worst = max(worst, float(mx.max(mx.abs(fa[k] - fb[k]))))
    return worst


def test_grad_checkpoint_loss_and_grads_match():
    """Checkpointed loss == non-checkpointed loss; grads match within tol."""
    input_ids, targets, loss_mask = _batch(2, 16, 512)

    base = _fresh_model(_tiny_cfg(grad_checkpoint=False), seed=0)
    ckpt = _fresh_model(_tiny_cfg(grad_checkpoint=True), seed=0)

    l0, g0 = _loss_and_grads(base, input_ids, targets, loss_mask)
    l1, g1 = _loss_and_grads(ckpt, input_ids, targets, loss_mask)
    mx.eval(l0, g0, l1, g1)

    # Same init (same seed) + same math => bit-identical loss.
    assert float(l0) == pytest.approx(float(l1), abs=1e-6, rel=1e-6)
    assert _max_grad_diff(g0, g1) < 1e-5


def test_chunked_ce_matches_full_ce_loss_and_grads():
    """Chunked-CE loss == full-CE loss; grads match within tol."""
    input_ids, targets, loss_mask = _batch(2, 16, 512)

    full = _fresh_model(_tiny_cfg(chunked_ce=False), seed=0)
    # Chunk size smaller than B*S=32 to actually exercise multiple chunks.
    chunk = _fresh_model(_tiny_cfg(chunked_ce=True, ce_chunk_size=10), seed=0)

    l0, g0 = _loss_and_grads(full, input_ids, targets, loss_mask)
    l1, g1 = _loss_and_grads(chunk, input_ids, targets, loss_mask)
    mx.eval(l0, g0, l1, g1)

    assert float(l0) == pytest.approx(float(l1), abs=1e-5, rel=1e-5)
    assert _max_grad_diff(g0, g1) < 1e-4


def test_chunked_ce_respects_loss_mask():
    """Chunked masked-mean CE == dense masked-mean CE with a partial mask."""
    input_ids, targets, _ = _batch(2, 16, 512)
    mask = mx.zeros((2, 16), dtype=mx.float32)
    mask[:, :8] = 1.0  # only first half of each row contributes

    full = _fresh_model(_tiny_cfg(chunked_ce=False), seed=3)
    chunk = _fresh_model(_tiny_cfg(chunked_ce=True, ce_chunk_size=7), seed=3)

    l0, g0 = _loss_and_grads(full, input_ids, targets, mask)
    l1, g1 = _loss_and_grads(chunk, input_ids, targets, mask)
    mx.eval(l0, g0, l1, g1)

    assert float(l0) == pytest.approx(float(l1), abs=1e-5, rel=1e-5)
    assert _max_grad_diff(g0, g1) < 1e-4


def test_both_opts_together_match_baseline():
    """grad_checkpoint + chunked_ce together == plain baseline (loss + grads)."""
    input_ids, targets, loss_mask = _batch(2, 16, 512)

    base = _fresh_model(_tiny_cfg(), seed=0)
    both = _fresh_model(
        _tiny_cfg(grad_checkpoint=True, chunked_ce=True, ce_chunk_size=9), seed=0
    )

    l0, g0 = _loss_and_grads(base, input_ids, targets, loss_mask)
    l1, g1 = _loss_and_grads(both, input_ids, targets, loss_mask)
    mx.eval(l0, g0, l1, g1)

    assert float(l0) == pytest.approx(float(l1), abs=1e-5, rel=1e-5)
    assert _max_grad_diff(g0, g1) < 1e-4


def test_default_path_unchanged_returns_logits():
    """Default config (no opts) still returns full logits alongside loss."""
    input_ids, targets, loss_mask = _batch(2, 16, 512)
    model = _fresh_model(_tiny_cfg(), seed=0)
    logits, loss = model(input_ids, targets=targets, loss_mask=loss_mask)
    mx.eval(logits, loss)
    assert logits is not None
    assert tuple(logits.shape) == (2, 16, 512)


def test_chunked_ce_returns_none_logits():
    """Chunked-CE deliberately does NOT materialize full logits."""
    input_ids, targets, loss_mask = _batch(2, 16, 512)
    model = _fresh_model(_tiny_cfg(chunked_ce=True, ce_chunk_size=8), seed=0)
    logits, loss = model(input_ids, targets=targets, loss_mask=loss_mask)
    mx.eval(loss)
    assert logits is None
    assert float(loss) > 0.0
