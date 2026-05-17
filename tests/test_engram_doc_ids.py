"""Tests that ``HybridTinyBlock`` / ``HybridTinyLM`` thread ``document_ids``
into the Engram (``N`` symbol) block so that n-gram aggregation respects
packed-document boundaries.

Mirrors the kwarg convention used in nanochat
(``nanochat/engram.py::EngramBranch.forward(self, x, doc_ids=None)`` and
``nanochat/unified_superblock.py`` which calls ``self.engram(x_norm, doc_ids=doc_ids)``):
the parent stack carries the raw ``(B, S)`` int document IDs and only the
engram branch consumes them via the ``doc_ids`` kwarg.
"""

from __future__ import annotations

import mlx.core as mx
import pytest

from cppmega_mlx.models.hybrid_lm import (
    HybridTinyBlock,
    HybridTinyConfig,
    HybridTinyLM,
)
from cppmega_mlx.nn.engram import EngramBranch


def _engram_only_config() -> HybridTinyConfig:
    return HybridTinyConfig(
        vocab_size=16,
        hidden_size=8,
        pattern="N",
        depth=1,
        num_attention_heads=2,
        max_seq_length=8,
        moe_num_experts=2,
        moe_top_k=1,
        moe_expert_hidden_size=4,
        moe_shared_expert_hidden_size=None,
        m2rnn_k_head_dim=2,
        m2rnn_v_head_dim=2,
        engram_ngram_orders=(2, 3),
        engram_conv_kernel=0,
    )


def _make_engram_model(*, conv_kernel: int = 0) -> HybridTinyLM:
    """Build an engram-only model and randomize ``out_proj`` so the branch is
    not a degenerate identity / all-zero map at init (the upstream
    ``EngramBranch`` deliberately zero-inits ``out_proj`` so the residual
    branch is identity at step 0)."""

    cfg = HybridTinyConfig(
        vocab_size=16,
        hidden_size=8,
        pattern="N",
        depth=1,
        num_attention_heads=2,
        max_seq_length=8,
        moe_num_experts=2,
        moe_top_k=1,
        moe_expert_hidden_size=4,
        moe_shared_expert_hidden_size=None,
        m2rnn_k_head_dim=2,
        m2rnn_v_head_dim=2,
        engram_ngram_orders=(2, 3),
        engram_conv_kernel=conv_kernel,
    )
    model = HybridTinyLM(cfg)
    engram = model.layers[0].block
    assert isinstance(engram, EngramBranch)
    mx.random.seed(7)
    engram.out_proj.weight = mx.random.normal(
        engram.out_proj.weight.shape, dtype=engram.out_proj.weight.dtype
    ) * 0.3
    return model


# ---------------------------------------------------------------------------
# (1) doc_ids actually flow through the model into the engram block
# ---------------------------------------------------------------------------


def test_engram_doc_ids_change_hidden_states():
    model = _make_engram_model()
    mx.random.seed(0)
    input_ids = mx.random.randint(0, 16, shape=(1, 6))
    document_ids = mx.array([[0, 0, 0, 1, 1, 1]], dtype=mx.int32)

    out_no_docs = model.decoder_hidden_states(input_ids)
    out_with_docs = model.decoder_hidden_states(input_ids, document_ids=document_ids)
    mx.eval(out_no_docs, out_with_docs)

    # The two outputs must differ — the engram n-gram averages span the
    # 0->1 doc boundary in the unmasked case and get clipped in the masked
    # case. We check absolute diff at the positions that the order=2/3
    # average windows touch around the boundary (positions 3 and 4).
    diff = mx.max(mx.abs(out_no_docs - out_with_docs), axis=-1)
    mx.eval(diff)
    assert float(diff[0, 3].item()) > 1e-4, (
        "engram delta at position 3 should change when its order>=2 average "
        "stops pulling in the prior document's tokens"
    )


def test_engram_doc_ids_flow_through_conv_path():
    """Same as above but with the depthwise causal conv branch active so the
    other doc_ids-aware code path inside ``EngramBranch`` is exercised too."""

    model = _make_engram_model(conv_kernel=3)
    mx.random.seed(1)
    input_ids = mx.random.randint(0, 16, shape=(1, 6))
    document_ids = mx.array([[0, 0, 0, 1, 1, 1]], dtype=mx.int32)

    out_no_docs = model.decoder_hidden_states(input_ids)
    out_with_docs = model.decoder_hidden_states(input_ids, document_ids=document_ids)
    mx.eval(out_no_docs, out_with_docs)

    diff = mx.max(mx.abs(out_no_docs - out_with_docs))
    mx.eval(diff)
    assert float(diff.item()) > 1e-4


# ---------------------------------------------------------------------------
# (2) No engram in the stack -> document_ids has no effect on non-attention
#     hidden states (we use a pattern that contains no attention layer so
#     the attention additive-mask path is not exercised either).
# ---------------------------------------------------------------------------


def test_document_ids_noop_when_no_engram_no_attention():
    cfg = HybridTinyConfig(
        vocab_size=16,
        hidden_size=8,
        pattern="C",
        depth=1,
        num_attention_heads=2,
        max_seq_length=8,
        moe_num_experts=2,
        moe_top_k=1,
        moe_expert_hidden_size=4,
        moe_shared_expert_hidden_size=None,
        m2rnn_k_head_dim=2,
        m2rnn_v_head_dim=2,
        concept_num_concepts=4,
        concept_num_heads=2,
    )
    model = HybridTinyLM(cfg)
    assert tuple(layer.backend for layer in model.layers) == ("concept",)

    mx.random.seed(2)
    input_ids = mx.random.randint(0, 16, shape=(1, 6))
    document_ids = mx.array([[0, 0, 0, 1, 1, 1]], dtype=mx.int32)

    out_no_docs = model.decoder_hidden_states(input_ids)
    out_with_docs = model.decoder_hidden_states(input_ids, document_ids=document_ids)
    mx.eval(out_no_docs, out_with_docs)

    assert mx.allclose(out_no_docs, out_with_docs, atol=0.0, rtol=0.0).item()


# ---------------------------------------------------------------------------
# (3) Regression: calling the block / model without doc_ids still produces
#     the same shape and dtype it always did.
# ---------------------------------------------------------------------------


def test_engram_block_without_doc_ids_unchanged_shape_dtype():
    cfg = _engram_only_config()
    layer = cfg.expanded_pattern().layers[0]
    block = HybridTinyBlock(layer, cfg)
    assert block.backend == "engram"

    x = mx.random.normal((2, 6, cfg.hidden_size))
    delta_no_docs = block.route_delta(x, mask=None)
    delta_kwarg_none = block.route_delta(x, mask=None, doc_ids=None)
    mx.eval(delta_no_docs, delta_kwarg_none)

    assert delta_no_docs.shape == x.shape
    assert delta_no_docs.dtype == x.dtype
    # Explicit doc_ids=None must be byte-identical to omitting the kwarg.
    assert mx.array_equal(delta_no_docs, delta_kwarg_none).item()


def test_hybrid_lm_forward_without_document_ids_runs():
    model = _make_engram_model()
    input_ids = mx.array([[1, 2, 3, 4, 5, 6]], dtype=mx.int32)
    out = model(input_ids)
    mx.eval(out)
    assert out.shape == (1, 6, model.config.vocab_size)


# ---------------------------------------------------------------------------
# (4) Wrong-shape document_ids fails closed via _validate_document_ids
# ---------------------------------------------------------------------------


def test_document_ids_wrong_shape_raises():
    model = _make_engram_model()
    input_ids = mx.array([[1, 2, 3, 4, 5, 6]], dtype=mx.int32)
    # Mismatched seq length: input is (1, 6), document_ids is (1, 5)
    bad_doc_ids = mx.array([[0, 0, 0, 1, 1]], dtype=mx.int32)
    with pytest.raises(ValueError, match="document_ids shape"):
        model.decoder_hidden_states(input_ids, document_ids=bad_doc_ids)


def test_document_ids_wrong_rank_raises():
    model = _make_engram_model()
    input_ids = mx.array([[1, 2, 3, 4, 5, 6]], dtype=mx.int32)
    # Rank 1 instead of (B, S)
    bad_doc_ids = mx.array([0, 0, 0, 1, 1, 1], dtype=mx.int32)
    with pytest.raises(ValueError, match="document_ids must be shaped"):
        model.decoder_hidden_states(input_ids, document_ids=bad_doc_ids)


def test_document_ids_negative_raises():
    model = _make_engram_model()
    input_ids = mx.array([[1, 2, 3, 4, 5, 6]], dtype=mx.int32)
    bad_doc_ids = mx.array([[0, 0, -1, 1, 1, 1]], dtype=mx.int32)
    with pytest.raises(ValueError, match="non-negative"):
        model.decoder_hidden_states(input_ids, document_ids=bad_doc_ids)
