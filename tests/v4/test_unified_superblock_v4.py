"""End-to-end tests: UnifiedSuperblockV4 + EngramV4Block doc_ids propagation."""

import mlx.core as mx
import numpy as np
import pytest

from cppmega_v4.models.unified_superblock_v4 import UnifiedSuperblockV4
from cppmega_v4.nn.engram_v4 import EngramV4Block, EngramV4Config
from cppmega_v4.run_template import BlockSpec, MTPSpec, RunTemplate


# ----- EngramV4Block standalone -----


def test_engram_v4_forward_shape():
    cfg = EngramV4Config(
        hidden_size=64, num_ngram_layers=2, max_ngram_size=3,
        num_embed_table_per_ngram=2, embed_dim=16, embed_table_size=64,
    )
    block = EngramV4Block(cfg)
    B, S = 1, 8
    x = mx.random.normal((B, S, cfg.hidden_size))
    tok = mx.array(np.random.randint(0, 100, (B, S)).astype(np.int32))
    delta = block(x, tok)
    assert delta.shape == (B, S, cfg.hidden_size)
    assert not bool(mx.any(mx.isnan(delta)).item())


def test_engram_v4_accepts_document_ids():
    cfg = EngramV4Config(
        hidden_size=32, num_ngram_layers=1, max_ngram_size=3,
        num_embed_table_per_ngram=2, embed_dim=8, embed_table_size=32,
    )
    block = EngramV4Block(cfg)
    B, S = 1, 6
    x = mx.random.normal((B, S, cfg.hidden_size))
    tok = mx.array(np.random.randint(0, 50, (B, S)).astype(np.int32))
    docs = mx.array([[0, 0, 0, 1, 1, 1]], dtype=mx.int32)
    delta = block(x, tok, document_ids=docs)
    assert delta.shape == (B, S, cfg.hidden_size)


def test_engram_v4_doc_ids_affect_output():
    """Output with doc-boundary at position 3 must differ from same input
    treated as one document."""
    cfg = EngramV4Config(
        hidden_size=32, num_ngram_layers=1, max_ngram_size=4,
        num_embed_table_per_ngram=2, embed_dim=8, embed_table_size=32,
    )
    block = EngramV4Block(cfg)
    B, S = 1, 8
    x = mx.random.normal((B, S, cfg.hidden_size))
    tok = mx.array(np.random.randint(1, 50, (B, S)).astype(np.int32))
    docs_two = mx.array([[0, 0, 0, 0, 1, 1, 1, 1]], dtype=mx.int32)
    docs_one = mx.array([[0] * S], dtype=mx.int32)
    delta_two = np.array(block(x, tok, document_ids=docs_two))
    delta_one = np.array(block(x, tok, document_ids=docs_one))
    # The n-gram window spans positions 5-7. With a doc-boundary at 4, the
    # window for position 5 cannot see positions 0-3; with one doc, it can.
    # → outputs at position 5 must differ.
    assert not np.allclose(delta_two[0, 5], delta_one[0, 5], atol=1e-6), (
        "doc_ids must affect the n-gram window at boundary-crossing positions"
    )


# ----- UnifiedSuperblockV4 -----


def _tiny_template_with_engram() -> RunTemplate:
    return RunTemplate(
        name="v4_tiny_engram",
        hidden_size=32,
        vocab_size=128,
        blocks=[
            BlockSpec(kind="mlp", repeat=1,
                      params={"intermediate_size": 64}),
            BlockSpec(kind="engram", repeat=1, params={
                "num_ngram_layers": 1, "max_ngram_size": 3,
                "num_embed_table_per_ngram": 2, "embed_dim": 8,
                "embed_table_size": 32,
            }),
        ],
    )


def test_unified_superblock_builds_from_template():
    t = _tiny_template_with_engram()
    sb = UnifiedSuperblockV4(t)
    assert len(sb.blocks) == 2
    assert sb.kinds() == ["mlp", "engram"]


def test_unified_superblock_forward_shape():
    t = _tiny_template_with_engram()
    sb = UnifiedSuperblockV4(t)
    B, S = 1, 6
    tok = mx.array(np.random.randint(1, 50, (B, S)).astype(np.int32))
    h = mx.random.normal((B, S, t.hidden_size))
    out = sb(tok, h)
    assert out.shape == (B, S, t.hidden_size)
    assert not bool(mx.any(mx.isnan(out)).item())


def test_unified_superblock_threads_document_ids_to_engram():
    """Output with non-trivial doc_ids must differ from same input with no doc_ids."""
    t = _tiny_template_with_engram()
    sb = UnifiedSuperblockV4(t)
    B, S = 1, 8
    tok = mx.array(np.random.randint(1, 50, (B, S)).astype(np.int32))
    h = mx.random.normal((B, S, t.hidden_size))
    docs = mx.array([[0, 0, 0, 0, 1, 1, 1, 1]], dtype=mx.int32)
    out_with = np.array(sb(tok, h, document_ids=docs))
    out_no = np.array(sb(tok, h, document_ids=None))
    # Outputs at position 5 (which would cross the doc boundary) must differ.
    assert not np.allclose(out_with[0, 5], out_no[0, 5], atol=1e-6)


def test_unified_superblock_repeat_count():
    t = RunTemplate(
        name="r2",
        hidden_size=16,
        blocks=[BlockSpec(kind="mlp", repeat=3, params={})],
    )
    sb = UnifiedSuperblockV4(t)
    assert len(sb.blocks) == 3
    assert sb.kinds() == ["mlp", "mlp", "mlp"]


def test_unified_superblock_rejects_doc_id_shape_mismatch():
    t = _tiny_template_with_engram()
    sb = UnifiedSuperblockV4(t)
    tok = mx.zeros((1, 4), dtype=mx.int32)
    h = mx.zeros((1, 4, t.hidden_size))
    bad_docs = mx.zeros((1, 5), dtype=mx.int32)
    with pytest.raises(ValueError, match="document_ids"):
        sb(tok, h, document_ids=bad_docs)


def test_unified_superblock_with_nsa_and_csa_hca_runs_end_to_end():
    """Mixed stack: NSA + Engram + CSA+HCA + MLP — all should compose."""
    t = RunTemplate(
        name="mixed",
        hidden_size=64,
        blocks=[
            BlockSpec(kind="nsa", repeat=1, params={
                "num_heads": 4, "head_dim": 16,
                "compress_block_size": 4, "select_topk": 2,
                "sliding_window": 4,
            }),
            BlockSpec(kind="engram", repeat=1, params={
                "num_ngram_layers": 1, "max_ngram_size": 3,
                "num_embed_table_per_ngram": 2, "embed_dim": 8,
                "embed_table_size": 32,
            }),
            BlockSpec(kind="csa_hca", repeat=1, params={
                "num_heads": 4, "head_dim": 16,
                "m_csa": 2, "m_hca": 4,
            }),
            BlockSpec(kind="mlp", repeat=1, params={}),
        ],
    )
    sb = UnifiedSuperblockV4(t)
    B, S = 1, 8
    tok = mx.array(np.random.randint(1, 50, (B, S)).astype(np.int32))
    h = mx.random.normal((B, S, t.hidden_size))
    docs = mx.array([[0] * 4 + [1] * 4], dtype=mx.int32)
    out = sb(tok, h, document_ids=docs)
    assert out.shape == (B, S, t.hidden_size)
    assert not bool(mx.any(mx.isnan(out)).item())
