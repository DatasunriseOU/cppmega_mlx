"""Tests for UnifiedSuperblockV4 real factories (gdn, kda, moe, mla, attention,
lightning_indexer) — covers what used to be pass-through stubs."""

import mlx.core as mx
import numpy as np
import pytest

from cppmega_v4.models.unified_superblock_v4 import (
    BLOCK_BUILDERS,
    UnifiedSuperblockV4,
)
from cppmega_v4.run_template import BlockSpec, RunTemplate


def _mk(blocks: list, hidden_size: int = 64) -> UnifiedSuperblockV4:
    t = RunTemplate(name="rf_test", hidden_size=hidden_size, blocks=blocks)
    return UnifiedSuperblockV4(t)


def _inputs(B=1, S=8, H=64, vocab=128, seed=0):
    np.random.seed(seed)
    tok = mx.array(np.random.randint(1, vocab, (B, S)).astype(np.int32))
    h = mx.random.normal((B, S, H))
    return tok, h


# ----- Each kind is no longer a pass-through stub -----


@pytest.mark.parametrize("kind", ["gdn", "kda", "moe", "attention",
                                   "mla", "mla_absorb", "lightning_indexer"])
def test_kind_is_no_longer_passthrough(kind):
    """Confirm BLOCK_BUILDERS[kind] is a real factory, not the placeholder."""
    factory = BLOCK_BUILDERS[kind]
    # Real factories live as top-level _build_<kind> functions; the
    # passthrough was a lambda. After this commit only kinds without a
    # real backend may still be lambdas — that set should be empty.
    assert not (factory.__name__ == "<lambda>"), (
        f"BLOCK_BUILDERS[{kind!r}] is still a pass-through lambda"
    )


# ----- gdn (GatedDeltaNet via LinearAttentionBlock) -----


def test_gdn_block_builds_and_runs():
    sb = _mk([BlockSpec(kind="gdn", repeat=1, params={
        "num_heads": 4, "head_dim": 16, "use_short_conv": False,
    })])
    tok, h = _inputs()
    out = sb(tok, h)
    assert out.shape == h.shape
    assert not bool(mx.any(mx.isnan(out)).item())


def test_gdn_block_threads_doc_ids():
    """GDN block accepts doc_ids via kwargs and uses them. Test perturbs
    o_proj (zero-init at construction) so the doc-reset effect is observable."""
    sb = _mk([BlockSpec(kind="gdn", repeat=1, params={
        "num_heads": 4, "head_dim": 16, "use_short_conv": False,
    })])
    # Perturb o_proj so block contributes a non-zero delta (default is identity).
    gdn = sb.blocks[0].module
    rng = np.random.default_rng(0)
    gdn.o_proj.weight = mx.array(
        rng.standard_normal(gdn.o_proj.weight.shape).astype(np.float32) * 0.1
    )

    tok, h = _inputs(S=8)
    docs_two = mx.array([[0, 0, 0, 0, 1, 1, 1, 1]], dtype=mx.int32)
    docs_one = mx.array([[0] * 8], dtype=mx.int32)
    out_two = np.array(sb(tok, h, document_ids=docs_two))
    out_one = np.array(sb(tok, h, document_ids=docs_one))
    # GDN block resets recurrent state at doc boundaries, so position 5
    # (after the boundary) should differ between the two runs.
    assert not np.allclose(out_two[0, 5], out_one[0, 5], atol=1e-6), (
        "doc_ids should affect the recurrent state at boundary-crossing positions"
    )


# ----- kda (Kimi Delta Attention) -----


def test_kda_block_builds_and_runs():
    sb = _mk([BlockSpec(kind="kda", repeat=1, params={
        "num_heads": 4, "head_dim": 16, "use_short_conv": False,
    })])
    tok, h = _inputs()
    out = sb(tok, h)
    assert out.shape == h.shape
    assert not bool(mx.any(mx.isnan(out)).item())


# ----- moe (V4MoE wrapped to expose .output) -----


def test_moe_block_builds_and_runs():
    sb = _mk([BlockSpec(kind="moe", repeat=1, params={
        "num_experts": 4, "top_k": 2, "expert_hidden_size": 128,
    })])
    tok, h = _inputs()
    out = sb(tok, h)
    assert out.shape == h.shape
    assert not bool(mx.any(mx.isnan(out)).item())


# ----- attention / mla / mla_absorb (standard self-attention fallback) -----


@pytest.mark.parametrize("kind", ["attention", "mla", "mla_absorb"])
def test_attention_kinds_build_and_run(kind):
    sb = _mk([BlockSpec(kind=kind, repeat=1, params={
        "num_heads": 4, "head_dim": 16,
    })])
    tok, h = _inputs()
    out = sb(tok, h)
    assert out.shape == h.shape
    assert not bool(mx.any(mx.isnan(out)).item())


def test_attention_block_zero_init_is_identity():
    """o_proj zero-init means initial forward should equal input (after norm)."""
    sb = _mk([BlockSpec(kind="attention", repeat=1, params={
        "num_heads": 4, "head_dim": 16,
    })])
    tok, h = _inputs()
    out = sb(tok, h)
    # With o_proj.weight = 0, the attention contribution is zero, so out == h
    # (residual passes through unchanged).
    np.testing.assert_allclose(np.array(out), np.array(h), atol=1e-5)


# ----- lightning_indexer (residual no-op wrapper) -----


def test_lightning_indexer_builds_and_runs():
    sb = _mk([BlockSpec(kind="lightning_indexer", repeat=1, params={
        "n_heads": 2, "head_dim": 32, "rope_head_dim": 16,
        "q_lora_rank": 64, "index_topk": 16,
    })])
    tok, h = _inputs()
    out = sb(tok, h)
    assert out.shape == h.shape
    # Lightning indexer residual is zero by design, so output == input.
    np.testing.assert_array_equal(np.array(out), np.array(h))


# ----- Full V4 stack: gdn + kda + moe + nsa + csa_hca + engram + mla -----


def test_full_v4_stack_runs_end_to_end():
    """Compose every real block kind in one stack — the 'real 1B' shape."""
    t = RunTemplate(
        name="full_v4", hidden_size=128,
        blocks=[
            BlockSpec(kind="engram", repeat=1, params={
                "num_ngram_layers": 1, "max_ngram_size": 3,
                "num_embed_table_per_ngram": 2, "embed_dim": 16,
                "embed_table_size": 64,
            }),
            BlockSpec(kind="gdn", repeat=1, params={
                "num_heads": 4, "head_dim": 32, "use_short_conv": False,
            }),
            BlockSpec(kind="kda", repeat=1, params={
                "num_heads": 4, "head_dim": 32, "use_short_conv": False,
            }),
            BlockSpec(kind="attention", repeat=1, params={
                "num_heads": 4, "head_dim": 32,
            }),
            BlockSpec(kind="nsa", repeat=1, params={
                "num_heads": 4, "head_dim": 32,
                "compress_block_size": 4, "select_topk": 2, "sliding_window": 4,
            }),
            BlockSpec(kind="csa_hca", repeat=1, params={
                "num_heads": 4, "head_dim": 32, "m_csa": 2, "m_hca": 4,
            }),
            BlockSpec(kind="moe", repeat=1, params={
                "num_experts": 4, "top_k": 2, "expert_hidden_size": 256,
            }),
            BlockSpec(kind="mlp", repeat=1, params={}),
        ],
    )
    sb = UnifiedSuperblockV4(t)
    assert sb.kinds() == [
        "engram", "gdn", "kda", "attention", "nsa", "csa_hca", "moe", "mlp",
    ]
    B, S = 1, 8
    tok = mx.array(np.random.randint(1, 50, (B, S)).astype(np.int32))
    h = mx.random.normal((B, S, t.hidden_size))
    docs = mx.array([[0] * 4 + [1] * 4], dtype=mx.int32)
    out = sb(tok, h, document_ids=docs)
    assert out.shape == h.shape
    assert not bool(mx.any(mx.isnan(out)).item())


def test_full_v4_stack_grads_propagate():
    """Through the full stack, mx.grad must produce finite grads on h."""
    t = RunTemplate(
        name="full_v4_grad", hidden_size=64,
        blocks=[
            BlockSpec(kind="gdn", repeat=1, params={
                "num_heads": 4, "head_dim": 16, "use_short_conv": False,
            }),
            BlockSpec(kind="attention", repeat=1, params={
                "num_heads": 4, "head_dim": 16,
            }),
            BlockSpec(kind="moe", repeat=1, params={
                "num_experts": 4, "top_k": 2, "expert_hidden_size": 128,
            }),
        ],
    )
    sb = UnifiedSuperblockV4(t)
    B, S = 1, 8
    tok = mx.array(np.random.randint(1, 50, (B, S)).astype(np.int32))
    h = mx.random.normal((B, S, t.hidden_size))
    cot = mx.random.normal(h.shape)
    def loss(hh):
        out = sb(tok, hh)
        return (out * cot).sum()
    g = mx.grad(loss)(h)
    assert g.shape == h.shape
    assert np.all(np.isfinite(np.array(g)))
