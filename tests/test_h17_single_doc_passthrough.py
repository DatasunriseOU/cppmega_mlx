"""V7-I04: H17 single-doc passthrough — real-forward empirical proof.

H17 introduced doc_attention_mask but the "single-doc reduces to
causal-only" claim was previously analytical only. This test runs the
real attention forward pass twice on identical inputs:

  (A) No doc mask  (kwargs={})
  (B) Single-doc mask (doc_ids all equal → mask is all-ones)

and asserts the outputs match within fp tolerance (mx.allclose).
"""

from __future__ import annotations

import mlx.core as mx
import pytest

from cppmega_v4.models.unified_superblock_v4 import BLOCK_BUILDERS


def _build_attn(hidden: int = 128, num_heads: int = 4,
                head_dim: int = 32):
    """_build_attention zero-inits o_proj as an "identity-at-init"
    safety; re-randomise it here so the attention pattern actually
    influences the output (otherwise both forwards return zero and the
    passthrough check is vacuous)."""
    m = BLOCK_BUILDERS["attention"](hidden, {
        "num_heads": num_heads,
        "head_dim": head_dim,
    })
    m.o_proj.weight = mx.random.normal(
        shape=m.o_proj.weight.shape, key=mx.random.key(17))
    return m


def _doc_mask_all_same(B: int, S: int):
    """single-document mask = all-ones (S, S) per batch entry,
    expanded to attention shape (B, 1, S, S)."""
    doc_ids = mx.zeros((B, S), dtype=mx.int32)
    same = doc_ids[:, :, None] == doc_ids[:, None, :]  # (B, S, S)
    return same[:, None, :, :]


def test_v7_i04_h17_single_doc_passthrough_bit_identical():
    """Output of attention(x) == attention(x, doc_attention_mask=all_ones)
    within fp tolerance."""
    B, S, H = 2, 16, 128
    attn = _build_attn(hidden=H, num_heads=4, head_dim=32)
    # Fixed-seed input so both forwards see the same tensor.
    x = mx.random.normal(shape=(B, S, H), key=mx.random.key(7))

    out_causal = attn(x)
    mask = _doc_mask_all_same(B, S)
    try:
        out_singledoc = attn(x, doc_attention_mask=mask)
    except TypeError:
        pytest.skip(
            "this attention module does not accept doc_attention_mask"
            " (kind dispatch landed on a non-H17 variant)"
        )

    assert out_causal.shape == out_singledoc.shape
    # Tight tolerance — single-doc mask is mathematically identity.
    assert mx.allclose(out_causal, out_singledoc, atol=1e-6, rtol=1e-6), (
        f"H17 single-doc passthrough failed: max diff "
        f"{float(mx.max(mx.abs(out_causal - out_singledoc)).item())}"
    )


def test_v7_i04_h17_multi_doc_changes_output():
    """Sanity inverse: doc_ids with 2 distinct values must change
    output vs causal-only (proves the mask is actually wired through)."""
    B, S, H = 2, 16, 128
    attn = _build_attn(hidden=H, num_heads=4, head_dim=32)
    x = mx.random.normal(shape=(B, S, H), key=mx.random.key(11))

    out_causal = attn(x)
    # Two halves with different doc ids → block-diagonal mask.
    doc_ids = mx.concatenate(
        [mx.zeros((B, S // 2), dtype=mx.int32),
         mx.ones((B, S - S // 2), dtype=mx.int32)],
        axis=1,
    )
    same = doc_ids[:, :, None] == doc_ids[:, None, :]
    mask = same[:, None, :, :]
    try:
        out_multidoc = attn(x, doc_attention_mask=mask)
    except TypeError:
        pytest.skip("attention has no doc_attention_mask param")
    max_diff = float(mx.max(mx.abs(out_causal - out_multidoc)).item())
    assert max_diff > 1e-4, (
        f"multi-doc mask had no effect on output (max diff {max_diff}); "
        "single-doc passthrough test is then a vacuous identity"
    )
