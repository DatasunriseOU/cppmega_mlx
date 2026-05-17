"""Tests for cppmega_v4.nn._external.tilekernels_engram — port of DeepSeek engram torch ref.

Parity tests against ``~/sources/TileKernels/tile_kernels/torch/engram.py``
when torch is available.
"""

from __future__ import annotations

import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from cppmega_v4.nn._external.tilekernels_engram import (
    engram_gate_ref,
    engram_hash_ref,
    make_offsets,
)


# ----- make_offsets shape + values -----


def test_make_offsets_shape_and_values():
    # 2 layers, 3 ngram positions, 2 tables/ngram -> flat per layer has 6 entries.
    vocab_sizes = mx.array([[[10, 20], [30, 40], [50, 60]],
                            [[5, 5], [5, 5], [5, 5]]], dtype=mx.int32)
    offsets = make_offsets(vocab_sizes)
    assert offsets.shape == (2, 6)
    # Layer 0 exclusive prefix sum of [10, 20, 30, 40, 50, 60] = [0,10,30,60,100,150]
    np.testing.assert_array_equal(
        np.array(offsets[0]), np.array([0, 10, 30, 60, 100, 150], dtype=np.int32)
    )
    np.testing.assert_array_equal(
        np.array(offsets[1]), np.array([0, 5, 10, 15, 20, 25], dtype=np.int32)
    )


def test_engram_hash_shape_contract():
    num_tokens = 4
    max_ngram_size = 3
    num_ngram_layers = 2
    num_embed = 2
    rng = np.random.default_rng(42)
    ngram_token_ids = mx.array(
        rng.integers(0, 100, size=(num_tokens, max_ngram_size)).astype(np.int32)
    )
    multipliers = mx.array(
        rng.integers(1, 10000, size=(num_ngram_layers, max_ngram_size)).astype(np.int64)
    )
    vocab_sizes = mx.array(
        rng.integers(5, 50, size=(num_ngram_layers, max_ngram_size - 1, num_embed)).astype(np.int32)
    )
    offsets = make_offsets(vocab_sizes)
    out = engram_hash_ref(ngram_token_ids, multipliers, vocab_sizes, offsets)
    # out shape (num_ngram_layers, num_tokens, (max_ngram_size - 1) * num_embed_table_per_ngram)
    assert out.shape == (num_ngram_layers, num_tokens, (max_ngram_size - 1) * num_embed)


def test_engram_gate_shape_and_dtype():
    num_tokens, hc_mult, hidden = 4, 2, 16
    rng = np.random.default_rng(7)
    hs = mx.array(rng.standard_normal((num_tokens, hc_mult, hidden)).astype(np.float32))
    k = mx.array(rng.standard_normal((num_tokens, hc_mult, hidden)).astype(np.float32))
    v = mx.array(rng.standard_normal((num_tokens, hidden)).astype(np.float32))
    wh = mx.ones((hc_mult, hidden))
    we = mx.ones((hc_mult, hidden))
    out = engram_gate_ref(
        hs.astype(mx.bfloat16), k.astype(mx.bfloat16), v.astype(mx.bfloat16),
        wh.astype(mx.bfloat16), we.astype(mx.bfloat16),
        clamp_value=1e-4, eps=1e-6,
    )
    assert out.shape == (num_tokens, hc_mult, hidden)
    assert out.dtype == mx.bfloat16


def test_engram_gate_save_for_backward_returns_tuple():
    num_tokens, hc_mult, hidden = 2, 1, 8
    hs = mx.random.normal((num_tokens, hc_mult, hidden)).astype(mx.bfloat16)
    k = mx.random.normal((num_tokens, hc_mult, hidden)).astype(mx.bfloat16)
    v = mx.random.normal((num_tokens, hidden)).astype(mx.bfloat16)
    wh = mx.ones((hc_mult, hidden)).astype(mx.bfloat16)
    we = mx.ones((hc_mult, hidden)).astype(mx.bfloat16)
    out, dot, gate, rstd_x, rstd_k = engram_gate_ref(
        hs, k, v, wh, we, clamp_value=1e-4, eps=1e-6, save_for_backward=True
    )
    assert out.shape == (num_tokens, hc_mult, hidden)
    assert dot.shape == (num_tokens, hc_mult)
    assert gate.shape == (num_tokens, hc_mult)
    assert rstd_x.shape == (num_tokens, hc_mult)
    assert rstd_k.shape == (num_tokens, hc_mult)


# ----- parity vs PyTorch TileKernels reference -----


@pytest.fixture(scope="module")
def tk_engram_torch():
    torch = pytest.importorskip("torch")
    repo = Path("/Users/dave/sources/TileKernels")
    if not repo.exists():
        pytest.skip("TileKernels repo not present at expected path")
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    try:
        from tile_kernels.torch.engram import (
            engram_gate_ref as tk_gate,
            engram_hash_ref as tk_hash,
            make_offsets as tk_offsets,
        )
    except Exception as exc:
        pytest.skip(f"could not import TileKernels torch.engram: {exc}")
    return {"torch": torch, "gate": tk_gate, "hash": tk_hash, "offsets": tk_offsets}


def test_parity_make_offsets(tk_engram_torch):
    torch = tk_engram_torch["torch"]
    rng = np.random.default_rng(11)
    vs_np = rng.integers(1, 100, size=(2, 3, 2)).astype(np.int32)
    t_out = tk_engram_torch["offsets"](torch.from_numpy(vs_np))
    m_out = make_offsets(mx.array(vs_np))
    np.testing.assert_array_equal(np.array(m_out), t_out.numpy())


def test_parity_engram_hash(tk_engram_torch):
    torch = tk_engram_torch["torch"]
    rng = np.random.default_rng(12)
    num_tokens, max_ngram, layers, num_embed = 4, 3, 2, 2
    ng = rng.integers(0, 100, size=(num_tokens, max_ngram)).astype(np.int32)
    mult = rng.integers(1, 10000, size=(layers, max_ngram)).astype(np.int64)
    vs = rng.integers(5, 50, size=(layers, max_ngram - 1, num_embed)).astype(np.int32)
    off = make_offsets(mx.array(vs))
    t_off = tk_engram_torch["offsets"](torch.from_numpy(vs))
    t_out = tk_engram_torch["hash"](
        torch.from_numpy(ng), torch.from_numpy(mult), torch.from_numpy(vs), t_off
    )
    m_out = engram_hash_ref(mx.array(ng), mx.array(mult), mx.array(vs), off)
    np.testing.assert_array_equal(np.array(m_out), t_out.numpy())


def test_parity_engram_gate(tk_engram_torch):
    torch = tk_engram_torch["torch"]
    rng = np.random.default_rng(13)
    nt, hc, h = 3, 2, 16
    hs = rng.standard_normal((nt, hc, h)).astype(np.float32)
    k = rng.standard_normal((nt, hc, h)).astype(np.float32)
    v = rng.standard_normal((nt, h)).astype(np.float32)
    wh = rng.standard_normal((hc, h)).astype(np.float32)
    we = rng.standard_normal((hc, h)).astype(np.float32)
    t_out = tk_engram_torch["gate"](
        torch.from_numpy(hs).bfloat16(),
        torch.from_numpy(k).bfloat16(),
        torch.from_numpy(v).bfloat16(),
        torch.from_numpy(wh).bfloat16(),
        torch.from_numpy(we).bfloat16(),
        clamp_value=1e-4, eps=1e-6,
    )
    m_out = engram_gate_ref(
        mx.array(hs).astype(mx.bfloat16),
        mx.array(k).astype(mx.bfloat16),
        mx.array(v).astype(mx.bfloat16),
        mx.array(wh).astype(mx.bfloat16),
        mx.array(we).astype(mx.bfloat16),
        clamp_value=1e-4, eps=1e-6,
    )
    # bfloat16 path — tolerate ~5e-2 max abs error.
    np.testing.assert_allclose(
        np.array(m_out.astype(mx.float32)),
        t_out.float().numpy(),
        atol=5e-2, rtol=5e-2,
    )
