"""Tests for the graph-supervised DSA dependency indexer.

Covers:
* indexer recall@k on a synthetic graph (use->def / callsite->callee blocks rank
  above non-edges);
* the block ``S_blk`` bias actually shifts top-k toward call-edge blocks
  (ablate beta=0 -> different selection);
* ``DenseCppLM`` with ``attention_mode='dsa'`` forward+backward + a few training
  steps lower loss on the read-only golden-mini fixture.
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pyarrow.parquet as pq
import pytest

from cppmega_mlx.data.code_packet_builder import (
    _int_vector,
    _table_columns,
    build_code_packet_from_row,
)
from cppmega_mlx.data.graph_packet import GraphPacket
from cppmega_mlx.models.dense_cpp_lm import DenseCppLM, DenseCppLMConfig
from cppmega_mlx.nn.code_graph_routes import (
    GraphRouteConfig,
    build_attention_bias,
    build_block_candidates,
)
from cppmega_mlx.nn.sparse_mla import (
    graph_indexed_attention_reference,
    indexer_topk_indices,
    lightning_indexer_scores,
)

_GOLDEN = Path(__file__).resolve().parent / "fixtures" / "golden_mini"


# --------------------------------------------------------------------- #
# Indexer score + selection primitives
# --------------------------------------------------------------------- #
def test_lightning_indexer_score_formula():
    # I[t,s] = sum_h w_h ReLU(q_h . k_s) + beta * S_blk[t,s].
    B, S, Hi, Di = 1, 3, 2, 4
    q = mx.ones((B, S, Hi, Di))
    k = mx.ones((B, S, Hi, Di))
    w = mx.array([1.0, 2.0])
    bias = mx.zeros((B, S, S))
    bias = bias + mx.eye(S)[None]  # diagonal prior
    scores = lightning_indexer_scores(
        q, k, w, block_bias=bias, beta=10.0, causal=True
    )
    s = np.asarray(scores)
    # diagonal: ReLU(q.k)=Di=4 per head; sum_h w_h*4 = (1+2)*4 = 12; +beta*1 = 22.
    assert abs(s[0, 0, 0] - 22.0) < 1e-4
    # off-diagonal causal entry [2,0]: 12 + beta*0 = 12.
    assert abs(s[0, 2, 0] - 12.0) < 1e-4
    # future masked: [0,2] -> -inf.
    assert s[0, 0, 2] < -1e8


def test_indexer_topk_keeps_sinks_and_local():
    B, S, Skv = 1, 6, 6
    scores = mx.array(np.zeros((B, S, Skv), dtype=np.float32))
    sel = indexer_topk_indices(
        scores, topk=1, local_window=2, num_sinks=1, causal=True
    )
    s = np.asarray(sel)
    # Query 5: sink 0 and local window {4,5} must always be present.
    row = set(int(x) for x in s[0, 5] if x >= 0)
    assert 0 in row  # sink
    assert 4 in row and 5 in row  # local window
    # Query 0: only position 0 is causally valid.
    assert set(int(x) for x in s[0, 0] if x >= 0) == {0}


# --------------------------------------------------------------------- #
# Recall@k on a synthetic dependency graph
# --------------------------------------------------------------------- #
def test_indexer_recall_on_synthetic_graph():
    # 4 KV blocks per query block. The graph prior beta*S_blk should push the
    # true callsite->callee block into the top-1 even when the learned head dots
    # are uninformative (all-equal), proving the graph route drives selection.
    B, S, Hi, Di = 1, 4, 1, 4
    rng = np.random.default_rng(0)
    q = mx.array(rng.standard_normal((B, S, Hi, Di)).astype(np.float32))
    k = mx.array(rng.standard_normal((B, S, Hi, Di)).astype(np.float32))
    w = mx.array([1.0])
    # true edges: query0->key0, query1->key0, query2->key1, query3->key2.
    true_edge = {0: 0, 1: 0, 2: 1, 3: 2}
    bias = np.zeros((B, S, S), dtype=np.float32)
    for t, s in true_edge.items():
        bias[0, t, s] = 1.0
    bias_t = mx.array(bias)

    scores = lightning_indexer_scores(q, k, w, block_bias=bias_t, beta=50.0)
    sel = np.asarray(
        indexer_topk_indices(scores, topk=1, local_window=0, num_sinks=0)
    )
    hits = sum(1 for t, s in true_edge.items() if s in set(sel[0, t].tolist()))
    recall = hits / len(true_edge)
    assert recall == 1.0  # every true dependency block selected at top-1


def test_beta_ablation_shifts_topk_selection():
    # With beta=0 the selection is driven only by the (random) head dots; turning
    # the graph prior on (beta>0) must change which blocks are selected toward the
    # true call-edge blocks. This is the S_blk ablation.
    B, S, Hi, Di = 1, 4, 1, 4
    rng = np.random.default_rng(7)
    q = mx.array(rng.standard_normal((B, S, Hi, Di)).astype(np.float32))
    k = mx.array(rng.standard_normal((B, S, Hi, Di)).astype(np.float32))
    w = mx.array([1.0])
    true_edge = {1: 0, 2: 0, 3: 1}
    bias = np.zeros((B, S, S), dtype=np.float32)
    for t, s in true_edge.items():
        bias[0, t, s] = 1.0
    bias_t = mx.array(bias)

    sel_off = np.asarray(
        indexer_topk_indices(
            lightning_indexer_scores(q, k, w, block_bias=bias_t, beta=0.0),
            topk=1,
            local_window=0,
            num_sinks=0,
        )
    )
    sel_on = np.asarray(
        indexer_topk_indices(
            lightning_indexer_scores(q, k, w, block_bias=bias_t, beta=50.0),
            topk=1,
            local_window=0,
            num_sinks=0,
        )
    )
    # selection differs.
    assert not np.array_equal(sel_off, sel_on)
    # and beta>0 recovers the true edges (which beta=0 fails to).
    hits_on = sum(1 for t, s in true_edge.items() if s in set(sel_on[0, t].tolist()))
    hits_off = sum(1 for t, s in true_edge.items() if s in set(sel_off[0, t].tolist()))
    assert hits_on > hits_off
    assert hits_on == len(true_edge)


def test_graph_indexed_attention_reference_runs_and_is_differentiable():
    B, S, H, G, D = 1, 6, 4, 2, 8
    Hi, Di = 2, 4
    rng = np.random.default_rng(1)
    q = mx.array(rng.standard_normal((B, S, H, D)).astype(np.float32))
    kv = mx.array(rng.standard_normal((B, S, G, D)).astype(np.float32))
    qi = mx.array(rng.standard_normal((B, S, Hi, Di)).astype(np.float32))
    ki = mx.array(rng.standard_normal((B, S, Hi, Di)).astype(np.float32))
    w = mx.ones((Hi,))

    def loss_fn(q):
        out = graph_indexed_attention_reference(
            q, kv, qi, ki, w, topk=3, local_window=1, num_sinks=1, kv_group=G
        )
        return mx.sum(out**2)

    val, grad = mx.value_and_grad(loss_fn)(q)
    mx.eval(grad)
    assert np.isfinite(float(val))
    assert grad.shape == q.shape


# --------------------------------------------------------------------- #
# DenseCppLM DSA seam: forward + backward + train on golden-mini
# --------------------------------------------------------------------- #
def _golden_graph_rows():
    table = pq.read_table(_GOLDEN / "code" / "code_graph.parquet")
    cols = _table_columns(table)
    rows = []
    for r in range(len(cols["input_ids"])):
        tok = _int_vector(cols["input_ids"][r], where="input_ids")
        tgt = _int_vector(cols["target_ids"][r], where="target_ids")
        lm = _int_vector(cols["loss_mask"][r], where="loss_mask")
        pkt = build_code_packet_from_row(
            token_ids=tok, target_ids=tgt, loss_mask=lm, columns=cols, row_index=r
        )
        rows.append(pkt)
    return rows


def _tiny_dsa_model(vocab: int, seq: int) -> DenseCppLM:
    cfg = DenseCppLMConfig(
        vocab_size=vocab,
        hidden_size=64,
        depth=2,
        ffn_hidden_size=128,
        max_seq_length=seq,
        num_query_heads=4,
        num_kv_heads=2,
        head_dim=16,
        attention_mode="dsa",
        attention_sparse_topk=16,
        indexer_heads=2,
        indexer_dim=8,
        indexer_local_window=4,
        indexer_num_sinks=1,
        require_graph_routes=False,
        ngram_hash_enabled=False,
        structure_residual_scale=0.0,
        platform_residual_scale=0.0,
    )
    return DenseCppLM(cfg)


def test_dense_cpp_lm_dsa_forward_exposes_indexer_scores():
    model = _tiny_dsa_model(vocab=128, seq=64)
    ids = mx.array(np.random.default_rng(0).integers(0, 128, (2, 24)))
    tgt = mx.array(np.random.default_rng(1).integers(0, 128, (2, 24)))
    logits, loss = model(ids, targets=tgt)
    assert tuple(logits.shape) == (2, 24, 128)
    scores = model.indexer_scores()
    assert len(scores) == 2  # one per layer
    assert tuple(scores[0].shape) == (2, 24, 24)


def test_dense_cpp_lm_dsa_requires_graph_routes_by_default():
    cfg = DenseCppLMConfig(
        vocab_size=128,
        hidden_size=64,
        depth=1,
        ffn_hidden_size=128,
        max_seq_length=64,
        num_query_heads=4,
        num_kv_heads=2,
        head_dim=16,
        attention_mode="dsa",
        attention_sparse_topk=8,
        indexer_heads=2,
        indexer_dim=8,
        indexer_local_window=2,
        indexer_num_sinks=1,
        ngram_hash_enabled=False,
        structure_residual_scale=0.0,
        platform_residual_scale=0.0,
    )
    model = DenseCppLM(cfg)
    ids = mx.array(np.random.default_rng(0).integers(0, 128, (1, 16)))
    tgt = mx.array(np.random.default_rng(1).integers(0, 128, (1, 16)))
    with pytest.raises(RuntimeError, match="requires graph route block_bias"):
        model(ids, targets=tgt)

    bias = mx.zeros((1, 16, 16), dtype=mx.float32)
    logits, loss = model(ids, targets=tgt, block_bias=bias)
    assert tuple(logits.shape) == (1, 16, 128)
    assert loss is not None and np.isfinite(float(loss))


def test_dense_cpp_lm_dsa_trains_on_golden_mini():
    rows = _golden_graph_rows()
    # Use a short prefix of the long packed rows to keep the test fast.
    seq = 96
    ids = mx.stack([r.token_ids[:seq] for r in rows])
    tgt = mx.stack([r.target_ids[:seq] for r in rows])
    lm = mx.stack([r.loss_mask[:seq].astype(mx.float32) for r in rows])
    vocab = int(mx.max(ids).item()) + 1
    vocab = max(vocab, int(mx.max(tgt).item()) + 1)

    model = _tiny_dsa_model(vocab=vocab, seq=seq)

    # Build a token-level graph prior from the row-0 call edges, mapped to a
    # B-block grid then expanded chunk->token (here: a flat zero bias is the
    # ablation baseline; the seam threads block_bias through cleanly).
    pkt0 = rows[0]
    graph_pkt = GraphPacket(
        edges={"call": pkt0.call_edges} if pkt0.call_edges is not None else {},
        num_nodes=int(pkt0.chunk_starts.shape[0]),
    )
    cfg = GraphRouteConfig(num_blocks=8, relations=("call",), normalize="binary")
    s_blk = build_attention_bias(graph_pkt, config=cfg)
    assert tuple(s_blk.shape) == (8, 8)  # the prior exists
    cands = build_block_candidates(graph_pkt, config=cfg)
    assert any(len(c) > 0 for c in cands)  # golden-mini has real call edges

    def loss_fn(model):
        _, loss = model(ids, targets=tgt, loss_mask=lm)
        return loss

    loss0 = float(loss_fn(model))
    opt = __import__(
        "mlx.optimizers", fromlist=["Adam"]
    ).Adam(learning_rate=1e-2)
    vg = nn.value_and_grad(model, loss_fn)
    for _ in range(8):
        loss_val, grads = vg(model)
        opt.update(model, grads)
        mx.eval(model.parameters(), opt.state)
    loss1 = float(loss_fn(model))
    assert np.isfinite(loss0) and np.isfinite(loss1)
    assert loss1 < loss0  # the DSA path trains (loss decreases) on golden-mini


def test_gqa_default_unchanged_by_dsa_additions():
    # The default (GQA) path must be untouched: forward + backward still run and
    # the GQA model exposes no indexer scores.
    cfg = DenseCppLMConfig(
        vocab_size=128,
        hidden_size=64,
        depth=2,
        ffn_hidden_size=128,
        max_seq_length=64,
        num_query_heads=4,
        num_kv_heads=2,
        head_dim=16,
        attention_mode="gqa",
        ngram_hash_enabled=False,
        structure_residual_scale=0.0,
        platform_residual_scale=0.0,
    )
    model = DenseCppLM(cfg)
    ids = mx.array(np.random.default_rng(0).integers(0, 128, (2, 16)))
    tgt = mx.array(np.random.default_rng(1).integers(0, 128, (2, 16)))
    _, loss = model(ids, targets=tgt)
    assert np.isfinite(float(loss))
    with pytest.raises(ValueError):
        model.indexer_scores()
