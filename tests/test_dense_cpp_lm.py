"""Tests for the Stage-1 dense C++ foundation LM (DenseCppLM).

Covers:
  * forward on tiny shapes returns correctly-shaped logits + a finite CE loss;
  * GQA kv-head broadcasting is shape-correct (q=8 heads attend over kv=2 heads,
    group size 4) and the SDPA path produces the right output shape;
  * side channels measurably change the loss (ablate the structure embedding
    on/off -> different loss);
  * a short overfit on one tiny batch drives the loss down;
  * the real ~500M profile builds and its transformer core is ~500M params;
  * the DSA seam: flipping attention_mode='dsa' rebuilds without restructuring.
"""

from __future__ import annotations

from typing import get_args

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from cppmega_mlx.data.batch import LMTokenBatch
from cppmega_mlx.data.code_packet import CodePacket
from cppmega_mlx.data.domain_packet import DomainEdgeIndex
from cppmega_mlx.data.graph_packet import EdgeIndex, GraphBatch, GraphPacket
from cppmega_mlx.models.dense_cpp_lm import (
    DenseCppBlock,
    DenseAttentionMode,
    DenseCppLM,
    DenseCppLMConfig,
)
from cppmega_mlx.training.loss import next_token_cross_entropy


def _smoke_config(**overrides) -> DenseCppLMConfig:
    base = dict(
        vocab_size=512,
        hidden_size=256,
        depth=4,
        ffn_hidden_size=512,
        max_seq_length=64,
        num_query_heads=8,
        num_kv_heads=2,
        head_dim=32,
        ngram_hash_table_size=4096,
        ngram_hash_embed_dim=16,
    )
    base.update(overrides)
    return DenseCppLMConfig(**base)


def _rand_batch(cfg: DenseCppLMConfig, batch: int, seq: int, *, seed: int = 0):
    mx.random.seed(seed)
    return {
        "input_ids": mx.random.randint(0, cfg.vocab_size, (batch, seq)),
        "targets": mx.random.randint(0, cfg.vocab_size, (batch, seq)),
        "structure_ids": mx.random.randint(0, cfg.structure_num_categories, (batch, seq)),
        "dep_levels": mx.random.randint(0, cfg.structure_max_dep_level, (batch, seq)),
        "ast_depth_ids": mx.random.randint(0, cfg.structure_max_ast_depth, (batch, seq)),
        "sibling_index_ids": mx.random.randint(
            0, cfg.structure_max_sibling_index, (batch, seq)
        ),
        "node_type_ids": mx.random.randint(0, cfg.structure_num_node_types, (batch, seq)),
    }


def test_forward_tiny_shapes():
    cfg = _smoke_config()
    model = DenseCppLM(cfg)
    b = _rand_batch(cfg, 2, 16)
    logits, loss = model(b["input_ids"], targets=b["targets"])
    mx.eval(logits, loss)
    assert tuple(logits.shape) == (2, 16, cfg.vocab_size)
    assert loss is not None
    assert mx.isfinite(loss).item()
    # No targets -> loss is None.
    logits_only, no_loss = model(b["input_ids"])
    assert no_loss is None
    assert tuple(logits_only.shape) == (2, 16, cfg.vocab_size)


def test_tied_lm_head():
    cfg = _smoke_config()
    model = DenseCppLM(cfg)
    # Tied head: lm_head.weight IS the token-embedding table (same object).
    assert model.lm_head.weight is model.token_embedding.weight
    assert tuple(model.lm_head.weight.shape) == (cfg.vocab_size, cfg.hidden_size)


def test_gqa_kv_head_broadcasting_shapes():
    cfg = _smoke_config(num_query_heads=8, num_kv_heads=2, head_dim=32)
    model = DenseCppLM(cfg)
    b, s = 3, 12
    shapes = model.gqa_attention_shapes(b, s)
    assert shapes["q"] == (b, 8, s, 32)
    assert shapes["k"] == (b, 2, s, 32)
    assert shapes["v"] == (b, 2, s, 32)
    assert shapes["kv_group_size"] == 4  # 8 q heads / 2 kv heads
    assert shapes["attention_mode"] == "gqa"

    # The attention block actually projects q/k/v to those head counts and the
    # SDPA path broadcasts kv across the group, returning a (B, S, d_model) delta.
    attn = model.layers[0].attention
    assert attn.config.is_gqa
    hidden = mx.random.normal((b, s, cfg.hidden_size))
    q, k, v = attn._project_qkv(hidden)
    mx.eval(q, k, v)
    assert tuple(q.shape) == (b, 8, s, 32)
    assert tuple(k.shape) == (b, 2, s, 32)
    assert tuple(v.shape) == (b, 2, s, 32)
    mask = nn.MultiHeadAttention.create_additive_causal_mask(s, dtype=hidden.dtype)
    out = attn(hidden, mask)
    mx.eval(out)
    assert tuple(out.shape) == (b, s, cfg.hidden_size)


def test_gqa_attention_consumes_graph_route_bias():
    cfg = _smoke_config(
        depth=1,
        ngram_hash_enabled=False,
        graph_routes_enabled=True,
    )
    model = DenseCppLM(cfg)
    b = _rand_batch(cfg, 1, 8, seed=17)
    zero_bias = mx.zeros((1, 8, 8), dtype=mx.float32)
    block_bias = mx.zeros((1, 8, 8), dtype=mx.float32)
    block_bias[:, 7, 1] = 50.0

    logits_plain, _ = model(
        b["input_ids"],
        block_bias=zero_bias,
        edge_kind_bias=zero_bias,
    )
    logits_biased, _ = model(
        b["input_ids"],
        block_bias=block_bias,
        edge_kind_bias=zero_bias,
    )
    mx.eval(logits_plain, logits_biased)

    delta = mx.abs(logits_plain - logits_biased).sum()
    assert float(delta.item()) > 1e-4


def test_gqa_attention_rejects_malformed_graph_route_bias():
    cfg = _smoke_config(depth=1, graph_routes_enabled=True)
    model = DenseCppLM(cfg)
    b = _rand_batch(cfg, 1, 8, seed=19)

    with pytest.raises(ValueError, match="attention_bias must be shaped"):
        model(
            b["input_ids"],
            block_bias=mx.zeros((1, 7, 8), dtype=mx.float32),
            edge_kind_bias=mx.zeros((1, 8, 8), dtype=mx.float32),
        )


def test_gqa_config_rejects_equal_kv_heads():
    # mode='gqa' requires num_kv_heads strictly < num_query_heads (the seam's
    # contract lives in AttentionConfig and is validated at config construction).
    with pytest.raises(ValueError):
        _smoke_config(num_query_heads=8, num_kv_heads=8)


def test_dense_stage1_rejects_advertised_mla_alias():
    assert "mla" not in get_args(DenseAttentionMode)
    with pytest.raises(ValueError, match="MLA.*not implemented"):
        _smoke_config(attention_mode="mla")


def test_side_channels_change_loss():
    cfg = _smoke_config()
    model = DenseCppLM(cfg)
    b = _rand_batch(cfg, 2, 16, seed=7)

    # Make the structure embedding non-trivial so ablation is observable: the
    # table is zero-initialized by design, so we randomize it for this test.
    se = model.structure_embedding
    se.stacked_emb.weight = mx.random.normal(se.stacked_emb.weight.shape) * 0.5
    se.up_proj.weight = mx.random.normal(se.up_proj.weight.shape) * 0.5

    _, loss_with = model(
        b["input_ids"],
        targets=b["targets"],
        structure_ids=b["structure_ids"],
        dep_levels=b["dep_levels"],
        ast_depth_ids=b["ast_depth_ids"],
        sibling_index_ids=b["sibling_index_ids"],
        node_type_ids=b["node_type_ids"],
    )
    _, loss_without = model(b["input_ids"], targets=b["targets"])
    mx.eval(loss_with, loss_without)
    # Feeding the structure side channels measurably changes the loss.
    assert abs(float(loss_with.item()) - float(loss_without.item())) > 1e-4


def test_residual_scale_zero_disables_structure():
    # structure_residual_scale=0 must make structure inputs a no-op (clean off
    # switch, not a silent fallback) -> identical loss with/without channels.
    cfg = _smoke_config(structure_residual_scale=0.0)
    model = DenseCppLM(cfg)
    se = model.structure_embedding
    se.stacked_emb.weight = mx.random.normal(se.stacked_emb.weight.shape) * 0.5
    b = _rand_batch(cfg, 2, 16, seed=3)
    _, loss_with = model(
        b["input_ids"], targets=b["targets"],
        structure_ids=b["structure_ids"], dep_levels=b["dep_levels"],
        ast_depth_ids=b["ast_depth_ids"], sibling_index_ids=b["sibling_index_ids"],
        node_type_ids=b["node_type_ids"],
    )
    _, loss_without = model(b["input_ids"], targets=b["targets"])
    mx.eval(loss_with, loss_without)
    assert abs(float(loss_with.item()) - float(loss_without.item())) < 1e-5


def test_overfit_tiny_batch_drives_loss_down():
    import mlx.optimizers as optim

    cfg = _smoke_config()
    model = DenseCppLM(cfg)
    b = _rand_batch(cfg, 2, 16, seed=11)
    opt = optim.AdamW(learning_rate=3e-3)

    def loss_fn(m):
        _, loss = m(
            b["input_ids"], targets=b["targets"],
            structure_ids=b["structure_ids"], dep_levels=b["dep_levels"],
            ast_depth_ids=b["ast_depth_ids"], sibling_index_ids=b["sibling_index_ids"],
            node_type_ids=b["node_type_ids"],
        )
        return loss

    vg = nn.value_and_grad(model, loss_fn)
    initial = None
    final = None
    for step in range(20):
        loss, grads = vg(model)
        opt.update(model, grads)
        mx.eval(model.parameters(), opt.state, loss)
        if step == 0:
            initial = float(loss.item())
        final = float(loss.item())
    assert final < initial
    # A 20-step overfit on a 2x16 batch should drop the loss substantially.
    assert final < initial - 1.0


def test_loss_mask_excludes_positions():
    cfg = _smoke_config()
    model = DenseCppLM(cfg)
    b = _rand_batch(cfg, 1, 16, seed=5)
    full_mask = mx.ones((1, 16))
    half_mask = mx.concatenate(
        [mx.ones((1, 8)), mx.zeros((1, 8))], axis=1
    )
    _, loss_full = model(b["input_ids"], targets=b["targets"], loss_mask=full_mask)
    _, loss_half = model(b["input_ids"], targets=b["targets"], loss_mask=half_mask)
    mx.eval(loss_full, loss_half)
    # Different masked sets -> different masked-mean loss.
    assert abs(float(loss_full.item()) - float(loss_half.item())) > 1e-5


def test_forward_packet_single_window():
    cfg = _smoke_config()
    model = DenseCppLM(cfg)
    s = 16
    packet = CodePacket(
        token_ids=mx.random.randint(0, cfg.vocab_size, (s,)),
        target_ids=mx.random.randint(0, cfg.vocab_size, (s,)),
        structure_ids=mx.random.randint(0, cfg.structure_num_categories, (s,)),
        dep_levels=mx.random.randint(0, cfg.structure_max_dep_level, (s,)),
    )
    logits, loss = model.forward_packet(packet)
    mx.eval(logits, loss)
    assert tuple(logits.shape) == (1, s, cfg.vocab_size)
    assert loss is not None and mx.isfinite(loss).item()


def test_forward_packet_promotes_platform_metadata_into_model():
    cfg = _smoke_config(
        depth=1,
        ngram_hash_enabled=False,
        structure_residual_scale=0.0,
    )
    model = DenseCppLM(cfg)
    mx.random.seed(211)
    model.platform_embedding.embedding.weight = mx.random.normal(
        model.platform_embedding.embedding.weight.shape
    )
    tokens = mx.array(np.arange(8, dtype=np.int32)) % cfg.vocab_size
    packet_one = CodePacket(
        token_ids=tokens,
        metadata={"platform_ids": mx.array([1, 2], dtype=mx.int32)},
    )
    packet_two = CodePacket(
        token_ids=tokens,
        metadata={"platform_ids": mx.array([3, 4], dtype=mx.int32)},
    )

    logits_one, _ = model.forward_packet(packet_one)
    logits_two, _ = model.forward_packet(packet_two)
    mx.eval(logits_one, logits_two)

    assert float(mx.sum(mx.abs(logits_one - logits_two)).item()) > 1e-4


def test_default_ngram_forward_packet_cannot_leak_between_packed_documents():
    cfg = _smoke_config(depth=1)
    model = DenseCppLM(cfg)
    assert model.ngram_hash_embedding is not None
    model.ngram_hash_embedding.out_proj.weight = mx.ones_like(
        model.ngram_hash_embedding.out_proj.weight
    )
    document_ids = mx.array([1, 1, 1, 1, 2, 2, 2, 2], dtype=mx.int32)
    suffix = mx.array([23, 29, 31, 37], dtype=mx.int32)
    left = CodePacket(
        token_ids=mx.concatenate(
            [mx.array([2, 3, 5, 7], dtype=mx.int32), suffix]
        ),
        document_ids=document_ids,
    )
    right = CodePacket(
        token_ids=mx.concatenate(
            [mx.array([11, 13, 17, 19], dtype=mx.int32), suffix]
        ),
        document_ids=document_ids,
    )

    left_logits, _ = model.forward_packet(left)
    right_logits, _ = model.forward_packet(right)
    mx.eval(left_logits, right_logits)

    np.testing.assert_allclose(
        np.asarray(left_logits[:, 4:]),
        np.asarray(right_logits[:, 4:]),
        rtol=0.0,
        atol=1e-6,
    )


def _single_token_chunk_graph(seq: int, pairs: list[list[int]]) -> GraphBatch:
    starts = mx.arange(seq, dtype=mx.int32)
    ends = starts + 1
    graph = EdgeIndex.from_pairs(pairs, relation="call", num_nodes=seq)
    return GraphBatch(
        graphs=(CodePacket(token_ids=mx.zeros((seq,), dtype=mx.int32), call_edges=graph).graph_packet(),),
        chunk_starts=(starts,),
        chunk_ends=(ends,),
        chunk_kinds=(mx.zeros((seq,), dtype=mx.int32),),
        chunk_dep_levels=(mx.zeros((seq,), dtype=mx.int32),),
    )


@pytest.mark.parametrize("attention_mode", ["gqa", "dsa"])
def test_dense_graph_routes_and_attention_do_not_cross_documents(
    attention_mode: str,
) -> None:
    cfg = _smoke_config(
        depth=1,
        ngram_hash_enabled=False,
        graph_routes_enabled=True,
        attention_mode=attention_mode,
        attention_sparse_topk=4,
        indexer_local_window=0,
        indexer_num_sinks=0,
    )
    mx.random.seed(149)
    model = DenseCppLM(cfg)
    document_ids = mx.array([[0, 0, 0, 0, 1, 1, 1, 1]], dtype=mx.int32)
    suffix = mx.array([[23, 29, 31, 37]], dtype=mx.int32)
    left = mx.concatenate(
        [mx.array([[2, 3, 5, 7]], dtype=mx.int32), suffix], axis=1
    )
    right = mx.concatenate(
        [mx.array([[11, 13, 17, 19]], dtype=mx.int32), suffix], axis=1
    )
    graph = _single_token_chunk_graph(8, [[7, 5]])

    left_logits = model.logits(
        left,
        graph_batch=graph,
        document_ids=document_ids,
    )
    right_logits = model.logits(
        right,
        graph_batch=graph,
        document_ids=document_ids,
    )
    mx.eval(left_logits, right_logits)

    np.testing.assert_allclose(
        np.asarray(left_logits[:, 4:]),
        np.asarray(right_logits[:, 4:]),
        rtol=0.0,
        atol=1e-6,
    )
    if attention_mode == "dsa":
        scores = model.indexer_scores(
            right,
            graph_batch=graph,
            document_ids=document_ids,
        )[0]
        mx.eval(scores)
        assert np.all(np.asarray(scores)[0, 4:, :4] < -1e8)


def test_forward_packet_consumes_codepacket_graph_edges_when_enabled():
    cfg = _smoke_config(
        depth=1,
        ngram_hash_enabled=False,
        graph_routes_enabled=True,
        graph_attention_bias_beta=25.0,
    )
    model = DenseCppLM(cfg)
    seq = 8
    tokens = mx.array(np.arange(seq, dtype=np.int32))[None, :] % cfg.vocab_size
    targets = (tokens + 1) % cfg.vocab_size
    chunk_starts = mx.arange(seq, dtype=mx.int32)
    chunk_ends = chunk_starts + 1
    with_graph = CodePacket(
        token_ids=tokens,
        target_ids=targets,
        call_edges=EdgeIndex.from_pairs([[7, 1]], relation="call", num_nodes=seq),
        chunk_starts=chunk_starts,
        chunk_ends=chunk_ends,
        chunk_kinds=mx.zeros((seq,), dtype=mx.int32),
        chunk_dep_levels=mx.zeros((seq,), dtype=mx.int32),
    )
    empty_graph = CodePacket(
        token_ids=tokens,
        target_ids=targets,
        call_edges=EdgeIndex.from_pairs([], relation="call", num_nodes=seq),
        chunk_starts=chunk_starts,
        chunk_ends=chunk_ends,
        chunk_kinds=mx.zeros((seq,), dtype=mx.int32),
        chunk_dep_levels=mx.zeros((seq,), dtype=mx.int32),
    )

    logits_graph, loss_graph = model.forward_packet(with_graph)
    logits_empty, loss_empty = model.forward_packet(empty_graph)
    mx.eval(logits_graph, logits_empty, loss_graph, loss_empty)

    assert float(mx.sum(mx.abs(logits_graph - logits_empty)).item()) > 1e-4
    assert abs(float(loss_graph.item()) - float(loss_empty.item())) > 1e-5


def test_forward_packet_rejects_graph_data_when_routes_are_disabled():
    cfg = _smoke_config(depth=1, ngram_hash_enabled=False, graph_routes_enabled=False)
    model = DenseCppLM(cfg)
    seq = 8
    tokens = mx.array(np.arange(seq, dtype=np.int32))[None, :] % cfg.vocab_size
    targets = (tokens + 1) % cfg.vocab_size
    chunk_starts = mx.arange(seq, dtype=mx.int32)
    chunk_ends = chunk_starts + 1
    with_graph = CodePacket(
        token_ids=tokens,
        target_ids=targets,
        call_edges=EdgeIndex.from_pairs([[7, 1]], relation="call", num_nodes=seq),
        chunk_starts=chunk_starts,
        chunk_ends=chunk_ends,
        chunk_kinds=mx.zeros((seq,), dtype=mx.int32),
        chunk_dep_levels=mx.zeros((seq,), dtype=mx.int32),
    )
    with pytest.raises(ValueError, match="graph data.*graph_routes_enabled is false"):
        model.forward_packet(with_graph)


def test_model_rejects_fixed_graph_bias_when_routes_are_disabled():
    cfg = _smoke_config(depth=1, ngram_hash_enabled=False, graph_routes_enabled=False)
    model = DenseCppLM(cfg)
    tokens = mx.array(np.arange(8, dtype=np.int32))[None, :] % cfg.vocab_size

    with pytest.raises(ValueError, match="graph data.*graph_routes_enabled is false"):
        model(tokens, block_bias=mx.zeros((1, 8, 8), dtype=mx.float32))


def test_forward_packet_rejects_malformed_graph_edges_when_enabled():
    cfg = _smoke_config(
        depth=1,
        ngram_hash_enabled=False,
        graph_routes_enabled=True,
    )
    model = DenseCppLM(cfg)
    seq = 8
    chunk_starts = mx.arange(seq, dtype=mx.int32)
    chunk_ends = chunk_starts + 1
    packet = CodePacket(
        token_ids=mx.array(np.arange(seq, dtype=np.int32))[None, :] % cfg.vocab_size,
        target_ids=mx.array(np.arange(1, seq + 1, dtype=np.int32))[None, :] % cfg.vocab_size,
        call_edges=EdgeIndex.from_pairs([[99, 0]], relation="call", num_nodes=seq),
        chunk_starts=chunk_starts,
        chunk_ends=chunk_ends,
        chunk_kinds=mx.zeros((seq,), dtype=mx.int32),
        chunk_dep_levels=mx.zeros((seq,), dtype=mx.int32),
    )

    with pytest.raises(ValueError, match="graph call edge.*out of range"):
        model.forward_packet(packet)


def test_graph_enabled_rejects_missing_typed_graph_batch():
    cfg = _smoke_config(
        depth=1,
        ngram_hash_enabled=False,
        graph_routes_enabled=True,
    )
    model = DenseCppLM(cfg)
    tokens = mx.array(np.arange(8, dtype=np.int32))[None, :] % cfg.vocab_size

    with pytest.raises(RuntimeError, match="graph routes are enabled.*GraphBatch"):
        model(tokens)


def test_graph_enabled_rejects_relationless_typed_graph_batch():
    cfg = _smoke_config(
        depth=1,
        ngram_hash_enabled=False,
        graph_routes_enabled=True,
    )
    model = DenseCppLM(cfg)
    tokens = mx.array(np.arange(8, dtype=np.int32))[None, :] % cfg.vocab_size
    relationless = GraphBatch(
        graphs=(GraphPacket(),),
        chunk_starts=(mx.zeros((0,), dtype=mx.int32),),
        chunk_ends=(mx.zeros((0,), dtype=mx.int32),),
    )

    with pytest.raises(KeyError, match="no route relations"):
        model(tokens, graph_batch=relationless)


def test_loss_path_consumes_lmtokenbatch_graph_batch_when_enabled():
    cfg = _smoke_config(
        depth=1,
        ngram_hash_enabled=False,
        graph_routes_enabled=True,
        graph_attention_bias_beta=25.0,
    )
    model = DenseCppLM(cfg)
    tokens = mx.array(np.arange(8, dtype=np.int32))[None, :] % cfg.vocab_size
    with_graph = LMTokenBatch(
        tokens=tokens,
        target_tokens=(tokens + 1) % cfg.vocab_size,
        graph_batch=_single_token_chunk_graph(8, [[7, 1]]),
    )
    empty_graph = LMTokenBatch(
        tokens=tokens,
        target_tokens=(tokens + 1) % cfg.vocab_size,
        graph_batch=_single_token_chunk_graph(8, []),
    )

    loss_graph, _ = next_token_cross_entropy(model, with_graph)
    loss_empty, _ = next_token_cross_entropy(model, empty_graph)
    mx.eval(loss_graph, loss_empty)

    assert abs(float(loss_graph.item()) - float(loss_empty.item())) > 1e-5


def test_loss_path_input_aligns_graph_batch_for_implicit_next_token_targets():
    cfg = _smoke_config(
        depth=1,
        ngram_hash_enabled=False,
        graph_routes_enabled=True,
        graph_attention_bias_beta=25.0,
    )
    model = DenseCppLM(cfg)
    tokens = mx.array(np.arange(8, dtype=np.int32))[None, :] % cfg.vocab_size
    with_graph = LMTokenBatch(
        tokens=tokens,
        graph_batch=_single_token_chunk_graph(8, [[6, 1]]),
    )
    empty_graph = LMTokenBatch(
        tokens=tokens,
        graph_batch=_single_token_chunk_graph(8, []),
    )

    loss_graph, ntokens_graph = next_token_cross_entropy(model, with_graph)
    loss_empty, ntokens_empty = next_token_cross_entropy(model, empty_graph)
    mx.eval(loss_graph, loss_empty, ntokens_graph, ntokens_empty)

    assert int(ntokens_graph.item()) == int(ntokens_empty.item()) == 7
    assert abs(float(loss_graph.item()) - float(loss_empty.item())) > 1e-5


def test_loss_path_rejects_source_invalid_edges_before_graph_input_alignment():
    cfg = _smoke_config(
        depth=1,
        ngram_hash_enabled=False,
        graph_routes_enabled=True,
    )
    model = DenseCppLM(cfg)
    tokens = mx.array(np.arange(8, dtype=np.int32))[None, :] % cfg.vocab_size
    batch = LMTokenBatch(
        tokens=tokens,
        graph_batch=_single_token_chunk_graph(8, [[8, 0]]),
    )

    with pytest.raises(ValueError, match="call edge.*source chunks"):
        next_token_cross_entropy(model, batch)


def test_typed_graph_batch_loss_backpropagates_through_dense_attention():
    cfg = _smoke_config(
        depth=1,
        ngram_hash_enabled=False,
        graph_routes_enabled=True,
    )
    model = DenseCppLM(cfg)
    tokens = mx.array(np.arange(8, dtype=np.int32))[None, :] % cfg.vocab_size
    batch = LMTokenBatch(
        tokens=tokens,
        graph_batch=_single_token_chunk_graph(8, [[6, 1]]),
    )
    loss_and_grad = nn.value_and_grad(model, next_token_cross_entropy)

    (loss, ntokens), grads = loss_and_grad(model, batch)
    attention_grad = grads["layers"][0]["attention"]["q_proj"]["weight"]
    mx.eval(loss, ntokens, attention_grad)

    assert mx.isfinite(loss).item()
    assert int(ntokens.item()) == 7
    assert float(mx.sum(mx.abs(attention_grad)).item()) > 0.0


def test_graph_beta_default_matches_megatron_contract():
    assert _smoke_config().graph_attention_bias_beta == 1.0

    dsa = DenseCppLM(
        _smoke_config(
            depth=1,
            ngram_hash_enabled=False,
            attention_mode="dsa",
            require_graph_routes=False,
        )
    )
    assert float(dsa.layers[0].attention.index_beta.item()) == 1.0


def test_domain_side_channels_change_loss_through_lm_batch():
    cfg = _smoke_config(
        depth=1,
        ngram_hash_enabled=False,
        domain_residual_scale=1.0,
        platform_residual_scale=0.0,
    )
    model = DenseCppLM(cfg)
    model.domain_embedding.stacked_emb.weight = mx.random.normal(
        model.domain_embedding.stacked_emb.weight.shape
    ) * 0.5
    model.domain_embedding.up_proj.weight = mx.random.normal(
        model.domain_embedding.up_proj.weight.shape
    ) * 0.5

    tokens = mx.array(np.arange(8, dtype=np.int32))[None, :] % cfg.vocab_size
    target_tokens = (tokens + 1) % cfg.vocab_size
    domain_one = LMTokenBatch(
        tokens=tokens,
        target_tokens=target_tokens,
        side_channels={
            "domain_routes": {
                "domain_ids": mx.ones((1, 8), dtype=mx.int32),
                "role_ids": mx.ones((1, 8), dtype=mx.int32),
                "confidence_ids": mx.ones((1, 8), dtype=mx.int32),
            }
        },
    )
    domain_two = LMTokenBatch(
        tokens=tokens,
        target_tokens=target_tokens,
        side_channels={
            "domain_routes": {
                "domain_ids": mx.full((1, 8), 2, dtype=mx.int32),
                "role_ids": mx.ones((1, 8), dtype=mx.int32),
                "confidence_ids": mx.ones((1, 8), dtype=mx.int32),
            }
        },
    )

    loss_one, _ = next_token_cross_entropy(model, domain_one)
    loss_two, _ = next_token_cross_entropy(model, domain_two)
    mx.eval(loss_one, loss_two)

    assert abs(float(loss_one.item()) - float(loss_two.item())) > 1e-5


def test_platform_side_channels_change_loss_through_lm_batch():
    cfg = _smoke_config(depth=1, ngram_hash_enabled=False)
    model = DenseCppLM(cfg)
    model.platform_embedding.embedding.weight = mx.random.normal(
        model.platform_embedding.embedding.weight.shape
    ) * 0.5
    tokens = mx.array(np.arange(8, dtype=np.int32))[None, :] % cfg.vocab_size
    targets = (tokens + 1) % cfg.vocab_size
    platform_one = LMTokenBatch(
        tokens=tokens,
        target_tokens=targets,
        platform_ids=mx.array([[1, 2]], dtype=mx.int32),
    )
    platform_two = LMTokenBatch(
        tokens=tokens,
        target_tokens=targets,
        platform_ids=mx.array([[3, 4]], dtype=mx.int32),
    )

    loss_one, _ = next_token_cross_entropy(model, platform_one)
    loss_two, _ = next_token_cross_entropy(model, platform_two)
    mx.eval(loss_one, loss_two)

    assert abs(float(loss_one.item()) - float(loss_two.item())) > 1e-5


def test_forward_packet_consumes_domain_graph_edges():
    cfg = _smoke_config(
        depth=1,
        ngram_hash_enabled=False,
        graph_routes_enabled=True,
        graph_attention_bias_beta=25.0,
    )
    model = DenseCppLM(cfg)
    seq = 8
    tokens = mx.array(np.arange(seq, dtype=np.int32))[None, :] % cfg.vocab_size
    targets = (tokens + 1) % cfg.vocab_size
    with_edge = CodePacket(
        token_ids=tokens,
        target_ids=targets,
        domain_edges=DomainEdgeIndex.from_triples([(7, 1, 20)]),
    )
    empty = CodePacket(
        token_ids=tokens,
        target_ids=targets,
        domain_edges=DomainEdgeIndex.empty(),
    )

    logits_edge, loss_edge = model.forward_packet(with_edge)
    logits_empty, loss_empty = model.forward_packet(empty)
    mx.eval(logits_edge, logits_empty, loss_edge, loss_empty)

    assert float(mx.sum(mx.abs(logits_edge - logits_empty)).item()) > 1e-4
    assert abs(float(loss_edge.item()) - float(loss_empty.item())) > 1e-5


def test_dsa_graph_batch_changes_indexer_scores_with_production_beta():
    cfg = _smoke_config(
        depth=1,
        ngram_hash_enabled=False,
        attention_mode="dsa",
        graph_routes_enabled=True,
        graph_attention_bias_beta=1.0,
        attention_sparse_topk=2,
        indexer_local_window=0,
        indexer_num_sinks=0,
    )
    model = DenseCppLM(cfg)
    tokens = mx.array(np.arange(8, dtype=np.int32))[None, :] % cfg.vocab_size
    graph = _single_token_chunk_graph(8, [[7, 1]])
    empty = _single_token_chunk_graph(8, [])

    graph_scores = model.indexer_scores(tokens, graph_batch=graph)[0]
    mx.eval(graph_scores)
    graph_scores_np = np.asarray(graph_scores).copy()
    empty_scores = model.indexer_scores(tokens, graph_batch=empty)[0]
    mx.eval(empty_scores)

    assert graph_scores_np[0, 7, 1] - np.asarray(empty_scores)[0, 7, 1] == pytest.approx(1.0)


def test_dsa_seam_rebuilds_without_restructuring():
    # The clean DSA seam: same config code path, attention_mode='dsa' swaps GQA
    # for the Sparse-MLA-capable attention block. Construction must succeed and
    # the block's live mode must be 'dsa'.
    cfg = _smoke_config(attention_mode="dsa")
    model = DenseCppLM(cfg)
    assert all(layer.attention.config.mode == "dsa" for layer in model.layers)
    # dsa block still owns the GQA head structure (q=8, kv=2).
    assert model.layers[0].attention.config.num_q_heads == 8
    assert model.layers[0].attention.config.kv_heads == 2


def test_dense_block_is_attention_plus_swiglu():
    # The block reuses our CausalSelfAttention + SwiGLU FeedForwardExpert leaves.
    from cppmega_mlx.nn.attention import CausalSelfAttention
    from cppmega_mlx.nn.moe import FeedForwardExpert

    cfg = _smoke_config()
    block = DenseCppBlock(cfg)
    assert isinstance(block.attention, CausalSelfAttention)
    assert isinstance(block.ffn, FeedForwardExpert)
    assert block.ffn.activation == "swiglu"
    assert block.ffn.up_proj is not None  # SwiGLU has gate + up + down


@pytest.mark.slow
def test_real_profile_core_is_about_500m():
    # Default config IS the real ~500M Stage-1 profile.
    cfg = DenseCppLMConfig()
    assert cfg.hidden_size == 1280
    assert cfg.depth == 24
    assert cfg.ffn_hidden_size == 3456
    assert cfg.vocab_size == 65536
    assert cfg.num_query_heads == 20
    assert cfg.num_kv_heads == 4
    assert cfg.head_dim == 64
    model = DenseCppLM(cfg)
    from mlx.utils import tree_flatten

    def count(mod):
        return sum(
            int(v.size)
            for _n, v in tree_flatten(mod.parameters())
            if isinstance(v, mx.array)
        )

    total = model.num_parameters(count_tied_once=True)
    core = total - count(model.ngram_hash_embedding)
    # Transformer core (everything except the n-gram hash feature table) is the
    # quoted model size and should land in the ~500M band.
    assert 480e6 < core < 520e6, f"core params {core/1e6:.1f}M outside ~500M band"
