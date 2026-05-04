from __future__ import annotations

import math
from collections.abc import Mapping

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten

from cppmega_mlx.data.batch import synthetic_token_batch
from cppmega_mlx.models.hybrid_lm import HybridTinyConfig, HybridTinyLM
from cppmega_mlx.nn.mhc import ManifoldBranchMixer
from cppmega_mlx.training.loss import next_token_cross_entropy


def _single_attention_config(*, mhc_enabled: bool) -> HybridTinyConfig:
    return HybridTinyConfig(
        vocab_size=32,
        hidden_size=8,
        pattern="A",
        depth=1,
        dsa_a_layer_ranks=(0,),
        num_attention_heads=1,
        max_seq_length=8,
        mamba_expand=1,
        mamba_head_dim=4,
        mamba_state_dim=4,
        mamba_groups=1,
        mamba_chunk_size=4,
        moe_num_experts=4,
        moe_top_k=4,
        moe_expert_hidden_size=16,
        moe_shared_expert_hidden_size=8,
        m2rnn_k_head_dim=2,
        m2rnn_v_head_dim=2,
        m2rnn_num_v_heads=1,
        m2rnn_num_f_heads=1,
        m2rnn_num_weight_heads=1,
        m2rnn_chunk_size=4,
        mhc_enabled=mhc_enabled,
    )


def _flat_tree(tree: Mapping[str, object]) -> dict[str, np.ndarray]:
    mx.eval(tree)
    return {name: np.array(value) for name, value in tree_flatten(tree)}


def _max_abs(tree: dict[str, np.ndarray], name: str) -> float:
    return float(np.max(np.abs(tree[name])))


def test_hybrid_lm_mhc_opt_in_installs_real_mixer_on_each_layer() -> None:
    base = HybridTinyLM(_single_attention_config(mhc_enabled=False))
    mhc_model = HybridTinyLM(_single_attention_config(mhc_enabled=True))

    assert base.layers[0].mhc is None
    assert isinstance(mhc_model.layers[0].mhc, ManifoldBranchMixer)
    assert mhc_model.layers[0].mhc.config.max_branches == 2
    assert mhc_model.layers[0].mhc.config.hidden_size == mhc_model.config.hidden_size


def test_hybrid_lm_mhc_changes_real_forward_path_from_residual_add() -> None:
    batch = synthetic_token_batch(
        batch_size=2,
        seq_length=6,
        vocab_size=_single_attention_config(mhc_enabled=False).vocab_size,
        seed=910,
        include_structure=True,
    )

    mx.random.seed(911)
    base = HybridTinyLM(_single_attention_config(mhc_enabled=False))
    mx.random.seed(911)
    mhc_model = HybridTinyLM(_single_attention_config(mhc_enabled=True))

    base_logits = base(
        batch.tokens,
        structure_ids=batch.structure_ids,
        dep_levels=batch.dep_levels,
        ast_depth_ids=batch.ast_depth_ids,
        sibling_index_ids=batch.sibling_index_ids,
        node_type_ids=batch.node_type_ids,
    )
    mhc_logits = mhc_model(
        batch.tokens,
        structure_ids=batch.structure_ids,
        dep_levels=batch.dep_levels,
        ast_depth_ids=batch.ast_depth_ids,
        sibling_index_ids=batch.sibling_index_ids,
        node_type_ids=batch.node_type_ids,
    )
    mx.eval(base_logits, mhc_logits)

    assert base_logits.shape == mhc_logits.shape == (2, 6, base.config.vocab_size)
    assert np.isfinite(np.array(mhc_logits)).all()
    assert not np.allclose(np.array(base_logits), np.array(mhc_logits))


def test_hybrid_lm_mhc_receives_gradients_and_optimizer_updates() -> None:
    mx.random.seed(920)
    model = HybridTinyLM(_single_attention_config(mhc_enabled=True))
    optimizer = optim.AdamW(learning_rate=1e-3, weight_decay=0.0)
    batch = synthetic_token_batch(
        batch_size=2,
        seq_length=6,
        vocab_size=model.config.vocab_size,
        seed=921,
        include_structure=True,
    )
    before = _flat_tree(model.parameters())

    (loss, ntokens), grads = nn.value_and_grad(model, next_token_cross_entropy)(model, batch)
    optimizer.update(model, grads)
    mx.eval(loss, ntokens, model.parameters(), optimizer.state)
    flat_grads = _flat_tree(grads)
    after = _flat_tree(model.parameters())
    optimizer_state = _flat_tree(optimizer.state)

    assert math.isfinite(float(loss.item()))
    assert int(ntokens.item()) == 10
    for param_name in ("layers.0.mhc.score_proj.weight", "layers.0.mhc.score_out.weight"):
        assert param_name in flat_grads
        assert np.isfinite(flat_grads[param_name]).all(), param_name
        assert _max_abs(flat_grads, param_name) > 0.0
        assert _max_abs({param_name: after[param_name] - before[param_name]}, param_name) > 0.0
        for suffix in (".m", ".v"):
            state_name = f"{param_name}{suffix}"
            assert state_name in optimizer_state
            assert np.isfinite(optimizer_state[state_name]).all(), state_name
            assert _max_abs(optimizer_state, state_name) > 0.0
