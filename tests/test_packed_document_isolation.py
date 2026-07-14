"""Focused regressions for packed-document isolation in the MLX runtime."""

from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from cppmega_mlx.data.code_packet import CodePacket
from cppmega_mlx.data.graph_packet import EdgeIndex
from cppmega_mlx.training.objective_mixer import (
    GraphAuxLossConfig,
    ObjectiveSource,
    production_training_loss,
)
from cppmega_mlx.training.objectives import build_causal_lm
from cppmega_mlx.training.task_mixer import TaskKind
from scripts.train_eval_stage1 import CHANNELS, _materialize_batch, _val_batch


def _array(values: list[int]) -> mx.array:
    return mx.array(np.asarray(values, dtype=np.int32))


def _aligned_packet(*, document_ids: mx.array | None) -> CodePacket:
    zeros = _array([0, 0, 0, 0, 0])
    return CodePacket(
        token_ids=_array([10, 11, 12, 13, 14]),
        document_ids=document_ids,
        structure_ids=_array([1, 1, 1, 1, 1]),
        dep_levels=zeros,
        ast_depth=zeros,
        sibling_index=zeros,
        ast_node_type=_array([1, 1, 1, 1, 1]),
        call_edges=EdgeIndex.from_pairs([], relation="call", num_nodes=2),
        type_edges=EdgeIndex.from_pairs([], relation="type", num_nodes=2),
        chunk_starts=_array([0, 2]),
        chunk_ends=_array([2, 5]),
        chunk_kinds=_array([1, 1]),
        chunk_dep_levels=_array([0, 0]),
    )


def test_objective_batch_preserves_document_ids_and_masks_boundary_target() -> None:
    packet = _aligned_packet(document_ids=_array([41, 41, 73, 73, 73]))
    example = build_causal_lm(packet)

    batch = _materialize_batch(
        TaskKind.CAUSAL_LM,
        [(example, ObjectiveSource(code_packet=packet))],
        seq_len=6,
    )

    assert np.asarray(batch.document_ids).tolist() == [[41, 41, 73, 73, 0, 0]]
    assert np.asarray(batch.loss_mask).tolist() == [[1.0, 0.0, 1.0, 1.0, 0.0, 0.0]]


def test_aligned_objective_batch_rejects_missing_document_ids() -> None:
    packet = _aligned_packet(document_ids=None)
    example = build_causal_lm(packet)

    with pytest.raises(ValueError, match="required aligned channel document_ids"):
        _materialize_batch(
            TaskKind.CAUSAL_LM,
            [(example, ObjectiveSource(code_packet=packet))],
            seq_len=6,
        )


def test_validation_batch_masks_cross_document_next_token_loss() -> None:
    row = {
        "token_ids": [10, 11, 12, 13, 14],
        "doc_ids": [1, 1, 2, 2, 2],
    }
    for source, _target in CHANNELS:
        row[source] = [0, 0, 0, 0, 0]

    _inputs, _targets, loss_mask, document_ids, _side = _val_batch(
        [row], [0], 4
    )

    assert np.asarray(document_ids).tolist() == [[1, 1, 2, 2]]
    assert np.asarray(loss_mask).tolist() == [[1.0, 0.0, 1.0, 1.0]]


def test_production_objective_requires_document_ids_before_model_forward() -> None:
    values = mx.array([[1, 2]], dtype=mx.int32)
    model = SimpleNamespace(
        config=SimpleNamespace(
            attention_mode="dsa",
            require_graph_routes=True,
            graph_routes_enabled=True,
            structure_residual_scale=1.0,
        )
    )
    side_channels = {
        name: mx.zeros_like(values)
        for name in (
            "structure_ids",
            "dep_levels",
            "ast_depth_ids",
            "sibling_index_ids",
            "node_type_ids",
        )
    }

    with pytest.raises(ValueError, match="requires document_ids"):
        production_training_loss(
            model,
            values,
            values,
            mx.ones_like(values),
            side_channels=side_channels,
            document_ids=None,
            block_bias=mx.zeros((1, 2, 2)),
            graph_targets=mx.zeros((1, 2, 2)),
            graph_pair_mask=mx.ones((1, 2, 2)),
            graph_config=GraphAuxLossConfig(relations=("call",)),
            graph_weight=1.0,
        )
