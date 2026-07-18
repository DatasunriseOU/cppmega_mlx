from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cppmega_mlx.data.graph_packet import GraphBatch
from cppmega_mlx.data.domain_schema import DomainEdgeKind
from cppmega_mlx.data.parquet_dataset import TokenParquetDataset
from cppmega_mlx.models.dense_cpp_lm import DenseCppLM, DenseCppLMConfig
from cppmega_mlx.training.compiled import normalize_compiled_batch
from cppmega_mlx.training.loss import next_token_cross_entropy


TOKENS = [1, 2, 3, 4, 5, 6, 7, 8]
CHUNK_STARTS = list(range(8))
CHUNK_ENDS = list(range(1, 9))


def _write_parquet(
    path: Path,
    *,
    edge: tuple[int, int] | None = None,
    include_chunks: bool = True,
    include_domain: bool = False,
) -> None:
    values: dict[str, pa.Array] = {
        "token_ids": pa.array([TOKENS], type=pa.large_list(pa.int32())),
    }
    if include_chunks:
        values.update(
            {
                "token_chunk_starts": pa.array(
                    [CHUNK_STARTS], type=pa.large_list(pa.int32())
                ),
                "token_chunk_ends": pa.array(
                    [CHUNK_ENDS], type=pa.large_list(pa.int32())
                ),
            }
        )
    if edge is not None:
        edge_type = pa.large_list(
            pa.struct([("from", pa.int32()), ("to", pa.int32())])
        )
        values["token_call_edges"] = pa.array(
            [[{"from": edge[0], "to": edge[1]}]], type=edge_type
        )
    if include_domain:
        token_type = pa.large_list(pa.int32())
        values.update(
            {
                "token_domain_ids": pa.array([[1, 1, 2, 2, 1, 1, 2, 2]], type=token_type),
                "token_role_ids": pa.array([[3] * 8], type=token_type),
                "token_confidence_ids": pa.array([[4] * 8], type=token_type),
                "token_entity_ids": pa.array([[5] * 8], type=token_type),
                "token_scope_ids": pa.array([[6] * 8], type=token_type),
            }
        )
    pq.write_table(pa.table(values), path)


def _dataset_batch(path: Path):
    return next(
        TokenParquetDataset(
            path,
            seq_len=len(TOKENS),
            batch_size=1,
            token_key="token_ids",
        ).iter_batches()
    )


def _model_config(**overrides: object) -> DenseCppLMConfig:
    values: dict[str, object] = {
        "vocab_size": 64,
        "hidden_size": 16,
        "depth": 1,
        "ffn_hidden_size": 32,
        "max_seq_length": len(TOKENS),
        "num_query_heads": 2,
        "num_kv_heads": 1,
        "head_dim": 8,
        "ngram_hash_enabled": False,
        "structure_residual_scale": 0.0,
        "platform_residual_scale": 0.0,
    }
    values.update(overrides)
    return DenseCppLMConfig(**values)


def test_real_parquet_domain_aliases_reach_dense_cpp_lm(tmp_path: Path) -> None:
    path = tmp_path / "domain_routes.parquet"
    _write_parquet(path, include_domain=True)
    batch = _dataset_batch(path)

    kwargs = batch.model_kwargs()
    assert {"domain_ids", "role_ids", "confidence_ids"} <= set(kwargs)
    assert batch.side_channels is not None
    assert "token_domain_ids" in batch.side_channels["domain_routes"]
    np.testing.assert_array_equal(
        np.asarray(kwargs["domain_ids"]), [[1, 1, 2, 2, 1, 1, 2]]
    )
    np.testing.assert_array_equal(np.asarray(kwargs["role_ids"]), [[3] * 7])
    np.testing.assert_array_equal(np.asarray(kwargs["confidence_ids"]), [[4] * 7])
    assert set(batch.side_channel_map()["domain_routes"]) == {
        "domain_ids",
        "role_ids",
        "confidence_ids",
        "entity_ids",
        "scope_ids",
    }

    model = DenseCppLM(
        _model_config(
            domain_residual_scale=1.0,
            require_domain_routes=True,
        )
    )
    normalized = normalize_compiled_batch(batch, graph_routes_enabled=False)
    model.validate_training_batch(normalized)
    _logits, loss = model(
        batch.inputs,
        targets=batch.targets,
        **kwargs,
    )
    assert loss is not None
    mx.eval(loss)
    assert np.isfinite(float(loss.item()))


def test_real_parquet_graph_edges_change_bias_forward_and_loss(tmp_path: Path) -> None:
    left_path = tmp_path / "graph_left.parquet"
    right_path = tmp_path / "graph_right.parquet"
    _write_parquet(left_path, edge=(6, 1))
    _write_parquet(right_path, edge=(6, 2))
    left = _dataset_batch(left_path)
    right = _dataset_batch(right_path)

    assert isinstance(left.graph_batch, GraphBatch)
    assert isinstance(right.graph_batch, GraphBatch)
    assert left.graph_batch.graphs[0].edge("call").to_pairs() == [(6, 1)]
    assert right.graph_batch.graphs[0].edge("call").to_pairs() == [(6, 2)]

    left_fixed = normalize_compiled_batch(left, graph_routes_enabled=True)
    right_fixed = normalize_compiled_batch(right, graph_routes_enabled=True)
    left_bias = left_fixed["graph_attention_bias"]
    right_bias = right_fixed["graph_attention_bias"]
    assert left_bias is not None and right_bias is not None
    assert float(left_bias[0, 6, 1].item()) == 1.0
    assert float(right_bias[0, 6, 2].item()) == 1.0
    assert not np.array_equal(np.asarray(left_bias), np.asarray(right_bias))

    model = DenseCppLM(
        _model_config(
            graph_routes_enabled=True,
            graph_attention_bias_beta=25.0,
        )
    )
    left_logits = model.logits(left.inputs, **left.model_kwargs())
    right_logits = model.logits(right.inputs, **right.model_kwargs())
    left_loss = next_token_cross_entropy(model, left)[0]
    right_loss = next_token_cross_entropy(model, right)[0]
    mx.eval(left_logits, right_logits, left_loss, right_loss)

    assert not np.array_equal(np.asarray(left_logits), np.asarray(right_logits))
    assert abs(float(left_loss.item()) - float(right_loss.item())) > 1e-6


def test_real_parquet_domain_graph_edges_reach_typed_edge_kind_bias(
    tmp_path: Path,
) -> None:
    path = tmp_path / "domain_graph.parquet"
    edge_type = pa.large_list(
        pa.struct(
            [
                ("from", pa.int32()),
                ("to", pa.int32()),
                ("kind", pa.int32()),
            ]
        )
    )
    pq.write_table(
        pa.table(
            {
                "token_ids": pa.array(
                    [TOKENS], type=pa.large_list(pa.int32())
                ),
                "token_diagnostic_edges": pa.array(
                    [
                        [
                            {
                                "from": 6,
                                "to": 1,
                                "kind": int(DomainEdgeKind.DIAG_PRIMARY_LOCATION),
                            }
                        ]
                    ],
                    type=edge_type,
                ),
            }
        ),
        path,
    )

    batch = _dataset_batch(path)
    assert batch.graph_batch is not None
    assert batch.graph_batch.graphs[0].edge("diagnostic").to_pairs() == [(6, 1)]
    fixed = normalize_compiled_batch(batch, graph_routes_enabled=True)
    assert float(fixed["graph_attention_bias"][0, 6, 1].item()) == 1.0
    assert float(fixed["graph_edge_kind_bias"][0, 6, 1].item()) == 1.0


@pytest.mark.parametrize(
    ("include_chunks", "edge", "message"),
    [
        (False, (0, 1), "require token_chunk_starts"),
        (True, (8, 1), "outside chunk range"),
    ],
)
def test_real_parquet_graph_routes_fail_closed(
    tmp_path: Path,
    include_chunks: bool,
    edge: tuple[int, int],
    message: str,
) -> None:
    path = tmp_path / "malformed_graph.parquet"
    _write_parquet(path, edge=edge, include_chunks=include_chunks)

    with pytest.raises(ValueError, match=message):
        _dataset_batch(path)
