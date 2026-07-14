from __future__ import annotations

import argparse
import inspect
from pathlib import Path
import subprocess
import sys

import mlx.core as mx
import pytest

from cppmega_mlx.data.batch import LMTokenBatch
from cppmega_mlx.data.domain_schema import DomainEdgeKind
from cppmega_mlx.data.graph_packet import EdgeIndex, GraphBatch, GraphPacket
from cppmega_mlx.models.dense_cpp_lm import GraphIndexedAttention
from cppmega_mlx.training.stage1_production import (
    STAGE1_GRAPH_DOMAIN_RECIPE,
    add_stage1_production_arguments,
    build_stage1_production_model,
    run_stage1_graph_domain_production,
    stage1_production_batch_receipt,
    stage1_production_config,
)
ROOT = Path(__file__).resolve().parents[1]


def _production_batch(
    *,
    include_graph: bool = True,
    include_domain: bool = True,
    domain_id: int = 3,
    include_edge_kinds: bool = True,
    include_document_ids: bool = True,
    edge_kind: DomainEdgeKind | int = DomainEdgeKind.DIAG_PRIMARY_LOCATION,
) -> LMTokenBatch:
    tokens = mx.array([[2, 3, 5, 7, 11, 13, 17, 19]], dtype=mx.int32)
    graph = None
    if include_graph:
        graph = GraphBatch(
            graphs=(
                GraphPacket(
                    edges={
                        "domain": EdgeIndex.from_pairs(
                            [(6, 5)], relation="domain", num_nodes=8
                        )
                    },
                    num_nodes=8,
                ),
            ),
            chunk_starts=(mx.arange(8, dtype=mx.int32),),
            chunk_ends=(mx.arange(1, 9, dtype=mx.int32),),
            edge_kinds=(
                {
                    "domain": mx.array(
                        [int(edge_kind)],
                        dtype=mx.int32,
                    )
                }
                if include_edge_kinds
                else {},
            ),
        )
    side_channels = None
    if include_domain:
        side_channels = {
            "domain_routes": {
                "domain_ids": mx.full(tokens.shape, domain_id, dtype=mx.int32),
                "role_ids": mx.full(tokens.shape, 2, dtype=mx.int32),
                "confidence_ids": mx.full(tokens.shape, 1, dtype=mx.int32),
            }
        }
    return LMTokenBatch(
        tokens=tokens,
        document_ids=(
            mx.array([[0, 0, 0, 0, 1, 1, 1, 1]], dtype=mx.int32)
            if include_document_ids
            else None
        ),
        graph_batch=graph,
        side_channels=side_channels,
    )


def test_stage1_production_config_enables_reference_graph_and_domain_routes() -> None:
    cfg = stage1_production_config(
        vocab_size=64,
        hidden_size=32,
        depth=1,
        ffn_hidden_size=64,
        max_seq_length=8,
        num_query_heads=4,
        num_kv_heads=2,
        head_dim=8,
        ngram_hash_enabled=False,
    )

    assert STAGE1_GRAPH_DOMAIN_RECIPE == "stage1_graph_domain_v1"
    assert cfg.attention_mode == "gqa"
    assert cfg.graph_routes_enabled is True
    assert cfg.require_graph_routes is True
    assert cfg.graph_attention_bias_beta == 1.0
    assert cfg.domain_residual_scale == 1.0
    assert cfg.require_domain_routes is True


def test_stage1_production_receipt_proves_nonzero_graph_and_domain_signal() -> None:
    cfg = stage1_production_config(
        vocab_size=64,
        hidden_size=32,
        depth=1,
        ffn_hidden_size=64,
        max_seq_length=8,
        num_query_heads=4,
        num_kv_heads=2,
        head_dim=8,
        ngram_hash_enabled=False,
    )

    receipt = stage1_production_batch_receipt(_production_batch(), config=cfg)

    assert receipt["recipe"] == STAGE1_GRAPH_DOMAIN_RECIPE
    assert receipt["graph_edges"] == 1
    assert receipt["graph_prior_nonzero"] == 1
    assert receipt["edge_kind_edges"] == 1
    assert receipt["edge_kind_ids"] == [int(DomainEdgeKind.DIAG_PRIMARY_LOCATION)]
    assert receipt["edge_kind_prior_nonzero"] == 1
    assert receipt["domain_tokens_nonzero"] == 8
    assert receipt["document_boundaries"] == 1
    assert receipt["domain_residual_scale"] == 1.0
    assert receipt["graph_attention_bias_beta"] == 1.0
    assert receipt["domain_sidecars"] == [
        "confidence_ids",
        "domain_ids",
        "role_ids",
    ]


def test_stage1_production_accepts_valid_categories_with_zero_kind_delta() -> None:
    cfg = stage1_production_config(
        vocab_size=64,
        hidden_size=32,
        depth=1,
        ffn_hidden_size=64,
        max_seq_length=8,
        num_query_heads=4,
        num_kv_heads=2,
        head_dim=8,
        ngram_hash_enabled=False,
    )

    receipt = stage1_production_batch_receipt(
        _production_batch(edge_kind=DomainEdgeKind.BUILD_TARGET_DEP),
        config=cfg,
    )

    assert receipt["edge_kind_ids"] == [int(DomainEdgeKind.BUILD_TARGET_DEP)]
    assert receipt["edge_kind_edges"] == 1
    assert receipt["edge_kind_prior_nonzero"] == 0


@pytest.mark.parametrize(
    ("batch", "error"),
    [
        (_production_batch(include_graph=False), "typed graph_batch"),
        (_production_batch(include_domain=False), "domain sidecars"),
        (_production_batch(domain_id=0), "nonzero domain tokens"),
        (_production_batch(include_edge_kinds=False), "edge-kind sidecars"),
        (_production_batch(edge_kind=999), "unsupported IDs"),
        (_production_batch(include_document_ids=False), "document_ids"),
    ],
)
def test_stage1_production_receipt_fails_closed_on_missing_signal(
    batch: LMTokenBatch, error: str
) -> None:
    cfg = stage1_production_config(
        vocab_size=64,
        hidden_size=32,
        depth=1,
        ffn_hidden_size=64,
        max_seq_length=8,
        num_query_heads=4,
        num_kv_heads=2,
        head_dim=8,
        ngram_hash_enabled=False,
    )

    with pytest.raises(ValueError, match=error):
        stage1_production_batch_receipt(batch, config=cfg)


def test_stage1_production_dsa_builds_graph_indexer_not_dense_alias() -> None:
    model = build_stage1_production_model(
        attention_mode="dsa",
        vocab_size=64,
        hidden_size=32,
        depth=2,
        ffn_hidden_size=64,
        max_seq_length=8,
        num_query_heads=4,
        num_kv_heads=2,
        head_dim=8,
        attention_sparse_topk=4,
        ngram_hash_enabled=False,
    )

    assert all(
        isinstance(layer.attention, GraphIndexedAttention) for layer in model.layers
    )
    assert all(layer.attention.config.mode == "dsa" for layer in model.layers)


def test_stage1_production_cli_does_not_offer_mla() -> None:
    parser = argparse.ArgumentParser()
    add_stage1_production_arguments(parser)

    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--production-graph-domain-data",
                "/tmp/data",
                "--production-attention-mode",
                "mla",
            ]
        )


def test_named_stage1_trainers_route_explicit_recipe_through_canonical_runner() -> None:
    for script in ("train_stage1.py", "train_eval_stage1.py", "train_realshard.py"):
        source = (ROOT / "scripts" / script).read_text(encoding="utf-8")
        assert "add_stage1_production_arguments" in source
        assert "run_stage1_graph_domain_production" in source


def test_stage1_runner_requires_and_uses_only_production_bundle_ingress() -> None:
    signature = inspect.signature(run_stage1_graph_domain_production)
    for name in ("bucket", "expected_bundle_id", "restore_receipt"):
        assert signature.parameters[name].default is inspect.Parameter.empty

    source = inspect.getsource(run_stage1_graph_domain_production)
    assert source.count("open_production_megatron_bundle(") == 1
    assert "open_megatron_indexed_dataset" not in source
    assert "provenance_receipt()" in source


def test_named_stage1_trainers_import_in_fresh_subprocess() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import scripts.train_stage1; "
                "import scripts.train_eval_stage1; "
                "import scripts.train_realshard"
            ),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_stage1_yaml_declares_fail_closed_graph_domain_recipe() -> None:
    text = (
        ROOT / "configs" / "stage1_cpp_foundation_dense500m.yaml"
    ).read_text(encoding="utf-8")

    assert f"name: {STAGE1_GRAPH_DOMAIN_RECIPE}" in text
    assert "beta: 1.0" in text
    assert "residual_scale: 1.0" in text
    assert "ingress: megatron_indexed" in text
    assert "required_sidecars:" in text
    assert "schema: cppmega_graph_routes_v2" in text
    assert "token_domain_edges" in text
    for required_key in (
        "token_chunk_kinds",
        "token_chunk_dep_levels",
        "token_call_edges",
        "token_type_edges",
        "graph_sidecar_schema",
        "graph_sidecar_paths",
    ):
        assert required_key in text
    assert "document_ids" in text
    assert "require_nonzero_document_boundaries: true" in text
    assert "require_valid_edge_kind_ids: true" in text
    assert "require_nonzero_edge_kind_prior" not in text
