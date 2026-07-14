from __future__ import annotations

import argparse
import importlib
import inspect
import subprocess
import sys
from pathlib import Path

import mlx.core as mx
import pytest

from cppmega_mlx.data.batch import LMTokenBatch
from cppmega_mlx.data.domain_schema import DomainEdgeKind
from cppmega_mlx.data.graph_packet import EdgeIndex, GraphBatch, GraphPacket
from cppmega_mlx.data.graph_recipe import STAGE1_GRAPH_RELATIONS
from cppmega_mlx.models.dense_cpp_lm import GraphIndexedAttention
from cppmega_mlx.training.stage1_production import (
    STAGE1_GRAPH_DOMAIN_RECIPE,
    _batch_route_counts,
    add_stage1_production_arguments,
    build_stage1_production_model,
    run_stage1_graph_domain_production,
    stage1_production_batch_receipt,
    stage1_production_config,
)

ROOT = Path(__file__).resolve().parents[1]
_TRAINER_MODULES = (
    "scripts.train_stage1",
    "scripts.train_eval_stage1",
    "scripts.train_realshard",
)
_BUNDLE_ROOT = Path("/tmp/cppmega-stage1-production-bundle")
_BUNDLE_ID = "cppmega-stage1-bundle-0123456789abcdef"
_RESTORE_RECEIPT = _BUNDLE_ROOT / "restore_receipt.json"
_BUNDLE_PROVENANCE_CLI = {
    "--production-graph-domain-data": str(_BUNDLE_ROOT),
    "--production-bucket": "4096",
    "--production-expected-bundle-id": _BUNDLE_ID,
    "--production-restore-receipt": str(_RESTORE_RECEIPT),
}


def _bundle_cli_args(*, missing: str | None = None) -> list[str]:
    return [
        item
        for flag, value in _BUNDLE_PROVENANCE_CLI.items()
        if flag != missing
        for item in (flag, value)
    ]


def _production_batch(
    *,
    include_graph: bool = True,
    include_domain: bool = True,
    domain_id: int = 3,
    include_edge_kinds: bool = True,
    include_document_ids: bool = True,
    with_edge: bool = True,
    edge_kind: DomainEdgeKind | int = DomainEdgeKind.DIAG_PRIMARY_LOCATION,
) -> LMTokenBatch:
    tokens = mx.array([[2, 3, 5, 7, 11, 13, 17, 19]], dtype=mx.int32)
    graph = None
    if include_graph:
        pairs = [(6, 5)] if with_edge else []
        graph = GraphBatch(
            graphs=(
                GraphPacket(
                    edges={
                        "domain": EdgeIndex.from_pairs(
                            pairs, relation="domain", num_nodes=8
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
                        [int(edge_kind)] if with_edge else [],
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


def test_stage1_production_receipt_and_counts_accept_graphless_batch() -> None:
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
    batch = _production_batch(with_edge=False)

    receipt = stage1_production_batch_receipt(batch, config=cfg)
    counts = _batch_route_counts(batch)

    assert receipt["graph_edges"] == 0
    assert receipt["graph_prior_nonzero"] == 0
    assert receipt["edge_kind_edges"] == 0
    assert receipt["edge_kind_ids"] == []
    assert counts["graph_edges"] == 0
    assert counts["graph_prior_nonzero"] == 0
    assert counts["edge_kind_edges"] == 0


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


def test_stage1_eval_cli_accepts_canonical_graph_relations() -> None:
    module = importlib.import_module("scripts.train_eval_stage1")
    parse_graph_relations = getattr(module, "_parse_graph_relations", None)

    assert callable(parse_graph_relations)
    assert parse_graph_relations(",".join(STAGE1_GRAPH_RELATIONS)) == (
        STAGE1_GRAPH_RELATIONS
    )


@pytest.mark.parametrize(
    ("module_name", "trainer_args", "expected_call"),
    [
        (
            "scripts.train_stage1",
            ["--steps", "9", "--seed", "17", "--lr", "0.0007"],
            {
                "data_path": _BUNDLE_ROOT,
                "bucket": 4096,
                "expected_bundle_id": _BUNDLE_ID,
                "restore_receipt": _RESTORE_RECEIPT,
                "steps": 9,
                "batch_size": 4,
                "seq_len": 4096,
                "hidden_size": 1280,
                "depth": 24,
                "ffn_hidden_size": 3456,
                "learning_rate": 0.0007,
                "seed": 17,
                "attention_mode": "dsa",
            },
        ),
        (
            "scripts.train_eval_stage1",
            [
                "--steps",
                "9",
                "--batch",
                "3",
                "--seq-len",
                "4096",
                "--hidden",
                "160",
                "--depth",
                "3",
                "--ffn",
                "480",
                "--lr",
                "0.0007",
                "--seed",
                "17",
                "--no-compile",
                "--bf16",
            ],
            {
                "data_path": _BUNDLE_ROOT,
                "bucket": 4096,
                "expected_bundle_id": _BUNDLE_ID,
                "restore_receipt": _RESTORE_RECEIPT,
                "steps": 9,
                "batch_size": 3,
                "seq_len": 4096,
                "hidden_size": 160,
                "depth": 3,
                "ffn_hidden_size": 480,
                "learning_rate": 0.0007,
                "seed": 17,
                "attention_mode": "dsa",
                "compile": False,
                "bf16": True,
            },
        ),
        (
            "scripts.train_realshard",
            [
                "--steps",
                "9",
                "--batch",
                "3",
                "--seq-len",
                "4096",
                "--hidden",
                "160",
                "--depth",
                "3",
                "--seed",
                "17",
                "--bf16",
            ],
            {
                "data_path": _BUNDLE_ROOT,
                "bucket": 4096,
                "expected_bundle_id": _BUNDLE_ID,
                "restore_receipt": _RESTORE_RECEIPT,
                "steps": 9,
                "batch_size": 3,
                "seq_len": 4096,
                "hidden_size": 160,
                "depth": 3,
                "ffn_hidden_size": 3456,
                "learning_rate": 3e-4,
                "seed": 17,
                "attention_mode": "dsa",
                "bf16": True,
            },
        ),
    ],
)
def test_named_stage1_trainers_pass_exact_bundle_provenance_to_runner(
    module_name: str,
    trainer_args: list[str],
    expected_call: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = importlib.import_module(module_name)
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        module,
        "run_stage1_graph_domain_production",
        lambda **kwargs: calls.append(kwargs),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            module_name,
            *_bundle_cli_args(),
            "--production-attention-mode",
            "dsa",
            *trainer_args,
        ],
    )

    module.main()

    assert calls == [expected_call]


@pytest.mark.parametrize("module_name", _TRAINER_MODULES)
@pytest.mark.parametrize("missing_flag", tuple(_BUNDLE_PROVENANCE_CLI))
def test_named_stage1_trainers_reject_incomplete_bundle_mode(
    module_name: str,
    missing_flag: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = importlib.import_module(module_name)

    def unexpected_runner_call(**_kwargs: object) -> None:
        pytest.fail("incomplete production bundle arguments reached the runner")

    monkeypatch.setattr(
        module,
        "run_stage1_graph_domain_production",
        unexpected_runner_call,
    )
    monkeypatch.setattr(
        sys, "argv", [module_name, *_bundle_cli_args(missing=missing_flag)]
    )

    with pytest.raises(SystemExit) as error:
        module.main()

    assert error.value.code == 2
    assert missing_flag in capsys.readouterr().err


@pytest.mark.parametrize("module_name", _TRAINER_MODULES)
def test_named_stage1_trainers_reject_legacy_data_only_bundle_mode(
    module_name: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = importlib.import_module(module_name)

    def unexpected_runner_call(**_kwargs: object) -> None:
        pytest.fail("legacy production bundle arguments reached the runner")

    monkeypatch.setattr(
        module,
        "run_stage1_graph_domain_production",
        unexpected_runner_call,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [module_name, "--production-graph-domain-data", str(_BUNDLE_ROOT)],
    )

    with pytest.raises(SystemExit) as error:
        module.main()

    stderr = capsys.readouterr().err
    assert error.value.code == 2
    for missing_flag in (
        "--production-bucket",
        "--production-expected-bundle-id",
        "--production-restore-receipt",
    ):
        assert missing_flag in stderr


def test_stage1_runner_requires_and_uses_only_production_bundle_ingress() -> None:
    signature = inspect.signature(run_stage1_graph_domain_production)
    for name in ("bucket", "expected_bundle_id", "restore_receipt"):
        assert signature.parameters[name].default is inspect.Parameter.empty

    source = inspect.getsource(run_stage1_graph_domain_production)
    assert source.count("open_production_megatron_bundle(") == 1
    assert "open_megatron_indexed_dataset" not in source
    assert "provenance_receipt()" in source


def test_generic_stage1_runner_threads_explicit_edge_kind_bias() -> None:
    module_source = inspect.getsource(
        importlib.import_module("scripts.train_eval_stage1")
    )

    assert "edge_kind_bias=mx.array(edge_kind_bias)" in module_source
    assert "edge_kind_bias=edge_kind_bias if graph_aux_enabled else None" in (
        module_source
    )
    assert "edge_kind_bias=mx.zeros_like(block_bias)" not in module_source


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
