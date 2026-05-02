from __future__ import annotations

import importlib
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any

import mlx.core as mx
import mlx.optimizers as optim
import numpy as np
import pytest

from cppmega_mlx.data.parquet_dataset import TokenParquetDataset
from cppmega_mlx.models.hybrid_lm import HybridTinyConfig, HybridTinyLM
from cppmega_mlx.training.compiled import CompiledPretrainingStep
from cppmega_mlx.training.eval import evaluate_batches
from cppmega_mlx.training.loss import next_token_cross_entropy


REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_HYBRID_TINY = REPO_ROOT / "scripts" / "train_hybrid_tiny.py"
GB10_SAMPLE_ROOT = REPO_ROOT / "data" / "parquet_samples" / "gb10"
REAL_COLUMNS = (
    "token_ids",
    "structure_ids",
)
EXPECTED_GB10_COLUMN_TYPES = {
    "token_ids": "large_list<element: uint32>",
    "structure_ids": "large_list<element: int8>",
}
STRUCTURE_MODEL_KWARG_KEYS = (
    "structure_ids",
    "dep_levels",
    "ast_depth_ids",
    "sibling_index_ids",
    "node_type_ids",
)


def test_local_parquet_sample_tree_is_git_ignored() -> None:
    result = subprocess.run(
        ["git", "check-ignore", "-q", "--", "data/parquet_samples"],
        cwd=REPO_ROOT,
        check=False,
    )

    assert result.returncode == 0


@pytest.mark.parametrize(
    ("dataset_name", "expected_min_rows"),
    [
        ("clang_semantic_4k_v10", 4),
        ("clang_commits_4k_v1", 4),
    ],
)
def test_gb10_real_parquet_schema_loads_token_only_training_columns(
    tmp_path: Path, dataset_name: str, expected_min_rows: int
) -> None:
    source_path = GB10_SAMPLE_ROOT / dataset_name / "val_00000.parquet"
    if not source_path.exists():
        pytest.skip(f"GB10 parquet sample is not present: {source_path}")

    pa, pq = _pyarrow_modules()
    sample_path = tmp_path / f"{dataset_name}_head.parquet"
    table = _copy_head_rows(
        pa=pa,
        pq=pq,
        source_path=source_path,
        sample_path=sample_path,
        row_count=expected_min_rows,
    )

    schema_names = set(table.column_names)
    missing = sorted(set(REAL_COLUMNS) - schema_names)
    assert missing == []
    assert _column_types(table, REAL_COLUMNS) == EXPECTED_GB10_COLUMN_TYPES

    first_token_ids = table["token_ids"].to_pylist()[0]
    first_source_structure = table["structure_ids"].to_pylist()[0]
    assert len(first_token_ids) >= 128
    assert len(first_source_structure) != len(first_token_ids)

    dataset = TokenParquetDataset(
        sample_path,
        seq_len=128,
        batch_size=2,
        token_key="token_ids",
    )
    batch = next(dataset.iter_batches())

    assert dataset.num_samples >= 2
    assert dataset.metadata.source_format == "parquet"
    assert dataset.parquet_receipt["column_types"] == EXPECTED_GB10_COLUMN_TYPES
    assert dataset.parquet_receipt["token_source"] == {
        "mode": "token_column",
        "column": "token_ids",
        "type": "large_list<element: uint32>",
    }
    assert dataset.parquet_receipt["side_channel_sources"] == {}
    assert dataset.parquet_receipt["skipped_side_channel_columns"] == [
        {
            "field": "structure_ids",
            "column": "structure_ids",
            "type": "large_list<element: int8>",
            "reason": "not_token_aligned",
        },
    ]
    assert tuple(batch.tokens.shape) == (2, 128)
    token_min, token_max = dataset.token_id_range()
    assert 0 <= token_min <= token_max <= np.iinfo(np.int32).max

    np.testing.assert_array_equal(np.array(batch.tokens[0]), first_token_ids[:128])
    assert batch.structure_ids is None
    assert batch.model_kwargs() == {}
    for field_name in (
        *STRUCTURE_MODEL_KWARG_KEYS,
    ):
        assert getattr(batch, field_name) is None


@pytest.mark.parametrize(
    "dataset_name",
    [
        "clang_semantic_4k_v10",
        "clang_commits_4k_v1",
    ],
)
def test_gb10_real_parquet_token_only_batches_reach_training_loss(
    tmp_path: Path, dataset_name: str
) -> None:
    source_path = GB10_SAMPLE_ROOT / dataset_name / "val_00000.parquet"
    if not source_path.exists():
        pytest.skip(f"GB10 parquet sample is not present: {source_path}")

    pa, pq = _pyarrow_modules()
    sample_path = tmp_path / f"{dataset_name}_train_head.parquet"
    _copy_head_rows(
        pa=pa,
        pq=pq,
        source_path=source_path,
        sample_path=sample_path,
        row_count=4,
    )

    dataset = TokenParquetDataset(
        sample_path,
        seq_len=128,
        batch_size=2,
        token_key="token_ids",
    )
    batch = next(dataset.iter_batches())
    token_min, token_max = dataset.token_id_range()
    assert 0 <= token_min <= token_max

    kwargs = batch.model_kwargs()
    assert kwargs == {}

    config = HybridTinyConfig(
        vocab_size=max(64, token_max + 1),
        hidden_size=8,
        pattern="M",
        depth=1,
        num_attention_heads=1,
        max_seq_length=127,
        structure_components="all",
        structure_bottleneck_dim=8,
        mamba_expand=1,
        mamba_head_dim=4,
        mamba_state_dim=4,
        mamba_groups=1,
        mamba_chunk_size=4,
        ngram_hash_enabled=True,
        ngram_hash_orders=(2,),
        ngram_hash_heads=1,
        ngram_hash_table_size=257,
        ngram_hash_embed_dim=4,
        ngram_hash_seed=17,
    )
    model = HybridTinyLM(config)

    loss, ntokens = next_token_cross_entropy(model, batch)
    mx.eval(loss, ntokens)

    assert int(ntokens.item()) == 2 * 127
    assert math.isfinite(float(loss.item()))
    assert float(loss.item()) > 0


@pytest.mark.parametrize(
    "dataset_name",
    [
        "clang_semantic_4k_v10",
        "clang_commits_4k_v1",
    ],
)
def test_gb10_real_parquet_runs_one_local_train_and_eval_step(
    tmp_path: Path, dataset_name: str
) -> None:
    source_path = GB10_SAMPLE_ROOT / dataset_name / "val_00000.parquet"
    if not source_path.exists():
        pytest.skip(f"GB10 parquet sample is not present: {source_path}")

    pa, pq = _pyarrow_modules()
    sample_path = tmp_path / f"{dataset_name}_local_train_eval_head.parquet"
    _copy_head_rows(
        pa=pa,
        pq=pq,
        source_path=source_path,
        sample_path=sample_path,
        row_count=4,
    )

    dataset = TokenParquetDataset(
        sample_path,
        seq_len=128,
        batch_size=2,
        token_key="token_ids",
    )
    train_batch = next(dataset.iter_batches())
    eval_batch = next(dataset.iter_batches())

    kwargs = train_batch.model_kwargs()
    assert kwargs == {}

    _, token_max = dataset.token_id_range()
    config = _real_parquet_tiny_config(token_max=token_max)
    mx.random.seed(1000 + sum(ord(char) for char in dataset_name))
    model = HybridTinyLM(config)
    optimizer = optim.AdamW(learning_rate=1e-3)
    step = CompiledPretrainingStep(model, optimizer, compile=False)

    before = evaluate_batches(model, [eval_batch])
    metrics = step(train_batch)
    after = evaluate_batches(model, [eval_batch])

    assert before.batches == 1
    assert before.ntokens == 2 * 127
    assert math.isfinite(before.loss)
    assert metrics.updated is True
    assert metrics.compiled is False
    assert metrics.step == 1
    assert metrics.ntokens == 2 * 127
    assert metrics.trained_tokens == 2 * 127
    assert math.isfinite(metrics.loss)
    assert after.batches == 1
    assert after.ntokens == 2 * 127
    assert math.isfinite(after.loss)


@pytest.mark.parametrize(
    "dataset_name",
    [
        "clang_semantic_4k_v10",
        "clang_commits_4k_v1",
    ],
)
def test_gb10_real_parquet_train_hybrid_tiny_cli_trains_and_evals(
    tmp_path: Path, dataset_name: str
) -> None:
    source_path = GB10_SAMPLE_ROOT / dataset_name / "val_00000.parquet"
    if not source_path.exists():
        pytest.skip(f"GB10 parquet sample is not present: {source_path}")

    pa, pq = _pyarrow_modules()
    sample_path = tmp_path / f"{dataset_name}_local_train_eval_head.parquet"
    _copy_head_rows(
        pa=pa,
        pq=pq,
        source_path=source_path,
        sample_path=sample_path,
        row_count=4,
    )

    result = subprocess.run(
        [
            sys.executable,
            str(TRAIN_HYBRID_TINY),
            str(sample_path),
            "--json",
            "--data-format",
            "parquet",
            "--token-key",
            "token_ids",
            "--batch-size",
            "1",
            "--seq-len",
            "64",
            "--steps",
            "1",
            "--hidden-size",
            "8",
            "--num-attention-heads",
            "1",
            "--pattern",
            "M",
            "--depth",
            "1",
            "--mamba-expand",
            "1",
            "--mamba-head-dim",
            "4",
            "--mamba-state-dim",
            "4",
            "--mamba-groups",
            "1",
            "--mamba-chunk-size",
            "4",
            "--vocab-size",
            "131072",
            "--valid-dataset-path",
            str(sample_path),
            "--valid-dataset-format",
            "parquet",
            "--eval-batches",
            "1",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
        timeout=60,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["status"] == "ok"
    assert payload["synthetic_npz"] is False
    assert payload["route_symbols"] == "M"
    assert payload["route_roles"] == ["mamba3"]
    assert payload["tokens_per_step"] == 63
    assert payload["trained_tokens"] == 63
    assert math.isfinite(payload["final_loss"])
    assert payload["final_loss"] > 0
    assert payload["step_metrics"][0]["ntokens"] == 63

    _assert_cli_dataset_receipt(payload["dataset"], sample_path, dataset_name)

    evaluation = payload["evaluation"]
    assert evaluation["requested_batches"] == 1
    assert evaluation["evaluated_batches"] == 1
    assert evaluation["metrics"]["batches"] == 1
    assert evaluation["metrics"]["ntokens"] == 63
    assert math.isfinite(evaluation["metrics"]["loss"])
    assert evaluation["metrics"]["loss"] > 0
    _assert_cli_dataset_receipt(evaluation["dataset"], sample_path, dataset_name)


def test_gb10_real_parquet_sample_receipt_names_available_samples() -> None:
    found = sorted(
        str(path.relative_to(REPO_ROOT))
        for path in GB10_SAMPLE_ROOT.glob("*/val_00000.parquet")
    )

    if not found:
        pytest.skip(f"GB10 parquet samples are not present under {GB10_SAMPLE_ROOT}")

    assert found == [
        "data/parquet_samples/gb10/clang_commits_4k_v1/val_00000.parquet",
        "data/parquet_samples/gb10/clang_semantic_4k_v10/val_00000.parquet",
    ]


def _pyarrow_modules() -> tuple[Any, Any]:
    try:
        return (
            importlib.import_module("pyarrow"),
            importlib.import_module("pyarrow.parquet"),
        )
    except ModuleNotFoundError as error:
        if error.name and error.name.startswith("pyarrow"):
            pytest.skip("pyarrow is required for real parquet sample tests")
        raise


def _copy_head_rows(
    *,
    pa: Any,
    pq: Any,
    source_path: Path,
    sample_path: Path,
    row_count: int,
) -> Any:
    parquet_file = pq.ParquetFile(source_path)
    batch = next(
        parquet_file.iter_batches(batch_size=row_count, columns=list(REAL_COLUMNS))
    )
    table = pa.Table.from_batches([batch])
    pq.write_table(table, sample_path)
    return table


def _column_types(table: Any, names: tuple[str, ...]) -> dict[str, str]:
    return {name: str(table.schema.field(name).type) for name in names}


def _assert_cli_dataset_receipt(
    dataset_payload: dict[str, Any],
    sample_path: Path,
    dataset_name: str,
) -> None:
    assert dataset_payload["path"] == str(sample_path)
    assert dataset_payload["metadata"]["source_format"] == "parquet"
    assert dataset_payload["token_key"] == "token_ids"
    assert dataset_payload["side_channels"] == []
    assert dataset_payload["side_channel_contract"]["structure_side_channels"] == {
        "attention_mask_is_loss_only": False,
        "batch_slice": "tokens[:, :-1]",
        "model_kwarg_names": list(STRUCTURE_MODEL_KWARG_KEYS),
        "threaded_to_model": [],
    }

    receipt = dataset_payload["dataset_receipt"]
    assert receipt == {
        "batch_size": 1,
        "dropped_samples": dataset_payload["dropped_samples"],
        "num_batches": dataset_payload["num_batches"],
        "num_samples": dataset_payload["num_samples"],
        "parquet_receipt": receipt["parquet_receipt"],
        "seq_len": 64,
        "side_channels": dataset_payload["side_channels"],
        "source_dataset_name": dataset_name,
        "source_format": "parquet",
        "source_path": str(sample_path),
        "token_key": "token_ids",
    }
    assert receipt["parquet_receipt"] == {
        "source_format": "parquet",
        "columns": sorted(REAL_COLUMNS),
        "column_types": EXPECTED_GB10_COLUMN_TYPES,
        "token_source": {
            "mode": "token_column",
            "column": "token_ids",
            "type": "large_list<element: uint32>",
        },
        "side_channel_sources": {},
        "skipped_side_channel_columns": [
            {
                "field": "structure_ids",
                "column": "structure_ids",
                "type": "large_list<element: int8>",
                "reason": "not_token_aligned",
            },
        ],
    }


def _real_parquet_tiny_config(*, token_max: int) -> HybridTinyConfig:
    return HybridTinyConfig(
        vocab_size=max(64, token_max + 1),
        hidden_size=8,
        pattern="M",
        depth=1,
        num_attention_heads=1,
        max_seq_length=127,
        structure_components="all",
        structure_bottleneck_dim=8,
        mamba_expand=1,
        mamba_head_dim=4,
        mamba_state_dim=4,
        mamba_groups=1,
        mamba_chunk_size=4,
        ngram_hash_enabled=True,
        ngram_hash_orders=(2,),
        ngram_hash_heads=1,
        ngram_hash_table_size=257,
        ngram_hash_embed_dim=4,
        ngram_hash_seed=17,
    )
