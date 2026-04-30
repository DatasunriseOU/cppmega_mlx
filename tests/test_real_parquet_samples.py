from __future__ import annotations

import importlib
import math
from pathlib import Path
import subprocess
from typing import Any

import mlx.core as mx
import numpy as np
import pytest

from cppmega_mlx.data.parquet_dataset import TokenParquetDataset
from cppmega_mlx.models.hybrid_lm import HybridTinyConfig, HybridTinyLM
from cppmega_mlx.training.loss import next_token_cross_entropy


REPO_ROOT = Path(__file__).resolve().parents[1]
GB10_SAMPLE_ROOT = REPO_ROOT / "data" / "parquet_samples" / "gb10"
REAL_COLUMNS = (
    "token_ids",
    "structure_ids",
    "token_structure_ids",
    "token_dep_levels",
    "token_ast_depth",
    "token_sibling_index",
    "token_ast_node_type",
)
EXPECTED_GB10_COLUMN_TYPES = {
    "token_ids": "large_list<element: uint32>",
    "structure_ids": "large_list<element: int8>",
    "token_structure_ids": "large_list<element: uint8>",
    "token_dep_levels": "large_list<element: uint16>",
    "token_ast_depth": "large_list<element: uint16>",
    "token_sibling_index": "large_list<element: uint16>",
    "token_ast_node_type": "large_list<element: uint16>",
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
def test_gb10_real_parquet_schema_loads_token_aligned_side_channels(
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
    first_token_structure = table["token_structure_ids"].to_pylist()[0]
    assert len(first_token_ids) >= 128
    assert len(first_token_structure) == len(first_token_ids)
    assert len(first_source_structure) != len(first_token_ids)
    for alias in (
        "token_dep_levels",
        "token_ast_depth",
        "token_sibling_index",
        "token_ast_node_type",
    ):
        assert len(table[alias].to_pylist()[0]) == len(first_token_ids)

    dataset = TokenParquetDataset(
        sample_path,
        seq_len=128,
        batch_size=2,
        token_key="token_ids",
    )
    batch = next(dataset.iter_batches())

    assert dataset.num_samples >= 2
    assert dataset.metadata.source_format == "parquet"
    assert tuple(batch.tokens.shape) == (2, 128)
    token_min, token_max = dataset.token_id_range()
    assert 0 <= token_min <= token_max <= np.iinfo(np.int32).max

    assert batch.structure_ids is not None
    np.testing.assert_array_equal(np.array(batch.tokens[0]), first_token_ids[:128])
    np.testing.assert_array_equal(
        np.array(batch.structure_ids[0]), first_token_structure[:128]
    )
    for field_name in (
        *STRUCTURE_MODEL_KWARG_KEYS,
    ):
        value = getattr(batch, field_name)
        assert value is not None
        assert tuple(value.shape) == tuple(batch.tokens.shape)
        assert np.array(value).dtype == np.int32


@pytest.mark.parametrize(
    "dataset_name",
    [
        "clang_semantic_4k_v10",
        "clang_commits_4k_v1",
    ],
)
def test_gb10_real_parquet_side_channels_reach_training_loss(
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
    assert tuple(kwargs) == STRUCTURE_MODEL_KWARG_KEYS
    for key in STRUCTURE_MODEL_KWARG_KEYS:
        assert tuple(kwargs[key].shape) == (2, 127)

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
