from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pytest

from cppmega_mlx.data.parquet_dataset import TokenParquetDataset
from cppmega_mlx.tokenizer.cpp_tokenizer import load_cppmega_tokenizer


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "materialize_token_coordinate_parquet.py"
TOKENIZER = ROOT / "cppmega_mlx" / "tokenizer" / "tokenizer.json"
REAL_PARQUET = (
    ROOT
    / "data"
    / "parquet_samples"
    / "gb10"
    / "clang_semantic_4k_v10"
    / "val_00000.parquet"
)


def _load_module() -> Any:
    spec = importlib.util.spec_from_file_location("materialize_token_coordinate_parquet", SCRIPT)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_materialize_token_coordinate_columns_maps_structure_and_chunks() -> None:
    module = _load_module()
    tokenizer = load_cppmega_tokenizer(TOKENIZER)
    text = "ab cd\nxy"
    token_ids = tokenizer.encode(text)
    assert isinstance(token_ids, list)

    columns, stats = module.materialize_token_coordinate_columns(
        {
            "text": [text],
            "actual_token_count": [len(token_ids)],
            "token_ids": [token_ids],
            "structure_ids": [list(range(len(text)))],
            "chunk_boundaries": [[{"start": 0, "end": len(text), "kind": 3, "dep_level": 2}]],
            "call_edges": [[]],
            "type_edges": [[]],
        },
        hf_tokenizer=tokenizer._tokenizer,
    )

    assert stats.rows == 1
    assert stats.retokenize_mismatches == 0
    assert columns["token_structure_ids"][0]
    assert len(columns["token_structure_ids"][0]) == len(token_ids)
    assert len(columns["token_dep_levels"][0]) == len(token_ids)
    assert columns["token_chunk_starts"][0] == [0]
    assert columns["token_chunk_ends"][0] == [len(token_ids)]
    assert columns["token_chunk_kinds"][0] == [3]
    assert columns["token_chunk_dep_levels"][0] == [2]


def test_real_parquet_head_materializes_token_side_channels(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    pytest.importorskip("pyarrow.parquet")
    if not REAL_PARQUET.exists():
        pytest.skip(f"local GB10 parquet sample is absent: {REAL_PARQUET}")

    module = _load_module()
    output = tmp_path / "val_00000_tokencoords.parquet"
    receipt = module.convert_parquet_file(
        REAL_PARQUET,
        output,
        tokenizer_path=TOKENIZER,
        max_rows=8,
        batch_size=4,
        row_group_size=4,
    )

    assert receipt["status"] == "ok"
    assert receipt["stats"]["rows"] == 8
    assert receipt["stats"]["retokenize_mismatches"] == 0
    assert receipt["stats"]["token_count_mismatches"] == 0

    dataset = TokenParquetDataset(
        output,
        seq_len=128,
        batch_size=2,
        token_key="token_ids",
    )
    batch = next(dataset.iter_batches())
    side_sources = dataset.parquet_receipt["side_channel_sources"]
    assert side_sources["structure_ids"]["column"] == "token_structure_ids"
    assert side_sources["dep_levels"]["column"] == "token_dep_levels"
    assert tuple(batch.tokens.shape) == (2, 128)
    assert batch.structure_ids is not None
    assert batch.dep_levels is not None
    assert tuple(batch.structure_ids.shape) == (2, 128)
    assert tuple(batch.dep_levels.shape) == (2, 128)
    assert np.array(batch.structure_ids).shape == np.array(batch.tokens).shape
