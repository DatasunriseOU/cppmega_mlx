from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from cppmega_mlx.data.parquet_dataset import TokenParquetDataset
import cppmega_mlx.data.parquet_dataset as parquet_dataset


class _FakeColumn:
    def __init__(self, values):
        self._values = values

    def to_pylist(self):
        return list(self._values)


class _FakeTable:
    def __init__(self, values, types=None):
        self._values = values
        self.column_names = list(values)
        self.schema = _FakeSchema(types) if types is not None else None

    def __getitem__(self, key):
        return _FakeColumn(self._values[key])


class _FakeField:
    def __init__(self, type_label):
        self.type = type_label


class _FakeSchema:
    def __init__(self, types):
        self._types = types

    def field(self, key):
        return _FakeField(self._types[key])


def _fake_pyarrow_reader(values, types=None):
    return SimpleNamespace(read_table=lambda path: _FakeTable(values, types=types))


def test_token_list_parquet_rows_yield_fixed_lm_batches(monkeypatch) -> None:
    values = {
        "tokens": [
            [0, 1, 2, 3, 99],
            [10, 11, 12, 13, 14, 15, 16, 17],
        ],
        "structure_ids": [
            [1, 1, 1, 1, 9],
            [2, 2, 2, 2, 3, 3, 3, 3],
        ],
    }
    monkeypatch.setattr(
        parquet_dataset.importlib,
        "import_module",
        lambda name: _fake_pyarrow_reader(values)
        if name == "pyarrow.parquet"
        else pytest.fail(f"unexpected import {name}"),
    )

    dataset = TokenParquetDataset("tokens.parquet", seq_len=4, batch_size=2)
    batch = next(dataset.iter_batches())

    assert dataset.num_samples == 3
    assert dataset.num_batches == 1
    assert dataset.dropped_samples == 1
    assert dataset.metadata.source_format == "parquet"
    np.testing.assert_array_equal(
        np.array(batch.tokens),
        np.array([[0, 1, 2, 3], [10, 11, 12, 13]], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        np.array(batch.structure_ids),
        np.array([[1, 1, 1, 1], [2, 2, 2, 2]], dtype=np.int32),
    )


def test_cppmega_token_side_channel_aliases_are_normalized(monkeypatch) -> None:
    values = {
        "token_ids": [[0, 1, 2, 3, 4, 5, 6, 7]],
        "structure_ids": [[101, 102]],
        "token_structure_ids": [[1, 1, 1, 1, 2, 2, 2, 2]],
        "token_dep_levels": [[0, 1, 2, 3, 0, 1, 2, 3]],
        "token_ast_depth": [[3, 3, 2, 2, 1, 1, 0, 0]],
        "token_sibling_index": [[0, 1, 0, 1, 0, 1, 0, 1]],
        "token_ast_node_type": [[9, 8, 7, 6, 5, 4, 3, 2]],
    }
    monkeypatch.setattr(
        parquet_dataset.importlib,
        "import_module",
        lambda name: _fake_pyarrow_reader(values)
        if name == "pyarrow.parquet"
        else pytest.fail(f"unexpected import {name}"),
    )

    dataset = TokenParquetDataset(
        "cppmega.parquet", seq_len=4, batch_size=2, token_key="token_ids"
    )
    batch = next(dataset.iter_batches())

    assert dataset.num_samples == 2
    np.testing.assert_array_equal(
        np.array(batch.structure_ids),
        np.array([[1, 1, 1, 1], [2, 2, 2, 2]], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        np.array(batch.dep_levels),
        np.array([[0, 1, 2, 3], [0, 1, 2, 3]], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        np.array(batch.ast_depth_ids),
        np.array([[3, 3, 2, 2], [1, 1, 0, 0]], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        np.array(batch.sibling_index_ids),
        np.array([[0, 1, 0, 1], [0, 1, 0, 1]], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        np.array(batch.node_type_ids),
        np.array([[9, 8, 7, 6], [5, 4, 3, 2]], dtype=np.int32),
    )


def test_source_level_structure_ids_are_ignored_when_not_token_aligned(
    monkeypatch,
) -> None:
    values = {
        "token_ids": [[0, 1, 2, 3, 4, 5, 6, 7]],
        "structure_ids": [[10, 20]],
    }
    monkeypatch.setattr(
        parquet_dataset.importlib,
        "import_module",
        lambda name: _fake_pyarrow_reader(values)
        if name == "pyarrow.parquet"
        else pytest.fail(f"unexpected import {name}"),
    )

    dataset = TokenParquetDataset(
        "cppmega.parquet", seq_len=4, batch_size=2, token_key="token_ids"
    )
    batch = next(dataset.iter_batches())

    assert batch.structure_ids is None


def test_token_aligned_alias_and_canonical_side_channel_collision_fails_closed(
    monkeypatch,
) -> None:
    values = {
        "token_ids": [[0, 1, 2, 3, 4, 5, 6, 7]],
        "structure_ids": [[10, 10, 10, 10, 20, 20, 20, 20]],
        "token_structure_ids": [[1, 1, 1, 1, 2, 2, 2, 2]],
    }
    monkeypatch.setattr(
        parquet_dataset.importlib,
        "import_module",
        lambda name: _fake_pyarrow_reader(values)
        if name == "pyarrow.parquet"
        else pytest.fail(f"unexpected import {name}"),
    )

    with pytest.raises(ValueError, match="structure_ids side-channel declared more than once"):
        TokenParquetDataset(
            "duplicate_structure.parquet",
            seq_len=4,
            batch_size=1,
            token_key="token_ids",
        )


def test_token_level_alias_shape_mismatch_fails_closed(monkeypatch) -> None:
    values = {
        "token_ids": [[0, 1, 2, 3, 4, 5, 6, 7]],
        "token_structure_ids": [[1, 1, 1, 1]],
    }
    monkeypatch.setattr(
        parquet_dataset.importlib,
        "import_module",
        lambda name: _fake_pyarrow_reader(values)
        if name == "pyarrow.parquet"
        else pytest.fail(f"unexpected import {name}"),
    )

    with pytest.raises(ValueError, match="token_structure_ids side-channel rows must be token-aligned"):
        TokenParquetDataset(
            "short_token_structure.parquet",
            seq_len=4,
            batch_size=1,
            token_key="token_ids",
        )


def test_scalar_token_column_is_treated_as_one_contiguous_stream(monkeypatch) -> None:
    values = {"tokens": list(range(12))}
    monkeypatch.setattr(
        parquet_dataset.importlib,
        "import_module",
        lambda name: _fake_pyarrow_reader(values)
        if name == "pyarrow.parquet"
        else pytest.fail(f"unexpected import {name}"),
    )

    dataset = TokenParquetDataset("tokens.parquet", seq_len=4, batch_size=3)
    batch = next(dataset.iter_batches())

    assert dataset.num_samples == 3
    np.testing.assert_array_equal(np.array(batch.tokens[2]), np.arange(8, 12))


def test_text_column_requires_and_uses_tokenizer(monkeypatch) -> None:
    values = {"text": ["abcd", "efgh"]}
    monkeypatch.setattr(
        parquet_dataset.importlib,
        "import_module",
        lambda name: _fake_pyarrow_reader(values)
        if name == "pyarrow.parquet"
        else pytest.fail(f"unexpected import {name}"),
    )

    class TinyTokenizer:
        def encode(self, text: str):
            return [ord(char) - 96 for char in text]

    dataset = TokenParquetDataset(
        "text.parquet",
        seq_len=4,
        batch_size=1,
        text_key="text",
        tokenizer=TinyTokenizer(),
    )
    batch = next(dataset.iter_batches())

    np.testing.assert_array_equal(np.array(batch.tokens[0]), np.array([1, 2, 3, 4]))


def test_pandas_backend_is_used_when_pyarrow_is_absent(monkeypatch) -> None:
    class FakeDataFrame:
        columns = ["tokens"]

        def __getitem__(self, key):
            assert key == "tokens"
            return SimpleNamespace(tolist=lambda: [[0, 1, 2, 3]])

    def fake_import(name):
        if name == "pyarrow.parquet":
            raise ModuleNotFoundError(name)
        if name == "pandas":
            return SimpleNamespace(read_parquet=lambda path: FakeDataFrame())
        raise AssertionError(f"unexpected import {name}")

    monkeypatch.setattr(parquet_dataset.importlib, "import_module", fake_import)

    dataset = TokenParquetDataset("tokens.parquet", seq_len=4, batch_size=1)
    batch = next(dataset.iter_batches())

    np.testing.assert_array_equal(np.array(batch.tokens[0]), np.arange(4))


def test_parquet_rejects_token_ids_outside_int32_range(monkeypatch) -> None:
    values = {
        "tokens": [[0, np.iinfo(np.int32).max + 1, 2, 3]],
    }
    monkeypatch.setattr(
        parquet_dataset.importlib,
        "import_module",
        lambda name: _fake_pyarrow_reader(values)
        if name == "pyarrow.parquet"
        else pytest.fail(f"unexpected import {name}"),
    )

    with pytest.raises(ValueError, match="token IDs exceed int32 range"):
        TokenParquetDataset("too_large_tokens.parquet", seq_len=4, batch_size=1)


def test_parquet_rejects_non_integer_token_ids(monkeypatch) -> None:
    values = {"tokens": [[0, 1.5, 2, 3]]}
    monkeypatch.setattr(
        parquet_dataset.importlib,
        "import_module",
        lambda name: _fake_pyarrow_reader(values)
        if name == "pyarrow.parquet"
        else pytest.fail(f"unexpected import {name}"),
    )

    with pytest.raises(ValueError, match="token IDs must be integer-valued"):
        TokenParquetDataset("float_tokens.parquet", seq_len=4, batch_size=1)


def test_parquet_rejects_declared_float_token_dtype(monkeypatch) -> None:
    values = {"tokens": [[0, 1, 2, 3]]}
    types = {"tokens": "large_list<element: double>"}
    monkeypatch.setattr(
        parquet_dataset.importlib,
        "import_module",
        lambda name: _fake_pyarrow_reader(values, types=types)
        if name == "pyarrow.parquet"
        else pytest.fail(f"unexpected import {name}"),
    )

    with pytest.raises(ValueError, match="token IDs parquet column 'tokens' must use an integer dtype"):
        TokenParquetDataset("float_token_type.parquet", seq_len=4, batch_size=1)


def test_parquet_rejects_negative_structure_side_channels(monkeypatch) -> None:
    values = {
        "tokens": [[0, 1, 2, 3]],
        "token_structure_ids": [[0, 1, -1, 3]],
    }
    monkeypatch.setattr(
        parquet_dataset.importlib,
        "import_module",
        lambda name: _fake_pyarrow_reader(values)
        if name == "pyarrow.parquet"
        else pytest.fail(f"unexpected import {name}"),
    )

    with pytest.raises(
        ValueError,
        match="structure_ids side-channel IDs must be non-negative",
    ):
        TokenParquetDataset("negative_structure.parquet", seq_len=4, batch_size=1)


def test_parquet_rejects_non_integer_structure_side_channels(monkeypatch) -> None:
    values = {
        "tokens": [[0, 1, 2, 3]],
        "token_structure_ids": [[0, 1.25, 2, 3]],
    }
    monkeypatch.setattr(
        parquet_dataset.importlib,
        "import_module",
        lambda name: _fake_pyarrow_reader(values)
        if name == "pyarrow.parquet"
        else pytest.fail(f"unexpected import {name}"),
    )

    with pytest.raises(ValueError, match="token_structure_ids side-channel must be integer-valued"):
        TokenParquetDataset("float_structure.parquet", seq_len=4, batch_size=1)


def test_missing_optional_parquet_backends_raise_clear_import_error(monkeypatch) -> None:
    def fake_import(name):
        if name in {"pyarrow.parquet", "pandas"}:
            raise ModuleNotFoundError(name)
        raise AssertionError(f"unexpected import {name}")

    monkeypatch.setattr(parquet_dataset.importlib, "import_module", fake_import)

    with pytest.raises(ImportError, match="pyarrow.*pandas"):
        TokenParquetDataset("tokens.parquet", seq_len=4, batch_size=1)


def test_text_column_without_tokenizer_is_rejected(monkeypatch) -> None:
    values = {"text": ["abcd"]}
    monkeypatch.setattr(
        parquet_dataset.importlib,
        "import_module",
        lambda name: _fake_pyarrow_reader(values)
        if name == "pyarrow.parquet"
        else pytest.fail(f"unexpected import {name}"),
    )

    with pytest.raises(ValueError, match="requires tokenizer"):
        TokenParquetDataset("text.parquet", seq_len=4, batch_size=1, text_key="text")
