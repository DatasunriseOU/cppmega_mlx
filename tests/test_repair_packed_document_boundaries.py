from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scripts.repair_packed_document_boundaries import _rewrite_file, repair_row


def _collapsed_row() -> dict[str, object]:
    return {
        "input_ids": [10, 11, 12, 13, 14, 0, 0, 0],
        "target_ids": [11, 12, 13, 14, 0, 0, 0, 0],
        "doc_ids": [11] * 8,
        "loss_mask": [1, 1, 1, 1, 0, 0, 0, 0],
        "source_doc_ids": [38, 41],
        "source_doc_token_lengths": [3, 2],
        "valid_token_count": 5,
        "trained_token_count": 4,
        "num_docs": 2,
    }


def test_repair_row_restores_independent_same_file_boundary() -> None:
    row = _collapsed_row()

    repaired, changed, restored = repair_row(row)

    assert changed is True
    assert restored == 1
    assert repaired["doc_ids"] == [1, 1, 1, 2, 2, 2, 2, 2]
    assert repaired["loss_mask"] == [1, 1, 0, 1, 0, 0, 0, 0]
    assert repaired["trained_token_count"] == 3
    assert repaired["target_ids"] == row["target_ids"]


def test_repair_row_rejects_lengths_that_do_not_partition_valid_tokens() -> None:
    row = _collapsed_row()
    row["source_doc_token_lengths"] = [3, 1]

    with pytest.raises(ValueError, match="source_doc_token_lengths sum=4"):
        repair_row(row, where="fixture")


def test_rewrite_file_replaces_hardlink_without_mutating_source(tmp_path: Path) -> None:
    source = tmp_path / "source.parquet"
    snapshot = tmp_path / "snapshot.parquet"
    pq.write_table(pa.Table.from_pylist([_collapsed_row()]), source, compression="zstd")
    snapshot.hardlink_to(source)
    source_inode = source.stat().st_ino

    _rewrite_file(str(snapshot), 3)

    assert source.stat().st_ino == source_inode
    assert snapshot.stat().st_ino != source_inode
    assert pq.read_table(source).column("doc_ids").to_pylist()[0] == [11] * 8
    repaired = pq.read_table(snapshot).to_pylist()[0]
    assert repaired["doc_ids"] == [1, 1, 1, 2, 2, 2, 2, 2]
    assert repaired["loss_mask"] == [1, 1, 0, 1, 0, 0, 0, 0]
