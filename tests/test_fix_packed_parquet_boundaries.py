from __future__ import annotations

import subprocess
import sys

import pyarrow as pa
import pyarrow.parquet as pq

from scripts.fix_packed_parquet_boundaries import TokenPieces, repair_row
from scripts.render_sidecar_example import get_tokenizer


def _row_for_text(text: str, *, slack: int = 8) -> dict:
    tok = get_tokenizer()
    ids = list(tok.encode(text))
    valid = len(ids)
    capacity = valid + slack
    padded = ids + [0] * slack
    return {
        "input_ids": padded,
        "target_ids": padded[1:] + [0],
        "loss_mask": [1] * max(valid - 1, 0) + [0] * (capacity - max(valid - 1, 0)),
        "doc_ids": [7] * valid + [0] * slack,
        "platform_ids": [1] * valid + [0] * slack,
        "token_structure_ids": [6] * valid + [0] * slack,
        "token_dep_levels": [0] * capacity,
        "token_ast_depth": [0] * capacity,
        "token_sibling_index": [0] * capacity,
        "token_ast_node_type": [0] * capacity,
        "token_symbol_ids": [0] * capacity,
        "token_call_targets": [0] * capacity,
        "token_type_refs": [0] * capacity,
        "token_def_use": [0] * capacity,
        "token_change_mask_pre": [0] * capacity,
        "token_change_mask_post": [0] * capacity,
        "hunk_id_per_token": [-1] * valid + [0] * slack,
        "edit_op_per_token": [0] * capacity,
        "token_chunk_starts": [0, max(1, valid // 2)],
        "token_chunk_ends": [max(1, valid // 2), valid],
        "changed_chunk_spans": [{"start": max(1, valid // 2), "end": valid}],
        "source_doc_ids": [7],
        "source_doc_token_lengths": [valid],
        "valid_token_count": valid,
        "trained_token_count": max(valid - 1, 0),
        "slack_tokens": slack,
    }


def test_repair_row_inserts_newlines_around_generated_commit_marker() -> None:
    tok = get_tokenizer()
    pieces = TokenPieces(tok)
    row = _row_for_text(
        "/* can't compile; see https://github.com/a/b. */// === PRE-COMMIT ===bool f();\n"
    )

    repaired, info = repair_row(row, pieces)

    text = tok.decode(repaired["input_ids"][: repaired["valid_token_count"]])
    assert info == {"changed": True, "insertions": 2, "overflow": False}
    assert "*/// === PRE-COMMIT" not in text
    assert "*/\n// === PRE-COMMIT ===\nbool f();" in text
    assert repaired["valid_token_count"] == row["valid_token_count"] + 2
    assert repaired["trained_token_count"] == repaired["valid_token_count"] - 1
    assert repaired["target_ids"] == repaired["input_ids"][1:] + [0]
    assert repaired["loss_mask"][repaired["trained_token_count"] - 1] == 1
    assert repaired["loss_mask"][repaired["trained_token_count"]] == 0
    assert repaired["source_doc_token_lengths"] == [row["source_doc_token_lengths"][0] + 2]
    assert repaired["token_chunk_ends"][-1] == row["token_chunk_ends"][-1] + 2
    assert repaired["changed_chunk_spans"][0]["end"] == row["changed_chunk_spans"][0]["end"] + 2


def test_repair_row_refuses_when_no_padding_slack() -> None:
    tok = get_tokenizer()
    pieces = TokenPieces(tok)
    row = _row_for_text(
        "/* can't compile. */// === DIFF ===diff --git a/x b/x\n",
        slack=0,
    )

    repaired, info = repair_row(row, pieces)

    assert repaired is row
    assert info == {"changed": False, "insertions": 2, "overflow": True}


def test_dry_run_fail_on_remaining_rejects_repairable_marker_rows(tmp_path) -> None:
    row = _row_for_text("/* can't compile. */// === PRE-COMMIT ===bool f();\n")
    path = tmp_path / "commits" / "1024" / "bad.parquet"
    path.parent.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist([row]), path)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/fix_packed_parquet_boundaries.py",
            str(tmp_path / "commits"),
            "--dry-run",
            "--workers",
            "1",
            "--report",
            str(tmp_path / "report.json"),
            "--fail-on-remaining",
        ],
        check=False,
    )

    assert result.returncode != 0


def test_real_run_fail_on_remaining_accepts_repaired_marker_rows(tmp_path) -> None:
    row = _row_for_text("/* can't compile. */// === PRE-COMMIT ===bool f();\n")
    root = tmp_path / "commits"
    path = root / "1024" / "bad.parquet"
    path.parent.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist([row]), path)

    subprocess.run(
        [
            sys.executable,
            "scripts/fix_packed_parquet_boundaries.py",
            str(root),
            "--workers",
            "1",
            "--report",
            str(tmp_path / "repair_report.json"),
            "--fail-on-remaining",
        ],
        check=True,
    )

    subprocess.run(
        [
            sys.executable,
            "scripts/fix_packed_parquet_boundaries.py",
            str(root),
            "--dry-run",
            "--workers",
            "1",
            "--report",
            str(tmp_path / "post_report.json"),
            "--fail-on-remaining",
        ],
        check=True,
    )
