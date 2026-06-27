from __future__ import annotations

import json
import subprocess
import sys

import pyarrow as pa
import pyarrow.parquet as pq


def _row(length: int, *, valid: int | None = None) -> dict:
    valid = length if valid is None else valid
    return {
        "input_ids": [1] * valid + [0] * (length - valid),
        "target_ids": [1] * valid + [0] * (length - valid),
        "loss_mask": [1] * valid + [0] * (length - valid),
        "doc_ids": [7] * valid + [0] * (length - valid),
        "valid_token_count": valid,
        "trained_token_count": valid,
        "slack_tokens": length - valid,
        "source_doc_ids": [1],
        "source_doc_token_lengths": [valid],
        "source_platform_ids": [7],
        "source_repo_stable_ids": [9],
        "source_filepath_stable_ids": [11],
        "source_file_local_commit_indices": [0],
        "platform_ids": [7],
        "token_structure_ids": [1] * valid + [0] * (length - valid),
        "token_dep_levels": [0] * length,
        "token_ast_depth": [0] * length,
        "token_sibling_index": [0] * length,
        "token_ast_node_type": [0] * length,
        "token_symbol_ids": [0] * length,
        "token_call_targets": [0] * length,
        "token_type_refs": [0] * length,
        "token_def_use": [0] * length,
        "token_change_mask_pre": [0] * length,
        "token_change_mask_post": [0] * length,
        "hunk_id_per_token": [-1] * valid + [0] * (length - valid),
        "edit_op_per_token": [0] * length,
        "token_chunk_starts": [0],
        "token_chunk_ends": [valid],
        "token_chunk_kinds": [1],
        "token_chunk_dep_levels": [0],
        "token_call_edges": [],
        "token_type_edges": [],
        "changed_chunk_ids": [],
        "changed_chunk_spans": [],
    }


def test_drop_invalid_rows_removes_over_bucket_rows_and_audit_passes(tmp_path):
    root = tmp_path / "commits"
    path = root / "8" / "sample.parquet"
    path.parent.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist([_row(8, valid=5), _row(11, valid=11)]), path)

    report = tmp_path / "drop_report.json"
    subprocess.run(
        [
            sys.executable,
            "scripts/drop_invalid_packed_parquet_rows.py",
            str(root),
            "--workers",
            "1",
            "--report",
            str(report),
        ],
        check=True,
    )

    summary = json.loads(report.read_text())
    assert summary["total"]["dropped_rows"] == 1
    assert (path.with_name(path.name + ".pre_validity_fix")).exists()
    assert pq.read_table(path).num_rows == 1

    out_dir = tmp_path / "audit"
    subprocess.run(
        [
            sys.executable,
            "scripts/audit_sidecar_parquet.py",
            "--code-root",
            str(tmp_path / "empty_code"),
            "--commit-root",
            str(root),
            "--pr-root",
            str(tmp_path / "empty_pr"),
            "--buckets",
            "8",
            "--workers",
            "1",
            "--out-dir",
            str(out_dir),
            "--fail-on-bad",
        ],
        check=True,
    )
    audit = json.loads((out_dir / "sidecar_parquet_audit.json").read_text())
    assert audit["total"]["bad_rows"] == 0


def test_drop_invalid_rows_dry_run_fails_when_rows_would_be_dropped(tmp_path):
    root = tmp_path / "code"
    path = root / "8" / "sample.parquet"
    path.parent.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist([_row(9, valid=9)]), path)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/drop_invalid_packed_parquet_rows.py",
            str(root),
            "--workers",
            "1",
            "--dry-run",
            "--fail-on-remaining",
            "--report",
            str(tmp_path / "dry_report.json"),
        ],
        check=False,
    )

    assert result.returncode == 2
    assert pq.read_table(path).num_rows == 1
