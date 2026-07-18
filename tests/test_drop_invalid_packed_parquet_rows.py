from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from cppmega_mlx.data.source_identity import source_identity
from scripts.nanochat_data.pack_enriched_rows import PACKED_ROW_OUTPUT_SCHEMA

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = "scripts/drop_invalid_packed_parquet_rows.py"


def _loss_mask_for_doc_ids(doc_ids: list[int], *, valid: int, length: int) -> list[int]:
    mask: list[int] = []
    for pos in range(length):
        if pos + 1 >= valid:
            mask.append(0)
        else:
            mask.append(1 if doc_ids[pos] == doc_ids[pos + 1] else 0)
    return mask


def _row(length: int, *, valid: int | None = None) -> dict:
    valid = length if valid is None else valid
    identity = source_identity({"source_path": "fixture.cpp"})
    doc_ids = [1] * length
    loss_mask = _loss_mask_for_doc_ids(doc_ids, valid=valid, length=length)
    return {
        "pack_id": 1,
        "input_ids": [1] * valid + [0] * (length - valid),
        "target_ids": [1] * max(valid - 1, 0) + [0] * (length - max(valid - 1, 0)),
        "loss_mask": loss_mask,
        "doc_ids": doc_ids,
        "valid_token_count": valid,
        "trained_token_count": sum(loss_mask),
        "num_docs": 1,
        "slack_tokens": length - valid,
        "source_doc_ids": [1],
        "source_doc_token_lengths": [valid],
        "source_platform_ids": [[7]],
        "source_repo_stable_ids": ["9"],
        "source_filepath_stable_ids": ["11"],
        "source_file_local_commit_indices": [0],
        "platform_ids": [7],
        "token_platform_ids": [0] * length,
        "token_structure_ids": [1] * valid + [0] * (length - valid),
        "token_dep_levels": [0] * length,
        "token_ast_depth": [0] * length,
        "token_sibling_index": [0] * length,
        "token_ast_node_type": [0] * length,
        "token_domain_ids": [0] * length,
        "token_role_ids": [0] * length,
        "token_entity_ids": [0] * length,
        "token_scope_ids": [0] * length,
        "token_source_doc_ids": [1] * valid + [0] * (length - valid),
        "token_source_identity_ids": [identity.source_identity_id] * valid
        + [0] * (length - valid),
        "token_confidence_ids": [0] * length,
        "token_symbol_ids": [0] * length,
        "token_call_targets": [0] * length,
        "token_type_refs": [0] * length,
        "token_def_use": [0] * length,
        "token_change_mask_pre": [0] * length,
        "token_change_mask_post": [0] * length,
        "hunk_id_per_token": [-1] * length,
        "edit_op_per_token": [0] * length,
        "token_chunk_starts": [0],
        "token_chunk_ends": [valid],
        "token_chunk_kinds": [1],
        "token_chunk_dep_levels": [0],
        "token_call_edges": [],
        "token_type_edges": [],
        "token_domain_edges": [],
        "token_build_edges": [],
        "token_shell_edges": [],
        "token_diagnostic_edges": [],
        "token_cross_domain_edges": [],
        "changed_chunk_ids": [],
        "changed_chunk_spans": [],
        "source_identity_registry": [identity.as_dict()],
    }


def _multidoc_row(length: int, *, doc_lengths: tuple[int, ...], valid: int | None = None) -> dict:
    """A realistic packed row spanning >=2 distinct documents (>=2 doc_ids segments)."""
    valid = length if valid is None else valid
    assert sum(doc_lengths) == valid, "doc_lengths must fill the valid region"
    doc_ids: list[int] = []
    for i, doc_len in enumerate(doc_lengths):
        doc_ids.extend([7 + 2 * i] * doc_len)
    doc_ids.extend([0] * (length - valid))
    row = _row(length, valid=valid)
    row["doc_ids"] = doc_ids
    row["loss_mask"] = _loss_mask_for_doc_ids(doc_ids, valid=valid, length=length)
    row["trained_token_count"] = sum(row["loss_mask"])
    return row


def test_broken_input_file_fails_loud_by_default(tmp_path):
    # Real, valid multi-document packed rows alongside a deliberately corrupt
    # ".parquet" file in the same numeric bucket directory. The corrupt file is
    # genuine on-disk bytes that are NOT a valid parquet container, so
    # pq.read_table raises while reading it (a real read/schema failure).
    root = tmp_path / "commits"
    bucket_dir = root / "8"
    bucket_dir.mkdir(parents=True)
    pq.write_table(
        pa.Table.from_pylist(
            [
                _multidoc_row(8, doc_lengths=(3, 5)),    # valid (kept)
                _multidoc_row(11, doc_lengths=(4, 7)),   # over-bucket (would drop)
            ],
            schema=PACKED_ROW_OUTPUT_SCHEMA,
        ),
        bucket_dir / "good.parquet",
    )
    broken = bucket_dir / "broken.parquet"
    broken.write_bytes(b"PAR0 this is not a real parquet container \x00\x01\x02\x03")

    report = tmp_path / "drop_report.json"
    result = subprocess.run(
        [
            sys.executable,
            SCRIPT,
            str(root),
            "--workers",
            "1",
            "--report",
            str(report),
        ],
        cwd=str(REPO_ROOT),
        check=False,
        capture_output=True,
        text=True,
    )

    # RULE #1: a per-file read/schema failure must be fatal BY DEFAULT, with no
    # opt-in flag required. The OLD code swallowed the exception into a report
    # entry and returned 0; this assertion fails against that behavior.
    assert result.returncode != 0, (
        "expected non-zero exit on a broken input file by default; "
        f"rc={result.returncode} stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    # Crash names WHERE it failed.
    assert "broken.parquet" in result.stderr, result.stderr


def test_continue_on_error_records_failure_but_still_exits_nonzero(tmp_path):
    # Same fixture as above, but with the explicit --continue-on-error opt-in:
    # the broken file is recorded and the healthy file is still cleaned, yet the
    # exit code is STILL non-zero because a file failed.
    root = tmp_path / "commits"
    bucket_dir = root / "8"
    bucket_dir.mkdir(parents=True)
    good = bucket_dir / "good.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                _multidoc_row(8, doc_lengths=(3, 5)),    # valid (kept)
                _multidoc_row(11, doc_lengths=(4, 7)),   # over-bucket (dropped)
            ],
            schema=PACKED_ROW_OUTPUT_SCHEMA,
        ),
        good,
    )
    broken = bucket_dir / "broken.parquet"
    broken.write_bytes(b"PAR0 this is not a real parquet container \x00\x01\x02\x03")

    report = tmp_path / "drop_report.json"
    result = subprocess.run(
        [
            sys.executable,
            SCRIPT,
            str(root),
            "--workers",
            "1",
            "--continue-on-error",
            "--report",
            str(report),
        ],
        cwd=str(REPO_ROOT),
        check=False,
        capture_output=True,
        text=True,
    )

    # Explicit opt-in keeps processing other files, but exit is STILL non-zero.
    assert result.returncode == 2, (
        f"rc={result.returncode} stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    summary = json.loads(report.read_text())
    assert summary["total"]["error_files"] == 1
    # The healthy file was still cleaned: its over-bucket row was dropped.
    assert summary["total"]["dropped_rows"] == 1
    assert pq.read_table(good).num_rows == 1
    assert good.with_name(good.name + ".pre_validity_fix").exists()


def test_drop_invalid_rows_removes_over_bucket_rows_and_audit_passes(tmp_path):
    root = tmp_path / "commits"
    path = root / "8" / "sample.parquet"
    path.parent.mkdir(parents=True)
    pq.write_table(
        pa.Table.from_pylist(
            [_row(8, valid=5), _row(11, valid=11)],
            schema=PACKED_ROW_OUTPUT_SCHEMA,
        ),
        path,
    )

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
    pq.write_table(
        pa.Table.from_pylist([_row(9, valid=9)], schema=PACKED_ROW_OUTPUT_SCHEMA),
        path,
    )

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
