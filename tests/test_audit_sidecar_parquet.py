from __future__ import annotations

import json
import subprocess
import sys

import pyarrow as pa
import pyarrow.parquet as pq


def _write_tiny_parquet(
    path,
    *,
    input_ids=(1, 2, 3, 4, 0, 0, 0, 0),
    target_ids=(2, 3, 4, 5, 0, 0, 0, 0),
    # Canonical single-doc loss_mask: 1 on every valid token EXCEPT the last
    # valid token (no next token to predict) and the whole pad region. With a
    # single document (doc_ids all equal) and valid=4 the rule yields
    # [1,1,1,0, 0,0,0,0] and trained_token_count = sum = 3 = valid - num_docs.
    loss_mask=(1, 1, 1, 0, 0, 0, 0, 0),
    doc_ids=(0, 0, 0, 0, 0, 0, 0, 0),
    valid_token_count=4,
    trained_token_count=3,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "input_ids": list(input_ids),
        "target_ids": list(target_ids),
        "loss_mask": list(loss_mask),
        "doc_ids": list(doc_ids),
        "valid_token_count": int(valid_token_count),
        "trained_token_count": int(trained_token_count),
        "slack_tokens": 4,
        "source_doc_ids": [1],
        "source_doc_token_lengths": [4],
        "source_platform_ids": [7],
        "source_repo_stable_ids": [9],
        "source_filepath_stable_ids": [11],
        "source_file_local_commit_indices": [0],
        "platform_ids": [7],
        "token_structure_ids": [1, 1, 1, 1, 0, 0, 0, 0],
        "token_dep_levels": [0, 1, 1, 0, 0, 0, 0, 0],
        "token_ast_depth": [1, 2, 2, 1, 0, 0, 0, 0],
        "token_sibling_index": [0, 1, 2, 3, 0, 0, 0, 0],
        "token_ast_node_type": [3, 4, 4, 3, 0, 0, 0, 0],
        "token_symbol_ids": [0, 21, 0, 0, 0, 0, 0, 0],
        "token_call_targets": [0, 0, 22, 0, 0, 0, 0, 0],
        "token_type_refs": [0, 0, 0, 23, 0, 0, 0, 0],
        "token_def_use": [0, 1, 2, 0, 0, 0, 0, 0],
        "token_change_mask_pre": [0, 1, 1, 0, 0, 0, 0, 0],
        "token_change_mask_post": [0, 1, 1, 0, 0, 0, 0, 0],
        "hunk_id_per_token": [-1, 0, 0, -1, -1, -1, -1, -1],
        "edit_op_per_token": [0, 1, 1, 0, 0, 0, 0, 0],
        "token_chunk_starts": [0, 2],
        "token_chunk_ends": [2, 4],
        "token_chunk_kinds": [1, 2],
        "token_chunk_dep_levels": [0, 1],
        "token_call_edges": [{"from": 0, "to": 1}],
        "token_type_edges": [{"from": 1, "to": 0}],
        "changed_chunk_ids": [1],
        "changed_chunk_spans": [{"start": 2, "end": 4}],
    }
    table = pa.Table.from_pylist([row])
    pq.write_table(table, path)


def _run_audit(tmp_path, code_root, commit_root, pr_root, *, extra_args=()):
    out_dir = tmp_path / "audit"
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/audit_sidecar_parquet.py",
            "--code-root",
            str(code_root),
            "--commit-root",
            str(commit_root),
            "--pr-root",
            str(pr_root),
            "--buckets",
            "8",
            "--workers",
            "1",
            "--out-dir",
            str(out_dir),
            *extra_args,
        ],
        capture_output=True,
        text=True,
    )
    report = None
    report_path = out_dir / "sidecar_parquet_audit.json"
    if report_path.exists():
        report = json.loads(report_path.read_text())
    return proc, report


def test_sidecar_audit_accepts_valid_chunk_indexed_edges(tmp_path):
    code_root = tmp_path / "code"
    commit_root = tmp_path / "commits"
    pr_root = tmp_path / "pr"
    _write_tiny_parquet(code_root / "8" / "code.parquet")
    _write_tiny_parquet(commit_root / "8" / "commit.parquet")
    _write_tiny_parquet(pr_root / "8" / "pr.parquet")

    # FAIL-CLOSED is now the default: a clean corpus must still exit 0.
    proc, report = _run_audit(tmp_path, code_root, commit_root, pr_root)
    assert proc.returncode == 0, proc.stderr

    assert report["total"]["files"] == 3
    assert report["total"]["rows"] == 3
    assert report["total"]["valid_tokens"] == 12
    assert report["total"]["bad_files"] == 0
    assert report["total"]["bad_rows"] == 0
    assert report["total"]["fields"]["loss_mask"]["bad_value_rows"] == 0
    assert report["total"]["edge_count"] == {
        "token_call_edges": 3,
        "token_type_edges": 3,
    }


def test_sidecar_audit_rejects_allones_loss_mask_on_multidoc_row(tmp_path):
    """C1 regression: an all-ones loss_mask over a MULTI-document packed row.

    The corruption preserves every array length, so the length checks pass; only
    the doc_ids-derived value check can catch it. With two documents
    (doc_ids = [7,7,7,9,9, pad...]) and valid=5 the canonical loss_mask is
    [1,1,0,1,0,...] (note the 0 at the inter-doc boundary, pos 2). The corrupted
    row stores [1,1,1,1,0,...], which trains the model to predict document B's
    first token from document A's last token. This MUST be flagged bad AND, under
    the fail-closed default, MUST block the upload (non-zero exit).
    """
    code_root = tmp_path / "code"
    commit_root = tmp_path / "commits"
    pr_root = tmp_path / "pr"

    # One clean code shard so the run is not trivially empty.
    _write_tiny_parquet(code_root / "8" / "code.parquet")
    _write_tiny_parquet(pr_root / "8" / "pr.parquet")

    # Corrupted multi-doc commit shard (C1): all-ones loss_mask over the valid
    # region despite a real document boundary at position 2->3.
    _write_tiny_parquet(
        commit_root / "8" / "commit.parquet",
        input_ids=(10, 11, 12, 13, 14, 0, 0, 0),
        target_ids=(11, 12, 13, 14, 0, 0, 0, 0),
        doc_ids=(7, 7, 7, 9, 9, 0, 0, 0),
        loss_mask=(1, 1, 1, 1, 0, 0, 0, 0),  # WRONG: boundary at pos 2 is masked 1
        valid_token_count=5,
        trained_token_count=4,
    )

    proc, report = _run_audit(tmp_path, code_root, commit_root, pr_root)

    # Fail-closed: the corrupted shard blocks the upload with a non-zero exit.
    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert report is not None
    assert report["total"]["bad_rows"] >= 1
    # The loss_mask value check (not a length check) is what flagged it.
    assert report["total"]["fields"]["loss_mask"]["bad_value_rows"] >= 1
    assert report["total"]["fields"]["loss_mask"]["bad_length_rows"] == 0
    # The corrupted commit shard is named in the bad-files list.
    assert any("commit.parquet" in p for p in report["bad_files"])


def test_sidecar_audit_accepts_correct_multidoc_loss_mask(tmp_path):
    """Control: the SAME multi-doc layout with the canonical mask passes.

    Proves the new check is precise (it pins the doc-boundary rule) rather than
    blanket-rejecting every multi-document row.
    """
    code_root = tmp_path / "code"
    commit_root = tmp_path / "commits"
    pr_root = tmp_path / "pr"

    _write_tiny_parquet(code_root / "8" / "code.parquet")
    _write_tiny_parquet(pr_root / "8" / "pr.parquet")
    _write_tiny_parquet(
        commit_root / "8" / "commit.parquet",
        input_ids=(10, 11, 12, 13, 14, 0, 0, 0),
        target_ids=(11, 12, 13, 14, 0, 0, 0, 0),
        doc_ids=(7, 7, 7, 9, 9, 0, 0, 0),
        loss_mask=(1, 1, 0, 1, 0, 0, 0, 0),  # canonical: 0 at the inter-doc boundary
        valid_token_count=5,
        trained_token_count=3,
    )

    proc, report = _run_audit(tmp_path, code_root, commit_root, pr_root)

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert report["total"]["bad_rows"] == 0
    assert report["total"]["fields"]["loss_mask"]["bad_value_rows"] == 0
