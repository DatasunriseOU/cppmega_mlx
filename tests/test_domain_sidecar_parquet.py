from __future__ import annotations

import json
import subprocess
import sys

import pyarrow as pa
import pyarrow.parquet as pq

from cppmega_mlx.data.domain_schema import DomainKind, DomainRoleKind, ParseConfidence
from cppmega_mlx.data.tokenizer_contract import DOMAIN_DELIMITER_TOKEN_IDS


def _write_parquet(path, *, extra):
    path.parent.mkdir(parents=True, exist_ok=True)
    start = DOMAIN_DELIMITER_TOKEN_IDS["COMPILER_ERROR_START"]
    end = DOMAIN_DELIMITER_TOKEN_IDS["COMPILER_ERROR_END"]
    row = {
        "input_ids": [start, 101, 102, end, 0, 0, 0, 0],
        "target_ids": [101, 102, end, 0, 0, 0, 0, 0],
        "loss_mask": [1, 1, 1, 0, 0, 0, 0, 0],
        "doc_ids": [0, 0, 0, 0, 0, 0, 0, 0],
        "valid_token_count": 4,
        "trained_token_count": 3,
        "slack_tokens": 4,
        "token_domain_ids": [int(DomainKind.COMPILER_ERROR)] * 4 + [0, 0, 0, 0],
        "token_role_ids": [
            int(DomainRoleKind.DELIMITER),
            int(DomainRoleKind.FILE),
            int(DomainRoleKind.MESSAGE),
            int(DomainRoleKind.DELIMITER),
            0,
            0,
            0,
            0,
        ],
        "token_entity_ids": [0] * 8,
        "token_scope_ids": [0] * 8,
        "token_source_doc_ids": [17, 17, 17, 17, 0, 0, 0, 0],
        "token_confidence_ids": [int(ParseConfidence.HEURISTIC)] * 4 + [0, 0, 0, 0],
        "token_diagnostic_edges": [],
    }
    row.update(extra)
    pq.write_table(pa.Table.from_pylist([row]), path)


def _run_audit(tmp_path, root):
    out_dir = tmp_path / "audit"
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/audit_sidecar_parquet.py",
            "--code-root",
            str(root),
            "--commit-root",
            str(root),
            "--pr-root",
            str(root),
            "--buckets",
            "8",
            "--workers",
            "1",
            "--out-dir",
            str(out_dir),
        ],
        capture_output=True,
        text=True,
    )
    report = json.loads((out_dir / "sidecar_parquet_audit.json").read_text())
    return proc, report


def test_audit_rejects_diagnostic_row_without_edges_or_raw_confidence(tmp_path):
    root = tmp_path / "code"
    _write_parquet(root / "8" / "diag.parquet", extra={})

    proc, report = _run_audit(tmp_path, root)

    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert report["total"]["fields"]["token_diagnostic_edges"]["bad_value_rows"] >= 1
    assert any("no token_diagnostic_edges" in err for err in report["total"]["errors"])


def test_audit_accepts_raw_diagnostic_row_without_edges(tmp_path):
    root = tmp_path / "code"
    _write_parquet(
        root / "8" / "diag.parquet",
        extra={
            "token_confidence_ids": [int(ParseConfidence.RAW)] * 4 + [0, 0, 0, 0],
        },
    )

    proc, report = _run_audit(tmp_path, root)

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert report["total"]["bad_rows"] == 0


def test_audit_accepts_structured_diagnostic_edges(tmp_path):
    root = tmp_path / "code"
    _write_parquet(
        root / "8" / "diag.parquet",
        extra={
            "token_diagnostic_edges": [{"from": 1, "to": 2, "kind": 60}],
        },
    )

    proc, report = _run_audit(tmp_path, root)

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert report["total"]["edge_count"]["token_diagnostic_edges"] == 3
