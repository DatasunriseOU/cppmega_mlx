from __future__ import annotations

import json
import subprocess
import sys

import pyarrow.parquet as pq

from cppmega_mlx.data.domain_schema import DomainKind, DomainRoleKind, ParseConfidence
from cppmega_mlx.data.source_identity import source_identity
from cppmega_mlx.data.tokenizer_contract import DOMAIN_DELIMITER_TOKEN_IDS
from scripts.nanochat_data.pack_enriched_rows import (
    normalize_document_record,
    pack_documents,
    rows_to_table,
)


def _write_parquet(path, *, extra):
    path.parent.mkdir(parents=True, exist_ok=True)
    start = DOMAIN_DELIMITER_TOKEN_IDS["COMPILER_ERROR_START"]
    end = DOMAIN_DELIMITER_TOKEN_IDS["COMPILER_ERROR_END"]
    identity = source_identity({"source_path": "diagnostic.log"})
    row = {
        "token_ids": [start, 101, 102, end],
        "token_domain_ids": [int(DomainKind.COMPILER_ERROR)] * 4,
        "token_role_ids": [
            int(DomainRoleKind.DELIMITER),
            int(DomainRoleKind.FILE),
            int(DomainRoleKind.MESSAGE),
            int(DomainRoleKind.DELIMITER),
        ],
        "token_entity_ids": [0] * 4,
        "token_scope_ids": [0] * 4,
        "token_source_doc_ids": [17] * 4,
        "token_source_identity_ids": [identity.source_identity_id] * 4,
        "source_identity_registry": [identity.as_dict()],
        "token_confidence_ids": [
            int(ParseConfidence.EXACT),
            int(ParseConfidence.HEURISTIC),
            int(ParseConfidence.HEURISTIC),
            int(ParseConfidence.EXACT),
        ],
        "token_diagnostic_edges": [],
    }
    row.update(extra)
    requested_confidence = list(row["token_confidence_ids"])
    doc = normalize_document_record(row, source_doc_index=0)
    packed, overflow = pack_documents([doc], target_length=8, strategy="sequential")
    assert overflow == []
    # Audit fixtures deliberately preserve the requested post-packer state so
    # the rejection test can write a producer-invalid confidence vector.
    packed[0]["token_confidence_ids"][: len(requested_confidence)] = (
        requested_confidence
    )
    pq.write_table(rows_to_table(packed), path)


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
            "token_confidence_ids": [
                int(ParseConfidence.EXACT),
                int(ParseConfidence.RAW),
                int(ParseConfidence.RAW),
                int(ParseConfidence.EXACT),
            ],
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
