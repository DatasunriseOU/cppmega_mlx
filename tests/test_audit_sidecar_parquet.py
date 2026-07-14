from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pyarrow as pa
import pyarrow.parquet as pq
import pytest


def _load_audit_module():
    path = Path(__file__).parents[1] / "scripts" / "audit_sidecar_parquet.py"
    spec = importlib.util.spec_from_file_location("cppmega_test_audit_sidecars", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_tiny_parquet(
    path,
    *,
    input_ids=(1, 2, 3, 4, 0, 0, 0, 0),
    target_ids=(2, 3, 4, 0, 0, 0, 0, 0),
    # Canonical single-doc loss_mask: 1 on every valid token EXCEPT the last
    # valid token (no next token to predict) and the whole pad region. With a
    # single document (doc_ids all equal) and valid=4 the rule yields
    # [1,1,1,0, 0,0,0,0] and trained_token_count = sum = 3 = valid - num_docs.
    loss_mask=(1, 1, 1, 0, 0, 0, 0, 0),
    doc_ids=(1, 1, 1, 1, 1, 1, 1, 1),
    valid_token_count=4,
    trained_token_count=3,
    extra=None,
    omit=(),
    token_source_doc_type=pa.uint32(),
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
    if extra:
        row.update(extra)
    if len(row["source_doc_ids"]) > 1 and not (
        extra and any(name in extra for name in ("token_chunk_starts", "token_call_edges"))
    ):
        starts = []
        ends = []
        offset = 0
        for length in row["source_doc_token_lengths"]:
            starts.append(offset)
            offset += int(length)
            ends.append(offset)
        row.update(
            {
                "token_chunk_starts": starts,
                "token_chunk_ends": ends,
                "token_chunk_kinds": [1] * len(starts),
                "token_chunk_dep_levels": [0] * len(starts),
                "token_call_edges": [],
                "token_type_edges": [],
            }
        )
    if "token_source_doc_ids" not in row:
        token_source_doc_ids = [
            int(source_id)
            for source_id, length in zip(
                row["source_doc_ids"],
                row["source_doc_token_lengths"],
                strict=True,
            )
            for _ in range(int(length))
        ]
        row["token_source_doc_ids"] = (
            token_source_doc_ids
            + [0] * (len(row["input_ids"]) - len(token_source_doc_ids))
        )
    for name in omit:
        row.pop(name, None)
    table = pa.Table.from_pylist([row])
    if "token_source_doc_ids" in row:
        index = table.schema.get_field_index("token_source_doc_ids")
        table = table.set_column(
            index,
            "token_source_doc_ids",
            pa.array([row["token_source_doc_ids"]], type=pa.list_(token_source_doc_type)),
        )
    pq.write_table(table, path)


def _run_audit(tmp_path, code_root, commit_root, pr_root, *, extra_args=()):
    out_dir = tmp_path / "audit"
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
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
        env=env,
        text=True,
    )
    report = None
    report_path = out_dir / "sidecar_parquet_audit.json"
    if report_path.exists():
        report = json.loads(report_path.read_text())
    return proc, report


def test_sidecar_audit_requires_all_source_roots_explicitly(capsys):
    audit = _load_audit_module()

    with pytest.raises(SystemExit) as exc:
        audit.build_arg_parser().parse_args([])

    assert exc.value.code == 2
    stderr = capsys.readouterr().err
    assert "--code-root" in stderr
    assert "--commit-root" in stderr
    assert "--pr-root" in stderr


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
    assert report["total"]["edge_count"]["token_call_edges"] == 3
    assert report["total"]["edge_count"]["token_type_edges"] == 3
    assert report["total"]["edge_count"]["token_domain_edges"] == 0


def test_sidecar_audit_reads_large_files_by_row_group(tmp_path, monkeypatch):
    path = tmp_path / "code" / "8" / "code.parquet"
    _write_tiny_parquet(path)
    one_row = pq.read_table(path)
    pq.write_table(pa.concat_tables([one_row, one_row]), path, row_group_size=1)

    audit = _load_audit_module()
    real_factory = audit.pq.ParquetFile
    calls: list[int] = []

    class RowGroupOnlyParquetFile:
        def __init__(self, parquet_path):
            self._inner = real_factory(parquet_path)
            self.schema_arrow = self._inner.schema_arrow
            self.metadata = self._inner.metadata

        def read(self, *args, **kwargs):
            raise AssertionError("whole-file parquet reads are forbidden")

        def read_row_group(self, index, *, columns):
            calls.append(index)
            return self._inner.read_row_group(index, columns=columns)

    monkeypatch.setattr(audit.pq, "ParquetFile", RowGroupOnlyParquetFile)
    result = audit._audit_file(str(path), "code", "8", 65536)

    assert calls == [0, 1]
    assert result["stats"]["rows"] == 2
    assert result["stats"]["valid_tokens"] == 8


def test_sidecar_audit_rejects_empty_or_reversed_chunk_span(tmp_path):
    code_root = tmp_path / "code"
    commit_root = tmp_path / "commits"
    pr_root = tmp_path / "pr"
    _write_tiny_parquet(
        code_root / "8" / "code.parquet",
        extra={"token_chunk_starts": [0, 2], "token_chunk_ends": [0, 4]},
    )
    _write_tiny_parquet(commit_root / "8" / "commit.parquet")
    _write_tiny_parquet(pr_root / "8" / "pr.parquet")

    proc, report = _run_audit(tmp_path, code_root, commit_root, pr_root)

    assert proc.returncode == 2
    assert report["total"]["bad_rows"] == 1
    assert report["total"]["fields"]["token_chunk_ends"]["bad_value_rows"] >= 1


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
        doc_ids=(1, 1, 1, 2, 2, 2, 2, 2),
        loss_mask=(1, 1, 1, 1, 0, 0, 0, 0),  # WRONG: boundary at pos 2 is masked 1
        valid_token_count=5,
        trained_token_count=4,
        extra={"source_doc_ids": [7, 9], "source_doc_token_lengths": [3, 2]},
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
        doc_ids=(1, 1, 1, 2, 2, 2, 2, 2),
        loss_mask=(1, 1, 0, 1, 0, 0, 0, 0),  # canonical: 0 at the inter-doc boundary
        valid_token_count=5,
        trained_token_count=3,
        extra={"source_doc_ids": [7, 9], "source_doc_token_lengths": [3, 2]},
    )

    proc, report = _run_audit(tmp_path, code_root, commit_root, pr_root)

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert report["total"]["bad_rows"] == 0
    assert report["total"]["fields"]["loss_mask"]["bad_value_rows"] == 0


def test_sidecar_audit_rejects_collapsed_doc_ids_even_when_mask_matches_them(tmp_path):
    code_root = tmp_path / "code"
    commit_root = tmp_path / "commits"
    pr_root = tmp_path / "pr"

    _write_tiny_parquet(code_root / "8" / "code.parquet")
    _write_tiny_parquet(pr_root / "8" / "pr.parquet")
    _write_tiny_parquet(
        commit_root / "8" / "commit.parquet",
        input_ids=(10, 11, 12, 13, 14, 0, 0, 0),
        target_ids=(11, 12, 13, 14, 0, 0, 0, 0),
        doc_ids=(11, 11, 11, 11, 11, 11, 11, 11),
        loss_mask=(1, 1, 1, 1, 0, 0, 0, 0),
        valid_token_count=5,
        trained_token_count=4,
        extra={"source_doc_ids": [38, 41], "source_doc_token_lengths": [3, 2]},
    )

    proc, report = _run_audit(tmp_path, code_root, commit_root, pr_root)

    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert report["total"]["fields"]["doc_ids"]["bad_value_rows"] >= 1
    assert report["total"]["fields"]["loss_mask"]["bad_value_rows"] >= 1


def test_sidecar_audit_rejects_bad_target_shift(tmp_path):
    code_root = tmp_path / "code"
    commit_root = tmp_path / "commits"
    pr_root = tmp_path / "pr"

    _write_tiny_parquet(code_root / "8" / "code.parquet")
    _write_tiny_parquet(pr_root / "8" / "pr.parquet")
    _write_tiny_parquet(
        commit_root / "8" / "commit.parquet",
        input_ids=(10, 11, 12, 13, 0, 0, 0, 0),
        target_ids=(11, 12, 99, 0, 0, 0, 0, 0),
    )

    proc, report = _run_audit(tmp_path, code_root, commit_root, pr_root)

    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert report is not None
    assert report["total"]["bad_rows"] >= 1
    assert report["total"]["fields"]["target_ids"]["bad_value_rows"] >= 1


def test_sidecar_audit_rejects_domain_delimiter_without_domain_sidecars(tmp_path):
    code_root = tmp_path / "code"
    commit_root = tmp_path / "commits"
    pr_root = tmp_path / "pr"

    _write_tiny_parquet(
        code_root / "8" / "code.parquet",
        input_ids=(195, 2, 3, 196, 0, 0, 0, 0),
        target_ids=(2, 3, 196, 0, 0, 0, 0, 0),
        valid_token_count=4,
        trained_token_count=3,
    )
    _write_tiny_parquet(commit_root / "8" / "commit.parquet")
    _write_tiny_parquet(pr_root / "8" / "pr.parquet")

    proc, report = _run_audit(tmp_path, code_root, commit_root, pr_root)

    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert report is not None
    assert report["total"]["bad_rows"] >= 1
    assert any("domain delimiter tokens" in err for err in report["total"]["errors"])


def test_sidecar_audit_accepts_domain_delimiter_with_domain_sidecars_and_edges(tmp_path):
    code_root = tmp_path / "code"
    commit_root = tmp_path / "commits"
    pr_root = tmp_path / "pr"

    extra = {
        "token_domain_ids": [2, 2, 2, 2, 0, 0, 0, 0],
        "token_role_ids": [1, 6, 4, 1, 0, 0, 0, 0],
        "token_entity_ids": [0, 1, 2, 0, 0, 0, 0, 0],
        "token_scope_ids": [0, 0, 1, 0, 0, 0, 0, 0],
        "token_source_doc_ids": [1, 1, 1, 1, 0, 0, 0, 0],
        "token_confidence_ids": [4, 4, 4, 4, 0, 0, 0, 0],
        "token_domain_edges": [{"from": 1, "to": 2, "kind": 5}],
        "token_build_edges": [{"from": 1, "to": 2, "kind": 26}],
        "token_shell_edges": [],
        "token_diagnostic_edges": [],
        "token_cross_domain_edges": [],
    }
    _write_tiny_parquet(
        code_root / "8" / "code.parquet",
        input_ids=(195, 2, 3, 196, 0, 0, 0, 0),
        target_ids=(2, 3, 196, 0, 0, 0, 0, 0),
        valid_token_count=4,
        trained_token_count=3,
        extra=extra,
    )
    _write_tiny_parquet(commit_root / "8" / "commit.parquet")
    _write_tiny_parquet(pr_root / "8" / "pr.parquet")

    proc, report = _run_audit(tmp_path, code_root, commit_root, pr_root)

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert report["total"]["edge_count"]["token_domain_edges"] == 1
    assert report["total"]["fields"]["token_domain_ids"]["bad_length_rows"] == 0
    assert report["total"]["fields"]["token_role_ids"]["bad_value_rows"] == 0


def test_sidecar_audit_requires_uint32_positive_token_source_doc_ids(tmp_path):
    cases = (
        ("missing", {"omit": ("token_source_doc_ids",)}),
        (
            "zero",
            {
                "extra": {
                    "token_source_doc_ids": [0, 0, 0, 0, 0, 0, 0, 0]
                }
            },
        ),
        ("wrong_dtype", {"token_source_doc_type": pa.int64()}),
    )
    for label, kwargs in cases:
        case_root = tmp_path / label
        code_root = case_root / "code"
        commit_root = case_root / "commits"
        pr_root = case_root / "pr"
        _write_tiny_parquet(code_root / "8" / "code.parquet", **kwargs)
        _write_tiny_parquet(commit_root / "8" / "commit.parquet")
        _write_tiny_parquet(pr_root / "8" / "pr.parquet")

        proc, report = _run_audit(case_root, code_root, commit_root, pr_root)

        assert proc.returncode == 2, (label, proc.stdout, proc.stderr)
        assert report["total"]["bad_rows"] >= 1


def test_sidecar_audit_rejects_zero_stable_source_id_inside_valid_prefix(tmp_path):
    code_root = tmp_path / "code"
    commit_root = tmp_path / "commits"
    pr_root = tmp_path / "pr"
    _write_tiny_parquet(
        code_root / "8" / "code.parquet",
        extra={"token_source_doc_ids": [7, 0, 7, 7, 0, 0, 0, 0]},
    )
    _write_tiny_parquet(commit_root / "8" / "commit.parquet")
    _write_tiny_parquet(pr_root / "8" / "pr.parquet")

    proc, report = _run_audit(tmp_path, code_root, commit_root, pr_root)

    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert report is not None
    assert report["total"]["fields"]["token_source_doc_ids"]["bad_value_rows"] >= 1


def test_sidecar_audit_rejects_edge_crossing_source_document_provenance(tmp_path):
    code_root = tmp_path / "code"
    commit_root = tmp_path / "commits"
    pr_root = tmp_path / "pr"
    _write_tiny_parquet(
        code_root / "8" / "code.parquet",
        doc_ids=(1, 1, 2, 2, 2, 2, 2, 2),
        loss_mask=(1, 0, 1, 0, 0, 0, 0, 0),
        trained_token_count=2,
        extra={
            "source_doc_ids": [11, 22],
            "source_doc_token_lengths": [2, 2],
            "token_source_doc_ids": [11, 11, 22, 22, 0, 0, 0, 0],
            "token_domain_edges": [{"from": 0, "to": 2, "kind": 5}],
        },
    )
    _write_tiny_parquet(commit_root / "8" / "commit.parquet")
    _write_tiny_parquet(pr_root / "8" / "pr.parquet")

    proc, report = _run_audit(tmp_path, code_root, commit_root, pr_root)

    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert report["total"]["fields"]["token_domain_edges"]["bad_value_rows"] >= 1


@pytest.mark.parametrize(
    ("field", "valid_kind", "wrong_kind"),
    [
        ("token_domain_edges", 5, 26),
        ("token_build_edges", 26, 5),
        ("token_shell_edges", 44, 26),
        ("token_diagnostic_edges", 60, 26),
        ("token_cross_domain_edges", 100, 26),
    ],
)
def test_sidecar_audit_enforces_independent_edge_kind_families(
    tmp_path, field, valid_kind, wrong_kind
):
    for label, kind, expected_code in (
        ("valid", valid_kind, 0),
        ("wrong", wrong_kind, 2),
    ):
        case_root = tmp_path / label
        code_root = case_root / "code"
        commit_root = case_root / "commits"
        pr_root = case_root / "pr"
        _write_tiny_parquet(
            code_root / "8" / "code.parquet",
            extra={field: [{"from": 1, "to": 2, "kind": kind}]},
        )
        _write_tiny_parquet(commit_root / "8" / "commit.parquet")
        _write_tiny_parquet(pr_root / "8" / "pr.parquet")

        proc, report = _run_audit(case_root, code_root, commit_root, pr_root)

        assert proc.returncode == expected_code, (label, proc.stdout, proc.stderr)
        if expected_code:
            assert report["total"]["fields"][field]["bad_value_rows"] >= 1


@pytest.mark.parametrize(
    ("label", "input_ids", "valid", "domains", "roles", "confidence"),
    [
        (
            "wrong_domain",
            (195, 2, 3, 196, 0, 0, 0, 0),
            4,
            [3, 2, 2, 2, 0, 0, 0, 0],
            [1, 6, 4, 1, 0, 0, 0, 0],
            [4, 4, 4, 4, 0, 0, 0, 0],
        ),
        (
            "swapped",
            (196, 2, 3, 195, 0, 0, 0, 0),
            4,
            [2, 2, 2, 2, 0, 0, 0, 0],
            [1, 6, 4, 1, 0, 0, 0, 0],
            [4, 4, 4, 4, 0, 0, 0, 0],
        ),
        (
            "crossing",
            (195, 193, 2, 196, 194, 3, 0, 0),
            6,
            [2, 3, 3, 2, 3, 0, 0, 0],
            [1, 1, 6, 1, 1, 2, 0, 0],
            [4, 4, 4, 4, 4, 4, 0, 0],
        ),
        (
            "unclosed",
            (195, 2, 3, 4, 0, 0, 0, 0),
            4,
            [2, 2, 2, 2, 0, 0, 0, 0],
            [1, 6, 4, 2, 0, 0, 0, 0],
            [4, 4, 4, 4, 0, 0, 0, 0],
        ),
        (
            "wrong_confidence",
            (195, 2, 3, 196, 0, 0, 0, 0),
            4,
            [2, 2, 2, 2, 0, 0, 0, 0],
            [1, 6, 4, 1, 0, 0, 0, 0],
            [3, 4, 4, 4, 0, 0, 0, 0],
        ),
    ],
)
def test_sidecar_audit_rejects_invalid_delimiter_semantics(
    tmp_path, label, input_ids, valid, domains, roles, confidence
):
    code_root = tmp_path / label / "code"
    commit_root = tmp_path / label / "commits"
    pr_root = tmp_path / label / "pr"
    target_ids = tuple(input_ids[1:valid]) + (0,) * (len(input_ids) - valid + 1)
    _write_tiny_parquet(
        code_root / "8" / "code.parquet",
        input_ids=input_ids,
        target_ids=target_ids,
        loss_mask=(1,) * (valid - 1) + (0,) * (len(input_ids) - valid + 1),
        valid_token_count=valid,
        trained_token_count=valid - 1,
        extra={
            "source_doc_token_lengths": [valid],
            "token_domain_ids": domains,
            "token_role_ids": roles,
            "token_entity_ids": [0] * len(input_ids),
            "token_scope_ids": [0] * len(input_ids),
            "token_confidence_ids": confidence,
        },
    )
    _write_tiny_parquet(commit_root / "8" / "commit.parquet")
    _write_tiny_parquet(pr_root / "8" / "pr.parquet")

    proc, report = _run_audit(tmp_path / label, code_root, commit_root, pr_root)

    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert report["total"]["bad_rows"] >= 1


def test_sidecar_audit_rejects_delimiter_role_on_arbitrary_token(tmp_path):
    code_root = tmp_path / "code"
    commit_root = tmp_path / "commits"
    pr_root = tmp_path / "pr"
    _write_tiny_parquet(
        code_root / "8" / "code.parquet",
        extra={
            "token_domain_ids": [0] * 8,
            "token_role_ids": [0, 1, 0, 0, 0, 0, 0, 0],
            "token_confidence_ids": [0] * 8,
        },
    )
    _write_tiny_parquet(commit_root / "8" / "commit.parquet")
    _write_tiny_parquet(pr_root / "8" / "pr.parquet")

    proc, report = _run_audit(tmp_path, code_root, commit_root, pr_root)

    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert report["total"]["fields"]["token_role_ids"]["bad_value_rows"] >= 1


def test_sidecar_audit_rejects_bad_trained_token_count(tmp_path):
    code_root = tmp_path / "code"
    commit_root = tmp_path / "commits"
    pr_root = tmp_path / "pr"

    _write_tiny_parquet(code_root / "8" / "code.parquet")
    _write_tiny_parquet(pr_root / "8" / "pr.parquet")
    _write_tiny_parquet(
        commit_root / "8" / "commit.parquet",
        loss_mask=(1, 1, 1, 0, 0, 0, 0, 0),
        trained_token_count=4,
    )

    proc, report = _run_audit(tmp_path, code_root, commit_root, pr_root)

    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert report is not None
    assert report["total"]["bad_rows"] >= 1
    assert any(
        "trained_token_count != sum(loss_mask)" in err
        for err in report["total"]["errors"]
    )
