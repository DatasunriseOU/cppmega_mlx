from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

import pyarrow.parquet as pq
import pytest

from cppmega_mlx.data.nanochat_pipeline.packed_rows_schema import (
    INPUT_IDS_COLUMN,
    LOSS_MASK_COLUMN,
    TARGET_IDS_COLUMN,
    VALID_TOKEN_COUNT_COLUMN,
)
from cppmega_mlx.data.nanochat_pipeline.tokenized_enriched_schema import (
    CHANGED_CHUNK_IDS_COLUMN,
    CHANGED_CHUNK_SPANS_COLUMN,
    TOKEN_BUILD_EDGES_COLUMN,
    TOKEN_CALL_EDGES_COLUMN,
    TOKEN_CHUNK_DEP_LEVELS_COLUMN,
    TOKEN_CHUNK_ENDS_COLUMN,
    TOKEN_CHUNK_KINDS_COLUMN,
    TOKEN_CHUNK_STARTS_COLUMN,
    TOKEN_CROSS_DOMAIN_EDGES_COLUMN,
    TOKEN_DIAGNOSTIC_EDGES_COLUMN,
    TOKEN_DOMAIN_EDGES_COLUMN,
    TOKEN_IDS_COLUMN,
    TOKEN_SHELL_EDGES_COLUMN,
    TOKEN_TYPE_EDGES_COLUMN,
)
from scripts.nanochat_data.token_budget import load_tokenizer
from scripts.audit_sidecar_parquet import (
    _discover as discover_audit_inputs,
    build_arg_parser as build_audit_arg_parser,
)
from scripts.tokenize_ci_enriched import (
    CI_MANIFEST_SCHEMA,
    DEFAULT_SEQ_LENGTHS,
    capture_ci_code_revision,
    discover_ci_input_files,
    iter_ci_jsonl_files,
    load_ci_log_completion,
    prepare_ci_document,
    split_tokenized_ci_document,
    tokenize_and_pack,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(repo: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _raw_ci_doc(index: int, diagnostic_lines: int) -> dict[str, object]:
    text = "".join(
        f"/src/f{line}.cpp:{line}:3: error: undefined symbol value_{line}\n"
        for line in range(diagnostic_lines)
    )
    return {
        "text": text,
        "source_doc_id": f"ci:{index}:{index}",
        "repo": "owner/repo",
        "filepath": "src/f.cpp",
        "commit_hash": "a" * 40,
        "doc_type": "diagnostic",
        "ci_metadata": {
            "run_id": index,
            "job_id": index,
            "conclusion": "failure",
            "platform": "ubuntu-24.04",
            "compiler_info": "clang",
            "diagnostics_count": diagnostic_lines,
            "severities": ["error"],
        },
        "domain_sidecars": {},
        "symbol_identities": [],
    }


def test_ci_input_directory_is_exact_and_malformed_rows_fail_closed(
    tmp_path: Path,
) -> None:
    logs = tmp_path / "ci_logs_enriched.jsonl"
    paired = tmp_path / "ci_paired_enriched.jsonl"
    logs.write_text(json.dumps(_raw_ci_doc(1, 2)) + "\n", encoding="utf-8")
    paired.write_text(json.dumps(_raw_ci_doc(2, 2)) + "\n", encoding="utf-8")

    assert discover_ci_input_files(tmp_path) == [logs.resolve(), paired.resolve()]

    extra = tmp_path / "unbound_enriched.jsonl"
    extra.write_text(json.dumps(_raw_ci_doc(3, 2)) + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="unbound extra JSONL"):
        discover_ci_input_files(tmp_path)
    extra.unlink()

    paired.write_text("{not-json}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="malformed JSON"):
        list(iter_ci_jsonl_files(discover_ci_input_files(tmp_path)))

    malformed = _raw_ci_doc(2, 2)
    malformed["ci_metadata"]["severities"] = "error"  # type: ignore[index]
    paired.write_text(json.dumps(malformed) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="severities must be a list"):
        list(iter_ci_jsonl_files(discover_ci_input_files(tmp_path)))


def test_ci_log_completion_binds_exact_output_and_http_410_gaps(
    tmp_path: Path,
) -> None:
    logs = tmp_path / "ci_logs_enriched.jsonl"
    paired = tmp_path / "ci_paired_enriched.jsonl"
    state = tmp_path / "ci_logs_enriched.fetch-state.jsonl"
    receipt = tmp_path / "ci_logs_enriched.completion.json"
    logs.write_text(json.dumps(_raw_ci_doc(1, 2)) + "\n", encoding="utf-8")
    paired.write_text(json.dumps(_raw_ci_doc(3, 2)) + "\n", encoding="utf-8")
    state.write_text(
        "\n".join(
            (
                json.dumps({"job_id": 1, "status": "fetched"}),
                json.dumps({"job_id": 2, "status": "expired"}),
            )
        )
        + "\n",
        encoding="utf-8",
    )
    payload = {
        "schema": "cppmega_ci_log_extraction_v1",
        "status": "complete",
        "unique_job_count": 2,
        "fetched_count": 1,
        "expired_count": 1,
        "too_short_count": 0,
        "unresolved_count": 0,
        "source_inventory_sha256": "a" * 64,
        "job_set_sha256": "b" * 64,
        "output": {
            "path": str(logs.resolve()),
            "row_count": 1,
            "size": logs.stat().st_size,
            "sha256": _sha256(logs),
        },
        "state": {
            "path": str(state.resolve()),
            "row_count": 2,
            "size": state.stat().st_size,
            "sha256": _sha256(state),
        },
        "expired_jobs": [
            {
                "job_id": 2,
                "repo": "owner/repo",
                "detail": "gh: Server Error (HTTP 410)",
            }
        ],
    }
    receipt.write_text(json.dumps(payload), encoding="utf-8")

    completion = load_ci_log_completion(receipt, logs_path=logs)

    assert completion["status"] == "complete"
    assert completion["fetched_count"] == 1
    assert completion["expired_count"] == 1
    assert completion["receipt_sha256"] == _sha256(receipt)
    assert discover_ci_input_files(
        tmp_path,
        allowed_auxiliary_jsonl=(state,),
    ) == [logs.resolve(), paired.resolve()]

    logs.write_text("drifted\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="output binding drifted"):
        load_ci_log_completion(receipt, logs_path=logs)


def test_ci_producer_revision_v2_binds_canonical_mlx_source_tree(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "cppmega.mlx"
    script = repo / "scripts" / "producer.py"
    script.parent.mkdir(parents=True)
    script.write_text("VALUE = 1\n", encoding="utf-8")
    _git(repo, "init", "-q")
    _git(repo, "config", "user.name", "CI Revision Test")
    _git(repo, "config", "user.email", "ci-revision@example.test")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "initial")
    commit = _git(repo, "rev-parse", "HEAD")

    receipt = capture_ci_code_revision(commit, repo_root=repo)

    assert receipt["schema"] == "cppmega_ci_code_revision_v2"
    assert receipt["schema_version"] == 2
    assert receipt["repository_identity"] == "cppmega.mlx"
    assert receipt["git_commit"] == commit
    assert len(str(receipt["source_tree_sha256"])) == 64
    assert receipt["dirty"] is False

    script.write_text("VALUE = 2\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="clean canonical"):
        capture_ci_code_revision(commit, repo_root=repo)


def test_ci_split_is_token_lossless_and_counts_unrepresentable_edges() -> None:
    record = {
        TOKEN_IDS_COLUMN: list(range(17_000)),
        "source_doc_id": "ci:1:2",
        TOKEN_CHUNK_STARTS_COLUMN: [0, 8_500, 16_000],
        TOKEN_CHUNK_ENDS_COLUMN: [8_500, 16_000, 17_000],
        TOKEN_CHUNK_KINDS_COLUMN: [1, 2, 3],
        TOKEN_CHUNK_DEP_LEVELS_COLUMN: [0, 1, 2],
        TOKEN_CALL_EDGES_COLUMN: [{"from": 0, "to": 1}, {"from": 0, "to": 2}],
        TOKEN_TYPE_EDGES_COLUMN: [],
        TOKEN_DOMAIN_EDGES_COLUMN: [],
        TOKEN_BUILD_EDGES_COLUMN: [],
        TOKEN_SHELL_EDGES_COLUMN: [],
        TOKEN_DIAGNOSTIC_EDGES_COLUMN: [
            {"from": 100, "to": 16_900, "kind": 60}
        ],
        TOKEN_CROSS_DOMAIN_EDGES_COLUMN: [],
        CHANGED_CHUNK_IDS_COLUMN: [],
        CHANGED_CHUNK_SPANS_COLUMN: [],
    }

    fragments, counters = split_tokenized_ci_document(
        record,
        seq_lengths=list(DEFAULT_SEQ_LENGTHS),
    )

    assert [len(fragment[TOKEN_IDS_COLUMN]) for fragment in fragments] == [
        8_500,
        8_500,
    ]
    assert [
        token
        for fragment in fragments
        for token in fragment[TOKEN_IDS_COLUMN]
    ] == record[TOKEN_IDS_COLUMN]
    assert counters["source_tokens"] == 17_000
    assert counters["fragment_tokens"] == 17_000
    assert counters["fragments"] == 2
    assert counters["split_source_docs"] == 1
    assert counters["cross_boundary_chunk_edges"] == 2
    assert counters["cross_boundary_token_edges"] == 1
    assert counters["source_token_call_edges"] == 2
    assert counters["fragment_token_call_edges"] == 0
    assert counters["source_token_diagnostic_edges"] == 1
    assert counters["fragment_token_diagnostic_edges"] == 0
    assert fragments[0][TOKEN_CALL_EDGES_COLUMN] == []
    assert fragments[1][TOKEN_CALL_EDGES_COLUMN] == []
    assert all(
        fragment[TOKEN_DIAGNOSTIC_EDGES_COLUMN] == []
        for fragment in fragments
    )


def test_ci_blank_lines_tokenize_and_malformed_explicit_edges_fail_closed(
    tmp_path: Path,
) -> None:
    document = _raw_ci_doc(1, 2)
    document["text"] = (
        "/src/a.cpp:1:3: error: first\n"
        "\n"
        "/src/b.cpp:2:3: error: second\n"
    )
    summary = tokenize_and_pack(
        [document],
        tokenizer=load_tokenizer(),
        seq_lengths=list(DEFAULT_SEQ_LENGTHS),
        output_dir=tmp_path,
        timestamp="blank-lines",
        dry_run=True,
        require_nonempty_buckets=False,
    )
    assert summary["counters"]["input_docs"] == 1
    assert summary["counters"]["source_tokens"] > 0
    assert summary["counters"]["source_tokens"] == summary["counters"][
        "fragment_tokens"
    ]

    document["domain_sidecars"] = {
        "diagnostic_edges": [{"from_char": 0, "kind": 1}]
    }
    with pytest.raises(ValueError, match="both from_char and to_char"):
        prepare_ci_document(document, doc_index=0)


def test_ci_pipeline_publishes_only_verified_five_bucket_manifest(
    tmp_path: Path,
) -> None:
    docs = [
        _raw_ci_doc(index, lines)
        for index, lines in enumerate((20, 40, 100, 200, 400, 800), start=1)
    ]
    manifest = tokenize_and_pack(
        docs,
        tokenizer=load_tokenizer(),
        seq_lengths=list(DEFAULT_SEQ_LENGTHS),
        output_dir=tmp_path,
        timestamp="fixture",
        batch_size=2,
    )

    output_dir = tmp_path / "reindexed_ci_fixture_ci"
    persisted = json.loads(
        (output_dir / "manifest.json").read_text(encoding="utf-8")
    )
    assert persisted["schema"] == CI_MANIFEST_SCHEMA
    assert persisted["kind"] == "ci"
    assert persisted["seq_lengths"] == list(DEFAULT_SEQ_LENGTHS)
    assert set(persisted["buckets"]) == {
        str(bucket) for bucket in DEFAULT_SEQ_LENGTHS
    }
    assert persisted["counters"]["input_docs"] == len(docs)
    assert persisted["counters"]["source_tokens"] == persisted["counters"][
        "fragment_tokens"
    ]
    assert persisted["counters"]["split_source_docs"] == 1
    assert persisted["counters"]["unexpected_rejects"] == 0
    assert persisted["counters"]["packing_overflow_docs"] == 0
    assert not (output_dir / "oversized").exists()
    assert manifest["output_dir"] == str(output_dir)

    for bucket in DEFAULT_SEQ_LENGTHS:
        receipt = persisted["buckets"][str(bucket)]
        parquet_path = output_dir / receipt["parquet"]["path"]
        bucket_manifest_path = output_dir / receipt["manifest"]["path"]
        assert _sha256(parquet_path) == receipt["parquet"]["sha256"]
        assert _sha256(bucket_manifest_path) == receipt["manifest"]["sha256"]
        rows = 0
        valid_tokens = 0
        for batch in pq.ParquetFile(parquet_path).iter_batches(
            columns=[
                VALID_TOKEN_COUNT_COLUMN,
                INPUT_IDS_COLUMN,
                TARGET_IDS_COLUMN,
                LOSS_MASK_COLUMN,
            ]
        ):
            for row in batch.to_pylist():
                rows += 1
                valid_tokens += int(row[VALID_TOKEN_COUNT_COLUMN])
                assert len(row[INPUT_IDS_COLUMN]) == bucket
                assert len(row[TARGET_IDS_COLUMN]) == bucket
                assert len(row[LOSS_MASK_COLUMN]) == bucket
        assert rows == receipt["packed_rows"]
        assert valid_tokens == receipt["valid_tokens"]


def test_ci_pipeline_refuses_incomplete_bucket_ladder_without_publish(
    tmp_path: Path,
) -> None:
    with pytest.raises(RuntimeError, match="production buckets are empty"):
        tokenize_and_pack(
            [_raw_ci_doc(1, 20)],
            tokenizer=load_tokenizer(),
            seq_lengths=list(DEFAULT_SEQ_LENGTHS),
            output_dir=tmp_path,
            timestamp="incomplete",
            batch_size=1,
        )

    assert not (tmp_path / "reindexed_ci_incomplete_ci").exists()
    assert not (tmp_path / ".reindexed_ci_incomplete_ci.partial").exists()


def test_sidecar_audit_exposes_ci_as_a_distinct_kind(tmp_path: Path) -> None:
    ci_parquet = tmp_path / "ci" / "1024" / "fixture.parquet"
    ci_parquet.parent.mkdir(parents=True)
    ci_parquet.touch()
    args = build_audit_arg_parser().parse_args(
        [
            "--code-root",
            str(tmp_path / "code"),
            "--commit-root",
            str(tmp_path / "commits"),
            "--pr-root",
            str(tmp_path / "pr"),
            "--ci-root",
            str(tmp_path / "ci"),
        ]
    )

    assert args.ci_root == tmp_path / "ci"
    assert discover_audit_inputs(args.ci_root, "ci", {"1024"}) == [
        (str(ci_parquet), "ci", "1024")
    ]
