from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path
from types import SimpleNamespace


CLANG_INDEXER = Path(__file__).resolve().parents[1] / "tools" / "clang_indexer"
if str(CLANG_INDEXER) not in sys.path:
    sys.path.insert(0, str(CLANG_INDEXER))

import process_commits  # noqa: E402
import pytest  # noqa: E402


def _stats(*, parse_errors: int) -> dict[str, int]:
    return {
        "records_read": 2,
        "documents_written": 1,
        "records_skipped": 0,
        "records_empty": 0,
        "parse_errors": parse_errors,
        "commit_chunks_claimed": 0,
        "commit_chunks_skipped": 0,
        "analysis_cache_hits": 0,
        "analysis_cache_misses": 0,
        "analysis_cache_evictions": 0,
    }


def _run_main(tmp_path: Path, monkeypatch, *, allow: bool) -> int:
    source = tmp_path / "records.jsonl"
    source.write_text("{}\n", encoding="utf-8")
    output = tmp_path / "out.jsonl"
    argv = [
        "process_commits.py",
        "--inputs",
        str(source),
        "--output",
        str(output),
    ]
    if allow:
        argv.append("--allow-parse-errors")
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(process_commits, "start_memory_guard", lambda *_a, **_k: None)
    monkeypatch.setattr(process_commits, "_configure_libclang", lambda _path: None)
    monkeypatch.setattr(process_commits, "_load_tokenizer", lambda _path: None)
    monkeypatch.setattr(process_commits, "Index", SimpleNamespace(create=lambda: object()))
    monkeypatch.setattr(
        process_commits,
        "process_jsonl_file",
        lambda *_args, **_kwargs: _stats(parse_errors=1),
    )
    return process_commits.main()


def test_process_commits_rejects_partial_range_by_default(tmp_path, monkeypatch) -> None:
    assert _run_main(tmp_path, monkeypatch, allow=False) == 1


def test_process_commits_allows_explicit_lossy_mode(tmp_path, monkeypatch) -> None:
    assert _run_main(tmp_path, monkeypatch, allow=True) == 0


def test_process_commits_rejects_missing_input_before_clang_init(tmp_path, monkeypatch) -> None:
    output = tmp_path / "out.jsonl"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "process_commits.py",
            "--inputs",
            str(tmp_path / "missing.jsonl"),
            "--output",
            str(output),
        ],
    )
    monkeypatch.setattr(
        process_commits,
        "_configure_libclang",
        lambda _path: (_ for _ in ()).throw(AssertionError("must not initialize clang")),
    )

    assert process_commits.main() == 1
    assert not output.exists()


def test_sha_pr_lookup_restores_number_title_and_uses_readonly_store(
    tmp_path: Path,
) -> None:
    store = tmp_path / "prs.sqlite"
    conn = process_commits._pr_store_mod.connect(str(store), create=True)
    process_commits._pr_store_mod.upsert_record(
        conn,
        {
            "repo": "owner/repo",
            "pr_number": 1467,
            "merge_commit_sha": "abc123",
            "pr_title": "Fix the build",
            "pr_body": "Body",
            "comments": [],
            "reviews": [],
            "linked_issues": [],
        },
    )
    conn.close()
    repo_list = tmp_path / "repos.json"
    repo_list.write_text(json.dumps({"repos": []}), encoding="utf-8")

    lookup = process_commits.PRDiscussionLookup(str(store), str(repo_list))
    record = {"repo": "owner/repo", "commit_hash": "abc123"}
    try:
        assert lookup.attach(record) is True
        assert record["pr_number"] == 1467
        assert record["pr_title"] == "Fix the build"
        assert "Body" in record["pr_discussion"]
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            lookup._conn.execute("CREATE TABLE forbidden(value INTEGER)")
    finally:
        lookup.close()
