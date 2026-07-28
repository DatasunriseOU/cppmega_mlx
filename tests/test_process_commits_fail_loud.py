from __future__ import annotations

import hashlib
import io
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
        "legacy_identity_overrides": 0,
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
    assert not (tmp_path / "out.jsonl").exists()


def test_process_commits_allows_explicit_lossy_mode(tmp_path, monkeypatch) -> None:
    assert _run_main(tmp_path, monkeypatch, allow=True) == 0
    assert (tmp_path / "out.jsonl").exists()


def test_allow_parse_errors_never_swallows_identity_failure(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "records.jsonl"
    source.write_text("{}\n", encoding="utf-8")
    output = tmp_path / "out.jsonl"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "process_commits.py",
            "--inputs",
            str(source),
            "--output",
            str(output),
            "--allow-parse-errors",
        ],
    )
    monkeypatch.setattr(process_commits, "start_memory_guard", lambda *_a, **_k: None)
    monkeypatch.setattr(process_commits, "_configure_libclang", lambda _path: None)
    monkeypatch.setattr(process_commits, "_load_tokenizer", lambda _path: None)
    monkeypatch.setattr(process_commits, "Index", SimpleNamespace(create=lambda: object()))

    def fail_identity(_input, out_f, *_args, **_kwargs):
        out_f.write('{"partial": true}\n')
        raise process_commits.SymbolIdentityError("synthetic identity collision")

    monkeypatch.setattr(process_commits, "process_jsonl_file", fail_identity)
    with pytest.raises(process_commits.SymbolIdentityError, match="identity collision"):
        process_commits.main()
    assert not output.exists()


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


def test_process_jsonl_overrides_legacy_identity_before_pr_lookup(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "records.jsonl"
    source.write_text(
        json.dumps(
            {
                "repo": "_src",
                "repo_stable_id": "stale-repo-id",
                "filepath": "source/blender/editors/object/object.cc",
                "filepath_stable_id": "stale-file-id",
                "commit_hash": "abc123",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    expected_repo = "blender/blender"
    expected_repo_id = hashlib.sha1(expected_repo.encode("utf-8")).hexdigest()[:16]
    expected_file_id = hashlib.sha1(
        f"{expected_repo}\0source/blender/editors/object/object.cc".encode("utf-8")
    ).hexdigest()[:16]

    class RecordingLookup:
        def attach(self, record: dict) -> bool:
            assert record["repo"] == expected_repo
            assert record["repo_stable_id"] == expected_repo_id
            assert record["filepath_stable_id"] == expected_file_id
            return False

    monkeypatch.setattr(process_commits, "process_record", lambda *_a, **_k: [])

    stats = process_commits.process_jsonl_file(
        str(source),
        io.StringIO(),
        object(),
        str(tmp_path / "work"),
        4096,
        500000,
        "both",
        5,
        10.0,
        pr_lookup=RecordingLookup(),
        project_id="blender/blender",
        analysis_cache_entries=0,
    )

    assert stats["legacy_identity_overrides"] == 1


def test_process_jsonl_rejects_conflicting_canonical_project_identity(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "records.jsonl"
    source.write_text(
        json.dumps(
            {
                "repo": "other/project",
                "filepath": "src/file.cc",
                "commit_hash": "abc123",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(process_commits, "process_record", lambda *_a, **_k: [])

    with pytest.raises(process_commits.SymbolIdentityError, match="conflicts"):
        process_commits.process_jsonl_file(
            str(source),
            io.StringIO(),
            object(),
            str(tmp_path / "work"),
            4096,
            500000,
            "both",
            5,
            10.0,
            project_id="blender/blender",
            analysis_cache_entries=0,
        )


def test_analyze_file_clang_surfaces_translation_unit_parse_failure(
    tmp_path: Path, monkeypatch
) -> None:
    class BrokenIndex:
        def parse(self, *_args, **_kwargs):
            raise RuntimeError("synthetic clang failure")

    monkeypatch.setattr(
        process_commits,
        "TranslationUnit",
        SimpleNamespace(PARSE_INCOMPLETE=0),
    )

    with pytest.raises(RuntimeError, match=r"libclang parse failed.*src/broken\.cpp"):
        process_commits.analyze_file_clang(
            "int broken(int value) { return value; }",
            "src/broken.cpp",
            BrokenIndex(),
            str(tmp_path / "clang-tmp"),
            repo_root=str(tmp_path),
            project_id="owner/repo-a",
        )


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


def test_pr_lookup_filters_stale_rows_by_exact_scan_id(tmp_path: Path) -> None:
    store = tmp_path / "prs.sqlite"
    conn = process_commits._pr_store_mod.connect(str(store), create=True)
    try:
        process_commits._pr_store_mod.upsert_record(
            conn,
            {
                "repo": "owner/repo",
                "pr_number": 1,
                "merge_commit_sha": "old-sha",
                "pr_title": "Old membership",
                "pr_body": "Must not be attached.",
                "comments": [],
                "reviews": [],
                "linked_issues": [],
            },
            scan_id="a" * 64,
        )
        process_commits._pr_store_mod.upsert_record(
            conn,
            {
                "repo": "owner/repo",
                "pr_number": 1,
                "merge_commit_sha": "current-sha",
                "pr_title": "Current membership",
                "pr_body": "Verified discussion.",
                "comments": [],
                "reviews": [],
                "linked_issues": [],
            },
            scan_id="a" * 64,
        )
        process_commits._pr_store_mod.upsert_record(
            conn,
            {
                "repo": "owner/repo",
                "pr_number": 2,
                "merge_commit_sha": "stale-sha",
                "pr_title": "Stale row",
                "pr_body": "Must not be attached.",
                "comments": [],
                "reviews": [],
                "linked_issues": [],
            },
            scan_id="b" * 64,
        )
    finally:
        conn.close()

    lookup = process_commits.PRDiscussionLookup(
        str(store),
        None,
        scan_id="a" * 64,
        owner_repo="owner/repo",
    )
    try:
        current = {"repo": "owner/repo", "commit_hash": "current-sha"}
        assert lookup.attach(current) is True
        assert current["pr_number"] == 1
        assert "Verified discussion." in current["pr_discussion"]

        assert lookup.attach(
            {"repo": "owner/repo", "commit_hash": "old-sha"}
        ) is False
        assert lookup.attach(
            {"repo": "owner/repo", "commit_hash": "stale-sha"}
        ) is False
        assert lookup.attach(
            {"repo": "owner/repo", "pr_number": 2}
        ) is False
    finally:
        lookup.close()

    with pytest.raises(ValueError, match="invalid PR scan_id"):
        process_commits.PRDiscussionLookup(
            str(store),
            None,
            scan_id="not-a-scan",
        )
