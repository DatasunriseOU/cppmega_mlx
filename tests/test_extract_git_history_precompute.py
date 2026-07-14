from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


MLX_ROOT = Path(__file__).resolve().parents[1]
NANOCHAT = MLX_ROOT / "scripts" / "nanochat_data"
SCRIPT = NANOCHAT / "extract_git_history.py"
if str(NANOCHAT) not in sys.path:
    sys.path.insert(0, str(NANOCHAT))


def _git(repo: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return proc.stdout.strip()


def _commit(repo: Path, message: str) -> str:
    _git(
        repo,
        "-c",
        "user.name=Test User",
        "-c",
        "user.email=test@example.com",
        "commit",
        "-am",
        message,
    )
    return _git(repo, "rev-parse", "HEAD")


def test_precomputed_cpp_files_feed_commit_diffs(tmp_path):
    import extract_git_history as egh

    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    source = repo / "main.cpp"
    source.write_text("int f() { return 1; }\n", encoding="utf-8")
    _git(repo, "add", "main.cpp")
    first = _commit(repo, "initial")

    source.write_text("int f() { return 2; }\n", encoding="utf-8")
    second = _commit(repo, "change value")

    commits = [second, first]
    indices, files_by_commit = egh.precompute_cpp_file_changes(str(repo), commits)

    assert files_by_commit[second] == ["main.cpp"]
    assert indices[(second, "main.cpp")] == 0

    diffs = egh.get_commit_diffs(
        str(repo),
        second,
        files=files_by_commit[second],
    )
    assert diffs is not None
    assert diffs[0]["filepath"] == "main.cpp"
    assert "return 1" in diffs[0]["old_content"]
    assert "return 2" in diffs[0]["new_content"]


def _write_main(repo: Path, first_line: str, last_line: str) -> None:
    """Write main.cpp with a fixed body so only the first/last line vary.

    The four "// pad" lines keep the (changeable) first line and the
    (oversizable) last line more than 3 lines apart, so a small edit to the
    first line never pulls the huge last line into the diff hunk's context.
    """
    (repo / "main.cpp").write_text(
        f"{first_line}\n"
        "// pad1\n"
        "// pad2\n"
        "// pad3\n"
        "// pad4\n"
        f"{last_line}\n",
        encoding="utf-8",
    )


def _build_diff_filtered_repo(tmp_path):
    """Repo whose middle commit is rejected by the diff-size gate.

    Returns ``(repo, c1, c2, c3)`` where main.cpp is modified by:
      * c1: small diff  -> accepted (expected file_local_commit_index 0)
      * c2: diff > MAX_DIFF_CHARS -> REJECTED by get_commit_diffs (no record)
      * c3: small diff  -> accepted (expected file_local_commit_index 1)
    """
    import extract_git_history as egh

    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")

    # Root commit: main.cpp is Added (status 'A'), so it is never an 'M' record
    # and never enters the index.
    _write_main(repo, "int v = 0;", "// small")
    _git(repo, "add", "main.cpp")
    _commit(repo, "initial")

    _write_main(repo, "int v = 1;", "// small")
    c1 = _commit(repo, "first real edit")

    # Oversized diff: the changed last line alone blows past MAX_DIFF_CHARS, so
    # get_commit_diffs rejects this commit. new_content stays under the 200000
    # content cap, so ONLY the diff-size gate filters it out.
    huge = "// " + ("x" * (egh.MAX_DIFF_CHARS + 10000))
    _write_main(repo, "int v = 1;", huge)
    c2 = _commit(repo, "oversized diff edit")

    _write_main(repo, "int v = 2;", huge)
    c3 = _commit(repo, "second real edit")

    return repo, c1, c2, c3


def test_file_local_commit_index_excludes_diff_filtered_commits(tmp_path):
    """A diff-filtered commit must NOT advance the per-file index counter.

    Pins the exact sequence so the documented "same semantics as the original
    post-diff-filter counting" cannot drift silently. Fails on the buggy
    name-status counting (which gives c3 -> 2 because the rejected c2 is
    counted) and passes on the post-diff-filter counting (c3 -> 1).
    """
    import extract_git_history as egh

    repo, c1, c2, c3 = _build_diff_filtered_repo(tmp_path)
    commits = [c3, c2, c1]  # newest-first, matching git log output

    # c2 is genuinely diff-filtered out of the emitted set; c1/c3 are not.
    assert egh.get_commit_diffs(str(repo), c2) is None
    assert egh.get_commit_diffs(str(repo), c1) is not None
    assert egh.get_commit_diffs(str(repo), c3) is not None

    indices, files_by_commit = egh.precompute_cpp_file_changes(str(repo), commits)

    assert indices[(c1, "main.cpp")] == 0
    assert indices[(c3, "main.cpp")] == 1  # contiguous: NOT 2
    assert (c2, "main.cpp") not in indices
    assert c2 not in files_by_commit
    # files_by_commit carries only ACCEPTED (post-diff-filter) files forward.
    assert files_by_commit[c1] == ["main.cpp"]
    assert files_by_commit[c3] == ["main.cpp"]


def test_emitted_records_carry_contiguous_indices_end_to_end(tmp_path):
    """End-to-end: the indices written into JSONL records are contiguous.

    Exercises the real process_repo emit path (no mocks). The rejected c2 is
    absent and c3's emitted file_local_commit_index is 1, not 2.
    """
    import extract_git_history as egh

    repo, c1, c2, c3 = _build_diff_filtered_repo(tmp_path)

    out = tmp_path / "commits.jsonl"
    with out.open("w", encoding="utf-8") as fh:
        egh.process_repo(str(repo), fh, repo_name="repo", notes="off")

    records = [
        json.loads(line) for line in out.read_text().splitlines() if line.strip()
    ]
    main_records = [r for r in records if r["filepath"] == "main.cpp"]
    index_by_commit = {
        r["commit_hash"]: r["file_local_commit_index"] for r in main_records
    }

    assert len(main_records) == 2  # c1 and c3 only; c2 filtered out
    assert c2 not in index_by_commit
    assert index_by_commit[c1] == 0
    assert index_by_commit[c3] == 1  # the rejected commit did not inflate it


def _make_cpp_repo(repo: Path) -> str:
    """Create a real 2-commit C/C++ repo whose HEAD has an emittable 'M' diff."""
    repo.mkdir()
    _git(repo, "init")
    source = repo / "main.cpp"
    source.write_text("int f() { return 1; }\n", encoding="utf-8")
    _git(repo, "add", "main.cpp")
    _commit(repo, "initial")
    source.write_text("int f() { return 2; }\n", encoding="utf-8")
    return _commit(repo, "change value")


def test_main_fails_loud_on_per_repo_extraction_failure(tmp_path):
    """A repo that fails to extract must fail the whole run loud (no silent exit 0).

    Real end-to-end run of the CLI over a --repo_dir with two real git repos:
      * ``repo_ok`` HAS refs/notes -> ``--notes on`` accepts it and it emits
        records (a partial corpus is written to the output JSONL).
      * ``repo_bad`` has NO refs/notes -> ``--notes on`` makes process_repo
        RAISE for that repo only.

    Before the fix, the per-repo ``except`` printed the error, the JSONL
    finalized, and main() returned (exit 0) -- a partial corpus looking like a
    clean run, with no durable record of the failure. After the fix the run must
    exit non-zero AND drop a durable failures manifest naming the failed repo.
    """
    repo_dir = tmp_path / "repos"
    repo_dir.mkdir()

    ok = repo_dir / "repo_ok"
    ok_head = _make_cpp_repo(ok)
    # Give repo_ok a git note so `--notes on` accepts it and it extracts.
    _git(
        ok,
        "-c",
        "user.name=Test User",
        "-c",
        "user.email=test@example.com",
        "notes",
        "add",
        "-m",
        "code review approved",
        ok_head,
    )

    bad = repo_dir / "repo_bad"
    _make_cpp_repo(bad)  # no notes -> `--notes on` raises for this repo

    out = tmp_path / "commits.jsonl"
    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--repo_dir",
            str(repo_dir),
            "--output",
            str(out),
            "--notes",
            "on",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    # Run failed loud (non-zero exit) instead of silently finalizing at exit 0.
    assert proc.returncode != 0, (
        f"expected non-zero exit; stdout=\n{proc.stdout}\nstderr=\n{proc.stderr}"
    )

    # Durable failure manifest written next to the output, naming the failed repo.
    manifest_path = Path(str(out) + ".failures.json")
    assert manifest_path.exists(), "expected a durable failures manifest"
    manifest = json.loads(manifest_path.read_text())
    failed_names = {item["repo_name"] for item in manifest["failed"]}
    assert "repo_bad" in failed_names
    assert "repo_ok" not in failed_names
    assert manifest["failed_count"] == 1
    assert manifest["total_repos"] == 2
    # The error is the real, specific extraction failure (not papered over).
    assert "refs/notes" in manifest["failed"][0]["error"]

    # The partial corpus (repo_ok's records) was still written -- but the run
    # is marked failed, so it can never be mistaken for a clean extraction.
    records = [
        json.loads(line) for line in out.read_text().splitlines() if line.strip()
    ]
    assert records, "repo_ok should have produced at least one record"
    assert all(r["repo"] == "repo_ok" for r in records)


def test_output_file_lock_rejects_second_process(tmp_path):
    import extract_git_history as egh

    output = tmp_path / "repo_commits.jsonl"
    lock = egh.OutputFileLock(output)
    lock.acquire()
    try:
        code = (
            "import sys\n"
            f"sys.path.insert(0, {str(NANOCHAT)!r})\n"
            "import extract_git_history as egh\n"
            f"lock = egh.OutputFileLock({str(output)!r})\n"
            "lock.acquire()\n"
        )
        proc = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode != 0
        assert "already held" in proc.stderr
    finally:
        lock.close()


def _build_linear_history(repo: Path, edits: int = 7) -> list[str]:
    """Create ``edits`` emittable modifications after one root commit."""
    repo.mkdir()
    _git(repo, "init")
    source = repo / "main.cpp"
    source.write_text("int f() { return 0; }\n", encoding="utf-8")
    _git(repo, "add", "main.cpp")
    _commit(repo, "initial")
    commits = []
    for value in range(1, edits + 1):
        source.write_text(f"int f() {{ return {value}; }}\n", encoding="utf-8")
        commits.append(_commit(repo, f"change value {value}"))
    return commits


def _open_checkpoint(
    egh,
    repo: Path,
    root: Path,
    *,
    checkpoint_commits: int = 2,
):
    source = egh._repo_source_context(
        str(repo),
        max_commits=0,
        repo_name=repo.name,
        notes="off",
    )
    checkpoint = egh.RepoExtractionCheckpoint(
        root,
        source=source,
        checkpoint_commits=checkpoint_commits,
    )
    return source, checkpoint


def test_transactional_chunks_preserve_exact_output_order_and_indices(tmp_path):
    import extract_git_history as egh

    repo = tmp_path / "repo"
    commits_oldest_first = _build_linear_history(repo, edits=7)

    legacy_output = tmp_path / "legacy.jsonl"
    with legacy_output.open("w", encoding="utf-8") as handle:
        egh.process_repo(str(repo), handle, repo_name="repo", notes="off")

    source, checkpoint = _open_checkpoint(
        egh,
        repo,
        tmp_path / "checkpoint",
        checkpoint_commits=2,
    )
    try:
        stats = egh.extract_repo_to_checkpoint(
            str(repo),
            checkpoint,
            memory_limit_gb=10.0,
            bad_unit_policy="fail",
            max_bad_units=0,
        )
        output = tmp_path / "transactional.jsonl"
        publication = egh.publish_checkpoints(output, [checkpoint])
    finally:
        checkpoint.close()

    assert output.read_bytes() == legacy_output.read_bytes()
    assert publication["line_count"] == stats["records_written"] == 7
    records = [json.loads(line) for line in output.read_text().splitlines()]
    assert [row["commit_hash"] for row in records] == list(
        reversed(commits_oldest_first)
    )
    assert [row["commit_hash"] for row in records] == source["commits"]
    assert [row["file_local_commit_index"] for row in records] == list(
        reversed(range(7))
    )


def test_transactional_index_preserves_subject_filtered_counter_semantics(tmp_path):
    import extract_git_history as egh

    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    source_file = repo / "main.cpp"
    source_file.write_text("int f() { return 0; }\n", encoding="utf-8")
    _git(repo, "add", "main.cpp")
    _commit(repo, "initial")
    source_file.write_text("int f() { return 1; }\n", encoding="utf-8")
    first = _commit(repo, "first emitted change")
    source_file.write_text("int f() { return 2; }\n", encoding="utf-8")
    filtered = _commit(repo, "clang-format source")
    source_file.write_text("int f() { return 3; }\n", encoding="utf-8")
    last = _commit(repo, "last emitted change")

    _source, checkpoint = _open_checkpoint(
        egh,
        repo,
        tmp_path / "checkpoint",
        checkpoint_commits=2,
    )
    output = tmp_path / "transactional.jsonl"
    try:
        egh.extract_repo_to_checkpoint(
            str(repo),
            checkpoint,
            memory_limit_gb=10.0,
            bad_unit_policy="fail",
            max_bad_units=0,
        )
        egh.publish_checkpoints(output, [checkpoint])
    finally:
        checkpoint.close()

    records = [json.loads(line) for line in output.read_text().splitlines()]
    index_by_commit = {
        row["commit_hash"]: row["file_local_commit_index"] for row in records
    }
    assert filtered not in index_by_commit
    assert index_by_commit[first] == 0
    # The historical precompute counted accepted diffs before subject filtering.
    assert index_by_commit[last] == 2


def test_resume_skips_committed_chunks_and_records_exact_bad_path(
    tmp_path,
    monkeypatch,
):
    import extract_git_history as egh

    repo = tmp_path / "repo"
    commits_oldest_first = _build_linear_history(repo, edits=7)
    failing_commit = commits_oldest_first[4]
    real_get_file_diff = egh.get_file_diff
    calls: list[str] = []

    def fail_one_path(repo_path: str, commit_hash: str, filepath: str):
        calls.append(commit_hash)
        if commit_hash == failing_commit:
            raise egh.UnitExtractionError(
                repo_path=repo_path,
                commit_hash=commit_hash,
                filepath=filepath,
                operation="new_blob",
                error_type="GitCommandError",
                detail="synthetic corrupt object",
            )
        return real_get_file_diff(repo_path, commit_hash, filepath)

    monkeypatch.setattr(egh, "get_file_diff", fail_one_path)
    checkpoint_root = tmp_path / "checkpoint"
    _source, checkpoint = _open_checkpoint(
        egh,
        repo,
        checkpoint_root,
        checkpoint_commits=2,
    )
    with pytest.raises(RuntimeError, match="bad extraction unit rejected"):
        egh.extract_repo_to_checkpoint(
            str(repo),
            checkpoint,
            memory_limit_gb=10.0,
            bad_unit_policy="fail",
            max_bad_units=0,
        )
    committed_before_resume = checkpoint.chunk_rows()
    checkpoint.close()

    assert [(row["start_index"], row["end_index"]) for row in committed_before_resume] == [
        (0, 2),
        (2, 4),
    ]
    ledger_path = checkpoint_root / "bad_units.jsonl"
    first_ledger = [json.loads(line) for line in ledger_path.read_text().splitlines()]
    assert first_ledger[0]["repo"] == "repo"
    assert first_ledger[0]["commit_hash"] == failing_commit
    assert first_ledger[0]["filepath"] == "main.cpp"
    assert first_ledger[0]["operation"] == "new_blob"
    assert first_ledger[0]["error"] == "synthetic corrupt object"

    calls.clear()
    _source, resumed = _open_checkpoint(
        egh,
        repo,
        checkpoint_root,
        checkpoint_commits=2,
    )
    try:
        stats = egh.extract_repo_to_checkpoint(
            str(repo),
            resumed,
            memory_limit_gb=10.0,
            bad_unit_policy="quarantine",
            max_bad_units=1,
        )
        output = tmp_path / "resumed.jsonl"
        egh.publish_checkpoints(output, [resumed])
    finally:
        resumed.close()

    assert not set(commits_oldest_first[:4]).intersection(calls)
    assert failing_commit in calls
    assert stats["records_written"] == 6
    resumed_records = [json.loads(line) for line in output.read_text().splitlines()]
    assert failing_commit not in {row["commit_hash"] for row in resumed_records}
    final_ledger = [json.loads(line) for line in ledger_path.read_text().splitlines()]
    assert len(final_ledger) == 1
    assert final_ledger[0]["attempts"] == 2


def test_quarantine_threshold_fails_after_recording_excess_unit(
    tmp_path,
    monkeypatch,
):
    import extract_git_history as egh

    repo = tmp_path / "repo"
    commits_oldest_first = _build_linear_history(repo, edits=4)
    failing_commits = set(commits_oldest_first[:2])
    real_get_file_diff = egh.get_file_diff

    def fail_two_paths(repo_path: str, commit_hash: str, filepath: str):
        if commit_hash in failing_commits:
            raise egh.UnitExtractionError(
                repo_path=repo_path,
                commit_hash=commit_hash,
                filepath=filepath,
                operation="old_blob",
                error_type="GitCommandError",
                detail=f"missing object for {commit_hash}",
            )
        return real_get_file_diff(repo_path, commit_hash, filepath)

    monkeypatch.setattr(egh, "get_file_diff", fail_two_paths)
    _source, checkpoint = _open_checkpoint(
        egh,
        repo,
        tmp_path / "checkpoint",
        checkpoint_commits=4,
    )
    try:
        with pytest.raises(RuntimeError, match="distinct_bad_units=2"):
            egh.extract_repo_to_checkpoint(
                str(repo),
                checkpoint,
                memory_limit_gb=10.0,
                bad_unit_policy="quarantine",
                max_bad_units=1,
            )
        assert checkpoint.chunk_rows() == []
    finally:
        checkpoint.close()

    ledger = [
        json.loads(line)
        for line in (tmp_path / "checkpoint" / "bad_units.jsonl")
        .read_text()
        .splitlines()
    ]
    assert {row["commit_hash"] for row in ledger} == failing_commits


def test_corrupt_committed_chunk_cannot_replace_published_output(tmp_path):
    import extract_git_history as egh

    repo = tmp_path / "repo"
    _build_linear_history(repo, edits=4)
    _source, checkpoint = _open_checkpoint(
        egh,
        repo,
        tmp_path / "checkpoint",
        checkpoint_commits=2,
    )
    output = tmp_path / "published.jsonl"
    output.write_bytes(b"previous-valid-output\n")
    try:
        egh.extract_repo_to_checkpoint(
            str(repo),
            checkpoint,
            memory_limit_gb=10.0,
            bad_unit_policy="fail",
            max_bad_units=0,
        )
        first_chunk = checkpoint.root / checkpoint.chunk_rows()[0]["artifact_name"]
        with first_chunk.open("ab") as handle:
            handle.write(b"corrupt")

        with pytest.raises(egh.CheckpointCorruptionError, match="changed during publication"):
            egh.publish_checkpoints(output, [checkpoint])
        assert checkpoint.meta("status") == "corrupt"
    finally:
        checkpoint.close()

    assert output.read_bytes() == b"previous-valid-output\n"


def test_cli_publishes_then_reuses_validated_output_without_chunk_replay(tmp_path):
    repo = tmp_path / "repo"
    _build_linear_history(repo, edits=5)
    output = tmp_path / "commits.jsonl"
    command = [
        sys.executable,
        str(SCRIPT),
        "--repo",
        str(repo),
        "--output",
        str(output),
        "--notes",
        "off",
        "--checkpoint-commits",
        "2",
    ]

    first = subprocess.run(command, capture_output=True, text=True, check=False)
    assert first.returncode == 0, f"stdout=\n{first.stdout}\nstderr=\n{first.stderr}"
    original = output.read_bytes()
    assert len(original.splitlines()) == 5
    checkpoint_root = Path(f"{output}.extract-checkpoint")
    publication = json.loads((checkpoint_root / "publication.json").read_text())
    assert publication["status"] == "done"
    assert publication["output"]["line_count"] == 5
    assert list(checkpoint_root.glob("repo-*/chunk-*.jsonl")) == []

    second = subprocess.run(command, capture_output=True, text=True, check=False)
    assert second.returncode == 0, (
        f"stdout=\n{second.stdout}\nstderr=\n{second.stderr}"
    )
    assert "no commit ranges reprocessed" in second.stdout
    assert output.read_bytes() == original
