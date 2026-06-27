from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


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
