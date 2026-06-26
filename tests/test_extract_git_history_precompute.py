from __future__ import annotations

import subprocess
import sys
from pathlib import Path


MLX_ROOT = Path(__file__).resolve().parents[1]
NANOCHAT = MLX_ROOT / "scripts" / "nanochat_data"
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
