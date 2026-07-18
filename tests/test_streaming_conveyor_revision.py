from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import streaming_conveyor as sc


def _git(repo: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        check=True,
    )
    return proc.stdout.strip()


@pytest.fixture
def source_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    (repo / "scripts" / "worker.py").write_text("VALUE = 1\n", encoding="utf-8")
    (repo / "tools" / "clang_indexer").mkdir(parents=True)
    (repo / "tools" / "clang_indexer" / "index_project.py").write_text(
        "INDEXER = 1\n", encoding="utf-8"
    )
    (repo / ".gitignore").write_text("outputs/\n", encoding="utf-8")
    _git(repo, "init", "-q")
    _git(repo, "config", "user.name", "Revision Test")
    _git(repo, "config", "user.email", "revision@example.test")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "initial")
    return repo


def _guard(repo: Path) -> sc.CodeRevisionGuard:
    head = _git(repo, "rev-parse", "HEAD")
    return sc.CodeRevisionGuard.for_production(head, repo_root=repo)


def test_code_revision_unchanged_passes_and_ignores_outputs(source_repo: Path) -> None:
    guard = _guard(source_repo)
    output = source_repo / "outputs" / "corpus" / "large.parquet"
    output.parent.mkdir(parents=True)
    output.write_bytes(b"not-source")

    guard.verify("unchanged test stage")


def test_code_revision_detects_head_drift(source_repo: Path) -> None:
    guard = _guard(source_repo)
    (source_repo / "scripts" / "worker.py").write_text("VALUE = 2\n", encoding="utf-8")
    _git(source_repo, "add", "scripts/worker.py")
    _git(source_repo, "commit", "-q", "-m", "next")

    with pytest.raises(sc.CodeRevisionDriftError, match="HEAD"):
        guard.verify("head drift test stage")


@pytest.mark.parametrize("staged", [False, True])
def test_code_revision_detects_tracked_edit(source_repo: Path, staged: bool) -> None:
    guard = _guard(source_repo)
    (source_repo / "scripts" / "worker.py").write_text("VALUE = 3\n", encoding="utf-8")
    if staged:
        _git(source_repo, "add", "scripts/worker.py")

    component = "index" if staged else "worktree"
    with pytest.raises(sc.CodeRevisionDriftError, match=component):
        guard.verify("tracked edit test stage")


def test_code_revision_detects_untracked_relevant_python(source_repo: Path) -> None:
    guard = _guard(source_repo)
    (source_repo / "scripts" / "new_parser.py").write_text(
        "PARSER_REVISION = 2\n",
        encoding="utf-8",
    )

    with pytest.raises(sc.CodeRevisionDriftError, match="untracked source"):
        guard.verify("untracked parser test stage")


def test_manifest_rejects_code_revision_mismatch(source_repo: Path, tmp_path: Path) -> None:
    first = sc.capture_code_revision(source_repo)
    manifest = sc.ConcurrentManifest.load(tmp_path / "conveyor" / "_done.json")
    manifest.bind_code_revision(first)

    (source_repo / "scripts" / "worker.py").write_text("VALUE = 4\n", encoding="utf-8")
    _git(source_repo, "add", "scripts/worker.py")
    _git(source_repo, "commit", "-q", "-m", "different revision")
    second = sc.capture_code_revision(source_repo)

    with pytest.raises(sc.CodeRevisionMismatchError, match="manifest code revision mismatch"):
        manifest.bind_code_revision(second)


def test_manifest_rejects_unpinned_legacy_resume(
    source_repo: Path,
    tmp_path: Path,
) -> None:
    manifest = sc.ConcurrentManifest.load(tmp_path / "legacy" / "_done.json")
    manifest.mark_done("repo::code", {"rows": 1})

    with pytest.raises(sc.CodeRevisionMismatchError, match="new conveyor/output root"):
        manifest.bind_code_revision(sc.capture_code_revision(source_repo))


def test_code_revision_receipt_contains_clang_indexer_dependency_binding(
    source_repo: Path,
) -> None:
    receipt = sc.capture_code_revision(source_repo)

    provenance = receipt["indexer_provenance"]
    assert provenance["schema"] == "cppmega_indexer_dependency_binding_v1"
    assert provenance["path"] == "tools/clang_indexer/index_project.py"
    assert len(provenance["source_sha256"]) == 64
    assert len(provenance["dependency_closure_sha256"]) == 64
    assert provenance["dependency_manifest"] == {
        "tools/clang_indexer/index_project.py": provenance["source_sha256"]
    }
    assert receipt["indexer_dependency_closure_sha256"] == (
        provenance["dependency_closure_sha256"]
    )


def test_production_revision_rejects_missing_clang_indexer(
    source_repo: Path, tmp_path: Path
) -> None:
    indexer = source_repo / "tools" / "clang_indexer" / "index_project.py"
    indexer.unlink()
    _git(source_repo, "add", "-u")
    _git(source_repo, "commit", "-q", "-m", "remove indexer")

    with pytest.raises(sc.CodeRevisionMismatchError, match="clang indexer"):
        sc.CodeRevisionGuard.for_production(
            _git(source_repo, "rev-parse", "HEAD"),
            repo_root=source_repo,
        )


def test_child_python_preflight_rejects_late_edit(
    source_repo: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    guard = _guard(source_repo)
    monkeypatch.setenv("PYTHONPATH", "")
    sc.install_code_revision_child_guard(guard, tmp_path / "conveyor")
    env = dict(os.environ)

    unchanged = subprocess.run(
        [sys.executable, "-c", "print('ok')"],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert unchanged.returncode == 0, unchanged.stderr

    (source_repo / "scripts" / "worker.py").write_text("VALUE = 5\n", encoding="utf-8")
    drifted = subprocess.run(
        [sys.executable, "-c", "raise AssertionError('must not import target')"],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert drifted.returncode == sc.CODE_REVISION_DRIFT_EXIT_CODE
    assert sc.CODE_REVISION_DRIFT_MARKER in drifted.stderr
