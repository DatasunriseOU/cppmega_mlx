from __future__ import annotations

import hashlib
import json
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


def _write_pr_completion_fixture(
    tmp_path: Path,
) -> tuple[Path, Path, Path]:
    pr_store = tmp_path / "prs.sqlite"
    repo_list = tmp_path / "repo_list.json"
    receipt_path = tmp_path / "pr_completion.json"
    pr_store.write_bytes(b"immutable sqlite fixture")
    repo_list.write_text('{"repos":["owner/repo"]}\n', encoding="utf-8")

    def sha256(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    receipt_path.write_text(
        json.dumps(
            {
                "schema": "cppmega_pr_completion_v2",
                "status": "verified",
                "pr_store": {
                    "path": str(pr_store.resolve()),
                    "sha256": sha256(pr_store),
                },
                "repo_list": {
                    "path": str(repo_list.resolve()),
                    "sha256": sha256(repo_list),
                },
                "expected_repos_sha256": "a" * 64,
                "scan_id": "1" * 64,
                "expected_repo_count": 1,
                "declared_pr_count": 7,
                "stored_pr_count": 7,
                "unverified_store_pr_count": 0,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return receipt_path, pr_store, repo_list


def test_pr_completion_binding_hashes_explicit_store_and_repo_list(
    tmp_path: Path,
) -> None:
    receipt_path, pr_store, repo_list = _write_pr_completion_fixture(tmp_path)
    wal = Path(f"{pr_store}-wal")
    wal.write_bytes(b"uncheckpointed")
    with pytest.raises(
        sc.PRCompletionBindingError,
        match="uncheckpointed WAL",
    ):
        sc.load_pr_completion_binding(
            receipt_path,
            pr_store=pr_store,
            repo_list=repo_list,
        )
    wal.unlink()

    binding = sc.load_pr_completion_binding(
        receipt_path,
        pr_store=pr_store,
        repo_list=repo_list,
    )

    assert binding == {
        "schema": "cppmega_pr_completion_v2",
        "status": "verified",
        "receipt_sha256": hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
        "pr_store_sha256": hashlib.sha256(pr_store.read_bytes()).hexdigest(),
        "repo_list_sha256": hashlib.sha256(repo_list.read_bytes()).hexdigest(),
        "expected_repos_sha256": "a" * 64,
        "scan_id": "1" * 64,
        "expected_repo_count": 1,
        "stored_pr_count": 7,
        "unverified_store_pr_count": 0,
    }

    pr_store.write_bytes(b"changed after verification")
    with pytest.raises(
        sc.PRCompletionBindingError,
        match="pr_store hash mismatch",
    ):
        sc.load_pr_completion_binding(
            receipt_path,
            pr_store=pr_store,
            repo_list=repo_list,
        )


def test_pr_completion_binding_rejects_legacy_receipt_without_scan_membership(
    tmp_path: Path,
) -> None:
    receipt_path, pr_store, repo_list = _write_pr_completion_fixture(tmp_path)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["schema"] = "cppmega_pr_completion_v1"
    receipt.pop("scan_id")
    receipt.pop("unverified_store_pr_count")
    receipt_path.write_text(
        json.dumps(receipt, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        sc.PRCompletionBindingError,
        match="unsupported PR completion schema",
    ):
        sc.load_pr_completion_binding(
            receipt_path,
            pr_store=pr_store,
            repo_list=repo_list,
        )


def test_pr_completion_finish_revalidation_detects_input_drift(
    tmp_path: Path,
) -> None:
    receipt_path, pr_store, repo_list = _write_pr_completion_fixture(tmp_path)
    binding = sc.load_pr_completion_binding(
        receipt_path,
        pr_store=pr_store,
        repo_list=repo_list,
    )
    sc.revalidate_pr_completion_binding(
        binding,
        receipt_path,
        pr_store=pr_store,
        repo_list=repo_list,
    )

    repo_list.write_text('{"repos":["other/repo"]}\n', encoding="utf-8")
    with pytest.raises(
        sc.PRCompletionBindingError,
        match="repo_list hash mismatch",
    ):
        sc.revalidate_pr_completion_binding(
            binding,
            receipt_path,
            pr_store=pr_store,
            repo_list=repo_list,
        )


def test_manifest_pr_completion_binding_is_preserved_and_resume_bound(
    tmp_path: Path,
) -> None:
    receipt_path, pr_store, repo_list = _write_pr_completion_fixture(tmp_path)
    binding = sc.load_pr_completion_binding(
        receipt_path,
        pr_store=pr_store,
        repo_list=repo_list,
    )
    manifest_path = tmp_path / "conveyor" / "_done.json"
    manifest = sc.ConcurrentManifest.load(manifest_path)
    manifest.bind_pr_completion(binding)
    manifest.mark_done("owner_repo::code", {"rows": 1})

    reloaded = sc.ConcurrentManifest.load(manifest_path)
    assert reloaded.pr_completion == binding

    changed = dict(binding)
    changed["receipt_sha256"] = "b" * 64
    with pytest.raises(
        sc.PRCompletionBindingError,
        match="PR completion mismatch",
    ):
        reloaded.bind_pr_completion(changed)


@pytest.mark.parametrize(
    "legacy_key",
    (
        "owner_repo::r0",
        "owner_repo::commits",
        "owner_repo::commit_plan",
        "owner_repo::no_git",
        "owner_repo::repo",
    ),
)
def test_manifest_rejects_legacy_commit_receipts_without_pr_binding(
    tmp_path: Path,
    legacy_key: str,
) -> None:
    receipt_path, pr_store, repo_list = _write_pr_completion_fixture(tmp_path)
    binding = sc.load_pr_completion_binding(
        receipt_path,
        pr_store=pr_store,
        repo_list=repo_list,
    )
    manifest = sc.ConcurrentManifest.load(tmp_path / "_done.json")
    manifest.mark_done(legacy_key, {"rows": 1})

    with pytest.raises(
        sc.PRCompletionBindingError,
        match="commit work receipts",
    ):
        manifest.bind_pr_completion(binding)


def test_commit_stream_requires_explicit_verified_pr_inputs() -> None:
    with pytest.raises(
        SystemExit,
        match=(
            r"commits/both/all requires explicit immutable PR inputs: "
            r"--pr-store, --repo-list, --pr-completion-receipt"
        ),
    ):
        sc.main(["--streams", "commits"])


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
