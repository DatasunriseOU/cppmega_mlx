from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from types import MappingProxyType

import pytest

from scripts import streaming_conveyor as conveyor
from scripts import streaming_reindex_commits as commits
from tools.clang_indexer import process_commits


ROOT = Path(__file__).resolve().parents[1]


def _write_v2_repo_list(path: Path, rows: list[dict[str, str]]) -> Path:
    by_bare_name = {
        row["bare_name"]: row["project_identity"]
        for row in rows
    }
    owner_repos = sorted(
        {
            row["owner_repo"]
            for row in rows
            if row.get("owner_repo") is not None
        }
    )
    path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "repos": rows,
                "by_bare_name": dict(sorted(by_bare_name.items())),
                "project_identities": sorted(set(by_bare_name.values())),
                "repo_names": owner_repos,
                "unresolved": [],
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _github_row(
    bare_name: str = "project",
    owner_repo: str = "owner/project",
) -> dict[str, str]:
    return {
        "bare_name": bare_name,
        "project_identity": owner_repo,
        "owner_repo": owner_repo,
    }


def _ownerless_row(
    bare_name: str,
    project_identity: str,
) -> dict[str, str]:
    return {
        "bare_name": bare_name,
        "project_identity": project_identity,
    }


def _verified_completion(
    snapshot: commits.RepoListSnapshot,
    *,
    scan_id: str = "a" * 64,
) -> dict[str, object]:
    return {
        "schema": "cppmega_pr_completion_v2",
        "status": "verified",
        "receipt_sha256": "1" * 64,
        "pr_store_sha256": "2" * 64,
        "repo_list_sha256": snapshot.sha256,
        "expected_repos_sha256": commits._canonical_json_sha256(
            list(snapshot.github_repos)
        ),
        "scan_id": scan_id,
        "expected_repo_count": len(snapshot.github_repos),
        "stored_pr_count": 0,
        "unverified_store_pr_count": 0,
    }


def test_strict_v2_pair_accepts_ownerless_and_source_only_rows(
    tmp_path: Path,
) -> None:
    shared_local = _ownerless_row("shared-local", "corpus.local/shared-local")
    source_only = _ownerless_row("source-only", "corpus.local/source-only")
    source_path = _write_v2_repo_list(
        tmp_path / "source.json",
        [_github_row(), shared_local, source_only],
    )
    pr_path = _write_v2_repo_list(
        tmp_path / "pr.json",
        [_github_row(), shared_local],
    )

    source, pr = commits.load_repo_list_contracts(source_path, pr_path)

    assert source.mapping_count == 3
    assert pr.mapping_count == 2
    assert source.owner_repo_by_bare_name["source-only"] is None
    assert pr.owner_repo_by_bare_name["shared-local"] is None
    assert pr.github_repos == ("owner/project",)


def test_strict_v2_pair_rejects_legacy_or_missing_github_scope(
    tmp_path: Path,
) -> None:
    legacy = tmp_path / "legacy.json"
    legacy.write_text('{"repos":[]}\n', encoding="utf-8")
    with pytest.raises(
        commits.RepoListBindingError,
        match="unsupported schema_version",
    ):
        commits.load_repo_list_snapshot(legacy, role="source")

    source_path = _write_v2_repo_list(
        tmp_path / "source.json",
        [_github_row(), _github_row("second", "owner/second")],
    )
    pr_path = _write_v2_repo_list(
        tmp_path / "pr.json",
        [_github_row()],
    )
    with pytest.raises(commits.RepoListBindingError, match="missing="):
        commits.load_repo_list_contracts(source_path, pr_path)


def test_process_command_separates_source_and_pr_identities() -> None:
    scan_id = "a" * 64
    command = commits.build_process_commits_command(
        [Path("commits.jsonl")],
        Path("enriched.jsonl"),
        Path("repo-root"),
        None,
        pr_store=Path("prs.sqlite"),
        project_id="corpus.local/archive-name",
        pr_owner_repo="owner/project",
        pr_scan_id=scan_id,
    )

    assert command[command.index("--project-id") + 1] == (
        "corpus.local/archive-name"
    )
    assert command[command.index("--pr-repo") + 1] == "owner/project"
    assert command[command.index("--pr-scan-id") + 1] == scan_id
    assert "--repo-list" not in command
    assert "--source-quarantine-manifest" not in command


def test_source_only_command_omits_every_pr_argument() -> None:
    command = commits.build_process_commits_command(
        [Path("commits.jsonl")],
        Path("enriched.jsonl"),
        Path("repo-root"),
        None,
        pr_store=Path("prs.sqlite"),
        project_id="corpus.local/source-only",
        pr_owner_repo=None,
    )

    assert "--project-id" in command
    assert "--pr-store" not in command
    assert "--pr-repo" not in command
    assert "--pr-scan-id" not in command


def test_neutral_completion_validator_binds_ordered_pr_scope(
    tmp_path: Path,
) -> None:
    pr_list = _write_v2_repo_list(
        tmp_path / "pr.json",
        [
            _github_row("zeta", "owner/zeta"),
            _ownerless_row("local", "corpus.local/local"),
            _github_row("alpha", "owner/alpha"),
        ],
    )
    snapshot = commits.load_repo_list_snapshot(pr_list, role="PR")
    store = tmp_path / "prs.sqlite"
    store.write_bytes(b"immutable sqlite fixture")
    receipt = tmp_path / "completion.json"
    ordered_membership = ["owner/zeta", "owner/alpha"]
    receipt.write_text(
        json.dumps(
            {
                "schema": "cppmega_pr_completion_v2",
                "status": "verified",
                "pr_store": {
                    "path": str(store.resolve()),
                    "sha256": hashlib.sha256(store.read_bytes()).hexdigest(),
                },
                "repo_list": {
                    "path": str(pr_list.resolve()),
                    "sha256": hashlib.sha256(pr_list.read_bytes()).hexdigest(),
                },
                "expected_repos_sha256": commits._canonical_json_sha256(
                    ordered_membership
                ),
                "scan_id": "b" * 64,
                "expected_repo_count": 2,
                "declared_pr_count": 0,
                "stored_pr_count": 0,
                "unverified_store_pr_count": 0,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    binding = commits.load_pr_completion_binding(
        receipt,
        pr_store=store,
        repo_list=pr_list,
        repo_list_snapshot=snapshot,
    )
    assert binding["scan_id"] == "b" * 64
    assert binding["expected_repo_count"] == 2
    commits.revalidate_pr_completion_binding(
        binding,
        receipt,
        pr_store=store,
        repo_list=pr_list,
        repo_list_snapshot=snapshot,
    )


def test_conveyor_source_manifest_is_bound_and_legacy_work_fails_closed(
    tmp_path: Path,
) -> None:
    source_path = _write_v2_repo_list(
        tmp_path / "source.json",
        [_github_row()],
    )
    binding = conveyor.build_source_repo_list_binding(
        conveyor.load_repo_list_snapshot(source_path, role="source")
    )
    manifest = conveyor.ConcurrentManifest.load(tmp_path / "fresh.json")
    manifest.bind_source_repo_list(binding)
    assert conveyor.ConcurrentManifest.load(
        tmp_path / "fresh.json"
    ).source_repo_list == binding

    legacy = conveyor.ConcurrentManifest.load(tmp_path / "legacy.json")
    legacy.mark_done("project::code", {"source": "code", "lengths": {}})
    with pytest.raises(
        commits.RepoListBindingError,
        match="no source repo-list binding",
    ):
        legacy.bind_source_repo_list(binding)


def test_process_commits_scan_cannot_bypass_fixed_pr_repo() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools" / "clang_indexer" / "process_commits.py"),
            "--inputs",
            str(ROOT / "missing.jsonl"),
            "--output",
            str(ROOT / "unused.jsonl"),
            "--pr-scan-id",
            "c" * 64,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 1
    assert "--pr-scan-id requires --pr-store and --pr-repo" in completed.stderr


def test_conveyor_reuses_the_neutral_completion_validator() -> None:
    assert conveyor.PRCompletionBindingError is commits.PRCompletionBindingError
    assert conveyor.load_pr_completion_binding is commits.load_pr_completion_binding
    assert (
        conveyor.revalidate_pr_completion_binding
        is commits.revalidate_pr_completion_binding
    )


def test_strict_v2_validation_rejects_duplicate_and_inconsistent_derived_map(
    tmp_path: Path,
) -> None:
    duplicate = _write_v2_repo_list(
        tmp_path / "duplicate.json",
        [_github_row(), _github_row()],
    )
    with pytest.raises(commits.RepoListBindingError, match="duplicate bare_name"):
        commits.load_repo_list_snapshot(duplicate, role="source")

    inconsistent = _write_v2_repo_list(
        tmp_path / "inconsistent.json",
        [_github_row()],
    )
    document = json.loads(inconsistent.read_text(encoding="utf-8"))
    document["by_bare_name"]["project"] = "different/project"
    inconsistent.write_text(
        json.dumps(document, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(
        commits.RepoListBindingError,
        match="by_bare_name does not match",
    ):
        commits.load_repo_list_snapshot(inconsistent, role="source")


@pytest.mark.parametrize(
    ("source_rows", "pr_rows"),
    [
        (
            [_github_row(), _github_row("second", "owner/second")],
            [_github_row()],
        ),
        (
            [_github_row()],
            [_github_row(), _github_row("unexpected", "owner/unexpected")],
        ),
        (
            [_github_row()],
            [_github_row("project", "different/project")],
        ),
    ],
)
def test_strict_v2_pair_rejects_every_scope_inconsistency(
    tmp_path: Path,
    source_rows: list[dict[str, str]],
    pr_rows: list[dict[str, str]],
) -> None:
    source_path = _write_v2_repo_list(tmp_path / "source.json", source_rows)
    pr_path = _write_v2_repo_list(tmp_path / "pr.json", pr_rows)
    with pytest.raises(
        commits.RepoListBindingError,
        match="PR repo list does not match",
    ):
        commits.load_repo_list_contracts(source_path, pr_path)


def test_source_binding_finish_revalidation_detects_drift(
    tmp_path: Path,
) -> None:
    source_path = _write_v2_repo_list(
        tmp_path / "source.json",
        [_github_row(), _ownerless_row("local", "corpus.local/local")],
    )
    snapshot = commits.load_repo_list_snapshot(source_path, role="source")
    binding = conveyor.build_source_repo_list_binding(snapshot)
    conveyor.revalidate_source_repo_list_binding(binding, source_path)

    _write_v2_repo_list(
        source_path,
        [_github_row(), _ownerless_row("local", "corpus.local/changed")],
    )
    with pytest.raises(
        commits.RepoListBindingError,
        match="changed while the conveyor was running",
    ):
        conveyor.revalidate_source_repo_list_binding(binding, source_path)


def test_standalone_manifest_and_scan_are_receipt_bound(
    tmp_path: Path,
) -> None:
    snapshot = commits.RepoListSnapshot(
        path=tmp_path / "repos.json",
        sha256="3" * 64,
        canonical_mapping_sha256="4" * 64,
        mapping_count=1,
        project_id_by_bare_name=MappingProxyType(
            {"project": "owner/project"}
        ),
        owner_repo_by_bare_name=MappingProxyType(
            {"project": "owner/project"}
        ),
        github_repos=("owner/project",),
    )
    completion = _verified_completion(snapshot)
    manifest_path = tmp_path / "done.json"
    manifest = commits.Manifest.load(manifest_path)
    commits.bind_verified_manifest_inputs(
        manifest,
        source=snapshot,
        pr=snapshot,
        pr_completion=completion,
    )
    commits.bind_verified_manifest_inputs(
        commits.Manifest.load(manifest_path),
        source=snapshot,
        pr=snapshot,
        pr_completion=completion,
    )
    assert commits.resolve_verified_pr_scan_id(completion, None) == "a" * 64

    changed = dict(completion)
    changed["receipt_sha256"] = "5" * 64
    with pytest.raises(
        commits.RepoListBindingError,
        match="verified input mismatch",
    ):
        commits.bind_verified_manifest_inputs(
            commits.Manifest.load(manifest_path),
            source=snapshot,
            pr=snapshot,
            pr_completion=changed,
        )
    with pytest.raises(
        commits.PRCompletionBindingError,
        match="requires --pr-completion-receipt",
    ):
        commits.resolve_verified_pr_scan_id(None, "a" * 64)


def test_pr_lookup_fixed_key_overrides_record_and_corpus_local_is_ineligible(
    tmp_path: Path,
) -> None:
    scan_id = "d" * 64
    store = tmp_path / "prs.sqlite"
    connection = process_commits._pr_store_mod.connect(str(store), create=True)
    try:
        process_commits._pr_store_mod.upsert_record(
            connection,
            {
                "repo": "owner/project",
                "pr_number": 7,
                "merge_commit_sha": "fixed-key-sha",
                "pr_title": "Fixed key",
                "pr_body": "Bound discussion.",
                "comments": [],
                "reviews": [],
                "linked_issues": [],
            },
            scan_id=scan_id,
        )
    finally:
        connection.close()

    lookup = process_commits.PRDiscussionLookup(
        str(store),
        None,
        scan_id=scan_id,
        owner_repo="owner/project",
    )
    try:
        record = {
            "repo": "different/source-identity",
            "commit_hash": "fixed-key-sha",
        }
        assert lookup.attach(record) is True
        assert record["pr_number"] == 7
    finally:
        lookup.close()

    ownerless_lookup = process_commits.PRDiscussionLookup(str(store), None)
    try:
        ownerless = {
            "repo": "corpus.local/local-source",
            "commit_hash": "fixed-key-sha",
        }
        assert ownerless_lookup._store_key(ownerless) is None
        assert ownerless_lookup.attach(ownerless) is False
    finally:
        ownerless_lookup.close()


def test_conveyor_never_aliases_a_missing_pr_repo_list(
    tmp_path: Path,
) -> None:
    source_path = _write_v2_repo_list(
        tmp_path / "source.json",
        [_github_row()],
    )
    with pytest.raises(SystemExit, match=r"--pr-repo-list"):
        conveyor.main(
            [
                "--streams",
                "commits",
                "--repo-list",
                str(source_path),
                "--pr-store",
                str(tmp_path / "prs.sqlite"),
                "--pr-completion-receipt",
                str(tmp_path / "completion.json"),
            ]
        )


def test_standalone_missing_receipt_does_not_initialize_dedup(
    tmp_path: Path,
) -> None:
    source_path = _write_v2_repo_list(
        tmp_path / "source.json",
        [_github_row()],
    )
    pr_path = _write_v2_repo_list(
        tmp_path / "pr.json",
        [_github_row()],
    )
    store = tmp_path / "prs.sqlite"
    store.write_bytes(b"immutable fixture")
    dedup = tmp_path / "dedup" / "global.sqlite"
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "streaming_reindex_commits.py"),
            "--repo-list",
            str(source_path),
            "--pr-repo-list",
            str(pr_path),
            "--pr-store",
            str(store),
            "--pr-completion-receipt",
            str(tmp_path / "missing.json"),
            "--dedup-db",
            str(dedup),
            "--max-repos",
            "0",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode != 0
    assert "PR completion receipt is missing" in completed.stderr
    assert not dedup.exists()
    assert not dedup.parent.exists()
