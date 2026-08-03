from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import urllib.parse
import urllib.request

import pytest

from scripts.pr_ingest import gitlab_mr_stream as gitlab
from scripts.pr_ingest import pr_store


def _exact_repo_list(tmp_path: Path) -> Path:
    identities = [
        "gitlab.com/libeigen%2FEigen",
        "gitlab.freedesktop.org/mesa%2Fmesa",
        "gitlab.torproject.org/tpo%2Fcore%2Ftor",
    ]
    path = tmp_path / "repo_list.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "unresolved": [],
                "repos": [
                    {
                        "bare_name": f"gitlab-{index}",
                        "project_identity": identity,
                    }
                    for index, identity in enumerate(identities)
                ],
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _empty_completion(
    tmp_path: Path,
) -> tuple[Path, Path, Path, Path, gitlab.Manifest]:
    repo_list = _exact_repo_list(tmp_path)
    projects = gitlab.load_gitlab_repos(repo_list)
    primary_store = tmp_path / "primary.sqlite"
    ancillary_store = tmp_path / "ancillary.sqlite"
    manifest_path = tmp_path / "manifest.json"
    receipt_path = tmp_path / "completion.json"
    sidecar_root = tmp_path / "sidecars"
    config = {
        "primary_store": str(primary_store.resolve()),
        "ancillary_store": str(ancillary_store.resolve()),
        "sidecar_root": str(sidecar_root.resolve()),
        "completion_receipt": str(receipt_path.resolve()),
        "window_days": 90,
        "page_size": 100,
        "max_window_pages": 100,
        "max_response_mib": 64,
        "max_detail_pages": 100,
        "max_detail_mib": 256,
    }
    manifest = gitlab.Manifest(
        manifest_path,
        repo_list=repo_list.resolve(),
        repo_list_sha256=gitlab._stable_file_sha256(
            repo_list,
            role="test repo list",
        ),
        projects=projects,
        config=config,
    )
    for project in projects:
        state = manifest.project(project.identity)
        state["inventory"]["empty_at_cutoff"] = True
        state["status"] = "done"
    manifest.save()

    for store in (primary_store, ancillary_store):
        conn = pr_store.connect(str(store), create=True)
        try:
            gitlab._checkpoint_store(conn, store)
        finally:
            conn.close()

    receipt = gitlab._build_completion_receipt(
        manifest=manifest,
        projects=projects,
        repo_list=repo_list.resolve(),
        primary_store=primary_store,
        ancillary_store=ancillary_store,
        sidecar_root=sidecar_root,
    )
    gitlab._atomic_write_json(receipt_path, receipt)
    return primary_store, receipt_path, repo_list, sidecar_root, manifest


def _capture_client_request(
    client: gitlab.GitLabClient,
    url: str,
) -> tuple[gitlab.APIResponse, urllib.request.Request]:
    class Response:
        status = 200
        headers: dict[str, str] = {}

        def __enter__(self) -> "Response":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def read(self, _limit: int) -> bytes:
            return b"[]"

    captured: list[object] = []

    class Opener:
        def open(self, request: object, *, timeout: float) -> Response:
            captured.append(request)
            return Response()

    client.opener = Opener()
    response = client.get(url)
    assert len(captured) == 1
    request = captured[0]
    assert isinstance(request, urllib.request.Request)
    return response, request


def test_canonical_gitlab_scope_is_exact_and_duplicate_closed(tmp_path: Path) -> None:
    repo_list = _exact_repo_list(tmp_path)
    projects = gitlab.load_gitlab_repos(repo_list)
    assert [project.identity for project in projects] == sorted(
        [
            "gitlab.com/libeigen%2FEigen",
            "gitlab.freedesktop.org/mesa%2Fmesa",
            "gitlab.torproject.org/tpo%2Fcore%2Ftor",
        ]
    )

    document = json.loads(repo_list.read_bytes())
    gitlab_rows = [
        row
        for row in document["repos"]
        if str(row.get("project_identity", "")).startswith("gitlab")
    ]
    duplicate_list = tmp_path / "duplicate_repo_list.json"
    duplicate_list.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "unresolved": [],
                "repos": [*gitlab_rows, dict(gitlab_rows[0])],
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(gitlab.GitLabIngestError, match="duplicates"):
        gitlab.load_gitlab_repos(duplicate_list)


def test_diff_paths_keep_python_and_javascript_out_of_primary() -> None:
    routed = gitlab.classify_diff_paths(
        [
            {"old_path": "src/solver.cpp", "new_path": "src/solver.cpp"},
            {"old_path": "schema.sql", "new_path": "schema.sql"},
            {"old_path": "tests/run.sh", "new_path": "tests/run.sh"},
            {"old_path": "tools/check.py", "new_path": "tools/check.py"},
            {"old_path": "web/build.js", "new_path": "web/build.js"},
            {"old_path": "README.md", "new_path": "README.md"},
        ]
    )

    assert [item["new_path"] for item in routed["primary"]] == [
        "src/solver.cpp",
        "schema.sql",
        "tests/run.sh",
    ]
    assert [item["new_path"] for item in routed["ancillary"]] == [
        "tools/check.py",
        "web/build.js",
    ]
    assert [item["new_path"] for item in routed["excluded"]] == ["README.md"]


def test_pagination_prefers_explicit_next_and_stops_at_exact_total() -> None:
    linked = gitlab.APIResponse(
        url="https://gitlab.com/api/v4/projects/a%2Fb/merge_requests?page=1",
        status=200,
        headers={
            "link": (
                "<https://gitlab.com/api/v4/projects/a%2Fb/merge_requests?page=2>; "
                'rel="next"'
            ),
            "x-total": "100",
        },
        body=[],
        body_sha256="0" * 64,
        byte_size=2,
    )
    assert gitlab._next_page_url(
        linked,
        page=1,
        page_size=100,
        item_count=100,
    ).endswith("page=2")

    final = gitlab.APIResponse(
        url="https://gitlab.com/api/v4/projects/a%2Fb/merge_requests?page=1",
        status=200,
        headers={"x-total": "100"},
        body=[],
        body_sha256="0" * 64,
        byte_size=2,
    )
    assert (
        gitlab._next_page_url(
            final,
            page=1,
            page_size=100,
            item_count=100,
        )
        is None
    )


def test_host_auth_coverage_requires_exactly_one_mode(tmp_path: Path) -> None:
    projects = gitlab.load_gitlab_repos(_exact_repo_list(tmp_path))
    token_env = gitlab._parse_token_env(["gitlab.com=GITLAB_TOKEN"])
    public_hosts = gitlab._parse_public_hosts(
        ["gitlab.freedesktop.org", "gitlab.torproject.org"]
    )

    resolved_tokens, resolved_public = gitlab._resolve_host_auth(
        projects,
        token_env_by_host=token_env,
        public_hosts=public_hosts,
    )
    assert resolved_tokens == token_env
    assert resolved_public == frozenset(public_hosts)

    with pytest.raises(gitlab.GitLabIngestError, match="missing=.*gitlab.torproject.org"):
        gitlab._resolve_host_auth(
            projects,
            token_env_by_host=token_env,
            public_hosts=["gitlab.freedesktop.org"],
        )

    with pytest.raises(gitlab.GitLabIngestError, match="overlap=.*gitlab.com"):
        gitlab._resolve_host_auth(
            projects,
            token_env_by_host=token_env,
            public_hosts=[
                "gitlab.com",
                "gitlab.freedesktop.org",
                "gitlab.torproject.org",
            ],
        )

    with pytest.raises(gitlab.GitLabIngestError, match="extra=.*example.invalid"):
        gitlab._resolve_host_auth(
            projects,
            token_env_by_host={},
            public_hosts=[
                "gitlab.com",
                "gitlab.freedesktop.org",
                "gitlab.torproject.org",
                "example.invalid",
            ],
        )

    with pytest.raises(gitlab.GitLabIngestError, match="unique"):
        gitlab._parse_public_hosts(["gitlab.com", "gitlab.com"])


def test_public_client_omits_private_token_header(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PUBLIC_HOST_TOKEN", "must-not-be-used")
    client = gitlab.GitLabClient(
        allowed_hosts={"gitlab.com"},
        token_env_by_host={},
        public_hosts={"gitlab.com"},
        max_response_bytes=1024,
        max_retries=0,
        timeout_s=1,
    )
    response, request = _capture_client_request(
        client,
        "https://gitlab.com/api/v4/projects/libeigen%2FEigen/merge_requests?page=1",
    )

    assert response.status == 200
    headers = {name.lower(): value for name, value in request.header_items()}
    assert "private-token" not in headers
    assert "authorization" not in headers
    assert headers["accept"] == "application/json"


def test_token_client_still_sends_private_token_header(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GITLAB_TOKEN", "token-value")
    client = gitlab.GitLabClient(
        allowed_hosts={"gitlab.com"},
        token_env_by_host={"gitlab.com": "GITLAB_TOKEN"},
        max_response_bytes=1024,
        max_retries=0,
        timeout_s=1,
    )
    _response, request = _capture_client_request(
        client,
        "https://gitlab.com/api/v4/projects/libeigen%2FEigen/merge_requests",
    )
    headers = {name.lower(): value for name, value in request.header_items()}
    assert headers["private-token"] == "token-value"


def test_token_client_missing_environment_does_not_fall_back_to_public(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MISSING_GITLAB_TOKEN", raising=False)
    with pytest.raises(gitlab.GitLabIngestError, match="environment variable is empty"):
        gitlab.GitLabClient(
            allowed_hosts={"gitlab.com"},
            token_env_by_host={"gitlab.com": "MISSING_GITLAB_TOKEN"},
            max_response_bytes=1024,
            max_retries=0,
            timeout_s=1,
        )


def test_sidecar_resume_requires_identical_frozen_content(tmp_path: Path) -> None:
    path = tmp_path / "sidecar.json"
    value = {
        "scan_id": "1" * 64,
        "project_identity": "gitlab.com/libeigen%2FEigen",
        "iid": 7,
        "metadata": {"title": "frozen"},
    }
    gitlab._write_bound_sidecar(path, value, scan_id="1" * 64)
    gitlab._write_bound_sidecar(path, dict(value), scan_id="1" * 64)
    changed = {**value, "metadata": {"title": "changed"}}
    with pytest.raises(gitlab.GitLabIngestError, match="content drifted"):
        gitlab._write_bound_sidecar(path, changed, scan_id="1" * 64)


def test_primary_sidecar_replay_materializes_exact_shas_without_fake_reviews(
    tmp_path: Path,
) -> None:
    repo_list = _exact_repo_list(tmp_path)
    projects = gitlab.load_gitlab_repos(repo_list)
    project = projects[0]
    primary_store = tmp_path / "primary.sqlite"
    ancillary_store = tmp_path / "ancillary.sqlite"
    manifest = gitlab.Manifest(
        tmp_path / "manifest.json",
        repo_list=repo_list,
        repo_list_sha256=gitlab._stable_file_sha256(repo_list, role="test repo list"),
        projects=projects,
        config={
            "max_detail_mib": 4,
            "max_response_mib": 2,
        },
    )
    source_sha = "1" * 40
    target_sha = "2" * 40
    base_sha = "3" * 40
    merge_sha = "4" * 40
    created_at = "2026-08-03T08:00:00.000000Z"
    metadata = {
        "id": 7001,
        "iid": 7,
        "project_id": 101,
        "state": "merged",
        "title": "Native fix",
        "created_at": created_at,
        "merge_commit_sha": merge_sha,
    }
    detail = {
        **metadata,
        "target_project_id": 101,
        "description": "Keep the native path exact",
        "merged_at": "2026-08-03T09:00:00.000000Z",
        "sha": source_sha,
        "diff_refs": {
            "head_sha": source_sha,
            "start_sha": target_sha,
            "base_sha": base_sha,
        },
        "squash_commit_sha": None,
        "author": {"username": "alice"},
        "reviewers": [{"id": 9, "username": "assigned-not-reviewed"}],
    }
    diff = {"old_path": "src/fix.cpp", "new_path": "src/fix.cpp"}
    record = {
        "schema": gitlab.GITLAB_RECORD_SCHEMA,
        "scan_id": manifest.scan_id,
        "platform": gitlab.GITLAB_PLATFORM,
        "project_identity": project.identity,
        "host": project.host,
        "project_path": "libeigen/Eigen",
        "iid": 7,
        "route": "primary",
        "fetched_at": "2026-08-03T10:00:00.000000Z",
        "metadata": metadata,
        "merge_request": detail,
        "shas": {
            "source": source_sha,
            "target": target_sha,
            "base": base_sha,
            "merge": merge_sha,
            "squash": None,
        },
        "diffs": {"primary": [diff], "ancillary": [], "excluded": []},
        "discussions": [
            {
                "id": "discussion-1",
                "notes": [
                    {
                        "id": 8001,
                        "system": False,
                        "type": "DiffNote",
                        "author": {"username": "bob"},
                        "body": "Please keep this native.",
                        "created_at": "2026-08-03T08:30:00.000000Z",
                        "position": {"new_path": "src/fix.cpp"},
                    }
                ],
            }
        ],
        "reviewer_assignments": detail["reviewers"],
        "linked_issues": [
            {
                "project_id": 101,
                "iid": 12,
                "title": "Native issue",
                "description": "Issue body",
            }
        ],
        "lineage": [],
    }
    record_path = gitlab._sidecar_path(
        tmp_path / "sidecars",
        "records/primary",
        project,
        7,
    )
    gitlab._atomic_write_json(record_path, record)
    primary_conn = pr_store.connect(str(primary_store), create=True)
    ancillary_conn = pr_store.connect(str(ancillary_store), create=True)
    try:
        for _ in range(2):
            assert (
                gitlab._materialize_record_sidecar(
                    record,
                    record_path,
                    manifest=manifest,
                    project=project,
                    primary_conn=primary_conn,
                    ancillary_conn=ancillary_conn,
                )
                == "primary"
            )
        stored = pr_store.get_by_pr(
            primary_conn,
            project.identity,
            7,
            scan_id=manifest.scan_id,
        )
        assert stored is not None
        assert stored["merge_commit_sha"] == merge_sha
        assert stored["raw"]["source_sha"] == source_sha
        assert stored["raw"]["target_sha"] == target_sha
        assert stored["raw"]["base_sha"] == base_sha
        assert stored["comments"][0]["kind"] == "review_comment"
        assert stored["reviews"] == []
        assert ancillary_conn.execute("SELECT COUNT(*) FROM prs").fetchone()[0] == 0
    finally:
        primary_conn.close()
        ancillary_conn.close()


def test_gitlab_completion_dispatch_is_verified_but_not_training_ready(
    tmp_path: Path,
) -> None:
    from scripts.pr_ingest import export_pr_parquet

    primary_store, receipt_path, repo_list, _sidecar_root, manifest = _empty_completion(
        tmp_path
    )
    binding = gitlab.load_gitlab_completion_binding(
        receipt_path,
        pr_store=primary_store,
        repo_list=repo_list,
    )
    assert binding["schema"] == gitlab.GITLAB_COMPLETION_SCHEMA
    assert binding["status"] == "verified"
    assert binding["platform"] == "gitlab"
    assert binding["scan_id"] == manifest.scan_id
    assert binding["training_ready_without_membership"] is False
    assert (
        gitlab.verify_gitlab_completion_receipt(
            receipt_path,
            pr_store=primary_store,
            repo_list=repo_list,
        )
        == binding
    )

    dispatched = export_pr_parquet._load_pr_completion(
        argparse.Namespace(
            store=str(primary_store),
            repo_list=str(repo_list),
            pr_completion_receipt=str(receipt_path),
        )
    )
    assert dispatched == binding


def test_case5_still_requires_exact_primary_membership(tmp_path: Path) -> None:
    from scripts.pr_ingest import export_pr_parquet

    primary_store, receipt_path, repo_list, _sidecar_root, _manifest = (
        _empty_completion(tmp_path)
    )
    missing_root = tmp_path / "missing-primary-membership"
    args = argparse.Namespace(
        store=str(primary_store),
        pr_completion_receipt=str(receipt_path),
        repo_list=str(repo_list),
        primary_membership_receipt=str(
            missing_root / "primary_pr_membership_receipt.json"
        ),
        primary_membership_root=str(missing_root),
        output_root=str(tmp_path / "out"),
        target_lengths="1024",
        repo=None,
        offset=0,
        limit=1,
        all=False,
        batch_size=1,
        max_shards=None,
        manifest=None,
        no_resume=False,
        memory_limit_gb=4.0,
    )
    with pytest.raises(FileNotFoundError, match="missing-primary-membership"):
        export_pr_parquet.export_pr_parquet_batches(args)


def test_completion_rejects_a_training_ready_claim(tmp_path: Path) -> None:
    primary_store, receipt_path, repo_list, _sidecar_root, _manifest = (
        _empty_completion(tmp_path)
    )
    receipt = json.loads(receipt_path.read_bytes())
    receipt["training_ready_without_membership"] = True
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(gitlab.GitLabIngestError, match="non-training-ready"):
        gitlab.load_gitlab_completion_binding(
            receipt_path,
            pr_store=primary_store,
            repo_list=repo_list,
        )


def test_inventory_window_query_brackets_inclusive_microseconds() -> None:
    start = datetime(2026, 8, 3, 8, 0, 0, 123456, tzinfo=timezone.utc)
    end = start.replace(microsecond=223456)
    project = gitlab._parse_project_identity("gitlab.com/libeigen%2FEigen")
    query = urllib.parse.parse_qs(
        urllib.parse.urlsplit(
            gitlab._inventory_url(
                project,
                window_start=start,
                window_end=end,
                page=3,
                page_size=100,
            )
        ).query
    )
    assert query["created_after"] == ["2026-08-03T08:00:00.123455Z"]
    assert query["created_before"] == ["2026-08-03T08:00:00.223457Z"]
    assert query["page"] == ["3"]
