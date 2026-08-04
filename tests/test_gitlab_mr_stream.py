from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import urllib.parse
from datetime import UTC, datetime
from pathlib import Path
import urllib.request

import pytest

from scripts.pr_ingest import gitlab_mr_stream as gitlab
from scripts.pr_ingest import pr_store

_EXPECTED_IDENTITIES = (
    "gitlab.com/libeigen%2FEigen",
    "gitlab.freedesktop.org/mesa%2Fmesa",
    "gitlab.torproject.org/tpo%2Fcore%2Ftor",
    "invent.kde.org/frameworks%2Fkconfig",
    "invent.kde.org/utilities%2Fkate",
)
_EXPECTED_HOSTS = tuple(
    sorted({identity.split("/", 1)[0] for identity in _EXPECTED_IDENTITIES})
)


def _exact_repo_list(tmp_path: Path) -> Path:
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
                        "remote_url": (
                            f"https://{identity.split('/', 1)[0]}/"
                            f"{urllib.parse.unquote(identity.split('/', 1)[1])}.git"
                        ),
                    }
                    for index, identity in enumerate(_EXPECTED_IDENTITIES)
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
    projects = gitlab.load_gitlab_repos(
        repo_list,
        expected_hosts=_EXPECTED_HOSTS,
    )
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
        expected_hosts=_EXPECTED_HOSTS,
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


def test_gitlab_scope_selects_all_exact_token_hosts_and_rejects_duplicates(
    tmp_path: Path,
) -> None:
    repo_list = _exact_repo_list(tmp_path)
    document = json.loads(repo_list.read_bytes())
    document["repos"].extend(
        [
            {
                "bare_name": "not-an-exact-host",
                "project_identity": "invent.kde.org.evil/group%2Frepo",
            },
            {
                "bare_name": "github",
                "project_identity": "owner/repo",
                "owner_repo": "owner/repo",
            },
        ]
    )
    repo_list.write_text(
        json.dumps(document, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    projects = gitlab.load_gitlab_repos(
        repo_list,
        expected_hosts=_EXPECTED_HOSTS,
    )
    assert [project.identity for project in projects] == sorted(_EXPECTED_IDENTITIES)
    assert [
        project.identity for project in projects if project.host == "invent.kde.org"
    ] == [
        "invent.kde.org/frameworks%2Fkconfig",
        "invent.kde.org/utilities%2Fkate",
    ]

    duplicate_list = tmp_path / "duplicate_repo_list.json"
    duplicate_list.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "unresolved": [],
                "repos": [*document["repos"], dict(document["repos"][0])],
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(gitlab.GitLabIngestError, match="duplicates"):
        gitlab.load_gitlab_repos(
            duplicate_list,
            expected_hosts=_EXPECTED_HOSTS,
        )


def test_gitlab_scope_rejects_remote_host_identity_mismatch(tmp_path: Path) -> None:
    repo_list = _exact_repo_list(tmp_path)
    document = json.loads(repo_list.read_bytes())
    document["repos"][-1]["remote_url"] = "https://gitlab.com/utilities/kate.git"
    repo_list.write_text(
        json.dumps(document, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(gitlab.GitLabIngestError, match="host/identity mismatch"):
        gitlab.load_gitlab_repos(
            repo_list,
            expected_hosts=_EXPECTED_HOSTS,
        )


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

    normalized = gitlab.APIResponse(
        url=(
            "https://gitlab.com/api/v4/projects/a%2Fb/merge_requests?"
            "state=all&created_before=2026-08-03T23%3A43%3A56.646941Z&page=1"
        ),
        status=200,
        headers={
            "link": (
                "<https://gitlab.com/api/v4/projects/a%2Fb/merge_requests?"
                "created_before=2026-08-03T23%3A43%3A56%2B00%3A00&"
                "id=a%2Fb&order_by=created_at&page=2&per_page=100&sort=asc&"
                "state=all&with_labels_details=false>; rel=\"next\""
            )
        },
        body=[],
        body_sha256="0" * 64,
        byte_size=2,
    )
    next_url = gitlab._next_page_url(
        normalized,
        page=1,
        page_size=100,
        item_count=100,
    )
    assert next_url == (
        "https://gitlab.com/api/v4/projects/a%2Fb/merge_requests?"
        "state=all&created_before=2026-08-03T23%3A43%3A56.646941Z&page=2"
    )

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

    for unsafe_next in (
        "https://invent.kde.org/api/v4/projects/a%2Fb/merge_requests?page=2",
        "https://gitlab.com:8443/api/v4/projects/a%2Fb/merge_requests?page=2",
        "https://gitlab.com/api/v4/projects/evil%2Frepo/merge_requests?page=2",
        "https://gitlab.com/api/v4/projects/a%2Fb/merge_requests?page=3",
    ):
        unsafe = gitlab.APIResponse(
            **{
                **linked.__dict__,
                "headers": {"link": f'<{unsafe_next}>; rel="next"'},
            }
        )
        with pytest.raises(gitlab.GitLabIngestError, match="pagination link"):
            gitlab._next_page_url(
                unsafe,
                page=1,
                page_size=100,
                item_count=100,
            )

    client = gitlab.GitLabClient(
        allowed_hosts={"gitlab.com"},
        token_env_by_host={},
        public_hosts={"gitlab.com"},
        max_response_bytes=1024,
        max_retries=0,
        timeout_s=1,
    )
    with pytest.raises(gitlab.GitLabIngestError, match="out-of-contract"):
        client._validate_url(
            "https://gitlab.com:8443/api/v4/projects/a%2Fb/merge_requests"
        )

    skipped = gitlab.APIResponse(
        **{
            **linked.__dict__,
            "headers": {"x-next-page": "999"},
        }
    )
    with pytest.raises(gitlab.GitLabIngestError, match="non-consecutive"):
        gitlab._next_page_url(
            skipped,
            page=1,
            page_size=100,
            item_count=100,
        )


def test_host_auth_coverage_requires_exactly_one_mode(tmp_path: Path) -> None:
    projects = gitlab.load_gitlab_repos(
        _exact_repo_list(tmp_path),
        expected_hosts=_EXPECTED_HOSTS,
    )
    token_env = gitlab._parse_token_env(["gitlab.com=GITLAB_TOKEN"])
    public_hosts = tuple(
        host for host in _EXPECTED_HOSTS if host != "gitlab.com"
    )

    resolved_tokens, resolved_public = gitlab._resolve_host_auth(
        projects,
        token_env_by_host=token_env,
        public_hosts=public_hosts,
    )
    assert resolved_tokens == token_env
    assert resolved_public == frozenset(public_hosts)

    with pytest.raises(gitlab.GitLabIngestError, match="missing=.*invent.kde.org"):
        gitlab._resolve_host_auth(
            projects,
            token_env_by_host=token_env,
            public_hosts=public_hosts[:-1],
        )

    with pytest.raises(gitlab.GitLabIngestError, match="overlap=.*gitlab.com"):
        gitlab._resolve_host_auth(
            projects,
            token_env_by_host=token_env,
            public_hosts=["gitlab.com", *public_hosts],
        )

    with pytest.raises(gitlab.GitLabIngestError, match="extra=.*example.invalid"):
        gitlab._resolve_host_auth(
            projects,
            token_env_by_host={},
            public_hosts=[*_EXPECTED_HOSTS, "example.invalid"],
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


def test_transient_retry_exhaustion_is_resumable(monkeypatch: pytest.MonkeyPatch) -> None:
    class Opener:
        def open(self, *_args: object, **_kwargs: object) -> object:
            raise TimeoutError("simulated timeout")

    client = gitlab.GitLabClient(
        allowed_hosts={"gitlab.com"},
        token_env_by_host={},
        public_hosts={"gitlab.com"},
        max_response_bytes=1024,
        max_retries=0,
        timeout_s=1,
    )
    client.opener = Opener()
    monkeypatch.setattr(gitlab.time, "sleep", lambda _delay: None)
    with pytest.raises(gitlab.GitLabTransientError, match="retry budget exhausted"):
        client.get(
            "https://gitlab.com/api/v4/projects/libeigen%2FEigen/merge_requests"
        )


def test_main_maps_transient_and_contract_failures_to_lane_exit_codes(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def transient(_args: argparse.Namespace) -> dict[str, object]:
        raise gitlab.GitLabTransientError("429 retry budget exhausted")

    monkeypatch.setattr(gitlab, "run", transient)
    assert gitlab.main([]) == 75
    assert "GITLAB_MR_INGEST_RETRYABLE" in capsys.readouterr().err

    def contract(_args: argparse.Namespace) -> dict[str, object]:
        raise gitlab.GitLabIngestError("credential scope is invalid")

    monkeypatch.setattr(gitlab, "run", contract)
    assert gitlab.main([]) == 2
    assert "GITLAB_MR_INGEST_FAILED" in capsys.readouterr().err


def test_sidecar_is_deterministic_gzip_and_resume_requires_frozen_content(
    tmp_path: Path,
) -> None:
    project = gitlab._parse_project_identity("invent.kde.org/utilities%2Fkate")
    path = gitlab._sidecar_path(tmp_path, "inventory", project, 7)
    value = {
        "scan_id": "1" * 64,
        "project_identity": project.identity,
        "iid": 7,
        "metadata": {"title": "frozen"},
    }
    gitlab._write_bound_sidecar(path, value, scan_id="1" * 64)
    first = path.read_bytes()
    assert path.name.endswith(".json.gz")
    assert int.from_bytes(first[4:8], "little") == 0
    assert gzip.decompress(first) == gitlab._canonical_bytes(value, pretty=True)

    path.unlink()
    gitlab._write_bound_sidecar(path, value, scan_id="1" * 64)
    assert path.read_bytes() == first
    gitlab._write_bound_sidecar(path, dict(value), scan_id="1" * 64)
    changed = {**value, "metadata": {"title": "changed"}}
    with pytest.raises(gitlab.GitLabIngestError, match="content drifted"):
        gitlab._write_bound_sidecar(path, changed, scan_id="1" * 64)

    binding = gitlab._sidecar_binding(path, value)
    assert binding == {
        "physical_sha256": hashlib.sha256(first).hexdigest(),
        "physical_byte_size": len(first),
        "logical_sha256": hashlib.sha256(
            gitlab._canonical_bytes(value, pretty=True)
        ).hexdigest(),
        "logical_byte_size": len(gitlab._canonical_bytes(value, pretty=True)),
    }


def test_sidecar_read_has_a_hard_decompression_bound(tmp_path: Path) -> None:
    path = tmp_path / "oversized.json.gz"
    payload = gitlab._canonical_bytes({"body": "x" * 4096}, pretty=True)
    path.write_bytes(gzip.compress(payload, mtime=0))

    with pytest.raises(gitlab.GitLabIngestError, match="decompression bound"):
        gitlab._read_gzip_json_object(
            path,
            role="test sidecar",
            max_compressed_bytes=4096,
            max_decompressed_bytes=128,
        )

    path.write_bytes(gzip.compress(payload, mtime=123))
    with pytest.raises(gitlab.GitLabIngestError, match="mtime=0"):
        gitlab._read_gzip_json_object(
            path,
            role="test sidecar",
            max_compressed_bytes=4096,
            max_decompressed_bytes=8192,
        )

    value = {"body": "canonical"}
    canonical = gitlab._canonical_gzip_bytes(value)
    named = io.BytesIO()
    with gzip.GzipFile(
        filename="sidecar.json",
        mode="wb",
        compresslevel=9,
        fileobj=named,
        mtime=0,
    ) as handle:
        handle.write(gitlab._canonical_bytes(value, pretty=True))
    for noncanonical in (
        canonical + b"\0" * 16,
        canonical + gzip.compress(b"", mtime=0),
        named.getvalue(),
    ):
        path.write_bytes(noncanonical)
        with pytest.raises(gitlab.GitLabIngestError, match="canonical gzip"):
            gitlab._read_gzip_json_object(
                path,
                role="test sidecar",
                max_compressed_bytes=4096,
                max_decompressed_bytes=8192,
            )


def test_sidecar_resume_discards_only_stale_atomic_temporaries(tmp_path: Path) -> None:
    root = tmp_path / "sidecars" / "inventory" / "project"
    root.mkdir(parents=True)
    stale = root / ".000000000007.json.gz.tmp.123"
    current = root / ".000000000008.json.gz.tmp"
    unrelated = root / "keep.txt"
    for path in (stale, current, unrelated):
        path.write_bytes(b"staging")

    gitlab._discard_stale_sidecar_temporaries(tmp_path / "sidecars")

    assert not stale.exists()
    assert not current.exists()
    assert unrelated.read_bytes() == b"staging"


def test_primary_sidecar_replay_materializes_exact_shas_without_fake_reviews(
    tmp_path: Path,
) -> None:
    repo_list = _exact_repo_list(tmp_path)
    projects = gitlab.load_gitlab_repos(
        repo_list,
        expected_hosts=_EXPECTED_HOSTS,
    )
    project = projects[0]
    primary_store = tmp_path / "primary.sqlite"
    ancillary_store = tmp_path / "ancillary.sqlite"
    manifest = gitlab.Manifest(
        tmp_path / "manifest.json",
        repo_list=repo_list,
        repo_list_sha256=gitlab._stable_file_sha256(repo_list, role="test repo list"),
        projects=projects,
        expected_hosts=_EXPECTED_HOSTS,
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
    gitlab._atomic_write_gzip_json(record_path, record)
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
        sidecar_binding = stored["raw"]["record_sidecar"]
        assert sidecar_binding["path"].endswith(".json.gz")
        assert (
            sidecar_binding["physical_sha256"]
            == hashlib.sha256(record_path.read_bytes()).hexdigest()
        )
        logical = gzip.decompress(record_path.read_bytes())
        assert sidecar_binding["physical_byte_size"] == record_path.stat().st_size
        assert sidecar_binding["logical_sha256"] == hashlib.sha256(logical).hexdigest()
        assert sidecar_binding["logical_byte_size"] == len(logical)
        manifest.project(project.identity)["inventory"]["max_iid"] = 7
        physical_set, logical_set, files, physical_bytes, logical_bytes = (
            gitlab._hash_sidecars(manifest, projects, tmp_path / "sidecars")
        )
        assert physical_set != logical_set
        assert (files, physical_bytes, logical_bytes) == (
            1,
            record_path.stat().st_size,
            len(logical),
        )
        assert stored["comments"][0]["kind"] == "review_comment"
        assert stored["reviews"] == []
        assert ancillary_conn.execute("SELECT COUNT(*) FROM prs").fetchone()[0] == 0
    finally:
        primary_conn.close()
        ancillary_conn.close()


@pytest.mark.parametrize("status", [401, 403])
@pytest.mark.parametrize("endpoint", ["discussions", "closes_issues"])
def test_primary_child_auth_fails_loudly_without_terminal_sidecar(
    tmp_path: Path,
    status: int,
    endpoint: str,
) -> None:
    repo_list = _exact_repo_list(tmp_path)
    projects = gitlab.load_gitlab_repos(
        repo_list,
        expected_hosts=_EXPECTED_HOSTS,
    )
    project = next(item for item in projects if item.host == "gitlab.com")
    primary_store = tmp_path / "primary.sqlite"
    ancillary_store = tmp_path / "ancillary.sqlite"
    sidecar_root = tmp_path / "sidecars"
    manifest = gitlab.Manifest(
        tmp_path / "manifest.json",
        repo_list=repo_list,
        repo_list_sha256=gitlab._stable_file_sha256(repo_list, role="test repo list"),
        projects=projects,
        expected_hosts=_EXPECTED_HOSTS,
        config={"max_detail_mib": 4, "max_response_mib": 2},
    )
    source_sha = "1" * 40
    target_sha = "2" * 40
    base_sha = "3" * 40
    merge_sha = "4" * 40
    iid = 175
    metadata = {
        "id": 7001,
        "iid": iid,
        "project_id": 101,
        "state": "merged",
        "title": "Native fix",
        "created_at": "2026-08-03T08:00:00.000000Z",
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
        "reviewers": [],
        "changes_count": "1",
    }
    diff = {"old_path": "src/fix.cpp", "new_path": "src/fix.cpp"}

    def response(
        url: str,
        response_status: int,
        body: object,
        *,
        headers: dict[str, str] | None = None,
    ) -> gitlab.APIResponse:
        payload = json.dumps(
            body,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return gitlab.APIResponse(
            url=url,
            status=response_status,
            headers=headers or {},
            body=body,
            body_sha256=hashlib.sha256(payload).hexdigest(),
            byte_size=len(payload),
        )

    class Client:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def get(
            self,
            url: str,
            *,
            terminal_statuses: set[int] | None = None,
        ) -> gitlab.APIResponse:
            self.calls.append(url)
            request_path = url.split("?", 1)[0]
            if request_path.endswith(f"/merge_requests/{iid}"):
                current = response(url, 200, detail)
            elif request_path.endswith("/diffs"):
                current = response(url, 200, [diff], headers={"x-total": "1"})
            elif request_path.endswith("/discussions"):
                if endpoint == "discussions":
                    assert terminal_statuses == {404, 410}
                    raise gitlab.GitLabIngestError(
                        f"GitLab HTTP {status} for {url}: Unauthorized"
                    )
                current = response(url, 200, [], headers={"x-total": "0"})
            elif request_path.endswith("/closes_issues"):
                assert endpoint == "closes_issues"
                assert terminal_statuses == {404, 410}
                raise gitlab.GitLabIngestError(
                    f"GitLab HTTP {status} for {url}: Unauthorized"
                )
            else:
                raise AssertionError(f"unexpected GitLab endpoint: {url}")
            if current.status not in {200, *(terminal_statuses or set())}:
                raise gitlab.GitLabIngestError(
                    f"unexpected status {current.status} for {url}"
                )
            return current

    primary_conn = pr_store.connect(str(primary_store), create=True)
    ancillary_conn = pr_store.connect(str(ancillary_store), create=True)
    try:
        client = Client()
        with pytest.raises(gitlab.GitLabIngestError, match=rf"GitLab HTTP {status}"):
            gitlab._process_candidate(
                client,  # type: ignore[arg-type]
                manifest,
                project,
                metadata,
                sidecar_root=sidecar_root,
                primary_conn=primary_conn,
                ancillary_conn=ancillary_conn,
                page_size=100,
                max_detail_pages=10,
                max_detail_bytes=4 * 1024 * 1024,
            )
        expected_calls = [
            str(iid),
            "diffs",
            "discussions",
        ]
        if endpoint == "closes_issues":
            expected_calls.append("closes_issues")
        assert [
            url.split("?", 1)[0].rsplit("/", 1)[-1] for url in client.calls
        ] == expected_calls
        terminal_path = gitlab._sidecar_path(
            sidecar_root,
            "records/terminal",
            project,
            iid,
        )
        assert not terminal_path.exists()
        assert primary_conn.execute("SELECT COUNT(*) FROM prs").fetchone()[0] == 0
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
    assert binding["expected_host_count"] == len(_EXPECTED_HOSTS)
    assert binding["expected_hosts_sha256"] == gitlab._canonical_sha256(
        list(_EXPECTED_HOSTS)
    )
    assert binding["training_ready_without_membership"] is False
    receipt = json.loads(receipt_path.read_bytes())
    assert receipt["expected_hosts"] == list(_EXPECTED_HOSTS)
    assert receipt["sidecars"] == {
        "root": str((tmp_path / "sidecars").resolve()),
        "format": "canonical-json-gzip",
        "gzip_mtime": 0,
        "physical_set_sha256": hashlib.sha256().hexdigest(),
        "logical_set_sha256": hashlib.sha256().hexdigest(),
        "files": 0,
        "physical_byte_size": 0,
        "logical_byte_size": 0,
    }
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

    loose = _sidecar_root / "loose.json.gz"
    gitlab._atomic_write_gzip_json(loose, {"unexpected": True})
    with pytest.raises(gitlab.GitLabIngestError, match="unexpected artifact"):
        gitlab.load_gitlab_completion_binding(
            receipt_path,
            pr_store=primary_store,
            repo_list=repo_list,
        )


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


def test_completion_rejects_expected_host_scope_drift(tmp_path: Path) -> None:
    primary_store, receipt_path, repo_list, _sidecar_root, _manifest = (
        _empty_completion(tmp_path)
    )
    receipt = json.loads(receipt_path.read_bytes())
    receipt["expected_hosts_sha256"] = "0" * 64
    gitlab._atomic_write_json(receipt_path, receipt)

    with pytest.raises(gitlab.GitLabIngestError, match="host scope drifted"):
        gitlab.load_gitlab_completion_binding(
            receipt_path,
            pr_store=primary_store,
            repo_list=repo_list,
        )


def test_inventory_window_query_brackets_inclusive_microseconds() -> None:
    start = datetime(2026, 8, 3, 8, 0, 0, 123456, tzinfo=UTC)
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
