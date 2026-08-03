#!/usr/bin/env python3
"""Resumable GitLab merge-request inventory and primary-domain routing.

The canonical mixed ``repo_list.json`` currently contains exactly three GitLab
projects.  This scanner freezes a ``created_at`` upper bound, inventories every
MR in stable ascending time windows, and only expands merged candidates.  Diff
paths are classified with the same primary commit-scope helper used by the
source conveyor.  Primary records use the existing PRStore schema; Python/JS-
only records go to a physically separate store, and all other records remain
receipt-bound sidecars only.

The completion receipt deliberately says ``training_ready_without_membership``
is false.  CASE5 export still requires the independent, exact primary-commit
membership receipt consumed by ``export_pr_parquet.py``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
import fcntl
import hashlib
import http.client
import json
import os
from pathlib import Path
import re
import sqlite3
import sys
import time
from typing import Any, Iterable, Mapping
import urllib.error
import urllib.parse
import urllib.request

if __package__ in (None, ""):
    _REPO_ROOT = Path(__file__).resolve().parents[2]
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))

from cppmega_mlx.data.commit_scope import classify_primary_commit_path  # noqa: E402
from scripts.pr_ingest import pr_store  # noqa: E402

GITLAB_COMPLETION_SCHEMA = "cppmega_gitlab_mr_completion_v1"
GITLAB_MANIFEST_SCHEMA = "cppmega_gitlab_mr_stream_manifest_v1"
GITLAB_INVENTORY_SCHEMA = "cppmega_gitlab_mr_inventory_v1"
GITLAB_INVENTORY_PAGE_SCHEMA = "cppmega_gitlab_mr_inventory_page_v1"
GITLAB_RECORD_SCHEMA = "cppmega_gitlab_mr_record_v1"
GITLAB_PLATFORM = "gitlab"
GITLAB_CONTRACT_VERSION = 1
_CONTRACT = {
    "version": GITLAB_CONTRACT_VERSION,
    "inventory": "project-MRs-created-at-ascending-inclusive-windows-v1",
    "candidate": "state-merged-and-merge-commit-sha-present-v1",
    "detail_endpoints": ["merge_request", "diffs"],
    "primary_endpoints": ["discussions", "closes_issues"],
    "preflight": "one-MR-list-credential-check-v1",
    "path_classifier": "cppmega_mlx.data.commit_scope.classify_primary_commit_path",
    "routes": ["primary", "ancillary", "excluded", "terminal"],
}
GITLAB_CONTRACT_SHA256 = hashlib.sha256(
    json.dumps(_CONTRACT, separators=(",", ":"), sort_keys=True).encode("utf-8")
).hexdigest()

_EXPECTED_GITLAB_REPOS = (
    "gitlab.com/libeigen%2FEigen",
    "gitlab.freedesktop.org/mesa%2Fmesa",
    "gitlab.torproject.org/tpo%2Fcore%2Ftor",
)
_TERMINAL_DETAIL_STATUSES = {404, 410}
_TRANSIENT_STATUSES = {408, 409, 425, 429, 500, 502, 503, 504}
_TRANSIENT_ERRORS = (
    http.client.IncompleteRead,
    http.client.RemoteDisconnected,
    TimeoutError,
    ConnectionError,
    urllib.error.URLError,
)
_SHA_RE = re.compile(r"^[0-9a-fA-F]{40,64}$")
_LINK_NEXT_RE = re.compile(r'(?:^|,)\s*<([^>]+)>\s*;\s*rel="next"')
_ANCILLARY_SUFFIXES = {".py", ".pyi", ".js", ".jsx", ".mjs", ".cjs", ".ts", ".tsx"}


class GitLabIngestError(RuntimeError):
    """The scan cannot continue without weakening its exactness contract."""


@dataclass(frozen=True)
class GitLabProject:
    identity: str
    host: str
    encoded_path: str

    @property
    def api_root(self) -> str:
        return f"https://{self.host}/api/v4/projects/{self.encoded_path}"

    @property
    def sidecar_key(self) -> str:
        return hashlib.sha256(self.identity.encode("utf-8")).hexdigest()[:20]


@dataclass(frozen=True)
class APIResponse:
    url: str
    status: int
    headers: dict[str, str]
    body: object
    body_sha256: str
    byte_size: int

    def lineage(self, endpoint: str, *, page: int | None = None) -> dict[str, object]:
        out: dict[str, object] = {
            "endpoint": endpoint,
            "url": self.url,
            "status": self.status,
            "body_sha256": self.body_sha256,
            "byte_size": self.byte_size,
            "headers": {
                key: self.headers[key]
                for key in (
                    "date",
                    "etag",
                    "link",
                    "ratelimit-limit",
                    "ratelimit-remaining",
                    "ratelimit-reset",
                    "x-next-page",
                    "x-page",
                    "x-per-page",
                    "x-request-id",
                    "x-total",
                    "x-total-pages",
                )
                if key in self.headers
            },
        }
        if page is not None:
            out["page"] = page
        return out


def _canonical_bytes(value: object, *, pretty: bool = False) -> bytes:
    if pretty:
        return (
            json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8")
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256_file(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_file_sha256(path: Path, *, role: str) -> str:
    before = path.stat()
    digest = _sha256_file(path)
    after = path.stat()
    identity = lambda stat: (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
    if identity(before) != identity(after):
        raise GitLabIngestError(f"{role} changed while hashing: {path}")
    return digest


def _atomic_write_json(path: Path, value: object) -> None:
    if path.is_symlink():
        raise GitLabIngestError(f"refusing to replace symlinked JSON artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _canonical_bytes(value, pretty=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    if temporary.is_symlink():
        raise GitLabIngestError(
            f"refusing symlinked JSON staging artifact: {temporary}"
        )
    try:
        with temporary.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()


def _read_json_object(
    path: Path, *, role: str, max_bytes: int = 32 * 1024 * 1024
) -> dict:
    if not path.is_file() or path.is_symlink():
        raise GitLabIngestError(f"{role} is missing or symlinked: {path}")
    before = path.stat()
    size = before.st_size
    if size < 1 or size > max_bytes:
        raise GitLabIngestError(f"{role} has invalid byte size {size}: {path}")
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise GitLabIngestError(f"{role} cannot be read: {path}: {exc}") from exc
    after = path.stat()
    identity = lambda stat: (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
    if len(payload) != size or identity(before) != identity(after):
        raise GitLabIngestError(f"{role} changed while reading: {path}")
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GitLabIngestError(f"{role} is invalid JSON: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise GitLabIngestError(f"{role} must be a JSON object: {path}")
    return value


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def _parse_time(value: object, *, field: str) -> datetime:
    if not isinstance(value, str) or not value:
        raise GitLabIngestError(f"{field} must be a non-empty ISO-8601 timestamp")
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise GitLabIngestError(f"{field} is not ISO-8601: {value!r}") from exc
    if parsed.tzinfo is None:
        raise GitLabIngestError(f"{field} must include a timezone: {value!r}")
    return parsed.astimezone(timezone.utc)


def _format_time(value: datetime) -> str:
    return (
        value.astimezone(timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def _require_int(value: object, *, field: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise GitLabIngestError(f"{field} must be an integer >= {minimum}: {value!r}")
    return value


def _require_sha(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _SHA_RE.fullmatch(value) is None:
        raise GitLabIngestError(f"{field} must be a 40-64 digit hexadecimal SHA")
    return value.lower()


def _parse_project_identity(identity: str) -> GitLabProject:
    if identity.count("/") != 1:
        raise GitLabIngestError(f"invalid GitLab project identity: {identity!r}")
    host, encoded_path = identity.split("/", 1)
    decoded = urllib.parse.unquote(encoded_path)
    normalized = urllib.parse.quote(decoded, safe="")
    if (
        not host.startswith("gitlab")
        or "/" not in decoded
        or normalized != encoded_path
    ):
        raise GitLabIngestError(f"non-canonical GitLab project identity: {identity!r}")
    return GitLabProject(identity=identity, host=host, encoded_path=encoded_path)


def load_gitlab_repos(repo_list: Path) -> tuple[GitLabProject, ...]:
    document = _read_json_object(repo_list, role="GitLab repo list")
    if document.get("schema_version") != 2 or document.get("unresolved") != []:
        raise GitLabIngestError("GitLab repo list must be resolved schema_version 2")
    rows = document.get("repos")
    if not isinstance(rows, list):
        raise GitLabIngestError("GitLab repo list lacks a repos array")
    identities: list[str] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise GitLabIngestError(f"repo_list repos[{index}] must be an object")
        identity = row.get("project_identity")
        if isinstance(identity, str) and identity.startswith("gitlab"):
            if row.get("owner_repo") is not None:
                raise GitLabIngestError(
                    f"GitLab repo_list row must not masquerade as GitHub: {identity}"
                )
            identities.append(identity)
    if len(identities) != len(set(identities)):
        raise GitLabIngestError("canonical GitLab repo scope contains duplicates")
    observed = tuple(sorted(identities))
    expected = tuple(sorted(_EXPECTED_GITLAB_REPOS))
    if observed != expected:
        raise GitLabIngestError(
            "canonical GitLab repo scope drifted: "
            f"missing={sorted(set(expected) - set(observed))} "
            f"extra={sorted(set(observed) - set(expected))}"
        )
    return tuple(_parse_project_identity(identity) for identity in observed)


def classify_diff_paths(diffs: Iterable[Mapping[str, object]]) -> dict[str, list[dict]]:
    """Route each exact GitLab diff without allowing Python/JS into primary."""

    routed: dict[str, list[dict]] = {"primary": [], "ancillary": [], "excluded": []}
    for index, raw in enumerate(diffs):
        item = dict(raw)
        paths = []
        for field in ("old_path", "new_path"):
            value = item.get(field)
            if not isinstance(value, str) or not value:
                raise GitLabIngestError(f"diff[{index}].{field} must be a path string")
            paths.append(value)
        if any(classify_primary_commit_path(path) is not None for path in paths):
            route = "primary"
        elif any(Path(path).suffix.lower() in _ANCILLARY_SUFFIXES for path in paths):
            route = "ancillary"
        else:
            route = "excluded"
        routed[route].append(item)
    return routed


def _candidate(metadata: Mapping[str, object]) -> bool:
    merge_sha = metadata.get("merge_commit_sha")
    return (
        metadata.get("state") == "merged"
        and isinstance(merge_sha, str)
        and _SHA_RE.fullmatch(merge_sha) is not None
    )


class _RejectRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001
        return None


class GitLabClient:
    def __init__(
        self,
        *,
        allowed_hosts: Iterable[str],
        token_env_by_host: Mapping[str, str],
        max_response_bytes: int,
        max_retries: int,
        timeout_s: float,
    ):
        self.allowed_hosts = frozenset(allowed_hosts)
        self.max_response_bytes = max_response_bytes
        self.max_retries = max_retries
        self.timeout_s = timeout_s
        self.tokens: dict[str, str] = {}
        for host, env_name in token_env_by_host.items():
            if host not in self.allowed_hosts:
                raise GitLabIngestError(
                    f"token configured for out-of-scope host {host!r}"
                )
            token = os.environ.get(env_name)
            if not token:
                raise GitLabIngestError(
                    f"token environment variable is empty: {env_name}"
                )
            self.tokens[host] = token
        self.opener = urllib.request.build_opener(_RejectRedirects())

    def _validate_url(self, url: str) -> str:
        parsed = urllib.parse.urlsplit(url)
        if (
            parsed.scheme != "https"
            or parsed.hostname not in self.allowed_hosts
            or parsed.username is not None
            or parsed.password is not None
            or not parsed.path.startswith("/api/v4/projects/")
        ):
            raise GitLabIngestError(f"refusing out-of-contract GitLab API URL: {url}")
        return str(parsed.hostname)

    def _read(self, handle, *, url: str) -> bytes:  # noqa: ANN001
        body = handle.read(self.max_response_bytes + 1)
        if len(body) > self.max_response_bytes:
            raise GitLabIngestError(
                f"GitLab response exceeded {self.max_response_bytes} bytes: {url}"
            )
        return body

    @staticmethod
    def _delay(headers: Mapping[str, str], attempt: int) -> float:
        raw = headers.get("retry-after")
        if raw:
            try:
                delay = float(raw)
            except ValueError:
                try:
                    parsed = parsedate_to_datetime(raw)
                except (TypeError, ValueError) as exc:
                    raise GitLabIngestError(
                        f"invalid Retry-After header: {raw!r}"
                    ) from exc
                delay = max(0.0, parsed.timestamp() - time.time())
            if delay < 0 or not delay < float("inf"):
                raise GitLabIngestError(f"invalid Retry-After delay: {raw!r}")
            return min(60.0, delay)
        return min(30.0, float(2 ** min(attempt, 5)))

    def get(
        self, url: str, *, terminal_statuses: set[int] | None = None
    ) -> APIResponse:
        host = self._validate_url(url)
        terminal_statuses = terminal_statuses or set()
        attempt = 0
        while True:
            headers = {
                "Accept": "application/json",
                "Accept-Encoding": "identity",
                "User-Agent": "cppmega-gitlab-mr-ingest/1",
            }
            if host in self.tokens:
                headers["PRIVATE-TOKEN"] = self.tokens[host]
            request = urllib.request.Request(url, headers=headers, method="GET")
            try:
                with self.opener.open(request, timeout=self.timeout_s) as response:
                    status = int(response.status)
                    response_headers = {
                        key.lower(): value for key, value in response.headers.items()
                    }
                    body_bytes = self._read(response, url=url)
            except urllib.error.HTTPError as exc:
                status = int(exc.code)
                response_headers = {
                    key.lower(): value for key, value in exc.headers.items()
                }
                body_bytes = self._read(exc, url=url)
            except _TRANSIENT_ERRORS as exc:
                attempt += 1
                if attempt > self.max_retries:
                    raise GitLabIngestError(
                        f"GitLab transport retry budget exhausted for {url}: {exc}"
                    ) from exc
                time.sleep(min(30.0, float(2 ** min(attempt - 1, 5))))
                continue

            if status in _TRANSIENT_STATUSES:
                attempt += 1
                if attempt > self.max_retries:
                    raise GitLabIngestError(
                        f"GitLab HTTP retry budget exhausted at status {status}: {url}"
                    )
                time.sleep(self._delay(response_headers, attempt - 1))
                continue
            if status not in {200, *terminal_statuses}:
                preview = body_bytes[:1000].decode("utf-8", errors="replace")
                raise GitLabIngestError(f"GitLab HTTP {status} for {url}: {preview}")
            try:
                body: object = json.loads(body_bytes) if body_bytes else None
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise GitLabIngestError(
                    f"GitLab returned invalid JSON for {url}"
                ) from exc
            return APIResponse(
                url=url,
                status=status,
                headers=response_headers,
                body=body,
                body_sha256=hashlib.sha256(body_bytes).hexdigest(),
                byte_size=len(body_bytes),
            )


def _url_with_query(url: str, **params: object) -> str:
    return url + "?" + urllib.parse.urlencode(params)


def _next_page_url(
    response: APIResponse,
    *,
    page: int,
    page_size: int,
    item_count: int,
) -> str | None:
    link = response.headers.get("link")
    if link:
        match = _LINK_NEXT_RE.search(link)
        if match:
            return urllib.parse.urljoin(response.url, match.group(1))
    raw_next = response.headers.get("x-next-page")
    if raw_next:
        try:
            next_page = int(raw_next)
        except ValueError as exc:
            raise GitLabIngestError(
                f"invalid X-Next-Page header: {raw_next!r}"
            ) from exc
        if next_page <= page:
            raise GitLabIngestError(
                f"non-advancing X-Next-Page header: {raw_next!r} after page {page}"
            )
        parsed = urllib.parse.urlsplit(response.url)
        query = urllib.parse.parse_qs(parsed.query, keep_blank_values=True)
        query["page"] = [str(next_page)]
        return urllib.parse.urlunsplit(
            parsed._replace(query=urllib.parse.urlencode(query, doseq=True))
        )
    raw_total = response.headers.get("x-total")
    if raw_total is not None:
        try:
            total = int(raw_total)
        except ValueError as exc:
            raise GitLabIngestError(f"invalid X-Total header: {raw_total!r}") from exc
        if page * page_size >= total:
            return None
    if item_count == page_size:
        parsed = urllib.parse.urlsplit(response.url)
        query = urllib.parse.parse_qs(parsed.query, keep_blank_values=True)
        query["page"] = [str(page + 1)]
        return urllib.parse.urlunsplit(
            parsed._replace(query=urllib.parse.urlencode(query, doseq=True))
        )
    return None


def _require_array(response: APIResponse, *, endpoint: str) -> list[dict]:
    if not isinstance(response.body, list) or any(
        not isinstance(item, dict) for item in response.body
    ):
        raise GitLabIngestError(f"{endpoint} response must be an array of objects")
    return [dict(item) for item in response.body]


def _preflight_project_access(client: GitLabClient, project: GitLabProject) -> None:
    """Fail before inventory if the configured host credential cannot list MRs."""

    listing = client.get(
        _url_with_query(
            f"{project.api_root}/merge_requests",
            state="all",
            order_by="updated_at",
            sort="desc",
            per_page=1,
            page=1,
        )
    )
    _require_array(listing, endpoint="preflight_merge_requests")


def _paged_get(
    client: GitLabClient,
    url: str,
    *,
    endpoint: str,
    page_size: int,
    max_pages: int,
    max_total_bytes: int,
    terminal_statuses: set[int] | None = None,
) -> tuple[list[dict], list[dict[str, object]], int | None, int]:
    if max_pages < 1 or max_total_bytes < 1:
        raise GitLabIngestError(
            f"{endpoint} has no remaining page/byte budget for exhaustive retrieval"
        )
    items: list[dict] = []
    lineage: list[dict[str, object]] = []
    page = 1
    total: int | None = None
    total_bytes = 0
    next_url: str | None = _url_with_query(url, per_page=page_size, page=page)
    while next_url is not None:
        if page > max_pages:
            raise GitLabIngestError(
                f"{endpoint} exceeded the configured {max_pages}-page bound"
            )
        response = client.get(next_url, terminal_statuses=terminal_statuses)
        lineage.append(response.lineage(endpoint, page=page))
        if response.status in (terminal_statuses or set()):
            return [], lineage, None, total_bytes
        total_bytes += response.byte_size
        if total_bytes > max_total_bytes:
            raise GitLabIngestError(
                f"{endpoint} exceeded the configured {max_total_bytes}-byte bound"
            )
        page_items = _require_array(response, endpoint=endpoint)
        raw_total = response.headers.get("x-total")
        if raw_total is not None:
            try:
                page_total = int(raw_total)
            except ValueError as exc:
                raise GitLabIngestError(
                    f"invalid X-Total header: {raw_total!r}"
                ) from exc
            if total is None:
                total = page_total
            elif total != page_total:
                raise GitLabIngestError(f"{endpoint} X-Total changed during pagination")
        items.extend(page_items)
        next_url = _next_page_url(
            response,
            page=page,
            page_size=page_size,
            item_count=len(page_items),
        )
        page += 1
    if total is not None and len(items) != total:
        raise GitLabIngestError(
            f"{endpoint} pagination incomplete: fetched={len(items)} total={total}"
        )
    return items, lineage, total, total_bytes


class Manifest:
    def __init__(
        self,
        path: Path,
        *,
        repo_list: Path,
        repo_list_sha256: str,
        projects: tuple[GitLabProject, ...],
        config: Mapping[str, object],
    ):
        self.path = path
        expected_config = dict(config)
        if path.exists():
            self.data = _read_json_object(path, role="GitLab MR manifest")
            if (
                self.data.get("schema") != GITLAB_MANIFEST_SCHEMA
                or self.data.get("contract_sha256") != GITLAB_CONTRACT_SHA256
                or self.data.get("repo_list")
                != {"path": str(repo_list.resolve()), "sha256": repo_list_sha256}
                or self.data.get("config") != expected_config
                or set(self.data.get("projects", {}))
                != {item.identity for item in projects}
            ):
                raise GitLabIngestError(
                    f"existing manifest is malformed or bound to different inputs: {path}"
                )
        else:
            started = _utc_now()
            scan_id = hashlib.sha256(
                f"{GITLAB_CONTRACT_SHA256}:{started}:{os.getpid()}".encode("utf-8")
            ).hexdigest()
            self.data = {
                "schema": GITLAB_MANIFEST_SCHEMA,
                "contract_sha256": GITLAB_CONTRACT_SHA256,
                "scan_id": scan_id,
                "scan_started_at": started,
                "repo_list": {
                    "path": str(repo_list.resolve()),
                    "sha256": repo_list_sha256,
                },
                "config": expected_config,
                "projects": {
                    project.identity: {
                        "status": "pending",
                        "inventory": {
                            "count": 0,
                            "pages": 0,
                            "max_iid": 0,
                            "completed_windows": [],
                        },
                        "details": {
                            "next_iid": 1,
                            "candidate_count": 0,
                            "primary_count": 0,
                            "ancillary_count": 0,
                            "excluded_count": 0,
                            "terminal_count": 0,
                        },
                    }
                    for project in projects
                },
            }
            self.save()
        scan_id = self.data.get("scan_id")
        if (
            not isinstance(scan_id, str)
            or re.fullmatch(r"[0-9a-f]{64}", scan_id) is None
        ):
            raise GitLabIngestError("GitLab manifest scan_id is invalid")
        _parse_time(self.data.get("scan_started_at"), field="manifest.scan_started_at")

    @property
    def scan_id(self) -> str:
        return str(self.data["scan_id"])

    def project(self, identity: str) -> dict:
        value = self.data["projects"][identity]
        if not isinstance(value, dict):
            raise GitLabIngestError(f"manifest project state is malformed: {identity}")
        return value

    def save(self) -> None:
        _atomic_write_json(self.path, self.data)


class RunLock:
    def __init__(self, manifest: Path):
        self.path = Path(f"{manifest}.lock")
        self.handle = None

    def __enter__(self):
        if self.path.is_symlink():
            raise GitLabIngestError(f"GitLab ingest lock is symlinked: {self.path}")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise GitLabIngestError(
                f"GitLab ingest is already running: {self.path}"
            ) from exc
        self.handle.seek(0)
        self.handle.truncate()
        self.handle.write(json.dumps({"pid": os.getpid(), "started_at": _utc_now()}))
        self.handle.write("\n")
        self.handle.flush()
        return self

    def __exit__(self, *_exc: object) -> None:
        assert self.handle is not None
        fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
        self.handle.close()


def _sidecar_path(root: Path, section: str, project: GitLabProject, iid: int) -> Path:
    return root / section / project.sidecar_key / f"{iid:012d}.json"


def _write_bound_sidecar(path: Path, value: dict, *, scan_id: str) -> None:
    if path.is_symlink():
        raise GitLabIngestError(f"GitLab MR sidecar is symlinked: {path}")
    if path.exists():
        existing = _read_json_object(
            path,
            role="GitLab MR sidecar",
            max_bytes=256 * 1024 * 1024,
        )
        if (
            existing.get("scan_id") != scan_id
            or existing.get("project_identity") != value.get("project_identity")
            or existing.get("iid") != value.get("iid")
        ):
            raise GitLabIngestError(
                f"sidecar belongs to a different scan or MR: {path}"
            )
        if existing != value:
            raise GitLabIngestError(f"sidecar content drifted on resume: {path}")
        return
    _atomic_write_json(path, value)


def _inventory_page_journal_path(manifest: Manifest) -> Path:
    return manifest.path.with_name(f"{manifest.path.name}.inventory-page.json")


def _discard_inventory_page_journal(manifest: Manifest) -> None:
    path = _inventory_page_journal_path(manifest)
    if path.is_symlink():
        raise GitLabIngestError(f"inventory page journal is symlinked: {path}")
    if not path.exists():
        return
    journal = _read_json_object(path, role="GitLab inventory page journal")
    if journal.get("scan_id") != manifest.scan_id:
        raise GitLabIngestError(
            f"inventory page journal belongs to another scan: {path}"
        )
    path.unlink()


def _inventory_url(
    project: GitLabProject,
    *,
    window_start: datetime,
    window_end: datetime,
    page: int,
    page_size: int,
) -> str:
    return _url_with_query(
        f"{project.api_root}/merge_requests",
        state="all",
        order_by="created_at",
        sort="asc",
        created_after=_format_time(window_start - timedelta(microseconds=1)),
        created_before=_format_time(window_end + timedelta(microseconds=1)),
        per_page=page_size,
        page=page,
    )


def _inventory_page(
    client: GitLabClient,
    manifest: Manifest,
    project: GitLabProject,
    *,
    window_start: datetime,
    window_end: datetime,
    page: int,
    page_size: int,
) -> dict[str, object]:
    """Fetch one page once, then replay its atomic journal after interruption."""

    path = _inventory_page_journal_path(manifest)
    identity = {
        "scan_id": manifest.scan_id,
        "project_identity": project.identity,
        "window_start": _format_time(window_start),
        "window_end": _format_time(window_end),
        "page": page,
    }
    if path.exists():
        existing = _read_json_object(
            path,
            role="GitLab inventory page journal",
            max_bytes=max(1024 * 1024, client.max_response_bytes * 4),
        )
        if existing.get("scan_id") != manifest.scan_id:
            raise GitLabIngestError(
                f"inventory page journal belongs to another scan: {path}"
            )
        if all(existing.get(key) == value for key, value in identity.items()):
            if existing.get("schema") != GITLAB_INVENTORY_PAGE_SCHEMA:
                raise GitLabIngestError(
                    f"inventory page journal schema drifted: {path}"
                )
            items = existing.get("items")
            if not isinstance(items, list) or any(
                not isinstance(item, dict) for item in items
            ):
                raise GitLabIngestError(
                    f"inventory page journal items are malformed: {path}"
                )
            if not isinstance(existing.get("has_next"), bool):
                raise GitLabIngestError(
                    f"inventory page journal has_next is malformed: {path}"
                )
            total = existing.get("total")
            if total is not None:
                _require_int(total, field="inventory page journal total")
            if not isinstance(existing.get("lineage"), dict):
                raise GitLabIngestError(
                    f"inventory page journal lineage is malformed: {path}"
                )
            return existing

    url = _inventory_url(
        project,
        window_start=window_start,
        window_end=window_end,
        page=page,
        page_size=page_size,
    )
    response = client.get(url)
    items = _require_array(response, endpoint="merge_requests")
    total: int | None = None
    raw_total = response.headers.get("x-total")
    if raw_total is not None:
        try:
            total = int(raw_total)
        except ValueError as exc:
            raise GitLabIngestError(
                f"invalid inventory X-Total: {raw_total!r}"
            ) from exc
    journal: dict[str, object] = {
        "schema": GITLAB_INVENTORY_PAGE_SCHEMA,
        **identity,
        "items": items,
        "total": total,
        "has_next": _next_page_url(
            response,
            page=page,
            page_size=page_size,
            item_count=len(items),
        )
        is not None,
        "lineage": response.lineage("merge_requests", page=page),
    }
    _atomic_write_json(path, journal)
    return journal


def _validate_inventory_item(
    project: GitLabProject, item: dict
) -> tuple[int, datetime]:
    iid = _require_int(item.get("iid"), field=f"{project.identity}.iid", minimum=1)
    created = _parse_time(
        item.get("created_at"), field=f"{project.identity}!{iid}.created_at"
    )
    _require_int(
        item.get("project_id"), field=f"{project.identity}!{iid}.project_id", minimum=1
    )
    _require_int(item.get("id"), field=f"{project.identity}!{iid}.id", minimum=1)
    if not isinstance(item.get("title"), str) or not isinstance(item.get("state"), str):
        raise GitLabIngestError(
            f"GitLab MR inventory identity is incomplete: {project.identity}!{iid}"
        )
    merge_sha = item.get("merge_commit_sha")
    if merge_sha is not None:
        item["merge_commit_sha"] = _require_sha(
            merge_sha,
            field=f"{project.identity}!{iid}.merge_commit_sha",
        )
    return iid, created


def _inventory_project(
    client: GitLabClient,
    manifest: Manifest,
    project: GitLabProject,
    *,
    sidecar_root: Path,
    window_days: int,
    page_size: int,
    max_window_pages: int,
) -> None:
    state = manifest.project(project.identity)
    if state.get("status") in {"inventory_done", "details", "done"}:
        _discard_inventory_page_journal(manifest)
        return
    inventory = state["inventory"]
    scan_started = _parse_time(
        manifest.data["scan_started_at"], field="scan_started_at"
    )
    if state.get("status") == "pending":
        probe_url = _url_with_query(
            f"{project.api_root}/merge_requests",
            state="all",
            order_by="created_at",
            sort="asc",
            created_before=_format_time(scan_started + timedelta(microseconds=1)),
            per_page=1,
            page=1,
        )
        probe = client.get(probe_url)
        items = _require_array(probe, endpoint="merge_requests_probe")
        inventory["probe"] = probe.lineage("merge_requests_probe", page=1)
        if not items:
            inventory["empty_at_cutoff"] = True
            state["status"] = "inventory_done"
            manifest.save()
            _discard_inventory_page_journal(manifest)
            return
        _iid, earliest = _validate_inventory_item(project, items[0])
        inventory.update(
            {
                "earliest_created_at": _format_time(earliest),
                "window_start": _format_time(earliest),
                "window_end": None,
                "next_page": 1,
                "window_seen": 0,
                "window_total": None,
                "window_pages": 0,
                "last_created_at": None,
            }
        )
        state["status"] = "inventory"
        manifest.save()

    while state.get("status") == "inventory":
        window_start = _parse_time(inventory["window_start"], field="window_start")
        if window_start > scan_started:
            state["status"] = "inventory_done"
            inventory.pop("window_end", None)
            inventory.pop("next_page", None)
            manifest.save()
            _discard_inventory_page_journal(manifest)
            break
        raw_end = inventory.get("window_end")
        if raw_end is None:
            window_end = min(
                scan_started,
                window_start + timedelta(days=window_days) - timedelta(microseconds=1),
            )
            inventory["window_end"] = _format_time(window_end)
            inventory["next_page"] = 1
            inventory["window_seen"] = 0
            inventory["window_total"] = None
            inventory["window_pages"] = 0
            inventory["last_created_at"] = None
            manifest.save()
        else:
            window_end = _parse_time(raw_end, field="window_end")
        page = _require_int(inventory["next_page"], field="next_page", minimum=1)
        if page > max_window_pages:
            raise GitLabIngestError(
                f"{project.identity} window exceeded {max_window_pages} pages; "
                "rerun a new scan with a smaller --window-days"
            )
        page_journal = _inventory_page(
            client,
            manifest,
            project,
            window_start=window_start,
            window_end=window_end,
            page=page,
            page_size=page_size,
        )
        raw_items = page_journal.get("items")
        if not isinstance(raw_items, list) or any(
            not isinstance(item, dict) for item in raw_items
        ):
            raise GitLabIngestError("inventory page journal items are malformed")
        page_items = [dict(item) for item in raw_items]
        observed_total = page_journal.get("total")
        if observed_total is not None:
            observed_total = _require_int(observed_total, field="inventory X-Total")
            if observed_total > max_window_pages * page_size:
                raise GitLabIngestError(
                    f"{project.identity} window has {observed_total} MRs, beyond the "
                    "configured exhaustive offset bound"
                )
            prior_total = inventory.get("window_total")
            if prior_total not in (None, observed_total):
                raise GitLabIngestError(f"{project.identity} window X-Total changed")
            inventory["window_total"] = observed_total

        page_iids: set[int] = set()
        last_created = (
            _parse_time(inventory["last_created_at"], field="last_created_at")
            if inventory.get("last_created_at")
            else None
        )
        for item in page_items:
            iid, created = _validate_inventory_item(project, item)
            if iid in page_iids:
                raise GitLabIngestError(
                    f"{project.identity} inventory page repeats IID {iid}"
                )
            page_iids.add(iid)
            if created < window_start or created > window_end:
                raise GitLabIngestError(
                    f"{project.identity}!{iid} escaped its created_at window"
                )
            if last_created is not None and created < last_created:
                raise GitLabIngestError(
                    f"{project.identity} created_at order regressed at !{iid}"
                )
            sidecar = {
                "schema": GITLAB_INVENTORY_SCHEMA,
                "scan_id": manifest.scan_id,
                "platform": GITLAB_PLATFORM,
                "project_identity": project.identity,
                "host": project.host,
                "project_path": urllib.parse.unquote(project.encoded_path),
                "iid": iid,
                "metadata": item,
                "lineage": page_journal["lineage"],
            }
            path = _sidecar_path(sidecar_root, "inventory", project, iid)
            _write_bound_sidecar(path, sidecar, scan_id=manifest.scan_id)
            inventory["count"] = _require_int(inventory["count"], field="count") + 1
            inventory["max_iid"] = max(
                _require_int(inventory["max_iid"], field="max_iid"), iid
            )
            last_created = created
        inventory["last_created_at"] = (
            _format_time(last_created) if last_created else None
        )
        inventory["pages"] = _require_int(inventory["pages"], field="pages") + 1
        inventory["window_pages"] = (
            _require_int(inventory["window_pages"], field="window_pages") + 1
        )
        inventory["window_seen"] = _require_int(
            inventory["window_seen"], field="window_seen"
        ) + len(page_items)
        if page_journal.get("has_next") is True:
            inventory["next_page"] = page + 1
            manifest.save()
            _discard_inventory_page_journal(manifest)
            continue
        window_total = inventory.get("window_total")
        if window_total is not None and inventory["window_seen"] != window_total:
            raise GitLabIngestError(
                f"{project.identity} window incomplete: "
                f"seen={inventory['window_seen']} total={window_total}"
            )
        completed_windows = inventory.get("completed_windows")
        if not isinstance(completed_windows, list):
            raise GitLabIngestError(
                f"{project.identity} completed-window journal is malformed"
            )
        completed_windows.append(
            {
                "start": _format_time(window_start),
                "end": _format_time(window_end),
                "pages": _require_int(inventory["window_pages"], field="window_pages"),
                "seen": _require_int(inventory["window_seen"], field="window_seen"),
                "reported_total": window_total,
            }
        )
        inventory["window_start"] = _format_time(window_end + timedelta(microseconds=1))
        inventory["window_end"] = None
        inventory["next_page"] = 1
        inventory["window_seen"] = 0
        inventory["window_total"] = None
        inventory["window_pages"] = 0
        inventory["last_created_at"] = None
        manifest.save()
        _discard_inventory_page_journal(manifest)


def _terminal_record(
    manifest: Manifest,
    project: GitLabProject,
    iid: int,
    metadata: dict,
    *,
    reason: str,
    lineage: list[dict[str, object]],
) -> dict:
    return {
        "schema": GITLAB_RECORD_SCHEMA,
        "scan_id": manifest.scan_id,
        "platform": GITLAB_PLATFORM,
        "project_identity": project.identity,
        "iid": iid,
        "route": "terminal",
        "terminal_reason": reason,
        "metadata": metadata,
        "lineage": lineage,
    }


def _normalize_comments(discussions: list[dict]) -> list[dict[str, object]]:
    comments: list[dict[str, object]] = []
    seen_note_ids: set[int] = set()
    for discussion in sorted(discussions, key=lambda item: str(item.get("id") or "")):
        notes = discussion.get("notes")
        if not isinstance(notes, list) or any(
            not isinstance(note, dict) for note in notes
        ):
            raise GitLabIngestError(
                "GitLab discussion notes must be an array of objects"
            )
        for note in sorted(
            notes,
            key=lambda item: (
                str(item.get("created_at") or ""),
                int(item.get("id") or 0),
            ),
        ):
            note_id = _require_int(
                note.get("id"), field="discussion note id", minimum=1
            )
            if note_id in seen_note_ids:
                raise GitLabIngestError(
                    f"duplicate GitLab discussion note id {note_id}"
                )
            seen_note_ids.add(note_id)
            if note.get("system") is True:
                continue
            author = note.get("author") or {}
            if not isinstance(author, dict):
                raise GitLabIngestError(
                    f"GitLab note {note_id} author must be an object"
                )
            position = note.get("position") or {}
            path = (
                position.get("new_path") or position.get("old_path")
                if isinstance(position, dict)
                else None
            )
            if path is not None and not isinstance(path, str):
                raise GitLabIngestError(
                    f"GitLab note {note_id} position path must be a string"
                )
            comments.append(
                {
                    "user": str(author.get("username") or author.get("name") or ""),
                    "body": str(note.get("body") or ""),
                    "path": path,
                    "created_at": str(note.get("created_at") or ""),
                    "kind": (
                        "review_comment"
                        if note.get("type") == "DiffNote"
                        else "comment"
                    ),
                }
            )
    comments.sort(
        key=lambda item: (
            str(item["created_at"]),
            str(item["user"]),
            str(item["body"]),
        )
    )
    return comments


def _normalize_linked_issues(issues: list[dict]) -> list[dict[str, object]]:
    linked: list[dict[str, object]] = []
    for item in sorted(
        issues,
        key=lambda issue: (
            int(issue.get("project_id") or 0),
            int(issue.get("iid") or 0),
        ),
    ):
        iid = item.get("iid")
        if isinstance(iid, int) and not isinstance(iid, bool) and iid > 0:
            linked.append(
                {
                    "number": iid,
                    "title": str(item.get("title") or ""),
                    "body": str(item.get("description") or ""),
                }
            )
    return linked


def _record_for_store(
    *,
    project: GitLabProject,
    detail: dict,
    comments: list[dict[str, object]],
    linked_issues: list[dict[str, object]],
    sidecar_path: Path,
    sidecar_sha256: str,
    scan_id: str,
    fetched_at: str,
) -> dict[str, object]:
    iid = _require_int(detail.get("iid"), field="detail.iid", minimum=1)
    author = detail.get("author") or {}
    if not isinstance(author, dict):
        raise GitLabIngestError(f"{project.identity}!{iid} author must be an object")
    return {
        "repo": project.identity,
        "pr_number": iid,
        "merge_commit_sha": _require_sha(
            detail.get("merge_commit_sha"),
            field=f"{project.identity}!{iid}.merge_commit_sha",
        ),
        "pr_title": str(detail.get("title") or ""),
        "pr_body": str(detail.get("description") or ""),
        "state": str(detail.get("state") or ""),
        "author": str(author.get("username") or author.get("name") or ""),
        "created_at": str(detail.get("created_at") or ""),
        "merged_at": str(detail.get("merged_at") or ""),
        "fetched_at": fetched_at,
        "scan_id": scan_id,
        "comments": comments,
        "reviews": [],
        "linked_issues": linked_issues,
        "raw": {
            "platform": GITLAB_PLATFORM,
            "project_identity": project.identity,
            "source_sha": detail["diff_refs"]["head_sha"],
            "target_sha": detail["diff_refs"]["start_sha"],
            "base_sha": detail["diff_refs"]["base_sha"],
            "merge_commit_sha": detail["merge_commit_sha"],
            "record_sidecar": {
                "path": str(sidecar_path.resolve()),
                "sha256": sidecar_sha256,
            },
        },
    }


def _validate_detail_shas(project: GitLabProject, iid: int, detail: dict) -> None:
    refs = detail.get("diff_refs")
    if not isinstance(refs, dict):
        raise GitLabIngestError(f"{project.identity}!{iid} lacks diff_refs")
    for field in ("base_sha", "head_sha", "start_sha"):
        refs[field] = _require_sha(
            refs.get(field), field=f"{project.identity}!{iid}.{field}"
        )
    detail["merge_commit_sha"] = _require_sha(
        detail.get("merge_commit_sha"),
        field=f"{project.identity}!{iid}.merge_commit_sha",
    )
    if (
        detail.get("sha") is not None
        and _require_sha(detail.get("sha"), field=f"{project.identity}!{iid}.sha")
        != refs["head_sha"]
    ):
        raise GitLabIngestError(f"{project.identity}!{iid} source head SHA drifted")
    if detail.get("squash_commit_sha") is not None:
        detail["squash_commit_sha"] = _require_sha(
            detail.get("squash_commit_sha"),
            field=f"{project.identity}!{iid}.squash_commit_sha",
        )


def _validate_detail_identity(
    project: GitLabProject,
    iid: int,
    detail: Mapping[str, object],
    metadata: Mapping[str, object],
) -> None:
    if detail.get("iid") != iid or detail.get("state") != "merged":
        raise GitLabIngestError(
            f"{project.identity}!{iid} detail identity/state drifted"
        )
    project_id = _require_int(
        detail.get("project_id"),
        field=f"{project.identity}!{iid}.detail_project_id",
        minimum=1,
    )
    if (
        project_id != metadata.get("project_id")
        or detail.get("id") != metadata.get("id")
        or detail.get("created_at") != metadata.get("created_at")
        or detail.get("target_project_id") not in (None, project_id)
    ):
        raise GitLabIngestError(
            f"{project.identity}!{iid} detail project/global identity drifted"
        )
    reviewers = detail.get("reviewers") or []
    if not isinstance(reviewers, list) or any(
        not isinstance(item, dict) for item in reviewers
    ):
        raise GitLabIngestError(f"{project.identity}!{iid} reviewers are malformed")


def _record_sidecar_max_bytes(manifest: Manifest) -> int:
    config = manifest.data.get("config")
    if not isinstance(config, dict):
        raise GitLabIngestError("GitLab manifest config is malformed")
    detail_mib = _require_int(
        config.get("max_detail_mib"), field="manifest.config.max_detail_mib", minimum=1
    )
    response_mib = _require_int(
        config.get("max_response_mib"),
        field="manifest.config.max_response_mib",
        minimum=1,
    )
    return (detail_mib + response_mib + 1) * 4 * 1024 * 1024


def _existing_record_sidecar(
    manifest: Manifest,
    project: GitLabProject,
    iid: int,
    *,
    sidecar_root: Path,
) -> tuple[dict, Path] | None:
    paths = [
        _sidecar_path(sidecar_root, f"records/{route}", project, iid)
        for route in ("primary", "ancillary", "excluded", "terminal")
    ]
    symlinked = [path for path in paths if path.is_symlink()]
    if symlinked:
        raise GitLabIngestError(
            f"routed sidecar paths are symlinked for {project.identity}!{iid}: {symlinked}"
        )
    existing = [path for path in paths if path.exists()]
    if len(existing) > 1:
        raise GitLabIngestError(
            f"multiple routed sidecars exist for {project.identity}!{iid}: {existing}"
        )
    if not existing:
        return None
    path = existing[0]
    record = _read_json_object(
        path,
        role="GitLab MR routed sidecar",
        max_bytes=_record_sidecar_max_bytes(manifest),
    )
    expected_route = path.parent.parent.name
    if (
        record.get("schema") != GITLAB_RECORD_SCHEMA
        or record.get("scan_id") != manifest.scan_id
        or record.get("platform") != GITLAB_PLATFORM
        or record.get("project_identity") != project.identity
        or record.get("iid") != iid
        or record.get("route") != expected_route
    ):
        raise GitLabIngestError(f"routed sidecar identity is malformed: {path}")
    return record, path


def _validate_routed_diffs(record: Mapping[str, object], *, route: str) -> None:
    routed = record.get("diffs")
    if not isinstance(routed, dict) or set(routed) != {
        "primary",
        "ancillary",
        "excluded",
    }:
        raise GitLabIngestError("GitLab routed sidecar has malformed diff groups")
    groups: dict[str, list[dict]] = {}
    for group in ("primary", "ancillary", "excluded"):
        value = routed[group]
        if not isinstance(value, list) or any(
            not isinstance(item, dict) for item in value
        ):
            raise GitLabIngestError(
                f"GitLab routed sidecar {group} diffs are malformed"
            )
        groups[group] = [dict(item) for item in value]
        for item in groups[group]:
            classified = classify_diff_paths([item])
            observed = next(key for key, rows in classified.items() if rows)
            if observed != group:
                raise GitLabIngestError(
                    f"GitLab routed sidecar diff moved from {group} to {observed}"
                )
    expected_route = (
        "primary"
        if groups["primary"]
        else "ancillary" if groups["ancillary"] else "excluded"
    )
    if route != expected_route:
        raise GitLabIngestError(
            f"GitLab routed sidecar route drifted: {route} != {expected_route}"
        )


def _materialize_record_sidecar(
    record: dict,
    record_path: Path,
    *,
    manifest: Manifest,
    project: GitLabProject,
    primary_conn: sqlite3.Connection,
    ancillary_conn: sqlite3.Connection,
) -> str:
    route = str(record.get("route") or "")
    if route == "terminal":
        return route
    if route not in {"primary", "ancillary", "excluded"}:
        raise GitLabIngestError(f"unsupported GitLab sidecar route: {route!r}")
    detail = record.get("merge_request")
    if not isinstance(detail, dict):
        raise GitLabIngestError(
            f"GitLab routed sidecar lacks merge_request: {record_path}"
        )
    iid = _require_int(record.get("iid"), field="record.iid", minimum=1)
    metadata = record.get("metadata")
    if not isinstance(metadata, dict):
        raise GitLabIngestError(
            f"GitLab routed sidecar metadata drifted: {record_path}"
        )
    _validate_detail_identity(project, iid, detail, metadata)
    _validate_detail_shas(project, iid, detail)
    shas = record.get("shas")
    expected_shas = {
        "source": detail["diff_refs"]["head_sha"],
        "target": detail["diff_refs"]["start_sha"],
        "base": detail["diff_refs"]["base_sha"],
        "merge": detail["merge_commit_sha"],
        "squash": detail.get("squash_commit_sha"),
    }
    if shas != expected_shas:
        raise GitLabIngestError(
            f"GitLab routed sidecar SHA binding drifted: {record_path}"
        )
    _validate_routed_diffs(record, route=route)
    fetched_at = record.get("fetched_at")
    _parse_time(fetched_at, field=f"{project.identity}!{iid}.fetched_at")
    if route == "excluded":
        return route
    discussions = record.get("discussions")
    linked_issues = record.get("linked_issues")
    if (
        not isinstance(discussions, list)
        or any(not isinstance(item, dict) for item in discussions)
        or not isinstance(linked_issues, list)
        or any(not isinstance(item, dict) for item in linked_issues)
    ):
        raise GitLabIngestError(
            f"GitLab routed sidecar children are malformed: {record_path}"
        )
    if route == "ancillary" and (discussions or linked_issues):
        raise GitLabIngestError(
            f"GitLab ancillary sidecar contains primary-only API data: {record_path}"
        )
    comments = _normalize_comments(discussions) if route == "primary" else []
    normalized_issues = (
        _normalize_linked_issues(linked_issues) if route == "primary" else []
    )
    store_record = _record_for_store(
        project=project,
        detail=detail,
        comments=comments,
        linked_issues=normalized_issues,
        sidecar_path=record_path,
        sidecar_sha256=_stable_file_sha256(
            record_path, role="GitLab MR routed sidecar"
        ),
        scan_id=manifest.scan_id,
        fetched_at=str(fetched_at),
    )
    conn = primary_conn if route == "primary" else ancillary_conn
    try:
        pr_store.upsert_record(
            conn,
            store_record,
            commit=False,
            replace_children=True,
            scan_id=manifest.scan_id,
        )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    return route


def _process_candidate(
    client: GitLabClient,
    manifest: Manifest,
    project: GitLabProject,
    metadata: dict,
    *,
    sidecar_root: Path,
    primary_conn: sqlite3.Connection,
    ancillary_conn: sqlite3.Connection,
    page_size: int,
    max_detail_pages: int,
    max_detail_bytes: int,
) -> str:
    iid = _require_int(metadata.get("iid"), field="metadata.iid", minimum=1)
    existing = _existing_record_sidecar(
        manifest,
        project,
        iid,
        sidecar_root=sidecar_root,
    )
    if existing is not None:
        record, record_path = existing
        if record.get("metadata") != metadata:
            raise GitLabIngestError(
                f"frozen inventory and routed sidecar disagree for {project.identity}!{iid}"
            )
        return _materialize_record_sidecar(
            record,
            record_path,
            manifest=manifest,
            project=project,
            primary_conn=primary_conn,
            ancillary_conn=ancillary_conn,
        )

    lineage: list[dict[str, object]] = []
    detail_response = client.get(
        f"{project.api_root}/merge_requests/{iid}",
        terminal_statuses=_TERMINAL_DETAIL_STATUSES,
    )
    lineage.append(detail_response.lineage("merge_request"))
    if detail_response.status in _TERMINAL_DETAIL_STATUSES:
        record = _terminal_record(
            manifest,
            project,
            iid,
            metadata,
            reason=f"merge_request_http_{detail_response.status}",
            lineage=lineage,
        )
        _write_bound_sidecar(
            _sidecar_path(sidecar_root, "records/terminal", project, iid),
            record,
            scan_id=manifest.scan_id,
        )
        return "terminal"
    if not isinstance(detail_response.body, dict):
        raise GitLabIngestError(
            f"{project.identity}!{iid} detail response must be an object"
        )
    detail = dict(detail_response.body)
    _validate_detail_identity(project, iid, detail, metadata)
    inventory_merge_sha = _require_sha(
        metadata.get("merge_commit_sha"),
        field=f"{project.identity}!{iid}.inventory_merge_commit_sha",
    )
    _validate_detail_shas(project, iid, detail)
    if detail["merge_commit_sha"] != inventory_merge_sha:
        raise GitLabIngestError(f"{project.identity}!{iid} merge_commit_sha drifted")

    diffs, diff_lineage, diff_total, diff_bytes = _paged_get(
        client,
        f"{project.api_root}/merge_requests/{iid}/diffs",
        endpoint="diffs",
        page_size=page_size,
        max_pages=max_detail_pages,
        max_total_bytes=max_detail_bytes,
        terminal_statuses=_TERMINAL_DETAIL_STATUSES,
    )
    lineage.extend(diff_lineage)
    terminal_status = next(
        (
            entry["status"]
            for entry in diff_lineage
            if entry["status"] in _TERMINAL_DETAIL_STATUSES
        ),
        None,
    )
    if terminal_status is not None:
        record = _terminal_record(
            manifest,
            project,
            iid,
            metadata,
            reason=f"diffs_http_{terminal_status}",
            lineage=lineage,
        )
        _write_bound_sidecar(
            _sidecar_path(sidecar_root, "records/terminal", project, iid),
            record,
            scan_id=manifest.scan_id,
        )
        return "terminal"
    changes_count = detail.get("changes_count")
    incomplete = (
        changes_count in (None, "")
        or str(changes_count).endswith("+")
        or not str(changes_count).isdigit()
        or int(str(changes_count)) != len(diffs)
        or (diff_total is not None and diff_total != len(diffs))
        or any(
            item.get("collapsed") is True or item.get("too_large") is True
            for item in diffs
        )
    )
    if incomplete:
        record = _terminal_record(
            manifest,
            project,
            iid,
            metadata,
            reason="diffs_incomplete_or_limited",
            lineage=lineage,
        )
        record["merge_request"] = detail
        record["diffs"] = diffs
        _write_bound_sidecar(
            _sidecar_path(sidecar_root, "records/terminal", project, iid),
            record,
            scan_id=manifest.scan_id,
        )
        return "terminal"

    routed = classify_diff_paths(diffs)
    route = (
        "primary"
        if routed["primary"]
        else "ancillary" if routed["ancillary"] else "excluded"
    )
    discussions: list[dict] = []
    linked_issues: list[dict] = []
    if route == "primary":
        discussions, discussion_lineage, _, discussion_bytes = _paged_get(
            client,
            f"{project.api_root}/merge_requests/{iid}/discussions",
            endpoint="discussions",
            page_size=page_size,
            max_pages=max_detail_pages,
            max_total_bytes=max_detail_bytes - diff_bytes,
            terminal_statuses=_TERMINAL_DETAIL_STATUSES,
        )
        lineage.extend(discussion_lineage)
        if any(
            item["status"] in _TERMINAL_DETAIL_STATUSES for item in discussion_lineage
        ):
            terminal = _terminal_record(
                manifest,
                project,
                iid,
                metadata,
                reason="primary_discussions_endpoint_terminal",
                lineage=lineage,
            )
            terminal["merge_request"] = detail
            terminal["diffs"] = diffs
            _write_bound_sidecar(
                _sidecar_path(sidecar_root, "records/terminal", project, iid),
                terminal,
                scan_id=manifest.scan_id,
            )
            return "terminal"
        linked_issues, issue_lineage, _, _issue_bytes = _paged_get(
            client,
            f"{project.api_root}/merge_requests/{iid}/closes_issues",
            endpoint="closes_issues",
            page_size=page_size,
            max_pages=max_detail_pages,
            max_total_bytes=max_detail_bytes - diff_bytes - discussion_bytes,
            terminal_statuses=_TERMINAL_DETAIL_STATUSES,
        )
        lineage.extend(issue_lineage)
        if any(item["status"] in _TERMINAL_DETAIL_STATUSES for item in issue_lineage):
            terminal = _terminal_record(
                manifest,
                project,
                iid,
                metadata,
                reason="primary_linked_issues_endpoint_terminal",
                lineage=lineage,
            )
            terminal["merge_request"] = detail
            terminal["diffs"] = diffs
            terminal["discussions"] = discussions
            _write_bound_sidecar(
                _sidecar_path(sidecar_root, "records/terminal", project, iid),
                terminal,
                scan_id=manifest.scan_id,
            )
            return "terminal"

    record = {
        "schema": GITLAB_RECORD_SCHEMA,
        "scan_id": manifest.scan_id,
        "platform": GITLAB_PLATFORM,
        "project_identity": project.identity,
        "host": project.host,
        "project_path": urllib.parse.unquote(project.encoded_path),
        "iid": iid,
        "route": route,
        "fetched_at": _utc_now(),
        "metadata": metadata,
        "merge_request": detail,
        "shas": {
            "source": detail["diff_refs"]["head_sha"],
            "target": detail["diff_refs"]["start_sha"],
            "base": detail["diff_refs"]["base_sha"],
            "merge": detail["merge_commit_sha"],
            "squash": detail.get("squash_commit_sha"),
        },
        "diffs": routed,
        "discussions": discussions,
        "reviewer_assignments": detail.get("reviewers") or [],
        "linked_issues": linked_issues,
        "lineage": lineage,
    }
    record_path = _sidecar_path(sidecar_root, f"records/{route}", project, iid)
    _write_bound_sidecar(record_path, record, scan_id=manifest.scan_id)
    return _materialize_record_sidecar(
        record,
        record_path,
        manifest=manifest,
        project=project,
        primary_conn=primary_conn,
        ancillary_conn=ancillary_conn,
    )


def _details_project(
    client: GitLabClient,
    manifest: Manifest,
    project: GitLabProject,
    *,
    sidecar_root: Path,
    primary_conn: sqlite3.Connection,
    ancillary_conn: sqlite3.Connection,
    page_size: int,
    max_detail_pages: int,
    max_detail_bytes: int,
) -> None:
    state = manifest.project(project.identity)
    if state.get("status") == "done":
        return
    if state.get("status") != "inventory_done" and state.get("status") != "details":
        raise GitLabIngestError(
            f"{project.identity} details started before inventory completion"
        )
    state["status"] = "details"
    details = state["details"]
    max_iid = _require_int(state["inventory"]["max_iid"], field="max_iid")
    iid = _require_int(details["next_iid"], field="next_iid", minimum=1)
    while iid <= max_iid:
        inventory_path = _sidecar_path(sidecar_root, "inventory", project, iid)
        if inventory_path.is_symlink():
            raise GitLabIngestError(
                f"GitLab MR inventory sidecar is symlinked: {inventory_path}"
            )
        if inventory_path.exists():
            inventory = _read_json_object(
                inventory_path,
                role="GitLab MR inventory sidecar",
                max_bytes=256 * 1024 * 1024,
            )
            if (
                inventory.get("schema") != GITLAB_INVENTORY_SCHEMA
                or inventory.get("scan_id") != manifest.scan_id
                or inventory.get("platform") != GITLAB_PLATFORM
                or inventory.get("project_identity") != project.identity
                or inventory.get("iid") != iid
            ):
                raise GitLabIngestError(
                    f"inventory sidecar identity is malformed: {inventory_path}"
                )
            metadata = inventory.get("metadata")
            if not isinstance(metadata, dict):
                raise GitLabIngestError(
                    f"inventory metadata is malformed: {inventory_path}"
                )
            if _candidate(metadata):
                route = _process_candidate(
                    client,
                    manifest,
                    project,
                    dict(metadata),
                    sidecar_root=sidecar_root,
                    primary_conn=primary_conn,
                    ancillary_conn=ancillary_conn,
                    page_size=page_size,
                    max_detail_pages=max_detail_pages,
                    max_detail_bytes=max_detail_bytes,
                )
                details["candidate_count"] = (
                    _require_int(details["candidate_count"], field="candidate_count")
                    + 1
                )
                key = f"{route}_count"
                details[key] = _require_int(details[key], field=key) + 1
        iid += 1
        details["next_iid"] = iid
        manifest.save()
    state["status"] = "done"
    manifest.save()


def _checkpoint_store(conn: sqlite3.Connection, path: Path) -> None:
    conn.commit()
    result = conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
    if result is None or int(result[0]) != 0:
        raise GitLabIngestError(f"SQLite WAL checkpoint failed for {path}: {result}")


def _hash_sidecars(
    manifest: Manifest,
    projects: tuple[GitLabProject, ...],
    sidecar_root: Path,
) -> tuple[str, int, int]:
    digest = hashlib.sha256()
    files = 0
    byte_size = 0
    for project in projects:
        state = manifest.project(project.identity)
        max_iid = _require_int(state["inventory"]["max_iid"], field="max_iid")
        for iid in range(1, max_iid + 1):
            paths = [_sidecar_path(sidecar_root, "inventory", project, iid)]
            paths.extend(
                _sidecar_path(sidecar_root, f"records/{route}", project, iid)
                for route in ("primary", "ancillary", "excluded", "terminal")
            )
            for path in paths:
                if path.is_symlink():
                    raise GitLabIngestError(f"GitLab MR sidecar is symlinked: {path}")
                if not path.exists():
                    continue
                if not path.is_file():
                    raise GitLabIngestError(
                        f"GitLab MR sidecar is not a regular file: {path}"
                    )
                relative = path.relative_to(sidecar_root).as_posix()
                sha = _stable_file_sha256(path, role="GitLab MR sidecar")
                size = path.stat().st_size
                encoded = f"{relative}\0{size}\0{sha}".encode("utf-8")
                digest.update(len(encoded).to_bytes(8, "big"))
                digest.update(encoded)
                files += 1
                byte_size += size
    return digest.hexdigest(), files, byte_size


def _store_counts(path: Path, scan_id: str) -> tuple[int, int, dict[str, int]]:
    conn = pr_store.connect(
        str(path),
        create=False,
        readonly=True,
    )
    try:
        total = int(conn.execute("SELECT COUNT(*) FROM prs").fetchone()[0])
        current = int(
            conn.execute(
                "SELECT COUNT(*) FROM prs WHERE scan_id=?", (scan_id,)
            ).fetchone()[0]
        )
        by_repo = {
            str(repo): int(count)
            for repo, count in conn.execute(
                "SELECT repo, COUNT(*) FROM prs WHERE scan_id=? GROUP BY repo ORDER BY repo",
                (scan_id,),
            )
        }
    finally:
        conn.close()
    return total, current, by_repo


def _update_membership_digest(digest: Any, repo: str, iid: int) -> None:
    encoded = f"{repo}\0{iid}".encode("utf-8")
    digest.update(len(encoded).to_bytes(8, "big"))
    digest.update(encoded)


def _sidecar_route_membership(
    manifest: Manifest,
    projects: tuple[GitLabProject, ...],
    sidecar_root: Path,
    route: str,
) -> tuple[int, str]:
    digest = hashlib.sha256()
    count = 0
    for project in projects:
        max_iid = _require_int(
            manifest.project(project.identity)["inventory"]["max_iid"],
            field="max_iid",
        )
        for iid in range(1, max_iid + 1):
            path = _sidecar_path(sidecar_root, f"records/{route}", project, iid)
            if path.is_symlink():
                raise GitLabIngestError(f"GitLab MR route sidecar is symlinked: {path}")
            if path.is_file():
                _update_membership_digest(digest, project.identity, iid)
                count += 1
    return count, digest.hexdigest()


def _store_membership(path: Path, scan_id: str) -> tuple[int, str]:
    digest = hashlib.sha256()
    count = 0
    conn = pr_store.connect(str(path), create=False, readonly=True)
    try:
        rows = conn.execute(
            "SELECT repo, pr_number FROM prs WHERE scan_id=? ORDER BY repo, pr_number",
            (scan_id,),
        )
        for repo, iid in rows:
            _update_membership_digest(digest, str(repo), int(iid))
            count += 1
    finally:
        conn.close()
    return count, digest.hexdigest()


def _validate_completed_inventory(manifest: Manifest, project: GitLabProject) -> None:
    inventory = manifest.project(project.identity).get("inventory")
    if not isinstance(inventory, dict):
        raise GitLabIngestError(f"{project.identity} inventory state is malformed")
    count = _require_int(inventory.get("count"), field="inventory.count")
    pages = _require_int(inventory.get("pages"), field="inventory.pages")
    max_iid = _require_int(inventory.get("max_iid"), field="inventory.max_iid")
    windows = inventory.get("completed_windows")
    if not isinstance(windows, list):
        raise GitLabIngestError(
            f"{project.identity} completed-window evidence is malformed"
        )
    if count == 0:
        if (
            inventory.get("empty_at_cutoff") is not True
            or pages != 0
            or max_iid != 0
            or windows
        ):
            raise GitLabIngestError(
                f"{project.identity} empty inventory lacks exact probe evidence"
            )
        return
    if not windows or max_iid < 1:
        raise GitLabIngestError(
            f"{project.identity} nonempty inventory lacks window evidence"
        )
    expected_start: datetime | None = None
    earliest = _parse_time(
        inventory.get("earliest_created_at"), field="inventory.earliest_created_at"
    )
    observed_pages = 0
    observed_items = 0
    for index, raw in enumerate(windows):
        if not isinstance(raw, dict) or set(raw) != {
            "start",
            "end",
            "pages",
            "seen",
            "reported_total",
        }:
            raise GitLabIngestError(
                f"{project.identity} completed window {index} is malformed"
            )
        start = _parse_time(raw["start"], field=f"completed_windows[{index}].start")
        end = _parse_time(raw["end"], field=f"completed_windows[{index}].end")
        if (
            end < start
            or (index == 0 and start != earliest)
            or (expected_start is not None and start != expected_start)
        ):
            raise GitLabIngestError(
                f"{project.identity} completed windows are not contiguous"
            )
        window_pages = _require_int(
            raw["pages"], field=f"completed_windows[{index}].pages", minimum=1
        )
        seen = _require_int(raw["seen"], field=f"completed_windows[{index}].seen")
        reported = raw["reported_total"]
        if (
            reported is not None
            and _require_int(
                reported, field=f"completed_windows[{index}].reported_total"
            )
            != seen
        ):
            raise GitLabIngestError(
                f"{project.identity} completed window total disagrees with seen rows"
            )
        observed_pages += window_pages
        observed_items += seen
        expected_start = end + timedelta(microseconds=1)
    scan_started = _parse_time(
        manifest.data["scan_started_at"], field="scan_started_at"
    )
    final_end = _parse_time(windows[-1]["end"], field="completed_windows[-1].end")
    if final_end != scan_started or observed_pages != pages or observed_items != count:
        raise GitLabIngestError(
            f"{project.identity} completed-window totals do not prove full inventory"
        )


def _build_completion_receipt(
    *,
    manifest: Manifest,
    projects: tuple[GitLabProject, ...],
    repo_list: Path,
    primary_store: Path,
    ancillary_store: Path,
    sidecar_root: Path,
) -> dict[str, object]:
    if any(
        manifest.project(project.identity).get("status") != "done"
        for project in projects
    ):
        raise GitLabIngestError(
            "cannot publish completion receipt with nonterminal projects"
        )
    for project in projects:
        _validate_completed_inventory(manifest, project)
    _require_checkpointed_store(primary_store)
    _require_checkpointed_store(ancillary_store)
    manifest_sha = _stable_file_sha256(manifest.path, role="GitLab MR manifest")
    repo_list_sha = _stable_file_sha256(repo_list, role="GitLab repo list")
    primary_sha = _stable_file_sha256(primary_store, role="GitLab primary PR store")
    ancillary_sha = _stable_file_sha256(
        ancillary_store, role="GitLab ancillary PR store"
    )
    primary_total, primary_current, primary_by_repo = _store_counts(
        primary_store, manifest.scan_id
    )
    ancillary_total, ancillary_current, ancillary_by_repo = _store_counts(
        ancillary_store, manifest.scan_id
    )
    expected_repos = [project.identity for project in projects]
    unexpected = sorted(
        (set(primary_by_repo) | set(ancillary_by_repo)) - set(expected_repos)
    )
    if unexpected:
        raise GitLabIngestError(
            f"GitLab stores contain out-of-scope current repos: {unexpected}"
        )
    declared_primary = sum(
        _require_int(
            manifest.project(project.identity)["details"]["primary_count"],
            field="primary_count",
        )
        for project in projects
    )
    declared_ancillary = sum(
        _require_int(
            manifest.project(project.identity)["details"]["ancillary_count"],
            field="ancillary_count",
        )
        for project in projects
    )
    if (primary_current, ancillary_current) != (declared_primary, declared_ancillary):
        raise GitLabIngestError(
            "GitLab store counts do not match routed manifest counts: "
            f"primary={primary_current}/{declared_primary} "
            f"ancillary={ancillary_current}/{declared_ancillary}"
        )
    for route, store_path in (
        ("primary", primary_store),
        ("ancillary", ancillary_store),
    ):
        sidecar_membership = _sidecar_route_membership(
            manifest,
            projects,
            sidecar_root,
            route,
        )
        store_membership = _store_membership(store_path, manifest.scan_id)
        if sidecar_membership != store_membership:
            raise GitLabIngestError(
                f"GitLab {route} store membership differs from routed sidecars"
            )
    sidecar_sha, sidecar_files, sidecar_bytes = _hash_sidecars(
        manifest, projects, sidecar_root
    )
    inventory_count = sum(
        _require_int(
            manifest.project(project.identity)["inventory"]["count"],
            field="inventory_count",
        )
        for project in projects
    )
    candidate_count = sum(
        _require_int(
            manifest.project(project.identity)["details"]["candidate_count"],
            field="candidate_count",
        )
        for project in projects
    )
    route_counts = {
        route: sum(
            _require_int(
                manifest.project(project.identity)["details"][f"{route}_count"],
                field=f"{route}_count",
            )
            for project in projects
        )
        for route in ("primary", "ancillary", "excluded", "terminal")
    }
    if sum(route_counts.values()) != candidate_count:
        raise GitLabIngestError("GitLab candidate route conservation failed")
    if candidate_count > inventory_count:
        raise GitLabIngestError("GitLab candidate count exceeds inventoried MRs")
    if sidecar_files != inventory_count + candidate_count:
        raise GitLabIngestError(
            "GitLab sidecar conservation failed: "
            f"files={sidecar_files} inventory={inventory_count} "
            f"candidates={candidate_count}"
        )
    return {
        "schema": GITLAB_COMPLETION_SCHEMA,
        "status": "verified",
        "platform": GITLAB_PLATFORM,
        "contract_sha256": GITLAB_CONTRACT_SHA256,
        "scan_id": manifest.scan_id,
        "scan_started_at": manifest.data["scan_started_at"],
        "repo_list": {"path": str(repo_list.resolve()), "sha256": repo_list_sha},
        "manifest": {"path": str(manifest.path.resolve()), "sha256": manifest_sha},
        "pr_store": {
            "path": str(primary_store.resolve()),
            "sha256": primary_sha,
            "size": primary_store.stat().st_size,
        },
        "ancillary_store": {
            "path": str(ancillary_store.resolve()),
            "sha256": ancillary_sha,
            "size": ancillary_store.stat().st_size,
        },
        "sidecars": {
            "root": str(sidecar_root.resolve()),
            "logical_set_sha256": sidecar_sha,
            "files": sidecar_files,
            "byte_size": sidecar_bytes,
        },
        "expected_repo_count": len(expected_repos),
        "expected_repos": expected_repos,
        "expected_repos_sha256": _canonical_sha256(expected_repos),
        "declared_mr_count": inventory_count,
        "candidate_mr_count": candidate_count,
        "noncandidate_mr_count": inventory_count - candidate_count,
        "stored_pr_count": primary_current,
        "unverified_store_pr_count": primary_total - primary_current,
        "ancillary_stored_count": ancillary_current,
        "ancillary_unverified_store_count": ancillary_total - ancillary_current,
        "route_counts": route_counts,
        "training_ready_without_membership": False,
        "required_training_gate": "exact_primary_pr_membership_receipt",
        "validation": {
            "inventory_complete": True,
            "candidate_route_conservation": True,
            "primary_ancillary_physical_separation": True,
            "store_counts_match_manifest": True,
            "immutable_artifact_hashes": True,
            "terminal_http_statuses_preserved": True,
            "exact_primary_membership_verified": False,
        },
    }


def _require_checkpointed_store(path: Path) -> None:
    wal = Path(f"{path}-wal")
    if wal.exists() and wal.stat().st_size:
        raise GitLabIngestError(f"GitLab completion store has a nonempty WAL: {wal}")


def load_gitlab_completion_binding(
    receipt_path: Path,
    *,
    pr_store: Path,
    repo_list: Path,
) -> dict[str, object]:
    """Validate one GitLab completion receipt for CASE5's existing exporter."""

    receipt_path = receipt_path.expanduser().resolve()
    pr_store = pr_store.expanduser().resolve()
    repo_list = repo_list.expanduser().resolve()
    receipt = _read_json_object(
        receipt_path, role="GitLab MR completion receipt", max_bytes=4 * 1024 * 1024
    )
    if (
        receipt.get("schema") != GITLAB_COMPLETION_SCHEMA
        or receipt.get("status") != "verified"
        or receipt.get("platform") != GITLAB_PLATFORM
        or receipt.get("contract_sha256") != GITLAB_CONTRACT_SHA256
    ):
        raise GitLabIngestError("GitLab MR completion receipt contract is unsupported")
    canonical = _canonical_bytes(receipt, pretty=True)
    if receipt_path.read_bytes() != canonical:
        raise GitLabIngestError("GitLab MR completion receipt is not canonical JSON")
    if (
        receipt.get("training_ready_without_membership") is not False
        or receipt.get("required_training_gate")
        != "exact_primary_pr_membership_receipt"
    ):
        raise GitLabIngestError(
            "GitLab MR completion must remain non-training-ready without membership"
        )
    validation = receipt.get("validation")
    if validation != {
        "inventory_complete": True,
        "candidate_route_conservation": True,
        "primary_ancillary_physical_separation": True,
        "store_counts_match_manifest": True,
        "immutable_artifact_hashes": True,
        "terminal_http_statuses_preserved": True,
        "exact_primary_membership_verified": False,
    }:
        raise GitLabIngestError("GitLab MR completion validation claims drifted")
    route_counts = receipt.get("route_counts")
    if not isinstance(route_counts, dict) or set(route_counts) != {
        "primary",
        "ancillary",
        "excluded",
        "terminal",
    }:
        raise GitLabIngestError("GitLab MR completion route counts are malformed")
    validated_routes = {
        route: _require_int(route_counts[route], field=f"route_counts.{route}")
        for route in sorted(route_counts)
    }
    candidate_count = _require_int(
        receipt.get("candidate_mr_count"), field="candidate_mr_count"
    )
    stored_count = _require_int(receipt.get("stored_pr_count"), field="stored_pr_count")
    ancillary_stored_count = _require_int(
        receipt.get("ancillary_stored_count"), field="ancillary_stored_count"
    )
    _require_int(
        receipt.get("ancillary_unverified_store_count"),
        field="ancillary_unverified_store_count",
    )
    if (
        sum(validated_routes.values()) != candidate_count
        or stored_count != validated_routes["primary"]
        or ancillary_stored_count != validated_routes["ancillary"]
    ):
        raise GitLabIngestError("GitLab MR completion route conservation drifted")
    declared_count = _require_int(
        receipt.get("declared_mr_count"), field="declared_mr_count"
    )
    noncandidate_count = _require_int(
        receipt.get("noncandidate_mr_count"), field="noncandidate_mr_count"
    )
    if declared_count != candidate_count + noncandidate_count:
        raise GitLabIngestError("GitLab MR completion inventory conservation drifted")
    _parse_time(receipt.get("scan_started_at"), field="scan_started_at")
    projects = load_gitlab_repos(repo_list)
    expected_repos = [project.identity for project in projects]
    if (
        receipt.get("expected_repos") != expected_repos
        or _require_int(
            receipt.get("expected_repo_count"),
            field="expected_repo_count",
            minimum=1,
        )
        != len(expected_repos)
        or receipt.get("expected_repos_sha256") != _canonical_sha256(expected_repos)
    ):
        raise GitLabIngestError("GitLab MR completion repo scope drifted")
    for field, expected_path in (("repo_list", repo_list), ("pr_store", pr_store)):
        artifact = receipt.get(field)
        if not isinstance(artifact, dict):
            raise GitLabIngestError(f"GitLab completion lacks {field} binding")
        if Path(str(artifact.get("path"))).expanduser().resolve() != expected_path:
            raise GitLabIngestError(f"GitLab completion {field} path mismatch")
        if field == "pr_store":
            _require_checkpointed_store(expected_path)
            if artifact.get("size") != expected_path.stat().st_size:
                raise GitLabIngestError("GitLab completion pr_store size mismatch")
        if artifact.get("sha256") != _stable_file_sha256(
            expected_path, role=f"GitLab completion {field}"
        ):
            raise GitLabIngestError(f"GitLab completion {field} hash mismatch")
    scan_id = receipt.get("scan_id")
    if not isinstance(scan_id, str) or re.fullmatch(r"[0-9a-f]{64}", scan_id) is None:
        raise GitLabIngestError("GitLab completion scan_id is invalid")
    total, current, by_repo = _store_counts(pr_store, scan_id)
    unverified = _require_int(
        receipt.get("unverified_store_pr_count"),
        field="unverified_store_pr_count",
    )
    if (
        current != stored_count
        or total - current != unverified
        or sorted(set(by_repo) - set(expected_repos))
    ):
        raise GitLabIngestError("GitLab completion store membership/count drifted")
    return {
        "schema": GITLAB_COMPLETION_SCHEMA,
        "status": "verified",
        "platform": GITLAB_PLATFORM,
        "receipt_sha256": _stable_file_sha256(
            receipt_path, role="GitLab MR completion receipt"
        ),
        "pr_store_sha256": str(receipt["pr_store"]["sha256"]),
        "repo_list_sha256": str(receipt["repo_list"]["sha256"]),
        "expected_repos_sha256": str(receipt["expected_repos_sha256"]),
        "scan_id": scan_id,
        "expected_repo_count": len(expected_repos),
        "stored_pr_count": current,
        "unverified_store_pr_count": total - current,
        "training_ready_without_membership": False,
    }


def verify_gitlab_completion_receipt(
    receipt_path: Path,
    *,
    pr_store: Path,
    repo_list: Path,
) -> dict[str, object]:
    """Rebuild the full ingest receipt from all immutable routed artifacts."""

    binding = load_gitlab_completion_binding(
        receipt_path,
        pr_store=pr_store,
        repo_list=repo_list,
    )
    receipt_path = receipt_path.expanduser().resolve()
    pr_store = pr_store.expanduser().resolve()
    repo_list = repo_list.expanduser().resolve()
    receipt = _read_json_object(
        receipt_path,
        role="GitLab MR completion receipt",
        max_bytes=4 * 1024 * 1024,
    )

    def bound_path(field: str, path_key: str) -> Path:
        descriptor = receipt.get(field)
        if not isinstance(descriptor, dict):
            raise GitLabIngestError(f"GitLab completion lacks {field} descriptor")
        raw = descriptor.get(path_key)
        if not isinstance(raw, str) or not raw:
            raise GitLabIngestError(
                f"GitLab completion {field}.{path_key} is malformed"
            )
        unexpanded = Path(raw).expanduser()
        if unexpanded.is_symlink():
            raise GitLabIngestError(
                f"GitLab completion {field}.{path_key} is symlinked: {unexpanded}"
            )
        return unexpanded.resolve()

    manifest_path = bound_path("manifest", "path")
    ancillary_store = bound_path("ancillary_store", "path")
    sidecar_root = bound_path("sidecars", "root")
    _require_checkpointed_store(ancillary_store)
    manifest_document = _read_json_object(manifest_path, role="GitLab MR manifest")
    config = manifest_document.get("config")
    if not isinstance(config, dict):
        raise GitLabIngestError("GitLab completion manifest config is malformed")
    projects = load_gitlab_repos(repo_list)
    manifest = Manifest(
        manifest_path,
        repo_list=repo_list,
        repo_list_sha256=_stable_file_sha256(repo_list, role="GitLab repo list"),
        projects=projects,
        config=config,
    )
    rebuilt = _build_completion_receipt(
        manifest=manifest,
        projects=projects,
        repo_list=repo_list,
        primary_store=pr_store,
        ancillary_store=ancillary_store,
        sidecar_root=sidecar_root,
    )
    if rebuilt != receipt:
        raise GitLabIngestError(
            "GitLab MR completion receipt differs from current routed artifacts"
        )
    return binding


def revalidate_gitlab_completion_binding(
    binding: Mapping[str, object],
    receipt_path: Path,
    *,
    pr_store: Path,
    repo_list: Path,
) -> None:
    current = load_gitlab_completion_binding(
        receipt_path,
        pr_store=pr_store,
        repo_list=repo_list,
    )
    if dict(binding) != current:
        raise GitLabIngestError("GitLab completion binding changed during CASE5 export")


def _parse_token_env(values: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        host, separator, env_name = value.partition("=")
        if (
            not separator
            or not host
            or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", env_name) is None
            or host in result
        ):
            raise GitLabIngestError(
                f"--token-env must be unique HOST=ENV_NAME values, got {value!r}"
            )
        result[host] = env_name
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    default_root = Path("outputs/pr_ingest/gitlab_mr")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-list", type=Path, default=Path("outputs/pr_ingest/repo_list.json")
    )
    parser.add_argument("--manifest", type=Path, default=default_root / "manifest.json")
    parser.add_argument(
        "--primary-store", type=Path, default=default_root / "primary.sqlite"
    )
    parser.add_argument(
        "--ancillary-store", type=Path, default=default_root / "ancillary.sqlite"
    )
    parser.add_argument("--sidecar-root", type=Path, default=default_root / "sidecars")
    parser.add_argument(
        "--completion-receipt",
        type=Path,
        default=default_root / "completion_receipt.json",
    )
    parser.add_argument("--window-days", type=int, default=90)
    parser.add_argument("--page-size", type=int, default=100)
    parser.add_argument("--max-window-pages", type=int, default=100)
    parser.add_argument("--max-response-mib", type=int, default=64)
    parser.add_argument("--max-detail-pages", type=int, default=100)
    parser.add_argument("--max-detail-mib", type=int, default=256)
    parser.add_argument("--max-retries", type=int, default=8)
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    parser.add_argument(
        "--token-env",
        action="append",
        default=[],
        metavar="HOST=ENV_NAME",
        help="Required host-specific PRIVATE-TOKEN environment variable; repeat per host.",
    )
    return parser


def run(args: argparse.Namespace) -> dict[str, object]:
    for field in (
        "window_days",
        "page_size",
        "max_window_pages",
        "max_response_mib",
        "max_detail_pages",
        "max_detail_mib",
    ):
        if int(getattr(args, field)) < 1:
            raise GitLabIngestError(f"--{field.replace('_', '-')} must be positive")
    if args.page_size > 100:
        raise GitLabIngestError("GitLab REST per_page maximum is 100")
    timeout_seconds = float(args.timeout_seconds)
    if int(args.max_retries) < 0 or not 0 < timeout_seconds < float("inf"):
        raise GitLabIngestError("--max-retries and --timeout-seconds are out of range")
    repo_list = args.repo_list.expanduser().resolve()
    projects = load_gitlab_repos(repo_list)
    repo_list_sha = _stable_file_sha256(repo_list, role="GitLab repo list")
    manifest_path = args.manifest.expanduser().resolve()
    primary_store = args.primary_store.expanduser().resolve()
    ancillary_store = args.ancillary_store.expanduser().resolve()
    sidecar_root = args.sidecar_root.expanduser().resolve()
    completion_path = args.completion_receipt.expanduser().resolve()
    page_journal = manifest_path.with_name(f"{manifest_path.name}.inventory-page.json")
    lock_path = Path(f"{manifest_path}.lock")
    paths = [
        manifest_path,
        page_journal,
        lock_path,
        primary_store,
        ancillary_store,
        sidecar_root,
        completion_path,
    ]
    if len({str(path) for path in paths}) != len(paths):
        raise GitLabIngestError(
            "manifest, stores, sidecars, and receipt paths must be distinct"
        )
    for artifact in paths:
        if artifact != sidecar_root and artifact.is_relative_to(sidecar_root):
            raise GitLabIngestError(
                f"non-sidecar artifact is nested under --sidecar-root: {artifact}"
            )
    if completion_path.exists():
        binding = verify_gitlab_completion_receipt(
            completion_path,
            pr_store=primary_store,
            repo_list=repo_list,
        )
        return {"status": "already_complete", "binding": binding}

    config = {
        "primary_store": str(primary_store),
        "ancillary_store": str(ancillary_store),
        "sidecar_root": str(sidecar_root),
        "completion_receipt": str(completion_path),
        "window_days": int(args.window_days),
        "page_size": int(args.page_size),
        "max_window_pages": int(args.max_window_pages),
        "max_response_mib": int(args.max_response_mib),
        "max_detail_pages": int(args.max_detail_pages),
        "max_detail_mib": int(args.max_detail_mib),
    }
    token_env = _parse_token_env(args.token_env)
    expected_token_hosts = {project.host for project in projects}
    if set(token_env) != expected_token_hosts:
        raise GitLabIngestError(
            "--token-env must cover every canonical GitLab host exactly: "
            f"missing={sorted(expected_token_hosts - set(token_env))} "
            f"extra={sorted(set(token_env) - expected_token_hosts)}"
        )
    client = GitLabClient(
        allowed_hosts=(project.host for project in projects),
        token_env_by_host=token_env,
        max_response_bytes=int(args.max_response_mib) * 1024 * 1024,
        max_retries=int(args.max_retries),
        timeout_s=timeout_seconds,
    )
    for project in projects:
        _preflight_project_access(client, project)
    with RunLock(manifest_path):
        manifest = Manifest(
            manifest_path,
            repo_list=repo_list,
            repo_list_sha256=repo_list_sha,
            projects=projects,
            config=config,
        )
        primary_conn = pr_store.connect(str(primary_store), create=True)
        ancillary_conn = pr_store.connect(str(ancillary_store), create=True)
        try:
            for project in projects:
                _inventory_project(
                    client,
                    manifest,
                    project,
                    sidecar_root=sidecar_root,
                    window_days=int(args.window_days),
                    page_size=int(args.page_size),
                    max_window_pages=int(args.max_window_pages),
                )
            for project in projects:
                _details_project(
                    client,
                    manifest,
                    project,
                    sidecar_root=sidecar_root,
                    primary_conn=primary_conn,
                    ancillary_conn=ancillary_conn,
                    page_size=int(args.page_size),
                    max_detail_pages=int(args.max_detail_pages),
                    max_detail_bytes=int(args.max_detail_mib) * 1024 * 1024,
                )
            _checkpoint_store(primary_conn, primary_store)
            _checkpoint_store(ancillary_conn, ancillary_store)
        finally:
            primary_conn.close()
            ancillary_conn.close()
        receipt = _build_completion_receipt(
            manifest=manifest,
            projects=projects,
            repo_list=repo_list,
            primary_store=primary_store,
            ancillary_store=ancillary_store,
            sidecar_root=sidecar_root,
        )
        _atomic_write_json(completion_path, receipt)
        verify_gitlab_completion_receipt(
            completion_path,
            pr_store=primary_store,
            repo_list=repo_list,
        )
        return receipt


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        result = run(args)
    except GitLabIngestError as exc:
        raise SystemExit(f"GITLAB_MR_INGEST_FAILED: {exc}") from exc
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
