"""Fail-closed inventory contract for portable cppmega sidecar snapshots."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
from typing import Any


MANIFEST_SCHEMA = "cppmega_verified_sidecar_manifest_v2"
AUDIT_SCHEMA = "cppmega_sidecar_audit_v2"
AUDIT_REMOTE = "audits/sidecar_audit_all_final_poststop_valid"
AUDIT_FILENAME = "sidecar_parquet_audit.json"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
REPO_ROOT = Path(__file__).resolve().parents[1]

TRAINING_BUCKETS: dict[str, tuple[int, ...]] = {
    "code": (1024, 2048, 4096, 8192, 16384),
    "commits": (1024, 2048, 4096, 8192, 16384),
    "pr": (1024, 2048, 4096, 8192, 16384),
}
OBJECTIVE_CONTRACT = {
    "schema": "cppmega_mlx_runtime_objectives_v1",
    "status": "runtime_materialized",
    "materialized_in_parquet": False,
    "runtime": "cppmega_mlx.training.task_mixer.TaskMixer",
    "stage": "stage1",
    "rates": {
        "causal_lm": "1/2",
        "ast_fim": "1/10",
        "ifim": "1/10",
        "commit_diff": "1/10",
        "pre_to_post": "1/10",
        "symbol_recovery": "1/30",
        "type_recovery": "1/30",
        "callee_recovery": "1/30",
    },
}
GRAPH_CONTRACT = {
    "schema": "cppmega_graph_sidecars_v1",
    "status": "required",
    "relations": [
        "call",
        "type",
        "domain",
        "build",
        "shell",
        "diagnostic",
        "cross_domain",
    ],
    "chunk_columns": [
        "token_chunk_starts",
        "token_chunk_ends",
        "token_chunk_kinds",
        "token_chunk_dep_levels",
    ],
}
AUDIT_GRAPH_FIELDS = (
    "token_call_edges",
    "token_type_edges",
    "token_domain_edges",
    "token_build_edges",
    "token_shell_edges",
    "token_diagnostic_edges",
    "token_cross_domain_edges",
)


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def audit_contract() -> dict[str, object]:
    return {
        "schema": "cppmega_sidecar_audit_contract_v2",
        "tokenizer_contract_sha256": sha256_file(
            REPO_ROOT / "cppmega_mlx" / "tokenizer" / "tokenizer_contract_v1.json"
        ),
        "domain_schema_sha256": sha256_file(
            REPO_ROOT / "cppmega_mlx" / "data" / "domain_schema_v1.json"
        ),
        "graph_fields": sorted(AUDIT_GRAPH_FIELDS),
        "chunk_fields": list(GRAPH_CONTRACT["chunk_columns"]),
        "objective_contract": json.loads(json.dumps(OBJECTIVE_CONTRACT)),
    }


def safe_relative_path(value: object, *, where: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{where} must be a string")
    path = PurePosixPath(value)
    if (
        not value
        or "\\" in value
        or path.is_absolute()
        or path.as_posix() != value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError(f"unsafe {where}: {value!r}")
    return value


def selection_key(remote: object) -> str | None:
    value = safe_relative_path(remote, where="selection remote")
    parts = value.split("/")
    if value == AUDIT_REMOTE:
        return None
    if len(parts) != 3 or parts[0] != "parquet":
        raise ValueError(f"unsupported selection remote: {value!r}")
    kind, bucket_text = parts[1], parts[2]
    if kind not in TRAINING_BUCKETS or not bucket_text.isdigit():
        raise ValueError(f"unsupported parquet selection: {value!r}")
    bucket = int(bucket_text)
    if bucket not in TRAINING_BUCKETS[kind]:
        raise ValueError(f"unknown parquet bucket selection: {value!r}")
    return f"{kind}/{bucket}"


def contained_path(root: Path, relative: object, *, where: str) -> Path:
    safe = safe_relative_path(relative, where=where)
    root = root.resolve()
    candidate = root.joinpath(*PurePosixPath(safe).parts)
    resolved_parent = candidate.parent.resolve()
    if resolved_parent != root and root not in resolved_parent.parents:
        raise ValueError(f"{where} escapes root: {safe!r}")
    return candidate


def _canonical_files(files: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    return [
        {
            "path": str(record["path"]),
            "size": int(record["size"]),
            "sha256": str(record["sha256"]),
        }
        for record in sorted(files, key=lambda item: str(item["path"]))
    ]


def inventory_directory(root: Path, *, remote: str) -> dict[str, object]:
    selection_key(remote)
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"selection source must be a regular directory: {root}")
    records: list[dict[str, object]] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"selection source contains a symlink: {path}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError(f"selection source contains a non-file entry: {path}")
        relative = path.relative_to(root).as_posix()
        safe_relative_path(relative, where="inventory file path")
        if remote.startswith("parquet/") and path.suffix != ".parquet":
            raise ValueError(f"parquet selection contains a non-parquet file: {path}")
        records.append(
            {
                "path": relative,
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    if not records:
        raise ValueError(f"selection source is empty: {root}")
    if remote == AUDIT_REMOTE and AUDIT_FILENAME not in {
        str(record["path"]) for record in records
    }:
        raise ValueError(f"audit selection lacks {AUDIT_FILENAME}: {root}")
    canonical = _canonical_files(records)
    return {
        "files": canonical,
        "file_count": len(canonical),
        "byte_count": sum(int(record["size"]) for record in canonical),
        "artifact_set_sha256": canonical_sha256(canonical),
    }


def validate_audit_receipt(
    receipt: object,
    *,
    selected_keys: Sequence[str],
) -> dict[str, Any]:
    if not isinstance(receipt, dict):
        raise ValueError("sidecar audit receipt must be an object")
    if receipt.get("schema") != AUDIT_SCHEMA or receipt.get("status") != "verified":
        raise ValueError("sidecar audit receipt is not schema-bound and verified")
    if receipt.get("contract") != audit_contract():
        raise ValueError("sidecar audit receipt contract does not match this checkout")
    total = receipt.get("total")
    by_kind_bucket = receipt.get("by_kind_bucket")
    if not isinstance(total, dict) or not isinstance(by_kind_bucket, dict):
        raise ValueError("sidecar audit receipt lacks totals or bucket accounting")
    if int(total.get("bad_files", -1)) != 0 or int(total.get("bad_rows", -1)) != 0:
        raise ValueError("sidecar audit receipt is not green")
    missing = [key for key in selected_keys if key not in by_kind_bucket]
    if missing:
        raise ValueError(f"sidecar audit receipt misses selected buckets: {missing}")
    for key in selected_keys:
        row = by_kind_bucket[key]
        if not isinstance(row, dict):
            raise ValueError(f"sidecar audit bucket is not an object: {key}")
        if int(row.get("bad_files", -1)) or int(row.get("bad_rows", -1)):
            raise ValueError(f"sidecar audit bucket is not green: {key}")
        if int(row.get("files", 0)) <= 0 or int(row.get("valid_tokens", 0)) <= 0:
            raise ValueError(f"sidecar audit bucket has no verified data: {key}")
    return receipt


def expected_bucket_remotes() -> set[str]:
    return {
        f"parquet/{kind}/{bucket}"
        for kind, buckets in TRAINING_BUCKETS.items()
        for bucket in buckets
    }


def selection_policy(
    selected_remotes: Sequence[str],
    *,
    include_standalone_pr: bool,
) -> dict[str, object]:
    selected = set(selected_remotes)
    expected = expected_bucket_remotes()
    unknown = sorted(selected - expected - {AUDIT_REMOTE})
    if unknown:
        raise ValueError(f"selection policy contains unknown remotes: {unknown}")
    excluded: list[dict[str, str]] = []
    for remote in sorted(expected - selected):
        if remote == "parquet/code/16384":
            reason = "no verified fixed-row code/16384 source bucket"
        elif remote.startswith("parquet/pr/") and not include_standalone_pr:
            reason = "standalone PR diagnostics excluded from the training profile"
        else:
            raise ValueError(f"selected snapshot leaves bucket unaccounted: {remote}")
        excluded.append({"remote": remote, "reason": reason})
    return {
        "schema": "cppmega_sidecar_selection_policy_v1",
        "expected_bucket_remotes": sorted(expected),
        "selected": sorted(selected - {AUDIT_REMOTE}),
        "excluded": excluded,
    }


def inventory_sha256(selections: Sequence[Mapping[str, object]]) -> str:
    payload = [
        {
            "remote": str(selection["remote"]),
            "artifact_set_sha256": str(selection["artifact_set_sha256"]),
            "file_count": int(selection["file_count"]),
            "byte_count": int(selection["byte_count"]),
        }
        for selection in sorted(selections, key=lambda item: str(item["remote"]))
    ]
    return canonical_sha256(payload)


def finalize_manifest(payload: Mapping[str, object]) -> dict[str, object]:
    manifest = dict(payload)
    manifest.pop("manifest_payload_sha256", None)
    manifest["manifest_payload_sha256"] = canonical_sha256(manifest)
    return manifest


def _validate_selection(selection: object) -> dict[str, object]:
    if not isinstance(selection, dict):
        raise ValueError("manifest selection must be an object")
    remote = safe_relative_path(selection.get("remote"), where="selection remote")
    key = selection_key(remote)
    local = safe_relative_path(selection.get("local"), where="selection local")
    files = selection.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError(f"manifest selection has no inventory: {remote}")
    canonical: list[dict[str, object]] = []
    seen: set[str] = set()
    for record in files:
        if not isinstance(record, dict):
            raise ValueError(f"manifest inventory record must be an object: {remote}")
        path = safe_relative_path(record.get("path"), where="inventory file path")
        if path in seen:
            raise ValueError(f"duplicate inventory file path: {remote}/{path}")
        seen.add(path)
        size = record.get("size")
        digest = record.get("sha256")
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise ValueError(f"invalid inventory size: {remote}/{path}")
        if not isinstance(digest, str) or SHA256_RE.fullmatch(digest) is None:
            raise ValueError(f"invalid inventory SHA-256: {remote}/{path}")
        if key is not None and not path.endswith(".parquet"):
            raise ValueError(f"parquet inventory contains non-parquet file: {remote}/{path}")
        canonical.append({"path": path, "size": size, "sha256": digest})
    canonical = _canonical_files(canonical)
    if int(selection.get("file_count", -1)) != len(canonical):
        raise ValueError(f"manifest file_count mismatch: {remote}")
    if int(selection.get("byte_count", -1)) != sum(
        int(record["size"]) for record in canonical
    ):
        raise ValueError(f"manifest byte_count mismatch: {remote}")
    if selection.get("artifact_set_sha256") != canonical_sha256(canonical):
        raise ValueError(f"manifest artifact set mismatch: {remote}")
    return {
        "remote": remote,
        "local": local,
        "files": canonical,
        "file_count": len(canonical),
        "byte_count": sum(int(record["size"]) for record in canonical),
        "artifact_set_sha256": canonical_sha256(canonical),
    }


def validate_manifest(
    value: object,
    *,
    expected_bucket: str | None = None,
    expected_prefix: str | None = None,
    expected_endpoint_url: str | None = None,
) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError("sidecar manifest must be an object")
    manifest = dict(value)
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise ValueError(f"unsupported sidecar manifest schema: {manifest.get('schema')!r}")
    for field, expected in (
        ("bucket", expected_bucket),
        ("prefix", expected_prefix),
        ("endpoint_url", expected_endpoint_url),
    ):
        actual = manifest.get(field)
        if not isinstance(actual, str) or not actual:
            raise ValueError(f"manifest {field} must be a nonempty string")
        if expected is not None and actual != expected:
            raise ValueError(f"manifest {field} mismatch: {actual!r} != {expected!r}")
    selections_raw = manifest.get("selections")
    if not isinstance(selections_raw, list) or not selections_raw:
        raise ValueError("manifest selections must be a nonempty list")
    selections = [_validate_selection(selection) for selection in selections_raw]
    remotes = [str(selection["remote"]) for selection in selections]
    locals_ = [str(selection["local"]) for selection in selections]
    if len(set(remotes)) != len(remotes) or len(set(locals_)) != len(locals_):
        raise ValueError("manifest selections contain duplicate remote/local paths")
    if AUDIT_REMOTE not in remotes:
        raise ValueError("manifest does not include the audit receipt selection")
    if manifest.get("inventory_sha256") != inventory_sha256(selections):
        raise ValueError("manifest inventory_sha256 mismatch")
    if manifest.get("objective_contract") != OBJECTIVE_CONTRACT:
        raise ValueError("manifest objective contract mismatch")
    if manifest.get("graph_contract") != GRAPH_CONTRACT:
        raise ValueError("manifest graph contract mismatch")

    include_pr = manifest.get("standalone_pr_included")
    if not isinstance(include_pr, bool):
        raise ValueError("manifest standalone_pr_included must be boolean")
    expected_profile = (
        "code_commits_plus_standalone_pr"
        if include_pr
        else "code_commits_integrated_pr"
    )
    if manifest.get("profile") != expected_profile:
        raise ValueError("manifest profile does not match standalone PR policy")
    expected_policy = selection_policy(remotes, include_standalone_pr=include_pr)
    if manifest.get("selection_policy") != expected_policy:
        raise ValueError("manifest selection policy mismatch")
    token_total = manifest.get("verified_valid_tokens")
    if not isinstance(token_total, int) or isinstance(token_total, bool) or token_total <= 0:
        raise ValueError("manifest verified_valid_tokens must be positive")
    audit = manifest.get("audit_receipt")
    if not isinstance(audit, dict):
        raise ValueError("manifest audit_receipt binding is missing")
    audit_sha = audit.get("sha256")
    if (
        audit.get("schema") != AUDIT_SCHEMA
        or audit.get("status") != "verified"
        or audit.get("remote") != AUDIT_REMOTE
        or audit.get("path") != AUDIT_FILENAME
        or not isinstance(audit_sha, str)
        or SHA256_RE.fullmatch(audit_sha) is None
    ):
        raise ValueError("manifest audit_receipt binding is invalid")
    audit_selection = next(
        selection for selection in selections if selection["remote"] == AUDIT_REMOTE
    )
    audit_inventory = [
        record
        for record in audit_selection["files"]
        if record["path"] == AUDIT_FILENAME
    ]
    if len(audit_inventory) != 1 or audit_inventory[0]["sha256"] != audit_sha:
        raise ValueError("manifest audit receipt does not match its inventory binding")

    expected_payload_sha = manifest.get("manifest_payload_sha256")
    unsigned = dict(manifest)
    unsigned.pop("manifest_payload_sha256", None)
    if expected_payload_sha != canonical_sha256(unsigned):
        raise ValueError("manifest payload SHA-256 mismatch")
    manifest["selections"] = selections
    return manifest


def verify_inventory(root: Path, manifest: Mapping[str, object]) -> dict[str, object]:
    selections = manifest.get("selections")
    if not isinstance(selections, list):
        raise ValueError("validated manifest selections are missing")
    verified_files = 0
    verified_bytes = 0
    for raw_selection in selections:
        selection = _validate_selection(raw_selection)
        local_root = contained_path(
            root, selection["local"], where="download selection local"
        )
        if local_root.is_symlink() or not local_root.is_dir():
            raise ValueError(f"download selection is not a regular directory: {local_root}")
        expected = {str(record["path"]): record for record in selection["files"]}
        actual: dict[str, Path] = {}
        for path in sorted(local_root.rglob("*")):
            if path.is_symlink():
                raise ValueError(f"download contains a symlink: {path}")
            if path.is_dir():
                continue
            if not path.is_file():
                raise ValueError(f"download contains a non-file entry: {path}")
            relative = path.relative_to(local_root).as_posix()
            safe_relative_path(relative, where="download inventory file")
            actual[relative] = path
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        if missing or extra:
            raise ValueError(
                f"download inventory mismatch for {selection['remote']}: "
                f"missing={missing[:5]} extra={extra[:5]}"
            )
        for relative, record in expected.items():
            path = actual[relative]
            if path.stat().st_size != int(record["size"]):
                raise ValueError(f"download size mismatch: {selection['remote']}/{relative}")
            if sha256_file(path) != record["sha256"]:
                raise ValueError(f"download SHA-256 mismatch: {selection['remote']}/{relative}")
            verified_files += 1
            verified_bytes += path.stat().st_size
    return {
        "status": "verified",
        "inventory_sha256": manifest["inventory_sha256"],
        "files": verified_files,
        "bytes": verified_bytes,
    }


def resolve_s3_env(source: Mapping[str, str]) -> dict[str, str]:
    env = dict(source)
    nebius_access = "NEBIUS_S3_ACCESS_KEY_ID"
    nebius_secret = "NEBIUS_S3_SECRET_ACCESS_KEY"
    if nebius_access in env or nebius_secret in env:
        access = env.get(nebius_access)
        secret = env.get(nebius_secret)
        if not access or not secret:
            raise SystemExit(
                "a complete Nebius S3 credential pair is required when either "
                "NEBIUS_S3 credential is set"
            )
        env["AWS_ACCESS_KEY_ID"] = access
        env["AWS_SECRET_ACCESS_KEY"] = secret
        env.pop("AWS_SESSION_TOKEN", None)
        env.pop("AWS_SECURITY_TOKEN", None)
        return env
    if not env.get("AWS_ACCESS_KEY_ID") or not env.get("AWS_SECRET_ACCESS_KEY"):
        raise SystemExit(
            "missing S3 credentials: export both NEBIUS_S3_ACCESS_KEY_ID and "
            "NEBIUS_S3_SECRET_ACCESS_KEY (or a complete AWS_ACCESS_KEY_ID/"
            "AWS_SECRET_ACCESS_KEY pair)"
        )
    return env


def write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


__all__ = [
    "AUDIT_FILENAME",
    "AUDIT_REMOTE",
    "AUDIT_SCHEMA",
    "AUDIT_GRAPH_FIELDS",
    "GRAPH_CONTRACT",
    "MANIFEST_SCHEMA",
    "OBJECTIVE_CONTRACT",
    "canonical_sha256",
    "audit_contract",
    "contained_path",
    "finalize_manifest",
    "inventory_directory",
    "inventory_sha256",
    "resolve_s3_env",
    "safe_relative_path",
    "selection_key",
    "selection_policy",
    "sha256_file",
    "validate_audit_receipt",
    "validate_manifest",
    "verify_inventory",
    "write_json_atomic",
]
