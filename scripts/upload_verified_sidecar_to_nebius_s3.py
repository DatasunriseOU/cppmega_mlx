#!/usr/bin/env python3
"""Upload verified cppmega sidecar parquet shards to Nebius Object Storage.

Default policy is the training contract: upload C/C++ code and commit parquet
only.  PR/discussion text is expected to be integrated into commit documents as
the HEAD docstring before PRE/POST/diff.  Standalone PR parquet is diagnostic
material and is uploaded only with explicit ``--include-standalone-pr``.

Nebius Object Storage exposes an S3-compatible object API.  The current Nebius
CLI manages buckets/transfers, but not individual object syncs, so this script
uses an S3 CLI client against the Nebius endpoint.  Prefer exporting
NEBIUS_S3_ACCESS_KEY_ID and NEBIUS_S3_SECRET_ACCESS_KEY; they are mapped to the
standard S3 client environment only for the subprocess.
"""

from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.sidecar_manifest_contract import (  # noqa: E402
    AUDIT_FILENAME,
    AUDIT_REMOTE,
    AUDIT_SCHEMA,
    GRAPH_CONTRACT,
    MANIFEST_SCHEMA,
    OBJECTIVE_CONTRACT,
    build_semantic_audit_binding,
    contained_path,
    finalize_manifest,
    inventory_directory,
    inventory_sha256,
    resolve_s3_env,
    selection_policy,
    sha256_file,
    validate_audit_receipt,
    validate_semantic_audit_receipt_binding,
    write_json_atomic,
)


DEFAULT_ENDPOINT = "https://storage.eu-north1.nebius.cloud"
DEFAULT_PREFIX = "cppmega-sidecar/code-commits-integrated-pr-20260627"

CODE_COMMIT_SELECTIONS = (
    ("parquet/code/1024", Path("outputs/reindexed/1024")),
    ("parquet/code/2048", Path("outputs/reindexed/2048")),
    ("parquet/code/4096", Path("outputs/reindexed/4096")),
    ("parquet/code/8192", Path("outputs/reindexed/8192")),
    ("parquet/commits/1024", Path("outputs/reindexed_commits/1024")),
    ("parquet/commits/2048", Path("outputs/reindexed_commits/2048")),
    ("parquet/commits/4096", Path("outputs/reindexed_commits/4096")),
    ("parquet/commits/8192", Path("outputs/reindexed_commits/8192")),
    ("parquet/commits/16384", Path("outputs/reindexed_commits/16384")),
    (
        "audits/sidecar_audit_all_final_poststop_valid",
        Path("outputs/sidecar_audit_all_final_poststop_valid"),
    ),
)

STANDALONE_PR_SELECTIONS = (
    ("parquet/pr/1024", Path("outputs/reindexed_pr/1024")),
    ("parquet/pr/2048", Path("outputs/reindexed_pr/2048")),
    ("parquet/pr/4096", Path("outputs/reindexed_pr/4096")),
    ("parquet/pr/8192", Path("outputs/reindexed_pr/8192")),
    ("parquet/pr/16384", Path("outputs/reindexed_pr/16384")),
)

# Backward-compatible names for older tests/imports.  "all valid" now means all
# training-valid buckets, not standalone PR diagnostics.
ALL_VALID_SELECTIONS = CODE_COMMIT_SELECTIONS
CODE_COMMITS_ONLY_SELECTIONS = CODE_COMMIT_SELECTIONS

FINAL_AUDIT_RECEIPTS = (
    Path("outputs/sidecar_audit_all_final_poststop_valid/sidecar_parquet_audit.json"),
)


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _s3_env() -> dict[str, str]:
    return resolve_s3_env(os.environ)


def _selection_items(
    *,
    include_standalone_pr: bool = False,
    code_commits_only: bool = False,
) -> tuple[tuple[str, Path], ...]:
    if include_standalone_pr and code_commits_only:
        raise SystemExit(
            "--include-standalone-pr conflicts with deprecated --code-commits-only"
        )
    if include_standalone_pr:
        return CODE_COMMIT_SELECTIONS + STANDALONE_PR_SELECTIONS
    return CODE_COMMIT_SELECTIONS


def _audit_receipts() -> tuple[Path, ...]:
    return FINAL_AUDIT_RECEIPTS


def _selected_bucket_keys(selections: tuple[tuple[str, Path], ...]) -> list[str]:
    """Map each ``parquet/<kind>/<bucket>`` selection to a receipt
    ``by_kind_bucket`` key ``<kind>/<bucket>``.

    Non-parquet selections (e.g. the audit-receipt mirror under ``audits/``)
    carry no token count and are not part of the token accounting, so they are
    excluded here.  Order is preserved for deterministic error messages.
    """
    keys: list[str] = []
    for remote, _local in selections:
        if remote.startswith("parquet/"):
            keys.append(remote[len("parquet/"):])
    return keys


def _load_verified_token_total(
    audit_receipts: tuple[Path, ...],
    selections: tuple[tuple[str, Path], ...],
) -> int:
    """Sum the receipt-verified valid token count for exactly the selected
    parquet buckets.

    The token total is profile-aware: under ``--code-commits-only`` only the
    selected ``code``/``commits`` buckets are summed, so the manifest never
    reports the all-valid (code+commits+pr) total for a subset upload.  Every
    selected ``(kind, bucket)`` must be covered by the receipt's
    ``by_kind_bucket`` map and be green (``bad_files == bad_rows == 0``);
    otherwise this raises ``SystemExit`` rather than uploading on an unverified
    or stale/narrowly-scoped receipt.
    """
    selected_keys = _selected_bucket_keys(selections)
    if not selected_keys:
        raise SystemExit(
            "no parquet buckets selected; refusing to write a verified token total"
        )

    # Merge by_kind_bucket across all receipts.  A bucket covered by more than
    # one receipt is ambiguous and would be silently double-counted, so fail loud.
    merged: dict[str, dict] = {}
    for path in audit_receipts:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("schema") != AUDIT_SCHEMA or data.get("status") != "verified":
            raise SystemExit(
                f"audit receipt is not a schema-bound verified receipt: {path}"
            )
        by_kind_bucket = data.get("by_kind_bucket")
        if not isinstance(by_kind_bucket, dict):
            raise SystemExit(f"audit receipt missing by_kind_bucket map: {path}")
        for key, entry in by_kind_bucket.items():
            if key in merged:
                raise SystemExit(
                    f"audit bucket {key!r} covered by multiple receipts; ambiguous coverage"
                )
            merged[key] = entry

    missing = [key for key in selected_keys if key not in merged]
    if missing:
        raise SystemExit(
            "audit receipt does not cover selected upload buckets: "
            + ", ".join(missing)
        )

    nongreen: list[str] = []
    total = 0
    for key in selected_keys:
        entry = merged[key]
        bad_files = int(entry["bad_files"])
        bad_rows = int(entry["bad_rows"])
        if bad_files or bad_rows:
            nongreen.append(f"{key} bad_files={bad_files} bad_rows={bad_rows}")
        total += int(entry["valid_tokens"])
    if nongreen:
        raise SystemExit(
            "audit receipt buckets are not green: " + "; ".join(nongreen)
        )
    return total


def _existing_sources(selections: tuple[tuple[str, Path], ...]) -> list[tuple[str, Path]]:
    missing: list[str] = []
    out: list[tuple[str, Path]] = []
    for remote, local in selections:
        if local.is_symlink() or not local.is_dir():
            missing.append(str(local))
        else:
            out.append((remote, local))
    if missing:
        raise SystemExit("missing verified upload inputs:\n" + "\n".join(missing))
    return out


def _run(cmd: list[str], *, dry_run: bool, env: dict[str, str] | None = None) -> None:
    print("+ " + " ".join(cmd), flush=True)
    if dry_run:
        return
    subprocess.run(cmd, check=True, env=env)


def _sync_one(
    *,
    endpoint_url: str,
    bucket: str,
    prefix: str,
    remote: str,
    local: Path,
    dry_run: bool,
    env: dict[str, str] | None,
) -> str:
    target = f"s3://{bucket}/{prefix.rstrip('/')}/{remote.strip('/')}/"
    cmd = [
        "aws",
        "s3",
        "sync",
        str(local),
        target,
        "--endpoint-url",
        endpoint_url,
        "--only-show-errors",
        "--no-progress",
    ]
    _run(cmd, dry_run=dry_run, env=env)
    return target


def _upload_manifest(
    *,
    endpoint_url: str,
    bucket: str,
    prefix: str,
    manifest_path: Path,
    dry_run: bool,
    env: dict[str, str] | None,
) -> dict[str, object]:
    target = f"s3://{bucket}/{prefix.rstrip('/')}/manifest.json"
    digest = sha256_file(manifest_path)
    size = manifest_path.stat().st_size
    _run(
        [
            "aws",
            "s3",
            "cp",
            str(manifest_path),
            target,
            "--endpoint-url",
            endpoint_url,
            "--metadata",
            f"sha256={digest}",
            "--checksum-algorithm",
            "SHA256",
            "--checksum-sha256",
            base64.b64encode(bytes.fromhex(digest)).decode("ascii"),
            "--only-show-errors",
            "--no-progress",
        ],
        dry_run=dry_run,
        env=env,
    )
    if dry_run:
        return {"status": "dry_run", "size": size, "sha256": digest}
    result = subprocess.run(
        [
            "aws",
            "s3api",
            "head-object",
            "--bucket",
            bucket,
            "--key",
            f"{prefix.rstrip('/')}/manifest.json",
            "--endpoint-url",
            endpoint_url,
            "--checksum-mode",
            "ENABLED",
            "--output",
            "json",
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"manifest HEAD failed after upload: {result.stderr.strip()}")
    head = json.loads(result.stdout)
    metadata = {
        str(key).lower(): value for key, value in (head.get("Metadata") or {}).items()
    }
    expected_checksum = base64.b64encode(bytes.fromhex(digest)).decode("ascii")
    if (
        int(head.get("ContentLength", -1)) != size
        or metadata.get("sha256") != digest
        or head.get("ChecksumSHA256") != expected_checksum
        or head.get("ChecksumType") != "FULL_OBJECT"
    ):
        raise RuntimeError("uploaded manifest lacks exact full-object SHA-256 verification")
    return {
        "status": "verified",
        "size": size,
        "sha256": digest,
        "checksum_sha256": head["ChecksumSHA256"],
    }


def _audit_selected_parquet_files(
    inventory: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Run the existing semantic parquet auditor over the exact inventory."""

    try:
        from scripts.audit_sidecar_parquet import _audit_file
    except Exception as exc:
        raise SystemExit(
            f"cannot load semantic parquet auditor for manifest publication: {exc}"
        ) from exc

    audited: list[dict[str, object]] = []
    for selection in inventory:
        remote = str(selection["remote"])
        if not remote.startswith("parquet/"):
            continue
        kind, bucket = remote.removeprefix("parquet/").split("/", 1)
        local_root = Path(str(selection["local"]))
        files = selection.get("files")
        if not isinstance(files, list):
            raise SystemExit(f"manifest inventory lacks file records for {remote}")
        for record in files:
            if not isinstance(record, dict):
                raise SystemExit(f"manifest inventory file record is invalid for {remote}")
            relative = str(record["path"])
            parquet_path = contained_path(
                local_root,
                relative,
                where=f"semantic audit file for {remote}",
            )
            try:
                result = _audit_file(str(parquet_path), kind, bucket, 65536)
            except Exception as exc:
                raise SystemExit(
                    f"semantic parquet audit failed for {remote}/{relative}: {exc}"
                ) from exc
            stats = result.get("stats")
            if not isinstance(stats, dict):
                raise SystemExit(
                    f"semantic parquet audit returned no stats for {remote}/{relative}"
                )
            if (
                int(stats.get("files", -1)) != 1
                or int(stats.get("bad_files", -1))
                or int(stats.get("bad_rows", -1))
            ):
                raise SystemExit(
                    f"semantic parquet audit is not green for {remote}/{relative}: "
                    f"files={stats.get('files')} bad_files={stats.get('bad_files')} "
                    f"bad_rows={stats.get('bad_rows')}"
                )
            audited.append(
                {
                    "remote": remote,
                    "path": relative,
                    "size": record["size"],
                    "sha256": record["sha256"],
                    "stats": stats,
                }
            )
    if not audited:
        raise SystemExit("semantic parquet audit selected no files")
    return audited


def _assert_inventory_stable(inventory: list[dict[str, object]]) -> None:
    """Reject source changes between inventory hashing and publication."""

    for selection in inventory:
        current = inventory_directory(
            Path(str(selection["local"])),
            remote=str(selection["remote"]),
        )
        for field in ("files", "file_count", "byte_count", "artifact_set_sha256"):
            if current[field] != selection[field]:
                raise SystemExit(
                    f"selection inventory changed during semantic audit: "
                    f"{selection['remote']} ({field})"
                )


def _write_manifest(
    path: Path,
    *,
    bucket: str,
    prefix: str,
    endpoint_url: str,
    token_total: int,
    selections: tuple[tuple[str, Path], ...],
    audit_receipts: tuple[Path, ...],
    include_standalone_pr: bool,
) -> dict[str, object]:
    selected_keys = _selected_bucket_keys(selections)
    if len(audit_receipts) != 1:
        raise SystemExit("exactly one final audit receipt is required")
    audit_path = audit_receipts[0]
    audit_payload = validate_audit_receipt(
        json.loads(audit_path.read_text(encoding="utf-8")),
        selected_keys=selected_keys,
    )
    inventory: list[dict[str, object]] = []
    for remote, local in selections:
        record = inventory_directory(local, remote=remote)
        inventory.append(
            {
                "remote": remote,
                "local": local.as_posix(),
                **record,
            }
        )
        key = remote.removeprefix("parquet/") if remote.startswith("parquet/") else None
        if key is not None:
            expected_files = int(audit_payload["by_kind_bucket"][key]["files"])
            if int(record["file_count"]) != expected_files:
                raise SystemExit(
                    f"inventory file count for {remote}={record['file_count']} "
                    f"differs from audit receipt={expected_files}"
                )
    audit_selection = next(
        selection for selection in inventory if selection["remote"] == AUDIT_REMOTE
    )
    audit_file = next(
        record
        for record in audit_selection["files"]
        if record["path"] == AUDIT_FILENAME
    )
    if audit_file["sha256"] != sha256_file(audit_path):
        raise SystemExit("audit receipt inventory SHA-256 drifted during manifest build")
    audited_files = _audit_selected_parquet_files(inventory)
    _assert_inventory_stable(inventory)
    semantic_audit = build_semantic_audit_binding(
        selections=inventory,
        audited_files=audited_files,
        source_receipt_sha256=str(audit_file["sha256"]),
    )
    try:
        validate_semantic_audit_receipt_binding(
            semantic_audit,
            audit_payload,
            selected_keys=selected_keys,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if int(semantic_audit["verified_valid_tokens"]) != token_total:
        raise SystemExit(
            "fresh semantic audit valid-token total differs from selected "
            "audit receipt total"
        )
    payload = {
        "schema": MANIFEST_SCHEMA,
        "bucket": bucket,
        "prefix": prefix,
        "endpoint_url": endpoint_url,
        "verified_valid_tokens": token_total,
        "profile": (
            "code_commits_plus_standalone_pr"
            if include_standalone_pr
            else "code_commits_integrated_pr"
        ),
        "standalone_pr_included": include_standalone_pr,
        "selections": inventory,
        "inventory_sha256": inventory_sha256(inventory),
        "selection_policy": selection_policy(
            [remote for remote, _local in selections],
            include_standalone_pr=include_standalone_pr,
        ),
        "audit_receipt": {
            "schema": AUDIT_SCHEMA,
            "status": "verified",
            "remote": AUDIT_REMOTE,
            "path": AUDIT_FILENAME,
            "sha256": audit_file["sha256"],
        },
        "semantic_audit": semantic_audit,
        "graph_contract": GRAPH_CONTRACT,
        "objective_contract": OBJECTIVE_CONTRACT,
    }
    manifest = finalize_manifest(payload)
    write_json_atomic(path, manifest)
    return manifest


def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bucket", required=True)
    ap.add_argument("--prefix", default=DEFAULT_PREFIX)
    ap.add_argument("--endpoint-url", default=DEFAULT_ENDPOINT)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--manifest", type=Path, default=Path("outputs/sidecar_upload_manifest.json"))
    ap.add_argument("--env-file", type=Path, default=Path(".env"))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--include-standalone-pr",
        action="store_true",
        help=(
            "Also upload standalone PR discussion parquet. This is diagnostic/"
            "curriculum material; default training upload excludes it because "
            "PR discussion is integrated into commit docstrings."
        ),
    )
    ap.add_argument(
        "--code-commits-only",
        action="store_true",
        help=(
            "Deprecated compatibility flag. Code+commit buckets are already the "
            "default training upload profile."
        ),
    )
    args = ap.parse_args(argv)
    receipt_path = args.manifest.with_name("sidecar_upload_receipt.json")
    receipt = {
        "schema": "cppmega_sidecar_upload_receipt_v1",
        "status": "starting",
        "bucket": args.bucket,
        "prefix": args.prefix.rstrip("/"),
        "endpoint_url": args.endpoint_url,
        "manifest": str(args.manifest),
        "data_object_verification": {
            "status": "pending",
            "completion_gate": "schema_bound_download_receipt",
        },
    }
    write_json_atomic(receipt_path, receipt)
    try:
        if args.jobs <= 0:
            raise ValueError("jobs must be positive")
        if args.include_standalone_pr and args.code_commits_only:
            raise SystemExit(
                "--include-standalone-pr conflicts with --code-commits-only"
            )
        _load_env_file(args.env_file)

        selections = _selection_items(
            include_standalone_pr=args.include_standalone_pr
        )
        audit_receipts = _audit_receipts()
        token_total = _load_verified_token_total(audit_receipts, selections)
        sources = _existing_sources(selections)
        manifest = _write_manifest(
            args.manifest,
            bucket=args.bucket,
            prefix=args.prefix,
            endpoint_url=args.endpoint_url,
            token_total=token_total,
            selections=selections,
            audit_receipts=audit_receipts,
            include_standalone_pr=args.include_standalone_pr,
        )
        receipt.update(
            {
                "status": "dry_run" if args.dry_run else "prepared",
                "manifest_payload_sha256": manifest["manifest_payload_sha256"],
                "inventory_sha256": manifest["inventory_sha256"],
                "data_object_verification": {
                    "status": "manifest_bound",
                    "completion_gate": "schema_bound_download_receipt",
                },
            }
        )
        write_json_atomic(receipt_path, receipt)
        print(
            json.dumps(
                {
                    "verified_valid_tokens": token_total,
                    "sources": len(sources),
                    "manifest": str(args.manifest),
                    "dry_run": args.dry_run,
                },
                indent=2,
            )
        )

        s3_env = None if args.dry_run else _s3_env()

        with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
            futures = [
                pool.submit(
                    _sync_one,
                    endpoint_url=args.endpoint_url,
                    bucket=args.bucket,
                    prefix=args.prefix,
                    remote=remote,
                    local=local,
                    dry_run=args.dry_run,
                    env=s3_env,
                )
                for remote, local in sources
            ]
            for future in as_completed(futures):
                print(f"synced {future.result()}", flush=True)

        manifest_verification = _upload_manifest(
            endpoint_url=args.endpoint_url,
            bucket=args.bucket,
            prefix=args.prefix,
            manifest_path=args.manifest,
            dry_run=args.dry_run,
            env=s3_env,
        )
    except (Exception, SystemExit) as error:
        receipt["status"] = "failed"
        receipt["error"] = f"{type(error).__name__}: {error}"
        write_json_atomic(receipt_path, receipt)
        raise
    receipt["status"] = "manifest_verified" if not args.dry_run else "dry_run"
    receipt["manifest_verification"] = manifest_verification
    write_json_atomic(receipt_path, receipt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
