#!/usr/bin/env python3
"""Download verified cppmega sidecar parquet shards from Nebius Object Storage.

Run this on the H200 host/container after exporting NEBIUS_S3_ACCESS_KEY_ID and
NEBIUS_S3_SECRET_ACCESS_KEY for the Nebius Object Storage access key.  Nebius
Object Storage exposes an S3-compatible object API; the S3 client is pointed at
the Nebius endpoint and does not use AWS cloud services.

The built-in default manifest is the training profile: C/C++ code plus commit
parquet, where PR discussion is already integrated into commit docstrings.
Standalone PR parquet is diagnostic material and is downloaded only with
explicit ``--include-standalone-pr``.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import subprocess
import shutil
import sys
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.sidecar_manifest_contract import (
    AUDIT_FILENAME,
    AUDIT_REMOTE,
    MANIFEST_SCHEMA,
    contained_path,
    resolve_s3_env,
    validate_audit_receipt,
    validate_manifest,
    validate_semantic_audit_receipt_binding,
    verify_inventory,
    write_json_atomic,
)


DEFAULT_ENDPOINT = "https://storage.eu-north1.nebius.cloud"
DEFAULT_PREFIX = "cppmega-sidecar/code-commits-integrated-pr-20260627"

# The audit receipt is downloaded as one of the selections; this is the remote
# subprefix and the JSON filename the producer (audit_sidecar_parquet.py) writes.
RECEIPT_REMOTE = AUDIT_REMOTE
RECEIPT_FILENAME = AUDIT_FILENAME
PARQUET_REMOTE_PREFIX = "parquet/"


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


def _run(cmd: list[str], *, dry_run: bool, env: dict[str, str] | None = None) -> None:
    print("+ " + " ".join(cmd), flush=True)
    if dry_run:
        return
    subprocess.run(cmd, check=True, env=env)


def _cp_manifest(
    *,
    endpoint_url: str,
    bucket: str,
    prefix: str,
    dest: Path,
    dry_run: bool,
    env: dict[str, str] | None,
) -> Path:
    target = dest / "manifest.json"
    uri = f"s3://{bucket}/{prefix.rstrip('/')}/manifest.json"
    _run(
        [
            "aws",
            "s3",
            "cp",
            uri,
            str(target),
            "--endpoint-url",
            endpoint_url,
            "--only-show-errors",
            "--no-progress",
        ],
        dry_run=dry_run,
        env=env,
    )
    return target


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
    source = f"s3://{bucket}/{prefix.rstrip('/')}/{remote.strip('/')}/"
    _run(
        [
            "aws",
            "s3",
            "sync",
            source,
            str(local),
            "--endpoint-url",
            endpoint_url,
            "--only-show-errors",
            "--no-progress",
        ],
        dry_run=dry_run,
        env=env,
    )
    return str(local)


def _default_selections(
    *,
    include_standalone_pr: bool = False,
    code_commits_only: bool = False,
) -> list[dict[str, str]]:
    if include_standalone_pr and code_commits_only:
        raise SystemExit(
            "--include-standalone-pr conflicts with deprecated --code-commits-only"
        )
    selections = [
        {"remote": "parquet/code/1024", "local": "outputs/reindexed/1024"},
        {"remote": "parquet/code/2048", "local": "outputs/reindexed/2048"},
        {"remote": "parquet/code/4096", "local": "outputs/reindexed/4096"},
        {"remote": "parquet/code/8192", "local": "outputs/reindexed/8192"},
        {"remote": "parquet/commits/1024", "local": "outputs/reindexed_commits/1024"},
        {"remote": "parquet/commits/2048", "local": "outputs/reindexed_commits/2048"},
        {"remote": "parquet/commits/4096", "local": "outputs/reindexed_commits/4096"},
        {"remote": "parquet/commits/8192", "local": "outputs/reindexed_commits/8192"},
        {"remote": "parquet/commits/16384", "local": "outputs/reindexed_commits/16384"},
        {
            "remote": "audits/sidecar_audit_all_final_poststop_valid",
            "local": "outputs/sidecar_audit_all_final_poststop_valid",
        },
    ]
    if include_standalone_pr:
        selections.extend(
            [
                {"remote": "parquet/pr/1024", "local": "outputs/reindexed_pr/1024"},
                {"remote": "parquet/pr/2048", "local": "outputs/reindexed_pr/2048"},
                {"remote": "parquet/pr/4096", "local": "outputs/reindexed_pr/4096"},
                {"remote": "parquet/pr/8192", "local": "outputs/reindexed_pr/8192"},
                {"remote": "parquet/pr/16384", "local": "outputs/reindexed_pr/16384"},
            ]
        )
    return selections


def _default_manifest(
    bucket: str,
    prefix: str,
    endpoint_url: str,
    *,
    include_standalone_pr: bool = False,
    code_commits_only: bool = False,
) -> dict:
    if include_standalone_pr and code_commits_only:
        raise SystemExit(
            "--include-standalone-pr conflicts with deprecated --code-commits-only"
        )
    return {
        "bucket": bucket,
        "prefix": prefix,
        "endpoint_url": endpoint_url,
        "profile": (
            "code_commits_plus_standalone_pr"
            if include_standalone_pr
            else "code_commits_integrated_pr"
        ),
        "standalone_pr_included": include_standalone_pr,
        "selections": _default_selections(
            include_standalone_pr=include_standalone_pr,
            code_commits_only=code_commits_only,
        ),
    }


def _receipt_path_for(dest: Path, manifest: dict) -> Path:
    """Locate the downloaded audit receipt from the manifest selections, or RAISE."""
    for item in manifest["selections"]:
        if item["remote"].strip("/") == RECEIPT_REMOTE:
            return contained_path(
                dest, item["local"], where="audit receipt local selection"
            ) / RECEIPT_FILENAME
    raise SystemExit(
        "manifest selections do not include the audit receipt "
        f"({RECEIPT_REMOTE!r}); the downloaded set cannot be verified."
    )


def _verify_downloaded_set(
    *,
    dest: Path,
    manifest: dict,
    require_token_total: bool,
) -> dict:
    """Verify the exact manifest-bound file inventory and audit accounting."""

    validated = validate_manifest(manifest)
    inventory = verify_inventory(dest, validated)
    receipt_path = _receipt_path_for(dest, validated)
    if receipt_path.is_symlink() or not receipt_path.is_file():
        raise SystemExit(f"audit receipt missing after download: {receipt_path}")
    selected_keys = [
        str(item["remote"])[len(PARQUET_REMOTE_PREFIX):]
        for item in validated["selections"]
        if str(item["remote"]).startswith(PARQUET_REMOTE_PREFIX)
    ]
    report = validate_audit_receipt(
        json.loads(receipt_path.read_text(encoding="utf-8")),
        selected_keys=selected_keys,
    )
    audit_binding = validated["audit_receipt"]
    if (
        not isinstance(audit_binding, dict)
        or hashlib.sha256(receipt_path.read_bytes()).hexdigest()
        != audit_binding["sha256"]
    ):
        raise SystemExit("downloaded audit receipt SHA-256 does not match manifest")
    validate_semantic_audit_receipt_binding(
        validated["semantic_audit"],
        report,
        selected_keys=selected_keys,
    )
    receipt_valid_tokens = sum(
        int(report["by_kind_bucket"][key]["valid_tokens"]) for key in selected_keys
    )
    if require_token_total and int(validated["verified_valid_tokens"]) != receipt_valid_tokens:
        raise SystemExit(
            "manifest verified_valid_tokens does not match selected audit receipt total"
        )
    return {
        "status": "verified",
        "receipt": str(receipt_path),
        "verified_valid_tokens": receipt_valid_tokens,
        "buckets_verified": len(selected_keys),
        "bad_files": 0,
        "bad_rows": 0,
        "inventory": inventory,
    }


def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bucket", required=True)
    ap.add_argument("--prefix", default=DEFAULT_PREFIX)
    ap.add_argument("--endpoint-url", default=DEFAULT_ENDPOINT)
    ap.add_argument("--dest", type=Path, default=Path("/data/cppmega_sidecar"))
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--env-file", type=Path, default=Path(".env"))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--use-default-manifest",
        action="store_true",
        help=(
            "Planning-only: skip the manifest copy and print the built-in "
            "training selection. Real downloads require a remote manifest."
        ),
    )
    ap.add_argument(
        "--include-standalone-pr",
        action="store_true",
        help=(
            "With --use-default-manifest, also download standalone PR discussion "
            "parquet. Default excludes it because PR discussion is integrated "
            "into commit docstrings."
        ),
    )
    ap.add_argument(
        "--code-commits-only",
        action="store_true",
        help=(
            "Deprecated compatibility flag. Code+commit buckets are already the "
            "built-in default manifest."
        ),
    )
    args = ap.parse_args(argv)
    if args.jobs <= 0:
        raise ValueError("jobs must be positive")
    if args.include_standalone_pr and args.code_commits_only:
        raise SystemExit("--include-standalone-pr conflicts with --code-commits-only")

    if args.dry_run:
        if args.use_default_manifest:
            manifest = _default_manifest(
                args.bucket,
                args.prefix,
                args.endpoint_url,
                include_standalone_pr=args.include_standalone_pr,
                code_commits_only=args.code_commits_only,
            )
        else:
            # Planning mode must be safe to run on a laptop with no S3
            # credentials.  Print the manifest fetch that the leader will run,
            # then use the deterministic profile only to enumerate sync URIs.
            _cp_manifest(
                endpoint_url=args.endpoint_url,
                bucket=args.bucket,
                prefix=args.prefix,
                dest=args.dest,
                dry_run=True,
                env=None,
            )
            manifest = _default_manifest(
                args.bucket,
                args.prefix,
                args.endpoint_url,
                include_standalone_pr=args.include_standalone_pr,
                code_commits_only=args.code_commits_only,
            )
        for item in manifest["selections"]:
            _sync_one(
                endpoint_url=args.endpoint_url,
                bucket=args.bucket,
                prefix=args.prefix,
                remote=item["remote"],
                local=contained_path(
                    args.dest,
                    item["local"],
                    where="planning manifest local selection",
                ),
                dry_run=True,
                env=None,
            )
        return 0

    _load_env_file(args.env_file)
    if args.use_default_manifest:
        # A built-in profile has no inventory or audit binding and therefore can
        # never certify a real restore.
        raise SystemExit(
            "--use-default-manifest is planning-only; real downloads require "
            "a schema-bound remote manifest"
        )
    else:
        if args.dest.exists():
            raise SystemExit(
                f"refusing stale download destination; remove or move it first: {args.dest}"
            )
        s3_env = _s3_env()
        stage_parent = args.dest.parent.resolve()
        stage_parent.mkdir(parents=True, exist_ok=True)
        stage = stage_parent / f".{args.dest.name}.download-{os.getpid()}"
        if stage.exists() or stage.is_symlink():
            raise SystemExit(f"refusing stale download staging directory: {stage}")
        stage.mkdir()
        try:
            manifest_path = _cp_manifest(
                endpoint_url=args.endpoint_url,
                bucket=args.bucket,
                prefix=args.prefix,
                dest=stage,
                dry_run=False,
                env=s3_env,
            )
            manifest = validate_manifest(
                json.loads(manifest_path.read_text(encoding="utf-8")),
                expected_bucket=args.bucket,
                expected_prefix=args.prefix.rstrip("/"),
                expected_endpoint_url=args.endpoint_url,
            )
            with ThreadPoolExecutor(max_workers=args.jobs) as pool:
                futures = []
                for item in manifest["selections"]:
                    remote = str(item["remote"])
                    local = contained_path(
                        stage, item["local"], where="manifest local selection"
                    )
                    futures.append(
                        pool.submit(
                            _sync_one,
                            endpoint_url=args.endpoint_url,
                            bucket=args.bucket,
                            prefix=args.prefix,
                            remote=remote,
                            local=local,
                            dry_run=False,
                            env=s3_env,
                        )
                    )
                for future in as_completed(futures):
                    print(f"downloaded {future.result()}", flush=True)
            verification = _verify_downloaded_set(
                dest=stage,
                manifest=manifest,
                require_token_total=True,
            )
            receipt = {
                "schema": "cppmega_sidecar_download_receipt_v1",
                "status": "verified",
                "bucket": args.bucket,
                "prefix": args.prefix.rstrip("/"),
                "endpoint_url": args.endpoint_url,
                "manifest_payload_sha256": manifest["manifest_payload_sha256"],
                "inventory_sha256": manifest["inventory_sha256"],
                "verification": verification,
            }
            write_json_atomic(stage / "download_receipt.json", receipt)
            os.replace(stage, args.dest)
            print(json.dumps({"verified": verification, "destination": str(args.dest)}, indent=2))
            return 0
        except Exception:
            if stage.is_symlink() or (stage.exists() and not stage.is_dir()):
                raise
            shutil.rmtree(stage, ignore_errors=True)
            raise

    raise AssertionError("unreachable download planning branch")


if __name__ == "__main__":
    raise SystemExit(main())
