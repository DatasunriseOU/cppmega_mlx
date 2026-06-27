#!/usr/bin/env python3
"""Upload verified cppmega sidecar parquet shards to Nebius Object Storage.

Default policy is fail-closed and broad: upload every parquet bucket covered by
the final green sidecar audit receipt.  The old code+commits-only H200 smoke
subset is still available, but only by explicit request via ``--code-commits-only``.

Nebius Object Storage exposes an S3-compatible object API.  The current Nebius
CLI manages buckets/transfers, but not individual object syncs, so this script
uses an S3 CLI client against the Nebius endpoint.  Prefer exporting
NEBIUS_S3_ACCESS_KEY_ID and NEBIUS_S3_SECRET_ACCESS_KEY; they are mapped to the
standard S3 client environment only for the subprocess.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
from typing import Iterable


DEFAULT_ENDPOINT = "https://storage.eu-north1.nebius.cloud"
DEFAULT_PREFIX = "cppmega-sidecar/valid-all-20260627"

ALL_VALID_SELECTIONS = (
    ("parquet/code/1024", Path("outputs/reindexed/1024")),
    ("parquet/code/2048", Path("outputs/reindexed/2048")),
    ("parquet/code/4096", Path("outputs/reindexed/4096")),
    ("parquet/code/8192", Path("outputs/reindexed/8192")),
    ("parquet/commits/1024", Path("outputs/reindexed_commits/1024")),
    ("parquet/commits/2048", Path("outputs/reindexed_commits/2048")),
    ("parquet/commits/4096", Path("outputs/reindexed_commits/4096")),
    ("parquet/commits/8192", Path("outputs/reindexed_commits/8192")),
    ("parquet/commits/16384", Path("outputs/reindexed_commits/16384")),
    ("parquet/pr/1024", Path("outputs/reindexed_pr/1024")),
    ("parquet/pr/2048", Path("outputs/reindexed_pr/2048")),
    ("parquet/pr/4096", Path("outputs/reindexed_pr/4096")),
    ("parquet/pr/8192", Path("outputs/reindexed_pr/8192")),
    ("parquet/pr/16384", Path("outputs/reindexed_pr/16384")),
    (
        "audits/sidecar_audit_all_final_poststop_valid",
        Path("outputs/sidecar_audit_all_final_poststop_valid"),
    ),
)

CODE_COMMITS_ONLY_SELECTIONS = (
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
    env = os.environ.copy()
    access_key = env.get("NEBIUS_S3_ACCESS_KEY_ID") or env.get("AWS_ACCESS_KEY_ID")
    secret_key = env.get("NEBIUS_S3_SECRET_ACCESS_KEY") or env.get("AWS_SECRET_ACCESS_KEY")
    if not access_key or not secret_key:
        raise SystemExit(
            "missing Nebius S3 credentials: export NEBIUS_S3_ACCESS_KEY_ID "
            "and NEBIUS_S3_SECRET_ACCESS_KEY for Nebius Object Storage."
        )
    env["AWS_ACCESS_KEY_ID"] = access_key
    env["AWS_SECRET_ACCESS_KEY"] = secret_key
    return env


def _selection_items(*, code_commits_only: bool = False) -> tuple[tuple[str, Path], ...]:
    return CODE_COMMITS_ONLY_SELECTIONS if code_commits_only else ALL_VALID_SELECTIONS


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
        if not local.exists():
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
) -> None:
    target = f"s3://{bucket}/{prefix.rstrip('/')}/manifest.json"
    _run(
        [
            "aws",
            "s3",
            "cp",
            str(manifest_path),
            target,
            "--endpoint-url",
            endpoint_url,
            "--only-show-errors",
            "--no-progress",
        ],
        dry_run=dry_run,
        env=env,
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
    code_commits_only: bool,
) -> None:
    payload = {
        "bucket": bucket,
        "prefix": prefix,
        "endpoint_url": endpoint_url,
        "verified_valid_tokens": token_total,
        "profile": "code_commits_only" if code_commits_only else "all_valid",
        "standalone_pr_included": not code_commits_only,
        "selections": [
            {"remote": remote, "local": str(local)}
            for remote, local in selections
        ],
        "audit_receipts": [str(path) for path in audit_receipts],
        "excluded": [],
    }
    if code_commits_only:
        payload["excluded"].append(
            "outputs/reindexed_pr/* (explicit --code-commits-only request)"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


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
            "Deprecated no-op. Standalone PR parquet is included by default in "
            "the all-valid upload profile. Use --code-commits-only to exclude it."
        ),
    )
    ap.add_argument(
        "--code-commits-only",
        action="store_true",
        help="Upload only code+commit parquet buckets. Default uploads every final-audit-valid bucket.",
    )
    args = ap.parse_args(argv)
    _load_env_file(args.env_file)

    selections = _selection_items(code_commits_only=args.code_commits_only)
    audit_receipts = _audit_receipts()
    token_total = _load_verified_token_total(audit_receipts, selections)
    sources = _existing_sources(selections)
    _write_manifest(
        args.manifest,
        bucket=args.bucket,
        prefix=args.prefix,
        endpoint_url=args.endpoint_url,
        token_total=token_total,
        selections=selections,
        audit_receipts=audit_receipts,
        code_commits_only=args.code_commits_only,
    )
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

    _upload_manifest(
        endpoint_url=args.endpoint_url,
        bucket=args.bucket,
        prefix=args.prefix,
        manifest_path=args.manifest,
        dry_run=args.dry_run,
        env=s3_env,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
