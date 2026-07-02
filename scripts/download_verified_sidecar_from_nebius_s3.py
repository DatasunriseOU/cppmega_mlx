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
import json
import os
from pathlib import Path
import subprocess
from typing import Iterable


DEFAULT_ENDPOINT = "https://storage.eu-north1.nebius.cloud"
DEFAULT_PREFIX = "cppmega-sidecar/code-commits-integrated-pr-20260627"

# The audit receipt is downloaded as one of the selections; this is the remote
# subprefix and the JSON filename the producer (audit_sidecar_parquet.py) writes.
RECEIPT_REMOTE = "audits/sidecar_audit_all_final_poststop_valid"
RECEIPT_FILENAME = "sidecar_parquet_audit.json"
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
            return dest / item["local"] / RECEIPT_FILENAME
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
    """Fail-closed consume-side gate over the freshly downloaded set.

    ``aws s3 sync`` ETag-checks individual objects, so a single truncated file is
    already caught, but a *missing-shards SET* (an absent or partial remote
    prefix) syncs "successfully" and would otherwise be trusted as the verified
    set. This proves the consume side by RAISING (``SystemExit``) unless ALL of
    the following hold:

      * every selected parquet bucket is present in the audit receipt and green;
      * the manifest's ``verified_valid_tokens`` matches the selected buckets'
        receipt total (when a real downloaded manifest is in use -- not the
        built-in default);
      * every downloaded parquet bucket has exactly the shard count the receipt
        certified for that ``kind/bucket`` -- this is what catches a
        missing-shards SET that the per-object ETag check cannot see.

    The receipt may cover diagnostic standalone PR buckets that this manifest
    intentionally does not select. Those unselected buckets do not participate
    in the consume-side training gate.
    """
    receipt_path = _receipt_path_for(dest, manifest)
    if not receipt_path.exists():
        raise SystemExit(
            f"audit receipt missing after download: {receipt_path}. "
            "Refusing to trust an unverified set."
        )
    report = json.loads(receipt_path.read_text(encoding="utf-8"))
    by_kind_bucket = report["by_kind_bucket"]
    receipt_valid_tokens = 0
    selected_bad_files = 0
    selected_bad_rows = 0
    nongreen: list[str] = []
    manifest_token_total = manifest.get("verified_valid_tokens")
    if require_token_total and manifest_token_total is None:
        raise SystemExit(
            "downloaded manifest is missing verified_valid_tokens; cannot "
            f"reconcile the downloaded set against the audit receipt {receipt_path}."
        )

    mismatches: list[str] = []
    checked = 0
    for item in manifest["selections"]:
        remote = item["remote"].strip("/")
        if not remote.startswith(PARQUET_REMOTE_PREFIX):
            continue
        key = remote[len(PARQUET_REMOTE_PREFIX):]  # e.g. "parquet/code/1024" -> "code/1024"
        local_dir = dest / item["local"]
        downloaded = len(list(local_dir.glob("*.parquet")))
        if key not in by_kind_bucket:
            mismatches.append(
                f"{remote}: downloaded {downloaded} shard(s) but the receipt has "
                f"no entry for {key!r}"
            )
            continue
        entry = by_kind_bucket[key]
        bad_files = int(entry["bad_files"])
        bad_rows = int(entry["bad_rows"])
        if bad_files or bad_rows:
            nongreen.append(f"{key} bad_files={bad_files} bad_rows={bad_rows}")
        selected_bad_files += bad_files
        selected_bad_rows += bad_rows
        receipt_valid_tokens += int(entry["valid_tokens"])
        expected = int(by_kind_bucket[key]["files"])
        if downloaded != expected:
            mismatches.append(
                f"{remote}: downloaded {downloaded} shard(s) into {local_dir} but "
                f"the receipt certified {expected}"
            )
        checked += 1
    if mismatches:
        raise SystemExit(
            f"downloaded set does not match the audit receipt ({receipt_path}):\n"
            + "\n".join(mismatches)
        )
    if checked == 0:
        raise SystemExit(
            "no parquet bucket selections were verified against the receipt "
            f"{receipt_path}; refusing to certify an empty set."
        )
    if nongreen:
        raise SystemExit(
            "selected audit receipt buckets are not green: "
            + "; ".join(nongreen)
        )
    if (
        manifest_token_total is not None
        and int(manifest_token_total) != receipt_valid_tokens
    ):
        raise SystemExit(
            "manifest verified_valid_tokens "
            f"({int(manifest_token_total)}) != selected receipt valid_tokens "
            f"({receipt_valid_tokens}); manifest/receipt pair is inconsistent. "
            f"Receipt={receipt_path}."
        )

    return {
        "receipt": str(receipt_path),
        "verified_valid_tokens": receipt_valid_tokens,
        "buckets_verified": checked,
        "bad_files": selected_bad_files,
        "bad_rows": selected_bad_rows,
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
        help="Do not download manifest first; use the built-in verified selection.",
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
    if args.include_standalone_pr and args.code_commits_only:
        raise SystemExit("--include-standalone-pr conflicts with --code-commits-only")
    _load_env_file(args.env_file)

    s3_env = None if args.dry_run else _s3_env()
    args.dest.mkdir(parents=True, exist_ok=True)

    if args.use_default_manifest:
        manifest = _default_manifest(
            args.bucket,
            args.prefix,
            args.endpoint_url,
            include_standalone_pr=args.include_standalone_pr,
            code_commits_only=args.code_commits_only,
        )
    else:
        manifest_path = _cp_manifest(
            endpoint_url=args.endpoint_url,
            bucket=args.bucket,
            prefix=args.prefix,
            dest=args.dest,
            dry_run=args.dry_run,
            env=s3_env,
        )
        manifest = (
            _default_manifest(
                args.bucket,
                args.prefix,
                args.endpoint_url,
                include_standalone_pr=args.include_standalone_pr,
                code_commits_only=args.code_commits_only,
            )
            if args.dry_run
            else json.loads(manifest_path.read_text(encoding="utf-8"))
        )

    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        futures = []
        for item in manifest["selections"]:
            remote = item["remote"]
            local = args.dest / item["local"]
            futures.append(
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
            )
        for future in as_completed(futures):
            print(f"downloaded {future.result()}", flush=True)

    if not args.dry_run:
        verification = _verify_downloaded_set(
            dest=args.dest,
            manifest=manifest,
            require_token_total=not args.use_default_manifest,
        )
        print(json.dumps({"verified": verification}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
