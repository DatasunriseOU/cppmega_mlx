#!/usr/bin/env python3
"""Download verified cppmega sidecar parquet shards from Nebius Object Storage.

Run this on the H200 host/container after exporting NEBIUS_S3_ACCESS_KEY_ID and
NEBIUS_S3_SECRET_ACCESS_KEY for the Nebius Object Storage access key.  Nebius
Object Storage exposes an S3-compatible object API; the S3 client is pointed at
the Nebius endpoint and does not use AWS cloud services.

The built-in default manifest is the final all-valid profile.  The old
code+commits-only subset is available only by explicit request via
``--code-commits-only``.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Iterable


DEFAULT_ENDPOINT = "https://storage.eu-north1.nebius.cloud"
DEFAULT_PREFIX = "cppmega-sidecar/valid-all-20260627"


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


def _default_selections(*, code_commits_only: bool = False) -> list[dict[str, str]]:
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
    if not code_commits_only:
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
    code_commits_only: bool = False,
) -> dict:
    return {
        "bucket": bucket,
        "prefix": prefix,
        "endpoint_url": endpoint_url,
        "profile": "code_commits_only" if code_commits_only else "all_valid",
        "standalone_pr_included": not code_commits_only,
        "selections": _default_selections(code_commits_only=code_commits_only),
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
            "Deprecated no-op for --use-default-manifest. Standalone PR parquet "
            "is included by default in the all-valid profile."
        ),
    )
    ap.add_argument(
        "--code-commits-only",
        action="store_true",
        help="With --use-default-manifest, download only code+commit buckets.",
    )
    args = ap.parse_args(argv)
    _load_env_file(args.env_file)

    s3_env = None if args.dry_run else _s3_env()
    args.dest.mkdir(parents=True, exist_ok=True)

    if args.use_default_manifest:
        manifest = _default_manifest(
            args.bucket,
            args.prefix,
            args.endpoint_url,
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
