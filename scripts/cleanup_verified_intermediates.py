#!/usr/bin/env python3
"""Purge source/intermediate files for repos whose parquet output is verified.

Streaming principle: download -> process -> verify parquet -> delete source+intermediate.
Parquet (zstd-max compressed) is the ONLY persistent artifact.

Usage:
    python scripts/cleanup_verified_intermediates.py [--dry-run] [--extract-cache] [--source-cache]
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pyarrow.parquet as pq

OUTPUT_BASE = Path("outputs")
EXTRACT_CACHE = OUTPUT_BASE / "extract_cache_case5_shared"
SOURCE_CACHE = OUTPUT_BASE / "source_cache" / "code"
CODE_PARQUET = OUTPUT_BASE / "reindexed_case5_v7_20260715_130725_code"
DONE_MANIFESTS = [
    OUTPUT_BASE / "conveyor_case5_v11_resume_20260719_120500" / "_done.json",
    OUTPUT_BASE / "conveyor_case5_v10_resume_20260718_150406" / "_done.json",
]


def load_done_repos() -> set[str]:
    done = set()
    for mf in DONE_MANIFESTS:
        if not mf.exists():
            continue
        data = json.loads(mf.read_text())
        for key, val in data.get("done", {}).items():
            if isinstance(val, dict) and val.get("source") == "code":
                done.add(key.split("::")[0])
    return done


def verify_parquet_exists(repo: str) -> bool:
    for length_dir in CODE_PARQUET.iterdir():
        if not length_dir.is_dir():
            continue
        if (length_dir / f"{repo}.parquet").exists():
            schema = pq.read_schema(length_dir / f"{repo}.parquet")
            if "trained_token_count" in schema.names:
                return True
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--extract-cache", action="store_true", help="Clean extract cache")
    parser.add_argument("--source-cache", action="store_true", help="Clean source cache")
    args = parser.parse_args()

    done_repos = load_done_repos()
    print(f"Repos with verified parquet: {len(done_repos)}")

    freed = 0

    if args.extract_cache and EXTRACT_CACHE.exists():
        for entry in EXTRACT_CACHE.iterdir():
            repo_name = entry.name
            if repo_name in done_repos and verify_parquet_exists(repo_name):
                size = sum(f.stat().st_size for f in entry.rglob("*") if f.is_file())
                if args.dry_run:
                    print(f"  [dry-run] would remove extract_cache/{repo_name} ({size/1e9:.2f} GB)")
                else:
                    shutil.rmtree(entry, ignore_errors=True)
                    print(f"  removed extract_cache/{repo_name} ({size/1e9:.2f} GB)")
                freed += size

    if args.source_cache and SOURCE_CACHE.exists():
        for entry in SOURCE_CACHE.iterdir():
            repo_name = entry.name
            if repo_name in done_repos and verify_parquet_exists(repo_name):
                size = sum(f.stat().st_size for f in entry.rglob("*") if f.is_file())
                if args.dry_run:
                    print(f"  [dry-run] would remove source_cache/{repo_name} ({size/1e9:.2f} GB)")
                else:
                    shutil.rmtree(entry, ignore_errors=True)
                    print(f"  removed source_cache/{repo_name} ({size/1e9:.2f} GB)")
                freed += size

    print(f"\nTotal freed: {freed/1e9:.2f} GB")


if __name__ == "__main__":
    main()
