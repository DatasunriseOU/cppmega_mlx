#!/usr/bin/env python3
"""Generate a machine-readable index of mlx_converted checkpoints.

The index maps a short checkpoint id to the directory, source checkpoint path,
conversion schema version, and whether the conversion includes a logit-parity
receipt. It is idempotent: re-run it after regenerating checkpoints to refresh
the index.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT_ROOT = Path(
    "/Volumes/external/sources/cppmega/outputs/checkpoints/mlx_converted"
)
DEFAULT_INDEX_PATH = DEFAULT_CHECKPOINT_ROOT / "index.json"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _inspect_checkpoint(directory: Path) -> dict:
    manifest_path = directory / "model.json"
    weights_path = directory / "model.safetensors"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    entry: dict = {
        "id": directory.name,
        "path": str(directory),
        "manifest": str(manifest_path),
        "schema": manifest.get("schema"),
        "source_checkpoint": manifest.get("source_checkpoint"),
        "dtype": manifest.get("dtype"),
        "has_logit_parity": "logit_parity" in manifest,
        "has_publish_receipt": isinstance(manifest.get("publish"), dict),
        "rope_only": manifest.get("rope_only"),
    }
    if weights_path.exists():
        entry["weights"] = str(weights_path)
        entry["weights_bytes"] = weights_path.stat().st_size
        entry["weights_sha256"] = _sha256_file(weights_path)
    return entry


def generate_index(checkpoint_root: Path) -> dict:
    entries = []
    for directory in sorted(checkpoint_root.iterdir()):
        if not directory.is_dir():
            continue
        if not (directory / "model.json").exists():
            continue
        entries.append(_inspect_checkpoint(directory))

    by_id = {entry["id"]: entry for entry in entries}
    statuses = {
        entry["id"]: (
            "v4_ready"
            if entry.get("has_logit_parity") and entry.get("has_publish_receipt")
            else "v1_superseded"
        )
        for entry in entries
    }
    return {
        "schema": "cppmega_mlx_converted_index_v1",
        "checkpoint_root": str(checkpoint_root),
        "count": len(entries),
        "statuses": statuses,
        "checkpoints": by_id,
    }


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint-root",
        default=str(DEFAULT_CHECKPOINT_ROOT),
        help="Root directory containing converted checkpoint subdirectories.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_INDEX_PATH),
        help="Path to write index.json.",
    )
    args = parser.parse_args(argv)

    index = generate_index(Path(args.checkpoint_root))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "count": index["count"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main([]))
