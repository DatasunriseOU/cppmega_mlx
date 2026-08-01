#!/usr/bin/env python3
"""Generate a machine-readable index of converted MLX checkpoints."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from scripts.nanochat_data.atomic_publish import atomic_output_file  # noqa: E402

DEFAULT_CHECKPOINT_ROOT = Path(
    "/Volumes/external/sources/cppmega/outputs/checkpoints/mlx_converted"
)
DEFAULT_INDEX_PATH = DEFAULT_CHECKPOINT_ROOT / "index.json"
CURRENT_CONVERSION_SCHEMA = "cppmega_megatron_dense500m_to_mlx_v4"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _weights_path(directory: Path, manifest: dict[str, object]) -> Path | None:
    output = manifest.get("output")
    if isinstance(output, str):
        candidate = directory / Path(output).name
        if candidate.is_file():
            return candidate
    candidates = sorted(directory.glob("*.safetensors"))
    return candidates[0] if len(candidates) == 1 else None


def _inspect_checkpoint(directory: Path) -> dict[str, object]:
    manifest_path = directory / "model.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError(f"{manifest_path}: expected a JSON object")
    weights_path = _weights_path(directory, manifest)
    actual_sha256 = _sha256_file(weights_path) if weights_path is not None else None
    publish = manifest.get("publish")
    has_publish_receipt = isinstance(publish, dict)
    published_sha256 = publish.get("weights_sha256") if has_publish_receipt else None
    reasons = []
    if manifest.get("schema") != CURRENT_CONVERSION_SCHEMA:
        reasons.append("conversion_schema")
    if not isinstance(manifest.get("logit_parity"), dict):
        reasons.append("logit_parity")
    if manifest.get("rope_only") is not True:
        reasons.append("rope_only")
    if weights_path is None:
        reasons.append("weights")
    if (
        not has_publish_receipt
        or publish.get("completion_marker") != "model.json"
        or published_sha256 != actual_sha256
    ):
        reasons.append("publish_receipt")

    entry: dict[str, object] = {
        "id": directory.name,
        "path": str(directory),
        "manifest": str(manifest_path),
        "schema": manifest.get("schema"),
        "source_checkpoint": manifest.get("source_checkpoint"),
        "dtype": manifest.get("dtype"),
        "has_logit_parity": isinstance(manifest.get("logit_parity"), dict),
        "has_publish_receipt": has_publish_receipt,
        "rope_only": manifest.get("rope_only"),
        "status": "ready" if not reasons else "superseded",
        "blockers": reasons,
    }
    if weights_path is not None:
        entry["weights"] = str(weights_path)
        entry["weights_bytes"] = weights_path.stat().st_size
        entry["weights_sha256"] = actual_sha256
    return entry


def generate_index(checkpoint_root: Path) -> dict[str, object]:
    if not checkpoint_root.is_dir():
        raise FileNotFoundError(checkpoint_root)
    entries = []
    for directory in sorted(checkpoint_root.iterdir()):
        if not directory.is_dir():
            continue
        if not (directory / "model.json").exists():
            continue
        entries.append(_inspect_checkpoint(directory))

    by_id = {entry["id"]: entry for entry in entries}
    return {
        "schema": "cppmega_mlx_converted_index_v2",
        "checkpoint_root": str(checkpoint_root),
        "count": len(entries),
        "ready": sum(entry["status"] == "ready" for entry in entries),
        "superseded": sum(entry["status"] == "superseded" for entry in entries),
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
    with atomic_output_file(output) as staged:
        staged.write_text(
            json.dumps(index, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps({"output": str(output), "count": index["count"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
