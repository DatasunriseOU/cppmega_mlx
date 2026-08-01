from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts import generate_mlx_converted_index as gen


def _checkpoint(root: Path, name: str, *, ready: bool) -> None:
    directory = root / name
    directory.mkdir(parents=True)
    weights = directory / "weights.safetensors"
    weights.write_bytes(name.encode("utf-8"))
    digest = hashlib.sha256(weights.read_bytes()).hexdigest()
    manifest = {
        "schema": (
            gen.CURRENT_CONVERSION_SCHEMA if ready else "legacy_conversion_v1"
        ),
        "source_checkpoint": f"/source/{name}",
        "output": str(weights),
        "dtype": "bfloat16",
        "rope_only": ready,
    }
    if ready:
        manifest["logit_parity"] = {"max_abs_logit_error": 0.0}
        manifest["publish"] = {
            "completion_marker": "model.json",
            "weights_sha256": digest,
        }
    (directory / "model.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )


def test_generate_index_is_portable_and_fail_closed(tmp_path: Path) -> None:
    root = tmp_path / "checkpoints"
    _checkpoint(root, "ready", ready=True)
    _checkpoint(root, "old", ready=False)
    _checkpoint(root, "tampered", ready=True)
    (root / "tampered" / "weights.safetensors").write_bytes(b"changed")

    payload = gen.generate_index(root)

    assert payload["schema"] == "cppmega_mlx_converted_index_v2"
    assert payload["count"] == 3
    assert payload["ready"] == 1
    assert payload["superseded"] == 2
    assert payload["checkpoints"]["ready"]["status"] == "ready"
    assert payload["checkpoints"]["ready"]["blockers"] == []
    assert payload["checkpoints"]["tampered"]["blockers"] == ["publish_receipt"]
    assert payload["checkpoints"]["old"]["status"] == "superseded"
    assert payload["checkpoints"]["old"]["blockers"] == [
        "conversion_schema",
        "logit_parity",
        "rope_only",
        "publish_receipt",
    ]


def test_main_honors_paths_and_publishes_idempotently(tmp_path: Path) -> None:
    root = tmp_path / "checkpoints"
    _checkpoint(root, "one", ready=True)
    output = tmp_path / "index.json"
    argv = ["--checkpoint-root", str(root), "--output", str(output)]

    assert gen.main(argv) == 0
    first = output.read_bytes()
    assert gen.main(argv) == 0
    assert output.read_bytes() == first
