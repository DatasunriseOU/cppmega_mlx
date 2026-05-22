from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "megatron_ingress_stress.py"


def run_script(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )


def load_json(result: subprocess.CompletedProcess[str]) -> dict[str, Any]:
    payload = json.loads(result.stdout)
    assert isinstance(payload, dict)
    return payload


def test_megatron_ingress_stress_generates_multishard_sidecar_receipt(
    tmp_path: Path,
) -> None:
    result = run_script(
        "--output-dir",
        str(tmp_path),
        "--token-count",
        "4096",
        "--shards",
        "2",
        "--seq-len",
        "128",
        "--batch-size",
        "2",
        "--batches",
        "3",
        "--chunk-tokens",
        "257",
        "--include-document-ids",
        "--include-structure-ids",
    )

    assert result.returncode == 0, result.stderr
    payload = load_json(result)
    assert payload["status"] == "ok"
    assert payload["receipt_scope"] == "local_megatron_indexed_ingress_stress"
    assert payload["token_count"] == 4096
    assert payload["shards"] == 2
    assert payload["tokens_read"] == 768
    assert payload["dataset"]["num_samples"] == 32
    assert payload["dataset"]["metadata"]["source_format"] == "megatron-multishard"
    assert payload["dataset"]["index_metadata"]["source_format"] == "megatron-multishard"
    assert payload["dataset"]["index_metadata"]["shard_count"] == 2
    assert payload["dataset"]["token_id_range"] == [0, 4095]
    assert payload["side_channel_presence"]["document_ids"] is True
    assert payload["side_channel_presence"]["structure_ids"] is True
    assert payload["memory_peak_within_limit"] is True
    assert payload["distributed_megatron_parity_claim"] is False
    assert payload["gb10_parity_claim"] is False


def test_megatron_ingress_stress_fails_memory_ceiling_with_json(
    tmp_path: Path,
) -> None:
    result = run_script(
        "--output-dir",
        str(tmp_path),
        "--token-count",
        "1024",
        "--seq-len",
        "128",
        "--batch-size",
        "1",
        "--batches",
        "1",
        "--max-peak-bytes",
        "1",
    )

    assert result.returncode == 2
    payload = load_json(result)
    assert payload["status"] == "error"
    assert payload["memory_peak_within_limit"] is False
    assert "peak memory exceeded" in payload["error"]
