"""Tests for the machine-readable mlx_converted checkpoint index."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import generate_mlx_converted_index as gen


REPO_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_ROOT = Path("/Volumes/external/sources/cppmega/outputs/checkpoints/mlx_converted")
INDEX_PATH = CHECKPOINT_ROOT / "index.json"


def test_index_file_exists_and_is_valid_json() -> None:
    assert INDEX_PATH.is_file(), f"missing index: {INDEX_PATH}"
    payload = json.loads(INDEX_PATH.read_text(encoding="utf-8"))
    assert payload["schema"] == "cppmega_mlx_converted_index_v1"
    assert payload["checkpoint_root"] == str(CHECKPOINT_ROOT)
    assert payload["count"] == len(payload["checkpoints"])
    assert set(payload["statuses"]) == set(payload["checkpoints"])


def test_all_entries_have_required_fields() -> None:
    payload = json.loads(INDEX_PATH.read_text(encoding="utf-8"))
    for cid, entry in payload["checkpoints"].items():
        assert entry["id"] == cid
        for key in ("path", "manifest", "schema", "source_checkpoint", "weights"):
            assert entry.get(key), f"{cid} missing {key}"
        assert isinstance(entry["weights_bytes"], int)
        assert len(entry["weights_sha256"]) == 64


def test_status_reflects_receipt_presence() -> None:
    payload = json.loads(INDEX_PATH.read_text(encoding="utf-8"))
    for cid, entry in payload["checkpoints"].items():
        status = payload["statuses"][cid]
        ready = entry.get("has_logit_parity") and entry.get("has_publish_receipt")
        expected = "v4_ready" if ready else "v1_superseded"
        assert status == expected, f"{cid}: expected {expected}, got {status}"


def test_generate_index_is_idempotent() -> None:
    first = gen.generate_index(CHECKPOINT_ROOT)
    second = gen.generate_index(CHECKPOINT_ROOT)
    assert first == second


def test_generate_index_matches_written_file() -> None:
    generated = gen.generate_index(CHECKPOINT_ROOT)
    written = json.loads(INDEX_PATH.read_text(encoding="utf-8"))
    assert generated == written
