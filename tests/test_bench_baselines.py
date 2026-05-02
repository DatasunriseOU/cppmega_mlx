from __future__ import annotations

import json
from pathlib import Path

import pytest

from cppmega_mlx.training.baselines import (
    BASELINE_INDEX_FILENAME,
    BaselineValidationError,
    archive_baseline_row,
    baseline_filename,
    validate_baseline_row,
)


def valid_local_row() -> dict[str, object]:
    return {
        "hardware": "M4 Max",
        "commit": "abc1234",
        "dtype": "bfloat16",
        "batch_size": 2,
        "seq_len": 64,
        "route": "mamba3",
        "model": "hybrid-m",
        "mode": "eager",
        "tokens_per_second": 123.5,
        "local_only": True,
        "gb10_parity_claim": False,
    }


def test_valid_local_baseline_row_archives_without_parity_claim(tmp_path: Path) -> None:
    row = valid_local_row()

    archived = archive_baseline_row(
        tmp_path,
        row,
        recorded_at_utc="2026-05-01T00:00:00Z",
    )

    assert archived["filename"] == (
        "m4-max__abc1234__bfloat16__b2__s64__"
        "route-mamba3__model-hybrid-m__eager.json"
    )
    payload = json.loads(Path(archived["path"]).read_text(encoding="utf-8"))
    assert payload["row"] == row
    assert payload["row"]["local_only"] is True
    assert payload["row"]["gb10_parity_claim"] is False
    assert payload["parity_evidence_policy"].startswith("GB10 parity claims require")


def test_invalid_baseline_row_missing_required_keys_fails_closed() -> None:
    row = valid_local_row()
    row.pop("tokens_per_second")

    with pytest.raises(BaselineValidationError, match="tokens_per_second"):
        validate_baseline_row(row)


def test_invalid_gb10_parity_claim_without_matched_evidence_fails_closed() -> None:
    row = valid_local_row()
    row["local_only"] = False
    row["gb10_parity_claim"] = True

    with pytest.raises(BaselineValidationError, match="matched evidence"):
        validate_baseline_row(row)


def test_deterministic_file_naming_and_indexing(tmp_path: Path) -> None:
    first = valid_local_row()
    second = {
        **first,
        "route": "plain",
        "model": "tiny",
        "mode": "compile",
        "seq_len": 32,
        "tokens_per_second": 99,
    }

    assert baseline_filename(first) == (
        "m4-max__abc1234__bfloat16__b2__s64__"
        "route-mamba3__model-hybrid-m__eager.json"
    )
    archive_baseline_row(
        tmp_path,
        second,
        recorded_at_utc="2026-05-01T00:00:01Z",
    )
    archive_baseline_row(
        tmp_path,
        first,
        recorded_at_utc="2026-05-01T00:00:02Z",
    )
    archive_baseline_row(
        tmp_path,
        {**first, "tokens_per_second": 125.0},
        recorded_at_utc="2026-05-01T00:00:03Z",
    )

    index_path = tmp_path / BASELINE_INDEX_FILENAME
    index = json.loads(index_path.read_text(encoding="utf-8"))
    paths = [entry["path"] for entry in index["entries"]]
    assert paths == sorted(paths)
    assert paths == [
        "m4-max__abc1234__bfloat16__b2__s32__route-plain__model-tiny__compile.json",
        "m4-max__abc1234__bfloat16__b2__s64__route-mamba3__model-hybrid-m__eager.json",
    ]
    updated_entry = index["entries"][1]
    assert updated_entry["tokens_per_second"] == 125.0
    assert updated_entry["recorded_at_utc"] == "2026-05-01T00:00:03Z"

