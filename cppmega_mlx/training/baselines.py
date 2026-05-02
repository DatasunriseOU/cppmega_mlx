"""Local benchmark baseline archive helpers.

These helpers intentionally archive evidence rows only. They do not compute or
claim throughput parity; matched-run comparison remains the job of
``scripts/compare_bench_rows.py``.
"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


BASELINE_INDEX_SCHEMA_VERSION = 1
BASELINE_ROW_SCHEMA_VERSION = 1
BASELINE_ARCHIVE_KIND = "cppmega.mlx.benchmark_baseline"
BASELINE_INDEX_KIND = "cppmega.mlx.benchmark_baseline_index"
BASELINE_INDEX_FILENAME = "index.json"

REQUIRED_BASELINE_ROW_KEYS = (
    "hardware",
    "commit",
    "dtype",
    "batch_size",
    "seq_len",
    "route",
    "model",
    "mode",
    "tokens_per_second",
    "local_only",
    "gb10_parity_claim",
)
VALID_MODES = {"compile", "compiled", "eager"}
PARITY_EVIDENCE_KEYS = (
    "matched_m4_row",
    "matched_gb10_row",
    "matched_comparison_key",
    "comparison_report",
)
PARITY_EVIDENCE_POLICY = (
    "GB10 parity claims require explicit matched_m4_row, matched_gb10_row, "
    "matched_comparison_key, and comparison_report evidence fields."
)


class BaselineValidationError(ValueError):
    """Raised when a benchmark baseline row is unsafe to archive."""


def validate_baseline_row(row: dict[str, Any]) -> dict[str, Any]:
    """Return a normalized benchmark row or raise on missing/unsafe fields."""

    if not isinstance(row, dict):
        raise BaselineValidationError("baseline row must be an object")

    missing = [key for key in REQUIRED_BASELINE_ROW_KEYS if key not in row]
    if missing:
        raise BaselineValidationError(
            f"baseline row missing required key(s): {', '.join(missing)}"
        )

    normalized = dict(row)
    for key in ("hardware", "commit", "dtype", "route", "model", "mode"):
        value = normalized[key]
        if not isinstance(value, str) or not value.strip():
            raise BaselineValidationError(f"{key} must be a non-empty string")
        normalized[key] = value.strip()

    if normalized["mode"] not in VALID_MODES:
        raise BaselineValidationError(
            f"mode must be one of {', '.join(sorted(VALID_MODES))}"
        )

    for key in ("batch_size", "seq_len"):
        value = normalized[key]
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise BaselineValidationError(f"{key} must be a positive integer")

    tokens_per_second = normalized["tokens_per_second"]
    if not isinstance(tokens_per_second, (int, float)) or isinstance(
        tokens_per_second, bool
    ):
        raise BaselineValidationError("tokens_per_second must be a positive number")
    if tokens_per_second <= 0:
        raise BaselineValidationError("tokens_per_second must be a positive number")
    normalized["tokens_per_second"] = float(tokens_per_second)

    if not isinstance(normalized["local_only"], bool):
        raise BaselineValidationError("local_only must be a boolean")
    if not isinstance(normalized["gb10_parity_claim"], bool):
        raise BaselineValidationError("gb10_parity_claim must be a boolean")

    if normalized["gb10_parity_claim"]:
        _validate_parity_evidence(normalized)
    elif normalized["local_only"] is not True:
        raise BaselineValidationError(
            "non-parity baseline rows must set local_only to true"
        )

    return normalized


def archive_baseline_row(
    baseline_dir: str | Path,
    row: dict[str, Any],
    *,
    recorded_at_utc: str | None = None,
) -> dict[str, Any]:
    """Validate and archive one benchmark row with deterministic naming.

    The row file is addressed by hardware, commit, dtype, batch, seq, route,
    model, and mode. Re-archiving the same row updates that file and replaces the
    matching index entry instead of appending a duplicate.
    """

    normalized = validate_baseline_row(row)
    recorded_at = recorded_at_utc or _utc_now()
    directory = Path(baseline_dir)
    filename = baseline_filename(normalized)
    relative_path = filename
    archive_path = directory / filename
    archived_row = {
        "schema_version": BASELINE_ROW_SCHEMA_VERSION,
        "kind": BASELINE_ARCHIVE_KIND,
        "recorded_at_utc": recorded_at,
        "parity_evidence_policy": PARITY_EVIDENCE_POLICY,
        "row": normalized,
    }

    directory.mkdir(parents=True, exist_ok=True)
    _write_json(archive_path, archived_row)
    index = _load_index(directory)
    entry = _index_entry(
        normalized,
        path=relative_path,
        recorded_at_utc=recorded_at,
    )
    _upsert_index_entry(index, entry)
    index["updated_at_utc"] = recorded_at
    _write_json(directory / BASELINE_INDEX_FILENAME, index)
    return {
        "path": str(archive_path),
        "index_path": str(directory / BASELINE_INDEX_FILENAME),
        "filename": filename,
        "index": index,
        "entry": entry,
        "archive": archived_row,
    }


def baseline_filename(row: dict[str, Any]) -> str:
    """Return the deterministic archive filename for a validated row."""

    normalized = validate_baseline_row(row)
    parts = (
        normalized["hardware"],
        normalized["commit"],
        normalized["dtype"],
        f"b{normalized['batch_size']}",
        f"s{normalized['seq_len']}",
        f"route-{normalized['route']}",
        f"model-{normalized['model']}",
        normalized["mode"],
    )
    return "__".join(_slug(part) for part in parts) + ".json"


def _validate_parity_evidence(row: dict[str, Any]) -> None:
    if row.get("local_only") is True:
        raise BaselineValidationError(
            "gb10_parity_claim rows must not also be local_only"
        )
    missing = [key for key in PARITY_EVIDENCE_KEYS if key not in row]
    if missing:
        raise BaselineValidationError(
            "gb10_parity_claim requires matched evidence field(s): "
            + ", ".join(missing)
        )
    if not isinstance(row["matched_m4_row"], dict) or not row["matched_m4_row"]:
        raise BaselineValidationError("matched_m4_row must be a non-empty object")
    if not isinstance(row["matched_gb10_row"], dict) or not row["matched_gb10_row"]:
        raise BaselineValidationError("matched_gb10_row must be a non-empty object")
    if (
        not isinstance(row["matched_comparison_key"], dict)
        or not row["matched_comparison_key"]
    ):
        raise BaselineValidationError(
            "matched_comparison_key must be a non-empty object"
        )
    if not isinstance(row["comparison_report"], dict) or not row["comparison_report"]:
        raise BaselineValidationError("comparison_report must be a non-empty object")


def _slug(value: Any) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value).strip().lower())
    slug = slug.strip("-._")
    if not slug:
        raise BaselineValidationError("baseline filename component is empty")
    return slug


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _new_index() -> dict[str, Any]:
    return {
        "schema_version": BASELINE_INDEX_SCHEMA_VERSION,
        "kind": BASELINE_INDEX_KIND,
        "updated_at_utc": None,
        "parity_evidence_policy": PARITY_EVIDENCE_POLICY,
        "entries": [],
    }


def _load_index(directory: Path) -> dict[str, Any]:
    path = directory / BASELINE_INDEX_FILENAME
    if not path.exists():
        return _new_index()
    with path.open("r", encoding="utf-8") as fh:
        index = json.load(fh)
    if index.get("schema_version") != BASELINE_INDEX_SCHEMA_VERSION:
        raise BaselineValidationError(
            f"unsupported baseline index schema_version {index.get('schema_version')!r}"
        )
    if index.get("kind") != BASELINE_INDEX_KIND:
        raise BaselineValidationError(
            f"unsupported baseline index kind {index.get('kind')!r}"
        )
    if not isinstance(index.get("entries"), list):
        raise BaselineValidationError("baseline index entries must be a list")
    return index


def _index_entry(
    row: dict[str, Any],
    *,
    path: str,
    recorded_at_utc: str,
) -> dict[str, Any]:
    return {
        "path": path,
        "recorded_at_utc": recorded_at_utc,
        "hardware": row["hardware"],
        "commit": row["commit"],
        "dtype": row["dtype"],
        "batch_size": row["batch_size"],
        "seq_len": row["seq_len"],
        "route": row["route"],
        "model": row["model"],
        "mode": row["mode"],
        "tokens_per_second": row["tokens_per_second"],
        "local_only": row["local_only"],
        "gb10_parity_claim": row["gb10_parity_claim"],
    }


def _upsert_index_entry(index: dict[str, Any], entry: dict[str, Any]) -> None:
    entries = index["entries"]
    path = entry["path"]
    for existing_index, existing in enumerate(entries):
        if isinstance(existing, dict) and existing.get("path") == path:
            entries[existing_index] = entry
            break
    else:
        entries.append(entry)
    entries.sort(key=lambda item: item["path"])


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.write("\n")

