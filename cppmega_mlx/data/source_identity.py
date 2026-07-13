"""Stable positive source identity, separate from document boundaries."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
from typing import Any


MAX_SOURCE_ID = (1 << 32) - 1


def _hash_source_signature(signature: str) -> int:
    digest = hashlib.sha256(signature.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % MAX_SOURCE_ID + 1


def stable_source_identity_id(record: Mapping[str, Any]) -> int:
    """Return a deterministic positive uint32 identity for a physical source.

    This deliberately ignores packed ``doc_ids`` and numeric ``source_doc_id``:
    those identify logical documents or row-local segment boundaries, not the
    source file from which multiple documents may have been assembled.
    """

    explicit = record.get("source_identity_id")
    if explicit is not None:
        value = int(explicit)
        if not 0 < value <= MAX_SOURCE_ID:
            raise ValueError(
                f"source_identity_id must be in [1, {MAX_SOURCE_ID}], got {value}"
            )
        return value

    signature_fields = {
        key: record.get(key)
        for key in (
            "repo_stable_id",
            "filepath_stable_id",
            "repo",
            "filepath",
            "source_path",
        )
        if record.get(key) not in (None, "")
    }
    legacy_source_doc_id = record.get("source_doc_id")
    if not signature_fields and isinstance(legacy_source_doc_id, str) and legacy_source_doc_id:
        signature_fields["legacy_source_doc_id"] = legacy_source_doc_id
    if signature_fields:
        signature = json.dumps(signature_fields, sort_keys=True, separators=(",", ":"))
    else:
        content = record.get("text", record.get("source_text"))
        if content is None:
            content = record.get("token_ids", [])
        serialized = (
            content
            if isinstance(content, str)
            else json.dumps(content, separators=(",", ":"), default=str)
        )
        signature = "content:" + hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    return _hash_source_signature(signature)


def stable_source_identity_for_path(
    path: str | Path,
    *,
    text: str = "",
    source_root: str | Path | None = None,
) -> int:
    path_obj = Path(path)
    if source_root is not None:
        try:
            source_path = path_obj.resolve().relative_to(Path(source_root).resolve()).as_posix()
        except (OSError, ValueError):
            source_path = path_obj.as_posix()
    else:
        source_path = path_obj.as_posix()
    return stable_source_identity_id({"source_path": source_path, "text": text})


def normalize_positive_source_ids(
    values: Sequence[Any] | None,
    *,
    length: int,
    fallback_source_id: int,
) -> list[int]:
    """Validate a source vector and repair legacy zero sentinels deterministically."""

    fallback = int(fallback_source_id)
    if not 0 < fallback <= MAX_SOURCE_ID:
        raise ValueError(
            f"fallback_source_id must be in [1, {MAX_SOURCE_ID}], got {fallback}"
        )
    if not values:
        return [fallback] * length
    if len(values) != length:
        raise ValueError(
            f"source identity vector length {len(values)} != expected {length}"
        )
    normalized: list[int] = []
    for raw in values:
        value = int(raw)
        if value < 0 or value > MAX_SOURCE_ID:
            raise ValueError(f"source identity must be uint32, got {value}")
        normalized.append(value or fallback)
    return normalized


__all__ = [
    "MAX_SOURCE_ID",
    "normalize_positive_source_ids",
    "stable_source_identity_for_path",
    "stable_source_identity_id",
]
