"""V7-G02: stable doc_id assignment across shards.

Today the corpus pipeline re-assigns doc_ids per shard, so a long
document split across shards N and N+1 ends up with two distinct
ids — breaking attention masking and FIM continuity.

`assign_stable_doc_ids` derives the doc id from a stable signature
(the canonical text content) so the SAME logical document gets the
SAME id regardless of which shard contains its rows.

  assign_stable_doc_ids(rows, signature_fn) → list[int]

The signature_fn extracts the stable identity for each row (default:
SHA-256 of the row's 'text' field). Rows with the same signature
share an id; new signatures get a monotonic counter.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence


_EXPLICIT_DOC_ID_COLUMNS = (
    "source_doc_id",
    "source_document_id",
    "document_id",
    "doc_id",
)
_PROVENANCE_SIGNATURE_COLUMNS = (
    "repo_stable_id",
    "filepath_stable_id",
    "commit_hash",
    "file_local_commit_index",
)


@dataclass(frozen=True)
class ShardedDocIdAssignment:
    """Stable doc ids plus a JSON-ready per-corpus span manifest."""

    doc_ids_by_shard: tuple[tuple[int, ...], ...]
    manifest: dict[str, list[dict[str, int]]]
    signature_to_doc_id: dict[str, int]


def stable_doc_signature(row: Mapping[str, Any]) -> str:
    """Return the stable logical-document signature for a row.

    Explicit source/document id columns win. If they are absent, use commit/file
    provenance when complete enough to identify the logical document; otherwise
    fall back to a text hash for backwards compatibility with older fixtures.
    """

    for column in _EXPLICIT_DOC_ID_COLUMNS:
        value = row.get(column)
        if value is not None:
            return f"{column}:{value}"

    provenance = tuple(row.get(column) for column in _PROVENANCE_SIGNATURE_COLUMNS)
    if any(value is not None for value in provenance):
        return "provenance:" + "\0".join("" if value is None else str(value)
                                        for value in provenance)

    return _text_signature(row)


def _text_signature(row: Mapping[str, Any]) -> str:
    text = row.get("text", "")
    if not isinstance(text, str):
        text = repr(text)
    return "text_sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def assign_stable_doc_ids(
    rows: Iterable[Mapping[str, Any]],
    *,
    signature_fn: Callable[[Mapping[str, Any]], str] = stable_doc_signature,
    seed_ids: Mapping[str, int] | None = None,
) -> list[int]:
    """Return a doc_id for each row, deterministic in row signature.

    Args:
        rows: any iterable of dict-like rows.
        signature_fn: extracts the stable doc identity from a row.
        seed_ids: optional pre-existing signature → id map from
            earlier shards. Allows cross-shard continuity by feeding
            the previous shard's manifest into the next call.
    """
    ids: list[int] = []
    seen: dict[str, int] = dict(seed_ids or {})
    counter = max(seen.values(), default=-1) + 1
    for row in rows:
        sig = signature_fn(row)
        if sig in seen:
            ids.append(seen[sig])
            continue
        seen[sig] = counter
        ids.append(counter)
        counter += 1
    return ids


def manifest_from_assignment(rows: Iterable[Mapping[str, Any]],
                              ids: list[int],
                              *, signature_fn: Callable[[Mapping[str, Any]], str]
                              = stable_doc_signature
                              ) -> dict[str, int]:
    """Round-trippable signature→id map for handing to the next shard."""
    out: dict[str, int] = {}
    for row, doc_id in zip(rows, ids):
        out[signature_fn(row)] = int(doc_id)
    return out


def assign_sharded_doc_ids(
    shards: Sequence[Sequence[Mapping[str, Any]]],
    *,
    signature_fn: Callable[[Mapping[str, Any]], str] = stable_doc_signature,
    seed_ids: Mapping[str, int] | None = None,
) -> ShardedDocIdAssignment:
    """Assign stable ids across all shards and emit doc_id→row-span manifest.

    The manifest is JSON-ready and keyed by stringified integer doc_id:
    ``{"3": [{"shard_index": 0, "start_row": 4, "end_row": 7}]}``.
    Adjacent rows with the same doc id inside a shard are coalesced into one
    half-open row span. If a row carries ``byte_offset``, the first offset in
    the span is preserved for producer/inspector alignment.
    """

    seen: dict[str, int] = dict(seed_ids or {})
    counter = max(seen.values(), default=-1) + 1
    ids_by_shard: list[tuple[int, ...]] = []
    manifest: dict[str, list[dict[str, int]]] = {}

    for shard_index, rows in enumerate(shards):
        shard_ids: list[int] = []
        active_doc_id: int | None = None
        active_start = 0
        active_byte_offset: int | None = None
        row_list = list(rows)

        def close_span(end_row: int) -> None:
            if active_doc_id is None:
                return
            span = {
                "shard_index": int(shard_index),
                "start_row": int(active_start),
                "end_row": int(end_row),
            }
            if active_byte_offset is not None:
                span["byte_offset"] = int(active_byte_offset)
            manifest.setdefault(str(active_doc_id), []).append(span)

        for row_index, row in enumerate(row_list):
            signature = signature_fn(row)
            doc_id = seen.get(signature)
            if doc_id is None:
                doc_id = counter
                seen[signature] = doc_id
                counter += 1

            shard_ids.append(int(doc_id))
            if active_doc_id != doc_id:
                close_span(row_index)
                active_doc_id = int(doc_id)
                active_start = row_index
                raw_offset = row.get("byte_offset")
                active_byte_offset = int(raw_offset) if raw_offset is not None else None
        close_span(len(row_list))
        ids_by_shard.append(tuple(shard_ids))

    return ShardedDocIdAssignment(
        doc_ids_by_shard=tuple(ids_by_shard),
        manifest=manifest,
        signature_to_doc_id=dict(seen),
    )


def write_doc_id_manifest(
    path: str | Path,
    assignment: ShardedDocIdAssignment,
) -> None:
    """Write a JSON sidecar for trainer/DataInspector cross-shard agreement."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "cppmega_doc_id_manifest_v1",
        "manifest": assignment.manifest,
        "signature_to_doc_id": assignment.signature_to_doc_id,
    }
    target.write_text(json.dumps(payload, sort_keys=True, indent=2), encoding="utf-8")


__all__ = [
    "ShardedDocIdAssignment",
    "assign_sharded_doc_ids",
    "assign_stable_doc_ids",
    "manifest_from_assignment",
    "stable_doc_signature",
    "write_doc_id_manifest",
]
