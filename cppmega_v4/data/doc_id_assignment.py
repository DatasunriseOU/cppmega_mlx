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

import hashlib
from typing import Callable, Iterable, Mapping


def _default_signature(row: Mapping) -> str:
    text = row.get("text", "")
    if not isinstance(text, str):
        text = repr(text)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def assign_stable_doc_ids(
    rows: Iterable[Mapping],
    *,
    signature_fn: Callable[[Mapping], str] = _default_signature,
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


def manifest_from_assignment(rows: Iterable[Mapping],
                              ids: list[int],
                              *, signature_fn: Callable[[Mapping], str]
                              = _default_signature
                              ) -> dict[str, int]:
    """Round-trippable signature→id map for handing to the next shard."""
    out: dict[str, int] = {}
    for row, doc_id in zip(rows, ids):
        out[signature_fn(row)] = int(doc_id)
    return out


__all__ = ["assign_stable_doc_ids", "manifest_from_assignment"]
