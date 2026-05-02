"""Reference sequence-packing helpers for local MLX data ingress.

This module intentionally stays tokenizer-free: callers pass token ID sequences
and the EOS ID from their tokenizer contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np

OversizedSamplePolicy = Literal["refuse", "truncate"]


@dataclass(frozen=True)
class PackedSequences:
    """Fixed-length concat-with-EOS packs plus document-boundary metadata."""

    tokens: np.ndarray
    token_mask: np.ndarray
    doc_ids: np.ndarray
    boundary_mask: np.ndarray


def pack_documents_with_eos(
    documents: Sequence[Sequence[int] | np.ndarray],
    *,
    seq_len: int,
    eos_token_id: int,
    pad_token_id: int = 0,
    oversized: OversizedSamplePolicy = "refuse",
) -> PackedSequences:
    """Pack tokenized documents into fixed-length rows with EOS separators.

    Documents are concatenated in input order.  Each document receives one EOS
    token unless it already ends with ``eos_token_id``.  Oversized documents are
    refused by default; with ``oversized="truncate"`` they are clipped to
    ``seq_len`` and the final token is forced to EOS.
    """

    if seq_len < 2:
        raise ValueError("seq_len must be at least 2")
    if oversized not in {"refuse", "truncate"}:
        raise ValueError("oversized must be 'refuse' or 'truncate'")
    _validate_token_id("eos_token_id", eos_token_id)
    _validate_token_id("pad_token_id", pad_token_id)

    packed_rows: list[np.ndarray] = []
    packed_masks: list[np.ndarray] = []
    current = np.full(seq_len, pad_token_id, dtype=np.int32)
    current_mask = np.zeros(seq_len, dtype=np.bool_)
    offset = 0

    for sample_idx, document in enumerate(documents):
        sample = _document_with_eos(
            document,
            eos_token_id=eos_token_id,
            sample_idx=sample_idx,
        )
        if sample.shape[0] > seq_len:
            if oversized == "refuse":
                raise ValueError(
                    f"sample {sample_idx} length {sample.shape[0]} exceeds seq_len "
                    f"{seq_len}; use oversized='truncate' to clip it"
                )
            sample = sample[:seq_len].copy()
            sample[-1] = eos_token_id

        if offset and offset + sample.shape[0] > seq_len:
            packed_rows.append(current)
            packed_masks.append(current_mask)
            current = np.full(seq_len, pad_token_id, dtype=np.int32)
            current_mask = np.zeros(seq_len, dtype=np.bool_)
            offset = 0

        end = offset + sample.shape[0]
        current[offset:end] = sample
        current_mask[offset:end] = True
        offset = end

        if offset == seq_len:
            packed_rows.append(current)
            packed_masks.append(current_mask)
            current = np.full(seq_len, pad_token_id, dtype=np.int32)
            current_mask = np.zeros(seq_len, dtype=np.bool_)
            offset = 0

    if offset:
        packed_rows.append(current)
        packed_masks.append(current_mask)

    if packed_rows:
        tokens = np.stack(packed_rows).astype(np.int32, copy=False)
        token_mask = np.stack(packed_masks).astype(np.bool_, copy=False)
    else:
        tokens = np.empty((0, seq_len), dtype=np.int32)
        token_mask = np.empty((0, seq_len), dtype=np.bool_)

    doc_ids = cumulative_doc_ids_from_eos(
        tokens,
        eos_token_id=eos_token_id,
        token_mask=token_mask,
    )
    boundary_mask = document_boundary_mask(doc_ids, token_mask=token_mask)
    return PackedSequences(
        tokens=tokens,
        token_mask=token_mask,
        doc_ids=doc_ids,
        boundary_mask=boundary_mask,
    )


def cumulative_doc_ids_from_eos(
    tokens: Sequence[Sequence[int]] | np.ndarray,
    *,
    eos_token_id: int,
    token_mask: Sequence[Sequence[bool]] | np.ndarray | None = None,
    pad_doc_id: int = -1,
) -> np.ndarray:
    """Build per-token document IDs from cumulative previous EOS positions."""

    _validate_token_id("eos_token_id", eos_token_id)
    token_array = _as_2d_integer_array("tokens", tokens)
    eos_hits = token_array == eos_token_id
    doc_ids = np.zeros(token_array.shape, dtype=np.int32)
    if token_array.shape[1] > 1:
        doc_ids[:, 1:] = np.cumsum(eos_hits[:, :-1], axis=1, dtype=np.int32)

    if token_mask is not None:
        mask_array = _as_2d_bool_array("token_mask", token_mask)
        if mask_array.shape != token_array.shape:
            raise ValueError(
                f"token_mask shape must match tokens {token_array.shape}, "
                f"got {mask_array.shape}"
            )
        doc_ids = np.where(mask_array, doc_ids, pad_doc_id).astype(
            np.int32,
            copy=False,
        )
    return doc_ids


def document_boundary_mask(
    doc_ids: Sequence[Sequence[int]] | np.ndarray,
    *,
    token_mask: Sequence[Sequence[bool]] | np.ndarray | None = None,
    causal: bool = False,
) -> np.ndarray:
    """Return a ``(B, S, S)`` mask that blocks attention across documents."""

    doc_id_array = _as_2d_integer_array("doc_ids", doc_ids, allow_negative=True)
    same_doc = doc_id_array[:, :, None] == doc_id_array[:, None, :]

    if token_mask is not None:
        mask_array = _as_2d_bool_array("token_mask", token_mask)
        if mask_array.shape != doc_id_array.shape:
            raise ValueError(
                f"token_mask shape must match doc_ids {doc_id_array.shape}, "
                f"got {mask_array.shape}"
            )
        same_doc &= mask_array[:, :, None] & mask_array[:, None, :]

    if causal:
        seq_len = doc_id_array.shape[1]
        same_doc &= np.tril(np.ones((seq_len, seq_len), dtype=np.bool_))[None, :, :]
    return same_doc.astype(np.bool_, copy=False)


def _document_with_eos(
    document: Sequence[int] | np.ndarray,
    *,
    eos_token_id: int,
    sample_idx: int,
) -> np.ndarray:
    sample = _as_1d_integer_array(f"sample {sample_idx}", document)
    if sample.shape[0] and sample[-1] == eos_token_id:
        return sample.astype(np.int32, copy=True)
    return np.concatenate(
        [sample.astype(np.int32, copy=True), np.array([eos_token_id], dtype=np.int32)]
    )


def _validate_token_id(name: str, value: int) -> None:
    if not isinstance(value, int):
        raise TypeError(f"{name} must be an int")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    if value > np.iinfo(np.int32).max:
        raise ValueError(f"{name} exceeds int32 range")


def _as_1d_integer_array(name: str, values: Sequence[int] | np.ndarray) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a 1D token sequence, got shape {array.shape}")
    _require_integer_values(name, array)
    return array


def _as_2d_integer_array(
    name: str,
    values: Sequence[Sequence[int]] | np.ndarray,
    *,
    allow_negative: bool = False,
) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 2:
        raise ValueError(f"{name} must be 2D, got shape {array.shape}")
    _require_integer_values(name, array, allow_negative=allow_negative)
    return array.astype(np.int32, copy=False)


def _as_2d_bool_array(
    name: str,
    values: Sequence[Sequence[bool]] | np.ndarray,
) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 2:
        raise ValueError(f"{name} must be 2D, got shape {array.shape}")
    return array.astype(np.bool_, copy=False)


def _require_integer_values(
    name: str,
    values: np.ndarray,
    *,
    allow_negative: bool = False,
) -> None:
    if not np.issubdtype(values.dtype, np.integer):
        raise ValueError(f"{name} must use an integer dtype")
    if not allow_negative and np.any(values < 0):
        raise ValueError(f"{name} token IDs must be non-negative")
    if np.any(values > np.iinfo(np.int32).max):
        raise ValueError(f"{name} token IDs exceed int32 range")


__all__ = [
    "OversizedSamplePolicy",
    "PackedSequences",
    "cumulative_doc_ids_from_eos",
    "document_boundary_mask",
    "pack_documents_with_eos",
]
