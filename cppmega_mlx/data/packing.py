"""Reference sequence-packing helpers for local MLX data ingress.

This module intentionally stays tokenizer-free: callers pass token ID sequences
and the BOS/EOS IDs from their tokenizer contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence, cast

import mlx.core as mx
import numpy as np

OversizedSamplePolicy = Literal["refuse", "truncate"]
PackingStrategy = Literal["best_fit", "sequential"]


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
    bos_token_id: int | None = None,
    pad_token_id: int = 0,
    oversized: OversizedSamplePolicy = "refuse",
    strategy: PackingStrategy = "sequential",
) -> PackedSequences:
    """Pack tokenized documents into fixed-length rows with EOS separators.

    ``strategy="sequential"`` preserves input-order concat behavior.
    ``strategy="best_fit"`` emits BOS-aligned rows by deterministically filling
    each pack with the largest remaining document that fits; ties keep input
    order.
    Every document receives one EOS token unless it already ends with
    ``eos_token_id``. Oversized documents are refused by default; with
    ``oversized="truncate"`` they are clipped to the usable row length and the
    final token is forced to EOS.
    """

    _validate_pack_options(
        seq_len=seq_len,
        eos_token_id=eos_token_id,
        bos_token_id=bos_token_id,
        pad_token_id=pad_token_id,
        oversized=oversized,
        strategy=strategy,
    )
    samples = _prepare_samples(
        documents,
        seq_len=seq_len,
        eos_token_id=eos_token_id,
        bos_token_id=bos_token_id,
        oversized=oversized,
    )
    if strategy == "sequential":
        return _pack_prepared_sequential(
            samples,
            seq_len=seq_len,
            eos_token_id=eos_token_id,
            bos_token_id=bos_token_id,
            pad_token_id=pad_token_id,
        )
    return _pack_prepared_best_fit(
        samples,
        seq_len=seq_len,
        eos_token_id=eos_token_id,
        bos_token_id=bos_token_id,
        pad_token_id=pad_token_id,
    )


def pack_bos_aligned_best_fit(
    documents: Sequence[Sequence[int] | np.ndarray],
    *,
    seq_len: int,
    eos_token_id: int,
    bos_token_id: int | None = None,
    pad_token_id: int = 0,
    oversized: OversizedSamplePolicy = "refuse",
) -> PackedSequences:
    """Pack documents with deterministic BOS-aligned best-fit placement."""

    return pack_documents_with_eos(
        documents,
        seq_len=seq_len,
        eos_token_id=eos_token_id,
        bos_token_id=bos_token_id,
        pad_token_id=pad_token_id,
        oversized=oversized,
        strategy="best_fit",
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
    expand_heads: bool = False,
) -> np.ndarray:
    """Return a boolean same-document mask.

    With ``expand_heads=True`` the shape is ``(B, 1, S, S)``, which broadcasts
    over attention heads for MLX SDPA without converting to an additive fp32
    mask.
    """

    doc_id_array = _as_2d_integer_array("doc_ids", doc_ids, allow_negative=True)
    valid_doc = doc_id_array >= 0
    same_doc = doc_id_array[:, :, None] == doc_id_array[:, None, :]
    same_doc &= valid_doc[:, :, None] & valid_doc[:, None, :]

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

    same_doc = same_doc.astype(np.bool_, copy=False)
    if expand_heads:
        return same_doc[:, None, :, :]
    return same_doc


def mlx_cumulative_doc_ids_from_eos(
    tokens: mx.array,
    *,
    eos_token_id: int,
    token_mask: mx.array | None = None,
    pad_doc_id: int = -1,
) -> mx.array:
    """MLX cumulative doc IDs using previous-EOS ``mx.cumsum`` semantics."""

    _validate_token_id("eos_token_id", eos_token_id)
    if len(tokens.shape) != 2:
        raise ValueError(f"tokens must be 2D, got shape {tokens.shape}")

    token_array = tokens.astype(mx.int32)
    eos_hits = cast(mx.array, token_array == eos_token_id)
    zeros = mx.zeros(token_array[:, :1].shape, dtype=mx.int32)
    if token_array.shape[1] == 0:
        doc_ids = token_array.astype(mx.int32)
    else:
        previous_eos = cast(mx.array, eos_hits[:, :-1]).astype(mx.int32)
        doc_ids = mx.concatenate(
            [zeros, mx.cumsum(previous_eos, axis=1)],
            axis=1,
        )

    if token_mask is not None:
        if token_mask.shape != token_array.shape:
            raise ValueError(
                f"token_mask shape must match tokens {token_array.shape}, "
                f"got {token_mask.shape}"
            )
        doc_ids = mx.where(token_mask.astype(mx.bool_), doc_ids, pad_doc_id)
    return doc_ids.astype(mx.int32)


def mlx_document_boundary_mask(
    doc_ids: mx.array,
    *,
    token_mask: mx.array | None = None,
    causal: bool = False,
    expand_heads: bool = False,
) -> mx.array:
    """Return a boolean same-document mask for MLX SDPA."""

    if len(doc_ids.shape) != 2:
        raise ValueError(f"doc_ids must be 2D, got shape {doc_ids.shape}")

    doc_id_array = doc_ids.astype(mx.int32)
    valid_doc = cast(mx.array, doc_id_array >= 0)
    same_doc = cast(mx.array, doc_id_array[:, :, None] == doc_id_array[:, None, :])
    same_doc = cast(
        mx.array,
        same_doc & valid_doc[:, :, None] & valid_doc[:, None, :],
    )

    if token_mask is not None:
        if token_mask.shape != doc_id_array.shape:
            raise ValueError(
                f"token_mask shape must match doc_ids {doc_id_array.shape}, "
                f"got {token_mask.shape}"
            )
        mask_array = token_mask.astype(mx.bool_)
        same_doc = cast(
            mx.array,
            same_doc & mask_array[:, :, None] & mask_array[:, None, :],
        )

    if causal:
        seq_len = doc_id_array.shape[1]
        causal_mask = mx.arange(seq_len)[:, None] >= mx.arange(seq_len)[None, :]
        same_doc = cast(mx.array, same_doc & causal_mask[None, :, :])

    same_doc = same_doc.astype(mx.bool_)
    if expand_heads:
        return same_doc[:, None, :, :]
    return same_doc


def mlx_sequence_packing_attention_mask(
    tokens: mx.array,
    *,
    eos_token_id: int,
    token_mask: mx.array | None = None,
    causal: bool = True,
    expand_heads: bool = True,
) -> mx.array:
    """Build a boolean SDPA mask from packed tokens without fp32 promotion."""

    doc_ids = mlx_cumulative_doc_ids_from_eos(
        tokens,
        eos_token_id=eos_token_id,
        token_mask=token_mask,
    )
    return mlx_document_boundary_mask(
        doc_ids,
        token_mask=token_mask,
        causal=causal,
        expand_heads=expand_heads,
    )


def _pack_prepared_best_fit(
    samples: Sequence[np.ndarray],
    *,
    seq_len: int,
    eos_token_id: int,
    bos_token_id: int | None,
    pad_token_id: int,
) -> PackedSequences:
    packed_rows: list[np.ndarray] = []
    packed_masks: list[np.ndarray] = []
    remaining = list(enumerate(samples))
    prefix = _row_prefix(bos_token_id)

    while remaining:
        current = np.full(seq_len, pad_token_id, dtype=np.int32)
        current_mask = np.zeros(seq_len, dtype=np.bool_)
        offset = _write_prefix(current, current_mask, prefix)

        while remaining:
            capacity = seq_len - offset
            candidate_pos = _largest_fitting_sample(remaining, capacity=capacity)
            if candidate_pos is None:
                break
            _, sample = remaining.pop(candidate_pos)
            end = offset + sample.shape[0]
            current[offset:end] = sample
            current_mask[offset:end] = True
            offset = end

        packed_rows.append(current)
        packed_masks.append(current_mask)

    return _finalize_packed_rows(
        packed_rows,
        packed_masks,
        seq_len=seq_len,
        eos_token_id=eos_token_id,
    )


def _pack_prepared_sequential(
    samples: Sequence[np.ndarray],
    *,
    seq_len: int,
    eos_token_id: int,
    bos_token_id: int | None,
    pad_token_id: int,
) -> PackedSequences:
    packed_rows: list[np.ndarray] = []
    packed_masks: list[np.ndarray] = []
    current = np.full(seq_len, pad_token_id, dtype=np.int32)
    current_mask = np.zeros(seq_len, dtype=np.bool_)
    prefix = _row_prefix(bos_token_id)
    offset = _write_prefix(current, current_mask, prefix)

    for sample in samples:
        if offset > prefix.size and offset + sample.shape[0] > seq_len:
            packed_rows.append(current)
            packed_masks.append(current_mask)
            current = np.full(seq_len, pad_token_id, dtype=np.int32)
            current_mask = np.zeros(seq_len, dtype=np.bool_)
            offset = _write_prefix(current, current_mask, prefix)

        end = offset + sample.shape[0]
        current[offset:end] = sample
        current_mask[offset:end] = True
        offset = end

        if offset == seq_len:
            packed_rows.append(current)
            packed_masks.append(current_mask)
            current = np.full(seq_len, pad_token_id, dtype=np.int32)
            current_mask = np.zeros(seq_len, dtype=np.bool_)
            offset = _write_prefix(current, current_mask, prefix)

    if offset > prefix.size:
        packed_rows.append(current)
        packed_masks.append(current_mask)

    return _finalize_packed_rows(
        packed_rows,
        packed_masks,
        seq_len=seq_len,
        eos_token_id=eos_token_id,
    )


def _finalize_packed_rows(
    packed_rows: Sequence[np.ndarray],
    packed_masks: Sequence[np.ndarray],
    *,
    seq_len: int,
    eos_token_id: int,
) -> PackedSequences:
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


def _prepare_samples(
    documents: Sequence[Sequence[int] | np.ndarray],
    *,
    seq_len: int,
    eos_token_id: int,
    bos_token_id: int | None,
    oversized: OversizedSamplePolicy,
) -> list[np.ndarray]:
    capacity = seq_len - len(_row_prefix(bos_token_id))
    if capacity < 1:
        raise ValueError("seq_len must leave room for at least one document token")

    samples: list[np.ndarray] = []
    for sample_idx, document in enumerate(documents):
        sample = _document_with_eos(
            document,
            eos_token_id=eos_token_id,
            sample_idx=sample_idx,
        )
        if sample.shape[0] > capacity:
            if oversized == "refuse":
                raise ValueError(
                    f"sample {sample_idx} length {sample.shape[0]} exceeds usable "
                    f"seq_len {capacity}; use oversized='truncate' to clip it"
                )
            sample = sample[:capacity].copy()
            sample[-1] = eos_token_id
        samples.append(sample)
    return samples


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


def _validate_pack_options(
    *,
    seq_len: int,
    eos_token_id: int,
    bos_token_id: int | None,
    pad_token_id: int,
    oversized: OversizedSamplePolicy,
    strategy: PackingStrategy,
) -> None:
    if seq_len < 2:
        raise ValueError("seq_len must be at least 2")
    if oversized not in {"refuse", "truncate"}:
        raise ValueError("oversized must be 'refuse' or 'truncate'")
    if strategy not in {"best_fit", "sequential"}:
        raise ValueError("strategy must be 'best_fit' or 'sequential'")
    _validate_token_id("eos_token_id", eos_token_id)
    if bos_token_id is not None:
        _validate_token_id("bos_token_id", bos_token_id)
    _validate_token_id("pad_token_id", pad_token_id)


def _row_prefix(bos_token_id: int | None) -> np.ndarray:
    if bos_token_id is None:
        return np.empty((0,), dtype=np.int32)
    return np.array([bos_token_id], dtype=np.int32)


def _write_prefix(
    tokens: np.ndarray,
    token_mask: np.ndarray,
    prefix: np.ndarray,
) -> int:
    if prefix.size:
        tokens[: prefix.size] = prefix
        token_mask[: prefix.size] = True
    return int(prefix.size)


def _largest_fitting_sample(
    remaining: Sequence[tuple[int, np.ndarray]],
    *,
    capacity: int,
) -> int | None:
    best_pos: int | None = None
    best_len = -1
    best_idx = 0
    for pos, (sample_idx, sample) in enumerate(remaining):
        sample_len = sample.shape[0]
        if sample_len <= capacity and (
            sample_len > best_len or (sample_len == best_len and sample_idx < best_idx)
        ):
            best_pos = pos
            best_len = sample_len
            best_idx = sample_idx
    return best_pos


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
    "PackingStrategy",
    "cumulative_doc_ids_from_eos",
    "document_boundary_mask",
    "mlx_cumulative_doc_ids_from_eos",
    "mlx_document_boundary_mask",
    "mlx_sequence_packing_attention_mask",
    "pack_bos_aligned_best_fit",
    "pack_documents_with_eos",
]
