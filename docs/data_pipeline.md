# MLX Data Pipeline

This document describes the current local data ingress and reference packing
contract for cppmega_mlx. It is intentionally scoped to local MLX data helpers
and the tiny training/model path. Full Stream D remains open until the tokenizer,
scale, and stress gates are proven.

## Token Ingress

cppmega_mlx.data accepts token IDs from three local ingress paths:

- NPZ shards through TokenNpzDataset / open_token_dataset, with tokens
  shaped (N, S) and optional side-channel arrays matching the token shape.
- Parquet files through TokenParquetDataset, with token columns configured by
  ParquetColumns and optional cppmega token-aligned structure side channels.
  Current copied GB10 parquet samples are token-only for MLX side-channel
  threading after the SPACE/NL retokenize: source-level structure_ids is
  recorded as not_token_aligned and skipped, while future regenerated
  token_* aliases remain supported when their rows match token_ids.
- Megatron Indexed `.bin/.idx` datasets through open_megatron_indexed_dataset,
  using the standalone reader seam without importing Megatron runtime code into
  the Mac path.

All paths are tokenizer-agnostic at this layer. Callers are responsible for
using a tokenizer contract that supplies the EOS token ID and any FIM/chat
special-token IDs needed before data reaches these helpers.

## Fixed-Length Sequence Packing

The public reference helpers are pack_documents_with_eos and
pack_bos_aligned_best_fit. They take tokenized documents and emit fixed-length
rows padded with pad_token_id. The packer appends one EOS token to each
document unless the document already ends with EOS.

The pack_bos_aligned_best_fit helper provides deterministic BOS-aligned best
fit: each packed row starts with bos_token_id when supplied, then repeatedly
selects the largest remaining EOS-terminated document that fits in the row.
Ties keep input order. The compatibility wrapper pack_documents_with_eos
defaults to input-order concatenation; callers can opt into best-fit with
strategy="best_fit".

Oversized documents fail closed by default. Callers can opt into
oversized="truncate" when they explicitly want a document clipped to
seq_len with the final token forced to EOS.

The helper returns PackedSequences:

- tokens: int32 fixed-length packed rows.
- token_mask: boolean mask marking real tokens and excluding padding.
- doc_ids: cumulative document IDs derived from previous EOS positions.
- boundary_mask: boolean (batch, seq, seq) same-document mask.

## Document-Boundary Mask Semantics

cumulative_doc_ids_from_eos assigns a token to the document that started after
the previous EOS in the same row. EOS belongs to the document it terminates.
Padding receives pad_doc_id=-1 when a token_mask is supplied.

document_boundary_mask compares those document IDs and returns True only
for token pairs that belong to the same document. Negative document IDs are
always treated as invalid padding IDs, even when the caller omits token_mask.
With token_mask, padding cannot attend to anything and real tokens cannot
attend to padding. With causal=True, the same-document mask is additionally
lower-triangular.

For MLX attention wiring, mlx_cumulative_doc_ids_from_eos derives the same
IDs with mx.cumsum over previous EOS hits, and
mlx_sequence_packing_attention_mask returns a boolean mask. By default the MLX
mask is shaped (batch, 1, seq, seq) so it broadcasts over heads for
mx.fast.scaled_dot_product_attention; it intentionally stays boolean and does
not promote to an additive float32 mask.

HybridTinyLM.__call__(document_ids=...) now consumes explicit packed document
IDs and uses mlx_document_boundary_mask(..., causal=True, expand_heads=True)
for attention routes. The next-token and MTP loss helpers accept exactly one of
document_ids, doc_ids, or packing_document_ids in mapping batches, validate
that it matches tokens, reject negative explicit IDs, slice it to model inputs,
and fail closed on alias conflicts.

Historical caveat: sequence packing is no longer simply "not wired into the
training loop" or "not consumed by the current attention implementation" for
mapping batches, but LMTokenBatch dataset ingress still lacks a first-class
document-id field.

## Optional PyTorch DataLoader Bridge

cppmega_mlx.data.dataloader_bridge provides an explicit, optional PyTorch
DataLoader handoff for already-local LMTokenBatch rows. Importing
cppmega_mlx.data does not import torch; callers must opt in by calling
build_spawn_dataloader(...).

The bridge is fail-closed by design:

- num_workers > 0 uses multiprocessing_context="spawn" by default and
  rejects any explicit non-spawn context.
- persistent_workers and prefetch_factor are accepted only when workers are
  enabled.
- Batch schemas are limited to tokens, attention_mask, and the existing
  token-aligned structure side-channel keys. Unknown keys are rejected instead
  of being silently dropped.
- If torch is not installed, bridge construction raises a clear optional
  dependency error; native MLX dataset iterators remain the default path.

This bridge covers the narrow Stream D PyTorch DataLoader seam only. M0.1
tokenizer parity is already closed by the vendored GB10 tokenizer contract and
explicit <SPACE>/<NL> sentinel decode receipt; the bridge does not make
packed document IDs first-class in LMTokenBatch, and does not satisfy the
100M-token stress gate.

## Current Guardrails

- Packing is exported from cppmega_mlx.data for callers and tests.
- The current training ingress still consumes dense LMTokenBatch rows.
- PyTorch DataLoader integration is explicit and optional; the MLX training hot
  path does not import torch unless the bridge is requested.
- Mapping-batch training can carry explicit packed document IDs through
  next-token and MTP loss paths into model attention.
- LMTokenBatch itself still has no persisted document-id field, and data
  loaders still need an owned schema pass before packed IDs are first-class.
- Full Stream D is still not closeable: multi-shard/scale validation, packed
  document-id schema ownership, and the 100M-token stress gate remain open.
