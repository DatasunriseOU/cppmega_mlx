# MLX Data Pipeline

This document describes the current local data ingress and reference packing
contract for `cppmega_mlx`. It is intentionally scoped to standalone MLX data
helpers. Sequence packing is not wired into the training loop, model attention,
or custom attention kernels yet.

## Token Ingress

`cppmega_mlx.data` accepts token IDs from three local ingress paths:

- NPZ shards through `TokenNpzDataset` / `open_token_dataset`, with `tokens`
  shaped `(N, S)` and optional side-channel arrays matching the token shape.
- Parquet files through `TokenParquetDataset`, with token columns configured by
  `ParquetColumns` and optional cppmega structure side channels.
- Megatron Indexed `.bin/.idx` datasets through `open_megatron_indexed_dataset`,
  using the standalone reader seam without importing Megatron runtime code into
  the Mac path.

All paths are tokenizer-agnostic at this layer. Callers are responsible for
using a tokenizer contract that supplies the EOS token ID and any FIM/chat
special-token IDs needed before data reaches these helpers.

## Fixed-Length Sequence Packing

The public reference helper is `pack_documents_with_eos`. It takes tokenized
documents, appends one EOS token to each document unless the document already
ends with EOS, concatenates documents in input order, and emits fixed-length
rows padded with `pad_token_id`.

Oversized documents fail closed by default. Callers can opt into
`oversized="truncate"` when they explicitly want a document clipped to
`seq_len` with the final token forced to EOS.

The helper returns `PackedSequences`:

- `tokens`: `int32` fixed-length packed rows.
- `token_mask`: boolean mask marking real tokens and excluding padding.
- `doc_ids`: cumulative document IDs derived from previous EOS positions.
- `boundary_mask`: boolean `(batch, seq, seq)` same-document mask.

## Document-Boundary Mask Semantics

`cumulative_doc_ids_from_eos` assigns a token to the document that started after
the previous EOS in the same row. EOS belongs to the document it terminates.
Padding receives `pad_doc_id=-1` when a `token_mask` is supplied.

`document_boundary_mask` compares those document IDs and returns `True` only
for token pairs that belong to the same document. With `token_mask`, padding
cannot attend to anything and real tokens cannot attend to padding. With
`causal=True`, the same-document mask is additionally lower-triangular.

This mask is reference metadata only today. It is not consumed by the current
attention implementation, training batches, or attention kernels.

## Current Guardrails

- Packing is exported from `cppmega_mlx.data` for callers and tests.
- The current training ingress still consumes dense `LMTokenBatch` rows.
- The sequence-packing mask is not wired into training loss, data loaders,
  model forward paths, or attention kernels yet.
- Future wiring must add targeted attention/training tests before claiming
  packed-document isolation in actual model execution.
