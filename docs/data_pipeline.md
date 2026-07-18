# MLX Data Pipeline

This document describes the current local data ingress and reference packing
contract for cppmega_mlx. It is intentionally scoped to local MLX data helpers
and the tiny training/model path. The local Stream D data-ingress acceptance is
closed for this scope: tokenizer parity is covered by the vendored GB10
tokenizer contract, local 100M-token Megatron indexed stress is covered by the
stress harness, and full-corpus pretraining is a post-M0 concern rather than a
local ingress blocker.

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
- Megatron Indexed `.bin/.idx` datasets through open_megatron_indexed_dataset:
  either a single suffixless prefix or a directory of `.idx`/`.bin` shard
  pairs, using the standalone reader seam without importing Megatron runtime
  code into the Mac path.

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
and fail closed on alias conflicts. LMTokenBatch exposes document_ids as a
first-class field, and NPZ, Parquet, and Megatron indexed ingress preserve
token-aligned document IDs when the shard provides exactly one supported alias.

## Production Objective Rows

Megatron objective materialization accepts either typed `token_ids` rows or
fixed-length packed `input_ids` rows with an integral `valid_token_count`; the
two representations cannot appear together. Every packed token-aligned sidecar
is length-checked and trimmed to the valid prefix. Single-document IFIM and
multi-document commit sections retain exact constituent bindings; ambiguous
provenance fails closed rather than selecting a candidate by position.
Objective loss masks are source-transition aligned: `loss_mask[i]` gates the
label predicted from source token `i`, and must be zero at document boundaries
and at the final valid token. The production mixer uses bounded lookahead to
realize exact task quotas when sparse objective eligibility requires additional
source rows.

The Stage-1 objective path has one task dispatcher: `EligibilityAwareTaskMixer`.
The streaming runner wraps it in `CanonicalObjectivePlanner`, remaps FIM/IFIM
routes before batch construction, and composes CE with the configured graph /
indexer loss through `production_training_loss`. The immutable Megatron ingress
used by the production bundle runner is the same path materialized ahead of
time; its `objective_materialized` contract and hashes are validated before any
training batch is opened. Runners do not call individual FIM, IFIM, or commit
builders. Missing graph route sidecars fail closed; independently tokenized
COMMIT_DIFF and PRE_TO_POST sections carry an explicit route exclusion rather
than fabricated graph alignment.

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
- Batch schemas are limited to tokens, attention_mask, document_ids, and the
  existing token-aligned structure/platform side-channel keys. Unknown keys are
  rejected instead of being silently dropped.
- If torch is not installed, bridge construction raises a clear optional
  dependency error; native MLX dataset iterators remain the default path.

This bridge covers the narrow Stream D PyTorch DataLoader seam only. M0.1
tokenizer parity is already closed by the vendored GB10 tokenizer contract and
explicit <SPACE>/<NL> sentinel decode receipt; the bridge preserves
LMTokenBatch document_ids but is separate from the Megatron indexed stress
harness.

## Megatron Indexed Stress Harness

scripts/megatron_ingress_stress.py generates local Megatron Indexed fixtures in
bounded chunks, optionally with document_ids and structure_ids sidecars, then
opens them through open_token_dataset/open_megatron_indexed_dataset and reads
fixed batches. The JSON receipt reports generated bytes, read throughput,
token_id_range, side-channel presence, index metadata, peak RSS, and the
configured memory ceiling. It is local-only and explicitly makes no GB10,
distributed Megatron, or M4-vs-GB10 parity claim.

The intended Stream D scale gate is:

```bash
./.venv/bin/python scripts/megatron_ingress_stress.py \
  --token-count 100000000 \
  --shards 4 \
  --seq-len 1024 \
  --batch-size 8 \
  --batches 64 \
  --include-document-ids \
  --include-structure-ids
```

## Local Forward Smoke

scripts/data_smoke.py remains ingress-only by default, but --forward-smoke
adds a forward-only tiny HybridTinyLM check on the first full batch. The script
threads LMTokenBatch.model_kwargs() side channels and token-aligned
document_ids into the model, reports finite next-token loss and logits shape,
and keeps training_wired=false with no GB10, distributed Megatron, or
M4-vs-GB10 parity claim.

## Current Guardrails

- Packing is exported from cppmega_mlx.data for callers and tests.
- The current training ingress still consumes dense LMTokenBatch rows.
- scripts/data_smoke.py --forward-smoke verifies local batch -> model forward
  closure without training, benchmarking, or parity claims.
- PyTorch DataLoader integration is explicit and optional; the MLX training hot
  path does not import torch unless the bridge is requested.
- Mapping-batch and LMTokenBatch training can carry explicit packed document
  IDs through next-token and MTP loss paths into model attention.
- NPZ, Parquet, and single- or multi-shard Megatron indexed loaders preserve
  persisted token-aligned document IDs.
- Multi-shard Megatron indexed directories batch across shard boundaries only
  when token dtype, side-channel keys/dtypes, document-id presence/dtype, and
  tokenizer metadata match across shards; schema drift fails closed.
- The Megatron indexed stress harness covers the local 100M-token ingress gate
  under an explicit peak-memory ceiling.
- Local Stream D data-ingress acceptance is closed for NPZ, Parquet, Megatron
  indexed, packing, document IDs, side-channel preservation, DataLoader bridge,
  data_smoke forward closure, tokenizer contract, and local 100M-token stress.
- Full-corpus pretraining remains a post-M0 concern and is not claimed by this
  local data-ingress contract.
