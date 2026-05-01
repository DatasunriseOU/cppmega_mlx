# Local GB10 Parquet Samples

Real cppmega Parquet samples are intentionally local-only and ignored by git.
Use them to exercise the MLX data path against the same row schema used before
the token-only Megatron indexed conversion.

Hygiene rule: do not add these files to git or move them into tracked test
fixtures. .gitignore must keep data/parquet_samples/ ignored, and tests that
use these samples must skip cleanly when the local files are absent.

Current GB10 sources:

- gb10:/home/dave/cppmega-root/data/parquet_samples/clang_semantic_4k_v10/val_00000.parquet
- gb10:/home/dave/cppmega-root/data/parquet_samples/clang_commits_4k_v1/val_00000.parquet

Local destination:

- data/parquet_samples/gb10/clang_semantic_4k_v10/val_00000.parquet
- data/parquet_samples/gb10/clang_commits_4k_v1/val_00000.parquet

Refresh command:

sh
mkdir -p data/parquet_samples/gb10/clang_semantic_4k_v10 \
  data/parquet_samples/gb10/clang_commits_4k_v1
scp gb10:/home/dave/cppmega-root/data/parquet_samples/clang_semantic_4k_v10/val_00000.parquet \
  data/parquet_samples/gb10/clang_semantic_4k_v10/val_00000.parquet
scp gb10:/home/dave/cppmega-root/data/parquet_samples/clang_commits_4k_v1/val_00000.parquet \
  data/parquet_samples/gb10/clang_commits_4k_v1/val_00000.parquet


The Parquet schema includes token-level side-channel aliases such as
token_structure_ids, token_dep_levels, token_ast_depth,
token_sibling_index, and token_ast_node_type.  Source-level columns such as
structure_ids are not assumed to be token-aligned.

The current GB10 samples are expected to expose these list element dtypes:

- token_ids: uint32
- structure_ids: int8 source-level AST IDs, not token-aligned
- token_structure_ids: uint8
- token_dep_levels: uint16
- token_ast_depth: uint16
- token_sibling_index: uint16
- token_ast_node_type: uint16

The MLX Parquet reader fails closed when token IDs or structure side-channel
aliases are non-integer, when token-level side-channel aliases are not
token-aligned, or when a canonical and alias side-channel both declare the same
batch field. Attention masks are the only numeric side channel that may use
floating-point values. TokenParquetDataset.parquet_receipt records the
physical Parquet columns and schema types it saw, the token source column, the
physical source column for each normalized side channel, and skipped
side-channel-looking columns. In the GB10 samples this should show
structure_ids skipped as not_token_aligned while token_structure_ids
supplies the normalized structure_ids batch field.

These samples validate the Parquet-side aliases before conversion. They do not
prove that ../cppmega Stage 3 preserves structure columns, because the current
source converter reads a single token column and emits token-only .bin/.idx
files. Local Megatron-indexed side channels require an explicit MLX sidecar as
documented in docs/megatron_indexed_ingress.md.

Current local smoke coverage:

sh
./.venv/bin/python -m pytest tests/test_parquet_dataset.py tests/test_token_dataset.py tests/test_megatron_indexed.py tests/test_real_parquet_samples.py -q
./.venv/bin/pyright cppmega_mlx/data tests/test_parquet_dataset.py tests/test_token_dataset.py tests/test_megatron_indexed.py tests/test_real_parquet_samples.py


The pytest receipt covers both clang_semantic_4k_v10 and
clang_commits_4k_v1.  tests/test_real_parquet_samples.py confirms each
sample produces fixed-shape token batches with token-aligned structure side
channels, slices those fields through LMTokenBatch.model_kwargs(), and runs a
one-step eager HybridTinyLM train/eval smoke on a copied local head of each
sample.

Train-script JSON receipts mirror this local-only provenance under
dataset.dataset_receipt. For copied Parquet heads the receipt includes
source_format: parquet, the normalized sample name such as
clang_semantic_4k_v10, source path, shape fields, token key, sample/batch
counts, dropped samples, side channels, and nested parquet_receipt provenance.
Suffixless Megatron .bin/.idx prefixes use source_format: megatron, preserve
the prefix name such as clang_semantic_4k_v10_train, and include parsed
index_metadata.

This proves local MLX tiny training on copied GB10 Parquet heads only. It does
not prove full Megatron distributed training, M4 Max vs GB10 throughput parity,
GB10 training correctness, or production-scale Parquet ingestion.
