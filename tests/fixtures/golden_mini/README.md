# golden_mini fixture

Deterministic, tiny GOLDEN fixture produced by the REAL modern data pipeline. Regenerate with:

```
/Volumes/external/sources/cppmega.mlx/.venv/bin/python /Volumes/external/sources/cppmega.mlx/scripts/data/make_golden_mini.py
```

All inputs are fixed Python constants in `make_golden_mini.py` (no randomness, no timestamps), so the synthetic C++ sources and the pipeline outputs are byte-stable across runs on the same toolchain.

## Pipeline

1. `tools/clang_indexer/index_project.py --enriched` (synth repo -> enriched JSONL)
2. `scripts/nanochat_data/clang_enriched_to_parquet.py --materialize-tokenized-enriched` (JSONL -> tokenized enriched parquet, tokenizer=tokenizer.json vocab=65536, size=4k)
3. `scripts/nanochat_data/pack_enriched_rows.py --target-length 4096` (tokenized parquet -> packed rows)
4. Commits: `tools/clang_indexer/process_commits.py` (before/after pairs -> enriched commit JSONL) -> step 2 conversion.

## Synthetic inputs

### Code mini-repos (`code/`)

- **container** (`code/code_container.parquet`, 2 packed row(s)) — class template `Stack<T>` (template_ref type edges); cross-file use in `use_stack.cpp`.
  - files: `include/stack.h`, `src/use_stack.cpp`
  - enriched docs: 2, call_edges: 0, type_edges: 0
- **graph** (`code/code_graph.parquet`, 3 packed row(s)) — 3-layer #include + call graph `util -> engine -> api`; struct `Config` field access.
  - files: `include/engine.h`, `include/util.h`, `src/api.cpp`, `src/engine.cpp`, `src/util.cpp`
  - enriched docs: 3, call_edges: 3, type_edges: 0
- **shapes** (`code/code_shapes.parquet`, 3 packed row(s)) — base class `Shape` + virtual `area()` override in `Circle`; struct `Point` field access; cross-file calls (`report.cpp` -> `Shape::area`/`centroid`, `circle_report` -> `area_report`).
  - files: `include/shape.h`, `src/report.cpp`, `src/shape.cpp`
  - enriched docs: 3, call_edges: 3, type_edges: 2

### Commit pairs (`commits/`)

- `src/scale.cpp` (repo `golden_mini/shapes`, commit `00000000`) — before/after of one function.
- `src/normalize.cpp` (repo `golden_mini/graph`, commit `00000000`) — before/after of one function.

Produced `commits/commits.parquet` with 4 row(s) from 4 enriched commit doc(s).

## Column inventory

### `code/code_container.parquet` (packed code rows; representative — all `code/*.parquet` share this schema)

  - `pack_id`: `int64`
  - `valid_token_count`: `int32`
  - `trained_token_count`: `int32`
  - `num_docs`: `int32`
  - `slack_tokens`: `int32`
  - `input_ids`: `list<element: uint32>`
  - `target_ids`: `list<element: uint32>`
  - `loss_mask`: `list<element: uint8>`
  - `doc_ids`: `list<element: uint32>`
  - `source_doc_ids`: `list<element: int64>`
  - `source_doc_token_lengths`: `list<element: int32>`
  - `source_platform_ids`: `list<element: list<element: uint16>>`
  - `source_repo_stable_ids`: `list<element: string>`
  - `source_filepath_stable_ids`: `list<element: string>`
  - `source_file_local_commit_indices`: `list<element: int32>`
  - `platform_ids`: `list<element: uint16>`
  - `repo`: `string`
  - `filepath`: `string`
  - `commit_hash`: `string`
  - `timestamp`: `string`
  - `parent_hashes`: `list<element: string>`
  - `parent_count`: `int32`
  - `is_merge_commit`: `bool`
  - `author_timestamp`: `string`
  - `commit_timestamp`: `string`
  - `repo_stable_id`: `string`
  - `filepath_stable_id`: `string`
  - `file_local_commit_index`: `int32`
  - `has_ambiguous_reconstruction`: `bool`
  - `has_rename_ambiguity`: `bool`
  - `token_platform_ids`: `list<element: uint16>`
  - `token_structure_ids`: `list<element: uint8>`
  - `token_dep_levels`: `list<element: uint16>`
  - `token_ast_depth`: `list<element: int32>`
  - `token_sibling_index`: `list<element: int32>`
  - `token_ast_node_type`: `list<element: int32>`
  - `token_symbol_ids`: `list<element: uint32>`
  - `token_call_targets`: `list<element: uint32>`
  - `token_type_refs`: `list<element: uint32>`
  - `token_def_use`: `list<element: uint8>`
  - `token_change_mask_pre`: `list<element: uint8>`
  - `token_change_mask_post`: `list<element: uint8>`
  - `hunk_id_per_token`: `list<element: int32>`
  - `edit_op_per_token`: `list<element: uint8>`
  - `token_chunk_starts`: `list<element: uint32>`
  - `token_chunk_ends`: `list<element: uint32>`
  - `token_chunk_kinds`: `list<element: uint8>`
  - `token_chunk_dep_levels`: `list<element: uint16>`
  - `token_call_edges`: `list<element: struct<from: uint16, to: uint16>>`
  - `token_type_edges`: `list<element: struct<from: uint16, to: uint16>>`
  - `changed_chunk_ids`: `list<element: uint32>`
  - `changed_chunk_spans`: `list<element: struct<start: uint32, end: uint32>>`

### `commits/commits.parquet` (tokenized enriched commit docs)

  - `text`: `string`
  - `source_text`: `string`
  - `source_doc_id`: `string`
  - `tokenizer_fingerprint`: `string`
  - `actual_token_count`: `int32`
  - `structure_ids`: `list<element: int8>`
  - `chunk_boundaries`: `list<element: struct<start: int32, end: int32, kind: int8, dep_level: int32, name: string>>`
  - `call_edges`: `list<element: struct<from: int32, to: int32>>`
  - `type_edges`: `list<element: struct<from: int32, to: int32>>`
  - `ast_depth`: `list<element: uint16>`
  - `sibling_index`: `list<element: uint16>`
  - `ast_node_type`: `list<element: uint16>`
  - `symbol_ids`: `list<element: uint32>`
  - `call_targets`: `list<element: uint32>`
  - `type_refs`: `list<element: uint32>`
  - `def_use`: `list<element: uint8>`
  - `change_mask_pre`: `list<element: uint8>`
  - `change_mask_post`: `list<element: uint8>`
  - `hunk_id_per_char`: `list<element: int32>`
  - `edit_op_per_char`: `list<element: uint8>`
  - `platform_info`: `string`
  - `language_info`: `string`
  - `build_info`: `string`
  - `constituent_provenance`: `list<element: struct<filepath: string, language_info: string, build_info: string>>`
  - `constituent_provenance_json`: `string`
  - `repo`: `string`
  - `filepath`: `string`
  - `commit_hash`: `string`
  - `timestamp`: `string`
  - `parent_hashes`: `list<element: string>`
  - `parent_count`: `int32`
  - `is_merge_commit`: `bool`
  - `author_timestamp`: `string`
  - `commit_timestamp`: `string`
  - `repo_stable_id`: `string`
  - `filepath_stable_id`: `string`
  - `file_local_commit_index`: `int32`
  - `has_ambiguous_reconstruction`: `bool`
  - `has_rename_ambiguity`: `bool`
  - `token_ids`: `list<element: uint32>`
  - `platform_ids`: `list<element: uint16>`
  - `token_structure_ids`: `list<element: uint8>`
  - `token_dep_levels`: `list<element: uint16>`
  - `token_ast_depth`: `list<element: uint16>`
  - `token_sibling_index`: `list<element: uint16>`
  - `token_ast_node_type`: `list<element: uint16>`
  - `token_symbol_ids`: `list<element: uint32>`
  - `token_call_targets`: `list<element: uint32>`
  - `token_type_refs`: `list<element: uint32>`
  - `token_def_use`: `list<element: uint8>`
  - `token_change_mask_pre`: `list<element: uint8>`
  - `token_change_mask_post`: `list<element: uint8>`
  - `hunk_id_per_token`: `list<element: int32>`
  - `edit_op_per_token`: `list<element: uint8>`
  - `token_chunk_starts`: `list<element: uint32>`
  - `token_chunk_ends`: `list<element: uint32>`
  - `token_chunk_kinds`: `list<element: uint8>`
  - `token_chunk_dep_levels`: `list<element: uint16>`
  - `changed_chunk_ids`: `list<element: uint32>`
  - `changed_chunk_spans`: `list<element: struct<start: uint32, end: uint32>>`
  - `token_call_edges`: `list<element: struct<from: uint16, to: uint16>>`
  - `token_type_edges`: `list<element: struct<from: uint16, to: uint16>>`
