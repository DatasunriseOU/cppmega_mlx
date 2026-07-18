# golden_mini fixture

Deterministic, tiny GOLDEN fixture produced by the REAL modern data pipeline. Regenerate with:

```
/Volumes/external/sources/nanochat/.venv/bin/python /Volumes/external/sources/cppmega.mlx/scripts/data/make_golden_mini.py
```

All inputs are fixed Python constants in `make_golden_mini.py` (no randomness, no timestamps), so the synthetic C++ sources and the pipeline outputs are byte-stable across runs on the same toolchain.

## Pipeline

1. `tools/clang_indexer/index_project.py --enriched` (synth repo -> enriched JSONL)
2. `scripts/nanochat_data/clang_enriched_to_parquet.py --materialize-tokenized-enriched` (JSONL -> tokenized enriched parquet, tokenizer=tokenizer.json vocab=65536, size=4k)
3. `scripts/nanochat_data/pack_enriched_rows.py --target-length 4096` (tokenized parquet -> packed rows)
4. Commits: `tools/clang_indexer/process_commits.py` (before/after pairs -> enriched commit JSONL) -> step 2 conversion.

## Synthetic inputs

### Code mini-repos (`code/`)

- **container** (`code/code_container.parquet`, 1 packed row(s)) — class template `Stack<T>` (template_ref type edges); cross-file use in `use_stack.cpp`.
  - files: `include/stack.h`, `src/use_stack.cpp`
  - enriched docs: 3, call_edges: 0, type_edges: 2
- **graph** (`code/code_graph.parquet`, 1 packed row(s)) — 3-layer #include + call graph `util -> engine -> api`; struct `Config` field access.
  - files: `include/engine.h`, `include/util.h`, `src/api.cpp`, `src/engine.cpp`, `src/util.cpp`
  - enriched docs: 4, call_edges: 3, type_edges: 3
- **shapes** (`code/code_shapes.parquet`, 1 packed row(s)) — base class `Shape` + virtual `area()` override in `Circle`; struct `Point` field access; cross-file calls (`report.cpp` -> `Shape::area`/`centroid`, `circle_report` -> `area_report`).
  - files: `include/shape.h`, `src/report.cpp`, `src/shape.cpp`
  - enriched docs: 6, call_edges: 4, type_edges: 11

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
  - `source_pr_numbers`: `list<element: int64>`
  - `source_has_pr_discussions`: `list<element: bool>`
  - `source_pr_discussion_chars`: `list<element: int32>`
  - `source_pr_discussion_lines`: `list<element: int32>`
  - `source_doc_types`: `list<element: string>`
  - `source_header_fragment_kinds`: `list<element: string>`
  - `source_ifim_instruction_token_ids`: `list<element: list<element: uint32>>`
  - `source_commit_msg_token_ids`: `list<element: list<element: uint32>>`
  - `source_pre_token_ids`: `list<element: list<element: uint32>>`
  - `source_post_token_ids`: `list<element: list<element: uint32>>`
  - `source_diff_token_ids`: `list<element: list<element: uint32>>`
  - `platform_ids`: `list<element: uint16>`
  - `symbol_identities`: `list<element: struct<symbol_id: uint64, symbol_key: string>>`
  - `repo`: `string`
  - `filepath`: `string`
  - `commit_hash`: `string`
  - `timestamp`: `string`
  - `pr_number`: `int64`
  - `has_pr_discussion`: `bool`
  - `pr_discussion_chars`: `int32`
  - `pr_discussion_lines`: `int32`
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
  - `token_domain_ids`: `list<element: uint16>`
  - `token_role_ids`: `list<element: uint16>`
  - `token_entity_ids`: `list<element: uint32>`
  - `token_scope_ids`: `list<element: uint32>`
  - `token_source_doc_ids`: `list<element: uint32>`
  - `token_source_identity_ids`: `list<element: uint64>`
  - `token_confidence_ids`: `list<element: uint8>`
  - `token_symbol_ids`: `list<element: uint64>`
  - `token_call_targets`: `list<element: uint64>`
  - `token_type_refs`: `list<element: uint64>`
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
  - `token_domain_edges`: `list<element: struct<from: uint32, to: uint32, kind: int32>>`
  - `token_build_edges`: `list<element: struct<from: uint32, to: uint32, kind: int32>>`
  - `token_shell_edges`: `list<element: struct<from: uint32, to: uint32, kind: int32>>`
  - `token_diagnostic_edges`: `list<element: struct<from: uint32, to: uint32, kind: int32>>`
  - `token_cross_domain_edges`: `list<element: struct<from: uint32, to: uint32, kind: int32>>`
  - `changed_chunk_ids`: `list<element: uint32>`
  - `changed_chunk_spans`: `list<element: struct<start: uint32, end: uint32>>`
  - `source_identity_registry`: `list<element: struct<source_identity_id: uint64 not null, canonical_sha256: string not null, source: string not null>>`

### `commits/commits.parquet` (tokenized enriched commit docs)

  - `text`: `string`
  - `symbol_identities`: `list<element: struct<symbol_id: uint64, symbol_key: string>>`
  - `source_text`: `string`
  - `ifim_instruction_text`: `string`
  - `commit_msg_text`: `string`
  - `pre_text`: `string`
  - `post_text`: `string`
  - `diff_text`: `string`
  - `source_doc_id`: `string`
  - `doc_type`: `string`
  - `header_fragment_kind`: `string`
  - `tokenizer_fingerprint`: `string`
  - `actual_token_count`: `int32`
  - `structure_ids`: `list<element: int8>`
  - `chunk_boundaries`: `list<element: struct<start: int32, end: int32, kind: int8, dep_level: int32, name: string, symbol_id: uint64>>`
  - `call_edges`: `list<element: struct<from: int32, to: int32>>`
  - `type_edges`: `list<element: struct<from: int32, to: int32>>`
  - `ast_depth`: `list<element: uint16>`
  - `sibling_index`: `list<element: uint16>`
  - `ast_node_type`: `list<element: uint16>`
  - `symbol_ids`: `list<element: uint64>`
  - `call_targets`: `list<element: uint64>`
  - `type_refs`: `list<element: uint64>`
  - `def_use`: `list<element: uint8>`
  - `domain_kind`: `uint16`
  - `domain_ids`: `list<element: uint16>`
  - `domain_role_ids`: `list<element: uint16>`
  - `domain_entity_ids`: `list<element: uint32>`
  - `domain_scope_ids`: `list<element: uint32>`
  - `domain_source_doc_ids`: `list<element: uint32>`
  - `domain_source_identity_ids`: `list<element: uint64>`
  - `domain_confidence_ids`: `list<element: uint8>`
  - `domain_edges`: `list<element: struct<from_char: int32, to_char: int32, kind: int32>>`
  - `build_edges`: `list<element: struct<from_char: int32, to_char: int32, kind: int32>>`
  - `shell_edges`: `list<element: struct<from_char: int32, to_char: int32, kind: int32>>`
  - `diagnostic_edges`: `list<element: struct<from_char: int32, to_char: int32, kind: int32>>`
  - `cross_domain_edges`: `list<element: struct<from_char: int32, to_char: int32, kind: int32>>`
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
  - `pr_number`: `int64`
  - `has_pr_discussion`: `bool`
  - `pr_discussion_chars`: `int32`
  - `pr_discussion_lines`: `int32`
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
  - `ifim_instruction_token_ids`: `list<element: uint32>`
  - `commit_msg_token_ids`: `list<element: uint32>`
  - `pre_token_ids`: `list<element: uint32>`
  - `post_token_ids`: `list<element: uint32>`
  - `diff_token_ids`: `list<element: uint32>`
  - `platform_ids`: `list<element: uint16>`
  - `token_structure_ids`: `list<element: uint8>`
  - `token_dep_levels`: `list<element: uint16>`
  - `token_ast_depth`: `list<element: uint16>`
  - `token_sibling_index`: `list<element: uint16>`
  - `token_ast_node_type`: `list<element: uint16>`
  - `token_symbol_ids`: `list<element: uint64>`
  - `token_call_targets`: `list<element: uint64>`
  - `token_type_refs`: `list<element: uint64>`
  - `token_def_use`: `list<element: uint8>`
  - `token_domain_ids`: `list<element: uint16>`
  - `token_role_ids`: `list<element: uint16>`
  - `token_entity_ids`: `list<element: uint32>`
  - `token_scope_ids`: `list<element: uint32>`
  - `token_source_doc_ids`: `list<element: uint32>`
  - `token_source_identity_ids`: `list<element: uint64>`
  - `token_confidence_ids`: `list<element: uint8>`
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
  - `token_domain_edges`: `list<element: struct<from: uint32, to: uint32, kind: int32>>`
  - `token_build_edges`: `list<element: struct<from: uint32, to: uint32, kind: int32>>`
  - `token_shell_edges`: `list<element: struct<from: uint32, to: uint32, kind: int32>>`
  - `token_diagnostic_edges`: `list<element: struct<from: uint32, to: uint32, kind: int32>>`
  - `token_cross_domain_edges`: `list<element: struct<from: uint32, to: uint32, kind: int32>>`
  - `source_identity_registry`: `list<element: struct<source_identity_id: uint64 not null, canonical_sha256: string not null, source: string not null>>`
