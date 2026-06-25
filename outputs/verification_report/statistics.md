# Sidecar fill statistics (real sample)

Per-channel FILL %, computed over a real sample of rows from EVERY input parquet file. `rows-filled %` = fraction of rows with >=1 non-zero/non-empty entry in that channel; `tokens-nonzero %` = fraction of all per-token slots that are non-zero (per-token channels only). hunk_id uses a -1 sentinel for "no hunk", so it is counted filled only when > 0.

## Roundtrip (our cpp_tokenizer)

Three honest metrics. `id_exact` is EXPECTED to be low: stored ids keep raw multi-space indentation (repeated literal-space token), while `encode()` canonicalizes whitespace runs to a single `<SPACE>`/`<NL>` sentinel. The load-bearing guarantee is `reencode_idempotent` (deterministic, self-consistent) and byte-exact `text_roundtrip` on content without collapsed indentation.

### CODE parquet

Roundtrip sample: **88 rows**.

| metric | % |
|---|---|
| text_roundtrip (byte-exact) | 4.55 |
| reencode_idempotent (load-bearing) | 100.0 |
| id_exact (literal stored ids) | 0.0 |
| id_match_modulo_ws_collapse | 3.41 |


### COMMIT parquet

Roundtrip sample: **40 rows**.

| metric | % |
|---|---|
| text_roundtrip (byte-exact) | 0.0 |
| reencode_idempotent (load-bearing) | 100.0 |
| id_exact (literal stored ids) | 0.0 |
| id_match_modulo_ws_collapse | 0.0 |


## FILL % — CODE parquet

Sample: **11 files**, **414 rows**.

| family | channel | kind | rows-filled % | tokens-nonzero % |
|---|---|---|---|---|
| A_platform | `token_platform_ids` | per-token | 0.0 | 0.0 |
| B_structure | `token_structure_ids` | per-token | 100.0 | 83.31 |
| B_structure | `token_dep_levels` | per-token | 65.7 | 36.41 |
| B_structure | `token_ast_depth` | per-token | 100.0 | 48.49 |
| B_structure | `token_sibling_index` | per-token | 100.0 | 30.25 |
| B_structure | `token_ast_node_type` | per-token | 100.0 | 48.49 |
| C_graph_semantic | `token_symbol_ids` | per-token | 100.0 | 83.31 |
| C_graph_semantic | `token_def_use` | per-token | 100.0 | 83.31 |
| C_graph_semantic | `token_call_targets` | per-token | 68.84 | 2.09 |
| C_graph_semantic | `token_type_refs` | per-token | 93.72 | 2.58 |
| D_commit_edit | `token_change_mask_pre` | per-token | 0.0 | 0.0 |
| D_commit_edit | `token_change_mask_post` | per-token | 0.0 | 0.0 |
| D_commit_edit | `hunk_id_per_token` | per-token | 0.0 | 0.0 |
| D_commit_edit | `edit_op_per_token` | per-token | 0.0 | 0.0 |
| A_platform | `platform_ids` | list | 100.0 | (n/a) |
| C_graph_semantic | `token_call_edges` | list | 65.46 | (n/a) |
| C_graph_semantic | `token_type_edges` | list | 93.48 | (n/a) |
| D_commit_edit | `changed_chunk_ids` | list | 0.0 | (n/a) |
| D_commit_edit | `changed_chunk_spans` | list | 0.0 | (n/a) |

> D-family (`token_change_mask_*`, `hunk_id`, `edit_op`, `changed_chunk_*`) is EXPECTED empty for plain code rows — those channels only carry signal for commit docs.

## FILL % — COMMIT parquet

Sample: **5 files**, **193 rows**.

| family | channel | kind | rows-filled % | tokens-nonzero % |
|---|---|---|---|---|
| A_platform | `token_platform_ids` | per-token | 0.0 | 0.0 |
| B_structure | `token_structure_ids` | per-token | 100.0 | 65.51 |
| B_structure | `token_dep_levels` | per-token | 46.11 | 25.4 |
| B_structure | `token_ast_depth` | per-token | 100.0 | 63.51 |
| B_structure | `token_sibling_index` | per-token | 100.0 | 48.24 |
| B_structure | `token_ast_node_type` | per-token | 100.0 | 63.51 |
| C_graph_semantic | `token_symbol_ids` | per-token | 100.0 | 63.26 |
| C_graph_semantic | `token_def_use` | per-token | 100.0 | 63.51 |
| C_graph_semantic | `token_call_targets` | per-token | 69.95 | 3.27 |
| C_graph_semantic | `token_type_refs` | per-token | 79.27 | 4.41 |
| D_commit_edit | `token_change_mask_pre` | per-token | 99.48 | 25.4 |
| D_commit_edit | `token_change_mask_post` | per-token | 54.92 | 4.56 |
| D_commit_edit | `hunk_id_per_token` | per-token | 56.99 | 18.68 |
| D_commit_edit | `edit_op_per_token` | per-token | 100.0 | 49.38 |
| A_platform | `platform_ids` | list | 100.0 | (n/a) |
| C_graph_semantic | `token_call_edges` | list | 45.08 | (n/a) |
| C_graph_semantic | `token_type_edges` | list | 2.07 | (n/a) |
| D_commit_edit | `changed_chunk_ids` | list | 100.0 | (n/a) |
| D_commit_edit | `changed_chunk_spans` | list | 100.0 | (n/a) |

## Known gaps (channels empty where signal could exist)

- `token_platform_ids` (per-token A-platform) is **0% in BOTH code and commit** parquet across the whole sample. Platform signal is carried by the row-level `platform_ids` LIST (100% filled) instead; the per-token mirror column is not populated by the current indexer. The A-platform family IS filled — via the list channel, not the per-token column.
- D-family (`token_change_mask_*`, `hunk_id_per_token`, `edit_op_per_token`, `changed_chunk_*`) is **0% in CODE parquet** — correct/expected, those channels only carry signal in commit docs.
- `token_change_mask_post` is filled on only 54.92% of commit rows (many commits touch only the PRE side / are pure deletions or have an empty POST), and `token_type_edges` on 2.07% of commit rows — lower fill but present where the diff actually adds type relationships.
- `token_call_targets` / `token_type_refs` have HIGH rows-filled % but LOW tokens-nonzero % (~1-4%): they are sparse by design (only the few call/type reference sites per window carry an id).

## Worked examples of each sidecar JSON family

### Family A (platform) + B (structure) + C (graph-semantic) — from a CODE row
```json
{
  "A_platform": {
    "platform_ids": [
      2,
      62,
      93,
      109
    ]
  },
  "B_C_per_token_window": [
    {
      "i": 0,
      "tok_id": 2,
      "tok": "<BOS>",
      "A_platform": 0,
      "B_structure": "0:none/ws",
      "B_dep_lvl": 0,
      "B_ast_depth": 0,
      "B_sibling": 0,
      "B_ast_node": 0,
      "C_symbol": 0,
      "C_def_use": "0:none",
      "C_call_tgt": 0,
      "C_type_ref": 0,
      "D_chg_pre": 0,
      "D_chg_post": 0,
      "D_hunk": -1,
      "D_edit_op": "0:UNCHANGED"
    },
    {
      "i": 1,
      "tok_id": 347,
      "tok": "//",
      "A_platform": 0,
      "B_structure": "0:none/ws",
      "B_dep_lvl": 0,
      "B_ast_depth": 0,
      "B_sibling": 0,
      "B_ast_node": 0,
      "C_symbol": 0,
      "C_def_use": "0:none",
      "C_call_tgt": 0,
      "C_type_ref": 0,
      "D_chg_pre": 0,
      "D_chg_post": 0,
      "D_hunk": -1,
      "D_edit_op": "0:UNCHANGED"
    },
    {
      "i": 2,
      "tok_id": 1214,
      "tok": " ",
      "A_platform": 0,
      "B_structure": "0:none/ws",
      "B_dep_lvl": 0,
      "B_ast_depth": 0,
      "B_sibling": 0,
      "B_ast_node": 0,
      "C_symbol": 0,
      "C_def_use": "0:none",
      "C_call_tgt": 0,
      "C_type_ref": 0,
      "D_chg_pre": 0,
      "D_chg_post": 0,
      "D_hunk": -1,
      "D_edit_op": "0:UNCHANGED"
    },
    {
      "i": 3,
      "tok_id": 923,
      "tok": "l",
      "A_platform": 0,
      "B_structure": "0:none/ws",
      "B_dep_lvl": 0,
      "B_ast_depth": 0,
      "B_sibling": 0,
      "B_ast_node": 0,
      "C_symbol": 0,
      "C_def_use": "0:none",
      "C_call_tgt": 0,
      "C_type_ref": 0,
      "D_chg_pre": 0,
      "D_chg_post": 0,
      "D_hunk": -1,
      "D_edit_op": "0:UNCHANGED"
    },
    {
      "i": 4,
      "tok_id": 7587,
      "tok": "ang",
      "A_platform": 0,
      "B_structure": "0:none/ws",
      "B_dep_lvl": 0,
      "B_ast_depth": 0,
      "B_sibling": 0,
      "B_ast_node": 0,
      "C_symbol": 0,
      "C_def_use": "0:none",
      "C_call_tgt": 0,
      "C_type_ref": 0,
      "D_chg_pre": 0,
      "D_chg_post": 0,
      "D_hunk": -1,
      "D_edit_op": "0:UNCHANGED"
    },
    {
      "i": 5,
      "tok_id": 921,
      "tok": "u",
      "A_platform": 0,
      "B_structure": "0:none/ws",
      "B_dep_lvl": 0,
      "B_ast_depth": 0,
      "B_sibling": 0,
      "B_ast_node": 0,
      "C_symbol": 0,
      "C_def_use": "0:none",
      "C_call_tgt": 0,
      "C_type_ref": 0,
      "D_chg_pre": 0,
      "D_chg_post": 0,
      "D_hunk": -1,
      "D_edit_op": "0:UNCHANGED"
    },
    {
      "i": 6,
      "tok_id": 7429,
      "tok": "age",
      "A_platform": 0,
      "B_structure": "0:none/ws",
      "B_dep_lvl": 0,
      "B_ast_depth": 0,
      "B_sibling": 0,
      "B_ast_node": 0,
      "C_symbol": 0,
      "C_def_use": "0:none",
      "C_call_tgt": 0,
      "C_type_ref": 0,
      "D_chg_pre": 0,
      "D_chg_post": 0,
      "D_hunk": -1,
      "D_edit_op": "0:UNCHANGED"
    },
    {
      "i": 7,
      "tok_id": 359,
      "tok": ":",
      "A_platform": 0,
      "B_structure": "0:none/ws",
      "B_dep_lvl": 0,
      "B_ast_depth": 0,
      "B_sibling": 0,
      "B_ast_node": 0,
      "C_symbol": 0,
      "C_def_use": "0:none",
      "C_call_tgt": 0,
      "C_type_ref": 0,
      "D_chg_pre": 0,
      "D_chg_post": 0,
      "D_hunk": -1,
      "D_edit_op": "0:UNCHANGED"
    },
    {
      "i": 8,
      "tok_id": 1214,
      "tok": " ",
      "A_platform": 0,
      "B_structure": "0:none/ws",
      "B_dep_lvl": 0,
      "B_ast_depth": 0,
      "B_sibling": 0,
      "B_ast_node": 0,
      "C_symbol": 0,
      "C_def_use": "0:none",
      "C_call_tgt": 0,
      "C_type_ref": 0,
      "D_chg_pre": 0,
      "D_chg_post": 0,
      "D_hunk": -1,
      "D_edit_op": "0:UNCHANGED"
    },
    {
      "i": 9,
      "tok_id": 9677,
      "tok": "prim",
      "A_platform": 0,
      "B_structure": "0:none/ws",
      "B_dep_lvl": 0,
      "B_ast_depth": 0,
      "B_sibling": 0,
      "B_ast_node": 0,
      "C_symbol": 0,
      "C_def_use": "0:none",
      "C_call_tgt": 0,
      "C_type_ref": 0,
      "D_chg_pre": 0,
      "D_chg_post": 0,
      "D_hunk": -1,
      "D_edit_op": "0:UNCHANGED"
    },
    {
      "i": 10,
      "tok_id": 4745,
      "tok": "ary",
      "A_platform": 0,
      "B_structure": "0:none/ws",
      "B_dep_lvl": 0,
      "B_ast_depth": 0,
      "B_sibling": 0,
      "B_ast_node": 0,
      "C_symbol": 0,
      "C_def_use": "0:none",
      "C_call_tgt": 0,
      "C_type_ref": 0,
      "D_chg_pre": 0,
      "D_chg_post": 0,
      "D_hunk": -1,
      "D_edit_op": "0:UNCHANGED"
    },
    {
      "i": 11,
      "tok_id": 373,
      "tok": "=",
      "A_platform": 0,
      "B_structure": "0:none/ws",
      "B_dep_lvl": 0,
      "B_ast_depth": 0,
      "B_sibling": 0,
      "B_ast_node": 0,
      "C_symbol": 0,
      "C_def_use": "0:none",
      "C_call_tgt": 0,
      "C_type_ref": 0,
      "D_chg_pre": 0,
      "D_chg_post": 0,
      "D_hunk": -1,
    
```
### Family D (commit-edit) — from a COMMIT row
```json
{
  "D_changed_chunks": {
    "changed_chunk_ids": [
      2,
      4
    ],
    "changed_chunk_spans": [
      {
        "start": 202,
        "end": 409
      },
      {
        "start": 443,
        "end": 769
      }
    ]
  },
  "D_per_token_window": [
    {
      "i": 198,
      "tok_id": 5501,
      "tok": "COMMIT",
      "D_chg_pre": 0,
      "D_chg_post": 0,
      "D_hunk": -1,
      "D_edit_op": "3:CONTEXT"
    },
    {
      "i": 199,
      "tok_id": 1214,
      "tok": " ",
      "D_chg_pre": 0,
      "D_chg_post": 0,
      "D_hunk": -1,
      "D_edit_op": "3:CONTEXT"
    },
    {
      "i": 200,
      "tok_id": 327,
      "tok": "==",
      "D_chg_pre": 0,
      "D_chg_post": 0,
      "D_hunk": -1,
      "D_edit_op": "3:CONTEXT"
    },
    {
      "i": 201,
      "tok_id": 373,
      "tok": "=",
      "D_chg_pre": 0,
      "D_chg_post": 0,
      "D_hunk": -1,
      "D_edit_op": "3:CONTEXT"
    },
    {
      "i": 202,
      "tok_id": 133,
      "tok": "int",
      "D_chg_pre": 1,
      "D_chg_post": 0,
      "D_hunk": 0,
      "D_edit_op": "3:CONTEXT"
    },
    {
      "i": 203,
      "tok_id": 1214,
      "tok": " ",
      "D_chg_pre": 1,
      "D_chg_post": 0,
      "D_hunk": 0,
      "D_edit_op": "3:CONTEXT"
    },
    {
      "i": 204,
      "tok_id": 7248,
      "tok": "s",
      "D_chg_pre": 1,
      "D_chg_post": 0,
      "D_hunk": 0,
      "D_edit_op": "3:CONTEXT"
    },
    {
      "i": 205,
      "tok_id": 4902,
      "tok": "2",
      "D_chg_pre": 1,
      "D_chg_post": 0,
      "D_hunk": 0,
      "D_edit_op": "3:CONTEXT"
    },
    {
      "i": 206,
      "tok_id": 7243,
      "tok": "n",
      "D_chg_pre": 1,
      "D_chg_post": 0,
      "D_hunk": 0,
      "D_edit_op": "3:CONTEXT"
    },
    {
      "i": 207,
      "tok_id": 377,
      "tok": "_",
      "D_chg_pre": 1,
      "D_chg_post": 0,
      "D_hunk": 0,
      "D_edit_op": "3:CONTEXT"
    },
    {
      "i": 208,
      "tok_id": 2268,
      "tok": "connect",
      "D_chg_pre": 1,
      "D_chg_post": 0,
      "D_hunk": 0,
      "D_edit_op": "3:CONTEXT"
    },
    {
      "i": 209,
      "tok_id": 7268,
      "tok": "ion",
      "D_chg_pre": 1,
      "D_chg_post": 0,
      "D_hunk": 0,
      "D_edit_op": "3:CONTEXT"
    }
  ]
}
```