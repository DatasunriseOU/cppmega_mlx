# Conveyor-output verification summary

Selective ~200-sample verification of the LIVE conveyor pipeline output, sampled
directly from the accumulating streams in `outputs/reindexed/{1024,2048,4096}`
(code) and `outputs/reindexed_commits/{1024,2048,4096,8192,16384}` (commits).
Each sample was detokenized through `CppMegaTokenizer.decode` (auto-converts
`<SPACE>`=46 / `<NL>`=47 to real whitespace), stripped of special tokens
(`<BOS>`=2, `<PAD>`=0, FIM 4/5/6/45) and the language/platform header comments,
then re-encoded for an idempotence check, run through clang-format
(`/opt/homebrew/opt/llvm/bin/clang-format`, LLVM style), and compile-probed
(`clang -std=c++17 -fsyntax-only`) to distinguish missing-include vs corrupt.
Sidecar channels were rendered as labelled JSON (families A platform / B structure
/ C graph / D commit-edit) and fill-counted.

- Samples: 210 (CODE 110, COMMIT 70, BUILD-FILE 30 = Bazel 13, CMake 15, Make 2)
- Read-only; concurrent-writer race handler (FileNotFoundError / ArrowInvalid
  skip-and-record) in place but **0 rows skipped** — never triggered.
- Corpus at sample time: 22,113 code rows, 1,709 commit rows.
- Driver: `scripts/_verify_conveyor_live.py`; report JSON:
  `outputs/verification_report/live_verification_report.json`; 211 rendered
  samples: `outputs/verification_report/conveyor_samples/`.

## Per-type results

| Metric | CODE (n=110) | COMMIT (n=70) | BUILD-FILE (n=30) |
|---|---|---|---|
| reencode idempotent (load-bearing) | 100% | 100% | 100% |
| text roundtrip | 98.2% | 100% | 100% |
| NL token present | 100% | 100% | 100% |
| clang-format ok | 100% | 100% | 100% |
| compile-probe: full-compile | 0% | n/a | 0%* |
| compile-probe: missing-include | 100% | n/a | 60%* |
| compile-probe: corrupt | 0% | n/a | 40%* (FALSE POSITIVE) |
| PR-as-docstring HEAD | n/a | 100% (`@brief`) | n/a |

\* CODE 0% full-compile is **expected** — snippets are header-less function/class
fragments (100% missing-include, 0% corrupt, not a defect). The BUILD-FILE 40%
"corrupt" is a **false positive** of the C++ probe: CMake/Bazel/Make are not C++,
so `clang -x c++` rejects Starlark/CMake syntax. Decoded build bodies are clean,
line-structured, `#` comments intact, clang-format ok = 100%. Correct
verification for build docs is JSON/structure render, not a C++ compile.

## Sidecar channel fill %

| Family / channel | CODE | COMMIT | BUILD-FILE |
|---|---|---|---|
| A platform_ids (per-doc) | 100 | 100 | 100 |
| A token_platform_ids (per-token) | **0** | **0** | **0** |
| B token_structure_ids | 72.4 | 72.5 | 81.1 |
| B token_dep_levels | 2.1 | 8.1 | 0 |
| B token_ast_depth | 65.6 | 43.4 | 0 |
| B token_sibling_index | 45.1 | 31.8 | 0 |
| B token_ast_node_type | 65.6 | 43.4 | 0 |
| C token_symbol_ids | 71.5 | 68.5 | 0 |
| C token_def_use | 71.5 | 68.5 | 0 |
| C token_call_targets | **0.3** | **1.1** | 0 |
| C token_type_refs | **0.6** | **0.9** | 0 |
| C token_call_edges (presence) | 12.7 | 28.6 | 0 |
| C token_type_edges (presence) | 30.9 | 21.4 | 0 |
| D token_change_mask_pre | — | 29.3 | — |
| D token_change_mask_post | — | 3.5 | — |
| D hunk_id_per_token | — | 19.8 | — |
| D edit_op_per_token | — | 59.5 | — |
| D changed_chunk_ids (presence) | — | 100 | — |
| D changed_chunk_spans (presence) | — | 100 | — |

Build docs carry platform (family A) + structure (B token_structure_ids) only —
no AST/graph/commit channels, as designed. NL content is present in every type
(47 distinct NL-bearing rows; CODE averages multiple-hundred NL tokens/sample).

## PR-as-docstring sample (CONFIRMED)

The commit docstring HEAD carries the full PR thread (`@brief` -> `@repo` ->
`File:` -> `@sha` -> `@pr` -> `@discussion`) before the PRE/POST/diff sections.
Verified order on `apex_r0` row 0: docstring ends idx 545, PRE-COMMIT at 549,
POST-COMMIT at 765. Rendered example:
`outputs/verification_report/conveyor_samples/commit_DISCUSSION_example.md`.

```c
/**
 * @brief use mem pool API for NCCL zero-copy (#1863)
 *
 * @repo NVIDIA/apex
 * File: apex/contrib/csrc/nccl_allocator/NCCLAllocator.cpp
 * @sha 94705018ceab50f5aa24c7c42c693868dcf042bc
 * @pr 1863
 *
 * @discussion
 * PR #1863: Use Mem Pool API for NCCL Zero-Copy
 *
 * per title
 * cc @syed-ahmed @xwang233
 *
 * --- Discussion (1 comments) ---
 * @syed-ahmed: LGTM.
 */
```

Corpus-wide over all 1,709 commit rows: `@brief` 100% (1709), `@repo` 99.9%
(1707), `@sha` 99.9% (1707), `@pr` 39.3% (671), `@discussion` 6.6% (112).
`@pr`/`@discussion` appear only when `pr_store` has a matching PR (per-repo:
s2n 548/935, apex 112/212, abseil 11/562). The 70-sample `pr_number_pct=0` was a
**sampling + regex artifact** (drew mostly s2n/abseil rows without PRs; the regex
matched a code-body `@pr` rather than the docstring) — **not a data defect**.

## Dedup

- `outputs/dedup_seen.sqlite` (~596 MB): `exact`=105,051 rows, `lsh`=2,533,975,
  `minhash`=101,359, `dedup_meta`=1.
- Spot-check of 140 sampled CODE+BUILD bodies (normalized-whitespace SHA1):
  **0 exact-duplicate function-body groups** -> dedup applied, no exact dupes
  across the sample.

## Build-file tagging (CONFIRMED)

Tagged via language header `language: primary={cmake|bazel|make|compile_commands}`
plus per-doc `platform_ids` (x86_64-linux-gnu) 100% filled. Build systems
identified across the sample: Bazel, CMake, Make all present.

## Gaps and flags

REAL FLAGS:
1. **`token_platform_ids` (per-token platform channel) is EMPTY (0%) across ALL
   210 samples / all types.** The current indexer does not populate the per-token
   platform stream. Platform info is NOT lost — it is carried at doc level
   (`platform_ids` family A = 100% filled, x86_64-linux-gnu) and in the language
   header — but the per-token channel is empty-where-arguably-expected.
2. **`token_call_targets` (0.3% CODE / 1.1% COMMIT) and `token_type_refs`
   (0.6% / 0.9%) are near-empty.** The per-token call/type-ref scalar pointers
   are essentially unfilled; the call/type relationships are instead carried by
   `token_call_edges` / `token_type_edges` (presence 12.7-30.9%), so family-C
   graph is present via edges but not via the per-token target/ref channels.
3. **`token_dep_levels` low on CODE (2.1%).**

NON-ISSUES (investigated, not defects):
- CODE 0% full-compile — header-less fragments (100% missing-include, 0% corrupt).
- BUILD-FILE 40% "corrupt" — false positive of the C++ probe; build files are not
  C++. Decoded bodies clean, clang-format ok 100%.
- `@pr`/`@discussion` < 100% — by-design; present only where a PR exists.

Overall: deterministic re-encode roundtrip is 100% across all types (the
load-bearing guarantee); detok/strip/format is clean; PR-as-docstring,
build-file tagging, and dedup are all confirmed. The single notable empty
channel is per-token platform (`token_platform_ids`), with platform still carried
at doc level.
