# Data, Parquet, and Megatron Production Review

Date: 2026-07-14 (Europe/Budapest)

## Scope and decision

This review covers the current generator, clang indexer, git/PR path, typed
domain and tokenizer contracts, packing, objective materialization, Megatron
conversion, bundle publication, and S3 restore implementation. It also audits
the live output run and bundle below without modifying either:

- Run: `macro_routes_v1_20260710_135335`
- Live output root: `/Volumes/external/sources/cppmega.mlx/outputs`
- Frozen bundle: `megatron_ready/macro_routes_v1_20260713`
- Implementation checkout: `/Volumes/external/sources/cppmega_mlx_full_integration`
- Converter/bundle checkout: `/Volumes/external/sources/cppmega_full_integration`

**Decision: do not train from, re-label, or republish the existing bundle as
the new production dataset.** The frozen bundle is internally consistent for
its legacy schema after the recorded boundary repair, but it is incomplete,
not objective-materialized, lacks a frozen tokenizer payload, lacks the new
source/symbol identity contract, and cannot pass the current restore validator.
The live parquet roots are also legacy and contain 39 unmanifested shards.

The production-critical source/objective provenance gap found during this review
was real in the inspected pre-fix code, but is now **fixed in code** by
`8a1b594` (objective materializer) and `ef44b42` (bundle builder). New objective
contracts hash the exact input parquet set, and the builder requires that byte
set to equal its repaired snapshot before conversion. The existing frozen bundle
and all old objective artifacts remain unbound legacy data and cannot be
grandfathered into the new contract.

The builder/publisher descriptor mismatch found immediately after the provenance
fix was closed during this review by `60f9891`; failed-freeze rejection and remote
logical-contract preflight were then added by `f8d5b30` and `e1a9a8c`. These are
current code guarantees, not guarantees retroactively present in the produced
20260713 bundle. No new source-bound bundle has yet completed build, publish, and
restore. Current Stage-1 also still bypasses the verified bundle and opens bare
same-schema prefixes without artifact identity checks, so the chain is not yet
end-to-end provenance-safe.

| ID | Severity | Current status |
|---|---|---|
| F1 | CRITICAL/P0 provenance | Fixed in code by `8a1b594` / `ef44b42`; existing produced data remains unbound and rejected |
| F2 | CRITICAL/P0 scalability | Open: all-document retention, whole-shard Python conversion, and graph-pair expansion make 1M+ materialization non-runnable |
| F3 | CRITICAL/P0 training ingress | Open: production MLX accepts bare/unlisted prefixes without immutable bundle, source, objective, tokenizer, or artifact hash verification |
| F4 | HIGH/P1 objective coverage | Open: 50% of the default objective mix drops all graph routes; 20% commit objectives also zero most semantic/change sidecars |
| F5 | HIGH/P1 integration | Fixed in code by `60f9891`; no real source-bound bundle has yet proved builder -> publisher -> restore composition |
| F6 | HIGH/P1 artifact | Existing uploaded bundle is legacy causal data, not current objective-materialized/restorable data |
| F7 | HIGH/P1 completeness | Existing freeze accepted 16 failures and live run has 18; current builder rejects this state via `f8d5b30` |
| F8 | HIGH/P1 schema/corruption | Old 73-column parquet is not upgradable; live code/1024 had 3,719,382 leaked cross-document loss targets before bundle repair |

## Evidence snapshot

Code was changing concurrently. Line references in this report are tied to
the inspected commits, not to later branch movement or uncommitted work:

| Item | Inspected identity |
|---|---|
| MLX integration checkout | `f9f1686f782229b740c9cae5b3509aa0f075c050`; objective remap `ddaafc34b8cf7f5cd64e8234ca25e2b86536815f` and source binding `8a1b5947890576c0aa6cab558f61b1f6a0294d64` are ancestors; worktree was clean except this report at final staging |
| Converter/bundle checkout | `e1a9a8c38f75dc88aee9d1ce6f494d0838cab389`; source binding `ef44b42`, descriptor validation `60f9891`, failed-freeze rejection `f8d5b30`, and remote preflight `e1a9a8c` are all included; only unrelated `outputs/review/` was untracked |
| Git recorded in frozen bundle | `cppmega=8eb95bd344dadabc2b0c677b83bc7008f2e8da18`; `cppmega_mlx=dd2b703bd69d197569f2faa41b51cbd5c83f0829` |
| Bundle ID | `macro_routes_v1_20260713-8c550514dd52ddc9` |
| Artifact set | `8c550514dd52ddc99ecad91cfd5e4b0355c194c5d3e38d6686550cf4fd2d088b` |
| Bundle logical manifest | `1e44fcd8ff9192a15e25b62c18ac71569ca7627e3f2221b29d249d5c2fb93391` |
| Frozen source manifest | `cf5e69f1aa67e4e00c29ed3c616e58214bac2b1c841ca8e33b1e6961f3f9090c` |
| Repaired source manifest | `c15901acc7821dae8b7c5056d694597bd5cdad674d5c583f326ed7dab9d547dd` |
| Audit receipt | `788a6b1d524e1bd9c04e110db9f18b35d92d8d31d5d2541e3d08fea8d3cb60be` |
| Current live `_done.json` | `ef9340ceec3b66dbbab07e83a3b2ab8589496f1e65f06415d2cb733297b3f710` |
| Frozen `_done.json` captured by source manifest | `319f2f6ffb416cb14d716346fb2bef47a1248510a1a8d4ea3f911279767e5199` |

The bundle records implementation hashes for the legacy builder, converter,
audit, and repair scripts respectively as `615a6e0e...3570a`,
`670435c6...dbf6`, `740b5793...f88f`, and `4eea9f67...8e34`.

## Findings

### F1 - CRITICAL/P0 historical gap: fixed in code; produced data remains unbound

At pre-fix commits `c54750e` / `5a72342`, the builder snapshot and audit chain
was separate from the external objective artifact chain. The materializer took a
free-form `--data-glob`; the artifact bound only its objective contract and output
parquet; and `_build_bucket` converted only that external artifact. Dataset B
could therefore provide the actual training objectives while the bundle claimed
source snapshot A. Equal row and token counts did not prevent substitution.

The live run provides a concrete accidental trigger: 39 parquet files are in
neither the frozen manifest nor current `_done.json` (8 each in
1024/2048/4096/8192 and 7 in 16384), totaling 3,887 rows, 8,599,023 valid tokens,
and 8,595,129 trained tokens. They are partial Paddle, radare2, and Valgrind range
publications. A directory glob saw them while the manifest-aware snapshot did
not. Publication precedes dedup promotion and the durable done mark
(`[mlx] scripts/streaming_reindex_commits.py:475-500,593-625` and
`scripts/streaming_conveyor.py:2203-2214`), so this is a real crash window.

The following current implementation closes the A/B byte-substitution path for
new artifacts:

1. `8a1b594` resolves and sorts the exact glob result, hashes every complete
   parquet, records absolute path, size, SHA-256, row count, bucket, and sampling
   schedule, and detects stat drift both while hashing and after materialization
   (`[mlx] scripts/materialize_megatron_objectives.py:103-198,441-496`).
2. That `cppmega_objective_source_snapshot_v1` object is part of the objective
   contract. The contract file hash and contract payload hash are in the
   objective artifact, whose `artifact_set_sha256` covers those references
   (`[converter] cppmega/megatron/objective_contract.py:474-629`).
3. `ef44b42` loads the source binding and requires the exact multiset of
   `(size_bytes, sha256)` records to equal every code and commit record for that
   bucket in the builder's repaired snapshot. It recomputes the binding digest
   and does this before audit and conversion
   (`[converter] scripts/data/build_macro_routes_megatron_bundle.py:460-553,913-926`).
4. The staged objective contract retains the full file list, while the bundle
   descriptor carries its schema, digest, file/row counts, and sampling summary
   (`build_macro_routes_megatron_bundle.py:955-972`). `60f9891` additionally
   recomputes the full file digest, row sum, and sampling equations and compares
   the staged contract to that descriptor during publish/restore validation
   (`publish_megatron_bundle_to_nebius_s3.py:1290-1405,1510-1565`).

Thus A/A passes and A/B with identical counts but different bytes fails before
conversion; an orphan admitted by `--data-glob` also makes the multiset differ
and fails assembly. Focused tests cover changed source bytes and A/B mismatch at
`[mlx] tests/test_materialize_megatron_contract_hashes.py:84-119` and
`[converter] tests/test_build_macro_routes_megatron_bundle.py:126-198`.

**Data status remains P0:** `macro_routes_v1_20260713` predates both commits, has
no objective materialization at all, and cannot acquire this provenance by
editing metadata. Old objective artifacts with no `source_snapshot` are rejected
by the new builder. Regenerate objectives from frozen/repaired source bytes.

Residual P2 hardening: the exact source byte multiset is bound, and `60f9891`
now catches row-count/sampling drift before publication, but the builder itself
still performs only the weaker count/mode checks. Neither layer binds the exact
repaired-manifest file hash, stream kind, or canonical relative path. This does
not reopen dataset A/B byte substitution, but it permits provenance relabeling of
identical bytes and delays malformed sampling rejection until publish. Share one
validator at build and publish, and add `repaired_manifest_sha256`, `kind`,
`bucket`, and canonical relative paths to the source contract. For operational
clarity, retain a two-phase `freeze -> materialize -> assemble` workflow keyed by
the immutable repaired-manifest hash.

### F2 - CRITICAL/P0: objective materialization is unbounded and will OOM at production scale

`scripts/materialize_megatron_objectives.py` is a correctness prototype, not a
production-scale materializer. It appends every fully materialized document to
one Python list, computes the contract only after the list is complete, and only
then writes parquet (`:457-510`). Each `MaterializedMegatronDocument` retains a
row dictionary containing `input_ids` plus 20 token-aligned sidecar lists
(`[mlx] cppmega_mlx/training/megatron_objectives.py:48-69,460-466,1026-1035`).
The source reader separately calls `ParquetFile.read()` and `to_pylist()` for all
selected columns of an entire source shard before yielding any row
(`scripts/materialize_megatron_objectives.py:295-320`).

This is a deterministic OOM, not merely a possible inefficiency. On this CPython
build a list slot is 8 bytes. Twenty-one token-aligned lists therefore have a
**168-byte/token pointer-array lower bound**, excluding every Python integer,
list header, row dictionary, edge/registry record, MLX/NumPy object, source-shard
buffer, and Arrow conversion. The fixed-width Arrow payload itself is 70 bytes
per token before list offsets, strings, graphs, registries, and compression.

| 1M objective documents | Pointer-only lower bound | Fixed-width Arrow payload |
|---|---:|---:|
| Full 1,025-token document | 160.4 GiB | 66.8 GiB |
| Full 4,097-token document | 641.0 GiB | 267.1 GiB |
| Full 16,385-token document | 2,563.6 GiB | 1,068.2 GiB |
| Frozen-run weighted mean, 1,570.513 valid tokens/document | **245.7 GiB** | **102.4 GiB** |

The review host has 128 GiB physical RAM. Even the pointer-only lower bound is
1.92x RAM for one million documents at the observed mix. Real resident memory is
substantially higher because non-interned semantic/source IDs are Python integers
(typically 28-32 bytes each). The default `--shard-rows=1024` adds another padded
copy of all 21 lists before `pa.Table.from_pylist`; for sequence length 16,384
that transient alone has a 2.625 GiB pointer floor plus 1.094 GiB Arrow payload
(`scripts/materialize_megatron_objectives.py:256-278,500-509`). Existing documents
remain resident throughout.

There is a second unbounded path in contract construction. Call/type chunk edges
are expanded into a Python `set` containing their cartesian token-span pairs
(`[mlx] cppmega_mlx/training/megatron_objectives.py:1164-1218`). One edge between
two 8K-token spans can attempt about 67 million tuples even when the spans are
disjoint within a 16K sequence; if overlapping full-length spans are accepted,
the loop has a 268-million-pair upper case. There is no expansion guard. A single
document can therefore OOM even before the corpus-retention limit.

**Verdict: do not launch 1M+ objective materialization with this script.** A CLI
memory estimate or larger swap is not an acceptable workaround. The required
patch shape is:

1. Replace `build_pre_materialized_objective_contract(documents, ...)` with an
   `ObjectiveContractAccumulator` that consumes each document once and tracks
   per-task samples/input/loss tokens, exact per-window quotas, graph-eligible
   samples, and positive-edge totals in O(number of tasks) state.
2. Retain at most one quota window (currently 60 sources/documents) plus one small
   Arrow write batch. Materialize, validate, account, write, and release each
   window before reading the next.
3. Use `ParquetWriter` with a separate `--write-batch-rows` chosen from a hard
   `--max-buffer-bytes` budget. Rotate a closed output shard after
   `--shard-rows`, but do not construct all 1,024 padded rows at once.
4. Replace whole-shard `read().to_pylist()` with bounded row-group/record-batch
   reads. Version the sampling mode as deterministic shard -> row-group -> row
   shuffle; record epoch/cursor/seed so the schedule is reproducible without
   random whole-shard residency, and update the builder's sampling-mode validator
   in the same change.
5. Replace graph pair materialization with interval-union counting per query token
   (merge destination intervals across configured relations, clipped by causal
   and same-document boundaries). Runtime must be bounded by token/chunk-edge
   counts, never by the number of cartesian token pairs.
6. Write into a new partial directory, close and hash each parquet before an
   atomic checkpoint, and persist source cursor, accumulator state, completed
   output hashes, and source-snapshot digest. On resume, revalidate all inputs and
   completed outputs; publish contract/artifact and rename final output only after
   exact totals and post-read source verification pass.
7. Add a streaming-vs-reference test on a small fixture, exact quota/contract and
   deterministic-resume tests, source-mutation failure, a large-edge test proving
   no pair set is built, and a subprocess RSS gate proving buffered documents
   never exceed `quota_window_samples + write_batch_rows`.

Production acceptance requires an observed peak-RSS receipt under a representative
multi-bucket load, a configured fail-closed RAM budget, restart equivalence, and
zero partial files in the published artifact. None exists for the current script.

### F3 - CRITICAL/P0: production MLX ingress bypasses immutable bundle provenance

The producer and restore paths can hash-bind a bundle, but the production MLX
consumer does not require that bundle. `run_stage1_graph_domain_production()`
accepts `--production-graph-domain-data` as a bare Megatron prefix/directory and
calls `open_megatron_indexed_dataset()` directly for both startup and training
(`[mlx] cppmega_mlx/training/stage1_production.py:216-289`). The opener accepts
either a directory or standalone prefix and has no bundle ID, manifest, expected
artifact-set digest, bucket, or restore-receipt argument
(`cppmega_mlx/data/megatron_indexed.py:827-876`).

The local reader's structural checks are useful but are not provenance checks:

- a missing JSON sidecar loads as `{}` (`megatron_indexed.py:1160-1178`), then
  tokenizer metadata defaults to local constants (`:1892-1905`);
- directory ingress enumerates every `.idx` file rather than a manifest allowlist
  (`:1030-1042`);
- multi-shard validation compares token dtype and sidecar shape/dtype/schema
  consistency, then compares tokenizer labels and vocab sizes, but never hashes
  tokenizer/domain/CASE5/objective/source/bundle artifacts (`:1045-1118`).

This is intentional in current unit coverage: a sidecar-less MMIDIDX fixture is
accepted and reported as standalone training ingress at
`tests/test_megatron_indexed.py:319-340,386-397`; both focused cases passed live.
Consequently same-shape shards from dataset B, an extra unlisted `.idx`, or a
post-restore byte mutation can enter a run that is operationally attributed to
bundle A. The upstream F1 binding does not protect the final training edge.

Fix: add a separate production-only
`open_production_megatron_bundle(bundle_root, bucket, expected_bundle_id)` and
make Stage-1 use it exclusively. It must run the shared current bundle validator,
require `objective_materialized`, select prefixes only from `bucket_results`,
verify the full artifact set plus tokenizer/domain/CASE5/objective/source hashes,
reject unlisted/symlinked files, and bind the retained restore receipt. Record
`bundle_id`, artifact-set SHA-256, source-snapshot digest, objective contract and
artifact hashes, tokenizer/domain hashes, and bucket in every run/checkpoint
receipt. One-byte mutation, mixed-bundle shard, unlisted `.idx`, missing sidecar,
wrong bucket, and stale objective tests must fail before mmap/model construction.
Keep the bare-prefix opener only as an explicitly non-production API.

### F4 - HIGH/P1: non-causal objective materialization drops graph and semantic coverage

`ddaafc3` materially improves domain and physical-source remapping for transformed
FIM/recovery tokens, but the final materialization step still clears every chunk
and graph sidecar whenever the objective is not aligned causal LM
(`[mlx] cppmega_mlx/training/megatron_objectives.py:105-120,1000-1014`). The
default mix is 50% causal and exactly 50% non-causal: 30% transformed code
(FIM/AST-FIM/IFIM/recovery) plus 20% commit/pre-to-post
(`cppmega_mlx/training/task_mixer.py:57-70`). Therefore half of all planned
objective documents have zero call/type/domain/build/shell/diagnostic/cross-domain
edges by construction, even when the source row had valid routes.

The commit branch first zeros every mapped token sidecar, then restores only
delimiter-derived domain/role/confidence and one source-document/physical-source
identity (`megatron_objectives.py:907-996`). Symbol IDs, call/type refs, def-use,
structure, AST, and both change masks remain zero for the 20% commit objective
share. This is especially weak for commit/pre-to-post objectives whose purpose is
temporal change modeling.

The contract records objective sample/token totals but only global graph
`eligible_samples` and `positive_edges`; it passes when any causal documents have
edges and has no per-objective semantic/graph coverage floor (`:1141-1240`). The
focused contract fixture itself gives a graph edge only to causal LM
(`tests/test_production_objective_mixer.py:1212-1223`). Thus exact quota success
does not prove the configured objectives retained the typed signals they were
meant to train.

Fix before production materialization: remap chunk spans and graph endpoints
through the exact transformed source-token map, splitting fragmented chunks and
dropping only individually unmappable endpoints with explicit reason counts. For
commit objectives, carry typed pre/post/diff section provenance, temporal change
masks, and source semantic IDs; inserted prompt/control tokens need explicit
sentinel roles rather than blanket zeroing. Extend the objective contract with
per-objective nonzero token counts for domain/source/symbol/temporal channels,
per-relation edge/sample totals, and ineligibility/drop reasons. Set reviewed
minimums and fail before writing the artifact when a required objective/channel
collapses to zero. Add all-nine-objective parquet round-trip tests with real
multi-document and graph-bearing source rows.

### F5 - HIGH/P1 integration gap: fixed in code, not yet proved on a real bundle

Immediately after `ef44b42`, the builder emitted a ninth `source_snapshot` field
while the publisher required the old exact eight-field descriptor. A bounded
composition check reproduced
`ValueError: bundle objective materialization descriptor is invalid for 1024`.

`60f9891` closes that code defect. The shared publish/restore validator now
requires the summary field, validates its exact shape, recomputes source file and
row counts, canonical source digest, and sampling equations, then requires the
staged full objective contract to produce exactly the same summary
(`[converter] scripts/data/publish_megatron_bundle_to_nebius_s3.py:1290-1465,1510-1562`).
Publisher fixtures now contain the source snapshot, and missing or altered
summaries fail (`tests/test_publish_megatron_bundle_to_nebius_s3.py:627-650`).
The current four-file converter/builder/publisher/restore suite passes 131 tests.

Residual acceptance risk: the tests still construct publisher fixtures
independently rather than running the complete, expensive builder and feeding its
actual manifest to publish and restore. No source-bound bundle has been produced,
published, or restored. Keep this P1 gate open operationally until one real
builder -> local validator -> immutable upload -> fresh restore receipt completes.
To prevent future schema drift, move the builder descriptor constructor and
validator to one shared contract module.

### F6 - HIGH/P1: the uploaded bundle is legacy, non-objective, and unsupported by current restore

The produced manifest has no `training_contract`, `objective_materialization`,
`tokenizer`, or `data_contracts`. Its per-prefix JSON has no objective contract
or objective materialization. It is therefore a legacy causal packed dataset,
not the current multi-objective training artifact. Objective coverage is exactly
zero materialized objective labels/contracts; no claimed quota can be audited.

The bundle also stages no tokenizer files. It claims only the string
`megacpp-vocab-65536`; neither source parquet metadata nor the bundle binds the
token IDs to tokenizer bytes. Current code validates tokenizer contents and
contract hashes (`[mlx] cppmega_mlx/data/tokenizer_contract.py:151-183`) and
requires frozen tokenizer/domain hashes in parquet
(`cppmega_mlx/data/domain_schema.py:159-181`). Those guarantees are absent here.

The current publisher rejects this local bundle immediately:

```text
ValueError: bundle training_contract must be 'objective_materialized'
```

That now comes from the pure logical contract validator at `[converter]
scripts/data/publish_megatron_bundle_to_nebius_s3.py:1408-1465`. `e1a9a8c`
imports the same validator into restore and calls it immediately after the small
remote logical manifest's size/SHA verification, before output allocation or
archive acquisition (`restore_megatron_bundle_from_nebius_s3.py:531-552`). The
focused regression proves a legacy contract cannot start archive download
(`tests/test_restore_megatron_bundle_from_nebius_s3.py:336-385`).

No `restore_receipt.json` exists, and current restore now correctly refuses to
create one for this legacy archive. The remote object is present but intentionally
unsupported. Preserve it as historical evidence; do not add a metadata shim or
label it restorable/current. Only regeneration can supply objective, tokenizer,
data-contract, and source bindings.

### F7 - HIGH/P1: existing freeze is incomplete; current code now rejects it

The frozen source manifest records 2,401 done units and **16 failed units**.
Current live state is 2,411 done and **18 failed**. The ten additional done entries
are commit-plan metadata, not new training shards; no manifest-complete training
tokens were added after freeze.

The builder version that produced the bundle derived an allowlist from `done` and
recorded, but did not reject, the failed count. This makes the artifact a completed
subset, not a completed run. The conveyor itself correctly returned nonzero when
failures remained (`[mlx] scripts/streaming_conveyor.py:3436-3469`), but the old
freeze bypassed that terminal condition. `f8d5b30` fixes current code:
`_load_manifest_allowlist()` now requires a failed map and rejects any non-empty
value before snapshot work (`[converter]
scripts/data/build_macro_routes_megatron_bundle.py:115-128`), with a focused
regression at `tests/test_build_macro_routes_megatron_bundle.py:359-382`.

Current unresolved units are:

| Unit(s) | Stage/evidence |
|---|---|
| `blender::commits`, `paraview::commits` | interrupted `extract_git_history`, exit `-2`; resumable caches retained |
| `mysql::commits`, `php-src::commits` | extraction exit `1` |
| `paddle::r13520` | interrupted `process_commits`, exit `-2` |
| `radare2::r0` | `pack_16384` rejected 22 unique platform IDs over `MAX_PLATFORM_IDS=20`; lower-bucket files remain orphaned |
| `dealii::code`, `intel-llvm-dpcpp::code`, `linux::code`, `llvm-project::code`, `mingw-w64::code` | index exit `1` |
| `dragonflybsd::code` | stalled for 1,800 seconds |
| `freebsd-src::code`, `nt5src::code`, `oceanbase::code` | exit `137` |
| `open-watcom-v2::code`, `open-watcom::code` | exit `-15` |
| `gsl::code` | successful index produced no training docs after dedup |

Do not bypass the new `failed == {}` gate to reproduce this subset. A deliberate
exclusion must instead be a reviewed, hash-bound policy listing every unit and
reason. All consumers, including objective materialization, must consume a
manifest, never directory globs. Orphan shards must be quarantined or removed
only after a dry-run inventory and operator approval; this review did not modify
them.

### F8 - HIGH/P1: old parquet is not upgradable in place and had a real boundary defect

All 11,771 frozen parquet files have the same 73-column legacy family and only
this footer metadata:

```text
cppmega.macro_routes_version=full_macro_concept_routes_v1
```

Representative validation against current objective ingress reports:

```text
token_symbol_ids      list<uint32>
token_call_targets    list<uint32>
token_type_refs       list<uint32>
token_source_doc_ids  present but all zero
token_source_identity_ids  missing
source_identity_registry   missing
ValueError: missing source_identity_registry, symbol_identities, token_ids,
            token_source_identity_ids
```

The current packed schema requires physical source identities, a registry,
`uint64` semantic IDs, typed objective source sections, and exact tokenizer,
domain, delimiter, symbol, and CASE5 metadata
(`[mlx] scripts/nanochat_data/pack_enriched_rows.py:270-360,1117-1165` and
`cppmega_mlx/training/objective_data.py:58-103`). Conversion cannot reconstruct
those semantics from old packed arrays. An extra post-freeze Paddle parquet was
sampled and is still the same legacy schema, proving that resume did not upgrade
the generation.

There was also a concrete loss-boundary defect in live source parquet. Before
repair, `code/1024` had 1,491,844,072 trained tokens where the canonical
`valid_tokens - source_documents` invariant requires 1,488,124,690. Exactly
3,719,382 cross-document targets leaked into training. No other kind/bucket
showed this mismatch. The bundle repair atomically rewrote 11,684 files and
2,617,421 rows, restoring 3,719,382 boundaries and reducing total trained tokens
from 4,127,683,222 to 4,123,963,840. The repair implementation derives one zero
loss target per logical source document and replaces hardlinks atomically
(`[mlx] scripts/repair_packed_document_boundaries.py:94-153,186-277`).

The repaired bundle is safe with respect to this invariant. The old live parquet
is not safe for direct conversion and must not be mixed with a new run. New schema
requires a new run ID, output roots, and dedup DB.

### F9 - MEDIUM/P2: receipts prove archive presence, not a completed current publish/restore workflow

`publish_receipt.json` is a loose-object **dry run**: 236 artifact records,
manifest, and latest pointer all have status `dry_run`. It is not upload evidence.

`archive_publish_receipt.json` records real `uploaded_verified` states, and live
S3 reads on 2026-07-14 confirmed:

| Remote object | Size/hash evidence |
|---|---|
| `latest_transport.json` | 695 bytes; `5e40289fbcb57a05c6c991bafe163acb9f6b5ab20585b5b70a4253bf75e2c6e6` |
| `transport.json` | 934 bytes; `1a2a14d3c56581d022eb3ae673ebeb1636f9e8c6575664193acd3e814edbdfba` |
| `logical_manifest.json` | 97,336 bytes; `1e44fcd8ff9192a15e25b62c18ac71569ca7627e3f2221b29d249d5c2fb93391` |
| `bundle.tar.zst` | 2,823,467,371 bytes; remote metadata `sha256=5ac3e095289b28d4924c3547ebae62ee79f002c4ad0c81467fda59b03a1c93a7` |

The transport binds 236 artifacts, 174,421,216,970 artifact bytes, and 237 archive
members. The remote archive key is the old non-content-addressed
`bundle.tar.zst`; transport SHA verification protects integrity on restore but
not availability against overwrite. Current publication code uses a hash-suffixed
key, conditional create/compare-and-swap, server checksum verification, and commits
the latest pointer last
(`[converter] publish_megatron_bundle_to_nebius_s3.py:1906-2006,2265-2388,2478-2562`).

The legacy receipts have no top-level `status`, `bundle_id`, artifact-set binding,
or receipt binding. Preserve them as historical evidence, but require the current
receipt schema and a successful fresh restore receipt for the next generation.

### F10 - MEDIUM/P2: run observability and disk state need a restart gate

The final commit restart ran 49,529.109 seconds but emitted
`commit_ranges_this_run=0` and `cumulative_valid_tokens_this_run=0`. That agrees
with the manifest: only ten commit-plan entries became done. The 39 visible shards
are uncommitted orphans and must not be reported as completed throughput.

`_reservations.json` still lists `intel-llvm-dpcpp::code` and `linux::code` under
PID 29653, which no longer exists. Current startup removes dead reservations
(`[mlx] scripts/streaming_conveyor.py:1208-1221,3049`), but the stale receipt is a
warning that the last process did not close cleanly.

Disk state at review time:

| Item | Size/state |
|---|---:|
| External volume | 2.7 TiB total, 2.2 TiB used, 575 GiB free, 80% |
| Frozen Megatron bundle | 162 GiB (`174,421,216,970` artifact bytes) |
| Frozen source parquet | 5.487 GiB; Megatron expansion is 29.6x |
| Conveyor run directory | 18 GiB, including retained temp trees of 11, 3.1, and 1.1 GiB |
| Dedup SQLite | 21 GiB; WAL 0 bytes at inspection |
| PR SQLite | 8.5 GiB |

The new objective sidecar contract widens `doc_ids` and three semantic IDs and
adds source-document and physical-source identity IDs
(`[converter] cppmega/megatron/objective_contract.py:30-51`). Relative to this
legacy layout that is about 26 additional bytes per token, or about 100.1 GiB for
4.133B tokens before registries/graph metadata. A new final bundle of roughly
262.5 GiB would leave about 312 GiB free if it were the only new copy. A workflow
that simultaneously retains objective parquet, snapshots, old bundle, new bundle,
archive, and failed temp can approach the 100 GiB conveyor floor. Require a
phase-by-phase projected-space receipt; do not infer safety from the tiny 2.63 GiB
compressed transport archive.

## What actually completed

### Run identity and timing

`run_env.sh` binds:

```bash
RUN_ID=macro_routes_v1_20260710_135335
CONV_ROOT=outputs/conveyor/macro_routes_v1_20260710_135335
CODE_ROOT=outputs/reindexed_macro_routes_v1_20260710_135335_code
COMMIT_ROOT=outputs/reindexed_macro_routes_v1_20260710_135335_commits
DEDUP_DB=outputs/dedup_seen_macro_routes_v1_20260710_135335.sqlite
```

Code and commit lanes first started at `2026-07-10T15:53:59Z`. The frozen source
snapshot was created at `2026-07-13T13:23:52.492713Z`, 69.498 hours later.
Gross overlapping-lane rate to freeze was about 16,518 valid tokens/s. Code
produced 2.482B valid tokens over about 71.38 wall hours (9,659 tokens/s gross);
the frozen commit lane produced 1.651B over 69.50 hours (6,598 tokens/s gross).
These are wall-clock ratios across interruptions, not steady-state benchmarks.

### Frozen parquet by stream and bucket

`trained` means nonzero loss-mask targets after repair.

| Stream | Bucket | Shards | Packed rows | Capacity | Valid tokens | Trained tokens | Padding |
|---|---:|---:|---:|---:|---:|---:|---:|
| code | 1024 | 293 | 1,509,110 | 1,545,328,640 | 1,495,779,442 | 1,488,124,690 | 3.206% |
| code | 2048 | 293 | 222,445 | 455,567,360 | 315,334,535 | 315,112,090 | 30.782% |
| code | 4096 | 289 | 100,355 | 411,054,080 | 282,151,520 | 282,051,165 | 31.359% |
| code | 8192 | 289 | 39,880 | 326,696,960 | 223,059,383 | 223,019,503 | 31.723% |
| code | 16384 | 277 | 14,863 | 243,515,392 | 165,555,100 | 165,540,237 | 32.015% |
| commits | 1024 | 2,073 | 294,972 | 302,051,328 | 216,519,785 | 216,218,523 | 28.317% |
| commits | 2048 | 2,074 | 218,695 | 447,887,360 | 314,170,230 | 313,951,535 | 29.855% |
| commits | 4096 | 2,074 | 129,649 | 531,042,304 | 371,189,937 | 371,060,288 | 30.102% |
| commits | 8192 | 2,069 | 70,160 | 574,750,720 | 398,656,066 | 398,585,906 | 30.638% |
| commits | 16384 | 2,040 | 31,334 | 513,376,256 | 350,331,237 | 350,299,903 | 31.759% |
| **code total** | | **1,441** | **1,886,653** | **2,982,162,432** | **2,481,879,980** | **2,473,847,685** | **16.776%** |
| **commit total** | | **10,330** | **744,810** | **2,369,107,968** | **1,650,867,255** | **1,650,116,155** | **30.317%** |
| **all** | | **11,771** | **2,631,463** | **5,351,270,400** | **4,132,747,235** | **4,123,963,840** | **22.771%** |

Code supplies 59.9871% of trained tokens; commit/PR supplies 40.0129%.

### Commit and PR composition

The attribution pass used per-constituent source lengths, not packed-row labels.
PR discussion is a subset of PR data.

| Bucket | Commit source docs | PR source docs | PR packed rows | PR trained tokens | Discussion rows | Discussion trained tokens |
|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 283,140 | 18,122 | 18,042 | 13,446,220 | 5,469 | 4,470,651 |
| 2048 | 200,238 | 18,457 | 18,457 | 27,064,973 | 12,606 | 18,887,818 |
| 4096 | 115,930 | 13,719 | 13,719 | 39,516,341 | 11,085 | 31,988,301 |
| 8192 | 62,382 | 7,778 | 7,778 | 44,285,585 | 6,198 | 35,319,735 |
| 16384 | 27,794 | 3,540 | 3,540 | 39,737,867 | 3,054 | 34,295,622 |
| **total** | **689,484** | **61,616** | **61,536** | **164,050,986** | **38,412** | **124,962,127** |

Non-PR commits carry 1,486,065,169 trained tokens. PRs are 9.9418% of commit
trained tokens and 3.9780% of all trained tokens. Discussion-bearing content is
47,144,052 characters / 974,029 lines and 7.5729% of commit trained tokens.
The PR store itself contains 1,292,734 PRs across 396 repositories, 893,571 with
a merge SHA, and 892,255 SHA index rows. The live lookup is read-only, resolves
PR number before SHA, and injects discussion before document construction
(`[mlx] tools/clang_indexer/process_commits.py:110-198,2644-2682`). Commit parse
errors fail the range unless explicitly allowed (`:2970-2985`).

### Delimiter, domain, graph, source, symbol, and objective coverage

A manifest-bound streaming pass decoded only `input_ids` and counted current
reserved IDs 191-244 over all 5,351,270,400 slots. It found exactly one balanced
start/end pair for each of the 8,783,395 constituent source documents:

| Delimiter family interpreted under current contract | Balanced pairs |
|---|---:|
| C/C++ | 8,731,003 |
| Make | 13,772 |
| CMake | 28,287 |
| Ninja | 7 |
| Bazel | 3,598 |
| Build diagnostic | 5,119 |
| Meson | 1,604 |
| compile_commands | 5 |
| All other 19 families | 0 |
| **Total** | **8,783,395** |

All 27 current start/end family counts balance exactly. This is strong raw-token
evidence, but the legacy parquet/bundle does not contain the current tokenizer
contract hash, so the semantic label remains an interpretation rather than a
cryptographic contract.

| Channel | Coverage/result |
|---|---|
| Domain IDs | 3,254,088,222 nonzero valid-token slots, 78.7391% of valid tokens; every row has a nonzero domain slot |
| Domain graph | 17,377,023 edges in 471,053 rows (17.9008%) |
| Symbols | 2,215,336,989 nonzero valid-token slots (53.6045%); 1,537,687 rows (58.4347%); legacy `uint32` qname IDs, no collision registry |
| Def/use | 2,261,726,979 nonzero slots (54.7270%); 1,652,523 rows |
| Call targets / type refs | 39,463,620 / 38,777,699 nonzero slots; 619,524 / 869,541 rows |
| Call/type/build graph | 644,450 / 452,370 / 572,229 edges; 222,783 / 199,503 / 19,761 rows |
| Shell/diagnostic/cross-domain graph | Exactly zero edges |
| Source documents | 8,783,395 constituent IDs/lengths; all length sums equal valid tokens |
| Token source provenance | `token_source_doc_ids` is zero in all 5,351,270,400 slots; physical `token_source_identity_ids` and registry absent |
| Token platform | `token_platform_ids` is zero in all slots; row/source-document platform lists are populated |
| Temporal | 1,045,679 changed chunks in 325,222 rows (12.3590%) |
| Objectives | No objective kind, materialization artifact, source binding, or quota receipt in the bundle |

Total graph edges are 19,046,072. Empty shell/diagnostic/cross-domain relations
are a declared limitation, not byte corruption, but cannot be represented as
active production graph objectives.

### Schema and row samples

Every sampled file has 73 columns and only the macro-routes footer metadata.
The table records row 0; `domain` and `symbol` are nonzero valid-token counts.

| Bucket | Stream/sample | Valid/trained | Docs or PR | Domain | Symbol | Selected graph/temporal evidence |
|---:|---|---:|---|---:|---:|---|
| 1024 | code `4.4bsd-lite2.parquet` | 1,024 / 1,022 | 2 source docs | 824 | 692 | 65 domain edges; 1 type edge |
| 1024 | commit `Windows-Driver-Frameworks_r0.parquet` | 676 / 675 | no PR | 576 | 427 | 1 changed chunk |
| 2048 | code `4.4bsd-lite2.parquet` | 1,427 / 1,426 | 1 source doc | 1,327 | 965 | 2 call edges; 36 domain edges |
| 2048 | commit `WindowsAppSDK_r0.parquet` | 1,901 / 1,900 | PR 6233, discussion | 1,801 | 282 | 2 changed chunks |
| 4096 | code `4.4bsd-lite2.parquet` | 2,359 / 2,358 | 1 source doc | 2,259 | 2,129 | 2 call edges; 20 domain edges |
| 4096 | commit `WindowsAppSDK_r0.parquet` | 3,684 / 3,683 | PR 6127 | 3,584 | 2,969 | 108 call targets; 156 type refs; 12 changed chunks |
| 8192 | code `4.4bsd-lite2.parquet` | 4,247 / 4,246 | 1 source doc | 4,147 | 4,123 | 7 call edges; 9 domain edges |
| 8192 | commit `Windows-Driver-Frameworks_r0.parquet` | 5,495 / 5,494 | no PR | 5,395 | 0 | no semantic symbol coverage in this row |
| 16384 | code `4.4bsd-lite2.parquet` | 9,266 / 9,265 | 1 source doc | 9,166 | 8,576 | 24 call and 22 type edges; 216 domain edges |
| 16384 | commit `WindowsAppSDK_r0.parquet` | 8,269 / 8,268 | PR 6068, discussion | 8,169 | 4,814 | 480 type refs; 20 changed chunks |

### Megatron conversion result

| Bucket | Documents | Capacity tokens | Valid tokens | Trained tokens | Padding | Graph edges |
|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 1,804,082 | 1,847,379,968 | 1,712,299,227 | 1,704,343,213 | 7.312% | 5,119,406 |
| 2048 | 441,140 | 903,454,720 | 629,504,765 | 629,063,625 | 30.322% | 2,890,737 |
| 4096 | 230,004 | 942,096,384 | 653,341,457 | 653,111,453 | 30.650% | 3,804,161 |
| 8192 | 110,040 | 901,447,680 | 621,715,449 | 621,605,409 | 31.031% | 3,893,194 |
| 16384 | 46,197 | 756,891,648 | 515,886,337 | 515,840,140 | 31.841% | 3,338,574 |
| **total** | **2,631,463** | **5,351,270,400** | **4,132,747,235** | **4,123,963,840** | **22.771%** | **19,046,072** |

Document/token totals match repaired parquet exactly. Ten deterministic local
spot checks rehashed every prefix JSON and main `.idx` across all five buckets;
all matched the artifact manifest. A full 162 GiB rehash was intentionally not
repeated. The original build manifest and archive descriptor carry the full set.

## SQLite state

Read-only bounded checks, not `quick_check` or full table scans, found:

- Dedup DB: 4,096-byte pages, 5,387,861 pages, no freelist pages,
  `next_doc_id=3,561,319`, min/max minhash doc IDs `1/3,561,319`.
- `dedup_stages` has zero rows; `exact_stage`, `minhash_stage`, `lsh_stage`, and
  `chunk_claims_stage` each return `EXISTS=0`.
- PR DB: 4,096-byte pages, 2,220,649 pages; 1,292,734 PR rows; 396 repositories;
  893,571 merge-SHA-bearing PRs; 892,255 SHA index rows. Max rowids are
  3,728,616 comments, 3,036,712 reviews, and 112,682 linked issues.

These checks support a clean staging boundary but are not an integrity scan of
the 21 GiB dedup database.

## Commands and verification

Producer entry points observed in logs/code:

```text
scripts/streaming_conveyor.py
tools/clang_indexer/index_project.py
scripts/nanochat_data/extract_git_history.py
tools/clang_indexer/process_commits.py
scripts/nanochat_data/clang_enriched_to_parquet.py
scripts/nanochat_data/pack_enriched_rows.py
scripts/repair_packed_document_boundaries.py
scripts/audit_sidecar_parquet.py
scripts/materialize_megatron_objectives.py
scripts/data_prep_parquet_to_megatron.py
scripts/data/build_macro_routes_megatron_bundle.py
scripts/data/publish_megatron_bundle_to_nebius_s3.py
scripts/data/restore_megatron_bundle_from_nebius_s3.py
```

Representative logged code pipeline command shape:

```bash
.venv/bin/python tools/clang_indexer/index_project.py \
  --project-dir <repo> --output <repo>.enriched.jsonl --enriched \
  --max-tokens 65536 --tokenizer-path cppmega_mlx/tokenizer/tokenizer.json \
  --dedup-db outputs/dedup_seen_macro_routes_v1_20260710_135335.sqlite \
  --dedup-stage-id code:<repo> --global-symbol-index outputs/crossrepo/global_symbols.sqlite
.venv/bin/python scripts/nanochat_data/clang_enriched_to_parquet.py \
  --input-file <repo>.enriched.jsonl --output-file <repo>.tok.parquet \
  --materialize-tokenized-enriched --overflow-policy drop --size 64k
.venv/bin/python scripts/nanochat_data/pack_enriched_rows.py \
  --input routed/route_<bucket>.parquet --output <repo>.packed.<bucket>.parquet \
  --target-length <bucket> --strategy best_fit
```

Review commands included:

```bash
shasum -a 256 \
  outputs/conveyor/macro_routes_v1_20260710_135335/_done.json \
  outputs/conveyor/macro_routes_v1_20260710_135335/progress.jsonl \
  outputs/conveyor/macro_routes_v1_20260710_135335/_reservations.json

sqlite3 -readonly outputs/dedup_seen_macro_routes_v1_20260710_135335.sqlite \
  'PRAGMA journal_mode; PRAGMA page_size; PRAGMA page_count; PRAGMA freelist_count; ...'
sqlite3 -readonly outputs/pr_ingest/prs.sqlite \
  'SELECT COUNT(*),COUNT(DISTINCT repo),SUM(merge_commit_sha IS NOT NULL) FROM prs; ...'

df -h /Volumes/external
du -sh outputs/megatron_ready/macro_routes_v1_20260713 \
  outputs/conveyor/macro_routes_v1_20260710_135335 \
  outputs/dedup_seen_macro_routes_v1_20260710_135335.sqlite \
  outputs/pr_ingest/prs.sqlite

sysctl -n hw.memsize
# 137438953472 (128 GiB)
python3 - <<'PY'
rows = 2_631_463
valid = 4_132_747_235
average = valid / rows
print(1_000_000 * average * (21 * 8) / 2**30)  # list-pointer floor
print(1_000_000 * average * 70 / 2**30)         # fixed Arrow payload
PY
# 245.72593024591782
# 102.38580426913242
```

The independent parquet review used PyArrow 21.0.0 from
`/Volumes/external/sources/nanochat/.venv/bin/python`. One 92.6-second pass read
only scalar/list sidecar columns from the exact 11,771 frozen manifest records;
it verified row lengths, source-length sums, metadata, PR attribution, and
coverage. A second 439.137-second bounded pass used
`ParquetFile.iter_batches(columns=['input_ids'], batch_size=512)` plus
`numpy.bincount` to count delimiter IDs over 5,351,270,400 slots. No table was
held corpus-wide in memory.

Live S3 verification read only the two small descriptors and 97 KB logical
manifest, then issued `aws s3api head-object` for the archive against
`https://storage.eu-north1.nebius.cloud`; it did not download the 2.8 GB archive
or expand the bundle.

Focused tests:

```bash
cd /Volumes/external/sources/cppmega_full_integration
pytest -q \
  tests/test_build_macro_routes_megatron_bundle.py \
  tests/test_megatron_objective_contract.py \
  tests/test_publish_megatron_bundle_to_nebius_s3.py \
  tests/test_restore_megatron_bundle_from_nebius_s3.py
# 131 passed in 10.56s

cd /Volumes/external/sources/cppmega_mlx_full_integration
/Volumes/external/sources/nanochat/.venv/bin/python -m pytest -q \
  tests/test_materialize_megatron_contract_hashes.py
# 5 passed in 0.42s

/Volumes/external/sources/nanochat/.venv/bin/python -m pytest -q \
  tests/test_megatron_indexed.py::test_mmididx_int32_reads_fixed_windows_without_crossing_sequences \
  tests/test_megatron_indexed.py::test_open_megatron_indexed_dataset_is_standalone_training_ingress
# 2 passed in 0.14s; confirms production-reachable bare ingress accepts no sidecar

/Volumes/external/sources/nanochat/.venv/bin/python -m pytest -q \
  tests/test_domain_schema.py tests/test_tokenizer_contract.py \
  tests/test_pack_enriched_rows.py \
  tests/test_materialize_megatron_contract_hashes.py \
  tests/test_materialize_megatron_dependency_provenance.py \
  tests/test_streaming_conveyor_progress.py \
  tests/test_extract_git_history_precompute.py \
  tests/test_process_commits_fail_loud.py
# 139 passed in 43.96s

/Volumes/external/sources/nanochat/.venv/bin/python -m pytest -q \
  tests/test_clang_indexer_regressions.py
# 15 passed in 2.48s
```

The objective/FIM work was committed by its owner as `ddaafc3` before the final
139-test rerun. The later indexer work was committed as `f9f1686` and its 15-test
regression suite passed separately. This review did not alter either change. The
focused source-binding test also passed all five cases independently.

## Restart, freeze, convert, upload acceptance checklist

### Restart/new generation

- [ ] Do not resume legacy output roots as the production schema generation.
- [ ] Allocate a new run ID, code root, commit root, conveyor root, and dedup DB.
- [ ] Pin and record exact clean git commits plus any approved patch digest for
  both checkouts; do not rely on a dirty worktree snapshot description.
- [ ] Preflight projected disk for source, objective parquet, final bundle,
  archive, and retained old data; maintain at least the configured 100 GiB floor
  through every simultaneous-copy phase.
- [ ] Require new footer metadata and schema on every output shard: frozen
  tokenizer/domain/delimiter hashes, symbol identity version and registry,
  `uint64` semantic/source identities, typed objective source sections.
- [ ] Fail if any legacy 73-column shard or unmanifested filename is present.
- [ ] Confirm reservation ledger empty after stale-PID cleanup and dedup staging
  tables empty before work starts.

### Freeze

- [ ] Stop lanes cleanly and require a stable `_done.json` hash.
- [ ] Require zero failed units, or a separately reviewed hash-bound exclusion
  manifest. A numeric failed count in provenance is not acceptance.
- [ ] Freeze only manifest-listed outputs; enumerate and reject/quarantine orphans.
- [ ] Use a two-phase immutable freeze receipt containing original and repaired
  manifest hashes, per-bucket shard-set hashes/counts, code commits, config hash,
  and audit receipt hash.
- [ ] Rehash source before/after repair; assert every row has
  `trained_token_count = valid_token_count - num_docs` and source lengths sum to
  valid tokens.
- [ ] Require audit `bad_files=0`, `bad_rows=0`, no missing required columns, and
  exact expected metadata on every shard.

### Objective materialization and conversion

- [ ] Replace all-document retention, whole-shard `to_pylist`, 1,024-row padded
  batches, and cartesian graph-pair sets with the bounded streaming design in F2.
- [ ] Record and enforce a memory budget, bounded source/write batch sizes, peak
  RSS, deterministic cursor/checkpoints, and byte-identical resume verification.
- [ ] Require `cppmega_objective_source_snapshot_v1` in every objective contract
  and exact per-bucket `(size, sha256)` equality with the repaired snapshot before
  conversion; retain the `8a1b594` / `ef44b42` regression cases.
- [ ] Prefer `--source-manifest` plus bucket over a production glob. Until that
  interface exists, prove the resolved glob set exactly matches the immutable
  repaired manifest and reject every orphan.
- [ ] Reverify every input shard after consumption and retain the source snapshot
  transitively inside objective contract/artifact hashes.
- [ ] Add the exact repaired-manifest hash, kind/path binding, row-count sum, and
  recomputed sampling equations described in F1 as provenance hardening.
- [ ] Prove exact per-window objective quotas and per-objective realized
  sample/loss-token, typed domain/source/symbol/temporal, and graph relation
  coverage; reject the silent non-causal collapses in F4.
- [ ] Stage and hash tokenizer and frozen data contracts in the bundle.
- [ ] Cross-check source binding in objective artifact, converted prefix JSON,
  bundle descriptor, and final artifact set.
- [ ] Retain the `60f9891` source-summary validation and pass a real
  builder -> publisher -> restore composition run as required by F5.
- [ ] Verify MMIDIDX document/token counts, every sidecar byte length/dtype,
  source registries, graph CSR endpoints/document masks, and prefix hashes.

### Publish and restore

- [ ] Current publisher validation passes locally before archive creation.
- [ ] Archive member set is exact and unique; archive and logical manifest are
  content-addressed and `uploaded_verified` (not `dry_run`).
- [ ] Immutable data/transport objects upload first; latest pointer updates last
  with compare-and-swap.
- [ ] Receipt has top-level complete status, bundle/artifact/source bindings,
  transport hash, logical manifest hash, archive hash/size, and current code/config
  binding.
- [ ] Current restore preflights logical manifest compatibility before download.
- [ ] Restore by explicit bundle ID into a fresh directory, verify archive SHA,
  every member SHA, artifact set, source/objective binding, and all five prefix
  manifests; then write `restored_verified` receipt.
- [ ] Production MLX training accepts only bundle root + bucket + expected bundle
  ID, revalidates the immutable manifest/restore receipt, and resolves an exact
  manifest-listed prefix; bare-prefix ingress is rejected in production mode.
- [ ] Smoke-open every `.idx`, read deterministic first/middle/last documents and
  sidecars for every bucket, and run the production dataset ingress validator.
- [ ] Only after the restore receipt and smoke results are retained may a latest
  pointer be treated as training-ready.

## Final disposition

The frozen 20260713 bytes are useful as a legacy, repaired, manifest-complete
subset and the remote archive objects are present. They are not the new required
production schema, do not prove objective provenance, and are not consumable by
the current supported restore validator. Preserve them read-only for comparison.
The A/B source/objective mismatch is fixed in code by `8a1b594` / `ef44b42`, but
no reviewed produced artifact contains that binding yet. Descriptor validation,
failed-freeze rejection, and remote preflight are now fixed in current converter
code, but still lack a real current-bundle receipt. The all-in-RAM materializer is
independently incapable of a 1M+ document production run on this host,
non-causal objectives still lose required graph/semantic coverage, and Stage-1
can bypass bundle provenance through bare-prefix ingress. Land the bounded
streaming/graph accounting and objective-remap fixes plus the production bundle
opener; pin clean passing commits; restart into new roots; use a two-phase
immutable freeze; reach zero failures (or explicit exclusions); then materialize,
convert, publish, restore, and open that exact restored bundle before training
acceptance.
