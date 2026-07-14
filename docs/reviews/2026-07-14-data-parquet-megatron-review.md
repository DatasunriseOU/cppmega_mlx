# Data / Parquet / Megatron Review

**Review date:** 2026-07-14

**Decision:** **NO-GO for Stage-1 training from CASE5 v4 or mini9**

**Reviewed MLX HEAD:** `b20db6ab90c642292ac09c0ecf2c77b5565ccd6c`

**Reviewed cppmega converter/publisher HEAD:** `9c4dd4b2ff257184dd9faffbc6add1a4db698608`

**Live conveyor code revision:** `63b0f587ce9504672bd58c964c6c4176fc10956d`

The immutable mini9 publication and cold restore are valid byte-integrity evidence,
but they do not make the data trainable. The materialized loss mask is shifted one
token relative to the runtime target mask, which trains 101 cross-document labels in
the 60-sequence mini9 sample. Its graph objective contract also covers only
`call,type` at `topk=64` while current Stage-1 requires all seven routes
at `topk=256`. Current HEAD additionally has a new tokenizer contract hash and
correctly rejects mini9 before mmap. Preserve mini9 as immutable negative evidence;
do not mutate, relabel, or train from it.

## Snapshot and review boundary

| Item | Frozen review value | Evidence |
|---|---|---|
| MLX source snapshot | `b20db6ab90c642292ac09c0ecf2c77b5565ccd6c`, clean worktree at freeze | [commit](https://github.com/DatasunriseOU/cppmega_mlx/commit/b20db6ab90c642292ac09c0ecf2c77b5565ccd6c) |
| Converter/publisher snapshot | `9c4dd4b2ff257184dd9faffbc6add1a4db698608` | [commit](https://github.com/DatasunriseOU/cppmega/commit/9c4dd4b2ff257184dd9faffbc6add1a4db698608) |
| Full CASE5 v4 conveyor | active; `251 done / 33 failed` at `2026-07-14T13:09:38+02:00` | [revision receipt](</Volumes/external/sources/cppmega.mlx/outputs/conveyor_case5_v4_20260714_093120/_done.json:2>), [code launch](</Volumes/external/sources/cppmega.mlx/outputs/conveyor_case5_v4_20260714_093120/launch_code.sh:4>), [commit launch](</Volumes/external/sources/cppmega.mlx/outputs/conveyor_case5_v4_20260714_093120/launch_commits_full.sh:4>) |
| Full conveyor runtime | clean checkout pinned to `63b0f587...`, not current HEAD | [launch receipt](</Volumes/external/sources/cppmega.mlx/outputs/conveyor_case5_v4_20260714_093120/code_revision_guard/63b0f587ce9504672bd58c964c6c4176fc10956d-be678ff3aeec94fd/launch_receipt.json:11>) |
| mini9 bundle | `case5_v4_20260714_093120_mini9-73a7931e9374bd29` | [manifest](</Volumes/external/sources/cppmega.mlx/outputs/megatron_ready/case5_v4_20260714_093120_mini9/manifest.json:2369>) |
| mini9 artifact set | `73a7931e9374bd29def594cd4cc4d031fbdd86cb2fe867318c3e56150710d5ca`, 65 files, 9,123,739 bytes | [manifest](</Volumes/external/sources/cppmega.mlx/outputs/megatron_ready/case5_v4_20260714_093120_mini9/manifest.json:1>) |
| mini9 source | mini3 snapshot, one code file plus 17 commit files, 18 done and 0 failed | [build plan](</Volumes/external/sources/cppmega.mlx/outputs/megatron_ready/case5_v4_20260714_093120_mini9/build_plan.json:8>) |
| mini9 producer revisions | cppmega `eb90059`, cppmega_mlx `8a8de80`; neither is current review HEAD | [manifest](</Volumes/external/sources/cppmega.mlx/outputs/megatron_ready/case5_v4_20260714_093120_mini9/manifest.json:2383>) |

The live conveyor is mutable. Counts and capacity below are timestamped planning
snapshots, not a completion receipt. The mini9 manifest, publish receipts, archive
receipt, and cold-restore receipt are immutable review inputs.

## Findings

### P0-1: loss-mask storage is one token out of phase with runtime targets

The materializer reconstructs the shifted LM document as
`[input_ids[0], *target_ids]`, but stores `[*loss_mask, 0]`. The runtime
reader derives targets as `tokens[:, 1:]` and derives a token-shaped target
mask as `loss_mask[:, 1:]`. Therefore target `i` receives source mask
`i + 1` instead of source mask `i`. With the current reader contract,
the token-shaped storage must be `[0, *loss_mask]`, or the reader and every
consumer must be versioned to an explicitly different alignment.

The cppmega converter independently cements the same wrong convention: it requires a
trailing zero, counts `mask[:-1]` as contract loss, then writes the entire
token-shaped mask. Existing tests assert that trailing-zero convention and never
compare it through `LMTokenBatch.target_mask`.

Direct mini9 MMIDIDX-reader audit:

| Measurement | Result |
|---|---:|
| Sequences | 60 |
| Contract/stored-alignment trained labels | 36,707 |
| Runtime `LMTokenBatch.target_mask` labels | 36,677 |
| Real inter-document target transitions | 101 |
| Inter-document labels trained under intended storage alignment | 0 |
| Inter-document labels trained by current runtime alignment | **101** |
| Intended same-document labels suppressed by the shift | 161 |
| Zero-to-one gains caused by the shift | 131, including 101 cross-document |

Sequence 16, target position 152 is a concrete witness: documents `1 -> 2`
have stored mask pair `[0, 1]`, so the runtime takes `1` and trains
across the boundary. The net token-count difference of only 30 hides the much larger
set substitution.

Attention and graph isolation are separate and are implemented: Stage-1 builds a
causal same-document pair mask, DenseCppLM uses a document-boundary attention mask,
and graph bias is zeroed across documents. The defect is label isolation.

**Sources:** [materializer construction](https://github.com/DatasunriseOU/cppmega_mlx/blob/b20db6ab90c642292ac09c0ecf2c77b5565ccd6c/cppmega_mlx/training/megatron_objectives.py#L803-L846), [runtime target alignment](https://github.com/DatasunriseOU/cppmega_mlx/blob/b20db6ab90c642292ac09c0ecf2c77b5565ccd6c/cppmega_mlx/data/batch.py#L216-L228), [shape alignment](https://github.com/DatasunriseOU/cppmega_mlx/blob/b20db6ab90c642292ac09c0ecf2c77b5565ccd6c/cppmega_mlx/data/batch.py#L348-L360), [Stage-1 consumption and pair mask](https://github.com/DatasunriseOU/cppmega_mlx/blob/b20db6ab90c642292ac09c0ecf2c77b5565ccd6c/cppmega_mlx/training/stage1_production.py#L183-L246), [DenseCppLM isolation](https://github.com/DatasunriseOU/cppmega_mlx/blob/b20db6ab90c642292ac09c0ecf2c77b5565ccd6c/cppmega_mlx/models/dense_cpp_lm.py#L655-L689), [graph-bias isolation](https://github.com/DatasunriseOU/cppmega_mlx/blob/b20db6ab90c642292ac09c0ecf2c77b5565ccd6c/cppmega_mlx/models/dense_cpp_lm.py#L767-L776), [converter convention](https://github.com/DatasunriseOU/cppmega/blob/9c4dd4b2ff257184dd9faffbc6add1a4db698608/scripts/data_prep_parquet_to_megatron.py#L1796-L1845), [raw sidecar write](https://github.com/DatasunriseOU/cppmega/blob/9c4dd4b2ff257184dd9faffbc6add1a4db698608/scripts/data_prep_parquet_to_megatron.py#L2128-L2138), [self-consistent test](https://github.com/DatasunriseOU/cppmega_mlx/blob/b20db6ab90c642292ac09c0ecf2c77b5565ccd6c/tests/test_production_objective_mixer.py#L1355-L1375), [mini9 totals](</Volumes/external/sources/cppmega.mlx/outputs/megatron_ready/case5_v4_20260714_093120_mini9/manifest.json:2350>).

**Required v5 action:** version the mask-alignment contract, fix producer/consumer
semantics, add an end-to-end multi-document regression through the actual MMIDIDX
reader and `LMTokenBatch.target_mask` for every objective, require runtime
loss sum to equal the objective contract, require zero trained cross-document labels,
and regenerate every objective parquet, prefix, manifest, and bundle ID.

### P0-2: mini9 graph objective contract cannot satisfy current Stage-1

mini9 records only `call,type` with `topk=64`. Current Stage-1
hard-codes `call,type,domain,build,shell,diagnostic,cross_domain` with
`topk=256` and rejects any different relation tuple or model top-k. The
materializer CLI defaults to a third contract, `call,type` at
`topk=8`.

Production bundle validation checks the objective schema, pair mask, and chunk-edge
expansion, but does not compare exact relations, top-k, or graph loss weights against
Stage-1. The immutable prefix physically has all route columns, but the mini9 sample
contains one call edge, zero type edges, 378 domain edges, and zero build, shell,
diagnostic, and cross-domain edges. This is not representative coverage for the
seven-relation training recipe.

**Sources:** [mini9 objective graph contract](</Volumes/external/sources/cppmega.mlx/outputs/megatron_ready/case5_v4_20260714_093120_mini9/provenance/objective_contract_seq1024.json:15>), [mini9 top-k](</Volumes/external/sources/cppmega.mlx/outputs/megatron_ready/case5_v4_20260714_093120_mini9/provenance/objective_contract_seq1024.json:684>), [Stage-1 relation recipe](https://github.com/DatasunriseOU/cppmega_mlx/blob/b20db6ab90c642292ac09c0ecf2c77b5565ccd6c/cppmega_mlx/training/stage1_production.py#L40-L64), [Stage-1 relation and top-k gates](https://github.com/DatasunriseOU/cppmega_mlx/blob/b20db6ab90c642292ac09c0ecf2c77b5565ccd6c/cppmega_mlx/training/stage1_production.py#L81-L92), [model top-k gate](https://github.com/DatasunriseOU/cppmega_mlx/blob/b20db6ab90c642292ac09c0ecf2c77b5565ccd6c/cppmega_mlx/training/stage1_production.py#L128-L142), [materializer defaults](https://github.com/DatasunriseOU/cppmega_mlx/blob/b20db6ab90c642292ac09c0ecf2c77b5565ccd6c/scripts/materialize_megatron_objectives.py#L1022-L1033), [incomplete ingress graph check](https://github.com/DatasunriseOU/cppmega_mlx/blob/b20db6ab90c642292ac09c0ecf2c77b5565ccd6c/cppmega_mlx/data/production_bundle.py#L808-L840), [serialized route counts](</Volumes/external/sources/cppmega.mlx/outputs/megatron_ready/case5_v4_20260714_093120_mini9/data/seq_1024/cppmega_macro_routes_seq1024_train.json:94>).

**Required v5 action:** define one versioned graph recipe shared by materializer,
bundle validator, model config, and Stage-1; validate exact relations, top-k, and all
weights at ingress; produce nonzero representative receipts per required relation or
an explicit approved zero-coverage exception; regenerate the bundle.

### P1-3: current HEAD correctly rejects the immutable mini9 tokenizer contract

mini9 binds tokenizer contract SHA-256
`c3bb669015c48e2049e3b82ccb8c98c6eceae0644f7da0b5b8600c573d7087a5`.
Current HEAD `b20db6a` binds
`80e73699e26d2c19fe4477cf8194886e52c7a5e114023df27e55d6a69b62c198`
and a delimiter-contract hash of
`1f2e35d7917409fc03704d32c2d55d0fb3e29f1bd9e60acca775a392cf2f53e6`.
The production opener fails closed with:

> `ValueError: bundle tokenizer contract does not match the local frozen hash`

This is correct behavior, not an ingress bug. It makes mini9 unusable under current
HEAD even before the semantic P0 blockers are considered. A v5 bundle must be
materialized and published under the current tokenizer/delimiter contract. Replacing
metadata on old token bytes is not acceptable.

**Sources:** [mini9 tokenizer hash](</Volumes/external/sources/cppmega.mlx/outputs/megatron_ready/case5_v4_20260714_093120_mini9/manifest.json:2377>), [current contract digest construction](https://github.com/DatasunriseOU/cppmega_mlx/blob/b20db6ab90c642292ac09c0ecf2c77b5565ccd6c/cppmega_mlx/data/tokenizer_contract.py#L451-L470), [production tokenizer gate](https://github.com/DatasunriseOU/cppmega_mlx/blob/b20db6ab90c642292ac09c0ecf2c77b5565ccd6c/cppmega_mlx/data/production_bundle.py#L615-L640), [CASE5 contract gate](https://github.com/DatasunriseOU/cppmega_mlx/blob/b20db6ab90c642292ac09c0ecf2c77b5565ccd6c/cppmega_mlx/data/production_bundle.py#L990-L1010), [mini9 build revisions](</Volumes/external/sources/cppmega.mlx/outputs/megatron_ready/case5_v4_20260714_093120_mini9/manifest.json:2383>).

### P1-4: objective bytes are immutable, but their producer implementation is not bound

The objective artifact binds schema, document count, objective contract, parquet
shards, and converter settings. It does not bind the materializer script hash, its
git commit and dirty state, command/config digest, or the checkout that executed it.
The cppmega build plan binds builder, converter, audit, and repair scripts, but not
the objective materializer. Its manifest derives `git.cppmega_mlx` from a
hard-coded sibling path `REPO_ROOT.parent / "cppmega.mlx"` rather than from
an explicit objective-producer root.

mini9 therefore proves exactly which objective bytes were bundled, but not which
materializer implementation produced them. The recorded `8a8de80` is not a
sufficient executable provenance witness.

**Sources:** [current objective artifact fields](https://github.com/DatasunriseOU/cppmega_mlx/blob/b20db6ab90c642292ac09c0ecf2c77b5565ccd6c/cppmega_mlx/training/megatron_objectives.py#L673-L800), [mini9 artifact payload](</Volumes/external/sources/cppmega.mlx/outputs/megatron_ready/case5_v4_20260714_093120_mini9/provenance/objective_artifact_seq1024.json:1>), [builder implementation record](https://github.com/DatasunriseOU/cppmega/blob/9c4dd4b2ff257184dd9faffbc6add1a4db698608/scripts/data/build_macro_routes_megatron_bundle.py#L1069-L1127), [hard-coded MLX git root](https://github.com/DatasunriseOU/cppmega/blob/9c4dd4b2ff257184dd9faffbc6add1a4db698608/scripts/data/build_macro_routes_megatron_bundle.py#L1437-L1448), [mini9 implementation list](</Volumes/external/sources/cppmega.mlx/outputs/megatron_ready/case5_v4_20260714_093120_mini9/build_plan.json:30>).

**Required v5 action:** bind materializer path and SHA-256, explicit producer repo
root, commit, dirty-state digest, command/config hash, Python/environment identity,
and output contract hash in the objective artifact and build plan.

### P1-5: all 33 live failures remain failed under the pinned v4 runtime

The active run is pinned to `63b0f587`. Current HEAD contains changes
intended to address every observed failure class, but none has been replayed under a
clean current-HEAD runtime. Code presence and unit tests are not live proof.

| Failure class | Count | Current failed units | Current-HEAD implementation |
|---|---:|---|---|
| Binary asset misclassified as domain text | 15 | ITK, SDL, VTK, apollo, apple-security, arrow, assimp, atlmfc, bazel, bgfx, bullet3, carla, cccl, ceph, clamav | exact build-name classification and text-like signature filtering |
| Oversized configure/build input | 3 | arangodb, aseprite, audacity | bounded typed chunking |
| Missing canonical source identity | 8 | Xbox Live Source, five AOSP repos, binutils-gdb, blender | forge-agnostic remote identity; repo list must be regenerated |
| Aggregate typed edge routed through wrong family | 3 | ProcMon-for-Linux, apple-dyld, bitcoin | canonical family routing |
| Character point anchor not mapped to a token | 2 | SPTAG, apple-mlx | deterministic right-token/final-left-token point mapping |
| Source/masked-line split mismatch | 1 | 4.4bsd-lite2 | shared physical line boundaries |
| Parse process killed | 1 | DirectXShaderCompiler, exit 137 | bounded in-flight parse future window |

**Sources:** [4.4BSD failure](</Volumes/external/sources/cppmega.mlx/outputs/conveyor_case5_v4_20260714_093120/code.log:77>), [typed-edge failure](</Volumes/external/sources/cppmega.mlx/outputs/conveyor_case5_v4_20260714_093120/code.log:171>), [point-anchor failure](</Volumes/external/sources/cppmega.mlx/outputs/conveyor_case5_v4_20260714_093120/code.log:309>), [exit 137](</Volumes/external/sources/cppmega.mlx/outputs/conveyor_case5_v4_20260714_093120/code.log:361>), [identity failure](</Volumes/external/sources/cppmega.mlx/outputs/conveyor_case5_v4_20260714_093120/code.log:485>), [binary failure](</Volumes/external/sources/cppmega.mlx/outputs/conveyor_case5_v4_20260714_093120/code.log:491>), [oversize failure](</Volumes/external/sources/cppmega.mlx/outputs/conveyor_case5_v4_20260714_093120/code.log:751>), [current exact build classification](https://github.com/DatasunriseOU/cppmega_mlx/blob/b20db6ab90c642292ac09c0ecf2c77b5565ccd6c/tools/clang_indexer/index_project.py#L706-L718), [bounded domain chunking](https://github.com/DatasunriseOU/cppmega_mlx/blob/b20db6ab90c642292ac09c0ecf2c77b5565ccd6c/cppmega_mlx/data/domain_ingestion.py#L441-L569), [remote identity](https://github.com/DatasunriseOU/cppmega_mlx/blob/b20db6ab90c642292ac09c0ecf2c77b5565ccd6c/cppmega_mlx/data/symbol_identity.py#L253-L301), [repo-list identity materialization](https://github.com/DatasunriseOU/cppmega_mlx/blob/b20db6ab90c642292ac09c0ecf2c77b5565ccd6c/scripts/pr_ingest/build_repo_list.py#L125-L150), [typed-edge routing](https://github.com/DatasunriseOU/cppmega_mlx/blob/b20db6ab90c642292ac09c0ecf2c77b5565ccd6c/cppmega_mlx/data/domain_schema.py#L308-L377), [point anchors](https://github.com/DatasunriseOU/cppmega_mlx/blob/b20db6ab90c642292ac09c0ecf2c77b5565ccd6c/cppmega_mlx/data/nanochat_pipeline/tokenized_enriched.py#L430-L510), [line split](https://github.com/DatasunriseOU/cppmega_mlx/blob/b20db6ab90c642292ac09c0ecf2c77b5565ccd6c/tools/clang_indexer/index_project.py#L3919-L3940), [bounded parse futures](https://github.com/DatasunriseOU/cppmega_mlx/blob/b20db6ab90c642292ac09c0ecf2c77b5565ccd6c/tools/clang_indexer/index_project.py#L7394-L7426).

**Latest binary-failure sources:** [ceph false positive](</Volumes/external/sources/cppmega.mlx/outputs/conveyor_case5_v4_20260714_093120/code.log:1902>), [clamav false positive](</Volumes/external/sources/cppmega.mlx/outputs/conveyor_case5_v4_20260714_093120/code.log:2015>).

**Required v5 action:** generate a fresh repo list, launch a clean current-HEAD
runtime with a revision receipt, replay all 33 named units, require zero failed units,
and retain per-unit output/metadata receipts before freezing the source snapshot.

### P1-6: current capacity is only 22.48% of a 5,000-step unique-token target

At the `2026-07-14T13:07:26+02:00` live planning snapshot, all five
bucket schedules use 196,608 tokens per optimizer step. The current code-plus-commit
parquet roots contain 221,012,987 trained tokens and 1,123 bucket-respecting full
steps. A 5,000-step run needs 983,040,000 trained tokens, leaving 762,027,013
tokens. The current unique-token corpus must grow by 4.448x, or the training plan
must explicitly approve and receipt epoch reuse. Reuse must not be silently
presented as unique coverage.

| Bucket | Files | Rows | Docs | Valid tokens | Trained tokens | Full steps |
|---:|---:|---:|---:|---:|---:|---:|
| 1,024 | 234 | 144,470 | 813,637 | 140,810,563 | 139,996,926 | 712 |
| 2,048 | 241 | 15,473 | 15,473 | 22,089,411 | 22,073,938 | 112 |
| 4,096 | 241 | 8,122 | 8,122 | 23,014,289 | 23,006,167 | 117 |
| 8,192 | 231 | 3,646 | 3,646 | 20,483,785 | 20,480,139 | 104 |
| 16,384 | 183 | 1,385 | 1,385 | 15,457,202 | 15,455,817 | 78 |
| **Total** | **1,130 bucket-file instances** | **173,096** | **842,263** | **221,855,250** | **221,012,987** | **1,123** |

The fixed-width storage floor is also above the current launch disk gate. The 20
token sidecars total 66 bytes per token. Final MMIDIDX uses another 2 bytes per token
and objective Arrow input uses 4 bytes:

| Retained payload at 983.04M tokens | Fixed-width floor |
|---|---:|
| Final prefix only: 68 B/token | 62.256 GiB |
| Objective parquet only: 70 B/token | 64.087 GiB |
| Objective plus one final prefix: 138 B/token | 126.343 GiB |
| Objective plus local and cold-restored final copies: 206 B/token | **188.599 GiB** |

These floors exclude graph ragged arrays, offsets, SQLite registries, JSON,
tokenizer, source parquet, temporary files, archive, filesystem overhead, and safety
headroom. The full conveyor launch gate is only 100 GiB free, so it is not a valid
v5 materialization gate.

**Sources:** [capacity reporter semantics](https://github.com/DatasunriseOU/cppmega_mlx/blob/b20db6ab90c642292ac09c0ecf2c77b5565ccd6c/scripts/report_training_steps.py#L1-L29), [step aggregation](https://github.com/DatasunriseOU/cppmega_mlx/blob/b20db6ab90c642292ac09c0ecf2c77b5565ccd6c/scripts/report_training_steps.py#L211-L253), [20 sidecar dtypes](</Volumes/external/sources/cppmega.mlx/outputs/megatron_ready/case5_v4_20260714_093120_mini9/provenance/objective_artifact_seq1024.json:68>), [token dtype and mini9 counts](</Volumes/external/sources/cppmega.mlx/outputs/megatron_ready/case5_v4_20260714_093120_mini9/manifest.json:2350>), [100 GiB launch gate](</Volumes/external/sources/cppmega.mlx/outputs/conveyor_case5_v4_20260714_093120/launch_code.sh:39>).

**Storage-type sources:** [objective `input_ids` int32 schema](https://github.com/DatasunriseOU/cppmega_mlx/blob/b20db6ab90c642292ac09c0ecf2c77b5565ccd6c/scripts/materialize_megatron_objectives.py#L315-L340), [final MMIDIDX uint16 prefix](</Volumes/external/sources/cppmega.mlx/outputs/megatron_ready/case5_v4_20260714_093120_mini9/data/seq_1024/cppmega_macro_routes_seq1024_train.json:1>).

### P2-7: source identity is valid in mini9, but production ingress does not prove it

The producer requires positive source document and identity IDs, preserves/remaps
them for transformed objectives, and requires a single physical identity where
appropriate. The converter derives uint64 IDs from SHA-256, detects collisions,
enforces token foreign keys, and writes a foreign-keyed SQLite registry.

Independent mini9 audit passed:

- `PRAGMA integrity_check` returned `ok` and
  `foreign_key_check` returned no rows.
- Registry: 28 identities, 95 references, 60 sequences, sequence indices 0 through 59.
- Token sidecar: 46,371 IDs, zero reserved-zero values, 28 used IDs, no missing or
  unreferenced registry IDs.

Production ingress only checks descriptor fields and that the registry path is
manifest-bound. It does not open SQLite or cross-check the token sidecar. The test
fixture writes arbitrary non-SQLite bytes as the registry and is accepted. mini9 is
good by a manual audit; the generic production contract is incomplete.

**Sources:** [objective source requirements](https://github.com/DatasunriseOU/cppmega_mlx/blob/b20db6ab90c642292ac09c0ecf2c77b5565ccd6c/cppmega_mlx/training/megatron_objectives.py#L900-L930), [transformed identity preservation](https://github.com/DatasunriseOU/cppmega_mlx/blob/b20db6ab90c642292ac09c0ecf2c77b5565ccd6c/cppmega_mlx/training/megatron_objectives.py#L1028-L1082), [converter registry validation](https://github.com/DatasunriseOU/cppmega/blob/9c4dd4b2ff257184dd9faffbc6add1a4db698608/scripts/data_prep_parquet_to_megatron.py#L1155-L1315), [mini9 registry receipt](</Volumes/external/sources/cppmega.mlx/outputs/megatron_ready/case5_v4_20260714_093120_mini9/manifest.json:2326>), [incomplete ingress check](https://github.com/DatasunriseOU/cppmega_mlx/blob/b20db6ab90c642292ac09c0ecf2c77b5565ccd6c/cppmega_mlx/data/production_bundle.py#L1059-L1076), [non-SQLite test fixture](https://github.com/DatasunriseOU/cppmega_mlx/blob/b20db6ab90c642292ac09c0ecf2c77b5565ccd6c/tests/test_production_megatron_bundle.py#L241-L265).

**Required v5 action:** production ingress must run SQLite integrity and foreign-key
checks, validate sequence coverage/counts, and compare the complete nonzero token-ID
set with registry keys before mmap.

### P2-8: S3 application-level immutability and cold restore pass; bucket-level WORM is unproven

The direct publisher uses conditional writes and exact post-upload HEAD checks. The
archive path is content-addressed, binds logical manifest and artifact set, and is
restored only after safe-member, size, and SHA-256 validation. mini9 receipts show all
65 direct artifacts uploaded and verified, a 66-member archive uploaded and verified,
and a cold restore with matching bundle, artifact, logical-manifest, transport,
command, config, checkpoint, and prefix hashes.

This proves application-level immutable naming and verified restore. It does not
prove Nebius bucket Object Lock, versioning, retention policy, or IAM denial of
overwrite/delete. A live remote HEAD recheck was attempted in this review environment
but AWS credentials were unavailable; the frozen receipts are the remote evidence.

**Sources:** [conditional multipart publication](https://github.com/DatasunriseOU/cppmega/blob/9c4dd4b2ff257184dd9faffbc6add1a4db698608/scripts/data/publish_megatron_bundle_to_nebius_s3.py#L2419-L2530), [conditional single-put and verification](https://github.com/DatasunriseOU/cppmega/blob/9c4dd4b2ff257184dd9faffbc6add1a4db698608/scripts/data/publish_megatron_bundle_to_nebius_s3.py#L2788-L2881), [content-addressed transport](https://github.com/DatasunriseOU/cppmega/blob/9c4dd4b2ff257184dd9faffbc6add1a4db698608/scripts/data/publish_megatron_bundle_to_nebius_s3.py#L2977-L3085), [safe restore validation](https://github.com/DatasunriseOU/cppmega/blob/9c4dd4b2ff257184dd9faffbc6add1a4db698608/scripts/data/restore_megatron_bundle_from_nebius_s3.py#L344-L443), [restore binding checks](https://github.com/DatasunriseOU/cppmega/blob/9c4dd4b2ff257184dd9faffbc6add1a4db698608/scripts/data/restore_megatron_bundle_from_nebius_s3.py#L664-L702), [direct publish receipt](</Volumes/external/sources/cppmega.mlx/outputs/megatron_ready/case5_v4_20260714_093120_mini9/publish_receipt.json:395>), [archive receipt](</Volumes/external/sources/cppmega.mlx/outputs/megatron_ready/case5_v4_20260714_093120_mini9/archive_publish_receipt.json:1>), [cold-restore receipt](</Volumes/external/sources/cppmega.mlx/outputs/s3_cold_restore_case5_v4_mini9/case5_v4_20260714_093120_mini9-73a7931e9374bd29/restore_receipt.json:1>).

**Required v5 action:** retain the conditional application protocol, capture current
remote HEAD receipts with credentials, and either capture bucket versioning/Object
Lock/retention/IAM evidence or state explicitly that the guarantee stops at the
application layer.

### P2-9: objective materialization is bounded and resumable, but only mini-scale is proven

Current code streams parquet record batches, caps write rows and estimated write
bytes, writes atomically, records a source cursor, and current production ingress
validates the bounded producer layout. mini9 proves a 60-document run with 106 source
rows consumed, a maximum source pool of 106 under a 240-sample cap, 46 unused
buffered sources, 18 source files, and 152 source rows.

This removes the old all-in-memory blocker. It is not yet a full-scale capacity
receipt: the eligibility pool is bounded by sample count rather than bytes, and there
is no retained peak-RSS, temporary-disk, or complete 16K-bucket materialization
receipt.

**Sources:** [streaming defaults](https://github.com/DatasunriseOU/cppmega_mlx/blob/b20db6ab90c642292ac09c0ecf2c77b5565ccd6c/scripts/materialize_megatron_objectives.py#L121-L128), [record-batch source reader](https://github.com/DatasunriseOU/cppmega_mlx/blob/b20db6ab90c642292ac09c0ecf2c77b5565ccd6c/scripts/materialize_megatron_objectives.py#L411-L508), [write-byte budget](https://github.com/DatasunriseOU/cppmega_mlx/blob/b20db6ab90c642292ac09c0ecf2c77b5565ccd6c/scripts/materialize_megatron_objectives.py#L652-L785), [atomic/bounded pool flow](https://github.com/DatasunriseOU/cppmega_mlx/blob/b20db6ab90c642292ac09c0ecf2c77b5565ccd6c/scripts/materialize_megatron_objectives.py#L803-L975), [cursor binding](https://github.com/DatasunriseOU/cppmega_mlx/blob/b20db6ab90c642292ac09c0ecf2c77b5565ccd6c/scripts/materialize_megatron_objectives.py#L1106-L1153), [production bounded-producer validation](https://github.com/DatasunriseOU/cppmega_mlx/blob/b20db6ab90c642292ac09c0ecf2c77b5565ccd6c/cppmega_mlx/data/production_bundle.py#L1306-L1472), [mini9 selection receipt](</Volumes/external/sources/cppmega.mlx/outputs/megatron_ready/case5_v4_20260714_093120_mini9/provenance/objective_contract_seq1024.json:770>), [source snapshot](</Volumes/external/sources/cppmega.mlx/outputs/megatron_ready/case5_v4_20260714_093120_mini9/provenance/objective_contract_seq1024.json:1308>).

**Required v5 action:** run every target bucket with a byte-bounded source pool,
retain peak RSS and peak temporary/free-disk receipts, exercise resume from a
recorded cursor, and production-open every resulting prefix.

## Objective and sidecar materialization status

mini9 contains 60 objective documents and 46,371 stored tokens. The objective
contract planned and realized all nine task families: 30 causal LM, 3 FIM, 3 AST-FIM,
6 IFIM, 6 commit-diff, 6 pre-to-post, and two each of symbol, type, and callee
recovery. The 36,707 contract loss-token total is internally consistent with the
stored trailing-zero convention, but not with runtime target alignment.

All 20 fixed-width token sidecars and all 11 graph sidecar columns are declared and
artifact-bound. Physical presence must not be confused with training compatibility:
the graph recipe mismatch and sparse relation coverage remain blockers.

**Sources:** [planned objective mix](</Volumes/external/sources/cppmega.mlx/outputs/megatron_ready/case5_v4_20260714_093120_mini9/provenance/objective_contract_seq1024.json:709>), [realized mix](</Volumes/external/sources/cppmega.mlx/outputs/megatron_ready/case5_v4_20260714_093120_mini9/provenance/objective_contract_seq1024.json:721>), [artifact sidecars](</Volumes/external/sources/cppmega.mlx/outputs/megatron_ready/case5_v4_20260714_093120_mini9/provenance/objective_artifact_seq1024.json:10>), [manifest totals](</Volumes/external/sources/cppmega.mlx/outputs/megatron_ready/case5_v4_20260714_093120_mini9/manifest.json:2350>).

## Evidence matrix

`Runtime-covered` means a repository test or local runtime path exercised
the code. `Live-proven` means the CASE5 v4 run or immutable mini9/cold
restore proves the behavior. These are intentionally separate.

| Area | Implemented at current HEAD | Runtime-covered | Live-proven | Receipt-gated | Verdict |
|---|---|---|---|---|---|
| Clean revision pin | Yes | Yes | Yes, v4 pinned to `63b0f58` | Yes | Pass for provenance; current fixes are outside that pin |
| Binary/oversize data ingestion fixes | Yes | Focused tests | No replay of 18 failed units | No | Current v4 failures remain |
| Forge-agnostic source project identity | Yes | Focused tests | No regenerated repo list/replay | No | Current v4 eight failures remain |
| Typed edge and point-anchor fixes | Yes | Focused tests | No replay of five failed units | No | Current v4 failures remain |
| Large-repo line/parse fixes | Yes | Focused tests | No 4.4BSD/DXC replay | No | Current v4 failures remain |
| Streaming objective materializer | Yes | Yes | Mini9 only | Partial | Full-scale RSS/disk/resume proof missing |
| Loss mask and label doc isolation | **No** | Reader audit exposes defect | **Broken: 101 cross-doc labels** | No | **P0 blocker** |
| Attention and graph doc isolation | Yes | Code/runtime path | mini9 document IDs present | Partial | Separate from broken label mask |
| Graph objective compatibility | **No** | Ingress gap demonstrated | mini9 is `call,type/64` | No | **P0 blocker** |
| Source identity producer/registry | Yes | Manual SQLite/token audit | mini9 passes | Partial | Ingress semantic validation missing |
| Tokenizer contract compatibility | Yes, fail closed | Current opener rejects mini9 | Rejection reproduced | Yes | mini9 stale under current HEAD |
| Direct S3 publish | Yes | Publisher tests/code | 65 artifacts verified | Yes | Application-level pass |
| Archive and cold restore | Yes | Restore code and production opener path | 66 members, restored verified | Yes | Byte-integrity pass |
| Bucket-level WORM | Outside current receipts | No | No | No | Object Lock/versioning/IAM evidence missing |
| 5,000-step capacity | Reporter exists | Live parquet scan, zero skips | 22.48% snapshot | No | Corpus and disk capacity insufficient |

## v5 prerequisites and acceptance gates

1. Freeze clean, explicit MLX and cppmega producer commits and contract hashes. The
   launch receipt, materializer receipt, build plan, and final manifest must agree.
2. Fix and version loss-mask alignment. Add end-to-end multi-document MMIDIDX tests
   through `LMTokenBatch.target_mask` for every objective. Gate on exact
   contract/runtime loss equality and zero trained cross-document targets.
3. Define one graph recipe for materializer, converter, ingress, model, and Stage-1.
   Gate exact relations, top-k, weights, route mapping, and per-relation coverage.
4. Bind objective materializer implementation, explicit repo root, commit, dirty
   state, command/config, environment, and output hashes.
5. Regenerate `repo_list.json` with forge-agnostic identities. Replay all
   33 currently failed units under the pinned v5 runtime and require zero failures.
6. Add production SQLite integrity/FK/token-ID registry validation before mmap.
7. Materialize every bucket with byte-bounded source and write buffers. Retain
   peak-RSS, peak temporary-disk, free-disk, source-cursor, and resume receipts.
8. Reach 983,040,000 trained tokens for one unique 5,000-step pass, or approve and
   receipt an explicit epoch-reuse schedule. Do not equate reuse with unique coverage.
9. Reserve more than the 188.599 GiB fixed payload floor plus measured graph, source,
   archive, temporary, filesystem, and safety headroom. The current 100 GiB gate is
   insufficient.
10. Build a new bundle ID with the current tokenizer/delimiter/domain contracts and
    current portable publisher. Production-open every bucket before publication.
11. Publish with conditional immutable keys, verify remote HEAD with credentials,
    cold-restore into an empty root, and production-open the restored copy. Capture
    bucket Object Lock/versioning/retention/IAM state or state the weaker boundary.
12. Preserve all v4/mini9 remote objects and receipts unchanged as negative evidence.

## Verification performed

| Check | Result |
|---|---|
| Actual MMIDIDX reader plus `LMTokenBatch.target_mask` audit | 60 sequences; stored 36,707 vs runtime 36,677; 101/101 real cross-document transitions trained |
| Source registry SQLite/token audit | `integrity_check=ok`; no FK failures; 28/28 token IDs resolved; no zero IDs |
| Production open of cold-restored mini9 at current HEAD | Correctly rejected: tokenizer contract does not match current frozen hash |
| Live parquet capacity scan | Zero skipped files; 221,012,987 trained tokens; 1,123 bucket-respecting steps |
| S3 live HEAD refresh | Not available in this shell because AWS credentials were absent; frozen publish/restore receipts reviewed |
| Current-HEAD focused tests | 175 passed across ingestion, identity, indexer, objective materialization, production bundle, and tokenizer suites |

## Final disposition

Do not start Stage-1 from mini9 or any v4 objective bundle. The cold restore is useful
proof that immutable bytes can be recovered, but those bytes contain a P0 label-mask
defect, an incompatible graph recipe, an old tokenizer contract, and incomplete
producer provenance. The active full conveyor also retains 33 failures under its old
pinned revision and does not yet meet unique-token or disk-capacity requirements.

The correct next artifact is a new v5 generation after all gates above pass. There is
no metadata-only repair path for the v4 objective parquet or final prefix.
