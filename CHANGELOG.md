# Changelog

All notable UI-surface changes since the V7 honest-closure pass.

## Receipt-bound GitLab MR ingestion (2026-08-03)

- Added bounded, resumable GitLab MR inventory for the three declared GitLab
  project identities, with atomic page journals, immutable per-MR sidecars,
  exact source/target/merge SHA lineage, and fail-closed credential preflight.
- Native/build/test/SQL diffs reuse the canonical primary commit-path
  classifier. Python/JavaScript-only MRs are routed to a physically separate
  ancillary SQLite store, while excluded and terminal candidates remain
  sidecar-only.
- GitLab completion receipts are verified but explicitly not training-ready;
  CASE5 export dispatches by completion schema and still requires the exact
  primary-commit membership receipt and artifact.
- Focused verification covers exact scope, path isolation, pagination,
  crash-resume immutability, full receipt rebuilding, exporter dispatch, and
  the retained primary-membership gate.

## Rerun-safe E2E matrix reports (2026-08-02)

- Matrix aggregation now reads only the artifacts named by
  `--require-artifact`, so an earlier attempt's `e2e_matrix_report` output
  cannot be mistaken for a Playwright input during a failed-job rerun.
- Unrequested screenshot trees are excluded from the same report boundary.

## Full local suite baseline (2026-08-01)

- Exact tree `deb221aff245de651ad1adf9b52d3b486fa82c0d` completed with
  7,860 passed, 70 skipped, 2 xfailed, and 0 failed in 1,790.17 seconds.
- The compact versioned receipt preserves the exact command and hashes of the
  removed reconstructible log/JUnit files at
  `outputs/reports/mlx_full_suite_baseline_2026_08_01_rerun.json`.

## FP8 NaN fail-fast (2026-08-01)

- TileLang CUDA and Metal amax kernels now emit a per-block NaN poison flag
  alongside the maximum. This keeps NaN out of `atomic_max`, avoids a
  full-tensor staging scan, and makes `fp8_pack_tilelang` raise instead of
  deriving a finite scale from poisoned input.
- The real-MPS engine test covers finite parity and mixed finite/NaN input;
  all 22 focused FP8 tests pass.

## Case-safe source artifact isolation (2026-08-01)

- Mirrored injective case-insensitive filesystem keys for source work,
  source/extract caches, and code/commit Parquet publication. Logical repository
  names and project identities remain unchanged in manifests and sidecars, so
  `DirectXTK` and `directxtk` no longer overwrite one physical APFS path.
- Cleanup verification consumes the manifest-bound artifact filename and
  recovers the logical repo from the published JSONL, while retaining legacy
  raw-name compatibility.
- Focused verification: 79 source-contract tests, 74 conveyor/cache tests,
  7 cleanup tests, and the real SIGTERM/resume subprocess test passed.
- Updated the two legacy source-cache fixtures to write the required logical
  repository identity; the fail-closed sentinel validation remains unchanged.
- Training-data status now binds conveyor progress to the 501-repository
  archive launch receipt instead of counting the 601-row mapping superset.
- Decoded Parquet rows prove the legacy `bionic` and `v8` archive aliases both
  contain NVC source, so `nickg/nvc` is the correct project identity; only the
  alias/snapshot provenance remains unresolved.

## Canonical live training-data inventory (2026-07-31)

- Mirrored the physically audited inventory generator, pinned input config,
  LaunchAgent, regression test, and operator documentation from `cppmega`.
- The durable current snapshot and append-only semantic changelog live under
  `/Volumes/external/sources/cppmega.mlx/outputs/training_data_status/`.
- Release-ready, staged, and legacy data are reported separately; overlapping
  stores are never added into a fictional global token total.

## V7-D-block: TileLang→Metal kernel epic closure (2026-05-23 14:30 GMT+2)

D-block (TileLang→Metal kernels, 8 subtickets + Path B adapter + epic
+ breakdown task = 13 tickets) closed as **stale bd vs reality**.

- **All 10 kernels shipped** in `cppmega_mlx/nn/_tilelang/`:
  `topk_selector` (Path B + Path C, 923 LOC), `sparse_mla` /
  `sparse_mla_path_c` (BF16 fwd+bwd), `sparse_mla_fp8` /
  `sparse_mla_fp8_path_c` (FP8 fwd+bwd), `sparse_mla_blockscaled` /
  `sparse_mla_blockscaled_path_c` (e8m0 fwd+bwd),
  `mamba3` / `mamba3_path_c` (fwd+bwd via Path C),
  `_mamba3_helpers_tilelang.compute_dacs_segsum` (Triton-free).
- **Path B adapter** (`0ce`): `TileLangMetalAdapter` +
  `wrap_tilelang_metal_kernel` in
  `cppmega_mlx/nn/_tilelang/_mlx_runtime.py:628-876`. Takes TileLang
  artifact → extracts MSL → `mx.fast.metal_kernel`.
- **Test fix**: `tests/test_path_c_sparse_mla_apply_emit_indent.py`
  was calling `_mamba3_fp8_train_acceptance_region(model_config=…)`
  but the signature was simplified to `include_backward` only.
  Dropped the stale kwarg; 3 tests now PASS.
- **Regression evidence**: 407/407 kernel tests PASS across 12
  test files at commit `77a1404`. bd open count: 57 → 44.

Closed tickets: `cppmega-mlx-za1`, `za1.1`, `za1.2`, `za1.3`, `za1.4`,
`za1.5`, `za1.6`, `za1.7`, `za1.8`, `za1.9`, `za1.10`, `za1.11`,
`0ce`.

## V7-M0.7 + M0.3 closure (2026-05-23 06:00 GMT+2)

- **M0.7 (resumable training) ✅ green** — `stage_train` now
  auto-derives a sidecar opt-state file `<ckpt>.opt` when
  `checkpoint_save_path` is given without an explicit
  `opt_state_save_path`; symmetric auto-resolve on load. The
  sidecar bundles AdamW m/v moments + `rng_key` so a single
  "checkpoint" path yields full resumable semantics. Flipped
  `tests/v4/test_m07_resumable_roundtrip.py` from xfail-strict to
  PASS — resumed loss continues the baseline within <0.1 per step
  (was Δ≈0.75).
- **M0.3 (capture_logits wiring) ✅ MLX-side green** —
  `dry_forward(graph, ..., capture_logits=True, seed=N)` surfaces
  `output_logits` (shape) + `output_values` (flat list) in
  `extras`. `bench/m03_cuda_logits_parity_harness.py` now writes
  `bench/baselines/m03_mlx_logits.npy` (shape `[1, S, H]`).
  Status: `awaiting_cuda_ref` — GB10 reference is the remaining
  external blocker (bd `cppmega-mlx-uwhj`).
- **M0.5 (FastMTP parity)**: refreshed
  `bench/baselines/m05_mlx_fastmtp.json` (`mlx_loss=5.7456`,
  `weight_delta_norm=0.4046`); still awaiting GB10 CUDA reference
  (bd `cppmega-mlx-hjfn`).
- **New regression**: `tests/v4/test_m03_capture_logits.py` (4
  tests) pins shape, finite-values, and same-seed determinism
  for the capture path.

## V7-P-block: full-scale UI + convergence e2e + M0 baseline (2026-05-23)

- **P5**: DimEnvEditor scale-preset dropdown — mini / dev_128 /
  small_512 / medium_1k / large_2k / llama3_8b / llama3_70b. Each
  preset self-consistent (nh*head_dim == H). Unblocks full-scale
  H=4096 through UI. testid `dim-env-preset` + per-option ids.
- **P6/P7**: `p6_real_convergence_curve.spec.ts` asserts visible
  LossChart shows monotonic decline across 16 steps via
  median-quartile comparison. Replaces tautological "num_steps == N"
  checks.
- **P10**: `p10_hybrid_muon_adamw.spec.ts` covers muon_adamw_hybrid
  optim end-to-end at H=512 via the P5 preset selector.
- **M0.6**: `bench/m06_memory_bench.py` + generated
  `bench/baselines/m06_memory.json` — the missing artefact from bd
  t8f.6. Walks verify_and_estimate to produce per-brick
  params/activations bytes breakdown. CLI knobs --H --S --B
  --preset --depth --out so dev-128 grad-checkpoint+AdamW can
  re-run.
- **M0.7**: initial honest finding (xfail-strict, Δ=0.75 loss
  divergence on resume). Superseded by the M0.7 closure entry
  above (sidecar opt-state landed 2026-05-23 06:00 GMT+2).
- **Deferred (need GB10/hardware/multi-hour)**: P8 multi-epoch UI
  e2e, P9 1k+ step Playwright run, M0.3 GB10 CUDA-side reference,
  M0.5 GB10 FastMTP CUDA reference. bd hjfn / uwhj / A05 carry the
  remaining external work; MLX side captured for both M0.3 + M0.5.

## V7-N-block: deferred-work bd tickets + gen.run UI (2026-05-23)

- **gen.run UI**: new Inference tab with `GenerationPanel` —
  prompt_tokens parser, sampler strategy selector (greedy /
  temperature / top_k / top_p), per-strategy hyperparams, smoke
  toggle, token chips + finish_reason + elapsed_ms surface. testid
  contract: gen-prompt-tokens / gen-strategy / gen-max-new-tokens /
  gen-temperature / gen-top-k / gen-top-p / gen-seed / gen-vocab-size /
  gen-smoke / gen-run / gen-result / gen-token-{i} / gen-finish-reason /
  gen-elapsed-ms / gen-error.
- **N01..N07 bd tickets created** for xfail markers + TODO comments
  + TileLang-Metal kernel breakdown: gemm_softmax emitter wiring,
  sparse_mla path-C custom-op, triton OP_TABLE coverage (RFC §5.5),
  mamba3 wave-6 DSL port, dsa_splitk_indexer KL term,
  training._quantize_8bit dDequantizeBlockwise, za1 epic per-kernel
  breakdown.

## V7-C-block: checkpoint robustness (closed via dedup, 2026-05-23)

C01..C06 had two parallel ticket trees — the closed cousins
(q3jq/z6dj/91h1/9t1m/fazl/ylyh) already shipped the backend +
TopBar ckpt-inspect-block + train_checkpoint_save/load paths +
K7 RunHistoryPicker warm-start UX. The 6 open dupes (1q0, 9vv,
4whl, 8t3b, bjiy, 7n4u) closed with DEDUP notes pointing to the
shipped twin.

## V7-H-block tail: pause rehydrate + project run-id + redo defense + tokenizer roundtrip (2026-05-23)

- **H46**: TokenizerPlayground byte-roundtrip pill from existing
  EncodeVisualizeResult.capabilities.byte_roundtrip bool. Green/red
  data-roundtrip=ok|fail testids.
- **H41**: App.tsx useEffect on mount reads localStorage
  activeTrainRunId, hits pipeline.status, restores trainRunId +
  trainInFlight + trainPaused when running && !aborted. Stale ids
  scrubbed.
- **H42**: trainRunId persisted to
  `vbgui_active_train_run_id_<project-uuid>` so concurrent project
  trains don't clobber.
- **H43**: useHistory.redo() now returns `{snapshot, rejected}`;
  new `markRejected()` API tags current top. App.tsx runVerify
  flags any error-severity-gotcha snapshot; handleRedo surfaces a
  4s toast when redo lands on a rejected one.

## V7-L-block tail: GotchaPayload extensions (2026-05-23)

- **L48**: GotchaPayload + GotchaState gain `suggested_fix`.
  verify RPC post-processes via backend
  `_SUGGESTED_FIX_BY_ID` lookup so new gotcha ids ship with their
  one-click fix label without a UI release. GotchasTab accepts
  EITHER backend suggested_fix OR legacy AUTO_FIXABLE id; when
  onAutoFix isn't passed it renders an italic hint.
- **L49**: `parseSourceFile()` extracts the last path segment from
  GotchaState.reference; renders as a small monospace `src:`
  chip with the full path on hover.
- **L50**: per-severity BG_TINT (red/amber/blue pastel) + a
  colored pill (`data-severity` attr) so warnings stop blending
  into info gotchas.

## V7-M-block: train-extras UI surfacing (2026-05-23)

Backend was emitting 16 train-stage extras the UI silently buried in
its generic JSON dl dump. Each got a dedicated visual surface.

- **M21 / M23**: LossChart now overlays `losses_smoothed` (green) and
  `val_losses` (purple) on top of the primary `losses` curve.
  testids: `extras-loss-chart-line` / `-line-smoothed` / `-line-val`.
- **M22**: dedicated LR chart (`extras-lr-chart-svg`,
  `extras-lr-chart-line-lr`) rendered from `lr_trajectory`.
- **M24**: `perplexity` + `bits_per_byte` scalar badges.
- **M25**: `master_dtype` + `dtype_actual` badges (shown side-by-side
  so request vs reality is one glance).
- **M26**: `fp8_active` ON badge (suppressed when false).
- **M27**: `sharding-panel` with `sharding_applied` + per-rank param
  bytes formatted via `Number#toLocaleString`.
- **M28**: FIM badge with `fim_ratio` percent.
- **M29**: `side-channels-panel` listing `side_channels_observed`
  with per-channel testid.
- **M30**: per-brick grad-norm horizontal bars with
  `data-grad-norm` attribute for e2e assertion.
- **M31**: `brick_kinds` pill row (handles array or comma-string).
- **M32**: MoE dashboard — 9 keys (`routing_entropy`,
  `load_balance_loss`, `per_expert_load` bars, `dropped`/`rerouted`/
  `overflow` ratios, `capacity_per_expert`, `capacity_factor`,
  `num_experts`).
- **M33**: grad-clip activity panel (`max_grad_norm_seen` + clip count).
- **M34**: `optimizer_kind` badge for the current run.
- **M35**: `gradient_reduce_ms` badge (fake_ranks proxy).
- **M36**: inference-steps flow trace in DimensionsTab (follow-up).

All surfaces share the existing `HelpIcon` pattern — clicking the "?"
opens a modal with what/why/example.

## V7-L-block: error/status visibility (2026-05-23)

- **L46**: `ErrorDetailsPanel` parses `RpcError.data.errors[]`
  (Pydantic shape) + traceback + stage metadata, instead of
  dropping the data blob and showing only `error.message`.
- **L47**: `RunResultModal` pre-expands the first `fail` stage on
  render and adds a red outline + light-red row background to the
  failing row (`data-first-failed='true'`).

## V7-K-block: 10 missing train controls (2026-05-23)

`TrainOptionsPanel` (K3-K6, K8) + `RunHistoryPicker` (K7) +
`TrainLiveControls` (K9, K10) + dim_env-into-train wiring (K2) +
the already-present TopBar num_steps input (K1). 7 Playwright e2e
green. See `vbgui/e2e/scenarios/k*.spec.ts`.

## V7-F-block: brick constructor (2026-05-23)

`LossChart` foundation + `DimEnvEditor` with snap-to-valid fixes
(F56b) + `GalleryTab` sortable cache (F58) + `SweepPanel` H-sweep
overlay (F53) + `TokenizerMatrixTab` (F55) + `BrickContextPanel`
swap-kind (F52) + `TransplantBar` (F54) + `InsertIntoEdgeBar` (F51)
+ `ParallelComposeBar` (F57). 8 Playwright e2e green.

## V7-J general-purpose surfacing primitives

- `HelpIcon` + `HELP_TOPICS` — central registry of explanation
  modals; every numeric / structural surface that lands in this
  changelog ships with a `?` icon linked to a topic entry.
