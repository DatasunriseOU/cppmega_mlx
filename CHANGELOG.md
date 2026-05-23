# Changelog

All notable UI-surface changes since the V7 honest-closure pass.

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
