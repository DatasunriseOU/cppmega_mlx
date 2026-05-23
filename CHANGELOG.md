# Changelog

All notable UI-surface changes since the V7 honest-closure pass.

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
