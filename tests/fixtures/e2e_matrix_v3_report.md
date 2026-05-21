# E2E Coverage Matrix v3 — Closure Report

Epic: `cppmega-mlx-y7f` (12 sub-tickets closed, 1 deferred). Tracks
delivery against `VisualBuilderPlan-v3.md` and `VisualBuilderSpec-v3.md`.

v3 closed the honest "от UI до real tiny model trainings" gap that v2
papered over with `expect(["ok","fail"]).toContain(status)` vacuous
assertions. Every claim a UI mutation makes about training behaviour is
now provable from a Playwright test that reads UI-surfaced extras.

## Stages delivered

| Stage | Ticket          | Commit    | Pytest+ | Vitest+ | E2E+ | Status |
|-------|-----------------|-----------|---------|---------|------|--------|
| V3-1  | cppmega-mlx-wly | `f443aaf` | +19     | —       | —    | ✅     |
| V3-2  | cppmega-mlx-4pp | `9eb0287` | +5      | —       | —    | ✅     |
| V3-3  | cppmega-mlx-0js | `f443aaf` | (V3-1)  | —       | —    | ✅     |
| V3-4  | cppmega-mlx-xzp | `5bf651c` | —       | +6      | —    | ✅     |
| V3-5  | cppmega-mlx-7g9 | `f53da81` | —       | —       | +5   | ✅     |
| V3-6  | cppmega-mlx-9wj | `7ba1661` | —       | (TopBar)| +4   | ✅     |
| V3-7  | cppmega-mlx-bws | `7526375` | —       | —       | +2   | ✅     |
| V3-8  | cppmega-mlx-cwz | `bbfc898` | —       | —       | +2   | ✅     |
| V3-9  | cppmega-mlx-7ut | `bbfc898` | —       | —       | +1   | ✅     |
| V3-10 | cppmega-mlx-e0t | `27e286d` | —       | —       | +1   | ✅     |
| V3-11 | cppmega-mlx-2ar | `27e286d` | —       | —       | +8   | ✅     |
| V3-12 | cppmega-mlx-1wm | —         | —       | —       | —    | ❄ deferred (P2, save/load UI scope creep) |
| V3-13 | cppmega-mlx-7yd | (this)    | —       | —       | —    | ✅     |

## Bugs fixed along the way (UI → API → training)

| ID  | Surface                                | Fix commit | Effect                                                                 |
|-----|----------------------------------------|------------|------------------------------------------------------------------------|
| B1  | `stage_train` hardcoded `optim.AdamW`  | `f443aaf`  | Lion/Muon/Lion8bit/Adam8bit/SGD UI choices now reach training math    |
| B1' | `build_model` rejected Lion/Lion8bit/Adam8bit | `f443aaf` | Pipeline no longer marks train "skipped" on those kinds, silently green-washing |
| B1''| `_make_optim` dropped `ns_steps`/`schedule` | `f443aaf` | Muon ns_steps + every schedule survive wire → buildspec hop           |
| B2  | `stage_train` only used synthetic embeds | `9eb0287` | stage_options.train.parquet_path consumed; `data_source` reported     |
| B3  | `RunResultModal` hid extras            | `5bf651c`  | Every extras key/array/object rendered with deterministic testid       |
| B4  | `buildVerifyParams` dropped `schedule`/`ns_steps` | `f53da81` | OptimTab ScheduleEditor was decorative; now schedule actually trains |
| BX  | `stage_train` forward chain zeroed grads| `f443aaf` | Residual fix — zero-init-out attention no longer kills q/k/v gradients |

## Acceptance criteria — Plan v3 §7

| Criterion                                          | Target | Actual | Status |
|----------------------------------------------------|--------|--------|--------|
| extras.optimizer_kind populated                    | yes    | yes    | ✅     |
| extras.model_summary populated                     | yes    | yes    | ✅     |
| extras.lr_trajectory[0] correct for linear_warmup  | 0      | 0      | ✅     |
| extras.data_source reports synthetic\|parquet      | yes    | yes    | ✅     |
| 0 vacuous expect(["ok","fail"]) in 1[1-5]_*.spec.ts| 0      | 0      | ✅     |
| Train button gated on gotcha severity=error        | yes    | yes    | ✅     |
| Roundtrip FAIL warning visible, non-blocking       | yes    | yes    | ✅     |
| Cross-arch deep verify (4 reps × 2 mutations)      | 8      | 8      | ✅     |
| stage_train OptimKind smoke pytest                 | 7      | 7      | ✅     |
| stage_train data ingestion pytest                  | 4+     | 5      | ✅     |
| Vitest target                                      | ≥166   | 166    | ✅     |
| Closure markdown ≤150 lines                        | yes    | 90     | ✅     |

## New surface

### Backend
- `cppmega_v4/runner/stages.py`: `_build_optimizer`, `_summarize_model`,
  `_read_first_n_tokens` helpers; extras gains `optimizer_kind`,
  `model_summary`, `data_source`, `token_count`.
- `cppmega_v4/buildspec/api.py`: `build_model` wires Lion/Lion8bit/
  Adam8bit through the cppmega_mlx factories.
- `cppmega_v4/jsonrpc/methods.py`: `_make_optim` threads ns_steps + schedule.

### UI
- `RunResultModal.tsx`: `StageExtras` + `ExtrasEntry` recursive components,
  data-testid'd cells for primitives / arrays / nested objects.
- `TopBar.tsx`: `train-num-steps` input, `trainDisabled` prop,
  `top-bar-train-disabled-reason` reason child.
- `App.tsx`: schedule + ns_steps threaded through wire payload,
  trainDisabled derived from `spec.gotchas` severity=error,
  stage_options.train.num_steps forwarded from TopBar opts.
- `AblationsTab.tsx`: per-row `ablation-final-{variant}` testid.

### E2E
- `vbgui/e2e/utils/train_extras.ts`: `readTrainExtras(page)` helper.
- `11_ui_to_train.spec.ts`: rewritten with 5 strict-content scenarios.
- `12_train_convergence.spec.ts`: 4 multi-step scenarios (convergence + sanity).
- `13_ablation_math.spec.ts`: 2 math-divergence scenarios.
- `14_cross_arch_deep.spec.ts`: 8 cross-arch scenarios.
- `15_gating.spec.ts`: 3 Train-button-gating scenarios.
- `17_roundtrip_warning.spec.ts`: 1 roundtrip-FAIL non-blocking scenario.

## Deferred / soft gaps

1. **V3-12 spec save/load roundtrip** — requires net-new Save/Load UI
   (TopBar buttons + localStorage + spec deserializer). Wire-level spec
   round-trip is already covered by `test_jsonrpc_methods.py`.
2. **UI plumbing for V3-2 parquet_path** — backend accepts
   `stage_options.train.parquet_path` but the DataInspector selection
   does not yet forward it. Follow-up UI ticket.
3. **Distributed / sharding apply → real distributed train** — out of
   scope (requires multi-device).
4. **Memory peak vs estimate comparison** — out of scope (requires
   instrumentation of mx.metal.get_peak_memory across runs).

## Pre-existing red (excluded from regression gates)

`tests/v4/test_path_d_runtime_adapter.py::test_gdn_chunk_o_metal_threadgroup_memory_fits_device_limit`
still fails (61440 > 32K threadgroup memory limit) — owned by parallel
work in `cppmega_mlx/runtime/path_c_fusion.py`, unchanged in this epic.
