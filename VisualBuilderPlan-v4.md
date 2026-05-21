# VisualBuilderPlan-v4 — Real Data Flow + Remaining UI→Train Coverage

**Status**: planned 2026-05-21 (epic `cppmega-mlx-br1` — id to fill after `bd create`)
**Driver**: honest V3 closure audit found 20 remaining un-tested UI→train flows.
Most damning: backend `stage_train` accepts `parquet_path` (V3-2) but the
DataInspector selection does not forward it, so user-chosen parquet shards
still have zero effect on training. UI Tokenizer Playground is fully decorative
for training. 6 of 10 activations, 5 of 6 schedules, 3 of 4 loss kinds untested
through full UI→train chain.

v4 supersedes v3 §7 "out of scope" entries 1, 2, 5, and adds 11 more.

## 1. Why v4 exists

Three classes of remaining gaps:

### A. Backend ready, UI not plumbed (data-flow honesty)

| ID  | Gap                                                       | Why bad                                                                                   |
|-----|-----------------------------------------------------------|-------------------------------------------------------------------------------------------|
| G1  | DataInspector parquet → train opts.parquet_path           | User selects parquet in Data tab; train silently uses synthetic random targets            |
| G2  | Tokenizer Playground tokenizer → train                    | Tokenizer selection is decorative; train uses raw int IDs from parquet, no tokenize step  |
| G3  | UI exposes parquet_path testid for e2e drive              | No way for Playwright to set parquet_path through UI walks                                |

### B. Untested mutations (UI changes but math effect never asserted)

| ID  | Gap                                                       | Coverage   |
|-----|-----------------------------------------------------------|------------|
| G4  | 9 of 10 ActivationName values untested through UI→train   | 1/10       |
| G5  | 5 of 6 schedule kinds untested through UI→train           | 1/6        |
| G6  | 3 of 4 LossKind values untested through UI→train          | 1/4        |
| G7  | RewritersTab apply → train math change                    | 0/4        |
| G8  | muon_adamw_hybrid split (which params → Muon vs AdamW)    | wire only  |
| G9  | side_channels (doc_ids, token_ids) reach forward pass     | 0          |

### C. Real-world honesty gaps

| ID  | Gap                                                       |
|-----|-----------------------------------------------------------|
| G10 | Convergence test uses synthetic targets; real-data conv untested |
| G11 | Inference after train: model output differs from initial — never asserted |
| G12 | activation/norm propagation across all 57 presets (V3-11 did 4 × 2)      |
| G13 | Real backend gotcha trigger (not page.route fake) — does incompatible loss config actually produce severity=error gotcha? |
| G14 | Tokenizer roundtrip OK actually means byte-exact decode (V3-10 only tested FAIL non-blocking) |

### Explicitly out of scope (defer to v5+)

- WS reconnect mid-train (lifecycle)
- Sharding apply → distributed train (needs multi-device)
- Memory peak vs estimate (needs runtime instrumentation)
- Concurrent Train clicks / Cancel button (lifecycle)
- Spec save/load (V3-12 deferred, keep deferred)
- fp8 / mixed_precision (subset of sharding)
- 100+ layer realistic depth on stage_train (synthetic-test architecture limit)
- Checkpoint save/resume (no stage_train checkpoint code)

## 2. Goal

After v4: for every UI surface that claims to mutate training, an e2e
test reads back the effect via extras or inference output. No silent
decoration left.

## 3. Stages

14 stages in epic `cppmega-mlx-br1`. P0/P1 first; defer P3 if context-limited.

| Stage | Title                                                                | Type    | Pri | Depends |
|-------|----------------------------------------------------------------------|---------|-----|---------|
| V4-1  | UI: DataInspector forwards parquet_path via stage_options.train       | feature | P0  | —       |
| V4-2  | Backend: stage_train accepts tokenizer_path; tokenize parquet text    | feature | P0  | —       |
| V4-3  | UI: Tokenizer Playground selection forwards tokenizer_path            | feature | P0  | V4-2    |
| V4-4  | E2E: real-data convergence (parquet+tokenizer → losses fall)          | test    | P1  | V4-1,3  |
| V4-5  | E2E: all 10 activations × UI→train propagation                        | test    | P1  | —       |
| V4-6  | E2E: all 6 schedule kinds × UI→train propagation                      | test    | P1  | —       |
| V4-7  | E2E: all 4 LossKind × UI→train + extras                               | test    | P1  | —       |
| V4-8  | Backend+E2E: rewriters apply → spec graph change → train extras       | test    | P1  | —       |
| V4-9  | E2E: muon_adamw_hybrid split — extras reports Muon/AdamW buckets      | test    | P2  | —       |
| V4-10 | Backend+E2E: side_channels reach forward → forward output differs     | feature | P2  | —       |
| V4-11 | E2E: inference after train — output diverges from initial             | feature | P2  | V4-1    |
| V4-12 | E2E: cross-arch activation/norm propagation × 12 presets              | test    | P2  | V4-5    |
| V4-13 | E2E: real config triggers backend gotcha (no page.route fake)         | test    | P3  | —       |
| V4-14 | Closure report `tests/fixtures/e2e_matrix_v4_report.md`               | doc     | P3  | all     |

## 4. Per-stage acceptance criteria

### V4-1 DataInspector parquet path threading

- DataInspector has a "Use for train" button (`data-testid="data-use-for-train"`)
  that records the loaded parquet path in App state (`trainParquetPath`).
- App.handleRunPipeline forwards `trainParquetPath` via
  `stage_options.train.parquet_path` when set.
- Visible indicator near Train button: `data-testid="train-data-source"`
  reads "parquet: <basename>" or "synthetic".
- pytest: 4 new tests (UI state, wire payload, backend receives it,
  fallback when null). vitest: 3 new (DataInspector button + App
  state + TopBar indicator).

### V4-2 Backend tokenize parquet text

- stage_train opts.tokenizer_path → loads tokenizer via existing
  `tokenizer.load_preset` or path-based loader. If parquet has a `text`
  column, encode rows; else fall through to V3-2's raw-int path.
- extras gains `tokenizer_used: str | null` (path basename when used).
- 5 new pytest in `test_stage_train_tokenize.py`: with text column,
  without, missing tokenizer (fallback), corrupted parquet, vocab clip.

### V4-3 UI tokenizer threading

- Tokenizer Playground "Use for train" button records the loaded
  preset name in App state. App forwards via stage_options.train.tokenizer_path.
- TopBar `train-data-source` upgrades to `parquet+tokenizer: <names>`.
- vitest: 3 new (Playground button, App state, TopBar text).

### V4-4 Real-data convergence

- New spec `18_real_data_convergence.spec.ts`. Picks a known-good
  parquet fixture from `tests/fixtures/build_e2e_matrix.py`, picks
  its matching tokenizer, sets N=8 steps, asserts:
  - `extras.data_source === "parquet"`
  - `extras.token_count > 0`
  - `extras.tokenizer_used` matches selection
  - For deep presets (llama3_8b, mistral_small_3_1):
    `losses[7] < losses[0] * 0.95` (real data convergence floor — stricter
    than synthetic V3-6's tailAvg<headAvg because real tokens give signal).

### V4-5 All 10 activations through UI→train

- New spec `19_activation_propagation.spec.ts`. For each
  ActivationName in `["glu", "gelu", "relu", "relu2", "sqrelu", "silu",
  "mish", "swiglu", "geglu", "reglu", "xielu"]` (11 entries — the IS_GATED
  set varies, BrickContextPanel filters compatible):
  - Drop mlp brick to canvas, open BrickContextPanel, set activation,
    Apply, Train. Assert `extras.model_summary.mlp_activation === selected`.
  - Loss is finite.

### V4-6 All 6 schedules through UI→train

- New spec `20_schedule_propagation.spec.ts`. For each ScheduleKind in
  `["constant", "linear_warmup", "cosine", "wsd", "inv_sqrt", "polynomial"]`:
  - Open OptimTab, toggle schedule, select kind, fill required fields
    (warmup_steps for warmup; total_steps for cosine/polynomial; etc),
    Apply, Train. Assert `extras.schedule_kind === selected`.
  - Assert `lr_trajectory` shape matches kind's analytical formula
    (e.g. cosine → monotone-non-increasing after warmup;
    polynomial → power-law decay).

### V4-7 All 4 LossKind through UI→train

- New spec `21_loss_kind_propagation.spec.ts`. For each LossKind in
  `["cross_entropy", "mtp_weighted", "ifim_weighted", "mhc_weighted"]`:
  - Open LossTab, set kind, set required params (k, betas for MTP),
    Apply, Train. Assert losses finite, model_summary contains
    `loss_kind` (NEW extras field — V4-7 backend work).
  - Skip kinds that have hard preconditions not present in default preset
    (e.g. mtp_weighted requires MTPRewriter — included or test selects
    a preset that has it).

### V4-8 Rewriters apply → train math change

- New backend: extras.model_summary gains `rewriters_applied: list[str]`.
- New spec `22_rewriter_propagation.spec.ts`. Enable MTPRewriter (or
  IFIMRewriter / MHCRewriter), Apply, Train. Assert:
  - `extras.model_summary.rewriters_applied` contains the rewriter name.
  - For MTP: extras.losses count multi-head loss contribution
    (or just verify shape change in some downstream extras key).

### V4-9 muon_adamw_hybrid split

- Backend: when `optimizer_kind === "muon_adamw_hybrid"`, extras gains
  `muon_group_size` and `adamw_group_size` (param counts in each bucket).
- New spec `23_hybrid_optimizer_split.spec.ts`. UI selects
  muon_adamw_hybrid, Train. Assert both counters > 0, sum equals total.

### V4-10 side_channels reach forward

- Backend: stage_train accepts `opts.side_channels: {doc_ids?, token_ids?}`.
  When present, asserts forward pass observes the channels (via a probe
  hook). extras gains `side_channels_observed: list[str]`.
- New spec `24_side_channels.spec.ts`. UI enables doc_ids side-channel,
  Train. Assert `extras.side_channels_observed` includes `"doc_ids"`.

### V4-11 Inference after train — output diverges

- Backend: extras gains `inference_probe`: result of a single
  forward pass over a fixed seed both before training (initial weights)
  and after. Returned as `{l2_diff: float, cos_sim: float}`.
- New spec `25_inference_after_train.spec.ts`. Run Train. Assert
  `extras.inference_probe.l2_diff > 0.01` (model actually changed
  observable output, not just optimizer state).

### V4-12 Cross-arch activation/norm × 12 presets

- New spec `26_cross_arch_brick_mutations.spec.ts`. 12 presets × 2
  brick-context mutations (activation + pre_norm) = 24 cells. Each
  scenario clicks the preset-suffixed brick node, opens the
  BrickContextPanel for its mlp/attention, mutates, runs Train, asserts
  propagation via model_summary.
- Solves the V3-11 testid gap by introspecting actual `data-testid`
  values (`brick-context-*`) at runtime instead of hardcoding.

### V4-13 Real backend gotcha trigger

- New spec `27_real_gotcha.spec.ts`. Set up an actually-incompatible
  config (e.g. MTPRewriter without enough head outputs, OR pre_norm=none
  + post_norm=none on attention). Wait for verify. Assert
  `top-bar-train-disabled-reason` appears WITHOUT page.route() shim.
- Proves V3-8/V3-9 gating logic catches real-world gotchas, not just
  injected ones.

### V4-14 Closure report

- ≤150-line markdown mirroring v3 layout. Stage table, gaps closed,
  remaining v5+ items, regression totals.

## 5. Workflow per stage (Goal directive)

Same as v3: claim → implement → review → perf → regression → commit
specific files → push → close bd ticket.

## 6. Acceptance counters

| Surface                                | Before v4 | Target v4 |
|----------------------------------------|-----------|-----------|
| pytest                                 | ~2330     | ≥2350     |
| vitest                                 | 166       | ≥176      |
| Playwright deep e2e (scenarios 11-17)  | 26        | ≥60       |
| Activations through UI→train           | 1         | 10        |
| Schedules through UI→train             | 1         | 6         |
| LossKinds through UI→train             | 1         | 4         |
| stage_train extras keys                | 9         | 13        |
| Real-data train scenarios              | 0         | 1+        |
| Inference-after-train scenarios        | 0         | 1+        |

## 7. Done definition

All P0+P1+P2 tickets closed (P3 may defer). Real parquet+tokenizer
selection in UI demonstrably reaches train. 10 activations + 6 schedules
+ 4 loss kinds all proven through UI→train. Inference output asserted
post-train. Closure doc at `tests/fixtures/e2e_matrix_v4_report.md`.
