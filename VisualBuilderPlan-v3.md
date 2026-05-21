# VisualBuilderPlan-v3 — Deep E2E Verification (UI → API → Real Training Math)

**Status**: in-progress 2026-05-21 (epic `cppmega-mlx-y7f`)
**Driver**: honest audit of v2 closure (bb0). v2 declared "1188 cells, 0 hard failures"
but the 5 "deep" UI→train scenarios used vacuous `expect(["ok","fail"]).toContain(status)`
assertions. The status was returned by `stage_train`; nothing proved the UI mutation
**affected the trained model**. v3 closes that gap.

This plan supersedes v2 §8 acceptance criteria ONLY for the UI→training assertions;
v2 stage delivery + 1188-cell breadth remains valid.

## 1. Why v3 exists

Audit on 2026-05-21 found 3 categories of unproven claims and 3 confirmed backend bugs:

### Confirmed backend bugs (v2 ignored these)

| ID  | Bug                                                           | File                                       | Effect                                                                |
|-----|---------------------------------------------------------------|--------------------------------------------|-----------------------------------------------------------------------|
| B1  | `stage_train` hardcodes `optim.AdamW`                         | `cppmega_v4/runner/stages.py:425`          | UI OptimTab choice (Lion/Muon/Adam8bit) is silently ignored at train  |
| B2  | `stage_train` uses synthetic Gaussian embeddings              | `cppmega_v4/runner/stages.py:447`          | tokenizer + parquet selection in UI has zero effect on training input |
| B3  | `RunResultModal` does not surface `extras` (losses, lr, delta)| `vbgui/src/components/RunResultModal.tsx`  | UI hides training math from user; tests cannot assert against extras  |

### Vacuous assertions in `11_ui_to_train.spec.ts`

| Test                           | Old assertion                                | Why vacuous                              |
|--------------------------------|----------------------------------------------|------------------------------------------|
| activation glu→swiglu          | `expect(["ok","fail"]).toContain(status)`    | passes even if activation ignored        |
| schedule linear_warmup w=4     | `await stage-train.waitFor()` (no content)   | no check that lr_trajectory shape matches|
| pre_norm rmsnorm→layernorm     | `expect(["ok","fail"]).toContain(status)`    | passes even if norm ignored              |
| Auto-group + Train             | `expect(["ok","fail"]).toContain(status)`    | passes even though B1 ignores optimizer  |
| AblationsTab Run               | `rows >= 2`                                  | passes even if all variants same loss    |

### Cross-cut gaps not tested at all

- **Multi-step convergence**: matrix uses N=2 (proves non-NaN, not learning).
- **Cross-architecture**: deep UI→train only run on `llama3_8b` (1/57 presets).
- **Sharding application**: UI shows proposals, never asserted that train uses them.
- **Memory match**: UI estimate vs real `mx.metal.get_peak_memory()` never compared.
- **Gotcha gating**: critical gotcha should disable Train; not verified.
- **Validator gating**: verify=error should block Train; not verified.
- **Lifecycle**: WS reconnect mid-train, spec save/load roundtrip.

## 2. Goal

Every UI mutation that claims to change training behavior **must produce an observable
difference in training math**, asserted from a Playwright test that reads the value
out of UI-surfaced extras.

Test pattern (the only one acceptable from v3 onward):

```ts
const { extras } = await trainResult(page);

expect(extras.model_summary.mlp_activation).toBe("swiglu");          // propagated
expect(extras.losses.length).toBeGreaterThanOrEqual(5);              // multi-step
expect(extras.losses[0]).toBeGreaterThan(extras.losses[N-1]);        // learned
expect(extras.weight_delta_norm).toBeGreaterThan(1e-3);              // moved
expect(extras.lr_trajectory[0]).toBeCloseTo(base_lr * 0.25, 4);      // schedule shape
expect(extras.optimizer_kind).toBe("lion");                          // B1 fixed
```

No more `["ok","fail"]`. Status check is a sanity gate only; the **content** is what
proves the claim.

## 3. Stages

13 stages in epic `cppmega-mlx-y7f`, each with bd ticket. Run in dependency order;
do not advance to next until current is green.

| Stage | Ticket          | Title                                                              | Type    | Pri | Depends |
|-------|-----------------|--------------------------------------------------------------------|---------|-----|---------|
| V3-1  | cppmega-mlx-wly | stage_train honors OptimKind (B1 fix + extras.optimizer_kind)      | bugfix  | P0  | —       |
| V3-2  | cppmega-mlx-4pp | stage_train consumes tokenizer+parquet from spec (B2 fix)          | bugfix  | P0  | —       |
| V3-3  | cppmega-mlx-0js | stage_train extras includes model_summary                          | feature | P1  | V3-1    |
| V3-4  | cppmega-mlx-xzp | RunResultModal surfaces extras with testids (B3 fix)               | feature | P0  | —       |
| V3-5  | cppmega-mlx-7g9 | Rewrite 11_ui_to_train.spec.ts with strict content assertions      | task    | P0  | V3-1,3,4|
| V3-6  | cppmega-mlx-9wj | Multi-step convergence (12_train_convergence.spec.ts)              | task    | P1  | V3-3,4  |
| V3-7  | cppmega-mlx-bws | Ablation loss-divergence (13_ablation_math.spec.ts)                | task    | P1  | V3-5    |
| V3-8  | cppmega-mlx-cwz | Critical gotcha disables Train + assertion                         | feature | P1  | V3-4    |
| V3-9  | cppmega-mlx-7ut | verify=error disables Train + assertion                            | feature | P1  | V3-4    |
| V3-10 | cppmega-mlx-e0t | Roundtrip FAIL warning, does not block train                       | task    | P2  | V3-4    |
| V3-11 | cppmega-mlx-2ar | Cross-architecture deep verify (14_cross_arch_deep.spec.ts)        | task    | P1  | V3-5    |
| V3-12 | cppmega-mlx-1wm | Spec save/load roundtrip → identical train extras                  | task    | P2  | V3-5    |
| V3-13 | cppmega-mlx-7yd | Closure report `tests/fixtures/e2e_matrix_v3_report.md`            | doc     | P3  | all     |

Deferred (not in this epic, follow-up):
- Sharding apply → real distributed train (needs multi-device, scope creep)
- Memory peak comparison (requires actual `mx.metal.get_peak_memory` calls + tolerance)
- WS reconnect mid-train (lifecycle, not "training math")

## 4. Per-stage acceptance criteria

### V3-1 stage_train honors OptimKind

- `stages.py` reads `spec.optim.groups[0].kind` and instantiates corresponding optimizer
  from `cppmega_mlx.training.optimizers` (AdamW / Lion / Lion8bit / Adam8bit / Muon /
  MuonAdamWHybrid / SGD)
- extras gains `optimizer_kind: str` field
- pytest: 7 new tests in `tests/v4/test_stage_train_optimizers.py`, one per OptimKind,
  asserts loss is finite and weight_delta > 0
- regression: existing 2306+ pytest stays green

### V3-2 stage_train consumes data

- If `spec.data` has `tokenizer_path` + `parquet_path`, train loads first N tokens
  from parquet, tokenizes via the configured tokenizer, uses those token IDs as
  targets (and a learned `nn.Embedding(vocab, hidden)` as input embeds)
- Falls back to synthetic embeds + random targets only when `spec.data` is None
  (backwards compat for matrix tests)
- extras gains `data_source: "synthetic"|"parquet"` and `token_count: int`
- pytest: 4 new tests in `tests/v4/test_stage_train_data.py` — synthetic path,
  parquet path, mismatched vocab raises clear error, tokenizer load failure surfaces

### V3-3 model_summary in extras

- `stage_train.extras["model_summary"]` is `{mlp_activation, attention_pre_norm,
  attention_post_norm, mlp_pre_norm, mlp_post_norm, optimizer_kind, schedule_kind,
  num_brick_kinds}`
- Helper `_summarize_model(spec, optimizer)` in stages.py
- pytest: 5 new tests asserting summary correctness for various configs

### V3-4 RunResultModal extras display

- For each stage with non-error extras, add a clickable disclosure row
- Inside expand: data-testid'd cells for every extras key:
  - `run-result-extras-{stage}-{key}` for primitives
  - `run-result-extras-{stage}-losses-{i}` for arrays
  - `run-result-extras-{stage}-model-{field}` for model_summary
- vitest: 6 new tests covering shape rendering

### V3-5 Rewrite 11_ui_to_train.spec.ts

Five scenarios, each with **at least 3 content assertions** beyond status row presence:

1. **Activation glu→swiglu**: extras.model_summary.mlp_activation === "swiglu",
   losses[0] !== losses[-1] (learning happened with the new activation).
2. **Schedule linear_warmup w=4**: extras.lr_trajectory[0] === base_lr * 0.25,
   extras.lr_trajectory[3] === base_lr (warmup completed), extras.schedule_kind ===
   "linear_warmup".
3. **pre_norm switch attention**: extras.model_summary.attention_pre_norm ===
   "layernorm", weight_delta_norm > 0.
4. **Auto-group + optimizer change to Lion**: extras.optimizer_kind === "lion",
   losses finite (B1 fix verified end-to-end).
5. **Optimizer change to Muon**: extras.optimizer_kind === "muon", weight_delta > 0
   (Muon != AdamW math regression).

### V3-6 multi-step convergence

- New test `12_train_convergence.spec.ts`
- Drive N=8 step training via stage_options.train.num_steps in OptimTab
- Assert `losses[0] > losses[7] * 1.05` (5% loss reduction floor)
- 4 preset variants: llama3_8b, gpt2_small_124m, mistral_7b, tiny_aya_parallel

### V3-7 ablation divergence

- `13_ablation_math.spec.ts`
- Activation axis: glu vs swiglu vs gelu, 4 steps each
- Assert pairwise `|final_loss_a - final_loss_b| > 1e-3` for at least one pair
- Optimizer axis: adamw vs lion → assert weight_delta_norm differs by >10%

### V3-8 Critical gotcha gates Train

- Force a gotcha-critical config (e.g., loss type incompatible with brick output)
- UI: `top-bar-train` button becomes disabled, tooltip "critical gotcha blocks train"
- Test asserts disabled state + tooltip

### V3-9 verify=error gates Train

- Same pattern with a validator error (e.g., parallel-block requires pre_norm)
- UI: train disabled, banner shows verify error count

### V3-10 roundtrip FAIL warning

- Force a parquet/tokenizer roundtrip mismatch
- UI shows yellow banner, but Train remains enabled (training is robust to roundtrip
  fail since synthetic targets bypass it)
- Assert banner + train still works (status="ok")

### V3-11 cross-architecture deep verify

- 6 family-reps × 3 mutations = 18 scenarios in `14_cross_arch_deep.spec.ts`
- Family reps: llama3_8b, mistral_7b, gpt2_small_124m, qwen3_8b, gemma2_2b, deepseek_v3
- Mutations: activation, schedule, optimizer
- Assert same strict content as V3-5

### V3-12 save/load roundtrip

- Build spec in UI → Save → Load → Train → assert extras identical to first run
- Tests load → save → diff JSON should be empty

### V3-13 Closure report

- ≤150 lines markdown report mirroring v2 layout
- Stage table with commits, before/after assertion counts, regression totals

## 5. Workflow per stage (Goal directive)

1. `bd update <id> --claim` → in_progress
2. Implement (smallest atomic change)
3. Code review pass (Self-review via diff; gsd-code-reviewer only if >100 LOC or
   touches >3 files)
4. Perf check (no >5% regression in pytest wall-clock for affected suite)
5. Regression (pytest + vitest + playwright affected files)
6. `git add` specific files (no `-A`) → commit → push to origin/main
7. `bd close <id>` only after push verified

## 6. Pre-existing red (excluded from regression gates)

- `tests/v4/test_path_d_runtime_adapter.py::test_gdn_chunk_o_metal_threadgroup_memory_fits_device_limit`
  (61440 > 32K limit, parallel agent's WIP in `path_c_fusion.py`, not owned by this work)

## 7. Done definition

All 13 V3 tickets closed. Playwright `11/12/13/14_*.spec.ts` green with strict
assertions (no `["ok","fail"]` in `vbgui/e2e/scenarios/1[1-4]_*.spec.ts`).
Closure doc at `tests/fixtures/e2e_matrix_v3_report.md`.
