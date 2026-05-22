# VisualBuilderPlan-v6 — UI Wiring + Real Math-Effect for V5 Placeholders

**Status**: planned 2026-05-22 (epic `cppmega-mlx-te6`)

## bd ticket ID mapping (post-creation)

| H#  | bd id           | Title                                            |
|-----|-----------------|--------------------------------------------------|
| H01 | cppmega-mlx-fjb | Sharding UI → train                              |
| H02 | cppmega-mlx-8p1 | fp8/mixed_precision toggles                      |
| H03 | cppmega-mlx-ou1 | Cancel button + WS abort                         |
| H04 | cppmega-mlx-c4f | train-warm-start checkbox                        |
| H05 | cppmega-mlx-fr5 | checkpoint path inputs                           |
| H06 | cppmega-mlx-ays | UI N=100 real-corpus walk                        |
| H07 | cppmega-mlx-75m | DimensionsTab Apply dispatch                     |
| H08 | cppmega-mlx-cem | train-probe-text textarea                        |
| H09 | cppmega-mlx-sq3 | Save/Load roundtrip parity                       |
| H10 | cppmega-mlx-v5i | SideChannelsTab proper                           |
| H11 | cppmega-mlx-i04 | memory peak parity                               |
| H12 | cppmega-mlx-vlh | WS reconnect recovery e2e                        |
| H13 | cppmega-mlx-65v | other-stages extras UI walk                      |
| H14 | cppmega-mlx-ep7 | AblationsTab full extras                         |
| H15 | cppmega-mlx-8rb | real per-rank shard simulation                   |
| H16 | cppmega-mlx-oku | real mlx dtype switching                         |
| H17 | cppmega-mlx-gqu | real cross-doc attention mask                    |
| H18 | cppmega-mlx-6ls | MoE forward hook (entropy + load balance)        |
| H19 | cppmega-mlx-56o | strict identical-loss-continuation               |
| H20 | cppmega-mlx-4o9 | fake-rank distributed smoke                      |
| H21 | cppmega-mlx-bqy | inference-after-resume bounded drift             |
| H22 | cppmega-mlx-1ne | concurrent Train clicks                          |
| H23 | cppmega-mlx-age | fp16 dtype option                                |
| H24 | cppmega-mlx-sby | 100+ step UI walk w/ checkpoint                  |
| H25 | cppmega-mlx-8px | honest categorisation audit                      |

**Driver**: V5 closure audit found that out of 25 closed tickets, only
~11 are real math-effect. The remaining 14 split between:
- **"backend ready + UI not wired"** — 10 gaps where pytest passes but
  pressing nothing through UI can reproduce the effect (G05/G07/G09/G10/
  G12/G17/G19/G20/G15/G11)
- **observation-only** — 4 gaps where extras report numbers but UI parity
  / forward effect / lifecycle recovery never asserted (G06/G08/G18/G21)
- **analytical proxies** — 3 gaps where extras reports `λ×something_norm`
  instead of running actual kernel (G07/G17/G25)

v6 closes the **backend-ready → UI-wired → real-math-effect** triangle.

## 1. Why v6 exists

V5 honest audit produced this taxonomy (25 gaps):

### A. UI wiring of V5 backends (10 gaps)
Backend contracts shipped, no UI surface or App dispatch route. Through
UI a user cannot reproduce what pytest covers.

| Gap  | V5 origin | UI work needed                                      |
|------|-----------|-----------------------------------------------------|
| H01  | G05       | Sharding ShardingTab → axis assignments → train     |
| H02  | G07       | TopBar fp8/mixed_precision toggles → train          |
| H03  | G09       | TopBar Cancel button + WS abort RPC + cancelled status row |
| H04  | G10       | TopBar train-warm-start checkbox → opts.continue_from_run_id |
| H05  | G12       | TopBar checkpoint-save/load path inputs → opts      |
| H06  | G15       | UI Train N=100 walk + e2e on real-corpus            |
| H07  | G19       | App dispatches DimensionsTab Apply → re-verify loop |
| H08  | G20       | TopBar train-probe-text textarea → opts.inference_probe_text |
| H09  | G11       | Save→Load roundtrip Train → extras parity e2e        |
| H10  | (V4-10)   | SideChannelsTab (proper) replacing dropdown checkboxes|

### B. Observation → math-effect upgrade (4 gaps)
Extras report numbers but no parity / recovery / forward-effect assertion.

| Gap  | V5 origin | Math claim to assert                                |
|------|-----------|-----------------------------------------------------|
| H11  | G06       | memory_peak_bytes within 30% of UI MemoryBar estimate|
| H12  | G08       | mid-train WS drop → reconnect → state preserved e2e |
| H13  | G21       | UI walk through smoke pipeline asserting each stage extras |
| H14  | G18       | AblationsTab table shows full extras per row (not just final) |

### C. Real math-effect for V5 analytical proxies (5 gaps)
V5 returned `λ×proxy_norm` instead of running the kernel.

| Gap  | V5 origin | Real-math upgrade                                   |
|------|-----------|-----------------------------------------------------|
| H15  | G05       | stage_train simulates per-rank shard (loss parity)  |
| H16  | G07       | mlx dtype switching actually casts params           |
| H17  | G17       | doc_ids → real attention mask in attention forward  |
| H18  | G25       | MoE forward hook → routing entropy + load balance   |
| H19  | G12       | Strict identical-loss-continuation: save→load→losses[0]==saved[-1] within 1e-5 |

### D. Multi-device / advanced (5 gaps)
| Gap  | Description                                          |
|------|------------------------------------------------------|
| H20  | Multi-device distributed train smoke (2 fake ranks)  |
| H21  | Inference-after-resume generation drift bounded      |
| H22  | Concurrent Train clicks → second click queued or rejected |
| H23  | fp16 dtype option alongside bf16/fp8                 |
| H24  | 100+ step UI walk → checkpoint → resume → continuation|

### E. Meta-honesty (1 gap)
| Gap  | Description                                          |
|------|------------------------------------------------------|
| H25  | Honest categorisation audit across V3+V4+V5 tests — every test labelled 🟢 math-effect / 🟡 propagation / 🔴 decorative. Target 0 🔴, no propagation in math-claimed surfaces. |

## 2. Goal

After v6: every V5 backend with a math claim has a UI surface that lets
the user trigger it AND an e2e test that walks UI→backend→numerical
assertion. Zero "backend pytest works but UI can't reach it" tests.

## 3. Stages — 25 H-gaps × 6-10 sub-tasks ≈ 200 sub-tickets

```
cppmega-mlx-v6 (epic)
├── H01-H10 UI wiring of V5 (10 × 6-8)
├── H11-H14 obs→math upgrade (4 × 7-9)
├── H15-H19 real-math for proxies (5 × 8-10)
├── H20-H24 advanced (5 × 8-10)
└── H25 meta-honesty (1 × 6)

Total: ~190 sub-tickets
```

## 4. Per-gap acceptance + sub-task breakdown

### H01 — Sharding ShardingTab → train (UI wiring V5-G05, 8)
- H01.1 ShardingTab accept proposal updates spec.sharding.axis_assignments
- H01.2 App.handleRunPipeline forwards via stage_options (verify already wired)
- H01.3 Topology-axis degree mismatch UI warning
- H01.4 Train button respects sharding-incompatible config (disabled+reason)
- H01.5 e2e: accept proposal → Train → extras.sharding_applied matches selection
- H01.6 e2e: change topology mid-canvas → re-verify → new axes shown
- H01.7 vitest ShardingTab dispatch
- H01.8 Closure regression: G05 e2e tightened (compile_mode propagation)

### H02 — fp8/mixed_precision UI → train (UI wiring V5-G07, 7)
- H02.1 TopBar `top-bar-mixed-precision` checkbox dispatches optim mutation
- H02.2 TopBar `top-bar-fp8-enabled` checkbox dispatches sharding mutation
- H02.3 App wire to spec.optim.mixed_precision / spec.sharding.fp8_enabled
- H02.4 e2e: toggle mixed_precision → extras.master_dtype changes
- H02.5 e2e: toggle fp8 → extras.fp8_active=true
- H02.6 vitest TopBar checkboxes
- H02.7 Honest doc: G07 propagation → UI-driven

### H03 — Cancel button + WS abort RPC (UI wiring V5-G09, 8)
- H03.1 TopBar Cancel button (`run-pipeline-cancel`) visible while train running
- H03.2 App tracks running run_id; click → rpc.call("pipeline.abort", {run_id})
- H03.3 Backend JSON-RPC method `pipeline.abort` → request_abort(token)
- H03.4 Modal shows "cancelled" status row with partial losses
- H03.5 e2e: start N=64 → click Cancel → modal shows partial losses
- H03.6 e2e: Cancel disabled when no train running
- H03.7 vitest Cancel button
- H03.8 Closure: G09 backend → UI-driven cancel

### H04 — train-warm-start checkbox (UI wiring V5-G10, 6)
- H04.1 TopBar `train-warm-start` checkbox + previousRunId state
- H04.2 App forwards continue_from_run_id when checked
- H04.3 e2e: enable warm-start, run twice, second extras.opt_state_carried=true
- H04.4 e2e: second-run losses[0] < first-run losses[0]
- H04.5 vitest TopBar checkbox
- H04.6 Honest doc: G10 backend → UI-driven

### H05 — checkpoint save/load path inputs (UI wiring V5-G12, 8)
- H05.1 TopBar `train-checkpoint-save-path` text input
- H05.2 TopBar `train-checkpoint-load-path` text input
- H05.3 App forwards via stage_options.train
- H05.4 e2e: set save-path → Train → extras.checkpoint.saved_path matches
- H05.5 e2e: set load-path → Train → extras.checkpoint.loaded_path matches
- H05.6 e2e: round-trip save → fresh page → load → Train → losses[0] within 1e-3 of saved[-1]
- H05.7 vitest TopBar inputs
- H05.8 Closure: G12 + H19 strict continuation

### H06 — UI N=100 real-corpus walk (UI wiring V5-G15, 7)
- H06.1 e2e: load real parquet+tokenizer, set N=100, Train (90s budget)
- H06.2 Assert extras.num_steps==100
- H06.3 Assert losses_smoothed monotone-window
- H06.4 Assert weight_delta_norm > 0.01
- H06.5 Assert inference_probe.l2_diff > 0.1 (100 steps significant)
- H06.6 Perf gate: completes within 120s on Apple Silicon
- H06.7 Mark long-running tag for CI selective run

### H07 — DimensionsTab Apply dispatches spec mutation (UI wiring V5-G19, 7)
- H07.1 App.tsx onApply handler maps entry → spec action
- H07.2 Dispatch sets brick param to suggested value
- H07.3 Re-verify removes the now-fulfilled suggestion
- H07.4 e2e: load preset → Apply → row disappears
- H07.5 e2e: applied suggestion reaches Train via extras.model_summary
- H07.6 vitest App.onApply mapping
- H07.7 Honest doc: G19 button → full feedback loop

### H08 — train-probe-text textarea (UI wiring V5-G20, 6)
- H08.1 TopBar `train-probe-text` textarea in train dropdown
- H08.2 App forwards via stage_options.train.inference_probe_text
- H08.3 e2e: provide "hello world", Train → extras.inference_probe.real_tokens=true
- H08.4 e2e: extras.inference_probe.text_len > 0
- H08.5 e2e: top1_token_drift >= 0
- H08.6 vitest TopBar textarea

### H09 — Save/Load roundtrip extras parity (UI wiring V5-G11, 8)
- H09.1 e2e: build spec → Save → capture Blob
- H09.2 Refresh page → Load same Blob
- H09.3 Train → capture extras_after_load
- H09.4 Build identical spec from scratch → Train → extras_baseline
- H09.5 Assert model_summary equal between two
- H09.6 Assert losses match within 1e-4
- H09.7 vitest serialiser + parser
- H09.8 Closure: G11 → identical-extras round-trip proven

### H10 — Proper SideChannelsTab (UI wiring V4-10/V5-G17, 9)
- H10.1 New sidebar tab `sidebar-tab-side-channels`
- H10.2 Per-family toggle + per-family data preview
- H10.3 Replace train-dropdown checkboxes
- H10.4 App wires via stage_options.train.side_channels
- H10.5 e2e: enable doc_ids in SideChannelsTab → Train → forward_effect populated
- H10.6 e2e: SideChannelsTab persists across tab switches
- H10.7 vitest SideChannelsTab toggles
- H10.8 Honest doc: V4-10 minimum-viable → proper tab
- H10.9 Closure: replace 24_side_channels.spec.ts with tab-driven version

### H11 — Memory peak vs estimate parity (obs→math V5-G06, 8)
- H11.1 verify RPC returns `estimated_peak_bytes` (separate from per-rank)
- H11.2 UI MemoryBar shows both estimate + actual after Train
- H11.3 vitest MemoryBar dual display
- H11.4 e2e: Train → assert |actual - estimate| / estimate < 0.5 (within 50%)
- H11.5 pytest: estimate accuracy on llama3_8b (tighter 30%)
- H11.6 testid `memory-bar-actual`, `memory-bar-estimate`
- H11.7 honest categorisation
- H11.8 closure regression

### H12 — WS reconnect mid-train recovery (obs→math V5-G08, 9)
- H12.1 e2e: start N=32 train
- H12.2 page.context().setOffline(true) mid-train
- H12.3 Wait 2s → setOffline(false)
- H12.4 Assert BottomStrip shows reconnecting then connected
- H12.5 Assert modal eventually shows completed train
- H12.6 Assert no spurious error modal
- H12.7 Backend WS heartbeat per-step
- H12.8 vitest reconnect UI states
- H12.9 Closure: G08 testid → recovery e2e

### H13 — Other-stages extras through UI (obs→math V5-G21, 7)
- H13.1 e2e: run Smoke pipeline (12 stages)
- H13.2 Expand each stage row in modal
- H13.3 Assert dry_forward extras present (batch, seq_len, hidden, num_nodes)
- H13.4 Assert loss_smoke extras (loss_value, loss_finite)
- H13.5 Assert optimizer_smoke extras (optimizer_kind, num_groups)
- H13.6 Negative: skipped stages don't crash on expand
- H13.7 Closure: G21 pytest → UI-walk e2e

### H14 — AblationsTab full extras display (obs→math V5-G18, 8)
- H14.1 AblationsTab row shows expand for full extras (like RunResultModal)
- H14.2 vitest expand renders extras subtree
- H14.3 e2e: run ablation → expand a variant row → full extras visible
- H14.4 Assert per-row losses array displayed
- H14.5 Assert per-row model_summary displayed
- H14.6 Strict parity: sequential UI Train vs ablation variant — same extras shape
- H14.7 Honest doc: G18 structural → user-visible
- H14.8 Closure regression

### H15 — Real per-rank shard simulation (real-math V5-G05, 9)
- H15.1 Backend: when sharding_applied has FSDP axis, simulate per-rank
  param shard for memory accounting
- H15.2 extras.sharding_applied gets `per_rank_param_bytes`
- H15.3 pytest: 8-way FSDP halves per-rank vs unsharded
- H15.4 pytest: loss bit-identical between sharded and unsharded
- H15.5 Strict tolerance < 1e-6
- H15.6 e2e: UI accept fsdp2 → per_rank_param_bytes populated
- H15.7 e2e: change to fsdp1 → per_rank_param_bytes differs
- H15.8 Honest doc: G05 propagation → real simulation
- H15.9 Closure regression

### H16 — Real mlx dtype switching (real-math V5-G07, 10)
- H16.1 Backend: when master_dtype="fp32", instantiate params in fp32
- H16.2 Backend: when train_dtype="fp16", cast forward inputs
- H16.3 Backend: when fp8_active=true, attempt fp8 (may fallback w/ reason)
- H16.4 extras.dtype_actual reports what mlx actually used
- H16.5 pytest: fp32 master + bf16 train produces different param norms vs all-bf16
- H16.6 pytest: fp16 train loss within tolerance of bf16
- H16.7 e2e: toggle precision → extras.dtype_actual reflects
- H16.8 Fallback path: fp8 unsupported → graceful + reason
- H16.9 Honest doc: G07 string-echo → real cast
- H16.10 Closure regression

### H17 — Real cross-doc attention mask (real-math V5-G17, 10)
- H17.1 Backend: _build_attention accepts doc_attention_mask param
- H17.2 stage_train.forward injects mask when doc_ids in side_channels
- H17.3 Mask: positions with different doc_ids → -inf in attention scores
- H17.4 extras.side_channels_forward_effect.doc_mask_applied = bool
- H17.5 pytest: with doc_ids vs without → losses differ > 1e-4
- H17.6 pytest: single-doc (all doc_ids equal) reduces to no-mask
- H17.7 e2e: UI enable doc_ids → losses differ from disabled
- H17.8 Token_ids: real conditional embed addition
- H17.9 Honest doc: G17 proxy → real kernel
- H17.10 Closure regression

### H18 — MoE forward hook (real-math V5-G25, 10)
- H18.1 Backend: when MoE brick present, wrap forward with routing hook
- H18.2 Capture routing weights per token (top-k softmax)
- H18.3 Compute routing_entropy = mean(-Σ p log p) over tokens
- H18.4 Compute load_balance_loss = Σ (expert_load - 1/num_experts)²
- H18.5 Compute dropped_token_ratio when capacity factor < 1
- H18.6 extras.moe populated with real values (not nulls)
- H18.7 pytest: gpt_oss_20b → routing_entropy ∈ (0, log(num_experts))
- H18.8 pytest: top_k=1 → routing_entropy < top_k=2
- H18.9 e2e: change top_k via brick context → routing_entropy changes
- H18.10 Honest doc: G25 propagation → real measurement

### H19 — Identical-loss-continuation strict (real-math V5-G12, 9)
- H19.1 pytest: save checkpoint after N=4 → load → run 1 more step
- H19.2 Assert losses[0] of resumed == losses[3] of original within 1e-5
- H19.3 pytest: same with optimizer state carried (Adam moments)
- H19.4 e2e: full UI walk Train N=4 save → fresh page Load Train N=1
- H19.5 Assert losses[0] match (within 1e-3 since UI noise)
- H19.6 Backend: opt.state also saved separately (state.safetensors)
- H19.7 extras.checkpoint.opt_state_path
- H19.8 Negative: missing opt-state file → warning + cold restart
- H19.9 Closure regression: G12 smoke → strict math

### H20 — Multi-device distributed smoke (advanced, 10)
- H20.1 Backend: stage_train accepts opts.fake_ranks=N
- H20.2 Forward + backward simulated on N fake ranks
- H20.3 Reduce gradients via mean
- H20.4 extras.fake_ranks + extras.gradient_reduce_ms
- H20.5 pytest: fake_ranks=2 vs 1 loss within 1e-6
- H20.6 pytest: fake_ranks=4 weight_delta similar magnitude
- H20.7 e2e: UI selects topology with degree=2 → fake_ranks=2 in extras
- H20.8 Sharding axis assignment respected per fake rank
- H20.9 Performance gate: fake_ranks=8 within 2× wall-clock vs 1
- H20.10 Closure: real distributed proxy

### H21 — Inference-after-resume bounded drift (advanced, 8)
- H21.1 Backend: extras.inference_probe.post_resume_drift = float
- H21.2 Save checkpoint after train; load fresh model; resume train 1 step
- H21.3 Compare inference probe between {train→save→reload} and {train continuous}
- H21.4 Assert drift < 1e-3 (proves checkpoint round-trip preserves weights bit-exact)
- H21.5 pytest end-to-end
- H21.6 e2e: UI save→fresh→resume→probe text → drift bounded
- H21.7 Negative: corrupt checkpoint → clear error not crash
- H21.8 Closure regression

### H22 — Concurrent Train clicks (advanced, 8)
- H22.1 App tracks trainInFlight state (one at a time)
- H22.2 Second click while running → queued indicator or rejected
- H22.3 TopBar Train button shows "Training..." while in-flight
- H22.4 e2e: click Train twice quickly → only one modal opens
- H22.5 e2e: second click after first completes → fresh run
- H22.6 vitest Train button state machine
- H22.7 Honest doc: lifecycle robust
- H22.8 Closure regression

### H23 — fp16 dtype option (advanced, 8)
- H23.1 Schema accepts train_dtype="fp16"
- H23.2 UI TopBar precision dropdown {bf16, fp16, fp32}
- H23.3 Backend casts forward inputs to fp16 when selected
- H23.4 extras.train_dtype reports
- H23.5 pytest: fp16 loss finite + within 1e-3 of bf16
- H23.6 pytest: fp16 weight_delta < fp32 weight_delta (less precision)
- H23.7 e2e: UI select fp16 → extras.train_dtype=="fp16"
- H23.8 Closure regression

### H24 — 100+ step UI walk with checkpoint (advanced, 9)
- H24.1 e2e: load preset + real corpus
- H24.2 Set N=100, set save-path
- H24.3 Train → modal completes
- H24.4 Fresh page → set load-path → set N=20
- H24.5 Train → assert losses[0] starts where saved[-1] ended
- H24.6 Assert continuation within 1e-3
- H24.7 Closure of G15 + G12 + H19 chain
- H24.8 Perf gate < 200s wall-clock
- H24.9 Mark long-running tag

### H25 — Honest categorisation audit (meta, 7)
- H25.1 Script audit_v3_v4_v5.py walks all tests/v3+v4+v5/ + vbgui/e2e/scenarios/
- H25.2 For each test extracts category (math-effect / propagation /
  decorative) from assertion patterns
- H25.3 Markdown report: per-test label + assertion summary
- H25.4 Target: 0 🔴 decorative, 🟡 propagation only on genuinely-config
  fields (e.g. project name)
- H25.5 vbgui-side counterpart for vitest
- H25.6 Closure report `tests/fixtures/honesty_audit_v6.md`
- H25.7 v7 follow-up list

## 5. Workflow per gap (same as v5)

1. `bd update <id> --claim`
2. Implement (each sub-task atomic)
3. Code review (self if <100 LOC, gsd-code-reviewer otherwise)
4. Perf check (no >5% regression on affected suite)
5. Regression (pytest + vitest + e2e affected)
6. `git add` specific files → commit → push → verify on origin/main
7. `bd close <id>` only after push verified

## 6. Acceptance counters

| Surface                                | Before v6 | Target v6 |
|----------------------------------------|-----------|-----------|
| pytest                                 | ~2400     | ≥2500     |
| vitest                                 | 189       | ≥220      |
| Playwright deep e2e cells              | 120+      | ≥200      |
| extras keys                            | 30        | ≥38       |
| backend-without-UI tests               | ~10       | 0         |
| 🟢 math-effect tests                   | ~25       | ≥55       |
| 🟡 propagation tests on math claims    | ~10       | 0         |
| 🔴 decorative tests                    | 0         | 0         |

## 7. Out of scope (defer to v7+)

- Real multi-mac distributed training (network protocol, gradient
  all-reduce, FSDP2 sharded forward)
- Production-scale (1B+ param) training matrix
- A/B hyperparameter search UI
- Anomaly detection on loss curves (NaN traps, gradient explosion)
- Live inference serving from trained checkpoint
- Model weight quantisation post-train
- LoRA / PEFT adapter wiring

## 8. Done definition

All 25 H-gaps closed. Zero "backend pytest but no UI walk" tests.
Closure report at `tests/fixtures/e2e_matrix_v6_report.md`. Honest
categorisation audit at `tests/fixtures/honesty_audit_v6.md`.
