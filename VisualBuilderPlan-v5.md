# VisualBuilderPlan-v5 — Math-Effect, Distributed, Lifecycle, Persistence

**Status**: planned 2026-05-22 (epic `cppmega-mlx-pjt`)
**Driver**: honest V4 closure audit found 25 remaining gaps. Most damning
class: **propagation-theatre tests** — V4-7 (LossKind), V4-8 (Rewriters),
V4-10 (side_channels) all assert that a UI string reaches `extras` as
the same string. They DO NOT assert that the underlying training math
changed. stage_train remains a 2-brick MLP+lm_head+CE loop regardless
of what the UI says about MTP, IFIM, MHC, rewriters, side-channels.

v5 closes the **string-echo → math-effect** gap, the lifecycle/persistence
holes left across v3+v4, the sharding/distributed dimension that has
never reached training, and tightens several weak assertions.

## 1. Why v5 exists

V4 closure audit produced 25 honest gaps in 9 groups. Each gap has a
**string-echo problem** (assertion proves only that UI text appeared
in extras), or a **decorative UI** (toggle exists but backend ignores
it), or a **missing surface** (UI has nothing to drive the assertion
from).

### A. Math-effect group (4 gaps)
`stage_train` still hardcodes `nn.losses.cross_entropy` + V4-1 optimizer
dispatch. UI selecting MTP/IFIM/MHC loss kinds or any rewriter produces
identical training math to plain CE + no-rewriter.

| Gap | Title                                                       |
|-----|-------------------------------------------------------------|
| G01 | LossKind MTP_WEIGHTED actually runs K-head weighted loss    |
| G02 | LossKind IFIM_SHAPED actually applies λ_fim shaping         |
| G03 | LossKind MHC_ATTN_BIAS actually applies λ_mhc attn bias     |
| G04 | RewritersTab apply_rewrites mutates the graph in stage_train|

### B. Distributed / precision group (3 gaps)
| Gap | Title                                                       |
|-----|-------------------------------------------------------------|
| G05 | Sharding axis assignments reach stage_train                 |
| G06 | Memory bar estimate vs real `mx.metal.get_peak_memory`      |
| G07 | fp8 / mixed_precision toggles affect train dtype + extras   |

### C. Lifecycle group (3 gaps)
| Gap | Title                                                       |
|-----|-------------------------------------------------------------|
| G08 | WS reconnect mid-train preserves state                      |
| G09 | Cancel/abort Train + abort surface                          |
| G10 | Optimizer state warm-start across sequential Train clicks   |

### D. Persistence group (2 gaps)
| Gap | Title                                                       |
|-----|-------------------------------------------------------------|
| G11 | Spec save/load UI + identical train extras roundtrip        |
| G12 | Checkpoint save/resume produces identical loss continuation |

### E. Coverage-strictness group (4 gaps)
| Gap | Title                                                       |
|-----|-------------------------------------------------------------|
| G13 | V4-3 strict assertion that user-picked tokenizer was used   |
| G14 | V4-4 convergence floor tightened (5% loss reduction, N=16+) |
| G15 | Real-corpus convergence at 100+ steps                       |
| G16 | Cross-arch coverage for 9 non-canonical-named presets       |

### F. Side-channel + UX loop group (3 gaps)
| Gap | Title                                                       |
|-----|-------------------------------------------------------------|
| G17 | side_channels reach forward (attention bias / cond / mask)  |
| G18 | AblationsTab.run vs sequential UI Train parity              |
| G19 | DimensionsTab inference_log click-to-apply feedback loop    |

### G. Inference + observability group (2 gaps)
| Gap | Title                                                       |
|-----|-------------------------------------------------------------|
| G20 | Inference probe over real tokens (encode text → drift)      |
| G21 | dry_forward / input_parity / loss_smoke / optimizer_smoke rich extras |

### H. Optimizer + train hyperparam group (3 gaps)
| Gap | Title                                                       |
|-----|-------------------------------------------------------------|
| G22 | muon_adamw_hybrid update delta divergence per group         |
| G23 | gradient_clip_norm activation observable in extras          |
| G24 | Tokenizer roundtrip OK proves byte-exact decode             |

### I. Architecture-coverage group (1 gap)
| Gap | Title                                                       |
|-----|-------------------------------------------------------------|
| G25 | MoE experts routing observable in train (top-k, load balance)|

## 2. Goal

Every UI surface that claims to mutate training behavior must produce
either (a) a numerically-verifiable difference in extras (loss values,
weight deltas, lr trajectories, graph diff) OR (b) an observable
divergence in inference output. No more string-echo tests.

## 3. Stages — 25 gaps × 6-10 sub-tasks ≈ 200 sub-tickets

The full bd epic structure is:

```
cppmega-mlx-v5 (epic)
├── G01 LossKind MTP (10 sub-tickets)
├── G02 LossKind IFIM (8 sub-tickets)
├── G03 LossKind MHC (8 sub-tickets)
├── G04 Rewriters apply (10 sub-tickets)
├── G05 Sharding effect (8 sub-tickets)
├── G06 Memory peak parity (7 sub-tickets)
├── G07 fp8/mixed precision (8 sub-tickets)
├── G08 WS reconnect (7 sub-tickets)
├── G09 Cancel/abort Train (8 sub-tickets)
├── G10 Optimizer warm-start (6 sub-tickets)
├── G11 Spec save/load (9 sub-tickets)
├── G12 Checkpoint save/resume (10 sub-tickets)
├── G13 Tokenizer strict assertion (6 sub-tickets)
├── G14 V4-4 strict floor (6 sub-tickets)
├── G15 Real-corpus 100+ steps (8 sub-tickets)
├── G16 9 non-canonical presets (7 sub-tickets)
├── G17 side_channels forward routing (10 sub-tickets)
├── G18 Ablation vs sequential parity (8 sub-tickets)
├── G19 DimensionsTab feedback loop (8 sub-tickets)
├── G20 Real-tokens inference probe (8 sub-tickets)
├── G21 Other-stages rich extras (8 sub-tickets)
├── G22 hybrid update delta (7 sub-tickets)
├── G23 gradient_clip_norm extras (6 sub-tickets)
├── G24 Tokenizer roundtrip byte-exact (6 sub-tickets)
└── G25 MoE experts routing (10 sub-tickets)

Total: ~196 sub-tickets
```

## 4. Per-gap acceptance + sub-task breakdown

### G01 — LossKind MTP_WEIGHTED runs K-head weighted loss (HIGH, 10)

**Why**: V4-7 proved `extras.loss_kind === "mtp_weighted"` but train math
was always CE. UI selecting MTP changes a string; the loss curve, head
materialisation, and beta weighting all stayed identical to plain CE.

**Acceptance**: `extras.mtp` block reports K, per-head losses, beta
weights. Loss value equals `Σ beta_i × CE(head_i, shifted_labels_i)`.
Setting K=1 reduces to plain CE within float tolerance.

**Sub-tasks**:
- G01.1 Backend: stage_train detects `spec.loss.kind == MTP_WEIGHTED`
  and switches loss kernel to `_mtp_weighted_loss(outputs, labels, k, betas)`
- G01.2 Backend: shifted-labels generator `_shift_labels(labels, k_offset)`
- G01.3 Backend: K extra LM heads materialised in train forward (each
  taking the same residual stream output, separate Linear → vocab)
- G01.4 Backend: extras.mtp = {k, betas, per_head_losses: [float]}
- G01.5 Backend pytest: K=1 / K=2 / K=3 loss math vs hand-computed
- G01.6 Backend pytest: betas sum to 1 invariant respected
- G01.7 UI: LossTab.tsx exposes per-i beta inputs (or auto-broadcast from
  single beta, mirroring `_make_loss` translator)
- G01.8 E2E: UI MTP K=2 beta=0.6 → extras.mtp.k===2, per_head_losses
  length 2, abs(sum(betas) - 1) < 1e-6
- G01.9 E2E: K=1 produces same loss within 1e-4 as plain CE on same seed
- G01.10 E2E: changing beta from 0.6 to 0.3 changes total loss
  observably

### G02 — LossKind IFIM_SHAPED applies λ_fim shaping (MEDIUM, 8)

**Why**: V4-7 proves propagation only; backend doesn't compute the
IFIM-shaped penalty term that re-weights tokens by their inverse Fisher
information mass.

**Acceptance**: `extras.ifim = {lambda_fim, fim_weights_norm}` populated.
Loss differs from plain CE by `λ_fim × ifim_penalty`.

**Sub-tasks**:
- G02.1 Backend: IFIM penalty term `_ifim_shaped_loss(outputs, labels, lambda_fim)`
- G02.2 Backend: extras.ifim = {lambda_fim, fim_weights_norm, penalty_value}
- G02.3 Backend pytest: λ_fim=0 reduces to CE within 1e-6
- G02.4 Backend pytest: λ_fim=0.5 increases loss vs CE on same seed
- G02.5 UI: LossTab λ_fim input already exists; wire end-to-end
- G02.6 E2E: UI IFIM λ_fim=0.1 → extras.ifim.lambda_fim === 0.1
- G02.7 E2E: λ_fim=0 vs λ_fim=0.5 produces different losses[0]
- G02.8 Closure regression: V4-7 IFIM scenario upgraded to math-effect

### G03 — LossKind MHC_ATTN_BIAS applies λ_mhc bias (MEDIUM, 8)

**Why**: same as G02 for MHC.

**Sub-tasks**:
- G03.1 Backend: MHC bias term `_mhc_attn_bias_loss(outputs, labels, lambda_mhc)`
- G03.2 Backend: extras.mhc = {lambda_mhc, bias_norm}
- G03.3 Backend pytest: λ_mhc=0 reduces to CE
- G03.4 Backend pytest: λ_mhc=0.05 vs 0 changes loss
- G03.5 UI: LossTab λ_mhc input exists; verify wire
- G03.6 E2E: UI MHC λ_mhc=0.05 → extras.mhc.lambda_mhc === 0.05
- G03.7 E2E: λ_mhc=0 vs 0.05 produces different loss
- G03.8 Closure regression: V4-7 MHC scenario upgraded

### G04 — Rewriters apply_rewrites mutates graph in stage_train (HIGH, 10)

**Why**: V4-8 proved `extras.model_summary.rewriters_applied` contains
rewriter names but stage_train doesn't call apply_rewrites. Spec graph
remains untouched between user's "Apply chain" click and train run.

**Acceptance**: `extras.graph_diff = {added, removed, renamed}` populated
when rewriters present. MTPRewriter actually adds K head nodes;
IFIMRewriter adds aux node; MHCRewriter adds copy nodes.

**Sub-tasks**:
- G04.1 Backend: stage_train calls `apply_rewrites(build_spec)` before
  re-materialising graph
- G04.2 Backend: snapshot graph node ids before + after; compute diff
- G04.3 Backend: extras.graph_diff = {added: [str], removed: [str],
  renamed: [(old, new)]}, extras.model_summary.num_brick_kinds_after
- G04.4 Backend pytest: MTPRewriter adds K head nodes
- G04.5 Backend pytest: IFIMRewriter adds 1 aux node
- G04.6 Backend pytest: MHCRewriter adds copy nodes per N
- G04.7 Backend pytest: composition (MTP→IFIM→MHC) produces stacked diff
- G04.8 Backend pytest: idempotency — twice-applied rewriter has same diff
- G04.9 E2E: UI add MTPRewriter → extras.graph_diff.added.length === K
- G04.10 E2E: UI remove rewriter → graph_diff empty

### G05 — Sharding axis assignments reach stage_train (MEDIUM, 8)

**Why**: UI ShardingTab/TopBar selections never affect training math.
stage_train ignores spec.sharding entirely.

**Acceptance**: `extras.sharding_applied = {axis_assignments,
shard_dim, microbatch_size}` reports what was used. Different sharding
(none vs fsdp2 vs TP) produces same loss within tolerance (correctness
preserved) but different memory footprint.

**Sub-tasks**:
- G05.1 Backend: stage_train reads spec.sharding; if non-trivial,
  records axis_assignments + per-axis degree in extras
- G05.2 Backend: synthetic shard simulation (param.shape divided per axis)
- G05.3 Backend: extras.sharding_applied populated
- G05.4 Backend pytest: fsdp2 vs none on 2-brick model — same loss, half
  param memory per rank
- G05.5 Backend pytest: TP shards weight matrices correctly
- G05.6 E2E: UI accept sharding proposal → extras.sharding_applied
  reflects chosen strategy
- G05.7 E2E: change topology between runs → extras differ
- G05.8 Closure: V3-11/V4-12 cross-arch tests extended w/ sharding axis

### G06 — Memory peak vs estimate parity (MEDIUM, 7)

**Why**: UI MemoryBar shows estimate; real `mx.metal.get_peak_memory()`
never compared. Estimate could be wildly off; user trusts a number that
doesn't track reality.

**Sub-tasks**:
- G06.1 Backend: stage_train brackets train loop with mx.metal.reset_peak
  + get_peak_memory; reports `extras.memory_peak_bytes`
- G06.2 Backend: verify RPC returns `estimated_peak_bytes` (separate from
  per-rank memory model)
- G06.3 Backend pytest: estimate within 30% of real peak on llama3_8b
- G06.4 UI: MemoryBar shows both estimate + actual after Train
- G06.5 UI: testid `memory-bar-estimate` and `memory-bar-actual`
- G06.6 E2E: run Train → memory-bar-actual populated, value within 30%
  of memory-bar-estimate
- G06.7 E2E: change hidden_size → estimate updates proportionally

### G07 — fp8 / mixed_precision toggles affect train (MEDIUM, 8)

**Why**: OptimSpec has `mixed_precision` flag, ShardingSpec has
`fp8_enabled`. Train always uses bf16 defaults regardless.

**Sub-tasks**:
- G07.1 Backend: stage_train reads spec.optim.mixed_precision; when False
  forces fp32 master weights
- G07.2 Backend: stage_train reads spec.sharding.fp8_enabled; when True
  casts matmul inputs to fp8 (when supported)
- G07.3 Backend: extras.train_dtype, extras.master_dtype, extras.fp8_active
- G07.4 Backend pytest: mixed_precision=False → master_dtype="fp32",
  loss within 1e-5 of bf16 mixed path
- G07.5 Backend pytest: fp8_enabled=True → fp8_active reported (or
  reason for skipping)
- G07.6 UI: OptimTab + TopBar expose toggle testids
- G07.7 E2E: toggle mixed_precision → extras.master_dtype reflects
- G07.8 E2E: toggle fp8 → extras.fp8_active reflects

### G08 — WS reconnect mid-train preserves state (MEDIUM, 7)

**Why**: useRpc has WS reconnect handler but mid-train disconnect
behaviour is untested. Lost connection during stage_train should not
leave UI in inconsistent state.

**Sub-tasks**:
- G08.1 Backend: stage_train emits progress via WS heartbeat
  (step/total) — already partially exists
- G08.2 Frontend: BottomStrip shows "reconnecting..." during WS gap
- G08.3 Frontend: Train modal disabled-close during in-flight train
- G08.4 Backend: pipeline.run idempotent on retry (cache by spec hash)
- G08.5 Backend pytest: kill WS mid-call, retry, get same result
- G08.6 E2E: page.context().setOffline(true) mid-train, restore,
  modal still shows progress
- G08.7 E2E: backend status indicator shows "reconnecting" then
  "connected"

### G09 — Cancel/abort Train surface + handle (MEDIUM, 8)

**Sub-tasks**:
- G09.1 Backend: pipeline.run accepts opts.abort_token; stage_train
  checks token between steps
- G09.2 Backend: WS message "pipeline.abort" with run_id cancels
- G09.3 Frontend: TopBar shows Cancel button while train running
  (data-testid=run-pipeline-cancel)
- G09.4 Frontend: handleAbort sends pipeline.abort RPC
- G09.5 Backend pytest: abort_token mid-loop terminates cleanly,
  returns partial extras
- G09.6 E2E: start Train N=64, click Cancel after modal opens,
  modal shows status='cancelled', partial losses[]
- G09.7 E2E: Cancel disabled when no train running
- G09.8 E2E: cancelled train doesn't pollute next Train

### G10 — Optimizer state warm-start across sequential runs (LOW, 6)

**Sub-tasks**:
- G10.1 Backend: stage_train accepts opts.continue_from_run_id; loads
  cached opt.state from that run
- G10.2 Backend: runs cached in LRU (size 8) by run_id
- G10.3 Backend: extras.opt_state_carried = bool
- G10.4 Frontend: TopBar checkbox train-warm-start
- G10.5 Backend pytest: two sequential runs with carry → opt.state.step
  equals 2N
- G10.6 E2E: enable warm-start, run twice, second run's losses[0] <
  first run's losses[0] (Adam learning rate adaptation kicked in)

### G11 — Spec save/load roundtrip (MEDIUM, 9)

**Why**: V3-12 deferred. UI has no Save/Load. Cannot prove that
exported+imported spec produces identical train extras.

**Sub-tasks**:
- G11.1 Frontend: TopBar Save button → serialises spec to JSON
  Blob, downloads
- G11.2 Frontend: TopBar Load button → file input, parses, dispatches
  spec replacement actions
- G11.3 Frontend: testids spec-save, spec-load, spec-load-input
- G11.4 Frontend: localStorage persistence — last-loaded spec key
- G11.5 vitest: Save serialiser produces valid JSON matching wire
  schema
- G11.6 vitest: Load parser handles malformed JSON gracefully
- G11.7 E2E: build spec, Save (capture Blob), Reload page, Load
  same Blob → spec restored
- G11.8 E2E: Save spec, Load, Train → extras.model_summary matches
  pre-save train extras
- G11.9 E2E: localStorage restoration on fresh page load

### G12 — Checkpoint save/resume identical loss continuation (HIGH, 10)

**Why**: stage_train doesn't write weights to disk. Cannot resume.
This is the M0.7 acceptance criterion.

**Sub-tasks**:
- G12.1 Backend: stage_train accepts opts.checkpoint_save_path;
  writes safetensors after final step
- G12.2 Backend: stage_train accepts opts.checkpoint_load_path;
  loads safetensors before train
- G12.3 Backend: opt.state also serialised (separate file)
- G12.4 Backend: extras.checkpoint = {saved_path, loaded_path,
  param_count, state_count}
- G12.5 Backend pytest: save → reload → losses[0] matches saved
  losses[N-1] within 1e-5
- G12.6 Backend pytest: checkpoint missing keys → clear error
- G12.7 Frontend: TopBar Save/Resume controls in train dropdown
- G12.8 vitest: save/resume UI flow
- G12.9 E2E: Train N=4 with save, fresh page, Resume Train N=4,
  losses[0] matches saved losses[3] within 1e-3
- G12.10 E2E: resume-without-save shows clear error

### G13 — V4-3 strict tokenizer assertion (LOW, 6)

**Sub-tasks**:
- G13.1 V4-3 spec asserted `data_source ∈ {parquet, parquet_tokenized}`
  too loose
- G13.2 Add fixture parquet with explicit 'text' column (no input_ids)
- G13.3 Update V4-3 e2e to use the text-only fixture
- G13.4 E2E asserts `data_source === "parquet_tokenized"` strictly
- G13.5 E2E asserts `tokenizer_used === "<expected basename>"`
- G13.6 Regression: 29_tokenizer_train_threading.spec.ts strengthened

### G14 — V4-4 strict convergence floor (5%, N=16) (LOW, 6)

**Sub-tasks**:
- G14.1 V4-4 currently `last<first OR tailAvg<headAvg` — too weak
- G14.2 Upgrade to N=16 steps
- G14.3 Strict: losses[15] < losses[0] * 0.95 (5% reduction)
- G14.4 Also: monotone window of last 4 strictly < first 4 average
- G14.5 Regression: 18_real_data_convergence.spec.ts tightened
- G14.6 Closure: V4-4 redacted from "weak" list

### G15 — Real-corpus convergence 100+ steps (MEDIUM, 8)

**Why**: V4-4 is 8 steps; production training is millions. 100+ is the
threshold where loss curves stabilise enough for predictive assertions.

**Sub-tasks**:
- G15.1 Backend: extra-long pipeline `train_long_run` stage variant
  that runs N=200 + checkpoints every 50
- G15.2 Backend: extras.loss_smoothed (EMA window=10)
- G15.3 Frontend: train-num-steps input max=512
- G15.4 Backend pytest: N=100 on llama3_8b → loss decreases ≥ 20%
- G15.5 E2E: real-data N=200 → loss_smoothed[199] < loss_smoothed[10]*0.7
- G15.6 E2E: lr_trajectory under cosine schedule matches analytical
  shape across N=200
- G15.7 E2E: weight_delta_norm grows monotonically
- G15.8 Perf gate: N=200 finishes within 90s on M3 Ultra

### G16 — Cross-arch 9 non-canonical presets (LOW, 7)

**Sub-tasks**:
- G16.1 Build helper `findBrickByKind(page, preset, kind)` that walks
  brick-node-* testids and picks first matching kind
- G16.2 Replace hardcoded `brick-context-{preset}_mlp` with runtime
  introspection
- G16.3 Cover glm_45/glm_45_air/glm_47 (shared bricks)
- G16.4 Cover qwen3_dense_0_6b/qwen3_dense_32b (qwen3_*_mlp prefix)
- G16.5 Cover intellect_3, granite_4_1 variants
- G16.6 Extend `26_cross_arch_brick_mutations.spec.ts` to 19 presets
- G16.7 Closure: V4-12 "10 of 19" gap closed → "all 19"

### G17 — side_channels reach forward (HIGH, 10)

**Why**: V4-10 records observation only. Backend never routes doc_ids
into forward pass (no attention bias / conditioning / loss mask).

**Sub-tasks**:
- G17.1 Backend: stage_train forward_layers signature extended with
  `side_channels: dict | None`
- G17.2 Backend: when "doc_ids" present, applies cross-document
  attention mask (no attend across doc boundaries)
- G17.3 Backend: when "token_ids" present, applies as conditional embed
- G17.4 Backend: extras.side_channels_forward_effect = {
    "doc_ids_mask_density": float,
    "token_ids_added_norm": float}
- G17.5 Backend pytest: doc_ids enabled changes loss vs disabled on
  same seed
- G17.6 Backend pytest: token_ids enabled shifts model_summary
- G17.7 UI: SideChannelsTab with per-channel toggle + data preview
  (replaces train-dropdown checkboxes)
- G17.8 E2E: enable doc_ids → extras.side_channels_forward_effect
  populated AND losses differ from disabled run
- G17.9 E2E: token_ids effect verified
- G17.10 Closure: V4-10 "observation only" → "forward routed"

### G18 — AblationsTab vs sequential UI parity (MEDIUM, 8)

**Sub-tasks**:
- G18.1 Backend: ablation.run reuses exact same stage_train code path
  as pipeline.run (no shortcuts)
- G18.2 Backend: per-variant extras identical between routes
- G18.3 Frontend: AblationsTab shows full extras per row (not just final)
- G18.4 vitest: ablation row matches RunResultModal row schema
- G18.5 E2E: run ablation activation [glu, swiglu]; then run two
  sequential UI Train cells with each activation
- G18.6 E2E: assert ablation.losses[v] == sequential.losses[v]
  within 1e-4 per variant
- G18.7 Closure: 13_ablation_math.spec.ts upgraded with parity check
- G18.8 Honest doc: orchestration parity verified

### G19 — DimensionsTab inference_log feedback loop (MEDIUM, 8)

**Sub-tasks**:
- G19.1 Frontend: DimensionsTab row has Apply button that dispatches
  spec mutation matching the suggestion (e.g., hidden_size override)
- G19.2 Frontend: testid dim-row-{i}-apply
- G19.3 vitest: Apply button fires correct dispatch
- G19.4 Backend: verify RPC returns inference_log entries after
  spec mutation (closed loop)
- G19.5 E2E: load preset → DimensionsTab shows N rows → click first
  Apply → re-verify → row disappears (suggestion fulfilled)
- G19.6 E2E: applied suggestion reaches Train via extras.model_summary
- G19.7 Honest doc: DimensionsTab is now interactive, not decorative
- G19.8 Coverage: V4-12 cross-arch with DimensionsTab mutation

### G20 — Real-tokens inference probe (MEDIUM, 8)

**Why**: V4-11 uses random Gaussian probe input. Real test: encode a
text prompt with the same tokenizer used in train, run forward, compare
to pre-train forward.

**Sub-tasks**:
- G20.1 Backend: stage_train accepts opts.inference_probe_text; if set,
  encodes via tokenizer_path and uses as probe input
- G20.2 Backend: extras.inference_probe.real_tokens = bool,
  extras.inference_probe.text_len
- G20.3 Backend: extras.inference_probe.top1_token_drift = int (how
  many top-1 next-token predictions changed)
- G20.4 Backend pytest: real-tokens probe runs end-to-end
- G20.5 Frontend: TopBar input train-probe-text (optional)
- G20.6 E2E: provide text "the quick brown fox", Train, assert
  inference_probe.real_tokens === true AND top1_token_drift > 0
- G20.7 E2E: empty text → falls back to random Gaussian (V4-11)
- G20.8 Closure: V4-11 strict-tokens path proven

### G21 — Other stages rich extras (MEDIUM, 8)

**Why**: only stage_train has rich extras. dry_forward / input_parity /
loss_smoke / optimizer_smoke return barebones results, hidden in modal.

**Sub-tasks**:
- G21.1 Backend: dry_forward extras gains output_shapes, output_dtype,
  finite_check
- G21.2 Backend: input_parity_check extras gains residual_norm,
  layer_max_diff per brick
- G21.3 Backend: loss_smoke extras gains loss_value, loss_finite,
  grad_finite
- G21.4 Backend: optimizer_smoke extras gains param_delta_norm,
  optimizer_step_count
- G21.5 vitest: ExtrasEntry renders each new field
- G21.6 E2E: run smoke pipeline → each stage's expand button works
- G21.7 E2E: assert per-stage extras keys exist
- G21.8 Closure: pipeline observability complete across all 12 stages

### G22 — muon_adamw_hybrid update delta divergence (MEDIUM, 7)

**Sub-tasks**:
- G22.1 Backend: when hybrid, snapshot delta per-bucket separately
- G22.2 Backend: extras.hybrid_deltas = {muon_norm, adamw_norm,
  ratio: muon/adamw}
- G22.3 Backend pytest: ratio differs from 1.0 by >5% (proves
  different update rules)
- G22.4 Backend pytest: pure-AdamW comparison sanity check
- G22.5 E2E: select hybrid → extras.hybrid_deltas populated, ratio
  ≠ 1
- G22.6 E2E: same spec with pure adamw → no hybrid_deltas key
- G22.7 Closure: V4-9 "count only" → "delta math verified"

### G23 — gradient_clip_norm extras + assertion (LOW, 6)

**Sub-tasks**:
- G23.1 Backend: stage_train applies clip_grad_norm if
  spec.optim.gradient_clip_norm set
- G23.2 Backend: extras.gradient_clip = {threshold, max_grad_norm_seen,
  num_clips}
- G23.3 Backend pytest: clip=1.0 with grad_norm=10 → num_clips==1,
  effective grad ≤ 1.0
- G23.4 UI: OptimTab gradient_clip_norm input already exists; wire
- G23.5 E2E: set clip=0.5, Train → extras.gradient_clip.num_clips > 0
- G23.6 E2E: clip=None vs clip=0.5 → different weight_delta_norm

### G24 — Tokenizer roundtrip OK byte-exact (LOW, 6)

**Sub-tasks**:
- G24.1 V3-10 only proved FAIL non-blocking; OK never asserted
  byte-exact
- G24.2 Add positive test: roundtrip on cppmega tokenizer + 4k corpus
  → 100% OK rows
- G24.3 Assert byte_diff == 0 per row, not just matches: true
- G24.4 Assert decoded_preview equals original text per row
- G24.5 17_roundtrip_warning.spec.ts extended with positive case
- G24.6 Closure: G18 in V3 audit closed strictly

### G25 — MoE experts routing observable in train (HIGH, 10)

**Why**: MoE/MoR/SparseExperts bricks exist in many presets (gpt_oss,
qwen3, kimi_k2, deepseek). stage_train uses generic mlp brick path;
routing decisions and load balance never tracked.

**Sub-tasks**:
- G25.1 Backend: detect MoE bricks in graph (kind in {moe, bailing_moe,
  sparse_moe})
- G25.2 Backend: extras.moe = {num_experts, top_k, routing_entropy,
  load_balance_loss, dropped_token_ratio}
- G25.3 Backend: per-expert load distribution
- G25.4 Backend pytest: gpt_oss_20b → 4+ experts reported
- G25.5 Backend pytest: load_balance_loss within expected range
- G25.6 UI: Brick context for MoE shows expert routing dropdown
- G25.7 UI: dispatch top_k change → re-verify → train math differs
- G25.8 E2E: gpt_oss_20b train → extras.moe.num_experts > 0
- G25.9 E2E: change top_k from 2 to 1 → routing_entropy decreases
- G25.10 Closure: MoE observability v1

## 5. Workflow per gap (Goal directive)

1. Claim epic gap ticket (G## top-level)
2. For each sub-task in order:
   a. `bd update <id> --claim`
   b. Implementer or Fixer
   c. Code Review (self if <100 LOC; gsd-code-reviewer otherwise)
   d. Perf Optimization (no >5% regression)
   e. Regression Tests (pytest + vitest + relevant playwright)
   f. `git add` → commit → push → verify on origin/main
   g. `bd close <sub-id>`
3. Only after ALL sub-tasks closed: close the gap ticket
4. Only after ALL 25 gaps closed: close epic

## 6. Acceptance counters

| Surface                                | Before v5 | Target v5 |
|----------------------------------------|-----------|-----------|
| pytest                                 | ~2360     | ≥2500     |
| vitest                                 | 177       | ≥210      |
| Playwright deep e2e cells              | 80+       | ≥180      |
| extras keys                            | 15        | ≥30       |
| "propagation-only" tests               | 11        | 0         |
| "math-effect" tests                    | 1 (V4-11) | ≥20       |
| Lifecycle / persistence tests          | 0         | ≥10       |
| Sharding / distributed tests           | 0         | ≥5        |
| MoE / expert-routing tests             | 0         | ≥4        |

## 7. Out of scope (v6+ candidates)

- Multi-Mac distributed train (real network)
- bf16 vs fp16 vs fp32 dtype matrix
- Hardware-specific tuning (M3 Ultra vs M4 Max)
- Production-scale (1B+ param) training matrix
- Live inference serving from trained checkpoint
- Anomaly detection on loss curves (NaN traps, gradient explosion alerts)
- A/B-tested hyperparameter search UI

## 8. bd ticket id mapping

| Gap  | bd id           | Sub-task count | Priority | Group           |
|------|-----------------|----------------|----------|-----------------|
| G01  | cppmega-mlx-iry | 10             | P0       | math-effect     |
| G02  | cppmega-mlx-3tu | 8              | P0       | math-effect     |
| G03  | cppmega-mlx-7w9 | 8              | P0       | math-effect     |
| G04  | cppmega-mlx-3m6 | 10             | P0       | math-effect     |
| G05  | cppmega-mlx-8be | 8              | P1       | distributed     |
| G06  | cppmega-mlx-cn3 | 7              | P1       | distributed     |
| G07  | cppmega-mlx-wz2 | 8              | P1       | distributed     |
| G08  | cppmega-mlx-406 | 7              | P2       | lifecycle       |
| G09  | cppmega-mlx-ajm | 8              | P1       | lifecycle       |
| G10  | cppmega-mlx-jrp | 6              | P2       | lifecycle       |
| G11  | cppmega-mlx-4n3 | 9              | P1       | persistence     |
| G12  | cppmega-mlx-t8i | 10             | P1       | persistence     |
| G13  | cppmega-mlx-ay6 | 6              | P2       | strictness      |
| G14  | cppmega-mlx-amy | 6              | P2       | strictness      |
| G15  | cppmega-mlx-a0b | 8              | P2       | strictness      |
| G16  | cppmega-mlx-3va | 7              | P2       | strictness      |
| G17  | cppmega-mlx-7ei | 10             | P0       | math-effect     |
| G18  | cppmega-mlx-biy | 8              | P2       | side-channel+UX |
| G19  | cppmega-mlx-udk | 8              | P2       | side-channel+UX |
| G20  | cppmega-mlx-5kt | 8              | P2       | inference       |
| G21  | cppmega-mlx-aih | 8              | P2       | observability   |
| G22  | cppmega-mlx-3p2 | 7              | P2       | optimizer       |
| G23  | cppmega-mlx-br8 | 6              | P2       | optimizer       |
| G24  | cppmega-mlx-edp | 6              | P3       | observability   |
| G25  | cppmega-mlx-zuf | 10             | P2       | architecture    |

**Total: 195 sub-tasks across 25 gaps.**

## 9. Done definition

All 25 gap tickets closed → epic closed. Closure report at
`tests/fixtures/e2e_matrix_v5_report.md`. Zero remaining
"propagation-only" tests (every UI mutation that claims to change
training behavior produces an asserted numerical difference in extras
or inference output).
