# VisualBuilderPlan-v8 — Raschka-Gallery Full-Loop Closure

**Status**: planned 2026-05-23 (epic `cppmega-mlx-v8`)

**Driver**: post-v7 audit. v7 wired the operator UX (file picker, edge
validation, checkpoint history, smoke_zero1). v8 closes the *gallery-
to-train round trip* the user actually asks for:

> "Из панели вытащить любую модель из galler.html, увидеть её
> архитектурный спек и оригинальные training hyperparams, скейл-даунить
> под мой devbox, увидеть memory matrix × precision × parallelism,
> мутировать (residual / norm / MTP / MHC / Engram / n-gram side-channel),
> увидеть что зафьюзится / материализуется / синхронизируется, поднять
> данные с HF в стиле nanochat ИЛИ с GitHub через tree-sitter+clang,
> прогнать N шагов трейна — и всё это покрыто Playwright e2e."

## bd ticket ID mapping (post-creation)

| R#   | scope                                              | bd parent            | priority |
|------|----------------------------------------------------|----------------------|----------|
| R01  | Per-preset training defaults (lr/bs/sched from paper) | new epic            | P1       |
| R02  | scale_down(preset, target_bytes) helper + UI slider | new epic            | P1       |
| R03  | Memory matrix UI (precision × parallelism)         | new epic             | P1       |
| R04  | Auto-fit to detected devbox                        | new epic             | P1       |
| R05  | MXFP4 on Apple Metal (e2m1 block-scaled)           | new epic             | P2       |
| R06  | Compile-trace panel (inductor / dlpack / fused)    | new epic             | P2       |
| R07  | Z3 sync-necessity checker + advice                 | new epic             | P2       |
| R08  | Mid-canvas feature injection (MTP/MHC/Engram/n-gram) | new epic           | P1       |
| R09  | HF nanochat-style data quick-start                 | new epic             | P1       |
| R10  | Tree-sitter / clang GitHub-corpus side-stream      | new epit             | P2       |
| R11  | N-step train end-to-end (full closure)             | new epic             | P0       |
| R12  | Playwright e2e harness for the full loop           | new epic             | P0       |

## 1. Why v8 exists

The audit dialogue with the operator confirmed the gap is no longer
"can we wire X" but "given any of the 57 presets, can you complete the
*single-pane* journey: drag → see paper defaults → scale-down → see
memory matrix → mutate (feature injection) → see compile/fuse trace →
attach data → run N steps → see live loss". v7-Q delivered the UX
plumbing; v8 delivers the journey *as one e2e flow*.

We have:
- 57 presets in `cppmega_v4/architectures/presets.py` (Raschka coverage)
- `verify_and_estimate` per-brick memory at every mutation
- 17 gotcha rules + `suggest_sharding` + 8 topology factories
- `mx.distributed` real DP/TP/PP (V7-B-real)
- bf16 / fp16 / fp32 / fp8 (e4m3, e8m0 blockscaled) precision toggles
- Plasticity (FIRE/DASH/ReDo), 8-bit Adam/Muon, LossScaler
- Live WS streaming (`/ws/train/{run_id}`, `/ws/gen/{job_id}`,
  `/ws/verify/{spec_hash}`) with sparkline + dead-man-switch
- 62 existing Playwright e2e specs

We *don't* have:
- Per-preset paper hyperparameter table
- Auto scale-down API
- MXFP4 (Metal e2m1 path)
- Compile-trace visualization
- Z3 sync-necessity check
- Tree-sitter/clang GitHub side-stream
- HF nanochat-style quick-attach
- The single integrated end-to-end e2e test

## 2. Goal

After v8 lands, an operator opens vbgui, picks any preset from the
gallery, and reaches a finished N-step train in one continuous flow
where every intermediate decision is informed by real backend data
(paper defaults, memory matrix, fuse-trace, z3 advice).

## 3. Stages — 12 R-epics × 2-5 sub-tasks ≈ 36 sub-tickets

Workflow per epic stays identical to v5/v6/v7:
discuss → plan → execute → verify, atomic commits per sub-task,
real backend changes never silently degraded to mocks.

## 4. Per-epic acceptance + sub-task breakdown

### R01 — Per-preset training defaults (P1, 3 subs)

**Why**: dragging llama3_8b currently inherits INITIAL_SPEC's
generic `lr=1e-3 / adamw / constant`, not the paper's
`lr=3e-4, wsd schedule, β=(0.9, 0.95)`. The operator must
re-discover paper hyperparams from scratch.

**Subs**:
- R01.1 `cppmega_v4/architectures/preset_training_defaults.py`:
  table keyed by preset → {lr, batch, schedule, betas, gradient_clip,
  warmup_steps, mixed_precision, source_paper_url}. ≥ 30 presets covered.
- R01.2 `build_preset_specs` RPC grows `defaults: TrainingDefaults`
  alongside `specs`; UI auto-fills LossTab / OptimTab / ScheduleEditor.
- R01.3 Vitest + pytest parity check that defaults round-trip.

**AC**: pick `llama3_8b` → OptimTab shows `lr=3e-4` automatically; pick
`gpt_oss_120b` → schedule pre-populates with `cosine warmup=2000`.

### R02 — scale_down(preset, target_bytes) helper + UI slider (P1, 4 subs)

**Subs**:
- R02.1 `cppmega_v4/architectures/scale_down.py:scale_down_preset(preset, target_bytes, *, min_dim=64) -> dict`:
  binary-search `hidden_size`, fall back to `num_layers` halving when
  hidden_size hits min_dim. Returns the resolved
  `(hidden_size, num_layers, est_bytes)`.
- R02.2 New RPC `architectures.scale_down` (registry + dispatcher).
- R02.3 UI: target-memory slider in GalleryTab → live `est_bytes`
  preview before drag-to-canvas.
- R02.4 pytest: scale_down(llama3_8b, 1GB) ≤ 1GB and ≥ 0.5GB.

### R03 — Memory matrix UI (P1, 3 subs)

**Subs**:
- R03.1 RPC `memory.matrix(spec, topologies, precisions) -> Matrix`
  returns 4 topologies × 4 precisions (fp32, bf16, fp16, fp8) per-rank
  worst-case bytes, plus a `fits` boolean per cell.
- R03.2 UI: new `MemoryMatrix.tsx` (16 cells, color-coded fit/no-fit,
  hover for breakdown). Lives in sidebar/MemoryMatrixTab.
- R03.3 Playwright: pick preset → matrix renders 16 cells → at least
  one cell green on `m3_ultra_solo`.

### R04 — Auto-fit to detected devbox (P1, 2 subs)

**Subs**:
- R04.1 `platform.get_info` already returns the host's RAM/Metal info.
  New RPC `architectures.auto_fit(preset, host_info)` chains scale_down
  + suggest_sharding to pick the smallest valid (hidden, layers, axes)
  that fits.
- R04.2 UI: "Auto-fit to this Mac" button next to GalleryTab preset →
  one-click drop to canvas with scaled spec.

### R05 — MXFP4 on Apple Metal (e2m1 block-scaled) (P2, 4 subs)

**Why**: user explicitly asked. NVIDIA NVFP4 is Hopper/Blackwell-only
but Apple MLX 0.16+ exposes a `mx.metal` block-scaled e2m1 quant
path the user nicknamed mx4. This is a real opportunity.

**Subs**:
- R05.1 `cppmega_mlx/quant/mxfp4_metal.py`: e2m1 block-scaled quant
  (16-elem blocks, fp8 e4m3 scales) using `mx.metal_kernel`. Round-trip
  RMSE ≤ bf16 → bf16 quant.
- R05.2 SchemeRouter (`_quantize_8bit.py`) gets `QUANT_SCHEME_MXFP4`.
- R05.3 Sharding spec gets `mxfp4_enabled` flag mirroring `fp8_enabled`.
- R05.4 Parity test: e2m1 quant ↔ bf16 within ≤ 5% RMSE on a random
  tensor; smoke-train at mxfp4 reaches finite loss.

### R06 — Compile-trace panel (P2, 4 subs)

**Why**: user wants to see *what* the compiler fuses, what materializes,
what passes through dlpack, and where MLX vs torch.compile vs TileLang
takes over.

**Subs**:
- R06.1 RPC `compile.trace(spec, backend in {mlx, tilelang, torch_inductor})`:
  reuses `torch_compile_backend.inspect_fx_graph` for the torch path
  and `dispatch_lower(prim, return_msl=True)` for the TileLang path.
  Returns `{ops: [{name, fused, materialised, dlpack_boundary}],
  fused_groups: [...], compile_artifact_path}`.
- R06.2 UI: `CompileTracePanel.tsx` — list ops with colored chips:
  green=fused, yellow=dlpack-crossing, red=materialised, gray=stays
  in source backend.
- R06.3 Hook into BrickContextPanel — clicking a brick highlights its
  ops in the trace.
- R06.4 pytest: trace on a small spec returns ≥ 1 fused group when the
  TileLang path is engaged.

### R07 — Z3 sync-necessity checker (P2, 3 subs)

**Why**: every dlpack boundary or `mx.eval` is a candidate sync point;
many are unnecessary. Z3 can prove necessity from the data-flow graph.

**Subs**:
- R07.1 `cppmega_v4/spec/sync_checker.py`: build a small SSA-style
  graph from the spec's brick chain; encode constraints
  ("op B reads tensor X which op A writes; if A and B are on same
  backend, no sync needed"); solve with z3 (`z3-solver` dep). Returns
  `{necessary_syncs: [...], redundant_syncs: [{op, reason}], advice: [...]}`.
- R07.2 RPC `sync.check(spec)`.
- R07.3 UI: badge in CompileTracePanel — "3 redundant syncs, click
  for advice"; advice modal explains each candidate.

### R08 — Mid-canvas feature injection (P1, 4 subs)

**Subs**:
- R08.1 `cppmega_v4/buildspec/rewriters.py` already has MTP /
  MHC / IFIM / Engram rewriters — confirm and surface in catalog.
- R08.2 `catalog.list_options('feature_injectors')` enumerates
  available rewriters with their effect descriptions.
- R08.3 UI: new `FeatureInjectionBar.tsx` between ParallelComposeBar
  and InsertIntoEdgeBar — dropdown of {MTP_WEIGHTED, MHC, IFIM, Engram,
  n_gram_side_channel} + Apply button. Mutation goes through existing
  rewriter pipeline.
- R08.4 Playwright: apply MTP rewriter via the bar → run train → assert
  `extras.train.mtp.per_head_losses` populates (proves the rewriter
  actually fired, not just listed).

### R09 — HF nanochat-style data quick-start (P1, 3 subs)

**Subs**:
- R09.1 `scripts/data/hf_quickstart.py`: takes
  `{dataset_id, split, tokenizer, n_tokens}`, streams from HF (datasets
  lib), tokenizes with the chosen tokenizer, writes a tmp parquet
  shard ready for stage_train. Matches the karpathy-nanochat "fineweb
  edu shuffle" pattern.
- R09.2 RPC `data.hf_quickstart` (long-running, publishes progress on
  `/ws/data/{job_id}`).
- R09.3 UI: button in DataInspector — "Quick-start from HF" → modal
  with dataset_id input + token-budget slider; on completion the
  parquet path auto-populates.

### R10 — Tree-sitter / clang GitHub-corpus side-stream (P2, 4 subs)

**Subs**:
- R10.1 We already have `scripts/nanochat_data/clang_enriched_to_parquet.py`
  + `extract_git_history.py`. Wrap them into a single
  `scripts/data/github_corpus.py {repo_url, max_commits, max_tokens}`.
- R10.2 RPC `data.github_corpus` parallel to `data.hf_quickstart`.
- R10.3 UI: tabs in the HF-quickstart modal — "HF Hub" vs "GitHub
  repo (tree-sitter / clang)".
- R10.4 Playwright: drive the GitHub branch against a small public
  repo; assert the resulting parquet has `source_doc_id` + clang
  side-channels populated.

### R11 — N-step train end-to-end (P0, 3 subs)

**Subs**:
- R11.1 No new RPC — this is the *integration*: pick preset → R01/R02/R03
  fire → R08 inject MTP → R09/R10 attach data → stage_train runs N
  steps with live WS sparkline (V7-L37..L41 already shipped).
- R11.2 Document the canonical flow in a new
  `docs/raschka_full_loop.md` cookbook.
- R11.3 Manual UAT script in `docs/uat/v8_raschka_full_loop.md`:
  10-step user journey with expected screenshots.

### R12 — Playwright e2e harness for the full loop (P0, 3 subs)

**Subs**:
- R12.1 `vbgui/e2e/scenarios/v8_raschka_full_loop.spec.ts`: a single
  end-to-end test that walks through R01 → R02 → R03 → R08 → R09 → R11
  on a synthetic 64-token quick-data path. Total time ≤ 90s on dev box.
- R12.2 Each intermediate assertion captures `page.on('console')` +
  `page.on('pageerror')` into `test.info().annotations` so flaky runs
  leave diagnosable traces (as the user requested earlier).
- R12.3 A `--full-loop` flag on the e2e runner that picks 5 different
  presets in a matrix and validates the same 10-step journey on each.

## 5. Workflow per epic (same as v5/v6/v7)

For each R-epic:
1. `bd update <id> --notes` capturing the audit context above.
2. Discuss → plan → execute → verify per gsd-quick template.
3. Atomic commit per sub-task with `feat(v8-rNN): ...` prefix.
4. Acceptance gate: pytest + vitest + Playwright cover every sub-task.

## 6. Acceptance counters

- **R01..R12**: 12 epics; 36 sub-tickets total.
- **Tests added**: ≈ 25 pytest, ≈ 10 vitest, ≈ 15 Playwright e2e.
- **New RPCs**: `architectures.scale_down`, `architectures.auto_fit`,
  `memory.matrix`, `compile.trace`, `sync.check`,
  `data.hf_quickstart`, `data.github_corpus`. (+ 3 ws endpoints:
  `/ws/data/{job_id}`, others reuse v7.)
- **Backend new files**: ≈ 9.
- **UI new components**: `MemoryMatrix`, `CompileTracePanel`,
  `FeatureInjectionBar`, HF/GitHub quickstart modal,
  GalleryTab scale-down slider, auto-fit button.

## 7. Out of scope (v9+)

- NVFP4 (Hopper/Blackwell only).
- Multi-host training launcher UI (CLI runbook only — `mlx.launch` /
  mpirun via `scripts/launch_multi.py` ships in v7-B-real).
- Replacing existing `tiny_aya` / `zaya1` preset internals.
- Cross-preset auto-arch-search (NAS).

## 8. Done definition

- All 12 R-epics closed in bd.
- VisualBuilderPlan-v8.md + VisualBuilderSpec-v8.md present.
- `v8_raschka_full_loop.spec.ts` green on llama3_8b, qwen3_dense_4b,
  gemma3_27b, deepseek_v3, gpt_oss_20b.
- `docs/raschka_full_loop.md` + `docs/uat/v8_raschka_full_loop.md`
  written.
