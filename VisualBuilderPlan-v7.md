# VisualBuilderPlan-v7 — UI → API → Training End-to-End Closure

**Status**: planned 2026-05-23 (epic `cppmega-mlx-v7-q`)

**Driver**: 8-lane parallel audit (Lanes 1–8: Gallery, Canvas, Tokenizer, Data,
Training, Checkpoints, Optim, Distributed) against the operator promise from
the Raschka LLM gallery walkthrough. Audit report at
`docs/UI-TO-TRAIN-AUDIT-2026-05-23.md`. Found 5 production-blocking gaps + 3
polish + 2 coverage; this plan closes them.

## bd ticket ID mapping (post-creation)

| Q#   | bd id           | Title                                                 |
|------|-----------------|-------------------------------------------------------|
| Q01  | cppmega-mlx-zcbs | DataInspector file picker (P0 BLOCKING, Lane 4)      |
| Q02  | cppmega-mlx-ofd2 | Edge validation in FlowCanvas (P0 BLOCKING, Lane 2)  |
| Q03  | cppmega-mlx-9avx | Checkpoint history UI + compress/strict (P1, Lane 6) |
| Q04  | cppmega-mlx-2d47 | Close 5 Raschka gallery preset gaps (P1, Lane 1)     |
| Q05  | cppmega-mlx-bfbi | smoke_zero1 CLI implementation (P1, Lane 8)          |
| Q06  | cppmega-mlx-sbz8 | Train UI polish — B/S + nesting + plasticity (P2)    |
| Q07  | cppmega-mlx-x4vd | E2E coverage gaps — extras + edge + adapter (P2)     |
| Q08  | cppmega-mlx-bpge | Late warnings + multi-node receipt (P3)              |

V7-Q is intentionally smaller than V5/V6 (8 epics, ~24 sub-tickets vs V6's
25 gaps / 190 sub-tickets) because the closure surface is narrower —
V5/V6 wired the *math*; V7-Q wires the *operator UX* on top.

## 1. Why v7-Q exists

V6 closed the backend-ready → UI-wired → real-math triangle. The honest
audit pass that followed (`docs/UI-TO-TRAIN-AUDIT-2026-05-23.md`) walked
the Raschka-gallery operator promise step-by-step and found that the
*math* paths work end-to-end, but several *operator surfaces* are stub
or text-only:

### Block A — File system access (P0 BLOCKING, 1 gap)
The promise "user uploads a parquet file (or picks one from the
matrix)" is half-shipped: matrix-pick works, upload doesn't.

| Q#   | Audit lane | UI work needed                                          |
|------|-----------|----------------------------------------------------------|
| Q01  | Lane 4     | `<input type="file">` + `POST /upload/parquet` endpoint |

### Block B — Compatibility-at-edit (P0 BLOCKING, 1 gap)
`FlowCanvas` accepts `isValidConnection?: IsValidConnection` (line 20)
but `App.tsx:1215` never passes it. Incompatible drops succeed silently
and fail at server verify. Client-side rejection contract exists in the
React Flow prop but is unused.

| Q#   | Audit lane | UI work needed                                          |
|------|-----------|----------------------------------------------------------|
| Q02  | Lane 2     | Pass `isValidConnection` from App, derive set from `catalog.list_options('compatible_edges')` |

### Block C — Operator ergonomics (P1, 3 gaps)
Backend supports save/load with arch-hash, compression, strict modes,
mid-run trigger; UI surfaces only the two raw text paths. 5 of 71
Raschka entries have bricks but no preset factory. `smoke_zero1` CLI
documented but unimplemented.

| Q#   | Audit lane | UI/CLI work needed                                       |
|------|-----------|-----------------------------------------------------------|
| Q03  | Lane 6     | `ckpt.list_history` RPC + dropdown + 3 toggles + V7-H05 WS |
| Q04  | Lane 1     | 4 preset factories: abs_pos_embed / mlstm / per_layer_embed; tiny_aya wire-in |
| Q05  | Lane 8     | `cppmega_mlx.cli.smoke_zero1` + mlx.launch wrapper + receipt regression |

### Block D — UX polish (P2, 2 gaps)
B and S not exposed as train controls. `loss_scaler.overflow_steps`
nests under `loss_scaler` dict in backend but UI reads flat
`extras.loss_scaler_overflows` — silent mismatch, overflow markers
don't render. Plasticity / MTP / IFIM / MHC extras flow through but
have no styled panel.

| Q#   | Audit lane | Work                                                      |
|------|-----------|-----------------------------------------------------------|
| Q06  | Lane 5     | B + S inputs; fix nesting at `stages.py:2245`; add 4 styled sub-panels |

### Block E — Coverage gaps (P2, 1 gap)
`readTrainExtras` reads 7/16 promised keys; no e2e for manual edge
connect or GotchasTab adapter suggestions.

| Q#   | Audit lane | Work                                                      |
|------|-----------|-----------------------------------------------------------|
| Q07  | Lane 2+5   | Extend readTrainExtras + 2 new Playwright specs           |

### Block F — Defer / external blocker (P3, 1 gap)
Tokenizer compatibility check at preview time, peak-RSS regression at
real >1 GiB checkpoint, multi-node throughput receipt. The last one is
**BLOCKED** on peer-48 hardware (external).

| Q#   | Audit lane | Work                                                      |
|------|-----------|-----------------------------------------------------------|
| Q08  | Lane 4+6+8 | tokenizer-at-preview, >1GB peak-RSS test, multi-node note |

## 2. Goal

After v7-Q every step of the operator promise from
`docs/UI-TO-TRAIN-AUDIT-2026-05-23.md §"happy path"` works without
copy-paste of paths, without silent shape-contract failures, and with
71/71 Raschka gallery coverage. The honest audit is re-runnable and
returns zero P0/P1 gaps.

## 3. Stages — 8 Q-epics × 3–5 sub-tasks ≈ 24 sub-tickets

```
cppmega-mlx-v7-q (epic)
├── Q01 file picker         (P0, 3 sub)
├── Q02 edge validation     (P0, 3 sub)
├── Q03 ckpt history        (P1, 4 sub)
├── Q04 5 gallery presets   (P1, 4 sub)
├── Q05 smoke_zero1 CLI     (P1, 3 sub)
├── Q06 train UI polish     (P2, 3 sub)
├── Q07 e2e coverage        (P2, 3 sub)
└── Q08 late warnings       (P3, 3 sub)

Total: ~24 sub-tickets (much smaller than V6 because closure surface
is narrow — V7-Q is the "polish on top of V6's wiring" pass)
```

## 4. Per-gap acceptance + sub-task breakdown

### Q01 — DataInspector file picker (P0 BLOCKING, 3)
- Q01.1 Backend `POST /upload/parquet` endpoint in `cppmega_v4/jsonrpc/server.py` with size-cap 2 GiB, suffix-validate, write to `${VBGUI_UPLOAD_DIR:-/tmp/vbgui_uploads}/<uuid>.parquet`, return `{"path": str}`. TTL cleanup 24 h.
- Q01.2 UI: `<input type="file" accept=".parquet">` in `DataInspector.tsx:194-198` with testid `data-inspector-file-upload`. On change: `FileReader.readAsArrayBuffer` → POST → set text field to returned path.
- Q01.3 Tests: vitest mock fetch + path-set; pytest `TestClient` POST a 1 KiB synthetic parquet; Playwright `page.setInputFiles` on the testid, assert preview RPC fires with the returned path.

### Q02 — Edge validation in FlowCanvas (P0 BLOCKING, 3)
- Q02.1 Backend `catalog.list_options("compatible_edges")` returns `[{src_kind, dst_kind}]` derived from `cppmega_v4/buildspec/shape_contract.SHAPE_CONTRACTS`.
- Q02.2 UI: `App.tsx` on preset/brick changes pulls compatible-edges set once; closures `validateEdge({source, target})` → `contractSet.has(`${srcKind}→${dstKind}`)`. Pass `isValidConnection={validateEdge}` to FlowCanvas at line 1215.
- Q02.3 Tests: vitest renders FlowCanvas with `isValidConnection=()=>false`, attempts onConnect, asserts edge not added; pytest catalog endpoint coverage for all 25 bricks; Playwright drops `attention` + `embed_lookup`, attempts connect, asserts no React Flow edge node appears.

### Q03 — Checkpoint history UI + compress/strict (P1, 4)
- Q03.1 RPC `ckpt.list_history(directory: str = ".") → list[CkptEntry]`. Scans recursively for `*.safetensors`, reads metadata via `read_ckpt_metadata` (header-only, no full load), sorts by mtime desc, caps at 100. CkptEntry: `{path, mtime, size_bytes, arch_hash, opt_kind, global_step, has_opt_sidecar}`.
- Q03.2 UI: new `CheckpointHistoryDropdown.tsx` component in `vbgui/src/components/`, wire into `TopBar.tsx` next to ckpt path input. Each row: `<basename> · arch_hash[:8] · step N · mtime`. Click → fills `ckpt_load_path` text field.
- Q03.3 UI: `TrainOptionsPanel.tsx` adds 3 controls — `compress` dropdown (none / weights-int8 / opt-fp16 / both), `ckpt_strict` checkbox, `opt_state_strict` checkbox. Forward via `stage_options.train`.
- Q03.4 V7-H05 mid-run trigger: backend adds `_TRIGGER_CHECKPOINT_TOKENS` set + WS message handler `ckpt.trigger {token, path}` → train loop checks each step → saves via existing `_save_compressed` + sidecar. UI: existing `TrainLiveControls` button's `onScheduleCheckpoint` callback rewired to send the WS message.

### Q04 — 5 Raschka gallery preset gaps (P1, 4)
- Q04.1 `gpt2_xl_specs(...)` in `cppmega_v4/architectures/presets.py` using `abs_pos_embed` brick. Closes Raschka #1.
- Q04.2 Wire existing `tiny_aya_parallel_specs()` into `PRESETS` dict with key `"tiny_aya"`. Closes Raschka #44.
- Q04.3 `xlstm_7b_specs(...)` using `mlstm` brick. Closes Raschka #50.
- Q04.4 `gemma_4_e2b_specs(...)` + `gemma_4_e4b_specs(...)` using `per_layer_embed` brick. Closes Raschka #57+58. Remove 5 xfail markers from `tests/v4/test_galcov_stage_d.py:204-211`. Acceptance: 71/71 (100%).

### Q05 — smoke_zero1 CLI (P1, 3)
- Q05.1 `cppmega_mlx/cli/smoke_zero1.py` entrypoint with argparse `--hosts H1,H2,...`, `--world-size W`, `--num-steps N`, `--out path`. Wraps `mlx.launch -n W --hosts H1,H2,...` around the `bench/bench_zero1_loopback` payload.
- Q05.2 Each rank runs ZeRO-1 wrapper over a tiny MLP; captures per-rank loss + `gradient_reduce_ms`. Rank 0 collects + writes receipt JSON `{world_size, hosts, losses, reduce_ms, parity_max_delta}`.
- Q05.3 Regression: receipt regeneratable via `python -m cppmega_mlx.cli.smoke_zero1 --hosts 127.0.0.1,127.0.0.1` matches existing `bench/baselines/zero1_loopback_2proc_m4.json` within tolerances. pytest hook for loopback receipt.

### Q06 — Train UI polish (P2, 3)
- Q06.1 `TrainOptionsPanel.tsx` adds `train-opt-B` (number 1-32) and `train-opt-S` (number 1-8192) inputs. App overrides `spec.dim_env.B/S` before `pipeline.run`.
- Q06.2 Fix nesting at `cppmega_v4/runner/stages.py:2245`: also flatten `extras["loss_scaler_overflows"] = loss_scaler_overflow_steps` (keep nested `loss_scaler.overflow_steps` for back-compat). vitest pins both.
- Q06.3 `TrainExtrasOverlay.tsx`: add conditional sub-panels — `PlasticityPanel` (FIRE/DASH/ReDo), `MTPPanel` (k-head losses), `IFIMPanel` (instr-FIM loss), `MHCPanel` (multi-head consistency). Each renders only when its key present in extras.

### Q07 — E2E coverage (P2, 3)
- Q07.1 Extend `vbgui/e2e/utils/train_extras.ts:readTrainExtras` to all 16 promised keys (losses, losses_smoothed, val_losses, lr_trajectory, perplexity, bpb, dtype_state, fp8_active, fim_active, sharding_applied, side_channels_observed, per_brick_grad_norms, moe.*, optimizer_kind, gradient_reduce_ms, weight_delta_norm). `12_train_convergence.spec.ts` asserts all 16 present + finite where applicable.
- Q07.2 New `vbgui/e2e/scenarios/74_manual_edge_connect.spec.ts`: drop two compatible bricks, manually draw edge via React Flow `connectionLine`, verify edge appears, assert verify RPC accepts.
- Q07.3 New `vbgui/e2e/scenarios/75_adapter_suggestions.spec.ts`: load preset with intentional shape mismatch, navigate to GotchasTab, click "Suggest adapters", assert chain renders and "Apply" inserts adapter nodes on canvas.

### Q08 — Late warnings + perf + external blocker note (P3, 3)
- Q08.1 Extend `data.preview_parquet` RPC: accept optional `tokenizer_source` arg → returns `roundtrip_pass_rate: float` field → DataInspector shows badge inline with preview, not waiting until train.
- Q08.2 pytest `test_checkpoint_streaming_peak_rss.py` generates 1.5 GiB synthetic checkpoint → `load_auto` → asserts `mx.metal.get_peak_memory()` < 200 MiB during load.
- Q08.3 Multi-node throughput receipt: file a `BLOCKED` note in bd ticket — peer-48 TB5 + macOS 26.2 + M4 family hardware required (per `docs/multimac_training.md:114-129`). Carry hardware-blocker notes; not implementable without external resource.

## 5. Workflow per gap (same as v5/v6)

1. `bd update <id> --claim`
2. Implement (each sub-task atomic)
3. Code review (self if <100 LOC, gsd-code-reviewer otherwise)
4. Perf check (no >5% regression on affected suite)
5. Regression (pytest + vitest + e2e affected)
6. `git add` specific files → commit → push → verify on origin/main
7. `bd close <id>` only after push verified

## 6. Acceptance counters

| Surface                                | Before v7-Q | Target v7-Q |
|----------------------------------------|-------------|-------------|
| pytest                                 | ~2540       | ≥2570       |
| vitest                                 | ~428        | ≥440        |
| Playwright deep e2e cells              | ~1170       | ≥1175       |
| Raschka gallery coverage               | 66/71 (93%) | 71/71 (100%) |
| extras keys read by e2e                | 7/16        | 16/16       |
| P0 audit gaps                          | 2           | 0           |
| P1 audit gaps                          | 3           | 0           |
| Hardware-blocked gaps                  | 1           | 1 (external) |

## 7. Out of scope (defer to v8+)

- Real cross-Mac ZeRO-1 receipt (BLOCKED on peer-48 hardware)
- 1B+ param training matrix on Mac
- A/B hyperparameter search UI
- LoRA / PEFT adapter wiring
- Tokenizer mutation UI (read-only by design per Lane 3)
- Sophia / Adafactor / Tiger / AdEMAMix optimizers (deferred per Lane 7)
- Live inference serving from trained checkpoint
- Model weight quantisation post-train

## 8. Done definition

All 8 Q-gaps closed (Q08.3 carried as hardware-blocker note). Zero P0/P1
audit findings on re-run. Closure report at
`tests/fixtures/e2e_matrix_v7_q_report.md`. Re-run audit
(`docs/UI-TO-TRAIN-AUDIT-2026-05-23.md` re-execution) returns 0 BLOCKING
gaps.
