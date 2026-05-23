# UI → API → Training End-to-End Audit
**Date**: 2026-05-23 14:45 GMT+2
**Scope**: Honest comparison of promised-vs-shipped across 8 vertical lanes of the Visual Builder GUI.
**Method**: 8 parallel `Explore` agents, each ≤80-line report mapping docs/specs/designs to actual code + test status.

## Executive summary

The system delivers the **core promise** end-to-end: a user can build a model from bricks, configure it, run training, and see loss. But there are **5 concrete production gaps** that block the full promise from the user description:

| # | Gap | Severity | Lane |
|---|---|---|---|
| 1 | **No real file upload** — DataInspector is text-input only, no `<input type="file">` | 🔴 BLOCKING | 4 |
| 2 | **No edge validation** — FlowCanvas accepts `isValidConnection` prop but App never passes it; edges fail silently at verify | 🔴 BLOCKING | 2 |
| 3 | **No checkpoint history UI** — no `ckpt.list` RPC, no picker widget; resumes require copy-paste paths | 🟡 HIGH | 6 |
| 4 | **5 Raschka gallery presets unwired** — GPT-2 XL, Tiny Aya, xLSTM, Gemma 4 E2B/E4B (bricks exist, factories don't) | 🟡 HIGH | 1 |
| 5 | **smoke_zero1 CLI missing** — documented but unimplemented; multi-Mac training is loopback-only | 🟡 HIGH | 8 |

Plus **3 medium-debt UX gaps**:

| # | Gap | Lane |
|---|---|---|
| 6 | B and S not exposed as train controls (embedded in preset) | 5 |
| 7 | `loss_scaler.overflow_steps` nesting mismatch between backend (nested) and UI (`loss_scaler_overflows` flat) | 5 |
| 8 | Compress/strict toggles missing from UI (backend supports `compress`, `ckpt_strict`, `opt_state_strict`) | 6 |

And **2 test-coverage gaps** (functionality works, regression doesn't pin it):

| # | Gap | Lane |
|---|---|---|
| 9 | E2E `readTrainExtras` only reads 7/16 promised extras keys | 5 |
| 10 | No e2e test for manual edge connection (only drag-drop bricks) or adapter-chain UI | 2 |

**Lane 3 (Tokenizer)** and **Lane 7 (Optimizer)** are clean — no production gaps; tokenizer mutation is out-of-scope by design.

---

## Lane-by-lane summary (compressed)

### Lane 1 — Gallery coverage: **93% (66/71)**
- 57 presets in `cppmega_v4/architectures/`, 57 in UI PRESETS array (dynamic via `architectures.list_presets` RPC).
- 25 brick kinds in BLOCK_BUILDERS, 25+6 adapters draggable from `Palette.tsx`.
- Tests: `tests/v4/test_galcov_stage_d.py` 203 passed, 15 xfail (locked gaps).
- **Gap**: 5 Raschka entries lack preset factories:
  - #1 GPT-2 XL → needs `abs_pos_embed` preset
  - #44 Tiny Aya → `tiny_aya_parallel_specs()` exists but not in PRESETS dict
  - #50 xLSTM 7B → needs `mlstm` preset
  - #57 Gemma 4 E2B, #58 E4B → needs `per_layer_embed` preset
- Bricks already exist for all five; only `presets.py` factory wiring missing.

### Lane 2 — Canvas drag/drop/connect/configure: **85%**
- 428 vitest tests passing (drag, connect, dim_env, undo/redo, verify-on-edit).
- E2E: `08_manual_drag_drop.spec.ts` (26 bricks) + `09_e2e_manual.spec.ts` (8 cross-product scenarios).
- **Critical gap**: `FlowCanvas` accepts `isValidConnection?: IsValidConnection` (line 20) but `App.tsx:1215` never passes it. Edges silently fail shape contracts.
- **Coverage gap**: no e2e exercises manual edge connection or `GotchasTab` suggest_adapters UI.

### Lane 3 — Tokenizer: **100% of in-scope promise**
- 12 tokenizer presets (cppmega_v3, nanochat, gpt-4o, gpt-3.5, gpt-2, llama-3, mistral, gemma, qwen, deepseek, phi-3, claude).
- Byte-exact roundtrip verified vs gb10 reference (`test_decode_parity_with_gb10_reference_receipt` PASS).
- 3-panel playground compare, FIM/SPACE/NL special-id contract exposed.
- 42/42 tests green.
- **Out-of-scope**: vocab editing (read-only by design; user "modify tokenizer" expectation = preset switch, which works).

### Lane 4 — Data pipeline: **60% — CRITICAL GAP**
- Backend complete: `data.preview_parquet`, `data.roundtrip_check`, glob/dir/manifest support, corpus_stats sidecar, real-token train consumption (parquet_path → data_source="parquet_tokenized").
- 16 vitest tests passing.
- E2E `18_real_data_convergence.spec.ts`: parquet + tokenizer + loss decay verified.
- **Critical gap**: DataInspector path field is plain `<input type="text">` (`DataInspector.tsx:194-198`). No file picker, no upload widget. User cannot point at arbitrary local parquet — only fixture paths.
- Secondary: no tokenizer compatibility check at preview time (only at train time → silent late failure).

### Lane 5 — Training loop UI: **88% — 14/16 extras keys surface**
- All 16 promised train-extras keys surface in TrainExtrasOverlay (losses, losses_smoothed, val_losses, lr_trajectory, perplexity, bpb, dtype_state, fp8, fim, sharding, side_channels, per_brick_grad_norms, MoE 9-key, optimizer_kind, gradient_reduce_ms, weight_delta_norm).
- All 10 promised train UI controls present except B and S (embedded in preset, not independent inputs).
- 75 pytest stage_train tests pass.
- **Gap 1**: B and S not exposed as train controls (workaround: spec edit pre-train).
- **Gap 2**: `loss_scaler.overflow_steps` nesting mismatch (`stages.py:2245` nests under `loss_scaler` dict, UI expects flat `extras.loss_scaler_overflows`). Silent: overflow markers don't render.
- **Gap 3**: Plasticity/MTP/IFIM/MHC extras flow through generic extras but no styled panel.
- **Coverage gap**: `e2e/utils/train_extras.ts` only reads 7/16 keys.

### Lane 6 — Checkpoint management: **85%**
- Backend complete: save/load, opt-state sidecar (just landed `05dcde8`), arch-hash validation, compression (none/int8/fp16/opt-fp16/both), sharded save/load, streaming load (peak RSS bound), backward compat (single-file + sharded + directory).
- 40+ test files pass. M0.7 resumable round-trip PASS.
- TopBar shows ckpt metadata (arch_hash, opt_kind, global_step) via debounced `ckpt.inspect` RPC.
- **Gap 1**: No `ckpt.list_history` RPC, no UI picker dropdown — resumes need copy-paste.
- **Gap 2**: Compression dropdown / `ckpt_strict` / `opt_state_strict` not surfaced in UI.
- **Gap 3**: V7-H05 live mid-run checkpoint button wired in `TrainLiveControls`; backend WS queue not implemented (`onScheduleCheckpoint` callback fires but is no-op).
- **Gap 4**: No peak-RSS regression test at real >1 GB scale.

### Lane 7 — Optimizer + Schedule: **100% — ready to ship**
- 7 OptimKinds wired (AdamW, Muon, Lion, Lion8bit, Adam8bit, muon_adamw_hybrid, SGD) — all in `OPTIM_BUILTINS` + `OptimTab` KINDS array.
- 6 ScheduleKinds wired (constant, linear_warmup, cosine, wsd, inv_sqrt, polynomial) — catalog'd with paper refs.
- 7 ParamGroup matchers + regex:* wildcard.
- `suggest_optim_groups` RPC + `AutoGroupButton` proven across 6 family presets.
- 45/45 tests pass.
- Sophia/Adafactor/Tiger/AdEMAMix explicitly deferred (P3 optional per E2E v2).

### Lane 8 — Distributed training: **75% — local sim done, multi-Mac stubbed**
- 8 device kinds (h100_80gb, h200_141gb, a100_40gb/80gb, b100_80gb, gb10, tpu_v5p/v6e, m3_ultra).
- 5 sharding factories + auto_shard 3-5 ranked proposals.
- 17 gotcha rules (15 promised, +2 bonus: incompatible_comm_backend, slow_loopback_ring).
- Full memory accounting (FP8/bf16/fp32 duplication, master fp32, FSDP peak, TP boundary).
- ShardingTab UI with proposals + axis editor.
- `fake_ranks` smoke + `gradient_reduce_ms` ✅.
- ZeRO-1 wrapper using real `mx.distributed.all_sum`/`all_gather`.
- 126 parallelism tests pass.
- **Gap 1**: `smoke_zero1` CLI documented but unimplemented (`docs/distributed_zero1_smoke_procedure.md:170`).
- **Gap 2**: Multi-mac launcher manual; no orchestration playbook (`docs/multimac_training.md` Phase 2 deferred to Stream F).
- **Gap 3**: Loopback receipt only — no real cross-Mac ZeRO-1 receipt (peer-48 hardware not connected, external blocker).

---

## What this means

The **happy path works**:
1. ✅ Drag a preset → 57/71 architectures (93%) auto-build
2. ✅ Drag individual bricks → 25 kinds + 6 adapters draggable
3. ✅ Connect them → React Flow edge creation works
4. ✅ Configure params → BrickContextPanel + DimEnvEditor + verify-on-edit
5. ❌ **Upload arbitrary parquet** — must copy paths into text field
6. ✅ Preview parquet → DataInspector + channel filter + roundtrip badges
7. ✅ Pick tokenizer → 12 presets, 3-panel compare, byte-exact roundtrip
8. ✅ Train N steps → loss chart + 14/16 extras + abort + live LR
9. ❌ **Resume from arbitrary checkpoint** — must copy paths into text field, no history list
10. ✅ Pick optimizer + schedule → 7 + 6 wired with auto-group
11. ⚠️ **Manage distributed training** — single-Mac + fake_ranks works, real multi-Mac is stubbed

The honest assessment: **the system is a working dev tool, not yet a polished operator UI**. Closing gaps 1–5 (file upload, edge validation, ckpt history, 5 missing presets, smoke_zero1) flips it to fully-promised behaviour.

---

## Recommended bd epics (8 epics, ~24 sub-tickets)

```
E-AUDIT-01  [P0] Lane 4 — DataInspector file picker (BLOCKING)
  ├─ 01.1 add <input type="file"> + onChange → FileReader → POST /upload
  ├─ 01.2 backend tmp-file persistence + path return
  └─ 01.3 e2e: pick local fixture, verify preview renders

E-AUDIT-02  [P0] Lane 2 — Edge validation (BLOCKING)
  ├─ 02.1 thread isValidConnection from App.tsx into FlowCanvas
  ├─ 02.2 use shape_contract.is_compatible(src, dst) as the check
  └─ 02.3 e2e: drop incompatible bricks, attempt connect, assert rejected

E-AUDIT-03  [P1] Lane 6 — Checkpoint history UI
  ├─ 03.1 ckpt.list_history RPC (scan dir, return [{path, mtime, arch_hash, step}])
  ├─ 03.2 CheckpointHistoryDropdown component in TopBar
  ├─ 03.3 compress dropdown + ckpt_strict + opt_state_strict checkboxes
  └─ 03.4 V7-H05 live mid-run checkpoint WS queue

E-AUDIT-04  [P1] Lane 1 — Close 5 gallery gaps
  ├─ 04.1 abs_pos_embed preset (GPT-2 XL)
  ├─ 04.2 tiny_aya_parallel wired into PRESETS dict
  ├─ 04.3 mlstm preset (xLSTM 7B)
  └─ 04.4 per_layer_embed preset (Gemma 4 E2B + E4B)

E-AUDIT-05  [P1] Lane 8 — smoke_zero1 CLI
  ├─ 05.1 cppmega_mlx.cli.smoke_zero1 entrypoint
  ├─ 05.2 multi-mac launcher script (mlx.launch wrapper)
  └─ 05.3 loopback receipt regression test

E-AUDIT-06  [P2] Lane 5 — Train UI polish
  ├─ 06.1 expose B + S as TrainOptionsPanel controls
  ├─ 06.2 fix loss_scaler_overflows nesting in stages.py:2245
  └─ 06.3 Plasticity/MTP/IFIM/MHC extras panels in TrainExtrasOverlay

E-AUDIT-07  [P2] Lane 5+2 — E2E coverage gaps
  ├─ 07.1 extend readTrainExtras to all 16 keys + e2e asserts
  ├─ 07.2 e2e for manual ReactFlow edge connect
  └─ 07.3 e2e for GotchasTab suggest_adapters panel

E-AUDIT-08  [P3] Lane 4+6 — Late warnings
  ├─ 08.1 tokenizer compatibility check at preview_parquet time
  ├─ 08.2 peak-RSS regression at >1GB checkpoint
  └─ 08.3 multi-node throughput receipt (BLOCKED on peer-48 hardware)
```

Total: 8 epics, 24 sub-tickets, ~3-5 dev-days each (except 08.3 external blocker).
