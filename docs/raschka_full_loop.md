# Raschka Full-Loop Cookbook (V8)

> Companion to `VisualBuilderPlan-v8.md` / `VisualBuilderSpec-v8.md`.
> This document is the **definitive end-to-end recipe** for taking any
> preset in the Raschka gallery (currently 62 presets) and running an
> N-step training job, with paper-anchored defaults, automatic
> scaling, memory-fit verification, mid-canvas feature injection, and
> streaming data, all from the Visual Builder UI.

## 0. Prerequisites

- A built `vbgui` (`pnpm -C vbgui install && pnpm -C vbgui dev`).
- The cppmega backend running on `127.0.0.1:8767` (`uvicorn
  cppmega_v4.jsonrpc.server:app --port 8767`).
- A working tokenizer (the default bundled `cppmega_v3` ships at
  `cppmega_mlx/tokenizer/tokenizer.json`).
- For the HF-quickstart path: outbound HTTPS to `huggingface.co`.
  Set `VBGUI_DISABLE_NETWORK=1` to refuse the RPC in offline runs.

## 1. Pick a preset (R01)

Open the Visual Builder, click the **Preset launcher** dropdown,
choose e.g. `llama3_8b`.

- `build_preset_specs` runs → canvas populates with the preset's
  brick chain.
- The `defaults` block in the same response auto-fills the OptimTab
  (`adamw`, lr=`3e-4`, mixed_precision on, grad-clip 1.0) and the
  first group's ScheduleEditor (`wsd`, warmup 2000).
- Paper anchor: <https://arxiv.org/abs/2407.21783>.

## 2. Scale down to fit (R02)

Switch to the **Gallery** tab.

- A `GalleryScaleDownSlider` lives above the per-preset table.
- Pick the target preset (same dropdown), drag the slider to a budget
  (default 1 GiB).
- The `architectures.scale_down` RPC fires (debounced 250 ms) and
  surfaces the chosen `(hidden_size, num_layers)` + estimated bytes
  + scaled-down-from canonical shape.
- Click **Apply scaled preset** → canvas swaps to the scaled chain
  and `dim_env.H` updates.

## 3. Verify it fits everywhere (R03)

Open the **Memory** sidebar tab.

- `memory.matrix` returns a 4×5 grid of (topology × precision) cells.
- Default axes:
  `[h100_8x, m3_ultra_solo, gb10_quarter, tpu_v6e_8]` ×
  `[fp32, bf16, fp16, fp8, mxfp4]`.
- Each cell shows estimated bytes + a `fits` chip (green/red) +
  full per-component breakdown on hover (weights / grads / optimizer
  / activations / kv-cache / edge handoff).
- Use this matrix to validate the scale-down decision matches the
  intended deployment target.

## 4. Auto-fit to your devbox (R04)

Click **Auto-fit to my devbox** in the GalleryScaleDownSlider.

- `architectures.auto_fit` chains `platform.get_info` → `scale_down`
  → `suggest_sharding` into a single result.
- The banner that appears reports the chosen topology, the largest
  (H, L) that still fits at 90 % headroom, and a sharding-proposal
  axis summary (`dp×N, tp×M, ...`).
- Apply slot still works the same; auto-fit just picks the slider
  value for you based on the detected hardware.

## 5. Inject a feature mid-canvas (R08)

In the **Canvas** tab the `FeatureInjectionBar` floats above the
flow chart.

- The dropdown is populated by
  `catalog.list_options('feature_injectors')` — 5 options today:
  - `mtp_weighted` → `rewriter:MTPRewriter` (K=2 head + weighted loss)
  - `ifim_shaped` → `rewriter:IFIMRewriter` (span-aware reshape)
  - `mhc_attn_bias` → `rewriter:MHCRewriter` (co-occurrence bias)
  - `engram` → `brick:engram` (standalone n-gram branch)
  - `ngram_2_3_4` → `brick:engram` with 2,3,4-gram defaults
- Apply dispatches either `rewriters.add` (mutates `spec.rewriters`,
  flows into `pipeline.run`) or inserts a new brick node into the
  canvas with a fresh edge from the tail.
- Subsequent training picks up the rewriter; e.g. MTP populates
  `extras.train.mtp` automatically.

## 6. Stream training data from HF Hub (R09)

In the **Data** tab → **HF quickstart** button.

- Pick a dataset (`HuggingFaceFW/fineweb-edu` is the default), set
  the `n_tokens` target, click **Run**.
- `data.hf_quickstart` streams the dataset, tokenizes each document
  with `CppMegaTokenizer`, and writes a parquet shard with the
  canonical 4-column schema
  (`token_ids / doc_ids / byte_offsets / byte_lengths`).
- Live progress streams on `/ws/data/{job_id}` (start / progress /
  done frames). The result parquet path lands in the inspector's
  path field — click **Use for training** to point the next train
  run at it.

## 7. Run N steps (R11)

Back on the **Canvas** tab:

- Click **Run pipeline** (or the split-button "Train" mode).
- The runner threads everything wired in steps 1–6 into a
  `pipeline.run` request: scaled spec + rewriters from the injection
  bar + parquet from the quickstart + topology from the matrix view +
  ScheduleEditor schedule kind.
- The `stage_train` stage publishes per-step
  `{step, loss, lr, overflow, grad_norms, mem_mb, ...}` events onto
  `train_event_bus`; the `LiveTrainPanel` subscribes via
  `/ws/train/{run_id}` and renders the sparkline + step counter.
- A `finish: "ok"` frame fires the toast; the `extras.train` block
  now contains:
  - `preset_origin` — the originally-picked preset name (R01).
  - `scale_down_factor` — `H_scaled / H_canonical` (R02).
  - `memory_matrix_cell_used` — the cell the run actually used (R03).
  - `feature_injections` — list of applied injections (R08).
  - `data_source` — `{kind: "hf_quickstart", dataset_id, n_tokens}`
    (R09 / R10).
  - All the existing V7 extras (schedule_kind, optimizer kind, ...).

## 8. Where this fits in the larger picture

This loop covers the **"preset → trainable model"** path. The
companions are:

- The Visual Builder distributed-semantics layer (V7) handles
  multi-rank training once you've finished step 7 and want to fan
  out across multiple devboxes.
- The Plasticity Toolkit (FIRE / DASH / ReDo) hooks happen inside
  `stage_train` — they don't surface in this cookbook but are
  active on every step.
- MXFP4 + compile-trace + sync-checker (R05 / R06 / R07) are the
  remaining V8 P2 items — they extend step 3 (memory matrix) and
  step 7 (extras.train) with deeper diagnostics but aren't gating
  for a successful loop.

## 9. Reference: extras.train v8 schema

```jsonc
{
  "preset_origin": "llama3_8b",          // R01
  "scale_down_factor": 0.125,            // R02
  "memory_matrix_cell_used":
    {"topology": "m3_ultra_solo", "precision": "bf16"},  // R03
  "compile_trace": {                                     // R06
    "fused_groups_count": 3, "dlpack_crossings": 1,
    "materialised_ops_count": 2
  },
  "sync_check": {"redundant_syncs_count": 0},            // R07
  "feature_injections":
    ["mtp_weighted", "engram"],                          // R08
  "data_source": {                                       // R09 / R10
    "kind": "hf_quickstart",
    "dataset_id": "HuggingFaceFW/fineweb-edu",
    "n_tokens": 100032
  }
}
```

## 10. Smoke test

The pure-Python integration test
`tests/v4/test_raschka_full_loop_integration.py` covers steps 1-3-9
without any UI:

```bash
.venv/bin/python -m pytest tests/v4/test_raschka_full_loop_integration.py -v
```

This proves the RPC plumbing end-to-end. The full UI loop is in
`vbgui/e2e/scenarios/v8_raschka_full_loop.spec.ts` (R12).
