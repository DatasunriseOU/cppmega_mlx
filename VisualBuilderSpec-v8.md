# VisualBuilderSpec-v8 — Raschka-Loop Technical Spec

Companion to `VisualBuilderPlan-v8.md`. Defines RPC contracts, extras
schema, file layout, data-testids, and the full e2e Playwright design.

## 1. New RPC contracts

### `architectures.scale_down`

```jsonc
// params
{ "preset": "llama3_8b",
  "target_bytes": 1073741824,    // 1 GB
  "min_hidden": 64,
  "min_layers": 1 }
// result
{ "hidden_size": 256, "num_layers": 4,
  "estimated_bytes": 950000000,
  "specs": [...],                // build_preset_specs output
  "scaled_down_from": { "hidden_size": 8192, "num_layers": 32 } }
```

### `architectures.auto_fit`

```jsonc
// params
{ "preset": "qwen3_dense_4b",
  "host_info": null }   // null → server-side platform.get_info
// result
{ "scaled": <scale_down result>,
  "sharding": <suggest_sharding result>,
  "fits": true,
  "reason": "hidden=384, layers=8, axis=dp×2, peak=3.1 GB / 16 GB" }
```

### `memory.matrix`

```jsonc
// params
{ "spec": <VerifyParams>,
  "topologies": ["h100_8x", "m3_ultra_solo", "gb10_quarter", "tpu_v6e_8"],
  "precisions": ["fp32", "bf16", "fp16", "fp8", "mxfp4"] }
// result
{ "cells": [
    {"topology": "h100_8x", "precision": "bf16", "bytes": 1.2e9,
     "fits": true, "breakdown": {"params": ..., "activations": ...}}
    // ...
  ] }
```

### `compile.trace`

```jsonc
// params
{ "spec": <VerifyParams>, "backend": "tilelang" | "torch_inductor" | "mlx" }
// result
{ "ops": [
    { "name": "matmul", "fused": true, "group": "gemm_softmax_0",
      "materialised": false, "dlpack_boundary": false,
      "backend": "tilelang" },
    // ...
  ],
  "fused_groups": ["gemm_softmax_0", "qk_reduce_sm_scale_1"],
  "dlpack_crossings": 2,
  "materialised_ops": ["mlx.softmax_long"],
  "compile_artifact_path": "/tmp/.tilelang/llama3_8b_0001.metal" }
```

### `sync.check`

```jsonc
// params
{ "spec": <VerifyParams> }
// result
{ "necessary_syncs": [{"after_op": "attention.0", "reason": "..."}],
  "redundant_syncs": [{"after_op": "mlp.1",
                        "reason": "same backend, no boundary crossing"}],
  "advice": [
    {"op": "mlp.1", "fix": "remove mx.eval after this brick",
     "confidence": "high"}
  ],
  "z3_solver_status": "sat",
  "z3_elapsed_ms": 12.3 }
```

### `data.hf_quickstart`

```jsonc
// params
{ "dataset_id": "HuggingFaceFW/fineweb-edu",
  "split": "train",
  "tokenizer": "cppmega_v3",
  "n_tokens": 100000,
  "job_id": "hf-2026-05-23-001" }
// result (sync RPC; data lands on /ws/data/{job_id} for progress)
{ "parquet_path": "/tmp/vbgui/hf-2026-05-23-001.parquet",
  "n_tokens_written": 100032,
  "elapsed_ms": 4231.0 }
```

### `data.github_corpus`

```jsonc
// params
{ "repo_url": "https://github.com/karpathy/nanochat",
  "max_commits": 50,
  "max_tokens": 50000,
  "tokenizer": "cppmega_v3",
  "use_clang": false,            // True → clang_enriched side-channels
  "use_treesitter": true,
  "job_id": "gh-2026-05-23-001" }
// result
{ "parquet_path": "...",
  "n_tokens_written": ...,
  "side_channels": ["doc_ids", "token_ids", "ast_node_kinds"],
  "elapsed_ms": ... }
```

## 2. Extras schema additions

`extras.train` grows:

```jsonc
{
  "preset_origin": "llama3_8b",          // R01
  "scale_down_factor": 0.125,            // R02
  "memory_matrix_cell_used":
    {"topology": "m3_ultra_solo", "precision": "bf16"},  // R03
  "compile_trace": {
    "fused_groups_count": 3,
    "dlpack_crossings": 1,
    "materialised_ops_count": 2
  },                                     // R06
  "sync_check": {"redundant_syncs_count": 0},   // R07
  "feature_injections":
    ["mtp_weighted_k2", "engram"],       // R08
  "data_source": {
    "kind": "hf_quickstart",
    "dataset_id": "HuggingFaceFW/fineweb-edu",
    "n_tokens": 100032
  }                                      // R09 / R10
}
```

## 3. New backend files

| Path | Purpose |
|------|---------|
| `cppmega_v4/architectures/preset_training_defaults.py` | R01: lr/bs/sched per preset |
| `cppmega_v4/architectures/scale_down.py` | R02 |
| `cppmega_v4/jsonrpc/memory_matrix_method.py` | R03 |
| `cppmega_v4/jsonrpc/auto_fit_method.py` | R04 |
| `cppmega_mlx/quant/mxfp4_metal.py` | R05 |
| `cppmega_v4/jsonrpc/compile_trace_method.py` | R06 |
| `cppmega_v4/spec/sync_checker.py` | R07 |
| `scripts/data/hf_quickstart.py` | R09 |
| `scripts/data/github_corpus.py` | R10 |

## 4. New UI components + testids

| Component | Path | Key data-testids |
|-----------|------|------------------|
| GalleryScaleDownSlider | `vbgui/src/components/GalleryScaleDownSlider.tsx` | `gallery-scaledown-slider`, `gallery-scaledown-est-bytes`, `gallery-scaledown-apply` |
| AutoFitButton | inline in GalleryTab | `gallery-auto-fit`, `gallery-auto-fit-result` |
| MemoryMatrix | `vbgui/src/components/sidebar/MemoryMatrixTab.tsx` | `memory-matrix`, `memory-matrix-cell-{topo}-{prec}`, `memory-matrix-cell-fits-{topo}-{prec}` |
| CompileTracePanel | `vbgui/src/components/CompileTracePanel.tsx` | `compile-trace`, `compile-trace-op-{i}`, `compile-trace-fused-group-{name}`, `compile-trace-dlpack-crossings` |
| SyncCheckBadge | inline in CompileTracePanel | `sync-check-badge`, `sync-check-redundant-count`, `sync-check-advice-modal` |
| FeatureInjectionBar | `vbgui/src/components/FeatureInjectionBar.tsx` | `feature-injection-bar`, `feature-injection-dropdown`, `feature-injection-apply`, `feature-injection-applied-list` |
| HFQuickStartModal | `vbgui/src/components/HFQuickStartModal.tsx` | `hf-quickstart-modal`, `hf-quickstart-dataset-id`, `hf-quickstart-n-tokens`, `hf-quickstart-run`, `hf-quickstart-result-path` |
| GitHubCorpusModal | shared with HF modal via tabs | `github-corpus-tab`, `github-corpus-repo-url`, `github-corpus-max-commits`, `github-corpus-run` |

## 5. End-to-end Playwright design (R12)

### 5.1 Single-preset full-loop (canonical)

`vbgui/e2e/scenarios/v8_raschka_full_loop.spec.ts`:

```typescript
test("V8 R12: full Raschka loop on llama3_8b — preset → defaults → scale → memory matrix → MTP inject → HF data → 4 steps", async ({ page }) => {
  test.setTimeout(120_000);
  const consoleFrames: Array<{type: string; text: string}> = [];
  page.on("console", (m) => consoleFrames.push({type: m.type(), text: m.text()}));
  page.on("pageerror", (e) => consoleFrames.push({type: "pageerror", text: String(e)}));

  await gotoApp(page);

  // R01: pick preset → paper defaults auto-fill.
  await selectPreset(page, "llama3_8b");
  await expect(page.getByTestId("optim-lr-input")).toHaveValue(/3e-04/);
  await expect(page.getByTestId("schedule-kind")).toHaveValue("wsd");

  // R02: scale-down to a 1GB target.
  await clickTab(page, "gallery");
  await page.getByTestId("gallery-scaledown-slider").fill("1073741824");
  await expect(page.getByTestId("gallery-scaledown-est-bytes"))
    .toContainText(/< 1 GB/);
  await page.getByTestId("gallery-scaledown-apply").click();

  // R03: memory matrix shows ≥ 1 green cell.
  await clickTab(page, "memory");
  await expect(page.getByTestId("memory-matrix")).toBeVisible();
  const greenCount = await page.locator(
    "[data-testid^='memory-matrix-cell-fits-'][data-fits='true']").count();
  expect(greenCount).toBeGreaterThan(0);

  // R06: compile trace exists.
  await clickTab(page, "compile");
  await expect(page.getByTestId("compile-trace")).toBeVisible();
  const fusedCount = await page.locator(
    "[data-testid^='compile-trace-fused-group-']").count();
  expect(fusedCount).toBeGreaterThanOrEqual(1);

  // R07: sync-check badge.
  await expect(page.getByTestId("sync-check-badge")).toBeVisible();

  // R08: inject MTP.
  await clickTab(page, "canvas");
  await page.getByTestId("feature-injection-dropdown").selectOption("mtp_weighted");
  await page.getByTestId("feature-injection-apply").click();
  await expect(page.getByTestId("feature-injection-applied-list"))
    .toContainText("mtp_weighted");

  // R09: HF quickstart data.
  await clickTab(page, "data");
  await page.getByTestId("hf-quickstart-modal-open").click();
  await page.getByTestId("hf-quickstart-dataset-id")
    .fill("HuggingFaceFW/fineweb-edu");
  await page.getByTestId("hf-quickstart-n-tokens").fill("8192");
  await page.getByTestId("hf-quickstart-run").click();
  await expect(page.getByTestId("hf-quickstart-result-path"))
    .toBeVisible({ timeout: 60_000 });

  // R11: train 4 steps.
  await page.getByTestId("run-pipeline-toggle").click();
  await page.getByTestId("train-num-steps").fill("4");
  await page.getByTestId("run-pipeline-train").click();
  await expect(page.getByTestId("live-train-panel")).toBeVisible({
    timeout: 30_000 });
  await expect(page.getByTestId("live-train-panel-toast")).toBeVisible({
    timeout: 90_000 });

  // R12: assert extras.train carries every v8 trace marker.
  const trainExtras = await readTrainExtras(page);
  expect(trainExtras.preset_origin).toBe("llama3_8b");
  expect(trainExtras.scale_down_factor).toBeLessThan(1.0);
  expect(trainExtras.memory_matrix_cell_used).toBeDefined();
  expect(trainExtras.compile_trace.fused_groups_count).toBeGreaterThanOrEqual(1);
  expect(trainExtras.feature_injections).toContain("mtp_weighted");
  expect(trainExtras.data_source.kind).toBe("hf_quickstart");

  // Console capture — annotation only.
  test.info().annotations.push({
    type: "console-frames",
    description: JSON.stringify(consoleFrames.slice(-50))
  });
});
```

### 5.2 Matrix variant (R12.3)

Same body, parametrized over `[llama3_8b, qwen3_dense_4b, gemma3_27b,
deepseek_v3, gpt_oss_20b]`. Skips R09 when the preset's tokenizer
isn't available in tokenizer.list_presets; uses the GitHub-corpus
branch as fallback (R10).

### 5.3 UAT manual script

`docs/uat/v8_raschka_full_loop.md` mirrors the e2e flow with
expected screenshots at each step.

## 6. Compile-trace heuristic (R06 implementation note)

```
fused        := op is part of a TileLang/inductor compile group
dlpack       := op crosses MLX↔Torch boundary via DLPack
materialised := op forces an mx.eval / cudaSynchronize
backend      := one of {mlx, tilelang, torch_inductor}
```

For the MLX backend the trace is built by intercepting
`mx.compile` graph emission; for TileLang via `dispatch_lower(prim,
return_msl=True)`; for torch.compile via `inspect_fx_graph`. Each
backend implementation lives behind a `BackendTracer` protocol so we
can add ONNX or others without touching the RPC.

## 7. Z3 sync-checker (R07 implementation note)

Encoded constraints (z3 boolean and Int variables):

- `produced_on(op) ∈ {mlx, tilelang, torch}`
- `consumed_on(op)` (forward graph)
- `needs_sync(op) ⇐ produced_on(prev) ≠ consumed_on(op)`
- `mx.eval(op)` flag

Solver returns SAT for the spec; any `needs_sync=False ∧ has_mx_eval=True`
op is a *redundant* sync and lands in `advice`.

## 8. MXFP4 (R05) numerical contract

Block size 16; per-block scale stored as fp8 e4m3; mantissas as e2m1
nibbles (2 mantissas / byte). Round-trip RMSE bound on a fp32
normal tensor: ≤ 5% of bf16→fp16→bf16 round-trip RMSE (tighter
near zero per the e2m1 dense-zero range).

Apple MLX exposes the underlying e2m1 path via `mx.metal_kernel` with
a custom shader; the helper `cppmega_mlx/quant/mxfp4_metal.py`
wraps the boilerplate.

## 9. Per-preset paper-defaults table (R01 schema)

`cppmega_v4/architectures/preset_training_defaults.py`:

```python
@dataclass(frozen=True)
class TrainingDefaults:
    lr: float
    batch_size: int
    schedule: str         # constant | linear_warmup | cosine | wsd
    warmup_steps: int
    betas: tuple[float, float] | None
    gradient_clip: float
    mixed_precision: bool
    optimizer: str        # adamw | muon | muon_adamw_hybrid | ...
    source_paper_url: str

DEFAULTS: dict[str, TrainingDefaults] = {
    "llama3_8b": TrainingDefaults(
        lr=3e-4, batch_size=1024, schedule="wsd",
        warmup_steps=2000, betas=(0.9, 0.95),
        gradient_clip=1.0, mixed_precision=True,
        optimizer="adamw",
        source_paper_url="https://arxiv.org/abs/2407.21783"),
    "qwen3_dense_8b": TrainingDefaults(
        lr=3e-4, batch_size=2048, schedule="cosine",
        warmup_steps=1000, betas=(0.9, 0.95),
        gradient_clip=1.0, mixed_precision=True,
        optimizer="adamw",
        source_paper_url="https://arxiv.org/abs/2412.15115"),
    # ... ≥ 30 entries
}
```

## 10. Acceptance counters (mirrors plan §6)

- 36 sub-tickets across 12 R-epics
- 7 new RPC methods + 1 new WS endpoint (`/ws/data/{job_id}`)
- 9 backend files, 7 UI components
- ≈ 50 new tests (25 pytest, 10 vitest, 15 Playwright)
- 2 new docs (`raschka_full_loop.md`, `uat/v8_raschka_full_loop.md`)

## 11. Out of scope (deferred to v9+)

- NVFP4 (Hopper/Blackwell only)
- Multi-host launcher UI
- Cross-preset auto-arch search
- ONNX export
