# UI → API → Training: Closure Spec

**Companion to**: `docs/UI-TO-TRAIN-AUDIT-2026-05-23.md`
**Goal**: ship E-AUDIT-01..08 to flip the system from "working dev tool" to "fully-promised operator UI".

## Design principles

1. **Backend remains source of truth** — every UI gap closes by adding a UI surface to an existing RPC, never by reimplementing logic UI-side.
2. **Test before merge** — every closure ships at least one vitest + one Playwright assertion.
3. **No monkeypatch** — production seams or real factory functions only.
4. **Backward-compat** — existing fixture-path workflow keeps working alongside new file-picker.

---

## E-AUDIT-01: DataInspector file picker (P0)

### Promise
User points at *any* parquet file on local filesystem and previews it. Today: must type a path into a text field.

### Design
Two-stage UX:
1. **Pick from fixtures** (existing) — text field + matrix dropdown.
2. **Upload from disk** (new) — `<input type="file" accept=".parquet">` → reads file via `FileReader.readAsArrayBuffer` → POSTs to `/upload/parquet` → backend writes to `/tmp/vbgui_uploads/<uuid>.parquet` → returns path string → text field auto-fills.

### Backend changes
- New endpoint `POST /upload/parquet` in `cppmega_v4/jsonrpc/server.py`:
  ```python
  @app.post("/upload/parquet")
  async def upload_parquet(file: UploadFile) -> dict[str, str]:
      # size-cap 2 GiB, suffix-validate, write to UPLOAD_DIR/uuid.parquet
      return {"path": str(target)}
  ```
- `UPLOAD_DIR` configurable via env (`VBGUI_UPLOAD_DIR`, default `/tmp/vbgui_uploads`).
- TTL cleanup: files older than 24h purged on next upload.

### UI changes
- `DataInspector.tsx:194-198`: add file input alongside text input:
  ```tsx
  <input type="file" accept=".parquet"
         data-testid="data-inspector-file-upload"
         onChange={async (e) => {
           const f = e.target.files?.[0]; if (!f) return;
           const buf = await f.arrayBuffer();
           const res = await fetch("/upload/parquet", {
             method: "POST", body: new Blob([buf]),
             headers: {"X-Filename": f.name},
           });
           const {path} = await res.json();
           setParquetPath(path);
         }} />
  ```

### Tests
- vitest: mock fetch, assert file → onChange → setParquetPath flow.
- pytest: TestClient POST a 1 KiB synthetic parquet, assert path written + readable.
- Playwright: upload `tests/fixtures/sample.parquet` via `page.setInputFiles`, assert preview renders.

### Acceptance
- `<input type="file">` present with testid `data-inspector-file-upload`.
- Uploaded file readable via existing `data.preview_parquet` RPC unchanged.
- Path text field auto-populated post-upload.

---

## E-AUDIT-02: Edge validation (P0)

### Promise
When user drags an edge between two incompatible bricks, the UI rejects the connection (or marks it red with a tooltip). Today: edges accepted silently, then verify fails at server-side.

### Design
`FlowCanvas` already accepts `isValidConnection?: IsValidConnection` (React Flow built-in prop). The function returns `boolean` synchronously. Wire it through from `App.tsx`.

### Source of truth
`cppmega_v4/buildspec/shape_contract.py` already exports `is_compatible(src_kind, dst_kind, dim_env)`. Expose via RPC if not already:
```python
@register("brick.compatible")
def brick_compatible(src_kind: str, dst_kind: str, dim_env: dict) -> bool: ...
```

But synchronous prop can't call RPC. Cache contract pairs client-side:
- On preset load / brick add, fetch `catalog.list_options("compatible_edges")` once → `Set<"src→dst">`.
- `isValidConnection({source, target}) → contractSet.has(`${srcKind}→${dstKind}`)`.

### UI changes
- `App.tsx:1215`: pass `isValidConnection={validateEdge}` to FlowCanvas.
- `App.tsx`: add `validateEdge` callback closure over `contractSet` state.
- Visual: rejected drops get React Flow's built-in "no" cursor.

### Backend changes
- New `catalog.list_options("compatible_edges")` returns list of `{src_kind, dst_kind}` derived from `shape_contract.SHAPE_CONTRACTS`.

### Tests
- vitest: pass `isValidConnection` returning false, drop incompatible pair, assert edge not in state.
- pytest: `catalog.list_options("compatible_edges")` returns expected pairs for all 25 bricks.
- Playwright: drop `attention` + `embed_lookup` (incompatible), attempt connect, assert no edge added.

### Acceptance
- Incompatible edge drops rejected client-side without round-trip.
- Compatible edges still flow through to verify.

---

## E-AUDIT-03: Checkpoint history UI (P1)

### Promise
User picks "Resume from..." dropdown, sees list of past checkpoints with arch_hash + step + timestamp, clicks → ckpt_load_path populates.

### Backend changes
New RPC `ckpt.list_history(directory: str = ".") → list[CkptEntry]`:
```python
class CkptEntry(BaseModel):
    path: str
    mtime: float
    size_bytes: int
    arch_hash: str | None        # via read_ckpt_metadata
    opt_kind: str | None
    global_step: int | None
    has_opt_sidecar: bool        # <path>.opt exists
```

Scan logic:
- Recursively walk `directory` for `*.safetensors`.
- For each, read header metadata (fast, no full load).
- Sort by mtime descending.
- Cap at 100 entries.

### UI changes
- New `CheckpointHistoryDropdown.tsx` component:
  - `<button>` opens dropdown listing entries.
  - Each row shows `<basename> · arch_hash[:8] · step N · mtime`.
  - Click → sets `ckpt_load_path` text field.
- Wire into `TopBar.tsx` next to ckpt path input.
- Three new checkboxes/dropdowns in `TrainOptionsPanel.tsx`:
  - `compress` dropdown: none / weights-int8 / opt-fp16 / both
  - `ckpt_strict` checkbox
  - `opt_state_strict` checkbox

### V7-H05 mid-run trigger
- Backend: add `_TRIGGER_CHECKPOINT_TOKENS` set + WS message `ckpt.trigger {token, path}` → adds to set → train loop checks each step → saves + opens sidecar.
- UI: existing `TrainLiveControls` button already wired; just point the callback at the new WS message.

### Tests
- pytest: 5 synthetic checkpoints in tmpdir → `ckpt.list_history` returns sorted-by-mtime with correct metadata.
- vitest: dropdown renders entries, click sets path.
- Playwright: end-to-end save → list → click → load → verify weights identical.

### Acceptance
- `ckpt.list_history` RPC registered + tested.
- `CheckpointHistoryDropdown` in TopBar.
- 3 new train-options controls.
- V7-H05 button actually triggers mid-run save.

---

## E-AUDIT-04: 5 gallery presets (P1)

### Backend changes
Add 4 preset factories to `cppmega_v4/architectures/presets.py`:

1. `def gpt2_xl_specs(...)` — uses `abs_pos_embed` brick.
2. Wire existing `tiny_aya_parallel_specs()` into `PRESETS` dict with `"tiny_aya"` key.
3. `def xlstm_7b_specs(...)` — uses `mlstm` brick.
4. `def gemma_4_e2b_specs(...)` + `def gemma_4_e4b_specs(...)` — uses `per_layer_embed` brick.

### Tests
- `tests/v4/test_galcov_stage_d.py`: remove 5 xfail markers, assert all 71 entries map to a preset.

### Acceptance
- Galcov coverage 71/71 (100%).
- Each preset passes verify + dry_forward.

---

## E-AUDIT-05: smoke_zero1 CLI (P1)

### Design
`cppmega_mlx/cli/smoke_zero1.py`:
- argparse `--hosts`, `--world-size`, `--num-steps`, `--out`.
- Wraps `mlx.launch -n W --hosts H1,H2,...` with the existing `scripts/bench_zero1_loopback.py` payload.
- Each rank runs ZeRO-1 wrapper over a tiny MLP, captures per-rank loss + reduce_ms.
- Rank 0 collects + writes `out.json` with `{world_size, hosts, losses, reduce_ms, parity_max_delta}`.

### Acceptance
- `python -m cppmega_mlx.cli.smoke_zero1 --hosts 127.0.0.1,127.0.0.1` runs loopback, writes receipt.
- Existing `bench/baselines/zero1_loopback_2proc_m4.json` regenerable via CLI.

---

## E-AUDIT-06..08: smaller polish items

- **06.1 B/S controls**: add two number inputs to `TrainOptionsPanel.tsx`, override spec.dim_env values before pipeline.run.
- **06.2 loss_scaler nesting fix**: change `stages.py:2245` to flatten `extras["loss_scaler_overflows"] = loss_scaler_overflow_steps` (keep nested key for compat).
- **06.3 plasticity panels**: extend `TrainExtrasOverlay.tsx` with conditional sub-panels.
- **07.1 readTrainExtras**: read all 16 keys; assert all in `12_train_convergence.spec.ts`.
- **07.2 manual edge e2e**: extend `08_manual_drag_drop.spec.ts` with React Flow `connectionLine` interaction.
- **07.3 adapter UI e2e**: new spec exercising GotchasTab fix-suggestion buttons.
- **08.1 tokenizer-at-preview check**: extend `data.preview_parquet` to accept `tokenizer_source` arg → returns `roundtrip_pass_rate` field → UI shows badge.
- **08.2 peak-RSS test**: generate 1.5 GB synthetic checkpoint → `load_auto` → assert mx peak memory < 200 MiB.
- **08.3 multi-node receipt**: BLOCKED on peer-48 hardware; ticket carries notes only.

---

## Rollout plan

Suggested order (parallelizable where independent):
1. **Wave 1 (P0, parallel)**: E-AUDIT-01 + E-AUDIT-02 — both critical, both touch UI + RPC.
2. **Wave 2 (P1, parallel)**: E-AUDIT-03 + E-AUDIT-04 + E-AUDIT-05.
3. **Wave 3 (P2/P3)**: E-AUDIT-06..08 polish & coverage.

Each wave gates the next on green vitest + pytest + Playwright regression.
