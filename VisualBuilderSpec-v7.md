# VisualBuilderSpec-v7 — Technical Spec (UI→API→Training E2E Closure)

Companion to `VisualBuilderPlan-v7.md`. Defines API contracts, schema
deltas, file layout, data-testids for the 8-gap V7-Q audit closure epic.

## 1. RPC contract deltas (post-v7-Q)

### New methods

```
POST /upload/parquet        Q01  body: multipart file, returns {"path": str}
ckpt.list_history           Q03  args: {directory: str} → list[CkptEntry]
ckpt.trigger                Q03.4 (WS) args: {token, path}
catalog.list_options('compatible_edges')   Q02  → list[CompatibleEdgePair]
data.preview_parquet        Q08.1 +arg: tokenizer_source? → +field: roundtrip_pass_rate
```

### Schema additions

```jsonc
// Q03.1 — ckpt.list_history response
class CkptEntry(BaseModel):
    path: str
    mtime: float
    size_bytes: int
    arch_hash: str | None        // via read_ckpt_metadata
    opt_kind: str | None
    global_step: int | None
    has_opt_sidecar: bool        // <path>.opt exists

// Q02 — catalog.list_options('compatible_edges') response
class CompatibleEdgePair(BaseModel):
    src_kind: str
    dst_kind: str

// Q08.1 — data.preview_parquet response extension
class PreviewParquetResponse(BaseModel):
    ...existing,
    roundtrip_pass_rate: float | None  // None when tokenizer_source not provided
```

## 2. stage_train extras schema deltas (post-v7-Q)

```jsonc
{
  // === v6 baseline preserved ===

  // === v7-Q additions ===
  "loss_scaler_overflows": [int],          // Q06.2 — FLATTENED from
                                            // loss_scaler.overflow_steps so
                                            // LossChart overflow markers
                                            // render. Nested key kept.

  "plasticity": {                          // Q06.3 — was generic, now styled
    "fire_rate": float | null,
    "dash_count": int | null,
    "redo_threshold": float | null
  },
  "mtp": {                                 // Q06.3 — per-head losses
    "k": int,
    "head_losses": [float],
    "beta": float
  },
  "ifim": {                                // Q06.3
    "instr_loss": float,
    "instr_token_count": int
  },
  "mhc": {                                 // Q06.3 — multi-head consistency
    "consistency_loss": float,
    "heads_correlated": float
  }
}
```

## 3. Backend changes by gap

### Q01 — Parquet upload endpoint

```python
# cppmega_v4/jsonrpc/server.py
import os, uuid
from fastapi import UploadFile, HTTPException
from pathlib import Path

UPLOAD_DIR = Path(os.environ.get("VBGUI_UPLOAD_DIR", "/tmp/vbgui_uploads"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
MAX_BYTES = 2 * 1024 * 1024 * 1024  # 2 GiB

@app.post("/upload/parquet")
async def upload_parquet(file: UploadFile) -> dict[str, str]:
    if not file.filename.endswith(".parquet"):
        raise HTTPException(400, "expected .parquet suffix")
    data = await file.read()
    if len(data) > MAX_BYTES:
        raise HTTPException(413, f"file >2GiB ({len(data)})")
    target = UPLOAD_DIR / f"{uuid.uuid4()}.parquet"
    target.write_bytes(data)
    _purge_older_than(UPLOAD_DIR, days=1)
    return {"path": str(target)}
```

### Q02 — Compatible-edges catalog

```python
# cppmega_v4/jsonrpc/catalog_methods.py
from cppmega_v4.buildspec.shape_contract import SHAPE_CONTRACTS

@register("catalog.list_options")
def list_options(name: str) -> dict:
    if name == "compatible_edges":
        return {
            "name": "compatible_edges",
            "items": [
                {"src_kind": s, "dst_kind": d}
                for (s, d), contract in SHAPE_CONTRACTS.items()
                if contract.is_strictly_valid()
            ]
        }
    # ...existing branches
```

### Q03 — Checkpoint history scan

```python
# cppmega_v4/jsonrpc/ckpt_history_method.py
from cppmega_v4.runtime.checkpoint_metadata import read_ckpt_metadata
from pathlib import Path

@register("ckpt.list_history")
def list_history(directory: str = ".") -> dict:
    root = Path(directory).expanduser().resolve()
    entries = []
    for path in root.rglob("*.safetensors"):
        if path.suffix.endswith(".opt"):
            continue  # sidecars exposed via has_opt_sidecar
        try:
            meta = read_ckpt_metadata(str(path)) or {}
            arch = meta.get("arch", {}) if isinstance(meta.get("arch"), dict) else {}
            opt = meta.get("opt", {}) if isinstance(meta.get("opt"), dict) else {}
            entries.append({
                "path": str(path),
                "mtime": path.stat().st_mtime,
                "size_bytes": path.stat().st_size,
                "arch_hash": arch.get("config_hash"),
                "opt_kind": opt.get("kind"),
                "global_step": opt.get("global_step"),
                "has_opt_sidecar": path.with_suffix(path.suffix + ".opt").is_file(),
            })
        except Exception:
            pass  # skip unreadable
    entries.sort(key=lambda e: e["mtime"], reverse=True)
    return {"entries": entries[:100]}
```

### Q03.4 — Mid-run checkpoint WS trigger

```python
# cppmega_v4/runner/stages.py — extend abort-token pattern
_TRIGGER_CHECKPOINT_TOKENS: set[str] = set()

def request_checkpoint(token: str, path: str) -> None:
    _TRIGGER_CHECKPOINT_TOKENS.add(f"{token}|{path}")

# Inside training loop, after each step:
for entry in list(_TRIGGER_CHECKPOINT_TOKENS):
    tok, save_path = entry.split("|", 1)
    if tok == abort_token:
        _save_compressed(flat, save_path, ...)
        _TRIGGER_CHECKPOINT_TOKENS.discard(entry)
```

### Q04 — 5 preset factories

```python
# cppmega_v4/architectures/presets.py
def gpt2_xl_specs(**overrides) -> list[BrickSpec]:
    """GPT-2 XL: abs_pos_embed + 48 transformer blocks + lm_head."""
    n_layer = overrides.get("n_layer", 48)
    hidden = overrides.get("hidden", 1600)
    return _stack(
        [{"kind": "abs_pos_embed", "params": {"max_seq": 1024}}]
        + _transformer_block_x_n(n_layer, hidden, num_heads=25)
        + [{"kind": "rmsnorm"}, {"kind": "lm_head"}]
    )

def xlstm_7b_specs(**overrides) -> list[BrickSpec]:
    """xLSTM 7B: mlstm-based long-range recurrence."""
    return _stack([{"kind": "mlstm", "params": {...}}, ...])

def gemma_4_e2b_specs(**overrides) -> list[BrickSpec]:
    """Gemma 4 E2B: per_layer_embed + gqa attention."""
    return _stack([{"kind": "per_layer_embed", "params": {...}}, ...])

def gemma_4_e4b_specs(**overrides) -> list[BrickSpec]:
    """Gemma 4 E4B: larger Gemma 4 variant."""
    ...

PRESETS["gpt2_xl"]      = gpt2_xl_specs
PRESETS["tiny_aya"]     = tiny_aya_parallel_specs  # already exists, just wire
PRESETS["xlstm_7b"]     = xlstm_7b_specs
PRESETS["gemma_4_e2b"]  = gemma_4_e2b_specs
PRESETS["gemma_4_e4b"]  = gemma_4_e4b_specs
```

### Q05 — smoke_zero1 CLI

```python
# cppmega_mlx/cli/smoke_zero1.py
import argparse, json, subprocess, sys
from pathlib import Path

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--hosts", required=True,
                   help="comma-separated host list (e.g. 127.0.0.1,127.0.0.1)")
    p.add_argument("--world-size", type=int, default=0,
                   help="defaults to len(hosts.split(','))")
    p.add_argument("--num-steps", type=int, default=16)
    p.add_argument("--out", default="bench/baselines/zero1_smoke.json")
    args = p.parse_args()
    hosts = args.hosts.split(",")
    world_size = args.world_size or len(hosts)
    subprocess.run([
        "mlx.launch", "-n", str(world_size),
        "--hosts", args.hosts,
        sys.executable, "-m", "scripts.bench_zero1_loopback",
        "--num-steps", str(args.num_steps),
        "--out", args.out,
    ], check=True)
```

### Q06.2 — Flatten loss_scaler overflow

```python
# cppmega_v4/runner/stages.py:~2245
extras["loss_scaler"] = {
    "overflow_steps": loss_scaler_overflow_steps,
    ...
}
# Q06.2: also flatten for LossChart overflow markers
extras["loss_scaler_overflows"] = list(loss_scaler_overflow_steps)
```

### Q08.1 — Tokenizer-at-preview check

```python
# cppmega_v4/jsonrpc/data_methods.py — preview_parquet extension
def preview_parquet(parquet_path, *, tokenizer_source=None, ...):
    rows = _load_preview(parquet_path, ...)
    response = {...existing fields...}
    if tokenizer_source:
        tok = _load_tokenizer(tokenizer_source)
        pass_count = sum(
            1 for r in rows
            if tok.decode(r["input_ids"]) == r.get("original_text", "")
        )
        response["roundtrip_pass_rate"] = pass_count / max(1, len(rows))
    return response
```

## 4. Frontend changes

### Q01 — File picker in DataInspector
```tsx
// vbgui/src/components/DataInspector.tsx:194
<div className="dataInspector__pathRow">
  <input data-testid="data-inspector-path" type="text"
         value={parquetPath} onChange={...} />
  <input data-testid="data-inspector-file-upload" type="file"
         accept=".parquet"
         onChange={async (e) => {
           const f = e.target.files?.[0]; if (!f) return;
           const fd = new FormData(); fd.append("file", f);
           const res = await fetch("/upload/parquet", {method: "POST", body: fd});
           const {path} = await res.json();
           setParquetPath(path);
         }} />
</div>
```

### Q02 — Edge validation wiring
```tsx
// vbgui/src/App.tsx
const [contractSet, setContractSet] = useState<Set<string>>(new Set());

useEffect(() => {
  rpc.call("catalog.list_options", {name: "compatible_edges"})
    .then((res) => {
      const items = (res as any).items as Array<{src_kind: string; dst_kind: string}>;
      setContractSet(new Set(items.map(p => `${p.src_kind}->${p.dst_kind}`)));
    });
}, [rpc]);

const validateEdge = useCallback(({source, target}: Connection) => {
  const srcKind = nodes.find(n => n.id === source)?.data?.kind;
  const dstKind = nodes.find(n => n.id === target)?.data?.kind;
  if (!srcKind || !dstKind) return true;  // unknowns fall through to verify
  return contractSet.has(`${srcKind}->${dstKind}`);
}, [nodes, contractSet]);

// at line 1215:
<FlowCanvas
  ...
  isValidConnection={validateEdge}
/>
```

### Q03.2 — Checkpoint history dropdown
```tsx
// vbgui/src/components/CheckpointHistoryDropdown.tsx
export function CheckpointHistoryDropdown({onSelect}: {onSelect: (path: string) => void}) {
  const [open, setOpen] = useState(false);
  const [entries, setEntries] = useState<CkptEntry[]>([]);

  const refresh = useCallback(async () => {
    const res = await rpc.call("ckpt.list_history", {directory: "."});
    setEntries((res as any).entries as CkptEntry[]);
  }, []);

  return (
    <div data-testid="ckpt-history-dropdown">
      <button onClick={() => { setOpen(!open); if (!open) refresh(); }}>
        History ▾
      </button>
      {open && (
        <ul>
          {entries.map(e => (
            <li key={e.path} data-testid={`ckpt-history-row-${e.path}`}
                onClick={() => { onSelect(e.path); setOpen(false); }}>
              <code>{basename(e.path)}</code> ·
              <span>{e.arch_hash?.slice(0,8) ?? "?"}</span> ·
              <span>step {e.global_step ?? "?"}</span> ·
              <span>{fmtMtime(e.mtime)}</span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
```

### Q03.3 — Compress + strict toggles in TrainOptionsPanel
```tsx
<select data-testid="train-opt-compress" value={compress}
        onChange={(e) => setCompress(e.target.value)}>
  <option value="none">none</option>
  <option value="weights-int8">weights-int8</option>
  <option value="opt-fp16">opt-fp16</option>
  <option value="both">both</option>
</select>
<label>
  <input data-testid="train-opt-ckpt-strict" type="checkbox"
         checked={ckptStrict} onChange={(e) => setCkptStrict(e.target.checked)} />
  ckpt_strict
</label>
<label>
  <input data-testid="train-opt-opt-state-strict" type="checkbox"
         checked={optStateStrict} onChange={(e) => setOptStateStrict(e.target.checked)} />
  opt_state_strict
</label>
```

### Q06.1 — B + S inputs
```tsx
// TrainOptionsPanel.tsx
<input data-testid="train-opt-B" type="number" min={1} max={32}
       value={trainB} onChange={...} />
<input data-testid="train-opt-S" type="number" min={1} max={8192}
       value={trainS} onChange={...} />
```

### Q06.3 — Plasticity/MTP/IFIM/MHC sub-panels
```tsx
// TrainExtrasOverlay.tsx
{extras.plasticity && <PlasticityPanel data={extras.plasticity} />}
{extras.mtp && <MTPPanel data={extras.mtp} />}
{extras.ifim && <IFIMPanel data={extras.ifim} />}
{extras.mhc && <MHCPanel data={extras.mhc} />}
```

## 5. Required data-testids (post-v7-Q)

| Testid                                            | Q#    | Used by |
|---------------------------------------------------|-------|---------|
| `data-inspector-file-upload`                      | Q01   | UI      |
| `ckpt-history-dropdown`                           | Q03.2 | UI      |
| `ckpt-history-row-{path}`                         | Q03.2 | UI      |
| `train-opt-compress`                              | Q03.3 | UI      |
| `train-opt-ckpt-strict`                           | Q03.3 | UI      |
| `train-opt-opt-state-strict`                      | Q03.3 | UI      |
| `train-opt-B`                                     | Q06.1 | UI      |
| `train-opt-S`                                     | Q06.1 | UI      |
| `extras-panel-plasticity`                         | Q06.3 | UI      |
| `extras-panel-mtp`                                | Q06.3 | UI      |
| `extras-panel-ifim`                               | Q06.3 | UI      |
| `extras-panel-mhc`                                | Q06.3 | UI      |
| `data-inspector-roundtrip-badge`                  | Q08.1 | UI      |

## 6. Test plan (file layout)

```
cppmega.mlx/tests/v7_q/
  test_upload_parquet_endpoint.py       # Q01 backend
  test_catalog_compatible_edges.py      # Q02 backend
  test_ckpt_list_history.py             # Q03.1 RPC
  test_ckpt_trigger_mid_run.py          # Q03.4 WS
  test_preset_gallery_71.py             # Q04 unblock xfails
  test_smoke_zero1_cli.py               # Q05 loopback receipt
  test_loss_scaler_overflows_flat.py    # Q06.2 backend
  test_preview_parquet_tokenizer.py     # Q08.1 backend
  test_checkpoint_peak_rss_1gib.py      # Q08.2 perf regression

vbgui/tests/                            # vitest
  DataInspectorFileUpload.test.tsx      # Q01
  FlowCanvasEdgeValidation.test.tsx     # Q02
  CheckpointHistoryDropdown.test.tsx    # Q03.2
  TrainOptionsPanelCompress.test.tsx    # Q03.3
  TrainOptionsPanelBS.test.tsx          # Q06.1
  TrainExtrasOverlayPanels.test.tsx     # Q06.3
  readTrainExtras16Keys.test.ts         # Q07.1

vbgui/e2e/scenarios/                    # Playwright
  74_manual_edge_connect.spec.ts        # Q07.2
  75_adapter_suggestions.spec.ts        # Q07.3
  76_file_upload.spec.ts                # Q01 e2e
  77_edge_rejection.spec.ts             # Q02 e2e
  78_checkpoint_history.spec.ts         # Q03 e2e
  79_gallery_71_smoke.spec.ts           # Q04 e2e

cppmega_mlx/cli/
  smoke_zero1.py                        # Q05

vbgui/src/components/
  CheckpointHistoryDropdown.tsx         # Q03.2
  panels/PlasticityPanel.tsx            # Q06.3
  panels/MTPPanel.tsx                   # Q06.3
  panels/IFIMPanel.tsx                  # Q06.3
  panels/MHCPanel.tsx                   # Q06.3

cppmega_v4/jsonrpc/
  ckpt_history_method.py                # Q03.1
  (server.py)                           # Q01 endpoint
  (catalog_methods.py)                  # Q02 branch
  (data_methods.py)                     # Q08.1 extension

cppmega_v4/architectures/
  (presets.py)                          # Q04 — 4 new factories + tiny_aya wire
```

## 7. Honest categorisation (same criteria as v6)

Every new test gets one label:

**🟢 math-effect**: assertion involves numerical value (loss continuation,
roundtrip rate, peak-RSS bytes, parity delta). All Q01/Q05/Q08
tests qualify.

**🟡 propagation**: only checks that a UI string reaches extras as the
same string. Allowed only for genuine config fields. Q03 path-fill
and Q06.1 B/S inputs qualify.

**🔴 decorative**: UI mutation has no observable effect. ZERO target.

## 8. Out of scope (v8+)

- Real cross-Mac ZeRO-1 receipt (BLOCKED on peer-48 hardware)
- 1B+ param training matrix on Mac
- A/B hyperparameter search UI
- LoRA / PEFT adapter wiring
- Tokenizer mutation UI (read-only by design per Lane 3)
- Sophia / Adafactor / Tiger / AdEMAMix (deferred per Lane 7)
- Live inference serving from trained checkpoint
- Model weight quantisation post-train

## 9. Acceptance counters (mirror Plan §6)

| Surface                                | Before v7-Q | Target v7-Q |
|----------------------------------------|-------------|-------------|
| pytest                                 | ~2540       | ≥2570       |
| vitest                                 | ~428        | ≥440        |
| Playwright deep e2e cells              | ~1170       | ≥1175       |
| Raschka gallery coverage               | 66/71 (93%) | 71/71 (100%)|
| extras keys read by e2e                | 7/16        | 16/16       |
| P0 audit gaps                          | 2           | 0           |
| P1 audit gaps                          | 3           | 0           |
| Hardware-blocked gaps                  | 1           | 1 (external)|
