# VisualBuilderSpec-v4 — Technical Spec (Real Data Flow + Remaining Coverage)

Companion to `VisualBuilderPlan-v4.md`. Defines API contracts, file layout,
data-testids, and extras schema for the 14-stage v4 epic.

## 1. Backend changes (`cppmega_v4/runner/stages.py`)

### 1.1 stage_train opts surface (post-v4)

```python
opts = {
  "num_steps": int = 2,
  "lr": float = 1e-3,
  "vocab_size": int = 256,
  "S": int = 8, "B": int = 1,
  # V3-2 (existing):
  "parquet_path": str | None,
  # V4-2 (new):
  "tokenizer_path": str | None,
  # V4-10 (new):
  "side_channels": dict[str, list[int]] | None,
}
```

### 1.2 Tokenizer integration (V4-2)

```python
def _tokenize_parquet_text(
    parquet_path: str, tokenizer_path: str, n_tokens: int,
) -> tuple[list[int], str | None]:
    """If parquet has a 'text' column, load tokenizer and encode rows
    until n_tokens collected. Returns (tokens, basename) or ([], None)
    on any failure. Falls through cleanly to V3-2 raw-int path."""
    ...
```

In stage_train: if `tokenizer_path` provided and parquet has text column,
use tokenize path; else V3-2 raw-int path; else synthetic.

### 1.3 extras additions

```jsonc
{
  // existing v3
  "losses": [...], "lr_trajectory": [...],
  "weight_delta_norm": ..., "num_steps": ..., "schedule_kind": ...,
  "optimizer_kind": ..., "data_source": ..., "token_count": ...,
  "model_summary": {...},
  // v4 new
  "tokenizer_used": string | null,         // V4-2 basename
  "loss_kind": string,                     // V4-7
  "side_channels_observed": [string],      // V4-10
  "muon_group_size": int | null,           // V4-9 (only when hybrid)
  "adamw_group_size": int | null,          // V4-9 (only when hybrid)
  "inference_probe": {                     // V4-11
    "l2_diff": float, "cos_sim": float,
  },
}
```

`model_summary` adds:
```jsonc
{
  // ... v3 fields
  "rewriters_applied": [string],   // V4-8
  "loss_kind": string,             // V4-7 (also at top level for convenience)
}
```

## 2. Frontend changes

### 2.1 DataInspector (V4-1)

```tsx
// New button below data-roundtrip
<button data-testid="data-use-for-train"
        disabled={!loadedPath}
        onClick={() => onUseForTrain(loadedPath)}>
  Use this parquet for training
</button>
```

Props addition: `onUseForTrain(path: string): void`. App.tsx stores in
`trainParquetPath` state. TopBar reads → indicator.

### 2.2 Tokenizer Playground (V4-3)

```tsx
<button data-testid="tokenizer-use-for-train"
        disabled={!loadedTokenizerPath}
        onClick={() => onUseForTrain(loadedTokenizerPath)}>
  Use this tokenizer for training
</button>
```

App.tsx stores in `trainTokenizerPath` state.

### 2.3 TopBar train-data indicator

```tsx
{(trainParquetPath || trainTokenizerPath) && (
  <span data-testid="train-data-source"
        style={{ fontSize: 10, color: "#16a34a" }}>
    {trainParquetPath ? `parquet: ${basename(trainParquetPath)}` : ""}
    {trainTokenizerPath ? ` · tokenizer: ${basename(trainTokenizerPath)}` : ""}
  </span>
)}
```

Default text (when both null): "synthetic".

### 2.4 App.handleRunPipeline

```ts
const stage_options: Record<string, Record<string, unknown>> = {};
if (mode === "train") {
  stage_options.train = {
    num_steps: opts?.num_steps,
    ...(trainParquetPath ? { parquet_path: trainParquetPath } : {}),
    ...(trainTokenizerPath ? { tokenizer_path: trainTokenizerPath } : {}),
    ...(activeSideChannels.length > 0
        ? { side_channels: { ...sideChannelData } } : {}),
  };
}
```

## 3. UI testids added

| Testid                                | Surface         | Used by  |
|---------------------------------------|-----------------|----------|
| `data-use-for-train`                  | DataInspector   | V4-1     |
| `tokenizer-use-for-train`             | Tokenizer pg    | V4-3     |
| `train-data-source`                   | TopBar          | V4-1/3   |
| `loss-kind`                           | LossTab         | V4-7     |
| `loss-mtp-k`, `loss-mtp-beta-{i}`     | LossTab MTP     | V4-7     |
| `rewriter-{name}-toggle`              | RewritersTab    | V4-8     |
| `side-channel-toggle-{name}`          | sidebar         | V4-10    |
| `run-result-extras-train-inference_probe-{l2_diff,cos_sim}` | Modal | V4-11 |

## 4. Tests file plan

```
cppmega.mlx/tests/v4/
  test_stage_train_tokenize.py       # V4-2: parquet text + tokenizer
  test_stage_train_loss_kinds.py     # V4-7: CE/MTP/IFIM/MHC backend
  test_stage_train_rewriters.py      # V4-8: rewriter application
  test_stage_train_hybrid.py         # V4-9: Muon/AdamW bucket sizes
  test_stage_train_inference_probe.py # V4-11: before/after diff
  test_stage_train_side_channels.py  # V4-10: doc_ids, token_ids

vbgui/e2e/scenarios/
  18_real_data_convergence.spec.ts   # V4-4
  19_activation_propagation.spec.ts  # V4-5 (10 cells)
  20_schedule_propagation.spec.ts    # V4-6 (6 cells)
  21_loss_kind_propagation.spec.ts   # V4-7 (4 cells)
  22_rewriter_propagation.spec.ts    # V4-8
  23_hybrid_optimizer_split.spec.ts  # V4-9
  24_side_channels.spec.ts           # V4-10
  25_inference_after_train.spec.ts   # V4-11
  26_cross_arch_brick_mutations.spec.ts # V4-12 (12 × 2 = 24 cells)
  27_real_gotcha.spec.ts             # V4-13

vbgui/src/components/DataInspector.tsx    # V4-1
vbgui/src/components/TokenizerPlayground.tsx # V4-3
vbgui/src/components/TopBar.tsx           # V4-1 (data-source indicator)
vbgui/src/App.tsx                         # V4-1/3/10 state + threading
```

## 5. Inference probe (V4-11) implementation

```python
def _inference_probe(all_modules, hidden, seq, vocab):
    """Snapshot model output at fixed seed before vs after train."""
    probe_input = mx.random.normal((1, seq, hidden), key=mx.random.key(42))
    return forward_layers(all_modules.layers[:-1], probe_input).reshape(-1)

# In stage_train:
probe_before = _inference_probe(all_modules, hidden, seq, vocab_size)
# ... N training steps ...
probe_after = _inference_probe(all_modules, hidden, seq, vocab_size)
l2 = float(mx.linalg.norm(probe_after - probe_before).item())
cos = float((probe_after @ probe_before /
             (mx.linalg.norm(probe_after) * mx.linalg.norm(probe_before))).item())
extras["inference_probe"] = {"l2_diff": l2, "cos_sim": cos}
```

## 6. Schedule trajectory assertions (V4-6)

Per kind, expected lr_trajectory shape for steps `[0..N-1]` with base_lr `B`:

| Kind            | Trajectory                                                        |
|-----------------|-------------------------------------------------------------------|
| constant        | `[B, B, ..., B]`                                                  |
| linear_warmup   | `[B*0, B*1/w, ..., B*(w-1)/w, B, B, ...]` for w=warmup_steps      |
| cosine          | `[B, B*cos(π/2 * t/T), ...]` after optional warmup                |
| wsd             | warmup → stable → cosine decay                                    |
| inv_sqrt        | `[B / sqrt(max(1, t-w))]`                                         |
| polynomial      | `[B * (1 - t/T)^power]`                                           |

V4-6 tests assert at least the monotonicity / value-at-step-0 / value-at-last-step
for each kind.

## 7. Out of scope (still — v5+ candidates)

- WS reconnect mid-train (lifecycle)
- Sharding apply → real distributed train (needs multi-device)
- Memory peak instrumentation
- Concurrent/abortable Train
- Spec save/load roundtrip
- Real 100+ layer realistic train depth
- Checkpoint save/resume
