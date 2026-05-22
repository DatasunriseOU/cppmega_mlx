# VisualBuilderSpec-v6 — Technical Spec (UI Wiring + Real Math for V5 Placeholders)

Companion to `VisualBuilderPlan-v6.md`. Defines API contracts, extras
schema, file layout, data-testids for the 25-gap epic.

## 1. stage_train extras schema (post-v6)

```jsonc
{
  // === v5 baseline preserved ===

  // === v6 additions ===
  "sharding_applied": {                    // H15 extends G05
    ...existing,
    "per_rank_param_bytes": int,           // H15.2
    "per_rank_activation_bytes": int       // H15.2
  },
  "dtype_actual": {                        // H16 — what mlx actually used
    "train": string, "master": string,
    "fp8_attempted": bool, "fp8_fallback_reason": string|null
  },
  "side_channels_forward_effect": {        // H17 extends G17
    ...existing (proxy),
    "doc_mask_applied": bool,              // H17.4
    "single_doc_passthrough": bool         // H17.6
  },
  "moe": {                                 // H18 extends G25
    ...existing,
    "routing_entropy": float,              // populated (was null)
    "load_balance_loss": float,
    "dropped_token_ratio": float,
    "per_expert_load": [float]
  },
  "checkpoint": {                          // H19 extends G12
    ...existing,
    "opt_state_path": string|null          // H19.6
  },
  "inference_probe": {                     // H21 extends G20
    ...existing,
    "post_resume_drift": float|null        // H21.1
  },
  "fake_ranks": int,                       // H20
  "gradient_reduce_ms": float,             // H20
  "train_in_flight_at_request": bool       // H22 echo
}
```

## 2. Backend changes by gap

### H15 — Per-rank shard simulation

```python
def _simulate_shard(params: dict, shard_dim: int) -> dict:
    """Return per-rank param byte count assuming row-major even split."""
    flat = dict(nn.utils.tree_flatten(params))
    per_rank = 0
    for k, v in flat.items():
        per_rank += int(v.size * v.dtype.size) // max(1, shard_dim)
    return per_rank
```

### H16 — Real dtype switching

```python
# Pre-instantiate cast:
if master_dtype == "fp32":
    all_modules = _cast_module(all_modules, mx.float32)
elif master_dtype == "fp16":
    all_modules = _cast_module(all_modules, mx.float16)
# else: keep bf16 default

if train_dtype == "fp8":
    try:
        # mlx fp8 path (if supported)
        ...
    except (AttributeError, NotImplementedError) as e:
        extras["dtype_actual"]["fp8_fallback_reason"] = str(e)
```

### H17 — Real doc attention mask

```python
# In _build_attention or wrapper:
if doc_ids is not None:
    # Create (S, S) mask: 1 where doc_ids[i] == doc_ids[j]
    doc_mask = mx.broadcast_to(doc_ids[:, None], (B, S, S)) == \
               mx.broadcast_to(doc_ids[None, :], (B, S, S))
    attention_scores = mx.where(doc_mask, attention_scores, -1e9)
```

### H18 — MoE routing hook

```python
class _RoutingProbe:
    def __init__(self):
        self.entropy_sum = 0.0
        self.load = mx.zeros(num_experts)
        self.dropped = 0
        self.total = 0

    def observe(self, routing_weights: mx.array):
        # routing_weights: (B*S, num_experts) softmax probabilities
        probs = mx.softmax(routing_weights, axis=-1)
        ent = -mx.sum(probs * mx.log(probs + 1e-12), axis=-1)
        self.entropy_sum += float(mx.mean(ent).item())
        top_k_idx = mx.argpartition(probs, -top_k, axis=-1)[:, -top_k:]
        # ... aggregate load + dropped ...
```

### H19 — Strict identical-loss-continuation

```python
def test_checkpoint_identical_continuation(tmp_path):
    save = tmp_path / "ckpt.safetensors"
    state_save = tmp_path / "state.safetensors"
    # Run A: N=4, save weights + state
    a = _run({"num_steps": 4,
              "checkpoint_save_path": str(save),
              "opt_state_save_path": str(state_save)})
    # Run B: N=1, load weights + state
    b = _run({"num_steps": 1,
              "checkpoint_load_path": str(save),
              "opt_state_load_path": str(state_save)})
    # B's losses[0] must equal A's losses[3] (last loss)
    assert abs(b["losses"][0] - a["losses"][3]) < 1e-5
```

### H20 — Fake-rank distributed simulation

```python
def _fake_rank_forward(modules, lm_head, input_embeds, fake_ranks):
    """Split batch across N fake ranks, accumulate gradients."""
    sub_losses = []
    sub_grads = []
    for rank in range(fake_ranks):
        sub_emb = input_embeds[rank::fake_ranks]
        # ... forward + backward ...
        sub_losses.append(loss)
        sub_grads.append(grads)
    # All-reduce
    reduced = _tree_mean(sub_grads)
    return mx.mean(mx.stack(sub_losses)), reduced
```

## 3. Frontend changes

### H01 — ShardingTab dispatch
Existing ShardingTab already dispatches sharding.set; verify wire.

### H02 — Precision toggles in TopBar
```tsx
<input data-testid="top-bar-mixed-precision" type="checkbox"
       checked={spec.optim.mixed_precision}
       onChange={(e) => dispatch({type: "optim.set",
         optim: {...spec.optim, mixed_precision: e.target.checked}})} />
<input data-testid="top-bar-fp8-enabled" type="checkbox"
       checked={spec.sharding.fp8_enabled}
       onChange={(e) => dispatch({type: "sharding.set",
         sharding: {...spec.sharding, fp8_enabled: e.target.checked}})} />
```

### H03 — Cancel button + abort RPC
```tsx
{trainInFlight && (
  <button data-testid="run-pipeline-cancel"
          onClick={() => rpc.call("pipeline.abort", {run_id: trainRunId})}>
    Cancel
  </button>
)}
```

### H04 — Warm-start checkbox
```tsx
<input data-testid="train-warm-start" type="checkbox"
       checked={warmStart}
       onChange={(e) => setWarmStart(e.target.checked)} />
```

App.handleRunPipeline:
```ts
if (warmStart && lastRunId) {
  trainOpts.continue_from_run_id = lastRunId;
}
```

### H05 — Checkpoint path inputs
```tsx
<input data-testid="train-checkpoint-save-path" type="text"
       placeholder="/tmp/ckpt.safetensors"
       value={ckptSavePath} onChange={...} />
<input data-testid="train-checkpoint-load-path" type="text"
       placeholder="/tmp/prev.safetensors"
       value={ckptLoadPath} onChange={...} />
```

### H07 — DimensionsTab Apply dispatch
```tsx
<DimensionsTab log={inferenceLog}
  onApply={(entry) => {
    // Map entry to brick params mutation
    const nodeIdx = nodes.findIndex(n => n.id === entry.brick);
    if (nodeIdx >= 0) {
      setNodes(nodes.map((n, i) => i === nodeIdx
        ? {...n, data: {...n.data, params: {
            ...n.data.params, [entry.param]: entry.value}}}
        : n));
    }
  }} />
```

### H08 — Probe text textarea
```tsx
<textarea data-testid="train-probe-text"
          placeholder="Optional: encode this for inference probe"
          value={probeText} onChange={(e) => setProbeText(e.target.value)} />
```

### H10 — SideChannelsTab (full sidebar tab)
```tsx
// vbgui/src/components/sidebar/SideChannelsTab.tsx
export function SideChannelsTab({sideChannels, onChange}: Props) {
  return (
    <div data-testid="side-channels-tab">
      {AVAILABLE_FAMILIES.map(family => (
        <label data-testid={`side-channel-${family}-toggle`}>
          <input type="checkbox" ... />
          {family}
        </label>
      ))}
    </div>
  );
}
```

### H11 — MemoryBar dual display
```tsx
<div>
  <span data-testid="memory-bar-estimate">{fmtBytes(estimate)}</span>
  / <span data-testid="memory-bar-actual">{fmtBytes(actual)}</span>
</div>
```

### H14 — AblationsTab expand rows
Mirror RunResultModal pattern: each variant row gets expand button that
shows full extras as ExtrasEntry recursive render.

### H22 — Train in-flight state
```tsx
const [trainInFlight, setTrainInFlight] = useState(false);
// disable Train button while inFlight; show "Training..."
```

### H23 — Precision dropdown (bf16/fp16/fp32)
Replace H02 checkboxes with a single select once fp16 supported.

## 4. Required data-testids (post-v6)

| Testid                                            | Stage | Used by |
|---------------------------------------------------|-------|---------|
| `top-bar-mixed-precision`                         | H02   | UI      |
| `top-bar-fp8-enabled`                             | H02   | UI      |
| `run-pipeline-cancel`                             | H03   | UI      |
| `train-warm-start`                                | H04   | UI      |
| `train-checkpoint-save-path`                      | H05   | UI      |
| `train-checkpoint-load-path`                      | H05   | UI      |
| `train-probe-text`                                | H08   | UI      |
| `sidebar-tab-side-channels`                       | H10   | UI      |
| `side-channel-{family}-toggle`                    | H10   | UI      |
| `memory-bar-estimate`                             | H11   | UI      |
| `memory-bar-actual`                               | H11   | UI      |
| `ablation-row-{variant}-expand`                   | H14   | UI      |
| `ablation-row-{variant}-extras`                   | H14   | UI      |
| `run-result-extras-train-sharding_applied-per_rank_param_bytes` | H15 | Modal |
| `run-result-extras-train-dtype_actual-fp8_attempted` | H16 | Modal |
| `run-result-extras-train-moe-routing_entropy`     | H18   | Modal   |
| `run-result-extras-train-checkpoint-opt_state_path` | H19 | Modal   |
| `run-result-extras-train-fake_ranks`              | H20   | Modal   |
| `top-bar-train-status`                            | H22   | UI      |

## 5. Test plan (file layout)

```
cppmega.mlx/tests/v6/
  test_stage_train_sharding_real.py     # H15
  test_stage_train_dtype_real.py        # H16
  test_stage_train_doc_mask.py          # H17
  test_stage_train_moe_routing.py       # H18
  test_stage_train_continuation.py      # H19
  test_stage_train_fake_ranks.py        # H20
  test_stage_train_resume_drift.py      # H21
  test_stage_train_fp16.py              # H23

vbgui/e2e/scenarios/
  55_sharding_apply.spec.ts             # H01
  56_precision_toggles.spec.ts          # H02
  57_cancel_train.spec.ts               # H03
  58_warm_start.spec.ts                 # H04
  59_checkpoint_save_load.spec.ts       # H05
  60_long_run_ui.spec.ts                # H06
  61_dimensions_feedback.spec.ts        # H07
  62_probe_text.spec.ts                 # H08
  63_save_load_roundtrip.spec.ts        # H09
  64_side_channels_tab.spec.ts          # H10
  65_memory_parity.spec.ts              # H11
  66_ws_reconnect.spec.ts               # H12
  67_other_stages_ui.spec.ts            # H13
  68_ablation_expand.spec.ts            # H14
  69_doc_mask_loss_differs.spec.ts      # H17 e2e
  70_moe_entropy_changes.spec.ts        # H18 e2e
  71_resume_continuation.spec.ts        # H19/H21/H24
  72_concurrent_train.spec.ts           # H22
  73_fp16_select.spec.ts                # H23

vbgui/src/components/
  sidebar/SideChannelsTab.tsx           # H10
  CancelTrainButton.tsx                 # H03 (inline OK)
  PrecisionSelect.tsx                   # H23 (inline OK)

scripts/
  audit_v3_v4_v5.py                     # H25
```

## 6. Honest categorisation criteria

Every test gets one label:

**🟢 math-effect**: assertion involves a numerical value from extras
(loss, weight delta, lr, entropy, drift, parity). Or assertion that
two routes produce same/different number within tolerance.

**🟡 propagation**: only checks that a UI string reaches extras as the
same string. Allowed only for genuine config fields with no math claim
(project name, save paths, dropdown selections that map 1:1 to enums).

**🔴 decorative**: UI mutation has no observable effect. Should be
removed or upgraded to 🟢.

Script `audit_v3_v4_v5.py` parses each test's assertions and classifies.

## 7. Out of scope (v7+)

- Real multi-mac distributed (network all-reduce)
- 1B+ param scale matrix
- A/B hyperparameter search
- Anomaly detection (NaN traps)
- Live inference serving
- Post-train quantisation
- LoRA / PEFT
