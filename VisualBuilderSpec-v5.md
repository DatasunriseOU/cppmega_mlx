# VisualBuilderSpec-v5 — Technical Spec (Math-Effect, Distributed, Lifecycle, Persistence)

Companion to `VisualBuilderPlan-v5.md`. Defines API contracts, extras
schema, file layout, data-testids for the 25-gap epic. Organised by gap
group A-I matching the plan.

## 1. Stage_train extras schema (post-v5)

```jsonc
{
  // === v4 baseline ===
  "losses": [number], "lr_trajectory": [number],
  "weight_delta_norm": number, "num_steps": int,
  "schedule_kind": string, "optimizer_kind": string,
  "data_source": "synthetic"|"parquet"|"parquet_tokenized",
  "token_count": int, "tokenizer_used": string|null,
  "loss_kind": string, "muon_group_size": int|null,
  "adamw_group_size": int|null,
  "inference_probe": { l2_diff, cos_sim },
  "side_channels_observed": [string],
  "model_summary": { mlp_activation, attention_pre_norm,
    attention_post_norm, mlp_pre_norm, mlp_post_norm,
    optimizer_kind, schedule_kind, num_brick_kinds,
    loss_kind, rewriters_applied: [string] },

  // === v5 — Math-effect (G01-G04) ===
  "mtp": {                                 // G01 — only when MTP_WEIGHTED
    "k": int, "betas": [float],
    "per_head_losses": [float]
  } | null,
  "ifim": {                                // G02 — only when IFIM_SHAPED
    "lambda_fim": float,
    "fim_weights_norm": float,
    "penalty_value": float
  } | null,
  "mhc": {                                 // G03 — only when MHC_ATTN_BIAS
    "lambda_mhc": float, "bias_norm": float
  } | null,
  "graph_diff": {                          // G04
    "added": [string],
    "removed": [string],
    "renamed": [[string, string]]
  },

  // === v5 — Distributed / precision (G05-G07) ===
  "sharding_applied": {                    // G05
    "axis_assignments": [
      { "axis_name": string, "kind": string, "degree": int }
    ],
    "shard_dim": int,
    "microbatch_size": int
  } | null,
  "memory_peak_bytes": int,                // G06
  "train_dtype": string,                   // G07 — bf16 / fp16 / fp32
  "master_dtype": string,                  // G07
  "fp8_active": bool,                      // G07

  // === v5 — Lifecycle / persistence (G08-G12) ===
  "ws_reconnects": int,                    // G08
  "abort_token_seen": bool,                // G09
  "opt_state_carried": bool,               // G10
  "checkpoint": {                          // G12
    "saved_path": string|null,
    "loaded_path": string|null,
    "param_count": int,
    "state_count": int
  } | null,

  // === v5 — Side-channels real routing (G17) ===
  "side_channels_forward_effect": {
    "doc_ids_mask_density": float,
    "token_ids_added_norm": float
  } | null,

  // === v5 — Inference / observability (G20) ===
  "inference_probe": {                     // EXTENDED from V4-11
    "l2_diff": float, "cos_sim": float,
    "real_tokens": bool,
    "text_len": int,
    "top1_token_drift": int                // G20
  },

  // === v5 — Optimizer (G22, G23) ===
  "hybrid_deltas": {                       // G22 — only when hybrid
    "muon_norm": float,
    "adamw_norm": float,
    "ratio": float
  } | null,
  "gradient_clip": {                       // G23
    "threshold": float|null,
    "max_grad_norm_seen": float,
    "num_clips": int
  },

  // === v5 — MoE (G25) ===
  "moe": {                                 // only when MoE bricks present
    "num_experts": int, "top_k": int,
    "routing_entropy": float,
    "load_balance_loss": float,
    "dropped_token_ratio": float,
    "per_expert_load": [float]
  } | null
}
```

## 2. Backend changes by gap

### G01 — MTP_WEIGHTED loss kernel (stages.py + loss_spec.py)

```python
def _mtp_weighted_loss(
    logits_per_head: list[mx.array],   # K arrays of shape (B, S, V)
    labels: mx.array,                   # (B, S)
    betas: list[float],                 # K weights, sum to 1
) -> tuple[mx.array, list[float]]:
    """Σ beta_i × CE(logits_i, labels shifted by i tokens)."""
    per_head: list[float] = []
    total = mx.zeros(())
    for i, (logits, beta) in enumerate(zip(logits_per_head, betas)):
        shifted = _shift_labels(labels, i)
        ce_i = nn.losses.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            shifted.reshape(-1), reduction="mean",
        )
        per_head.append(float(ce_i.item()))
        total = total + beta * ce_i
    return total, per_head


def _shift_labels(labels: mx.array, k_offset: int) -> mx.array:
    """Right-shift labels by k_offset; pad last positions with last label."""
    ...
```

In stage_train: when `spec.loss.kind == MTP_WEIGHTED`:
1. Materialise K extra `nn.Linear(hidden, vocab)` heads
2. Forward produces K logits arrays
3. `_mtp_weighted_loss` over all K
4. extras["mtp"] populated

### G02-G03 — IFIM / MHC loss kernels

Same pattern. Either modulate CE per-token (IFIM with FIM weights) or
add bias term to attention scores (MHC).

### G04 — apply_rewrites in stage_train

```python
from cppmega_v4.buildspec.rewriters import apply_rewriters
from cppmega_v4.buildspec.diagnostics import verify_build_spec

# After ctx.build_spec assembled:
graph_before = {n.name for n in ctx.build_spec.graph.nodes}
ctx.build_spec = apply_rewriters(ctx.build_spec, spec_rewriters)
graph_after = {n.name for n in ctx.build_spec.graph.nodes}

extras["graph_diff"] = {
    "added": sorted(graph_after - graph_before),
    "removed": sorted(graph_before - graph_after),
    "renamed": [],  # rewriters don't rename in v1
}
```

### G05 — Sharding effect

```python
# Read spec.sharding into stage_train context.
if spec_sharding and spec_sharding.axis_assignments:
    extras["sharding_applied"] = {
        "axis_assignments": [
            {"axis_name": a.axis_name, "kind": a.kind, "degree": a.degree}
            for a in spec_sharding.axis_assignments
        ],
        "shard_dim": prod(a.degree for a in spec_sharding.axis_assignments),
        "microbatch_size": batch // shard_dim,
    }
```

### G06 — Memory peak

```python
mx.metal.reset_peak_memory()  # if available
# ... train loop ...
extras["memory_peak_bytes"] = int(mx.metal.get_peak_memory())
```

Wrap in try/except for non-Metal backends.

### G07 — Precision toggles

```python
master_dtype = mx.float32 if spec_optim.mixed_precision is False else mx.bfloat16
train_dtype = mx.float8 if spec_sharding.fp8_enabled else mx.bfloat16
extras["train_dtype"] = str(train_dtype)
extras["master_dtype"] = str(master_dtype)
extras["fp8_active"] = bool(spec_sharding.fp8_enabled)
```

### G08 — WS reconnect emit progress

Backend already has heartbeat via WS. Add per-step:
```python
for step in range(n_steps):
    ws_emit({"type": "train.progress", "step": step, "total": n_steps,
             "loss": float(loss.item())})
```

### G09 — Abort token

```python
abort_token = opts.get("abort_token")
for step in range(n_steps):
    if abort_token is not None and _is_aborted(abort_token):
        return StageResult(name="train", status="cancelled",
                           extras={"abort_token_seen": True,
                                   "losses_so_far": losses})
    ...
```

WS handler for `pipeline.abort` sets a process-global flag indexed by
run_id.

### G10 — Warm-start

```python
continue_from = opts.get("continue_from_run_id")
if continue_from and continue_from in _RUN_CACHE:
    opt.state = _RUN_CACHE[continue_from]["opt_state"]
    extras["opt_state_carried"] = True
# ... train ...
_RUN_CACHE[run_id] = {"opt_state": opt.state}
```

### G11 — Spec save/load

Pure frontend. JSON.stringify of wireSpecRef.current.spec + nodes + edges.
On load, dispatch resetSpec + setNodes + setEdges.

### G12 — Checkpoint save/resume

```python
import safetensors.mlx
ckpt_save = opts.get("checkpoint_save_path")
if ckpt_save:
    weights = dict(nn.utils.tree_flatten(all_modules.parameters()))
    safetensors.mlx.save_file(weights, ckpt_save)
    extras["checkpoint"]["saved_path"] = ckpt_save

ckpt_load = opts.get("checkpoint_load_path")
if ckpt_load:
    loaded = safetensors.mlx.load_file(ckpt_load)
    all_modules.update(nn.utils.tree_unflatten(list(loaded.items())))
    extras["checkpoint"]["loaded_path"] = ckpt_load
```

### G15 — Long-run smoothing

```python
losses_smoothed = []
window = 10
for i, l in enumerate(losses):
    start = max(0, i - window + 1)
    losses_smoothed.append(sum(losses[start:i+1]) / (i - start + 1))
extras["losses_smoothed"] = losses_smoothed
```

### G17 — Side-channel forward routing

```python
def forward_layers(layers, x, side_channels=None):
    doc_mask = None
    cond_embed = None
    if side_channels:
        if "doc_ids" in side_channels:
            doc_mask = _doc_attention_mask(side_channels["doc_ids"])
        if "token_ids" in side_channels:
            cond_embed = nn.Embedding(MAX_VOCAB, hidden)(
                mx.array(side_channels["token_ids"]))
    for mod in layers[:-1]:
        out = mod(x, attention_mask=doc_mask) if doc_mask else mod(x)
        ...
    if cond_embed is not None:
        x = x + cond_embed
    return layers[-1](x)

extras["side_channels_forward_effect"] = {
    "doc_ids_mask_density": float(doc_mask.sum() / doc_mask.size)
                            if doc_mask is not None else 0.0,
    "token_ids_added_norm": float(mx.linalg.norm(cond_embed).item())
                            if cond_embed is not None else 0.0,
}
```

### G20 — Real-tokens inference probe

```python
probe_text = opts.get("inference_probe_text")
if probe_text and tokenizer_used:
    tok = Tokenizer.from_file(tokenizer_path)
    probe_ids = tok.encode(probe_text).ids[:seq]
    probe_input = embed_layer(mx.array(probe_ids).reshape(1, -1))
    extras["inference_probe"]["real_tokens"] = True
    extras["inference_probe"]["text_len"] = len(probe_ids)
    # ... forward before + after, compute top1 drift ...
    extras["inference_probe"]["top1_token_drift"] = int(
        (logits_before.argmax(-1) != logits_after.argmax(-1)).sum().item()
    )
```

### G22 — Hybrid update delta per group

```python
if optimizer_kind == "muon_adamw_hybrid":
    muon_before = {k: mx.array(v) for k, v in muon_params.items()}
    adamw_before = {k: mx.array(v) for k, v in adamw_params.items()}
    # ... train ...
    muon_after = ...
    muon_norm = sum_param_norm_diff(muon_before, muon_after)
    adamw_norm = sum_param_norm_diff(adamw_before, adamw_after)
    extras["hybrid_deltas"] = {
        "muon_norm": muon_norm,
        "adamw_norm": adamw_norm,
        "ratio": muon_norm / max(adamw_norm, 1e-12),
    }
```

### G23 — gradient_clip_norm

```python
clip_threshold = spec_optim.gradient_clip_norm if spec_optim else None
extras["gradient_clip"] = {
    "threshold": clip_threshold,
    "max_grad_norm_seen": 0.0,
    "num_clips": 0,
}
for step in range(n_steps):
    loss, grads = loss_and_grad(...)
    grad_norm = optim.clip_grad_norm(grads, clip_threshold) if clip_threshold \
                else compute_grad_norm(grads)
    extras["gradient_clip"]["max_grad_norm_seen"] = max(
        extras["gradient_clip"]["max_grad_norm_seen"],
        float(grad_norm.item())
    )
    if clip_threshold and float(grad_norm.item()) > clip_threshold:
        extras["gradient_clip"]["num_clips"] += 1
```

### G25 — MoE routing

```python
moe_node = next((n for n in graph.nodes
                 if n.kind in ("moe", "bailing_moe", "sparse_moe")), None)
if moe_node is not None:
    # Forward with MoE hook capturing routing weights
    routing_log = {"per_expert_load": [], "entropy": 0.0,
                   "load_balance_loss": 0.0, "dropped": 0}
    # ... attach hook, forward, collect ...
    extras["moe"] = {
        "num_experts": int(moe_node.params.get("num_experts", 1)),
        "top_k": int(moe_node.params.get("top_k", 1)),
        "routing_entropy": routing_log["entropy"],
        "load_balance_loss": routing_log["load_balance_loss"],
        "dropped_token_ratio": routing_log["dropped"] / token_count,
        "per_expert_load": routing_log["per_expert_load"],
    }
```

## 3. Frontend changes

### G01 — LossTab per-i beta inputs (or kept flat with auto-broadcast)

Current LossTab sends `params: {k, beta}`. Either:
- (Easier) Keep flat, rely on `_make_loss` broadcast (existing V4-7).
  Add `loss-mtp-beta-{i}` testids for visible per-head values rendered
  after Apply.
- (Cleaner) Inputs per-i (`loss-mtp-beta-0`, `loss-mtp-beta-1`...).

### G05-G07 — Sharding / precision toggles already exist in OptimTab +
TopBar + ShardingTab. Verify wire end-to-end.

### G08 — BottomStrip reconnect indicator (exists; verify behaviour)

### G09 — Cancel button in TopBar dropdown
```tsx
{trainRunning && (
  <button data-testid="run-pipeline-cancel"
          onClick={onCancel}>Cancel</button>
)}
```

### G11 — TopBar Save / Load buttons
```tsx
<button data-testid="spec-save" onClick={() => downloadSpec(spec)}>Save</button>
<input data-testid="spec-load-input" type="file" accept=".json"
       onChange={(e) => loadSpec(e.target.files?.[0])} />
<button data-testid="spec-load"
        onClick={() => fileInputRef.current?.click()}>Load</button>
```

### G12 — Checkpoint controls in train dropdown
```tsx
<input data-testid="train-checkpoint-save-path" type="text"
       placeholder="/tmp/ckpt.safetensors" />
<input data-testid="train-checkpoint-load-path" type="text"
       placeholder="/tmp/prev_ckpt.safetensors" />
```

### G17 — SideChannelsTab (replaces train-dropdown checkboxes)
New sidebar tab with per-family toggle + data picker + preview. Replaces
the minimum-viable V4-10 toggles.

### G19 — DimensionsTab Apply per-row
```tsx
<button data-testid={`dim-row-${i}-apply`}
        onClick={() => applyDimensionSuggestion(row)}>Apply</button>
```

### G20 — train-probe-text input in TopBar dropdown
```tsx
<textarea data-testid="train-probe-text"
          placeholder="Optional: encode this text for inference probe" />
```

## 4. Test plan (file layout)

```
cppmega.mlx/tests/v5/
  test_stage_train_mtp.py            # G01
  test_stage_train_ifim.py           # G02
  test_stage_train_mhc.py            # G03
  test_stage_train_rewriters.py      # G04
  test_stage_train_sharding.py       # G05
  test_stage_train_memory_peak.py    # G06
  test_stage_train_precision.py      # G07
  test_stage_train_lifecycle.py      # G08+G09
  test_stage_train_warm_start.py     # G10
  test_stage_train_checkpoint.py     # G12
  test_stage_train_long_run.py       # G15
  test_stage_train_side_channels.py  # G17
  test_stage_train_ablation_parity.py # G18
  test_stage_train_real_probe.py     # G20
  test_stage_train_other_extras.py   # G21
  test_stage_train_hybrid_delta.py   # G22
  test_stage_train_clip.py           # G23
  test_stage_train_moe.py            # G25

vbgui/e2e/scenarios/
  30_mtp_math.spec.ts                # G01
  31_ifim_math.spec.ts               # G02
  32_mhc_math.spec.ts                # G03
  33_rewriter_graph_diff.spec.ts     # G04
  34_sharding_apply.spec.ts          # G05
  35_memory_peak_parity.spec.ts      # G06
  36_precision_toggles.spec.ts       # G07
  37_ws_reconnect.spec.ts            # G08
  38_cancel_train.spec.ts            # G09
  39_warm_start.spec.ts              # G10
  40_spec_save_load.spec.ts          # G11
  41_checkpoint_resume.spec.ts       # G12
  42_tokenizer_strict.spec.ts        # G13
  43_convergence_strict.spec.ts      # G14
  44_long_run_convergence.spec.ts    # G15
  45_cross_arch_non_canonical.spec.ts # G16
  46_side_channels_forward.spec.ts   # G17
  47_ablation_parity.spec.ts         # G18
  48_dimensions_feedback.spec.ts     # G19
  49_real_tokens_probe.spec.ts       # G20
  50_other_stages_extras.spec.ts     # G21
  51_hybrid_delta.spec.ts            # G22
  52_clip_norm.spec.ts               # G23
  53_roundtrip_exact.spec.ts         # G24
  54_moe_routing.spec.ts             # G25

vbgui/src/components/
  sidebar/SideChannelsTab.tsx        # G17
  CancelTrainButton.tsx              # G09 (inline in TopBar OK too)

vbgui/src/state/
  specSerializer.ts                  # G11
```

## 5. Required data-testids (post-v5)

| Testid                                              | Used by | Gap   |
|-----------------------------------------------------|---------|-------|
| `loss-mtp-beta-{i}`                                 | LossTab | G01   |
| `run-result-extras-train-mtp-k`                     | Modal   | G01   |
| `run-result-extras-train-mtp-per_head_losses-{i}`   | Modal   | G01   |
| `run-result-extras-train-ifim-lambda_fim`           | Modal   | G02   |
| `run-result-extras-train-mhc-lambda_mhc`            | Modal   | G03   |
| `run-result-extras-train-graph_diff-added-{i}`      | Modal   | G04   |
| `run-result-extras-train-sharding_applied-shard_dim`| Modal   | G05   |
| `run-result-extras-train-memory_peak_bytes`         | Modal   | G06   |
| `memory-bar-actual`                                 | TopBar  | G06   |
| `run-result-extras-train-train_dtype`               | Modal   | G07   |
| `run-result-extras-train-fp8_active`                | Modal   | G07   |
| `bottom-strip-reconnecting`                         | BottomStrip | G08 |
| `run-pipeline-cancel`                               | TopBar  | G09   |
| `train-warm-start`                                  | TopBar  | G10   |
| `run-result-extras-train-opt_state_carried`         | Modal   | G10   |
| `spec-save`, `spec-load`, `spec-load-input`         | TopBar  | G11   |
| `train-checkpoint-save-path`                        | TopBar  | G12   |
| `train-checkpoint-load-path`                        | TopBar  | G12   |
| `run-result-extras-train-checkpoint-saved_path`     | Modal   | G12   |
| `sidebar-tab-side-channels`                         | Sidebar | G17   |
| `side-channel-{name}-toggle`                        | SCTab   | G17   |
| `run-result-extras-train-side_channels_forward_effect-doc_ids_mask_density` | Modal | G17 |
| `dim-row-{i}-apply`                                 | DimTab  | G19   |
| `train-probe-text`                                  | TopBar  | G20   |
| `run-result-extras-train-inference_probe-top1_token_drift` | Modal | G20 |
| `run-result-extras-train-hybrid_deltas-ratio`       | Modal   | G22   |
| `run-result-extras-train-gradient_clip-num_clips`   | Modal   | G23   |
| `run-result-extras-train-moe-num_experts`           | Modal   | G25   |
| `run-result-extras-train-moe-routing_entropy`       | Modal   | G25   |

## 6. Honest categorisation post-v5

Every UI mutation gets one of three labels in closure report:
- **🟢 math-effect** — UI change observably changes loss/weight/output
- **🟡 propagation** — UI change reaches extras (still valuable for
  config audit, but not training math proof) — ALLOWED only for fields
  where there is no math (e.g. project name, save paths)
- **🔴 decorative** — UI change has no observable effect; should be
  fixed or removed

v5 target: 0 🔴 decorative, 🟡 propagation reduced to config-only fields,
≥20 🟢 math-effect tests.

## 7. Out of scope (v6+ candidates) — see Plan §7
