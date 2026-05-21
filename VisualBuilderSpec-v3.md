# VisualBuilderSpec-v3 — Technical Spec (Deep E2E Verification)

Companion to `VisualBuilderPlan-v3.md`. Defines API contracts, file layout, and
testids for the 13-stage v3 epic.

## 1. Backend: `cppmega_v4/runner/stages.py` changes

### 1.1 stage_train OptimKind dispatch (V3-1)

Replace hardcoded `opt = optim.AdamW(learning_rate=lr)` with:

```python
opt, optimizer_kind = _build_optimizer(spec_optim, lr)
```

`_build_optimizer(spec_optim: OptimSpec|None, base_lr: float) -> (mlx.optim, str)`:

| OptimKind value      | Builder                                                | Notes                              |
|----------------------|--------------------------------------------------------|------------------------------------|
| `adamw` (default)    | `optim.AdamW(learning_rate=base_lr)`                   | legacy fallback                    |
| `lion`               | `cppmega_mlx.training.optimizers.make_lion(base_lr)`   | warn if lr > 5e-4                  |
| `lion8bit`           | `make_lion8bit(base_lr)`                               |                                    |
| `adam8bit`           | `make_adam8bit(base_lr)`                               |                                    |
| `muon`               | `make_muon(base_lr)`                                   |                                    |
| `muon_adamw_hybrid`  | `make_muon_adamw_hybrid(base_lr)`                      |                                    |
| `sgd`                | `optim.SGD(learning_rate=base_lr)`                     |                                    |

`extras["optimizer_kind"]` set to second return value.

### 1.2 stage_train data consumption (V3-2)

`StageContext.spec.data` is `DataSpec(tokenizer_path: str|None, parquet_path: str|None)`.
New helper `_load_real_batch(data_spec, B, S, vocab)` returns
`(embeds: mx.array, targets: mx.array)`:

```python
if data_spec and data_spec.parquet_path and data_spec.tokenizer_path:
    tokens = _read_first_n_tokens(data_spec, B * S)
    targets = mx.array(tokens).reshape(B, S)
    embed_layer = nn.Embedding(vocab, hidden)
    embeds = embed_layer(targets)
    return embeds, targets, "parquet", len(tokens)
# fallback
return _synthetic_batch(B, S, hidden, vocab), "synthetic", 0
```

`extras["data_source"]` and `extras["token_count"]` reported.

### 1.3 model_summary helper (V3-3)

```python
def _summarize_model(spec: BuildSpec, optimizer_kind: str, schedule_kind: str) -> dict:
    mlp_node = next((n for n in spec.graph.nodes if n.kind == "mlp"), None)
    attn_node = next((n for n in spec.graph.nodes if n.kind == "attention"), None)
    return {
        "mlp_activation": mlp_node.params.get("activation", "swiglu") if mlp_node else None,
        "attention_pre_norm": attn_node.params.get("pre_norm", "none") if attn_node else None,
        "attention_post_norm": attn_node.params.get("post_norm", "rmsnorm") if attn_node else None,
        "mlp_pre_norm": mlp_node.params.get("pre_norm", "none") if mlp_node else None,
        "mlp_post_norm": mlp_node.params.get("post_norm", "none") if mlp_node else None,
        "optimizer_kind": optimizer_kind,
        "schedule_kind": schedule_kind,
        "num_brick_kinds": len(set(n.kind for n in spec.graph.nodes)),
    }
```

### 1.4 Extras schema (post-V3)

```jsonc
{
  "losses": [number],            // existing
  "lr_trajectory": [number],     // existing
  "weight_delta_norm": number,   // existing
  "num_steps": int,              // existing
  "schedule_kind": string,       // existing
  "optimizer_kind": string,      // NEW (V3-1)
  "data_source": "synthetic"|"parquet",  // NEW (V3-2)
  "token_count": int,            // NEW (V3-2)
  "model_summary": { ... }       // NEW (V3-3)
}
```

## 2. Frontend: RunResultModal extras display (V3-4)

### 2.1 Expand toggle change

Currently the expand button only renders when `s.error` is truthy. Change to: render
for any stage where `extras` is present OR `error` is present. Renamed prop test-ids
preserved.

### 2.2 Extras render

```tsx
{open && s.extras && (
  <tr data-testid={`run-result-extras-row-${s.name}`}>
    <td colSpan={5}>
      <StageExtras stage={s.name} extras={s.extras} />
    </td>
  </tr>
)}
```

`StageExtras` component:

```tsx
function StageExtras({ stage, extras }: { stage: string; extras: Record<string, unknown> }) {
  return (
    <dl data-testid={`run-result-extras-${stage}`}>
      {Object.entries(extras).map(([k, v]) => (
        <ExtrasEntry key={k} stage={stage} k={k} v={v} />
      ))}
    </dl>
  );
}
```

`ExtrasEntry` renders based on shape:
- primitive → `<dd data-testid={`run-result-extras-${stage}-${k}`}>{String(v)}</dd>`
- array of primitives → `<ol>` with `data-testid={`run-result-extras-${stage}-${k}-${i}`}`
- object → recurse with key `run-result-extras-${stage}-${k}-{subkey}`

### 2.3 Required testids for v3 tests

| Path                                                              | Used by      |
|-------------------------------------------------------------------|--------------|
| `run-result-stage-train`                                          | all          |
| `run-result-expand-train`                                         | V3-5+        |
| `run-result-extras-train-losses-{i}`                              | V3-6 conv    |
| `run-result-extras-train-lr_trajectory-{i}`                       | V3-5.2       |
| `run-result-extras-train-weight_delta_norm`                       | V3-5.3       |
| `run-result-extras-train-optimizer_kind`                          | V3-5.4/5     |
| `run-result-extras-train-data_source`                             | V3-10        |
| `run-result-extras-train-model_summary-mlp_activation`            | V3-5.1       |
| `run-result-extras-train-model_summary-attention_pre_norm`        | V3-5.3       |
| `run-result-extras-train-model_summary-optimizer_kind`            | V3-5.4       |
| `top-bar-train` (button)                                          | V3-8/9       |
| `top-bar-train-disabled-reason`                                   | V3-8/9       |

## 3. Tests file plan

```
vbgui/e2e/scenarios/
  11_ui_to_train.spec.ts          # rewrite (V3-5, strict assertions)
  12_train_convergence.spec.ts    # new (V3-6, multi-step loss decrease)
  13_ablation_math.spec.ts        # new (V3-7, loss divergence between variants)
  14_cross_arch_deep.spec.ts      # new (V3-11, 6 reps × 3 mutations)
  15_gating.spec.ts               # new (V3-8/9/10, critical gotcha / verify error / roundtrip)
  16_spec_roundtrip.spec.ts       # new (V3-12, save/load → train identical)

vbgui/e2e/utils/
  train_extras.ts                 # readTrainExtras(page) → typed extras object

tests/v4/
  test_stage_train_optimizers.py  # new (V3-1, 7 OptimKind smoke)
  test_stage_train_data.py        # new (V3-2, parquet+tokenizer + fallback)
  test_stage_train_summary.py     # new (V3-3, model_summary correctness)

vbgui/src/components/__tests__/
  RunResultModal.extras.test.tsx  # new (V3-4, extras rendering)
```

## 4. UI gating logic (V3-8/9/10)

### 4.1 Train button state machine

```ts
type TrainButtonState = "enabled" | "disabled-gotcha" | "disabled-verify" | "running";

function trainButtonState(
  gotchas: GotchaResult[],
  verify: VerifyResult,
  running: boolean,
): TrainButtonState {
  if (running) return "running";
  if (gotchas.some(g => g.severity === "critical")) return "disabled-gotcha";
  if (verify.errors > 0) return "disabled-verify";
  return "enabled";
}
```

Render button with `disabled` attr when not "enabled". `data-testid="top-bar-train"`,
`data-testid="top-bar-train-disabled-reason"` shows the reason when hovered/focused.

### 4.2 Roundtrip is a warning, not a block

Roundtrip FAIL surfaces yellow banner in DataInspector but Train remains enabled
(synthetic-targets fallback makes it independent of tokenizer correctness).

## 5. train_extras.ts helper

```ts
export type TrainExtras = {
  losses: number[];
  lr_trajectory: number[];
  weight_delta_norm: number;
  num_steps: number;
  schedule_kind: string;
  optimizer_kind: string;
  data_source: "synthetic" | "parquet";
  token_count: number;
  model_summary: {
    mlp_activation: string | null;
    attention_pre_norm: string;
    attention_post_norm: string;
    mlp_pre_norm: string;
    mlp_post_norm: string;
    optimizer_kind: string;
    schedule_kind: string;
    num_brick_kinds: number;
  };
};

export async function readTrainExtras(page: Page): Promise<TrainExtras> {
  await page.getByTestId("run-result-expand-train").click();
  const row = page.getByTestId("run-result-extras-row-train");
  await row.waitFor();
  // Parse from rendered DOM
  const text = async (k: string) =>
    (await page.getByTestId(`run-result-extras-train-${k}`).textContent())!;
  const lossesCount = await page.locator(
    "[data-testid^='run-result-extras-train-losses-']").count();
  const losses: number[] = [];
  for (let i = 0; i < lossesCount; i++) {
    losses.push(parseFloat(
      await page.getByTestId(`run-result-extras-train-losses-${i}`)
        .textContent() ?? "NaN"));
  }
  // ... lr_trajectory, model_summary recursion
  return { losses, /* ... */ } as TrainExtras;
}
```

## 6. Acceptance counters

| Surface                | Before v3 | Target v3 |
|------------------------|-----------|-----------|
| pytest                 | 2321      | ≥2350     |
| vitest                 | 160       | ≥166      |
| Playwright (deep cells)| 5         | ≥40       |
| stage_train extras keys| 5         | 9         |
| Lines in `11_*` (strict)| 128      | ≥180      |
| Vacuous assertions     | ~7        | 0         |

## 7. Out of scope (deferred to v4 if needed)

- Sharding apply → real distributed train
- Memory peak comparison (peak vs estimate within tolerance)
- WS reconnect mid-train
- Spec save/load via filesystem (in-memory only for V3-12)
- Real backward through tokenizer (we use embed layer, not raw bytes)
