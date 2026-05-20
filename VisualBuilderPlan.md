План: Визуальный конструктор моделей — компонентный стек, UI, API, маппинг

Финальный план GUI-слоя поверх трёх уже шипнутых backend-слоёв
(VBSpec / MBSpec / PSpec / Auto-Fusion). Синтез 9 параллельных
research-агентов: 3 code-extract (cppmega.mlx / ../cppmega / ../nanochat) +
6 web-research (Perplexity deep / Perplexity Pro Gemini 3.1 / Brave /
Exa neural + content / Perplexity reasoning Gemini 3.1; Tavily вылетел
на исчерпанных ключах).

Источники для каждого раздела перечислены inline.

---

0. TL;DR

Стек: **anywidget + React Flow 12 + Rust/WASM core + ELK.js layout + Python
JSON-RPC backend**. Шипается двумя путями из одного codebase:

  (a) Jupyter widget — anywidget traitlets, backend = реальный
      cppmega_v4.* в notebook'е (instantiate=True, MLX runtime).
  (b) JupyterLite/Pyodide static bundle — тот же React + WASM-core,
      backend = чистый-Python estimator из cppmega_v4.spec/buildspec/
      parallelism (instantiate=False, только sizing).

WASM-only путь (Rust+egui-graph-edit) рассмотрен и **отвергнут** для v1:
UX незрелый, ecosystem мелкий, custom HTML в нодах невозможен (а нам
нужны memory bars/tooltips/inline param editors). WASM появляется в нашем
стеке как **исполнительное ядро для shape-inference и memory accounting**
(Rust→WASM или Pyodide), а UI shell — React.

---

1. Что у нас уже есть (backend готов)

Все эти модули шипнуты на main и протестированы (1079/5/0):

  - **cppmega_v4.spec** (VBSpec, 167 тестов):
      ShapeExpr, BrickShapeContract, ResolvedBrickGraph, resolve_shapes,
      MemoryReport, estimate_memory, AdapterRule, suggest_adapter_chain,
      insert_adapter_chain, verify_and_estimate, suggest_dim_env, suggest_adapters.

  - **cppmega_v4.buildspec** (MBSpec, 146 тестов):
      LossSpec/LossKind (CE/MTP/IFIM/MHC/CUSTOM), OptimSpec/OptimKind
      (AdamW/Muon/Hybrid/SGD) + ParamGroup matchers, ModelBuildSpec,
      verify_build_spec, Rewriter protocol, MTPRewriter, IFIMRewriter,
      MHCRewriter, build_model + BuiltSequentialModel.

  - **cppmega_v4.parallelism** (PSpec, 164 теста):
      DeviceKind (H100/H200/A100/B100/GB10/TPU/M3Ultra), DeviceTopology,
      ParallelismKind (DP/FSDP1/FSDP2/ZERO1/ZERO2/TP/SP/EP/PP/PP_VPP),
      ShardingSpec, estimate_distributed_memory, 15 GOTCHAS,
      suggest_sharding, verify_distributed_plan.

  - **cppmega_v4.fusion** (Auto-Fusion, 175 тестов):
      BrickGraph, plan_fusion_regions, auto_compile_region.

  - **cppmega_v4.architectures**: 12 preset factories (Qwen3-Next, Kimi
      Linear/K2, DeepSeek V3/V4, Gemma4, Mistral4, Ling26, LongCat,
      Nemotron3, ZAYA1, Arcee Trinity).

  - **cppmega_v4.models.unified_superblock_v4**: 22 bricks в BLOCK_BUILDERS.

Итого: backend ~80% готов. **GUI plan фокусируется на front-end shell +
wire protocol + executor wrapper**.

---

2. Что НЕ хватает (gap vs готовый backend)

┌─────────────────────────────────────────┬────────────────────────────────┐
│         Слой                            │             Сейчас             │
├─────────────────────────────────────────┼────────────────────────────────┤
│ React Flow UI shell с типизированными   │ ✗                              │
│ портами (per-tensor-shape)              │                                │
├─────────────────────────────────────────┼────────────────────────────────┤
│ Custom node renderer с memory-bar +     │ ✗                              │
│ inline param editor                     │                                │
├─────────────────────────────────────────┼────────────────────────────────┤
│ Side panel: Loss / Optim / Rewriter     │ ✗                              │
├─────────────────────────────────────────┼────────────────────────────────┤
│ Topology selector + sharding overlay    │ ✗                              │
├─────────────────────────────────────────┼────────────────────────────────┤
│ Real-time JSON-RPC bridge React↔Python  │ ✗                              │
├─────────────────────────────────────────┼────────────────────────────────┤
│ Anywidget shim для Jupyter              │ ✗                              │
├─────────────────────────────────────────┼────────────────────────────────┤
│ JupyterLite/Pyodide bundle              │ ✗                              │
├─────────────────────────────────────────┼────────────────────────────────┤
│ Gotcha-chip UI + reference-link tooltip │ ✗                              │
├─────────────────────────────────────────┼────────────────────────────────┤
│ Preset/architecture launcher            │ ✗                              │
├─────────────────────────────────────────┼────────────────────────────────┤
│ Live "Build & Train Step" button с      │ ✗                              │
│ executor wrapper над build_model        │                                │
└─────────────────────────────────────────┴────────────────────────────────┘

---

3. Component Stack — конкретно

### 3.1 Front-end shell

**Pick**: `@xyflow/react` v12 (React Flow) — MIT, 36.2k stars, активный
2026-04-10, latest 12.x. Лучший custom-node ergonomics (React component
внутри node), `isValidConnection` для type-checked edges, edge styling
on state.

**Альтернативы рассмотренные**:
  - Rete.js v2.0.6 — лучшая typed-socket model, но React Flow выигрывает
    по экосистеме и custom rendering.
  - Comfy-Org litegraph (canvas-based) — единственный с verified
    1000+ nodes performance, но HTML widgets в нодах painful.
  - egui-graph-edit (Rust+WASM) — единственный pure-WASM, но egui UX
    утилитарный, no HTML overlays, лимитированный ecosystem.

**Auto-layout**: ELK.js v0.10+ (layered DAG + orthogonal routing). Запускаем
в Web Worker чтобы не блокировать canvas при больших графах.

**Bundler**: Vite 5 + TypeScript 5.3 + React 18.

### 3.2 Validation/sizing core (the WASM-relevant slice)

Два режима, оба с одинаковым JSON contract:

**Mode A — Jupyter notebook backend (default development)**:
  Python kernel загружает `cppmega_v4.spec` / `buildspec` / `parallelism` /
  `fusion`, отвечает через anywidget traitlets. Performance: <5ms на
  verify_distributed_plan уже измерено (PSpec Stage E perf gate).

**Mode B — JupyterLite/Pyodide standalone (WASM path для shareable demos)**:
  Чистый-Python Pyodide bundle с тем же кодом, только `instantiate=False`
  на BrickGraph (никакой MLX runtime в браузере, только symbolic shape +
  memory math). Pyodide ~5MB загружается single-shot.

**Mode C (опционально, Phase 2) — Rust shape engine для real-time hot path**:
  Только формулы из `cppmega_v4/spec/shape_contract.py` и
  `cppmega_v4/spec/memory_report.py` переписываем в Rust и компилируем в
  WASM. Это даёт <1ms validation на каждом drag/edit. Вся остальная
  логика (rewriters, gotcha checker, suggest_sharding) остаётся в Python.

### 3.3 Wire protocol

JSON-RPC 2.0 envelope. В Jupyter режиме идёт через anywidget traitlets
(одно сообщение = одна `traitlet.set()`). В standalone режиме идёт через
WebSocket к локальному Python-серверу или прямо в Pyodide worker.

### 3.4 Python backend (опциональный standalone)

FastAPI 0.110+ + uvicorn + python-jsonrpc-server. Только если юзер хочет
desktop launcher с реальным MLX runtime вне Jupyter. Anywidget mode
покрывает 80% use cases.

### 3.5 Packaging

  - **NPM package** `@cppmega/visual-builder` — React Flow + custom
    nodes + JSON-RPC client. Bundle ~1.5MB gzipped.
  - **PyPI package** `cppmega-builder-widget` — anywidget shim
    привязывающий NPM frontend к cppmega_v4.* backend. Установка:
    `pip install cppmega-builder-widget`.
  - **JupyterLite static site** — GitHub Pages deployment, single URL,
    no install. Pyodide preloaded.

---

4. UI — детальное описание

### 4.1 Главная зона (Canvas)

React Flow canvas во всю высоту окна минус 56px топ-бар + 320px sidebar.

**Палитра нод** (drag source, левая колонка 240px) с категориями:

  - **Attention** (раскрывается): gated_attention, attention, mla,
    mla_absorb, mistral4_mla, bailing_mla, dsv4_attention, gqa_sliding,
    cca_attention
  - **Linear-attn**: gdn, kda, bailing_linear
  - **SSM**: mamba3
  - **MoE**: moe, bailing_moe
  - **Sparse-attn**: nsa, csa_hca
  - **MTP / Cross**: gemma4_drafter, nemotron_h_mtp
  - **Projection/Norm**: mlp, engram, lightning_indexer
  - **Adapters** (auto-insertable): merge_heads, split_heads,
    transpose_bnsd, linear_bridge, rmsnorm, residual
  - **Heads** (loss target): logits-head (mlp)

Каждый кирпичик в палитре — миниатюра 80×60px с иконкой категории
(color-coded) и значком ⓘ → hover показывает 1-line role и required
side-channels (`doc_ids` для gdn/kda/bailing_linear; `token_ids` для
engram; etc.).

**Custom node на canvas** (180×N px, N зависит от количества raised params):

```
┌────────────────────────────────────┐
│ ◉ gdn_0          [linear_attn]  ⚙ │  ← input port left, kind badge, settings cog
│ ────────────────────────────────── │
│ Params:                            │
│  num_heads:     [32       ▼]       │  ← inline editor, dropdown for enums
│  head_dim:      [128      ]        │  ← numeric input
│ ────────────────────────────────── │
│ Memory:    [████████░░] 4.2 GB    │  ← per-node memory bar (params+act)
│ Shape:     (B,S,4096)              │  ← resolved output shape
│ ✓ doc_ids supplied                 │  ← side-channel status
└─────────────◉──────────────────────┘  ← output port right
```

Цветовой код категорий (фон ноды):
  - linear_attn   — голубой (#3B82F6 50% alpha)
  - sdpa_attention— фиолетовый (#8B5CF6 50%)
  - ssm           — циан (#06B6D4 50%)
  - moe           — оранжевый (#F97316 50%)
  - sparse_attn   — красно-серый (#EF4444 30%)
  - cross_attn    — янтарный (#F59E0B 50%)
  - norm_or_proj  — серый (#6B7280 50%)
  - adapter_*     — зелёный пунктир (#10B981, dashed border)

**Ребро (edge)** показывает resolved shape в badge на середине:
  `[1, 4096, 4096]`

Состояние ребра:
  - **green** (default) — shapes match, side-channels supplied
  - **yellow** — shape match но есть WARNING (opaque-brick boundary,
    side-channel может быть не supplied)
  - **red** + лейбл с suggested_fix — ERROR (shape mismatch); клик на
    лейбл вставляет предложенный adapter chain (Stage C VBSpec)
  - **dashed grey** — adapter-вставленное ребро

### 4.2 Top bar (56px)

Слева направо:
  1. Logo + project name (editable inline)
  2. **Preset launcher** dropdown: 12 presets (qwen3_next / kimi_linear /
     kimi_k2 / deepseek_v3 / deepseek_v4_flash / gemma4 / mistral4 /
     ling26 / longcat / nemotron3 / zaya1 / arcee_trinity) +
     `build_preset_specs(name, hidden_size, num_layers)` — drag preset
     на canvas создаёт всю цепочку.
  3. **Repeat-N selector** — сколько раз повторить unit (1..N), preset
     раскрывается accordion-style.
  4. **Topology selector** dropdown: 8 builtin (h100_8x / h200_8x / a100_8x /
     b100_8x / gb10_quarter / tpu_v6e_8 / tpu_v5p_4 / m3_ultra_solo) +
     "custom..." → opens DeviceTopology builder modal с mesh-axes inputs.
  5. **Compile mode** dropdown: off / **regional** (default) / whole_model
     (с красным предупреждением — сразу firing fsdp2_whole_compile /
     megatron_tp_whole_compile gotchas).
  6. **Memory bar** (всю оставшуюся ширину): worst-rank total /
     device HBM, sectioned coloured по components
     (weights / grads / optim / activations / kv_cache / overhead).
     На hover — tooltip с per-component bytes.
     Цвет:
       - green если total < device_hbm * 0.7
       - yellow если 0.7..0.9
       - red если >0.9 (включая duplication, kernel_boundary)
  7. **Build & Step** button (primary): запускает `build_model(spec)` +
     один forward+loss+optimizer.update.

### 4.3 Right sidebar (320px)

Tabbed: **Loss** / **Optim** / **Rewriters** / **Sharding** / **Gotchas**.

**Tab "Loss"** — карточка `LossSpec`:
  - Dropdown: CE / MTP-weighted / IFIM-shaped / MHC-attn-bias / Custom
  - Conditional inputs:
    * MTP: `k` slider (1..8), per-head `beta_i` sliders (visible когда
      k>1), label-source dropdown
    * IFIM: `lambda_fim` slider (0..1), head-output-name input
    * MHC: `lambda_mhc` slider (0..0.5)
    * Custom: function-name input (resolved через registry)
  - "Apply" button → отправляет `loss.update` event.

**Tab "Optim"** — карточка `OptimSpec`:
  - Dropdown: AdamW / Muon / Muon-AdamW Hybrid / SGD
  - Per-group panel (table): matcher (`all`/`moe_experts`/`embeddings`/
    `attention`/`mlp`/`head`/`regex:...`) + lr + weight_decay + betas /
    ns_steps.
  - Global knobs: gradient_clip_norm input, mixed_precision toggle.
  - **+ Add group** button → user wires e.g. `regex:.*expert.*` с
    другим lr.

**Tab "Rewriters"** — порядковый список chip'ов:
  - Drag-drop reorder.
  - Chips: `MTPRewriter(k=2, beta=(1,0.6))`, `IFIMRewriter(λ=0.1)`,
    `MHCRewriter(N=2, λ=0.05)`.
  - Каждый chip раскрывается → внутри inline param editor.
  - Preview button — показывает post-rewrite graph как ghost overlay на
    canvas (head_0/head_1 nodes появляются translucent).
  - Apply button → коммитит rewrites в спеку (canvas обновляется).

**Tab "Sharding"** — два раздела:
  - **Proposals** (top 3-5 от `suggest_sharding`): каждое — карточка
    с strategy_name + per-rank bytes badge + fits/✗ badge + reason +
    "Accept" button.
  - **Custom** — manual axis-assignment table (axis_name + ParallelismKind
    + degree). Toggles для master_weights_fp32 / grad_reduce_dtype /
    fp8_enabled / activation_checkpointing.
  - На accept proposal — топ bar's memory bar пересчитывается, на canvas
    overlay показывает per-axis colour-coding (которые ноды на каком
    мешевом axis).

**Tab "Gotchas"** — список fired-gotchas сгруппированных по severity:
  - Каждый — chip с цветом (red ERROR / yellow WARNING / blue INFO)
  - Клик → раскрывается с message + reference link (открывает
    nanochat/cppmega file в внешнем окне или показывает inline code
    excerpt если есть API).
  - "Auto-fix" button где применимо (например, `compile_mode='regional'`
    fix для fsdp2_whole_compile / megatron_tp_whole_compile).

### 4.4 Bottom strip (32px)

  - Status indicator: "Backend connected" (green dot) / "Reconnecting..."
    (yellow pulsing) / "Disconnected" (red).
  - Last verify latency: "Verify: 4.2ms"
  - Total brick count: "22 bricks, 3 fused regions"
  - "?" help → toggles tutorial overlay.

### 4.5 Modal dialogs

  - **DeviceTopology builder**: text inputs for device kind / hbm / count
    / interconnect / bandwidth + mesh-axis table (axis name + degree) +
    validation hint when product != device count.
  - **Build & Step results**: после нажатия Build & Step — modal с loss
    scalar (finite/NaN), shape of head output, optimizer state-bytes
    delta, error stack trace если что-то упало.
  - **Save / Load**: graph + loss/optim/rewrites/sharding → JSON file.
  - **Export**: код на Python (MLX) — generates a complete
    `cppmega_v4.buildspec.build_model(spec, ...)` invocation как .py
    file. (TorchStack-style Graph2Code.)

### 4.6 Empty state

При пустом canvas — splash card:
  - "Start from a preset" → 12 chips
  - "Drop a brick from the palette"
  - "Read the docs" → link

### 4.7 Real-time interaction loop

1. Юзер тянет ноду на canvas → `graph.mutate` event с дебаунсом 0ms.
2. Юзер редактирует параметр в node-side panel → `param.edit` с дебаунсом
   150ms.
3. После каждого `graph.mutate` или `param.edit` (debounced) — `verify`
   request fires → backend возвращает `DistributedVerificationResult`
   за <5ms (Mode A) или <50ms (Pyodide Mode B).
4. UI обновляет:
   - per-node memory bars
   - edge colours + suggested_fix labels
   - top memory bar
   - gotcha tab badge count
   - sharding tab recompute (если sharding tab открыт)

---

5. API contract — JSON-RPC 2.0

Один envelope для обоих транспортов (anywidget traitlets и WebSocket).

### 5.1 Event taxonomy + latency tiers

| Event              | Trigger                       | Debounce | Latency target | Backend? |
|--------------------|-------------------------------|----------|----------------|----------|
| `node.move`        | drag (position update)        | —        | local-only     | No       |
| `graph.mutate`     | add/remove node OR edge       | 0ms      | <100ms         | Yes      |
| `param.edit`       | edit node params              | 150ms    | <100ms         | Yes      |
| `loss.update`      | LossSpec change               | 0ms      | <50ms          | Yes      |
| `optim.update`     | OptimSpec change              | 0ms      | <50ms          | Yes      |
| `rewriter.add/rm/reorder` | Rewriter chain mutation| 0ms      | <50ms          | Yes      |
| `sharding.update`  | ShardingSpec change           | 0ms      | <100ms         | Yes      |
| `verify.request`   | follows graph/param/etc       | 150ms coalesced | <100ms  | Yes      |
| `sharding.request` | explicit "Suggest sharding"   | n/a      | <2s            | Yes      |
| `build.request`    | "Build & Step" button         | n/a      | <5s            | Yes      |
| `backend.status`   | heartbeat 1Hz                 | —        | —              | Server-push |

Ключ кеша: `sha256(canonical_json(spec))`, исключая позиции узлов.
Frontend держит LRU(50). Backend держит per-step LRU.

### 5.2 Core request/response shapes

**`verify` request:**
```json
{
  "jsonrpc": "2.0", "id": "v_42", "method": "verify",
  "params": {
    "graph": {
      "nodes": [
        { "id": "g0", "kind": "gdn", "params": {"num_heads": 32, "head_dim": 128} },
        { "id": "g1", "kind": "gdn", "params": {"num_heads": 32, "head_dim": 128} },
        { "id": "attn", "kind": "gated_attention",
          "params": {"num_attention_heads": 32, "num_key_value_heads": 4, "head_dim": 128} }
      ],
      "edges": [
        { "src": "g0", "dst": "g1" },
        { "src": "g1", "dst": "attn" }
      ]
    },
    "dim_env": {"B": 1, "S": 4096, "H": 4096, "nh": 32, "nkv": 4, "head_dim": 128,
                "num_experts": 8, "top_k": 2},
    "loss":     {"kind": "cross_entropy", "head_outputs": ["logits"]},
    "optim":    {"kind": "adamw", "groups": [{"matcher": "all", "lr": 3e-4,
                 "weight_decay": 0.01, "betas": [0.9, 0.95]}]},
    "rewriters": [],
    "sharding": {
      "topology": {"factory": "h100_8x", "kwargs": {"dp": 8, "tp": 1, "ep": 1, "pp": 1}},
      "axis_assignments": [{"axis_name": "dp", "kind": "fsdp2", "degree": 8}],
      "compile_mode": "regional", "fp8_enabled": false
    },
    "training": true,
    "available_side_channels": ["doc_ids", "token_ids"]
  }
}
```

**`verify` response:**
```json
{
  "jsonrpc": "2.0", "id": "v_42",
  "result": {
    "resolved": {
      "edges": [
        { "src": "g0", "dst": "g1", "shape": [1, 4096, 4096],
          "matched": true, "severity": "info" },
        { "src": "g1", "dst": "attn", "shape": [1, 4096, 4096],
          "matched": true, "severity": "info" }
      ],
      "diagnostics": [],
      "has_errors": false
    },
    "memory_per_brick": {
      "g0":   { "params_bytes": 16777216, "activations_bytes": 4194304, "kv_cache_bytes": 0 },
      "g1":   { "params_bytes": 16777216, "activations_bytes": 4194304, "kv_cache_bytes": 0 },
      "attn": { "params_bytes": 67108864, "activations_bytes": 16777216,
                "kv_cache_bytes": 8388608 }
    },
    "memory_distributed": {
      "worst_rank_idx": 0,
      "worst_rank": {
        "weights_bytes": 12345678, "grads_bytes": 12345678,
        "optimizer_state_bytes": 49382712,
        "activations_bytes": 8388608,
        "fsdp_allgather_peak_bytes": 1234567,
        "kv_cache_bytes": 0, "moe_routing_buffers_bytes": 0,
        "collective_workspace_bytes": 0,
        "framework_overhead_bytes": 2147483648,
        "master_weights_bytes": 0,
        "total_bytes": 2233445566
      },
      "duplication_bytes": 0,
      "master_weights_overhead_bytes": 0,
      "kernel_boundary_materialisation_bytes": 0,
      "fits_on_topology": true
    },
    "gotchas": [
      { "id": "fsdp_allgather_peak_unsharded", "severity": "info",
        "message": "FSDP all-gather peak == unsharded parameter size...",
        "reference": "nanochat/fsdp_cuda.py" }
    ],
    "fusion_plan": [
      { "brick_names": ["g0", "g1"], "backend": "path_c", "is_fused": true,
        "estimated_savings_us": 8.5 },
      { "brick_names": ["attn"], "backend": "dlpack_handoff", "is_fused": false }
    ],
    "elapsed_ms": 4.2
  }
}
```

**`suggest_sharding` response:**
```json
{
  "result": {
    "proposals": [
      { "strategy_name": "fsdp2_only", "fits": true,
        "estimated_per_rank_bytes": 2233445566,
        "reason": "FSDP2 across dp axis...",
        "num_errors": 0, "sharding": {/* full ShardingSpec */} },
      { "strategy_name": "fsdp2_plus_tp", "fits": true,
        "estimated_per_rank_bytes": 1845000000,
        "reason": "3D parallelism: FSDP2 + TP=2",
        "num_errors": 0, "sharding": {/* */} }
    ]
  }
}
```

**`build.request` response:**
```json
{
  "result": {
    "built": true,
    "spec_applied": { "graph": {/*post-rewrite*/}, "loss": {/*post-rewrite*/} },
    "forward_step": {
      "loss_scalar": 10.42, "finite": true,
      "output_shape": [1, 4096, 32000],
      "optim_state_delta_bytes": 4096
    },
    "elapsed_ms": 1234
  }
}
```

### 5.3 Error envelope

```json
{
  "jsonrpc": "2.0", "id": "v_42",
  "error": {
    "code": -32602, "message": "Invalid params",
    "data": {
      "type": "ResolveError",
      "detail": "missing dim_env entries: ['head_dim']; expr=('B', 'S', 'nh * head_dim')",
      "node_id": "g0"
    }
  }
}
```

UI poсредник: ошибки кода маппит на in-place toast + node-highlight.

### 5.4 Edge states

  - `backend.status` лоу 1Hz. При отсутствии → top-bar status badge становится
    yellow "Reconnecting...", structural edits dimmed.
  - `verify.request` дольше 500ms → ноды показывают pulsing animation
    "evaluating".
  - Partial result (build_model упал в середине) → backend всё равно
    возвращает frame с error.data.partial=true и greyed-out failed nodes.

---

6. Mapping — UI control → cppmega_v4.* entity → последствия

| UI surface          | Triggers backend call         | Maps to spec field                        |
|---------------------|-------------------------------|--------------------------------------------|
| Drag brick to canvas| `graph.mutate` → `verify`     | `BrickGraph.nodes[+1]`, `BrickNode(kind, name, params)` |
| Edit node param     | `param.edit` (debounced) → `verify` | `BrickNode.params[key] = value`       |
| Connect two ports   | `graph.mutate` → `verify`     | `BrickGraph.edges[+1]`                     |
| Disconnect          | `graph.mutate` → `verify`     | `BrickGraph.edges[-1]`                     |
| Click red edge label| `graph.mutate` (apply adapter)| `insert_adapter_chain(graph, p, c, suggestion)` |
| Preset dropdown     | `graph.mutate` (replace all)  | `build_preset_specs(name, H, num_layers)`  |
| Loss dropdown       | `loss.update` → `verify`      | `LossSpec(kind, params, head_outputs)`     |
| Optim dropdown      | `optim.update` → `verify`     | `OptimSpec(kind, groups, ...)`             |
| Add param group     | `optim.update` → `verify`     | `OptimSpec.groups[+1]`                     |
| Add MTPRewriter chip| `rewriter.add` → `verify`     | `ModelBuildSpec.rewrites[+1]`              |
| Reorder rewriters   | `rewriter.reorder` → `verify` | Tuple permutation                          |
| Topology dropdown   | `sharding.update` → `verify`  | `DeviceTopology` factory                   |
| Mesh axis editor    | `sharding.update` → `verify`  | `DeviceTopology.mesh_axes`                 |
| Sharding proposal accept | `sharding.update` → `verify` | `ShardingSpec` replaced                |
| compile_mode dropdown | `sharding.update` → `verify`| `ShardingSpec.compile_mode`                |
| fp8_enabled toggle  | `sharding.update` → `verify`  | `ShardingSpec.fp8_enabled`                 |
| master_weights toggle | `sharding.update` → `verify`| `ShardingSpec.master_weights_fp32`         |
| activation_checkpointing | `sharding.update` → `verify` | `ShardingSpec.activation_checkpointing` |
| Auto-fix gotcha     | `sharding.update`             | `ShardingSpec.compile_mode='regional'`     |
| Build & Step button | `build.request`               | `build_model(spec)` → `BuiltModel`         |
| Export button       | (local download)              | `python_codegen(spec)` → .py file          |

---

7. Реализация поэтапно

5 этапов, ~2-3 недели на каждый при solo developer; ~10 недель end-to-end.

**Этап F-A — JSON-RPC contract + Python jsonrpc server (1 неделя)**
  - Зафиксировать JSON schema (TypeScript + Pydantic в shared package)
  - FastAPI standalone server обёртка над `verify_distributed_plan`,
    `suggest_sharding`, `build_model`, `verify_and_estimate`
  - Test suite: golden round-trip JSON для каждого endpoint
  - Cache layer с LRU(50)

**Этап F-B — React Flow canvas + 22 brick custom nodes (2 недели)**
  - Vite + React 18 + xyflow scaffolding
  - 22 brick custom node components (color-coded, inline param editor,
    memory bar placeholder)
  - 6 adapter nodes (dashed border, ghost preview)
  - ELK.js auto-layout в Web Worker
  - Palette на dragstart, canvas drop zones
  - Edge validation hook (`isValidConnection` через cached shape data)

**Этап F-C — Sidebar (Loss/Optim/Rewriters/Sharding/Gotchas) (2 недели)**
  - 5 tabs с form components
  - Real-time wire to verify endpoint
  - Sharding proposal cards с accept button
  - Gotcha chips с reference links
  - Top bar memory aggregate

**Этап F-D — Anywidget shim + Jupyter packaging (1 неделя)**
  - `cppmega-builder-widget` PyPI package
  - Anywidget traitlets binding
  - Example notebook
  - JupyterLab + JupyterLite compatibility test

**Этап F-E — JupyterLite/Pyodide static bundle + GH Pages deploy (1 неделя)**
  - Pyodide preload `cppmega_v4` (отделить от MLX runtime)
  - GitHub Pages auto-deploy on main commit
  - Public demo URL
  - Telemetry-free, no backend

**Phase 2 (optional, 4-6 недель)**:
  - Rust shape engine → WASM (sub-ms validation hot path)
  - Multi-user CRDT collaboration (Yjs)
  - GraphQL gateway (если нужны облачные deployments)
  - Custom-brick plugin SDK (researcher определяет свой brick через
    Python decorator + JSON schema, попадает в палитру)

---

8. Что заимствуем у production-инструментов

  - **PerceptiLabs** — live shape inference + per-node warnings/tips
  - **Houdini SOPs/VOPs** — типизированные сокеты + cook-on-change
    deferred evaluation
  - **TouchDesigner** — per-node cost overlay в реальном времени
  - **Blender Geometry Nodes** — color-coded sockets по type
  - **Figma** — headless core (Rust→WASM) + thin React shell, shareable
    URLs, "one codebase, multiple shells"
  - **TorchStack** — live Graph2Code (always-up-to-date Python export)
  - **NodeTool** — workflow portable между desktop/cloud/notebook
  - **ComfyUI** — Python ↔ JS WebSocket reference architecture
  - **Rerun** (Rust+egui, multimodal) — single-binary distribution
    inspiration для desktop variant

---

9. Что у нас будет уникально (не существует ни у кого)

Из 9 research-агентов: **никто не решает следующее**, мы займём white space:

  1. **Distributed-topology auto-shard visualisation** — единственные
     с per-axis colour overlay и `suggest_sharding` proposals в одном UI
  2. **FP8/bf16 duplication callouts** — никто не показывает gap "fp8
     forward + bf16 grad = 3× param-elements"
  3. **Gotcha-checker UI с провенансом** — 15 known footguns с прямыми
     ссылками на nanochat/cppmega files
  4. **Per-node memory bar с fusion-aware activations** — внутри fused
     region активации `max`, не `sum` — это видно в реальном времени
  5. **3D-parallelism (TP×DP×EP×PP) planner** integrated с shape
     validation и memory accounting
  6. **Hybrid block-pattern editor** для Jamba/Mamba-MoE архитектур
     (повторяющиеся `[M, M, A, M, MoE] × N` patterns)
  7. **Recompute/AC region painter** на canvas с FLOP/memory tradeoff
  8. **Kernel-path selector overlay** (TileLang/MSL/Triton/MLX path per
     node)

---

10. Бюджет + риски

Бюджет: 7 недель (F-A..F-E) для v1 production-ready Jupyter widget +
static demo. Phase 2 опционально через 3-6 месяцев.

**Главный риск**: React Flow DOM ceiling ~1000 nodes. Mitigation:
collapse repeated-layer stacks в один "×N" node (UI-side abstraction);
expand on demand. Если упрёмся реально, переходим на Comfy-Org litegraph
fork (canvas-based, 1000+ verified).

**Второй риск**: Pyodide ↔ real MLX divergence. Mitigation: estimator
math должен быть byte-identical в обоих runtime'ах. Golden-output test
suite в CI.

**Третий риск**: Anywidget + JupyterLite + React Flow совместимость —
flaky в реальном мире. Mitigation: ранний spike в F-D перед finalizing
архитектуру.

**Четвёртый риск**: Sharding cost-model fidelity. Our PSpec layer уже
±10-15% точности; для GUI "OOM / fits" этого хватает, но юзеры могут
ожидать ±5%. Mitigation: calibration loop с реальными H100 замерами в
Phase 2.

**Пятый риск**: Custom-brick plugin SDK не дизайнен. Mitigation: Phase 2
work — для v1 поддерживаем только 22 hardcoded brick kinds.

---

Источники research-агентов:
  - Code extraction: cppmega.mlx (22 bricks + полный feature catalogue),
    ../cppmega (Megatron stack catalogue), ../nanochat (~280 GPTConfig
    fields + memory_estimator.py reference)
  - Web (6 agents): xyflow.com, retejs.org, comfy-org github, Figma blog,
    Modular synthesis blackjack repo, awesome-node-based-uis index,
    Jamba paper (arXiv:2403.19887, 2408.12570), MambaVision (CVPR 2025),
    PerceptiLabs docs, TorchStack landing, NodeTool docs, anywidget docs.
