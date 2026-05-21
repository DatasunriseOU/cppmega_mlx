План: Visual-Builder Spec Layer v2 — Pluggable Optimizers + Activations
  + Norms + LR Schedules + Tooltips + Ablations + Dim Feedback

Цель: дать GUI выбор любой комбинации (optimizer × activation × norm ×
schedule) ВЕЗДЕ где это имеет смысл — с tooltip-обоснованием каждого
варианта и recommended params — чтобы инженер не угадывал "почему swiglu"
или "почему Lion с 1e-4 не 1e-3", а видел рядом ответ. Поверх этого
живёт ablation runner: выбрал ось замены → жмёшь Run → видишь как
изменение влияет на loss трактории за 20 шагов, side-by-side.

Лежит **поверх** VBSpec / MBSpec / PSpec / E2E Coverage Matrix v1.
Закрывает дыры, найденные при аудите перед эпиком cppmega-mlx-bb0:

  - Lion есть в `cppmega_mlx/training/optimizers.py` но не зарегистрирован
    в OptimKind enum (UI его не видит).
  - `ActivationName` Literal содержит только `["gelu","relu2","swiglu"]`,
    хотя в коде уже есть `silu`, `squared_relu`, и нужны ещё
    `mish/geglu/reglu/xielu` для покрытия LLM-state-of-the-art.
  - Норма везде хардкод RMSNorm (LayerNorm только в lightning_indexer).
  - Schedules нет вообще (LR только float или Callable, юзер пишет сам).
  - Tooltips нет — пользователь не понимает что такое Muon ns_steps или
    почему Lion=1e-4.
  - Ablation runner отсутствует — нельзя за один клик увидеть как замена
    swiglu→gelu повлияет на loss.

---
1. Что уже есть (карта существующего)

Optimizers (cppmega_mlx/training/optimizers*.py)
  AdamWFP32Moments, LionFP32Moments, MuonWithNSCarrier, SGD,
  Lion8bit, Adam8bit, QuantizedMuonWithNSCarrier, MuonAdamWMulti.
  Все производят `mlx.optimizers.Optimizer`-совместимые объекты, у всех
  поддерживается `learning_rate: float | Callable[[mx.array], mx.array]`.

OptimKind enum (cppmega_v4/buildspec/optim_spec.py)
  Содержит: ADAMW, MUON, MUON_ADAMW_HYBRID, SGD.
  Factory functions: adamw(), muon(), muon_adamw_hybrid(), sgd().
  Не содержит: LION, LION_8BIT, ADAM_8BIT.

ParamGroup matchers (7 builtin)
  "all", "moe_experts", "embeddings", "attention", "mlp", "head",
  "regex:<pattern>".

Activations (cppmega_mlx/nn/moe.py)
  `ActivationName = Literal["gelu","relu2","swiglu"]`.
  Реализации: `gelu` = `nn.gelu_approx`, `silu` = `nn.silu` (но не в
  Literal), `swiglu` = `silu(gate) * up`, `relu2` = `square(max(0, x))`.
  Metal kernel `squared_relu` в `kernels/metal_ops.py:143` — не подключён.

Norms
  Везде `mlx.nn.RMSNorm` хардкод. `mlx.nn.LayerNorm` только в
  `cppmega_v4/nn/lightning_indexer.py`. Custom норм нет.

Schedules
  Отсутствуют. `optim_spec.OptimSpec` принимает `lr: float | Callable`,
  но никаких готовых `cosine_annealing()` / `linear_warmup()` нет.

JSON-RPC methods (cppmega_v4/jsonrpc/, 10 шт. сейчас)
  verify, suggest_sharding, suggest_adapters, build_preset_specs,
  probe.run, pipeline.run, tokenizer.encode_visualize,
  tokenizer.list_presets, data.preview_parquet, backend.status.

UI sidebar tabs (vbgui/src/components/sidebar/, 5 шт.)
  LossTab, OptimTab, RewritersTab, ShardingTab, GotchasTab.
  OptimTab.KINDS = `["adamw","muon","muon_adamw_hybrid","sgd"]` —
  без Lion. Каждая `ParamGroup` редактируется (matcher / lr / wd), но
  schedules нет.

---
2. Что НЕ хватает (gap-analysis)

┌──────────────────────────────────────────────┬──────────────────────────┐
│              Что нужно                       │           Сейчас         │
├──────────────────────────────────────────────┼──────────────────────────┤
│ Lion / Lion8bit / Adam8bit в OptimKind enum  │ ✗                        │
├──────────────────────────────────────────────┼──────────────────────────┤
│ Расширенный ActivationName (10 вариантов)    │ 3 из 10                  │
├──────────────────────────────────────────────┼──────────────────────────┤
│ Activation параметр в _build_mlp (не моё MoE)│ ✗ (SwiGLU хардкод)       │
├──────────────────────────────────────────────┼──────────────────────────┤
│ pre_norm / post_norm параметры в BrickSpec   │ ✗                        │
├──────────────────────────────────────────────┼──────────────────────────┤
│ LR Schedule factories (cosine/warmup/wsd/…)  │ ✗                        │
├──────────────────────────────────────────────┼──────────────────────────┤
│ Tooltip catalogue (что/зачем/recommended)    │ ✗                        │
├──────────────────────────────────────────────┼──────────────────────────┤
│ catalog.explain RPC                          │ ✗                        │
├──────────────────────────────────────────────┼──────────────────────────┤
│ ablation.run RPC                             │ ✗                        │
├──────────────────────────────────────────────┼──────────────────────────┤
│ suggest_optim_groups RPC                     │ ✗                        │
├──────────────────────────────────────────────┼──────────────────────────┤
│ data.roundtrip_check RPC                     │ ✗ (есть только cap flag) │
├──────────────────────────────────────────────┼──────────────────────────┤
│ architectures.list_presets RPC               │ ✗ (UI хардкод 57/62)     │
├──────────────────────────────────────────────┼──────────────────────────┤
│ inference_log в resolve_shapes response      │ ✗                        │
├──────────────────────────────────────────────┼──────────────────────────┤
│ UI: per-brick context panel (click→edit)     │ ✗                        │
├──────────────────────────────────────────────┼──────────────────────────┤
│ UI: Dimensions sidebar tab                   │ ✗                        │
├──────────────────────────────────────────────┼──────────────────────────┤
│ UI: Ablations sidebar tab                    │ ✗                        │
├──────────────────────────────────────────────┼──────────────────────────┤
│ UI: Schedule editor в OptimTab               │ ✗                        │
├──────────────────────────────────────────────┼──────────────────────────┤
│ UI: tooltips на каждый dropdown option       │ ✗                        │
├──────────────────────────────────────────────┼──────────────────────────┤
│ UI: Auto-group button в OptimTab             │ ✗                        │
└──────────────────────────────────────────────┴──────────────────────────┘

---
3. Архитектура (новые модули)

3.1 cppmega_v4/buildspec/optim_spec.py — Lion / Lion8bit / Adam8bit
    регистрация (для bb0.12)

  class OptimKind(str, Enum):
      ADAMW             = "adamw"
      MUON              = "muon"
      MUON_ADAMW_HYBRID = "muon_adamw_hybrid"
      LION              = "lion"        # ← new
      LION_8BIT         = "lion8bit"    # ← new
      ADAM_8BIT         = "adam8bit"    # ← new
      SGD               = "sgd"

  def lion(
      lr: float = 1e-4,                 # ← intentionally 3-10x < adamw
      weight_decay: float = 0.0,
      betas: tuple[float, float] = (0.9, 0.99),
  ) -> OptimSpec: ...

  def lion8bit(
      lr: float = 1e-4,
      weight_decay: float = 0.0,
      betas: tuple[float, float] = (0.9, 0.99),
      quant_scheme: str = "linear",
      block_size: int = 256,
  ) -> OptimSpec: ...

  def adam8bit(
      lr: float = 3e-4,
      weight_decay: float = 0.01,
      betas: tuple[float, float] = (0.9, 0.999),
      eps: float = 1e-8,
      quant_scheme: str = "linear",
  ) -> OptimSpec: ...

  Validation в verify_optim_spec:
    - LION/LION_8BIT lr > 5e-4 → WARNING ("Lion recommended ≤5e-4 to
      avoid NaN; gradient magnitude unused, only sign")
    - LION требует betas (default OK)

3.2 cppmega_v4/buildspec/schedules.py — LR Schedule factories (для bb0.9)

  @dataclass(frozen=True)
  class ScheduleSpec:
      kind: Literal["constant","cosine","linear_warmup","wsd",
                    "inv_sqrt","polynomial"]
      warmup_steps: int = 0
      total_steps: int | None = None        # None for inv_sqrt/constant
      min_lr_ratio: float = 0.1             # for cosine/wsd
      decay_steps: int | None = None        # for wsd
      power: float = 2.0                    # for polynomial

      def build(self, base_lr: float) -> Callable[[int], float]: ...

  ParamGroup.lr теперь:
      lr: float | ScheduleSpec

  Factory wrappers:
      def cosine_annealing(total_steps, min_lr_ratio=0.1, warmup_steps=0)
          → ScheduleSpec
      def linear_warmup_then_constant(warmup_steps) → ScheduleSpec
      def wsd(warmup_steps, decay_steps, min_lr_ratio=0.1) → ScheduleSpec
      def inv_sqrt(warmup_steps) → ScheduleSpec
      def polynomial(total_steps, power=2.0) → ScheduleSpec
      def constant() → ScheduleSpec

  Validation:
    - kind ∈ ("cosine","wsd","polynomial") → total_steps обязателен
    - kind="wsd" → decay_steps ≥ 1, total_steps ≥ warmup + decay
    - warmup_steps ≥ 0, total_steps ≥ warmup_steps

3.3 cppmega_mlx/nn/activations.py — Activation registry (новый, для bb0.13)

  ActivationName = Literal[
      "gelu","relu","relu2","sqrelu","silu","mish",
      "swiglu","geglu","reglu","xielu",
  ]

  IS_GATED: dict[ActivationName, bool] = {
      "gelu": False, "relu": False, "relu2": False, "sqrelu": False,
      "silu": False, "mish":  False,
      "swiglu": True, "geglu": True, "reglu": True, "xielu": True,
  }

  def apply_activation(name: ActivationName, x: mx.array,
                       gate: mx.array | None = None) -> mx.array:
      """Dispatch to the right impl. Gated activations require `gate`."""
      if IS_GATED[name] and gate is None:
          raise ValueError(f"{name} requires gate input")
      ...

  Реализации:
    - gelu → nn.gelu_approx
    - relu → mx.maximum(x, 0)
    - relu2 → mx.square(mx.maximum(x, 0))
    - sqrelu → metal kernel squared_relu (training-aware)
    - silu → nn.silu
    - mish → x * mx.tanh(nn.softplus(x))
    - swiglu → nn.silu(gate) * up
    - geglu → nn.gelu_approx(gate) * up
    - reglu → mx.maximum(gate, 0) * up
    - xielu → nn.gelu_approx(linear_gate(gate)) * up  (extended)

3.4 cppmega_v4/buildspec/api.py — _build_mlp параметризация (для bb0.5)

  Сейчас _build_mlp всегда делает SwiGLU (sigmoid(gate)*up).
  Расширить:

  def _build_mlp(hidden_size, params):
      activation = params.get("activation", "swiglu")
      pre_norm = params.get("pre_norm", "rmsnorm")
      post_norm = params.get("post_norm", "none")
      intermediate_size = params.get("intermediate_size", 4 * hidden_size)
      ...
      # Validation: dense activation? → один up projection, без gate.
      # Gated activation? → два projections (gate, up), apply_activation
      #                     с обоими.

  Расширить BrickSpec.params validation: activation должен быть в
  ActivationName Literal. Несовместимая gated activation на брике-без-
  gating → BuildDiagnostic ERROR.

3.5 cppmega_v4/buildspec/api.py — Norm parameterization (для bb0.6)

  Каждый builder (attention/mlp/moe/gated_attention/...) принимает
  pre_norm / post_norm:

      def _build_attention(hidden_size, params):
          pre_norm = params.get("pre_norm", "rmsnorm")
          post_norm = params.get("post_norm", "none")
          ...

  norm_kind: Literal["rmsnorm", "layernorm", "none"]

  При "none" — норма не вставляется. Validation: pre_norm и post_norm
  не могут оба быть "none" (тогда residual без нормы взрывает grad).
  Для parallel-block (attention || mlp) — оба брика должны иметь
  pre_norm != "none".

3.6 cppmega_v4/explain/catalog.py — Tooltip catalogue (для bb0.10)

  @dataclass(frozen=True)
  class ExplainEntry:
      category: Literal["optimizer","activation","norm","schedule",
                        "loss","rewriter","brick"]
      name: str
      summary: str                  # 1-line, для tooltip hover
      when_to_use: str              # 2-3 sentences
      when_to_avoid: str            # 1-2 sentences
      recommended_params: dict[str, Any]  # auto-populate в UI
      paper_ref: str | None         # "Chen et al. 2023"
      paper_url: str | None
      gotchas: tuple[str, ...]      # известные грабли

  CATALOG: dict[tuple[str, str], ExplainEntry] = {
      ("optimizer", "adamw"): ExplainEntry(
          summary="Adam with decoupled weight decay (Loshchilov 2017).",
          when_to_use="Default for transformer pretraining. Robust to "
                      "hyperparams. Use for any dense LLM <100B.",
          when_to_avoid="Memory-constrained training (2× momentum "
                        "buffers); large 2D weight matrices where Muon "
                        "is 1.5-2× faster.",
          recommended_params={"lr": 3e-4, "betas": (0.9, 0.95),
                              "weight_decay": 0.01, "eps": 1e-8},
          paper_ref="Loshchilov & Hutter, 2017",
          paper_url="https://arxiv.org/abs/1711.05101",
          gotchas=("eps too small → NaN on bf16",
                   "lr > 1e-3 typically diverges for >1B models"),
      ),
      ("optimizer", "lion"): ExplainEntry(
          summary="Sign-based momentum, 50% less state than AdamW "
                  "(Chen et al. 2023).",
          when_to_use="Memory-constrained; works well from 100M-7B. "
                      "Single momentum buffer = half AdamW state.",
          when_to_avoid="Small batch sizes (<64) — sign updates noisy.",
          recommended_params={"lr": 1e-4, "betas": (0.9, 0.99),
                              "weight_decay": 0.01},
          paper_ref="Chen et al., 2023",
          paper_url="https://arxiv.org/abs/2302.06675",
          gotchas=("lr > 5e-4 → NaN (gradient magnitude not used, only "
                   "sign — lr needs to be 3-10x smaller than AdamW)",
                   "Less stable on <100M params"),
      ),
      ("optimizer", "muon"): ExplainEntry(
          summary="Newton-Schulz orthogonalization of 2D weight grads "
                  "(Jordan 2024).",
          when_to_use="2D linear weight matrices in transformer backbone "
                      "(QKV/output proj, FFN). 1.5-2× faster than AdamW.",
          when_to_avoid="1D params (biases, norms) — use AdamW. "
                        "Embeddings (1D lookup) — use AdamW.",
          recommended_params={"lr": 2e-3, "momentum": 0.95, "ns_steps": 5,
                              "weight_decay": 0.01, "ns_carrier": "fp32"},
          paper_ref="Jordan, 2024 (Modula)",
          paper_url="https://kellerjordan.github.io/posts/muon/",
          gotchas=("ns_steps < 3 → poor orthogonalization",
                   "fp16 ns_carrier loses precision on large matrices",
                   "1D params SKIP Muon update — use hybrid"),
      ),
      ("optimizer", "muon_adamw_hybrid"): ExplainEntry(
          summary="Muon на 2D weights, AdamW на embeddings/norms/head/MoE.",
          when_to_use="Default for >1B LLM pretraining. Best of both: "
                      "Muon speed where it applies, AdamW where Muon "
                      "skips (1D + lookup tables).",
          when_to_avoid="Memory-constrained — use Lion or AdamW.",
          recommended_params={"muon_lr": 2e-3, "adam_lr": 1e-4,
                              "ns_steps": 5, "weight_decay": 0.01},
          paper_ref="Jordan, 2024 + Loshchilov, 2017",
          paper_url="https://kellerjordan.github.io/posts/muon/",
          gotchas=("matcher misclassification → Muon update applied to "
                   "1D bias → degraded loss",
                   "MoE experts MUST be in AdamW group, not Muon"),
      ),
      ("activation", "swiglu"): ExplainEntry(
          summary="Gated SiLU (Shazeer 2020). Default LLM activation.",
          when_to_use="Best quality across model scales. Used in "
                      "LLaMA/Qwen/DeepSeek/Mistral.",
          when_to_avoid="Memory-bound training — 1.5× param footprint "
                        "vs dense MLP (gate + up).",
          recommended_params={"intermediate_size": "auto (4*H or 8/3*H)"},
          paper_ref="Shazeer, 2020",
          paper_url="https://arxiv.org/abs/2002.05202",
          gotchas=("Requires gated_mlp brick (with gate projection)",
                   "Not compatible with dense mlp"),
      ),
      ("activation", "relu2"): ExplainEntry(
          summary="Squared ReLU (Hua et al. 2022). Fast, sparse-ish.",
          when_to_use="TPU/GPU pretraining where compute matters. "
                      "Sparser activation than GELU (some units exactly "
                      "0) — slightly better for downstream pruning.",
          when_to_avoid="Small models <100M — sparsity hurts capacity.",
          recommended_params={},
          paper_ref="Hua et al., 2022 (T5-Pile)",
          paper_url="https://arxiv.org/abs/2109.08668",
          gotchas=("Larger output magnitude than ReLU → may need scaled "
                   "weight init",),
      ),
      ("activation", "geglu"): ExplainEntry(
          summary="Gated GELU (Shazeer 2020). GLU variant с gelu gate.",
          when_to_use="LLM finetuning от моделей которые тренились с "
                      "GELU; GLM/Falcon variants.",
          when_to_avoid="Same as swiglu (memory).",
          recommended_params={},
          paper_ref="Shazeer, 2020",
          paper_url="https://arxiv.org/abs/2002.05202",
          gotchas=("Requires gated_mlp brick",),
      ),
      ("schedule", "cosine"): ExplainEntry(
          summary="Cosine annealing (Loshchilov & Hutter 2016).",
          when_to_use="Default for LLM pretraining (Chinchilla). "
                      "Smooth decay, no abrupt LR drops.",
          when_to_avoid="Checkpoint reuse / continued training — "
                        "decay state hard to resume; use WSD instead.",
          recommended_params={"warmup_steps": 2000,
                              "total_steps": "auto",
                              "min_lr_ratio": 0.1},
          paper_ref="Loshchilov & Hutter, 2016",
          paper_url="https://arxiv.org/abs/1608.03983",
          gotchas=("total_steps must match actual training duration",),
      ),
      ("schedule", "wsd"): ExplainEntry(
          summary="Warmup → Steady → Decay (DeepSeek-V2). Allows "
                  "checkpoint reuse mid-training.",
          when_to_use="Long training runs (>100K steps) where "
                      "checkpoints may be reused. DeepSeek default.",
          when_to_avoid="Short runs (<10K steps) — cosine is simpler.",
          recommended_params={"warmup_steps": 2000, "decay_steps": 5000,
                              "min_lr_ratio": 0.1},
          paper_ref="DeepSeek-V2 tech report, 2024",
          paper_url="https://arxiv.org/abs/2405.04434",
          gotchas=("decay phase requires fixed total_steps — set "
                   "carefully if extending training",),
      ),
      ("norm", "rmsnorm"): ExplainEntry(
          summary="Root mean square normalization (Zhang & Sennrich 2019).",
          when_to_use="Default for LLM pretraining. Simpler/faster than "
                      "LayerNorm (no mean centering).",
          when_to_avoid="Models where mean offset matters (BERT-style "
                        "encoder, some vision).",
          recommended_params={"eps": 1e-6},
          paper_ref="Zhang & Sennrich, 2019",
          paper_url="https://arxiv.org/abs/1910.07467",
          gotchas=("eps < 1e-8 → NaN on bf16",),
      ),
      ("norm", "layernorm"): ExplainEntry(
          summary="LayerNorm (Ba et al. 2016) — mean + variance norm.",
          when_to_use="GPT-style models; finetuning от моделей которые "
                      "тренились с LayerNorm.",
          when_to_avoid="LLaMA-family (use RMSNorm).",
          recommended_params={"eps": 1e-5},
          paper_ref="Ba et al., 2016",
          paper_url="https://arxiv.org/abs/1607.06450",
          gotchas=("~10% slower than RMSNorm",),
      ),
      # ... (полный каталог — ~50 entries: 7 optimizers × 10 activations
      # × 3 norms × 6 schedules × 5 losses × N rewriters × 26 bricks)
  }

3.7 cppmega_v4/jsonrpc/methods/catalog.py — catalog.explain RPC

  class CatalogExplainParams(BaseModel):
      category: Literal["optimizer","activation","norm","schedule",
                        "loss","rewriter","brick"]
      name: str

  class CatalogExplainResult(BaseModel):
      entry: ExplainEntry | None
      not_found_message: str | None

  Метод просто читает CATALOG[(category, name)] и возвращает.
  Cacheable (LRU).

  Также добавить catalog.list_options:
    Params: {category}
    Result: {options: [{name, summary, paper_ref}, ...]} — для
    populate dropdowns.

3.8 cppmega_v4/jsonrpc/methods/ablation.py — ablation.run RPC (для bb0.11)

  class AblationRunParams(BaseModel):
      base_spec: VerifyParams           # canvas state
      ablation_axis: Literal["optimizer","activation","norm","schedule"]
      variants: list[str]               # ["swiglu","gelu","relu2"]
      num_steps: int = 20
      step_options: dict[str, Any] = {} # lr / betas overrides

  class AblationVariantResult(BaseModel):
      variant: str
      status: Literal["ok","fail"]
      losses: list[float]               # per step
      elapsed_ms: float
      weight_delta_norm: float
      error: dict | None

  class AblationRunResult(BaseModel):
      results: list[AblationVariantResult]
      ranked_by_final_loss: list[str]
      baseline_variant: str             # тот что в base_spec
      elapsed_ms_total: float

  Реализация: для каждого variant клонировать base_spec, заменить
  ablation_axis (через mutate_spec helper), запустить
  stage_train с num_steps шагов, собрать losses.
  Ablations выполняются последовательно (mlx single-GPU); параллелизация
  через subprocess pool — Phase 2.

3.9 cppmega_v4/jsonrpc/methods/suggest_optim_groups.py — для bb0.4

  class SuggestOptimGroupsParams(BaseModel):
      graph: BrickGraph
      optim_kind: OptimKind             # лидер группа (например muon_adamw_hybrid)
      hidden_size: int = 128

  class ProposedGroup(BaseModel):
      matcher: str
      optim_kind: OptimKind             # e.g. Muon for 2D, AdamW for 1D
      lr: float
      weight_decay: float
      betas: tuple[float, float] | None
      param_count: int                  # сколько params покрывает
      rationale: str                    # "12 2D linear weights"

  class SuggestOptimGroupsResult(BaseModel):
      proposals: list[ProposedGroup]
      total_params: int
      uncovered_params: int             # warning if > 0

  Эвристика:
    - Сначала materialize graph (instantiate=True), собрать все
      parameters().
    - Группировать по rules:
        Embedding/lm_head (1D lookup) → AdamW
        MoE expert weights → AdamW (не Muon — экспертные апдейты)
        2D linear weights (Q/K/V/O, MLP gate/up/down) → Muon
        1D bias / norm gain → AdamW
    - Возвращать matcher regex для каждой группы.

3.10 cppmega_v4/jsonrpc/methods/architectures.py — list_presets (для bb0.8)

  class ListArchitecturesResult(BaseModel):
      presets: list[str]                # sorted

  Просто `sorted(PRESETS.keys())`. UI вызывает один раз на mount.

3.11 cppmega_v4/jsonrpc/methods/data.py — roundtrip_check (для bb0.3)

  class RoundtripCheckParams(BaseModel):
      parquet_path: str
      tokenizer_source: str
      max_rows: int = 8

  class RoundtripRow(BaseModel):
      row_idx: int
      original_bytes: int
      decoded_bytes: int
      matches: bool
      byte_diff: int                    # 0 if perfect roundtrip

  class RoundtripCheckResult(BaseModel):
      rows: list[RoundtripRow]
      tokenizer_capability: Literal["exact","approx","none"]
      pass_rate: float                  # fraction matching
      elapsed_ms: float

  Использует tokenizer.encode_visualize capabilities + decode по
  input_ids → сравнивает с 'original_text' колонкой parquet.

3.12 cppmega_v4/spec/resolver.py — inference_log (для bb0.2)

  Расширить ResolvedBrickGraph:

  @dataclass(frozen=True)
  class InferenceEntry:
      brick: str
      param: str
      value: int | str
      source: Literal["user","auto"]
      reason: str                       # "H=128/head_dim=64 → 2"

  @dataclass(frozen=True)
  class ResolvedBrickGraph:
      # ... existing fields ...
      inference_log: tuple[InferenceEntry, ...]

  Источник 'user' если параметр был передан в BrickSpec.params, 'auto'
  если выведен в resolve_shapes из dim_env / правил.

---
4. UI Surface (новые компоненты + расширения)

4.1 vbgui/src/components/sidebar/OptimTab.tsx — расширение (bb0.4, bb0.9, bb0.12)

  KINDS array → ["adamw","muon","muon_adamw_hybrid","lion","lion8bit",
                 "adam8bit","sgd"]

  Новая кнопка [Auto-group from graph] → вызывает suggest_optim_groups
  RPC → заменяет groups с показом rationale в tooltip:
    "Muon on regex:.*\\.weight$ — covers 12 2D linear weights (q_proj,
     k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj × 2 layers)"

  Schedule editor per ParamGroup:
    Schedule kind dropdown ["constant","cosine","linear_warmup","wsd",
                            "inv_sqrt","polynomial"]
    Conditional fields: warmup_steps / total_steps / min_lr_ratio / decay_steps
    Mini-sparkline preview (SVG): рендерит LR curve по 50 точкам.

  Info icon (ⓘ) рядом с каждым label → modal с explanation
  (catalog.explain RPC).

4.2 vbgui/src/components/sidebar/DimensionsTab.tsx (НОВЫЙ, bb0.2)

  Таблица:
    | Brick     | Param            | Value | Source | Reason          |
    |-----------|------------------|-------|--------|-----------------|
    | attn_0    | num_heads        | 2     | auto   | H=128/head_dim=64|
    | attn_0    | head_dim         | 64    | user   | provided        |
    | mlp_0     | intermediate_size| 512   | auto   | 4*H             |

  Click на row → emit `node.highlight(brick)` через store → FlowCanvas
  добавляет selected=true на соответствующий node на 2 сек.

  Filter by brick / source / param. Source 'auto' выделен синим бейджем.

4.3 vbgui/src/components/sidebar/AblationsTab.tsx (НОВЫЙ, bb0.11)

  Axis selector (radio): activation / optimizer / norm / schedule
  Variants multi-select (depends on axis): 2-4 чекбокса с tooltip.
  Num steps slider (10-100, default 20).
  [Run ablation] кнопка → progress bar → results table:
    | Variant   | Final loss | Δ vs baseline | Time   | Status |
    |-----------|-----------|---------------|--------|--------|
    | swiglu ★  | 5.21      | (baseline)    | 14.7ms | ok     |
    | gelu      | 5.34      | +2.5%         | 12.1ms | ok     |
    | relu2     | 5.51      | +5.8%         | 11.4ms | ok     |

  Per row: mini-chart loss curve (10×40 px SVG).
  Ranked by final_loss ascending; baseline marked.

4.4 vbgui/src/components/BrickContextPanel.tsx (НОВЫЙ, bb0.5, bb0.6)

  Открывается при click на node в FlowCanvas (overlay справа от canvas).
  Поля per brick kind (зависят от brick category):

  Для attention/gated_attention/...:
    - pre_norm dropdown (rmsnorm/layernorm/none) + info icon
    - post_norm dropdown
    - num_heads (locked если auto-inferred)
    - head_dim
    - ...

  Для mlp/gated_mlp:
    - activation dropdown (gelu/relu/relu2/sqrelu/silu/mish/swiglu/
                            geglu/reglu/xielu)
    - intermediate_size
    - pre_norm / post_norm

  Для moe:
    - activation (тот же list но default swiglu)
    - num_experts / top_k
    - pre_norm / post_norm

  [Apply] кнопка → dispatch brick.update action.

4.5 vbgui/src/components/Tooltip.tsx (НОВЫЙ, bb0.10)

  Generic hover tooltip с lazy catalog.explain fetch.
  Usage:
    <OptionWithTooltip
      value="lion"
      label="Lion"
      category="optimizer"
      onSelect={...}
    />

  Hover 250ms → fetch + display summary + when_to_use.
  Click ⓘ → opens explanation modal с full entry + paper link.

  Cache: per-session memo по (category, name).

4.6 vbgui/src/components/ExplainModal.tsx (НОВЫЙ, bb0.10)

  Полная карточка ExplainEntry:
    - Title + category badge
    - Summary
    - When to use / when to avoid (две колонки)
    - Recommended params (key-value table)
    - Gotchas (warning list, ⚠ icon)
    - Paper reference + link
    - [Apply recommended params] кнопка (для optimizer/schedule)

---
5. Tooltip catalogue — recommended params lookup table

Полный CATALOG растёт во время разработки. Минимум для bb0.10 launch:

5.1 Optimizers (7)
  adamw, muon, muon_adamw_hybrid, lion, lion8bit, adam8bit, sgd

5.2 Activations (10)
  gelu, relu, relu2, sqrelu, silu, mish, swiglu, geglu, reglu, xielu

5.3 Norms (3)
  rmsnorm, layernorm, none

5.4 Schedules (6)
  constant, cosine, linear_warmup, wsd, inv_sqrt, polynomial

5.5 Losses (5)
  cross_entropy, mtp_weighted, ifim_shaped, mhc_attn_bias, custom

5.6 Rewriters (3+)
  mtp, ifim, mhc

5.7 Bricks (26)
  attention, gated_attention, mla, mla_absorb, mistral4_mla,
  dsv4_attention, gqa_sliding, cca_attention, gemma4_drafter,
  nemotron_h_mtp, gdn, kda, bailing_linear, mamba3, moe, bailing_moe,
  nsa, lightning_indexer, csa_hca, engram, mlp, mlstm, abs_pos_embed,
  per_layer_embed, gated_mlp (новый), squared_relu (kernel-only)

Итого: ~60 entries. Минимум 10-line summary per entry; реальный текст
~50-100 слов per entry.

---
6. Validation rules (новые)

6.1 Optim validation (build_spec)
  - LION/LION_8BIT lr > 5e-4 → WARNING ("recommended ≤5e-4")
  - LION требует betas
  - ADAM_8BIT eps must be set (1e-8 default)
  - MUON+gated_mlp.gate_proj в одной группе → WARNING (gate often
    benefits from AdamW)

6.2 Schedule validation
  - kind ∈ (cosine, wsd, polynomial) → total_steps required
  - wsd → decay_steps required, total_steps >= warmup+decay
  - warmup_steps > 0 и schedule kind="constant" → WARNING ("constant
    ignores warmup — use linear_warmup_then_constant")

6.3 Activation × Brick validation
  - Gated activation (swiglu/geglu/reglu/xielu) requires brick of type
    gated_mlp or moe (which has gate projection)
  - Dense activation (gelu/relu/relu2/sqrelu/silu/mish) compatible с
    любым brick

6.4 Norm validation
  - pre_norm="none" AND post_norm="none" → ERROR
  - parallel-block (attention || mlp): both bricks pre_norm != "none"
  - LayerNorm + RMSNorm mix в одном блоке → WARNING

6.5 Roundtrip validation
  - tokenizer.byte_roundtrip="none" AND any non-ASCII в text → WARNING

---
7. Public API surface

  # New RPC methods (delta over v1's 10 methods):

  catalog.explain(category, name) → ExplainEntry
  catalog.list_options(category) → [{name, summary, paper_ref}]
  ablation.run(base_spec, ablation_axis, variants, num_steps) →
    AblationRunResult
  suggest_optim_groups(graph, optim_kind, hidden_size) →
    SuggestOptimGroupsResult
  data.roundtrip_check(parquet_path, tokenizer_source, max_rows) →
    RoundtripCheckResult
  architectures.list_presets() → ListArchitecturesResult

  # New ScheduleSpec accepted by:
  optim_spec.OptimSpec.groups[].lr  (теперь float | ScheduleSpec)

  # Updated VerifyResult fields:
  resolved.inference_log: list[InferenceEntry]   (для DimensionsTab)

---
8. Что вне scope (явно)

  - Pyodide/JupyterLite — sidebar tabs работают только в Jupyter
    notebook anywidget mode (или в standalone FastAPI dev mode).
    Pyodide path откладывается до Phase 3.
  - Distributed ablations (multi-device parallel variants) — все
    варианты последовательно на single device.
  - Activation forward parity vs reference (mlx-lm baselines) — Phase 3.
  - Real-time ablation during training (live loss curves streaming) —
    Phase 3, требует WS push.
  - Save/load spec as JSON — отдельный тикет (E7-X future).
  - Undo/redo — отдельный тикет.
  - Rewriter param editing — отдельный тикет.

---
9. Связь с другими спеками

  - VBSpec / MBSpec / PSpec / Auto-Fusion: используются как backend
    через JSON-RPC; не модифицируем (кроме MBSpec optim_spec.py:
    добавление Lion в enum + новый ScheduleSpec).
  - ContractProbe.md: probe.run остаётся, новый ablation.run живёт
    рядом — общий стиль params/result.
  - VisualBuilderPlan.md / VisualBuilderSpec.md: расширяем UI sidebar
    (новые tabs DimensionsTab / AblationsTab + BrickContextPanel),
    остальные components (FlowCanvas / Palette / TopBar / BottomStrip /
    RunResultModal) не трогаем.
  - E2EMatrix.md (v1, эпик cppmega-mlx-pa3 закрыт): mini-spec
    (H=128, depth=2, S=64) переиспользуется. E2E матрица v1 (912 + 192
    cells) НЕ перезапускается — этот эпик добавляет новые матрицы (54
    activation cells, 24 norm cells, ablation runs).
