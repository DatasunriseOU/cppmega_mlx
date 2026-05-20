План: ModelBuildSpec Layer (Loss + Optim + Graph Rewrites поверх Bricks)

Цель: расширить IR за пределы forward-цепочки, чтобы MTP / IFIM / MHC
можно было собрать визуально. Сейчас `BrickGraph` описывает только
forward; loss и optim живут руками в `scripts/m04_train_step.py`. Для
MTP это смертельно — он требует переписать сам forward (K параллельных
голов) ПЛЮС multi-loss с per-head весами ПЛЮС маршрутизацию градиентов.

Лежит между BrickGraph и executable runtime; потребляет всё из
`cppmega_v4.spec.*` (ShapeContract, ResolvedBrickGraph, MemoryReport).

---
1. Что уже есть (карта существующего)

Forward IR (cppmega_v4/fusion/brick_graph.py)
  BrickGraph (nodes + edges) — топология forward. Знает kind, params,
  module. Не знает loss, optim, rewrites.

Spec layer (cppmega_v4/spec/, шипнуто на main, 167 тестов)
  - ShapeContract / ResolvedBrickGraph / MemoryReport
  - verify_and_estimate(graph) → одношаговая проверка для GUI
  - adapters (6 правил с авто-вставкой)

Auto-Fusion (cppmega_v4/fusion/, шипнуто, 175 тестов)
  FusionRegionPlan + auto_compile_region + 12 architecture presets.

Тренировочный код (scripts/m04_train_step.py + recipes/)
  Hardcoded: один CE loss, AdamW, нет K-head, нет per-head βi/λi.
  Если пользователь захочет MTP K=2 — надо ПИСАТЬ КОД, а не натянуть.

Brick `nemotron_h_mtp` (есть в BLOCK_BUILDERS)
  Sort-of MTP-блок, но не настоящий K-head rewrite — просто
  re-export из mlx-lm PR #1161. Loss-сторону не трогает.

---
2. Что НЕ хватает (gap-analysis)

┌─────────────────────────────────────────┬──────────────────────────────────┐
│             Что нужно GUI                │             Сейчас               │
├─────────────────────────────────────────┼──────────────────────────────────┤
│ Декларировать loss (CE / MTP / IFIM)    │ ✗ (hardcoded в scripts/)         │
├─────────────────────────────────────────┼──────────────────────────────────┤
│ Декларировать optimizer + hyperparams   │ ✗ (hardcoded — AdamW lr=3e-4)    │
├─────────────────────────────────────────┼──────────────────────────────────┤
│ Описать модель целиком (forward+L+O)    │ ✗ (только BrickGraph)            │
├─────────────────────────────────────────┼──────────────────────────────────┤
│ Применить MTP rewrite к графу           │ ✗ (одна голова в forward)        │
├─────────────────────────────────────────┼──────────────────────────────────┤
│ Применить IFIM (inverse FIM) shaping    │ ✗                                │
├─────────────────────────────────────────┼──────────────────────────────────┤
│ Применить MHC (multi-head copy) attn    │ ✗                                │
├─────────────────────────────────────────┼──────────────────────────────────┤
│ Per-head weights βi / λi для multi-loss │ ✗                                │
├─────────────────────────────────────────┼──────────────────────────────────┤
│ Routing градиентов (head_i -> backbone) │ ✗                                │
├─────────────────────────────────────────┼──────────────────────────────────┤
│ Проверить coherence loss vs graph       │ ✗                                │
└─────────────────────────────────────────┴──────────────────────────────────┘

Конкретный gap: на vapor MTP K=2 (см. M0.5 bead `cppmega-mlx-t8f.5`) — у
нас нет ни LossSpec, ни Rewriter — только мечта о том, что m04_train_step
"должен это делать". Этот roadmap закрывает gap визуально-описуемым
ModelBuildSpec.

---
3. Архитектура (новый пакет cppmega_v4/build/)

3.1 cppmega_v4/build/loss_spec.py — декларация loss

  class LossKind(str, Enum):
      CROSS_ENTROPY = "cross_entropy"
      MTP_WEIGHTED  = "mtp_weighted"      # Σ βi * CE(head_i, shifted_label_i)
      IFIM_SHAPED   = "ifim_shaped"        # CE + λ * IFIM penalty
      MHC_ATTN_BIAS = "mhc_attn_bias"      # CE + multi-head copy auxiliary
      CUSTOM        = "custom"             # caller supplies callable

  @dataclass(frozen=True)
  class LossSpec:
      kind: LossKind
      params: Mapping[str, float]   # {"k": 2, "beta_0": 1.0, "beta_1": 0.6, ...}
      head_outputs: tuple[str, ...] # names of brick outputs that feed loss
                                    # (must reference nodes that exist in graph
                                    #  after rewrites)
      label_source: str             # "next_token" | "next_k_tokens" | "doc_ids"
      reduction: str = "mean"

  # Built-ins:
  def cross_entropy_loss(head_output_name="logits") -> LossSpec: ...
  def mtp_weighted_loss(k=2, beta=(1.0, 0.6), lambda_=0.3) -> LossSpec: ...
  def ifim_shaped_loss(lambda_fim=0.1, head_output_name="logits") -> LossSpec: ...

3.2 cppmega_v4/build/optim_spec.py — декларация optimizer

  class OptimKind(str, Enum):
      ADAMW = "adamw"
      MUON  = "muon"
      MUON_ADAMW_HYBRID = "muon_adamw_hybrid"  # 2D params on Muon, rest AdamW
      SGD   = "sgd"

  @dataclass(frozen=True)
  class ParamGroup:
      """One param group. The matcher selects parameters; the hyperparams
      apply to the matched subset (per-group lr / wd / betas)."""
      matcher: str        # "all" | "moe_experts" | "embeddings" | regex
      lr: float
      weight_decay: float = 0.01
      betas: tuple[float, float] | None = None     # AdamW-only
      ns_steps: int | None = None                  # Muon-only

  @dataclass(frozen=True)
  class OptimSpec:
      kind: OptimKind
      groups: tuple[ParamGroup, ...]
      gradient_clip_norm: float | None = 1.0
      mixed_precision: bool = True

  # Built-ins:
  def adamw(lr=3e-4, wd=0.01, betas=(0.9, 0.95)) -> OptimSpec: ...
  def muon(lr=1e-2, ns_steps=5) -> OptimSpec: ...
  def muon_adamw_hybrid(muon_lr=1e-2, adam_lr=3e-4) -> OptimSpec: ...

3.3 cppmega_v4/build/model_build_spec.py — composer

  @dataclass(frozen=True)
  class ModelBuildSpec:
      graph:  BrickGraph
      loss:   LossSpec
      optim:  OptimSpec
      rewrites: tuple["Rewriter", ...] = ()   # applied in order
      dim_env: Mapping[str, int] = field(default_factory=dict)

      def apply_rewrites(self) -> "ModelBuildSpec":
          """Run every Rewriter; return new spec (frozen, immutable)."""
          ...

  def verify_build_spec(spec) -> BuildDiagnostics:
      """Check coherence:
         - loss.head_outputs ⊆ rewritten_graph.nodes
         - optim.groups[*].matcher matches at least one param
         - rewrites apply cleanly (no rewrite cycles, no name conflicts)
         - shape contracts still valid after rewrites
      """

3.4 cppmega_v4/build/rewriters.py — graph-rewrite слой

  class Rewriter(Protocol):
      """A pure graph transformation. Takes a ModelBuildSpec, returns a
      new ModelBuildSpec. Must not mutate inputs."""
      def __call__(self, spec: ModelBuildSpec) -> ModelBuildSpec: ...

  Built-ins:
    MTPRewriter(k=2, share_backbone=True)
      - Find the head brick (usually `logits` projection)
      - Materialise K copies, each predicting shifted labels [+0, +1, ..., +k-1]
      - Adjust LossSpec: convert single-output CE into mtp_weighted
        (if loss is CE; otherwise raise)
      - Update graph: K new head nodes; backbone shared (share_backbone=True)
      - Update OptimSpec: optional separate param group for new heads

    IFIMRewriter(lambda_fim=0.1)
      - Wrap the final head's output in an IFIM auxiliary loss node
      - Adds a `ifim_aux` virtual brick that consumes logits + computes
        the Fisher-information penalty
      - LossSpec rewritten to ifim_shaped

    MHCRewriter(num_copies=2)
      - For each attention brick, materialise N copies sharing weights
      - Add a "multi-head copy" auxiliary loss

3.5 cppmega_v4/build/api.py — public

  def build_model(spec: ModelBuildSpec) -> BuiltModel:
      """Apply rewrites, verify, materialise nn.Module + loss callable +
      optimizer instance."""

  @dataclass(frozen=True)
  class BuiltModel:
      module: nn.Module             # post-rewrite forward
      loss_fn: Callable[..., mx.array]
      optimizer: object             # MLX-native optim instance
      param_groups: tuple[tuple[str, ...], ...]  # group_idx -> param names
      spec_applied: ModelBuildSpec  # post-rewrite snapshot for telemetry

  def verify_build_spec(spec) -> BuildDiagnostics:
      """Shape-coherence + loss-head-output coverage + optim-matcher coverage."""

---
4. Какие rewrites нужны для текущих presets

┌─────────────────────────┬───────────────────────────────────────────────┐
│         Preset          │             Recommended rewrites              │
├─────────────────────────┼───────────────────────────────────────────────┤
│ qwen3_next              │ MTPRewriter(k=2, beta=(1.0,0.6))              │
│ kimi_linear / kimi_k2   │ MTPRewriter(k=2)                              │
│ deepseek_v3 / v4_flash  │ MTPRewriter(k=2) — DeepSeek уже использует     │
│ gemma4                  │ MTPRewriter(k=3) — Gemma-4 drafter            │
│ mistral4                │ — (нет MTP)                                   │
│ ling26                  │ MTPRewriter(k=2) + IFIMRewriter(λ=0.05)       │
│ longcat                 │ —                                              │
│ nemotron3               │ замена `nemotron_h_mtp` brick на MTPRewriter  │
│ zaya1                   │ MTPRewriter(k=2) + MHCRewriter(num_copies=2)  │
│ arcee_trinity           │ MTPRewriter(k=2)                               │
└─────────────────────────┴───────────────────────────────────────────────┘

---
5. Что GUI получает

  - В правой панели: «Loss» dropdown + per-head βi sliders + checkbox для
    IFIM / MHC. Сразу видно когда βi не складываются в 1.0 (warning).
  - В правой панели: «Optimizer» dropdown (AdamW / Muon / Hybrid) + lr
    slider; per-group панель когда у юзера несколько param-groups (MoE
    experts vs main).
  - Над графом: цепочка «Rewrites applied» (chips) — кликабельные
    MTPRewriter(k=2), IFIMRewriter(λ=0.05). Юзер может drag-drop порядок,
    выкинуть, добавить.
  - После применения rewrites — preview графа с подсвеченными новыми
    нодами (head_1, head_2, ifim_aux) в другом цвете.
  - Memory report пересчитывается на post-rewrite graph — юзер видит
    «MTP K=2 adds 1.8 GB» прежде чем нажмёт Train.

---
6. План реализации поэтапно

Этап A — LossSpec + OptimSpec (1 заход)
  - cppmega_v4/build/loss_spec.py — LossKind, LossSpec, built-ins
  - cppmega_v4/build/optim_spec.py — OptimKind, ParamGroup, OptimSpec, built-ins
  - Чистый data-layer, никакой MLX runtime; всё валидируется в __post_init__.
  - Тесты: каждый built-in возвращает валидный spec; rejection-тесты
    на негативные lr, плохой kind, пустые head_outputs.

Этап B — ModelBuildSpec + verify_build_spec (1 заход)
  - cppmega_v4/build/model_build_spec.py — composer (immutable dataclass)
  - apply_rewrites — chain order semantics
  - BuildDiagnostics + verify_build_spec (shape-coherence через
    cppmega_v4.spec, loss head_outputs ⊆ graph.names, optim matcher
    coverage)
  - Тесты: верифицирует Qwen3-Next + AdamW + CE — clean; искусственный
    bad head_output — ERROR; пустой matcher — WARNING.

Этап C — Rewriter protocol + MTPRewriter (1 заход)
  - cppmega_v4/build/rewriters.py — Rewriter Protocol + MTPRewriter
  - MTPRewriter материализует K head копий, перепишет LossSpec
    CE → mtp_weighted, добавит head-only param group в OptimSpec
  - Тесты: K=1 — no-op; K=2 — 2 head nodes + mtp_weighted loss; K=3 +
    share_backbone=False — duplicate-backbone path; ошибка когда loss
    не CE (нельзя автоматически переписать)
  - System: применить к qwen3_next preset, verify — clean, memory rate
    растёт ровно на N(heads-1) * params(head).

Этап D — IFIMRewriter + MHCRewriter (1 заход)
  - cppmega_v4/build/rewriters.py — добавить два rewriter
  - IFIMRewriter — добавит aux node + λ-loss
  - MHCRewriter — на каждом attention брике сделает copies
  - Тесты: composition (MTPRewriter + IFIMRewriter), порядок
    применения, anti-cycle проверка.

Этап E — build_model + executable wiring + GUI integration test (1 заход)
  - cppmega_v4/build/api.py — build_model(spec) → BuiltModel
  - Wire BrickGraph nodes (с module) в nn.Module-композицию
  - Wire LossSpec → callable (CE / mtp_weighted / IFIM)
  - Wire OptimSpec → mlx.optimizers.AdamW / Muon / Hybrid
  - Perf-критерий: build_model для каждого из 12 presets < 200 ms
  - Тесты: построить qwen3_next + MTPRewriter(k=2) + AdamW → forward
    проходит, loss скаляр финитен, optimizer.update без crash, GIU
    workflow integration (build_spec → verify → build_model → step).

---
7. Бюджет и риски

Бюджет: 5 этапов × ~2 часа = ~10 часов чистого кодинга. A и B —
data-layer, последовательны. C/D — rewriter implementations, частично
параллелятся (MTP vs IFIM/MHC независимы). E — wiring + integration.

Главный риск: MTPRewriter должен корректно "найти head brick" в графе.
Сейчас наш `logits` head — это часть UnifiedSuperblockV4, не отдельный
brick. Mitigation: ввести опциональный marker `is_head=True` в BrickNode
или соглашение «head — последний node в графе». Документировать
жёстко.

Второй риск: LossSpec coherence vs Rewriter. Если юзер ставит CE,
потом MTPRewriter автоматически перепишет в mtp_weighted — это сюрприз.
Mitigation: verify_build_spec показывает что **итоговая** loss будет
mtp_weighted, ДО build_model.

Третий риск: composition rewrites в неправильном порядке. IFIMRewriter
после MTPRewriter работает на K голов или на одну? Mitigation:
Rewriter имеет required_precondition + provided_postcondition; verifier
выкинет ERROR если порядок неверен.
