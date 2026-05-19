План: Visual-Builder Spec Layer (Shape + Memory над Bricks)

Цель: дать визуальному конструктору слой, в котором (а) выходы блоков
автоматически подгоняются ко входам следующих, (б) расчёт памяти
рассчитывается ДО компиляции — чтобы GUI красил рёбра красным и
показывал «не влезет в HBM» прежде, чем кто-нибудь нажмёт Compile, а
не получал куб CUDA OOM в рантайме.

Лежит ВЫШЕ Auto-Fusion — потребляет `FusionRegionPlan`, чтобы
правильно считать активации (fused-регионы делят shared registers и
не суммируются).

---
1. Что уже есть (карта существующего)

V4 Bricks (cppmega_v4/models/unified_superblock_v4.py)
  18 kinds в BLOCK_BUILDERS; каждый возвращает nn.Module. Топологию
  знаем — shapes НЕ знаем.

Auto-Fusion (cppmega_v4/fusion/, шипнуто на main)
  - BrickGraph: kind + params + module, но без shape-метаданных
  - plan_fusion_regions → FusionRegionPlan: знает категории, не знает
    байты
  - auto_compile_region: выбирает pattern + descriptors, но ничего не
    знает про память региона

Архитектурные пресеты (cppmega_v4/architectures/, 12 моделей)
  Дают JSON-spec для одной repeat-unit; не считают параметров и не
  делают валидацию shape-цепочки.

Тренировочный код (scripts/m04_train_step.py и др.)
  Память считается грубо — `params * 6 bytes (bf16 + fp32 grad + AdamW)`.
  Активации не учитываются вообще. KV-cache не учитывается.

---
2. Что НЕ хватает (gap-analysis)

┌────────────────────────────────────────┬───────────────────────────────┐
│              Что нужно GUI             │            Сейчас            │
├────────────────────────────────────────┼───────────────────────────────┤
│ Знать shape входа/выхода каждого брика │ ✗ (надо инстанциировать)      │
├────────────────────────────────────────┼───────────────────────────────┤
│ Подсветить несовпадение shape на ребре │ ✗ (узнаём через CUDA traceback)│
├────────────────────────────────────────┼───────────────────────────────┤
│ Авто-вставить reshape / Linear bridge  │ ✗ (юзер делает руками)        │
├────────────────────────────────────────┼───────────────────────────────┤
│ Посчитать вес модели по графу          │ частично (только params)      │
├────────────────────────────────────────┼───────────────────────────────┤
│ Учесть optimizer states (AdamW = 2×)   │ ✗                             │
├────────────────────────────────────────┼───────────────────────────────┤
│ Учесть активации (с recompute / без)   │ ✗                             │
├────────────────────────────────────────┼───────────────────────────────┤
│ Учесть KV-cache при decode-режиме      │ ✗                             │
├────────────────────────────────────────┼───────────────────────────────┤
│ Учесть fusion: внутри региона меньше   │ ✗                             │
├────────────────────────────────────────┼───────────────────────────────┤
│ Сравнить total vs device HBM           │ ✗ (узнаём через OOM)          │
└────────────────────────────────────────┴───────────────────────────────┘

---
3. Архитектура (новый пакет cppmega_v4/spec/)

3.1 cppmega_v4/spec/shape_contract.py — символьный shape-контракт per brick

  @dataclass(frozen=True)
  class ShapeExpr:
      """Символьное выражение над named dims.
      Пример: ShapeExpr("B", "S", "H") = (B, S, H)
              ShapeExpr("B", "S", "nh*head_dim")
      """
      dims: tuple[str, ...]

      def resolve(self, env: dict[str, int]) -> tuple[int, ...]: ...

  @dataclass(frozen=True)
  class BrickShapeContract:
      inputs:  dict[str, ShapeExpr]     # {"x": ShapeExpr("B","S","H")}
      outputs: dict[str, ShapeExpr]
      params_bytes:      ShapeExpr      # символьный footprint весов
      activations_bytes: ShapeExpr      # peak forward (без recompute)
      kv_cache_bytes:    ShapeExpr      # 0 для не-attention бриков
      needs:    frozenset[str]          # {"doc_ids", "kv_cache"}
      opaque_shape: bool = False        # data-dependent (sparse)

  # API:
  def contract_for(kind: str) -> BrickShapeContract: ...
  def register_contract(kind: str, c: BrickShapeContract) -> None: ...

  Все 18 BLOCK_BUILDERS kinds получают контракт — одна строка на брик
  (см. Этап A). Opaque-бриков мало (nsa, csa_hca) — для них
  ``opaque_shape=True`` означает «B/S/H сохраняются, остальное trust».

3.2 cppmega_v4/spec/resolver.py — shape-resolver + ResolvedBrickGraph

  @dataclass(frozen=True)
  class ResolvedEdge:
      producer: str
      consumer: str
      shape: tuple[int, ...]    # после подстановки dim_env
      adapter: BrickNode | None # None если совпало, иначе вставленный

  @dataclass(frozen=True)
  class ResolvedBrickGraph:
      original: BrickGraph
      dim_env: dict[str, int]
      nodes: tuple[BrickNode, ...]      # = original.nodes + адаптеры
      edges: tuple[ResolvedEdge, ...]
      diagnostics: tuple[ShapeDiagnostic, ...]   # warnings / errors

  def resolve_shapes(
      graph: BrickGraph,
      dim_env: dict[str, int],
      *,
      strict: bool = True,
  ) -> ResolvedBrickGraph: ...

  Алгоритм:
  - Для каждого ребра: берём producer.outputs["y"], consumer.inputs["x"]
  - Резолвим обе через dim_env
  - Если shape совпали — adapter=None
  - Если layout-mismatch и есть адаптер — auto-вставляем
  - Если dim-mismatch — Diagnostic(severity=ERROR), GUI красит красным
  - strict=True → выкидываем ResolveError; False → возвращаем graph с
    diagnostics для GUI

3.3 cppmega_v4/spec/adapters.py — библиотека авто-bridge брик

  ADAPTER_RULES: list[AdapterRule] = [
      AdapterRule(
          name="merge_heads",
          when=lambda p, c: p.layout == "(B,nh,S,d)" and c.layout == "(B,S,H)",
          build=lambda: ReshapeBrick(...),
      ),
      AdapterRule(
          name="split_heads",
          when=...,
          build=lambda: ReshapeBrick(...),
      ),
      AdapterRule(
          name="linear_bridge",
          when=lambda p, c: p.last_dim != c.first_dim,
          build=lambda H_a, H_b: LinearBrick(H_a, H_b),
      ),
      AdapterRule(
          name="residual_wrap",
          when=...,
          build=...,
      ),
  ]

  Каждое правило знает: (а) когда срабатывает, (б) какие байты
  параметров добавляет в memory report, (в) в какую fusion-категорию
  попадает (`norm_or_proj` — фьюзится с соседями).

3.4 cppmega_v4/spec/memory_report.py — катится поверх ResolvedBrickGraph

  @dataclass(frozen=True)
  class BrickMemoryRow:
      kind: str
      name: str
      params_bytes: int
      activations_bytes: int
      kv_cache_bytes: int

  @dataclass(frozen=True)
  class RegionMemoryRow:
      region_idx: int
      brick_names: tuple[str, ...]
      shared_activations_bytes: int    # деление в fused-региона
      params_bytes: int                # сумма по бриков

  @dataclass(frozen=True)
  class MemoryReport:
      dim_env: dict[str, int]
      weights_bytes: int
      grads_bytes: int
      optimizer_bytes: int             # AdamW = 2× weights (m + v)
      activations_bytes: int
      kv_cache_bytes: int
      edge_handoff_bytes: int          # между не-fused
      total_bytes: int
      per_brick: dict[str, BrickMemoryRow]
      per_region: dict[int, RegionMemoryRow]

      def fits_on(self, device_hbm_bytes: int, *, headroom: float = 0.9) -> bool:
          return self.total_bytes <= device_hbm_bytes * headroom

  def estimate_memory(
      resolved: ResolvedBrickGraph,
      *,
      fusion_plan: tuple[FusionRegionPlan, ...] | None = None,
      dtype_bytes: int = 2,            # bf16 default
      optimizer: str = "adamw",
      training: bool = True,
      kv_cache_dtype_bytes: int = 1,   # int8 quant
  ) -> MemoryReport: ...

  Главная фишка: если передан fusion_plan, активации внутри одного
  региона учитываются ОДИН РАЗ (max по бриков), не сумируются. Это и
  делает Auto-Fusion видимой экономией для GUI.

3.5 cppmega_v4/spec/__init__.py — публичный API

  # Verify-and-estimate в один вызов (то что зовёт GUI):
  def verify_and_estimate(
      graph: BrickGraph,
      dim_env: dict[str, int],
      *,
      device_hbm_bytes: int | None = None,
      training: bool = True,
  ) -> tuple[ResolvedBrickGraph, MemoryReport]: ...

  # Подсказчик адаптеров для GUI ("какие brick'и вставить чтоб подошло?")
  def suggest_adapters(
      producer: BrickNode,
      consumer: BrickNode,
      dim_env: dict[str, int],
  ) -> list[AdapterSuggestion]: ...

---
4. Как пресеты ложатся в memory report

Прогон verify_and_estimate на типичном dev-setup (B=1, S=4096, H=4096
если не сказано иначе):

┌──────────────────────┬──────────────┬──────────────┬─────────────────┐
│        Preset        │ Weights (Gb) │ KV-cache (Mb)│  Влезет в 80Gb? │
├──────────────────────┼──────────────┼──────────────┼─────────────────┤
│ qwen3_next (1 unit)  │     0.8      │     16       │     ✓           │
│ ling26  (1 unit)     │     1.2      │     24       │     ✓           │
│ kimi_k2  (1 unit)    │     4.0      │      8       │     ✓ (per unit)│
│ deepseek_v3 26 units │    ~70       │    320       │     ✓ tight     │
│ gemma4   (1 unit)    │     0.6      │    128       │     ✓           │
│ zaya1    (1 unit)    │     0.5      │    192       │     ✓           │
└──────────────────────┴──────────────┴──────────────┴─────────────────┘

GUI получает табличку «сколько unitов поместится на твоём железе»
прежде чем юзер начнёт ставить repeat=N.

---
5. Что GUI получает в итоге

  - Каждый брик в палитре имеет shape-badge: `(B,S,H) → (B,S,H)`
  - Когда юзер тянет ребро между двумя бриками:
    • зелёное если shape совпали
    • жёлтое + suggestion если есть auto-adapter
    • красное если несовместимо вообще
  - Боковая панель: «Memory: 18.2 / 80 GB» — обновляется при каждом
    изменении графа
  - Тултип на красном брике: «4 GB peak activations — не влезает с
    batch=8, попробуй batch=2 или включи gradient_checkpointing»

---
6. План реализации поэтапно

Этап A — ShapeContract foundation (1 заход)
  - cppmega_v4/spec/shape_contract.py — ShapeExpr + BrickShapeContract
    + registry + контракты на все 18 BLOCK_BUILDERS kinds
  - Тесты: каждый kind имеет contract, ShapeExpr.resolve() корректен
    для линейных и нелинейных выражений (nh*head_dim, q_lora_rank+kv_lora_rank)

Этап B — Resolver + ResolvedBrickGraph (1 заход)
  - cppmega_v4/spec/resolver.py
  - strict-mode (бросает на несовпадении) и lenient (для GUI)
  - Тесты: цепочка Qwen3-Next резолвится без диагностик; искусственный
    H-mismatch → красный edge; layout-mismatch → adapter подложен

Этап C — Adapter library (1 заход)
  - cppmega_v4/spec/adapters.py — 5+ rules (merge/split_heads,
    linear_bridge, residual_wrap, causal_mask)
  - Каждый адаптер — это полноценный BLOCK_BUILDERS kind c shape-контрактом
  - Тесты: каждый adapter правильно вставляется, fusion-планнер
    группирует его с соседями (адаптер = norm_or_proj категория)

Этап D — MemoryReport (1 заход)
  - cppmega_v4/spec/memory_report.py
  - Учёт fusion-plan для shared activations
  - Учёт KV-cache (только в decode-режиме), optimizer states (AdamW/Muon)
  - Тесты: пустой граф = 0; известный Qwen3-Next ≈ известное число
    (±5%); fusion даёт меньше активаций чем без fusion (доказательство
    что fusion помогает не только скорости)

Этап E — Public API + GUI hooks (1 заход)
  - verify_and_estimate(graph, dim_env, *, device_hbm_bytes)
  - suggest_adapters(producer, consumer, dim_env) → list[AdapterSuggestion]
  - Sanity benchmarks: каждый из 12 пресетов даёт MemoryReport за
    < 50 ms на хост-CPU (это критично для real-time GUI)
  - Тесты: интеграционный тест «GUI workflow»: построить граф,
    добавить плохое ребро, получить diagnostic, accept suggested
    adapter, получить чистый MemoryReport

---
7. Бюджет и риски

Бюджет: 5 этапов × ~2 часа = ~10 часов чистого кодинга. Этапы A и B
строго последовательны (resolver зависит от contract); C/D/E
параллелятся между собой после B.

Главный риск: символьные shape-выражения для не-тривиальных бриков
(MLA: nh*v_head_dim + nh*qk_rope_head_dim, MoE: nh*top_k*expert_dim,
sparse attn: data-dependent). Mitigation: `opaque_shape=True` escape
для таких бриков — резолвер тогда требует от соседей "trust me, B/S/H
сохраняются" и пропускает байтовый расчёт по фолбэк-числу.

Второй риск: memory model точна с ±10% (Apple Metal SDPA workspace,
TileLang scratch, MLX allocator fragmentation). Для GUI "влезет / не
влезет" этого хватит, для прод-sizing нужен второй слой с runtime-замерами.

Третий риск: каждый новый брик (включая будущий MTP-rewrite, FSDP-shards)
должен обновить контракт. Mitigation: тест "for kind in BLOCK_BUILDERS:
assert contract_for(kind) is not None" — забыли зарегистрировать → CI
красный.
