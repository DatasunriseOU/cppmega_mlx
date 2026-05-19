  План: Auto-Fusion Layer над Bricks

  1. Что уже есть (карта существующего)

  Брики (cppmega_v4/models/unified_superblock_v4.py)
  
  15+ kinds в BLOCK_BUILDERS, каждый возвращает nn.Module. Каждый брик уже знает себя как self-contained вычисление.

  Path C fusion infrastructure (cppmega_mlx/runtime/path_c_fusion.py — 2523 строк)

  Готовое:
  - FusionKernelSurface — описывает один op (name/op_name/inputs/outputs/backend)
  - FusionNode / FusionEdge — граф producer→consumer
  - PathCFusionRegion — целый регион
  - PathCFusionRegionBuilder.add_kernel().add_kernels().build() — fluent API
  - compile_path_c_region(region, schedule_template, ...) — отдаёт CompiledPathCRegion
  - build_path_c_aot_autograd_region() — добавляет backward поверх forward графа
  
  Schedule descriptors (cppmega_mlx/runtime/path_c_fusion_schedules.py — 2038 строк)

  - PathCBrickScheduleDescriptor — рецепт codegen для одного op_name
  - PathCBrickScheduleDescriptorRegistry — карта op_name → descriptor
  - default_path_c_brick_schedule_descriptor_registry() — сейчас покрывает 5 ops: mamba3_mimo, residual_rmsnorm, m2rnn, attention_qkv_projection, sparse_mla_fp8_apply
  - make_path_c_descriptor_schedule_template(region) — берёт descriptors из registry, сшивает в один tir.PrimFunc template

  TileLang fusion engine (tilelang/engine/fusion.py — 1228 строк)

  - FusionRegion — region graph для TileLang
  - ScheduleTemplate — функция (region) → tir.PrimFunc
  - Compile в один metallib через tvm_ffi
  - ОТКАЗ принимать уже-лоwered MSL: требует pre-source IR

  DLPack

  Проверил: MLX поддерживает __dlpack__() и from_dlpack(), в т.ч. для Metal-resident тензоров. RFC ml-explore/mlx#2848 — наш PoC уже умеет zero-copy mlx ↔ TileLang/TVM Metal handoff.

  Triton frontend → TileLang
  
  poc.triton_frontend.from_ttir(...) — TTIR→TileLang IR с OP_TABLE walker + PtrAnalysis pre-pass. Полный путь Triton→TileLang→Metal работает.

  ---
  2. Что НЕ хватает (gap-analysis)
  
  ┌────────────────────────────────────────────────────────────────────┬────────────────────┬────────────────────────────┬───────────────┐
  │                          Brick (наши 15+)                          │   Path B (Metal)   │ Path C schedule descriptor │ Fusion-готов? │
  ├────────────────────────────────────────────────────────────────────┼────────────────────┼────────────────────────────┼───────────────┤
  │ gdn                                                                │ ✓                  │ ✗ (только mamba3)          │ нет           │
  ├────────────────────────────────────────────────────────────────────┼────────────────────┼────────────────────────────┼───────────────┤
  │ kda                                                                │ ✓                  │ ✗                          │ нет           │
  ├────────────────────────────────────────────────────────────────────┼────────────────────┼────────────────────────────┼───────────────┤
  │ gated_attention                                                    │ через mx.fast.SDPA │ ✗                          │ нет           │
  ├────────────────────────────────────────────────────────────────────┼────────────────────┼────────────────────────────┼───────────────┤
  │ mistral4_mla, dsv4_attention, bailing_*                            │ через mlx-lm SDPA  │ ✗                          │ нет           │
  ├────────────────────────────────────────────────────────────────────┼────────────────────┼────────────────────────────┼───────────────┤
  │ nsa, csa_hca, lightning_indexer                                    │ свой Metal         │ ✗                          │ нет           │
  ├────────────────────────────────────────────────────────────────────┼────────────────────┼────────────────────────────┼───────────────┤
  │ mla, mla_absorb                                                    │ mx.fast.SDPA       │ ✗                          │ нет           │
  ├────────────────────────────────────────────────────────────────────┼────────────────────┼────────────────────────────┼───────────────┤
  │ moe, bailing_moe, nemotron_h_mtp                                   │ свой               │ ✗                          │ нет           │
  ├────────────────────────────────────────────────────────────────────┼────────────────────┼────────────────────────────┼───────────────┤
  │ mamba3 (cppmega_mlx)                                               │ ✓                  │ ✓                          │ да            │
  ├────────────────────────────────────────────────────────────────────┼────────────────────┼────────────────────────────┼───────────────┤
  │ m2rnn                                                              │ ✓                  │ ✓                          │ да            │
  ├────────────────────────────────────────────────────────────────────┼────────────────────┼────────────────────────────┼───────────────┤
  │ attention_qkv_projection + sparse_mla_fp8_apply + residual_rmsnorm │ ✓                  │ ✓                          │ да            │
  └────────────────────────────────────────────────────────────────────┴────────────────────┴────────────────────────────┴───────────────┘

  Конкретный gap: 5 из 15+ bricks имеют Path C descriptor (только legacy v3 train block). Для 10 bricks нужно:
  1. Сгенерировать TIR-уровневый descriptor (или wrapper "fall through to Path B kernel")
  2. Зарегистрировать в default_path_c_brick_schedule_descriptor_registry 
  3. Решить какие пары bricks МОГУТ фузиться (compatibility matrix)

  ---
  3. Архитектура auto-fusion layer (новый модуль)
  
  Предлагаю cppmega_v4/fusion/ как новый пакет с этими слоями:

  3.1 cppmega_v4/fusion/brick_graph.py — введение graph IR над нашими блоками

  @dataclass
  class BrickNode:
      kind: str              # ключ из BLOCK_BUILDERS
      name: str              # уникальное имя в графе
      params: dict           # параметры конструктора
      module: nn.Module      # инстанс (для shape inference + fallback execution)

  @dataclass
  class BrickGraph:
      nodes: list[BrickNode]
      edges: list[tuple[str, str]]   # (producer, consumer) — линейная цепочка или MoE-stage граф

  def from_unified_superblock(block_specs: list[dict]) -> BrickGraph: ...
  def from_mlx_model(model: nn.Module) -> BrickGraph: ...   # walk nn.Module tree

  3.2 cppmega_v4/fusion/compatibility.py — fusion eligibility оракул

  @dataclass
  class FusionEligibility:
      can_fuse: bool
      reason: str
      backend: str           # "path_c" | "metal_inline" | "dlpack_handoff"

  def can_fuse_pair(a: BrickNode, b: BrickNode) -> FusionEligibility: ...

  Правила (table-driven):
  - linear-attn (gdn/kda) + rmsnorm + residual → можно (общий проход по T)
  - nonlinear M2RNN + что угодно → fusion невозможен ВНУТРИ recurrence (но fwd-RMSNorm можно вставить ДО)
  - mx.fast.SDPA-based bricks (gated_attention/mla/mistral4_mla/dsv4_attention) → fusion только на output side (gate*output, o_proj)  — потому что SDPA это монолитный Apple Metal kernel
  - MoE → fusion внутри expert FFN можно, между experts — нет (route disjoint)
  - TurboQuantKVCache → cross-cutting, плюс на attention бриках в decode
  
  3.3 cppmega_v4/fusion/auto_planner.py — auto-detection fusion regions

  def plan_fusion_regions(graph: BrickGraph) -> list[FusionRegionPlan]:
      """Greedy: walk graph, group adjacent fusable nodes into regions."""

  Стратегии:
  - Bottom-up greedy: начинаем с single-node regions, пробуем расширить если can_fuse_pair returns True для (region.tail, next_node)
  - Cost model: для каждой кандидатной region считаем estimated speedup vs separately-compiled bricks (на основе DLPack hand-off overhead, kernel launch overhead, register pressure)
  - Hard limits: max region size = 8 bricks (register pressure), max shared mem = 32KB Apple Metal
  
  3.4 cppmega_v4/fusion/descriptor_synthesizer.py — auto-gen schedule descriptors для отсутствующих 10 bricks

  Подход:
  - Для bricks с уже существующим Path B Metal kernel (gdn, kda, nsa, etc.): wrap as an opaque FusionKernelSurface который TileLang не пытается фузить дальше, а просто вызывает через DLPack handoff. Это "fallback descriptor" — даёт корректность без оптимизации
  - Для bricks based on mx.fast.scaled_dot_product_attention (gated_attention, mla*, mistral4_mla, etc.): сгенерировать descriptor с явным граничным DLPack handoff — SDPA внутри opaque, но gate-multiply и o_proj могут быть fused внутри TileLang region
  - Для linear-attn (gdn, kda, bailing_linear): попробовать сгенерировать честный TileLang scan template из FLA reference

  3.5 cppmega_v4/fusion/dlpack_bridge.py — zero-copy boundary

  def mlx_to_tilelang(arr: mx.array) -> "tvm.ffi.NDArray":
      """Wrap mx.array as TVM NDArray via DLPack (Metal-resident, no copy)."""
      return tvm.runtime.from_dlpack(arr.__dlpack__())

  def tilelang_to_mlx(nda: "tvm.ffi.NDArray") -> mx.array:
      return mx.from_dlpack(nda)
  Используется на каждой границе fusion region ↔ остальной MLX-граф.

  3.6 cppmega_v4/fusion/__init__.py — public API

  def auto_fuse_model(model: nn.Module) -> nn.Module:
      """Walk model, plan regions, compile each via path_c_fusion, return new model."""

  def auto_fuse_block_specs(specs: list[dict]) -> nn.Module:
      """Same but from BLOCK_BUILDERS-shaped JSON."""

  ---
  4. Как из наших bricks строится каждая модель из галереи

  ┌──────────────────────┬────────────────────────────────────────┬─────────────────────────────────────────────────────────────────────────────────────┐
  │        Модель        │                Pattern                 │                                     Наши bricks                                     │
  ├──────────────────────┼────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────┤
  │ Qwen3-Next/3.5/3.6   │ 3:1 GDN + Gated Attention + MoE        │ gdn×3 + gated_attention + moe повторить N раз                                       │
  ├──────────────────────┼────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────┤
  │ Kimi Linear 48B-A3B  │ 3:1 KDA + MLA                          │ kda×3 + mla + MoE expert                                                            │
  ├──────────────────────┼────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────┤
  │ Kimi K2/K2.5         │ 100% MLA + 384 experts                 │ mla_absorb + moe                                                                    │
  ├──────────────────────┼────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────┤
  │ DeepSeek V3          │ MLA + sparse MoE                       │ mla + moe                                                                           │
  ├──────────────────────┼────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────┤
  │ DeepSeek V4 Flash    │ hash-indexed sparse MLA                │ dsv4_attention + moe                                                                │
  ├──────────────────────┼────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────┤
  │ Gemma 4 26B-A4B      │ 5:1 sliding/global GQA + QK-norm + MoE │ новый brick: gqa_sliding (можем взять из mlx-lm), gated_attention (для global), moe │
  ├──────────────────────┼────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────┤
  │ Mistral Small 4 119B │ MLA + 128 sparse MoE                   │ mistral4_mla (с INT4 cache) + moe                                                   │
  ├──────────────────────┼────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────┤
  │ Ling 2.6             │ 7:1 Lightning + MLA + MoE              │ bailing_linear×7 + bailing_mla + bailing_moe                                        │
  ├──────────────────────┼────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────┤
  │ LongCat              │ MLA + RoPE+NoPE + Shortcut MoE         │ mla + bailing_moe + новый brick shortcut_route                                      │
  ├──────────────────────┼────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────┤
  │ Nemotron 3 Super     │ Mamba-2 + GQA + MoE                    │ mamba3 + attention + moe                                                            │
  ├──────────────────────┼────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────┤
  │ ZAYA1-8B             │ CCA + 4:1 GQA + top-1 MoE              │ новый brick cca_attention, gated_attention (4:1 GQA), moe (top-1)                   │
  ├──────────────────────┼────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────┤
  │ Arcee Trinity Large  │ 3:1 sliding/global gated GQA + MoE     │ gated_attention (sliding) + gated_attention (global) + moe                          │
  └──────────────────────┴────────────────────────────────────────┴─────────────────────────────────────────────────────────────────────────────────────┘

  Каждая запись — это JSON spec для UnifiedSuperblock (1 строка на architecture). Нужно добавить 2 новых bricks: gqa_sliding (для Gemma 4) и cca_attention (для ZAYA1) — оба легко вытащить из mlx-lm или написать тонкий wrapper.

  ---
  5. Что path C должен сам делать (auto-fusion внутри path C)
  
      # 1. Look up each node's descriptor in default_registry (or auto-synth)
      # 2. Detect pattern: linear scan? attn-with-proj? MoE+route?
      # 3. Pick matching pre-baked schedule_template OR synth a new one via
      #    make_path_c_descriptor_schedule_template
      # 4. compile_path_c_region(region, schedule_template=picked)

  Это уже почти есть в make_path_c_descriptor_schedule_template — он берёт descriptors из registry и сшивает template. Нужно дополнить:
  - pattern matcher что выбирает right template per region shape
  - fallback на "compile each node separately, glue via DLPack" если pattern не распознан

  ---
  6. План реализации поэтапно

  Этап A — Foundation (1 заход)

  - cppmega_v4/fusion/brick_graph.py — graph IR + walker
  - cppmega_v4/fusion/dlpack_bridge.py — zero-copy helpers
  - cppmega_v4/fusion/compatibility.py — table-driven eligibility
  - Тесты: walk a 5-brick UnifiedSuperblock spec, assert graph correct

  Этап B — Descriptor synthesis (1-2 захода)

  - cppmega_v4/fusion/descriptor_synthesizer.py — auto-gen "fallback descriptors" для каждого нашего brick (opaque kernel + DLPack handoff)
  - Расширить default_path_c_brick_schedule_descriptor_registry на наши 10 bricks
  - Тесты: каждый brick получает working descriptor, compile_path_c_region не падает

  Этап C — Auto-planner (1 заход)

  - cppmega_v4/fusion/auto_planner.py — greedy region grouping
  - auto_fuse_model(model) → model
  - Тесты: pass Qwen3-Next-shaped 4-block sequence, verify regions detected (3 GDN fused, 1 attn separate, или все 4 в одну если cost-model говорит OK)

  Этап D — Architecture presets (1 заход)

  - cppmega_v4/architectures/ — JSON specs для 12 моделей галереи
  - Loader: build_model_from_preset("qwen3.6_27b") → nn.Module
  - 2 новых bricks: gqa_sliding, cca_attention
  - Тесты: каждый preset инстанциируется без ошибок, тривиальный forward проходит

  Этап E — Fusion auto-detection within Path C (1 заход)

  - Добавить auto_compile(region) в path_c_fusion
  - Pattern matcher: "если region содержит scan+rmsnorm — used scan-with-rmsnorm template"
  - Fallback на per-node compile + DLPack glue
  - Тесты: compile fusion region из (kda, residual, rmsnorm) — выбирает правильный template

  ---
  7. Бюджет и риски

  Бюджет: 5 этапов × ~2 часа каждый = ~10 часов чистого кодинга. Параллелится плохо (каждый этап зависит от предыдущего), но Этапы A и B независимы — можно делать одновременно.

  Главный риск: Path C TileLang lowering для наших bricks может оказаться долгим — некоторые брики (например nsa_v4 sparse) имеют data-dependent control flow которое TileLang не умеет. Mitigation: "fallback descriptor" в Этапе B — оставляем такие bricks как opaque kernel + DLPack handoff, не пытаемся вфузить.

  Второй риск: DLPack overhead на каждой границе. Mitigation: cost model в Этапе C — если region < N bricks, fused лучше; иначе оставить раздельно.

  Третий риск: Cross-brick optimization (e.g. fuse adjacent rmsnorm+linear) требует SHAPE-AWARE matcher. Mitigation: начать с трёх hand-coded patterns (LinearAttn+RMSNorm+Residual, SDPA+Gate+OProj, MoE+Route+Combine), расширять по мере надобности.