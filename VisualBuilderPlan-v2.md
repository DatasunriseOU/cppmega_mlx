План: Visual-Builder v2 — Manual Assembly + Auto-Tune + Roundtrip + Tooltips

Roadmap для эпика cppmega-mlx-bb0 (E2E Coverage Matrix v2). Закрывает
дыры, найденные при post-mortem аудите эпика cppmega-mlx-pa3:

  - Pa3 проверил только preset launcher path (готовые архитектуры),
    но не ручную сборку с нуля.
  - Activation/norm/schedule selectors отсутствуют как в backend, так и
    в UI.
  - Lion и quant variants есть в backend но не зарегистрированы в
    OptimKind enum (UI их не видит).
  - Нет tooltip system для объяснения вариантов; пользователь угадывает
    что выбрать.
  - Нет ablation runner для side-by-side сравнения вариантов.
  - 5 PRESETS из 62 пропущены в UI (хардкод 57 в App.tsx).

См. **VisualBuilderSpec-v2.md** для технической спецификации API и
data structures. Этот файл — pure roadmap (stages, tests, criteria,
budget).

---
0. TL;DR

15 stages (E7-1 .. E7-14, плюс эпик-обёртка bb0). Бюджет ~25-30 часов
effort. Делим на 3 wave-а:

  **Wave 1 — Backend foundation** (~7 ч):
    E7-12 (Lion/8bit registration), E7-9 (LR Schedules),
    E7-10 (Tooltip catalogue + catalog.explain RPC),
    E7-8 (Dynamic PRESETS list).

  **Wave 2 — Per-brick controls + auto-tune** (~12 ч):
    E7-4 (suggest_optim_groups + Auto-group UI),
    E7-5 (Activation selector + BrickContextPanel),
    E7-6 (Norm selector),
    E7-2 (Dim feedback + DimensionsTab),
    E7-13 (Extended activations: geglu/reglu/mish/xielu).

  **Wave 3 — Verification + final E2E** (~8 ч):
    E7-1 (Manual drag-drop тесты 26 бриксов),
    E7-3 (Parquet roundtrip + UI badge),
    E7-11 (Ablation runner + AblationsTab),
    E7-7 (E2E cross-product финал).

  **Optional** P3: E7-14 (Sophia/Adafactor/Tiger/AdEMAMix).

---
1. Что уже есть (backend ready + UI gaps)

См. VisualBuilderSpec-v2.md §1 для полного inventory. Краткая сводка:

  Backend готовых ~70% (Lion / Lion8bit / Adam8bit реализованы в
  cppmega_mlx/training/, MoE activation параметр в V4MoEConfig,
  ParamGroup matchers, ContractProbe, 10 JSON-RPC методов).

  UI готовых ~50% (5 sidebar tabs CRUD-полные, drag-drop работает,
  modal/tabs/playground/inspector смонтированы), но:
    - OptimTab KINDS = только [adamw,muon,muon_adamw_hybrid,sgd]
    - PRESETS array = 57 хардкод
    - per-brick context panel = нет
    - tooltips = нет
    - schedule editor = нет
    - dim feedback panel = нет
    - ablation runner = нет
    - parquet roundtrip badge = нет
    - auto-group button = нет

---
2. Что НЕ хватает (mapping to 14 stages)

См. VisualBuilderSpec-v2.md §2 для детальной таблицы. Краткое маппинг
gap → ticket:

  Gap                                     →  Ticket
  ─────────────────────────────────────────────────────
  Lion in OptimKind enum                  →  E7-12
  ActivationName 3→10                     →  E7-13
  _build_mlp activation param             →  E7-5 (backend часть)
  pre/post_norm params                    →  E7-6 (backend часть)
  LR Schedules backend                    →  E7-9 (backend часть)
  Tooltip catalogue                       →  E7-10 (backend часть)
  catalog.explain RPC                     →  E7-10
  ablation.run RPC                        →  E7-11 (backend часть)
  suggest_optim_groups RPC                →  E7-4 (backend часть)
  data.roundtrip_check RPC                →  E7-3 (backend часть)
  architectures.list_presets RPC          →  E7-8
  inference_log в resolve_shapes          →  E7-2 (backend часть)
  UI: per-brick context panel             →  E7-5 + E7-6 (UI часть)
  UI: DimensionsTab                       →  E7-2 (UI часть)
  UI: AblationsTab                        →  E7-11 (UI часть)
  UI: Schedule editor в OptimTab          →  E7-9 (UI часть)
  UI: tooltips dropdown                   →  E7-10 (UI часть)
  UI: Auto-group button                   →  E7-4 (UI часть)
  Manual drag-drop тесты                  →  E7-1
  E2E финальный cross-product             →  E7-7

---
3. Component stack (новые модули + расширения)

См. VisualBuilderSpec-v2.md §3 для детальных контрактов. Краткая
mapping:

  Backend additions:
    cppmega_v4/buildspec/schedules.py            (E7-9)
    cppmega_v4/buildspec/optim_spec.py           (E7-12, +Lion factories)
    cppmega_mlx/nn/activations.py                (E7-13, новый registry)
    cppmega_v4/explain/catalog.py                (E7-10, новый)
    cppmega_v4/jsonrpc/methods/catalog.py        (E7-10, RPC)
    cppmega_v4/jsonrpc/methods/ablation.py       (E7-11, RPC)
    cppmega_v4/jsonrpc/methods/suggest_optim_groups.py  (E7-4)
    cppmega_v4/jsonrpc/methods/architectures.py  (E7-8)
    cppmega_v4/jsonrpc/methods/data_roundtrip.py (E7-3)
    cppmega_v4/spec/resolver.py                  (E7-2, +inference_log)
    cppmega_v4/buildspec/api.py                  (E7-5/E7-6, params)

  UI additions:
    vbgui/src/components/sidebar/DimensionsTab.tsx  (E7-2)
    vbgui/src/components/sidebar/AblationsTab.tsx   (E7-11)
    vbgui/src/components/BrickContextPanel.tsx      (E7-5/E7-6)
    vbgui/src/components/Tooltip.tsx                (E7-10)
    vbgui/src/components/ExplainModal.tsx           (E7-10)
    vbgui/src/components/ScheduleEditor.tsx         (E7-9)
    vbgui/src/components/AutoGroupButton.tsx        (E7-4)
    vbgui/src/state/spec.ts                         (extend для всех)
    vbgui/src/hooks/useCatalog.ts                   (E7-10)
    vbgui/src/lib/activations.ts                    (E7-13)

  Test fixtures:
    tests/fixtures/build_e2e_matrix.py              (E7-3, +original_text)
    tests/fixtures/build_catalog_explain_fixtures.py (E7-10, для unit)

---
4. UI — детальное описание новых компонентов

### 4.1 OptimTab расширение (E7-9, E7-12, E7-4)

Текущий компонент остаётся; добавляются:

  - Раздел "Schedule" для каждой ParamGroup (collapsible, default
    closed):
      Schedule kind dropdown
      Conditional fields warmup_steps / total_steps / min_lr_ratio
      Mini-sparkline preview (50-point SVG curve)

  - Info icon (ⓘ) рядом с label "Kind" → opens ExplainModal с
    catalog.explain("optimizer", current_kind).

  - Кнопка [Auto-group from graph] над таблицей групп:
      Disabled если canvas пустой
      Loading state во время RPC
      После RPC: заменяет groups в draft, показывает баннер
      "Auto-generated 3 groups: 12 2D weights → Muon, 8 1D params →
       AdamW, 2 embeddings → AdamW"

### 4.2 DimensionsTab (E7-2, новый sidebar tab)

  Header: "Inferred Dimensions"
  Filter row: brick (multi-select) / source (auto/user/all) /
              search by param name
  Table: 5 columns (Brick / Param / Value / Source / Reason)
  Source 'auto' — синий бейдж, 'user' — серый
  Click на row → emit highlight event → FlowCanvas selectionChange

### 4.3 AblationsTab (E7-11, новый sidebar tab)

  Header: "Ablation Runner"
  Axis selector (radio): activation / optimizer / norm / schedule
  Variants section (depends on axis):
    Multi-select (2-4 items) с tooltip per option
    Recommended params per variant (read-only)
  Num steps input (10-100, default 20)
  Step options (collapsible advanced): lr / betas overrides

  [Run ablation] button:
    Disabled если variants < 2 или canvas пустой
    Progress bar (variants done / total)
    Cancellable

  Results section (после run):
    Sorted table by final_loss ascending
    Per row: variant name (★ baseline marker if matches base_spec) /
              final loss / Δ vs baseline (% green/red) / time / status
    Mini-chart loss curve per row (10×40px SVG)
    [Export JSON] button → download ablation_result.json

### 4.4 BrickContextPanel (E7-5, E7-6, новый overlay)

  Открывается при click на brick node в FlowCanvas.
  Размер: 320×500 px, поверх правой части canvas (НЕ внутри Sidebar —
  чтоб не закрывал графы).
  Header: brick kind + name + close button
  Tabs внутри: "Params" / "Norms" / "Activation" (если применимо)

  Params tab:
    Все params из ResolvedBrickGraph.inference_log для этого brick
    Editable если source='user', read-only badge если 'auto'
    [Apply] / [Reset to inferred] buttons

  Norms tab:
    pre_norm dropdown (rmsnorm / layernorm / none) + tooltip
    post_norm dropdown
    eps field

  Activation tab (только для mlp/gated_mlp/moe):
    Activation dropdown (10 options) с tooltip per option
    intermediate_size

  Все dropdowns используют <OptionWithTooltip> из E7-10.

### 4.5 Tooltip и ExplainModal (E7-10)

  Tooltip: lightweight, hover-only, 250ms delay, lazy fetch.
  Position: above dropdown option, arrow pointer.
  Content: summary + first sentence of when_to_use + "Click ⓘ for more".

  ExplainModal: full ExplainEntry display.
    Header: name + category badge + close
    Sections (с разделителями):
      Summary
      When to use
      When to avoid
      Recommended params (table)
      Gotchas (warning list)
      Paper reference (link)
    Footer: [Apply recommended params] button (для optim/schedule)

---
5. Stage breakdown — детально

### Stage E7-12 — Lion / Lion8bit / Adam8bit / SqReLU registration

  Bd ticket: cppmega-mlx-bb0.12 (P1, quick win)

  Files:
    cppmega_v4/buildspec/optim_spec.py (extend OptimKind enum + 3 factories)
    cppmega_v4/buildspec/validate_optim.py (Lion lr warning)
    cppmega_mlx/nn/moe.py (extend ActivationName Literal до 6)
    cppmega_mlx/nn/activations.py (новый — apply_activation dispatch)
    vbgui/src/components/sidebar/OptimTab.tsx (KINDS array)
    vbgui/src/state/spec.ts (OptimKind type)

  Implementation steps:
    1. Добавить LION, LION_8BIT, ADAM_8BIT в OptimKind enum.
    2. Написать lion() / lion8bit() / adam8bit() factories с
       правильными defaults (Lion lr=1e-4).
    3. Добавить validation в verify_optim_spec: warning если
       Lion lr > 5e-4.
    4. Расширить ActivationName Literal до 6 (gelu/relu/relu2/sqrelu/
       silu/swiglu) — без gated extensions (тех в E7-13).
    5. Добавить apply_activation dispatch в activations.py.
    6. UI: расширить KINDS array, опционально показывать
       recommended_lr рядом с kind dropdown.

  Unit tests (+10):
    - lion() возвращает OptimSpec с lr=1e-4
    - lion(lr=1e-3) → validate выдаёт warning
    - lion(lr=6e-4) → validate проходит (warning только >5e-4)
    - apply_activation("silu", x) parity vs nn.silu(x)
    - apply_activation("sqrelu", x) parity vs metal kernel
    - apply_activation("swiglu", x) без gate → ValueError
    - OptimKind.LION зарегистрирован
    - validate_optim_spec(LION_8BIT) с block_size=0 → error

  E2E tests (+3):
    - В UI: select kind=lion → assert dropdown показывает lion
    - Train с lion на mini-spec — assert loss finite за 2 шага
    - Train с lion lr=1e-3 на mini-spec — modal показывает warning

  Acceptance criteria:
    ✅ OptimKind содержит 7 entries (4 старых + 3 новых)
    ✅ ActivationName Literal содержит 6 entries
    ✅ vitest: 104 + 3 новых = 107 passed
    ✅ pytest: 2150 + 10 новых = 2160 passed
    ✅ E2E: 1135 + 3 = 1138 cells passed

  Budget: 2 часа.

### Stage E7-9 — LR Schedules + UI editor

  Bd ticket: cppmega-mlx-bb0.9 (P1, БЛОКЕР для real training)

  Files:
    cppmega_v4/buildspec/schedules.py (новый)
    cppmega_v4/buildspec/optim_spec.py (ScheduleSpec в ParamGroup.lr)
    cppmega_v4/buildspec/validate_optim.py (schedule validation)
    cppmega_v4/runner/stages.py (stage_train реально использует schedule)
    vbgui/src/components/ScheduleEditor.tsx (новый)
    vbgui/src/components/sidebar/OptimTab.tsx (intergrate ScheduleEditor)
    vbgui/src/state/spec.ts (ParamGroupState.schedule field)

  Implementation steps:
    1. ScheduleSpec dataclass + 6 factory functions.
    2. ScheduleSpec.build(base_lr) → Callable[[int], float] для каждого
       kind.
    3. Validation: см. VisualBuilderSpec-v2.md §6.2.
    4. stage_train передаёт schedule.build() в optimizer как
       learning_rate Callable.
    5. UI ScheduleEditor: kind dropdown + conditional fields +
       mini-sparkline (50 точек SVG).
    6. OptimTab: для каждой ParamGroup → expandable Schedule секция.

  Unit tests (+15):
    - cosine_annealing(100).build(1e-3)(0) > cosine_annealing(100).build(1e-3)(100)
    - linear_warmup_then_constant(10).build(1e-3)(5) == 1e-3 * 0.5
    - wsd(10, 20).build(1e-3)(15) == 1e-3 (steady phase)
    - inv_sqrt(10).build(1e-3)(0) == 0 (warmup)
    - polynomial(100, power=2.0).build(1e-3) монотонно убывает
    - constant().build(1e-3)(any_step) == 1e-3
    - ScheduleSpec(kind="cosine") без total_steps → validate error
    - ScheduleSpec(kind="wsd", warmup=10, decay=20, total=25) → error
      (10+20 > 25)
    - Train stage with cosine schedule reports per-step LRs

  E2E tests (+6):
    - В UI: ScheduleEditor для каждой ParamGroup отображается
    - Выбрать cosine schedule → sparkline появилась → train 5 шагов
      → loss finite
    - Выбрать wsd → conditional fields warmup/decay/min_lr_ratio
      появились
    - Выбрать constant → conditional fields скрыты
    - Train с warmup_steps=5 на 5 шагов train → lr_step_0 < lr_step_4
    - Validation: схема wsd без decay → red banner в UI

  Acceptance criteria:
    ✅ 6 schedule kinds работают для adamw/muon/lion
    ✅ UI ScheduleEditor отображает sparkline
    ✅ pytest: +15 = 2175
    ✅ vitest: +6 = 113
    ✅ E2E: +6 = 1144

  Budget: 4 часа.

### Stage E7-10 — Tooltip catalogue + catalog.explain RPC

  Bd ticket: cppmega-mlx-bb0.10 (P1, UX foundation)
  Depends on: E7-12 (Lion registered to be tooltippable)

  Files:
    cppmega_v4/explain/__init__.py (новый)
    cppmega_v4/explain/catalog.py (~60 ExplainEntry литералов)
    cppmega_v4/jsonrpc/methods/catalog.py (catalog.explain + list_options)
    cppmega_v4/jsonrpc/dispatcher.py (register methods)
    vbgui/src/lib/rpc.ts (typed wrappers)
    vbgui/src/hooks/useCatalog.ts (новый, memoized)
    vbgui/src/components/Tooltip.tsx (новый)
    vbgui/src/components/ExplainModal.tsx (новый)
    vbgui/src/components/OptionWithTooltip.tsx (wrapper)
    vbgui/src/components/sidebar/OptimTab.tsx (use Tooltip)
    vbgui/src/components/sidebar/LossTab.tsx (use Tooltip)
    vbgui/src/components/sidebar/ShardingTab.tsx (для compile_mode)

  Implementation steps:
    1. Написать ~60 ExplainEntry для CATALOG. Заголовки взять из
       paper'ов; recommended_params взять из реальных factories.
    2. catalog.explain(category, name) RPC.
    3. catalog.list_options(category) RPC (для populate dropdowns).
    4. Pydantic schemas в schema.py.
    5. UI Tooltip component с 250ms hover delay + lazy fetch.
    6. ExplainModal с full entry rendering + "Apply recommended"
       button.
    7. Wrap все dropdown options в OptionWithTooltip.
    8. useCatalog hook с per-session memo.

  Unit tests (+25):
    - CATALOG содержит entry для каждого OptimKind value
    - CATALOG содержит entry для каждой ActivationName value
    - CATALOG entry для cosine schedule содержит recommended warmup
    - catalog.explain("optimizer", "lion") → правильный ExplainEntry
    - catalog.explain("optimizer", "unknown") → not_found_message
    - catalog.list_options("schedule") → 6 entries
    - useCatalog memoizes по (category, name)
    - Tooltip lazy-loads только на hover
    - ExplainModal renders all sections
    - "Apply recommended params" обновляет state

  E2E tests (+8):
    - Hover на "Lion" в OptimTab kind dropdown → tooltip появляется с
      summary "Sign-based momentum..."
    - Click ⓘ рядом с "Kind" в OptimTab → ExplainModal открывается
    - В ExplainModal: paper link → clickable
    - "Apply recommended" в Lion modal → lr автоматом 1e-4
    - Hover на "swiglu" в BrickContextPanel → tooltip про gated
    - Hover на "wsd" в ScheduleEditor → tooltip про checkpoint reuse
    - catalog.list_options("activation") в UI → 10 entries в dropdown
    - Tooltip не появляется до 250ms (timing test)

  Acceptance criteria:
    ✅ CATALOG ≥ 60 entries (covers all OptimKind, ActivationName,
       Schedule kinds, LossKind, norms, rewriters, bricks)
    ✅ catalog.explain RPC < 5ms response time (cached)
    ✅ pytest: +25 = 2200
    ✅ vitest: +10 = 123
    ✅ E2E: +8 = 1152

  Budget: 4 часа (текст занимает много времени).

### Stage E7-8 — Dynamic PRESETS list

  Bd ticket: cppmega-mlx-bb0.8 (P2, quick fix для прошлой ошибки)

  Files:
    cppmega_v4/jsonrpc/methods/architectures.py (новый)
    cppmega_v4/jsonrpc/dispatcher.py (register)
    vbgui/src/App.tsx (replace hardcoded PRESETS array)
    vbgui/src/hooks/usePresets.ts (новый, fetch once)
    vbgui/tests/App.integration.test.tsx (update mock)
    vbgui/e2e/scenarios/02_preset_matrix.spec.ts (dynamic list)

  Implementation steps:
    1. Написать architectures.list_presets() RPC →
       sorted(cppmega_v4.architectures.PRESETS.keys()).
    2. UI: usePresets hook с useEffect on mount.
    3. App.tsx: PRESETS = usePresets() (list state).
    4. Update vitest mocks.
    5. Update e2e: scenario reads dynamic list из MATRIX.json
       (предварительно построенного через fixture generator).

  Unit tests (+3):
    - architectures.list_presets returns 62 entries (или сколько на
      момент запуска)
    - Все entries — valid keys в PRESETS dict
    - Sorted alphabetically

  E2E tests:
    - 02_preset_matrix.spec.ts: 62 × 4 × 4 = 992 cells (vs 912 раньше)

  Acceptance criteria:
    ✅ PRESETS list dynamic (no hardcode in App.tsx)
    ✅ Все 62 preset reachable через UI dropdown
    ✅ pytest: +3 = 2203
    ✅ E2E preset matrix: 992 cells passed (заменяет старые 912)

  Budget: 1 час.

### Stage E7-4 — suggest_optim_groups + Auto-group UI

  Bd ticket: cppmega-mlx-bb0.4 (P1)
  Depends on: E7-12 (Lion в OptimKind для AdamW+Lion hybrids)

  Files:
    cppmega_v4/jsonrpc/methods/suggest_optim_groups.py (новый)
    cppmega_v4/buildspec/group_inference.py (новый, heuristic)
    vbgui/src/components/AutoGroupButton.tsx (новый)
    vbgui/src/components/sidebar/OptimTab.tsx (integrate)

  Implementation steps:
    1. group_inference.py: эвристика для классификации параметров:
       - 1D shape ∧ name contains "embedding"|"lm_head" → embeddings
         group (AdamW)
       - 2D shape ∧ name contains "expert" → moe_experts group (AdamW)
       - 2D shape ∧ name contains "weight" → backbone (Muon если
         hybrid, AdamW если pure adamw)
       - 1D shape ∧ name contains "bias"|"norm" → 1d group (AdamW)
    2. RPC материализует graph (instantiate=True), собирает parameters,
       прогоняет эвристику, возвращает ProposedGroup list с rationale.
    3. UI AutoGroupButton: вызывает RPC, заменяет groups в draft.
    4. Inline баннер показывает rationale.

  Unit tests (+12):
    - Llama-style graph (attention+mlp) + muon_adamw_hybrid →
      2 группы: Muon на *.weight, AdamW на embeddings + 1D
    - Pure AdamW → 1 группа matcher="all" (тривиально)
    - MoE preset → группа moe_experts AdamW с правильным matcher
    - uncovered_params == 0 для всех 12 family-reps
    - rationale содержит количество параметров

  E2E tests (+5):
    - Load llama3_8b preset → OptimTab → Auto-group → 3 группы
      проявились
    - rationale tooltip содержит "Muon on 12 2D weights"
    - Train после auto-group → finite loss
    - Pure AdamW kind → auto-group выдаёт 1 группу
    - MoE preset → auto-group отдельная expert группа

  Acceptance criteria:
    ✅ suggest_optim_groups для всех 62 presets uncovered_params == 0
    ✅ Auto-group button visible в OptimTab
    ✅ pytest: +12 = 2215
    ✅ E2E: +5 = 1157

  Budget: 3 часа.

### Stage E7-5 — Activation selector + backend параметризация

  Bd ticket: cppmega-mlx-bb0.5 (P1)
  Depends on: E7-12 (расширенный ActivationName)

  Files:
    cppmega_v4/buildspec/api.py (_build_mlp / _build_gated_mlp принимают
                                 activation param)
    cppmega_v4/buildspec/validate.py (validate gated activation requires
                                      gated brick)
    cppmega_v4/spec/brick_metadata.py (per-brick activation defaults)
    vbgui/src/components/BrickContextPanel.tsx (новый, Activation tab)
    vbgui/src/components/FlowCanvas.tsx (click → emit selectBrick)
    vbgui/src/state/spec.ts (brick.update action)

  Implementation steps:
    1. Создать новый brick kind `gated_mlp` (отдельно от `mlp` для
       чистоты — dense vs gated имеют разные shapes/params).
    2. _build_mlp(activation="gelu" by default) — dense path.
    3. _build_gated_mlp(activation="swiglu" by default) — gated path с
       gate + up projections.
    4. Validation: BrickSpec(kind=mlp, params={activation: "swiglu"}) →
       ERROR ("swiglu requires gated_mlp").
    5. FlowCanvas onNodeClick → store.selectBrick(node.id).
    6. BrickContextPanel mount при selectedBrick != null.
    7. Activation dropdown в BrickContextPanel использует
       OptionWithTooltip из E7-10.
    8. apply активирует BrickSpec.update RPC (новый или через verify).

  Unit tests (+18):
    - _build_mlp с activation="gelu" → forward вернёт правильную shape
    - _build_mlp с activation="swiglu" → BuildDiagnostic error
    - _build_gated_mlp с activation="swiglu" → forward OK
    - _build_gated_mlp с activation="gelu" → BuildDiagnostic warning
      (gelu без gate — теряет gating capacity)
    - 10 activations × 3 brick variants = 30 combos: assert правильные
      pass/fail по IS_GATED tables
    - mish forward parity vs reference impl
    - relu2 vs sqrelu numerical equivalence test

  E2E tests (+10):
    - Click на mlp node → BrickContextPanel открывается
    - Сменить activation → Apply → trigger verify → no errors
    - Сменить mlp.activation на swiglu → red banner "requires gated_mlp"
    - Сменить gated_mlp.activation на swiglu→geglu → train OK
    - 5 activations × 2 brick variants = 10 cells, все train finite
    - Hover на "swiglu" в dropdown → tooltip про gated
    - BrickContextPanel close button работает
    - Click на другой node → панель обновляется

  Acceptance criteria:
    ✅ 10 activations × 3 brick variants validated (60 combinations)
    ✅ BrickContextPanel mountable per click
    ✅ pytest: +18 = 2233
    ✅ vitest: +10 = 133
    ✅ E2E: +10 = 1167

  Budget: 5 часов.

### Stage E7-6 — Norm selector

  Bd ticket: cppmega-mlx-bb0.6 (P1)

  Files:
    cppmega_v4/buildspec/api.py (builders принимают pre/post_norm)
    cppmega_v4/buildspec/validate.py (norm validation rules)
    vbgui/src/components/BrickContextPanel.tsx (Norms tab)

  Implementation steps:
    1. Расширить ~10 builders (_build_attention/_build_gqa_sliding/...)
       чтобы принимать pre_norm/post_norm params.
    2. Switch nn.RMSNorm / nn.LayerNorm / None в builder.
    3. Validation: pre=none && post=none → ERROR; parallel-block оба
       brick → both pre != none.
    4. BrickContextPanel.NormsTab: 2 dropdowns + eps + Apply.

  Unit tests (+15):
    - Attention с pre_norm="layernorm" → forward OK
    - Attention с pre=none post=none → BuildDiagnostic ERROR
    - Parallel-block с одним брика pre=none → WARNING
    - Mix RMSNorm + LayerNorm в одном блоке → WARNING
    - eps < 1e-8 на bf16 → WARNING
    - Все 10 builders принимают norm params

  E2E tests (+8):
    - Сменить pre_norm в UI → train OK с layernorm
    - pre=none post=none → red banner в UI
    - 3 norm × 4 family-reps = 12 cells (pick 8 для covarage), все train

  Acceptance criteria:
    ✅ 3 norm kinds × 10 builders supported
    ✅ pytest: +15 = 2248
    ✅ vitest: +8 = 141
    ✅ E2E: +8 = 1175

  Budget: 4 часа.

### Stage E7-2 — Dim auto-adjust feedback + DimensionsTab

  Bd ticket: cppmega-mlx-bb0.2 (P1)

  Files:
    cppmega_v4/spec/resolver.py (extend с inference_log)
    cppmega_v4/spec/inference_log.py (новый — InferenceEntry + builder)
    cppmega_v4/jsonrpc/schema.py (VerifyResult.inference_log)
    vbgui/src/components/sidebar/DimensionsTab.tsx (новый)
    vbgui/src/components/Sidebar.tsx (add tab)
    vbgui/src/state/spec.ts (verify.complete include inference_log)
    vbgui/src/components/FlowCanvas.tsx (highlight on tab click)

  Implementation steps:
    1. resolve_shapes: для каждого auto-inferred parameter создавать
       InferenceEntry с reason ("H/head_dim=128/64=2").
    2. Сохранять source='user' для params переданных явно.
    3. ResolvedBrickGraph.inference_log → VerifyResult →
       App.spec.inference_log.
    4. DimensionsTab: таблица с фильтрами.
    5. Click на row → emit highlight event → FlowCanvas selectsNode для
       2 секунд.

  Unit tests (+10):
    - num_heads inferred from H/head_dim → InferenceEntry source=auto
    - num_heads provided explicitly → InferenceEntry source=user
    - reason содержит формулу
    - intermediate_size auto-inferred для mlp без явного значения
    - q_lora_rank для mla auto-inferred

  E2E tests (+5):
    - Load llama3_8b → DimensionsTab показывает ≥3 inference rows
    - Click на row attn_0/num_heads → attn_0 node получает selected ring
    - Filter "auto" → только auto entries видны
    - Filter by brick name → отфильтровано
    - Mismatch (user override invalid) → red badge на entry

  Acceptance criteria:
    ✅ inference_log populated для всех 12 family-reps
    ✅ DimensionsTab functional
    ✅ pytest: +10 = 2258
    ✅ vitest: +5 = 146
    ✅ E2E: +5 = 1180

  Budget: 3 часа.

### Stage E7-13 — Extended activations (GeGLU/ReGLU/Mish/xIELU)

  Bd ticket: cppmega-mlx-bb0.13 (P2)
  Depends on: E7-12 (registry foundation)

  Files:
    cppmega_mlx/nn/activations.py (extend с 4 новыми + IS_GATED update)
    cppmega_v4/buildspec/validate.py (validate gated set)

  Implementation steps:
    1. Реализовать mish (x * tanh(softplus(x))) — dense.
    2. Реализовать geglu (gelu(gate) * up) — gated.
    3. Реализовать reglu (relu(gate) * up) — gated.
    4. Реализовать xielu (gelu(linear(gate)) * up) — gated.
    5. Extend ActivationName Literal до 10.
    6. Extend IS_GATED map.

  Unit tests (+12):
    - Forward parity per новой activation vs reference torch impl
    - Backward gradient finite для всех 4
    - IS_GATED correctly identifies gated set
    - Validation: reglu на dense_mlp → error

  E2E tests (+4):
    - 4 new activations × gated_mlp = 4 cells train finite
    - Hover на "mish" → tooltip про smooth nonlinearity

  Acceptance criteria:
    ✅ 10 activations всего; 4 gated; 6 dense
    ✅ pytest: +12 = 2270
    ✅ E2E: +4 = 1184

  Budget: 3 часа.

### Stage E7-1 — Manual brick assembly via drag-drop tests

  Bd ticket: cppmega-mlx-bb0.1 (P1)

  Files:
    vbgui/e2e/scenarios/08_manual_assembly.spec.ts (новый)
    vbgui/e2e/fixtures.ts (extend с dragBrickToCanvas helper)

  Implementation steps:
    1. 26 single-drop scenarios: для каждого brick kind drop из palette
       на canvas, assert brick-node появился.
    2. 8 multi-brick assembly scenarios: drop embedding → attention →
       mlp → lm_head, connect через handle drag.
    3. engram сценарий: load parquet с call_edges первым.

  Unit tests:
    - existing FlowCanvas.test.tsx уже покрывает drag-drop infra

  E2E tests (+34):
    - 26 single-brick drops
    - 8 multi-brick chains

  Acceptance criteria:
    ✅ 26 brick kinds successfully dropped via UI
    ✅ 8 multi-brick chains assembled
    ✅ E2E: +34 = 1218

  Budget: 3 часа.

### Stage E7-3 — Parquet roundtrip с UI badge

  Bd ticket: cppmega-mlx-bb0.3 (P1)

  Files:
    cppmega_v4/jsonrpc/methods/data_roundtrip.py (новый)
    tests/fixtures/build_e2e_matrix.py (add original_text column)
    vbgui/src/components/DataInspector.tsx (roundtrip badge per row)

  Implementation steps:
    1. data.roundtrip_check RPC (см. Spec §3.11).
    2. Extend fixture generator: добавить 'original_text' колонку в
       parquet (исходный текст до tokenize).
    3. DataInspector: per row → fetch roundtrip status → badge.

  Unit tests (+8):
    - T1/T2 roundtrip 100% match для ASCII text
    - T3 (256 vocab) partial fail на non-ASCII
    - byte_diff > 0 → matches=False

  E2E tests (+6):
    - DataInspector показывает Roundtrip OK для T1×P1
    - Roundtrip FAIL для T3×P4 (unicode) → red badge

  Acceptance criteria:
    ✅ 16 (tok × parq) combos: T1/T2 100% pass, T3 expected partial
    ✅ pytest: +8 = 2278
    ✅ E2E: +6 = 1224

  Budget: 3 часа.

### Stage E7-11 — Ablation runner + AblationsTab

  Bd ticket: cppmega-mlx-bb0.11 (P1)
  Depends on: E7-9 (schedules), E7-12 (Lion)

  Files:
    cppmega_v4/jsonrpc/methods/ablation.py (новый)
    cppmega_v4/buildspec/mutate_spec.py (helper для replace axis)
    vbgui/src/components/sidebar/AblationsTab.tsx (новый)
    vbgui/src/components/Sidebar.tsx (add tab)
    vbgui/src/hooks/useAblation.ts (новый)

  Implementation steps:
    1. mutate_spec(spec, axis, value) → new spec с заменённым
       компонентом.
    2. ablation.run RPC: для каждого variant клонировать → run
       stage_train с num_steps → collect losses.
    3. AblationsTab UI: axis selector + variants multi-select + run
       button + results table с mini-charts.
    4. Export JSON button.

  Unit tests (+15):
    - mutate_spec(spec, "activation", "gelu") → spec с MLP activation=gelu
    - mutate_spec(spec, "optimizer", "lion") → spec с lion + lr=1e-4
    - ablation.run([swiglu, gelu, relu2]) на llama3_8b → 3 results
    - ranked_by_final_loss сортирует ascending
    - Variant fail → status="fail" но другие продолжаются

  E2E tests (+8):
    - Select axis=activation → variants=[swiglu, gelu] → run → 2 rows
    - Loss curve mini-chart рендерится per row
    - Export JSON → download triggered
    - Optimizer axis: [adamw, lion] → both finite (Lion с auto lr=1e-4)
    - Schedule axis: [cosine, constant] → both finite

  Acceptance criteria:
    ✅ ablation.run < 60s для 4 variants × 20 steps
    ✅ AblationsTab отрендерен и functional
    ✅ pytest: +15 = 2293
    ✅ vitest: +10 = 156
    ✅ E2E: +8 = 1232

  Budget: 5 часов.

### Stage E7-7 — E2E cross-product финал

  Bd ticket: cppmega-mlx-bb0.7 (P1, финальный gate)
  Depends on: E7-1, E7-2, E7-4, E7-5, E7-6

  Files:
    vbgui/e2e/scenarios/09_e2e_manual.spec.ts (новый)

  Implementation steps:
    8 scenarios manual model assembly:
      1. Tiny mlp-only (embedding + mlp + lm_head)
      2. Attention+mlp (классический GPT block)
      3. Attention+moe
      4. MLA+mlp
      5. Mamba3+attention+mlp (hybrid)
      6. Parallel-block (attention || mlp с fan-in mean)
      7. Sliding+global mix
      8. Pure-ssm (mamba3-only stack)

    Per scenario:
      - Drop bricks via drag-drop
      - Connect
      - Click brick → BrickContextPanel → set activation + norm
      - OptimTab → Auto-group
      - Load tokenizer + parquet
      - Train → assert overall=ok + loss finite + weight delta > 0

  E2E tests (+8):
    - 8 manual assembly cells, all PASS

  Acceptance criteria:
    ✅ Все 8 manual models build + train successfully
    ✅ E2E total: +8 = 1240

  Budget: 4 часа.

### Stage E7-14 — Research optimizers (OPTIONAL)

  Bd ticket: cppmega-mlx-bb0.14 (P3, optional)

  Запускается только если пользователь явно скажет "хочу SOTA-tier
  ablation". Скорее всего НЕ нужен.

  Sophia / Adafactor / Tiger / AdEMAMix реализации.

  Budget: 8-12 часов.

---
6. Cross-cutting tests (regression + perf)

После каждой stage завершения:
  pytest tests/v4/ -q                  # full python regression
  cd vbgui && npx vitest run           # full UI unit regression
  cd vbgui && npx playwright test      # full e2e matrix (с retries=1)

Final regression target после всех wave 3:
  Python pytest:    ~2293 passed
  Vbgui vitest:     ~156 passed
  Playwright e2e:   ~1240 scenarios passed
  Wall-clock CI:    ~25 мин для preset matrix shard, ~5 мин unit

Performance gates:
  catalog.explain RPC < 5ms (cached)
  suggest_optim_groups < 100ms для 200-brick graph
  ablation.run 4×20 steps < 60s
  data.roundtrip_check 32 rows < 200ms

---
7. Бюджет + риски

Бюджет суммарно: ~32 часов (wave 1: 7ч + wave 2: 18ч + wave 3: 15ч).
В реальности ожидаемо ~25-30ч (часть stage'ей параллелятся внутри
wave).

Главные риски:

  **Risk 1**: Расширение builders для activation/norm params может
  сломать existing presets (gallery coverage test может стать
  красным). Mitigation: backwards-compat defaults — старые presets
  без явного activation/norm dict работают как раньше.

  **Risk 2**: CATALOG content (60 entries × ~50-100 слов каждый)
  занимает много времени на написание корректного текста с paper
  refs. Mitigation: первый коммит с stub-ами, второй с reviewed
  content; UI работает и со stub-ами.

  **Risk 3**: Lion lr автоматическая подстройка может ломаться когда
  пользователь явно переопределяет lr. Mitigation: validation
  warning, но НЕ блокирующий error — пользователь может настаивать.

  **Risk 4**: ablation.run серверная нагрузка — 4 variants × 20 steps
  = ~5-20s per ablation. UI должен показывать progress. Mitigation:
  cancellable run, WS progress updates (если успеется в Phase 1) или
  long-polling.

  **Risk 5**: BrickContextPanel UX — где он рендерится? Overlay поверх
  canvas или в Sidebar? Compromise: overlay (320×500px поверх правой
  части canvas), потому что Sidebar уже занят 5-ю tabs.

---
8. Acceptance criteria для closure эпика

✅ E7-12: OptimKind содержит 7 entries, ActivationName 6
✅ E7-9: 6 schedules работают + UI editor
✅ E7-10: ≥60 ExplainEntry в CATALOG + Tooltip + ExplainModal
✅ E7-8: PRESETS dynamic + 992-cell matrix (заменяет 912)
✅ E7-4: suggest_optim_groups для всех 62 presets uncovered=0
✅ E7-5: 10 activations × 3 brick variants validated; BrickContextPanel
✅ E7-6: 3 norm kinds × 10 builders supported
✅ E7-2: inference_log + DimensionsTab functional
✅ E7-13: ActivationName total 10 (4 gated + 6 dense)
✅ E7-1: 26 brick drag-drop scenarios + 8 multi-brick chains
✅ E7-3: Roundtrip badge + data.roundtrip_check RPC
✅ E7-11: AblationsTab + ablation.run для 4 axes
✅ E7-7: 8 manual assembly E2E scenarios passed
✅ Regression: pytest ≥2293, vitest ≥156, playwright ≥1240
✅ Markdown summary `tests/fixtures/e2e_matrix_v2_report.md` ≤200 lines
✅ All artefacts committed and pushed to main

---
9. Связь с другими спеками

  - **VisualBuilderSpec-v2.md** — техническая спецификация всех API,
    data structures, tooltip catalogue. Этот Plan ссылается на §X.Y
    Spec для деталей.
  - **VisualBuilderPlan.md / VisualBuilderSpec.md** (v1) — GUI shell
    + 5 sidebar tabs готовы. Этот Plan расширяет sidebar новыми tabs
    (Dimensions / Ablations) + добавляет BrickContextPanel overlay.
  - **E2EMatrix.md** (v1, эпик cppmega-mlx-pa3 закрыт) — mini-spec
    (H=128, depth=2, S=64) и fixture infra (4 tokenizer × 4 parquet)
    переиспользуются. Этот Plan добавляет к ним: 26 brick drag-drop
    scenarios, 10 activation × 3 brick combos, 3 norm × 10 builder,
    ablation runs, 8 manual assembly scenarios.
  - **ContractProbe.md** — probe.run остаётся, новый ablation.run
    живёт рядом по тому же API стилю.
  - **ModelBuildSpec.md** — extending LossSpec/OptimSpec/ParamGroup
    minor (ScheduleSpec inserted in ParamGroup.lr type union).
  - **ParallelismSpec.md** — без изменений.
