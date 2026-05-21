План: End-to-End Coverage Matrix (GUI → API → Real Training)

Цель: доказать пошагово, как живой пользователь, что Visual Builder
действительно работает — открываешь GUI, тянешь preset / bricks на
canvas, настраиваешь loss + optim + sharding в sidebar, грузишь свой
parquet shard через Data Inspector, выбираешь tokenizer в Playground,
жмёшь "Run" и видишь модель действительно строится, делает forward,
делает loss, делает 2 gradient steps. И это работает для всей галереи
Раschka на всех вариантах токенайзеров и parquet'ов.

Никаких backend-only smoke тестов в этом слое — каждая ячейка матрицы
прогоняется через **Playwright против реального React+FastAPI стека**,
со скриншотами в каждом нетривиальном промежуточном шаге.

Лежит **поверх** всего стека (VBSpec / MBSpec / PSpec / Probe / JSON-RPC
/ Runner / VBGui) — это финальный gate, который не пускает регрессии
на main.

---
1. Что уже есть (карта существующего)

Backend (всё shipped на main):
  - cppmega_v4.jsonrpc — 9 методов (verify / suggest_sharding /
    suggest_adapters / build_preset_specs / probe.run / pipeline.run /
    tokenizer.encode_visualize / tokenizer.list_presets /
    data.preview_parquet / backend.status). FastAPI server.
  - cppmega_v4.runner — Pipeline с 12 stages, run_pipeline, cppmega-run CLI.
  - cppmega_v4.probe — Contract Probe (tokenizer + parquet capabilities,
    BRICK_REQUIREMENTS, alternatives generator).
  - cppmega_v4.widget — anywidget VisualBuilderWidget.
  - cppmega_v4.architectures — 57 PRESETS покрывают всю Раschka галерею
    (через 71-entry mapping в test_galcov_stage_d.py).

Frontend (vbgui/, shipped):
  - React Flow canvas + Palette + Sidebar (5 tabs) + TopBar + BottomStrip.
  - TokenizerPlayground + DataInspector components (но НЕ смонтированы в
    App — только standalone, без routing).
  - Anywidget bundle (npm run build:widget).

GUI tests:
  - 77 vitest unit tests (jsdom) — компоненты в изоляции.
  - 0 browser E2E против реального HTTP+WS стека.
  - 0 скриншотов.
  - 0 проверок что preset.button → backend → canvas пайплайн реально
    работает end-to-end.

Backend tests:
  - 2103 pytest тестов — все unit + system-level (TestClient / subprocess
    cli). 71-arch smoke matrix (test_galcov_stage_d.py) — но без
    tokenizer × parquet комбинаторики и без real gradient steps.

---
2. Что НЕ хватает (gap-analysis)

┌────────────────────────────────────────────┬──────────────────────────────┐
│              Что нужно                      │           Сейчас             │
├────────────────────────────────────────────┼──────────────────────────────┤
│ Preset launcher реально создаёт ноды       │ stub в App.tsx (пустой       │
│ на canvas через RPC                         │ handler)                     │
├────────────────────────────────────────────┼──────────────────────────────┤
│ Run pipeline button реально вызывает       │ stub                         │
│ pipeline.run и показывает modal             │                              │
├────────────────────────────────────────────┼──────────────────────────────┤
│ TokenizerPlayground + DataInspector        │ компоненты есть, не в App    │
│ доступны через UI                           │                              │
├────────────────────────────────────────────┼──────────────────────────────┤
│ Backend heartbeat обновляет BottomStrip    │ status всегда disconnected   │
├────────────────────────────────────────────┼──────────────────────────────┤
│ Sharding proposals видны юзеру             │ proposals=[] hardcoded       │
├────────────────────────────────────────────┼──────────────────────────────┤
│ Real Playwright против полного React+FastAPI │ ✗                         │
├────────────────────────────────────────────┼──────────────────────────────┤
│ Tokenizer matrix (3-4 варианта в            │ ✗ (только vendored)         │
│ fixtures, под все размеры)                  │                              │
├────────────────────────────────────────────┼──────────────────────────────┤
│ Parquet matrix (4 варианта enrichment)      │ ✗ (tmp_path per test)       │
├────────────────────────────────────────────┼──────────────────────────────┤
│ Real mini-train: 2 gradient steps,          │ ✗ (только smoke pipeline)   │
│ loss finite, weight delta > 0               │                              │
├────────────────────────────────────────────┼──────────────────────────────┤
│ Все 66 не-xfail Раschka archs × 4 tok       │ ✗                            │
│ × 4 parquet = 1056 ячеек проверены через GUI│                              │
├────────────────────────────────────────────┼──────────────────────────────┤
│ Screenshot regression matrix                 │ ✗                            │
├────────────────────────────────────────────┼──────────────────────────────┤
│ Probe failure → alternative → re-run path   │ ✗                            │
│ проверен через UI                            │                              │
└────────────────────────────────────────────┴──────────────────────────────┘

Конкретный gap: разработчик жмёт preset "qwen3_next" в top bar и НИЧЕГО
не происходит. Никто это не ловит — unit-тесты проверяют что компонент
рендерится, но не что preset-кнопка реально создаёт BrickGraph через
backend. Этот roadmap закрывает gap end-to-end matrix-тестами через
Playwright.

---
3. Архитектура (новый пакет: vbgui/e2e/ + tests/fixtures/)

3.1 Mini-model спецификация (фикс на весь матрикс)

**Hard cap: 200M parameters per model** (≈400 MB bf16 weights).
**Preferred: <100M**. Реальная мотивация — 3B на 2k blocks ест ~60 GB,
1B на 2k тоже впритык. На тестах нам нужно прогнать пару сот моделей
сегодня же, значит на каждой 30s бюджета, значит каждая обязана влезть
в RAM+VRAM мгновенно и forward'нуть за <1s.

Все архитектуры строятся с теми же micro-params:

  DIM_ENV_MINI = {
      "B": 1, "S": 64, "H": 128,
      "nh": 2, "nkv": 1, "head_dim": 64,
      "num_experts": 4, "top_k": 2,
  }
  DEPTH = 2  # repeat unit повторяется дважды

Параметры по архитектурам (теоретический расчёт):
  - Embedding (доминирует): 50257 × 128 = 6.4M (GPT-2 small class)
  - Attention block ≈ 200K params (Q/K/V/O проекции 128×128 × 4)
  - MLP block ≈ 200K params (128×512×128)
  - MoE block ≈ 800K params (4 experts × 200K)
  - Repeat unit ≈ 400K-1M в зависимости от family
  - Всего: **7-15M params** на типичной архитектуре, **<30M** в худшем
    случае (MoE-heavy presets с big embedding).

Гарантия cap'а: в Phase 0 wire App.tsx добавляется precondition check
`assert total_params < 200_000_000` в bridge между preset_launcher и
canvas; если preset на mini-spec выдаёт >200M, тест отмечается xfail
с reason `preset_too_large_for_e2e_mini`. На практике 57 наших PRESETS
на mini-spec все укладываются <30M (verified analytically в design
phase, регрессион-тест в Stage E-0).

S=64 (не S=512 и не S=2048) — обоснование: на E2E матрице нам нужно
быстрое доказательство корректности pipeline'а, не реалистичный training.
S=64 даёт kv_cache 64 × 128 × 2bytes = 16 KB per layer × 2 layers = 32 KB.
Activations 1 × 64 × 128 × 2 = 16 KB. Полная activation memory <1 MB
на любой архитектуре. Forward < 50ms.

3.2 Tokenizer matrix — tests/fixtures/tokenizers/

  | Имя              | Vocab | Спецтокены           | Зачем                   | Embed @H=128 |
  |------------------|-------|----------------------|-------------------------|--------------|
  | T1_cppmega_v3    | 65536 | Full FIM + SPACE/NL  | Наш golden path         | 8.4 M        |
  | T2_gpt2_small    | 50257 | BOS/EOS только       | Industry baseline       | 6.4 M        |
  | T3_minimal_no_fim| 256   | PAD/UNK/BOS/EOS      | Должен ломать IFIM      | 33 K         |
  | T4_fim_only      | 1024  | FIM + BOS/EOS        | FIM без vocab излишеств | 131 K        |

Большие вокабы (200K+) НЕ берём — embedding на 200K × 128 = 25M уже
съедает половину 50M cap'а просто на эмбеддинге, не оставляя места
для bricks. GPT-2 small (50257) — industry sweet spot.

T1 — копия `cppmega_mlx/tokenizer/tokenizer.json`.
T2 — `tokenizers.Tokenizer.from_pretrained('gpt2')` (офлайн-cache в
fixtures/, не сетевой запрос в CI).
T3/T4 — генерятся скриптом BPE-trainer-ом на small corpus (clang
sample 100 строк), checked-in как .json (256/1024 vocab — маленькие).

3.3 Parquet matrix — tests/fixtures/parquet/

32 row × 64 tokens per shard, deterministic seed.

  | Имя             | Колонки                                                       | Покрывает          |
  |-----------------|---------------------------------------------------------------|--------------------|
  | P1_minimal      | input_ids                                                     | bare minimum       |
  | P2_doc          | + doc_ids                                                     | + boundaries       |
  | P3_engram       | + call_edges                                                  | engram / csa_hca   |
  | P4_full         | + loss_mask, chunk_boundaries, type_edges, constituent_*      | enriched corpus    |

Один shard per tokenizer (4 × 4 = 16 shards). Encoded под выбранный
tokenizer, чтобы input_ids действительно резолвились.

3.4 Phase 0 — wire App.tsx (Stage F-I — pre-condition)

App.tsx сейчас имеет stub handlers. Без этого никакой Playwright
никаких реальных сценариев гонять не может.

Новые модули frontend:
  - vbgui/src/hooks/useRpc.ts — singleton RpcClient + WS subscribe
    для backend.status heartbeat.
  - vbgui/src/hooks/useVerifyAfter.ts — debounced verify.request после
    каждой mutation (graph / loss / optim / sharding).
  - vbgui/src/components/RunResultModal.tsx — таблица 8+ stages с
    ✓/✗/elapsed_ms, expanded detail per failed stage, кнопка Download JSON.
  - vbgui/src/components/AppTabs.tsx — переключатель между
    Canvas / Tokenizer / Data вьюхами.

Handlers в App.tsx (раньше stub, теперь real):
  - handlePresetDrop → rpc.call('build_preset_specs') → emit nodes/edges
  - handleRunPipeline → rpc.call('pipeline.run') → open RunResultModal
  - handleShardingAccept → apply proposal axes → trigger re-verify
  - Backend heartbeat WS → dispatch({type: 'backend.status'})
  - useVerifyAfter wraps spec dispatcher → debounce 150ms → verify.request

3.5 Phase 1 — fixture generator

tests/fixtures/build_e2e_matrix.py:
  - generate_tokenizers() → 4 .json файла
  - generate_parquets() → 16 shards (4 tokenizer × 4 schema)
  - validate_round_trip() → каждый shard декодится обратно в текст,
    sanity check
  - Idempotent: hash inputs → skip regeneration

3.6 Phase 2 — Playwright scaffolding

vbgui/e2e/
  playwright.config.ts — chromium only headless, 1280×800, retries=1
  globalSetup.ts:
    - spawn `uvicorn cppmega_v4.jsonrpc.server:create_app --port 8765`
    - spawn `npm run dev -- --port 5173`
    - wait для обоих ready (poll /health и localhost:5173)
    - teardown в globalTeardown.ts
  fixtures.ts — per-test page + utils:
    - selectPreset(name) — open dropdown, click, wait nodes
    - setLossKind(k) — switch sidebar tab, select, fill params, Apply
    - loadTokenizer(path) — Tokenizer tab → input path → Encode
    - loadParquet(path) — Data tab → input path → Load
    - clickRun(mode) — top bar button → wait for modal → return result
    - assertNoErrorBadge() — verify zero red edge/node/gotcha chips
  utils/screenshot.ts — `screenshot(page, name)` → vbgui/e2e/screenshots/
  utils/matrix.ts — readonly список all combinations, экспорт для
    parametrised tests

3.7 Phase 3 — scenarios

  scenarios/
    01_canvas_smoke.spec.ts        # manual brick-by-brick (5 archs)
    02_preset_matrix.spec.ts       # 66 archs × 4 tok × 4 parq = 1056
    03_train_matrix.spec.ts        # 12 family-rep × 4 tok × 4 parq = 192
    04_tokenizer_playground.spec.ts # 3 panels side-by-side compare
    05_data_inspector.spec.ts      # 4 parquet variants, pagination
    06_probe_failure_path.spec.ts  # IFIM × T3 → alternative → re-run
    07_sharding_proposals.spec.ts  # accept proposal → mem bar updates
    08_gotchas_autofix.spec.ts     # fsdp2_whole_compile → auto-fix

Каждый scenario — full GUI walkthrough, with screenshots after key
actions (open preset / after run modal / after fix etc).

3.8 Phase 4 — CI workflow

.github/workflows/e2e-matrix.yml:
  - На каждый push в main: Phase 1 + 2 + 3 (preset matrix only)
  - Раз в день (cron): + Phase 3.03 (train matrix, реальный mlx)
  - Artifacts: screenshots/*.png (3-day retention)
  - Report: e2e_matrix_report.md загружается в Pages

---
4. Стадии (mapping на bd-эпик `cppmega-mlx-e2e`)

Stage E-0 — Wire App.tsx (новый ticket cppmega-mlx-e2e.0)
  Files: vbgui/src/hooks/*, vbgui/src/components/RunResultModal.tsx,
         AppTabs.tsx, App.tsx rewrite
  Tests: +20 vitest для новых hooks/modal/tabs
  Budget: ~1.5 дня

Stage E-1 — Fixtures + matrix generator (cppmega-mlx-e2e.1)
  Files: tests/fixtures/build_e2e_matrix.py, tests/fixtures/{tokenizers,parquet}/
  Tests: +10 pytest — generate_*, validate_round_trip
  Budget: ~0.5 дня

Stage E-2 — Playwright scaffolding (cppmega-mlx-e2e.2)
  Files: vbgui/e2e/{playwright.config.ts, globalSetup.ts, fixtures.ts, utils/*}
  Tests: 01_canvas_smoke.spec.ts — proof что инфраструктура работает на
         5 manual scenarios
  Budget: ~0.5 дня

Stage E-3 — Preset matrix (cppmega-mlx-e2e.3)
  Files: vbgui/e2e/scenarios/02_preset_matrix.spec.ts +
         tests/fixtures/e2e_matrix_report.json generator
  Tests: 1056 cells × headless chromium workers=4 → ~17 мин wall-clock
  Budget: ~1 день (build + fix iterations)

Stage E-4 — Mini-train matrix (cppmega-mlx-e2e.4)
  Files: 03_train_matrix.spec.ts + backend pipeline 'optimizer_real' stage
         (заменяет текущий optimizer_smoke placeholder)
  Tests: 192 cells × ~30s each = ~25 мин wall-clock с workers=4
  Budget: ~1 день

Stage E-5 — Playground / Inspector / Probe-failure / Sharding / Gotchas
  scenarios (cppmega-mlx-e2e.5)
  Files: 04..08 spec.ts
  Tests: ~50 scenarios всего
  Budget: ~0.5 дня

Stage E-6 — CI workflow + matrix report (cppmega-mlx-e2e.6)
  Files: .github/workflows/e2e-matrix.yml, tests/fixtures/e2e_matrix_report.md
  Tests: workflow syntax check + smoke run в actions/setup-node
  Budget: ~0.5 дня

Итог: ~5 дней effort, ~4000-5000 LoC (TS + Python), ~100 новых тестов
(unit + e2e cells), 1100+ screenshots.

---
5. API entry point + интеграция

Researcher / developer flow:

  # Один раз — сгенерить fixture matrix
  python tests/fixtures/build_e2e_matrix.py

  # Локально гонять E2E (требует node + playwright browsers installed)
  cd vbgui && npm run e2e             # full matrix, ~50 мин
  cd vbgui && npm run e2e -- -g "preset_matrix" --workers=8  # narrow

  # Сгенерить report
  npm run e2e:report                  # читает test-results/, emits md

CI flow:
  - main push → workflow gates на Stage 1+2+3 + 1 smoke train scenario.
  - Daily schedule → full Stage 4 train matrix (macos-latest runner).
  - PR diff → run scenarios filtered по changed files (только
    affected архитектуры если поменялись presets).

---
6. Бюджет + риски

Бюджет: ~5 дней effort. CI gate сразу gates на Stage 3 (preset matrix);
Stage 4 mini-train запускается на каждое изменение в cppmega_v4/ но
schedule-only для main. Phase 0 (wire App) — pre-condition, без него
nothing else стартует.

**Главный риск**: некоторые MLX bricks (ssm, linear_attn варианты) не
поддерживают gradient через mx.eval на момент написания. Mitigation:
xfail с явным reason в matrix report; не блокируем остальные ячейки.

**Второй риск**: Playwright HTML5 drag-drop капризный (требует
`page.dispatchEvent` ручками для dragstart/dragover/drop). Mitigation:
для preset-launcher path используем select/click, для manual-brick
test (только 5 scenarios) пишем явный helper `dragBrickToCanvas` с
ручными синтетическими событиями. Если совсем не пойдёт — fallback на
keyboard shortcut "drop attention" через store action.

**Третий риск**: vite dev server + uvicorn в CI — port collisions,
зомби процессы. Mitigation: dynamic port в globalSetup, child_process
с explicit kill в globalTeardown, retry с 5s timeout на ready check.

**Четвёртый риск**: 1100+ screenshots раздувают git и CI artifacts.
Mitigation: .gitignore vbgui/e2e/screenshots/; в CI — artifact-retention
3 дня; для on-disk regression только diff против baseline (Playwright
toMatchSnapshot со threshold 0.05).

**Пятый риск**: mlx-on-ubuntu CI — нет нативной поддержки. Mitigation:
train matrix только на macos-latest. На ubuntu skip с reason
"mlx_not_available". Stage 3 (preset matrix без gradient) гоняем на
обоих — там MLX используется только для instantiate, без real eval.

**Шестой риск**: разрастающийся wall-clock CI. Mitigation:
sharding scenarios через `playwright --shard 1/4 ... 4/4` parallel jobs.

---
7. Что вне scope (явно)

  - **Pixel-perfect visual regression** — скриншоты для проверки
    отрисовки/артефактов, не для byte-identical diff. Использовать
    Playwright toMatchSnapshot с threshold (0.05) только для
    стабильных layouts (TopBar / BottomStrip).
  - **Performance benchmarking** — Playwright не для replay-нагрузки,
    есть отдельный bench-уровень.
  - **Multi-user concurrent сессии** — single-user mode достаточно.
  - **Mobile / responsive layouts** — desktop 1280×800 ONLY (Visual
    Builder — desktop tool, не mobile).
  - **Cross-browser fully** — только Chromium. WebKit/Firefox добавим
    отдельным тикетом если будет реальный bug report.
  - **Internationalisation** — английский UI only.
  - **Full Раschka 71 × 4 × 4 = 1136 для train** — только smoke (1056).
    Train на 12 family-rep × 16 fixture combos = 192.
  - **100+ epoch convergence** — только 2 grad steps, проверяем что
    loss finite + weight delta > 0.
  - **Multi-device (FSDP/TP)** — только sharding proposals через UI,
    без реального distributed запуска. PSpec verify покрывает.
  - **WebSocket heartbeat real-clock timing** — мокаем frequency в
    test mode (1 Hz слишком долго для CI).

---
8. Связь с другими спеками

  - **VBSpec / MBSpec / PSpec / Auto-Fusion**: потребляются через
    cppmega_v4.jsonrpc.dispatch как backend; не модифицируем.
  - **ContractProbe.md**: probe.run RPC — гарантирует что probe
    отвечает за <2 сек, что критично для UX. Stage E-3 включает
    probe-call для каждой combination.
  - **VisualBuilderPlan.md (epic cppmega-mlx-o0k, ✅ done)**:
    Stage F-A..F-H предоставили компоненты, Stage E-0 их склеивает
    реальной wiring; Stage E-1..6 — это **итоговый proof-of-life**
    что F-A..F-H реально работают вместе на полной матрице.
  - **Gallery Coverage Completion (cppmega-mlx-1t0, ✅ done)**:
    71-entry GALLERY переиспользуется как источник архитектур.
    Stage D test становится строжайшим super-set'ом Stage E-3.

---
9. Acceptance criteria для closure эпика

✅ Phase 0: все vitest tests green + новые hooks/modal/tabs покрыты
✅ Phase 1: 4 tokenizer + 16 parquet shards генерируются deterministic
✅ Phase 2: Playwright globalSetup поднимает оба server в <30s
✅ Phase 3: 1056-cell preset matrix — все non-xfail PASS
✅ Phase 4: 192-cell train matrix — loss finite + weight Δ > 0 per cell
✅ Phase 5: 50+ specialised scenarios — все green
✅ Phase 6: CI workflow зелёный на каждом push + daily train run
✅ Markdown summary `tests/fixtures/e2e_matrix_report.md` ≤ 200 lines
✅ Полный v4 регрессион passing: 2103+ pytest, 77+ vitest, 1300+ playwright
