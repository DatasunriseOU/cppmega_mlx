План: Contract Probe Layer (dry-run capability check над Model + Data + Tokenizer)

Цель: дать пользователю — до того, как он запустит первый training step
на дорогом железе — **explicit, machine-readable report**: «вот эти
блоки твоей модели требуют side-channel X, твой parquet его не даёт,
твой tokenizer не умеет special-token Y. Вот N альтернатив: (а) ослабь
loss, (б) добавь колонку Z, (в) переключи tokenizer на W». Без волшебной
авто-адаптации в рантайме — пользователь принимает решение явно, диффом
к спеке, и только потом мы строим модель.

Лежит **поверх** всех трёх backend-слоёв (VBSpec / MBSpec / PSpec) и
**рядом** с Visual Builder GUI: GUI получает `ContractProbeReport` как
data-структуру, рендерит чипы, проксирует accept/reject обратно в diff.

---
1. Что уже есть (карта существующего)

V4 Bricks (cppmega_v4/models/unified_superblock_v4.py)
  25 kinds в BLOCK_BUILDERS. Каждый — `nn.Module`. ShapeContract
  знает inputs/outputs, но НЕ знает «нужны ли мне FIM-токены», «жду
  ли я doc_ids side-channel», «требую ли я K shifted-label streams».
  Эти требования сейчас живут implicit в коде брика — узнаются через
  runtime crash.

LossSpec (cppmega_v4/buildspec/loss_spec.py)
  Знает свой kind + head_outputs. НЕ декларирует data dependencies
  (MTP_WEIGHTED требует K сдвинутых label-streams; IFIM_SHAPED требует
  FIM_PREFIX/MIDDLE/SUFFIX в vocab; MHC_ATTN_BIAS требует multi-stream
  copy-source поля в parquet).

Tokenizer (cppmega_mlx/tokenizer/_nanochat_decoder.py)
  Production tokenizer: vocab=65536 BPE, 48 special-id contract.
  НЕ умеет себя описывать: «есть ли у меня FIM_INSTRUCTION (id=45)?»,
  «знаю ли я <NL>/<SPACE> spans?». Caller узнаёт через try-except.

Parquet datasets (scripts/nanochat_data/*.py)
  Production schemas: clang_semantic_4k_v10 + clang_commits_4k_v1.
  12 token-aligned колонок были дропнуты после 2026-05-02 redesign
  (см. bd memory `after-2026-05-02-space-nl-tokenizer-redesign`).
  Нет API «что эта parquet вообще предлагает в качестве side-channels».

Rewriters (cppmega_v4/buildspec/rewriters.py)
  MTPRewriter / IFIMRewriter / MHCRewriter знают свои preconditions
  (state-token set). НЕ знают, как сообщить наружу «мне нужны эти
  колонки в parquet» — только взрывают на verify_build_spec.

---
2. Что НЕ хватает (gap-analysis)

┌─────────────────────────────────────────┬──────────────────────────────────┐
│              Что нужно тренинг-pipeline │             Сейчас               │
├─────────────────────────────────────────┼──────────────────────────────────┤
│ Спросить tokenizer: «какие специальные  │ ✗ (читаем code+constants руками) │
│ id ты предоставляешь, vocab size?»      │                                  │
├─────────────────────────────────────────┼──────────────────────────────────┤
│ Спросить parquet: «какие колонки + dtypes│ ✗ (открываем .schema руками)    │
│ + сколько rows + side-channel мета?»    │                                  │
├─────────────────────────────────────────┼──────────────────────────────────┤
│ Спросить каждый brick: «какие side-     │ ✗ (live в коде модуля)           │
│ channels мне нужны / опциональны?»      │                                  │
├─────────────────────────────────────────┼──────────────────────────────────┤
│ Спросить LossSpec: «какие data deps?»   │ ✗ (только head_outputs)          │
├─────────────────────────────────────────┼──────────────────────────────────┤
│ Решатель: проверить все requirements vs │ ✗ (узнаём через NaN на step 1)   │
│ всё что предоставлено                   │                                  │
├─────────────────────────────────────────┼──────────────────────────────────┤
│ Для каждого unmet — сгенерить N         │ ✗                                │
│ explicit альтернатив (no auto-magic)    │                                  │
├─────────────────────────────────────────┼──────────────────────────────────┤
│ Эмитнуть diff-к-спеке для каждой        │ ✗                                │
│ принятой альтернативы                   │                                  │
├─────────────────────────────────────────┼──────────────────────────────────┤
│ Dry-run forward на минимальной модели   │ ✗ (только реальный train)        │
│ для верификации                         │                                  │
└─────────────────────────────────────────┴──────────────────────────────────┘

Конкретный gap: пользователь нарисовал в GUI Tiny Aya + MTP K=2, грузит
наш clang_semantic_4k_v10 parquet → crash на step 1 потому что MTPRewriter
требует K shifted-label streams, а наш parquet даёт один token stream
(`input_ids`) и provenance, а shifted labels вообще должны строиться в
collate. Сейчас способа узнать об этом до запуска НЕТ.

---
3. Архитектура (новый пакет cppmega_v4/probe/)

Контракт пакета: всё read-only (никакого мутирования спеки), всё
deterministic (один и тот же вход → один и тот же отчёт), всё
дешёвое (probe должен укладываться в <2 сек на minimal-hidden-64
preset; цель — запускать его на каждое изменение в GUI).

### 3.1 cppmega_v4/probe/capabilities.py — два introspector-а

  @dataclass(frozen=True)
  class TokenizerCapabilities:
      """Read-only snapshot tokenizer-а. Что предлагает наружу."""
      vocab_size:        int
      special_ids:       Mapping[str, int]    # {"BOS":2,"FIM_PREFIX":4,...}
      has_fim:           bool                  # FIM_PREFIX+MIDDLE+SUFFIX все есть
      has_space_nl:      bool                  # SPACE(46)+NL(47) explicit
      has_code_start:    bool                  # CODE_START(7) для дисамбигации
      has_instruction:   bool                  # FIM_INSTRUCTION(45)
      byte_roundtrip:    Literal["exact","approx","none"]  # decode parity
      decoder_kind:      Literal["custom","hf","none"]
      source:            str                   # путь / hub-id / "vendored"

  @dataclass(frozen=True)
  class ParquetCapabilities:
      """Read-only snapshot parquet shard. Что в нём лежит."""
      schema_columns:    tuple[ColumnSpec, ...]
      row_count:         int
      total_bytes:       int
      has_token_ids:     bool
      has_doc_ids:       bool
      has_chunk_spans:   bool                  # chunk_boundaries column
      has_call_edges:    bool                  # for engram / mhc
      has_type_edges:    bool
      has_provenance:    bool                  # constituent_provenance*
      side_channels:     frozenset[str]        # union всех not-NULL метаканалов
      sample_seq_lens:   tuple[int, ...]       # первые N seq длин для оценки

  @dataclass(frozen=True)
  class ColumnSpec:
      name:           str
      arrow_dtype:    str               # "int32", "list<int32>", "string"
      nullable:       bool
      non_null_ratio: float             # 0..1 на sample

  def introspect_tokenizer(source: str | Path) -> TokenizerCapabilities: ...
  def introspect_parquet(path: Path, sample_rows: int = 256) -> ParquetCapabilities: ...

### 3.2 cppmega_v4/probe/requirements.py — что просят бриксы и loss

Каждый брик + каждая loss декларируют что им нужно из data. Это
полностью dataclass-table, без runtime-инспекции — детерминировано.

  @dataclass(frozen=True)
  class DataRequirement:
      key:        str                   # "token_ids", "fim_prefix_id", "doc_ids"
      origin:     Literal["tokenizer","parquet","derived"]
      required:   bool                  # hard fail vs soft warning
      reason:     str                   # human readable for the report
      satisfied_by: tuple[str, ...] = () # equivalent provider keys (any-of)

  # Static table — единственное место где описаны deps; growing list.
  BRICK_REQUIREMENTS: Mapping[str, tuple[DataRequirement, ...]] = {
      "engram":           (DataRequirement("call_edges", "parquet", True, ...),),
      "mhc":              (DataRequirement("type_edges", "parquet", True, ...),),
      "abs_pos_embed":    (),  # nothing beyond input_ids
      "mlstm":            (),
      "per_layer_embed":  (),
      ...
  }

  LOSS_REQUIREMENTS: Mapping[LossKind, tuple[DataRequirement, ...]] = {
      LossKind.CROSS_ENTROPY: (DataRequirement("labels", "derived", True, ...),),
      LossKind.MTP_WEIGHTED:  (DataRequirement("labels_k_shifted", "derived",
                                              True, "MTP needs K shifted streams"),),
      LossKind.IFIM_SHAPED:   (DataRequirement("fim_prefix_id", "tokenizer", True, ...),
                              DataRequirement("fim_middle_id",  "tokenizer", True, ...),
                              DataRequirement("fim_suffix_id",  "tokenizer", True, ...)),
      LossKind.MHC_ATTN_BIAS: (DataRequirement("type_edges", "parquet", True, ...),),
  }

### 3.3 cppmega_v4/probe/probe.py — solver

  @dataclass(frozen=True)
  class ProbeFinding:
      """Один пункт отчёта."""
      kind:        Literal["satisfied","unsatisfied","warning"]
      component:   str                  # "brick:engram_3" / "loss:mtp"
      requirement: DataRequirement
      message:     str
      alternatives: tuple[Alternative, ...] = ()   # пусто если satisfied

  @dataclass(frozen=True)
  class Alternative:
      action:     Literal["swap_loss","swap_tokenizer","add_column",
                          "drop_brick","relax_requirement"]
      target:     str                   # component being changed
      diff:       Mapping[str, object]  # JSON-Patch op to apply to spec
      cost:       Literal["low","medium","high"]
      reason:     str                   # why this alternative was generated

  @dataclass(frozen=True)
  class ContractProbeReport:
      tokenizer:    TokenizerCapabilities
      parquet:      ParquetCapabilities
      findings:     tuple[ProbeFinding, ...]
      elapsed_ms:   float
      probe_hidden_size: int            # what hidden the dry-forward used
      dry_forward_ok: bool              # forward of size H@batch=1,seq=8

      @property
      def is_clean(self) -> bool:
          return all(f.kind != "unsatisfied" for f in self.findings)

      @property
      def blocking(self) -> tuple[ProbeFinding, ...]:
          return tuple(f for f in self.findings if f.kind == "unsatisfied")

  def contract_probe(
      build_spec: ModelBuildSpec,
      tokenizer_source: str | Path,
      parquet_path: Path,
      *,
      probe_hidden_size: int = 64,
      sample_rows: int = 256,
  ) -> ContractProbeReport: ...

### 3.4 cppmega_v4/probe/alternatives.py — генератор альтернатив

Каждый тип unsatisfied requirement имеет свой генератор. Чистые
функции `(requirement, current_spec, capabilities) -> tuple[Alternative, ...]`.
Никаких side effects, никакого LLM.

Примеры:
  - unmet `fim_prefix_id` from tokenizer →
      Alternative(swap_tokenizer, target="...", suggest a tokenizer
                  from registry that has FIM ids), cost=medium
      OR
      Alternative(swap_loss, target="loss",
                  diff={LossKind: CROSS_ENTROPY}), cost=low
  - unmet `call_edges` from parquet →
      Alternative(add_column, target="parquet", suggest enrichment
                  pipeline scripts/nanochat_data/...), cost=high
      OR
      Alternative(drop_brick, target="brick:engram_3"), cost=medium
  - unmet `labels_k_shifted` (MTP) →
      Alternative(swap_loss, diff={LossKind: CROSS_ENTROPY}, cost=low)

### 3.5 cppmega_v4/probe/dry_forward.py — последний gate

Если все requirements satisfied — запускаем минимальный forward на
`probe_hidden_size=64, batch=1, seq=8` с **синтетическими input_ids**
(никакой реальной parquet-загрузки). Цель — поймать residual mismatch
который static анализ пропустил (например, brick кладёт hardcoded
`assert seq_len % 8 == 0`).

  def dry_forward(
      build_spec: ModelBuildSpec,
      hidden_size: int = 64,
      seq_len: int = 8,
      batch: int = 1,
  ) -> Literal["ok", "shape_mismatch", "exception"]: ...

---
4. Стадии (mapping на bd-тикеты cppmega-mlx-728.1/.2/.3)

Stage A — capabilities introspectors (`cppmega-mlx-728.1`)
  Файлы: probe/capabilities.py
  Тесты: tests/v4/test_probe_stage_a.py
    - introspect_tokenizer на vendored nanochat tokenizer → 48 special_ids
    - introspect_parquet на минимальной фикстуре в tests/fixtures/
    - golden-output check schema_columns / non_null_ratio
    - performance gate < 200ms на real-shard tokenizer + 256-row sample
  Бюджет: 1 день

Stage B — requirements + probe + alternatives (`cppmega-mlx-728.2`)
  Файлы: probe/requirements.py, probe/probe.py, probe/alternatives.py,
         probe/dry_forward.py
  Тесты: tests/v4/test_probe_stage_b.py
    - contract_probe на 12 пресетов × 4 loss-kinds = 48 cases
    - per-loss-kind: правильный set of unsatisfied при заведомо-плохом
      tokenizer (vocab=256 no-FIM) + parquet (token_ids only)
    - per-alternative: каждая action возвращает валидный JSON-Patch
      который verify_build_spec принимает после apply
    - dry_forward на 25 brick-kinds @ hidden=64 = 25 cases
    - probe sub-2s gate на самой большой preset (qwen3_235b_a22b @ h=64)
  Бюджет: 2-3 дня

Stage C — GUI panel hand-off (`cppmega-mlx-728.3`)
  Этот эпик НЕ строит GUI — он строит **stable API surface для GUI**:
    - ContractProbeReport must be JSON-serialisable (asdict → json.dumps)
    - JSON schema published в docs/contract_probe_schema.json для F-A
      JSON-RPC contract
    - cppmega-run subcommand: `cppmega-run probe --spec spec.json
      --tokenizer ... --parquet ... --json` → emits report to stdout
  Тесты: tests/v4/test_probe_stage_c.py
    - serialise → deserialise round-trip identity
    - JSON schema validation на образцах report
    - CLI subcommand smoke
  Бюджет: 1 день

Итог по эпику: 4-5 дней effort, ~1500-2000 LoC + ~50 тестов.

---
5. API entry point + интеграция

Production-facing:

  from cppmega_v4.probe import contract_probe, ContractProbeReport

  report = contract_probe(
      build_spec=my_spec,
      tokenizer_source="/path/to/tokenizer.json",
      parquet_path="/data/clang_semantic_4k_v10/shard_00.parquet",
  )
  if not report.is_clean:
      for finding in report.blocking:
          print(finding.message)
          for alt in finding.alternatives:
              print(f"  - [{alt.cost}] {alt.action}: {alt.reason}")
      raise RuntimeError("contract probe failed; resolve before training")

GUI-facing (Stage C):
  - frontend sends `probe.run` JSON-RPC с спекой
  - backend возвращает ContractProbeReport JSON
  - frontend рендерит чипы red/yellow/green per finding
  - user кликает "accept alternative #2" → frontend применяет
    `alt.diff` к спеке локально, шлёт `probe.run` ещё раз — пока
    `report.is_clean` не станет true

CLI-facing (Stage C):
  $ cppmega-run probe --spec model.json --tokenizer tokenizer.json \
      --parquet data/shard_00.parquet --format json
  → exits 0 if clean, 1 if blocking findings; report on stdout

---
6. Бюджет + риски

Бюджет: 4-5 дней wall-clock на implementation (3 stages). Полный
эпик `cppmega-mlx-728` укладывается в одну рабочую неделю одного
backend-инженера.

**Главный риск**: BRICK_REQUIREMENTS table становится rotten — добавили
новый brick, забыли вписать requirements, probe пропускает реальный
crash. Mitigation: nyquist-style coverage test в Stage B —
parametrized над всеми BLOCK_BUILDERS keys, требует наличия записи в
BRICK_REQUIREMENTS (может быть пустой tuple, но запись должна быть).

**Второй риск**: tokenizer introspection дорогая на huge BPE (Llama 3
128k vocab). Mitigation: lazy-fetch special_ids only, cache snapshot
по mtime+path key.

**Третий риск**: parquet introspection читает целый файл если мы не
ограничим. Mitigation: `sample_rows=256` default, использовать
pyarrow `read_row_group(0)` для schema + первой партии.

**Четвёртый риск**: alternatives genrate noise (10+ предложений на одно
unmet). Mitigation: жёсткий cap `max_alternatives=3` per finding,
deterministic sort by cost+specificity.

**Пятый риск**: dry_forward на синтетике пропускает real-data corner-
cases (пользователь думает что probe = full validation). Mitigation:
явный disclaimer в report.dry_forward_ok docstring + GUI tooltip;
report.dry_forward_ok != .is_clean, два разных гейта.

---
7. Что вне scope (явно)

  - **auto-apply** альтернативы без согласия пользователя — НЕ делаем.
    Probe только ПРЕДЛАГАЕТ, человек ВЫБИРАЕТ, GUI/CLI ЭМИТИТ diff.
  - **performance profiling** — probe не меряет throughput; для этого
    есть отдельный bench-уровень.
  - **multi-shard parquet aggregation** — probe смотрит один shard,
    предполагая representative; multi-shard разнятие — отдельный
    data-quality эпик.
  - **probe-time training** — никаких real gradients, никакого
    optimizer.step(); только forward.
  - **LLM-generated alternatives** — все альтернативы из статической
    таблицы; никаких model calls в probe-pipeline.

---
8. Связь с другими спеками

  - **VBSpec**: probe потребляет `ResolvedBrickGraph` (для нахождения
    bricks по kind+name); не модифицирует.
  - **MBSpec**: probe потребляет `ModelBuildSpec` (LossSpec даёт
    LOSS_REQUIREMENTS lookup); JSON-Patch diff target = ModelBuildSpec.
  - **PSpec**: probe **независим** от sharding — capability check
    одинаков на single-device и distributed; orthogonal axis.
  - **Auto-Fusion**: probe **независим** от fusion plan — capabilities
    проверяются на не-fused графе.
  - **VBGui (cppmega-mlx-o0k)**: Stage F-C sidebar получает probe report
    как один из 5 тэбов; F-A JSON-RPC contract включает `probe.run`.
  - **Gallery Coverage (cppmega-mlx-1t0, ✅ done)**: probe запускается
    на любой preset из 57; gallery tests становятся стронг-form probe
    тестов после Stage B.
