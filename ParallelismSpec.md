План: ParallelismSpec Layer (DeviceTopology + Sharding + Gotchas)

Цель: визуальный конструктор должен предсказывать "влезет/не влезет на
устройство X с топологией Y" с учётом FSDP/TP/EP/PP/ZeRO нюансов + всех
тех граблей которые мы накопили в ../cppmega и ../nanochat:

  - FSDP2 + whole-model torch.compile = flat loss (gradients не sync-аются)
  - Megatron + whole-model compile = NaN step 1 (hooks reorder, param.grad=None)
  - FP8 forward с bf16/fp32 backward — duplicate grad copies
  - master-fp32 weights дублируют optimizer state поверх bf16 compute
  - Megatron RowParallelLinear материализует full tensor на AllReduce boundary
  - ZeRO-3 peak memory = unsharded params в forward all-gather
  - PP comm-stream separation patch сломан в torch.distributed.pipelining
  - XLA TPU 4GB tensor limit на MoE expert fusion
  - Optimizer state duplication под TP + master weights
  - SP all-reduce overhead на norms/qk-norms без overlap

Источники практики: ../cppmega (Megatron EP=4/8 production) и
../nanochat (FSDP2 + Megatron TP + SPMD multi-stack, /memory_estimator.py
1265 строк с формулами).

Наш target — MLX/Apple Silicon, runtime распределёнки у нас нет. Этот
слой — PLANNER/SIZER который моделирует распределённую память для GUI
**до** запуска в reality на H100/TPU/etc. Лежит над cppmega_v4.spec
(MemoryReport) и cppmega_v4.buildspec (ModelBuildSpec).

---
1. Что уже есть (карта существующего)

cppmega_v4/spec/memory_report.py (Stage D VBSpec, шипнуто)
  Per-brick + per-region byte accounting. fusion-aware activations.
  **Single-device** — не знает про TP/EP/FSDP/replication.

cppmega_v4/buildspec/ (5 стадий MBSpec, шипнуто)
  Loss/Optim/Rewriters/build_model. Знает что есть MoE, MTP heads,
  IFIM/MHC. **Не знает** на скольких устройствах будем тренировать.

cppmega_v4/architectures/ — 12 presets
  Per-architecture topology suggestions появятся в Stage D ниже.

В ../nanochat/memory_estimator.py — готовые формулы:
  - params_gb с TP/EP sharding factor
  - gradients_gb (eager full vs SPMD reduced)
  - optimizer_gb (Muon 2 bytes vs AdamW 8 bytes per param)
  - activations_gb (Megatron 34*N*L*bpe формула + checkpoint variant)
  - moe_routing_gb (route + dispatch + EP all-to-all buffers)
  - feature_activations_gb (MTP/IFIM/MHC/Engram heads)
  - overhead_gb (CUDA baseline + XLA compiler temps)
  - LLO 4GB warning for TPU MoE

---
2. Что НЕ хватает (gap-analysis)

┌────────────────────────────────────────────┬──────────────────────────────┐
│              Что нужно GUI                  │             Сейчас            │
├────────────────────────────────────────────┼──────────────────────────────┤
│ Описать device topology (GPU/TPU/mesh)     │ ✗                            │
├────────────────────────────────────────────┼──────────────────────────────┤
│ Описать sharding strategy (DP/FSDP/TP/EP)  │ ✗                            │
├────────────────────────────────────────────┼──────────────────────────────┤
│ Посчитать память per-rank с TP/EP factors  │ ✗ (single-device only)       │
├────────────────────────────────────────────┼──────────────────────────────┤
│ Учесть FP8/bf16/fp32 mix duplication       │ ✗                            │
├────────────────────────────────────────────┼──────────────────────────────┤
│ Учесть FSDP all-gather peak                │ ✗                            │
├────────────────────────────────────────────┼──────────────────────────────┤
│ Учесть Megatron AllReduce materialisation  │ ✗                            │
├────────────────────────────────────────────┼──────────────────────────────┤
│ Учесть master-fp32 weights overhead        │ ✗                            │
├────────────────────────────────────────────┼──────────────────────────────┤
│ Подсказать sharding strategy для модели    │ ✗                            │
├────────────────────────────────────────────┼──────────────────────────────┤
│ Surface known gotcha (compile+FSDP/Megatron│ ✗                            │
│ pain, PP comm-stream, XLA 4GB limit)       │                              │
└────────────────────────────────────────────┴──────────────────────────────┘

---
3. Архитектура (новый пакет cppmega_v4/parallelism/)

3.1 cppmega_v4/parallelism/topology.py — declarative device + mesh

  class DeviceKind(str, Enum):
      H100_80GB = "h100_80gb"        # NVIDIA H100 SXM5 80GB HBM3
      H200_141GB = "h200_141gb"      # NVIDIA H200 141GB HBM3e
      A100_40GB = "a100_40gb"
      A100_80GB = "a100_80gb"
      B100_80GB = "b100_80gb"
      GB10 = "gb10"                   # Apple/NVIDIA hybrid (mac quarter)
      TPU_V5P = "tpu_v5p"
      TPU_V6E = "tpu_v6e"             # 32GB HBM
      M3_ULTRA = "m3_ultra"           # Apple Silicon (~512GB unified)

  @dataclass(frozen=True)
  class DeviceSpec:
      kind: DeviceKind
      hbm_bytes: int                  # absolute usable HBM
      interconnect: str               # "nvlink" | "infiniband" | "ethernet" | "uci"
      bandwidth_gbps: float           # rough device-to-device BW (TP overhead)

  @dataclass(frozen=True)
  class DeviceTopology:
      devices: tuple[DeviceSpec, ...]
      mesh_axes: dict[str, int]       # {"dp": 4, "tp": 2, "ep": 4}
      # Topology constraints: product(mesh_axes.values()) == len(devices)

  Built-ins:
  def h100_8x() -> DeviceTopology    # 8x H100:80GB on one node, NVLink
  def h200_8x() -> DeviceTopology    # 8x H200:141GB
  def gb10_quarter() -> DeviceTopology  # 1x GB10 (Mac quarter)
  def tpu_v6e_8() -> DeviceTopology   # 8x TPU v6e
  def m3_ultra_solo() -> DeviceTopology  # 1x Mac Studio M3 Ultra

3.2 cppmega_v4/parallelism/sharding_spec.py — declarative sharding strategy

  class ParallelismKind(str, Enum):
      DP        = "dp"                 # replicate weights, shard data
      FSDP1     = "fsdp1"               # full sharding (peak = unsharded)
      FSDP2     = "fsdp2"               # ZeRO-3 modern API
      ZERO1     = "zero1"               # optimizer state only
      ZERO2     = "zero2"               # optim state + grads
      TP        = "tp"                 # tensor parallel intra-layer
      SP        = "sp"                 # sequence parallel (with TP)
      EP        = "ep"                 # expert parallel for MoE
      PP        = "pp"                 # pipeline parallel
      PP_VPP    = "pp_vpp"             # virtual pipeline parallel

  @dataclass(frozen=True)
  class AxisAssignment:
      axis_name: str                   # "dp" | "tp" | "ep" | "pp"
      kind: ParallelismKind
      degree: int                      # parallelism degree
      degree_dim: str | None = None    # which mesh axis (for 3D parallelism)

  @dataclass(frozen=True)
  class ShardingSpec:
      topology: DeviceTopology
      axis_assignments: tuple[AxisAssignment, ...]
      # Per-brick override: which axes to apply
      brick_axes: dict[str, tuple[str, ...]] = ()  # {"attn_0": ("tp", "fsdp")}
      # Knobs
      master_weights_fp32: bool = False    # fp32 master weights duplicate
      grad_reduce_dtype: str = "bf16"      # "bf16" | "fp32"
      compile_mode: str = "regional"       # "off" | "regional" | "whole_model"
      fp8_enabled: bool = False
      activation_checkpointing: str = "full"  # "off" | "full" | "selective"

3.3 cppmega_v4/parallelism/distributed_memory.py — per-rank accounting

  @dataclass(frozen=True)
  class PerRankMemory:
      rank_idx: int
      device_idx: int
      weights_bytes: int               # post-sharding
      grads_bytes: int                 # may differ from weights (bf16 grad of fp8 fwd)
      optimizer_state_bytes: int
      master_weights_bytes: int        # 0 unless fp32 master enabled
      activations_bytes: int           # forward peak + checkpoint variant
      fsdp_allgather_peak_bytes: int   # extra during FSDP all-gather
      kv_cache_bytes: int              # inference only
      moe_routing_buffers_bytes: int
      collective_workspace_bytes: int  # AllReduce / AllGather / ReduceScatter
      framework_overhead_bytes: int    # ~2GB CUDA baseline / 6.4GB XLA
      total_bytes: int

      def fits_on_device(self, device: DeviceSpec, headroom: float = 0.9) -> bool

  @dataclass(frozen=True)
  class DistributedMemoryReport:
      sharding: ShardingSpec
      per_rank: tuple[PerRankMemory, ...]     # one per rank
      duplication_bytes: int                  # FP8 fwd + bf16 grad overhead
      master_weights_overhead_bytes: int
      kernel_boundary_materialisation_bytes: int  # Megatron RowParallel
      worst_rank_idx: int
      fits_on_topology: bool
      bottleneck_diagnostics: tuple[str, ...]

  def estimate_distributed_memory(
      build_spec: ModelBuildSpec,
      sharding: ShardingSpec,
      *,
      training: bool = True,
  ) -> DistributedMemoryReport

  Реализация формул:
    - FSDP2: peak weights = unsharded (all-gather), optimizer = sharded
    - TP: divide all per-layer params by tp_degree, replicate norms/embeds
    - EP: MoE experts divided by ep_degree, router replicated
    - FP8 + bf16 grad: weights in fp8 (1 byte/param), grads materialise in
      bf16 (2 bytes/param) → 3× memory vs pure bf16
    - master_weights_fp32: +4 bytes/param duplicate (cf. nanochat #5)
    - Megatron RowParallel AllReduce: +full activation tensor at boundary
    - SP + small replicated params: separate AllReduce buffer (no overlap)

3.4 cppmega_v4/parallelism/gotcha_checker.py — table-driven известные грабли

  @dataclass(frozen=True)
  class Gotcha:
      gotcha_id: str           # "fsdp2_whole_compile" | "megatron_nan_step1" | ...
      severity: str            # "ERROR" | "WARNING" | "INFO"
      condition: Callable[[ShardingSpec, ModelBuildSpec], bool]
      message: str
      reference: str           # nanochat/cppmega file:line where lesson learned

  GOTCHAS: tuple[Gotcha, ...] = (
      Gotcha("fsdp2_whole_compile", "ERROR",
             lambda s, b: ParallelismKind.FSDP2 in axis_kinds(s)
                          and s.compile_mode == "whole_model",
             "FSDP2 + whole-model torch.compile produces flat loss "
             "(gradients never sync; PyTorch #144376). Use compile_mode='regional'.",
             "nanochat/CLAUDE.md:fsdp2_compile_section"),
      Gotcha("megatron_nan_step1", "ERROR",
             lambda s, b: ParallelismKind.TP in axis_kinds(s)
                          and s.compile_mode == "whole_model",
             "Megatron TP + whole-model compile = NaN step 1 (hooks "
             "reorder, param.grad=None triggers recompile). "
             "Use compile_mode='regional'.",
             "nanochat/scripts/base_train.py:regional_compile_flag"),
      Gotcha("fp8_grad_duplication", "WARNING", ...),
      Gotcha("master_fp32_duplication", "WARNING", ...),
      Gotcha("ep_more_than_16_experts_xla", "WARNING", ...),  # 4GB tensor limit
      Gotcha("pp_comm_stream_broken", "INFO", ...),
      Gotcha("megatron_row_parallel_boundary", "INFO", ...),
      Gotcha("fsdp_allgather_peak_unsharded", "INFO", ...),
      ...
  )

  def check_gotchas(sharding, build_spec) -> tuple[Gotcha, ...]

3.5 cppmega_v4/parallelism/auto_shard.py — propose sharding strategy

  @dataclass(frozen=True)
  class ShardingProposal:
      strategy_name: str           # "h100_8x_fsdp2_ep4_tp2"
      sharding: ShardingSpec
      reason: str                  # why this picked
      estimated_per_rank_bytes: int
      fits: bool

  def suggest_sharding(
      build_spec: ModelBuildSpec,
      topology: DeviceTopology,
  ) -> list[ShardingProposal]
  """Returns 3-5 proposals ranked by fit + throughput estimate.

  Heuristics (informed by nanochat practice):
    - MoE present → EP first (degree = sqrt(num_experts) clamped to mesh)
    - >70B params on 80GB device → FSDP2 mandatory
    - Attention layers > 4096 H → TP=2 reduces activation memory
    - Small base model (<10B) → DP-only, no FSDP
    - Always compile_mode="regional" (avoid known footguns)
    - master_weights_fp32=False unless explicitly requested (Muon path)
  """

3.6 cppmega_v4/parallelism/api.py — public + GUI hooks

  def verify_distributed_plan(
      build_spec: ModelBuildSpec,
      sharding: ShardingSpec,
  ) -> tuple[DistributedMemoryReport, tuple[Gotcha, ...]]
      """One-shot: gotcha check + memory estimate. GUI consumes both."""

---
4. Какие topologies + sharding нужны для текущих presets

┌──────────────────────┬──────────────────────────────────────────┐
│       Preset         │           Recommended topology           │
├──────────────────────┼──────────────────────────────────────────┤
│ qwen3_next (small)   │ gb10_quarter / m3_ultra_solo (no shard) │
│ qwen3_next (full)    │ h100_8x: FSDP2 + EP=4 + TP=1            │
│ kimi_k2 (1T param)   │ h200_8x: FSDP2 + EP=8 + TP=2            │
│ deepseek_v3          │ h100_8x: FSDP2 + EP=4                   │
│ gemma4 (small)       │ tpu_v6e_8: SPMD + EP=2                  │
│ ling26               │ h100_8x: FSDP2 + EP=4                   │
│ nemotron3            │ h100_8x: FSDP2 + TP=1                   │
└──────────────────────┴──────────────────────────────────────────┘

---
5. Что GUI получает

  - Topology selector: dropdown с пресетами (h100_8x, tpu_v6e_8, m3_ultra_solo)
  - Sharding canvas: визуальная mesh с axes labels (dp/tp/ep/pp)
  - Per-rank memory bar: "Rank 0: 67.2 / 80 GB (worst case)"
  - Gotcha pane: красные/жёлтые/синие чипы со ссылкой на nanochat docs
  - Sharding suggestions: 3-5 strategies + "fit / no fit" badge
  - Memory duplication callouts:
    "FP8 fwd + bf16 grad: +9.3 GB duplication overhead"
    "Master fp32 weights: +18.6 GB duplication"
    "Megatron RowParallel boundary: +12 GB peak materialisation"

---
6. План реализации поэтапно

Этап A — Topology + ShardingSpec data layer (1 заход)
  - cppmega_v4/parallelism/topology.py — DeviceKind, DeviceSpec,
    DeviceTopology + built-in factories (h100_8x, h200_8x, gb10_quarter,
    tpu_v6e_8, m3_ultra_solo)
  - cppmega_v4/parallelism/sharding_spec.py — ParallelismKind,
    AxisAssignment, ShardingSpec
  - Validation: mesh product == num devices; axis degree ≥ 1; compile_mode
    in {"off", "regional", "whole_model"}
  - Тесты: builtin factories well-formed; rejection on bad mesh;
    parametrized device coverage

Этап B — distributed_memory.py (1 заход)
  - cppmega_v4/parallelism/distributed_memory.py — PerRankMemory,
    DistributedMemoryReport, estimate_distributed_memory
  - Порт формул из nanochat/memory_estimator.py
  - FP8/bf16/fp32 duplication accounting
  - FSDP all-gather peak (= unsharded)
  - Megatron RowParallel kernel-boundary materialisation
  - SP all-reduce buffers
  - Тесты: бэйзлайн без шардинга = single-device MemoryReport;
    FSDP2 reduces optim state by dp_degree; TP reduces per-layer params;
    EP reduces expert params; FP8+bf16 grad accounting >3× pure-bf16

Этап C — gotcha_checker.py (1 заход)
  - cppmega_v4/parallelism/gotcha_checker.py — Gotcha + GOTCHAS table
    (≥10 entries, each with reference to nanochat/cppmega lesson)
  - check_gotchas(sharding, build_spec) → tuple[Gotcha]
  - Тесты: каждый gotcha срабатывает в нужной комбинации; не срабатывает
    в clean spec; severity assignments

Этап D — auto_shard.py (1 заход)
  - cppmega_v4/parallelism/auto_shard.py — suggest_sharding heuristics
  - Тесты: small model на gb10_quarter → no-shard proposed;
    MoE → EP-first; >70B на 80GB → FSDP2 mandatory;
    always compile_mode="regional"

Этап E — public API + GUI integration (1 заход)
  - cppmega_v4/parallelism/api.py — verify_distributed_plan
  - Perf-критерий: per-preset verify_distributed_plan <100ms
  - System: full GUI workflow — pick preset → pick topology → suggest →
    accept → verify → show report + gotcha chips
  - End-to-end integration with cppmega_v4.spec (single-device MemoryReport)
    and cppmega_v4.buildspec (ModelBuildSpec)

---
7. Бюджет и риски

Бюджет: 5 этапов × ~2 часа = ~10 часов. A→B→D последовательны;
C параллелится с B-D; E финальный.

Главный риск: формулы memory accounting приблизительны (±10-15%).
Mitigation: tag-on calibration слой в будущем — empirically замеряем
на nanochat runtime, корректируем коэффициенты. Для GUI "OOM warning"
точности ±15% хватает.

Второй риск: GOTCHAS table может устареть. Mitigation: каждая запись
имеет ссылку на конкретный файл/коммит в nanochat — когда там фиксят,
мы видим diff и обновляем условие.

Третий риск: 3D parallelism (TP+EP+FSDP composition) — самый сложный
случай для accounting. Mitigation: ограничиться валидным product=mesh
constraint; не обещать идеальную точность для >3 одновременных axes
в Stage B; добавить calibration в будущем.

Четвёртый риск: MLX runtime у нас нет — это PLANNING layer. Не выдавать
себя за runtime. Документировать в каждом docstring "this is a sizing
estimate; actual training happens on CUDA/TPU via nanochat".
