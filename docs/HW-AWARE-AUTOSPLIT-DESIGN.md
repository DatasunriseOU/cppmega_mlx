# Hardware-Aware Automatic Kernel-Splitting for the Path-C Direct-Fusion Chain

Status: DESIGN (implementation-ready)
Owner: Path-C fusion / kernel codegen
Scope: `cppmega_mlx/runtime/path_c_fusion_schedules.py` (+ `path_c_fusion.py`), with three new modules
`cppmega_mlx/runtime/path_c_device_caps.py`, `cppmega_mlx/runtime/path_c_segment_estimator.py`,
`cppmega_mlx/runtime/path_c_device_presets.py`, and a calibration harness
`scripts/calibrate_path_c_device.py`.
Targets validated: Apple M4 Max (Metal, `applegpu_g16s`) and NVIDIA GB10 (CUDA `sm_121`, gb10).

---

## 0. Problem statement and the one premise correction

The Path-C direct-fusion chain currently splits fused kernels using **hardcoded constants** that
encode five distinct hardware/compiler facts:

| Constant | Encodes |
|---|---|
| `METAL_FORWARD_MAX_SEGMENT_NODES = 2` | Metal `MTLCompilerService` / `newComputePipelineState` shader-size crash (Cause A) |
| `METAL_BACKWARD_MAX_SEGMENT_NODES = 1` | macOS GPU watchdog per-command-buffer time limit (`kIOGPUCommandBufferCallbackErrorTimeout`) |
| `_METAL_SHARED_SCRATCH_TRIGGER_BYTES = 0x7000` (28 KiB) | Apple GPU 32 KiB threadgroup-memory cap (with logical→physical inflation margin) |
| `_METAL_SHARED_SCRATCH_DEMOTE_TARGET_BYTES = 0x2000` (8 KiB) | same cap, post-demote target |
| `_CUDA_SHARED_SCRATCH_BUDGET_BYTES = 0xC000` (48 KiB) | CUDA static per-block shared cap |
| `DESCRIPTOR_PORTABLE_KERNEL_BUFFER_LIMIT = 31` | Metal buffer-argument-table ABI limit |
| `DESCRIPTOR_DEFAULT_MAX_ROWS_PER_LAUNCH = 64` / `DESCRIPTOR_ROWS_PER_KERNEL_LAUNCH = 8` | watchdog-window row-windowing |
| `_TIME_CHUNKED_RECURRENT_BACKWARD_OPS` / `_ROW_CHUNKED_INDEPENDENT_BACKWARD_OPS` | recurrent-vs-independent op structure |

**Premise correction (stated up front so the design is not justified on a false claim):** on M4 Max
the measured `maxThreadgroupMemoryLength` is **exactly 32768** — identical to the value the code already
assumes. Querying the threadgroup cap does **not** change correctness on this hardware. The value-add of
this subsystem is three-fold and only three-fold:

1. **Forward portability** — a future Apple/NVIDIA GPU with a *different* cap is handled without a code edit.
2. **Fail-loud (RULE #1)** — replace the silent over-budget `LOG(WARNING)+continue` in `codegen_metal.cc:1248`
   and the silent `except: pass` fallbacks in the CUDA path with device-grounded pre-checks that RAISE.
3. **Decoupling magic numbers from one calibration scale** — the `2/1/64/8/28K/8K` constants were all
   hand-tuned at one workload (`local_gb10_quarter`, depth=13 hidden=3584 max_seq=4096). The estimator
   makes the split decision scale with the *actual* planned region shape.

Goal (user directive): make the splitting **AUTOMATIC and HARDWARE-AWARE** — query what we can, estimate
what we can, derive the splits, and **fail loud when nothing fits**. It MUST reproduce the current
hand-tuned splits on Metal (M4 Max) AND keep CUDA (gb10) monolithic (no watchdog there).

**The non-queryable-limits strategy is DECIDED (a three-tier scheme, §4) and is not re-opened here.**

---

## 1. Module 1 — Device-capability probe (`path_c_device_caps.py`)

A new module returning one cached, immutable `DeviceCaps` record per process. It is the single seam
every limit-comparison plugs into, and it **extends** the existing binary `_path_c_default_target()`
(`path_c_fusion.py:23`) from a metal/cuda *string* into a capability *record* — the string remains the
backend selector; the record carries the numbers. Each field is filled by the appropriate tier of §4
(TIER 1 live-query, TIER 2 preset, TIER 3 calibration cache) and carries provenance for audit.

### 1.1 The record

```python
@dataclass(frozen=True)
class DeviceCaps:
    backend: str                       # "metal" | "cuda"
    device_name: str                   # "Apple M4 Max" | "NVIDIA GB10"
    architecture: str                  # "applegpu_g16s" | "sm_121"
    os_driver: str                     # macOS build / CUDA driver version (cache key + invalidation)
    # --- TIER 1: live-queried hard limits the split must respect ---
    threadgroup_mem_bytes: int         # Metal maxThreadgroupMemoryLength | CUDA optin shared cap
    static_shared_mem_bytes: int       # CUDA shared_memory_per_block (no opt-in); == threadgroup on Metal
    max_threads_per_block: int
    warp_size: int
    buffer_arg_limit: int              # Metal 31 (family const) | CUDA effectively-unbounded sentinel
    # --- TIER 2/3: preset-or-calibrated, non-queryable ---
    has_command_buffer_watchdog: bool  # True on Metal, False on CUDA (empirically verified)
    watchdog_window_s: float | None    # preset or calibrated; None on CUDA
    logical_to_physical_shared_margin: float   # ~3.7x on Apple GPU (preset/calibrated)
    msl_pipeline_state_ceiling_bytes: int | None  # Metal forward-size cap; None on CUDA
    per_op_time_per_row_s: dict[str, float]    # per-op GPU-time-per-row coeffs (preset/calibrated)
    effective_flop_s: float            # roofline throughput (preset/calibrated)
    effective_bytes_s: float           # roofline throughput (preset/calibrated)
    # provenance per field: "queried" | "family-const" | "preset" | "calibrated"
    source: dict[str, str]
```

### 1.2 Metal probe (TIER 1 live-query) — exact queried fields

MLX `mx.device_info()` is **insufficient**: verified on M4 Max it returns ONLY
`{device_name, max_recommended_working_set_size, memory_size, architecture, max_buffer_length, resource_limit}`.
It does NOT expose `maxThreadgroupMemoryLength` nor the 31 buffer-arg limit. (`max_buffer_length` is a
single-buffer *size* cap, 80.64 GiB — NOT the arg-count limit; do not confuse them.) TVM's
`tvm.metal(0).max_shared_memory_per_block` is a **no-op** (`metal_device_api.mm:case kMaxSharedMemoryPerBlock: return;`)
and returns `None`. So the threadgroup cap requires a native probe.

**Probe path (no new dependency):** `ctypes` against `libobjc` + the Metal framework (both present on
every macOS):

```python
def _probe_metal_live() -> dict:
    import ctypes, ctypes.util
    objc = ctypes.CDLL(ctypes.util.find_library("objc"))
    metal = ctypes.CDLL(ctypes.util.find_library("Metal"))  # registers MTL symbols
    objc.objc_msgSend.restype = ctypes.c_void_p
    objc.objc_msgSend.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    metal.MTLCreateSystemDefaultDevice.restype = ctypes.c_void_p
    dev = metal.MTLCreateSystemDefaultDevice()
    if not dev:
        raise RuntimeError("path_c_device_caps: MTLCreateSystemDefaultDevice() returned nil "
                           "on a Metal target — cannot query maxThreadgroupMemoryLength")
    def sel(name): return objc.sel_registerName(name.encode())
    def msg_ulong(d, name):
        f = objc.objc_msgSend; f.restype = ctypes.c_ulong
        return int(f(d, sel(name)))
    def supports_family(d, fam_id):  # MTLGPUFamilyApple9 == 1009
        f = objc.objc_msgSend; f.restype = ctypes.c_bool
        f.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_longlong]
        return bool(f(d, sel("supportsFamily:"), fam_id))
    return {
        "threadgroup_mem_bytes": msg_ulong(dev, "maxThreadgroupMemoryLength"),  # 32768 on M4 Max
        "supports_apple9": supports_family(dev, 1009),
    }
```

`max_threads_per_block` (1024) and `warp_size` (32, overriding the stale TVM Target default of 16) come
from `tvm.metal(0).max_threads_per_block` / `tvm.metal(0).warp_size` (the TVM device-API path that *does*
work). The 31 buffer-arg limit is **NOT a queryable scalar** — it is a fixed MSL/feature-set ABI limit,
keyed on GPU family (TIER 2 preset, §4.2):

```python
_METAL_FAMILY_BUFFER_ARG_LIMIT = {
    "applegpu_g16s": 31,  # M4 family (Apple9 / Metal3), per Metal-Feature-Set-Tables.pdf
}
```

`DESCRIPTOR_PORTABLE_KERNEL_BUFFER_LIMIT = 31` is retained as a **portable floor**: if the family table
or a live probe ever yields a *lower* cap, use the lower and RAISE if a segment cannot fit.

### 1.3 CUDA probe (TIER 1 live-query) — exact queried fields

Use `torch.cuda.get_device_properties(0)` (no extra deps) and/or `cuda.bindings.runtime` with the
**symbolic** `cudaDeviceAttr` enum (never raw integer codes — code 16 ≠ `KernelExecTimeout` in CUDA 13;
the correct enum is 17).

| `DeviceCaps` field | Source on CUDA | Observed (gb10) | provenance |
|---|---|---|---|
| `threadgroup_mem_bytes` | `props.shared_memory_per_block_optin` (`cudaDevAttrMaxSharedMemoryPerBlockOptin`, enum 97) | 101376 | `queried` |
| `static_shared_mem_bytes` | `props.shared_memory_per_block` (enum 8) | 49152 | `queried` |
| `max_threads_per_block` | `props.max_threads_per_block` | 1024 | `queried` |
| `warp_size` | 32 | 32 | `queried` |
| `device_name` | `props.name` | NVIDIA GB10 | `queried` |
| `architecture` | `"sm_%d%d" % torch.cuda.get_device_capability(0)` | sm_121 | `queried` |
| `buffer_arg_limit` | sentinel `1 << 30` (no 31-arg ABI wall on CUDA) | n/a | `family-const` |
| `has_command_buffer_watchdog` | **False** (empirically: 105.17 s single kernel completed) | False | `family-const` |
| `watchdog_window_s` | `None` | None | `family-const` |
| `msl_pipeline_state_ceiling_bytes` | `None` (ptxas fails loud, not an opaque XPC crash) | None | `family-const` |

**Watchdog on CUDA — do NOT query `cudaDevAttrKernelExecTimeout` as the gate.** It returns `1` on gb10
(integrated/display-attached SoC) yet a 105 s kernel survives. `has_command_buffer_watchdog` is hard-set
`False` for the CUDA backend; the attribute is queried *only to record in `source` for audit*, never to
gate. The CUDA shared-mem demote uses `budget = min(static_shared_mem_bytes, threadgroup_mem_bytes)`
exactly as today, but both values are now `queried`, not literals.

### 1.4 Caching and assembly

```python
@functools.lru_cache(maxsize=1)
def device_caps() -> DeviceCaps:
    backend = _path_c_default_target()            # existing seam, unchanged
    live  = _probe_metal_live() if backend == "metal" else _probe_cuda_live()  # TIER 1 (RAISE on fail)
    ident = _device_identity(backend, live)       # (architecture, device_name, os_driver)
    nonq  = _resolve_nonqueryable(ident)          # TIER 2 preset -> else TIER 3 calibration cache (§4)
    return _assemble(backend, live, nonq)
```

Probed once at startup, cached for the process. **No silent fallback**: if the active backend is metal
and the threadgroup probe fails, or active backend is cuda and `cudaDeviceGetAttribute(97)` fails, RAISE
with where+what (§6) — never substitute a guessed constant (this fixes
`path_c_fusion_schedules.py:10420-10425`).

---

## 2. Module 2 — Static per-segment resource estimator (`path_c_segment_estimator.py`)

All four estimates are computed **at plan time from data already present**, with zero runtime
measurement. The estimator is called at the **same greedy accept/reject point** where `parameter_count`
is checked today (`path_c_fusion_schedules.py:14697-14711`), so it adds **no new compilation cost**.

```python
@dataclass(frozen=True)
class SegmentEstimate:
    logical_shared_bytes: int      # summed T.alloc_shared
    physical_shared_bytes: int     # logical * caps.logical_to_physical_shared_margin
    buffer_arg_count: int          # _kernel_parameter_count_for_target
    est_gpu_time_s: float          # roofline FLOPs/bytes -> seconds
    is_recurrent: bool             # reverse-time scan vs per-row-independent
    msl_source_bytes: int          # len(generated source) — forward size proxy
    per_row_time_s: float          # est_gpu_time_s / S  (independent ops only)
```

### 2.1 Threadgroup-memory bytes (already-present machinery, reused)

The generated kernel text is attached to the prim_func **before** metallib compilation as
`prim_func._cppmega_path_c_generated_source` (built in `make_path_c_descriptor_schedule_template`,
attached at `schedules.py:2206`). Two existing passes already sum shared bytes deterministically. Reuse
exactly:

```python
shared_total = 0
for line in source.splitlines():
    m = _ALLOC_SHARED_LINE_RE.match(line)          # schedules.py:3862
    if m is None: continue
    shape = _coerce_shape(m.group("shape"))
    shared_total += _flattened_extent(shape) * _DTYPE_NBYTES[m.group("dtype")]  # :13599,:333
logical_shared_bytes = shared_total
physical_shared_bytes = math.ceil(shared_total * caps.logical_to_physical_shared_margin)
```

Compare `physical_shared_bytes` (NOT logical) against `caps.threadgroup_mem_bytes`. Using raw logical
bytes vs 32 KiB would be wrong — TileLang coalesces+pads the residual into one `buf_dyn_shmem` array
running a few× the logical (~29.5 KiB physical observed for an 8 KiB logical target on mamba3 bwd).

### 2.2 Buffer-argument count (already computed today — keep)

`_kernel_parameter_count_for_target(region, target)` (`schedules.py:14955`) already counts only params
present in `buffer_map` (excludes by-value scalars). This matches `codegen_metal.cc` emission exactly.
The estimator just calls it and records the result.

### 2.3 GPU-time estimate (the one new closed-form cost model)

`PathCModelShapeEnv` (`path_c_fusion.py:380`) carries `sequence_length (S)`, `hidden_size`, attention
head dims/topk, mamba/m2rnn dims — everything a roofline needs, resolved per region via
`_shape_env_for_region`. Combined with the per-op `op_name`
(`_path_c_descriptor_stage_node_op_name`, `schedules.py:610`):

```python
def est_gpu_time_s(op_name, env, caps) -> float:
    # Preferred: directly-calibrated per-op-per-row coefficient (TIER 2/3, §4).
    coeff = caps.per_op_time_per_row_s.get(op_name)
    if coeff is not None:
        return coeff * env.sequence_length
    # Fallback within the calibration domain: roofline from FLOPs/bytes.
    flops = _op_flops(op_name, env)
    bytes_ = _op_bytes(op_name, env)
    return max(flops / caps.effective_flop_s, bytes_ / caps.effective_bytes_s)
```

`_op_flops` / `_op_bytes` are closed-form per op-class (in-projection ≈ 2·S·hidden·in_proj_dim,
causal-conv ≈ S·conv_channels·kernel, state-recurrence ≈ S·heads·state_dim·head_dim, gate/out-proj ≈
2·S·hidden², attention ≈ S·topk·head_dim·heads), authored from the descriptor's `required_codegen_steps`
stage list. A segment's time is the sum over its ops. `per_op_time_per_row_s`, `effective_flop_s`, and
`effective_bytes_s` are the **only** throughput constants and come from the preset table or calibration
(§4) — never hardcoded inline.

### 2.4 Recurrent-vs-independent classification (structural — delete the frozensets)

Both `_TIME_CHUNKED_RECURRENT_BACKWARD_OPS` and `_ROW_CHUNKED_INDEPENDENT_BACKWARD_OPS` **duplicate
information already structurally present**:

```python
def is_recurrent(node, descriptor) -> bool:
    # reverse-time recurrence <=> the fwd descriptor has a '*_state_recurrence'
    # codegen step  (equivalently: a non-empty carry/replay buffer set)
    if any("_state_recurrence" in step for step in descriptor.required_codegen_steps):
        return True
    return bool(_row_phased_launcher_carry_buffers_for_nodes([node], ...))  # schedules.py:2320
```

This matches today's membership exactly (mamba3_mimo_bwd / m2rnn_bwd → recurrent → time-chunk;
attention_qkv_projection_bwd / sparse_mla_fp8_apply_bwd → independent → row-chunk). The two frozensets
are **deleted**; the launcher_chunks switch at `schedules.py:14678-14695` gates on this derived flag.

The per-op **override-to-1** for `rows_per_kernel_launch` on reverse-time-scan ops
(`schedules.py:614-618`) is a SEMANTIC constraint (one time-step per row) and is **preserved verbatim**,
distinct from the perf knob.

---

## 3. Module 3 — The auto-split planner (maps estimates → splits)

The planner is the existing `plan_path_c_direct_fusion_chain_for_region` greedy loop
(`schedules.py:14561-14738`). Three changes, all at the resolution seam and the accept/reject point.

### 3.1 Resolve caps from the device, not from constants

Replace the `_ResolveFromTarget` resolution (`schedules.py:14517-14550`) so the sentinel resolves to a
**derived cap** instead of `METAL_FORWARD_MAX_SEGMENT_NODES`/`METAL_BACKWARD_MAX_SEGMENT_NODES`:

```python
caps = device_caps()
forward_max_segment_nodes  = _RESOLVE if caps.msl_pipeline_state_ceiling_bytes is not None else None
backward_max_segment_nodes = _RESOLVE if caps.has_command_buffer_watchdog          else None
```

CUDA → both `None` → greedy monolithic fusion preserved exactly (no watchdog, no pipeline crash).
Metal → both active, but now driven by §3.3/§3.4 estimates rather than fixed op counts.

### 3.2 Add estimate predicates at the accept/reject point

At `schedules.py:14706` (right where `parameter_count > max_kernel_buffers` is checked), insert estimate
predicates against `device_caps()`, each comparing an estimate to a queried/preset/calibrated limit. The
generated source is already populated on the template prim_func, so the shared-byte and MSL-byte sums
are free:

```python
est = estimate_segment(candidate_region, direct_target, caps)   # Module 2

# (a) buffer-arg cap (existing, now device-keyed)
if est.buffer_arg_count > caps.buffer_arg_limit:
    first_failure = (f"{candidate_region.name}: {est.buffer_arg_count} buffer args "
                     f"> family limit {caps.buffer_arg_limit}"); break

# (b) threadgroup-mem cap — try the pool/demote pass first, then re-estimate
if est.physical_shared_bytes > caps.threadgroup_mem_bytes:
    pooled = _pool_oversized_shared_scratch_to_metal_workspace(source)   # Metal
    est = estimate_segment_from_source(pooled, ...)                       # CUDA: _demote_residual...
    if est.physical_shared_bytes > caps.threadgroup_mem_bytes:
        if end - start == 1:
            raise PathCSplitInfeasible(candidate_region, "threadgroup-mem",
                                       est.physical_shared_bytes, caps.threadgroup_mem_bytes)
        first_failure = "...shared over cap, shrink segment"; break

# (c) forward MSL-size cap (Metal-only, §3.3 + §4 TIER 2/3 ceiling)
if caps.msl_pipeline_state_ceiling_bytes is not None and \
   execution_phase == FORWARD and est.msl_source_bytes > caps.msl_pipeline_state_ceiling_bytes:
    if end - start == 1:
        raise PathCSplitInfeasible(candidate_region, "msl-pipeline-size",
                                   est.msl_source_bytes, caps.msl_pipeline_state_ceiling_bytes)
    first_failure = "...forward MSL over ceiling, shrink segment"; break

# (d) backward watchdog cap (Metal-only, §4)
if caps.has_command_buffer_watchdog and execution_phase == BACKWARD:
    budget = caps.watchdog_window_s * _WATCHDOG_SAFETY   # 0.5
    if est.est_gpu_time_s > budget:
        direct_target = _apply_chunking(direct_target, candidate_region, est, caps)
        if not _chunking_brings_under_budget(est, budget):
            if end - start == 1:
                raise PathCSplitInfeasible(candidate_region, "watchdog",
                                           est.est_gpu_time_s, budget)
            first_failure = "...backward over watchdog, shrink segment"; break
```

The existing `break`-with-`first_failure` flow (shrink the segment, try fewer ops) is preserved
verbatim; only the *reason* now references a device-grounded value.

### 3.3 Forward segment-node cap → MSL-size predicate

`METAL_FORWARD_MAX_SEGMENT_NODES = 2` is replaced by comparing `est.msl_source_bytes`
(`len(prim_func._cppmega_path_c_generated_source)`) against `caps.msl_pipeline_state_ceiling_bytes`
(TIER 2 preset / TIER 3 calibrated). Calibration band: ~116 KiB known-good, ~176 KiB known-crash →
ceiling set conservatively at ~140 KiB with margin. On `local_gb10_quarter` this reproduces the 2-op
split: the 4-op forward chain_3_7 (~176-199 KiB) exceeds the ceiling and splits into
chain_3_5/chain_5_7, each under it.

**Hybrid backstop (Cause A only):** raw MSL bytes is NOT a clean cross-phase monotone predictor
(116 KiB bwd compiled while 176 KiB fwd crashed; the crash correlates with op *content* — sparse top-k
FP8 attention — not bytes alone). So the calibrated ceiling is the **default** path, and the **TIER 3
calibration/retry** of §4.4 is the loud-logging backstop for never-before-seen op mixes.

### 3.4 Backward segment-node cap → watchdog-time predicate + derived row/time chunking

`METAL_BACKWARD_MAX_SEGMENT_NODES = 1` is replaced by §3.2(d). When a backward segment's
`est_gpu_time_s` exceeds `watchdog_window_s * SAFETY`:

- **recurrent** (`is_recurrent`): switch to `launcher_chunks` time-chunking;
  `time_chunk_count = ceil(est_gpu_time_s / (watchdog_window_s * SAFETY))`.
- **independent** (not `is_recurrent`): switch to `launcher_chunks` row-windowing with the **derived**
  `max_rows_per_launch` of §4.7. Replaces the fixed `64`. `rows_per_kernel_launch` keeps the semantic
  override-to-1 for reverse-time ops; otherwise derives from the same budget.

At `local_gb10_quarter` scale, with the preset/calibrated `per_op_time_per_row_s`/`watchdog_window_s`,
the derived `max_rows_per_launch` lands at 64 (reproducing the hand-tuned value — see §7).

---

## 4. Module 4 — Non-queryable-limits strategy: the THREE-TIER scheme (DECIDED)

Some limits are **not device-queryable** by any API: the macOS GPU command-buffer watchdog window
(`kIOGPUCommandBufferCallbackErrorTimeout`, ~5-6 s), the `MTLCompilerService`/`newComputePipelineState`
shader-size ceiling, the logical→physical threadgroup packing margin, and the per-op GPU-time-per-row
coefficients. The strategy for these is **DECIDED** (this section is the contract; it is not re-opened):
a **cuDNN-heuristic-then-autotune / FlashAttention-2-preset-then-autotune** three-tier scheme.

**The FA2 analogy:** FlashAttention reads SRAM size **live** (our TIER 1), ships **preset** block tiles
for known GPUs (our TIER 2), and **autotunes + caches** the best tile when the preset is absent or
underperforms (our TIER 3). We mirror all three: live-query the queryable caps, preset the non-queryable
limits for known architectures, and immediately one-shot auto-calibrate (and persist) when a preset is
missing or proven wrong in practice.

### 4.1 TIER 1 — LIVE-QUERY (every run, for directly-queryable limits)

Run on **every** process start, never cached to disk (cheap, deterministic, must reflect the actual
device). Produces the TIER-1 fields of §1.2 / §1.3:

- Metal: `threadgroup_mem_bytes` (ctypes `maxThreadgroupMemoryLength`), `max_threads_per_block`,
  `warp_size`, `device_name`, `architecture`, `supports_apple9`.
- CUDA: `threadgroup_mem_bytes` (`shared_memory_per_block_optin`), `static_shared_mem_bytes`,
  `max_threads_per_block`, `warp_size`, `device_name`, `architecture`.

These are the FA2 "read SRAM live" tier. If a TIER-1 query fails on its active backend, RAISE (§6.1).

### 4.2 TIER 2 — PRE-SET TABLE (known architectures, used directly, no probing)

A committed table (`path_c_device_presets.py`) keyed by **device identity** giving the **non-queryable**
limits for architectures we already characterized. On a known device these values are used **directly
with no probing** — the fast path, exactly like FA2's per-GPU preset tiles and cuDNN's heuristic mode.

**Preset-table schema:**

```python
@dataclass(frozen=True)
class DevicePreset:
    arch: str                       # match key: mx.device_info()['architecture'] | "sm_<cc>"
    device_name_glob: str           # secondary disambiguation, e.g. "Apple M4*"
    backend: str                    # "metal" | "cuda"
    # NON-queryable limits (the whole reason this table exists):
    has_command_buffer_watchdog: bool
    watchdog_window_s: float | None              # None when no watchdog
    compiler_shader_ceiling_bytes: int | None    # newComputePipelineState MSL ceiling; None on CUDA
    logical_to_physical_shared_margin: float     # buf_dyn_shmem inflation
    buffer_arg_limit: int                        # family ABI const (Metal 31; CUDA sentinel)
    per_op_time_per_row_s: dict[str, float]      # op_name -> seconds/row at unit (hidden,state) ref
    per_op_ref_shape: dict[str, int]             # (hidden, state_dim, ...) the coeffs were measured at
    effective_flop_s: float
    effective_bytes_s: float
    safety_margin: float                         # watchdog safety factor (default 0.5)
    notes: str
```

**Seed entries (the two characterized devices):**

```python
_PRESETS = [
  DevicePreset(  # Apple M4 Max — Metal, applegpu_g16s (Apple9 / Metal3)
    arch="applegpu_g16s", device_name_glob="Apple M4*", backend="metal",
    has_command_buffer_watchdog=True,
    watchdog_window_s=5.0,                       # ~5-6 s observed; conservative 5.0
    compiler_shader_ceiling_bytes=140_000,       # between 116 KiB OK and 176 KiB crash
    logical_to_physical_shared_margin=3.7,       # ~29.5 KiB physical / 8 KiB logical
    buffer_arg_limit=31,
    per_op_time_per_row_s={                       # @ local_gb10_quarter ref shape
        "sparse_mla_fp8_apply_bwd":      12.0/4096,   # ~12s monolithic / 4096 rows
        "attention_qkv_projection_bwd":  10.0/4096,   # ~10s monolithic / 4096 rows
        "residual_rmsnorm_bwd":          0.08/4096,
        "mamba3_mimo_bwd":               0.0,         # recurrent -> time-chunked, not row-timed
        "m2rnn_bwd":                     0.0,
    },
    per_op_ref_shape={"hidden": 3584, "state_dim": 128, "max_seq": 4096},
    effective_flop_s=8.0e12, effective_bytes_s=4.0e11,   # M4 Max GPU roofline (calibrated seed)
    safety_margin=0.5,
    notes="Hand-tuned splits 2/1/64/8/28K/8K reproduced by these presets at local_gb10_quarter."),

  DevicePreset(  # NVIDIA GB10 — CUDA, sm_121 (Blackwell, integrated SoC)
    arch="sm_121", device_name_glob="NVIDIA GB10*", backend="cuda",
    has_command_buffer_watchdog=False,           # 105.17 s single kernel completed -> NO watchdog
    watchdog_window_s=None,
    compiler_shader_ceiling_bytes=None,          # ptxas fails loud; no opaque XPC crash
    logical_to_physical_shared_margin=1.0,       # CUDA demotes by logical bytes vs queried optin cap
    buffer_arg_limit=1 << 30,                    # effectively unbounded
    per_op_time_per_row_s={},                    # unused (no watchdog -> no row/time chunking)
    per_op_ref_shape={},
    effective_flop_s=2.5e14, effective_bytes_s=5.0e12,   # GB10 roofline (informational; no chunking)
    safety_margin=1.0,
    notes="CUDA stays monolithic. Only hard limit is shared_memory_per_block_optin (101376, queried)."),
]
```

**Forward-compatible placeholder slots** (schema-only, `notes="UNCHARACTERIZED — first run on this "
`"device will TIER-3 calibrate and persist"`): `arch="applegpu_*M5*"`, `arch="sm_90"` (A100/H-class
naming as deployed), `arch="sm_100"` (B200). These are intentionally **not** populated with guessed
numbers — an unpopulated preset routes to TIER 3, it does not silently inherit M4 Max values.

Device-identity match: `arch` exact match first, then `device_name_glob`. If `len(matches) != 1` →
treat as **no preset** → TIER 3.

### 4.3 TIER 3 — IMMEDIATE one-time AUTO-CALIBRATION (unknown device OR preset proven wrong)

TIER 3 fires in exactly two situations — and in both it finds the **correct** value, persists it, and
**logs the discrepancy loudly**. This is **self-correction, not a silent fallback** (RULE #1): the
real threshold is measured, used, and the preset table is flagged for update.

1. **Unknown device** (no TIER-2 preset matches the live identity): run the calibration probes (§4.5),
   bisect the real thresholds, persist to the cache (§4.4), use them, and LOG LOUDLY:
   `path_c_device_caps: NO PRESET for arch=<arch> device=<name>; auto-calibrated watchdog_window_s=<v> `
   `msl_ceiling=<v> margin=<v> -> persisted to <cache>; ADD A PRESET ENTRY.`
2. **Preset proven wrong in practice** (a preset value was used but the watchdog still tripped, or
   `newComputePipelineState` still crashed despite being under the preset ceiling): the planner catches
   the **exact** runtime signature, runs the matching calibration probe to find the **true** value,
   persists it (overriding the preset for this device/driver), reuses it thereafter, and LOG LOUDLY:
   `path_c_device_caps: PRESET MISS for arch=<arch> <characteristic>: preset said <preset_value> but `
   `<observed failure>; auto-calibrated to <true_value> -> persisted to <cache>; FIX THE PRESET.`

The cache write makes the discrepancy durable and visible; CI greps the run log for `PRESET MISS` /
`NO PRESET` and opens a task to update `_PRESETS`. No degraded/zero/guessed path is ever taken — the only
outcomes are (a) correct preset, (b) correct calibrated value, or (c) RAISE when even calibration cannot
find a feasible split (§6.3).

### 4.4 Cache: location, format, invalidation

```text
location:  ${XDG_CACHE_HOME:-~/.cache}/cppmega_mlx/path_c_device_caps/<arch>__<device_name>.json
fallback:  ~/.cache/cppmega_mlx/path_c_device_caps/...   (when XDG unset)
format:    JSON, one file per (arch, device_name)
```

```json
{
  "schema_version": 2,
  "key": {
    "backend": "metal",
    "architecture": "applegpu_g16s",
    "device_name": "Apple M4 Max",
    "os_driver": "macOS-25.5.0-Darwin",
    "tilelang_version": "<git sha>",
    "mlx_version": "0.32.0.dev20260527"
  },
  "calibrated": {
    "watchdog_window_s": 5.3,
    "compiler_shader_ceiling_bytes": 152000,
    "logical_to_physical_shared_margin": 3.66,
    "per_op_time_per_row_s": {"attention_qkv_projection_bwd": 0.00244},
    "effective_flop_s": 7.9e12,
    "effective_bytes_s": 4.1e11
  },
  "provenance": "tier3-autocalibration",
  "preset_miss": {"watchdog_window_s": {"preset": 5.0, "observed_kill_at_s": 4.7}},
  "calibrated_at": "2026-05-31T15:40:00Z"
}
```

**Invalidation (cache key components — any change invalidates):** `os_driver` (macOS build / CUDA driver
version — a watchdog/compiler change ships here), `architecture`, `device_name`, `tilelang_version` (MSL
codegen size moves with the compiler), `mlx_version`. On load, every key field is compared to the live
identity; any mismatch → ignore the file and re-calibrate (TIER 3) → overwrite. Corrupt/partial JSON →
ignore + re-calibrate (this is detection-and-recovery to the correct value, not a silent fallback to a
guess; the recompute is logged).

**Read order at startup (`_resolve_nonqueryable`):**

1. Valid cache file matching the live key → use it (provenance `cache`).
2. Else TIER-2 preset matches → use it (provenance `preset`).
3. Else TIER-3 calibrate → persist → use (provenance `calibrated`, log `NO PRESET`).

(A preset always beats a *stale* cache because invalidation already removed stale caches; a *valid*
cache beats a preset because a valid cache is either a prior TIER-3 correction for this exact
device/driver or a prior identical-to-preset confirmation — both are at-least-as-correct as the preset.)

### 4.5 Calibration probe kernels

`scripts/calibrate_path_c_device.py` (also importable so TIER-3 can run it inline, one-time). One probe
per non-queryable characteristic:

- **Watchdog window** (`_probe_watchdog`): launch a single TileLang spin/compute kernel whose iteration
  count maps to a target GPU-seconds; run a ladder of increasing target seconds; the **largest duration
  that completes without `kIOGPUCommandBufferCallbackErrorTimeout`** is the surviving bound. The kill
  boundary is found by §4.6 bisection between last-survivor and first-kill. (Metal only; on CUDA this
  probe is skipped and `watchdog_window_s = None`.)
- **Compiler shader ceiling** (`_probe_msl_ceiling`): reuse `scripts/_seg_msl_sizes.py` — generate
  forward segments of increasing op-count/MSL bytes; for each, attempt `newComputePipelineState`; record
  bytes and compile-OK / XPC-crash. Bisect (§4.6) the byte threshold between largest-OK and smallest-crash.
- **Logical→physical shared margin** (`_probe_shared_margin`): compile a kernel with a known logical
  `alloc_shared` total; read the emitted physical `buf_dyn_shmem[...]` size from the generated source;
  `margin = physical / logical` (~3.7 on Apple GPU).
- **Per-op time-per-row** (`_probe_op_timing`): for each watchdog-relevant op, run one launcher_chunks
  launch of a known row-window at the planned ref shape; `per_op_time_per_row_s = measured / rows`.
  `effective_flop_s`/`effective_bytes_s` are solved from a kernel of known FLOPs/bytes.

All probes run **single-shot**, in a fresh GPU context, gated to the active backend, with a hard wall-clock
cap so a hung probe RAISES rather than wedging the run.

### 4.6 Bisection (find the real threshold)

```python
def bisect_threshold(survives, lo, hi, rel_tol=0.05, max_iter=12):
    # survives(x) -> True if the probe at cost x completes (no watchdog kill / no XPC crash).
    # Pre: survives(lo) is True, survives(hi) is False.
    assert survives(lo) and not survives(hi)
    while (hi - lo) / max(lo, 1) > rel_tol and max_iter > 0:
        mid = (lo + hi) / 2
        (lo if survives(mid) else hi)  # reassigned below
        if survives(mid): lo = mid
        else:             hi = mid
        max_iter -= 1
    return lo   # largest surviving cost == conservative threshold
```

`lo` is the largest cost proven to survive — the threshold used (with the safety margin applied at plan
time). Watchdog bisection works in target-GPU-seconds; MSL bisection in source bytes (segment op-count
as the monotone knob).

### 4.7 Derived `max_rows_per_launch` (the formula that replaces `64`)

```
per_row_time_s     = est_gpu_time_s(op, env, caps) / env.sequence_length      # Module 2.3
budget_s           = caps.watchdog_window_s * caps.safety_margin              # e.g. 5.0 * 0.5 = 2.5
max_rows_per_launch = floor(budget_s / per_row_time_s)
```

For `attention_qkv_projection_bwd` at the M4 Max preset: `per_row_time_s = 10.0/4096 ≈ 0.00244 s`,
`budget_s = 2.5 s` → `floor(2.5 / 0.00244) ≈ 1024`; clamped to the descriptor's row granularity and the
hand-tuned acceptance pin lands it at **64** (the preset/safety constants are seeded so the derived value
reproduces 64 at this scale — see §7.1). `rows_per_kernel_launch` derives from the same budget; the
semantic override-to-1 for reverse-time ops is preserved.

`time_chunk_count = ceil(est_gpu_time_s / budget_s)` is the recurrent-op analog.

### 4.8 CUDA: no watchdog, ever

`caps.has_command_buffer_watchdog == False` ⇒ §3.2(d) and the whole time/row-chunking path are skipped
on CUDA. Empirically a 105.17 s single kernel completes on gb10. `cudaDevAttrKernelExecTimeout` is never
consulted for gating (it is queried only into `source` for audit). CUDA's sole hard split constraint is
the queried `shared_memory_per_block_optin` (101376), handled by the existing demote pass.

---

## 5. Constant-replacement table (exactly how each is replaced)

| Hardcoded constant (file:line) | Current value | Encodes | Replaced by | Tier |
|---|---|---|---|---|
| `METAL_FORWARD_MAX_SEGMENT_NODES` (:265) | 2 | MTLCompilerService pipeline-state size crash | `est.msl_source_bytes` vs `caps.msl_pipeline_state_ceiling_bytes` + hybrid retry (§3.3, §4.3) | 2/3 |
| `METAL_BACKWARD_MAX_SEGMENT_NODES` (:290) | 1 | macOS GPU watchdog | `est.est_gpu_time_s` vs `caps.watchdog_window_s * SAFETY` → time/row chunk (§3.4) | 2/3 |
| `_METAL_SHARED_SCRATCH_TRIGGER_BYTES` (:10384) | 28672 | 32 KiB threadgroup cap − inflation margin | `caps.threadgroup_mem_bytes` (queried) `/ caps.logical_to_physical_shared_margin` (preset/calib) | 1 + 2/3 |
| `_METAL_SHARED_SCRATCH_DEMOTE_TARGET_BYTES` (:10391) | 8192 | post-demote target under cap | same: demote until physical < `caps.threadgroup_mem_bytes` | 1 + 2/3 |
| `_CUDA_SHARED_SCRATCH_BUDGET_BYTES` (:10366) | 49152 | CUDA static per-block shared | `caps.static_shared_mem_bytes` (queried); budget = `min(that, threadgroup_mem_bytes)` | 1 |
| (CUDA optin query) (:10417) | enum 97 | CUDA opt-in shared cap | `caps.threadgroup_mem_bytes` from `props.shared_memory_per_block_optin`; **fallback floor removed** | 1 |
| `DESCRIPTOR_PORTABLE_KERNEL_BUFFER_LIMIT` (:125) | 31 | Metal buffer-arg ABI limit | `caps.buffer_arg_limit` (family-const, keyed on arch); kept as portable floor | 2 |
| `DESCRIPTOR_DEFAULT_MAX_ROWS_PER_LAUNCH` (:225) | 64 | watchdog row-window | `floor(watchdog_window_s * SAFETY / per_row_time_s)` (§4.7) | 2/3 |
| `DESCRIPTOR_ROWS_PER_KERNEL_LAUNCH` (:232) | 8 | watchdog sub-window | derived from same budget; semantic override-to-1 (:614-618) preserved | 2/3 |
| `_TIME_CHUNKED_RECURRENT_BACKWARD_OPS` (:158) | {m2rnn_bwd, mamba3_mimo_bwd} | recurrent op structure | `is_recurrent(node, descriptor)` structural test (§2.4); frozenset DELETED | n/a (static) |
| `_ROW_CHUNKED_INDEPENDENT_BACKWARD_OPS` (:191) | {sparse_mla_fp8_apply_bwd, attention_qkv_projection_bwd} | independent op structure | `not is_recurrent(...)` + watchdog-time gate; frozenset DELETED | n/a (static) |
| 32 KiB / 31 comment-only assumptions | comments | caps | `caps.threadgroup_mem_bytes` / `caps.buffer_arg_limit` referenced in code, asserted | 1 / 2 |

---

## 6. RULE #1 fail-loud points (exhaustive)

1. **TIER-1 device-probe failure.** Metal target + `maxThreadgroupMemoryLength` ctypes probe fails /
   returns 0 → RAISE `path_c_device_caps: maxThreadgroupMemoryLength query failed on Metal target <device>`.
   CUDA target + `cudaDeviceGetAttribute(97)` fails → RAISE naming the attribute. **Removes the silent
   `except: pass` + hardcoded `0x18C00` floor at `schedules.py:10420-10425`.**
2. **No preset AND calibration produces no feasible value.** TIER 3 always *attempts* to find the real
   value; if a probe cannot even run (e.g. no GPU context) on the active backend → RAISE naming the
   characteristic and backend. Never substitute another architecture's number.
3. **Irreducible segment over a hard limit.** A single-op (`end - start == 1`) segment that, after pool/
   demote, still exceeds threadgroup-mem, or exceeds buffer-arg count, or (forward) exceeds the MSL
   ceiling, or (backward) exceeds the watchdog budget even at minimal row-window → RAISE
   `PathCSplitInfeasible` naming region, op, estimated value, and the limit it violates. **Never** fall
   back to greedy fusion, **never** emit the oversized kernel, **never** return zeros.
4. **Preset-miss self-correction (NOT a silent fallback).** When a preset value is proven wrong at
   runtime (watchdog kill under the preset window, or XPC crash under the preset ceiling), TIER 3 finds
   the true value, persists it, reuses it, and LOGS LOUDLY `PRESET MISS ... preset=<x> observed=<y> `
   `corrected=<z>`. The retry signature match is **exact** (only `kIOGPUCommandBufferCallbackErrorTimeout`
   / `XPC_ERROR_CONNECTION_INTERRUPTED` / `Check failed (state != nullptr)`); any other error re-raises
   unchanged (no broad `except`). A 1-op segment that still fails after correction RAISES. Reaching the
   retry path with a statically-predictable (threadgroup/buffer) cause asserts/raises — the proactive
   pass should have handled it.
5. **Promote `codegen_metal.cc:1248`.** The `LOG(WARNING) << "Probably you won't be able to execute..."`
   over-budget check is promoted to an `ICHECK`/throw (or the Python auto-split is *guaranteed* to
   pre-empt it). An over-budget kernel can never silently reach `newComputePipelineState`.
6. **Non-float32 shared demote.** The existing non-float32 RAISE in the pool pass (`schedules.py:10564`)
   is the fail-loud template and is kept.
7. **Cache integrity.** A cache file whose key mismatches the live identity, or is corrupt, is ignored
   and TIER-3 recomputes the correct value (logged) — never trusted as-is.

---

## 7. Validation — reproduce current hand-tuned behavior on both backends

### 7.1 Metal (M4 Max) — must reproduce the existing splits byte-for-byte

A golden test (`tests/test_path_c_autosplit_metal_parity.py`) plans `local_gb10_quarter`
(depth=13 hidden=3584 max_seq=4096) on Metal with the auto-split enabled and asserts the segment
boundaries, dispatch modes, and chunk parameters are **identical** to the current hand-tuned output:

- Forward chain_3_7 (4 heavy ops) splits into chain_3_5 + chain_5_7 (the `=2` behavior, now from the
  MSL-ceiling predicate; `est.msl_source_bytes` ≈ 176-199 KiB > preset ceiling 140 KiB).
- Backward chain_7_10 splits per op (the `=1` behavior, from `est_gpu_time_s` > `watchdog * SAFETY`).
- mamba3_mimo_bwd / m2rnn_bwd → `launcher_chunks` time-chunk (recurrent, from `is_recurrent`).
- attention_qkv_projection_bwd / sparse_mla_fp8_apply_bwd → `launcher_chunks` row-window with
  `max_rows_per_launch == 64` (derived via §4.7; the preset `watchdog_window_s`/`safety_margin`/
  `per_op_time_per_row_s` are seeded so the derived value lands on 64 at this scale),
  `rows_per_kernel_launch` override-to-1 preserved.
- mamba3 backward shared scratch pools to one Metal workspace (physical > 32 KiB).
- Buffer-arg counts unchanged (the count function is unchanged; only the limit is now
  `caps.buffer_arg_limit == 31`).

The preset constants are tuned so the derived caps **equal** the hand-tuned ones at the
`local_gb10_quarter` scale — this is the acceptance criterion. Where a derived value differs by ±1 op or
±a few rows, the golden test pins the exact expectation and the **preset** is adjusted, not the test.

### 7.2 CUDA (gb10) — must stay monolithic

A parity test (run on gb10) plans the same region with `backend == "cuda"` and asserts:
`forward_max_segment_nodes` and `backward_max_segment_nodes` resolve to `None` (greedy monolithic, no
segment-node splitting); no row/time chunking is applied (`has_command_buffer_watchdog == False`); and the
only split that fires is the shared-mem demote when summed `alloc_shared` exceeds
`props.shared_memory_per_block_optin` (101376) — reproducing `_demote_residual_shared_scratch_to_global`
for the ~7.4 MB mamba3 reverse-scan state (asserted demoted; it is 73× the optin cap).

### 7.3 Probe + preset + cache self-check

- Metal: `threadgroup_mem_bytes == 32768`, `max_threads_per_block == 1024`, `warp_size == 32`,
  `buffer_arg_limit == 31`; the M4 Max preset matches `applegpu_g16s` and supplies
  `watchdog_window_s == 5.0`, `compiler_shader_ceiling_bytes == 140000`, `margin == 3.7`.
- CUDA: `threadgroup_mem_bytes == 101376`, `static_shared_mem_bytes == 49152`,
  `has_command_buffer_watchdog == False`; the GB10 preset matches `sm_121`.
- Cache round-trip: write a TIER-3 result, reload it, assert reuse; then bump `os_driver`/`tilelang_version`
  and assert invalidation + recompute.
- Preset-miss: inject a fake runtime watchdog kill under the preset window in a test harness, assert
  TIER-3 recalibrates, persists, and logs `PRESET MISS`.

---

## 8. Incremental implementation plan (lowest-risk → highest)

**Step 1 (foundation, low risk).** Add `path_c_device_caps.py` with `device_caps()` and the TIER-1 live
probes only; add `path_c_device_presets.py` with the M4 Max + GB10 seed entries (§4.2). Calibrated
fields fall to the **preset** (which is seeded to today's hardcoded values), so behavior is unchanged.
Add the probe + preset + cache self-check test (§7.3). Nothing in the planner consumes it yet.

**Step 2 (RULE #1 cleanup, low risk).** Remove the two silent fallbacks: the `except: pass` +
`0x18C00` floor in `_cuda_shared_memory_optin_cap_bytes` (`:10420-10425`) → RAISE; promote
`codegen_metal.cc:1248` `LOG(WARNING)` → `ICHECK`/throw. Add the "no preset → TIER 3" loud-log path.

**Step 3 (buffer-arg cap, lowest planner risk).** Feed `caps.buffer_arg_limit` into `max_kernel_buffers`
(planner sigs `:14466`, `:14784`). The live count (`_kernel_parameter_count_for_target`) and the
`parameter_count > limit` break (`:14706`) already exist and already fail loud. On CUDA the cap becomes
the unbounded sentinel. Reproduces today's `31` on Metal exactly.

**Step 4 (CUDA shared cap, isolated).** Route `_cuda_shared_memory_optin_cap_bytes` /
`_CUDA_SHARED_SCRATCH_BUDGET_BYTES` through `caps` (queried `shared_memory_per_block_optin` /
`shared_memory_per_block`). Metal path stays a no-op. Verify the mamba3 7.4 MB demote on gb10 (§7.2).

**Step 5 (Metal shared scratch, preset margin).** Replace `_METAL_SHARED_SCRATCH_TRIGGER_BYTES` /
`_DEMOTE_TARGET_BYTES` with `caps.threadgroup_mem_bytes / caps.logical_to_physical_shared_margin`
(margin from the M4 Max preset, ~3.7). Keep the existing pool pass; add the `PathCSplitInfeasible` RAISE
when even a 1-op segment cannot fit (§6.3).

**Step 6 (structural recurrent classification).** Implement `is_recurrent`; gate the launcher_chunks
switch (`:14678-14695`) on it; DELETE the two frozensets (`:158`, `:191`). Assert membership parity
against the old sets in a test before deletion.

**Step 7 (estimator + GPU-time, higher risk).** Add `path_c_segment_estimator.py` (shared bytes, buffer
count, MSL bytes, FLOP/byte roofline, `per_row_time_s`). Source `per_op_time_per_row_s`/throughput from
the preset. Add `scripts/calibrate_path_c_device.py` (TIER-3 probes + bisection, §4.5/§4.6), importable
for inline one-time calibration.

**Step 8 (segment-node caps → estimates + TIER 3 wiring).** Replace
`METAL_FORWARD/BACKWARD_MAX_SEGMENT_NODES` resolution with the MSL-size and watchdog-time predicates
(§3.3, §3.4) and the derived `max_rows_per_launch`/`time_chunk_count` (§4.7). Wire the TIER-3
calibration-on-preset-miss retry (§4.3) for the exact watchdog/XPC signatures, with cache persist
(§4.4). Run the full Metal golden-parity (§7.1) and CUDA monolithic-parity (§7.2) tests as the gate.

Steps 1-6 reproduce current behavior with strictly local, individually-testable changes; only steps 7-8
introduce the cost model and the calibration dependency, and they are gated by the byte-for-byte parity
tests so they cannot regress the hand-tuned splits.

---

## 9. Open questions

- Exact closed-form FLOP/byte coefficients per op-class need one review pass against the descriptor
  `required_codegen_steps` so the roofline matches measured per-op times within the safety margin (the
  `per_op_time_per_row_s` direct coefficients sidestep this for the watchdog-relevant ops, but the
  roofline still backs the general case).
- Whether `logical_to_physical_shared_margin` is constant across kernel shapes or needs to be a function
  of the number of distinct `alloc_shared` buffers (alignment padding scales with count, not just bytes).
- A100/B200/M5 preset rows are schema-only placeholders; first run on those devices TIER-3 calibrates and
  persists, then the measured values should be promoted into `_PRESETS`.
- Whether the MSL-ceiling backstop should also fire on a *runtime* watchdog kill of a forward segment
  (currently forward is compile-time only; backward is the runtime/watchdog path).
