# Performance Baseline Plan

This lane establishes a local MLX benchmark harness for Apple Silicon. It does
not prove M4 Max parity with GB10 by itself; it defines how to measure that claim
without mixing hardware, software stack, or model-shape effects.

## Benchmark Harness

Run the synthetic tiny benchmark from the repository root:

```bash
./.venv/bin/python scripts/bench_tiny.py \
  --batch-size 2 \
  --seq-len 64 \
  --vocab-size 2048 \
  --d-model 128 \
  --n-heads 4 \
  --n-layers 2 \
  --mlp-dim 512 \
  --dtype bfloat16 \
  --warmup-steps 5 \
  --steps 20 \
  --hardware-label "M4 Max" \
  --json
```

For side-by-side spreadsheets or log scraping, emit one stable comparison row:

```bash
./.venv/bin/python scripts/bench_tiny.py \
  --batch-size 2 \
  --seq-len 64 \
  --dtype bfloat16 \
  --warmup-steps 5 \
  --steps 20 \
  --hardware-label "M4 Max" \
  --compare-line
```

For memory-limit experiments on Apple Silicon, keep the JSON output and opt in
explicitly:

```bash
./.venv/bin/python scripts/bench_tiny.py \
  --batch-size 2 \
  --seq-len 64 \
  --dtype bfloat16 \
  --warmup-steps 5 \
  --steps 20 \
  --hardware-label "M4 Max" \
  --auto-wired-limit \
  --json
```

Do not use wired-limit changes as a parity claim. They are allocator residency
controls for MLX/Metal runs and should be reported alongside the exact memory
telemetry and software stack.

The current script reports:

- Top-level comparison fields: `hardware_label`, `dtype`, `batch_size`,
  `seq_len`, `warmup_steps`, `measured_steps`, `compile`,
  `include_structure`, `tokens_per_second`, and `peak_memory_bytes`. These are
  also the only fields in `--compare-line`; keep that one-line contract stable
  for spreadsheets and existing log scrapers.
- `compile_time_s`: first compiled-call wall time, including graph build and
  Metal code generation. This is `0.0` when `--no-compile` is used.
- `first_call_time_s`: first-call wall time for either compiled or eager mode.
- `mean_step_time_s` and `median_step_time_s`: synchronized steady-state
  training step times after the compile call and warmup steps.
- `tokens_per_second`: `batch_size * seq_len / mean_step_time_s`.
- `peak_memory_bytes` and `peak_memory_gib`: MLX peak memory since the reset
  immediately before steady-state measurement.
- Extended JSON memory telemetry under `memory`: active, peak, and cache bytes
  plus GiB conversions after warmup and after measured steady-state steps.
  `memory.active_bytes` comes from `mx.get_active_memory()` and excludes cached
  unused buffers; `memory.cache_bytes` comes from `mx.get_cache_memory()`.
- Profile hook output under `profile`: `first_call`, optional `warmup`, and
  `measured_steps` scopes are recorded through
  `cppmega_mlx.training.profile.profile_step` when the helper module is present.
  Each scope includes synchronized wall time, tokens, tokens/sec when available,
  active/peak/cache allocator bytes, whether peak memory was reset, and whether
  MLX arrays/state were forced with `mx.eval`.
- Optional wired-limit reporting under `memory.wired_limit`: mode, requested
  bytes, applied bytes, previous limit, MLX-reported `memory_size`, and
  `max_recommended_working_set_size`. Use `--auto-wired-limit` to request the
  device max recommended working set, or `--wired-limit-bytes N` for an explicit
  value. The script validates that the requested limit is strictly below total
  memory and not above the MLX-reported max recommended working set before it
  calls `mx.set_wired_limit`.
- `run_metadata`: a schema-versioned metadata block that records workload shape,
  synthetic-data contract, model source, optimizer learning rate, Python/OS,
  MLX/MLX-LM/MLX-Metal versions, default MLX device, MLX device info, Metal
  availability/capture support, and the exact profiling helper names used.
- `matched_run`: a compact comparison key derived from workload and stack
  metadata. It is a guardrail, not a claim: a single JSON row must not be used
  to claim anything about GB10.
- Matrix rows additionally expose `workload_key`, `software_key`, and
  `comparison_key`. For M4-vs-GB10 claims, require identical
  `comparison_key.workload` and identical `comparison_key.software`; labels such
  as `hardware_label` are only selectors.
- Matrix route aliases are workload identity, not display-only labels:
  `mamba3` resolves to model route `hybrid-m`, `m2rnn` resolves to
  `hybrid-r`, and `hybrid-aemr` resolves to the full hybrid route. Rows preserve
  both `route` and `resolved_model_route`; compare helpers require the alias
  label itself to match so `mamba3` is not silently compared to `hybrid-m`.
- Matrix rows also expose `bench_receipt`, a schema-versioned receipt block that
  repeats the hardware label, route, resolved model route, alias metadata,
  `seq_len`, `batch_size`, warmup/measured step counts, compile mode, tokens/sec
  or step-time fields, device, software stack, workload key, comparison key,
  matched-run guard, and timing metadata in one stable object for log collectors.
- Matrix summary, case rows, `matched_run`, `bench_receipt`, and local baseline
  archives mark single-host evidence with `receipt_scope: local_only`,
  `local_only: true`, and `gb10_parity_claim: false`. Keep those fields visible
  when copying rows into reports so a local Apple Silicon matrix is not quoted
  as a GB10 comparison before matched GB10 rows exist.
- `model_source`: the project tiny model when available, otherwise the
  self-contained fallback.

Imported GB10 Torch/CUDA rows are accepted when they preserve the same workload
fields plus explicit stack and device metadata. Use top-level fields or nested
`framework`/`device` objects, but keep these values machine-readable:

- `framework`: `torch`.
- `backend`: the exact execution path, for example `torch_sdpa`, `flash`, or a
  project-specific CUDA backend name.
- `torch_version`: or `torch`.
- `cuda_version`: or `cuda` / `torch_cuda`.
- `driver_version`: or `driver` / `cuda_driver`.
- `device_name`: or `cuda_device` / string-valued `device`.
- `device_capability`: or `cuda_capability` / `cuda_cap` / `capability`.
- Throughput and memory aliases: `tokens_per_second` or `tokens_per_sec`, and
  `peak_memory_bytes` or `max_alloc_gib`.

These are comparison identity fields, not decorative labels. A Torch/CUDA GB10
row and an MLX/Metal M4 row can be stored in the same input file, but
`scripts/compare_bench_rows.py` must reject the cross-stack pair and emit no
ratio. Only rows with matching workload fields and matching framework/backend
metadata may produce M4-vs-GB10 ratios.

The train step follows the current MLX optimizer contract: compute
`nn.value_and_grad`, call `optimizer.update(model, grads)`, and evaluate
`model.state` plus `optimizer.state`. The compiled path captures
`[model.state, optimizer.state, mx.random.state]` as compile inputs/outputs.
Do not manually assign `optimizer.state` or pass parameter dictionaries through
the benchmark loop when changing this script.

The benchmark prefers the local `cppmega_mlx.models.tiny_lm` and
`cppmega_mlx.data.batch.synthetic_token_batch` APIs when they are present. It keeps a
self-contained fallback model for partial checkouts; always record
`model_source` from the JSON before comparing runs. `--include-structure` adds
the tiny cppmega structure side-channel tensors when the project tiny API is
available.

## Why Compile Warmup Is Separate

MLX compilation is not just a timer detail. The first call to a compiled
function builds the compute graph, optimizes it, and generates/compiles code;
subsequent calls reuse the cached compiled function for the same compatible
input contract. Therefore, GB10-vs-M4 comparisons must report first-call compile
time separately from steady-state training throughput.

For this harness:

1. The first compiled call is recorded as `compile_time_s`.
2. `warmup_steps` are run and synchronized but excluded from throughput.
3. Peak memory is reset before measured steady-state steps.
4. `steps` synchronized training updates produce the reported mean/median step
   time and tokens/sec.

For matrix JSON, the same measurement contract is mirrored under
`bench_receipt.timing`: `first_call_time_s`, `compile_time_s`,
`mean_step_time_s`, `median_step_time_s`, `tokens_per_second`, warmup/measured
step counts, compile mode, raw per-step timing arrays, and a
`synchronized_timing` flag. A row without `tokens_per_second` or step-time fields
is planning metadata only, not benchmark evidence.

## Fair GB10 vs M4 Max Methodology

The claim "M4 Max is not worse than GB10" is only meaningful under a matched
workload. At minimum, compare runs with identical:

- Model source and commit: same fallback tiny model or same future cppmega tiny
  adapter.
- Shape: `batch_size`, `seq_len`, `vocab_size`, `d_model`, `n_heads`,
  `n_layers`, and `mlp_dim`.
- Precision: same dtype, preferably `bfloat16` for first parity work.
- Optimizer and update rule: same optimizer family, learning rate, weight decay,
  and gradient accumulation policy.
- Data contract: synthetic data for synthetic benchmarks, or the same tokenized
  real batches for model-port benchmarks.
- Measurement window: compile excluded from steady-state throughput, same
  warmup count, same measured step count, and synchronized timers on both hosts.
- Memory accounting: report peak allocator memory and any hardware-level memory
  telemetry separately; do not compare a framework allocator peak on one machine
  to a whole-device peak on another.
- Cache accounting: compare `memory.active_bytes`, `memory.peak_bytes`, and
  `memory.cache_bytes` separately because active MLX arrays and cached allocator
  buffers answer different questions.
- Wired-limit mode: record `memory.wired_limit.mode` and whether the request was
  actually applied. A run with `--auto-wired-limit` is not directly comparable to
  one without it unless the residency setting is treated as part of the stack.
- Thermals and power: record whether either host was thermally throttled, on
  battery, in low-power mode, or sharing the GPU/accelerator with other work.
- Software stack: MLX/Metal/macOS versions for M4 Max; CUDA, driver, framework,
  and kernel versions for GB10.
- Matched-run key: the JSON `matched_run.key` values must match across rows.
  For matrix output, use `matched_run_key`, which additionally includes
  `profile` and `route`.
- Matrix comparison key: for matrix output, prefer `comparison_key.workload` and
  `comparison_key.software`. They split workload identity from software-stack
  identity so parity claims cannot accidentally compare M4 MLX/Metal rows
  against GB10 rows collected with different framework, backend, Python, MLX,
  MLX-LM, Metal, or platform metadata.
  `scripts/compare_bench_rows.py` requires the selected explicit workload and
  software keys to match as complete objects, not just as normalized display
  fields. Extra stack flags or key-only data-contract differences deliberately
  block ratios.

If batch sizes differ because one host runs out of memory, report the largest
common batch first. Additional max-throughput runs can be useful, but they are
capacity studies, not a same-workload parity claim.

Matched-run guard: do not write "M4 Max matches GB10", "GB10 is slower", or any
equivalent parity/win claim unless both rows are collected and their
`comparison_key.workload` and `comparison_key.software` values are identical.
For non-matrix `bench_tiny.py` rows, use `matched_run.key` plus the nested
`run_metadata.framework` stack block. If only one host has been measured, report
it as a single-host baseline. If keys differ, describe the difference first and
classify the result as a capacity or stack study rather than parity.

## Reporting Template

Capture each run as JSON and attach the command line:

```bash
./.venv/bin/python scripts/bench_tiny.py --json ... > runs/m4max_tiny_bf16.json
```

Or capture comparable one-line rows:

```bash
./.venv/bin/python scripts/bench_tiny.py --compare-line ... >> runs/tiny_compare.txt
```

For a small matched-shape matrix across batch size, sequence length, model
profile, route, and compile mode, use the matrix wrapper:

```bash
./.venv/bin/python scripts/bench_matrix.py \
  --hardware-label "M4 Max" \
  --batch-sizes 1,2 \
  --seq-lens 32,64 \
  --profiles smoke,tiny,hybrid-smoke \
  --routes plain,structure,mamba3,m2rnn,hybrid-aemr \
  --compile-modes eager,compiled \
  --dtype bfloat16 \
  --warmup-steps 5 \
  --steps 20 \
  --json > runs/m4max_matrix.json
```

For streaming log collectors, `--jsonl` emits one JSON object per matrix row.
The matrix runner is still a wrapper over the tiny benchmark; it does not make
GB10 claims. Treat its `matched_run_guard` field as binding: compare M4 Max and
GB10 only when rows have identical profile, route, dtype, batch size, sequence
length, compile mode, warmup, measured step count, data contract, and software
stack fields. The per-case `comparison_key` is the machine-checkable key for
that comparison; `matched_run_key` remains available for compatibility and is
mirrored as `workload_key`. The per-case `bench_receipt` block is the preferred
artifact for archiving or forwarding matrix rows because it keeps device,
software, workload, timing, and guardrail fields together.

Every matrix receipt emitted by `scripts/bench_matrix.py` is local-only until it
is paired with a GB10 row that has identical `comparison_key.workload` and
`comparison_key.software`. The receipt fields
`receipt_scope: local_only`, `local_only: true`, and `gb10_parity_claim: false`
must remain visible in archived summaries and copied rows.

To maintain a local M4 Max regression ledger, append matrix summaries to a
schema-versioned archive:

```bash
./.venv/bin/python scripts/bench_matrix.py \
  --hardware-label "M4 Max" \
  --batch-sizes 1,2 \
  --seq-lens 32,64 \
  --profiles smoke,tiny,hybrid-smoke \
  --routes plain,structure,mamba3,m2rnn,hybrid-aemr \
  --compile-modes eager,compiled \
  --dtype bfloat16 \
  --warmup-steps 5 \
  --steps 20 \
  --json \
  --archive-baseline runs/m4max_baseline_archive.json \
  --baseline-note "local M4 Max regression baseline"
```

The archive schema is intentionally conservative:

- Archive object: `schema_version: 1`, kind
  `cppmega.mlx.local_m4_benchmark_baselines`, top-level `guards`, and append-only
  `records`.
- Record object: `schema_version: 1`, kind
  `cppmega.mlx.local_m4_benchmark_baseline_record`, timestamp, hardware label,
  profile/route inventory, source schema versions, compare-line contract, guard
  policy, and one row per matrix case.
- Row object: stable `case_id`, status, hardware/profile/route selectors,
  `comparison_key`, `workload_key`, `software_key`, `bench_receipt`, compact
  timing/memory metrics, `receipt_scope: local_only`, `local_only: true`, and
  `gb10_parity_claim: false`.
- Compare-line contract: the archived field order mirrors
  `scripts/bench_tiny.py --compare-line` and is guarded by tests:
  `hardware_label`, `dtype`, `batch_size`, `seq_len`, `warmup_steps`,
  `measured_steps`, `compile`, `include_structure`, `tokens_per_second`,
  `peak_memory_bytes`. Treat this field order as append-only by explicit
  migration; do not reorder it for display preferences.
- Append guard: before appending to an existing archive, the writer validates
  the archive, every prior record, every row, and every nested `bench_receipt`
  still carry `receipt_scope: local_only`, `local_only: true`, and
  `gb10_parity_claim: false`. If a prior archive is missing those guards or
  contains a parity claim, the append fails instead of rewriting history.

This archive is not a GB10 comparison file. It is a local M4 baseline and
regression history so later Apple Silicon changes can be compared against the
same MLX/Metal workload. To make an M4-vs-GB10 claim, export matched M4 and GB10
rows and run `scripts/compare_bench_rows.py`; if strict workload and software
keys do not match, report no ratio.

`scripts/compare_bench_rows.py` treats `bench_receipt` as a first-class input
source, not only as an archived copy. If an exported row only keeps top-level
`hardware_label` plus nested `bench_receipt.workload`,
`bench_receipt.software`, `bench_receipt.timing`, or
`bench_receipt.comparison_key`, the parser reconstructs the normalized match
fields from those nested blocks. Mismatched `data_contract` values, including
future parquet labels such as `parquet_clang_v10_code`, intentionally block
ratios in the same way as MLX/Metal/Python/software-stack mismatches.
If a row contains more than one modern key source, for example top-level
`comparison_key`, `bench_receipt.comparison_key`, and `workload_key` plus
`software_key`, those sources must agree inside the row before the helper will
use the row. A row-local conflict is reported through
`matched_comparison_key_conflicts` and is treated as missing matched-key
provenance, so no M4-vs-GB10 ratio is emitted.

For local Mamba3/M2RNN/hybrid receipts, prefer the named aliases so exported rows
are stable across the MLX and Megatron/CUDA inventories:

```bash
./.venv/bin/python scripts/bench_matrix.py \
  --json \
  --hardware-label "M4 Max" \
  --batch-sizes 1 \
  --seq-lens 8 \
  --profiles hybrid-smoke \
  --routes mamba3,m2rnn,hybrid-aemr \
  --compile-modes eager \
  --dtype float32 \
  --warmup-steps 1 \
  --steps 3
```

## Local M4 Max Smoke Hotspot Pass

On 2026-04-30, this local Apple M4 Max ran a small eager smoke matrix to rank
the current MLX tiny and hybrid route costs before considering any custom Metal
work:

```bash
./.venv/bin/python scripts/bench_matrix.py \
  --json \
  --hardware-label "M4 Max" \
  --batch-sizes 1 \
  --seq-lens 8 \
  --profiles smoke \
  --routes tiny,hybrid-a,hybrid-e,hybrid-m,hybrid-r \
  --compile-modes eager \
  --dtype float32 \
  --warmup-steps 1 \
  --steps 3
```

The run reported device `Apple M4 Max`, MLX `0.31.1`, and MLX-LM `0.31.2`.
These are single-host smoke results, not GB10 parity evidence.

| Route | Backend summary | Mean step ms | Tokens/s | Peak bytes |
| --- | --- | ---: | ---: | ---: |
| `hybrid-m` | `mamba3: 1` | 2.8190 | 2837.84 | 49492 |
| `hybrid-r` | `m2rnn: 1` | 1.6091 | 4971.86 | 60164 |
| `hybrid-e` | `moe: 1` | 1.5748 | 5080.04 | 85840 |
| `tiny` | `mlx.nn.MultiHeadAttention: 1` | 1.1685 | 6846.22 | 54301 |
| `hybrid-a` | `attention: 1` | 1.1680 | 6849.48 | 53566 |

For this shape, the observed hotspot order was
`hybrid-m > hybrid-r > hybrid-e > tiny > hybrid-a` by mean step time. Re-run the
same matrix with larger sequence lengths and measured-step counts before
starting custom kernel work; this smoke pass is only enough to prioritize the
next profiling target.

No matched GB10 row is recorded in this document. Until a GB10 run is collected
with the same workload, stack metadata, timing window, and data contract, the
M4 Max rows above remain local regression baselines only. Local real-data
Parquet smoke tests under `data/parquet_samples/` should also be kept separate
from synthetic benchmark rows unless both hosts consume the same token batches.

Wave 9 lane 5 also produced a local-only smoke receipt at
`/tmp/cppmega_mlx_wave9_lane5_m4_smoke.json` with this command:

```bash
./.venv/bin/python scripts/bench_matrix.py \
  --json \
  --hardware-label "M4 Max local-only wave9 lane5" \
  --batch-sizes 1 \
  --seq-lens 4 \
  --profiles smoke \
  --routes plain \
  --compile-modes eager \
  --dtype float32 \
  --warmup-steps 0 \
  --steps 1 > /tmp/cppmega_mlx_wave9_lane5_m4_smoke.json
```

That receipt reported one `smoke-route_plain-b1-s4-float32-eager` row on
`Apple M4 Max`, `receipt_scope: local_only`, `local_only: true`,
`gb10_parity_claim: false`, tokens/sec `3150.2268407971783`, and mean step time
`0.0012697498314082623` seconds. It is single-host smoke evidence only.

Wave 10 lane 6 produced another local-only smoke receipt at
`/tmp/cppmega_mlx_wave10_lane6_m4_smoke.json` with the same smoke workload and
`--hardware-label "M4 Max local-only wave10 lane6"`. The receipt reported one
`smoke-route_plain-b1-s4-float32-eager` row on `Apple M4 Max`,
`receipt_scope: local_only`, `local_only: true`, `gb10_parity_claim: false`,
tokens/sec `3640.3619589873447`, and mean step time `0.0010987918358296156`
seconds. It is single-host smoke evidence only and is not a GB10 comparison.

Package matched M4 Max and GB10 rows with the comparison helper rather than by
hand-copying local-only numbers into a table:

```bash
./.venv/bin/python scripts/bench_matrix.py \
  --hardware-label "M4 Max" \
  --batch-sizes 1,2 \
  --seq-lens 32,64 \
  --profiles smoke,tiny,hybrid-smoke \
  --routes plain,structure,mamba3,m2rnn,hybrid-aemr \
  --compile-modes eager,compiled \
  --dtype bfloat16 \
  --warmup-steps 5 \
  --steps 20 \
  --json > runs/m4max_matrix.json

./.venv/bin/python scripts/bench_matrix.py \
  --hardware-label "GB10" \
  --batch-sizes 1,2 \
  --seq-lens 32,64 \
  --profiles smoke,tiny,hybrid-smoke \
  --routes plain,structure,mamba3,m2rnn,hybrid-aemr \
  --compile-modes eager,compiled \
  --dtype bfloat16 \
  --warmup-steps 5 \
  --steps 20 \
  --json > runs/gb10_matrix.json

./.venv/bin/python scripts/compare_bench_rows.py \
  --input runs/m4max_matrix.json \
  --input runs/gb10_matrix.json \
  --package-dir runs/gb10_matched_package > runs/matched_compare.json
```

The helper accepts matrix summary JSON, JSON arrays/objects, or NDJSON rows. Add
`--jsonl` when downstream tooling wants one comparison row per line. It reports
throughput, memory, and compile-time ratios only after a strict match on the
selected explicit `comparison_key.workload` and `comparison_key.software`
objects plus normalized model source, synthetic data contract, dtype, compile
mode, warmup/measured steps, shape, route/profile, framework/backend, and
software versions. For MLX rows it requires MLX/Metal version metadata. For
Torch rows it requires the Torch version and device name; CUDA-style rows also
require CUDA version, driver version, and device capability. If only M4 Max or
only GB10 rows are present, the status is `insufficient_matched_rows`; if both
hosts are present but no strict key matches, the status is `no_matching_rows`.
In both cases, `comparisons` stays empty and no ratio should be reported.

The packaged handoff writes `manifest.json`, `compare_report.json`,
`matched_comparisons.jsonl`, and `refused_pairs.jsonl`. Packaged
`compare_report.json` keeps the guard/refusal summary without ratios. Only
`matched_comparisons.jsonl` may contain `ratios`; `refused_pairs.jsonl` is
archiveable evidence that a pair was rejected because the workload/software keys
or required metadata did not match.

The GB10 Megatron/Torch launchers in the sibling `../cppmega` checkout use a
different CUDA/Megatron training stack from this MLX matrix. Those logs are
useful grounding for GB10 runtime constraints, but they are not automatically
comparable to M4 Max MLX rows. Add explicit stack fields to any imported GB10
rows and let `scripts/compare_bench_rows.py` reject mismatches instead of
claiming parity across framework boundaries.

Use a table like this for comparisons:

| Host | Device | Stack | Model source | Dtype | Batch | Seq | Params | Compile s | Mean step s | Tokens/s | Peak GiB |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| M4 Max | Apple M4 Max | MLX/Metal | fallback tiny | bf16 | 2 | 64 | fill from JSON | fill | fill | fill | fill |
| GB10 | fill exact device | fill exact stack | fallback tiny equivalent | bf16 | 2 | 64 | fill from JSON | fill | fill | fill | fill |

Only mark the target as met when the steady-state tokens/sec and memory numbers
come from matched rows. Keep compile-time as its own result because it affects
interactive iteration but should not be folded into steady-state throughput.

## References Checked

- MLX compile documentation: https://ml-explore.github.io/mlx/build/html/usage/compile.html
  The benchmark follows the documented split between first compiled call and
  cached repeated calls, uses warmup before timing, materializes MLX work with
  `mx.eval`, and keeps `mx.random.state` in the compiled training state.
- MLX peak-memory API: https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.get_peak_memory.html
- MLX active-memory API: https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.get_active_memory.html
- MLX cache-memory API: https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.get_cache_memory.html
- MLX wired-limit API: https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.set_wired_limit.html
- Local MLX stubs: `.venv/lib/python3.13/site-packages/mlx/core/__init__.pyi`
  and `.venv/lib/python3.13/site-packages/mlx/core/metal.pyi`.
- Local MLX-LM trainer pattern:
  `.venv/lib/python3.13/site-packages/mlx_lm/tuner/trainer.py`.
- MLX-LM benchmark docs/source: https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/BENCHMARKS.md and `mlx_lm/benchmark.py`
- MLX custom Metal kernels: https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html
- Hugging Face Apple M4 kernel listing: https://huggingface.co/kernels?hardware=apple-m4&sort=trending.
  On 2026-04-30 the visible trending Apple M4 filter returned 10 kernels; the
  Metal-capable entries included
  `kernels-community/mlx-rmsnorm`, `kernels-community/relu`,
  `kernels-community/paged-attention`,
  `kernels-community/mlx-quantization-metal-kernels`,
  `kernels-community/metal-flash-sdpa`,
  `kernels-community/gpt-oss-metal-kernels`,
  `kernels-community/bitsandbytes-mps`, and
  `kernels-community/activation`.
