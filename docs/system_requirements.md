# MLX Mac Local Training System Requirements

This document is the fail-closed readiness slice for local cppmega.mlx training
on Apple Silicon Macs. It is a preflight checklist, not a performance receipt.
Fail closed means report `NOT READY` and do not start a training run when a
required probe is missing, stale, ambiguous, or below the threshold.

## Scope And Non-Claims

- This checks local Mac MLX training readiness for the small and incremental
  cppmega.mlx lanes.
- M4-vs-GB10 parity is not proven by this document or by a Mac-only run.
- Distributed Megatron parity is not claimed from `mlx.core.distributed`,
  JACCL availability, or single-host distributed probes.
- Trainable Metal-kernel adoption is not claimed. A training-path Metal kernel
  still requires a pure-MLX fallback, dtype and gradient parity, local profiling
  evidence, and VJP/JVP coverage before it can replace the reference path.
- Full NAM56R readiness is not claimed. This repo has local scaffolds and tiny
  training lanes; it does not prove full capacity, distributed launch, CUDA,
  GB10, H200, or production readiness.

## Required Readiness Receipt

Every local training receipt must record these fields before the first measured
step:

- Date, git commit, script, CLI args, dataset shape, dtype, optimizer, warmup
  steps, measured steps, and whether `mx.compile` is enabled.
- `importlib.metadata.version("mlx")`.
- `platform.platform()`, `platform.machine()`, and macOS version.
- `mx.default_device()` and `mx.metal.is_available()`.
- `mx.device_info()` with at least `device_name`, `memory_size`,
  `max_recommended_working_set_size`, and `architecture`.
- File descriptor soft and hard limits from `resource.RLIMIT_NOFILE`.
- Memory-limit mode, requested bytes, applied bytes, previous limit bytes, and
  whether the run used `mx.set_wired_limit`, `mx.set_memory_limit`, or the
  compatibility `mx.metal.set_memory_limit` path.
- Thermal status source: `powermetrics` available/unavailable, sample window if
  collected, and any thermal-pressure or throttling observation.
- Distributed status only as a measured experimental field:
  `mx.distributed.is_available("ring")`,
  `mx.distributed.is_available("jaccl")`,
  `mx.distributed.is_available("mpi")`, and
  `mx.distributed.is_available("nccl")`.

Current CN probe, captured on 2026-05-01 in this checkout, is an example of the
required shape only: MLX `0.31.1`, `Device(gpu, 0)`, Apple M4 Max,
`memory_size=137438953472`, `max_recommended_working_set_size=115448725504`,
file descriptor soft limit `131072`, ring and JACCL available, MPI and NCCL not
available. Re-probe on every host; do not reuse this row as a readiness claim
for another machine.

## Host Requirements

- macOS on Apple Silicon (`arm64`) with the Metal backend available is required
  for local Mac training readiness. CPU-only fallback can be useful for import
  tests, but it is `NOT READY` for a Mac training receipt.
- Unified RAM must fit the model weights, gradients, optimizer state,
  activations, compilation/cache overhead, dataloader buffers, and OS headroom.
  If the budget does not fit under the configured MLX limits with headroom, the
  run is `NOT READY`; shrink the shape or model before training.
- Do not treat a RAM tier as proof of NAM56R readiness. A 64 GB or 128 GB Mac
  can be a local development host only after the concrete model/data shape has a
  passing memory budget and smoke receipt.

Minimal preflight probe:

```bash
./.venv/bin/python - <<'PY'
import importlib.metadata as md
import platform
import resource

import mlx.core as mx

print("mlx", md.version("mlx"))
print("platform", platform.platform())
print("machine", platform.machine())
print("default_device", mx.default_device())
print("metal_available", mx.metal.is_available())
print("device_info", mx.device_info())
print("nofile", resource.getrlimit(resource.RLIMIT_NOFILE))
for backend in ("ring", "jaccl", "mpi", "nccl"):
    print(f"distributed_{backend}", mx.distributed.is_available(backend))
PY
```

Fail closed if the script cannot import MLX, cannot query device information,
uses a CPU default device for a training receipt, reports no Metal backend, or
does not expose enough memory information to compute the training budget.

## File Descriptor Limit

Local training launchers should require `ulimit -n >= 65536`, measured through
`resource.RLIMIT_NOFILE`, before opening dataset shards, logs, checkpoints, or
distributed sockets. If the soft limit is below `65536`, report `NOT READY` and
raise it before the run:

```bash
ulimit -n 65536
```

Do not silently continue with a lower limit. File-descriptor exhaustion can look
like flaky dataset, checkpoint, or launcher behavior and should be rejected at
startup.

## Memory Limit Policy

Use the repo helper in `cppmega_mlx/runtime/memory.py` for arithmetic and
receipts. The documented default planning ratios are:

- `DEFAULT_WIRED_RATIO = 0.70`
- `DEFAULT_METAL_RATIO = 0.85`

The helper is dry-run by default. It must not mutate process-global MLX limits
unless the caller explicitly applies the plan. For an applied training run,
record both requested and previous limits.

Current MLX documentation exposes `mx.set_wired_limit(limit_bytes)` and
`mx.set_memory_limit(limit_bytes)`. The current local install and helper also
support `mx.metal.set_memory_limit(limit_bytes)` as the compatibility Metal
allocator path. A training preflight must report which API path is present and
fail closed if `mx.set_wired_limit` is unavailable or if no supported memory
limit setter is available.

Use `mx.device_info()["memory_size"]` for total device memory and
`mx.device_info()["max_recommended_working_set_size"]` for the system wired
limit guidance. Keep the wired limit strictly below total memory.
Do not raise `iogpu.wired_limit_mb` inside a training script; if an operator
changes the system wired limit, record it in the receipt and re-run the
preflight.

### M0.6 Memory Receipt Gate

For M0.6 (`cppmega-mlx-t8f.6`), the dev-128 arithmetic target is:

- `total_memory_bytes = 137438953472`
- `wired_limit_bytes = int(0.70 * total) = 96207267430`
- `metal_limit_bytes = int(0.85 * total) = 116823110451`
- strict peak-memory threshold `< int(0.75 * total) = 103079215104`

The receipt at `bench/baselines/m06_memory.json` is allowed to exist as a
partial/blocker receipt while upstream M0.4 dependencies are still open. It must
not claim full M0.6 acceptance until a 100-step `local_gb10_quarter` run with
AdamW, grad-checkpoint, documented memory-limit application, recorded
`mx.clear_cache()` cadence, and a peak-memory profile below the threshold has
been captured. The documented memory-limit application must include previous
limit values from `mx.set_wired_limit` and the selected Metal memory-limit API
path (`mx.set_memory_limit` or `mx.metal.set_memory_limit`) so the receipt can
distinguish applied limits from arithmetic-only planning. A full-acceptance
receipt must also record the exact command/provenance for the target run,
including `local_gb10_quarter`, `--grad-checkpoint`,
`--apply-memory-limit-plan`, and `--clear-cache-every-steps`, and each
`mx.clear_cache()` event must report the configured cadence. It must also record
the local runtime stack: MLX version, MLX-Metal version when installed, default
MLX device, Metal availability, macOS/platform fields, and `mx.device_info()`
including device name and memory size. Missing command, cadence, or
runtime-stack evidence keeps the receipt partial even if the arithmetic and
peak-memory gates pass.

## Thermal And Power Caveat

`powermetrics` is a reporting tool, not a correctness oracle. A readiness
receipt should record whether `powermetrics` is present and, for performance
receipts, should capture a bounded sample window outside the Python process.

Thermal throttling invalidates performance receipts. If throttling, low-power
mode, or unstable package power is observed, keep correctness smoke results
separate from throughput rows and do not publish tokens/sec, MFU, M4-vs-GB10, or
speedup claims from that run.

## Distributed And JACCL

`mlx.core.distributed` and `mx.distributed.is_available("jaccl")` are
future measured-only inputs for this repo.
Availability means the backend can be selected by MLX on the host;
it does not prove that cppmega.mlx implements Megatron TP/PP/VPP/EP/SP,
distributed optimizer behavior, ZeRO/FSDP semantics, rank-deterministic
checkpointing, or production launch parity.

Before any distributed cppmega.mlx claim, require a separate measured receipt
with:

- MLX version and backend selected by `mx.distributed.init(backend=...)`.
- `mlx.launch` command, hostfile, rank count, and world size.
- For JACCL, macOS version, Thunderbolt 5 RDMA enablement, `ibv_devices`
  output, fully connected mesh evidence, and `MLX_METAL_FAST_SYNCH` setting.
- Correctness parity for the intended local operation and communication
  throughput for the exact host topology.
- Explicit statement that the result is MLX distributed evidence only,
  not distributed Megatron parity.

Until those receipts exist, JACCL remains a future backend candidate and
all distributed training readiness is `NOT READY`.

## Source Notes

- Official MLX memory docs list `mlx.core.set_wired_limit` and
  `mlx.core.set_memory_limit`; `set_wired_limit` is macOS 15+ oriented and the
  wired limit must remain below total memory.
- Official MLX distributed docs list `ring`, `jaccl`, `mpi`, and `nccl`
  backend selection, with JACCL tied to macOS 26.2+ Thunderbolt 5 RDMA setup and
  a fully connected mesh.
- Official MLX-LM large-model notes describe wired memory behavior on macOS 15+
  and the operator-controlled `sudo sysctl iogpu.wired_limit_mb=N` setting.
