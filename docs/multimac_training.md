# Multi-Mac topology and roles for cppmega.mlx

**Status:** scope and role-definition stub. Full playbook (run scripts, network topology details, JACCL/ring decision tree, throughput receipts) lands at Stream F step 120. Until then, this file is the source of truth for **which Mac plays which role**.

This is planning material, not a production-readiness claim.

---

## Hardware inventory

| Node              | Model                       | Chip      | RAM    | macOS  | TB                      | JACCL prereqs      |
| ----------------- | --------------------------- | --------- | ------ | ------ | ----------------------- | ------------------ |
| dev-128           | Mac Studio                  | M4 Max    | 128 GB | 26.4.1 | 4× TB5 (up to 120 Gb/s) | ✓ all met          |
| peer-48           | TBD (M4 Pro or M4 Max base) | M4 family | 48 GB  | TBD    | TBD                     | TBD until verified |
| dev-128b (future) | Mac Studio                  | M4 Max    | 128 GB | TBD    | TBD                     | TBD when added     |

Run system_profiler SPHardwareDataType SPThunderboltDataType && sw_vers on each node before assigning roles; record the output here.

---

## Role definitions

### inference_scout (default role for any non-128 GB peer)

Purpose: serve inference, run continuous validation, host the speculative-decode draft model. Does **not** participate in training.

Workloads it handles:
- q4 quantized inference of mini (1.2B at q4 ≈ 0.7 GB weights + KV) with full headroom for batch and prompt cache.
- Eval / parity / regression CI runs against fresh dev-128 checkpoints.
- Speculative-decode draft model server for Stream I (steps 167–169): vanilla acceptance-rejection, EAGLE-2 draft, or MTP self-spec — pick per workload.
- Long-context benchmarks (NIAH, RULER) on KV-q4 path.

Memory budget on a 48 GB peer:
- Quantized weights + KV: 5–10 GB
- Inference activations + batch: 5–15 GB
- macOS + MLX cache: 8–12 GB (run **headless** — no GUI session, sshd only)
- Comfortable headroom: 15+ GB

### training_peer (only on 128 GB nodes by default; 48 GB peer in Lion+ZeRO-1 smoke mode)

Purpose: hold a rank of a distributed training run.

Per-rank memory budget for local_gb10_quarter mini (1.2B):

| Optimizer | Sharding                 | Per-rank memory | 128 GB? | 48 GB?     |
| --------- | ------------------------ | --------------- | ------- | ---------- |
| AdamW     | none (full DP replica)   | ~22–26 GB       | ✓       | ✓          |
| AdamW     | ZeRO-1 (opt state shard) | ~18–22 GB       | ✓       | ✓          |
| AdamW     | ZeRO-2 (grads + opt)     | ~16–20 GB       | ✓       | ✓          |
| Lion      | none (full DP replica)   | ~16–20 GB       | ✓       | ✓          |
| Lion      | ZeRO-1 (opt state shard) | ~12–14 GB       | ✓       | ✓ headless |
| Lion      | ZeRO-2 (grads + opt)     | ~10–12 GB       | ✓       | ✓          |

**The 48 GB peer is feasible as training_peer across all optimizer/sharding combinations once we reach the calibrated 1.2B size**; the previous 3.79B figure was wrong and overstated memory pressure by ~3×. Smoke configuration goal is to prove the distributed code path works end-to-end, not a production throughput target.

---

## Topology decisions

### Phase 1 — single-Mac M0 (current)
- dev-128 only.
- AdamW + grad-checkpoint, no distributed.

### Phase 2 — heterogeneous Stream F smoke (~1–2 weeks after M0 starts)
- dev-128 (training_peer) + peer-48 (training_peer for smoke, then inference_scout).
- Lion + ZeRO-1, batch=1, headless on peer-48.
- TB5 cable; JACCL if both nodes hit prereqs, ring fallback otherwise.
- Goal: prove mx.distributed + ZeRO-1 plumbing works on real hardware. Don't chase throughput parity with single-Mac AdamW.

### Phase 3 — homogeneous production (when dev-128b arrives)
- dev-128 + dev-128b as paired training_peers.
- AdamW + ZeRO-1 (or full DP if memory permits).
- peer-48 demoted to inference_scout permanent role.
- This is the path that lets us scale past mini.

---

## JACCL vs ring backend decision

| Both nodes meet TB5 + macOS ≥ 26.2 + M3 Ultra/M4 Pro/Max? | Backend                                             |
| --------------------------------------------------------- | --------------------------------------------------- |
| Yes                                                       | JACCL (RDMA over TB5, ~10× lower latency than ring) |
| No (TB4 cable, older macOS, mismatched chips)             | Ring backend over TCP/Thunderbolt                   |
| Can't init either                                         | Fail loudly; do not silently downgrade              |

peer-48 JACCL prereqs need verification at connection time. If the 48 GB unit is a MacBook Pro M4 Max, it has TB5; if it's an M4 Pro variant, check spec — TB ports vary across M4-family chips.

---

## Connection trigger and verification checklist

When peer-48 is connected, run this checklist before a distributed run:

1. system_profiler SPHardwareDataType SPThunderboltDataType > peer-48-hw.txt and update the inventory table above.
2. sw_vers > peer-48-os.txt. Confirm macOS ≥ 26.2 if JACCL is wanted.
3. TB5 cable physically connected, both ends seeing the link via system_profiler SPThunderboltDataType (link speed Up to 120 Gb/s).
4. ssh peer-48 'mlx version' matches the version pinned on dev-128.
5. mlx.distributed_config --hosts dev-128,peer-48 --backend auto reports the chosen backend (JACCL or ring).
6. Smoke: mlx.launch -n 2 --hosts dev-128,peer-48 -- python -m cppmega_mlx.cli.smoke_distributed runs to completion, emits a baseline row in bench/baselines/m4max_heterogeneous_2node.json.

Only after all six pass do we promote peer-48 to training_peer for actual Stream F runs. Until then, it remains inference_scout.

---

## Non-claims

- This document does not claim cppmega.mlx has reached distributed Megatron parity. Stream F is greenfield work in MLX.
- The 48 GB training_peer smoke proves the code path runs; it does not claim production-readiness or matched throughput vs CUDA/H200.
- JACCL throughput numbers cited elsewhere (e.g., "10× lower latency than ring") are external Apple/MLX references, not local receipts. Replace with bench/baselines/ rows once the rig is up.

---

## Runbook (V7-B-real)

The implementation pieces that back this document:

| Need                            | Code                                                              |
| ------------------------------- | ----------------------------------------------------------------- |
| World init + primitives         | `cppmega_v4/runtime/distributed.py`                               |
| TP Column/Row Linear            | `cppmega_v4/runtime/tp_proxy.py`                                  |
| PP 1F1B + GPipe schedule        | `cppmega_v4/runtime/pp_proxy.py`                                  |
| Process groups (intra/inter)    | `cppmega_v4/runtime/multi_node_topology.py:build_process_groups`  |
| Real all-reduce bench           | `scripts/bench_allreduce_distributed.py`                          |
| Multi-process launcher          | `scripts/launch_multi.py`                                         |
| Stage_train DP path             | `cppmega_v4/runner/stages.py` (uses `distributed.all_reduce` when `is_distributed()` is True) |

### Two-host smoke

```bash
# On dev-128:
python -m scripts.launch_multi --np 2 --backend mlx -- \
    python -m scripts.bench_allreduce_distributed --buf-mb 1 4 16

# Multi-host via mpirun + hostfile:
python -m scripts.launch_multi --np 2 --backend mpi \
    --hostfile peers.hostfile -- \
    python -m scripts.bench_allreduce_distributed --buf-mb 4 16 64
```

Expected: `world_size=2`, `backend in {mpi, ring, jaccl}`, `real=true`
in the emitted JSON. `reports/bench_allreduce_distributed_<ts>.csv`
carries per-buffer ms-per-iter for all-reduce, all-gather, and
reduce-scatter.

### Real DP training run

`stage_train` switches to real all-reduce automatically when the
launcher has produced `world_size > 1`. The same `pipeline.run` call
that worked single-process now mean-reduces grads across ranks every
step. Surface in `extras.distributed`:

```json
{
  "world_size": 2,
  "rank": 0,
  "backend": "mpi",
  "real": true
}
```

Single-process runs keep the legacy `fake_ranks` replay path
untouched, so existing CI passes.

### Megatron-style TP

```python
from cppmega_v4.runtime.tp_proxy import (
    ColumnParallelLinear, RowParallelLinear,
)

# When world_size==2, each rank stores half of the output dim.
qkv = ColumnParallelLinear(in_features=H, out_features=3 * H,
                            gather_output=False)
out = RowParallelLinear(in_features=H, out_features=H,
                         input_is_parallel=True)
```

`gather_output=False` paired with `input_is_parallel=True` is the
canonical Megatron pattern — the column output stays sharded, the row
input consumes the matching shard, and the only collectives needed
are the implicit `all_reduce` inside `RowParallelLinear` after its
local matmul.

### Pipeline parallelism

```python
from cppmega_v4.runtime.pp_proxy import (
    pipeline_forward_real, one_f_one_b_schedule,
)

out = pipeline_forward_real(
    x, stages=[stage0, stage1, stage2, stage3],
    num_microbatches=8, schedule="1f1b",
)
```

When `world_size == len(stages)` the call routes activations between
ranks via `distributed.send / recv_like`; otherwise it runs the same
schedule sequentially on one process for verification.
