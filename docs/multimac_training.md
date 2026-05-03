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
