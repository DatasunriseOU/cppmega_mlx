# ZeRO-1 multi-node smoke procedure (Stream F, peer-48 hand-off)

**Status:** procedure document. No 2-node receipt exists yet; this file
captures the exact steps to produce one once the 48 GB peer
(`docs/multimac_training.md` Phase 2 hardware) is connected. Until that
receipt lands, the
[`cppmega_mlx.training.distributed_optimizer.DistributedZeRO1Optimizer`](../cppmega_mlx/training/distributed_optimizer.py)
wrapper has only single-rank receipts plus simulated W=2 tests; see
`tests/test_distributed_zero1.py`.

This document is a planning aid, not a production-readiness claim.

---

## Pre-flight checklist (must all pass before step 1)

Run `docs/multimac_training.md` "Connection trigger and verification
checklist" first. Specifically:

1. `system_profiler SPHardwareDataType SPThunderboltDataType > peer-48-hw.txt`
   on peer-48; confirm TB5 ports if JACCL is wanted.
2. `sw_vers > peer-48-os.txt`; confirm macOS >= 26.2 if JACCL is wanted.
3. TB5 cable physically connected; both ends report "Up to 120 Gb/s".
4. `ssh peer-48 'mlx version'` matches the version pinned on dev-128.
5. `mlx.distributed_config --hosts dev-128,peer-48 --backend auto` reports
   the chosen backend (JACCL or ring).
6. Smoke baseline `bench/baselines/m4max_heterogeneous_2node.json` exists
   from the connection-time smoke (separate from this ZeRO-1 receipt).

If any step fails, do not proceed; peer-48 stays in `inference_scout` role.

---

## Step 1 -- launch a 2-rank ZeRO-1 toy run

```sh
# On dev-128, the orchestrator host.
mlx.launch \
    -n 2 \
    --hosts dev-128,peer-48 \
    --backend auto \
    -- python -m cppmega_mlx.cli.smoke_zero1 \
    --steps 100 \
    --batch-size 1 \
    --seq-len 1024 \
    --depth 13 \
    --vocab-size 65536 \
    --optimizer lion \
    --output bench/baselines/zero1_smoke_2node.json
```

The launcher binds rank 0 to dev-128 and rank 1 to peer-48; both nodes import
the same model graph and call:

```python
from cppmega_mlx.training import (
    DistributedZeRO1Optimizer,
    make_distributed_optimizer,
    make_lion,
)

optimizer = make_distributed_optimizer(make_lion(learning_rate=1e-4))
optimizer.init(model.trainable_parameters())
```

When `mx.distributed.init(backend='auto')` has already produced a 2-rank
group (which `mlx.launch -n 2` guarantees), the wrapper auto-detects
`world_size = 2` and `rank in {0, 1}` from the global group and only the
local-shard optimizer state is allocated.

Note: `cppmega_mlx.cli.smoke_zero1` is **not yet implemented**. It will be
added alongside the first real receipt; for now, the procedure is documented
ahead of the launcher script so we have a target signature.

---

## Step 2 -- verify per-rank peak memory budgets

While the smoke run executes, monitor:

- `dev-128`: `psutil.Process().memory_info().rss` under
  `~14 GB` ceiling (Lion + ZeRO-1 from `docs/multimac_training.md`'s 1.2B
  per-rank memory table).
- `peer-48`: same monitor, headless mode (`sudo launchctl unload
  /System/Library/LaunchDaemons/com.apple.coreservices.useractivityd.plist`
  to reduce GUI overhead) -- ceiling identical at ~14 GB.

Any rank exceeding 14 GB peak fails the receipt; investigate before
continuing to step 3.

---

## Step 3 -- collect the receipt

The smoke writes a baseline row matching the schema in
`cppmega_mlx.training.baselines.REQUIRED_BASELINE_ROW_KEYS`:

```json
{
    "hardware": "M4 Max 128 GB + M4 (variant) 48 GB pair",
    "commit": "<git rev-parse HEAD>",
    "dtype": "bfloat16",
    "batch_size": 1,
    "seq_len": 1024,
    "route": "lion_zero1",
    "model": "mini-1.2B",
    "mode": "distributed_zero1",
    "tokens_per_second": <measured>,
    "local_only": true,
    "gb10_parity_claim": false,
    "world_size": 2,
    "backend": "<jaccl_or_ring>",
    "per_rank_peak_memory_gb": [<rank0_peak>, <rank1_peak>],
    "zero1_state_bytes_per_rank": [<rank0_state>, <rank1_state>],
    "zero1_state_bytes_full_replica_estimate": <full>
}
```

The extra fields beyond the standard baseline schema (`world_size`,
`backend`, `per_rank_peak_memory_gb`, `zero1_state_bytes_*`) are receipt
metadata for this scaffold; if they enter the standard schema we can promote
them. Validate the row with `validate_baseline_row()` and archive via
`archive_baseline_row()`.

Save to `bench/baselines/zero1_smoke_2node.json`. Until that file exists, the
ZeRO-1 wrapper is **scaffold only**; do not claim distributed parity.

---

## Step 4 -- numerical sanity (optional, recommended)

Run a 20-step single-Mac Lion baseline at the same hyperparameters:

```sh
python -m cppmega_mlx.cli.smoke_zero1 \
    --steps 20 \
    --world-size 1 \
    --output bench/baselines/zero1_smoke_1node.json
```

Compare the final loss between the 1-node and 2-node runs. Per
`tests/test_distributed_zero1.py::test_zero1_simulation_w2_loss_matches_non_sharded_within_tolerance`
the expected relative error is < 1% (ZeRO-1 is mathematically exact;
deviations beyond 1% indicate either gradient-reduction bugs or RNG /
data-loader divergence between ranks).

---

## Failure modes (not yet observed; documented for the receipt run)

| Symptom                                  | Likely cause                     | Mitigation                                                        |
| ---------------------------------------- | -------------------------------- | ----------------------------------------------------------------- |
| Rank 1 peak memory > 14 GB                | peer-48 carrying full opt state  | Confirm `is_sharded == True` and `owned_param_names` is half-set  |
| `mx.distributed.all_sum` raises          | backend init failed              | Drop to ring; re-run mlx.distributed_config and check link        |
| Loss diverges between W=1 and W=2 runs   | grad reduce missing factor 1/W   | Re-check `_all_reduce_mean` scale factor                          |
| Hang on `_gather_full_params`            | uneven leaf count between ranks  | Verify `_shard_assignment(num_leaves, 2)` is symmetric            |

---

## Non-claims

- This procedure does not claim cppmega.mlx ZeRO-1 reaches Megatron
  `DistributedOptimizer` parity. The MLX wrapper is a small scaffold;
  Megatron's CUDA implementation has 50+ engineering details (bucketed
  contiguous grouping, NCCL fusion, optimizer-overlap) that the scaffold
  intentionally omits.
- The 1% loss-tolerance bound is a sanity check, not a numerical-equivalence
  proof. Real cross-node receipts may show larger drift due to floating-point
  reduction order; record observed drift and tighten the bound only if
  warranted.
- Until `bench/baselines/zero1_smoke_2node.json` exists, the wrapper status
  is "scaffold + single-rank receipts; multi-node receipt pending peer-48
  hardware".
