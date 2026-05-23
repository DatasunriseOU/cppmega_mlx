# Multi-node throughput receipt — BLOCKED on peer-48 hardware

**Status**: BLOCKED (external)
**bd**: tracks under `cppmega-mlx-bpge` (V7-Q08.3 sub-task)
**Last updated**: 2026-05-23

## What the receipt would prove

A real cross-Mac ZeRO-1 wall-clock receipt showing:
- `world_size = 2` across two distinct Mac hosts (dev-128 + peer-48)
- bf16 + Lion or AdamW + grad-checkpoint
- N-step parity vs single-host loopback (loss within 1% relative)
- Per-rank peak memory ≤ documented headroom (`docs/multimac_training.md` §"Phase 2")
- `gradient_reduce_ms` profile under TB5 JACCL (or ring fallback if JACCL unavailable)

## What's already shipped (loopback CAN run today)

- `cppmega_mlx.cli.smoke_zero1` CLI (V7-Q05, commit `19cb7a8`)
- `scripts/bench_zero1_loopback.py` with `--simulate` and real
  `mlx.launch -n 2 --hosts 127.0.0.1` paths
- `bench/baselines/zero1_loopback_2proc_m4.json` baseline
- `DistributedZeRO1Optimizer` wrapper with real `mx.distributed`
  all_sum / all_gather collectives
- 17 gotcha rules (`cppmega_v4/parallelism/gotcha_checker.py`)
- 8 device kinds + 5 sharding factories
- Loopback receipt invariant: `|loss_W2 - loss_W1| / loss_W1 < 1%`

## What's missing (hardware blocker)

Per `docs/multimac_training.md` lines 114-129 + Stream F step 120:
- peer-48 hardware not yet connected (no SSH, no TB5 cable mapped)
- macOS 26.2 + M4 family availability checklist not yet ticked
- No JACCL throughput baseline numbers
- Single-host loopback uses TCP via 127.0.0.1; cross-host uses real NIC

## Unblock procedure (when hardware is up)

1. Verify peer-48 SSH access from dev-128 (passwordless).
2. Verify TB5 cable + thunderbolt-net interface visible on both hosts.
3. Run smoke:
   ```bash
   python -m cppmega_mlx.cli.smoke_zero1 \
     --hosts dev-128.local,peer-48.local \
     --num-steps 20 \
     --out bench/baselines/zero1_multimac_2host.json
   ```
4. Compare `losses[-1]` against `bench/baselines/zero1_loopback_2proc_m4.json` —
   should be within 1% relative.
5. Update this doc with throughput numbers + close
   `cppmega-mlx-bpge` Q08.3.

## Why this is not a regression risk

The CLI, the wrapper math, and the gotcha checker are all
production-tested via loopback. The only missing piece is the physical
network — the code path is identical between `127.0.0.1,127.0.0.1` and
`dev-128.local,peer-48.local`. Hardware-blocked, not code-blocked.
