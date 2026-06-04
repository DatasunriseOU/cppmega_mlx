"""E2E step A/B harness: TE DelayedScaling(E4M3) fp8 transformer-block GEMMs vs bf16.

Measures the REALIZED end-to-end training-step effect of wiring the MEASURED-1.7x
TE fp8 GEMM (docs/RELAX-GRAPH-VS-MEGATRON.md §19/§20, R1 tensorwise) into the model's
dense transformer-block Linears (attention q/k/v/out, MoE gate/up/down), gated by
``CPPMEGA_FP8_LINEAR``. It runs the SAME local_gb10_quarter model fwd+bwd(+optimizer)
under the bf16 path (gate OFF) and the fp8 path (gate ON), times N steps each, and
reports median step time -> tok/s, MLX + torch-CUDA peak GB, and the final loss of
each arm so the GB10 phase sees a loss-parity sanity check.

This is the runnable e2e MEASURE harness for lever r1-e2e-wire. It is DISJOINT from
lever 1's fp8_matmul_path_c.py and from scratch/fp8_gemm_microbench.py: this one
exercises the FULL model step, not a standalone GEMM.

RULE #1: when the fp8 arm is selected (gate ON), every wired Linear that clears the
fp8 floor goes through the ONE TE fp8 path; a TE/bridge failure RAISES (it is NOT
caught here and NOT degraded to bf16). The bf16 arm (gate OFF) is byte-identical to
the un-wired model. We report MEASURED numbers only; no fabricated tok/s.

Usage (on gb10):
  python scripts/bench_fp8_e2e_step_20260604.py --batch 1 --seq 4096 \
      --steps 6 --warmup 2 --optimizer
  # A/B both arms (default). To run a single arm: --arm bf16  or  --arm fp8.
  # Force fp8 on EVERY Linear (stress small-shape case): --fp8-floor 0.

SAFETY (the GB10 phase enforces exclusive ownership + the >105 GB budget). This
harness only builds the model and runs the configured steps; pick --batch/--seq to
stay inside the budget (bs1 @ default layers is safe; project to bs4 = EXTRAPOLATION).
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time

# RULE #1 (lever dlpack-fix): bound torch/TE's cudaMallocAsync pool so it does NOT
# reserve-and-retain unified memory that contends with MLX's pool on the single
# GB10 117 GB unified part (the §21 fp8-backward OOM root cause). This env MUST be
# set BEFORE the first ``import torch`` anywhere in the process; torch reads it once
# at CUDA-context init. release_threshold:0 makes torch's stream-ordered pool return
# reserved-but-idle memory instead of hoarding it, so the combined MLX+torch+TE
# reserved set fits. Overridable via CPPMEGA_TORCH_ALLOC_CONF for A/B. This is NOT a
# fallback — it only makes the zero-copy fp8 bridge FIT; the bridge still RAISES on a
# genuinely-uncrossable tensor.
if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = os.environ.get(
        "CPPMEGA_TORCH_ALLOC_CONF",
        "backend:cudaMallocAsync,release_threshold:0",
    )

# SECONDARY (§22) export-doubling fix: when CPPMEGA_FP8_BRIDGE_UNIFIED is requested,
# propagate it to MLX_CUDA_BRIDGE_UNIFIED so the MLX CUDA allocator (rebuilt with the
# allocator.cpp change) allocates bridge arrays as unified up front — killing the
# per-tensor cudaMallocManaged doubling that move_to_unified_memory does on each
# bridge crossing. MUST be set BEFORE ``import mlx.core`` (MLX reads it once at
# allocator construction). Default OFF so the bf16 arm and un-rebuilt MLX are
# unchanged; the GB10 phase opts in after rebuilding MLX. RULE #1: managed memory is
# the same bytes, device-resident — correctness-neutral, never a degrade.
_bridge_unified = os.environ.get("CPPMEGA_FP8_BRIDGE_UNIFIED", "").strip().lower()
if _bridge_unified in ("1", "true", "yes", "on") and "MLX_CUDA_BRIDGE_UNIFIED" not in os.environ:
    os.environ["MLX_CUDA_BRIDGE_UNIFIED"] = "1"

# RULE #1: fix the NVRTC loader BEFORE torch / TE are imported anywhere (gb10).
# Safe no-op off-gb10. Must precede the first transformer_engine import (which
# happens lazily inside fp8_te_linear when the fp8 arm runs).
from cppmega_mlx._gb10_nvrtc_env import ensure_nvrtc_builtins_path

ensure_nvrtc_builtins_path()

import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402
import numpy as np  # noqa: E402

from cppmega_mlx.nn._tilelang.fp8_te_linear import FP8_LINEAR_ENV  # noqa: E402
from cppmega_mlx.recipes.model_factory import local_gb10_quarter  # noqa: E402
from cppmega_mlx.training.loss import next_token_cut_cross_entropy  # noqa: E402


def _peak_gb() -> float:
    fn = getattr(mx, "get_peak_memory", None)
    return float(fn()) / (1024**3) if fn else float("nan")


def _reset_peak() -> None:
    if hasattr(mx, "reset_peak_memory"):
        mx.reset_peak_memory()


def _torch_peaks():
    try:
        import torch

        if not torch.cuda.is_available():
            return None, None
        return (
            round(float(torch.cuda.max_memory_allocated()) / (1024**3), 3),
            round(float(torch.cuda.max_memory_reserved()) / (1024**3), 3),
        )
    except Exception:
        return None, None


def _torch_reset_peak() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


def _budget_mlx_torch() -> None:
    """Wire the §22 shared-pool fix + budget split for the fp8 arm.

    §22 root cause: the fp8-backward OOM is dual-pool reservation contention (MLX's
    AND torch/TE's cudaMallocAsync pools each reserve-and-retain the 117 GB unified
    pool), not activation size. The PRIMARY fix is a SINGLE shared pool; the budget
    knobs below are complementary bounds (NOT fallbacks):

      (0, PRIMARY) ``install_shared_mempool()`` — route the device-current pool to
          MLX's default pool (``cudaDeviceSetMemPool``) so MLX and torch/TE draw from
          ONE pool: one reservation, no double-count. Done ONCE here AFTER torch's
          CUDA context is initialized (so torch's later allocations honor it). RAISES
          if the runtime lacks cudaDeviceSetMemPool (then the trim-both fallback in
          relieve_bridge_memory_pressure still runs each backward). Gated
          CPPMEGA_FP8_SHARED_POOL (default ON).
      (1) ``mx.set_cache_limit(0)`` — MLX returns freed buffers immediately so the
          bridge trim (``relieve_bridge_memory_pressure``) can reclaim them, instead
          of MLX hoarding them in its cache.
      (2) ``torch.cuda.set_per_process_memory_fraction(F)`` — caps torch/TE's share
          so MLX + torch/TE + the transient per-tensor export doubling sum < the
          physical 117 GB part. F is set conservatively (env CPPMEGA_TORCH_MEM_FRAC,
          default 0.45 = ~52 GB on a 117 GB box) leaving MLX the majority.
      (3) ``mx.set_memory_limit`` honored via CPPMEGA_MLX_MEMORY_LIMIT_GB (in main).
    RULE #1: if the split is too tight the bridge still RAISES with where+what — it
    never host-copies or degrades to bf16.
    """

    set_cache = getattr(mx, "set_cache_limit", None)
    if callable(set_cache):
        set_cache(0)

    from cppmega_mlx.nn._tilelang._cuda_zerocopy import (
        install_shared_mempool,
        shared_pool_enabled,
    )

    # §24 MEASURED root cause: on this MLX(0.32.0.dev+6680d75a)+torch build the
    # device's CURRENT cudaMallocAsync pool ALREADY EQUALS the device DEFAULT pool,
    # and BOTH MLX's and torch/TE's plain cudaMallocAsync(&p,size,stream) draw from
    # it (MEASURED: a 2 GB torch alloc + a 2 GB MLX alloc grow the SAME default-pool
    # ReservedMemCurrent 2->4 GB; a 40 GB MLX + 10 GB torch co-allocation reaches 50
    # GB on the one pool with NO double-count). So the §22 "two pools double-reserve"
    # premise does NOT hold here — there is already ONE shared pool.
    #
    # The persistent fp8-backward OOM was NOT dual-pool contention; it was
    # ``torch.cuda.set_per_process_memory_fraction(F)`` POISONING that shared pool:
    # MEASURED, the fraction call flips the SHARED default pool's
    # cudaMemPoolAttrReleaseThreshold 0 -> UINT64_MAX (retain-everything) AND installs
    # a hard per-process ceiling that MLX's ``mx.eval`` growth — drawing from the SAME
    # pool — trips, OOMing at fp8_te_linear.py:321 even at seq128/frac0.95. Dropping
    # the fraction cap and re-asserting ReleaseThreshold=0 lets the shared pool grow
    # for both allocators (MEASURED: the fp8 backward then COMPLETES). The fraction
    # split is therefore ONLY meaningful in the genuine dual-pool fallback
    # (CPPMEGA_FP8_SHARED_POOL=0); with the shared pool it starves MLX. RULE #1: this
    # is not a degrade — it removes a self-inflicted cap so the ONE zero-copy path FITS.
    try:
        import torch

        if torch.cuda.is_available():
            # Initialize torch's CUDA context so its cudaMallocAsync pool exists
            # BEFORE we (re)assert the shared-pool release threshold below.
            torch.cuda.init()
            if not shared_pool_enabled():
                # DUAL-POOL FALLBACK only: cap torch's share so the two distinct pools
                # fit. With the shared pool this cap poisons the one pool (see above),
                # so it is skipped.
                frac = float(os.environ.get("CPPMEGA_TORCH_MEM_FRAC", "0.45"))
                torch.cuda.set_per_process_memory_fraction(frac)
    except Exception as exc:  # RULE #1: a budget-cap failure is a config bug, surface it.
        raise RuntimeError(
            "bench_fp8_e2e: could not apply the torch CUDA memory budget for the fp8 "
            f"arm: {type(exc).__name__}: {exc}. The fp8 bridge needs the MLX+torch "
            "pool config to fit the GB10 unified pool."
        ) from exc

    # PRIMARY §22/§24 shared-pool install (once, up front, AFTER torch's CUDA context
    # so it re-asserts ReleaseThreshold=0 on the shared pool even if torch's init
    # bumped it to UINT64_MAX). install_shared_mempool() is a no-op repoint here (the
    # current pool already IS the default pool) BUT its ReleaseThreshold=0 re-assert is
    # the load-bearing step that undoes any retain-everything torch left behind.
    if shared_pool_enabled():
        try:
            install_shared_mempool()
        except Exception as exc:  # RULE #1: surface where+what; do NOT silently skip.
            raise RuntimeError(
                "bench_fp8_e2e: could not install the single shared cudaMemPool for "
                f"the fp8 arm (the §22 PRIMARY fix): {type(exc).__name__}: {exc}. "
                "Set CPPMEGA_FP8_SHARED_POOL=0 to fall to the trim-both path (and "
                "report that honestly), or rebuild MLX with the allocator shared-pool "
                "change if cudaDeviceSetMemPool is unavailable."
            ) from exc


def _run_arm(arm: str, args, tokens) -> dict:
    """Run one arm (bf16 | fp8) for warmup+steps timed steps; return the metrics."""

    fp8_on = arm == "fp8"
    if fp8_on:
        os.environ[FP8_LINEAR_ENV] = "1"
        # Floor controls which Linears are eligible (see fp8_te_linear). 0 = every
        # wired Linear takes fp8 (stress the small MoE-expert shapes).
        if args.fp8_floor is not None:
            os.environ["CPPMEGA_FP8_LINEAR_MIN_K"] = str(args.fp8_floor)
            os.environ["CPPMEGA_FP8_LINEAR_MIN_N"] = str(args.fp8_floor)
            os.environ["CPPMEGA_FP8_LINEAR_MIN_M"] = str(args.fp8_floor)
        # lever dlpack-fix: split the single GB10 unified pool between MLX and
        # torch/TE so neither hoards. (1) Stop MLX from caching freed buffers so the
        # bridge trim can return them. (2) Cap torch's share so MLX + torch/TE + the
        # transient export doubling sum < physical. These are bounds, not fallbacks:
        # if the split is too tight the bridge RAISES (no host-copy degrade).
        _budget_mlx_torch()
    else:
        os.environ.pop(FP8_LINEAR_ENV, None)

    model = local_gb10_quarter(
        dtype=mx.bfloat16, grad_checkpoint=args.grad_checkpoint
    )
    mx.eval(model.parameters())
    mx.synchronize()

    optimizer = None
    if args.optimizer:
        from cppmega_mlx.training.optimizers import make_adamw

        optimizer = make_adamw(learning_rate=1e-4, weight_decay=0.0)
        optimizer.init(model.trainable_parameters())
        mx.eval(model.parameters(), optimizer.state)

    batch = {"tokens": tokens}

    def loss_fn(m, b):
        return next_token_cut_cross_entropy(m, b, eval_chunks=False)

    vag = nn.value_and_grad(model, loss_fn)

    def one_step():
        (loss, ntok), grads = vag(model, batch)
        if optimizer is not None:
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state, loss, ntok)
        else:
            mx.eval(loss, ntok, grads)
        mx.synchronize()
        return float(loss)

    # Warmup (also lets DelayedScaling amax history converge for the fp8 arm).
    last_loss = float("nan")
    for _ in range(max(0, args.warmup)):
        last_loss = one_step()

    _torch_reset_peak()
    _reset_peak()

    step_times = []
    for _ in range(max(1, args.steps)):
        t0 = time.time()
        last_loss = one_step()
        step_times.append(time.time() - t0)

    median_s = statistics.median(step_times)
    tok_per_step = args.batch * args.seq
    tok_s = tok_per_step / median_s if median_s > 0 else float("nan")
    torch_peak, torch_reserved = _torch_peaks()

    return {
        "arm": arm,
        "fp8_enabled": fp8_on,
        "fp8_floor": (args.fp8_floor if fp8_on else None),
        "batch": args.batch,
        "seq": args.seq,
        "tokens_per_step": tok_per_step,
        "optimizer": bool(args.optimizer),
        "grad_checkpoint": bool(args.grad_checkpoint),
        "warmup": args.warmup,
        "steps": args.steps,
        "median_step_s": round(median_s, 4),
        "p10_step_s": round(min(step_times), 4),
        "p90_step_s": round(max(step_times), 4),
        "tok_per_s": round(tok_s, 1),
        "mlx_peak_gb": round(_peak_gb(), 3),
        "torch_cuda_peak_gb": torch_peak,
        "torch_cuda_reserved_gb": torch_reserved,
        "final_loss": round(last_loss, 5),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--seq", type=int, default=4096)
    ap.add_argument("--vocab", type=int, default=65_536)
    ap.add_argument("--steps", type=int, default=6, help="timed steps per arm")
    ap.add_argument("--warmup", type=int, default=2, help="untimed warmup steps")
    ap.add_argument("--optimizer", action="store_true", help="run AdamW update too")
    ap.add_argument(
        "--grad-checkpoint",
        dest="grad_checkpoint",
        action="store_true",
        default=False,
    )
    ap.add_argument(
        "--arm",
        choices=["both", "bf16", "fp8"],
        default="both",
        help="which arm(s) to run (default both = A/B)",
    )
    ap.add_argument(
        "--fp8-floor",
        dest="fp8_floor",
        type=int,
        default=None,
        help="set MIN_K/N/M eligibility floor for the fp8 arm (0 = every Linear)",
    )
    args = ap.parse_args()

    cap_gb = os.environ.get("CPPMEGA_MLX_MEMORY_LIMIT_GB", "").strip()
    if cap_gb and hasattr(mx, "set_memory_limit"):
        mx.set_memory_limit(int(float(cap_gb) * (1024**3)))

    rng = np.random.RandomState(0)
    tokens = mx.array(
        rng.randint(0, args.vocab, size=(args.batch, args.seq + 1)).astype(np.int32)
    )

    arms = ["bf16", "fp8"] if args.arm == "both" else [args.arm]
    results = {}
    for arm in arms:
        res = _run_arm(arm, args, tokens)
        results[arm] = res
        print("FP8_E2E_ARM " + json.dumps(res), flush=True)

    summary = {"arms": results}
    if "bf16" in results and "fp8" in results:
        b = results["bf16"]["tok_per_s"]
        f = results["fp8"]["tok_per_s"]
        summary["fp8_vs_bf16_tok_s_speedup"] = (
            round(f / b, 4) if b and b > 0 else None
        )
        summary["fp8_vs_bf16_loss_delta"] = round(
            results["fp8"]["final_loss"] - results["bf16"]["final_loss"], 5
        )
        bp = results["bf16"]["mlx_peak_gb"]
        fp = results["fp8"]["mlx_peak_gb"]
        summary["fp8_vs_bf16_mlx_peak_ratio"] = (
            round(fp / bp, 4) if bp and bp > 0 else None
        )
    print("FP8_E2E_SUMMARY " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
