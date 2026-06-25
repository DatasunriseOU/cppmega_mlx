"""NUMERICAL PROBE (no production edit, no GPU dispatch -- pure fp32 numpy sim of
the path-c reverse-scan h re-materialization). Answers the design question:

  Q1 (cumulative underflow): at s4096, what is the magnitude range of the
      CUMULATIVE log-decay dA_cumsum = sum_{0..t} A*dt, and exp2(dA_cumsum) /
      its inverse 1/exp(dA_cumsum)? Does the FULL-cumulative inverse underflow
      /overflow in fp32 -> i.e. is the snapshot needed to avoid forming it?

  Q2 (adjacent-diff / per-step remat): the single-dispatch bwd_simd
      (mamba3_path_c.py:1414) re-materializes h_prev = (h_t - x*B)*inv_decay
      with inv_decay = 1/decay, decay = exp(A[t]*dt[t]) -- the BOUNDED per-step
      (adjacent-diff) decay, NEVER exp2 of the cumulative. Does this per-step
      remat stay bounded + bit-accurate vs the snapshot-cached forward h at
      s4096? And -- per the kernel docstring (:1259) -- when does decay->0 make
      inv_decay=1/decay blow up (0*inf NaN)?

We sweep TWO regimes:
  * BENIGN (the probe_fp32_bwd_simd nam56r config): A in [-1,0], dt in [0,0.05]
    -> A*dt in [-0.05,0] -> decay in [~0.95,1].
  * STRESS: drive A*dt strongly negative (large |A|, large dt) so decay -> 0,
    the regime the snapshot docstring warns about ("real bf16 weights drive
    decay to zero -> inverse walk 0*inf NaN").

All math is fp32 to mirror the kernel accum_dtype. memguard 70 mandatory.
RULE #1: no fallback; report the measured numbers honestly.
"""
from __future__ import annotations
import os, sys, threading, time

_MEMGUARD_LIMIT_KB = 70 * 1024 * 1024
_PEAK = 0
def _rss_kb():
    import resource
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024)
def _memguard():
    global _PEAK
    while True:
        r = _rss_kb()
        if r > _PEAK: _PEAK = r
        if r > _MEMGUARD_LIMIT_KB:
            sys.stderr.write(f"[memguard70] KILL rss_kb={r}\n"); sys.stderr.flush(); os._exit(137)
        time.sleep(0.25)
threading.Thread(target=_memguard, daemon=True).start()

import numpy as np

F32 = np.float32
SEQ = 4096
STATE = 64
np.random.seed(0)


def run_regime(name, A_scale, dt_lo, dt_hi):
    """Simulate ONE (b,h,p) lane forward scan in fp32, then probe remat."""
    rng = np.random.RandomState(1234)
    # per-step A*dt (the log-decay increment). A in [-A_scale, 0], dt in [dt_lo,dt_hi].
    A = (-rng.rand(SEQ).astype(F32) * F32(A_scale))           # <= 0
    dt = (rng.rand(SEQ).astype(F32) * F32(dt_hi - dt_lo) + F32(dt_lo)).astype(F32)
    logdecay = (A * dt).astype(F32)            # per-step A[t]*dt[t]  (<=0)
    decay = np.exp(logdecay).astype(F32)       # per-step bounded decay in (0,1]
    x = (rng.randn(SEQ).astype(F32) * F32(0.1))
    Bv = (rng.randn(SEQ, STATE).astype(F32) * F32(0.1))

    # ---- forward scan (fp32), cache EVERY boundary h_t (the snapshot) ----
    h = np.zeros(STATE, dtype=F32)
    h_snap = np.zeros((SEQ + 1, STATE), dtype=F32)
    h_snap[0] = h
    for t in range(SEQ):
        h = (decay[t] * h + x[t] * Bv[t]).astype(F32)
        h_snap[t + 1] = h

    # ---- Q1: cumulative log-decay and its exp / inverse ----
    # dA_cumsum[t] = sum_{0..t} logdecay  (fp32 running sum, как в forward)
    dA_cumsum = np.cumsum(logdecay.astype(np.float64)).astype(F32)  # cumulative (<=0, monotone down)
    cum_min = float(dA_cumsum.min()); cum_max = float(dA_cumsum.max())
    # forward cumulative decay exp(dA_cumsum) (->0 underflow) and inverse 1/exp (->inf overflow)
    with np.errstate(over="ignore", under="ignore"):
        exp_cum = np.exp(dA_cumsum.astype(F32)).astype(F32)            # -> 0
        inv_exp_cum = (F32(1.0) / exp_cum).astype(F32)                 # -> inf
        exp2_cum = np.exp2(dA_cumsum.astype(F32)).astype(F32)
    n_exp_underflow = int(np.sum(exp_cum == 0.0))
    n_inv_overflow = int(np.sum(~np.isfinite(inv_exp_cum)))
    # first step index where the cumulative-inverse overflows fp32 (becomes inf)
    inv_inf_idx = int(np.argmax(~np.isfinite(inv_exp_cum))) if n_inv_overflow else -1
    exp_zero_idx = int(np.argmax(exp_cum == 0.0)) if n_exp_underflow else -1

    # ---- Q2: per-step (adjacent-diff) remat that bwd_simd ACTUALLY does ----
    # walk h backwards from h_snap[SEQ] using ONLY per-step inv_decay = 1/decay[t]:
    #   h_prev = (h_t - x[t]*B[t]) * inv_decay[t]
    inv_decay = (F32(1.0) / decay).astype(F32)     # per-step (bounded unless decay->0)
    inv_decay_max = float(np.max(inv_decay))
    inv_decay_finite = bool(np.all(np.isfinite(inv_decay)))
    decay_min = float(np.min(decay))
    n_decay_zero = int(np.sum(decay == 0.0))       # decay underflowed to exactly 0 -> inv = inf

    h_back = h_snap[SEQ].astype(F32).copy()
    remat_err = np.zeros(SEQ, dtype=np.float64)
    remat_finite = True
    for r in range(SEQ):
        t = SEQ - 1 - r
        h_prev = ((h_back - x[t] * Bv[t]).astype(F32) * inv_decay[t]).astype(F32)
        # compare reconstructed h_prev vs the cached forward boundary h_snap[t]
        err = np.abs(h_prev.astype(np.float64) - h_snap[t].astype(np.float64)).max()
        remat_err[t] = err
        if not np.all(np.isfinite(h_prev)):
            remat_finite = False
        h_back = h_prev
    remat_worst = float(np.nanmax(remat_err))
    # relative error vs the magnitude of the cached h (avoid div-by-0)
    denom = np.maximum(np.abs(h_snap[:SEQ]).max(axis=1), F32(1e-20))
    rel = (remat_err / denom.astype(np.float64))
    remat_rel_worst = float(np.nanmax(rel))

    print(f"\n===== REGIME [{name}] A in [-{A_scale},0], dt in [{dt_lo},{dt_hi}] =====")
    print(f"  per-step logdecay (A*dt): min={float(logdecay.min()):.4e} max={float(logdecay.max()):.4e}")
    print(f"  per-step decay=exp(A*dt): min={decay_min:.6e} max={float(decay.max()):.6f}  (decay==0 count={n_decay_zero})")
    print(f"  per-step inv_decay=1/decay: max={inv_decay_max:.6e} all_finite={inv_decay_finite}")
    print(f"  [Q1 CUMULATIVE] dA_cumsum: min={cum_min:.4e} max={cum_max:.4e}")
    print(f"  [Q1] exp(dA_cumsum): min={float(exp_cum.min()):.4e} ; ==0 (underflow) count={n_exp_underflow}"
          + (f" first@t={exp_zero_idx}" if exp_zero_idx>=0 else ""))
    print(f"  [Q1] 1/exp(dA_cumsum) (full-cumulative INVERSE): non-finite (overflow) count={n_inv_overflow}"
          + (f" first@t={inv_inf_idx}" if inv_inf_idx>=0 else "") + f" ; max_finite={float(np.nanmax(inv_exp_cum[np.isfinite(inv_exp_cum)])) if np.any(np.isfinite(inv_exp_cum)) else float('nan'):.4e}")
    print(f"  [Q2 ADJ-DIFF REMAT] per-step h_prev=(h-xB)*inv_decay vs snapshot h:")
    print(f"        finite={remat_finite}  worst_abs={remat_worst:.4e}  worst_rel={remat_rel_worst:.4e}")
    return {
        "name": name, "decay_min": decay_min, "inv_decay_max": inv_decay_max,
        "inv_decay_finite": inv_decay_finite, "n_decay_zero": n_decay_zero,
        "cum_min": cum_min, "n_exp_underflow": n_exp_underflow,
        "n_inv_overflow": n_inv_overflow, "remat_finite": remat_finite,
        "remat_worst": remat_worst, "remat_rel_worst": remat_rel_worst,
    }


print(f"SEQ={SEQ} STATE={STATE} (fp32 sim of path-c reverse-scan remat)")
results = []
# benign nam56r-like config (probe_fp32_bwd_simd): decay ~ [0.95,1]
results.append(run_regime("BENIGN_nam56r", A_scale=1.0, dt_lo=0.0, dt_hi=0.05))
# moderate: decay can reach ~exp(-0.5)
results.append(run_regime("MODERATE", A_scale=2.0, dt_lo=0.0, dt_hi=0.25))
# stress: large |A|*dt so per-step decay -> tiny -> inv_decay huge (docstring regime)
results.append(run_regime("STRESS_decay0", A_scale=16.0, dt_lo=0.1, dt_hi=2.0))
# extreme: push per-step decay to fp32 underflow (decay==0 -> inv=inf)
results.append(run_regime("EXTREME_underflow", A_scale=128.0, dt_lo=1.0, dt_hi=4.0))

print(f"\nPEAK_RSS_KB={_PEAK} (~{_PEAK/1048576:.3f}GB) memguard70=ON")
print("\n==== SUMMARY ====")
for r in results:
    print(f"  {r['name']:20s} decay_min={r['decay_min']:.3e} inv_decay_max={r['inv_decay_max']:.3e} "
          f"cum_min={r['cum_min']:.2e} exp_underflow={r['n_exp_underflow']} inv_overflow={r['n_inv_overflow']} "
          f"| remat finite={r['remat_finite']} worst_abs={r['remat_worst']:.2e} worst_rel={r['remat_rel_worst']:.2e}")
print("RC=0")
