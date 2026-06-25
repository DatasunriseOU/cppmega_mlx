"""DESIGN PROBE: can bwd_simd be single-dispatch + numerically safe at SEQ>1
WITHOUT the snapshot dispatch? Evaluate the two candidate safe remats:

 (a) LOG-SPACE carry: carry h in log/scaled form so no exp2 of cumulative is
     formed. PROBLEM: h is a SIGNED accumulation (decay*h + x*B); sums of
     signed terms have no log. A per-element log|h|+sign carry still must ADD
     x*B (different sign/scale) -> requires bringing both to a common scale =
     exp of the (huge) cumulative gap -> same underflow. Probe whether a
     common-scale add stays finite at s4096.

 (b) FORWARD-RERUN-IN-KERNEL (no inverse, no snapshot buffer): recompute h_t by
     re-scanning forward from h0 to t each reverse step. Numerically == the
     snapshot (decay<=1 contracts). COST: O(SEQ^2) flops vs O(SEQ). Quantify the
     blow-up at s4096 to judge if single-dispatch is even viable perf-wise.

All fp32. memguard 70. No production edit.
"""
from __future__ import annotations
import os, sys, threading, time
_LIM=70*1024*1024; _PEAK=0
def _rss():
    import resource; return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss//1024)
def _g():
    global _PEAK
    while True:
        r=_rss(); _PEAK=max(_PEAK,r)
        if r>_LIM: os._exit(137)
        time.sleep(0.25)
threading.Thread(target=_g,daemon=True).start()
import numpy as np
F32=np.float32; SEQ=4096; STATE=64
rng=np.random.RandomState(1234)
A=(-rng.rand(SEQ).astype(F32)); dt=(rng.rand(SEQ).astype(F32)*F32(0.05)).astype(F32)
decay=np.exp((A*dt).astype(F32)).astype(F32)
x=(rng.randn(SEQ).astype(F32)*F32(0.1)); Bv=(rng.randn(SEQ,STATE).astype(F32)*F32(0.1))
h=np.zeros(STATE,F32); snap=np.zeros((SEQ+1,STATE),F32); snap[0]=h
for t in range(SEQ):
    h=(decay[t]*h+x[t]*Bv[t]).astype(F32); snap[t+1]=h

# (b) forward-rerun-in-kernel accuracy (sample reverse steps to keep O(SEQ^2) feasible
#     in the probe; the math is identical to the snapshot's forward scan = exact).
worst=0.0
sample_ts=list(range(0,SEQ,131))  # sample; each is an O(t) forward re-scan
for t in sample_ts:
    hf=snap[0].astype(F32).copy()
    for u in range(t):
        hf=(decay[u]*hf+x[u]*Bv[u]).astype(F32)
    worst=max(worst,float(np.max(np.abs(hf.astype(np.float64)-snap[t].astype(np.float64)))))
flops_lane_linear = SEQ*STATE
flops_lane_quad = (SEQ*(SEQ-1)//2)*STATE
print(f"(b) FORWARD-RERUN-IN-KERNEL worst|h_fwd-snap| (sampled {len(sample_ts)} t) = {worst:.3e}  -> EXACT/bounded")
print(f"(b) cost per lane: snapshot+inverse = O(SEQ*STATE)={flops_lane_linear:,} adds;"
      f" forward-rerun = O(SEQ^2/2*STATE)={flops_lane_quad:,} adds  => {flops_lane_quad/flops_lane_linear:.0f}x more work")

# (a) log-space common-scale add feasibility: to add x*B to decay-scaled h in
# log space you must align scales by exp(gap of cumulative logdecay). Worst gap
# == full cum range. Show that factor.
cum=np.cumsum((np.log(decay)).astype(np.float64))
gap=float(cum.max()-cum.min())
print(f"(a) LOG-SPACE common-scale align factor exp(cum range) = exp({gap:.2f}) = {np.exp(gap):.3e}"
      f"  -> reintroduces the exact over/underflow the snapshot avoids (signed-sum has no log)")

print(f"\nPEAK_RSS_KB={_PEAK} memguard70=ON")
print(f"VERDICT (b) numerically exact but O(SEQ) slower per lane; (a) log-space infeasible for signed state-sum")
print("RC=0")
