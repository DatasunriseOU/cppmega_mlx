"""FOLLOW-UP PROBE: prove the BENIGN-config s4096 remat blow-up is CUMULATIVE
error amplification through the inverse recurrence (NOT inv_decay overflow),
and that the snapshot (forward re-run) is the stable alternative.

The single-dispatch bwd_simd walks h backward: h_prev=(h-xB)*inv_decay, inv_decay>=1.
The product of inv_decay over a span = 1/exp(sum logdecay) = 1/exp(dA_cumsum span)
which reaches ~1e22 at s4096 even in BENIGN. So an fp32 ULP in h_snap[SEQ] grows by
~1e22. We confirm by:
 (A) seeding h_snap[SEQ] with one ULP perturbation and watching the back-walk error;
 (B) comparing per-step inverse-walk h vs FORWARD-rerun h (the snapshot method) --
     the forward re-run is bounded (decay<=1 contracts), the inverse amplifies.
All fp32. memguard 70.
"""
from __future__ import annotations
import os, sys, threading, time
_LIM = 70*1024*1024; _PEAK=0
def _rss():
    import resource; return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss//1024)
def _g():
    global _PEAK
    while True:
        r=_rss(); _PEAK=max(_PEAK,r)
        if r>_LIM: sys.stderr.write(f"[memguard70]KILL {r}\n"); os._exit(137)
        time.sleep(0.25)
threading.Thread(target=_g,daemon=True).start()

import numpy as np
F32=np.float32
SEQ=4096; STATE=64
rng=np.random.RandomState(1234)

# BENIGN nam56r config (the one probe_fp32_bwd_simd uses, decay in [0.95,1])
A=(-rng.rand(SEQ).astype(F32)*F32(1.0))
dt=(rng.rand(SEQ).astype(F32)*F32(0.05)).astype(F32)
logdecay=(A*dt).astype(F32)
decay=np.exp(logdecay).astype(F32)
inv_decay=(F32(1.0)/decay).astype(F32)
x=(rng.randn(SEQ).astype(F32)*F32(0.1))
Bv=(rng.randn(SEQ,STATE).astype(F32)*F32(0.1))

# forward, cache boundaries
h=np.zeros(STATE,F32); snap=np.zeros((SEQ+1,STATE),F32); snap[0]=h
for t in range(SEQ):
    h=(decay[t]*h+x[t]*Bv[t]).astype(F32); snap[t+1]=h

# cumulative inverse-decay product over a back-span of length L from the end
csum=np.cumsum(logdecay[::-1].astype(np.float64))  # sum of last-k logdecays
inv_prod=np.exp(-csum)  # product of inv_decay over last k steps == amplification factor
print(f"[amplification] product(inv_decay) over last-k steps to t=0: "
      f"max={float(inv_prod.max()):.4e} (==1/exp(dA_cumsum full))")

# (A) ULP perturbation at the END boundary, back-walk in fp32
eps = np.spacing(snap[SEQ].astype(F32)).astype(F32)  # 1 ULP per element
hb_clean = snap[SEQ].astype(F32).copy()
hb_pert  = (snap[SEQ].astype(F32)+eps).astype(F32)
div_at = -1
for r in range(SEQ):
    t=SEQ-1-r
    hb_clean=((hb_clean-x[t]*Bv[t]).astype(F32)*inv_decay[t]).astype(F32)
    hb_pert =((hb_pert -x[t]*Bv[t]).astype(F32)*inv_decay[t]).astype(F32)
    d=float(np.max(np.abs(hb_pert.astype(np.float64)-hb_clean.astype(np.float64))))
    if div_at<0 and d>1.0:
        div_at=r
print(f"[A: ULP back-walk] 1-ULP seed at t=SEQ -> after full back-walk divergence "
      f"max|pert-clean|={float(np.max(np.abs(hb_pert.astype(np.float64)-hb_clean.astype(np.float64)))):.4e}"
      f"  (crossed 1.0 at back-step r={div_at})")

# (B) inverse-walk vs snapshot accuracy, AND forward-rerun-from-h0 accuracy
# inverse walk error vs cached forward:
hb=snap[SEQ].astype(F32).copy(); inv_err=0.0
for r in range(SEQ):
    t=SEQ-1-r
    hb=((hb-x[t]*Bv[t]).astype(F32)*inv_decay[t]).astype(F32)
    inv_err=max(inv_err,float(np.max(np.abs(hb.astype(np.float64)-snap[t].astype(np.float64)))))
print(f"[B: inverse-walk]  worst |h_inv - snap| over s4096 = {inv_err:.4e}")

# forward re-run from h0 (== the snapshot kernel's method: re-scan forward, decay<=1) :
hf=snap[0].astype(F32).copy(); fwd_err=0.0
for t in range(SEQ):
    hf=(decay[t]*hf+x[t]*Bv[t]).astype(F32)
    fwd_err=max(fwd_err,float(np.max(np.abs(hf.astype(np.float64)-snap[t+1].astype(np.float64)))))
print(f"[B: forward-rerun] worst |h_fwd - snap| over s4096 = {fwd_err:.4e}  (snapshot method = bounded)")

print(f"\nPEAK_RSS_KB={_PEAK} memguard70=ON")
print(f"VERDICT inv_decay_max={float(inv_decay.max()):.4f} (BOUNDED+finite) "
      f"BUT inverse-walk worst_err={inv_err:.2e} vs forward-rerun worst_err={fwd_err:.2e}")
print("RC=0")
