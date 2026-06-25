"""SMOKE: path_c_fwd_path_c_bwd end-to-end fwd+bwd on the nam56r-config surface.

Confirms the NEW mode (Path C DSL fwd + chunked B2->B1->B0 backward) RUNS end to
end through the MLX custom_function and returns 8 grads of correct shape/dtype.
This is NOT a parity check (next phase). memguard 70. No fabrication.

Surface = the _dispatch_mamba3_scan surface: B/C per-head (b,seq,H,N), A
per-head-CONSTANT (b,seq,H) (the chunked kernels' validated regime). nam56r dims:
P=64, N=64, chunk=64, H=128; small batch + short seq for the smoke.
"""
import os
import sys
import threading
import time

_MEMGUARD_LIMIT_KB = 70 * 1024 * 1024
_PEAK = 0


def _rss_kb():
    import resource
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024)


def _guard():
    global _PEAK
    while True:
        r = _rss_kb()
        if r > _PEAK:
            _PEAK = r
        if r > _MEMGUARD_LIMIT_KB:
            sys.stderr.write(f"[memguard70] KILL rss_kb={r}\n")
            sys.stderr.flush()
            os._exit(137)
        time.sleep(0.25)


threading.Thread(target=_guard, daemon=True).start()
sys.path.insert(0, "/Volumes/external/sources/cppmega.mlx")

import numpy as np
import mlx.core as mx

from cppmega_mlx.nn._tilelang.mamba3_path_c import (
    mamba3_mimo_apply_with_state_path_c_fwd_path_c_bwd,
)

# nam56r config-exact dims (P,N,chunk per the prompt). Short seq + b=1 for smoke.
b, seq, H, P, N, chunk = 1, 128, 128, 64, 64, 64
rng = np.random.RandomState(0)


def f32(*shape, s=0.1):
    return mx.array((rng.randn(*shape) * s).astype(np.float32))


x = f32(b, seq, H, P)
B = f32(b, seq, H, N)              # per-head (G==H at this surface)
C = f32(b, seq, H, N)
z = f32(b, seq, H, P, s=0.5)
# A per-head-CONSTANT across seq (the chunked kernels' validated regime).
A_head = (-rng.rand(H)).astype(np.float32)        # (H,)
A = mx.array(np.broadcast_to(A_head[None, None, :], (b, seq, H)).copy())
dt = mx.array((rng.rand(b, seq, H) * 0.05).astype(np.float32))
D = mx.array((rng.randn(H)).astype(np.float32))
h0 = f32(b, H, P, N)

primal_shapes = {
    "x": x.shape, "B": B.shape, "C": C.shape, "z": z.shape,
    "A": A.shape, "dt": dt.shape, "D": D.shape, "h0": h0.shape,
}

# ---- FORWARD ----
y, h_last, cb, dA_cumsum, prev_states, y_stash = (
    mamba3_mimo_apply_with_state_path_c_fwd_path_c_bwd(x, B, C, z, A, dt, D, h0)
)
mx.eval(y, h_last, cb, dA_cumsum, prev_states, y_stash)
print(f"FWD y {y.shape}/{y.dtype}  h_last {h_last.shape}/{h_last.dtype}")
print(f"STASH cb {cb.shape}/{cb.dtype}  dA_cumsum {dA_cumsum.shape}/{dA_cumsum.dtype} "
      f"prev_states {prev_states.shape}/{prev_states.dtype}  y_stash {y_stash.shape}/{y_stash.dtype}")

# ---- BACKWARD via mx.vjp (cotangent only on y; h_last/stash get zero) ----
def fwd_y(x, B, C, z, A, dt, D, h0):
    out = mamba3_mimo_apply_with_state_path_c_fwd_path_c_bwd(x, B, C, z, A, dt, D, h0)
    return out[0]  # y only


cot_y = mx.array((rng.randn(b, seq, H, P) * 0.1).astype(np.float32))
_, grads = mx.vjp(
    fwd_y, (x, B, C, z, A, dt, D, h0), (cot_y,)
)
mx.eval(*grads)

names = ["dx", "dB", "dC", "dz", "dA", "ddt", "dD", "dh0"]
prim_names = ["x", "B", "C", "z", "A", "dt", "D", "h0"]
print(f"\n8 GRADS (count={len(grads)}):")
ok = True
shapes_out = {}
for nm, pn, g in zip(names, prim_names, grads):
    shp = tuple(g.shape)
    shapes_out[nm] = (shp, str(g.dtype))
    match = shp == tuple(primal_shapes[pn])
    ok = ok and match
    finite = bool(mx.all(mx.isfinite(g.astype(mx.float32))))
    print(f"  {nm:4s} shape={shp} dtype={g.dtype} (primal {pn}={primal_shapes[pn]}) "
          f"shape_match={match} finite={finite}")

print(f"\nALL_8_GRADS_SHAPE_MATCH_PRIMALS={ok}")
print(f"PEAK_RSS_KB={_PEAK} (~{_PEAK/1048576:.3f}GB) memguard70=ON")
print("SMOKE_RESULT:", "RUNS_8_GRADS_OK" if (ok and len(grads) == 8) else "FAIL")
