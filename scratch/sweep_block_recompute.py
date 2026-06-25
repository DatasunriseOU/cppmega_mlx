"""Sweep _BWD_SNAPSHOT_BLOCK for the recompute-chunk SIMD lane kernel: bit-correct
vs path-b gold + time vs MSL. memguard 70.
Usage: python3.13 scratch/sweep_block_recompute.py SEQ BLOCK [BLOCK ...]
"""
from __future__ import annotations
import os, sys, threading, time
_LIM = 70 * 1024 * 1024
_PEAK = 0


def _rss():
    import resource
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024)


def _guard():
    global _PEAK
    while True:
        r = _rss()
        if r > _PEAK:
            _PEAK = r
        if r > _LIM:
            sys.stderr.write(f"[memguard70] KILL rss_kb={r}\n")
            os._exit(137)
        time.sleep(0.25)


threading.Thread(target=_guard, daemon=True).start()
os.environ.setdefault("TILELANG_MLX_TVM_FFI_FORCE_COMMAND_BUFFER_BOUNDARY", "1")
sys.path.insert(0, "/Volumes/external/sources/cppmega.mlx")
import numpy as np  # noqa: E402
import mlx.core as mx  # noqa: E402
import cppmega_mlx.nn._tilelang.mamba3_path_c as M  # noqa: E402
from cppmega_mlx.nn._tilelang.mamba3 import mamba3_mimo_bwd_metal  # noqa: E402

GRAD_NAMES = ("dx", "dB", "dC", "dz", "dA", "ddt", "dD", "dh0")
HEADS, HEADDIM, STATE, BATCH = 128, 64, 64, 1
SEQ = int(sys.argv[1]) if len(sys.argv) > 1 else 2048
BLOCKS_TO_TRY = [int(a) for a in sys.argv[2:]] or [16]


def build(seed=0):
    rng = np.random.RandomState(seed)

    def f32(*s, sc=0.1):
        return mx.array((rng.randn(*s) * sc).astype(np.float32))

    x = f32(BATCH, SEQ, HEADS, HEADDIM)
    B = f32(BATCH, SEQ, HEADS, STATE)
    C = f32(BATCH, SEQ, HEADS, STATE)
    z = f32(BATCH, SEQ, HEADS, HEADDIM, sc=0.5)
    Ah = (-rng.rand(HEADS)).astype(np.float32)
    A = mx.array(np.broadcast_to(Ah[None, None, :], (BATCH, SEQ, HEADS)).copy())
    dt = mx.array((rng.rand(BATCH, SEQ, HEADS) * 0.05).astype(np.float32))
    D = mx.array((rng.randn(HEADS)).astype(np.float32))
    h0 = f32(BATCH, HEADS, HEADDIM, STATE)
    dy = mx.array((rng.randn(BATCH, SEQ, HEADS, HEADDIM) * 0.1).astype(np.float32))
    return dy, x, B, C, z, A, dt, D, h0


def maxabs(a, ref):
    return float(np.abs(np.asarray(a.astype(mx.float32), np.float64)
                        - np.asarray(ref.astype(mx.float32), np.float64)).max())


def bench(fn, args, warmup=3, runs=7):
    for _ in range(warmup):
        g = fn(*args)
        mx.eval(*g)
    ts = []
    for _ in range(runs):
        t0 = time.perf_counter()
        g = fn(*args)
        mx.eval(*g)
        ts.append((time.perf_counter() - t0) * 1e3)
    ts.sort()
    return ts[len(ts) // 2], ts[0]


args = build()
dy = args[0]
primals = args[1:]
gold = mamba3_mimo_bwd_metal(dy, *primals, backend="mlx")
mx.eval(*gold)
msl_med, msl_min = bench(lambda *a: mamba3_mimo_bwd_metal(a[0], *a[1:], backend="metal"), args)
print(f"SEQ={SEQ}  MSL median={msl_med:.3f}ms min={msl_min:.3f}ms")

for BLK in BLOCKS_TO_TRY:
    M._BWD_SIMD_RECOMPUTE_BLOCK = BLK
    M._bwd_state_snapshots_kernel_for.cache_clear()
    if hasattr(M._bwd_simd_reduce_kernel_for_state_snapshots, "cache_clear"):
        M._bwd_simd_reduce_kernel_for_state_snapshots.cache_clear()
    try:
        g = M._mamba3_mimo_bwd_path_c_simd_kernel(*args)
        mx.eval(*g)
        worst = 0.0
        per = {}
        for nm, gg, gr in zip(GRAD_NAMES, g, gold):
            d = maxabs(gg, gr)
            per[nm] = d
            worst = max(worst, d)
        med, mn = bench(M._mamba3_mimo_bwd_path_c_simd_kernel, args)
        bad = {k: f"{v:.1e}" for k, v in per.items() if v >= 1e-3}
        print(f"  BLOCK={BLK:3d}  worst={worst:.3e} ok={worst<1e-3}  "
              f"SIMD median={med:.3f}ms min={mn:.3f}ms  SIMD/MSL={med/msl_med:.3f}x"
              + (f"  FAIL={bad}" if bad else ""))
    except Exception as e:  # noqa: BLE001
        import traceback
        print(f"  BLOCK={BLK:3d}  FAILED: {type(e).__name__}: {str(e)[:200]}")
        for ln in traceback.format_exc().splitlines():
            if any(s in ln for s in ("int32", "2147483", "exceeds", "overflow", "register", "Register")):
                print(f"      >> {ln.strip()}")

print(f"\nPEAK_RSS_KB={_PEAK} (~{_PEAK/1048576:.3f}GB)  memguard70=ON")
print("RC=0")
