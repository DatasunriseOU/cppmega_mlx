"""Build a float64 numpy autodiff-free analytic backward oracle for the Mamba3
MIMO scan and compare BOTH the gold (mlx ref) AND the Path-C SIMD route against
it at s4096. Determines whether the s4096 dC/dz divergence is a float32 precision
artifact (both float32 paths differ from f64 by similar magnitude / rel-error
small) or a genuine Path-C logic bug (Path-C far from f64, gold close).
Also report relative error and the magnitude of dC/dz themselves. memguard 70.
"""
from __future__ import annotations

import os
import sys
import threading
import time

_MEMGUARD_LIMIT_KB = 70 * 1024 * 1024
_PEAK_RSS_KB = 0


def _rss_kb() -> int:
    import resource
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024)


def _memguard_thread():
    global _PEAK_RSS_KB
    while True:
        r = _rss_kb()
        if r > _PEAK_RSS_KB:
            _PEAK_RSS_KB = r
        if r > _MEMGUARD_LIMIT_KB:
            sys.stderr.write(f"[memguard70] KILL self rss_kb={r} > 70GB\n")
            sys.stderr.flush()
            os._exit(137)
        time.sleep(0.25)


threading.Thread(target=_memguard_thread, daemon=True).start()

os.environ.setdefault("TILELANG_MLX_TVM_FFI_FORCE_COMMAND_BUFFER_BOUNDARY", "1")
sys.path.insert(0, "/Volumes/external/sources/cppmega.mlx")

import numpy as np
import mlx.core as mx
from cppmega_mlx.nn._tilelang.mamba3 import mamba3_mimo_bwd_metal
from cppmega_mlx.nn._tilelang.mamba3_path_c import _mamba3_mimo_bwd_path_c_simd_kernel

GRAD_NAMES = ("dx", "dB", "dC", "dz", "dA", "ddt", "dD", "dh0")
HEADS, HEADDIM, STATE = 128, 64, 64
BATCH = 1


def build_inputs_np(SEQ, seed=0):
    rng = np.random.RandomState(seed)
    x = (rng.randn(BATCH, SEQ, HEADS, HEADDIM) * 0.1).astype(np.float32)
    B = (rng.randn(BATCH, SEQ, HEADS, STATE) * 0.1).astype(np.float32)
    C = (rng.randn(BATCH, SEQ, HEADS, STATE) * 0.1).astype(np.float32)
    z = (rng.randn(BATCH, SEQ, HEADS, HEADDIM) * 0.5).astype(np.float32)
    A_head = (-rng.rand(HEADS)).astype(np.float32)
    A = np.broadcast_to(A_head[None, None, :], (BATCH, SEQ, HEADS)).copy().astype(np.float32)
    dt = (rng.rand(BATCH, SEQ, HEADS) * 0.05).astype(np.float32)
    D = (rng.randn(HEADS)).astype(np.float32)
    h0 = (rng.randn(BATCH, HEADS, HEADDIM, STATE) * 0.1).astype(np.float32)
    dy = (rng.randn(BATCH, SEQ, HEADS, HEADDIM) * 0.1).astype(np.float32)
    return dy, x, B, C, z, A, dt, D, h0


def silu(v):
    return v / (1.0 + np.exp(-v))


def dsilu(v):
    s = 1.0 / (1.0 + np.exp(-v))
    return s * (1.0 + v * (1.0 - s))


def f64_bwd(dy, x, B, C, z, A, dt, D, h0):
    """Analytic float64 backward of the reference scan. Shapes B,T,H,P / B,T,H,N."""
    dy = dy.astype(np.float64); x = x.astype(np.float64); B = B.astype(np.float64)
    C = C.astype(np.float64); z = z.astype(np.float64); A = A.astype(np.float64)
    dt = dt.astype(np.float64); D = D.astype(np.float64); h0 = h0.astype(np.float64)
    bb, T, H, P = x.shape
    N = B.shape[-1]
    log_decay = (A * dt)  # (b,T,H)
    dec = np.exp(log_decay)  # (b,T,H)
    # forward, store h_t (after update at step t), h_{-1}=h0
    h = h0.copy()  # (b,H,P,N)
    h_list = []
    for t in range(T):
        inp = x[:, t, :, :, None] * B[:, t, :, None, :]  # (b,H,P,N)
        h = dec[:, t, :, None, None] * h + inp
        h_list.append(h.copy())
    # gradients
    dx = np.zeros_like(x); dB = np.zeros_like(B); dC = np.zeros_like(C)
    dz = np.zeros_like(z); dA = np.zeros_like(A); ddt = np.zeros_like(dt)
    dD = np.zeros((H,), dtype=np.float64)
    dh = np.zeros_like(h0)  # carry d/dh_t
    for t in range(T - 1, -1, -1):
        ht = h_list[t]  # (b,H,P,N)
        inp_acc = ht  # h after update
        # y_pre = sum_n h*C + D*x   (b,H,P)
        y_pre = np.sum(ht * C[:, t, :, None, :], axis=-1) + D[None, :, None] * x[:, t]
        zt = z[:, t]  # (b,H,P)
        sz = silu(zt)
        # out = sz * y_pre ; dout = dy
        dyt = dy[:, t]  # (b,H,P)
        dz[:, t] = dyt * dsilu(zt) * y_pre
        dy_pre = dyt * sz  # (b,H,P)
        # y_pre = sum_n h*C + D*x
        dC[:, t] = np.sum(dy_pre[:, :, :, None] * ht, axis=2)  # sum over P -> (b,H,N)
        dD += np.sum(dy_pre * x[:, t], axis=(0, 2))  # (H,)
        dx[:, t] += dy_pre * D[None, :, None]
        # dh from y: dy_pre * C broadcast over N
        dh = dh + dy_pre[:, :, :, None] * C[:, t, :, None, :]  # (b,H,P,N)
        # h_t = dec_t * h_{t-1} + inp_t ; inp = x⊗B
        # grad wrt inp
        dinp = dh
        dx[:, t] += np.sum(dinp * B[:, t, :, None, :], axis=-1)
        dB[:, t] = np.sum(dinp * x[:, t, :, :, None], axis=2)
        # grad wrt dec and h_{t-1}
        h_prev = h_list[t - 1] if t > 0 else h0
        ddec = np.sum(dh * h_prev, axis=(2, 3))  # (b,H)
        # dec = exp(A*dt) -> d/d(A*dt) = dec*ddec ; A*dt -> dA += dt*., ddt += A*.
        dlog = ddec * dec[:, t]
        dA[:, t] += dlog * dt[:, t]
        ddt[:, t] += dlog * A[:, t]
        dh = dh * dec[:, t, :, None, None]  # propagate to h_{t-1}
    dh0 = dh
    return dict(dx=dx, dB=dB, dC=dC, dz=dz, dA=dA, ddt=ddt, dD=dD, dh0=dh0)


def maxabs(a, ref):
    return float(np.abs(np.asarray(a, np.float64) - np.asarray(ref, np.float64)).max())


def relerr(a, ref):
    a = np.asarray(a, np.float64); ref = np.asarray(ref, np.float64)
    denom = np.maximum(np.abs(ref).max(), 1e-30)
    return float(np.abs(a - ref).max() / denom)


for SEQ in [2048, 4096]:
    print(f"\n=== SEQ={SEQ} ===")
    dy, x, B, C, z, A, dt, D, h0 = build_inputs_np(SEQ)
    oracle = f64_bwd(dy, x, B, C, z, A, dt, D, h0)
    print("  |dC|max(f64)=%.4e  |dz|max(f64)=%.4e" %
          (np.abs(oracle["dC"]).max(), np.abs(oracle["dz"]).max()))

    mxargs = [mx.array(a) for a in (dy, x, B, C, z, A, dt, D, h0)]
    gold = mamba3_mimo_bwd_metal(mxargs[0], *mxargs[1:], backend="mlx")
    mx.eval(*gold)
    simd = _mamba3_mimo_bwd_path_c_simd_kernel(*mxargs)
    mx.eval(*simd)

    for nm, gg, sg in zip(GRAD_NAMES, gold, simd):
        ora = oracle[nm]
        gg_np = np.asarray(gg.astype(mx.float32), np.float64)
        sg_np = np.asarray(sg.astype(mx.float32), np.float64)
        if nm == "dD":
            ora = ora.reshape(gg_np.shape)
        d_g = maxabs(gg_np, ora); r_g = relerr(gg_np, ora)
        d_s = maxabs(sg_np, ora); r_s = relerr(sg_np, ora)
        d_gs = maxabs(gg_np, sg_np)
        flag = "  <== over 1e-3 vs gold" if d_gs >= 1e-3 else ""
        print(f"  {nm:4s} gold-vs-f64 abs={d_g:.2e} rel={r_g:.1e} | "
              f"simd-vs-f64 abs={d_s:.2e} rel={r_s:.1e} | gold-vs-simd abs={d_gs:.2e}{flag}")

print(f"\nPEAK_RSS_KB={_PEAK_RSS_KB} (~{_PEAK_RSS_KB/1048576:.3f}GB)  memguard70=ON")
print("RC=0")
