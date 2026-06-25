"""STRUCTURAL ABLATION of the de-funneled B2 (chunk_scan_combine_bwd) Metal prim.

RULE #1: these are TIMING PROBES, NOT correctness paths. Each variant stubs ONE
block (DYX build, dseg-atomic scatter, dC dstate loop incl. dC_off atomic) to
ISOLATE its share of the B2 wall-time. The stubbed kernels produce WRONG outputs
ON PURPOSE -- they are NEVER fed into the bit-correct chain, only timed. We do
NOT touch the production builder. No fabrication: every number printed is a real
torch.mps-synchronized median over BENCH_ITERS GPU dispatches; memguard 70 on.

Variants (Metal, same grid/threads/smem as the production serial prim):
  FULL    : exact copy of chunk_scan_combine_bwd_metal_prim (parity ref timing)
  NO_DYX  : DYX[ll,ss] build inner-headdim reduction stubbed (DYX:=0)
  NO_DSEG : the dseg L*L atomic_add(+/-) scatter into dAcs_acc stubbed (no atomics)
  NO_DCBLK: the dC (L*dstate) loop -- dC_off inner + dC_diag inner + dC_off atomic
            into dAcs_acc -- stubbed (writes 0, no inner reductions, no atomic)
  NO_ATOMS: BOTH dC_off atomic AND dseg +/- atomics stubbed, reductions KEPT
            (isolates the atomic-contention cost ALONE from the arithmetic)
  NO_YDIAG: the Y_diag dinp (L*headdim * dstate * tri) serial block stubbed
The delta FULL-minus-variant == that block's contribution to B2 wall-time.
"""
import os, sys, threading, time
import numpy as np

_LIM = 70 * 1024 * 1024
_PEAK = 0
def _rss():
    import resource
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024)
def _guard():
    global _PEAK
    while True:
        r = _rss()
        if r > _PEAK: _PEAK = r
        if r > _LIM:
            sys.stderr.write(f"[memguard70] KILL rss_kb={r}\n"); sys.stderr.flush(); os._exit(137)
        time.sleep(0.25)
threading.Thread(target=_guard, daemon=True).start()

sys.path.insert(0, "/Volumes/external/sources/cppmega.mlx")
import torch  # noqa
import tilelang  # noqa
import tilelang.language as T  # noqa
from cppmega_mlx.nn._tilelang.mamba3_chunked_backward_core import _LOG2E, _silu_grad_expr
from cppmega_mlx.nn._tilelang.mamba3_chunked_scan_core import _resolve_chunked_compile_target

DEV = "mps"
ITERS = int(os.environ.get("BENCH_ITERS", "40"))
WARMUP = int(os.environ.get("BENCH_WARMUP", "8"))


def build_prim(batch, seqlen, chunk_size, ngroups, nheads, headdim, dstate, *, threads=128, ablate=""):
    dtype = T.float16
    accum_dtype = T.float32
    nchunks = seqlen // chunk_size
    heads_per_group = nheads // ngroups
    L = chunk_size
    p = _LOG2E
    AB = ablate

    @T.prim_func
    def main(
        dout: T.Tensor((batch, seqlen, nheads, headdim), dtype),
        cb: T.Tensor((batch, nchunks, ngroups, chunk_size, chunk_size), dtype),
        x: T.Tensor((batch, seqlen, nheads, headdim), dtype),
        z: T.Tensor((batch, seqlen, nheads, headdim), dtype),
        dt: T.Tensor((batch, nheads, nchunks, chunk_size), dtype),
        dA_cumsum: T.Tensor((batch, nheads, nchunks, chunk_size), dtype),
        C: T.Tensor((batch, seqlen, ngroups, dstate), dtype),
        B: T.Tensor((batch, seqlen, ngroups, dstate), dtype),
        prev_states: T.Tensor((batch, nchunks, nheads, headdim, dstate), accum_dtype),
        D: T.Tensor((nheads), dtype),
        y: T.Tensor((batch, seqlen, nheads, headdim), dtype),
        gate_buf: T.Tensor((1,), accum_dtype),  # runtime ablation gate (==1.0); opaque to compiler -> NO DCE of stubbed math
        dC: T.Tensor((batch, seqlen, nheads, dstate), accum_dtype),
        dx: T.Tensor((batch, seqlen, nheads, headdim), accum_dtype),
        dz: T.Tensor((batch, seqlen, nheads, headdim), accum_dtype),
        dchunk_states: T.Tensor((batch, nchunks, nheads, headdim, dstate), accum_dtype),
        dinp: T.Tensor((batch, seqlen, nheads, headdim, dstate), accum_dtype),
        dA_cumsum_y: T.Tensor((batch, nheads, nchunks, chunk_size), accum_dtype),
        dD: T.Tensor((nheads), accum_dtype),
    ):
        with T.Kernel(batch * nchunks, nheads, threads=threads) as (bx, by):
            batch_idx = bx % batch
            chunk_idx = bx // batch
            head_idx = by
            group_idx = head_idx // heads_per_group
            base = chunk_idx * chunk_size

            dacs = T.alloc_shared((L,), accum_dtype)
            dY = T.alloc_shared((L, headdim), accum_dtype)
            dAcs_acc = T.alloc_shared((L,), accum_dtype)
            DYX = T.alloc_shared((L, L), dtype)
            g = gate_buf[0]  # runtime ==1.0; opaque -> stores stay live (no DCE cascade)

            for l in T.Parallel(L):
                dacs[l] = T.Cast(accum_dtype, dA_cumsum[batch_idx, head_idx, chunk_idx, l])
                dAcs_acc[l] = T.Cast(accum_dtype, 0)
            T.sync_threads()

            dD_local = T.alloc_local((1,), accum_dtype)
            dD_local[0] = T.Cast(accum_dtype, 0)
            for lp in T.Parallel(L * headdim):
                ll = lp // headdim
                pp = lp % headdim
                s = base + ll
                z_v = T.Cast(accum_dtype, z[batch_idx, s, head_idx, pp])
                gate = z_v / (T.Cast(accum_dtype, 1.0) + T.exp(-z_v))
                y_v = T.Cast(accum_dtype, y[batch_idx, s, head_idx, pp])
                dout_v = T.Cast(accum_dtype, dout[batch_idx, s, head_idx, pp])
                dgate = dout_v * y_v
                dy_v = dout_v * gate
                dz[batch_idx, s, head_idx, pp] = dgate * _silu_grad_expr(T, z_v, accum_dtype)
                d_v = T.Cast(accum_dtype, D[head_idx])
                x_v = T.Cast(accum_dtype, x[batch_idx, s, head_idx, pp])
                dx[batch_idx, s, head_idx, pp] = d_v * dy_v
                dD_local[0] = dD_local[0] + dy_v * x_v
                dY[ll, pp] = dy_v
            T.sync_threads()
            T.atomic_add(dD[head_idx], dD_local[0])

            for ln in T.Parallel(L * dstate):
                ll = ln // dstate
                nn = ln % dstate
                dC[batch_idx, base + ll, head_idx, nn] = T.Cast(accum_dtype, 0)
            for pn in T.Parallel(headdim * dstate):
                pp = pn // dstate
                nn = pn % dstate
                dchunk_states[batch_idx, chunk_idx, head_idx, pp, nn] = T.Cast(accum_dtype, 0)
            for lpn0 in T.serial(0, L * headdim * dstate, threads):
                lane = T.get_thread_binding(0)
                idx = lpn0 + lane
                if idx < L * headdim * dstate:
                    ll = idx // (headdim * dstate)
                    rem = idx % (headdim * dstate)
                    pp = rem // dstate
                    nn = rem % dstate
                    dinp[batch_idx, base + ll, head_idx, pp, nn] = T.Cast(accum_dtype, 0)
            T.sync_threads()

            # ---- DYX build ----
            for ls in T.Parallel(L * L):
                ll = ls // L
                ss = ls % L
                acc = T.alloc_local((1,), accum_dtype)
                acc[0] = T.Cast(accum_dtype, 0)
                if "NO_DYX" not in AB:
                    if ss <= ll:
                        for pp in T.serial(headdim):
                            acc[0] = acc[0] + dY[ll, pp] * T.Cast(
                                accum_dtype, x[batch_idx, base + ss, head_idx, pp])
                else:
                    acc[0] = g * dY[ll, 0]  # cheap, runtime-live (no DCE)
                DYX[ll, ss] = T.Cast(dtype, acc[0])
            T.sync_threads()

            # ---- dC block ----
            for ln in T.Parallel(L * dstate):
                ll = ln // dstate
                nn = ln % dstate
                s = base + ll
                sd = T.exp2(dacs[ll] * p)
                accn = T.alloc_local((1,), accum_dtype)
                accn[0] = T.Cast(accum_dtype, 0)
                cdiag = T.alloc_local((1,), accum_dtype)
                cdiag[0] = T.Cast(accum_dtype, 0)
                if "NO_DCBLK" not in AB:
                    for pp in T.serial(headdim):
                        cs = prev_states[batch_idx, chunk_idx, head_idx, pp, nn]
                        accn[0] = accn[0] + dY[ll, pp] * cs
                    for ss in T.serial(0, ll + 1):
                        lmat = T.exp2((dacs[ll] - dacs[ss]) * p)
                        dt_s = T.Cast(accum_dtype, dt[batch_idx, head_idx, chunk_idx, ss])
                        b_v = T.Cast(accum_dtype, B[batch_idx, base + ss, group_idx, nn])
                        cdiag[0] = cdiag[0] + lmat * dt_s * DYX[ll, ss] * b_v
                else:
                    accn[0] = g * dY[ll, 0]      # cheap, runtime-live
                    cdiag[0] = g * T.Cast(accum_dtype, DYX[ll, nn])
                dC[batch_idx, s, head_idx, nn] = accn[0] * sd + cdiag[0]
                c_v = T.Cast(accum_dtype, C[batch_idx, s, group_idx, nn])
                if "NO_DCBLK" in AB:
                    pass
                elif "NO_ATOMS" in AB:
                    dAcs_acc[ll] = accn[0] * c_v * sd  # racy plain store, probe only
                else:
                    T.atomic_add(dAcs_acc[ll], accn[0] * c_v * sd)
            T.sync_threads()

            # ---- dchunk_states ----
            for pn in T.Parallel(headdim * dstate):
                pp = pn // dstate
                nn = pn % dstate
                acc = T.alloc_local((1,), accum_dtype)
                acc[0] = T.Cast(accum_dtype, 0)
                for ll in T.serial(L):
                    sd = T.exp2(dacs[ll] * p)
                    c_v = T.Cast(accum_dtype, C[batch_idx, base + ll, group_idx, nn])
                    acc[0] = acc[0] + dY[ll, pp] * c_v * sd
                dchunk_states[batch_idx, chunk_idx, head_idx, pp, nn] = acc[0]
            T.sync_threads()

            # ---- Y_diag dinp ----
            for sp in T.serial(0, L * headdim, threads):
                lane = T.get_thread_binding(0)
                spi = sp + lane
                if spi < L * headdim:
                    ss = spi // headdim
                    pp = spi % headdim
                    sidx = base + ss
                    if "NO_YDIAG" not in AB:
                        for nn in T.serial(dstate):
                            acc = T.alloc_local((1,), accum_dtype)
                            acc[0] = T.Cast(accum_dtype, 0)
                            for ll in T.serial(ss, L):
                                lmat = T.exp2((dacs[ll] - dacs[ss]) * p)
                                c_v = T.Cast(accum_dtype, C[batch_idx, base + ll, group_idx, nn])
                                acc[0] = acc[0] + dY[ll, pp] * c_v * lmat
                            dinp[batch_idx, sidx, head_idx, pp, nn] = (
                                dinp[batch_idx, sidx, head_idx, pp, nn] + acc[0])
                    else:
                        for nn in T.serial(dstate):  # keep L*headdim*dstate store traffic + 1 read/store, drop the ll reduction
                            dinp[batch_idx, sidx, head_idx, pp, nn] = g * (
                                dY[0, pp] + T.Cast(accum_dtype, C[batch_idx, base + ss, group_idx, nn]))
            T.sync_threads()

            # ---- dseg atomic scatter ----
            for ls in T.Parallel(L * L):
                ll = ls // L
                ss = ls % L
                cb_v = T.Cast(accum_dtype, cb[batch_idx, chunk_idx, group_idx, ll, ss])
                lmat = T.exp2((dacs[ll] - dacs[ss]) * p)
                dt_s = T.Cast(accum_dtype, dt[batch_idx, head_idx, chunk_idx, ss])
                tri = T.if_then_else(ss < ll, T.Cast(accum_dtype, 1), T.Cast(accum_dtype, 0))
                dseg = DYX[ll, ss] * cb_v * lmat * dt_s * tri
                if "NO_DSEG" in AB:
                    pass  # whole dseg block (arith+atomic) elided -> isolates dseg total
                elif "NO_ATOMS" in AB:
                    # keep dseg arithmetic live (racy plain store, timing probe only)
                    # -> isolates the CAS-atomic-contention cost ALONE from the arith
                    dAcs_acc[ll] = dseg
                else:
                    T.atomic_add(dAcs_acc[ll], dseg)
                    T.atomic_add(dAcs_acc[ss], -dseg)
            T.sync_threads()

            for l in T.Parallel(L):
                dA_cumsum_y[batch_idx, head_idx, chunk_idx, l] = dAcs_acc[l]

    return main


def compile_variant(shape, ablate):
    prim = build_prim(**shape, ablate=ablate)
    tgt = _resolve_chunked_compile_target(None)
    # gate_buf is INPUT slot 11; the 7 outputs shift to slots 12..18.
    return tilelang.compile(prim, out_idx=[12, 13, 14, 15, 16, 17, 18], target=tgt)


def make_inputs(shape, seed=0):
    b, S, ch, G, H, P, N = (shape["batch"], shape["seqlen"], shape["chunk_size"],
                            shape["ngroups"], shape["nheads"], shape["headdim"], shape["dstate"])
    nchunks = S // ch
    rng = np.random.RandomState(seed)
    def t16(*s): return torch.tensor((rng.randn(*s) * 0.1).astype(np.float32), device=DEV, dtype=torch.float16).contiguous()
    def t32(*s): return torch.tensor((rng.randn(*s) * 0.1).astype(np.float32), device=DEV, dtype=torch.float32).contiguous()
    ins = dict(
        dout=t16(b, S, H, P), cb=t16(b, nchunks, G, ch, ch), x=t16(b, S, H, P),
        z=t16(b, S, H, P), dt=t16(b, H, nchunks, ch), dA=t16(b, H, nchunks, ch),
        C=t16(b, S, G, N), B=t16(b, S, G, N), prev=t32(b, nchunks, H, P, N),
        D=t16(H), y=t16(b, S, H, P),
        gate=torch.ones(1, device=DEV, dtype=torch.float32),  # ablation gate ==1.0
    )
    return ins, (b, S, ch, nchunks, G, H, P, N)


def time_variant(k, ins, dims):
    b, S, ch, nchunks, G, H, P, N = dims
    def alloc():
        return (torch.zeros(b, S, H, N, device=DEV, dtype=torch.float32),
                torch.zeros(b, S, H, P, device=DEV, dtype=torch.float32),
                torch.zeros(b, S, H, P, device=DEV, dtype=torch.float32),
                torch.zeros(b, nchunks, H, P, N, device=DEV, dtype=torch.float32),
                torch.zeros(b, S, H, P, N, device=DEV, dtype=torch.float32),
                torch.zeros(b, H, nchunks, ch, device=DEV, dtype=torch.float32),
                torch.zeros(H, device=DEV, dtype=torch.float32))
    def run():
        outs = alloc()
        k(ins["dout"], ins["cb"], ins["x"], ins["z"], ins["dt"], ins["dA"],
          ins["C"], ins["B"], ins["prev"], ins["D"], ins["y"], ins["gate"], *outs)
        torch.mps.synchronize()
    for _ in range(WARMUP): run()
    ts = []
    for _ in range(ITERS):
        t0 = time.perf_counter(); run(); ts.append(time.perf_counter() - t0)
    return float(np.median(ts) * 1e6), min(ts) * 1e6


SHAPES = {
    "tested-S512": dict(batch=1, seqlen=512, chunk_size=64, ngroups=1, nheads=2, headdim=64, dstate=16),
    "nam56r-H128": dict(batch=1, seqlen=512, chunk_size=64, ngroups=8, nheads=128, headdim=64, dstate=64),
}
VARIANTS = ["FULL", "NO_DYX", "NO_DSEG", "NO_DCBLK", "NO_ATOMS", "NO_YDIAG",
            "NO_DYX,NO_DSEG,NO_DCBLK", "NO_YDIAG,NO_DSEG,NO_DCBLK,NO_DYX"]

for sname, shape in SHAPES.items():
    print(f"\n===== {sname} =====")
    ins, dims = make_inputs(shape)
    base_us = None
    for v in VARIANTS:
        ab = "" if v == "FULL" else v
        k = compile_variant(shape, ab)
        med, mn = time_variant(k, ins, dims)
        if v == "FULL": base_us = med
        delta = (base_us - med) if base_us is not None else 0.0
        pct = (100.0 * delta / base_us) if base_us else 0.0
        print(f"  {v:9s} median={med:9.1f}us  min={mn:9.1f}us  delta_from_FULL={delta:8.1f}us ({pct:5.1f}%)")

print(f"\nPEAK_RSS_KB={_PEAK} (~{_PEAK/1048576:.3f}GB) memguard70=ON")
print("RC=0")
