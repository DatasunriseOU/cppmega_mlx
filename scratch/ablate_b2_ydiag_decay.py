"""Y_diag DECAY/exp2 isolation + LEVER-A prototype timing probe (RULE #1: TIMING
PROBES, NOT correctness paths -- the stubbed/const variants produce WRONG outputs
on purpose and are NEVER fed into the bit-correct chain; only timed).

Builds a SLIM kernel that ONLY runs the Y_diag dinp block (everything else stubbed
to the cheapest runtime-live op) so the wall-time delta is attributable to Y_diag
arithmetic alone. Variants:

  YD_FULL   : exact production Y_diag inner loop -- exp2 RECOMPUTED inside the
              (pp,nn) iteration (ll-loop), L*headdim*dstate*tri exp2 total.
  YD_CONST  : decay/exp2 replaced by a runtime-live CONSTANT (g). Isolates the
              transcendental (exp2) cost: YD_FULL - YD_CONST == exp2 wall share.
  YD_NORED  : the ll-reduction body stubbed to a single runtime-live store (drops
              the O(L) inner sweep). Isolates the reduction/MAC cost.
  YD_LEVERA : LEVER A -- DECAY[L,L] shared tile precomputed ONCE over T.Parallel(L*L)
              (L*L/2 exp2), then the inner ll-loop is a PURE FMA dY*C*DECAY (zero
              exp2 in the (pp,nn) iteration). Mathematically identical decay values
              -> WORST-preserving; this measures the LEVER-A speedup.

memguard 70 mandatory.
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
from cppmega_mlx.nn._tilelang.mamba3_chunked_backward_core import _LOG2E
from cppmega_mlx.nn._tilelang.mamba3_chunked_scan_core import _resolve_chunked_compile_target

DEV = "mps"
ITERS = int(os.environ.get("BENCH_ITERS", "40"))
WARMUP = int(os.environ.get("BENCH_WARMUP", "8"))


def build_prim(batch, seqlen, chunk_size, ngroups, nheads, headdim, dstate, *, threads=128, variant="YD_FULL"):
    dtype = T.float16
    accum_dtype = T.float32
    nchunks = seqlen // chunk_size
    heads_per_group = nheads // ngroups
    L = chunk_size
    p = _LOG2E
    V = variant

    @T.prim_func
    def main(
        dout: T.Tensor((batch, seqlen, nheads, headdim), dtype),
        C: T.Tensor((batch, seqlen, ngroups, dstate), dtype),
        dA_cumsum: T.Tensor((batch, nheads, nchunks, chunk_size), dtype),
        gate_buf: T.Tensor((1,), accum_dtype),
        dinp: T.Tensor((batch, seqlen, nheads, headdim, dstate), accum_dtype),
    ):
        with T.Kernel(batch * nchunks, nheads, threads=threads) as (bx, by):
            batch_idx = bx % batch
            chunk_idx = bx // batch
            head_idx = by
            group_idx = head_idx // heads_per_group
            base = chunk_idx * chunk_size

            dacs = T.alloc_shared((L,), accum_dtype)
            dY = T.alloc_shared((L, headdim), accum_dtype)
            # DECAY as fp16 (mirrors the existing DYX[L,L] fp16 precompute slot) so
            # the L*L tile coexists with dY within 32KB threadgroup memory. exp2 is
            # evaluated in fp32 then narrowed -- decay in [0,1], fp16 ~1e-3 relerr,
            # WITHIN the 1e-3 gate (same as DYX which already round-trips fp16).
            DECAY = T.alloc_shared((L, L), dtype)
            g = gate_buf[0]

            for l in T.Parallel(L):
                dacs[l] = T.Cast(accum_dtype, dA_cumsum[batch_idx, head_idx, chunk_idx, l])
            for lp in T.Parallel(L * headdim):
                ll = lp // headdim
                pp = lp % headdim
                dY[ll, pp] = T.Cast(accum_dtype, dout[batch_idx, base + ll, head_idx, pp])
            T.sync_threads()

            if V == "YD_LEVERA":
                # LEVER A: precompute DECAY[ll,ss] = exp2((dacs[ll]-dacs[ss])*p) ONCE
                # over the L*L grid (L*L/2 useful exp2). Mirrors the DYX precompute.
                for ls in T.Parallel(L * L):
                    ll = ls // L
                    ss = ls % L
                    DECAY[ll, ss] = T.Cast(dtype, T.exp2((dacs[ll] - dacs[ss]) * p))
                T.sync_threads()

            for spn in T.serial(0, L * headdim * dstate, threads):
                lane = T.get_thread_binding(0)
                spni = spn + lane
                if spni < L * headdim * dstate:
                    ss = spni // (headdim * dstate)
                    rem = spni % (headdim * dstate)
                    pp = rem // dstate
                    nn = rem % dstate
                    sidx = base + ss
                    acc = T.alloc_local((1,), accum_dtype)
                    acc[0] = T.Cast(accum_dtype, 0)
                    if V == "YD_NORED":
                        # drop the O(L) ll-sweep; keep one runtime-live read+store
                        acc[0] = g * (dY[ss, pp] + T.Cast(
                            accum_dtype, C[batch_idx, sidx, group_idx, nn]))
                    else:
                        for ll in T.serial(ss, L):
                            if V == "YD_FULL":
                                lmat = T.exp2((dacs[ll] - dacs[ss]) * p)
                            elif V == "YD_CONST":
                                lmat = g  # runtime-live const; NO exp2
                            elif V == "YD_NOCREAD":
                                lmat = T.exp2((dacs[ll] - dacs[ss]) * p)
                            else:  # YD_LEVERA: pure shared read, NO exp2
                                lmat = T.Cast(accum_dtype, DECAY[ll, ss])
                            if V == "YD_NOCREAD":
                                c_v = g  # stub the C GLOBAL read with runtime-live const
                            else:
                                c_v = T.Cast(accum_dtype, C[batch_idx, base + ll, group_idx, nn])
                            acc[0] = acc[0] + dY[ll, pp] * c_v * lmat
                    dinp[batch_idx, sidx, head_idx, pp, nn] = acc[0]
            T.sync_threads()

    return main


def compile_variant(shape, variant):
    prim = build_prim(**shape, variant=variant)
    tgt = _resolve_chunked_compile_target(None)
    return tilelang.compile(prim, out_idx=[4], target=tgt)


def make_inputs(shape, seed=0):
    b, S, ch, G, H, P, N = (shape["batch"], shape["seqlen"], shape["chunk_size"],
                            shape["ngroups"], shape["nheads"], shape["headdim"], shape["dstate"])
    nchunks = S // ch
    rng = np.random.RandomState(seed)
    def t16(*s): return torch.tensor((rng.randn(*s) * 0.1).astype(np.float32), device=DEV, dtype=torch.float16).contiguous()
    ins = dict(
        dout=t16(b, S, H, P), C=t16(b, S, G, N), dA=t16(b, H, nchunks, ch),
        gate=torch.ones(1, device=DEV, dtype=torch.float32),
    )
    return ins, (b, S, ch, nchunks, G, H, P, N)


def run_once(k, ins, dims):
    b, S, ch, nchunks, G, H, P, N = dims
    out = torch.zeros(b, S, H, P, N, device=DEV, dtype=torch.float32)
    k(ins["dout"], ins["C"], ins["dA"], ins["gate"], out)
    torch.mps.synchronize()
    return out


def time_variant(k, ins, dims):
    b, S, ch, nchunks, G, H, P, N = dims
    def run():
        run_once(k, ins, dims)
    for _ in range(WARMUP): run()
    ts = []
    for _ in range(ITERS):
        t0 = time.perf_counter(); run(); ts.append(time.perf_counter() - t0)
    return float(np.median(ts) * 1e6), min(ts) * 1e6


SHAPES = {
    "tested-S512": dict(batch=1, seqlen=512, chunk_size=64, ngroups=1, nheads=2, headdim=64, dstate=16),
    "nam56r-H128": dict(batch=1, seqlen=512, chunk_size=64, ngroups=8, nheads=128, headdim=64, dstate=64),
}
VARIANTS = ["YD_FULL", "YD_CONST", "YD_NORED", "YD_LEVERA"]

for sname, shape in SHAPES.items():
    print(f"\n===== {sname} =====")
    ins, dims = make_inputs(shape)
    res = {}
    out_full = None
    out_levera = None
    for v in VARIANTS:
        k = compile_variant(shape, v)
        med, mn = time_variant(k, ins, dims)
        res[v] = med
        if v == "YD_FULL": out_full = run_once(k, ins, dims)
        if v == "YD_LEVERA": out_levera = run_once(k, ins, dims)
        print(f"  {v:9s} median={med:9.1f}us  min={mn:9.1f}us")
    if out_full is not None and out_levera is not None:
        worst = float((out_full - out_levera).abs().max().item())
        denom = float(out_full.abs().max().item()) + 1e-12
        print(f"  -- LEVER A WORST abs diff vs YD_FULL = {worst:.3e} (rel {worst/denom:.3e}); gate<1e-3")
    full = res["YD_FULL"]
    print(f"  -- exp2 wall share (YD_FULL - YD_CONST) = {full - res['YD_CONST']:8.1f}us "
          f"({100.0*(full-res['YD_CONST'])/full:5.1f}% of Y_diag)")
    print(f"  -- reduction wall share (YD_FULL - YD_NORED) = {full - res['YD_NORED']:8.1f}us "
          f"({100.0*(full-res['YD_NORED'])/full:5.1f}% of Y_diag)")
    print(f"  -- LEVER A speedup (YD_FULL - YD_LEVERA) = {full - res['YD_LEVERA']:8.1f}us "
          f"({100.0*(full-res['YD_LEVERA'])/full:5.1f}% of Y_diag; LEVERA={res['YD_LEVERA']:.1f}us)")

print(f"\nPEAK_RSS_KB={_PEAK} (~{_PEAK/1048576:.3f}GB) memguard70=ON")
print("RC=0")
