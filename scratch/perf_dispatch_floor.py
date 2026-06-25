"""Pin the MLX tvm_ffi PER-DISPATCH floor: time the SMALLEST chunked kernel (B1,
~18us real compute on torch/mps) on the MLX route, with mx.eval per call, and
compare to (a) the same kernel WITHOUT a forced per-call eval (queued, 1 eval at
end of N), and (b) a bare mx.eval sync of a trivial op. This separates the
per-dispatch bridge/encoder-finalize/commit cost from the per-eval sync cost.
memguard 70. base 5d5c878.
"""
import os, sys, threading, time
_LIM = 70 * 1024 * 1024; _PEAK = 0
def _g():
    global _PEAK
    import resource
    while True:
        r = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024)
        if r > _PEAK: _PEAK = r
        if r > _LIM: os._exit(137)
        time.sleep(0.25)
threading.Thread(target=_g, daemon=True).start()
sys.path.insert(0, "/Volumes/external/sources/cppmega.mlx")
os.environ["TILELANG_MLX_TVM_FFI_FORCE_COMMAND_BUFFER_BOUNDARY"] = "1"
import numpy as np
import mlx.core as mx
from cppmega_mlx.nn._tilelang.mamba3_path_c import (
    mamba3_mimo_fwd_path_c, _mamba3_chunked_fwd_intermediates_path_c,
    _chunked_bwd_b2_kernel, _chunked_bwd_b1_kernel)

SEQ = int(sys.argv[1]) if len(sys.argv) > 1 else 128
b, seq, H, P, N, chunk = 1, SEQ, 128, 64, 64, 64
G = H; nchunks = seq // chunk
rng = np.random.RandomState(0)
def f32(*s, sc=0.1): return mx.array((rng.randn(*s) * sc).astype(np.float32))
x = f32(b, seq, H, P); B = f32(b, seq, H, N); C = f32(b, seq, H, N)
z = f32(b, seq, H, P, sc=0.5)
A_head = (-rng.rand(H)).astype(np.float32)
A = mx.array(np.broadcast_to(A_head[None, None, :], (b, seq, H)).copy())
dt = mx.array((rng.rand(b, seq, H) * 0.05).astype(np.float32))
D = mx.array((rng.randn(H)).astype(np.float32)); h0 = f32(b, H, P, N)
cot_y = mx.array((rng.randn(b, seq, H, P) * 0.1).astype(np.float32))
cb, dA_cumsum, prev_states = _mamba3_chunked_fwd_intermediates_path_c(x, B, C, A, dt, h0)
mx.eval(cb, dA_cumsum, prev_states)
y_lane, _ = mamba3_mimo_fwd_path_c(x, B, C, z, A, dt, D, h0); mx.eval(y_lane)
y16 = mx.contiguous(y_lane.astype(mx.float16)); mx.eval(y16)
def f16(a): return mx.contiguous(a.astype(mx.float16))
x16=f16(x); B16=f16(B); C16=f16(C); z16=f16(z); D16=f16(D); dout16=f16(cot_y)
dt16=dt.astype(mx.float16)
dt_k=mx.contiguous(mx.transpose(dt16.reshape(b,nchunks,chunk,H),(0,3,1,2)))
cb16=cb.astype(mx.float16); dA16=dA_cumsum.astype(mx.float16)
A_head16=A[:,0,:][0].astype(mx.float16); prev32=prev_states.astype(mx.float32)
mx.eval(x16,B16,C16,z16,D16,dout16,dt_k,cb16,dA16,A_head16,prev32)
k_b2=_chunked_bwd_b2_kernel(b,seq,chunk,G,H,P,N)
_o2=k_b2(dout16,cb16,x16,z16,dt_k,dA16,C16,B16,prev32,D16,y16); mx.eval(*_o2)
dchunk=_o2[3]
k_b1=_chunked_bwd_b1_kernel(b,seq,chunk,G,H,P,N)
dh0z=mx.zeros((b,H,P,N),dtype=mx.float32); mx.eval(dh0z)
def run_b1():
    return k_b1(dchunk,dA16,dh0z,prev32)
def med(ts): return float(np.median(np.asarray(ts,np.float64)))*1e6

# (1) B1 with eval-per-call (the production pattern: each kernel eval'd / synced)
ts=[]
for i in range(40):
    t0=time.perf_counter(); o=run_b1(); mx.eval(*o);
    if i>=10: ts.append(time.perf_counter()-t0)
m_evalpercall=med(ts)

# (2) N B1 dispatches QUEUED then ONE eval (amortize per-eval sync over N)
N=16
ts=[]
for i in range(20):
    t0=time.perf_counter()
    outs=[run_b1() for _ in range(N)]
    mx.eval(*[a for o in outs for a in o])
    if i>=5: ts.append((time.perf_counter()-t0)/N)
m_queued=med(ts)

# (3) bare mx.eval sync floor of a trivial op (no tvm_ffi)
ts=[]
for i in range(60):
    t0=time.perf_counter(); t=x16*1.0; mx.eval(t)
    if i>=10: ts.append(time.perf_counter()-t0)
m_trivial=med(ts)

print(f"\n=== MLX tvm_ffi PER-DISPATCH floor (B1 smallest kernel) S={SEQ} ===")
print(f"  B1 eval-per-call (prod pattern) = {m_evalpercall:9.1f} us")
print(f"  B1 queued (N={N}, 1 eval/N)     = {m_queued:9.1f} us  (amortized)")
print(f"  per-eval sync component         = {m_evalpercall-m_queued:+9.1f} us")
print(f"  bare mx.eval trivial-op floor   = {m_trivial:9.1f} us")
print(f"PEAK_RSS_KB={_PEAK} (~{_PEAK/1048576:.3f}GB) memguard70=ON")
print("DISPATCH_FLOOR_DONE")
