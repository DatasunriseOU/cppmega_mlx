import os, sys, threading, time
_LIM=70*1024*1024
def _rss():
    import resource; return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss//1024)
def _g():
    while True:
        if _rss()>_LIM: os._exit(137)
        time.sleep(0.25)
threading.Thread(target=_g,daemon=True).start()
import tilelang, tilelang.language as T, re
L,P,N=64,64,64
NT=N//2  # 32

# dchunk[p,n] = sum_l dY[l,p]*opB[l,n] (transpose_A). N-split: slice opB cols + output cols.
def build():
    @T.prim_func
    def main(dY: T.Tensor((L,P), T.float16), Cm: T.Tensor((L,N), T.float16),
             Out: T.Tensor((P,N), T.float32)):
        with T.Kernel(1, threads=128) as bx:
            dY16=T.alloc_shared((L,P),T.float16)
            opBh=T.alloc_shared((L,NT),T.float16)   # half-N B-operand (offset 0)
            stg=T.alloc_shared((P,NT),T.float16)   # half-N staging
            dcf=T.alloc_fragment((P,NT),T.float32)
            T.copy(dY,dY16)
            for n0 in T.serial(0, N, NT):
                for ln in T.Parallel(L*NT):
                    opBh[ln//NT, ln%NT] = Cm[ln//NT, n0 + ln%NT]
                T.sync_threads()
                T.clear(dcf)
                T.gemm(dY16, opBh, dcf, transpose_A=True)
                T.copy(dcf, stg)
                for pn in T.Parallel(P*NT):
                    Out[pn//NT, n0 + pn%NT] = T.Cast(T.float32, stg[pn//NT, pn%NT])
                T.sync_threads()
    return main

mod=tilelang.lower(build(), target="metal")
src=mod.kernel_source if hasattr(mod,'kernel_source') else str(mod)
pool=sum(int(n) for n in re.findall(r'threadgroup\s+uchar\s+\w+\[(\d+)\]', src))
print(f"pool={pool} smma={src.count('simdgroup_multiply_accumulate')} s8x8={src.count('simdgroup_float8x8')}")
k=tilelang.compile(build(), out_idx=[2], target="metal")
print("PSO BUILT OK:", k is not None)
