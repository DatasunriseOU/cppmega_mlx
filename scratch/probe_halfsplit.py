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
L,P,N=64,64,16
H=L//2

# Split DYX output rows into two halves via two separate offset-0 A-operand tiles.
def build():
    @T.prim_func
    def main(dY: T.Tensor((L,P), T.float16), Xm: T.Tensor((L,P), T.float16),
             Cm: T.Tensor((L,N), T.float16),
             OutDYX: T.Tensor((L,L), T.float16), OutDC: T.Tensor((P,N), T.float32)):
        with T.Kernel(1, threads=128) as bx:
            dY16=T.alloc_shared((L,P),T.float16)
            DYX=T.alloc_shared((L,L),T.float16)
            opA=T.alloc_shared((L,P),T.float16)
            opB=T.alloc_shared((L,N),T.float16)
            dyA=T.alloc_shared((H,P),T.float16)   # half A-operand (offset 0)
            hfrag=T.alloc_fragment((H,L),T.float32) # half DYX frag staging = 8192B
            dcf=T.alloc_fragment((P,N),T.float32)
            T.copy(dY,dY16); T.copy(Xm,opA)
            for hh in T.serial(0,2):
                for ip in T.Parallel(H*P):
                    dyA[ip//P, ip%P] = dY16[hh*H + ip//P, ip%P]
                T.clear(hfrag)
                T.gemm(dyA, opA, hfrag, transpose_B=True)
                T.copy(hfrag, DYX[hh*H:hh*H+H, 0:L])
            T.copy(Cm,opB)
            T.clear(dcf); T.gemm(dY16,opB,dcf,transpose_A=True); T.copy(dcf,OutDC)
            for ls in T.Parallel(L*L):
                OutDYX[ls//L,ls%L]=DYX[ls//L,ls%L]
    return main

mod=tilelang.lower(build(), target="metal")
src=mod.kernel_source if hasattr(mod,'kernel_source') else str(mod)
pools=re.findall(r'threadgroup\s+uchar\s+(\w+)\[(\d+)\]', src)
tot=sum(int(n) for _,n in pools)
print(f"pools={pools} total={tot} smma={src.count('simdgroup_multiply_accumulate')}")
k=tilelang.compile(build(), out_idx=[3,4], target="metal")
print("PSO BUILT OK:", k is not None, "total", tot)
