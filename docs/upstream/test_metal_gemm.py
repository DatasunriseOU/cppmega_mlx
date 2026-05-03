import tilelang
import tilelang.language as T
from tilelang import tvm as tvm_inner


@T.prim_func
def metal_gemm_test(
    A: T.Tensor((64, 32), "float16"),
    B: T.Tensor((32, 64), "float16"),
    C: T.Tensor((64, 64), "float16"),
):
    with T.Kernel(T.ceildiv(64, 16), T.ceildiv(64, 16), threads=64) as (bx, by):
        A_shared = T.alloc_shared((16, 32), "float16", scope="shared")
        B_shared = T.alloc_shared((32, 16), "float16", scope="shared")
        C_local = T.alloc_fragment((16, 16), "float16")
        T.clear(C_local)
        T.copy(A[by * 16, 0], A_shared)
        T.copy(B[0, bx * 16], B_shared)
        T.gemm(A_shared, B_shared, C_local)
        T.copy(C_local, C[by * 16, bx * 16])


target = tvm_inner.target.Target("metal")
try:
    from tilelang.engine.lower import lower
    result = lower(metal_gemm_test, target=target)
    print("METAL T.GEMM LOWERING: OK")
    print("type:", type(result).__name__)
except Exception as e:
    print("METAL T.GEMM LOWERING: FAILED")
    print("Error type:", type(e).__name__)
    print("Error:", str(e)[:600])
