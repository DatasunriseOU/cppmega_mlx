import pathlib
import subprocess

import tvm
from tvm.script import tirx as T


@T.prim_func
def fp8_scalar(
    A: T.Buffer((16,), "float8_e4m3fn"),
    B: T.Buffer((16,), "float8_e5m2"),
    S: T.Buffer((16,), "float8_e8m0fnu"),
    C: T.Buffer((16,), "float16"),
    D: T.Buffer((16,), "float8_e4m3fn"),
    E: T.Buffer((16,), "float8_e5m2"),
):
    T.func_attr({"tirx.noalias": True, "global_symbol": "fp8_scalar"})
    sh = T.sblock_alloc_buffer((16,), "float16", scope="shared")
    for i in T.thread_binding(16, thread="threadIdx.x"):
        with T.sblock("decode"):
            vi = T.axis.spatial(16, i)
            sh[vi] = T.Cast("float16", A[vi]) + T.Cast("float16", B[vi]) + T.Cast("float16", S[vi])
    T.tvm_storage_sync("shared")
    for i in T.thread_binding(16, thread="threadIdx.x"):
        with T.sblock("encode"):
            vi = T.axis.spatial(16, i)
            C[vi] = sh[vi]
            D[vi] = T.Cast("float8_e4m3fn", sh[vi])
            E[vi] = T.Cast("float8_e5m2", sh[vi])


@T.prim_func
def fp8_vector(
    A: T.Buffer((16,), "float8_e4m3fn"),
    B: T.Buffer((16,), "float8_e5m2"),
    S: T.Buffer((16,), "float8_e8m0fnu"),
    C: T.Buffer((16,), "float16"),
    D: T.Buffer((16,), "float8_e4m3fn"),
    E: T.Buffer((16,), "float8_e5m2"),
):
    T.func_attr({"tirx.noalias": True, "global_symbol": "fp8_vector"})
    for i in T.thread_binding(4, thread="threadIdx.x"):
        for j in T.vectorized(4):
            with T.sblock("block"):
                vi = T.axis.spatial(16, i * 4 + j)
                h = T.Cast("float16", A[vi]) + T.Cast("float16", B[vi]) + T.Cast("float16", S[vi])
                C[vi] = h
                D[vi] = T.Cast("float8_e4m3fn", h)
                E[vi] = T.Cast("float8_e5m2", h)


def emit_and_compile(func, name):
    mod = tvm.IRModule({name: func})
    built = tvm.tirx.build(mod, target="metal")
    src = built.imports[0].inspect_source()
    path = pathlib.Path(f"/tmp/{name}.metal")
    out = pathlib.Path(f"/tmp/{name}.air")
    path.write_text(src)
    subprocess.run(["xcrun", "metal", "-c", str(path), "-o", str(out)], check=True)
    print(f"{name}: compiled {path} -> {out}")
    for needle in [
        "__tvm_fp8_e4m3_to_half",
        "__tvm_fp8_e5m2_to_half",
        "__tvm_fp8_e8m0_to_half",
        "__tvm_half_to_fp8_e4m3",
        "__tvm_half_to_fp8_e5m2",
    ]:
        assert needle in src, needle
    return src


scalar_src = emit_and_compile(fp8_scalar, "fp8_scalar")
vector_src = emit_and_compile(fp8_vector, "fp8_vector")
assert "threadgroup half" in scalar_src
assert "threadgroup_barrier(mem_flags::mem_threadgroup)" in scalar_src
assert "uchar4" in vector_src
assert "half4" in vector_src
