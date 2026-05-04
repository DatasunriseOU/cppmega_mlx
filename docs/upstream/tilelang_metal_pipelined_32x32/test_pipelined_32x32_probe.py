"""Regression probes for Metal 32x32 pipelined fragments.

These kernels stack on the 3D leading-dimension fix in
../tilelang_metal_pipelined and exercise the follow-up C++ StorageRewrite fix:
metal.simdgroup buffers must keep scalar pointer element types instead of being
rewritten to float32x4 when 32x32 fragments produce ramp accesses.
"""

from __future__ import annotations

import os
import pickle
import re
import shutil
import subprocess
import sys
import tempfile
import types
from pathlib import Path

import pytest

_THIS_DIR = Path(__file__).resolve().parent
_PATCH_PATH = (
    _THIS_DIR
    / "0001-metal-keep-simdgroup-storage-scalar-for-pipelined-32x32.patch"
)


def _install_import_stubs() -> None:
    if "psutil" not in sys.modules:
        psutil_stub = types.ModuleType("psutil")
        psutil_stub.Process = lambda *args, **kwargs: None
        psutil_stub.cpu_count = lambda logical=True: 1
        sys.modules["psutil"] = psutil_stub

    if "cloudpickle" not in sys.modules:
        cloudpickle_stub = types.ModuleType("cloudpickle")
        cloudpickle_stub.dumps = pickle.dumps
        cloudpickle_stub.loads = pickle.loads
        sys.modules["cloudpickle"] = cloudpickle_stub

    if "tqdm" not in sys.modules:
        tqdm_stub = types.ModuleType("tqdm")
        tqdm_auto_stub = types.ModuleType("tqdm.auto")

        def _tqdm(iterable=None, *args, **kwargs):
            return iterable if iterable is not None else []

        _tqdm.write = lambda *args, **kwargs: None
        tqdm_stub.tqdm = _tqdm
        tqdm_auto_stub.tqdm = _tqdm
        sys.modules["tqdm"] = tqdm_stub
        sys.modules["tqdm.auto"] = tqdm_auto_stub


_install_import_stubs()

_TILELANG_IMPORT_ERROR = None
try:
    import tilelang
    import tilelang.language as T
    from tilelang import tvm
    from tvm.target import Target
except Exception as exc:  # pragma: no cover - depends on local build env
    tilelang = None
    T = None
    tvm = None
    Target = None
    _TILELANG_IMPORT_ERROR = exc


def _require_tilelang() -> None:
    if _TILELANG_IMPORT_ERROR is not None:
        pytest.skip(f"tilelang/tvm Metal lowering deps unavailable: {_TILELANG_IMPORT_ERROR}")


def _assert_simdgroup_scalar_guard_text(text: str, *, require_diff_header: bool) -> None:
    normalized = re.sub(r"(?m)^[+]", "", text)
    guard = 'GetPtrStorageScope(var_info.var) == "metal.simdgroup"'
    if require_diff_header:
        assert "src/transform/storage_rewrite.cc" in text
    assert "simdgroup_matrix<T, 8, 8>" in normalized
    assert guard in normalized
    assert re.search(
        r"if\s*\(\s*GetPtrStorageScope\(var_info\.var\)\s*==\s*"
        r'"metal\.simdgroup"\s*\)\s*\{\s*continue;\s*\}',
        normalized,
        flags=re.DOTALL,
    )
    assert normalized.index(guard) < normalized.index(
        "DataType preferred = var_info.get_preferred_dtype();"
    )
    assert "TODO" not in normalized
    assert "future patch" not in normalized.lower()


def _find_storage_rewrite_source() -> Path | None:
    if tilelang is None:
        return None
    package_init = Path(tilelang.__file__).resolve()
    for parent in package_init.parents:
        candidate = parent / "src" / "transform" / "storage_rewrite.cc"
        if candidate.exists():
            return candidate
    return None


def _has_metal_sdk() -> bool:
    if shutil.which("xcrun") is None:
        return False
    result = subprocess.run(
        ["xcrun", "--sdk", "macosx", "--find", "metal"],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


_HAS_METAL_SDK = _has_metal_sdk()


def _make_gemm_32x32_pipe2_kernel():
    _require_tilelang()

    @T.prim_func
    def gemm_32x32_pipe2(
        A: T.Tensor((64, 64), "float16"),
        B: T.Tensor((64, 64), "float16"),
        C: T.Tensor((64, 64), "float16"),
    ):
        with T.Kernel(2, 2, threads=256) as (bx, by):
            A_shared = T.alloc_shared((32, 32), "float16")
            B_shared = T.alloc_shared((32, 32), "float16")
            C_local = T.alloc_fragment((32, 32), "float32")
            T.clear(C_local)
            for ko in T.Pipelined(2, num_stages=2):
                T.copy(
                    A[by * 32 : (by + 1) * 32, ko * 32 : (ko + 1) * 32],
                    A_shared,
                )
                T.copy(
                    B[ko * 32 : (ko + 1) * 32, bx * 32 : (bx + 1) * 32],
                    B_shared,
                )
                T.gemm(A_shared, B_shared, C_local, policy=T.GemmWarpPolicy.FullCol)
            T.copy(C_local, C[by * 32 : (by + 1) * 32, bx * 32 : (bx + 1) * 32])

    return gemm_32x32_pipe2


def _make_sparse_mla_32x32_pipe2_kernel():
    _require_tilelang()

    @T.prim_func
    def sparse_mla_32x32_pipe2(
        Q: T.Tensor((1, 32, 32), "float16"),
        Q_pe: T.Tensor((1, 32, 32), "float16"),
        KV: T.Tensor((1, 64, 1, 32), "float16"),
        K_pe: T.Tensor((1, 64, 1, 32), "float16"),
        Output: T.Tensor((1, 32, 32), "float16"),
    ):
        with T.Kernel(1, 1, threads=256) as (hid, bid):
            Q_shared = T.alloc_shared((32, 32), "float16")
            Q_pe_shared = T.alloc_shared((32, 32), "float16")
            KV_shared = T.alloc_shared((32, 32), "float16")
            K_pe_shared = T.alloc_shared((32, 32), "float16")
            S_shared = T.alloc_shared((32, 32), "float16")
            O_shared = T.alloc_shared((32, 32), "float16")
            acc_s = T.alloc_fragment((32, 32), "float32")
            acc_o = T.alloc_fragment((32, 32), "float32")

            T.copy(Q[bid, hid * 32 : (hid + 1) * 32, :], Q_shared)
            T.copy(Q_pe[bid, hid * 32 : (hid + 1) * 32, :], Q_pe_shared)
            T.clear(acc_o)

            for k in T.Pipelined(2, num_stages=2):
                T.copy(KV[bid, k * 32 : (k + 1) * 32, 0, :], KV_shared)
                T.copy(K_pe[bid, k * 32 : (k + 1) * 32, 0, :], K_pe_shared)
                T.gemm(
                    Q_shared,
                    KV_shared,
                    acc_s,
                    transpose_B=True,
                    policy=T.GemmWarpPolicy.FullCol,
                    clear_accum=True,
                )
                T.gemm(
                    Q_pe_shared,
                    K_pe_shared,
                    acc_s,
                    transpose_B=True,
                    policy=T.GemmWarpPolicy.FullCol,
                )
                for i, j in T.Parallel(32, 32):
                    acc_s[i, j] = T.exp2(acc_s[i, j] * 1.4426950408889634)
                T.copy(acc_s, S_shared)
                T.gemm(S_shared, KV_shared, acc_o, policy=T.GemmWarpPolicy.FullCol)

            T.copy(acc_o, O_shared)
            T.copy(O_shared, Output[bid, hid * 32 : (hid + 1) * 32, :])

    return sparse_mla_32x32_pipe2


def _lower_source(fn) -> str:
    _require_tilelang()
    with tvm.transform.PassContext(), tvm.target.Target("metal"):
        artifact = tilelang.lower(fn, target=Target("metal"))
    source = getattr(artifact, "kernel_source", None)
    if source:
        return source
    rt_mod = getattr(artifact, "rt_mod", None)
    if rt_mod is not None and hasattr(rt_mod, "get_source"):
        return rt_mod.get_source()
    raise AssertionError("lower() did not expose generated Metal source")


def _kernel_body(source: str) -> str:
    idx = source.find("kernel void")
    assert idx >= 0, "lowered Metal source must contain a kernel entry point"
    return source[idx:]


def _threadgroup_half_allocs(source: str) -> dict[str, int]:
    allocs: dict[str, int] = {}
    pattern = re.compile(r"\bthreadgroup\s+half\s+(\w+)\[(\d+)\]")
    for name, size in pattern.findall(source):
        allocs[name] = int(size)
    return allocs


def _assert_32x32_pipeline_msl(source: str, *, min_mma: int, min_half_storage: int) -> None:
    body = _kernel_body(source)
    lowered = body.lower()
    half_allocs = _threadgroup_half_allocs(body)

    assert "threadgroup half" in lowered
    assert sum(half_allocs.values()) >= min_half_storage, half_allocs
    assert "threadgroup_barrier" in lowered
    assert "simdgroup_multiply_accumulate" in body
    assert body.count("simdgroup_multiply_accumulate") >= min_mma
    assert "simdgroup_matrix<float, 8, 8>" in body
    assert "simdgroup_matrix<float32x4" not in body
    assert "simdgroup_matrix<float4" not in body
    assert "metal.simdgroup" not in lowered
    assert "3-dimensional" not in source


def test_patch_is_concrete_storage_rewrite_cpp_fix():
    patch = _PATCH_PATH.read_text(encoding="utf-8")
    assert "diff --git a/src/transform/storage_rewrite.cc" in patch
    assert "+++ b/src/transform/storage_rewrite.cc" in patch
    _assert_simdgroup_scalar_guard_text(patch, require_diff_header=True)


def test_local_storage_rewrite_contains_simdgroup_scalar_guard_when_available():
    source_path = _find_storage_rewrite_source()
    if source_path is None:
        pytest.skip("TileLang source checkout with src/transform/storage_rewrite.cc not found")
    _assert_simdgroup_scalar_guard_text(
        source_path.read_text(encoding="utf-8"),
        require_diff_header=False,
    )


def _xcrun_compile(msl_source: str) -> tuple[int, str]:
    with tempfile.NamedTemporaryFile(suffix=".metal", delete=False) as f:
        f.write(msl_source.encode("utf-8"))
        msl_path = f.name
    try:
        air_path = msl_path + ".air"
        res = subprocess.run(
            ["xcrun", "--sdk", "macosx", "metal", "-c", msl_path, "-o", air_path],
            capture_output=True,
            text=True,
        )
        return res.returncode, res.stderr or ""
    finally:
        for path in (msl_path, msl_path + ".air"):
            if os.path.exists(path):
                os.remove(path)


def _collect_metrics() -> list[dict[str, object]]:
    kernels = [
        ("gemm_32x32_pipe2", _make_gemm_32x32_pipe2_kernel(), 2, 4096),
        ("sparse_mla_32x32_pipe2", _make_sparse_mla_32x32_pipe2_kernel(), 3, 10240),
    ]
    rows = []
    for name, kernel, min_mma, min_half_storage in kernels:
        source = _lower_source(kernel)
        body = _kernel_body(source)
        half_allocs = _threadgroup_half_allocs(body)
        rows.append(
            {
                "name": name,
                "lines": len(source.splitlines()),
                "bytes": len(source.encode("utf-8")),
                "half_allocs": half_allocs,
                "half_storage": sum(half_allocs.values()),
                "mma": body.count("simdgroup_multiply_accumulate"),
                "min_mma": min_mma,
                "min_half_storage": min_half_storage,
            }
        )
    return rows


def test_gemm_32x32_pipeline_lowers_without_simdgroup_vector_type():
    source = _lower_source(_make_gemm_32x32_pipe2_kernel())
    _assert_32x32_pipeline_msl(source, min_mma=2, min_half_storage=4096)


def test_sparse_mla_32x32_pipeline_lowers_without_simdgroup_vector_type():
    source = _lower_source(_make_sparse_mla_32x32_pipe2_kernel())
    _assert_32x32_pipeline_msl(source, min_mma=3, min_half_storage=10240)
    assert "exp2" in source.lower()


@pytest.mark.skipif(not _HAS_METAL_SDK, reason="macOS Metal SDK is unavailable")
def test_gemm_32x32_pipeline_xcrun_compiles():
    source = _lower_source(_make_gemm_32x32_pipe2_kernel())
    rc, stderr = _xcrun_compile(source)
    assert rc == 0, f"xcrun metal -c failed:\n{stderr}"


if __name__ == "__main__":
    _require_tilelang()
    print(
        f"{'kernel':<28} {'lines':>5} {'bytes':>7} {'half':>7} "
        f"{'mma':>4} threadgroup half allocs"
    )
    for row in _collect_metrics():
        allocs = ", ".join(
            f"{name}[{size}]" for name, size in sorted(row["half_allocs"].items())
        )
        print(
            f"{row['name']:<28} {row['lines']:>5} {row['bytes']:>7} "
            f"{row['half_storage']:>7} {row['mma']:>4} {allocs}"
        )
