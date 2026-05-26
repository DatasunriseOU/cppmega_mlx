#!/usr/bin/env python3
"""Profile Path C fused train-block compile phases on minimal route mixes."""

from __future__ import annotations

import argparse
import cProfile
import contextlib
import io
import json
import pstats
import re
import resource
import time
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

from cppmega_mlx.recipes.model_factory import local_gb10_quarter_profile
from cppmega_mlx.runtime.path_c_fusion import (
    build_path_c_aot_autograd_region,
    build_path_c_model_regions_from_model,
    tilelang_single_entry_lowerer,
)
from cppmega_mlx.runtime.path_c_fusion_schedules import (
    select_path_c_fusion_schedule_target,
)


_DEFAULT_CASES = ("M", "A", "MRA")
_DTYPE_MAP = {
    "float32": "float32",
    "float16": "float16",
    "bfloat16": "bfloat16",
    "uint8": "uint8",
    "int32": "int32",
}


@dataclass(frozen=True)
class TimedValue:
    value: Any
    seconds: float


def _rss_bytes() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def _timed(fn: Any) -> TimedValue:
    started = time.perf_counter()
    value = fn()
    return TimedValue(value=value, seconds=time.perf_counter() - started)


def _count_source(source: str) -> dict[str, int | bool]:
    shmem_match = re.search(r"threadgroup float buf_dyn_shmem\[(\d+)\];", source)
    shmem_elements = int(shmem_match.group(1)) if shmem_match is not None else 0
    return {
        "bytes": len(source.encode("utf-8")),
        "lines": len(source.splitlines()),
        "t_min": source.count("T.min"),
        "t_max": source.count("T.max"),
        "alloc_shared": source.count("T.alloc_shared"),
        "alloc_local": source.count("T.alloc_local"),
        "atomic_add": source.count("T.atomic_add"),
        "tl_atomic_add": source.count("tl::AtomicAdd"),
        "buf_dyn_shmem": source.count("buf_dyn_shmem"),
        "threadgroup_dynamic_shared_bytes": shmem_elements * 4,
        "threadgroup_atomic_add": "AtomicAdd((&(buf_dyn_shmem" in source,
    }


def _case_region(case: str, *, seq: int, include_backward: bool) -> Any:
    cfg = local_gb10_quarter_profile().tiny_smoke_config(
        pattern=case,
        depth=len(case),
        dsa_a_layer_ranks=tuple(range(case.count("A"))),
        max_seq_length=seq,
    )
    model = SimpleNamespace(
        name=f"profile_path_c_{case.lower()}",
        route_symbols=tuple(case),
        config=cfg,
    )
    regions = build_path_c_model_regions_from_model(
        model,
        region_prefix=f"profile_path_c_{case.lower()}",
        min_route_bricks=1,
        sequence_length=seq,
    )
    if not regions:
        raise RuntimeError(f"no Path C regions generated for case {case!r}")
    region = max(regions, key=lambda item: (len(item.nodes), len(item.edges), item.name))
    return build_path_c_aot_autograd_region(region) if include_backward else region


def _compile_case(case: str, *, seq: int, include_backward: bool, launch: bool) -> dict[str, Any]:
    rss_before = _rss_bytes()
    region_t = _timed(lambda: _case_region(case, seq=seq, include_backward=include_backward))
    region = region_t.value
    target = select_path_c_fusion_schedule_target(region)
    if target is None:
        raise RuntimeError(f"no schedule target selected for case {case!r}")

    prim_t = _timed(lambda: target.schedule_template(region))
    prim_func = prim_t.value
    generated_source = str(getattr(prim_func, "_cppmega_path_c_generated_source", ""))
    script_t = _timed(lambda: str(prim_func.script()))
    tilelang_t = _timed(lambda: tilelang_single_entry_lowerer(prim_func, target="metal"))
    artifact = tilelang_t.value
    kernel_source = str(artifact.get_kernel_source())

    launch_payload: dict[str, Any] = {"enabled": False}
    if launch:
        launch_payload = _launch_artifact_once(artifact)

    return {
        "case": case,
        "sequence_length": seq,
        "include_backward": include_backward,
        "node_ops": [str(node.op_name) for node in region.nodes],
        "timing_seconds": {
            "region": region_t.seconds,
            "prim_func": prim_t.seconds,
            "prim_func_script": script_t.seconds,
            "tilelang_lowerer": tilelang_t.seconds,
            **(
                {"first_launch": launch_payload["seconds"]}
                if launch_payload.get("enabled")
                else {}
            ),
        },
        "generated_source": _count_source(generated_source),
        "tir_script": _count_source(script_t.value),
        "metal_kernel_source": _count_source(kernel_source),
        "prim_func": {
            "params": len(tuple(prim_func.params)),
            "spilled_shared_scratch_count": len(
                getattr(prim_func, "_cppmega_path_c_spilled_shared_scratch_shapes", {})
            ),
            "internal_scratch_abi_count": len(
                getattr(prim_func, "_cppmega_path_c_internal_scratch_abi_buffers", ())
            ),
        },
        "rss_bytes": {
            "before": rss_before,
            "after": _rss_bytes(),
        },
        "launch": launch_payload,
    }


def _launch_artifact_once(artifact: Any) -> dict[str, Any]:
    import mlx.core as mx

    args: list[Any] = []
    prim_func = artifact.prim_func
    for param in prim_func.params:
        buffer = prim_func.buffer_map.get(param)
        if buffer is None:
            args.append(0)
            continue
        dtype_name = str(buffer.dtype)
        mx_dtype_name = _DTYPE_MAP.get(dtype_name)
        if mx_dtype_name is None:
            raise RuntimeError(f"unsupported launch dtype {dtype_name!r}")
        shape = tuple(int(dim) for dim in buffer.shape)
        args.append(mx.zeros(shape, dtype=getattr(mx, mx_dtype_name)))
    mx_args = [arg for arg in args if isinstance(arg, mx.array)]
    if mx_args:
        mx.eval(*mx_args)
    started = time.perf_counter()
    returned = artifact(*args)
    elapsed = time.perf_counter() - started
    returned_arrays = (
        (returned,)
        if isinstance(returned, mx.array)
        else tuple(item for item in (returned or ()) if isinstance(item, mx.array))
    )
    if returned_arrays:
        mx.eval(*returned_arrays)
    if mx_args:
        mx.eval(*mx_args)
    return {
        "enabled": True,
        "seconds": elapsed,
        "argument_count": len(args),
        "mx_argument_count": len(mx_args),
        "returned_array_count": len(returned_arrays),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        action="append",
        choices=("M", "A", "MR", "RA", "MRA"),
        help="Route-symbol mix to profile. Repeatable. Defaults to M, A, MRA.",
    )
    parser.add_argument("--seq", type=int, default=64)
    parser.add_argument("--forward-only", action="store_true")
    parser.add_argument("--launch", action="store_true", help="Also force first kernel launch.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    parser.add_argument("--cprofile", action="store_true", help="Print Python cProfile top entries.")
    parser.add_argument("--profile-limit", type=int, default=30)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    cases = tuple(args.case or _DEFAULT_CASES)
    include_backward = not bool(args.forward_only)

    def run() -> list[dict[str, Any]]:
        return [
            _compile_case(
                case,
                seq=int(args.seq),
                include_backward=include_backward,
                launch=bool(args.launch),
            )
            for case in cases
        ]

    if args.cprofile:
        profiler = cProfile.Profile()
        profiler.enable()
        results = run()
        profiler.disable()
        stats_out = io.StringIO()
        stats = pstats.Stats(profiler, stream=stats_out).sort_stats("cumtime")
        stats.print_stats(int(args.profile_limit))
    else:
        results = run()
        stats_out = None

    if args.json:
        print(json.dumps({"results": results}, indent=2, sort_keys=True))
    else:
        for result in results:
            print(
                f"{result['case']}: tilelang={result['timing_seconds']['tilelang_lowerer']:.3f}s "
                f"metal_bytes={result['metal_kernel_source']['bytes']} "
                f"threadgroup_atomic={result['metal_kernel_source']['threadgroup_atomic_add']} "
                f"rss={result['rss_bytes']['after']}"
            )
    if stats_out is not None:
        print(stats_out.getvalue())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
