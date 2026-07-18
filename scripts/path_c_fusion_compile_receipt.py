#!/usr/bin/env python3
"""Emit a receipt for the model-derived Path C fused schedule compile.

The training matrix measures the current runtime route. This receipt is the
separate proof for the dynamic fused train-block schedule selected from the
local_gb10_quarter model bricks, so reports can distinguish "runtime split
surfaces" from "one generated fused schedule compiles".
"""

from __future__ import annotations

import argparse
import contextlib
from dataclasses import replace
import json
import math
import os
import platform
import re
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cppmega_mlx.recipes.model_factory import local_gb10_quarter_profile  # noqa: E402
from cppmega_mlx.runtime.path_c_fusion import (  # noqa: E402
    CompiledPathCRegion,
    FusionCompilePlan,
    build_path_c_aot_autograd_region,
    build_path_c_model_regions_from_model,
    compile_path_c_region,
    mark_path_c_schedule_template_for_region,
    tilelang_single_entry_lowerer,
)
from cppmega_mlx.runtime.path_c_fusion_schedules import (  # noqa: E402
    DESCRIPTOR_DEFAULT_MAX_ROWS_PER_LAUNCH,
    DESCRIPTOR_EXECUTION_STAGE_BACKWARD,
    DESCRIPTOR_EXECUTION_STAGE_FORWARD,
    DESCRIPTOR_PHYSICAL_ABI_POLICY_DIRECT,
    DESCRIPTOR_ROW_DISPATCH_LAUNCHER_CHUNKS,
    DESCRIPTOR_ROWS_PER_KERNEL_LAUNCH,
    make_path_c_descriptor_schedule_template,
    make_path_c_descriptor_stage_schedule_template,
    merge_path_c_physical_abi_for_prim_funcs,
    path_c_fusion_schedule_spec,
    plan_path_c_descriptor_phase_groups,
    plan_path_c_descriptor_stage_groups,
    plan_path_c_direct_fusion_chain_for_region,
    plan_path_c_direct_fusion_chains_for_model,
    select_path_c_fusion_schedule_target,
)
from cppmega_mlx.runtime.path_c_physical_abi import (  # noqa: E402
    make_physical_abi_bank_owner,
    path_c_kernel_buffer_order,
    plan_physical_abi_runtime_bridge,
    validate_physical_abi_map,
    validate_physical_abi_runtime_bindings,
)
from cppmega_mlx.training.compiled import (  # noqa: E402
    PathCFusedTrainBlockCallableArtifact,
    PathCGeneratedStageTrainBlockCallableArtifact,
)


DEFAULT_OUTPUT = ROOT / "reports" / "path_c_fusion_compile_receipt.json"
DEFAULT_SOURCE_OUTPUT = ROOT / "reports" / "path_c_fusion_compile_source.py"
RECEIPT_SCHEMA_VERSION = 1
METAL_KERNEL_BUFFER_LIMIT = 31


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compile the selected local_gb10_quarter Path C fused schedule.",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--source-out", type=Path, default=DEFAULT_SOURCE_OUTPUT)
    parser.add_argument(
        "--no-native-compile",
        action="store_true",
        help="Generate the schedule and plan receipt without calling tilelang.compile.",
    )
    parser.add_argument("--target", default="metal")
    parser.add_argument("--execution-backend", default="tvm_ffi")
    parser.add_argument(
        "--runtime-smoke",
        choices=("none", "tiny", "production"),
        default="tiny",
        help=(
            "Optionally compile and execute a model-derived banked-ABI fused "
            "schedule. The production mode uses the selected full 1B-shaped "
            "region and is byte-budget gated."
        ),
    )
    parser.add_argument(
        "--runtime-smoke-max-bytes",
        type=int,
        default=256 * 1024 * 1024,
        help="Skip runtime smoke execution if the smoke ABI exceeds this byte budget.",
    )
    parser.add_argument(
        "--runtime-smoke-max-launches",
        type=int,
        default=None,
        help=(
            "Optionally cap generated-stage forward and backward launches. "
            "This validates binding without claiming a full production execution."
        ),
    )
    parser.add_argument(
        "--runtime-smoke-max-stages",
        type=int,
        default=None,
        help=(
            "Optionally cap generated stages per launch. This is a diagnostic "
            "binding smoke budget and always prevents a full-execution claim."
        ),
    )
    parser.add_argument(
        "--runtime-smoke-artifact",
        choices=("single", "staged", "phased"),
        default="single",
        help=(
            "Choose whether runtime smoke dispatches the single generated "
            "train-block artifact, the diagnostic generated-stage artifact, "
            "or the descriptor-fused forward/backward phase artifact."
        ),
    )
    return parser


def _git_sha(cwd: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=cwd,
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _tilelang_root() -> Path | None:
    sibling = ROOT.parent / "tilelang"
    if (sibling / "tilelang").exists() and (sibling / ".git").exists():
        return sibling
    return None


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


@contextlib.contextmanager
def _capture_native_output() -> Any:
    """Capture Python and C/C++ writes to stdout/stderr during native lowering."""

    capture = SimpleNamespace(stdout="", stderr="")
    sys.stdout.flush()
    sys.stderr.flush()
    saved_stdout = os.dup(1)
    saved_stderr = os.dup(2)
    with tempfile.TemporaryFile() as stdout_file, tempfile.TemporaryFile() as stderr_file:
        try:
            os.dup2(stdout_file.fileno(), 1)
            os.dup2(stderr_file.fileno(), 2)
            yield capture
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            stdout_file.flush()
            stderr_file.flush()
            stdout_file.seek(0)
            stderr_file.seek(0)
            capture.stdout = stdout_file.read().decode(errors="replace")
            capture.stderr = stderr_file.read().decode(errors="replace")
            os.dup2(saved_stdout, 1)
            os.dup2(saved_stderr, 2)
            os.close(saved_stdout)
            os.close(saved_stderr)


def _plan_payload(plan: FusionCompilePlan) -> dict[str, Any]:
    contract = plan.schedule_contract
    return {
        "region_name": plan.region_name,
        "lowering_boundary": plan.lowering_boundary,
        "backend": plan.backend,
        "compiler": plan.compiler,
        "fusion_kind": plan.fusion_kind,
        "schedule_name": plan.schedule_name,
        "schedule_status": plan.schedule_status,
        "single_kernel_fused": plan.single_kernel_fused,
        "backward_graph": plan.backward_graph,
        "autograd_status": plan.autograd_status,
        "autograd_backward_nodes": list(plan.autograd_backward_nodes),
        "autograd_missing_backward_nodes": list(plan.autograd_missing_backward_nodes),
        "semantic_blockers": [
            {
                "kind": blocker.kind,
                "producer": blocker.producer,
                "consumer": blocker.consumer,
                "required_node": blocker.required_node,
                "reason": blocker.reason,
            }
            for blocker in plan.semantic_blockers
        ],
        "schedule_contract": None
        if contract is None
        else {
            "name": contract.name,
            "key": contract.key,
            "status": contract.status,
            "reason": contract.reason,
            "shape_env_key": contract.shape_env_key,
            "declared_key": contract.declared_key,
            "declared_implementation_kind": contract.declared_implementation_kind,
            "declared_schedule_id": contract.declared_schedule_id,
            "declared_required_real_abi_inputs": list(
                contract.declared_required_real_abi_inputs
            ),
            "missing_real_abi_inputs": list(contract.missing_real_abi_inputs),
            "op_signature": list(contract.op_signature),
            "required_internal_buffers": list(contract.required_internal_buffers),
            "required_external_buffers": list(contract.required_external_buffers),
        },
    }


def _fusion_cache_key_payload(compiled: FusionCompilePlan | CompiledPathCRegion) -> dict[str, Any]:
    plan = compiled.plan if isinstance(compiled, CompiledPathCRegion) else compiled
    lowered_module = (
        compiled.lowered_module if isinstance(compiled, CompiledPathCRegion) else None
    )
    material = list(plan.cache_key_parts)
    payload: dict[str, Any] = {
        "cache_key_material": material,
        "cache_key_material_sha256": __import__("hashlib")
        .sha256(json.dumps(material, sort_keys=True).encode("utf-8"))
        .hexdigest(),
        "lowered_module_digest": None,
        "status": "plan_only_no_lowered_module",
        "reason": (
            "compile plan cache-key material is available, but no lowered "
            "TileLang IRModule was captured"
        ),
    }
    if lowered_module is None:
        return payload
    try:
        try:
            module_script = lowered_module.script(show_meta=True)
        except TypeError:
            module_script = lowered_module.script()
        digest_payload = {
            "cache_key_material": material,
            "lowered_module": str(module_script),
        }
        payload["lowered_module_digest"] = __import__("hashlib").sha256(
            json.dumps(digest_payload, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()
        payload["status"] = "lowered_module_digest_recorded"
        payload["reason"] = (
            "digest covers compile plan cache-key material and the lowered "
            "single-entry TileLang IRModule"
        )
    except Exception as exc:  # pragma: no cover - defensive receipt path
        payload["status"] = "digest_failed"
        payload["reason"] = f"{type(exc).__name__}: {exc}"
    return payload


def _compile_path_c_region_with_native_capture(
    *,
    region: Any,
    schedule_template: Callable[[Any], Any],
    schedule_name: str,
    schedule_status: str,
    native_lowerer: Callable[..., Any] | None,
    target_name: str,
) -> tuple[CompiledPathCRegion | FusionCompilePlan, str, str, str | None, float]:
    """Compile once and capture Python/C++ output plus recoverable exceptions."""

    native_output = SimpleNamespace(stdout="", stderr="")
    started = time.perf_counter()
    try:
        with _capture_native_output() as native_output:
            compiled = compile_path_c_region(
                region,
                schedule_template=schedule_template,
                schedule_name=schedule_name,
                schedule_status=schedule_status,
                tilelang_lowerer=native_lowerer,
                target=target_name,
            )
        elapsed_s = time.perf_counter() - started
        return (
            compiled,
            native_output.stdout,
            native_output.stderr,
            None,
            elapsed_s,
        )
    except Exception as exc:
        elapsed_s = time.perf_counter() - started
        return (
            compile_path_c_region(
                region,
                schedule_template=schedule_template,
                schedule_name=schedule_name,
                schedule_status=schedule_status,
            ),
            native_output.stdout,
            native_output.stderr,
            f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
            elapsed_s,
        )


def _cache_key_recompile_audit_payload(
    *,
    native_compile: bool,
    primary_compiled: FusionCompilePlan | CompiledPathCRegion,
    primary_compile_error: str | None,
    region: Any,
    schedule_template: Callable[[Any], Any],
    schedule_name: str,
    schedule_status: str,
    native_lowerer: Callable[..., Any] | None,
    target_name: str,
) -> dict[str, Any]:
    """Repeat compile once to prove the production receipt key is stable."""

    primary_cache = _fusion_cache_key_payload(primary_compiled)
    if not native_compile:
        return {
            "status": "skipped_no_native_compile",
            "reason": "native compile was not requested for this receipt",
            "cache_hit_observed": None,
            "cache_hit_status": "not_observed",
            "primary": primary_cache,
            "second": None,
        }
    if primary_compile_error is not None:
        return {
            "status": "skipped_primary_compile_failed",
            "reason": "primary native compile failed, so recompile audit would not compare equivalent lowered modules",
            "cache_hit_observed": None,
            "cache_hit_status": "not_observed",
            "primary": primary_cache,
            "second": None,
        }

    second_compiled, second_stdout, second_stderr, second_error, second_elapsed_s = (
        _compile_path_c_region_with_native_capture(
            region=region,
            schedule_template=schedule_template,
            schedule_name=schedule_name,
            schedule_status=schedule_status,
            native_lowerer=native_lowerer,
            target_name=target_name,
        )
    )
    second_cache = _fusion_cache_key_payload(second_compiled)
    second_payload = {
        **second_cache,
        "elapsed_s": second_elapsed_s,
        "native_compile_ok": second_error is None
        and isinstance(second_compiled, CompiledPathCRegion)
        and second_compiled.artifact is not None,
        "native_compile_error": second_error,
        "stdout_tail": second_stdout[-2000:],
        "stderr_tail": second_stderr[-2000:],
    }
    if second_error is not None:
        return {
            "status": "second_compile_failed",
            "reason": "second native compile failed while auditing equivalent production receipt",
            "cache_hit_observed": None,
            "cache_hit_status": "not_observed",
            "primary": primary_cache,
            "second": second_payload,
        }

    primary_material = primary_cache.get("cache_key_material_sha256")
    second_material = second_cache.get("cache_key_material_sha256")
    primary_lowered = primary_cache.get("lowered_module_digest")
    second_lowered = second_cache.get("lowered_module_digest")
    key_stable = bool(primary_material and primary_material == second_material)
    lowered_stable = bool(primary_lowered and primary_lowered == second_lowered)
    if not key_stable or not lowered_stable:
        return {
            "status": "key_changed",
            "reason": "equivalent native compile attempts produced different cache material or lowered module digests",
            "cache_hit_observed": None,
            "cache_hit_status": "not_observed_by_tilelang_api",
            "primary": primary_cache,
            "second": second_payload,
        }
    return {
        "status": "key_stable",
        "reason": "equivalent native compile attempts produced identical cache material and lowered module digests",
        "cache_hit_observed": None,
        "cache_hit_status": "not_observed_by_tilelang_api",
        "primary": primary_cache,
        "second": second_payload,
    }


def _lane0_fragment_markers(generated_source: str) -> tuple[list[str], list[str]]:
    lines = generated_source.splitlines()
    markers: list[str] = []
    production: list[str] = []
    production_ops = {
        "mamba3_mimo",
        "m2rnn",
        "attention_qkv_projection",
        "sparse_mla_fp8_apply",
        "attention_qkv_projection_bwd",
        "m2rnn_bwd",
        "mamba3_mimo_bwd",
    }
    for index, line in enumerate(lines):
        if line.strip() != "if lane == 0:":
            continue
        marker = "lane0_scalar_reduction"
        for lookahead in lines[index + 1 : index + 16]:
            stripped = lookahead.strip()
            if not stripped:
                continue
            match = re.match(r"\s*#\s+[^:]+:\s+([A-Za-z0-9_]+)\b", lookahead)
            if match:
                marker = match.group(1)
            break
        if marker == "lane0_scalar_reduction":
            context = " ".join(item.strip() for item in lines[index + 1 : index + 8])
            if "row_sum_sq" in context and "row_dot" in context:
                marker = "residual_rmsnorm_bwd_row_reduction"
            elif "row_sum_sq" in context:
                marker = "residual_rmsnorm_row_reduction"
        markers.append(marker)
        if marker in production_ops and marker not in production:
            production.append(marker)
    return markers, production


def _runtime_execution_contract(
    *,
    generated_source: str,
    schedule_spec: Any,
    plan: FusionCompilePlan,
    kernel_parameter_count: int,
    target_name: str,
    physical_abi_runtime_bridge: Mapping[str, Any] | None = None,
    physical_abi_runtime_binding: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Describe whether the compiled region is safe to route at runtime."""

    loop_policy = getattr(schedule_spec, "loop_policy", None)
    single_thread_kernel = "with T.Kernel(1, threads=1):" in generated_source
    lane0_markers, lane0_production_fragments = _lane0_fragment_markers(generated_source)
    lane0_serial_fragments = bool(lane0_markers)
    lane_strided_row_loops = "step=" in generated_source and "lane" in generated_source
    contract = plan.schedule_contract
    blockers: list[str] = []
    bottlenecks: list[str] = []
    next_codegen_steps: list[str] = []
    metal_buffer_limit_exceeded = (
        target_name == "metal"
        and kernel_parameter_count > METAL_KERNEL_BUFFER_LIMIT
    )

    if metal_buffer_limit_exceeded:
        blockers.append(
            "Metal runtime execution is blocked because the fused schedule ABI uses "
            "more kernel buffer arguments than Metal exposes: "
            f"{kernel_parameter_count} > {METAL_KERNEL_BUFFER_LIMIT}"
        )
        bottlenecks.append(
            "TileLang JIT wrapper creation is not enough proof: first Metal "
            "source compile/execution fails when generated buffer indices exceed 30"
        )
        next_codegen_steps.append(
            "reduce fused-region ABI below the Metal buffer slot limit with "
            "generic model parameter banks instead of per-weight kernel arguments"
        )
    if not plan.single_kernel_fused:
        blockers.append(
            "compile plan is not verified as the runtime single-kernel fused path "
            f"(single_kernel_fused={plan.single_kernel_fused}, "
            f"schedule_status={plan.schedule_status})"
        )
    if contract is None or contract.status != "verified":
        contract_status = contract.status if contract is not None else None
        contract_reason = contract.reason if contract is not None else None
        blockers.append(
            "schedule contract is not verified by this build "
            f"(status={contract_status}, reason={contract_reason})"
        )
    bridge_status = (
        physical_abi_runtime_bridge.get("status")
        if isinstance(physical_abi_runtime_bridge, Mapping)
        else None
    )
    binding_status = (
        physical_abi_runtime_binding.get("status")
        if isinstance(physical_abi_runtime_binding, Mapping)
        else None
    )
    missing_bank_buffers = (
        list(physical_abi_runtime_binding.get("missing_bank_buffers", ()))
        if isinstance(physical_abi_runtime_binding, Mapping)
        else []
    )
    if binding_status != "ok":
        blockers.append(
            "physical ABI runtime binding is not ready "
            f"(bridge_status={bridge_status}, binding_status={binding_status}, "
            f"missing_bank_buffers={missing_bank_buffers})"
        )
        next_codegen_steps.append(
            "bind the training runtime to caller-owned physical ABI bank buffers "
            "without hidden tensor packing or copies"
        )
    if single_thread_kernel:
        blockers.append(
            "generated row-phased schedule lowers as one TileLang block with one thread"
        )
        bottlenecks.extend(
            [
                "mamba3_mimo and m2rnn recurrence rows are serialized",
                "residual/RMSNorm row reductions are scalarized",
                "attention_qkv_projection and sparse_mla_fp8_apply inner loops remain serial",
            ]
        )
        next_codegen_steps.extend(
            [
                "replace row-local hidden loops with lane-strided loops using T.get_thread_binding()",
                "insert explicit T.sync_threads() barriers around row-local reductions and recurrent state updates",
                "reuse production brick fragments from the existing Path C Mamba3 and sparse-MLA kernels instead of scalar descriptor fallbacks",
            ]
        )
    if lane0_production_fragments:
        blockers.append(
            "generated row-phased schedule still serializes production fragments "
            f"behind lane == 0: {', '.join(lane0_production_fragments)}"
        )
        bottlenecks.extend(
            [
                "lane-0 production fragments remain: "
                + ", ".join(lane0_production_fragments),
                "threaded row-local shared buffers exist, but the heavy fragments have not been rewritten to consume them from all lanes",
            ]
        )
        next_codegen_steps.extend(
            [
                "replace each lane-0 guarded production fragment with a lane-strided implementation",
                "keep shared row-local intermediates and barriers as the handoff ABI between threaded fragments",
            ]
        )
    elif lane0_serial_fragments:
        bottlenecks.append(
            "no lane-0 production fragments were detected, but scalar row reductions still use lane 0"
        )

    status = "runtime_ready" if not blockers else "compile_only_not_runtime_ready"
    runtime_route_uses_fused_region = status == "runtime_ready"
    return {
        "status": status,
        "single_entry_tilelang_ir": bool(generated_source.strip()),
        "runtime_route_uses_fused_region": runtime_route_uses_fused_region,
        "kernel_parameter_count": kernel_parameter_count,
        "metal_buffer_limit": METAL_KERNEL_BUFFER_LIMIT,
        "metal_buffer_limit_exceeded": metal_buffer_limit_exceeded,
        "loop_policy": loop_policy,
        "plan_single_kernel_fused": plan.single_kernel_fused,
        "plan_schedule_status": plan.schedule_status,
        "physical_abi_runtime_bridge_status": bridge_status,
        "physical_abi_runtime_binding_status": binding_status,
        "physical_abi_missing_bank_buffers": missing_bank_buffers,
        "schedule_contract_status": contract.status if contract is not None else None,
        "schedule_contract_reason": contract.reason if contract is not None else None,
        "schedule_contract_declared_kind": (
            contract.declared_implementation_kind if contract is not None else None
        ),
        "schedule_contract_declared_schedule_id": (
            contract.declared_schedule_id if contract is not None else None
        ),
        "single_thread_kernel": single_thread_kernel,
        "lane0_serial_fragments": lane0_serial_fragments,
        "lane0_fragment_markers": lane0_markers,
        "lane0_production_fragments": lane0_production_fragments,
        "lane_strided_row_loops": lane_strided_row_loops,
        "blockers": blockers,
        "expected_bottlenecks": bottlenecks,
        "next_codegen_steps": next_codegen_steps,
    }


def _production_runtime_smoke_binding(
    *,
    runtime_smoke_payload: Mapping[str, Any],
    region_name: str,
) -> Mapping[str, Any] | None:
    """Return production smoke binding evidence for the selected fused region."""

    if runtime_smoke_payload.get("status") != "ok":
        return None
    if runtime_smoke_payload.get("full_execution") is False:
        return None
    if runtime_smoke_payload.get("mode") != "production_1b":
        return None
    if runtime_smoke_payload.get("region_name") != region_name:
        return None
    fused_route = runtime_smoke_payload.get("fused_train_block_route")
    if not isinstance(fused_route, Mapping) or fused_route.get("status") != "ok":
        return None
    route = fused_route.get("route")
    if not isinstance(route, Mapping):
        return None
    if route.get("selected_action") != "run_path_c_fused_train_block_route":
        return None
    if route.get("single_fused_train_block_runtime_available") is not True:
        return None
    binding = runtime_smoke_payload.get("physical_abi_runtime_binding")
    if not isinstance(binding, Mapping):
        return None
    if binding.get("status") != "ok":
        return None
    return binding


def _direct_logical_abi_alternative_payload(
    *,
    region: Any,
    target: Any,
    target_name: str,
    shape_env: Any,
    model: Any | None = None,
    selected_forward_region: Any | None = None,
) -> dict[str, Any]:
    """Describe the no-pack direct logical-buffer ABI alternative."""

    chained_plan = None
    chained_construction: dict[str, Any] = {
        "status": "not_planned",
        "reason": "direct alternative planning did not start",
    }
    try:
        if not getattr(target, "brick_descriptors", ()):
            return {
                "status": "unsupported_target",
                "physical_abi_policy": DESCRIPTOR_PHYSICAL_ABI_POLICY_DIRECT,
                "reason": "selected schedule target does not expose brick descriptors",
            }
        chained_plan, chained_construction = _direct_chain_for_selected_region(
            model=model,
            selected_region=selected_forward_region or region,
            already_has_backward=selected_forward_region is None,
        )
        direct_template = make_path_c_descriptor_schedule_template(
            target.brick_descriptors,
            entry_symbol=getattr(region, "entry_symbol", None)
            or getattr(region, "name", None),
            buffer_extent=target.buffer_extent,
            shape_env=shape_env,
            internal_buffer_policy=target.internal_buffer_policy,
            loop_policy=target.loop_policy,
            physical_abi_policy=DESCRIPTOR_PHYSICAL_ABI_POLICY_DIRECT,
        )
        direct_prim_func = direct_template(region)
        physical_abi_map = dict(
            getattr(direct_prim_func, "_cppmega_path_c_physical_buffer_abi_map", {})
            or {}
        )
        physical_abi_shapes = dict(
            getattr(
                direct_prim_func,
                "_cppmega_path_c_physical_buffer_abi_shapes",
                {},
            )
            or {}
        )
        bridge = plan_physical_abi_runtime_bridge(
            physical_abi_map,
            physical_abi_shapes,
        )
        kernel_parameter_count = _kernel_parameter_count(direct_prim_func)
        metal_buffer_limit_exceeded = (
            target_name == "metal"
            and kernel_parameter_count > METAL_KERNEL_BUFFER_LIMIT
        )
        status = (
            "blocked_metal_buffer_limit"
            if metal_buffer_limit_exceeded
            else "direct_logical_binding_candidate"
        )
        return {
            "status": status,
            "physical_abi_policy": DESCRIPTOR_PHYSICAL_ABI_POLICY_DIRECT,
            "reason": (
                "direct logical-buffer ABI would avoid hidden bank packing, but "
                "the selected train-block exceeds the Metal kernel buffer limit"
                if metal_buffer_limit_exceeded
                else
                "direct logical-buffer ABI can bind caller-owned tensors without "
                "prepacked dtype banks"
            ),
            "kernel_parameter_count": kernel_parameter_count,
            "logical_parameter_count": len(physical_abi_map),
            "metal_buffer_limit": METAL_KERNEL_BUFFER_LIMIT,
            "metal_buffer_limit_exceeded": metal_buffer_limit_exceeded,
            "logical_tensor_binding_supported": bool(
                bridge.get("logical_tensor_binding_supported")
            ),
            "prepacked_bank_binding_supported": bool(
                bridge.get("prepacked_bank_binding_supported")
            ),
            "required_kernel_buffers": list(bridge.get("required_bank_buffers", ())),
            "no_hidden_allocation_policy": bool(
                bridge.get("no_hidden_allocation_policy", True)
            ),
            "direct_chained_fusion_plan": _direct_chained_fusion_plan_payload(
                chained_plan
            ),
            "direct_chained_fusion_construction": chained_construction,
            "next_codegen_steps": [
                "route runtime through direct-buffer chained fused segments to stay under Metal limits without dtype-bank packing",
                "compile and call each chained segment with caller-owned logical tensor buffers",
            ]
            if (
                metal_buffer_limit_exceeded
                and chained_plan.status == "ready"
            )
            else [
                "keep production ABI under Metal buffer limits without runtime packing",
                "use model/training-owned physical banks or a generic chained fusion planner",
            ]
            if metal_buffer_limit_exceeded
            else [],
        }
    except Exception as exc:  # pragma: no cover - defensive receipt metadata
        error = f"{type(exc).__name__}: {exc}"
        return {
            "status": "unavailable",
            "physical_abi_policy": DESCRIPTOR_PHYSICAL_ABI_POLICY_DIRECT,
            "reason": str(exc),
            "error": error,
            "logical_tensor_binding_supported": True,
            "no_hidden_allocation_policy": True,
            "direct_chained_fusion_plan": _direct_chained_fusion_plan_payload(
                chained_plan
            ),
            "direct_chained_fusion_construction": chained_construction,
        }


def _direct_chained_fusion_plan_payload(chain: Any) -> dict[str, Any]:
    if chain is None:
        return {
            "status": "unavailable",
            "reason": "direct chained fusion plan was not produced",
            "max_kernel_buffers": METAL_KERNEL_BUFFER_LIMIT,
            "segment_count": 0,
            "covers_full_region": False,
            "source_region_name": "",
            "source_node_count": 0,
            "segments": [],
        }
    segments = []
    for segment in getattr(chain, "segments", ()):
        plan = getattr(segment, "plan", None)
        target = getattr(segment, "schedule_target", None)
        contract = getattr(plan, "schedule_contract", None)
        segments.append(
            {
                "index": int(segment.index),
                "node_start": int(segment.node_start),
                "node_end": int(segment.node_end),
                "region_name": getattr(segment.region, "name", ""),
                "node_names": list(getattr(segment.region, "node_names", ())),
                "op_signature": [
                    str(node.op_name)
                    for node in getattr(segment.region, "nodes", ())
                ],
                "status": str(segment.status),
                "reason": str(segment.reason),
                "physical_abi_policy": str(segment.physical_abi_policy),
                "kernel_parameter_count": segment.kernel_parameter_count,
                "schedule_id": getattr(target, "schedule_id", None),
                "schedule_name": getattr(target, "schedule_name", None),
                "schedule_contract_status": getattr(contract, "status", None),
            }
        )
    source_nodes = list(getattr(chain.source_region, "node_names", ()))
    covers_full_region = bool(segments) and (
        segments[0]["node_start"] == 0
        and segments[-1]["node_end"] == len(source_nodes)
        and all(
            left["node_end"] == right["node_start"]
            for left, right in zip(segments[:-1], segments[1:], strict=True)
        )
    )
    return {
        "status": str(chain.status),
        "reason": str(chain.reason),
        "max_kernel_buffers": int(chain.max_kernel_buffers),
        "segment_count": len(segments),
        "covers_full_region": covers_full_region,
        "source_region_name": getattr(chain.source_region, "name", ""),
        "source_node_count": len(source_nodes),
        "segments": segments,
    }


def _select_path_c_direct_chain_for_region(
    chains: tuple[Any, ...],
    selected_region: Any,
) -> Any | None:
    if not chains:
        return None
    selected_name = str(getattr(selected_region, "name", ""))
    for chain in chains:
        source_region = getattr(chain, "source_region", None)
        if str(getattr(source_region, "name", "")) == selected_name:
            return chain
    return max(
        chains,
        key=lambda chain: (
            len(getattr(getattr(chain, "source_region", None), "nodes", ())),
            len(getattr(getattr(chain, "source_region", None), "edges", ())),
            str(getattr(getattr(chain, "source_region", None), "name", "")),
        ),
    )


def _direct_chain_for_selected_region(
    *,
    model: Any | None,
    selected_region: Any,
    already_has_backward: bool,
) -> tuple[Any, dict[str, Any]]:
    if model is not None:
        region_prefix = str(
            getattr(model, "path_c_region_prefix", None)
            or f"{getattr(model, 'name', 'model')}_path_c"
        )
        chains = plan_path_c_direct_fusion_chains_for_model(
            model,
            region_prefix=region_prefix,
            include_backward=True,
            max_kernel_buffers=METAL_KERNEL_BUFFER_LIMIT,
        )
        chain = _select_path_c_direct_chain_for_region(chains, selected_region)
        if chain is None:
            raise RuntimeError("selected model did not produce any direct Path C chain")
        return chain, {
            "planner": "plan_path_c_direct_fusion_chains_for_model",
            "region_prefix": region_prefix,
            "candidate_chain_count": len(chains),
            "selected_source_region": getattr(
                getattr(chain, "source_region", None),
                "name",
                None,
            ),
        }
    chain = plan_path_c_direct_fusion_chain_for_region(
        selected_region,
        include_backward=not already_has_backward,
        max_kernel_buffers=METAL_KERNEL_BUFFER_LIMIT,
    )
    return chain, {
        "planner": "plan_path_c_direct_fusion_chain_for_region",
        "region_prefix": None,
        "candidate_chain_count": 1,
        "selected_source_region": getattr(
            getattr(chain, "source_region", None),
            "name",
            None,
        ),
    }


def _native_compile_direct_chain_payload(
    *,
    model: Any | None,
    selected_region: Any,
    already_has_backward: bool,
    native_compile: bool,
    native_lowerer: Callable[..., Any] | None,
    target_name: str,
) -> dict[str, Any]:
    if not native_compile:
        return {
            "status": "not_requested",
            "native_compile_requested": False,
        }
    if native_lowerer is None:
        return {
            "status": "unavailable",
            "native_compile_requested": True,
            "reason": "no native lowerer was configured",
        }

    chain, construction = _direct_chain_for_selected_region(
        model=model,
        selected_region=selected_region,
        already_has_backward=already_has_backward,
    )
    segment_payloads: list[dict[str, Any]] = []
    for segment in chain.segments:
        target = segment.schedule_target
        if target is None:
            segment_payloads.append(
                {
                    "index": segment.index,
                    "status": "blocked",
                    "native_compile_ok": False,
                    "reason": segment.reason,
                }
            )
            continue
        schedule_template = mark_path_c_schedule_template_for_region(
            target.schedule_template,
            segment.region,
            implementation_kind=target.implementation_kind,
            production_schedule_id=target.schedule_id
            if target.implementation_kind == "production"
            else "",
            required_real_abi_inputs=target.required_real_abi_inputs,
        )
        compiled, stdout, stderr, error, elapsed_s = (
            _compile_path_c_region_with_native_capture(
                region=segment.region,
                schedule_template=schedule_template,
                schedule_name=target.schedule_name,
                schedule_status=target.schedule_status,
                native_lowerer=native_lowerer,
                target_name=target_name,
            )
        )
        artifact = compiled.artifact if isinstance(compiled, CompiledPathCRegion) else None
        plan = compiled.plan if isinstance(compiled, CompiledPathCRegion) else compiled
        contract = getattr(plan, "schedule_contract", None)
        native_compile_ok = bool(error is None and artifact is not None)
        segment_payloads.append(
            {
                "index": segment.index,
                "node_start": segment.node_start,
                "node_end": segment.node_end,
                "region_name": segment.region.name,
                "op_signature": [node.op_name for node in segment.region.nodes],
                "kernel_parameter_count": segment.kernel_parameter_count,
                "schedule_id": target.schedule_id,
                "schedule_contract_status": getattr(contract, "status", None),
                "native_compile_ok": native_compile_ok,
                "artifact_type": type(artifact).__name__ if artifact is not None else None,
                "elapsed_s": elapsed_s,
                "error": error,
                "stdout_tail": stdout[-2000:],
                "stderr_tail": stderr[-2000:],
            }
        )
    all_ok = bool(segment_payloads) and all(
        bool(segment.get("native_compile_ok"))
        for segment in segment_payloads
    )
    return {
        "status": "ok" if chain.status == "ready" and all_ok else "failed",
        "native_compile_requested": True,
        "target": target_name,
        "chain_status": chain.status,
        "construction": construction,
        "segment_count": len(segment_payloads),
        "covers_full_region": _direct_chained_fusion_plan_payload(chain)[
            "covers_full_region"
        ],
        "segments": segment_payloads,
    }


def _local_gb10_model_descriptor() -> tuple[Any, Any]:
    profile = local_gb10_quarter_profile()
    model = SimpleNamespace(
        name=profile.name,
        path_c_profile_name=profile.name,
        path_c_input_model_name=f"{profile.name}.path_c_bricks",
        path_c_region_prefix=f"{profile.name}_path_c",
        path_c_bricks=profile.path_c_bricks,
        config=profile.hybrid_config(),
    )
    return profile, model


def _selected_region_and_target() -> tuple[Any, Any, Any, Any, Any]:
    profile, model = _local_gb10_model_descriptor()
    regions = build_path_c_model_regions_from_model(
        model,
        region_prefix=model.path_c_region_prefix,
        include_backward=False,
    )
    if not regions:
        raise RuntimeError("local_gb10_quarter has no Path C fusion regions")
    fwd_region = max(
        regions,
        key=lambda region: (len(region.nodes), len(region.edges), region.name),
    )
    region = build_path_c_aot_autograd_region(fwd_region)
    target = select_path_c_fusion_schedule_target(region)
    if target is None:
        raise RuntimeError(
            "selected Path C fusion region did not resolve to a schedule target"
        )
    return profile, model, fwd_region, region, target


def _kernel_parameter_count(prim_func: Any) -> int:
    params = getattr(prim_func, "params", ())
    return len(tuple(params))


_DTYPE_NBYTES = {
    "bool": 1,
    "uint8": 1,
    "int8": 1,
    "float16": 2,
    "bfloat16": 2,
    "uint16": 2,
    "int16": 2,
    "float32": 4,
    "uint32": 4,
    "int32": 4,
    "float64": 8,
    "uint64": 8,
    "int64": 8,
}


def _shape_tuple(shape: Any) -> tuple[int, ...]:
    return tuple(int(extent) for extent in tuple(shape))


def _buffer_abi_payload(prim_func: Any) -> tuple[list[dict[str, Any]], int]:
    entries: list[dict[str, Any]] = []
    total_bytes = 0
    for param in tuple(getattr(prim_func, "params", ())):
        buffer_map = getattr(prim_func, "buffer_map", {})
        try:
            buffer = buffer_map[param]
        except KeyError:
            continue
        dtype = str(getattr(buffer, "dtype", "unknown"))
        shape = _shape_tuple(getattr(buffer, "shape", ()))
        elements = 1
        for extent in shape:
            elements *= extent
        byte_count = elements * _DTYPE_NBYTES.get(dtype, 0)
        total_bytes += byte_count
        entries.append(
            {
                "name": str(getattr(buffer, "name", param)),
                "dtype": dtype,
                "shape": list(shape),
                "elements": elements,
                "bytes": byte_count,
            }
        )
    return entries, total_bytes


def _runtime_smoke_full_kernel_args(
    prim_func: Any,
    kernel_buffers: Mapping[str, Any],
) -> tuple[tuple[Any, ...], list[dict[str, Any]]]:
    """Return full kernel args in PrimFunc order, including scalar ABI params."""

    args: list[Any] = []
    scalar_args: list[dict[str, Any]] = []
    buffer_map = getattr(prim_func, "buffer_map", {})
    missing: list[str] = []
    for param in tuple(getattr(prim_func, "params", ())):
        buffer = buffer_map.get(param)
        if buffer is None:
            name = str(param)
            value = 1 if name == "path_c_run_backward" else 0
            args.append(value)
            scalar_args.append({"name": name, "value": value})
            continue
        name = str(getattr(buffer, "name", param))
        if name not in kernel_buffers:
            missing.append(name)
            continue
        args.append(kernel_buffers[name])
    if missing:
        raise ValueError(
            "physical ABI runtime bindings are not executable: "
            + "; ".join(
                f"{name}: missing caller-owned kernel buffer" for name in missing
            )
        )
    return tuple(args), scalar_args


def _tiny_runtime_smoke_region_target_model() -> tuple[Any, Any, Any]:
    profile = local_gb10_quarter_profile(
        name="runtime_smoke",
        pattern="MRA",
        depth=3,
        dsa_a_layer_ranks=(0,),
    )
    model = profile.build_tiny_smoke_model(
        pattern="MRA",
        depth=3,
        dsa_a_layer_ranks=(0,),
        max_seq_length=32,
        hidden_size=32,
        num_attention_heads=4,
        mamba_head_dim=8,
        mamba_state_dim=4,
        mamba_groups=1,
        m2rnn_k_head_dim=8,
        m2rnn_v_head_dim=8,
    )
    regions = build_path_c_model_regions_from_model(
        model,
        region_prefix=model.path_c_region_prefix,
        include_backward=False,
    )
    if not regions:
        raise RuntimeError("tiny MRA smoke model did not produce a Path C region")
    region = build_path_c_aot_autograd_region(regions[0])
    target = select_path_c_fusion_schedule_target(region)
    if target is None:
        raise RuntimeError("tiny MRA smoke region did not resolve to a schedule target")
    return region, target, model


def _tiny_runtime_smoke_region_and_target() -> tuple[Any, Any]:
    region, target, _model = _tiny_runtime_smoke_region_target_model()
    return region, target


def _runtime_smoke_region_target_model(mode: str) -> tuple[Any, Any, Any]:
    if mode == "production":
        _profile, _model, _fwd_region, region, target = (
            _selected_region_and_target()
        )
        return region, target, _model
    return _tiny_runtime_smoke_region_target_model()


def _runtime_smoke_region_and_target(mode: str) -> tuple[Any, Any]:
    region, target, _model = _runtime_smoke_region_target_model(mode)
    return region, target


def _mlx_dtype(dtype: str) -> Any:
    import mlx.core as mx  # noqa: PLC0415

    mapping = {
        "bool": mx.bool_,
        "uint8": mx.uint8,
        "int8": mx.int8,
        "float16": mx.float16,
        "bfloat16": mx.bfloat16,
        "uint16": mx.uint16,
        "int16": mx.int16,
        "float32": mx.float32,
        "uint32": mx.uint32,
        "int32": mx.int32,
        "float64": mx.float64,
        "uint64": mx.uint64,
        "int64": mx.int64,
    }
    return mapping[dtype]


def _artifact_kernel_source_stats(artifact: Any) -> dict[str, Any]:
    kernel_source = str(getattr(artifact, "kernel_source", "") or "")
    stats: dict[str, Any] = {
        "kernel_source_bytes": len(kernel_source.encode("utf-8")),
        "kernel_source_lines": kernel_source.count("\n") + 1
        if kernel_source
        else 0,
        "threadgroup_barrier_count": kernel_source.count("threadgroup_barrier"),
        "for_loop_count": kernel_source.count("for ("),
        "if_count": kernel_source.count("if ("),
        "threadgroup_dynamic_shared_bytes": None,
    }
    match = re.search(
        r"threadgroup\s+(?P<dtype>uchar|float|int)\s+buf_dyn_shmem\[(?P<count>\d+)\]",
        kernel_source,
    )
    if match is not None:
        scale = {"uchar": 1, "float": 4, "int": 4}[match.group("dtype")]
        stats["threadgroup_dynamic_shared_bytes"] = int(match.group("count")) * scale
    return stats


def _stage_kernel_source_payload(
    *,
    monolithic_artifact: Any | None,
    forward_artifacts: Sequence[Any],
    backward_artifacts: Sequence[Any],
    artifact_kind: str = "staged",
) -> dict[str, Any]:
    monolithic = _artifact_kernel_source_stats(monolithic_artifact)
    stages = {
        **{
            f"forward_{index}": _artifact_kernel_source_stats(artifact)
            for index, artifact in enumerate(forward_artifacts)
        },
        **{
            f"backward_{index}": _artifact_kernel_source_stats(artifact)
            for index, artifact in enumerate(backward_artifacts)
        },
    }
    max_stage = max(
        stages.values(),
        key=lambda item: int(item.get("kernel_source_bytes", 0) or 0),
    )
    return {
        **max_stage,
        "generated_stage_artifact": True,
        "generated_phase_artifact": artifact_kind == "phased",
        "stage_count": len(stages),
        "max_stage_kernel_source_bytes": max(
            int(item.get("kernel_source_bytes", 0) or 0)
            for item in stages.values()
        ),
        "total_stage_kernel_source_bytes": sum(
            int(item.get("kernel_source_bytes", 0) or 0)
            for item in stages.values()
        ),
        "monolithic_contract": monolithic,
        "monolithic_native_compile_skipped": monolithic_artifact is None,
        "stages": stages,
    }


def _generated_stage_schedule_template(
    *,
    target: Any,
    region: Any,
    abi_prim_func: Any,
    execution_stage: str,
    active_node_names: Sequence[str] | None = None,
    stage_suffix: str = "",
    row_dispatch_mode: str = DESCRIPTOR_ROW_DISPATCH_LAUNCHER_CHUNKS,
    rows_per_kernel_launch: int | None = None,
) -> Callable[[Any], Any]:
    return make_path_c_descriptor_stage_schedule_template(
        schedule_target=target,
        region=region,
        abi_prim_func=abi_prim_func,
        execution_stage=execution_stage,
        active_node_names=active_node_names,
        stage_suffix=stage_suffix,
        row_dispatch_mode=row_dispatch_mode,
        rows_per_kernel_launch=(
            DESCRIPTOR_ROWS_PER_KERNEL_LAUNCH
            if rows_per_kernel_launch is None
            else int(rows_per_kernel_launch)
        ),
    )


def _runtime_smoke_launcher_chunked_target(target: Any, region: Any) -> Any:
    shape_env = getattr(
        getattr(target, "schedule_template", None),
        "_cppmega_path_c_shape_env",
        None,
    )
    max_rows_per_launch = (
        target.max_rows_per_launch
        if getattr(target, "max_rows_per_launch", None) is not None
        else DESCRIPTOR_DEFAULT_MAX_ROWS_PER_LAUNCH
    )
    schedule_template = make_path_c_descriptor_schedule_template(
        target.brick_descriptors,
        entry_symbol=getattr(region, "entry_symbol", None)
        or getattr(region, "name", None),
        buffer_extent=target.buffer_extent,
        shape_env=shape_env,
        internal_buffer_policy=target.internal_buffer_policy,
        loop_policy=target.loop_policy,
        physical_abi_policy=target.physical_abi_policy,
        train_step_output_abi=bool(
            getattr(
                target.schedule_template,
                "_cppmega_path_c_train_step_output_abi_enabled",
                False,
            )
        ),
        max_rows_per_launch=max_rows_per_launch,
        row_dispatch_mode=DESCRIPTOR_ROW_DISPATCH_LAUNCHER_CHUNKS,
    )
    return replace(
        target,
        schedule_template=schedule_template,
        max_rows_per_launch=max_rows_per_launch,
        row_dispatch_mode=DESCRIPTOR_ROW_DISPATCH_LAUNCHER_CHUNKS,
    )


def _kernel_buffer_shapes_for_prim_func(prim_func: Any) -> dict[str, tuple[int, ...]]:
    shapes: dict[str, tuple[int, ...]] = {}
    buffer_map = getattr(prim_func, "buffer_map", {}) or {}
    for param in tuple(getattr(prim_func, "params", ())):
        buffer = buffer_map.get(param)
        if buffer is None:
            continue
        shapes[str(getattr(buffer, "name", param))] = tuple(
            int(dim) for dim in tuple(getattr(buffer, "shape", ()))
        )
    return shapes


def _path_c_int_prim_attr(prim_func: Any, attr_name: str) -> int | None:
    value = getattr(prim_func, attr_name, None)
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def _path_c_str_prim_attr(prim_func: Any, attr_name: str) -> str | None:
    value = getattr(prim_func, attr_name, None)
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _path_c_stage_row_launch_specs(
    prim_funcs: Sequence[Any],
) -> tuple[dict[str, Any], ...]:
    specs: list[dict[str, Any]] = []
    for index, prim_func in enumerate(prim_funcs):
        specs.append(
            {
                "stage_index": index,
                "row_chunk_count": _path_c_int_prim_attr(
                    prim_func,
                    "_cppmega_path_c_row_chunk_count",
                ),
                "row_chunk_index_param": _path_c_str_prim_attr(
                    prim_func,
                    "_cppmega_path_c_row_chunk_index_param",
                ),
                "row_subchunk_count": _path_c_int_prim_attr(
                    prim_func,
                    "_cppmega_path_c_row_subchunk_count",
                ),
                "row_subchunk_index_param": _path_c_str_prim_attr(
                    prim_func,
                    "_cppmega_path_c_row_subchunk_index_param",
                ),
                "rows_per_kernel_launch": _path_c_int_prim_attr(
                    prim_func,
                    "_cppmega_path_c_rows_per_kernel_launch",
                ),
            }
        )
    return tuple(specs)


def _merged_physical_abi_for_prim_funcs(
    prim_funcs: Sequence[Any],
) -> tuple[dict[str, Any], dict[str, tuple[int, ...]]]:
    return merge_path_c_physical_abi_for_prim_funcs(prim_funcs)


def _runtime_smoke_route_args(mode: str) -> SimpleNamespace:
    if mode == "production":
        profile = local_gb10_quarter_profile()
        return SimpleNamespace(
            dtype="fp8_path_c",
            model_profile="local_gb10_quarter",
            seq_len=1,
            pattern=profile.pattern,
            depth=profile.depth,
            dsa_a_layer_ranks=profile.dsa_a_layer_ranks,
        )
    return SimpleNamespace(
        dtype="fp8_path_c",
        model_profile="runtime_smoke",
        seq_len=1,
        pattern="MRA",
        depth=3,
        dsa_a_layer_ranks=(0,),
    )


def _runtime_smoke_fused_train_block_route_payload(
    *,
    mode: str,
    model: Any,
    artifact: Any,
    physical_buffer_abi_map: Mapping[str, Any],
    physical_buffer_abi_shapes: Mapping[str, Any],
    physical_bank_buffers: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind the smoke artifact through the m04 fused train-block route."""

    try:
        from scripts import m04_train_step  # noqa: PLC0415

        profile_name = str(
            getattr(model, "path_c_profile_name", None)
            or getattr(model, "name", "runtime_smoke")
        )
        bank_owner = make_physical_abi_bank_owner(
            f"{profile_name}.path_c_physical_abi_banks",
            physical_buffer_abi_map,
            physical_buffer_abi_shapes,
            physical_bank_buffers,
        )
        install = m04_train_step.install_path_c_fused_train_block_runtime_for_model(
            model=model,
            bank_owner=bank_owner,
            fused_artifact=artifact,
            sequence_length=None,
        )
        route = m04_train_step.fp8_path_c_training_route_payload_for_model(
            _runtime_smoke_route_args(mode),
            model,
        )
        runtime_binding = dict(
            route.get("path_c_fusion", {}).get("runtime_training_binding", {})
        )
        selected_action = route.get("selected_action")
        standalone_bound = bool(
            install.get("runtime_uses_fused_train_block") is True
            and runtime_binding.get("runtime_uses_fused_train_block") is True
        )
        training_route_bound = bool(
            standalone_bound
            and selected_action == "run_path_c_fused_train_block_route"
            and route.get("single_fused_train_block_runtime_available") is True
        )
        status = (
            "ok"
            if training_route_bound
            else "standalone_only_not_training_route"
            if standalone_bound
            else "blocked"
        )
        return {
            "status": status,
            "install": install,
            "route": {
                "status": route.get("status"),
                "selected_action": selected_action,
                "full_end_to_end_training_available": route.get(
                    "full_end_to_end_training_available"
                ),
                "single_fused_train_block_standalone_dispatch_available": (
                    route.get(
                        "single_fused_train_block_standalone_dispatch_available"
                    )
                ),
                "single_fused_train_block_runtime_available": route.get(
                    "single_fused_train_block_runtime_available"
                ),
                "fused_train_block_training_runtime_available": route.get(
                    "fused_train_block_training_runtime_available"
                ),
                "fused_train_block_training_runtime_contract": route.get(
                    "fused_train_block_training_runtime_contract"
                ),
                "path_c_fusion": {
                    "runtime_training_binding": runtime_binding,
                },
            },
            "hidden_packing_performed": bool(
                install.get("hidden_packing_performed", False)
                or runtime_binding.get("hidden_packing_performed", False)
            ),
        }
    except Exception as exc:
        return {
            "status": "unavailable",
            "error": f"{type(exc).__name__}: {exc}",
            "hidden_packing_performed": False,
        }


def _runtime_smoke_payload(
    *,
    mode: str,
    target_name: str,
    execution_backend: str,
    max_bytes: int,
    max_launches: int | None = None,
    max_stages: int | None = None,
    artifact_kind: str = "single",
    lowerer: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    if mode == "none":
        return {
            "status": "not_requested",
            "mode": "none",
            "actually_executed": False,
        }
    if mode not in {"tiny", "production"}:
        return {
            "status": "unsupported_mode",
            "mode": mode,
            "actually_executed": False,
            "reason": f"unsupported runtime smoke mode: {mode}",
        }
    if artifact_kind not in {"single", "staged", "phased"}:
        return {
            "status": "unsupported_artifact_kind",
            "mode": mode,
            "actually_executed": False,
            "reason": f"unsupported runtime smoke artifact: {artifact_kind}",
        }

    started = time.perf_counter()
    region: Any | None = None
    target: Any | None = None
    prim_func: Any | None = None
    buffer_abi: list[dict[str, Any]] = []
    total_bytes: int | None = None
    launch_limit = None if max_launches is None else max(0, int(max_launches))
    stage_limit = None if max_stages is None else max(0, int(max_stages))
    forward_stage_artifacts: tuple[Any, ...] = ()
    backward_stage_artifacts: tuple[Any, ...] = ()
    forward_stage_prim_funcs: tuple[Any, ...] = ()
    backward_stage_prim_funcs: tuple[Any, ...] = ()
    smoke_mode = "production_1b" if mode == "production" else "tiny_mra"
    try:
        generated_stage_artifact = artifact_kind in {"staged", "phased"}
        region, target, model = _runtime_smoke_region_target_model(mode)
        target = _runtime_smoke_launcher_chunked_target(target, region)
        prim_func = target.schedule_template(region)
        buffer_abi, total_bytes = _buffer_abi_payload(prim_func)
        if total_bytes > max_bytes:
            return {
                "status": "skipped_large_abi",
                "mode": smoke_mode,
                "actually_executed": False,
                "reason": (
                    f"runtime smoke ABI uses {total_bytes} bytes, above "
                    f"budget {max_bytes}"
                ),
                "region_name": region.name,
                "schedule_id": target.schedule_id,
                "kernel_parameter_count": _kernel_parameter_count(prim_func),
                "buffer_abi": buffer_abi,
                "total_buffer_bytes": total_bytes,
                "max_buffer_bytes": max_bytes,
            }

        schedule_template = mark_path_c_schedule_template_for_region(
            lambda _region: prim_func,
            region,
            implementation_kind=target.implementation_kind,
            production_schedule_id=target.schedule_id
            if target.implementation_kind == "production"
            else "",
            required_real_abi_inputs=target.required_real_abi_inputs,
        )
        native_lowerer = lowerer
        if native_lowerer is None:

            def native_lowerer(func_or_mod: Any, *, target: str, **kwargs: Any) -> Any:
                return tilelang_single_entry_lowerer(
                    func_or_mod,
                    target=target,
                    execution_backend=execution_backend,
                    **kwargs,
                )

        if artifact_kind == "single":
            compiled, stdout, stderr, compile_error, compile_elapsed_s = (
                _compile_path_c_region_with_native_capture(
                    region=region,
                    schedule_template=schedule_template,
                    schedule_name=target.schedule_name,
                    schedule_status=target.schedule_status,
                    native_lowerer=native_lowerer,
                    target_name=target_name,
                )
            )
        else:
            plan_started = time.perf_counter()
            compiled = compile_path_c_region(
                region=region,
                schedule_template=schedule_template,
                schedule_name=target.schedule_name,
                schedule_status=target.schedule_status,
            )
            stdout = ""
            stderr = ""
            compile_error = None
            compile_elapsed_s = time.perf_counter() - plan_started
        artifact = (
            compiled.artifact if isinstance(compiled, CompiledPathCRegion) else None
        )
        if compile_error is not None or (artifact_kind == "single" and artifact is None):
            return {
                "status": "failed_compile",
                "mode": smoke_mode,
                "actually_executed": False,
                "region_name": region.name,
                "schedule_id": target.schedule_id,
                "compile_elapsed_s": compile_elapsed_s,
                "native_compile_error": compile_error,
                "artifact_type": type(artifact).__name__ if artifact is not None else None,
                "kernel_source": _artifact_kernel_source_stats(artifact)
                if artifact is not None
                else None,
                "stdout_tail": stdout[-2000:],
                "stderr_tail": stderr[-2000:],
            }
        descriptor_stage_groups = (
            plan_path_c_descriptor_phase_groups(region)
            if artifact_kind == "phased"
            else plan_path_c_descriptor_stage_groups(region)
        )
        forward_stage_groups = tuple(
            group
            for group in descriptor_stage_groups
            if group.execution_stage == DESCRIPTOR_EXECUTION_STAGE_FORWARD
        )
        backward_stage_groups = tuple(
            group
            for group in descriptor_stage_groups
            if group.execution_stage == DESCRIPTOR_EXECUTION_STAGE_BACKWARD
        )
        stage_output = SimpleNamespace(stdout="", stderr="")
        stage_compile_elapsed_s = 0.0
        if generated_stage_artifact:
            forward_stage_templates = tuple(
                _generated_stage_schedule_template(
                    target=target,
                    region=region,
                    abi_prim_func=prim_func,
                    execution_stage=group.execution_stage,
                    active_node_names=group.active_node_names,
                    stage_suffix=group.stage_suffix,
                    row_dispatch_mode=group.row_dispatch_mode,
                    rows_per_kernel_launch=group.rows_per_kernel_launch,
                )
                for group in forward_stage_groups
            )
            backward_stage_templates = tuple(
                _generated_stage_schedule_template(
                    target=target,
                    region=region,
                    abi_prim_func=prim_func,
                    execution_stage=group.execution_stage,
                    active_node_names=group.active_node_names,
                    stage_suffix=group.stage_suffix,
                    row_dispatch_mode=group.row_dispatch_mode,
                    rows_per_kernel_launch=group.rows_per_kernel_launch,
                )
                for group in backward_stage_groups
            )
            forward_stage_prim_funcs = tuple(
                template(region) for template in forward_stage_templates
            )
            backward_stage_prim_funcs = tuple(
                template(region) for template in backward_stage_templates
            )
            stage_started = time.perf_counter()
            with _capture_native_output() as captured_stage_output:
                forward_stage_artifacts = tuple(
                    native_lowerer(
                        stage_prim_func,
                        target=target_name,
                    )
                    for stage_prim_func in forward_stage_prim_funcs
                )
                backward_stage_artifacts = tuple(
                    native_lowerer(
                        stage_prim_func,
                        target=target_name,
                    )
                    for stage_prim_func in backward_stage_prim_funcs
                )
            stage_output = captured_stage_output
            stage_compile_elapsed_s = time.perf_counter() - stage_started

        import mlx.core as mx  # noqa: PLC0415

        kernel_buffers = {
            str(entry["name"]): mx.zeros(
                tuple(entry["shape"]),
                dtype=_mlx_dtype(str(entry["dtype"])),
            )
            for entry in buffer_abi
        }
        stage_buffer_abi: list[dict[str, Any]] = []
        if generated_stage_artifact:
            for stage_prim_func in (
                *forward_stage_prim_funcs,
                *backward_stage_prim_funcs,
            ):
                stage_entries, _stage_total_bytes = _buffer_abi_payload(stage_prim_func)
                stage_buffer_abi.extend(stage_entries)
                for entry in stage_entries:
                    name = str(entry["name"])
                    shape = tuple(entry["shape"])
                    existing = kernel_buffers.get(name)
                    if existing is not None and int(existing.size) >= math.prod(shape):
                        continue
                    kernel_buffers[name] = mx.zeros(
                        shape,
                        dtype=_mlx_dtype(str(entry["dtype"])),
                    )
            physical_buffer_abi_map, physical_buffer_abi_shapes = (
                _merged_physical_abi_for_prim_funcs(
                    (
                        prim_func,
                        *forward_stage_prim_funcs,
                        *backward_stage_prim_funcs,
                    )
                )
            )
        else:
            physical_buffer_abi_map = dict(
                getattr(prim_func, "_cppmega_path_c_physical_buffer_abi_map", {})
                or {}
            )
            physical_buffer_abi_shapes = {
                str(name): tuple(int(dim) for dim in tuple(shape))
                for name, shape in dict(
                    getattr(
                        prim_func,
                        "_cppmega_path_c_physical_buffer_abi_shapes",
                        {},
                    )
                    or {}
                ).items()
            }
        physical_bank_buffers = {
            name: kernel_buffers[name]
            for name in physical_buffer_abi_shapes
        }
        physical_abi_runtime_binding = validate_physical_abi_runtime_bindings(
            physical_buffer_abi_map,
            physical_buffer_abi_shapes,
            physical_bank_buffers,
        )
        try:
            from scripts import m04_train_step  # noqa: PLC0415

            training_abi_contract = (
                m04_train_step._path_c_fused_train_block_training_abi_contract_payload(
                    prim_func
                )
            )
            parameter_gradient_aliases = (
                model.path_c_parameter_gradient_aliases()
                if callable(getattr(model, "path_c_parameter_gradient_aliases", None))
                else {}
            )
            trainable_parameter_names = tuple(
                sorted(m04_train_step._path_c_model_trainable_parameter_names(model))
            )
            selected_region_metadata = (
                m04_train_step._path_c_fused_train_block_selected_region_metadata(
                    region
                )
            )
        except Exception:
            training_abi_contract = {"generated_stage_artifact": True}
            parameter_gradient_aliases = {}
            trainable_parameter_names = ()
            selected_region_metadata = {}
        if generated_stage_artifact:
            runtime_artifact = PathCGeneratedStageTrainBlockCallableArtifact(
                forward_kernel=forward_stage_artifacts[0],
                forward_kernels=forward_stage_artifacts,
                backward_kernel=backward_stage_artifacts[0]
                if backward_stage_artifacts
                else None,
                backward_kernels=backward_stage_artifacts,
                physical_abi_map=physical_buffer_abi_map,
                physical_abi_shapes=physical_buffer_abi_shapes,
                training_abi_contract=training_abi_contract,
                parameter_gradient_aliases=parameter_gradient_aliases,
                trainable_parameter_names=trainable_parameter_names,
                selected_region_metadata=selected_region_metadata,
                forward_kernel_buffer_order=path_c_kernel_buffer_order(
                    forward_stage_prim_funcs[0]
                ),
                forward_kernel_buffer_orders=tuple(
                    path_c_kernel_buffer_order(stage_prim_func)
                    for stage_prim_func in forward_stage_prim_funcs
                ),
                backward_kernel_buffer_order=path_c_kernel_buffer_order(
                    backward_stage_prim_funcs[0]
                ),
                backward_kernel_buffer_orders=tuple(
                    path_c_kernel_buffer_order(stage_prim_func)
                    for stage_prim_func in backward_stage_prim_funcs
                ),
                forward_kernel_buffer_shapes=_kernel_buffer_shapes_for_prim_func(
                    forward_stage_prim_funcs[0]
                ),
                forward_kernel_buffer_shapes_by_stage=tuple(
                    _kernel_buffer_shapes_for_prim_func(stage_prim_func)
                    for stage_prim_func in forward_stage_prim_funcs
                ),
                backward_kernel_buffer_shapes=_kernel_buffer_shapes_for_prim_func(
                    backward_stage_prim_funcs[0]
                ),
                backward_kernel_buffer_shapes_by_stage=tuple(
                    _kernel_buffer_shapes_for_prim_func(stage_prim_func)
                    for stage_prim_func in backward_stage_prim_funcs
                ),
                backward_gate_param=getattr(
                    prim_func,
                    "_cppmega_path_c_backward_gate_param",
                    "path_c_run_backward",
                )
                or "path_c_run_backward",
                row_chunk_count=getattr(
                    prim_func,
                    "_cppmega_path_c_row_chunk_count",
                    None,
                ),
                row_chunk_index_param=getattr(
                    prim_func,
                    "_cppmega_path_c_row_chunk_index_param",
                    None,
                ),
                row_subchunk_count=getattr(
                    prim_func,
                    "_cppmega_path_c_row_subchunk_count",
                    None,
                ),
                row_subchunk_index_param=getattr(
                    prim_func,
                    "_cppmega_path_c_row_subchunk_index_param",
                    None,
                ),
                rows_per_kernel_launch=getattr(
                    prim_func,
                    "_cppmega_path_c_rows_per_kernel_launch",
                    None,
                ),
                forward_stage_row_launches=_path_c_stage_row_launch_specs(
                    forward_stage_prim_funcs
                ),
                backward_stage_row_launches=_path_c_stage_row_launch_specs(
                    backward_stage_prim_funcs
                ),
            )
        else:
            runtime_artifact = PathCFusedTrainBlockCallableArtifact(
                kernel=artifact,
                physical_abi_map=physical_buffer_abi_map,
                physical_abi_shapes=physical_buffer_abi_shapes,
                training_abi_contract=training_abi_contract,
                parameter_gradient_aliases=parameter_gradient_aliases,
                trainable_parameter_names=trainable_parameter_names,
                selected_region_metadata=selected_region_metadata,
                kernel_buffer_order=path_c_kernel_buffer_order(prim_func),
            )
        smoke_bank_owner = make_physical_abi_bank_owner(
            "runtime_smoke.path_c_physical_abi_banks",
            physical_buffer_abi_map,
            physical_buffer_abi_shapes,
            {
                **physical_bank_buffers,
                **{
                    name: value
                    for name, value in kernel_buffers.items()
                    if name not in physical_bank_buffers
                },
            },
        )
        fused_train_block_route = _runtime_smoke_fused_train_block_route_payload(
            mode=mode,
            model=model,
            artifact=runtime_artifact,
            physical_buffer_abi_map=physical_buffer_abi_map,
            physical_buffer_abi_shapes=physical_buffer_abi_shapes,
            physical_bank_buffers=physical_bank_buffers,
        )
        kernel_buffer_order = tuple(str(entry["name"]) for entry in buffer_abi)
        kernel_args, scalar_kernel_args = _runtime_smoke_full_kernel_args(
            prim_func,
            kernel_buffers,
        )
        arrays = [arg for arg in kernel_args if isinstance(arg, mx.array)]
        mx.eval(*arrays)

        row_chunk_param = str(
            getattr(prim_func, "_cppmega_path_c_row_chunk_index_param", "") or ""
        )
        row_subchunk_param = str(
            getattr(prim_func, "_cppmega_path_c_row_subchunk_index_param", "") or ""
        )
        row_chunk_count = (
            max(1, int(getattr(prim_func, "_cppmega_path_c_row_chunk_count", 1) or 1))
            if row_chunk_param
            else 1
        )
        row_subchunk_count = (
            max(
                1,
                int(getattr(prim_func, "_cppmega_path_c_row_subchunk_count", 1) or 1),
            )
            if row_subchunk_param
            else 1
        )
        backward_gate_param = str(
            getattr(
                prim_func,
                "_cppmega_path_c_backward_gate_param",
                "path_c_run_backward",
            )
            or "path_c_run_backward"
        )

        def launch_generated_artifact(*, run_backward: bool) -> tuple[Any, int, bool]:
            result: Any = None
            executed = 0
            for chunk_index in range(row_chunk_count):
                for subchunk_index in range(row_subchunk_count):
                    if launch_limit is not None and executed >= launch_limit:
                        return result, executed, True
                    scalar_params = {backward_gate_param: 1 if run_backward else 0}
                    if row_chunk_param:
                        scalar_params[row_chunk_param] = int(chunk_index)
                    if row_subchunk_param:
                        scalar_params[row_subchunk_param] = int(subchunk_index)
                    dispatch_kwargs: dict[str, Any] = {
                        "bank_owner": smoke_bank_owner,
                        "kernel_scalar_params": scalar_params,
                    }
                    if generated_stage_artifact:
                        dispatch_kwargs["max_stages"] = stage_limit
                    result = runtime_artifact.forward(**dispatch_kwargs)
                    executed += 1
            return result, executed, False

        execute_started = time.perf_counter()
        forward_result, executed_forward_launches, forward_launch_limited = (
            launch_generated_artifact(run_backward=False)
        )
        mx.eval(*physical_bank_buffers.values())
        backward_result, executed_backward_launches, backward_launch_limited = (
            launch_generated_artifact(run_backward=True)
        )
        mx.eval(*arrays)
        execute_elapsed_s = time.perf_counter() - execute_started
        total_forward_launches = row_chunk_count * row_subchunk_count
        total_backward_launches = row_chunk_count * row_subchunk_count
        launch_limited = bool(forward_launch_limited or backward_launch_limited)
        stage_limited = bool(
            generated_stage_artifact
            and
            stage_limit is not None
            and (
                stage_limit < len(forward_stage_artifacts)
                or stage_limit < len(backward_stage_artifacts)
            )
        )
        executed_launches = executed_forward_launches + executed_backward_launches
        return {
            "status": "ok",
            "mode": smoke_mode,
            "actually_executed": executed_launches > 0,
            "full_execution": not launch_limited and not stage_limited,
            "launch_limited": launch_limited,
            "stage_limited": stage_limited,
            "max_launches": launch_limit,
            "max_stages": stage_limit,
            "target": target_name,
            "execution_backend": execution_backend,
            "region_name": region.name,
            "node_names": list(region.node_names),
            "schedule_id": target.schedule_id,
            "implementation_kind": target.implementation_kind,
            "physical_abi_policy": getattr(
                prim_func,
                "_cppmega_path_c_physical_abi_policy",
                "unknown",
            ),
            "physical_abi_runtime_binding": physical_abi_runtime_binding,
            "fused_train_block_route": fused_train_block_route,
            "runtime_smoke_uses_fused_train_block": (
                fused_train_block_route.get("status") == "ok"
            ),
            "runtime_kernel_buffers": list(kernel_buffer_order),
            "runtime_stage_kernel_buffers": [
                str(entry["name"]) for entry in stage_buffer_abi
            ],
            "runtime_kernel_arg_count": len(kernel_args),
            "runtime_scalar_args": scalar_kernel_args,
            "runtime_launch_grid": {
                "row_chunk_param": row_chunk_param or None,
                "row_chunk_count": row_chunk_count,
                "row_subchunk_param": row_subchunk_param or None,
                "row_subchunk_count": row_subchunk_count,
                "total_forward_launches": total_forward_launches,
                "total_backward_launches": total_backward_launches,
                "executed_forward_launches": executed_forward_launches,
                "executed_backward_launches": executed_backward_launches,
                "skipped_forward_launches": (
                    total_forward_launches - executed_forward_launches
                ),
                "skipped_backward_launches": (
                    total_backward_launches - executed_backward_launches
                ),
                "forward_stages_per_launch": len(forward_stage_artifacts)
                if generated_stage_artifact
                else 1,
                "backward_stages_per_launch": len(backward_stage_artifacts)
                if generated_stage_artifact
                else 1,
                "executed_forward_stages_per_launch": min(
                    len(forward_stage_artifacts),
                    stage_limit
                    if stage_limit is not None
                    else len(forward_stage_artifacts),
                )
                if generated_stage_artifact
                else 1,
                "executed_backward_stages_per_launch": min(
                    len(backward_stage_artifacts),
                    stage_limit
                    if stage_limit is not None
                    else len(backward_stage_artifacts),
                )
                if generated_stage_artifact
                else 1,
            },
            "kernel_parameter_count": _kernel_parameter_count(prim_func),
            "logical_parameter_count": len(
                getattr(prim_func, "_cppmega_path_c_buffer_abi_shapes", {}) or {}
            ),
            "buffer_abi": buffer_abi,
            "total_buffer_bytes": total_bytes,
            "max_buffer_bytes": max_bytes,
            "compile_elapsed_s": compile_elapsed_s,
            "stage_compile_elapsed_s": stage_compile_elapsed_s,
            "execute_elapsed_s": execute_elapsed_s,
            "elapsed_s": time.perf_counter() - started,
            "artifact_type": type(runtime_artifact).__name__,
            "runtime_artifact_kind": artifact_kind,
            "single_generated_artifact": artifact_kind == "single",
            "generated_stage_artifact": generated_stage_artifact,
            "generated_phase_artifact": artifact_kind == "phased",
            "generated_stage_groups": [
                {
                    "index": group.index,
                    "execution_stage": group.execution_stage,
                    "active_node_names": list(group.active_node_names),
                    "stage_suffix": group.stage_suffix,
                    "row_dispatch_mode": group.row_dispatch_mode,
                    "rows_per_kernel_launch": group.rows_per_kernel_launch,
                    "reason": group.reason,
                }
                for group in (*forward_stage_groups, *backward_stage_groups)
            ],
            "generated_forward_stage_row_launches": list(
                _path_c_stage_row_launch_specs(forward_stage_prim_funcs)
            ),
            "generated_backward_stage_row_launches": list(
                _path_c_stage_row_launch_specs(backward_stage_prim_funcs)
            ),
            "kernel_source": (
                {
                    **_artifact_kernel_source_stats(artifact),
                    "single_generated_artifact": True,
                    "generated_stage_artifact": False,
                }
                if artifact_kind == "single"
                else _stage_kernel_source_payload(
                    monolithic_artifact=artifact,
                    forward_artifacts=forward_stage_artifacts,
                    backward_artifacts=backward_stage_artifacts,
                    artifact_kind=artifact_kind,
                )
            ),
            "stage_source_bytes": {
                **{
                    f"forward_{index}": len(
                        (
                            getattr(
                                stage_prim_func,
                                "_cppmega_path_c_generated_source",
                                "",
                            )
                            or ""
                        ).encode("utf-8")
                    )
                    for index, stage_prim_func in enumerate(forward_stage_prim_funcs)
                },
                **{
                    f"backward_{index}": len(
                        (
                            getattr(
                                stage_prim_func,
                                "_cppmega_path_c_generated_source",
                                "",
                            )
                            or ""
                        ).encode("utf-8")
                    )
                    for index, stage_prim_func in enumerate(backward_stage_prim_funcs)
                },
            },
            "result_type": type((forward_result, backward_result)).__name__,
            "result_repr": repr((forward_result, backward_result))[:500],
            "stdout_tail": stdout[-2000:],
            "stderr_tail": (stderr + stage_output.stderr)[-2000:],
        }
    except Exception as exc:  # pragma: no cover - defensive receipt path
        if (
            "forward_stage_artifacts" in locals()
            and (forward_stage_artifacts or backward_stage_artifacts)
        ):
            kernel_source = _stage_kernel_source_payload(
                monolithic_artifact=artifact if "artifact" in locals() else None,
                forward_artifacts=forward_stage_artifacts,
                backward_artifacts=backward_stage_artifacts,
                artifact_kind=artifact_kind,
            )
        else:
            kernel_source = (
                _artifact_kernel_source_stats(artifact)
                if "artifact" in locals() and artifact is not None
                else None
            )
        return {
            "status": "failed_execute",
            "mode": smoke_mode,
            "actually_executed": False,
            "full_execution": False,
            "launch_limited": False,
            "max_launches": launch_limit,
            "stage_limited": False,
            "max_stages": stage_limit,
            "runtime_artifact_kind": artifact_kind,
            "target": target_name,
            "execution_backend": execution_backend,
            "region_name": getattr(region, "name", None),
            "schedule_id": getattr(target, "schedule_id", None),
            "kernel_parameter_count": _kernel_parameter_count(prim_func)
            if prim_func is not None
            else None,
            "buffer_abi": buffer_abi,
            "total_buffer_bytes": total_bytes,
            "max_buffer_bytes": max_bytes,
            "elapsed_s": time.perf_counter() - started,
            "kernel_source": kernel_source,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }


def build_compile_receipt(
    *,
    native_compile: bool,
    source_out: Path | None,
    target_name: str = "metal",
    execution_backend: str = "tvm_ffi",
    lowerer: Callable[..., Any] | None = None,
    tilelang_entry_lowerer: Callable[..., Any] | None = None,
    runtime_smoke: str = "none",
    runtime_smoke_max_bytes: int = 256 * 1024 * 1024,
    runtime_smoke_max_launches: int | None = None,
    runtime_smoke_max_stages: int | None = None,
    runtime_smoke_artifact: str = "single",
    runtime_smoke_lowerer: Callable[..., Any] | None = None,
) -> tuple[int, dict[str, Any]]:
    started = time.perf_counter()
    profile, model, fwd_region, region, target = _selected_region_and_target()
    prim_func = target.schedule_template(region)
    descriptor_stage_groups = plan_path_c_descriptor_stage_groups(region)
    descriptor_stage_plan = [
        {
            "index": group.index,
            "execution_stage": group.execution_stage,
            "active_node_names": list(group.active_node_names),
            "stage_suffix": group.stage_suffix,
            "row_dispatch_mode": group.row_dispatch_mode,
            "rows_per_kernel_launch": group.rows_per_kernel_launch,
            "reason": group.reason,
        }
        for group in descriptor_stage_groups
    ]
    generated_source = str(
        getattr(prim_func, "_cppmega_path_c_generated_source", "")
        or prim_func.script()
    )
    spilled_shared_scratch_shapes = dict(
        getattr(prim_func, "_cppmega_path_c_spilled_shared_scratch_shapes", {})
        or {}
    )
    physical_buffer_abi_map = dict(
        getattr(prim_func, "_cppmega_path_c_physical_buffer_abi_map", {})
        or {}
    )
    physical_buffer_abi_shapes = dict(
        getattr(
            prim_func,
            "_cppmega_path_c_physical_buffer_abi_shapes",
            {},
        )
        or {}
    )
    physical_abi_runtime_bridge = plan_physical_abi_runtime_bridge(
        physical_buffer_abi_map,
        physical_buffer_abi_shapes,
    )
    physical_abi_runtime_binding = validate_physical_abi_runtime_bindings(
        physical_buffer_abi_map,
        physical_buffer_abi_shapes,
        None,
    )
    direct_logical_abi_alternative = _direct_logical_abi_alternative_payload(
        region=region,
        target=target,
        target_name=target_name,
        shape_env=getattr(prim_func, "_cppmega_path_c_shape_env", None),
        model=model,
        selected_forward_region=fwd_region,
    )
    internal_scratch_abi_buffers = tuple(
        getattr(prim_func, "_cppmega_path_c_internal_scratch_abi_buffers", ())
        or ()
    )
    shared_scratch_abi_bytes = sum(
        int(info.get("bytes", 0))
        for info in spilled_shared_scratch_shapes.values()
        if isinstance(info, dict)
    )
    if source_out is not None:
        source_out.parent.mkdir(parents=True, exist_ok=True)
        source_out.write_text(generated_source, encoding="utf-8")

    schedule_template = mark_path_c_schedule_template_for_region(
        lambda _region: prim_func,
        region,
        implementation_kind=target.implementation_kind,
        production_schedule_id=target.schedule_id
        if target.implementation_kind == "production"
        else "",
        required_real_abi_inputs=target.required_real_abi_inputs,
    )
    schedule_spec = path_c_fusion_schedule_spec(
        region,
        target=target,
    )
    compile_stdout = ""
    compile_stderr = ""
    compile_error: str | None = None
    compiled: FusionCompilePlan | CompiledPathCRegion
    native_lowerer = lowerer
    if native_compile and native_lowerer is None:
        entry_lowerer = tilelang_entry_lowerer or tilelang_single_entry_lowerer

        def native_lowerer(func_or_mod: Any, *, target: str, **kwargs: Any) -> Any:
            return entry_lowerer(
                func_or_mod,
                target=target,
                execution_backend=execution_backend,
                **kwargs,
            )

    direct_logical_abi_alternative["direct_chained_fusion_native_compile"] = (
        _native_compile_direct_chain_payload(
            model=model,
            selected_region=fwd_region,
            already_has_backward=False,
            native_compile=native_compile,
            native_lowerer=native_lowerer,
            target_name=target_name,
        )
    )

    if native_compile:
        compiled, compile_stdout, compile_stderr, compile_error, _compile_elapsed_s = (
            _compile_path_c_region_with_native_capture(
                region=region,
                schedule_template=schedule_template,
                schedule_name=target.schedule_name,
                schedule_status=target.schedule_status,
                native_lowerer=native_lowerer,
                target_name=target_name,
            )
        )
    else:
        compiled = compile_path_c_region(
            region,
            schedule_template=schedule_template,
            schedule_name=target.schedule_name,
            schedule_status=target.schedule_status,
        )
        _compile_elapsed_s = 0.0

    plan = compiled.plan if isinstance(compiled, CompiledPathCRegion) else compiled
    artifact = compiled.artifact if isinstance(compiled, CompiledPathCRegion) else None
    contract = plan.schedule_contract
    native_compile_ok = bool(native_compile and compile_error is None and artifact is not None)
    status = "ok" if (not native_compile or native_compile_ok) else "failed"
    elapsed_s = time.perf_counter() - started
    tilelang_root = _tilelang_root()
    runtime_smoke_payload = _runtime_smoke_payload(
        mode=runtime_smoke,
        target_name=target_name,
        execution_backend=execution_backend,
        max_bytes=runtime_smoke_max_bytes,
        max_launches=runtime_smoke_max_launches,
        max_stages=runtime_smoke_max_stages,
        artifact_kind=runtime_smoke_artifact,
        lowerer=runtime_smoke_lowerer,
    )
    production_runtime_smoke_binding = _production_runtime_smoke_binding(
        runtime_smoke_payload=runtime_smoke_payload,
        region_name=region.name,
    )
    runtime_contract_binding = (
        production_runtime_smoke_binding
        if production_runtime_smoke_binding is not None
        else physical_abi_runtime_binding
    )
    production_runtime_smoke_uses_fused_train_block = (
        production_runtime_smoke_binding is not None
    )
    runtime_smoke_fused_route = runtime_smoke_payload.get("fused_train_block_route")
    runtime_smoke_uses_fused_train_block = bool(
        isinstance(runtime_smoke_fused_route, Mapping)
        and runtime_smoke_fused_route.get("status") == "ok"
    )
    payload = {
        "kind": "cppmega_path_c_fusion_compile_receipt",
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "status": status,
        "native_compile_requested": native_compile,
        "native_compile_ok": native_compile_ok,
        "native_compile_error": compile_error,
        "target": target_name,
        "execution_backend": execution_backend,
        "elapsed_s": elapsed_s,
        "model_profile": profile.name,
        "route_symbols": list(profile.expanded_pattern.symbols),
        "selected_forward_region": {
            "name": fwd_region.name,
            "node_names": list(fwd_region.node_names),
            "op_signature": [node.op_name for node in fwd_region.nodes],
        },
        "selected_aot_region": {
            "name": region.name,
            "node_names": list(region.node_names),
            "op_signature": [node.op_name for node in region.nodes],
        },
        "schedule_target": {
            "schedule_id": target.schedule_id,
            "schedule_name": target.schedule_name,
            "schedule_status": target.schedule_status,
            "implementation_kind": target.implementation_kind,
            "schedule_generator": target.schedule_generator,
            "required_real_abi_inputs": list(target.required_real_abi_inputs),
            "brick_ops": [
                descriptor.op_name for descriptor in target.brick_descriptors
            ],
        },
        "descriptor_stage_plan": descriptor_stage_plan,
        "schedule_spec": {
            "schedule_id": schedule_spec.schedule_id,
            "implementation_kind": schedule_spec.implementation_kind,
            "implementation_status": schedule_spec.implementation_status,
            "trusted_by_default": schedule_spec.trusted_by_default,
            "real_abi_contract_complete": schedule_spec.real_abi_contract_complete,
            "missing_real_abi_inputs": list(schedule_spec.missing_real_abi_inputs),
            "required_real_abi_input_shapes": list(
                schedule_spec.required_real_abi_input_shapes
            ),
            "production_fragments_complete": (
                schedule_spec.production_fragments_complete
            ),
            "brick_production_fragment_statuses": list(
                schedule_spec.brick_production_fragment_statuses
            ),
            "loop_policy": schedule_spec.loop_policy,
            "internal_buffer_policy": schedule_spec.internal_buffer_policy,
            "loop_extent": schedule_spec.loop_extent,
        },
        "compile_plan": _plan_payload(plan),
        "fusion_cache_key": _fusion_cache_key_payload(compiled),
        "cache_key_recompile_audit": _cache_key_recompile_audit_payload(
            native_compile=native_compile,
            primary_compiled=compiled,
            primary_compile_error=compile_error,
            region=region,
            schedule_template=schedule_template,
            schedule_name=target.schedule_name,
            schedule_status=target.schedule_status,
            native_lowerer=native_lowerer if native_compile else None,
            target_name=target_name,
        ),
        "runtime_execution_contract": _runtime_execution_contract(
            generated_source=generated_source,
            schedule_spec=schedule_spec,
            plan=plan,
            kernel_parameter_count=_kernel_parameter_count(prim_func),
            target_name=target_name,
            physical_abi_runtime_bridge=physical_abi_runtime_bridge,
            physical_abi_runtime_binding=runtime_contract_binding,
        ),
        "direct_logical_abi_alternative": direct_logical_abi_alternative,
        "runtime_smoke": runtime_smoke_payload,
        "default_eligible": bool(
            plan.single_kernel_fused
            and contract is not None
            and contract.status == "verified"
            and schedule_spec.trusted_by_default
        ),
        "artifact": {
            "type": type(artifact).__name__ if artifact is not None else None,
            "repr": repr(artifact)[:500] if artifact is not None else None,
        },
        "generated_source": {
            "path": str(source_out) if source_out is not None else None,
            "bytes": len(generated_source.encode("utf-8")),
            "sha256": __import__("hashlib")
            .sha256(generated_source.encode("utf-8"))
            .hexdigest(),
            "physical_abi_policy": getattr(
                prim_func,
                "_cppmega_path_c_physical_abi_policy",
                "unknown",
            ),
            "logical_parameter_count": len(
                getattr(prim_func, "_cppmega_path_c_buffer_abi_shapes", {}) or {}
            ),
            "physical_parameter_count": _kernel_parameter_count(prim_func),
            "physical_buffer_abi_shapes": physical_buffer_abi_shapes,
            "physical_buffer_abi_map": physical_buffer_abi_map,
            "logical_buffer_abi_map_count": len(physical_buffer_abi_map),
            "physical_abi_validation": validate_physical_abi_map(
                physical_buffer_abi_map,
                physical_buffer_abi_shapes,
            ),
            "physical_abi_runtime_bridge": physical_abi_runtime_bridge,
            "physical_abi_runtime_binding": physical_abi_runtime_binding,
            "compile_pass_configs": dict(
                getattr(prim_func, "_cppmega_path_c_compile_pass_configs", {}) or {}
            ),
            "spilled_shared_scratch_shapes": spilled_shared_scratch_shapes,
            "spilled_shared_scratch_count": len(spilled_shared_scratch_shapes),
            "shared_scratch_abi_bytes": shared_scratch_abi_bytes,
            "internal_scratch_abi_buffers": list(internal_scratch_abi_buffers),
            "internal_scratch_abi_count": len(internal_scratch_abi_buffers),
        },
        "compile_log": {
            "stdout_tail": compile_stdout[-4000:],
            "stderr_tail": compile_stderr[-4000:],
        },
        "software": {
            "cppmega_sha": _git_sha(ROOT),
            "tilelang_sha": _git_sha(tilelang_root) if tilelang_root else "unknown",
            "tilelang_root": str(tilelang_root) if tilelang_root else None,
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "reporting_contract": {
            "matrix_measures_current_runtime_route": True,
            "compile_receipt_measures_fused_schedule_compile": True,
            "runtime_uses_fused_train_block": False,
            "runtime_smoke_uses_fused_train_block": (
                runtime_smoke_uses_fused_train_block
            ),
            "production_runtime_smoke_uses_fused_train_block": (
                production_runtime_smoke_uses_fused_train_block
            ),
            "path_c_default_allowed": False,
        },
    }
    return (0 if status == "ok" else 2), _json_ready(payload)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    exit_code, payload = build_compile_receipt(
        native_compile=not bool(args.no_native_compile),
        source_out=args.source_out,
        target_name=args.target,
        execution_backend=args.execution_backend,
        runtime_smoke=args.runtime_smoke,
        runtime_smoke_max_bytes=args.runtime_smoke_max_bytes,
        runtime_smoke_max_launches=args.runtime_smoke_max_launches,
        runtime_smoke_max_stages=args.runtime_smoke_max_stages,
        runtime_smoke_artifact=args.runtime_smoke_artifact,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(args.out)
    if exit_code != 0 and payload.get("native_compile_error"):
        print(payload["native_compile_error"], file=sys.stderr)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
