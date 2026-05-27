#!/usr/bin/env python3
"""Fail-closed guardrails for cppmega path taxonomies and benchmark reports."""

from __future__ import annotations

import argparse
import ast
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


ROOT = Path(__file__).resolve().parents[1]
ALLOWED_CONTRACT_STATES = {
    "runnable",
    "runnable_if_available",
    "runnable_baseline",
    "runnable_candidate",
    "retired",
    "test_only",
}
PATH_B_BLOCKER_MARKERS = (
    "Path B kernel unavailable",
    "direct-MSL Path B is retired",
    "Path B is retired",
)
REASON_KEYS = {
    "blocker",
    "blockers",
    "error",
    "failure_reason",
    "fallback_reason",
    "pass_fail_reason",
    "reason",
    "status_reason",
}
PATH_D_PROBE_MODULES = (
    Path("cppmega_v4/_tilelang/linear_attention_path_d.py"),
    Path("cppmega_v4/_tilelang/kda_path_d.py"),
)
PATH_D_FLA_PROBES = {
    Path("cppmega_v4/_tilelang/linear_attention_path_d.py"): "_fla_chunk_kernel_importable",
    Path("cppmega_v4/_tilelang/kda_path_d.py"): "_fla_kda_chunk_importable",
}
PATH_D_UNSAFE_IMPORT_ENV = "CPPMEGA_V4_ENABLE_UNSAFE_TRITON_IMPORT"
PATH_D_DEFAULT_STATUS_TIMEOUT_SECONDS = 20
PATH_D_DEFAULT_STATUS_PROBE_CODE = r"""
import json
import os
import sys

UNSAFE_ENV = "CPPMEGA_V4_ENABLE_UNSAFE_TRITON_IMPORT"
os.environ.pop(UNSAFE_ENV, None)

from cppmega_v4._tilelang.kda_path_d import (
    _fla_kda_chunk_importable,
    _path_d_runtime_status as kda_runtime_status,
    _triton_frontend_importable as kda_triton_importable,
)
from cppmega_v4._tilelang.linear_attention_path_d import (
    _fla_chunk_kernel_importable,
    _path_d_runtime_status as gdn_runtime_status,
    _triton_frontend_importable as gdn_triton_importable,
)


def call(label, fn):
    ok, reason = fn()
    return {"label": label, "ok": bool(ok), "reason": str(reason)}


payload = {
    "probes": [
        call("gdn.runtime_status", gdn_runtime_status),
        call("gdn.triton_frontend", gdn_triton_importable),
        call("gdn.fla_chunk", _fla_chunk_kernel_importable),
        call("kda.runtime_status", kda_runtime_status),
        call("kda.triton_frontend", kda_triton_importable),
        call("kda.fla_chunk", _fla_kda_chunk_importable),
    ],
    "unsafe_modules": sorted(
        name
        for name in sys.modules
        if name == "triton"
        or name.startswith("triton.")
        or name == "fla"
        or name.startswith("fla.")
    ),
}
print(json.dumps(payload, sort_keys=True))
"""
PATH_E_ADAPTER_MODULES = {
    "v4.gdn.path_e": Path("cppmega_v4/nn/_external/mlx_lm_gated_delta_update.py"),
    "v4.kda.path_e": Path("cppmega_v4/nn/_external/mlx_lm_kda_update.py"),
}
PATH_E_STATUS_IMPORTS = {
    "v4.gdn.path_e": (
        Path("cppmega_v4/_tilelang/linear_attention_paths.py"),
        "_path_e_status",
        "cppmega_v4.nn._external.mlx_lm_gated_delta_update",
    ),
    "v4.kda.path_e": (
        Path("cppmega_v4/_tilelang/kda_paths.py"),
        "_path_e_status",
        "cppmega_v4.nn._external.mlx_lm_kda_update",
    ),
}
BLOCKED_BENCHMARK_ROLES = {"blocked_candidate"}
BLOCKED_CONTRACT_STATES = {"retired", "test_only"}


@dataclass(frozen=True)
class Finding:
    code: str
    severity: str
    location: str
    detail: str


@dataclass(frozen=True)
class PathContract:
    surface: str
    path: str
    state: str
    benchmark_role: str
    reason: str


PATH_CONTRACTS: tuple[PathContract, ...] = (
    PathContract(
        "m04.training_matrix",
        "path_b",
        "runnable_baseline",
        "baseline",
        "Path B is the 1B matrix baseline; failed or retired raw receipts make comparisons invalid.",
    ),
    PathContract(
        "m04.training_matrix",
        "path_c_cold",
        "runnable_candidate",
        "candidate",
        "Cold Path C measures compile/cache setup cost against a valid Path B baseline.",
    ),
    PathContract(
        "m04.training_matrix",
        "path_c_warm",
        "runnable_candidate",
        "candidate",
        "Warm Path C measures steady-state candidate speed against a valid Path B baseline.",
    ),
    PathContract(
        "v4.gdn",
        "path_a",
        "runnable",
        "reference",
        "Pure MLX GDN reference path.",
    ),
    PathContract(
        "v4.gdn",
        "path_b",
        "runnable_if_available",
        "candidate",
        "GDN hand-MSL path is eligible only when its status reports backend_available.",
    ),
    PathContract(
        "v4.gdn",
        "path_c",
        "runnable_if_available",
        "candidate",
        "GDN TileLang/TVM-FFI path is eligible only when its status reports backend_available.",
    ),
    PathContract(
        "v4.gdn",
        "path_d",
        "test_only",
        "blocked_candidate",
        "GDN Triton-frontend Path D remains fallback/test-only until the runtime adapter is wired.",
    ),
    PathContract(
        "v4.gdn",
        "path_e",
        "runnable_if_available",
        "candidate",
        "GDN vendored mlx-lm Path E is eligible only when its status reports backend_available.",
    ),
    PathContract(
        "v4.kda",
        "path_a",
        "runnable",
        "reference",
        "Pure MLX KDA reference path.",
    ),
    PathContract(
        "v4.kda",
        "path_b",
        "runnable_if_available",
        "candidate",
        "KDA hand-MSL path is eligible only when its status reports backend_available.",
    ),
    PathContract(
        "v4.kda",
        "path_c",
        "runnable_if_available",
        "candidate",
        "KDA TileLang/TVM-FFI path is eligible only when its status reports backend_available.",
    ),
    PathContract(
        "v4.kda",
        "path_d",
        "runnable_if_available",
        "candidate",
        "KDA Triton-frontend Path D is eligible only when its runtime adapter reports backend_available.",
    ),
    PathContract(
        "v4.kda",
        "path_e",
        "runnable_if_available",
        "candidate",
        "KDA vendored mlx-lm vectorised-gate Path E is eligible only when its status reports backend_available.",
    ),
)
CONTRACT_BY_KEY = {(item.surface, item.path): item for item in PATH_CONTRACTS}


def _read_ast(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _literal_strings(node: ast.AST) -> tuple[str, ...]:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return (node.value,)
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        values: list[str] = []
        for item in node.elts:
            values.extend(_literal_strings(item))
        return tuple(values)
    if isinstance(node, ast.Subscript):
        return _literal_strings(node.slice)
    return ()


def _tuple_assignment(module: ast.Module, name: str) -> tuple[str, ...]:
    for node in module.body:
        if isinstance(node, ast.Assign):
            if not any(
                isinstance(target, ast.Name) and target.id == name
                for target in node.targets
            ):
                continue
            return _literal_strings(node.value)
        if (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == name
            and node.value is not None
        ):
            return _literal_strings(node.value)
    return ()


def _function_assignment_dict_keys(
    module: ast.Module,
    function_name: str,
    assignment_name: str,
) -> tuple[str, ...]:
    def keys_from_dict(node: ast.AST | None) -> tuple[str, ...]:
        if isinstance(node, ast.Subscript):
            node = node.value
        if not isinstance(node, ast.Dict):
            return ()
        keys: list[str] = []
        for key in node.keys:
            if isinstance(key, ast.Constant) and isinstance(key.value, str):
                keys.append(key.value)
        return tuple(keys)

    for node in module.body:
        if not isinstance(node, ast.FunctionDef) or node.name != function_name:
            continue
        for child in ast.walk(node):
            if isinstance(child, ast.Assign):
                if not any(
                    isinstance(target, ast.Name) and target.id == assignment_name
                    for target in child.targets
                ):
                    continue
                keys = keys_from_dict(child.value)
                if keys:
                    return keys
            if (
                isinstance(child, ast.AnnAssign)
                and isinstance(child.target, ast.Name)
                and child.target.id == assignment_name
            ):
                keys = keys_from_dict(child.value)
                if keys:
                    return keys
    return ()


def _function_def(module: ast.Module, function_name: str) -> ast.FunctionDef | None:
    for node in module.body:
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            return node
    return None


def _call_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _call_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    return None


def _first_call_line(function: ast.FunctionDef, call_name: str) -> int | None:
    lines = [
        node.lineno
        for node in ast.walk(function)
        if isinstance(node, ast.Call) and _call_name(node.func) == call_name
    ]
    return min(lines) if lines else None


def _first_import_line(function: ast.FunctionDef, module_name: str) -> int | None:
    lines: list[int] = []
    for node in ast.walk(function):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == module_name:
                    lines.append(node.lineno)
        if isinstance(node, ast.ImportFrom) and node.module == module_name:
            lines.append(node.lineno)
    return min(lines) if lines else None


def _first_triton_import_probe_line(function: ast.FunctionDef) -> int | None:
    lines = [
        line
        for line in (
            _first_import_line(function, "triton"),
            _first_call_line(function, "import_triton_with_local_symbols"),
        )
        if line is not None
    ]
    return min(lines) if lines else None


def _first_import_prefix_line(function: ast.FunctionDef, module_prefix: str) -> int | None:
    lines: list[int] = []
    for node in ast.walk(function):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == module_prefix or alias.name.startswith(f"{module_prefix}."):
                    lines.append(node.lineno)
        if isinstance(node, ast.ImportFrom) and node.module is not None:
            if node.module == module_prefix or node.module.startswith(f"{module_prefix}."):
                lines.append(node.lineno)
    return min(lines) if lines else None


def _function_string_literals(function: ast.FunctionDef) -> tuple[str, ...]:
    return tuple(
        node.value
        for node in ast.walk(function)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    )


def _pathname_literal(module: ast.Module) -> tuple[str, ...]:
    for node in module.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(target, ast.Name) and target.id == "PathName" for target in node.targets):
            continue
        return _literal_strings(node.value)
    return ()


def discover_declared_paths(repo_root: Path = ROOT) -> dict[str, tuple[str, ...]]:
    """Discover path declarations without importing MLX or TileLang modules."""

    bench = _read_ast(repo_root / "scripts" / "bench_1b_training_matrix.py")
    dispatch = _read_ast(repo_root / "cppmega_v4" / "_tilelang" / "_dispatch.py")
    matrix = _read_ast(repo_root / "cppmega_v4" / "_tilelang" / "benchmark_matrix.py")
    gdn = _read_ast(repo_root / "cppmega_v4" / "_tilelang" / "linear_attention_paths.py")
    kda = _read_ast(repo_root / "cppmega_v4" / "_tilelang" / "kda_paths.py")
    return {
        "m04.training_matrix": _tuple_assignment(bench, "PATH_CHOICES"),
        "v4.dispatch.PathName": _pathname_literal(dispatch),
        "v4.gdn": _tuple_assignment(matrix, "GDN_PATHS"),
        "v4.kda": _tuple_assignment(matrix, "KDA_PATHS"),
        "v4.gdn.statuses": _function_assignment_dict_keys(
            gdn,
            "linear_attention_path_statuses",
            "statuses",
        ),
        "v4.gdn.dispatch": _function_assignment_dict_keys(
            gdn,
            "gated_delta_recurrent_dispatch",
            "fn",
        ),
        "v4.kda.statuses": _function_assignment_dict_keys(
            kda,
            "kda_path_statuses",
            "statuses",
        ),
        "v4.kda.dispatch": _function_assignment_dict_keys(
            kda,
            "kda_recurrent_dispatch",
            "fn",
        ),
    }


def _check_v4_path_d_import_guards(repo_root: Path) -> list[Finding]:
    findings: list[Finding] = []
    for rel_path in PATH_D_PROBE_MODULES:
        module = _read_ast(repo_root / rel_path)
        function = _function_def(module, "_triton_frontend_importable")
        location = str(rel_path)
        if function is None:
            findings.append(
                Finding(
                    code="missing_path_d_triton_probe",
                    severity="error",
                    location=location,
                    detail="_triton_frontend_importable() is required for Path D status reporting",
                )
            )
            continue
        guard_line = _first_call_line(
            function,
            "unsafe_triton_frontend_import_enabled",
        )
        import_line = _first_triton_import_probe_line(function)
        if import_line is None:
            findings.append(
                Finding(
                    code="missing_path_d_triton_import_probe",
                    severity="error",
                    location=location,
                    detail=(
                        "_triton_frontend_importable() must explicitly probe "
                        "Triton with local native symbols after the unsafe-import guard"
                    ),
                )
            )
            continue
        if guard_line is None or guard_line > import_line:
            findings.append(
                Finding(
                    code="unsafe_path_d_import_probe",
                    severity="error",
                    location=location,
                    detail=(
                        "Path D status probe imports triton before checking "
                        "unsafe_triton_frontend_import_enabled(); local Triton "
                        "checkouts can abort the interpreter during import"
                    ),
                )
            )
        fla_function = _function_def(module, PATH_D_FLA_PROBES[rel_path])
        if fla_function is None:
            findings.append(
                Finding(
                    code="missing_path_d_fla_probe",
                    severity="error",
                    location=location,
                    detail="Path D FLA import probe is required for status reporting",
                )
            )
            continue
        fla_guard_line = _first_call_line(
            fla_function,
            "unsafe_triton_frontend_import_enabled",
        )
        fla_import_line = _first_import_prefix_line(fla_function, "fla")
        if fla_import_line is None:
            findings.append(
                Finding(
                    code="missing_path_d_fla_import_probe",
                    severity="error",
                    location=location,
                    detail="Path D FLA probe must explicitly import the FLA source module after the unsafe-import guard",
                )
            )
            continue
        if fla_guard_line is None or fla_guard_line > fla_import_line:
            findings.append(
                Finding(
                    code="unsafe_path_d_fla_import_probe",
                    severity="error",
                    location=location,
                    detail=(
                        "Path D FLA probe imports FLA before checking "
                        "unsafe_triton_frontend_import_enabled(); FLA imports "
                        "can transitively import Triton and abort the interpreter"
                    ),
                )
            )
    return findings


def _path_d_disabled_reason_is_clear(reason: str) -> bool:
    return (
        "unsafe" in reason.lower()
        and "runtime adapter not reached" in reason
        and PATH_D_UNSAFE_IMPORT_ENV in reason
    )


def check_path_d_default_status_no_unsafe_imports(
    repo_root: Path = ROOT,
) -> list[Finding]:
    """Run default Path D status probes in a subprocess.

    Local Triton/FLA imports can abort the interpreter. This guard keeps that
    failure out of the main test process and verifies the default path stays
    fail-closed with an actionable disabled reason.
    """

    env = os.environ.copy()
    env.pop(PATH_D_UNSAFE_IMPORT_ENV, None)
    pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        f"{repo_root}{os.pathsep}{pythonpath}" if pythonpath else str(repo_root)
    )
    try:
        proc = subprocess.run(
            [sys.executable, "-c", PATH_D_DEFAULT_STATUS_PROBE_CODE],
            cwd=repo_root,
            env=env,
            text=True,
            capture_output=True,
            timeout=PATH_D_DEFAULT_STATUS_TIMEOUT_SECONDS,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return [
            Finding(
                code="path_d_default_status_probe_timeout",
                severity="error",
                location="v4.path_d",
                detail=(
                    "default Path D status subprocess timed out before proving "
                    "unsafe Triton/FLA imports stay disabled"
                ),
            )
        ]

    if proc.returncode != 0:
        detail = _first_line(proc.stderr or proc.stdout or "no output")
        return [
            Finding(
                code="path_d_default_status_probe_crashed",
                severity="error",
                location="v4.path_d",
                detail=(
                    "default Path D status subprocess exited with "
                    f"{proc.returncode}: {detail}"
                ),
            )
        ]

    try:
        last_line = next(
            line for line in reversed(proc.stdout.splitlines()) if line.strip()
        )
        payload = json.loads(last_line)
    except (StopIteration, json.JSONDecodeError) as exc:
        return [
            Finding(
                code="path_d_default_status_probe_bad_output",
                severity="error",
                location="v4.path_d",
                detail=f"default Path D status subprocess did not emit JSON: {exc}",
            )
        ]

    findings: list[Finding] = []
    unsafe_modules = payload.get("unsafe_modules")
    if not isinstance(unsafe_modules, list):
        findings.append(
            Finding(
                code="path_d_default_status_probe_bad_output",
                severity="error",
                location="v4.path_d",
                detail="default Path D status subprocess omitted unsafe_modules list",
            )
        )
    elif unsafe_modules:
        findings.append(
            Finding(
                code="path_d_default_status_imported_unsafe_deps",
                severity="error",
                location="v4.path_d",
                detail=(
                    "default Path D status probe imported unsafe modules in "
                    f"the subprocess: {unsafe_modules[:8]!r}"
                ),
            )
        )

    probes = payload.get("probes")
    if not isinstance(probes, list):
        findings.append(
            Finding(
                code="path_d_default_status_probe_bad_output",
                severity="error",
                location="v4.path_d",
                detail="default Path D status subprocess omitted probes list",
            )
        )
        return findings

    for probe in probes:
        if not isinstance(probe, dict):
            findings.append(
                Finding(
                    code="path_d_default_status_probe_bad_output",
                    severity="error",
                    location="v4.path_d",
                    detail=(
                        "default Path D status probe payload item is invalid: "
                        f"{probe!r}"
                    ),
                )
            )
            continue
        label = str(probe.get("label") or "unknown")
        reason = str(probe.get("reason") or "")
        if bool(probe.get("ok")):
            findings.append(
                Finding(
                    code="path_d_default_status_marked_available",
                    severity="error",
                    location=label,
                    detail=(
                        "Path D default status probe reported available "
                        "without unsafe import opt-in"
                    ),
                )
            )
        if not _path_d_disabled_reason_is_clear(reason):
            findings.append(
                Finding(
                    code="path_d_default_status_missing_disabled_reason",
                    severity="error",
                    location=label,
                    detail=(
                        "Path D default status probe did not return the "
                        f"unsafe-import disabled reason: {_first_line(reason)}"
                    ),
                )
            )
    return findings


def _check_v4_path_e_adapters(repo_root: Path) -> list[Finding]:
    findings: list[Finding] = []
    for location, rel_path in PATH_E_ADAPTER_MODULES.items():
        if not (repo_root / rel_path).exists():
            findings.append(
                Finding(
                    code="missing_path_e_adapter",
                    severity="error",
                    location=location,
                    detail=f"expected vendored Path E adapter at {rel_path}",
                )
            )
    for location, (rel_path, function_name, import_name) in PATH_E_STATUS_IMPORTS.items():
        module = _read_ast(repo_root / rel_path)
        function = _function_def(module, function_name)
        if function is None:
            findings.append(
                Finding(
                    code="missing_path_e_status",
                    severity="error",
                    location=location,
                    detail=f"{rel_path}:{function_name} is required",
                )
            )
            continue
        if import_name not in _function_string_literals(function):
            findings.append(
                Finding(
                    code="path_e_status_missing_adapter_import",
                    severity="error",
                    location=location,
                    detail=f"{function_name} must status-check {import_name}",
                )
            )
    return findings


def check_path_contracts(repo_root: Path = ROOT) -> list[Finding]:
    declared = discover_declared_paths(repo_root)
    findings: list[Finding] = []
    for contract in PATH_CONTRACTS:
        if contract.state not in ALLOWED_CONTRACT_STATES:
            findings.append(
                Finding(
                    code="invalid_path_contract_state",
                    severity="error",
                    location=f"{contract.surface}.{contract.path}",
                    detail=f"unknown state {contract.state!r}",
                )
            )
        if not contract.reason.strip():
            findings.append(
                Finding(
                    code="empty_path_contract_reason",
                    severity="error",
                    location=f"{contract.surface}.{contract.path}",
                    detail="path contract must explain runnable, retired, or test-only status",
                )
            )

    for surface in ("m04.training_matrix", "v4.gdn", "v4.kda"):
        for path in declared.get(surface, ()):
            if (surface, path) not in CONTRACT_BY_KEY:
                findings.append(
                    Finding(
                        code="missing_path_contract",
                        severity="error",
                        location=f"{surface}.{path}",
                        detail="declared path has no explicit runnable, retired, or test-only contract",
                    )
                )

    dispatch_paths = set(declared.get("v4.dispatch.PathName", ()))
    for surface in ("v4.gdn", "v4.kda"):
        paths = set(declared.get(surface, ()))
        if paths != dispatch_paths:
            findings.append(
                Finding(
                    code="path_taxonomy_mismatch",
                    severity="error",
                    location=surface,
                    detail=(
                        f"{surface} declares {sorted(paths)} but PathName declares "
                        f"{sorted(dispatch_paths)}"
                    ),
                )
            )
        status_keys = set(declared.get(f"{surface}.statuses", ()))
        if status_keys != paths:
            findings.append(
                Finding(
                    code="missing_status_declaration",
                    severity="error",
                    location=f"{surface}.statuses",
                    detail=f"status keys {sorted(status_keys)} do not match declared paths {sorted(paths)}",
                )
            )
        dispatch_keys = set(declared.get(f"{surface}.dispatch", ()))
        if dispatch_keys != paths:
            findings.append(
                Finding(
                    code="missing_dispatch_declaration",
                    severity="error",
                    location=f"{surface}.dispatch",
                    detail=f"dispatch keys {sorted(dispatch_keys)} do not match declared paths {sorted(paths)}",
                )
            )
    findings.extend(_check_v4_path_d_import_guards(repo_root))
    findings.extend(check_path_d_default_status_no_unsafe_imports(repo_root))
    findings.extend(_check_v4_path_e_adapters(repo_root))
    findings.extend(_check_m04_path_b_m2rnn_direct_msl(repo_root))
    findings.extend(_check_m04_path_b_mamba3_fast_bwd(repo_root))
    return findings


def _check_m04_path_b_mamba3_fast_bwd(repo_root: Path) -> list[Finding]:
    """Keep Path B Mamba3 backward on the measured fast partial-reduce route."""

    module_path = repo_root / "cppmega_mlx" / "nn" / "_tilelang" / "mamba3.py"
    location = str(module_path.relative_to(repo_root))
    if not module_path.exists():
        return [
            Finding(
                code="m04_path_b_mamba3_surface_missing",
                severity="error",
                location=location,
                detail=(
                    "Mamba3 Path B module is missing; the 1B Path B baseline "
                    "cannot be trusted."
                ),
            )
        ]

    text = module_path.read_text(encoding="utf-8")
    required = (
        "_FWD_KERNEL_SOURCE",
        "_BWD_KERNEL_SOURCE",
        "@mx.custom_function",
        "mamba3_mimo_apply.vjp",
        "dB_partial",
        "dC_partial",
        "dA_partial",
        "ddt_partial",
        "dD_partial",
        "mx.sum(dB_partial, axis=3)",
        "mx.sum(dD_partial, axis=(0, 2))",
    )
    missing = [symbol for symbol in required if symbol not in text]
    slow_owner_output_markers = [
        marker
        for marker in (
            "cppmega_atomic_add_float",
            "init_value=0",
            "&dB[bc_idx + n]",
            "&dC[bc_idx + n]",
        )
        if marker in text
    ]
    if not missing and not slow_owner_output_markers:
        return []

    detail_parts: list[str] = []
    if missing:
        detail_parts.append("missing fast partial-reduce symbols: " + ", ".join(missing))
    if slow_owner_output_markers:
        detail_parts.append(
            "slow owner-output atomics present: " + ", ".join(slow_owner_output_markers)
        )
    return [
        Finding(
            code="m04_path_b_mamba3_atomic_owner_bwd",
            severity="error",
            location=location,
            detail=(
                "; ".join(detail_parts)
                + "; Path B Mamba3 backward must keep per-lane partial outputs "
                "plus MLX reduction unless a replacement proves 1B Path B "
                "throughput parity."
            ),
        )
    ]


def _check_m04_path_b_m2rnn_direct_msl(repo_root: Path) -> list[Finding]:
    """Keep the 1B Path B baseline from silently degrading to the MLX oracle."""

    module_path = repo_root / "cppmega_mlx" / "nn" / "_tilelang" / "m2rnn.py"
    location = str(module_path.relative_to(repo_root))
    if not module_path.exists():
        return [
            Finding(
                code="m04_path_b_m2rnn_surface_missing",
                severity="error",
                location=location,
                detail=(
                    "M2RNN Path B module is missing; the 1B Path B baseline "
                    "cannot be trusted."
                ),
            )
        ]

    text = module_path.read_text(encoding="utf-8")
    missing = [
        symbol
        for symbol in (
            "_FWD_KERNEL_SOURCE",
            "_BWD_KERNEL_SOURCE",
            "@mx.custom_function",
            "m2rnn_apply_with_state.vjp",
        )
        if symbol not in text
    ]
    retired = "direct-MSL Path B is retired" in text or "_RETIRED_REASON" in text
    if not missing and not retired:
        return []

    detail_parts: list[str] = []
    if retired:
        detail_parts.append("module declares M2RNN direct-MSL Path B retired")
    if missing:
        detail_parts.append("missing direct-MSL symbols: " + ", ".join(missing))
    return [
        Finding(
            code="m04_path_b_m2rnn_direct_msl_retired",
            severity="error",
            location=location,
            detail=(
                "; ".join(detail_parts)
                + "; Path B must keep the direct-MSL forward/backward VJP or "
                "the speed matrix baseline is invalid."
            ),
        )
    ]


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_receipt_path(raw: Any, *, report_path: Path, repo_root: Path) -> Path | None:
    if raw is None:
        return None
    candidate = Path(str(raw))
    candidates = [candidate] if candidate.is_absolute() else [
        repo_root / candidate,
        report_path.parent / candidate,
    ]
    for item in candidates:
        if item.exists():
            return item
    return candidates[0] if candidates else None


def _interesting_strings(value: Any, *, key: str | None = None) -> Iterable[str]:
    if isinstance(value, dict):
        for item_key, item_value in value.items():
            yield from _interesting_strings(item_value, key=str(item_key))
        return
    if isinstance(value, list):
        for item in value:
            yield from _interesting_strings(item, key=key)
        return
    if isinstance(value, str) and (key in REASON_KEYS if key else False):
        text = value.strip()
        if text:
            yield text


def _path_b_blocker(strings: Iterable[str]) -> str | None:
    for text in strings:
        if any(marker in text for marker in PATH_B_BLOCKER_MARKERS):
            return text
    return None


def _first_line(text: str, *, limit: int = 500) -> str:
    first = text.strip().splitlines()[0] if text.strip() else ""
    return first[:limit]


def _row_id(row: dict[str, Any]) -> str:
    return str(
        row.get("case_id")
        or "_".join(
            str(row.get(key) or "unknown")
            for key in ("dtype", "optimizer", "path")
        )
    )


def _rows(payload: Any) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        return []
    rows = payload.get("results")
    if not isinstance(rows, list):
        return []
    return [row for row in rows if isinstance(row, dict)]


def _v4_cells(payload: Any) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        return []
    cells = payload.get("cells")
    if not isinstance(cells, list):
        return []
    return [cell for cell in cells if isinstance(cell, dict)]


def _row_status(row: dict[str, Any]) -> str:
    return str(row.get("status") or "")


def _raw_status(raw_receipt: Any) -> str | None:
    if isinstance(raw_receipt, dict) and raw_receipt.get("status") is not None:
        return str(raw_receipt.get("status"))
    return None


def _path_c_claims_fused_train_block_runtime(raw_receipt: Any) -> bool:
    if not isinstance(raw_receipt, dict):
        return False
    training = raw_receipt.get("training")
    if not isinstance(training, dict):
        return False
    payloads = (
        training.get("fp8_path_c_training_route"),
        training.get("fp8_path_c_post_step_runtime_capture"),
    )
    for payload in payloads:
        if not isinstance(payload, dict):
            continue
        if bool(payload.get("fused_train_block_training_critical_path")):
            return True
        contract = payload.get("fused_train_block_training_runtime_contract")
        if isinstance(contract, dict) and bool(
            contract.get("training_critical_path_verified")
        ):
            return True
        path_c_fusion = payload.get("path_c_fusion")
        if isinstance(path_c_fusion, dict) and bool(
            path_c_fusion.get("runtime_uses_fused_train_block")
        ):
            return True
    return False


def _stepper_attached_path_c_runtime(raw_receipt: Any) -> bool:
    if not isinstance(raw_receipt, dict):
        return False
    training = raw_receipt.get("training")
    if not isinstance(training, dict):
        return False
    stepper_state = training.get("stepper_state")
    return isinstance(stepper_state, dict) and bool(
        stepper_state.get("path_c_training_runtime_installed")
    )


def _raw_receipt_for_row(
    row: dict[str, Any],
    *,
    report_path: Path,
    repo_root: Path,
) -> tuple[Path | None, Any | None]:
    path = _resolve_receipt_path(
        row.get("receipt_path"),
        report_path=report_path,
        repo_root=repo_root,
    )
    if path is None or not path.exists():
        return path, None
    try:
        return path, _load_json(path)
    except json.JSONDecodeError:
        return path, None


def _contract_for_v4_cell(payload: dict[str, Any], cell: dict[str, Any]) -> PathContract | None:
    shape = payload.get("shape")
    block = cell.get("block")
    if not block and isinstance(shape, dict):
        block = shape.get("block")
    path = cell.get("path")
    if not isinstance(block, str) or not isinstance(path, str):
        return None
    return CONTRACT_BY_KEY.get((f"v4.{block}", path))


def _v4_shape_path_e_uses_metal_kernel(payload: dict[str, Any]) -> bool:
    shape = payload.get("shape")
    if not isinstance(shape, dict):
        return False
    try:
        head_dim_k = int(shape.get("head_dim_k"))
        head_dim_v = int(shape.get("head_dim_v"))
    except (TypeError, ValueError):
        return False
    return head_dim_k % 32 == 0 and head_dim_v % 4 == 0


def _check_v4_matrix_report(payload: Any) -> list[Finding]:
    if not isinstance(payload, dict):
        return []
    cells = _v4_cells(payload)
    if not cells:
        return []
    findings: list[Finding] = []
    promotion = payload.get("promotion")
    shape = payload.get("shape")
    shape_block = shape.get("block") if isinstance(shape, dict) else None
    promoted_path = None
    promotion_applied = False
    if isinstance(promotion, dict):
        promoted_path = promotion.get("winning_path")
        promotion_applied = bool(promotion.get("promotion_applied"))

    for cell in cells:
        path = str(cell.get("path") or "")
        block = str(cell.get("block") or shape_block or "")
        location = f"v4.{block}.{path}"
        contract = _contract_for_v4_cell(payload, cell)
        if contract is None:
            findings.append(
                Finding(
                    code="v4_matrix_missing_path_contract",
                    severity="error",
                    location=location,
                    detail="v4 matrix cell has no explicit path contract",
                )
            )
            continue
        blocked = (
            contract.state in BLOCKED_CONTRACT_STATES
            or contract.benchmark_role in BLOCKED_BENCHMARK_ROLES
        )
        backend_available = bool(cell.get("backend_available"))
        if blocked and backend_available:
            findings.append(
                Finding(
                    code="v4_blocked_path_marked_available",
                    severity="error",
                    location=location,
                    detail=(
                        f"{contract.state}/{contract.benchmark_role} path "
                        "cannot be benchmark-available until its contract is updated"
                    ),
                )
            )
        if blocked and promotion_applied and promoted_path == path:
            findings.append(
                Finding(
                    code="v4_blocked_path_promoted",
                    severity="error",
                    location=location,
                    detail="matrix promotion selected a test-only or blocked candidate path",
                )
            )
        measured_path = cell.get("measured_path")
        fallback_used = bool(cell.get("fallback_used"))
        if backend_available and (fallback_used or measured_path not in (None, path)):
            findings.append(
                Finding(
                    code="v4_available_path_measured_fallback",
                    severity="error",
                    location=location,
                    detail=(
                        "matrix cell is backend_available but receipt says "
                        f"measured_path={measured_path!r}, fallback_used={fallback_used}"
                    ),
                )
            )
        if path == "path_e" and backend_available and not _v4_shape_path_e_uses_metal_kernel(payload):
            findings.append(
                Finding(
                    code="v4_path_e_ops_fallback_marked_available",
                    severity="error",
                    location=location,
                    detail=(
                        "Path E cell is marked backend_available for a shape "
                        "that cannot use the vendored Metal kernel "
                        "(requires head_dim_k%32==0 and head_dim_v%4==0)"
                    ),
                )
            )
    return findings


def check_matrix_report(report_path: Path, *, repo_root: Path = ROOT) -> list[Finding]:
    payload = _load_json(report_path)
    rows = _rows(payload)
    findings: list[Finding] = []
    findings.extend(_check_v4_matrix_report(payload))
    keyed: dict[tuple[str, str, str], dict[str, Any]] = {}
    ok_path_c_keys: set[tuple[str, str]] = set()

    for row in rows:
        path = str(row.get("path") or "")
        dtype = str(row.get("dtype") or "")
        optimizer = str(row.get("optimizer") or "")
        status = _row_status(row)
        keyed[(dtype, optimizer, path)] = row
        if path.startswith("path_c") and status == "ok":
            ok_path_c_keys.add((dtype, optimizer))

        receipt_path, raw_receipt = _raw_receipt_for_row(
            row,
            report_path=report_path,
            repo_root=repo_root,
        )
        raw_status = _raw_status(raw_receipt)
        if status == "ok" and raw_status not in (None, "ok"):
            findings.append(
                Finding(
                    code="aggregate_ok_raw_not_ok",
                    severity="error",
                    location=_row_id(row),
                    detail=(
                        f"aggregate row is ok but raw receipt {receipt_path} "
                        f"has status {raw_status!r}"
                    ),
                )
            )

        if path != "path_b":
            if (
                path.startswith("path_c")
                and status == "ok"
                and _path_c_claims_fused_train_block_runtime(raw_receipt)
                and not _stepper_attached_path_c_runtime(raw_receipt)
            ):
                findings.append(
                    Finding(
                        code="path_c_fused_runtime_claimed_but_not_attached",
                        severity="error",
                        location=_row_id(row),
                        detail=(
                            "raw receipt claims a fused Path C train-block "
                            "critical path, but CompiledPretrainingStep did not "
                            "record an attached path_c_training_runtime"
                        ),
                    )
                )
            continue
        raw_blocker = _path_b_blocker(_interesting_strings(raw_receipt))
        if raw_blocker is None:
            continue
        row_reason = str(row.get("pass_fail_reason") or "")
        blocker_summary = _first_line(raw_blocker)
        if status == "ok":
            findings.append(
                Finding(
                    code="retired_broken_baseline_marked_ok",
                    severity="error",
                    location=_row_id(row),
                    detail=(
                        "Path B baseline is marked ok even though raw receipt "
                        f"contains blocker: {blocker_summary}"
                    ),
                )
            )
        if blocker_summary not in row_reason and not any(
            marker in row_reason for marker in PATH_B_BLOCKER_MARKERS
        ):
            findings.append(
                Finding(
                    code="matrix_report_reason_masked",
                    severity="error",
                    location=_row_id(row),
                    detail=(
                        "aggregate pass_fail_reason does not preserve raw Path B blocker; "
                        f"raw blocker: {blocker_summary}; aggregate reason: "
                        f"{_first_line(row_reason)}"
                    ),
                )
            )

    for dtype, optimizer in sorted(ok_path_c_keys):
        baseline = keyed.get((dtype, optimizer, "path_b"))
        if baseline is None:
            findings.append(
                Finding(
                    code="path_c_without_runnable_path_b_baseline",
                    severity="error",
                    location=f"{dtype}_{optimizer}",
                    detail="Path C produced an ok row but no Path B baseline row exists.",
                )
            )
            continue
        if _row_status(baseline) not in {"ok", "not_applicable"}:
            findings.append(
                Finding(
                    code="path_c_without_runnable_path_b_baseline",
                    severity="error",
                    location=f"{dtype}_{optimizer}",
                    detail=(
                        "Path C produced an ok row but Path B baseline status is "
                        f"{_row_status(baseline)!r}; do not accept speed/default "
                        "comparisons until the baseline is runnable or explicitly retired."
                    ),
                )
            )
    return findings


def _print_findings(findings: Sequence[Finding], *, as_json: bool) -> None:
    if as_json:
        print(json.dumps([asdict(finding) for finding in findings], indent=2))
        return
    if not findings:
        print("path sanity guard: ok")
        return
    for finding in findings:
        print(
            f"{finding.severity.upper()} {finding.code} "
            f"{finding.location}: {finding.detail}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate declared path contracts and benchmark matrix reports so "
            "broken baselines cannot be accepted silently."
        )
    )
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument(
        "--matrix-report",
        type=Path,
        action="append",
        default=[],
        help="Matrix JSON report to validate. May be passed more than once.",
    )
    parser.add_argument(
        "--contracts-only",
        action="store_true",
        help="Only validate static path declarations and contracts.",
    )
    parser.add_argument("--json", action="store_true", help="Emit findings as JSON.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = args.repo_root.resolve()
    findings = check_path_contracts(repo_root)
    if not args.contracts_only:
        for report_path in args.matrix_report:
            findings.extend(
                check_matrix_report(report_path, repo_root=repo_root)
            )
    _print_findings(findings, as_json=bool(args.json))
    return 2 if any(finding.severity == "error" for finding in findings) else 0


if __name__ == "__main__":
    raise SystemExit(main())
