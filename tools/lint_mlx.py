#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
import re


_PYTHON_SUFFIX = ".py"
_DIRECT_MSL_LEGACY_DEBUG_PATHS = {
    ("cppmega_mlx", "nn", "_tilelang", "fp8_msl_kernels.py"),
    ("cppmega_mlx", "nn", "_tilelang", "m2rnn.py"),
    ("cppmega_mlx", "nn", "_tilelang", "mamba3.py"),
    ("cppmega_mlx", "nn", "_tilelang", "sparse_mla.py"),
    ("cppmega_mlx", "nn", "_tilelang", "sparse_mla_blockscaled.py"),
    ("cppmega_mlx", "nn", "_tilelang", "topk_selector.py"),
}
_DIRECT_MSL_MARKER = re.compile(
    r"\b(legacy|debug|path\s+b|direct[- ]msl|raw[- ]msl)\b",
    re.IGNORECASE,
)
_AUTODIFF_NAMES = {"grad", "jvp", "value_and_grad", "vjp"}
_CUSTOM_GRADIENT_NAMES = {"jvp", "vjp"}
_MONKEYPATCH_METHOD_NAMES = {
    "setattr",
    "delattr",
    "setenv",
    "delenv",
    "setitem",
    "delitem",
    "syspath_prepend",
    "chdir",
    "context",
}
_THROUGHPUT_COMPILE_TIMING_NAMES = {
    "compile_time_s",
    "compile_profile",
    "first_call_profile",
    "first_call_seconds",
    "first_call_time_s",
}
_COMPILE_FROM_STEADY_TIMING_NAMES = {
    "measured_profile",
    "mean_step_s",
    "mean_step_time_s",
    "step_times",
    "step_times_s",
    "steady_times",
    "tokens_per_second",
    "total_measured_wall_time_s",
    "warmup_step_times_s",
    "warmup_times",
}
_FORBIDDEN_NATIVE_TVM_FFI_MODULES = {
    "_tilelang_mlx_tvm_ffi",
    "tilelang.contrib.mlx_tvm_ffi",
    "tilelang.jit.adapter._mlx_tvm_ffi",
}


@dataclass(frozen=True)
class Finding:
    path: Path
    line: int
    rule: str
    message: str

    def format(self) -> str:
        return f"{self.path}:{self.line}: {self.rule} {self.message}"


class _MlxLintVisitor(ast.NodeVisitor):
    def __init__(self, path: Path, source: str) -> None:
        self._path = path
        self._source = source
        self._mlx_core_aliases: set[str] = set()
        self._mlx_fast_aliases: set[str] = set()
        self._mlx_nn_aliases: set[str] = set()
        self._msl_transform_aliases: set[str] = set()
        self._msl_dispatch_names: set[str] = set()
        self._metal_kernel_names: set[str] = set()
        self._custom_function_names: set[str] = set()
        self._autodiff_names: set[str] = set()
        self._importlib_aliases: set[str] = {"importlib"}
        self._import_module_names: set[str] = set()
        self._monkeypatch_aliases: set[str] = {"monkeypatch"}
        self._mock_patch_names: set[str] = set()
        self._metal_kernel_lines: list[int] = []
        self._uses_autodiff = False
        self._uses_custom_function = False
        self._defines_custom_gradient = False
        self.findings: list[Finding] = []

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            imported_name = alias.asname or alias.name
            if alias.name == "mlx.core":
                self._mlx_core_aliases.add(imported_name)
            elif alias.name == "mlx.core.fast":
                self._mlx_fast_aliases.add(imported_name)
            elif alias.name == "mlx.nn":
                self._mlx_nn_aliases.add(imported_name)
            elif alias.name == "cppmega_mlx.nn._tilelang._msl_transform":
                self._msl_transform_aliases.add(imported_name)
            elif alias.name == "importlib":
                self._importlib_aliases.add(imported_name)
            if self._is_production_module() and (
                alias.name == "pytest"
                or alias.name == "unittest.mock"
                or alias.name.startswith("unittest.mock.")
            ):
                self._add_monkeypatch_finding(node.lineno)
            if (
                self._is_production_module()
                and self._is_forbidden_native_tvm_ffi_module(alias.name)
            ):
                self._add_native_tvm_ffi_import_finding(node.lineno)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        for alias in node.names:
            imported_name = alias.asname or alias.name
            if node.module == "mlx":
                if alias.name == "core":
                    self._mlx_core_aliases.add(imported_name)
                elif alias.name == "nn":
                    self._mlx_nn_aliases.add(imported_name)
            elif node.module == "mlx.core":
                if alias.name == "fast":
                    self._mlx_fast_aliases.add(imported_name)
                elif alias.name == "custom_function":
                    self._custom_function_names.add(imported_name)
                elif alias.name in _AUTODIFF_NAMES:
                    self._autodiff_names.add(imported_name)
            elif node.module == "mlx.core.fast" and alias.name == "metal_kernel":
                self._metal_kernel_names.add(imported_name)
            elif node.module == "mlx.nn" and alias.name == "value_and_grad":
                self._autodiff_names.add(imported_name)
            elif (
                node.module == "cppmega_mlx.nn._tilelang"
                and alias.name == "_msl_transform"
            ):
                self._msl_transform_aliases.add(imported_name)
            elif node.module == "cppmega_mlx.nn._tilelang._msl_transform":
                if alias.name == "dispatch":
                    self._msl_dispatch_names.add(imported_name)
            elif node.module == "importlib" and alias.name == "import_module":
                self._import_module_names.add(imported_name)
            if self._is_production_module() and (
                node.module == "pytest"
                or node.module == "unittest.mock"
                or node.module == "mock"
            ):
                self._add_monkeypatch_finding(node.lineno)
                if alias.name == "patch":
                    self._mock_patch_names.add(imported_name)
            if (
                self._is_production_module()
                and (
                    self._is_forbidden_native_tvm_ffi_module(node.module)
                    or self._is_forbidden_native_tvm_ffi_module(
                        f"{node.module}.{alias.name}" if node.module else alias.name
                    )
                )
            ):
                self._add_native_tvm_ffi_import_finding(node.lineno)
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._check_monkeypatch_args(node)
        self._visit_decorators(node.decorator_list)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._check_monkeypatch_args(node)
        self._visit_decorators(node.decorator_list)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if self._is_mlx_array_call(node) and self._has_unsafe_scalar_literal_arg(node):
            self.findings.append(
                Finding(
                    self._path,
                    node.lineno,
                    "MLX001",
                    "avoid mx.array(<python scalar literal>) without dtype; "
                    "keep the Python scalar or pass an explicit dtype",
                )
            )
        if self._is_metal_kernel_call(node.func):
            self._metal_kernel_lines.append(node.lineno)
            if not self._is_direct_msl_allowed_module():
                self.findings.append(
                    Finding(
                        self._path,
                        node.lineno,
                        "MLX002",
                        "construct raw mx.fast.metal_kernel only in explicitly "
                        "allowlisted legacy/debug direct-MSL modules; new "
                        "production paths must use native TileLang/TVM-FFI "
                        "or an owned fail-closed wrapper",
                    )
                )
        if self._is_msl_transform_dispatch_call(node.func):
            if not self._is_direct_msl_allowed_module():
                self.findings.append(
                    Finding(
                        self._path,
                        node.lineno,
                        "MLX005",
                        "call _msl_transform.dispatch only from explicitly "
                        "allowlisted legacy/debug direct-MSL modules; new "
                        "production paths must use native TileLang/TVM-FFI",
                    )
                )
        if self._is_autodiff_call(node.func):
            self._uses_autodiff = True
        if self._is_custom_function_marker(node.func):
            self._uses_custom_function = True
        if self._is_custom_gradient_marker(node.func):
            self._defines_custom_gradient = True
        if self._is_production_monkeypatch_call(node.func):
            self._add_monkeypatch_finding(node.lineno)
        if self._is_native_tvm_ffi_dynamic_import_call(node):
            self._add_native_tvm_ffi_import_finding(node.lineno)
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        self._check_timing_assignment(node.targets, node.value, node.lineno)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if node.value is not None:
            self._check_timing_assignment([node.target], node.value, node.lineno)
        self.generic_visit(node)

    def finalize(self) -> None:
        if (
            self._metal_kernel_lines
            and self._uses_autodiff
            and not (self._uses_custom_function and self._defines_custom_gradient)
        ):
            self.findings.append(
                Finding(
                    self._path,
                    self._metal_kernel_lines[0],
                    "MLX003",
                    "differentiated custom Metal kernels require "
                    "@mx.custom_function plus an explicit .vjp or .jvp before "
                    "they can enter training",
                )
            )

    def _visit_decorators(self, decorators: list[ast.expr]) -> None:
        for decorator in decorators:
            target = decorator.func if isinstance(decorator, ast.Call) else decorator
            if self._is_custom_function_marker(target):
                self._uses_custom_function = True
            if self._is_custom_gradient_marker(target):
                self._defines_custom_gradient = True

    def _is_mlx_array_call(self, node: ast.Call) -> bool:
        chain = _attribute_chain(node.func)
        if chain is None or chain[-1] != "array":
            return False
        return self._is_mlx_core_chain(chain[:-1])

    @staticmethod
    def _has_unsafe_scalar_literal_arg(node: ast.Call) -> bool:
        if not node.args:
            return False
        if any(keyword.arg == "dtype" for keyword in node.keywords):
            return False
        first_arg = node.args[0]
        return (
            isinstance(first_arg, ast.Constant)
            and isinstance(first_arg.value, (bool, int, float, complex))
        )

    def _is_metal_kernel_call(self, func: ast.expr) -> bool:
        if isinstance(func, ast.Name):
            return func.id in self._metal_kernel_names
        chain = _attribute_chain(func)
        if chain is None or chain[-1] != "metal_kernel":
            return False
        return self._is_mlx_fast_chain(chain[:-1])

    def _is_msl_transform_dispatch_call(self, func: ast.expr) -> bool:
        if isinstance(func, ast.Name):
            return func.id in self._msl_dispatch_names
        chain = _attribute_chain(func)
        if chain is None or chain[-1] != "dispatch":
            return False
        owner = chain[:-1]
        if len(owner) == 1 and owner[0] in self._msl_transform_aliases:
            return True
        return owner == ("cppmega_mlx", "nn", "_tilelang", "_msl_transform")

    def _is_autodiff_call(self, func: ast.expr) -> bool:
        if isinstance(func, ast.Name):
            return func.id in self._autodiff_names
        chain = _attribute_chain(func)
        if chain is None or chain[-1] not in _AUTODIFF_NAMES:
            return False
        owner = chain[:-1]
        return self._is_mlx_core_chain(owner) or self._is_mlx_nn_chain(owner)

    def _is_custom_function_marker(self, target: ast.expr) -> bool:
        if isinstance(target, ast.Name):
            return target.id in self._custom_function_names
        chain = _attribute_chain(target)
        if chain is None or chain[-1] != "custom_function":
            return False
        return self._is_mlx_core_chain(chain[:-1])

    def _is_custom_gradient_marker(self, target: ast.expr) -> bool:
        chain = _attribute_chain(target)
        if chain is None or chain[-1] not in _CUSTOM_GRADIENT_NAMES:
            return False
        return not self._is_mlx_core_chain(chain[:-1])

    def _is_mlx_core_chain(self, chain: tuple[str, ...]) -> bool:
        if len(chain) == 1 and chain[0] in self._mlx_core_aliases:
            return True
        return chain == ("mlx", "core")

    def _is_mlx_fast_chain(self, chain: tuple[str, ...]) -> bool:
        if len(chain) == 1 and chain[0] in self._mlx_fast_aliases:
            return True
        if len(chain) == 2 and chain[0] in self._mlx_core_aliases and chain[1] == "fast":
            return True
        return chain == ("mlx", "core", "fast")

    def _is_mlx_nn_chain(self, chain: tuple[str, ...]) -> bool:
        if len(chain) == 1 and chain[0] in self._mlx_nn_aliases:
            return True
        return chain == ("mlx", "nn")

    def _is_direct_msl_allowed_module(self) -> bool:
        return (
            _path_matches_any_tail(self._path, _DIRECT_MSL_LEGACY_DEBUG_PATHS)
            and _DIRECT_MSL_MARKER.search(self._source) is not None
        )

    def _is_production_module(self) -> bool:
        return "cppmega_mlx" in self._path.parts

    @staticmethod
    def _is_forbidden_native_tvm_ffi_module(module: str | None) -> bool:
        if module is None:
            return False
        return any(
            module == forbidden or module.startswith(f"{forbidden}.")
            for forbidden in _FORBIDDEN_NATIVE_TVM_FFI_MODULES
        )

    def _is_native_tvm_ffi_dynamic_import_call(self, node: ast.Call) -> bool:
        if not self._is_production_module() or not node.args:
            return False
        module_arg = node.args[0]
        if not isinstance(module_arg, ast.Constant) or not isinstance(module_arg.value, str):
            return False
        func = node.func
        if isinstance(func, ast.Name):
            is_import_call = func.id == "__import__" or func.id in self._import_module_names
        else:
            chain = _attribute_chain(func)
            is_import_call = (
                chain is not None
                and len(chain) == 2
                and chain[0] in self._importlib_aliases
                and chain[1] == "import_module"
            )
        return is_import_call and self._is_forbidden_native_tvm_ffi_module(module_arg.value)

    def _check_monkeypatch_args(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> None:
        if not self._is_production_module():
            return
        args = [
            *node.args.posonlyargs,
            *node.args.args,
            *node.args.kwonlyargs,
        ]
        if node.args.vararg is not None:
            args.append(node.args.vararg)
        if node.args.kwarg is not None:
            args.append(node.args.kwarg)
        if any(arg.arg == "monkeypatch" for arg in args):
            self._add_monkeypatch_finding(node.lineno)

    def _is_production_monkeypatch_call(self, func: ast.expr) -> bool:
        if not self._is_production_module():
            return False
        if isinstance(func, ast.Name):
            return func.id in self._mock_patch_names
        chain = _attribute_chain(func)
        if chain is None:
            return False
        if chain[0] in self._monkeypatch_aliases and chain[-1] in _MONKEYPATCH_METHOD_NAMES:
            return True
        return chain in {
            ("unittest", "mock", "patch"),
            ("mock", "patch"),
        }

    def _add_monkeypatch_finding(self, line: int) -> None:
        self.findings.append(
            Finding(
                self._path,
                line,
                "MLX006",
                "do not use pytest monkeypatch or mock.patch patterns in "
                "production code; expose explicit fail-closed seams instead",
            )
        )

    def _add_native_tvm_ffi_import_finding(self, line: int) -> None:
        self.findings.append(
            Finding(
                self._path,
                line,
                "MLX007",
                "production code must not import the native MLX TVM-FFI bridge "
                "directly; call compiled TileLang kernels through the standard "
                "tilelang -> tvm -> tvm-ffi adapter",
            )
        )

    def _check_timing_assignment(
        self,
        targets: Iterable[ast.expr],
        value: ast.expr,
        line: int,
    ) -> None:
        assigned_names = {name for target in targets for name in _target_names(target)}
        referenced_names = _referenced_names(value)
        if (
            "tokens_per_second" in assigned_names
            and referenced_names & _THROUGHPUT_COMPILE_TIMING_NAMES
        ):
            self.findings.append(
                Finding(
                    self._path,
                    line,
                    "MLX004",
                    "compute tokens_per_second only from steady measured steps; "
                    "keep first-call/compile timing as separate receipt fields",
                )
            )
        if "compile_time_s" in assigned_names and referenced_names & _COMPILE_FROM_STEADY_TIMING_NAMES:
            self.findings.append(
                Finding(
                    self._path,
                    line,
                    "MLX004",
                    "compile_time_s must not be derived from warmup or steady-state "
                    "step timings",
                )
            )


def _attribute_chain(node: ast.expr) -> tuple[str, ...] | None:
    if isinstance(node, ast.Name):
        return (node.id,)
    if isinstance(node, ast.Attribute):
        parent = _attribute_chain(node.value)
        if parent is None:
            return None
        return (*parent, node.attr)
    return None


def _target_names(node: ast.expr) -> set[str]:
    if isinstance(node, ast.Name):
        return {node.id}
    if isinstance(node, ast.Tuple | ast.List):
        return {name for item in node.elts for name in _target_names(item)}
    if (
        isinstance(node, ast.Subscript)
        and isinstance(node.slice, ast.Constant)
        and isinstance(node.slice.value, str)
    ):
        return {node.slice.value}
    return set()


def _referenced_names(node: ast.AST) -> set[str]:
    names: set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Name):
            names.add(child.id)
        elif isinstance(child, ast.Attribute):
            names.add(child.attr)
        elif isinstance(child, ast.Constant) and isinstance(child.value, str):
            names.add(child.value)
    return names


def _path_matches_tail(path: Path, tail: tuple[str, ...]) -> bool:
    return path.parts[-len(tail) :] == tail


def _path_matches_any_tail(path: Path, tails: Iterable[tuple[str, ...]]) -> bool:
    return any(_path_matches_tail(path, tail) for tail in tails)


def _iter_python_files(paths: Iterable[Path]) -> Iterable[Path]:
    for path in paths:
        if path.is_dir():
            yield from sorted(
                child
                for child in path.rglob(f"*{_PYTHON_SUFFIX}")
                if child.is_file()
            )
        elif path.is_file() and path.suffix == _PYTHON_SUFFIX:
            yield path


def lint_file(path: Path) -> list[Finding]:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    visitor = _MlxLintVisitor(path, source)
    visitor.visit(tree)
    visitor.finalize()
    return visitor.findings


def lint_paths(paths: Iterable[Path], *, select: set[str] | None = None) -> list[Finding]:
    findings: list[Finding] = []
    for path in _iter_python_files(paths):
        findings.extend(
            finding
            for finding in lint_file(path)
            if select is None or finding.rule in select
        )
    return findings


def _parse_select(values: Iterable[str]) -> set[str] | None:
    rules = {
        rule.strip().upper()
        for value in values
        for rule in value.split(",")
        if rule.strip()
    }
    return rules or None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Lint MLX anti-patterns for scalar dtype, custom Metal, and timing guardrails.",
    )
    parser.add_argument(
        "--select",
        action="append",
        default=[],
        help="Comma-separated rule ids to report, for example MLX002,MLX005.",
    )
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="Python files or directories to scan.",
    )
    args = parser.parse_args(argv)

    findings = lint_paths(args.paths, select=_parse_select(args.select))
    for finding in findings:
        print(finding.format())
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
