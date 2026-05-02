#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path


_PYTHON_SUFFIX = ".py"
_OWNED_METAL_KERNEL_PATH = ("cppmega_mlx", "kernels", "metal_ops.py")
_AUTODIFF_NAMES = {"grad", "jvp", "value_and_grad", "vjp"}
_CUSTOM_GRADIENT_NAMES = {"jvp", "vjp"}
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


@dataclass(frozen=True)
class Finding:
    path: Path
    line: int
    rule: str
    message: str

    def format(self) -> str:
        return f"{self.path}:{self.line}: {self.rule} {self.message}"


class _MlxLintVisitor(ast.NodeVisitor):
    def __init__(self, path: Path) -> None:
        self._path = path
        self._mlx_core_aliases: set[str] = set()
        self._mlx_fast_aliases: set[str] = set()
        self._mlx_nn_aliases: set[str] = set()
        self._metal_kernel_names: set[str] = set()
        self._custom_function_names: set[str] = set()
        self._autodiff_names: set[str] = set()
        self._metal_kernel_lines: list[int] = []
        self._uses_autodiff = False
        self._uses_custom_function = False
        self._defines_custom_gradient = False
        self.findings: list[Finding] = []

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            if alias.name == "mlx.core":
                self._mlx_core_aliases.add(alias.asname or "mlx.core")
            elif alias.name == "mlx.core.fast":
                self._mlx_fast_aliases.add(alias.asname or "mlx.core.fast")
            elif alias.name == "mlx.nn":
                self._mlx_nn_aliases.add(alias.asname or "mlx.nn")
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
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_decorators(node.decorator_list)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
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
            if not self._is_owned_metal_kernel_module():
                self.findings.append(
                    Finding(
                        self._path,
                        node.lineno,
                        "MLX002",
                        "construct custom mx.fast.metal_kernel only in "
                        "cppmega_mlx/kernels/metal_ops.py behind the owned "
                        "fallback/parity/VJP/JVP/profile-evidence policy seam",
                    )
                )
        if self._is_autodiff_call(node.func):
            self._uses_autodiff = True
        if self._is_custom_function_marker(node.func):
            self._uses_custom_function = True
        if self._is_custom_gradient_marker(node.func):
            self._defines_custom_gradient = True
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

    def _is_owned_metal_kernel_module(self) -> bool:
        return self._path.parts[-len(_OWNED_METAL_KERNEL_PATH) :] == _OWNED_METAL_KERNEL_PATH

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
    visitor = _MlxLintVisitor(path)
    visitor.visit(tree)
    visitor.finalize()
    return visitor.findings


def lint_paths(paths: Iterable[Path]) -> list[Finding]:
    findings: list[Finding] = []
    for path in _iter_python_files(paths):
        findings.extend(lint_file(path))
    return findings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Lint MLX anti-patterns for scalar dtype, custom Metal, and timing guardrails.",
    )
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="Python files or directories to scan.",
    )
    args = parser.parse_args(argv)

    findings = lint_paths(args.paths)
    for finding in findings:
        print(finding.format())
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
