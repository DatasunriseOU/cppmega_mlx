#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path


_PYTHON_SUFFIX = ".py"
_RULE_ID = "MLX001"


@dataclass(frozen=True)
class Finding:
    path: Path
    line: int
    message: str

    def format(self) -> str:
        return f"{self.path}:{self.line}: {_RULE_ID} {self.message}"


class _MlxArrayScalarVisitor(ast.NodeVisitor):
    def __init__(self, path: Path) -> None:
        self._path = path
        self._mlx_aliases: set[str] = set()
        self.findings: list[Finding] = []

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            if alias.name == "mlx.core":
                self._mlx_aliases.add(alias.asname or "mlx.core")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module != "mlx":
            self.generic_visit(node)
            return
        for alias in node.names:
            if alias.name == "core":
                self._mlx_aliases.add(alias.asname or alias.name)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if self._is_mlx_array_call(node) and self._has_unsafe_scalar_literal_arg(node):
            self.findings.append(
                Finding(
                    self._path,
                    node.lineno,
                    "avoid mx.array(<python scalar literal>) without dtype; "
                    "keep the Python scalar or pass an explicit dtype",
                )
            )
        self.generic_visit(node)

    def _is_mlx_array_call(self, node: ast.Call) -> bool:
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr == "array":
            if isinstance(func.value, ast.Name):
                return func.value.id in self._mlx_aliases
            if (
                isinstance(func.value, ast.Attribute)
                and func.value.attr == "core"
                and isinstance(func.value.value, ast.Name)
                and func.value.value.id == "mlx"
            ):
                return "mlx.core" in self._mlx_aliases
        return False

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
    visitor = _MlxArrayScalarVisitor(path)
    visitor.visit(tree)
    return visitor.findings


def lint_paths(paths: Iterable[Path]) -> list[Finding]:
    findings: list[Finding] = []
    for path in _iter_python_files(paths):
        findings.extend(lint_file(path))
    return findings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Lint MLX scalar anti-patterns that can silently promote fp32.",
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
