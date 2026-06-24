"""Thin eval harness wiring the ground-truth verifier as the HARD GATE.

An :class:`EvalCase` describes a repo snapshot + a task + an oracle command. The
:class:`EvalRunner` executes the candidate (via the orchestrator/policy) and then
runs the oracle command through :class:`~cppmega_mlx.runtime.code_verifier.CodeVerifier`
as the *non-negotiable* pass/fail gate (compile / test / exit-code).

Any LLM-judge signal is ADVISORY ONLY and is recorded *behind* the hard gate —
a case can never be marked passed unless the compiler/test oracle exits 0
(project RULE #1: no fake pass).

Dataset adapters (Defects4C / ComBench / Multi-SWE-bench C-C++ / InterCode) are
provided as *interface stubs* that map each dataset's row shape onto
:class:`EvalCase`. No datasets are downloaded here — only the shape + verifier
wiring. Each adapter RAISES ``NotImplementedError`` for the actual row loading so
nobody mistakes a stub for a working loader.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from cppmega_mlx.runtime.code_verifier import CodeVerifier, Outcome

__all__ = [
    "EvalCase",
    "EvalResult",
    "EvalRunner",
    "DatasetAdapter",
    "Defects4CAdapter",
    "ComBenchAdapter",
    "MultiSWEBenchCppAdapter",
    "InterCodeAdapter",
]

# A candidate maps a working repo path -> nothing (it mutates the repo in place).
Candidate = Callable[[Path], None]
# An advisory judge maps (case, oracle_outcome) -> float score in [0,1].
Judge = Callable[["EvalCase", Outcome], float]


@dataclass(frozen=True)
class EvalCase:
    """One evaluation case.

    Attributes
    ----------
    name:
        Stable identifier.
    repo_snapshot:
        Path to a directory tree representing the starting repo state.
    task:
        Natural-language task description (for the policy / record only).
    oracle_cmd:
        Command run by the verifier as the hard gate (e.g. a test command).
    oracle_kind:
        ``"test"`` | ``"build"`` | ``"syntax"`` | ``"sql"`` — selects the
        verifier method used as the gate.
    oracle_target:
        For ``syntax``: the file to check. For ``sql``: the SQL text. Ignored for
        build/test (which use ``oracle_cmd``).
    """

    name: str
    repo_snapshot: Path
    task: str
    oracle_cmd: str = ""
    oracle_kind: str = "test"
    oracle_target: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvalResult:
    name: str
    passed: bool
    oracle_outcome: dict[str, Any]
    advisory_score: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "oracle_outcome": self.oracle_outcome,
            "advisory_score": self.advisory_score,
        }


class EvalRunner:
    """Runs an :class:`EvalCase` through a candidate then the HARD oracle gate."""

    def __init__(self, *, timeout_s: float = 120.0) -> None:
        self.timeout_s = timeout_s

    def _oracle(self, verifier: CodeVerifier, case: EvalCase) -> Outcome:
        kind = case.oracle_kind
        if kind == "test":
            return verifier.run_tests(case.oracle_cmd)
        if kind == "build":
            return verifier.build(case.oracle_cmd, copy_tree=False)
        if kind == "syntax":
            if not case.oracle_target:
                raise ValueError(
                    f"case {case.name!r}: syntax oracle needs oracle_target"
                )
            return verifier.syntax_check(case.oracle_target)
        if kind == "sql":
            return verifier.validate_sql(case.oracle_target)
        raise ValueError(f"case {case.name!r}: unknown oracle_kind {kind!r}")

    def run_case(
        self,
        case: EvalCase,
        candidate: Candidate | None = None,
        *,
        judge: Judge | None = None,
    ) -> EvalResult:
        """Execute one case in an isolated copy; oracle exit-code is the gate."""
        snap = Path(case.repo_snapshot).resolve(strict=True)
        with tempfile.TemporaryDirectory(prefix=f"cppmega_eval_{case.name}_") as tmp:
            work = Path(tmp) / "repo"
            shutil.copytree(snap, work, symlinks=True)
            if candidate is not None:
                candidate(work)  # mutate the working copy (apply the fix)
            verifier = CodeVerifier(work, timeout_s=self.timeout_s)
            oracle = self._oracle(verifier, case)

            advisory: float | None = None
            if judge is not None:
                advisory = float(judge(case, oracle))  # advisory only

            # HARD GATE: pass iff the oracle exited 0. Judge cannot override.
            return EvalResult(
                name=case.name,
                passed=oracle.ok,
                oracle_outcome=oracle.to_dict(),
                advisory_score=advisory,
            )

    def run_suite(
        self,
        cases: list[EvalCase],
        candidates: dict[str, Candidate] | None = None,
        *,
        judge: Judge | None = None,
    ) -> list[EvalResult]:
        results: list[EvalResult] = []
        for case in cases:
            cand = (candidates or {}).get(case.name)
            results.append(self.run_case(case, cand, judge=judge))
        return results


# --------------------------------------------------------------------------- #
# Dataset adapters (interface stubs — verifier wiring only, no downloads).
# --------------------------------------------------------------------------- #
class DatasetAdapter:
    """Base adapter: maps a dataset row dict to an :class:`EvalCase`."""

    name = "base"

    def to_case(self, row: dict[str, Any]) -> EvalCase:  # pragma: no cover - stub
        raise NotImplementedError(
            f"{type(self).__name__}.to_case is a stub; wire the real row schema"
        )

    def load_rows(self, root: str | os.PathLike[str]) -> list[dict[str, Any]]:
        raise NotImplementedError(
            f"{type(self).__name__}.load_rows: no dataset downloaded in base infra"
        )


class Defects4CAdapter(DatasetAdapter):
    """Defects4C (C/C++ real-world bugs): row -> compile/test oracle case."""

    name = "defects4c"

    def to_case(self, row: dict[str, Any]) -> EvalCase:
        return EvalCase(
            name=f"defects4c-{row['bug_id']}",
            repo_snapshot=Path(row["buggy_snapshot"]),
            task=row.get("issue_text", "fix the defect"),
            oracle_cmd=row.get("test_cmd", "ctest --output-on-failure"),
            oracle_kind="test",
            metadata={"dataset": self.name, "raw": row},
        )


class ComBenchAdapter(DatasetAdapter):
    """ComBench (compilation benchmark): row -> build/syntax oracle case."""

    name = "combench"

    def to_case(self, row: dict[str, Any]) -> EvalCase:
        return EvalCase(
            name=f"combench-{row['id']}",
            repo_snapshot=Path(row["snapshot"]),
            task=row.get("task", "make it compile"),
            oracle_cmd=row.get("build_cmd", ""),
            oracle_kind=row.get("oracle_kind", "build"),
            oracle_target=row.get("target_file", ""),
            metadata={"dataset": self.name, "raw": row},
        )


class MultiSWEBenchCppAdapter(DatasetAdapter):
    """Multi-SWE-bench (C/C++ slice): row -> test oracle case."""

    name = "multi-swe-bench-cpp"

    def to_case(self, row: dict[str, Any]) -> EvalCase:
        return EvalCase(
            name=f"mswe-{row['instance_id']}",
            repo_snapshot=Path(row["repo_snapshot"]),
            task=row.get("problem_statement", ""),
            oracle_cmd=row.get("test_cmd", "ctest --output-on-failure"),
            oracle_kind="test",
            metadata={"dataset": self.name, "raw": row},
        )


class InterCodeAdapter(DatasetAdapter):
    """InterCode (interactive, container-state): row -> command oracle case."""

    name = "intercode"

    def to_case(self, row: dict[str, Any]) -> EvalCase:
        return EvalCase(
            name=f"intercode-{row['id']}",
            repo_snapshot=Path(row["snapshot"]),
            task=row.get("query", ""),
            oracle_cmd=row.get("gold_cmd", ""),
            oracle_kind=row.get("oracle_kind", "test"),
            oracle_target=row.get("oracle_target", ""),
            metadata={"dataset": self.name, "raw": row},
        )
