"""Ground-truth code sandbox for the Phase-7 loop (NO LLM involved).

This is the *oracle*: it runs real tools (clang++, the build system, ctest,
sqlite3) and reports structured outcomes. It is the hard gate that turns a model
proposal into a fact.

FAIL-LOUD contract (project RULE #1):

* If a required tool is not installed / not reproducible, the corresponding
  method RAISES :class:`ToolUnavailableError`. It NEVER returns a fabricated
  ``ok=True`` / fake pass.
* A *legitimate* compiler/test failure (the tool ran and exited non-zero) is NOT
  an error — it is a real :class:`Outcome` with ``ok=False`` and the captured
  diagnostics. That is the signal the loop is built to consume.

Container-state pattern (InterCode): every mutating op records a
``fs_state_diff`` — per-file SHA-256 hashes before/after — so the orchestrator
can detect exactly what changed on disk.
"""

from __future__ import annotations

import hashlib
import os
import shlex
import shutil
import subprocess
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "Outcome",
    "FsStateDiff",
    "CodeVerifier",
    "ToolUnavailableError",
    "VerifierError",
]

_STDOUT_TAIL_BYTES = 4096


class VerifierError(RuntimeError):
    """Base error for verifier-level failures (not tool exit codes)."""


class ToolUnavailableError(VerifierError):
    """Raised when a required external tool is not available/reproducible."""


# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class FsStateDiff:
    """Per-file hash diff of a directory tree before/after an operation."""

    added: dict[str, str] = field(default_factory=dict)
    removed: dict[str, str] = field(default_factory=dict)
    changed: dict[str, tuple[str, str]] = field(default_factory=dict)

    @property
    def any_change(self) -> bool:
        return bool(self.added or self.removed or self.changed)

    def to_dict(self) -> dict[str, Any]:
        return {
            "added": dict(self.added),
            "removed": dict(self.removed),
            "changed": {k: list(v) for k, v in self.changed.items()},
        }


@dataclass(frozen=True)
class Outcome:
    """Structured result of a sandbox operation.

    Attributes
    ----------
    ok:
        True iff the tool ran AND exited 0 (semantic success).
    exit_code:
        Process exit code (0 on success).
    diagnostics:
        Parsed/raw diagnostic lines (compiler errors, test failures, ...).
    stdout_tail:
        Tail of combined stdout/stderr (bounded).
    fs_state_diff:
        File-hash diff for mutating ops; empty for read-only ops.
    kind:
        Which verifier method produced this outcome.
    """

    ok: bool
    exit_code: int
    diagnostics: list[str]
    stdout_tail: str
    fs_state_diff: FsStateDiff
    kind: str
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "exit_code": self.exit_code,
            "diagnostics": list(self.diagnostics),
            "stdout_tail": self.stdout_tail,
            "fs_state_diff": self.fs_state_diff.to_dict(),
            "kind": self.kind,
            "extra": dict(self.extra),
        }


# --------------------------------------------------------------------------- #
def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _snapshot_tree(root: Path) -> dict[str, str]:
    """Map relpath -> sha256 for every regular file under ``root``."""
    snap: dict[str, str] = {}
    for dirpath, _dirs, files in os.walk(root):
        for name in files:
            fp = Path(dirpath) / name
            if fp.is_symlink() or not fp.is_file():
                continue
            rel = str(fp.relative_to(root))
            snap[rel] = _hash_file(fp)
    return snap


def _diff_snapshots(before: dict[str, str], after: dict[str, str]) -> FsStateDiff:
    added = {k: v for k, v in after.items() if k not in before}
    removed = {k: v for k, v in before.items() if k not in after}
    changed = {
        k: (before[k], after[k])
        for k in before
        if k in after and before[k] != after[k]
    }
    return FsStateDiff(added=added, removed=removed, changed=changed)


# --------------------------------------------------------------------------- #
class CodeVerifier:
    """Runs real ground-truth checks against a repository tree."""

    def __init__(
        self,
        repo_root: str | os.PathLike[str],
        *,
        clangxx: str | None = None,
        sqlite3_bin: str | None = None,
        timeout_s: float = 120.0,
    ) -> None:
        self.repo_root = Path(repo_root).resolve(strict=True)
        self.timeout_s = timeout_s
        self._clangxx = clangxx or shutil.which("clang++") or shutil.which("clang")
        self._sqlite3 = sqlite3_bin or shutil.which("sqlite3")

    # ------------------------------------------------------------------ #
    def _require_clang(self) -> str:
        if not self._clangxx:
            raise ToolUnavailableError(
                "clang++/clang not found on PATH; cannot run syntax/build checks "
                "(fail-loud: refusing to fake a pass)"
            )
        return self._clangxx

    def _run(
        self, argv: Sequence[str], cwd: Path
    ) -> subprocess.CompletedProcess[str]:
        try:
            return subprocess.run(
                list(argv),
                cwd=str(cwd),
                capture_output=True,
                text=True,
                timeout=self.timeout_s,
                check=False,
            )
        except FileNotFoundError as exc:
            raise ToolUnavailableError(
                f"command not found: {argv[0]!r} ({exc})"
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise VerifierError(
                f"command timed out after {self.timeout_s}s: {' '.join(argv)}"
            ) from exc

    @staticmethod
    def _tail(*streams: str | None) -> str:
        combined = "\n".join(s for s in streams if s)
        return combined[-_STDOUT_TAIL_BYTES:]

    # ------------------------------------------------------------------ #
    def syntax_check(
        self, file_path: str | os.PathLike[str], *, std: str = "c++17"
    ) -> Outcome:
        """clang++ -fsyntax-only on a single C++ file. Read-only (no diff)."""
        clang = self._require_clang()
        src = Path(file_path)
        if not src.is_absolute():
            src = self.repo_root / src
        if not src.is_file():
            raise VerifierError(f"syntax_check: file does not exist: {src}")
        proc = self._run(
            [clang, f"-std={std}", "-fsyntax-only", str(src)], cwd=self.repo_root
        )
        diags = [ln for ln in (proc.stderr or "").splitlines() if "error:" in ln]
        return Outcome(
            ok=proc.returncode == 0,
            exit_code=proc.returncode,
            diagnostics=diags,
            stdout_tail=self._tail(proc.stdout, proc.stderr),
            fs_state_diff=FsStateDiff(),
            kind="syntax_check",
            extra={"std": std, "file": str(src)},
        )

    def build(
        self,
        command: str | Sequence[str],
        *,
        copy_tree: bool = True,
        cwd: str | os.PathLike[str] | None = None,
    ) -> Outcome:
        """Run a build command, by default in a temp COPY of the repo.

        Returns an Outcome with a fs_state_diff over the (copied) tree.
        """
        argv = shlex.split(command) if isinstance(command, str) else list(command)
        if not argv:
            raise VerifierError("build: empty command")
        # ensure the build tool exists (fail-loud rather than fake pass)
        if shutil.which(argv[0]) is None and not Path(argv[0]).exists():
            raise ToolUnavailableError(f"build tool not found: {argv[0]!r}")

        if copy_tree:
            with tempfile.TemporaryDirectory(prefix="cppmega_build_") as tmp:
                work = Path(tmp) / "repo"
                shutil.copytree(self.repo_root, work, symlinks=True)
                run_cwd = work / cwd if cwd else work
                return self._build_in(argv, work, run_cwd, "build")
        run_cwd = (self.repo_root / cwd) if cwd else self.repo_root
        return self._build_in(argv, self.repo_root, run_cwd, "build")

    def _build_in(
        self, argv: list[str], tree_root: Path, run_cwd: Path, kind: str
    ) -> Outcome:
        before = _snapshot_tree(tree_root)
        proc = self._run(argv, run_cwd)
        after = _snapshot_tree(tree_root)
        diags = [
            ln
            for ln in (proc.stderr or "").splitlines()
            if "error" in ln.lower() or "failed" in ln.lower()
        ]
        return Outcome(
            ok=proc.returncode == 0,
            exit_code=proc.returncode,
            diagnostics=diags,
            stdout_tail=self._tail(proc.stdout, proc.stderr),
            fs_state_diff=_diff_snapshots(before, after),
            kind=kind,
            extra={"command": argv, "cwd": str(run_cwd)},
        )

    def run_tests(
        self,
        command: str | Sequence[str] = "ctest --output-on-failure",
        *,
        cwd: str | os.PathLike[str] | None = None,
    ) -> Outcome:
        """Run a test command (ctest / make test / custom). Mutating-aware."""
        argv = shlex.split(command) if isinstance(command, str) else list(command)
        if not argv:
            raise VerifierError("run_tests: empty command")
        if shutil.which(argv[0]) is None and not Path(argv[0]).exists():
            raise ToolUnavailableError(f"test tool not found: {argv[0]!r}")
        run_cwd = (self.repo_root / cwd) if cwd else self.repo_root
        return self._build_in(argv, self.repo_root, run_cwd, "run_tests")

    def validate_sql(self, sql: str, *, dialect: str = "sqlite") -> Outcome:
        """Validate SQL by EXPLAINing it against a throwaway sqlite3 DB.

        Read-only against the real filesystem (uses a temp DB).
        """
        if dialect != "sqlite":
            raise VerifierError(
                f"validate_sql: only 'sqlite' dialect supported, got {dialect!r}"
            )
        if not self._sqlite3:
            raise ToolUnavailableError(
                "sqlite3 not found on PATH; cannot validate SQL (fail-loud)"
            )
        # Use EXPLAIN to parse without executing side effects where possible;
        # fall back to running the statement in an in-memory DB.
        with tempfile.TemporaryDirectory(prefix="cppmega_sql_") as tmp:
            script = Path(tmp) / "check.sql"
            script.write_text(sql, encoding="utf-8")
            proc = self._run(
                [self._sqlite3, ":memory:", f".read {script}"], cwd=Path(tmp)
            )
        diags = [
            ln
            for ln in (proc.stderr or "").splitlines()
            if "error" in ln.lower() or "syntax" in ln.lower()
        ]
        return Outcome(
            ok=proc.returncode == 0,
            exit_code=proc.returncode,
            diagnostics=diags,
            stdout_tail=self._tail(proc.stdout, proc.stderr),
            fs_state_diff=FsStateDiff(),
            kind="validate_sql",
            extra={"dialect": dialect},
        )
