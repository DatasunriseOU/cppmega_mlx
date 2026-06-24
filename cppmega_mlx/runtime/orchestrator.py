"""Deterministic Phase-7 agentic loop (the loop itself is NOT the LLM).

Flow per step::

    event -> load repo state -> policy proposes ToolCall (intent)
          -> ToolRouter validates (typed-action safety gate)
          -> CodeVerifier executes (ground-truth sandbox)
          -> structured observation fed back to the policy

The policy is an injected callable: ``policy(state, history) -> intent_text``.
The orchestrator never trusts the policy — every proposal passes through the
typed-action validation gate before the verifier touches the filesystem.

A :class:`StubPolicy` is provided so the whole loop is runnable + testable
without any model weights.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from cppmega_mlx.inference.tool_router_schema import ToolRouter
from cppmega_mlx.inference.typed_actions import ActionKind, ToolCall
from cppmega_mlx.runtime.code_verifier import CodeVerifier, Outcome

__all__ = [
    "RepoState",
    "Observation",
    "Step",
    "Orchestrator",
    "StubPolicy",
    "Policy",
]

# A policy maps (state, history) to raw model text proposing the next tool call.
Policy = Callable[["RepoState", "list[Step]"], str]


@dataclass(frozen=True)
class RepoState:
    """Minimal loaded view of the repo at the start of a step."""

    repo_root: Path
    files: tuple[str, ...]

    @classmethod
    def load(cls, repo_root: str | os.PathLike[str]) -> "RepoState":
        root = Path(repo_root).resolve(strict=True)
        files = tuple(
            sorted(
                str(p.relative_to(root))
                for p in root.rglob("*")
                if p.is_file() and not p.is_symlink()
            )
        )
        return cls(repo_root=root, files=files)


@dataclass(frozen=True)
class Observation:
    """What the loop hands back to the policy after executing a tool call."""

    tool_call: dict[str, Any]
    outcome: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {"tool_call": self.tool_call, "outcome": self.outcome}


@dataclass
class Step:
    """One iteration of the loop."""

    intent_text: str
    tool_call: ToolCall | None = None
    observation: Observation | None = None
    error: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


class StubPolicy:
    """A fixed-script policy for testing the loop without a model.

    Yields the supplied raw model-text proposals in order, then STOP.
    """

    def __init__(self, scripted_texts: list[str]) -> None:
        self._scripted = list(scripted_texts)
        self._i = 0

    def __call__(self, state: RepoState, history: list[Step]) -> str:
        if self._i < len(self._scripted):
            text = self._scripted[self._i]
            self._i += 1
            return text
        return '{"kind": "stop", "args": {}}'


class Orchestrator:
    """Drives the deterministic event -> validate -> execute -> observe loop."""

    def __init__(
        self,
        repo_root: str | os.PathLike[str],
        policy: Policy,
        *,
        verifier: CodeVerifier | None = None,
        max_steps: int = 16,
    ) -> None:
        self.repo_root = Path(repo_root).resolve(strict=True)
        self.policy = policy
        self.router = ToolRouter(str(self.repo_root))
        self.verifier = verifier or CodeVerifier(self.repo_root)
        self.max_steps = max_steps

    # ------------------------------------------------------------------ #
    def _execute(self, call: ToolCall) -> Outcome:
        """Dispatch a validated ToolCall to the ground-truth verifier."""
        kind = call.kind
        args = call.args
        if kind is ActionKind.READ_FILE:
            # READ_FILE is satisfied by a syntax-free existence + content read;
            # we surface it through the verifier as a syntax check when it is a
            # C/C++ source, else a plain read outcome.
            path = call.resolved_path("path", self.repo_root)
            if path.suffix in {".cc", ".cpp", ".cxx", ".h", ".hpp", ".c"}:
                return self.verifier.syntax_check(path)
            return self._read_outcome(path)
        if kind is ActionKind.RUN_BUILD:
            return self.verifier.build(args["command"], cwd=args.get("cwd"))
        if kind is ActionKind.RUN_TEST:
            return self.verifier.run_tests(args["command"], cwd=args.get("cwd"))
        if kind is ActionKind.VALIDATE_SQL:
            return self.verifier.validate_sql(
                args["sql"], dialect=args.get("dialect", "sqlite")
            )
        # INSPECT_SYMBOL / GET_DEP_BLOCKS / QUERY_CMAKE / APPLY_PATCH are static
        # analyses owned by other tracks; here they resolve their path safely and
        # return a structured "not-executed-by-verifier" outcome (fail-loud if a
        # required path escapes is already handled upstream by ToolCall).
        raise NotImplementedError(
            f"orchestrator._execute: kind {kind.value!r} has no verifier binding "
            "in this base-independent infra (owned by analysis tracks)"
        )

    @staticmethod
    def _read_outcome(path: Path) -> Outcome:
        from cppmega_mlx.runtime.code_verifier import FsStateDiff

        text = path.read_text(encoding="utf-8", errors="replace")
        return Outcome(
            ok=True,
            exit_code=0,
            diagnostics=[],
            stdout_tail=text[-4096:],
            fs_state_diff=FsStateDiff(),
            kind="read_file",
            extra={"path": str(path), "bytes": path.stat().st_size},
        )

    # ------------------------------------------------------------------ #
    def run(self) -> list[Step]:
        """Run the loop until STOP or ``max_steps``; returns the trace."""
        history: list[Step] = []
        for _ in range(self.max_steps):
            state = RepoState.load(self.repo_root)
            intent_text = self.policy(state, history)
            step = Step(intent_text=intent_text)

            # SAFETY GATE: validate the intent into a typed, contained ToolCall.
            call = self.router.parse(intent_text)  # RAISES on bad/escape/etc.
            step.tool_call = call

            if call.kind is ActionKind.STOP:
                history.append(step)
                break

            outcome = self._execute(call)
            step.observation = Observation(
                tool_call=call.to_wire(), outcome=outcome.to_dict()
            )
            history.append(step)
        return history


# --------------------------------------------------------------------------- #
def demo(repo_root: str | None = None) -> list[Step]:
    """Runnable demo: stub policy + real verifier over a tiny temp repo."""
    import tempfile

    if repo_root is None:
        tmp = tempfile.mkdtemp(prefix="cppmega_orch_demo_")
        (Path(tmp) / "ok.cpp").write_text(
            "int add(int a, int b){ return a + b; }\n", encoding="utf-8"
        )
        repo_root = tmp

    policy = StubPolicy(
        [
            '{"kind": "read_file", "args": {"path": "ok.cpp"}}',
            '{"kind": "stop", "args": {"reason": "done"}}',
        ]
    )
    orch = Orchestrator(repo_root, policy)
    trace = orch.run()
    for i, step in enumerate(trace):
        kind = step.tool_call.kind.value if step.tool_call else "?"
        ok = step.observation.outcome["ok"] if step.observation else "—"
        print(f"step {i}: {kind} ok={ok}")
    return trace


if __name__ == "__main__":
    demo()
