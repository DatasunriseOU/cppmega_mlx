"""Phase-8 agent-trajectory extractor (C/C++-only).

Walks recorded coding-agent sessions -- Claude Code ``*.jsonl`` transcripts
(``~/.claude/projects``) and Codex ``rollout-*.jsonl`` rollouts
(``~/.codex/sessions``) -- into temporally-ordered ``(observation, action,
result, verifiable_outcome)`` transitions for RL / world-model training.

RULE #1 (fail fast / fail loud, NO fabricated rewards):

* ``reward`` / ``exit_code`` are populated ONLY when the session data carries a
  REAL verifiable outcome:
    - Codex: the ``function_call_output`` body contains a literal
      ``Process exited with code N`` line -> ``exit_code = N`` and (for
      build/test actions) ``reward = 1.0 if N == 0 else 0.0``.
    - Claude: a ``Bash`` ``toolUseResult`` with ``interrupted`` /
      ``returnCodeInterpretation`` -> for build/test actions parse
      pass/fail into a reward; otherwise leave ``reward = None``.
* Non-build / non-test steps NEVER receive a reward (``reward = None``) -- we
  never invent a signal where the action is not reward-bearing.
* A missing / unparseable outcome leaves ``exit_code = None`` and
  ``reward = None``. We do not guess.

The output schema is a flat per-transition record (:class:`AgentTransition`)
suited to a parquet table; it is the agent-trajectory analogue of the
:class:`~cppmega_mlx.data.trajectory_packet.Transition` contract (real
transitions carry ``reward = None`` unless a genuine outcome exists).
"""

from __future__ import annotations

import json
import re
import shlex
import subprocess
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from cppmega_mlx.data.build_parsers.base import ParsedDomainDocument
from cppmega_mlx.data.diagnostic_parsers import (
    parse_build_error,
    parse_clang_diagnostic,
    parse_linker_error,
    parse_sanitizer_output,
    parse_test_output,
)
from cppmega_mlx.data.shell_parsers import parse_bash, parse_sh, parse_tcsh, parse_zsh

# --------------------------------------------------------------------------- #
# Action-kind classification
# --------------------------------------------------------------------------- #
ACTION_EDIT = "edit"
ACTION_BUILD = "build"
ACTION_TEST = "test"
ACTION_READ = "read"
ACTION_SEARCH = "search"
ACTION_OTHER = "other"

# C/C++ provenance signals (paths, build systems, compilers).
_CPP_PATH_RE = re.compile(r"\.(c|cc|cpp|cxx|c\+\+|h|hh|hpp|hxx|cu|cuh|inl|ipp)\b", re.IGNORECASE)
_CPP_TOKEN_RE = re.compile(
    r"\b(cmake|cmakelists|makefile|clang\+\+|clang|gcc|g\+\+|nvcc|ninja|"
    r"meson|bazel|ctest|gtest|catch2|conan|vcpkg|"
    r"\-std=c\+\+|\-std=gnu\+\+|\.so\b|\.a\b|libstdc\+\+|cuda)\b",
    re.IGNORECASE,
)

# Build / test command signals (used for action classification + reward gating).
_BUILD_RE = re.compile(
    r"\b(cmake|make\b|ninja|nvcc|clang\+\+|g\+\+|gcc\b|clang\b|bazel build|"
    r"meson|\bbuild\b|setup\.py build|pip install -e|cargo build)\b",
    re.IGNORECASE,
)
_TEST_RE = re.compile(
    r"\b(ctest|gtest|catch2|pytest|\btest\b|\btests\b|google ?test|"
    r"unittest|--gtest|run_tests|check\b)\b",
    re.IGNORECASE,
)
_READ_TOOLS = {"Read"}
_SEARCH_TOOLS = {"Grep", "Glob"}
_EDIT_TOOLS = {"Edit", "Write", "MultiEdit", "NotebookEdit"}

# "Process exited with code N" (Codex verifiable reward).
_EXIT_RE = re.compile(r"Process exited with code\s+(-?\d+)")


def classify_cpp(text: str) -> bool:
    """True if ``text`` shows C/C++ provenance (path extension or toolchain token)."""

    if not text:
        return False
    return bool(_CPP_PATH_RE.search(text) or _CPP_TOKEN_RE.search(text))


def _classify_command(cmd: str) -> str:
    """Classify a shell command string into build/test/search/read/other."""

    if not cmd:
        return ACTION_OTHER
    low = cmd
    if _TEST_RE.search(low):
        return ACTION_TEST
    if _BUILD_RE.search(low):
        return ACTION_BUILD
    head = cmd.strip().split()
    first = head[0] if head else ""
    if first in {"rg", "grep", "ag", "ack", "find", "fd"}:
        return ACTION_SEARCH
    if first in {"cat", "less", "head", "tail", "bat"}:
        return ACTION_READ
    return ACTION_OTHER


def classify_action(tool_name: str, command: str | None) -> str:
    """Classify a (tool, command) pair into one action kind."""

    if tool_name in _EDIT_TOOLS:
        return ACTION_EDIT
    if tool_name in _READ_TOOLS:
        return ACTION_READ
    if tool_name in _SEARCH_TOOLS:
        return ACTION_SEARCH
    if command is not None:
        return _classify_command(command)
    return ACTION_OTHER


def parse_shell_action_domain(
    command: str,
    *,
    shell_kind: str | None = None,
) -> ParsedDomainDocument:
    """Parse a trajectory shell action into the matching shell domain.

    Unknown shell stays POSIX ``sh``. We do not label it bash unless the caller
    or command itself gives evidence, because bash/zsh/tcsh have different
    syntax and should remain separate domains for training.
    """

    kind = (shell_kind or "").strip().lower()
    stripped = command.lstrip()
    if not kind:
        if stripped.startswith("#!"):
            first = stripped.splitlines()[0]
            if "zsh" in first:
                kind = "zsh"
            elif "tcsh" in first or "csh" in first:
                kind = "tcsh"
            elif "bash" in first:
                kind = "bash"
            else:
                kind = "sh"
        else:
            head = stripped.split(maxsplit=1)[0] if stripped else ""
            kind = head if head in {"bash", "zsh", "tcsh", "sh"} else "sh"

    if kind == "bash":
        return parse_bash(command)
    if kind == "zsh":
        return parse_zsh(command)
    if kind == "tcsh":
        return parse_tcsh(command)
    return parse_sh(command)


def parse_result_diagnostic_domain(result_text: str) -> ParsedDomainDocument | None:
    """Parse a build/test/tool result into a diagnostic domain when possible."""

    if not result_text.strip():
        return None
    lower = result_text.lower()
    if any(
        marker.lower() in lower
        for marker in (
            "AddressSanitizer",
            "LeakSanitizer",
            "MemorySanitizer",
            "ThreadSanitizer",
            "UndefinedBehaviorSanitizer",
        )
    ):
        return parse_sanitizer_output(result_text)
    if re.search(r"(?m)^(FAILED|PASSED)\s+\S+::", result_text) or (
        "assertionerror" in lower and "test" in lower
    ):
        return parse_test_output(result_text)
    if "undefined reference" in lower or "unresolved external symbol" in lower:
        return parse_linker_error(result_text)
    if re.search(
        r"^[^\n:]+:\d+:(?:\d+:)?\s*(fatal error|error|warning|note):",
        result_text,
        re.MULTILINE,
    ):
        return parse_clang_diagnostic(result_text)
    if "cmake error" in lower or "ninja:" in lower or "build stopped" in lower:
        return parse_build_error(result_text)
    return None


# --------------------------------------------------------------------------- #
# Transition record (flat parquet-friendly schema)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class AgentTransition:
    """One agent step. ``reward`` / ``exit_code`` are None unless REAL.

    Columns map 1:1 to the emitted parquet table.
    """

    session_id: str
    source: str  # "claude" | "codex"
    repo: str | None
    step_idx: int
    obs_text: str
    action_kind: str
    action_payload: str  # JSON string {tool, ...args}
    result_text: str
    exit_code: int | None
    is_build: bool
    is_test: bool
    reward: float | None
    edit_diff: str | None

    def __post_init__(self) -> None:
        if self.source not in ("claude", "codex"):
            raise ValueError(
                f"AgentTransition.source must be 'claude' or 'codex', got {self.source!r}"
            )
        if self.reward is not None and not isinstance(self.reward, (int, float)):
            raise TypeError(
                f"AgentTransition.reward must be a number or None, got "
                f"{type(self.reward).__name__}"
            )
        # RULE #1: a reward may exist ONLY on a build/test step (the
        # reward-bearing ones). A reward on any other kind is fabricated.
        if self.reward is not None and not (self.is_build or self.is_test):
            raise ValueError(
                f"AgentTransition step {self.step_idx} ({self.action_kind}) carries "
                f"reward={self.reward} but is neither build nor test -- non-build/"
                f"non-test steps must have reward=None (no fabricated reward)."
            )

    def as_row(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "source": self.source,
            "repo": self.repo,
            "step_idx": self.step_idx,
            "obs_text": self.obs_text,
            "action_kind": self.action_kind,
            "action_payload": self.action_payload,
            "result_text": self.result_text,
            "exit_code": self.exit_code,
            "is_build": self.is_build,
            "is_test": self.is_test,
            "reward": self.reward,
            "edit_diff": self.edit_diff,
        }


PARQUET_COLUMNS = (
    "session_id",
    "source",
    "repo",
    "step_idx",
    "obs_text",
    "action_kind",
    "action_payload",
    "result_text",
    "exit_code",
    "is_build",
    "is_test",
    "reward",
    "edit_diff",
)

_MAX_TEXT = 16384  # keep obs/result text bounded for parquet


def _truncate(text: str, limit: int = _MAX_TEXT) -> str:
    if text is None:
        return ""
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n...[truncated {len(text) - limit} chars]"


def _reward_from_exit(exit_code: int | None) -> float | None:
    """Map a real exit code to a reward (1.0 pass / 0.0 fail). None if no code."""

    if exit_code is None:
        return None
    return 1.0 if exit_code == 0 else 0.0


# --------------------------------------------------------------------------- #
# Claude Code transcript parsing
# --------------------------------------------------------------------------- #
def _content_to_text(content: Any) -> str:
    """Flatten a Claude message ``content`` (str or list of blocks) into text."""

    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                bt = block.get("type")
                if bt == "text":
                    parts.append(block.get("text", ""))
                elif bt == "thinking":
                    parts.append(block.get("thinking", ""))
                elif bt == "tool_result":
                    parts.append(_content_to_text(block.get("content")))
        return "\n".join(p for p in parts if p)
    return str(content)


def _structured_patch_to_diff(structured_patch: Any) -> str | None:
    """Render a Claude ``structuredPatch`` (list of hunks) into a unified diff."""

    if not structured_patch or not isinstance(structured_patch, list):
        return None
    lines: list[str] = []
    for hunk in structured_patch:
        if not isinstance(hunk, dict):
            continue
        os_, ol = hunk.get("oldStart"), hunk.get("oldLines")
        ns_, nl = hunk.get("newStart"), hunk.get("newLines")
        lines.append(f"@@ -{os_},{ol} +{ns_},{nl} @@")
        for ln in hunk.get("lines", []):
            lines.append(ln)
    return "\n".join(lines) if lines else None


def parse_claude_session(path: Path) -> tuple[list[dict[str, Any]], str | None, str | None]:
    """Parse a Claude transcript into raw events + (session_id, cwd/repo).

    Returns ``(events, session_id, cwd)`` where ``events`` is the ordered list of
    line dicts. Raises on unreadable file (fail loud).
    """

    events: list[dict[str, Any]] = []
    session_id: str | None = None
    cwd: str | None = None
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                # A single malformed transcript line is skipped (the harness
                # itself can flush a partial line); we do not abort the whole
                # session, but we never fabricate content for it.
                continue
            events.append(obj)
            if session_id is None and obj.get("sessionId"):
                session_id = obj["sessionId"]
            if cwd is None and obj.get("cwd"):
                cwd = obj["cwd"]
    if session_id is None:
        session_id = path.stem
    return events, session_id, cwd


def _bash_outcome(tool_result: dict[str, Any]) -> tuple[int | None, str]:
    """Extract (exit_code, result_text) from a Claude Bash toolUseResult.

    Claude does not record a raw numeric exit code; it records ``interrupted``
    and (sometimes) ``returnCodeInterpretation`` (e.g. "exit code 1"). We parse
    a real numeric code from that interpretation when present; otherwise the
    code is None (we never guess).
    """

    stdout = tool_result.get("stdout", "") or ""
    stderr = tool_result.get("stderr", "") or ""
    interp = tool_result.get("returnCodeInterpretation", "") or ""
    body = stdout
    if stderr:
        body = body + ("\n" if body else "") + "[stderr]\n" + stderr
    exit_code: int | None = None
    m = re.search(r"exit(?:ed with)? code\s+(-?\d+)", interp, re.IGNORECASE)
    if m:
        exit_code = int(m.group(1))
    elif interp:
        low = interp.lower()
        if "success" in low or "completed successfully" in low:
            exit_code = 0
        elif "error" in low or "failed" in low or "failure" in low:
            exit_code = 1
    if tool_result.get("interrupted"):
        # Interrupted runs carry no honest pass/fail; force code unknown.
        exit_code = None
        body = (body + "\n[interrupted]").strip()
    return exit_code, body


def walk_claude(
    events: list[dict[str, Any]], session_id: str, repo: str | None
) -> Iterator[AgentTransition]:
    """Walk parsed Claude events into ordered transitions.

    Pairs each assistant ``tool_use`` block with the following user
    ``toolUseResult``. Observation = the most recent user prompt text plus any
    assistant reasoning leading into the action, plus the cwd.
    """

    # Index toolUseResults: a user event after a tool_use carries the result;
    # the tool_use_id appears in the user message content tool_result block.
    # We build a map id -> toolUseResult by scanning user events.
    result_by_id: dict[str, dict[str, Any]] = {}
    for ev in events:
        if ev.get("type") != "user":
            continue
        tur = ev.get("toolUseResult")
        if not isinstance(tur, dict):
            continue
        content = ev.get("message", {}).get("content")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    tid = block.get("tool_use_id")
                    if tid:
                        result_by_id[tid] = tur

    step = 0
    last_prompt = ""
    pending_reasoning = ""
    cwd = repo
    for ev in events:
        et = ev.get("type")
        if ev.get("cwd"):
            cwd = ev["cwd"]
        if et == "user":
            content = ev.get("message", {}).get("content")
            # Only treat genuine user prompts (str content) as observation text.
            if isinstance(content, str) and content.strip():
                last_prompt = content
            continue
        if et != "assistant":
            continue
        msg = ev.get("message", {})
        for block in msg.get("content", []):
            if not isinstance(block, dict):
                continue
            bt = block.get("type")
            if bt == "thinking":
                pending_reasoning = block.get("thinking", "")[:2000]
                continue
            if bt == "text":
                pending_reasoning = (pending_reasoning + "\n" + block.get("text", ""))[:2000]
                continue
            if bt != "tool_use":
                continue
            tool_name = block.get("name", "")
            tinput = block.get("input", {}) or {}
            tid = block.get("id")
            command = tinput.get("command") if tool_name == "Bash" else None
            action_kind = classify_action(tool_name, command)
            is_build = action_kind == ACTION_BUILD
            is_test = action_kind == ACTION_TEST

            tur = result_by_id.get(tid) if tid else None
            exit_code: int | None = None
            result_text = ""
            edit_diff: str | None = None
            if isinstance(tur, dict):
                if "structuredPatch" in tur:
                    edit_diff = _structured_patch_to_diff(tur.get("structuredPatch"))
                    result_text = (
                        f"applied edit to {tur.get('filePath', '')}"
                        if not tur.get("userModified")
                        else f"edit (user-modified) to {tur.get('filePath', '')}"
                    )
                elif "stdout" in tur or "stderr" in tur or "interrupted" in tur:
                    exit_code, result_text = _bash_outcome(tur)
                elif "content" in tur:  # Write tool result
                    result_text = f"wrote {tur.get('filePath', '')}"
                else:
                    result_text = _truncate(json.dumps(tur)[:2000])

            reward = _reward_from_exit(exit_code) if (is_build or is_test) else None

            obs = f"[cwd={cwd}]\n{last_prompt}"
            if pending_reasoning.strip():
                obs += f"\n[reasoning] {pending_reasoning.strip()}"

            payload = {"tool": tool_name, **{k: tinput[k] for k in tinput}}
            yield AgentTransition(
                session_id=session_id,
                source="claude",
                repo=cwd,
                step_idx=step,
                obs_text=_truncate(obs),
                action_kind=action_kind,
                action_payload=json.dumps(payload, default=str)[:_MAX_TEXT],
                result_text=_truncate(result_text),
                exit_code=exit_code,
                is_build=is_build,
                is_test=is_test,
                reward=reward,
                edit_diff=_truncate(edit_diff) if edit_diff else None,
            )
            step += 1
            pending_reasoning = ""


# --------------------------------------------------------------------------- #
# Codex rollout parsing
# --------------------------------------------------------------------------- #
def parse_codex_session(path: Path) -> tuple[list[dict[str, Any]], str | None, str | None]:
    """Parse a Codex rollout into ordered events + (session_id, cwd/repo)."""

    events: list[dict[str, Any]] = []
    session_id: str | None = None
    cwd: str | None = None
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            events.append(obj)
            payload = obj.get("payload", {})
            if obj.get("type") == "session_meta" and isinstance(payload, dict):
                session_id = payload.get("id", session_id)
                cwd = payload.get("cwd", cwd)
    if session_id is None:
        session_id = path.stem
    return events, session_id, cwd


def _codex_exit_code(output_text: str) -> int | None:
    """Parse the literal 'Process exited with code N' line. None if absent."""

    m = _EXIT_RE.search(output_text or "")
    return int(m.group(1)) if m else None


def walk_codex(
    events: list[dict[str, Any]], session_id: str, repo: str | None
) -> Iterator[AgentTransition]:
    """Walk Codex events into transitions, pairing function_call + output by call_id."""

    # Map call_id -> output text.
    output_by_call: dict[str, str] = {}
    for ev in events:
        p = ev.get("payload", {})
        if isinstance(p, dict) and p.get("type") == "function_call_output":
            cid = p.get("call_id")
            out = p.get("output")
            if cid is not None and isinstance(out, str):
                output_by_call[cid] = out

    step = 0
    last_user = ""
    last_reasoning = ""
    for ev in events:
        p = ev.get("payload", {})
        if not isinstance(p, dict):
            continue
        pt = p.get("type")
        if pt in ("user_message", "message") and p.get("role") in (None, "user"):
            txt = p.get("text") or p.get("content")
            if isinstance(txt, str) and txt.strip():
                last_user = txt
            continue
        if pt == "reasoning":
            summ = p.get("summary") or p.get("text")
            if isinstance(summ, list):
                summ = " ".join(str(s) for s in summ)
            if isinstance(summ, str):
                last_reasoning = summ[:2000]
            continue
        if pt != "function_call":
            continue
        name = p.get("name", "")
        if name != "exec_command":
            continue
        raw_args = p.get("arguments", "{}")
        try:
            args = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args)
        except json.JSONDecodeError:
            args = {}
        cmd = args.get("cmd", "")
        workdir = args.get("workdir", repo)
        cid = p.get("call_id")
        output_text = output_by_call.get(cid, "")

        action_kind = _classify_command(cmd)
        is_build = action_kind == ACTION_BUILD
        is_test = action_kind == ACTION_TEST
        exit_code = _codex_exit_code(output_text)
        reward = _reward_from_exit(exit_code) if (is_build or is_test) else None

        obs = f"[cwd={workdir}]\n{last_user}"
        if last_reasoning.strip():
            obs += f"\n[reasoning] {last_reasoning.strip()}"

        payload = {"tool": "exec_command", "cmd": cmd, "workdir": workdir}
        yield AgentTransition(
            session_id=session_id,
            source="codex",
            repo=workdir,
            step_idx=step,
            obs_text=_truncate(obs),
            action_kind=action_kind,
            action_payload=json.dumps(payload, default=str)[:_MAX_TEXT],
            result_text=_truncate(output_text),
            exit_code=exit_code,
            is_build=is_build,
            is_test=is_test,
            reward=reward,
            edit_diff=None,
        )
        step += 1
        last_reasoning = ""


# --------------------------------------------------------------------------- #
# Session enumeration + C/C++ classification
# --------------------------------------------------------------------------- #
@dataclass
class SessionRef:
    """A locatable session file (local or remote)."""

    path: str
    source: str  # "claude" | "codex"
    host: str | None = None  # None = local, else ssh host


def enumerate_local_sessions(
    claude_root: Path | None = None, codex_root: Path | None = None
) -> list[SessionRef]:
    """Enumerate local Claude + Codex session files."""

    if claude_root is None:
        claude_root = Path.home() / ".claude" / "projects"
    if codex_root is None:
        codex_root = Path.home() / ".codex" / "sessions"
    refs: list[SessionRef] = []
    if claude_root.exists():
        for p in sorted(claude_root.rglob("*.jsonl")):
            refs.append(SessionRef(path=str(p), source="claude"))
    if codex_root.exists():
        for p in sorted(codex_root.rglob("rollout-*.jsonl")):
            refs.append(SessionRef(path=str(p), source="codex"))
    return refs


_SSH_BASE = [
    "ssh",
    "-o",
    "BatchMode=yes",
    "-o",
    "ConnectTimeout=15",
    "-o",
    "StrictHostKeyChecking=accept-new",
]


def enumerate_remote_sessions(host: str = "dave@10.0.0.25") -> list[SessionRef]:
    """Enumerate remote session files over read-only ssh. Fails loud on error."""

    cmd = (
        "find ~/.claude/projects -name '*.jsonl' 2>/dev/null; "
        "find ~/.codex/sessions -name 'rollout-*.jsonl' 2>/dev/null"
    )
    proc = subprocess.run(
        _SSH_BASE + [host, cmd],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"remote enumeration on {host} failed (rc={proc.returncode}): "
            f"{proc.stderr.strip()[:400]}"
        )
    refs: list[SessionRef] = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        src = "codex" if "/rollout-" in line else "claude"
        refs.append(SessionRef(path=line, source=src, host=host))
    return refs


def _read_session_text(ref: SessionRef) -> str:
    """Read a session file's bytes (local or via ssh cat, read-only)."""

    if ref.host is None:
        return Path(ref.path).read_text(encoding="utf-8", errors="replace")
    proc = subprocess.run(
        _SSH_BASE + [ref.host, f"cat {shlex.quote(ref.path)}"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"remote read of {ref.path} on {ref.host} failed (rc={proc.returncode}): "
            f"{proc.stderr.strip()[:200]}"
        )
    return proc.stdout


def _events_from_text(text: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events


def _session_is_cpp(events: list[dict[str, Any]], source: str, cwd: str | None) -> bool:
    """Classify a session as C/C++ by cwd/project name AND content scan."""

    if cwd and classify_cpp(cwd):
        return True
    # Scan a bounded amount of event text for C/C++ signals.
    scanned = 0
    for ev in events:
        if scanned > 400_000:
            break
        chunk = json.dumps(ev, default=str)
        scanned += len(chunk)
        if classify_cpp(chunk):
            return True
    return False


@dataclass
class ExtractionStats:
    sessions_seen: int = 0
    sessions_kept: int = 0
    sessions_dropped: int = 0
    transitions: int = 0
    build_steps: int = 0
    test_steps: int = 0
    build_test_with_exit: int = 0
    rewards_emitted: int = 0
    by_source: dict[str, int] = field(default_factory=dict)


def extract_session(ref: SessionRef) -> tuple[str | None, list[AgentTransition], bool]:
    """Extract one session. Returns (session_id, transitions, is_cpp)."""

    text = _read_session_text(ref)
    events = _events_from_text(text)
    if not events:
        return None, [], False

    if ref.source == "claude":
        session_id = None
        cwd = None
        for ev in events:
            if session_id is None and ev.get("sessionId"):
                session_id = ev["sessionId"]
            if cwd is None and ev.get("cwd"):
                cwd = ev["cwd"]
        session_id = session_id or Path(ref.path).stem
        is_cpp = _session_is_cpp(events, "claude", cwd)
        if not is_cpp:
            return session_id, [], False
        transitions = list(walk_claude(events, session_id, cwd))
        return session_id, transitions, True

    # codex
    session_id = None
    cwd = None
    for ev in events:
        p = ev.get("payload", {})
        if ev.get("type") == "session_meta" and isinstance(p, dict):
            session_id = p.get("id", session_id)
            cwd = p.get("cwd", cwd)
    session_id = session_id or Path(ref.path).stem
    is_cpp = _session_is_cpp(events, "codex", cwd)
    if not is_cpp:
        return session_id, [], False
    transitions = list(walk_codex(events, session_id, cwd))
    return session_id, transitions, True


def extract_all(
    refs: list[SessionRef], max_sessions: int | None = None
) -> tuple[list[AgentTransition], ExtractionStats]:
    """Extract transitions from all C/C++ sessions in ``refs``."""

    stats = ExtractionStats()
    out: list[AgentTransition] = []
    considered = refs if max_sessions is None else refs[:max_sessions]
    for ref in considered:
        stats.sessions_seen += 1
        try:
            _sid, transitions, is_cpp = extract_session(ref)
        except (OSError, RuntimeError) as exc:
            raise RuntimeError(f"failed extracting {ref.path}: {exc}") from exc
        if not is_cpp:
            stats.sessions_dropped += 1
            continue
        stats.sessions_kept += 1
        stats.by_source[ref.source] = stats.by_source.get(ref.source, 0) + 1
        for tr in transitions:
            out.append(tr)
            stats.transitions += 1
            if tr.is_build:
                stats.build_steps += 1
            if tr.is_test:
                stats.test_steps += 1
            if (tr.is_build or tr.is_test) and tr.exit_code is not None:
                stats.build_test_with_exit += 1
            if tr.reward is not None:
                stats.rewards_emitted += 1
    return out, stats


def write_parquet(transitions: list[AgentTransition], out_path: Path) -> Path:
    """Write transitions to a parquet table with the fixed schema."""

    import pyarrow as pa
    import pyarrow.parquet as pq

    if not transitions:
        raise ValueError("refusing to write an empty transitions parquet (no data)")
    rows = [t.as_row() for t in transitions]
    cols = {c: [r[c] for r in rows] for c in PARQUET_COLUMNS}
    schema = pa.schema(
        [
            ("session_id", pa.string()),
            ("source", pa.string()),
            ("repo", pa.string()),
            ("step_idx", pa.int32()),
            ("obs_text", pa.string()),
            ("action_kind", pa.string()),
            ("action_payload", pa.string()),
            ("result_text", pa.string()),
            ("exit_code", pa.int32()),
            ("is_build", pa.bool_()),
            ("is_test", pa.bool_()),
            ("reward", pa.float32()),
            ("edit_diff", pa.string()),
        ]
    )
    table = pa.table(cols, schema=schema)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, out_path)
    return out_path


__all__ = [
    "ACTION_BUILD",
    "ACTION_EDIT",
    "ACTION_OTHER",
    "ACTION_READ",
    "ACTION_SEARCH",
    "ACTION_TEST",
    "AgentTransition",
    "ExtractionStats",
    "PARQUET_COLUMNS",
    "SessionRef",
    "classify_action",
    "classify_cpp",
    "enumerate_local_sessions",
    "enumerate_remote_sessions",
    "extract_all",
    "extract_session",
    "parse_result_diagnostic_domain",
    "parse_shell_action_domain",
    "walk_claude",
    "walk_codex",
    "write_parquet",
]
