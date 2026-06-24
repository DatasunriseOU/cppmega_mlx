"""Typed action layer for the Phase-7 agentic loop.

The model core emits an *intent* (free text / constrained JSON). This module is
the SAFE TYPED LAYER that sits between that intent and the ground-truth sandbox
(:mod:`cppmega_mlx.runtime.code_verifier`).

Design rules (per project RULE #1 — fail fast, fail loud):

* There is **no** raw-shell action kind. The only commands that can ever run
  are the ones explicitly modeled by an :class:`ActionKind` and validated here.
* Validation **RAISES** (never silently degrades / clamps / drops) on:
    - unknown action kind,
    - a missing required argument,
    - a path argument that escapes the repository root,
    - a disallowed command token for command-bearing kinds.

Nothing in this module imports model weights or any heavy runtime; it is pure
interface + validation so it can be unit-tested in isolation.
"""

from __future__ import annotations

import enum
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "ActionKind",
    "ToolCall",
    "ActionValidationError",
    "PathEscapeError",
    "MissingArgumentError",
    "DisallowedCommandError",
    "UnknownActionError",
    "ACTION_REQUIRED_ARGS",
    "ACTION_PATH_ARGS",
    "ALLOWED_BUILD_TOOLS",
    "ALLOWED_TEST_TOOLS",
]


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #
class ActionValidationError(ValueError):
    """Base class for all typed-action validation failures."""


class UnknownActionError(ActionValidationError):
    """Raised when an action kind is not a member of :class:`ActionKind`."""


class MissingArgumentError(ActionValidationError):
    """Raised when a required argument for an action kind is absent."""


class PathEscapeError(ActionValidationError):
    """Raised when a path argument resolves outside the repository root."""


class DisallowedCommandError(ActionValidationError):
    """Raised when a command-bearing action names a non-allowlisted tool."""


# --------------------------------------------------------------------------- #
# Frozen action vocabulary
# --------------------------------------------------------------------------- #
@enum.unique
class ActionKind(enum.Enum):
    """The complete, frozen set of actions the core may request.

    This enum is the security boundary: anything not listed here cannot be
    expressed, let alone executed. There is intentionally **no** generic
    ``SHELL`` / ``EXEC`` kind.
    """

    READ_FILE = "read_file"
    INSPECT_SYMBOL = "inspect_symbol"
    GET_DEP_BLOCKS = "get_dep_blocks"
    RUN_BUILD = "run_build"
    RUN_TEST = "run_test"
    APPLY_PATCH = "apply_patch"
    VALIDATE_SQL = "validate_sql"
    QUERY_CMAKE = "query_cmake"
    STOP = "stop"

    @classmethod
    def from_str(cls, value: str) -> "ActionKind":
        """Resolve a wire string to an :class:`ActionKind` or RAISE.

        Accepts either the value (``"read_file"``) or the member name
        (``"READ_FILE"``); anything else is a hard error.
        """
        if isinstance(value, ActionKind):
            return value
        if not isinstance(value, str):
            raise UnknownActionError(
                f"action kind must be a string, got {type(value).__name__!r}"
            )
        # value form
        for member in cls:
            if member.value == value:
                return member
        # name form
        try:
            return cls[value]
        except KeyError as exc:
            valid = ", ".join(sorted(m.value for m in cls))
            raise UnknownActionError(
                f"unknown action kind {value!r}; valid kinds: {valid}"
            ) from exc


# --------------------------------------------------------------------------- #
# Per-kind argument contracts
# --------------------------------------------------------------------------- #
# Required argument names per action kind.
ACTION_REQUIRED_ARGS: dict[ActionKind, tuple[str, ...]] = {
    ActionKind.READ_FILE: ("path",),
    ActionKind.INSPECT_SYMBOL: ("symbol",),
    ActionKind.GET_DEP_BLOCKS: ("path",),
    ActionKind.RUN_BUILD: ("command",),
    ActionKind.RUN_TEST: ("command",),
    ActionKind.APPLY_PATCH: ("path", "patch"),
    ActionKind.VALIDATE_SQL: ("sql",),
    ActionKind.QUERY_CMAKE: ("query",),
    ActionKind.STOP: (),
}

# Argument names whose values are filesystem paths and MUST stay inside the
# repository root. Keyed by action kind.
ACTION_PATH_ARGS: dict[ActionKind, tuple[str, ...]] = {
    ActionKind.READ_FILE: ("path",),
    ActionKind.GET_DEP_BLOCKS: ("path",),
    ActionKind.INSPECT_SYMBOL: ("path",),  # optional path; validated if present
    ActionKind.APPLY_PATCH: ("path",),
    ActionKind.RUN_BUILD: ("cwd",),  # optional working dir
    ActionKind.RUN_TEST: ("cwd",),
    ActionKind.QUERY_CMAKE: ("path",),  # optional CMakeLists path
}

# Command-bearing kinds: the FIRST token of ``command`` must be in the allowlist.
ALLOWED_BUILD_TOOLS: frozenset[str] = frozenset(
    {"cmake", "make", "ninja", "clang++", "clang", "g++", "gcc", "bazel"}
)
ALLOWED_TEST_TOOLS: frozenset[str] = frozenset(
    {"ctest", "make", "ninja", "bazel", "./run_tests.sh", "pytest"}
)

_COMMAND_ALLOWLISTS: dict[ActionKind, frozenset[str]] = {
    ActionKind.RUN_BUILD: ALLOWED_BUILD_TOOLS,
    ActionKind.RUN_TEST: ALLOWED_TEST_TOOLS,
}

# Shell metacharacters that must never appear in a modeled command — their
# presence means someone is trying to smuggle a raw shell in through an
# allowlisted tool name.
_SHELL_METACHARS = (";", "|", "&", "$(", "`", ">", "<", "\n", "&&", "||")


# --------------------------------------------------------------------------- #
# The typed tool call
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ToolCall:
    """A validated request from the core to the sandbox.

    Construct via :meth:`validated` (the only safe entry point). Direct
    construction does *not* validate, mirroring how dataclasses behave; callers
    in the orchestrator always go through :meth:`validated`.
    """

    kind: ActionKind
    args: Mapping[str, Any] = field(default_factory=dict)

    # ----------------------------------------------------------------- #
    @staticmethod
    def _command_tokens(command: Any) -> list[str]:
        if isinstance(command, str):
            return command.strip().split()
        if isinstance(command, (list, tuple)):
            return [str(tok) for tok in command]
        raise MissingArgumentError(
            f"command must be a string or list, got {type(command).__name__!r}"
        )

    @classmethod
    def validated(
        cls,
        kind: ActionKind | str,
        args: Mapping[str, Any] | None,
        repo_root: str | os.PathLike[str],
    ) -> "ToolCall":
        """Build a :class:`ToolCall`, RAISING on any contract violation.

        Parameters
        ----------
        kind:
            An :class:`ActionKind` or its wire string.
        args:
            The argument mapping emitted by the core.
        repo_root:
            Absolute repository root; every path argument must resolve inside
            it (defends against ``../`` escapes and absolute-path escapes).
        """
        resolved_kind = ActionKind.from_str(kind)
        arg_map: dict[str, Any] = dict(args or {})

        root = Path(repo_root).resolve(strict=False)
        if not root.is_absolute():
            raise ActionValidationError(
                f"repo_root must be absolute, got {repo_root!r}"
            )

        # 1) required-arg presence
        for required in ACTION_REQUIRED_ARGS[resolved_kind]:
            if required not in arg_map or arg_map[required] in (None, ""):
                raise MissingArgumentError(
                    f"action {resolved_kind.value!r} requires arg {required!r}; "
                    f"got keys {sorted(arg_map)}"
                )

        # 2) path-escape containment
        for path_arg in ACTION_PATH_ARGS.get(resolved_kind, ()):  # noqa: B007
            if path_arg not in arg_map or arg_map[path_arg] in (None, ""):
                continue  # optional path absent -> nothing to check
            cls._assert_within_root(str(arg_map[path_arg]), root, resolved_kind, path_arg)

        # 3) command allowlist + no-shell-smuggling
        allowlist = _COMMAND_ALLOWLISTS.get(resolved_kind)
        if allowlist is not None:
            raw_command = arg_map["command"]
            cls._assert_no_shell(raw_command, resolved_kind)
            tokens = cls._command_tokens(raw_command)
            if not tokens:
                raise MissingArgumentError(
                    f"action {resolved_kind.value!r} command is empty"
                )
            tool = tokens[0]
            if tool not in allowlist:
                raise DisallowedCommandError(
                    f"action {resolved_kind.value!r} command tool {tool!r} not "
                    f"allowlisted; allowed: {sorted(allowlist)}"
                )

        return cls(kind=resolved_kind, args=arg_map)

    # ----------------------------------------------------------------- #
    @staticmethod
    def _assert_no_shell(command: Any, kind: ActionKind) -> None:
        text = command if isinstance(command, str) else " ".join(map(str, command))
        for meta in _SHELL_METACHARS:
            if meta in text:
                raise DisallowedCommandError(
                    f"action {kind.value!r} command contains forbidden shell "
                    f"metacharacter {meta!r}: {text!r}"
                )

    @staticmethod
    def _assert_within_root(
        raw_path: str, root: Path, kind: ActionKind, arg_name: str
    ) -> Path:
        """Resolve ``raw_path`` against ``root`` and RAISE if it escapes."""
        candidate = Path(raw_path)
        if candidate.is_absolute():
            resolved = candidate.resolve(strict=False)
        else:
            resolved = (root / candidate).resolve(strict=False)
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise PathEscapeError(
                f"action {kind.value!r} arg {arg_name!r} path {raw_path!r} "
                f"resolves to {resolved} which escapes repo root {root}"
            ) from exc
        return resolved

    # ----------------------------------------------------------------- #
    def resolved_path(self, arg_name: str, repo_root: str | os.PathLike[str]) -> Path:
        """Return the absolute, root-contained path for ``arg_name`` (RAISES)."""
        root = Path(repo_root).resolve(strict=False)
        if arg_name not in self.args:
            raise MissingArgumentError(
                f"tool call {self.kind.value!r} has no arg {arg_name!r}"
            )
        return self._assert_within_root(
            str(self.args[arg_name]), root, self.kind, arg_name
        )

    def to_wire(self) -> dict[str, Any]:
        """Serialize back to the constrained-decoding JSON shape."""
        return {"kind": self.kind.value, "args": dict(self.args)}
