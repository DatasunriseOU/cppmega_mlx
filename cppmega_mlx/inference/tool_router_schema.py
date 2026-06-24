"""Constrained-decoding schema + grammar for the typed tool-call JSON.

This module provides three things, in order of preference for a serving stack:

1. :data:`TOOL_CALL_JSON_SCHEMA` — a JSON Schema (draft-2020-12) describing the
   ``{"kind": ..., "args": {...}}`` envelope with **one variant per**
   :class:`~cppmega_mlx.inference.typed_actions.ActionKind`, including each
   kind's required arg fields. Suitable for Outlines / xgrammar JSON-schema
   constrained decoding.

2. :data:`TOOL_CALL_GBNF` — an explicit GBNF/EBNF-style grammar string (one rule
   per ActionKind) for grammar-based constrained decoding when a schema path is
   not wired up.

3. :class:`ToolRouter` — a pure-python validator + round-trip:
   ``model_text -> grammar/schema-valid JSON -> ToolCall`` that does NOT depend
   on xgrammar/outlines (those are *optional*; if importable we additionally
   compile + validate a sample to prove the grammar is well-formed).

Round-trip contract: :meth:`ToolRouter.parse` takes raw model text, extracts the
first JSON object, validates it against the schema, and returns a validated
:class:`ToolCall`. It RAISES on malformed JSON / schema violation / typed-action
violation — no silent best-effort.
"""

from __future__ import annotations

import json
import re
from typing import Any

from cppmega_mlx.inference.typed_actions import (
    ACTION_PATH_ARGS,
    ACTION_REQUIRED_ARGS,
    ActionKind,
    ActionValidationError,
    ToolCall,
)

__all__ = [
    "TOOL_CALL_JSON_SCHEMA",
    "TOOL_CALL_GBNF",
    "ToolRouter",
    "SchemaValidationError",
    "build_json_schema",
    "build_gbnf",
    "xgrammar_available",
]


class SchemaValidationError(ActionValidationError):
    """Raised when tool-call JSON does not match the schema."""


# Optional arg fields per kind (validated for type if present, never required).
_OPTIONAL_ARGS: dict[ActionKind, tuple[str, ...]] = {
    ActionKind.READ_FILE: ("start_line", "end_line"),
    ActionKind.INSPECT_SYMBOL: ("path",),
    ActionKind.GET_DEP_BLOCKS: ("depth",),
    ActionKind.RUN_BUILD: ("cwd",),
    ActionKind.RUN_TEST: ("cwd",),
    ActionKind.QUERY_CMAKE: ("path",),
    ActionKind.VALIDATE_SQL: ("dialect",),
    ActionKind.STOP: ("reason",),
}

# JSON-schema type per arg field name (kept simple: strings + ints).
_INT_ARGS = frozenset({"start_line", "end_line", "depth"})


def build_json_schema() -> dict[str, Any]:
    """Construct the oneOf JSON schema, one branch per :class:`ActionKind`."""
    branches: list[dict[str, Any]] = []
    for kind in ActionKind:
        required = ACTION_REQUIRED_ARGS[kind]
        optional = _OPTIONAL_ARGS.get(kind, ())
        props: dict[str, Any] = {}
        for name in (*required, *optional):
            props[name] = (
                {"type": "integer"} if name in _INT_ARGS else {"type": "string"}
            )
        args_schema: dict[str, Any] = {
            "type": "object",
            "properties": props,
            "required": list(required),
            "additionalProperties": False,
        }
        branches.append(
            {
                "type": "object",
                "properties": {
                    "kind": {"const": kind.value},
                    "args": args_schema,
                },
                "required": ["kind", "args"],
                "additionalProperties": False,
            }
        )
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "cppmega tool call",
        "oneOf": branches,
    }


TOOL_CALL_JSON_SCHEMA: dict[str, Any] = build_json_schema()


def _gbnf_arg_rules(kind: ActionKind) -> tuple[str, list[str]]:
    """Return (object-rule-body, helper-rules) for a kind's args object."""
    required = ACTION_REQUIRED_ARGS[kind]
    helpers: list[str] = []
    members: list[str] = []
    for name in required:
        val = "integer" if name in _INT_ARGS else "string"
        members.append(f'"\\"{name}\\"" ws ":" ws {val}')
    if not members:
        body = '"{" ws "}"'
    else:
        body = '"{" ws ' + ' ws "," ws '.join(members) + ' ws "}"'
    return body, helpers


def build_gbnf() -> str:
    """Construct a GBNF grammar string, one ``call-<kind>`` rule per kind."""
    lines: list[str] = []
    call_alts: list[str] = []
    for kind in ActionKind:
        rule_name = f"call-{kind.value.replace('_', '-')}"
        args_body, _ = _gbnf_arg_rules(kind)
        lines.append(
            f'{rule_name} ::= "{{" ws "\\"kind\\"" ws ":" ws "\\"{kind.value}\\"" '
            f'ws "," ws "\\"args\\"" ws ":" ws ({args_body}) ws "}}"'
        )
        call_alts.append(rule_name)
    header = ["root ::= " + " | ".join(call_alts)]
    terminals = [
        r'ws ::= [ \t\n\r]*',
        r'string ::= "\"" ([^"\\] | "\\" .)* "\""',
        r"integer ::= [0-9]+",
    ]
    return "\n".join(header + lines + terminals) + "\n"


TOOL_CALL_GBNF: str = build_gbnf()


def xgrammar_available() -> bool:
    try:
        import xgrammar  # noqa: F401

        return True
    except Exception:
        return False


_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)


class ToolRouter:
    """Pure-python validator + round-trip from model text to :class:`ToolCall`.

    Parameters
    ----------
    repo_root:
        Absolute repository root, forwarded to typed-action path containment.
    """

    def __init__(self, repo_root: str) -> None:
        self.repo_root = repo_root
        self.schema = TOOL_CALL_JSON_SCHEMA
        self.gbnf = TOOL_CALL_GBNF

    # ------------------------------------------------------------------ #
    @staticmethod
    def extract_json(model_text: str) -> dict[str, Any]:
        """Extract the first JSON object from model text or RAISE."""
        match = _JSON_OBJ_RE.search(model_text)
        if match is None:
            raise SchemaValidationError(
                f"no JSON object found in model text: {model_text!r:.200}"
            )
        try:
            obj = json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            raise SchemaValidationError(
                f"tool-call JSON is malformed: {exc}"
            ) from exc
        if not isinstance(obj, dict):
            raise SchemaValidationError(
                f"tool-call JSON must be an object, got {type(obj).__name__!r}"
            )
        return obj

    def validate_schema(self, obj: dict[str, Any]) -> ActionKind:
        """Validate ``obj`` against the schema; return the resolved kind (RAISES)."""
        if "kind" not in obj:
            raise SchemaValidationError("tool-call JSON missing 'kind'")
        if "args" not in obj or not isinstance(obj["args"], dict):
            raise SchemaValidationError(
                "tool-call JSON missing object-typed 'args'"
            )
        kind = ActionKind.from_str(obj["kind"])  # RAISES UnknownActionError

        required = set(ACTION_REQUIRED_ARGS[kind])
        allowed = required | set(_OPTIONAL_ARGS.get(kind, ()))
        args = obj["args"]

        missing = required - set(args)
        if missing:
            raise SchemaValidationError(
                f"kind {kind.value!r} missing required args {sorted(missing)}"
            )
        extra = set(args) - allowed
        if extra:
            raise SchemaValidationError(
                f"kind {kind.value!r} has unexpected args {sorted(extra)}; "
                f"allowed {sorted(allowed)}"
            )
        for name, value in args.items():
            if name in _INT_ARGS and not isinstance(value, int):
                raise SchemaValidationError(
                    f"arg {name!r} for {kind.value!r} must be integer, got "
                    f"{type(value).__name__!r}"
                )
            if name not in _INT_ARGS and not isinstance(value, str):
                raise SchemaValidationError(
                    f"arg {name!r} for {kind.value!r} must be string, got "
                    f"{type(value).__name__!r}"
                )
        return kind

    def parse(self, model_text: str) -> ToolCall:
        """Full round-trip: text -> schema-valid JSON -> validated ToolCall."""
        obj = self.extract_json(model_text)
        kind = self.validate_schema(obj)
        # typed-action layer enforces path-escape + command allowlist + raises
        return ToolCall.validated(kind, obj["args"], self.repo_root)

    # ------------------------------------------------------------------ #
    def compile_with_xgrammar(self, sample_text: str) -> bool:
        """If xgrammar is importable, compile the grammar + check a sample.

        Returns ``True`` when xgrammar validated the sample, ``False`` when
        xgrammar is unavailable (caller can fall back to :meth:`parse`).
        RAISES if xgrammar is present but the sample is rejected — we do not
        silently swallow a real grammar failure.
        """
        try:
            import xgrammar as xgr
        except Exception:
            return False
        compiler = xgr.GrammarCompiler(xgr.TokenizerInfo([]))  # type: ignore[arg-type]
        grammar = xgr.Grammar.from_json_schema(json.dumps(self.schema))
        compiler.compile_grammar(grammar)
        matcher = xgr.GrammarMatcher(grammar)
        if not matcher.accept_string(sample_text):
            raise SchemaValidationError(
                f"xgrammar rejected sample tool call: {sample_text!r}"
            )
        return True

    def note(self) -> str:
        return (
            "xgrammar/outlines are OPTIONAL. When absent, use TOOL_CALL_JSON_SCHEMA "
            "or TOOL_CALL_GBNF with your serving stack's constrained decoder, and "
            "ToolRouter.parse() as the pure-python validating round-trip."
        )
