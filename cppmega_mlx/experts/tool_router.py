"""ToolRouter expert: state + available typed actions -> validated ToolCall.

Inference-time wrapper around (Qwen3-4B base + tool_router LoRA). It renders the
prompt with the base's native tool-calling chat template, decodes under the
tool-call JSON schema / GBNF grammar
(:mod:`cppmega_mlx.inference.tool_router_schema`), and round-trips the model text
through :meth:`ToolRouter.parse`, which RAISES on any malformed / schema-invalid
/ typed-action-invalid output. No silent best-effort: a bad decode is an error.
"""

from __future__ import annotations

from typing import Any

from cppmega_mlx.experts._base import ExpertDecodeError, LoadedExpert, load_expert
from cppmega_mlx.inference.tool_router_schema import (
    TOOL_CALL_GBNF,
    TOOL_CALL_JSON_SCHEMA,
    ToolRouter,
)
from cppmega_mlx.inference.typed_actions import ToolCall

__all__ = ["ToolRouterExpert", "TOOL_ROUTER_SYSTEM"]

TOOL_ROUTER_SYSTEM = (
    "You are the ToolRouter for a C/C++ build-and-fix agent. Given the current "
    "observation/state and the available typed actions, emit EXACTLY ONE JSON "
    'object {"kind": <action>, "args": {...}} that is valid against the tool-call '
    "schema. Do not emit prose, multiple objects, or any action not in the schema. "
    "If no action is appropriate, emit the stop action with a reason."
)


class ToolRouterExpert:
    """Decode a schema-/grammar-valid :class:`ToolCall` from agent state."""

    def __init__(
        self,
        repo_root: str,
        *,
        base: str = "Qwen/Qwen3-4B-Instruct-2507",
        adapter_path: str | None = None,
        loaded: LoadedExpert | None = None,
    ) -> None:
        self.repo_root = repo_root
        self.router = ToolRouter(repo_root)
        self._base = base
        self._adapter = adapter_path
        self._loaded = loaded

    # ------------------------------------------------------------------ #
    def _ensure_loaded(self) -> LoadedExpert:
        if self._loaded is None:
            self._loaded = load_expert(self._base, self._adapter)
        return self._loaded

    def build_prompt(self, obs_text: str) -> str:
        """Render the chat/tool-call prompt for ``obs_text`` (RAISES on bad tmpl)."""
        loaded = self._ensure_loaded()
        messages = [
            {"role": "system", "content": TOOL_ROUTER_SYSTEM},
            {
                "role": "user",
                "content": (
                    f"OBSERVATION:\n{obs_text}\n\n"
                    "Available typed actions are constrained by the tool-call "
                    "JSON schema below. Respond with one schema-valid JSON object.\n"
                    f"SCHEMA:\n{_schema_hint()}"
                ),
            },
        ]
        return loaded.render_chat(messages)

    @property
    def grammar(self) -> str:
        """The GBNF grammar string to drive constrained decoding."""
        return TOOL_CALL_GBNF

    @property
    def json_schema(self) -> dict[str, Any]:
        """The JSON schema to drive constrained decoding."""
        return TOOL_CALL_JSON_SCHEMA

    # ------------------------------------------------------------------ #
    def route(self, obs_text: str, *, max_tokens: int = 256) -> ToolCall:
        """Full inference: obs -> generated text -> validated ToolCall (RAISES)."""
        loaded = self._ensure_loaded()
        prompt = self.build_prompt(obs_text)
        text = loaded.generate(prompt, max_tokens=max_tokens, temp=0.0)
        # ToolRouter.parse RAISES on malformed/schema/typed-action violation.
        return self.router.parse(text)

    def parse_only(self, model_text: str) -> ToolCall:
        """Validate already-generated text (RAISES). Useful for offline checks."""
        if not model_text:
            raise ExpertDecodeError("empty model text for ToolRouter.parse_only")
        return self.router.parse(model_text)


def _schema_hint() -> str:
    """A compact human/LLM-readable rendering of the kinds + required args."""
    import json

    branches = TOOL_CALL_JSON_SCHEMA["oneOf"]
    lines = []
    for b in branches:
        kind = b["properties"]["kind"]["const"]
        req = b["properties"]["args"].get("required", [])
        lines.append(f"{kind}: required args {req}")
    return json.dumps(lines, indent=0)
