"""BuildOps expert: failing build log span -> {cause, file:line, fix}.

Inference-time wrapper around (Qwen3-4B base + buildops LoRA). It renders the
build-system + failing-log span into the chat template and decodes a small JSON
object describing the diagnosis. The output is parsed and FAILS LOUD if it is not
the expected JSON shape -- we never return a guessed/empty fix silently.

The TRAINING targets for this expert are produced only from real-exit-code
transitions (see ``scripts/build_expert_sft_data.py``); this wrapper is the
matching decode side.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from cppmega_mlx.experts._base import ExpertDecodeError, LoadedExpert, load_expert

__all__ = ["BuildOpsExpert", "BuildOpsResult", "BUILDOPS_SYSTEM"]

BUILDOPS_SYSTEM = (
    "You are BuildOps, a C/C++ build-failure diagnostician. Given the build "
    "system and the failing log span, respond with EXACTLY ONE JSON object: "
    '{"cause": <short cause>, "file": <path or null>, "line": <int or null>, '
    '"fix": <typed-action patch or short fix, or null>}. '
    "Base every field on the log; never invent a file:line not present in it."
)

_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)


@dataclass(frozen=True)
class BuildOpsResult:
    """Structured BuildOps diagnosis."""

    cause: str
    file: str | None
    line: int | None
    fix: str | None

    def to_dict(self) -> dict[str, Any]:
        return {"cause": self.cause, "file": self.file, "line": self.line, "fix": self.fix}


class BuildOpsExpert:
    def __init__(
        self,
        *,
        base: str = "Qwen/Qwen3-4B-Instruct-2507",
        adapter_path: str | None = None,
        loaded: LoadedExpert | None = None,
    ) -> None:
        self._base = base
        self._adapter = adapter_path
        self._loaded = loaded

    def _ensure_loaded(self) -> LoadedExpert:
        if self._loaded is None:
            self._loaded = load_expert(self._base, self._adapter)
        return self._loaded

    def build_prompt(self, build_system: str, log_span: str) -> str:
        loaded = self._ensure_loaded()
        messages = [
            {"role": "system", "content": BUILDOPS_SYSTEM},
            {
                "role": "user",
                "content": f"BUILD_SYSTEM: {build_system}\nFAILING_LOG:\n{log_span}",
            },
        ]
        return loaded.render_chat(messages)

    def diagnose(
        self, build_system: str, log_span: str, *, max_tokens: int = 256
    ) -> BuildOpsResult:
        loaded = self._ensure_loaded()
        prompt = self.build_prompt(build_system, log_span)
        text = loaded.generate(prompt, max_tokens=max_tokens, temp=0.0)
        return self.parse_only(text)

    def parse_only(self, model_text: str) -> BuildOpsResult:
        """Parse the JSON diagnosis. RAISES on any shape violation (fail-loud)."""
        match = _JSON_OBJ_RE.search(model_text or "")
        if match is None:
            raise ExpertDecodeError(
                f"BuildOps produced no JSON object: {model_text!r:.200}"
            )
        try:
            obj = json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            raise ExpertDecodeError(f"BuildOps JSON malformed: {exc}") from exc
        if not isinstance(obj, dict) or "cause" not in obj:
            raise ExpertDecodeError(
                f"BuildOps JSON missing required 'cause': {obj!r}"
            )
        line = obj.get("line")
        if line is not None and not isinstance(line, int):
            raise ExpertDecodeError(
                f"BuildOps 'line' must be int or null, got {type(line).__name__!r}"
            )
        return BuildOpsResult(
            cause=str(obj["cause"]),
            file=(str(obj["file"]) if obj.get("file") is not None else None),
            line=line,
            fix=(str(obj["fix"]) if obj.get("fix") is not None else None),
        )
