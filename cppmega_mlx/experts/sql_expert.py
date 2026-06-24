"""SQL expert: embedded C/C++ SQL + schema -> {valid, repaired_sql}.

Inference-time wrapper around (Qwen3-4B base + sql LoRA). It renders the C++
context + candidate SQL (+ schema, if any) into the chat template, decodes a
small JSON object, and -- crucially -- RE-VALIDATES the model's ``repaired_sql``
against the ground-truth sqlite verifier
(:meth:`cppmega_mlx.runtime.code_verifier.CodeVerifier.validate_sql`) when a
verifier is supplied. The decode FAILS LOUD on a bad JSON shape, and the
``validated`` flag reflects the REAL verifier outcome, never the model's claim.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from cppmega_mlx.experts._base import ExpertDecodeError, LoadedExpert, load_expert

__all__ = ["SqlExpert", "SqlRepairResult", "SQL_SYSTEM"]

SQL_SYSTEM = (
    "You are a SQL repair expert for SQL embedded in C/C++ source. Given the C++ "
    "context, the candidate SQL, and the schema (if any), respond with EXACTLY ONE "
    'JSON object: {"valid": <bool>, "repaired_sql": <string>}. If the SQL is '
    "already valid, set valid=true and return it unchanged in repaired_sql. Repair "
    "only syntax/schema issues; do not change query intent."
)

_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)


@dataclass(frozen=True)
class SqlRepairResult:
    """Structured SQL repair result.

    ``validated`` is the REAL sqlite verifier outcome on ``repaired_sql`` when a
    verifier was provided, else ``None`` (we never claim validation we did not
    run). ``model_claimed_valid`` is the model's own (untrusted) flag.
    """

    repaired_sql: str
    model_claimed_valid: bool
    validated: bool | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "repaired_sql": self.repaired_sql,
            "model_claimed_valid": self.model_claimed_valid,
            "validated": self.validated,
        }


class SqlExpert:
    def __init__(
        self,
        *,
        base: str = "Qwen/Qwen3-4B-Instruct-2507",
        adapter_path: str | None = None,
        loaded: LoadedExpert | None = None,
        verifier: Any | None = None,
    ) -> None:
        self._base = base
        self._adapter = adapter_path
        self._loaded = loaded
        self._verifier = verifier

    def _ensure_loaded(self) -> LoadedExpert:
        if self._loaded is None:
            self._loaded = load_expert(self._base, self._adapter)
        return self._loaded

    def build_prompt(
        self, cpp_context: str, sql: str, schema: str | None = None
    ) -> str:
        loaded = self._ensure_loaded()
        schema_block = f"\nSCHEMA:\n{schema}" if schema else "\nSCHEMA: (none)"
        messages = [
            {"role": "system", "content": SQL_SYSTEM},
            {
                "role": "user",
                "content": f"C++_CONTEXT:\n{cpp_context}\nSQL:\n{sql}{schema_block}",
            },
        ]
        return loaded.render_chat(messages)

    def repair(
        self,
        cpp_context: str,
        sql: str,
        schema: str | None = None,
        *,
        max_tokens: int = 256,
    ) -> SqlRepairResult:
        loaded = self._ensure_loaded()
        prompt = self.build_prompt(cpp_context, sql, schema)
        text = loaded.generate(prompt, max_tokens=max_tokens, temp=0.0)
        return self.parse_and_verify(text)

    def parse_and_verify(self, model_text: str) -> SqlRepairResult:
        """Parse JSON, then RE-VALIDATE repaired_sql via the sqlite verifier."""
        match = _JSON_OBJ_RE.search(model_text or "")
        if match is None:
            raise ExpertDecodeError(
                f"SQL expert produced no JSON object: {model_text!r:.200}"
            )
        try:
            obj = json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            raise ExpertDecodeError(f"SQL expert JSON malformed: {exc}") from exc
        if not isinstance(obj, dict) or "repaired_sql" not in obj:
            raise ExpertDecodeError(
                f"SQL expert JSON missing 'repaired_sql': {obj!r}"
            )
        repaired = str(obj["repaired_sql"])
        claimed = bool(obj.get("valid", False))

        validated: bool | None = None
        if self._verifier is not None:
            # REAL ground-truth check; RAISES if sqlite is unavailable (fail-loud).
            outcome = self._verifier.validate_sql(repaired, dialect="sqlite")
            validated = bool(outcome.ok)
        return SqlRepairResult(
            repaired_sql=repaired,
            model_claimed_valid=claimed,
            validated=validated,
        )
