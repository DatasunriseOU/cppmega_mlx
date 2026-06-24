"""Per-expert inference wrappers (base Qwen3-4B-Instruct-2507 + per-expert LoRA).

Three experts share the SAME base model + tokenizer / chat template and differ
only by their LoRA adapter and their structured-output contract:

* :class:`~cppmega_mlx.experts.tool_router.ToolRouterExpert` -- decodes a
  schema-valid typed :class:`~cppmega_mlx.inference.typed_actions.ToolCall`
  under the tool-call grammar.
* :class:`~cppmega_mlx.experts.buildops.BuildOpsExpert` -- maps a failing build
  log span to ``{cause, file:line, fix}``.
* :class:`~cppmega_mlx.experts.sql_expert.SqlExpert` -- validates / repairs
  embedded SQL and returns ``{valid, repaired_sql}``.

These are INFERENCE-TIME ONLY. There is no training code here (training lives in
``scripts/train_expert_lora.py``). Per RULE #1, every wrapper FAILS LOUD: a
missing base/adapter, a model load error, or an unparseable / schema-invalid
decode RAISES -- there is no silent fallback to an untyped string.
"""

from __future__ import annotations

from cppmega_mlx.experts.buildops import BuildOpsExpert, BuildOpsResult
from cppmega_mlx.experts.sql_expert import SqlExpert, SqlRepairResult
from cppmega_mlx.experts.tool_router import ToolRouterExpert

__all__ = [
    "EXPERT_NAMES",
    "DEFAULT_BASE_MODEL",
    "CHEAP_BASE_MODEL",
    "ToolRouterExpert",
    "BuildOpsExpert",
    "BuildOpsResult",
    "SqlExpert",
    "SqlRepairResult",
]

#: The three trainable experts. Keys are the ``--expert`` CLI choices.
EXPERT_NAMES: tuple[str, ...] = ("tool_router", "buildops", "sql")

#: Shared base model (Apache-2.0, native tool-calling chat template).
DEFAULT_BASE_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
#: Cheap tier for latency-sensitive / smoke runs.
CHEAP_BASE_MODEL = "Qwen/Qwen3-1.7B"
