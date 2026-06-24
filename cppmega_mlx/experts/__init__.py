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
    "DEFAULT_USE_LORA",
    "ToolRouterExpert",
    "BuildOpsExpert",
    "BuildOpsResult",
    "SqlExpert",
    "SqlRepairResult",
]

#: The three trainable experts. Keys are the ``--expert`` CLI choices.
EXPERT_NAMES: tuple[str, ...] = ("tool_router", "buildops", "sql")

#: Shared base model: text-only Qwen3-4B-Instruct-2507, 4-bit MLX quant
#: (Apache-2.0, native tool-calling). NOT Qwen3.5-4B (that is a vision-language
#: model — dead weight + unverified quant vision paths for a code agent).
DEFAULT_BASE_MODEL = "mlx-community/Qwen3-4B-Instruct-2507-4bit"
#: Cheap tier for latency-sensitive / smoke runs (verify repo on first pull).
CHEAP_BASE_MODEL = "mlx-community/Qwen3-1.7B-4bit"
#: DEFAULT EXPERT PATH = base + typed-action grammar + verifier loop, NO training.
#: Native FC + constrained decoding already guarantee schema-valid output; the
#: LoRA trainer (scripts/train_expert_lora.py) is OPTIONAL, not the default.
DEFAULT_USE_LORA = False
