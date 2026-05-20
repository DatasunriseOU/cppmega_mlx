"""Generators for explicit alternatives to unsatisfied requirements.

Each generator is a pure function:
   (requirement, build_spec, tokenizer_caps, parquet_caps) -> tuple[Alternative, ...]

No side effects, no LLM, no model calls. Alternatives are ranked
deterministically by ``cost`` (low < medium < high) and then by
``action`` to keep test output stable across runs.

Each Alternative carries a JSON-Patch-shaped ``diff`` that the GUI/CLI
applies to the canonical spec representation. The probe never applies
the diff itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping

from cppmega_v4.buildspec.loss_spec import LossKind
from cppmega_v4.buildspec.model_build_spec import ModelBuildSpec
from cppmega_v4.probe.capabilities import (
    ParquetCapabilities,
    TokenizerCapabilities,
)
from cppmega_v4.probe.requirements import DataRequirement


_COST_ORDER: dict[str, int] = {"low": 0, "medium": 1, "high": 2}


@dataclass(frozen=True)
class Alternative:
    """One way the user can resolve an unsatisfied requirement."""

    action: Literal[
        "swap_loss", "swap_tokenizer", "add_column",
        "drop_brick", "relax_requirement",
    ]
    target: str
    diff: Mapping[str, object]
    cost: Literal["low", "medium", "high"]
    reason: str


_MAX_ALTERNATIVES_PER_FINDING: int = 3


def generate_alternatives(
    requirement: DataRequirement,
    component: str,
    build_spec: ModelBuildSpec,
    tokenizer_caps: TokenizerCapabilities,
    parquet_caps: ParquetCapabilities,
) -> tuple[Alternative, ...]:
    """Return ordered alternatives that, if applied, would satisfy
    ``requirement``. Caps the result at three per design ceiling."""
    out: list[Alternative] = []

    # Loss-level requirements — most cheaply resolved by swapping loss.
    if component.startswith("loss:"):
        if build_spec.loss.kind != LossKind.CROSS_ENTROPY:
            out.append(Alternative(
                action="swap_loss", target=component,
                diff={"op": "replace", "path": "/loss/kind",
                      "value": LossKind.CROSS_ENTROPY.value},
                cost="low",
                reason="cross-entropy has no side-channel data requirements",
            ))
        # FIM-specific: suggest tokenizer swap when FIM ids are missing.
        if requirement.origin == "tokenizer" and requirement.key.startswith("FIM_"):
            out.append(Alternative(
                action="swap_tokenizer", target="tokenizer",
                diff={"op": "replace", "path": "/tokenizer/source",
                      "value": "<tokenizer with full FIM trio>"},
                cost="medium",
                reason=f"need a tokenizer that defines {requirement.key}",
            ))

    # Brick-level requirements — drop the brick OR add the missing column.
    if component.startswith("brick:"):
        brick_name = component.split(":", 1)[1]
        if requirement.origin == "parquet":
            out.append(Alternative(
                action="add_column", target=f"parquet:{requirement.key}",
                diff={"op": "add", "path": f"/parquet/columns/{requirement.key}",
                      "value": "<run enrichment pipeline>"},
                cost="high",
                reason=f"parquet shard is missing {requirement.key!r}; "
                       "enrichment scripts live under scripts/nanochat_data/",
            ))
            out.append(Alternative(
                action="drop_brick", target=component,
                diff={"op": "remove", "path": f"/graph/nodes/{brick_name}"},
                cost="medium",
                reason=f"remove brick {brick_name!r} which requires "
                       f"{requirement.key!r}",
            ))

    # Universal escape hatch — only when the requirement is non-required.
    if not requirement.required:
        out.append(Alternative(
            action="relax_requirement", target=component,
            diff={"op": "add", "path": "/probe/allowlist",
                  "value": requirement.key},
            cost="low",
            reason="requirement is soft; explicitly accept missing data",
        ))

    out.sort(key=lambda a: (_COST_ORDER[a.cost], a.action))
    return tuple(out[:_MAX_ALTERNATIVES_PER_FINDING])
