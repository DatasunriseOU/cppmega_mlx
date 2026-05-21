"""Dim auto-adjust feedback log (E7-2).

Walks a BrickGraph + dim_env and synthesises an :class:`InferenceEntry`
per (brick, parameter) row describing whether the value was provided by
the user (source='user') or auto-derived from dim_env (source='auto'),
plus a one-line ``reason`` like ``"H=128/head_dim=64 → 2"``.

Renders in the Dimensions sidebar tab. Click a row → highlights the
node on the canvas.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


@dataclass(frozen=True)
class InferenceEntry:
    brick: str
    param: str
    value: Any
    source: Literal["user", "auto"]
    reason: str


def _attention_entries(name: str, params: dict, dim_env: dict) -> list[InferenceEntry]:
    H = dim_env.get("H", 128)
    head_dim = params.get("head_dim", dim_env.get("head_dim", 64))
    out: list[InferenceEntry] = []

    if "head_dim" in params:
        out.append(InferenceEntry(
            brick=name, param="head_dim", value=params["head_dim"],
            source="user", reason="provided in BrickSpec.params",
        ))
    else:
        out.append(InferenceEntry(
            brick=name, param="head_dim", value=head_dim,
            source="auto",
            reason=f"default from dim_env.head_dim={dim_env.get('head_dim', 64)}",
        ))

    if "num_heads" in params:
        out.append(InferenceEntry(
            brick=name, param="num_heads", value=params["num_heads"],
            source="user", reason="provided in BrickSpec.params",
        ))
    else:
        derived = max(1, H // max(1, head_dim))
        out.append(InferenceEntry(
            brick=name, param="num_heads", value=derived,
            source="auto",
            reason=f"H={H}/head_dim={head_dim} → {derived}",
        ))
    return out


def _mlp_entries(name: str, params: dict, dim_env: dict) -> list[InferenceEntry]:
    H = dim_env.get("H", 128)
    out: list[InferenceEntry] = []
    if "intermediate_size" in params:
        out.append(InferenceEntry(
            brick=name, param="intermediate_size",
            value=params["intermediate_size"],
            source="user", reason="provided in BrickSpec.params",
        ))
    else:
        out.append(InferenceEntry(
            brick=name, param="intermediate_size", value=4 * H,
            source="auto", reason=f"4 * H ({H}) = {4 * H}",
        ))
    if "activation" in params:
        out.append(InferenceEntry(
            brick=name, param="activation", value=params["activation"],
            source="user", reason="set via BrickContextPanel",
        ))
    else:
        out.append(InferenceEntry(
            brick=name, param="activation", value="glu",
            source="auto",
            reason="GLU default preserves legacy sigmoid(gate)*up math",
        ))
    return out


def _moe_entries(name: str, params: dict, dim_env: dict) -> list[InferenceEntry]:
    out: list[InferenceEntry] = []
    if "num_experts" in params:
        out.append(InferenceEntry(
            brick=name, param="num_experts", value=params["num_experts"],
            source="user", reason="provided in BrickSpec.params",
        ))
    else:
        ne = dim_env.get("num_experts", 4)
        out.append(InferenceEntry(
            brick=name, param="num_experts", value=ne,
            source="auto", reason=f"from dim_env.num_experts={ne}",
        ))
    if "top_k" in params:
        out.append(InferenceEntry(
            brick=name, param="top_k", value=params["top_k"],
            source="user", reason="provided in BrickSpec.params",
        ))
    else:
        tk = dim_env.get("top_k", 2)
        out.append(InferenceEntry(
            brick=name, param="top_k", value=tk,
            source="auto", reason=f"from dim_env.top_k={tk}",
        ))
    return out


_ATTN_KINDS = frozenset({
    "attention", "gated_attention", "mla", "mla_absorb",
    "mistral4_mla", "dsv4_attention", "gqa_sliding", "cca_attention",
    "gdn", "kda", "nsa",
})
_MLP_KINDS = frozenset({"mlp", "gated_mlp"})
_MOE_KINDS = frozenset({"moe", "bailing_moe"})


def build_inference_log(
    graph: dict,
    dim_env: dict,
) -> list[InferenceEntry]:
    """Build the full inference log for a graph + dim_env pair.

    ``graph`` is the wire-form dict with ``nodes: [{id, kind, params}, …]``.
    """
    entries: list[InferenceEntry] = []
    for node in graph.get("nodes", []):
        kind = node.get("kind")
        name = node.get("id") or node.get("name") or "<unnamed>"
        params = node.get("params", {}) or {}
        if kind in _ATTN_KINDS:
            entries.extend(_attention_entries(name, params, dim_env))
        elif kind in _MLP_KINDS:
            entries.extend(_mlp_entries(name, params, dim_env))
        elif kind in _MOE_KINDS:
            entries.extend(_moe_entries(name, params, dim_env))
        # Other brick kinds: emit one stub row per provided param so the
        # Dimensions tab still shows them.
        else:
            for k, v in params.items():
                entries.append(InferenceEntry(
                    brick=name, param=k, value=v,
                    source="user", reason="provided in BrickSpec.params",
                ))
    return entries


__all__ = ["InferenceEntry", "build_inference_log"]
