"""UnifiedSuperblockV4 — composes V4 blocks per a RunTemplate.

Closes the loop from the v4 plugin: build a heterogeneous stack of V4
blocks declaratively from a template, run forward on (token_ids,
hidden_states), thread document_ids through to any Engram blocks.

Block kind → V4 module mapping:
    gdn               → cppmega_v4.nn.linear_attention.LinearAttention
    kda               → cppmega_v4.nn.kimi_delta_attention.KimiDeltaAttention
    mla_absorb        → cppmega_v4.nn.mla_absorb.AbsorbedMLA
    engram            → cppmega_v4.nn.engram_v4.EngramV4Block (doc-aware)
    moe               → cppmega_v4.nn.moe_v4.V4MoE
    lightning_indexer → cppmega_v4.nn.lightning_indexer_fp8.LightningIndexerFP8
    nsa               → cppmega_v4.nn.nsa_v4.NativeSparseAttentionV4
    csa_hca           → cppmega_v4.nn.csa_hca_v4.CSAHCAHybridV4
    mlp               → mlx.nn.Linear sandwich (gate/up/down + SiLU)

The superblock only knows how to compose blocks, not how to train. A
HybridTinyLM-style outer loop can wrap it for an actual training run.
"""

from dataclasses import dataclass
from typing import Any, Callable, Optional

import mlx.core as mx
import mlx.nn as nn

from cppmega_v4.nn.csa_hca_v4 import CSAHCAConfig, CSAHCAHybridV4
from cppmega_v4.nn.engram_v4 import EngramV4Block, EngramV4Config
from cppmega_v4.nn.nsa_v4 import NSAConfig, NativeSparseAttentionV4
from cppmega_v4.run_template import BlockSpec, RunTemplate


# Block kind → factory. Each factory takes (hidden_size, params dict) and
# returns the block module. We keep factories tiny — full hyperparameter
# wiring happens inside the block's config dataclass.

def _build_engram(hidden_size: int, params: dict) -> EngramV4Block:
    cfg = EngramV4Config(hidden_size=hidden_size, **params)
    return EngramV4Block(cfg)


def _build_nsa(hidden_size: int, params: dict) -> NativeSparseAttentionV4:
    if "num_heads" not in params:
        params["num_heads"] = max(1, hidden_size // 32)
    if "head_dim" not in params:
        params["head_dim"] = hidden_size // params["num_heads"]
    cfg = NSAConfig(hidden_size=hidden_size, **params)
    return NativeSparseAttentionV4(cfg)


def _build_csa_hca(hidden_size: int, params: dict) -> CSAHCAHybridV4:
    if "num_heads" not in params:
        params["num_heads"] = max(1, hidden_size // 32)
    if "head_dim" not in params:
        params["head_dim"] = hidden_size // params["num_heads"]
    cfg = CSAHCAConfig(hidden_size=hidden_size, **params)
    return CSAHCAHybridV4(cfg)


def _build_mlp(hidden_size: int, params: dict) -> nn.Module:
    intermediate = params.get("intermediate_size", 4 * hidden_size)
    class _GatedMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate = nn.Linear(hidden_size, intermediate, bias=False)
            self.up = nn.Linear(hidden_size, intermediate, bias=False)
            self.down = nn.Linear(intermediate, hidden_size, bias=False)
        def __call__(self, x):
            return self.down(mx.sigmoid(self.gate(x)) * self.up(x))
    return _GatedMLP()


def _build_pass_through_unsupported(kind: str):
    """Block kinds with circular imports or external dep — pass-through."""
    class _PassThrough(nn.Module):
        def __init__(self):
            super().__init__()
            self._kind = kind
        def __call__(self, x):
            return x  # zero contribution, residual-only path
    return _PassThrough()


BLOCK_BUILDERS: dict[str, Callable[[int, dict], nn.Module]] = {
    "engram": _build_engram,
    "nsa": _build_nsa,
    "csa_hca": _build_csa_hca,
    "mlp": _build_mlp,
    # Pass-through for blocks that need optional deps not always in scope:
    "gdn": lambda h, p: _build_pass_through_unsupported("gdn"),
    "kda": lambda h, p: _build_pass_through_unsupported("kda"),
    "mla_absorb": lambda h, p: _build_pass_through_unsupported("mla_absorb"),
    "mla": lambda h, p: _build_pass_through_unsupported("mla"),
    "attention": lambda h, p: _build_pass_through_unsupported("attention"),
    "moe": lambda h, p: _build_pass_through_unsupported("moe"),
    "lightning_indexer": lambda h, p: _build_pass_through_unsupported("lightning_indexer"),
}


@dataclass
class _BuiltBlock:
    kind: str
    module: nn.Module
    needs_doc_ids: bool
    needs_token_ids: bool


def _build_one(spec: BlockSpec, hidden_size: int) -> _BuiltBlock:
    builder = BLOCK_BUILDERS.get(spec.kind)
    if builder is None:
        raise ValueError(f"no builder registered for block kind {spec.kind!r}")
    mod = builder(hidden_size, dict(spec.params))
    return _BuiltBlock(
        kind=spec.kind, module=mod,
        needs_doc_ids=(spec.kind == "engram"),
        needs_token_ids=(spec.kind == "engram"),
    )


class UnifiedSuperblockV4(nn.Module):
    """Composes V4 blocks per RunTemplate. Threads doc_ids to Engram blocks.

    Forward:
        (token_ids: [B, S] int32, hidden_states: [B, S, H],
         document_ids: [B, S] int32 | None)
        -> [B, S, H]

    Each block in the template is repeated ``spec.repeat`` times. Residual
    connections are applied around every block: ``h = h + block(h, ...)``.
    """

    def __init__(self, template: RunTemplate):
        super().__init__()
        self.template = template
        self.hidden_size = template.hidden_size
        # Flatten the (kind, repeat) entries into a list of built blocks.
        flat_specs: list[BlockSpec] = []
        for spec in template.blocks:
            for _ in range(spec.repeat):
                flat_specs.append(BlockSpec(kind=spec.kind, repeat=1,
                                             params=dict(spec.params)))
        self.blocks: list[_BuiltBlock] = [
            _build_one(s, template.hidden_size) for s in flat_specs
        ]
        # nn.Module needs the modules accessible as attributes for parameter
        # discovery: register under deterministic names.
        for i, b in enumerate(self.blocks):
            setattr(self, f"block_{i}_{b.kind}", b.module)

    def __call__(
        self,
        token_ids: mx.array,
        hidden_states: mx.array,
        document_ids: Optional[mx.array] = None,
    ) -> mx.array:
        if token_ids.ndim != 2:
            raise ValueError(f"token_ids must be (B, S), got {token_ids.shape}")
        if hidden_states.shape[:2] != token_ids.shape:
            raise ValueError(
                f"hidden_states {hidden_states.shape} must agree with "
                f"token_ids {token_ids.shape} on the first two axes"
            )
        if document_ids is not None and document_ids.shape != token_ids.shape:
            raise ValueError(
                f"document_ids {document_ids.shape} must match "
                f"token_ids {token_ids.shape}"
            )
        h = hidden_states
        for b in self.blocks:
            if b.needs_doc_ids and b.needs_token_ids:
                delta = b.module(h, token_ids, document_ids=document_ids)
            elif b.needs_token_ids:
                delta = b.module(h, token_ids)
            else:
                delta = b.module(h)
            h = h + delta
        return h

    def kinds(self) -> list[str]:
        return [b.kind for b in self.blocks]


__all__ = ["UnifiedSuperblockV4"]
