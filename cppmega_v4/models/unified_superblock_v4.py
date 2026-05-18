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
    """Last-resort no-op for kinds with no factory yet. Residual-only."""
    class _PassThrough(nn.Module):
        def __init__(self):
            super().__init__()
            self._kind = kind
        def __call__(self, x):
            return x  # zero contribution
    return _PassThrough()


def _build_gdn(hidden_size: int, params: dict) -> nn.Module:
    """Real GatedDeltaNet block (Path A — Path B/C/E land via env override)."""
    from cppmega_v4.nn.linear_attention import (  # local import: avoids circular
        LinearAttentionBlock, LinearAttentionConfig,
    )
    # Defaults sensible for our 1B smoke configs.
    params.setdefault("num_heads", max(1, hidden_size // 64))
    params.setdefault("head_dim", hidden_size // params["num_heads"])
    cfg = LinearAttentionConfig(hidden_size=hidden_size, **params)
    return LinearAttentionBlock(cfg)


def _build_kda(hidden_size: int, params: dict) -> nn.Module:
    """Real Kimi Delta Attention block."""
    from cppmega_v4.nn.kimi_delta_attention import (
        KimiDeltaAttentionBlock, KimiDeltaAttentionConfig,
    )
    params.setdefault("num_heads", max(1, hidden_size // 64))
    params.setdefault("head_dim", hidden_size // params["num_heads"])
    cfg = KimiDeltaAttentionConfig(hidden_size=hidden_size, **params)
    return KimiDeltaAttentionBlock(cfg)


def _build_moe(hidden_size: int, params: dict) -> nn.Module:
    """Real V4 MoE. V4MoE returns MoEOutput — wrap to expose .output."""
    from cppmega_v4.nn.moe_v4 import V4MoE, V4MoEConfig

    params.setdefault("expert_hidden_size", hidden_size * 4)
    cfg = V4MoEConfig(d_model=hidden_size, **params)
    inner = V4MoE(cfg)

    class _MoEWrap(nn.Module):
        def __init__(self):
            super().__init__()
            self.moe = inner

        def __call__(self, x):
            return self.moe(x).output

    return _MoEWrap()


def _build_attention(hidden_size: int, params: dict) -> nn.Module:
    """Standard multi-head self-attention (causal). Used for `attention` and
    as the fallback for `mla` / `mla_absorb` until we land a full MLA block.
    """
    num_heads = params.get("num_heads", max(1, hidden_size // 64))
    head_dim = params.get("head_dim", hidden_size // num_heads)
    norm_eps = params.get("norm_eps", 1e-6)

    class _SelfAttn(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
            self.k_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
            self.v_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
            self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)
            self.norm = nn.RMSNorm(hidden_size, eps=norm_eps)
            # Zero-init out so the block is identity at init.
            self.o_proj.weight = mx.zeros_like(self.o_proj.weight)

        def __call__(self, x):
            B, S, _ = x.shape
            q = self.q_proj(x).reshape(B, S, num_heads, head_dim)
            k = self.k_proj(x).reshape(B, S, num_heads, head_dim)
            v = self.v_proj(x).reshape(B, S, num_heads, head_dim)
            q = mx.transpose(q, (0, 2, 1, 3))
            k = mx.transpose(k, (0, 2, 1, 3))
            v = mx.transpose(v, (0, 2, 1, 3))
            scale = head_dim ** -0.5
            scores = mx.matmul(q, mx.transpose(k, (0, 1, 3, 2))) * scale
            mask = mx.tril(mx.ones((S, S), dtype=mx.bool_))
            scores = mx.where(mask, scores, mx.full(scores.shape, -1e9,
                                                     dtype=scores.dtype))
            w = mx.softmax(scores.astype(mx.float32), axis=-1).astype(scores.dtype)
            o = mx.matmul(w, v)
            o = mx.transpose(o, (0, 2, 1, 3)).reshape(B, S, num_heads * head_dim)
            return self.norm(self.o_proj(o))

    return _SelfAttn()


def _build_lightning_indexer(hidden_size: int, params: dict) -> nn.Module:
    """LightningIndexer is a top-k helper, not a residual block. Wrap as a
    residual no-op for stack composition — the real callsite is inside
    CSA+HCA. RunTemplate users who want the indexer as an inline gate-pre
    pass get a configurable hidden-pass-through wrapper.
    """
    from cppmega_v4.nn.lightning_indexer_fp8 import (
        LightningIndexerFP8, LightningIndexerFP8Config,
    )
    n_heads = params.get("n_heads", max(1, hidden_size // 64))
    cfg = LightningIndexerFP8Config(
        hidden_size=hidden_size,
        n_heads=n_heads,
        head_dim=params.get("head_dim", 32),
        rope_head_dim=params.get("rope_head_dim", 16),
        q_lora_rank=params.get("q_lora_rank", hidden_size),
        index_topk=params.get("index_topk", 64),
        fp8_blocks=params.get("fp8_blocks", True),
    )
    indexer = LightningIndexerFP8(cfg)

    class _IndexerResidualNoOp(nn.Module):
        """Residual pass-through wrapper for LightningIndexer.

        Lightning Indexer's natural output is top-k indices (int32), not a
        residual. In a RunTemplate context, the block contributes zero
        delta — its presence in the stack signals that downstream
        CSA/HCA / sparse-MLA layers should consume the indexer's outputs.
        Real wiring lives in CSA+HCA's select_indices argument.
        """
        def __init__(self):
            super().__init__()
            self.indexer = indexer

        def __call__(self, x):
            return mx.zeros_like(x)

    return _IndexerResidualNoOp()


BLOCK_BUILDERS: dict[str, Callable[[int, dict], nn.Module]] = {
    "engram": _build_engram,
    "nsa": _build_nsa,
    "csa_hca": _build_csa_hca,
    "mlp": _build_mlp,
    "gdn": _build_gdn,
    "kda": _build_kda,
    "moe": _build_moe,
    "attention": _build_attention,
    # mla / mla_absorb fall back to standard attention until we land a
    # full MLA block (mla_absorb.py is a pure algebra module, not nn.Module).
    "mla": _build_attention,
    "mla_absorb": _build_attention,
    "lightning_indexer": _build_lightning_indexer,
}


@dataclass
class _BuiltBlock:
    kind: str
    module: nn.Module
    needs_doc_ids: bool
    needs_token_ids: bool


_DOC_ID_KW_KINDS = {"gdn", "kda"}     # accept doc_ids via *kwargs
_TOKEN_ID_POS_KINDS = {"engram"}      # take token_ids positional


def _build_one(spec: BlockSpec, hidden_size: int) -> _BuiltBlock:
    builder = BLOCK_BUILDERS.get(spec.kind)
    if builder is None:
        raise ValueError(f"no builder registered for block kind {spec.kind!r}")
    mod = builder(hidden_size, dict(spec.params))
    return _BuiltBlock(
        kind=spec.kind, module=mod,
        needs_doc_ids=(spec.kind in _DOC_ID_KW_KINDS or spec.kind == "engram"),
        needs_token_ids=(spec.kind in _TOKEN_ID_POS_KINDS),
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
            if b.needs_token_ids and b.needs_doc_ids:
                # Engram: positional token_ids + keyword document_ids.
                delta = b.module(h, token_ids, document_ids=document_ids)
            elif b.needs_token_ids:
                delta = b.module(h, token_ids)
            elif b.needs_doc_ids:
                # GDN/KDA: doc_ids keyword (block uses it for doc-reset).
                delta = b.module(h, doc_ids=document_ids)
            else:
                delta = b.module(h)
            h = h + delta
        return h

    def kinds(self) -> list[str]:
        return [b.kind for b in self.blocks]


__all__ = ["UnifiedSuperblockV4"]
