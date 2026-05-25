"""Symbolic shape contracts for V4 bricks.

A :class:`ShapeExpr` is a tuple of textual expressions over a fixed
vocabulary of named dimensions (``B`` batch, ``S`` seq, ``H`` hidden,
``nh`` num attention heads, ``nkv`` num key/value heads, ``head_dim``,
``kv_lora_rank``, ``qk_rope_head_dim``, ``qk_nope_head_dim``,
``v_head_dim``, ``num_experts``, ``top_k``, ``fine_window``,
``coarse_block_size``, ``sliding_window_size``). Each expression is a
Python arithmetic snippet that evaluates against a ``dim_env`` dict of
ints; missing names yield a :class:`ResolveError`.

The contract per brick declares:
  - what positional inputs the brick consumes (by name → shape)
  - what positional outputs it produces
  - symbolic byte footprints for params / activations / KV-cache
  - the set of side-channel inputs it needs (``"doc_ids"``,
    ``"kv_cache"``, ``"token_ids"``)
  - whether shape is opaque (data-dependent: sparse attention, MoE
    post-route token-disjoint slabs). Opaque means: producers must hand
    over ``(B, S, H)`` and consumers must accept ``(B, S, H)`` — we
    trust the brick to preserve it internally without auditing.

Stage A scope: contracts only. Resolution (:func:`ShapeExpr.resolve`)
plus the per-brick registry. The resolver / planner (Stage B+) reads
these without modifying them.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Final


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class ResolveError(ValueError):
    """Raised when a ShapeExpr can't be evaluated under a dim_env."""


# ---------------------------------------------------------------------------
# Symbolic shape expression
# ---------------------------------------------------------------------------


_ALLOWED_NAMES: Final[frozenset[str]] = frozenset({
    # Universal model dims
    "B", "S", "H",
    # Attention head topology
    "nh", "nkv", "head_dim",
    # MLA-style LoRA splits
    "q_lora_rank", "kv_lora_rank",
    "qk_rope_head_dim", "qk_nope_head_dim", "v_head_dim",
    # MoE
    "num_experts", "top_k", "expert_dim",
    # ZAYA1 / Gemma sliding params
    "fine_window", "coarse_block_size", "sliding_window_size",
    # SSM (mamba3)
    "d_state", "d_conv", "expand",
    # Drafter / MTP
    "k_predict",
})


@dataclass(frozen=True)
class ShapeExpr:
    """A tuple of textual expressions over :data:`_ALLOWED_NAMES`.

    Examples::

        ShapeExpr(("B", "S", "H"))                    # standard
        ShapeExpr(("B", "S", "nh*head_dim"))          # heads-flattened
        ShapeExpr(("B", "nh", "S", "head_dim"))       # heads-major
        ShapeExpr(("B", "S", "nh*(qk_nope_head_dim+qk_rope_head_dim)"))

    The expressions are pure arithmetic — no calls, no attribute
    lookups, no comprehensions. Evaluation is sandboxed (``eval`` with
    empty globals + the dim_env dict).
    """

    dims: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.dims, tuple):
            raise TypeError(
                f"ShapeExpr.dims must be tuple of str, got {type(self.dims).__name__}"
            )
        for d in self.dims:
            if not isinstance(d, str) or not d.strip():
                raise ValueError(
                    f"ShapeExpr.dims entries must be non-empty str, got {d!r}"
                )

    @property
    def rank(self) -> int:
        return len(self.dims)

    def free_names(self) -> frozenset[str]:
        """Return the set of named-dim references used by this expression.

        Only walks ``_ALLOWED_NAMES`` — unknown identifiers raise on
        resolve. The walker is intentionally simple (substring match
        against allowed names); good enough for the arithmetic-only
        subset we support."""
        found: set[str] = set()
        for d in self.dims:
            for name in _ALLOWED_NAMES:
                # word-boundary check via character-class trick: surround
                # the expression with non-word chars so single-letter names
                # ("B", "S", "H") don't match inside "head_dim".
                padded = f" {d} "
                idx = padded.find(name)
                while idx >= 0:
                    left = padded[idx - 1]
                    right = padded[idx + len(name)]
                    if not (left.isalnum() or left == "_") and not (
                        right.isalnum() or right == "_"
                    ):
                        found.add(name)
                        break
                    idx = padded.find(name, idx + 1)
        return frozenset(found)

    def resolve(self, env: Mapping[str, int]) -> tuple[int, ...]:
        """Evaluate every dim under ``env``. Raises :class:`ResolveError`
        on unknown identifier or non-int result."""
        used = self.free_names()
        missing = used - env.keys()
        if missing:
            raise ResolveError(
                f"missing dim_env entries: {sorted(missing)}; "
                f"expr={self.dims!r}, provided={sorted(env.keys())}"
            )
        # Constrain eval to allowed names only — defence in depth in case
        # an expression slipped a foreign identifier past free_names().
        scope = {name: env[name] for name in used}
        resolved: list[int] = []
        for d in self.dims:
            try:
                # Empty globals (no builtins) + named-dims-only locals.
                val = eval(d, {"__builtins__": {}}, scope)  # noqa: S307
            except Exception as exc:  # pragma: no cover - defensive
                raise ResolveError(
                    f"failed to evaluate dim {d!r} with env={scope}: {exc}"
                ) from exc
            if not isinstance(val, int) or isinstance(val, bool):
                raise ResolveError(
                    f"dim {d!r} resolved to non-int {val!r} (type={type(val).__name__})"
                )
            if val <= 0:
                raise ResolveError(
                    f"dim {d!r} resolved to non-positive {val!r}"
                )
            resolved.append(val)
        return tuple(resolved)


# ---------------------------------------------------------------------------
# Brick shape contract
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BrickShapeContract:
    """Static shape + memory contract for one brick kind.

    All bytes-valued ShapeExprs evaluate to a single integer (one-element
    "shape"). Conventionally the byte expressions are written as scalar
    arithmetic, e.g. ``ShapeExpr(("4 * H * H",))`` for a 4×H² Linear
    in bf16 (the ×2 dtype factor is supplied by the caller).
    """

    inputs: dict[str, ShapeExpr]
    outputs: dict[str, ShapeExpr]
    params_elems: ShapeExpr           # element count (multiply by dtype_bytes)
    activations_elems: ShapeExpr      # peak forward activations, elements
    kv_cache_elems: ShapeExpr         # decode-only; 0 for non-attention
    needs: frozenset[str] = field(default_factory=frozenset)
    opaque_shape: bool = False
    description: str = ""

    def __post_init__(self) -> None:
        if not self.inputs:
            raise ValueError("BrickShapeContract.inputs must not be empty")
        if not self.outputs:
            raise ValueError("BrickShapeContract.outputs must not be empty")
        for name, expr in (*self.inputs.items(), *self.outputs.items()):
            if not isinstance(expr, ShapeExpr):
                raise TypeError(
                    f"contract field {name!r} must be ShapeExpr, "
                    f"got {type(expr).__name__}"
                )
        for byte_field, expr in (
            ("params_elems", self.params_elems),
            ("activations_elems", self.activations_elems),
            ("kv_cache_elems", self.kv_cache_elems),
        ):
            if not isinstance(expr, ShapeExpr):
                raise TypeError(
                    f"contract field {byte_field!r} must be ShapeExpr, "
                    f"got {type(expr).__name__}"
                )
            if expr.rank != 1:
                raise ValueError(
                    f"{byte_field!r} must be rank-1 ShapeExpr (scalar elem "
                    f"count), got rank={expr.rank}"
                )


# ---------------------------------------------------------------------------
# Per-kind contracts
# ---------------------------------------------------------------------------


def _bsh() -> ShapeExpr:
    """Shorthand: standard ``(B, S, H)`` activation tensor."""
    return ShapeExpr(("B", "S", "H"))


def _zero() -> ShapeExpr:
    return ShapeExpr(("0",))


# --- norm / projection family --------------------------------------------

_MLP_CONTRACT = BrickShapeContract(
    inputs={"x": _bsh()},
    outputs={"y": _bsh()},
    # SwiGLU-style: gate + up + down ≈ 3 * H * (8/3 * H) ≈ 8 * H^2
    params_elems=ShapeExpr(("8 * H * H",)),
    activations_elems=ShapeExpr(("B * S * 4 * H",)),
    kv_cache_elems=_zero(),
    description="MLP / SwiGLU FFN — pointwise, fuses with neighbours",
)


_ENGRAM_CONTRACT = BrickShapeContract(
    inputs={"x": _bsh()},
    outputs={"y": _bsh()},
    params_elems=ShapeExpr(("2 * H * H",)),
    activations_elems=ShapeExpr(("B * S * H",)),
    kv_cache_elems=_zero(),
    needs=frozenset({"token_ids"}),
    description="Engram retrieval block (norm_or_proj category)",
)


_LIGHTNING_INDEXER_CONTRACT = BrickShapeContract(
    inputs={"x": _bsh()},
    outputs={"y": _bsh()},
    params_elems=ShapeExpr(("4 * H * H",)),
    activations_elems=ShapeExpr(("B * S * H",)),
    kv_cache_elems=_zero(),
    description="Lightning indexer + CSA/HCA bundle",
)


# --- standard SDPA attention family --------------------------------------

def _SDPA_BASE(extra_params="0"):
    return BrickShapeContract(
    inputs={"x": _bsh()},
    outputs={"y": _bsh()},
    # q + k + v + o projections; GQA shrinks kv to nkv heads
    params_elems=ShapeExpr((
        f"H * nh * head_dim + 2 * H * nkv * head_dim + nh * head_dim * H + ({extra_params})",
    )),
    activations_elems=ShapeExpr(("B * S * nh * head_dim",)),
    # KV-cache: 2 (k,v) × nkv × head_dim per token
    kv_cache_elems=ShapeExpr(("2 * B * S * nkv * head_dim",)),
    description="GQA SDPA attention",
)


_ATTENTION_CONTRACT = _SDPA_BASE()
_GATED_ATTENTION_CONTRACT = _SDPA_BASE(extra_params="nh * head_dim")  # gate proj
_GQA_SLIDING_CONTRACT = _SDPA_BASE()


# --- CCA (ZAYA1) — coarse causal attention ------------------------------

_CCA_CONTRACT = BrickShapeContract(
    inputs={"x": _bsh()},
    outputs={"y": _bsh()},
    params_elems=ShapeExpr((
        "H * nh * head_dim + 2 * H * nkv * head_dim + nh * head_dim * H",
    )),
    activations_elems=ShapeExpr((
        "B * S * nh * head_dim + B * (S // coarse_block_size) * nkv * head_dim",
    )),
    kv_cache_elems=ShapeExpr(("2 * B * S * nkv * head_dim",)),
    description="ZAYA1 Coarse Causal Attention — fine + coarse stream",
)


# --- MLA family (LoRA Q + LoRA KV + RoPE on rope-only split) -------------

_MLA_CONTRACT = BrickShapeContract(
    inputs={"x": _bsh()},
    outputs={"y": _bsh()},
    params_elems=ShapeExpr((
        # q_a + q_b + kv_a + kv_b + o
        "H * q_lora_rank "
        "+ q_lora_rank * nh * (qk_nope_head_dim + qk_rope_head_dim) "
        "+ H * (kv_lora_rank + qk_rope_head_dim) "
        "+ kv_lora_rank * nh * (qk_nope_head_dim + v_head_dim) "
        "+ nh * v_head_dim * H",
    )),
    activations_elems=ShapeExpr((
        "B * S * nh * (qk_nope_head_dim + qk_rope_head_dim + v_head_dim)",
    )),
    # MLA stores latent KV + rope-K only
    kv_cache_elems=ShapeExpr((
        "B * S * (kv_lora_rank + qk_rope_head_dim)",
    )),
    description="MLA — LoRA Q + LoRA KV + RoPE split",
)

# mla_absorb shares the contract — it's the same block w/ absorb fast-path.
_MLA_ABSORB_CONTRACT = _MLA_CONTRACT


_MISTRAL4_MLA_CONTRACT = BrickShapeContract(
    inputs={"x": _bsh()},
    outputs={"y": _bsh()},
    params_elems=_MLA_CONTRACT.params_elems,
    activations_elems=_MLA_CONTRACT.activations_elems,
    # Mistral 4 uses INT4-quantised latent cache → 0.5 byte/element later;
    # element count is the same as MLA, the dtype_bytes factor drops it.
    kv_cache_elems=_MLA_CONTRACT.kv_cache_elems,
    description="Mistral Small 4 MLA absorbed + INT4 latent cache",
)


_BAILING_MLA_CONTRACT = _MLA_CONTRACT


_DSV4_ATTENTION_CONTRACT = BrickShapeContract(
    inputs={"x": _bsh()},
    outputs={"y": _bsh()},
    params_elems=ShapeExpr((
        "H * nh * head_dim + 2 * H * nkv * head_dim + nh * head_dim * H",
    )),
    activations_elems=ShapeExpr(("B * S * nh * head_dim",)),
    kv_cache_elems=ShapeExpr(("2 * B * S * nkv * head_dim",)),
    opaque_shape=True,
    description="DeepSeek V4 Flash — hash-indexed sparse MLA (data-dependent)",
)


# --- Linear-attention family (GDN / KDA / Bailing Linear) ----------------

_LINEAR_ATTN_CONTRACT = BrickShapeContract(
    inputs={"x": _bsh()},
    outputs={"y": _bsh()},
    # Q + K + V + Out projections + state RMS-norm scales
    params_elems=ShapeExpr((
        "3 * H * nh * head_dim + nh * head_dim * H + 2 * nh * head_dim",
    )),
    # recurrent state across heads
    activations_elems=ShapeExpr(("B * nh * head_dim * head_dim + B * S * H",)),
    kv_cache_elems=_zero(),  # linear-attn carries recurrent state, not KV
    needs=frozenset({"doc_ids"}),
    description="Linear-attn (GDN / KDA / Bailing linear scan)",
)


# --- SSM / Mamba-3 -------------------------------------------------------

_MAMBA3_CONTRACT = BrickShapeContract(
    inputs={"x": _bsh()},
    outputs={"y": _bsh()},
    # in_proj + conv + ssm dt/A/D + out_proj — rough estimate
    params_elems=ShapeExpr(("8 * H * H",)),
    # chunkwise scan buffer
    activations_elems=ShapeExpr(("B * S * H + B * H * d_state",)),
    kv_cache_elems=_zero(),
    description="Mamba-3 SSM reference block",
)


# --- Nonlinear RNN (M2RNN) ----------------------------------------------

_M2RNN_CONTRACT = BrickShapeContract(
    inputs={"x": _bsh()},
    outputs={"y": _bsh()},
    params_elems=ShapeExpr(("4 * H * H",)),
    activations_elems=ShapeExpr(("B * S * H + B * H",)),
    kv_cache_elems=_zero(),
    description="M2RNN nonlinear tanh recurrence",
)


# --- MoE family ----------------------------------------------------------

_MOE_CONTRACT = BrickShapeContract(
    inputs={"x": _bsh()},
    outputs={"y": _bsh()},
    # Router (H * num_experts) + per-expert SwiGLU (8 * H^2 each)
    params_elems=ShapeExpr((
        "H * num_experts + num_experts * 8 * H * H",
    )),
    # Routed activations: top_k experts process the token
    activations_elems=ShapeExpr(("B * S * top_k * 4 * H",)),
    kv_cache_elems=_zero(),
    opaque_shape=False,
    description="MoE block — router + experts FFN",
)


_BAILING_MOE_CONTRACT = _MOE_CONTRACT


# --- Sparse attention (NSA / CSA/HCA) -----------------------------------

_SPARSE_ATTN_CONTRACT = BrickShapeContract(
    inputs={"x": _bsh()},
    outputs={"y": _bsh()},
    params_elems=ShapeExpr((
        "H * nh * head_dim + 2 * H * nkv * head_dim + nh * head_dim * H",
    )),
    activations_elems=ShapeExpr(("B * S * nh * head_dim",)),
    kv_cache_elems=ShapeExpr(("2 * B * S * nkv * head_dim",)),
    opaque_shape=True,
    description="Sparse attention (NSA / CSA/HCA) — data-dependent indexing",
)


# --- Cross-attention drafter / MTP --------------------------------------

_GEMMA4_DRAFTER_CONTRACT = BrickShapeContract(
    inputs={"x": _bsh()},
    outputs={"y": _bsh()},
    params_elems=ShapeExpr((
        "2 * H * H + H * nh * head_dim + H * nkv * head_dim * 2 + nh * head_dim * H",
    )),
    activations_elems=ShapeExpr(("B * S * nh * head_dim",)),
    kv_cache_elems=ShapeExpr(("2 * B * S * nkv * head_dim",)),
    description="Gemma 4 MTP-drafter cross-attn decoder layer",
)


_NEMOTRON_MTP_CONTRACT = BrickShapeContract(
    inputs={"x": _bsh()},
    outputs={"y": _bsh()},
    params_elems=ShapeExpr(("4 * H * H",)),
    activations_elems=ShapeExpr(("B * S * k_predict * H",)),
    kv_cache_elems=_zero(),
    description="Nemotron-H Multi-Token-Prediction block",
)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


_CONTRACTS: dict[str, BrickShapeContract] = {
    # norm / projection
    "mlp": _MLP_CONTRACT,
    "engram": _ENGRAM_CONTRACT,
    "lightning_indexer": _LIGHTNING_INDEXER_CONTRACT,
    # SDPA attention
    "attention": _ATTENTION_CONTRACT,
    "gated_attention": _GATED_ATTENTION_CONTRACT,
    "gqa_sliding": _GQA_SLIDING_CONTRACT,
    # MLA / latent attention
    "mla": _MLA_CONTRACT,
    "mla_absorb": _MLA_ABSORB_CONTRACT,
    "mistral4_mla": _MISTRAL4_MLA_CONTRACT,
    "bailing_mla": _BAILING_MLA_CONTRACT,
    "dsv4_attention": _DSV4_ATTENTION_CONTRACT,
    # ZAYA1
    "cca_attention": _CCA_CONTRACT,
    # linear-attn family
    "gdn": _LINEAR_ATTN_CONTRACT,
    "kda": _LINEAR_ATTN_CONTRACT,
    "bailing_linear": _LINEAR_ATTN_CONTRACT,
    # SSM
    "mamba3": _MAMBA3_CONTRACT,
    # MoE
    "moe": _MOE_CONTRACT,
    "bailing_moe": _BAILING_MOE_CONTRACT,
    # sparse attention
    "nsa": _SPARSE_ATTN_CONTRACT,
    "csa_hca": _SPARSE_ATTN_CONTRACT,
    # cross-attn / MTP
    "gemma4_drafter": _GEMMA4_DRAFTER_CONTRACT,
    "nemotron_h_mtp": _NEMOTRON_MTP_CONTRACT,
    # GalCov-B bricks
    "mlstm": BrickShapeContract(
        inputs={"x": _bsh()},
        outputs={"y": _bsh()},
        params_elems=ShapeExpr(("7 * H * head_dim",)),
        activations_elems=ShapeExpr(("B * S * H + B * head_dim * head_dim",)),
        kv_cache_elems=_zero(),
        description="xLSTM matrix-LSTM block (no self-attention)",
    ),
    "abs_pos_embed": BrickShapeContract(
        inputs={"x": _bsh()},
        outputs={"y": _bsh()},
        params_elems=ShapeExpr(("4096 * H",)),
        activations_elems=ShapeExpr(("B * S * H",)),
        kv_cache_elems=_zero(),
        description="GPT-2-style learned absolute positional embedding",
    ),
    "per_layer_embed": BrickShapeContract(
        inputs={"x": _bsh()},
        outputs={"y": _bsh()},
        params_elems=ShapeExpr(("2 * H",)),
        activations_elems=ShapeExpr(("B * S * H",)),
        kv_cache_elems=_zero(),
        description="Per-layer scaled embedding (Gemma 4 E2B/E4B)",
    ),
    "embedding_table": BrickShapeContract(
        inputs={"x": _bsh()},
        outputs={"y": _bsh()},
        params_elems=ShapeExpr(("65536 * H",)),
        activations_elems=ShapeExpr(("B * S * H",)),
        kv_cache_elems=_zero(),
        description="Standard learned embedding / output de-embedding projection table",
    ),
    "rmsnorm": BrickShapeContract(
        inputs={"x": _bsh()},
        outputs={"y": _bsh()},
        params_elems=ShapeExpr(("H",)),
        activations_elems=ShapeExpr(("B * S * H",)),
        kv_cache_elems=_zero(),
        description="Root-Mean-Square Normalization layer",
    ),
    "layernorm": BrickShapeContract(
        inputs={"x": _bsh()},
        outputs={"y": _bsh()},
        params_elems=ShapeExpr(("2 * H",)),
        activations_elems=ShapeExpr(("B * S * H",)),
        kv_cache_elems=_zero(),
        description="Layer Normalization layer (scale + bias)",
    ),
    "residual": BrickShapeContract(
        inputs={"x": _bsh()},
        outputs={"y": _bsh()},
        params_elems=_zero(),
        activations_elems=_zero(),
        kv_cache_elems=_zero(),
        description="Residual skip-connection addition / identity passthrough",
    ),
}


def contract_for(kind: str) -> BrickShapeContract:
    """Return the shape contract for a brick kind.

    Raises :class:`KeyError` for unknown kinds — every kind in
    :data:`cppmega_v4.models.unified_superblock_v4.BLOCK_BUILDERS`
    is required to have a contract (the CI test
    ``test_every_block_builder_kind_has_contract`` enforces this).
    """
    try:
        return _CONTRACTS[kind]
    except KeyError as exc:
        raise KeyError(
            f"no shape contract registered for kind={kind!r}; "
            "register one in cppmega_v4/spec/shape_contract.py"
        ) from exc


def register_contract(kind: str, contract: BrickShapeContract) -> None:
    """Register / override a contract. Used by adapters and tests."""
    if not isinstance(contract, BrickShapeContract):
        raise TypeError(
            f"contract must be BrickShapeContract, got {type(contract).__name__}"
        )
    _CONTRACTS[kind] = contract


def registered_kinds() -> tuple[str, ...]:
    return tuple(sorted(_CONTRACTS.keys()))


def compatible_edges() -> tuple[tuple[str, str], ...]:
    """V7-E-AUDIT-02: enumerate (src_kind, dst_kind) pairs that the
    shape-contract layer considers well-typed.

    Rule: edge is compatible iff src declares at least one output AND
    dst declares at least one input. cppmega bricks use a canonical
    ``x → y`` channel naming convention; every transformer-block brick
    can feed every other at the contract layer. Numerical compatibility
    (dim_env, head_dim, ...) is enforced separately by
    ``verify_and_estimate``; this helper answers the UI's coarse
    'can I drop an edge here' question to reject obvious mis-wiring
    (e.g. dropping an edge into a brick with no inputs).
    Opaque-shape bricks are universally compatible.
    """
    pairs: list[tuple[str, str]] = []
    kinds = sorted(_CONTRACTS.keys())
    for src in kinds:
        src_c = _CONTRACTS[src]
        for dst in kinds:
            dst_c = _CONTRACTS[dst]
            if src_c.opaque_shape or dst_c.opaque_shape:
                pairs.append((src, dst))
                continue
            if src_c.outputs and dst_c.inputs:
                pairs.append((src, dst))
    return tuple(pairs)


__all__ = [
    "BrickShapeContract",
    "ResolveError",
    "ShapeExpr",
    "compatible_edges",
    "contract_for",
    "register_contract",
    "registered_kinds",
]
