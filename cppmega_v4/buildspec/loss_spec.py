"""LossSpec — declarative loss-function specification.

A :class:`LossSpec` carries:
  - the kind of loss (CE / MTP-weighted / IFIM-shaped / MHC-attn-bias /
    custom)
  - free-form numeric params keyed by string
  - the head output names the loss reads from
  - the label source ("next_token", "next_k_tokens", "doc_ids")
  - reduction strategy ("mean" | "sum" | "none")

Builders in :data:`LOSS_BUILTINS` (and the free-standing :func:`cross_entropy_loss`,
:func:`mtp_weighted_loss`, :func:`ifim_shaped_loss` helpers) cover the
patterns the gallery actually uses. Custom callables are wrapped via
``LossSpec(kind=CUSTOM, ...)``; the actual function is plugged in at
build time (Stage E) via a side-channel dict.

Pure data — no MLX runtime. Validation lives entirely in
``__post_init__``; CI tests cover every rejection branch.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Final


class LossKind(str, Enum):
    """Recognised loss families. Strings so JSON-friendly."""

    CROSS_ENTROPY = "cross_entropy"
    MTP_WEIGHTED  = "mtp_weighted"
    IFIM_SHAPED   = "ifim_shaped"
    MHC_ATTN_BIAS = "mhc_attn_bias"
    CUSTOM        = "custom"


_VALID_REDUCTIONS: Final[frozenset[str]] = frozenset({"mean", "sum", "none"})
_VALID_LABEL_SOURCES: Final[frozenset[str]] = frozenset({
    "next_token", "next_k_tokens", "doc_ids", "custom",
})


@dataclass(frozen=True)
class LossSpec:
    """Declarative loss specification.

    Fields:
      kind: one of :class:`LossKind`.
      params: numeric params (per-kind contract — MTP wants ``k`` and
        ``beta_0..beta_{k-1}``; IFIM wants ``lambda_fim``; etc.).
      head_outputs: tuple of brick output names the loss reads. After
        :func:`apply_rewrites` runs (Stage B+), these must reference
        nodes that exist in the rewritten graph.
      label_source: how the supervision labels are constructed
        (``"next_token"`` for standard CE, ``"next_k_tokens"`` for MTP,
        ``"doc_ids"`` for doc-conditioned bricks).
      reduction: ``"mean"`` (default) | ``"sum"`` | ``"none"``.
    """

    kind: LossKind
    params: Mapping[str, float] = field(default_factory=dict)
    head_outputs: tuple[str, ...] = ()
    label_source: str = "next_token"
    reduction: str = "mean"

    def __post_init__(self) -> None:
        if not isinstance(self.kind, LossKind):
            raise TypeError(
                f"LossSpec.kind must be LossKind, got {type(self.kind).__name__}"
            )
        if not self.head_outputs:
            raise ValueError(
                "LossSpec.head_outputs must not be empty — at least one "
                "brick output name is required"
            )
        for name in self.head_outputs:
            if not isinstance(name, str) or not name.strip():
                raise ValueError(
                    f"LossSpec.head_outputs entries must be non-empty str, "
                    f"got {name!r}"
                )
        if self.label_source not in _VALID_LABEL_SOURCES:
            raise ValueError(
                f"LossSpec.label_source={self.label_source!r} not in "
                f"{sorted(_VALID_LABEL_SOURCES)}"
            )
        if self.reduction not in _VALID_REDUCTIONS:
            raise ValueError(
                f"LossSpec.reduction={self.reduction!r} not in "
                f"{sorted(_VALID_REDUCTIONS)}"
            )
        # Per-kind sanity
        if self.kind is LossKind.MTP_WEIGHTED:
            k = self.params.get("k")
            if not isinstance(k, (int, float)) or int(k) < 1:
                raise ValueError(
                    "MTP_WEIGHTED requires params['k'] ≥ 1; "
                    f"got {k!r}"
                )
            if len(self.head_outputs) != int(k):
                raise ValueError(
                    f"MTP_WEIGHTED: head_outputs length ({len(self.head_outputs)}) "
                    f"must equal params['k'] ({int(k)})"
                )
            for i in range(int(k)):
                key = f"beta_{i}"
                if key not in self.params:
                    raise ValueError(
                        f"MTP_WEIGHTED missing required param {key!r}"
                    )
                if self.params[key] < 0:
                    raise ValueError(
                        f"MTP_WEIGHTED params[{key!r}] must be ≥ 0, "
                        f"got {self.params[key]!r}"
                    )
        elif self.kind is LossKind.IFIM_SHAPED:
            lam = self.params.get("lambda_fim")
            if lam is None or lam < 0:
                raise ValueError(
                    "IFIM_SHAPED requires params['lambda_fim'] ≥ 0; "
                    f"got {lam!r}"
                )
        elif self.kind is LossKind.MHC_ATTN_BIAS:
            lam = self.params.get("lambda_mhc")
            if lam is None or lam < 0:
                raise ValueError(
                    "MHC_ATTN_BIAS requires params['lambda_mhc'] ≥ 0; "
                    f"got {lam!r}"
                )


# ---------------------------------------------------------------------------
# Built-in factories
# ---------------------------------------------------------------------------


def cross_entropy_loss(head_output_name: str = "logits") -> LossSpec:
    """Standard next-token cross-entropy."""
    return LossSpec(
        kind=LossKind.CROSS_ENTROPY,
        params={},
        head_outputs=(head_output_name,),
        label_source="next_token",
    )


def mtp_weighted_loss(
    k: int = 2,
    beta: tuple[float, ...] | None = None,
    head_output_prefix: str = "logits",
) -> LossSpec:
    """Multi-Token Prediction weighted CE.

    Generates ``k`` head outputs named ``{prefix}_0 ... {prefix}_{k-1}``;
    each head predicts ``label_t+i``. The aggregate loss is
    ``Σ_i β_i * CE(head_i, label_t+i)``. When ``beta`` is None, defaults
    to ``(1.0, 0.6, 0.4, 0.3, 0.2, ...)`` (geometric-ish decay).
    """
    if k < 1:
        raise ValueError(f"k must be ≥ 1, got {k}")
    if beta is None:
        beta = tuple(1.0 if i == 0 else 0.6 ** i for i in range(k))
    if len(beta) != k:
        raise ValueError(
            f"len(beta)={len(beta)} must equal k={k}"
        )
    params: dict[str, float] = {"k": float(k)}
    for i, b in enumerate(beta):
        params[f"beta_{i}"] = float(b)
    heads = tuple(f"{head_output_prefix}_{i}" for i in range(k))
    return LossSpec(
        kind=LossKind.MTP_WEIGHTED,
        params=params,
        head_outputs=heads,
        label_source="next_k_tokens",
    )


def ifim_shaped_loss(
    lambda_fim: float = 0.1,
    head_output_name: str = "logits",
) -> LossSpec:
    """Standard CE plus an Inverse-Fisher-Information shaping penalty."""
    return LossSpec(
        kind=LossKind.IFIM_SHAPED,
        params={"lambda_fim": float(lambda_fim)},
        head_outputs=(head_output_name,),
        label_source="next_token",
    )


def mhc_attn_bias_loss(
    lambda_mhc: float = 0.05,
    head_output_name: str = "logits",
) -> LossSpec:
    """CE plus multi-head-copy attention-bias auxiliary loss."""
    return LossSpec(
        kind=LossKind.MHC_ATTN_BIAS,
        params={"lambda_mhc": float(lambda_mhc)},
        head_outputs=(head_output_name,),
        label_source="next_token",
    )


def custom_loss(
    head_outputs: tuple[str, ...],
    *,
    label_source: str = "next_token",
    reduction: str = "mean",
    **params: float,
) -> LossSpec:
    """Build a CUSTOM-kind spec. The actual callable is wired in at
    :func:`build_model` time via a side-channel ``custom_loss_fn`` arg."""
    return LossSpec(
        kind=LossKind.CUSTOM,
        params=dict(params),
        head_outputs=head_outputs,
        label_source=label_source,
        reduction=reduction,
    )


# ---------------------------------------------------------------------------
# Registry of built-in loss builders (used by Stage E / GUI dropdown)
# ---------------------------------------------------------------------------


LOSS_BUILTINS: dict[str, str] = {
    "cross_entropy": "cppmega_v4.buildspec.loss_spec:cross_entropy_loss",
    "mtp_weighted":  "cppmega_v4.buildspec.loss_spec:mtp_weighted_loss",
    "ifim_shaped":   "cppmega_v4.buildspec.loss_spec:ifim_shaped_loss",
    "mhc_attn_bias": "cppmega_v4.buildspec.loss_spec:mhc_attn_bias_loss",
}


__all__ = [
    "LOSS_BUILTINS",
    "LossKind",
    "LossSpec",
    "cross_entropy_loss",
    "custom_loss",
    "ifim_shaped_loss",
    "mhc_attn_bias_loss",
    "mtp_weighted_loss",
]
