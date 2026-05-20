"""ModelBuildSpec layer — Loss + Optim + Graph Rewriters.

See ``ModelBuildSpec.md`` (repo root) for the design.

Stage A surface (this commit):
  - loss_spec.LossKind / LossSpec / cross_entropy_loss / mtp_weighted_loss
    / ifim_shaped_loss / mhc_attn_bias_loss / custom_loss / LOSS_BUILTINS
  - optim_spec.OptimKind / OptimSpec / ParamGroup / adamw / muon /
    muon_adamw_hybrid / sgd / OPTIM_BUILTINS
"""

from __future__ import annotations

from cppmega_v4.buildspec.loss_spec import (
    LOSS_BUILTINS,
    LossKind,
    LossSpec,
    cross_entropy_loss,
    custom_loss,
    ifim_shaped_loss,
    mhc_attn_bias_loss,
    mtp_weighted_loss,
)
from cppmega_v4.buildspec.optim_spec import (
    OPTIM_BUILTINS,
    OptimKind,
    OptimSpec,
    ParamGroup,
    adamw,
    muon,
    muon_adamw_hybrid,
    sgd,
)

__all__ = [
    "LOSS_BUILTINS",
    "LossKind",
    "LossSpec",
    "OPTIM_BUILTINS",
    "OptimKind",
    "OptimSpec",
    "ParamGroup",
    "adamw",
    "cross_entropy_loss",
    "custom_loss",
    "ifim_shaped_loss",
    "mhc_attn_bias_loss",
    "mtp_weighted_loss",
    "muon",
    "muon_adamw_hybrid",
    "sgd",
]
