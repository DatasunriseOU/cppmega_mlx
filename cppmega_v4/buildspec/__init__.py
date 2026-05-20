"""ModelBuildSpec layer — Loss + Optim + Graph Rewriters.

See ``ModelBuildSpec.md`` (repo root) for the design.

Stage A+B surface:
  - loss_spec: LossKind / LossSpec / cross_entropy_loss / mtp_weighted_loss
    / ifim_shaped_loss / mhc_attn_bias_loss / custom_loss / LOSS_BUILTINS
  - optim_spec: OptimKind / OptimSpec / ParamGroup / adamw / muon /
    muon_adamw_hybrid / sgd / OPTIM_BUILTINS
  - model_build_spec: ModelBuildSpec / Rewriter / RewriteOrderError
  - diagnostics: BuildDiagnostic / BuildDiagnosticSeverity /
    BuildDiagnostics / verify_build_spec
"""

from __future__ import annotations

from cppmega_v4.buildspec.diagnostics import (
    BuildDiagnostic,
    BuildDiagnosticSeverity,
    BuildDiagnostics,
    verify_build_spec,
)
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
from cppmega_v4.buildspec.model_build_spec import (
    ModelBuildSpec,
    RewriteOrderError,
    Rewriter,
)
from cppmega_v4.buildspec.rewriters import (
    HeadDetectionError,
    LossRewriteError,
    MTPRewriter,
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
    "BuildDiagnostic",
    "BuildDiagnosticSeverity",
    "BuildDiagnostics",
    "HeadDetectionError",
    "LOSS_BUILTINS",
    "LossRewriteError",
    "MTPRewriter",
    "LossKind",
    "LossSpec",
    "ModelBuildSpec",
    "OPTIM_BUILTINS",
    "OptimKind",
    "OptimSpec",
    "ParamGroup",
    "RewriteOrderError",
    "Rewriter",
    "adamw",
    "cross_entropy_loss",
    "custom_loss",
    "ifim_shaped_loss",
    "mhc_attn_bias_loss",
    "mtp_weighted_loss",
    "muon",
    "muon_adamw_hybrid",
    "sgd",
    "verify_build_spec",
]
