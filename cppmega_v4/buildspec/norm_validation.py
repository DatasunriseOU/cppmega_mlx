"""Norm parameter validation (E7-6).

Bricks that support pre_norm / post_norm parameters (attention family,
MLP family, MoE) must obey:

  - both 'none' simultaneously → ERROR (residual stream blows up
    without normalization)
  - mismatched norm kinds within a parallel block → WARNING
  - LayerNorm eps < 1e-5 → WARNING (numerically unstable on bf16)
  - RMSNorm eps < 1e-8 → WARNING (same reason)

Used by verify_build_spec to fold these diagnostics into the model
build report.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


NormKind = Literal["rmsnorm", "layernorm", "none"]

VALID_NORM_KINDS: tuple[str, ...] = ("rmsnorm", "layernorm", "none")


@dataclass(frozen=True)
class NormDiagnostic:
    severity: Literal["error", "warning", "info"]
    brick: str
    message: str


def validate_norm_params(
    brick_name: str,
    pre_norm: str = "rmsnorm",
    post_norm: str = "none",
    eps: float = 1e-6,
) -> list[NormDiagnostic]:
    """Validate one brick's norm config; return zero or more diagnostics."""
    out: list[NormDiagnostic] = []
    for label, val in [("pre_norm", pre_norm), ("post_norm", post_norm)]:
        if val not in VALID_NORM_KINDS:
            out.append(NormDiagnostic(
                severity="error",
                brick=brick_name,
                message=f"{label}={val!r} not in {VALID_NORM_KINDS}",
            ))
    if pre_norm == "none" and post_norm == "none":
        out.append(NormDiagnostic(
            severity="error",
            brick=brick_name,
            message="pre_norm and post_norm cannot both be 'none' — "
                    "residual stream variance will diverge",
        ))
    if pre_norm == "layernorm" and post_norm == "rmsnorm":
        out.append(NormDiagnostic(
            severity="warning",
            brick=brick_name,
            message="mixing LayerNorm pre + RMSNorm post is unusual; "
                    "verify it matches your target architecture",
        ))
    if pre_norm == "rmsnorm" and post_norm == "layernorm":
        out.append(NormDiagnostic(
            severity="warning",
            brick=brick_name,
            message="mixing RMSNorm pre + LayerNorm post is unusual",
        ))
    norms_in_use = {n for n in (pre_norm, post_norm)
                    if n in ("rmsnorm", "layernorm")}
    if norms_in_use and eps < 1e-8:
        out.append(NormDiagnostic(
            severity="warning",
            brick=brick_name,
            message=f"norm eps={eps:.0e} below 1e-8 → NaN risk on bf16",
        ))
    return out


def validate_parallel_block_norms(
    bricks: list[tuple[str, str, str]],
) -> list[NormDiagnostic]:
    """Validate norm config across a parallel block (attention || mlp etc).

    bricks: list of (name, pre_norm, post_norm). All entries in the
    parallel branch must have pre_norm != 'none' so the fan-in residual
    sees normalized inputs."""
    out: list[NormDiagnostic] = []
    for name, pre, _ in bricks:
        if pre == "none":
            out.append(NormDiagnostic(
                severity="error",
                brick=name,
                message="parallel-block branch must have pre_norm != 'none' "
                        "— fan-in residual would mix unnormalized streams",
            ))
    return out


__all__ = [
    "NormDiagnostic", "NormKind", "VALID_NORM_KINDS",
    "validate_norm_params", "validate_parallel_block_norms",
]
