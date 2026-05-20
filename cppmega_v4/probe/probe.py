"""contract_probe — top-level solver.

Pulls in:
  - capability snapshots (TokenizerCapabilities + ParquetCapabilities)
  - per-brick + per-loss requirement tables
  - alternative generator
  - dry-forward gate

Emits one ContractProbeReport — read-only, JSON-serialisable, the
single artefact the GUI/CLI consume.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from cppmega_v4.buildspec.model_build_spec import ModelBuildSpec
from cppmega_v4.probe.alternatives import Alternative, generate_alternatives
from cppmega_v4.probe.capabilities import (
    ParquetCapabilities,
    TokenizerCapabilities,
    introspect_parquet,
    introspect_tokenizer,
)
from cppmega_v4.probe.dry_forward import dry_forward
from cppmega_v4.probe.requirements import (
    BRICK_REQUIREMENTS,
    LOSS_REQUIREMENTS,
    DataRequirement,
)


@dataclass(frozen=True)
class ProbeFinding:
    """One entry of the probe report."""

    kind: Literal["satisfied", "unsatisfied", "warning"]
    component: str
    requirement: DataRequirement
    message: str
    alternatives: tuple[Alternative, ...] = ()


@dataclass(frozen=True)
class ContractProbeReport:
    """Single-shot capability check result."""

    tokenizer: TokenizerCapabilities
    parquet: ParquetCapabilities
    findings: tuple[ProbeFinding, ...]
    elapsed_ms: float
    probe_hidden_size: int
    dry_forward_verdict: Literal["ok", "shape_mismatch", "exception", "skipped"]
    dry_forward_detail: str = ""

    @property
    def is_clean(self) -> bool:
        return (
            all(f.kind != "unsatisfied" for f in self.findings)
            and self.dry_forward_verdict in ("ok", "skipped")
        )

    @property
    def blocking(self) -> tuple[ProbeFinding, ...]:
        return tuple(f for f in self.findings if f.kind == "unsatisfied")


def _provided_keys(
    tokenizer: TokenizerCapabilities,
    parquet: ParquetCapabilities,
) -> frozenset[str]:
    keys: set[str] = set(tokenizer.special_ids.keys())
    for col in parquet.schema_columns:
        keys.add(col.name)
    # Derived keys are "always producible" from input_ids by the collate.
    if parquet.has_token_ids:
        keys.add("input_ids")
        keys.add("labels")              # next-token shift
        keys.add("labels_k_shifted")    # K-shift for MTP
    return frozenset(keys)


def _evaluate(
    component: str,
    requirements: tuple[DataRequirement, ...],
    provided: frozenset[str],
    build_spec: ModelBuildSpec,
    tokenizer: TokenizerCapabilities,
    parquet: ParquetCapabilities,
) -> list[ProbeFinding]:
    out: list[ProbeFinding] = []
    for req in requirements:
        if req.is_satisfied_by(provided):
            out.append(ProbeFinding(
                kind="satisfied", component=component, requirement=req,
                message=f"{component}: {req.key!r} satisfied",
            ))
            continue
        kind: Literal["unsatisfied", "warning"] = (
            "unsatisfied" if req.required else "warning"
        )
        alts = generate_alternatives(
            req, component, build_spec, tokenizer, parquet,
        )
        out.append(ProbeFinding(
            kind=kind, component=component, requirement=req,
            message=f"{component}: {req.key!r} missing ({req.reason})",
            alternatives=alts,
        ))
    return out


def contract_probe(
    build_spec: ModelBuildSpec,
    tokenizer_source: str | Path,
    parquet_path: str | Path,
    *,
    probe_hidden_size: int = 64,
    sample_rows: int = 256,
    run_dry_forward: bool = True,
) -> ContractProbeReport:
    """Run the full Contract Probe pipeline and return the report.

    Args:
      build_spec: the ModelBuildSpec to probe.
      tokenizer_source / parquet_path: paths to artefacts the spec will
        be paired with at training time.
      probe_hidden_size: hidden dim for the synthetic dry-forward gate.
      sample_rows: how many parquet rows to sample for non-null ratios.
      run_dry_forward: when False the dry-forward gate is skipped — the
        report carries verdict ``"skipped"``.
    """
    t0 = time.perf_counter()

    tok_caps = introspect_tokenizer(tokenizer_source)
    pq_caps = introspect_parquet(parquet_path, sample_rows=sample_rows)
    provided = _provided_keys(tok_caps, pq_caps)

    findings: list[ProbeFinding] = []

    # Loss-side requirements
    loss_reqs = LOSS_REQUIREMENTS.get(build_spec.loss.kind, ())
    findings.extend(_evaluate(
        f"loss:{build_spec.loss.kind.value}", loss_reqs,
        provided, build_spec, tok_caps, pq_caps,
    ))

    # Brick-side requirements
    for node in build_spec.graph.nodes:
        brick_reqs = BRICK_REQUIREMENTS.get(node.kind, ())
        findings.extend(_evaluate(
            f"brick:{node.name}", brick_reqs,
            provided, build_spec, tok_caps, pq_caps,
        ))

    # Dry-forward only if no unsatisfied requirements would block training
    has_unsat = any(f.kind == "unsatisfied" for f in findings)
    if run_dry_forward and not has_unsat:
        dry = dry_forward(build_spec.graph, hidden_size=probe_hidden_size)
        dry_verdict = dry.verdict
        dry_detail = dry.detail
    else:
        dry_verdict = "skipped"
        dry_detail = (
            "skipped: blocking findings present"
            if has_unsat else "skipped: run_dry_forward=False"
        )

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return ContractProbeReport(
        tokenizer=tok_caps,
        parquet=pq_caps,
        findings=tuple(findings),
        elapsed_ms=elapsed_ms,
        probe_hidden_size=probe_hidden_size,
        dry_forward_verdict=dry_verdict,
        dry_forward_detail=dry_detail,
    )
