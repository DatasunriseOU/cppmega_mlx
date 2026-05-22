"""ablation.run RPC handler (E7-11).

Runs N variants of a base spec where one axis (activation / optimizer
/ norm / schedule) is swapped, executes stage_train per variant with
the same num_steps, and returns side-by-side loss trajectories.

Used by the Ablations sidebar tab to show 'replace swiglu with gelu →
+2.5% final loss after 20 steps' style results.
"""

from __future__ import annotations

import copy
import time
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from cppmega_v4.jsonrpc.cache import LRUCache
from cppmega_v4.jsonrpc.schema import VerifyParams


AblationAxis = Literal["activation", "optimizer", "norm", "schedule"]


class AblationRunParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    base_spec: VerifyParams
    ablation_axis: AblationAxis
    variants: list[str]
    num_steps: int = 20


class AblationVariantResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    variant: str
    status: Literal["ok", "fail"]
    losses: list[float] = Field(default_factory=list)
    elapsed_ms: float
    weight_delta_norm: float = 0.0
    error: dict[str, Any] | None = None
    # H14: full train extras subtree so the UI can render a
    # per-variant detail view (model_summary, optimizer_kind,
    # schedule_kind, data_source, etc.) instead of just final loss.
    extras: dict[str, Any] = Field(default_factory=dict)


class AblationRunResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    results: list[AblationVariantResult] = Field(default_factory=list)
    ranked_by_final_loss: list[str] = Field(default_factory=list)
    baseline_variant: str = ""
    elapsed_ms_total: float = 0.0


def _mutate(base: VerifyParams, axis: AblationAxis,
            variant: str) -> VerifyParams:
    """Return a deep-copy with axis swapped to variant."""
    d = base.model_dump(mode="json")
    if axis == "activation":
        # Apply to every mlp/gated_mlp/moe brick.
        for node in d.get("graph", {}).get("nodes", []):
            if node.get("kind") in ("mlp", "gated_mlp", "moe"):
                node.setdefault("params", {})["activation"] = variant
    elif axis == "optimizer":
        d["optim"]["kind"] = variant
        # Replace first group's lr/wd/betas with recommended values
        # from the catalogue so e.g. Lion gets lr=1e-4 automatically.
        from cppmega_v4.explain import get_entry
        entry = get_entry("optimizer", variant)
        if entry and entry.recommended_params:
            rp = entry.recommended_params
            g = d["optim"]["groups"][0]
            if "lr" in rp:
                g["lr"] = float(rp["lr"])
            if "weight_decay" in rp:
                g["weight_decay"] = float(rp["weight_decay"])
            if "betas" in rp and isinstance(rp["betas"], (list, tuple)):
                g["betas"] = list(rp["betas"])
        if variant == "muon":
            d["optim"]["groups"][0].setdefault("ns_steps", 5)
    elif axis == "norm":
        for node in d.get("graph", {}).get("nodes", []):
            node.setdefault("params", {})["pre_norm"] = variant
    elif axis == "schedule":
        for grp in d["optim"]["groups"]:
            if variant == "constant":
                grp.pop("schedule", None)
            else:
                grp["schedule"] = {"kind": variant, "warmup_steps": 2,
                                   "total_steps": 50}
    return VerifyParams.model_validate(d)


def ablation_run(
    params: AblationRunParams,
    *,
    cache: LRUCache | None = None,
) -> AblationRunResult:
    """Execute the train stage per variant; return collected losses."""
    from cppmega_v4.runner import Pipeline, run_pipeline

    t0 = time.perf_counter()
    results: list[AblationVariantResult] = []

    for variant in params.variants:
        v_start = time.perf_counter()
        try:
            spec = _mutate(params.base_spec, params.ablation_axis, variant)
            pipeline = Pipeline.from_dict({
                "stages": ["parse", "verify_build_spec", "resolve_shapes",
                           "build_model", "train"],
                "stage_options": {
                    "train": {"num_steps": params.num_steps},
                },
            })
            report = run_pipeline(spec, pipeline)
            train = next((s for s in report.stages if s.name == "train"),
                         None)
            if train is None or train.status != "ok":
                results.append(AblationVariantResult(
                    variant=variant, status="fail",
                    losses=[],
                    elapsed_ms=(time.perf_counter() - v_start) * 1000,
                    weight_delta_norm=0.0,
                    error=train.error if train else
                          {"detail": "train stage missing"},
                ))
                continue
            extras = train.extras or {}
            results.append(AblationVariantResult(
                variant=variant, status="ok",
                losses=[float(l) for l in extras.get("losses", [])],
                elapsed_ms=(time.perf_counter() - v_start) * 1000,
                weight_delta_norm=float(extras.get("weight_delta_norm", 0)),
                # H14: forward the full extras subtree so the UI can
                # render a per-variant model_summary + auxiliary fields.
                extras=dict(extras),
            ))
        except Exception as exc:
            results.append(AblationVariantResult(
                variant=variant, status="fail",
                losses=[],
                elapsed_ms=(time.perf_counter() - v_start) * 1000,
                weight_delta_norm=0.0,
                error={"type": type(exc).__name__, "detail": str(exc)},
            ))

    ranked = sorted(
        (r for r in results if r.status == "ok" and r.losses),
        key=lambda r: r.losses[-1],
    )
    return AblationRunResult(
        results=results,
        ranked_by_final_loss=[r.variant for r in ranked],
        baseline_variant=params.variants[0] if params.variants else "",
        elapsed_ms_total=(time.perf_counter() - t0) * 1000,
    )
