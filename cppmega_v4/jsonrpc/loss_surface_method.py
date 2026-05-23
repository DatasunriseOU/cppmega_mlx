"""V7-H09: explore.loss_surface RPC — N×M (lr_delta × wd_delta) sweep.

Runs a grid of short k-step Train evaluations around the current
schedule's lr / wd, returns a 2D matrix of {final_loss, throughput,
mem_mb} cells the UI renders as a heatmap.
"""

from __future__ import annotations

import copy
import time
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from cppmega_v4.jsonrpc.schema import VerifyParams
# NOTE: lazy import inside the handler — importing Pipeline/run_pipeline
# at module load makes cppmega_v4.runner -> cppmega_v4.jsonrpc.schema ->
# cppmega_v4.jsonrpc.__init__ -> dispatcher -> loss_surface_method a
# circular path that breaks any caller (e.g.
# `python -m cppmega_v4.tools.ckpt_inspect`) that imports
# cppmega_v4.runner before cppmega_v4.jsonrpc finished initialising.


class LossSurfaceParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    spec: VerifyParams
    lr_deltas: list[float] = Field(default_factory=lambda: [0.5, 1.0, 2.0])
    wd_deltas: list[float] = Field(default_factory=lambda: [0.5, 1.0, 2.0])
    k_steps: int = Field(2, ge=1, le=64)


class LossSurfaceCell(BaseModel):
    model_config = ConfigDict(extra="allow")

    lr_mult: float
    wd_mult: float
    status: str
    final_loss: float | None = None
    throughput_tok_s: float | None = None
    mem_mb: float | None = None
    elapsed_ms: float = 0.0


class LossSurfaceResult(BaseModel):
    model_config = ConfigDict(extra="allow")

    rows: list[list[LossSurfaceCell]]
    lr_deltas: list[float]
    wd_deltas: list[float]
    best_lr_mult: float | None = None
    best_wd_mult: float | None = None
    best_loss: float | None = None


def _mutate(spec_dict: dict, *, lr_mult: float,
            wd_mult: float) -> dict:
    out = copy.deepcopy(spec_dict)
    opt = out.get("optim", {})
    for g in opt.get("groups", []):
        if "lr" in g:
            g["lr"] = float(g["lr"]) * lr_mult
        if "weight_decay" in g:
            g["weight_decay"] = float(g["weight_decay"]) * wd_mult
    return out


def loss_surface_run(params: LossSurfaceParams,
                      *, cache: Any | None = None
                      ) -> LossSurfaceResult:
    spec_dict = params.spec.model_dump()
    rows: list[list[LossSurfaceCell]] = []
    best: tuple[float, float, float] | None = None
    for lr_m in params.lr_deltas:
        row: list[LossSurfaceCell] = []
        for wd_m in params.wd_deltas:
            t0 = time.perf_counter()
            try:
                mutated = VerifyParams.model_validate(
                    _mutate(spec_dict, lr_mult=lr_m, wd_mult=wd_m))
                from cppmega_v4.runner import Pipeline, run_pipeline
                rep = run_pipeline(mutated, Pipeline.from_dict({
                    "stages": ["parse", "verify_build_spec",
                               "build_model", "train"],
                    "stage_options": {"train": {
                        "num_steps": params.k_steps}},
                }))
                tr = next(s for s in rep.stages if s.name == "train")
                if tr.status != "ok":
                    row.append(LossSurfaceCell(
                        lr_mult=lr_m, wd_mult=wd_m, status="fail",
                        elapsed_ms=(time.perf_counter() - t0) * 1000))
                    continue
                losses = tr.extras.get("losses", [])
                final = float(losses[-1]) if losses else None
                mem = tr.extras.get("memory_peak_bytes")
                mem_mb = (round(int(mem) / (1024 * 1024), 4)
                           if mem else None)
                elapsed_ms = (time.perf_counter() - t0) * 1000
                throughput = (
                    params.k_steps * 1000.0 / max(elapsed_ms, 1e-3))
                cell = LossSurfaceCell(
                    lr_mult=lr_m, wd_mult=wd_m, status="ok",
                    final_loss=final, throughput_tok_s=throughput,
                    mem_mb=mem_mb, elapsed_ms=elapsed_ms)
                row.append(cell)
                if final is not None and (
                        best is None or final < best[2]):
                    best = (lr_m, wd_m, final)
            except Exception as exc:
                row.append(LossSurfaceCell(
                    lr_mult=lr_m, wd_mult=wd_m,
                    status=f"fail:{type(exc).__name__}",
                    elapsed_ms=(time.perf_counter() - t0) * 1000))
        rows.append(row)
    return LossSurfaceResult(
        rows=rows,
        lr_deltas=params.lr_deltas,
        wd_deltas=params.wd_deltas,
        best_lr_mult=best[0] if best else None,
        best_wd_mult=best[1] if best else None,
        best_loss=best[2] if best else None,
    )


__all__ = ["LossSurfaceParams", "LossSurfaceResult",
           "LossSurfaceCell", "loss_surface_run"]
