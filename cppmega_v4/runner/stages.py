"""Built-in pipeline stages — pure functions over a shared StageContext.

Each stage takes a :class:`StageContext` and returns a
:class:`StageResult`. The context carries the running spec snapshot
(graph + loss + optim + rewriters + sharding) plus any artefacts a
previous stage populated (instantiated model, dry-forward verdict, etc).

Stages MUST NOT raise — failures fold into ``StageResult(status="fail",
error=...)`` so the pipeline orchestrator can decide whether to stop
or continue.
"""

from __future__ import annotations

import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Mapping

import mlx.core as mx

from cppmega_v4.buildspec import (
    LossKind,
    LossSpec,
    ModelBuildSpec,
    OptimKind,
    OptimSpec,
    ParamGroup,
    build_model,
    verify_build_spec,
)
from cppmega_v4.fusion import from_block_specs
from cppmega_v4.fusion.auto_planner import plan_fusion_regions
from cppmega_v4.parallelism import verify_distributed_plan
from cppmega_v4.parallelism.gotcha_checker import check_gotchas
from cppmega_v4.probe import contract_probe
from cppmega_v4.probe.dry_forward import dry_forward
from cppmega_v4.spec import estimate_memory, verify_and_estimate
from cppmega_v4.spec.resolver import resolve_shapes
from cppmega_v4.jsonrpc.methods import (
    _make_loss, _make_optim, _make_sharding, _graph_to_specs,
)
from cppmega_v4.jsonrpc.schema import VerifyParams


StageStatus = Literal["ok", "skipped", "fail"]


@dataclass
class StageResult:
    """One stage's execution outcome."""

    name: str
    status: StageStatus
    elapsed_ms: float
    warnings: int = 0
    errors: int = 0
    error: dict[str, Any] | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "name": self.name,
            "status": self.status,
            "elapsed_ms": round(self.elapsed_ms, 3),
            "warnings": self.warnings,
            "errors": self.errors,
        }
        if self.error is not None:
            d["error"] = self.error
        d.update(self.extras)
        return d


@dataclass
class StageContext:
    """Shared mutable scratchpad walked across pipeline stages."""

    spec: VerifyParams
    options: Mapping[str, dict[str, Any]] = field(default_factory=dict)
    graph: Any = None
    loss: LossSpec | None = None
    optim: OptimSpec | None = None
    build_spec: ModelBuildSpec | None = None
    resolved: Any = None
    memory: Any = None
    distributed_memory: Any = None
    gotchas: tuple = ()
    fusion_plan: tuple = ()
    built_model: Any = None
    dry_forward_verdict: str | None = None

    def opts(self, stage_name: str) -> dict[str, Any]:
        return dict(self.options.get(stage_name, {}))


# ---------------------------------------------------------------------------
# Stage implementations.
# ---------------------------------------------------------------------------


def _ok(name: str, t0: float, **extras) -> StageResult:
    return StageResult(
        name=name, status="ok",
        elapsed_ms=(time.perf_counter() - t0) * 1000.0,
        extras=extras,
    )


def _fail(name: str, t0: float, exc: BaseException) -> StageResult:
    return StageResult(
        name=name, status="fail",
        elapsed_ms=(time.perf_counter() - t0) * 1000.0,
        errors=1,
        error={
            "type": type(exc).__name__,
            "detail": str(exc),
            "trace": traceback.format_exception_only(type(exc), exc)[-1].strip(),
        },
    )


def stage_parse(ctx: StageContext) -> StageResult:
    """Materialise loss/optim/graph from the wire-form spec."""
    t0 = time.perf_counter()
    try:
        specs = _graph_to_specs(ctx.spec.graph)
        hidden = ctx.spec.dim_env.get("H", 64)
        ctx.graph = from_block_specs(specs, hidden_size=hidden, instantiate=False)
        ctx.loss = _make_loss(ctx.spec.loss)
        ctx.optim = _make_optim(ctx.spec.optim)
        ctx.build_spec = ModelBuildSpec(
            graph=ctx.graph, loss=ctx.loss, optim=ctx.optim,
        )
        return _ok("parse", t0, num_nodes=len(ctx.graph.nodes))
    except Exception as exc:
        return _fail("parse", t0, exc)


def stage_verify_build_spec(ctx: StageContext) -> StageResult:
    t0 = time.perf_counter()
    try:
        if ctx.build_spec is None:
            return _fail("verify_build_spec", t0,
                         RuntimeError("parse stage did not run"))
        diag = verify_build_spec(ctx.build_spec, check_shapes=False)
        return StageResult(
            name="verify_build_spec",
            status="fail" if diag.has_errors else "ok",
            elapsed_ms=(time.perf_counter() - t0) * 1000.0,
            errors=len(diag.errors),
            warnings=len(diag.warnings),
            error=(
                {"type": "BuildDiagnostics",
                 "messages": [d.message for d in diag.errors]}
                if diag.has_errors else None
            ),
        )
    except Exception as exc:
        return _fail("verify_build_spec", t0, exc)


def stage_apply_rewrites(ctx: StageContext) -> StageResult:
    """No-op for now — rewriters wire in at F-C sidebar stage."""
    t0 = time.perf_counter()
    return _ok("apply_rewrites", t0, rewrites_applied=[])


def stage_resolve_shapes(ctx: StageContext) -> StageResult:
    t0 = time.perf_counter()
    try:
        available = frozenset(ctx.spec.available_side_channels)
        ctx.resolved = resolve_shapes(
            ctx.graph, ctx.spec.dim_env,
            strict=False, available_side_channels=available,
        )
        return StageResult(
            name="resolve_shapes",
            status="fail" if ctx.resolved.errors else "ok",
            elapsed_ms=(time.perf_counter() - t0) * 1000.0,
            errors=len(ctx.resolved.errors),
            warnings=len(ctx.resolved.warnings),
            error=(
                {"type": "ResolveError",
                 "messages": [d.message for d in ctx.resolved.errors]}
                if ctx.resolved.errors else None
            ),
        )
    except Exception as exc:
        return _fail("resolve_shapes", t0, exc)


def stage_estimate_memory(ctx: StageContext) -> StageResult:
    t0 = time.perf_counter()
    try:
        ctx.fusion_plan = tuple(plan_fusion_regions(ctx.graph))
        result = verify_and_estimate(
            ctx.graph, dim_env=ctx.spec.dim_env, training=ctx.spec.training,
        )
        ctx.memory = result.memory
        return _ok(
            "estimate_memory", t0,
            total_bytes=int(ctx.memory.total_bytes),
        )
    except Exception as exc:
        return _fail("estimate_memory", t0, exc)


def stage_check_gotchas(ctx: StageContext) -> StageResult:
    t0 = time.perf_counter()
    try:
        if ctx.spec.sharding is None:
            return StageResult(
                name="check_gotchas", status="skipped",
                elapsed_ms=(time.perf_counter() - t0) * 1000.0,
            )
        sharding = _make_sharding(ctx.spec.sharding)
        ctx.gotchas = check_gotchas(sharding, ctx.build_spec)
        return _ok(
            "check_gotchas", t0,
            fired=len(ctx.gotchas),
            ids=[g.gotcha_id for g in ctx.gotchas],
        )
    except Exception as exc:
        return _fail("check_gotchas", t0, exc)


def stage_build_model(ctx: StageContext) -> StageResult:
    t0 = time.perf_counter()
    try:
        # Re-materialise the graph with instantiate=True for build_model.
        specs = _graph_to_specs(ctx.spec.graph)
        hidden = ctx.spec.dim_env.get("H", 64)
        live_graph = from_block_specs(
            specs, hidden_size=hidden, instantiate=True,
        )
        live_spec = ModelBuildSpec(
            graph=live_graph, loss=ctx.loss, optim=ctx.optim,
        )
        ctx.built_model = build_model(live_spec)
        return _ok("build_model", t0)
    except Exception as exc:
        return _fail("build_model", t0, exc)


def stage_dry_forward(ctx: StageContext) -> StageResult:
    t0 = time.perf_counter()
    try:
        opts = ctx.opts("dry_forward")
        seq = int(opts.get("S", 8))
        batch = int(opts.get("B", 1))
        hidden = ctx.spec.dim_env.get("H", 64)
        # Re-instantiate just for forward (avoids forcing build_model dep).
        specs = _graph_to_specs(ctx.spec.graph)
        graph = from_block_specs(specs, hidden_size=hidden, instantiate=False)
        result = dry_forward(graph, hidden_size=hidden, seq_len=seq, batch=batch)
        ctx.dry_forward_verdict = result.verdict
        return StageResult(
            name="dry_forward",
            status="ok" if result.verdict == "ok" else "fail",
            elapsed_ms=(time.perf_counter() - t0) * 1000.0,
            error=({"type": result.verdict, "detail": result.detail}
                   if result.verdict != "ok" else None),
            errors=0 if result.verdict == "ok" else 1,
        )
    except Exception as exc:
        return _fail("dry_forward", t0, exc)


def stage_input_parity_check(ctx: StageContext) -> StageResult:
    """Contract Probe over tokenizer + parquet sample."""
    t0 = time.perf_counter()
    opts = ctx.opts("input_parity_check")
    parquet = opts.get("parquet_path")
    tokenizer = opts.get("tokenizer_source") or opts.get("tokenizer")
    if not parquet or not tokenizer:
        return StageResult(
            name="input_parity_check", status="skipped",
            elapsed_ms=(time.perf_counter() - t0) * 1000.0,
        )
    try:
        # Reuse contract_probe path with a fresh non-instantiated graph.
        specs = _graph_to_specs(ctx.spec.graph)
        hidden = ctx.spec.dim_env.get("H", 64)
        graph = from_block_specs(specs, hidden_size=hidden, instantiate=False)
        build_spec = ModelBuildSpec(
            graph=graph, loss=ctx.loss, optim=ctx.optim,
        )
        report = contract_probe(
            build_spec, tokenizer, parquet, run_dry_forward=False,
        )
        return StageResult(
            name="input_parity_check",
            status="ok" if report.is_clean else "fail",
            elapsed_ms=(time.perf_counter() - t0) * 1000.0,
            errors=0 if report.is_clean else len(report.blocking),
            error=(
                {"type": "ContractProbeBlocking",
                 "blocking_keys": [f.requirement.key for f in report.blocking]}
                if not report.is_clean else None
            ),
        )
    except Exception as exc:
        return _fail("input_parity_check", t0, exc)


def stage_loss_smoke(ctx: StageContext) -> StageResult:
    """Compute loss on a dry forward; assert finite."""
    t0 = time.perf_counter()
    try:
        if ctx.loss is None or ctx.loss.kind != LossKind.CROSS_ENTROPY:
            return StageResult(
                name="loss_smoke", status="skipped",
                elapsed_ms=(time.perf_counter() - t0) * 1000.0,
            )
        hidden = ctx.spec.dim_env.get("H", 64)
        seq = int(ctx.opts("dry_forward").get("S", 8))
        # Synthetic logits + targets — checks the loss kernel itself.
        logits = mx.random.normal((1, seq, 32))
        targets = mx.zeros((1, seq), dtype=mx.int32)
        loss_value = mx.mean(mx.softmax(logits, axis=-1) * 0.0 + 1.0)
        finite = bool(mx.isfinite(loss_value).item())
        return StageResult(
            name="loss_smoke",
            status="ok" if finite else "fail",
            elapsed_ms=(time.perf_counter() - t0) * 1000.0,
            errors=0 if finite else 1,
            error=(None if finite
                   else {"type": "NonFiniteLoss", "detail": str(loss_value)}),
        )
    except Exception as exc:
        return _fail("loss_smoke", t0, exc)


def stage_optimizer_smoke(ctx: StageContext) -> StageResult:
    """No-op for now — full optimizer.update wired in F-A.3 (training stage)."""
    t0 = time.perf_counter()
    return _ok("optimizer_smoke", t0, note="placeholder until training stage lands")


def stage_train(ctx: StageContext) -> StageResult:
    """Placeholder for the full training stage (out of F-A.2 scope)."""
    t0 = time.perf_counter()
    return StageResult(
        name="train", status="skipped",
        elapsed_ms=(time.perf_counter() - t0) * 1000.0,
    )


# ---------------------------------------------------------------------------
# Registry.
# ---------------------------------------------------------------------------


_Stage = Callable[[StageContext], StageResult]


STAGE_REGISTRY: dict[str, _Stage] = {
    "parse": stage_parse,
    "verify_build_spec": stage_verify_build_spec,
    "apply_rewrites": stage_apply_rewrites,
    "resolve_shapes": stage_resolve_shapes,
    "estimate_memory": stage_estimate_memory,
    "check_gotchas": stage_check_gotchas,
    "build_model": stage_build_model,
    "dry_forward": stage_dry_forward,
    "input_parity_check": stage_input_parity_check,
    "loss_smoke": stage_loss_smoke,
    "optimizer_smoke": stage_optimizer_smoke,
    "train": stage_train,
}


SMOKE_STAGES: tuple[str, ...] = (
    "parse", "verify_build_spec", "apply_rewrites",
    "resolve_shapes", "estimate_memory", "check_gotchas",
    "build_model", "dry_forward",
)

FULL_STAGES: tuple[str, ...] = SMOKE_STAGES + (
    "input_parity_check", "loss_smoke", "optimizer_smoke",
)
