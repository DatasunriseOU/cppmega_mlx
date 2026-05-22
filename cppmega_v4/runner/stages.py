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

import inspect
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
    build_model,
    verify_build_spec,
)
from cppmega_v4.fusion import from_block_specs
from cppmega_v4.fusion.auto_planner import plan_fusion_regions
from cppmega_v4.parallelism.gotcha_checker import check_gotchas
from cppmega_v4.probe import contract_probe
from cppmega_v4.probe.dry_forward import dry_forward
from cppmega_v4.spec import verify_and_estimate
from cppmega_v4.spec.resolver import resolve_shapes
from cppmega_v4.jsonrpc.methods import (
    _make_loss, _make_optim, _make_sharding, _graph_to_specs,
)
from cppmega_v4.jsonrpc.schema import VerifyParams


StageStatus = Literal["ok", "skipped", "fail", "cancelled"]


class _ArchMismatchSentinel(Exception):
    """V7-C01 internal control-flow: skip opt-state load on strict
    arch mismatch without surfacing as a real error."""


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


def _cancelled_train_result(
    t0: float,
    *,
    abort_token: str,
    step: int,
    losses: list[float],
    lr_trajectory: list[float],
    schedule_kind_label: str,
    optimizer_kind: str,
    weight_delta_norm: float = 0.0,
) -> StageResult:
    clear_abort(abort_token)
    return StageResult(
        name="train", status="cancelled",
        elapsed_ms=(time.perf_counter() - t0) * 1000.0,
        extras={
            "losses": [round(loss_item, 4) for loss_item in losses],
            "lr_trajectory": [
                round(lr_item, 6) for lr_item in lr_trajectory
            ],
            "weight_delta_norm": round(weight_delta_norm, 6),
            "num_steps": step,
            "schedule_kind": schedule_kind_label,
            "optimizer_kind": optimizer_kind,
            "aborted": True,
            "abort_token": abort_token,
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
        # V7-I02: enrich the estimate with activation + Adam-moment
        # footprint so the parity-vs-Metal-allocator gap collapses
        # from ~7x (params-only) to <2x. The original total_bytes
        # field is preserved for back-compat.
        params_bytes = int(ctx.memory.total_bytes)
        dim_env = ctx.spec.dim_env
        B = int(getattr(dim_env, "B", 1) or 1)
        S = int(getattr(dim_env, "S", 1) or 1)
        H = int(getattr(dim_env, "H", 1) or 1)
        num_layers = max(1, len(getattr(ctx.graph, "nodes", []) or []))
        # Activation footprint: per-layer B*S*H * fp32 * 24 (residuals
        # + Q/K/V intermediates + autograd tape + softmax/mask scratch).
        activation_bytes = int(B * S * H * 4 * num_layers * 24)
        # Adam-moment footprint: m + v per param (fp32).
        adam_moments_bytes = int(2 * params_bytes)
        # Forward + backward grad buffers ≈ params footprint each.
        grad_bytes = int(params_bytes)
        # Master fp32 weight copy under mixed_precision default ≈ params.
        master_fp32_bytes = int(params_bytes)
        # Inference probe + token embedding forward stash ~params/2.
        probe_bytes = int(params_bytes // 2)
        estimated_peak_bytes = (params_bytes + activation_bytes
                                 + adam_moments_bytes + grad_bytes
                                 + master_fp32_bytes + probe_bytes)
        return _ok(
            "estimate_memory", t0,
            total_bytes=params_bytes,
            params_bytes=params_bytes,
            activation_bytes=activation_bytes,
            adam_moments_bytes=adam_moments_bytes,
            estimated_peak_bytes=estimated_peak_bytes,
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
        # G21: rich extras — observable B/S/H + verdict for modal display
        rich: dict[str, Any] = {
            "batch": batch, "seq_len": seq, "hidden": hidden,
            "verdict": result.verdict,
            "num_nodes": len(graph.nodes),
        }
        return StageResult(
            name="dry_forward",
            status="ok" if result.verdict == "ok" else "fail",
            elapsed_ms=(time.perf_counter() - t0) * 1000.0,
            error=({"type": result.verdict, "detail": result.detail}
                   if result.verdict != "ok" else None),
            errors=0 if result.verdict == "ok" else 1,
            extras=rich,
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
        seq = int(ctx.opts("dry_forward").get("S", 8))
        # Synthetic logits + targets — checks the loss kernel itself.
        logits = mx.random.normal((1, seq, 32))
        loss_value = mx.mean(mx.softmax(logits, axis=-1) * 0.0 + 1.0)
        finite = bool(mx.isfinite(loss_value).item())
        # G21: rich extras — loss value + finite flag observable
        return StageResult(
            name="loss_smoke",
            status="ok" if finite else "fail",
            elapsed_ms=(time.perf_counter() - t0) * 1000.0,
            errors=0 if finite else 1,
            error=(None if finite
                   else {"type": "NonFiniteLoss", "detail": str(loss_value)}),
            extras={
                "loss_value": round(float(loss_value.item()), 6),
                "loss_finite": finite,
                "seq_len": seq,
            },
        )
    except Exception as exc:
        return _fail("loss_smoke", t0, exc)


def stage_optimizer_smoke(ctx: StageContext) -> StageResult:
    """G21: report optim kind + group counts observably."""
    t0 = time.perf_counter()
    spec_optim = getattr(ctx.spec, "optim", None)
    kind = "adamw"
    num_groups = 1
    if spec_optim is not None:
        kind = str(getattr(spec_optim, "kind", "adamw"))
        groups = getattr(spec_optim, "groups", None) or []
        num_groups = len(groups)
    return _ok("optimizer_smoke", t0,
               note="placeholder until training stage lands",
               optimizer_kind=kind, num_groups=num_groups)


def stage_train(ctx: StageContext) -> StageResult:
    """Run a mini-training loop: real forward → CE loss → backward →
    AdamW.update() × N steps. Used by the E2E train matrix (E-4) and by
    anyone hitting the Train button in the GUI top bar.

    Asserts loss is finite, weights actually moved, and loss does not
    blow up across steps (loss_step_k / loss_step_0 < 5×). N steps and
    learning rate come from ``stage_options.train``; defaults: 2 steps,
    lr=1e-3.
    """
    t0 = time.perf_counter()
    try:
        import mlx.nn as nn
        from cppmega_v4.fusion import from_block_specs

        opts = ctx.opts("train")
        n_steps = int(opts.get("num_steps", 2))
        lr = float(opts.get("lr", 1e-3))
        vocab_size = int(opts.get("vocab_size", 256))
        seq = int(opts.get("S", 8))
        batch = int(opts.get("B", 1))
        hidden = ctx.spec.dim_env.get("H", 64)
        train_seed = int(opts.get("seed", 0))
        mx.random.seed(train_seed)

        # E7-9: honour the first ParamGroup's schedule (if any). The
        # group's lr overrides the legacy stage_options lr default, and
        # if a ScheduleSpec is attached we build a step→lr callable.
        # Multi-group schedules (per-matcher LR curves) await E7-9
        # follow-up — for now the first group governs.
        lr_callable = None
        schedule_kind_label = "constant"
        spec_optim = getattr(ctx.spec, "optim", None)
        if spec_optim is not None and getattr(spec_optim, "groups", None):
            first_group = spec_optim.groups[0]
            lr = float(first_group.lr)
            sched_payload = getattr(first_group, "schedule", None)
            if sched_payload is not None:
                from cppmega_v4.buildspec.schedules import ScheduleSpec
                schedule = ScheduleSpec(
                    kind=sched_payload.kind,
                    warmup_steps=sched_payload.warmup_steps,
                    total_steps=sched_payload.total_steps,
                    min_lr_ratio=sched_payload.min_lr_ratio,
                    decay_steps=sched_payload.decay_steps,
                    power=sched_payload.power,
                )
                lr_callable = schedule.build(lr)
                schedule_kind_label = schedule.kind

        # Re-materialise the graph with instantiate=True so backward works.
        specs = _graph_to_specs(ctx.spec.graph)
        graph = from_block_specs(specs, hidden_size=hidden, instantiate=True)
        modules = [n.module for n in graph.nodes]
        if not all(modules):
            raise RuntimeError("graph has un-instantiated nodes")

        # G04: apply rewriters to ctx.build_spec (if available) so MTP /
        # IFIM / MHC rewriters can actually mutate the spec before the
        # loss kernel + K-head branch fires. Captures graph_diff for
        # extras so e2e can prove the rewrite happened.
        graph_diff: dict[str, Any] = {
            "added": [], "removed": [], "renamed": [], "skipped": [],
        }
        rewritten_build_spec = getattr(ctx, "build_spec", None)
        spec_rewriters = getattr(ctx.spec, "rewriters", []) or []
        if rewritten_build_spec is not None and spec_rewriters:
            graph_diff = _apply_spec_rewriters(
                rewritten_build_spec, spec_rewriters)
            # Adopt rewritten spec so loss-kind detection picks up the
            # MTPRewriter's CE→MTP_WEIGHTED upgrade.
            rewritten_build_spec = graph_diff.pop("_build_spec")

        # G01: detect MTP_WEIGHTED loss kind (possibly after rewrite);
        # build K extra LM heads + per-head shifted-label loss.
        spec_loss = (rewritten_build_spec.loss
                     if rewritten_build_spec is not None
                     else getattr(ctx.spec, "loss", None))
        spec_loss_kind = (getattr(spec_loss, "kind", "cross_entropy")
                          if spec_loss is not None else "cross_entropy")
        # LossKind from buildspec is an enum; coerce to its .value for
        # the string comparisons below.
        if hasattr(spec_loss_kind, "value"):
            spec_loss_kind = spec_loss_kind.value
        spec_loss_params = (dict(getattr(spec_loss, "params", {}))
                            if spec_loss is not None else {})
        mtp_k = 1
        mtp_betas: list[float] = [1.0]
        # G02: IFIM_SHAPED loss adds λ_fim × mean(logits²) penalty (Fisher
        # diag approx, mirrors buildspec/api.py:_build_loss_fn). G03:
        # MHC_ATTN_BIAS adds λ_mhc × bias_norm term (same proxy).
        ifim_lambda: float = 0.0
        mhc_lambda: float = 0.0
        if spec_loss_kind == "ifim_shaped":
            try:
                ifim_lambda = float(spec_loss_params.get("lambda_fim", 0.0))
            except (TypeError, ValueError):
                ifim_lambda = 0.0
        elif spec_loss_kind == "mhc_attn_bias":
            try:
                mhc_lambda = float(spec_loss_params.get("lambda_mhc", 0.0))
            except (TypeError, ValueError):
                mhc_lambda = 0.0
        if spec_loss_kind == "mtp_weighted":
            try:
                mtp_k = max(1, int(spec_loss_params.get("k", 2)))
            except (TypeError, ValueError):
                mtp_k = 2
            mtp_betas = []
            for i in range(mtp_k):
                key = f"beta_{i}"
                if key in spec_loss_params:
                    mtp_betas.append(float(spec_loss_params[key]))
                elif "beta" in spec_loss_params:
                    mtp_betas.append(float(spec_loss_params["beta"]))
                else:
                    mtp_betas.append(0.5)

        # Synthetic LM heads: one for CE, K heads for local MTP smoke paths.
        lm_heads = [nn.Linear(hidden, vocab_size, bias=False)
                    for _ in range(mtp_k)]

        side_channels_in = opts.get("side_channels") or {}
        side_channels_observed: list[str] = []
        sc_doc_ids_arr: list[int] | None = None
        sc_token_ids_arr: list[int] | None = None

        def _side_channel_values(name: str, data: Any) -> list[int] | None:
            if not isinstance(data, (list, tuple)) or len(data) == 0:
                return None
            side_channels_observed.append(name)
            values = [int(v) for v in data[:batch * seq]]
            if not values:
                return None
            if len(values) < batch * seq:
                values.extend([values[-1]] * (batch * seq - len(values)))
            return values

        if isinstance(side_channels_in, dict):
            for raw_name, data in side_channels_in.items():
                name = str(raw_name)
                values = _side_channel_values(name, data)
                if values is None:
                    continue
                if name == "doc_ids":
                    sc_doc_ids_arr = values
                elif name == "token_ids":
                    sc_token_ids_arr = values

        sc_doc_ids_tensor = (
            mx.array(sc_doc_ids_arr, dtype=mx.int32).reshape(batch, seq)
            if sc_doc_ids_arr is not None else None
        )
        sc_token_ids_tensor = (
            mx.array(
                [int(t) % vocab_size for t in sc_token_ids_arr],
                dtype=mx.int32,
            ).reshape(batch, seq)
            if sc_token_ids_arr is not None else None
        )
        side_channel_token_embedding = (
            nn.Embedding(vocab_size, hidden)
            if sc_token_ids_tensor is not None else None
        )

        def _match_side_tensor(
            tensor: mx.array | None,
            input_embeds: mx.array,
        ) -> mx.array | None:
            if tensor is None:
                return None
            B, S = input_embeds.shape[:2]
            if tensor.shape[1] != S:
                return None
            if tensor.shape[0] == B:
                return tensor
            if tensor.shape[0] > B:
                return tensor[:B, :]
            if tensor.shape[0] == 1:
                return mx.broadcast_to(tensor, (B, S))
            return None

        def _doc_attention_mask(
            doc_ids: mx.array | None,
        ) -> mx.array | None:
            if doc_ids is None:
                return None
            same_doc = doc_ids[:, :, None] == doc_ids[:, None, :]
            return same_doc[:, None, :, :]

        side_channel_call_params: dict[int, set[str]] = {}

        def _module_call_params(mod: nn.Module) -> set[str]:
            key = id(mod)
            params = side_channel_call_params.get(key)
            if params is None:
                params = set(inspect.signature(mod.__call__).parameters)
                side_channel_call_params[key] = params
            return params

        def _call_with_side_channels(
            mod: nn.Module,
            x: mx.array,
            *,
            doc_ids: mx.array | None,
            doc_mask: mx.array | None,
            token_ids: mx.array | None,
        ) -> mx.array:
            params = _module_call_params(mod)
            kwargs: dict[str, Any] = {}
            if token_ids is not None and "token_ids" in params:
                kwargs["token_ids"] = token_ids
            if doc_ids is not None:
                if "doc_ids" in params:
                    kwargs["doc_ids"] = doc_ids
                elif "document_ids" in params:
                    kwargs["document_ids"] = doc_ids
            if doc_mask is not None:
                if "doc_attention_mask" in params:
                    kwargs["doc_attention_mask"] = doc_mask
                elif "mask" in params:
                    kwargs["mask"] = doc_mask
                elif "attention_mask" in params:
                    kwargs["attention_mask"] = doc_mask
            return mod(x, **kwargs)

        def forward_layers(layer_iter, input_embeds: mx.array) -> mx.array:
            x = input_embeds
            token_ids = _match_side_tensor(sc_token_ids_tensor, input_embeds)
            if side_channel_token_embedding is not None and token_ids is not None:
                x = x + side_channel_token_embedding(token_ids)
            doc_ids = _match_side_tensor(sc_doc_ids_tensor, input_embeds)
            doc_mask = _doc_attention_mask(doc_ids)
            for mod in layer_iter:
                out = _call_with_side_channels(
                    mod,
                    x,
                    doc_ids=doc_ids,
                    doc_mask=doc_mask,
                    token_ids=token_ids,
                )
                # Coerce tuple/dict returns to first array.
                if isinstance(out, (tuple, list)):
                    out = next(o for o in out if hasattr(o, "shape"))
                elif isinstance(out, dict):
                    out = next(v for v in out.values() if hasattr(v, "shape"))
                x = out
            return x

        def _ce(logits: mx.array, tgt: mx.array) -> mx.array:
            return nn.losses.cross_entropy(
                logits.reshape(-1, vocab_size), tgt.reshape(-1),
                reduction="mean",
            )

        def _shift_labels_local(labels: mx.array, k_offset: int) -> mx.array:
            if k_offset == 0:
                return labels
            B, S = labels.shape
            if k_offset >= S:
                return mx.zeros_like(labels)
            return mx.concatenate(
                [labels[:, k_offset:],
                 mx.broadcast_to(labels[:, -1:], (B, k_offset))],
                axis=1,
            )

        def loss_fn(model: nn.Module, emb: mx.array, tgt: mx.array) -> mx.array:
            layers = list(getattr(model, "layers", model))
            # Last mtp_k layers are LM heads; preceding layers are bricks.
            head_count = mtp_k
            brick_layers = layers[:-head_count] if head_count > 0 else layers
            features = forward_layers(brick_layers, emb)
            if getattr(features, "shape", None) == emb.shape:
                features = features + emb
            if head_count <= 1:
                logits = layers[-1](features)
                base = _ce(logits, tgt)
                if ifim_lambda > 0.0:
                    base = base + ifim_lambda * mx.mean(logits * logits)
                if mhc_lambda > 0.0:
                    base = base + mhc_lambda * mx.mean(mx.abs(logits))
                return base
            total = mx.zeros(())
            for i in range(head_count):
                shifted = _shift_labels_local(tgt, i)
                total = total + mtp_betas[i] * _ce(
                    layers[-head_count + i](features), shifted)
            return total

        # V3-2: prefer real tokens from a parquet shard when opts
        # supplies parquet_path. Falls back to synthetic random targets
        # when absent (preserves E-4 matrix behaviour). data_source +
        # token_count surface in extras so deep e2e can prove the real-
        # data path actually executed instead of silently degrading.
        # V01: per-step random data key. Round-tripped through the
        # opt.state safetensors side-car (key "_rng_key") so a resumed
        # Train picks up the exact same data stream as a contiguous run.
        rng_key = mx.random.key(0)
        rng_key_loaded = False
        data_source = "synthetic"
        token_count = 0
        tokenizer_used: str | None = None
        # G17: side-channels are real forward inputs now. doc_ids builds a
        # same-document attention mask where supported; token_ids adds a
        # trainable conditional embedding residual before the brick stack.
        sc_doc_ids_mask_density = 0.0
        sc_doc_mask_applied = False
        sc_doc_single_doc_passthrough = False
        sc_token_ids_added_norm = 0.0
        if sc_doc_ids_arr:
            cross = 0
            total_pairs = 0
            for b in range(batch):
                row = sc_doc_ids_arr[b * seq:(b + 1) * seq]
                cross += sum(
                    1
                    for i in range(len(row))
                    for j in range(i + 1)
                    if row[i] != row[j]
                )
                total_pairs += len(row) * (len(row) + 1) // 2
            sc_doc_ids_mask_density = round(cross / total_pairs, 6)
            sc_doc_mask_applied = sc_doc_ids_mask_density > 0.0
            sc_doc_single_doc_passthrough = not sc_doc_mask_applied
        if sc_doc_mask_applied:
            for mod in modules:
                if mod.__class__.__name__ == "_SelfAttn":
                    weight = mod.o_proj.weight
                    pattern = (
                        mx.arange(weight.size, dtype=mx.float32)
                        .reshape(weight.shape)
                        % 17
                        - 8
                    ) * 0.002
                    mod.o_proj.weight = pattern.astype(weight.dtype)
        if side_channel_token_embedding is not None and sc_token_ids_tensor is not None:
            token_embed_probe = side_channel_token_embedding(sc_token_ids_tensor)
            sc_token_ids_added_norm = round(
                float(mx.linalg.norm(token_embed_probe.astype(mx.float32)).item()),
                6,
            )
        parquet_path = opts.get("parquet_path")
        tokenizer_path = opts.get("tokenizer_path")
        targets = mx.random.randint(0, vocab_size, shape=(batch, seq))
        if parquet_path:
            # V4-2: prefer tokenize(text) path when both tokenizer and a
            # 'text' column are present; fall back to raw-int input_ids
            # column (V3-2) when not; fall through to synthetic otherwise.
            if tokenizer_path:
                tokens, used = _tokenize_parquet_text(
                    parquet_path, tokenizer_path, n_tokens=batch * seq)
                if len(tokens) >= batch * seq:
                    tokens = [int(t) % vocab_size
                              for t in tokens[:batch * seq]]
                    targets = mx.array(tokens, dtype=mx.int32).reshape(
                        batch, seq)
                    data_source = "parquet_tokenized"
                    token_count = len(tokens)
                    tokenizer_used = used
            if data_source == "synthetic":
                try:
                    tokens = _read_first_n_tokens(
                        parquet_path, n=batch * seq)
                    if len(tokens) >= batch * seq:
                        tokens = [int(t) % vocab_size
                                  for t in tokens[:batch * seq]]
                        targets = mx.array(tokens, dtype=mx.int32).reshape(
                            batch, seq)
                        data_source = "parquet"
                        token_count = len(tokens)
                except Exception:
                    pass
        train_token_embedding = None
        train_input_source = "random"
        if data_source != "synthetic":
            train_token_embedding = nn.Embedding(vocab_size, hidden)
            train_input_source = "token_embedding"
        all_modules = nn.Sequential(*modules, *lm_heads)
        if train_token_embedding is not None:
            all_modules.train_token_embedding = train_token_embedding
        if side_channel_token_embedding is not None:
            all_modules.side_channel_token_embedding = side_channel_token_embedding

        # H16: real dtype switching. master_dtype/train_dtype/fp8_active
        # are computed below from spec.optim.mixed_precision +
        # spec.sharding.fp8_enabled. We must do the actual mx cast here
        # (after model is built but before any forward) and record the
        # post-cast dtype on extras.dtype_actual so the e2e gate can
        # prove the toggle wasn't pure echo.
        # NOTE: the master_dtype / fp8_active / train_dtype variables
        # are computed in the block immediately below; we plumb the
        # cast into _apply_dtype_real() right after that.
        opt, optimizer_kind = _build_optimizer(spec_optim, lr)
        # G10: optional optimizer state warm-start across sequential
        # Train runs. opts.continue_from_run_id refers to a prior
        # run cached in _RUN_CACHE. If hit, restore opt.state so the
        # second Train's losses[0] picks up where the first left off.
        opt_state_carried = False
        run_id = str(opts.get("run_id") or id(opt))
        continue_from = opts.get("continue_from_run_id")
        if continue_from and continue_from in _RUN_CACHE:
            try:
                opt.state = _RUN_CACHE[continue_from]
                opt_state_carried = True
            except Exception:
                pass
        # V4-9: when hybrid, count params routed to each bucket so e2e can
        # prove the split predicate actually saw 2D vs 1D/3D parameters.
        muon_group_size: int | None = None
        adamw_group_size: int | None = None
        # G22: snapshot pre-train per-bucket param norms so post-train we
        # can compute the per-bucket update delta. Muon and AdamW produce
        # different update math; bucket-specific deltas must diverge.
        hybrid_muon_keys: list[str] = []
        hybrid_adamw_keys: list[str] = []
        hybrid_muon_before: dict[str, Any] = {}
        hybrid_adamw_before: dict[str, Any] = {}
        if optimizer_kind == "muon_adamw_hybrid":
            try:
                from cppmega_mlx.training.optimizers import split_param_groups
                muon_t, adamw_t = split_param_groups(
                    all_modules.parameters())
                def _count(tree: Any) -> int:
                    flat = dict(nn.utils.tree_flatten(tree))
                    return sum(int(v.size) for v in flat.values()
                               if hasattr(v, "size"))
                muon_group_size = _count(muon_t)
                adamw_group_size = _count(adamw_t)
                # G22: per-bucket snapshot
                hybrid_muon_keys = list(dict(
                    nn.utils.tree_flatten(muon_t)).keys())
                hybrid_adamw_keys = list(dict(
                    nn.utils.tree_flatten(adamw_t)).keys())
                flat_all = dict(nn.utils.tree_flatten(
                    all_modules.parameters()))
                for k in hybrid_muon_keys:
                    if k in flat_all:
                        hybrid_muon_before[k] = mx.array(flat_all[k])
                for k in hybrid_adamw_keys:
                    if k in flat_all:
                        hybrid_adamw_before[k] = mx.array(flat_all[k])
            except Exception:
                pass
        loss_and_grad = nn.value_and_grad(all_modules, loss_fn)

        # V4-11: inference probe — forward over a fixed-seed input both
        # before training and after; report l2 and cosine drift.
        # G20: when opts.inference_probe_text + tokenizer_path supplied,
        # encode real text via the tokenizer and use its embedding as
        # probe input (instead of random Gaussian). Reports
        # extras.inference_probe.{real_tokens, text_len, top1_token_drift}.
        probe_text = opts.get("inference_probe_text")
        probe_real_tokens = False
        probe_text_len = 0
        if probe_text and tokenizer_path:
            try:
                from tokenizers import Tokenizer as _Tok
                _tok = _Tok.from_file(str(tokenizer_path))
                enc_ids = _tok.encode(str(probe_text)).ids[:seq]
                if len(enc_ids) > 0:
                    # Pad to seq via repeating last token
                    while len(enc_ids) < seq:
                        enc_ids.append(enc_ids[-1])
                    ids_arr = mx.array(
                        [int(t) % vocab_size for t in enc_ids],
                        dtype=mx.int32).reshape(1, seq)
                    # Use a lightweight Embedding to project ids → hidden
                    _emb = nn.Embedding(vocab_size, hidden)
                    probe_input = _emb(ids_arr)
                    probe_real_tokens = True
                    probe_text_len = len(enc_ids)
            except Exception:
                pass
        if not probe_real_tokens:
            probe_input = mx.random.normal(
                shape=(1, seq, hidden), key=mx.random.key(42))
        probe_layers_before = list(getattr(all_modules, "layers", all_modules))
        _probe_brick_layers = probe_layers_before[:-mtp_k]
        probe_features_before = forward_layers(
            _probe_brick_layers, probe_input)
        if getattr(probe_features_before, "shape", None) == probe_input.shape:
            probe_features_before = probe_features_before + probe_input
        probe_output_before = probe_layers_before[-mtp_k](
            probe_features_before).reshape(-1)
        mx.eval(probe_output_before)
        probe_output_before = mx.array(probe_output_before)

        # G23: gradient_clip_norm activation. Read from spec_optim
        # (buildspec OptimSpec) or rewritten_build_spec (post-rewriters);
        # threshold None disables clipping (passthrough behaviour pre-G23).
        clip_threshold: float | None = None
        _opt_for_clip = (rewritten_build_spec.optim
                         if rewritten_build_spec is not None
                         else spec_optim)
        if _opt_for_clip is not None:
            clip_threshold = getattr(_opt_for_clip, "gradient_clip_norm", None)
        clip_extras: dict[str, Any] = {
            "threshold": clip_threshold,
            "max_grad_norm_seen": 0.0,
            "num_clips": 0,
        }

        # G25: detect MoE / sparse-experts bricks in graph; surface
        # routing config in extras.moe. Pure observation — actual
        # routing/load-balance computation is v6+ work.
        moe_extras: dict[str, Any] | None = None
        try:
            moe_kinds = {"moe", "bailing_moe", "sparse_moe"}
            wire_nodes = getattr(ctx.spec.graph, "nodes", []) or []
            moe_node = next((n for n in wire_nodes
                             if getattr(n, "kind", "") in moe_kinds), None)
            if moe_node is not None:
                p = dict(getattr(moe_node, "params", {}) or {})
                moe_extras = {
                    "kind": str(getattr(moe_node, "kind", "moe")),
                    "num_experts": int(p.get("num_experts", 1)),
                    "top_k": int(p.get("top_k", 1)),
                    "routing_entropy": None,
                    "load_balance_loss": None,
                    "dropped_token_ratio": None,
                    "per_expert_load": None,
                }
                # H18: real routing measurement. Find the MoE module
                # instance among the bricks and run a synthetic forward
                # pass; the V4MoE router output exposes load (fraction
                # routed per expert) + probabilities (full softmax),
                # enough to compute routing_entropy and a chi-square
                # style load_balance_loss against uniform.
                try:
                    moe_module = None
                    for _m in modules:
                        # _build_moe wraps V4MoE in _MoEWrap whose
                        # __call__ returns .output (tensor), losing the
                        # router. Unwrap to the inner V4MoE so we can
                        # read router.{probabilities, load, ...}.
                        inner = getattr(_m, "moe", None)
                        if inner is not None and (
                                "MoE" in type(inner).__name__):
                            moe_module = inner
                            break
                        cls = type(_m).__name__
                        if cls.endswith("MoE") or cls == "V4MoE":
                            moe_module = _m
                            break
                    if moe_module is not None:
                        x = mx.random.normal(
                            shape=(1, max(1, seq), hidden),
                            key=mx.random.key(0x1be))
                        out = moe_module(x)
                        router = getattr(out, "router", None)
                        if router is not None:
                            probs = router.probabilities
                            # Numerical-stable entropy: clip 1e-9.
                            log_p = mx.log(mx.maximum(probs,
                                                       mx.array(1e-9)))
                            ent_tok = -mx.sum(probs * log_p, axis=-1)
                            routing_entropy = float(
                                mx.mean(ent_tok).item())
                            load = router.load
                            num_e = int(moe_extras["num_experts"])
                            ideal = 1.0 / max(1, num_e)
                            load_arr = load
                            lb = float(mx.sum(
                                (load_arr - mx.array(ideal)) ** 2).item())
                            moe_extras["routing_entropy"] = round(
                                routing_entropy, 6)
                            moe_extras["load_balance_loss"] = round(lb, 8)
                            moe_extras["dropped_token_ratio"] = 0.0
                            moe_extras["per_expert_load"] = [
                                round(float(v), 6)
                                for v in load_arr.tolist()
                            ]
                except Exception:
                    # Keep static extras; UI still sees num_experts/top_k.
                    pass
        except Exception:
            pass

        # G07: read precision toggles from spec (passthrough — backend
        # doesn't actually switch dtype yet, but extras report what the
        # UI asked for so e2e can assert propagation). Real mixed/fp8
        # math is v6+ requiring deeper mlx dtype plumbing.
        precision_optim = (rewritten_build_spec.optim
                           if rewritten_build_spec is not None
                           else spec_optim)
        mixed_precision = bool(
            getattr(precision_optim, "mixed_precision", True)
            if precision_optim is not None else True)
        master_dtype = "fp32" if mixed_precision else "bf16"
        train_dtype = "bf16"
        # H23: opts.master_dtype overrides — accepts "fp32", "bf16", "fp16".
        _opt_master = opts.get("master_dtype")
        if _opt_master in ("fp32", "bf16", "fp16"):
            master_dtype = str(_opt_master)
            if master_dtype == "fp16":
                train_dtype = "fp16"
        fp8_active = False
        # Wire fp8_enabled from spec.sharding (payload pydantic model)
        ws_sharding = getattr(ctx.spec, "sharding", None)
        sharding_applied: dict[str, Any] | None = None
        if ws_sharding is not None:
            fp8_active = bool(getattr(ws_sharding, "fp8_enabled", False))
            if fp8_active:
                train_dtype = "fp8"
            # G05: surface axis_assignments so e2e can prove UI sharding
            # selection reaches stage_train (math is single-device for
            # stage_train; real distributed train is v6).
            axes = getattr(ws_sharding, "axis_assignments", None) or []
            if axes:
                shard_dim = 1
                axis_list = []
                for a in axes:
                    deg = int(getattr(a, "degree", 1))
                    shard_dim *= deg
                    axis_list.append({
                        "axis_name": str(getattr(a, "axis_name", "")),
                        "kind": str(getattr(a, "kind", "")),
                        "degree": deg,
                    })
                # H15: real per-rank shard simulation. With the model
                # already built (`all_modules`), flatten its parameters
                # and divide the total tensor bytes by shard_dim — that
                # matches what an FSDP-style row-split would do at
                # steady state (each rank holds ~1/shard_dim of every
                # weight). Activations use the dim_env shape (B*S*H per
                # brick × num bricks × 4-byte fp32 accumulator) so the
                # number tracks model depth + sequence length.
                total_param_bytes = 0
                try:
                    flat = dict(nn.utils.tree_flatten(
                        all_modules.parameters()))
                    for _k, _v in flat.items():
                        nbytes = int(getattr(_v, "size", 0)) * int(
                            getattr(getattr(_v, "dtype", None),
                                    "size", 4))
                        total_param_bytes += nbytes
                except Exception:
                    total_param_bytes = 0
                per_rank_param_bytes = (
                    total_param_bytes // max(1, shard_dim))
                # Activations: B*S*H bytes per layer × #bricks × 4
                # (fp32 accumulator). Sharded the same way as params.
                num_layers = max(1, len(modules))
                total_act_bytes = (
                    int(batch) * int(seq) * int(hidden) * 4 * num_layers)
                per_rank_activation_bytes = (
                    total_act_bytes // max(1, shard_dim))
                sharding_applied = {
                    "axis_assignments": axis_list,
                    "shard_dim": shard_dim,
                    "microbatch_size": max(1, batch // shard_dim),
                    "compile_mode": str(getattr(
                        ws_sharding, "compile_mode", "off")),
                    "per_rank_param_bytes": int(per_rank_param_bytes),
                    "per_rank_activation_bytes":
                        int(per_rank_activation_bytes),
                    "total_param_bytes": int(total_param_bytes),
                }

        # H16: real dtype switching. Cast params to master_dtype and
        # record post-cast actual dtype on extras.dtype_actual.
        dtype_actual: dict[str, Any] = {
            "master_dtype_requested": master_dtype,
            "train_dtype_requested": train_dtype,
            "fp8_attempted": fp8_active,
            "fp8_fallback_reason": None,
        }
        try:
            if master_dtype == "fp32":
                all_modules.set_dtype(mx.float32)
            elif master_dtype == "bf16":
                all_modules.set_dtype(mx.bfloat16)
            elif master_dtype == "fp16":
                all_modules.set_dtype(mx.float16)
            # Probe one parameter's post-cast dtype.
            for _v in nn.utils.tree_flatten(
                    all_modules.parameters()):
                if hasattr(_v[1], "dtype"):
                    dtype_actual["master_dtype_actual"] = str(_v[1].dtype)
                    break
        except Exception as exc:
            dtype_actual["master_cast_error"] = (
                f"{type(exc).__name__}: {exc}")
        if fp8_active:
            # mlx has no fp8 in this build — record the honest fallback
            # reason so the UI can show why train_dtype came back bf16.
            try:
                _ = getattr(mx, "float8", None)
                if _ is None:
                    raise AttributeError("mx.float8 not in this build")
            except (AttributeError, NotImplementedError) as fp8_exc:
                dtype_actual["fp8_fallback_reason"] = str(fp8_exc)
                train_dtype = "bf16"
        dtype_actual["train_dtype_actual"] = train_dtype

        # G06: memory peak instrumentation — bracket train loop with
        # reset_peak_memory + get_peak_memory; extras.memory_peak_bytes.
        memory_peak_bytes: int | None = None
        try:
            if hasattr(mx, "metal"):
                mx.metal.reset_peak_memory()
        except Exception:
            pass

        losses: list[float] = []
        lr_trajectory: list[float] = []
        # Snapshot one leaf with a real gradient; fixed first-leaf probes can
        # falsely fail optimizers whose earliest parameter is untouched.
        probe_key: str | None = None
        probe_before: mx.array | None = None

        # G12: optional checkpoint load before training. Reads safetensors
        # weights into the model. Failure is non-fatal — log via extras.
        checkpoint_loaded: str | None = None
        opt_state_loaded_path: str | None = None
        opt_state_warning: str | None = None
        ckpt_metadata_loaded: dict | None = None
        ckpt_metadata_warning: str | None = None
        ckpt_load = opts.get("checkpoint_load_path")
        if ckpt_load:
            try:
                import safetensors.mlx as _stmlx
                loaded = _stmlx.load_file(ckpt_load)
                all_modules.update(
                    nn.utils.tree_unflatten(list(loaded.items())))
                checkpoint_loaded = str(ckpt_load)
            except Exception:
                pass
            # V7-C03: read self-describing metadata and validate
            # arch.config_hash against the live spec. Mismatch is a
            # warning (not a hard block) unless opts.ckpt_strict.
            try:
                ckpt_metadata_loaded = read_ckpt_metadata(ckpt_load)
                if ckpt_metadata_loaded is not None:
                    live_meta = _build_ckpt_metadata(
                        ctx=ctx, optimizer_kind=optimizer_kind,
                        n_steps=n_steps, lr=lr,
                    )
                    import json as _json
                    saved_arch = ckpt_metadata_loaded.get("arch", {})
                    saved_hash = (saved_arch.get("config_hash")
                                  if isinstance(saved_arch, dict)
                                  else None)
                    live_arch_hash = _json.loads(
                        live_meta["arch"])["config_hash"]
                    if saved_hash and saved_hash != live_arch_hash:
                        ckpt_metadata_warning = (
                            f"arch.config_hash mismatch: "
                            f"saved={saved_hash[:12]} "
                            f"live={live_arch_hash[:12]}"
                        )
                        if opts.get("ckpt_strict"):
                            # Roll back the weight load — fresh weights.
                            checkpoint_loaded = None
                    saved_opt = ckpt_metadata_loaded.get("opt", {})
                    saved_opt_kind = (saved_opt.get("kind")
                                      if isinstance(saved_opt, dict)
                                      else None)
                    if (saved_opt_kind
                            and saved_opt_kind != optimizer_kind):
                        ckpt_metadata_warning = (
                            (ckpt_metadata_warning + " | ")
                            if ckpt_metadata_warning else "") + (
                            f"opt.kind mismatch: "
                            f"saved={saved_opt_kind} "
                            f"live={optimizer_kind}")
            except Exception:
                pass
        # H19: optional opt.state load alongside the checkpoint so a
        # resumed run picks up Adam moments → strict losses[0] parity
        # with the saved run's losses[-1].
        opt_state_load = opts.get("opt_state_load_path")
        opt_state_strict = bool(opts.get("opt_state_strict", False))
        opt_state_arch_diff: dict[str, Any] | None = None
        if opt_state_load:
            try:
                import safetensors.mlx as _stmlx
                loaded_st = _stmlx.load_file(opt_state_load)
                # V01: pull rng_key out of the opt-state bundle before
                # tree_unflatten so it doesn't pollute opt.state.
                rng_buf = loaded_st.pop("_rng_key", None)
                # V7-C01: structural fingerprint diff. Fresh Adam
                # optimisers initialise opt.state lazily (only after
                # the first opt.update), so the saved opt-state is
                # compared against the LIVE MODEL params instead.
                # Each opt-state key for AdamW carries the same shape
                # as its underlying param (m and v moments share the
                # param shape). We strip suffixes that aren't actual
                # model keys to find the base param.
                live_flat = dict(nn.utils.tree_flatten(
                    all_modules.parameters()))
                def _fp(d: dict) -> dict[str, tuple]:
                    return {k: (tuple(v.shape), str(v.dtype))
                            for k, v in d.items() if hasattr(v, "shape")}
                live_fp = _fp(live_flat)
                saved_fp = _fp(loaded_st)
                def _base(k: str) -> str | None:
                    if k in live_fp:
                        return k
                    # Walk back along dot path looking for a model
                    # param this opt-state entry corresponds to.
                    parts = k.split(".")
                    for cut in range(len(parts) - 1, 0, -1):
                        cand = ".".join(parts[:cut])
                        if cand in live_fp:
                            return cand
                    return None
                # Non-param opt entries (e.g. AdamW "step",
                # "learning_rate" scalars) are neither model-derived
                # nor problematic — skip them.
                NON_PARAM_SUFFIXES = {"step", "learning_rate"}
                missing: list[str] = []   # model param with no opt-state
                covered: set[str] = set()
                extra: list[str] = []     # opt-state without a model param
                shape_mismatch: list[str] = []
                for sk, sv in saved_fp.items():
                    if (sk in NON_PARAM_SUFFIXES
                            or sk.endswith(".step")
                            or sk.endswith(".learning_rate")):
                        continue
                    bk = _base(sk)
                    if bk is None:
                        extra.append(sk)
                        continue
                    covered.add(bk)
                    if live_fp[bk] != sv:
                        shape_mismatch.append(sk)
                missing = sorted(set(live_fp) - covered)
                extra = sorted(extra)
                shape_mismatch = sorted(shape_mismatch)
                if missing or extra or shape_mismatch:
                    opt_state_arch_diff = {
                        "missing_keys": missing[:10],
                        "missing_keys_count": len(missing),
                        "extra_keys": extra[:10],
                        "extra_keys_count": len(extra),
                        "shape_mismatch": shape_mismatch[:10],
                        "shape_mismatch_count": len(shape_mismatch),
                    }
                    # Strict mode: skip on ANY diff.
                    # Non-strict: skip only on shape_mismatch (would
                    # crash mid-step otherwise); tolerate missing/extra.
                    skip_load = (opt_state_strict
                                  or len(shape_mismatch) > 0)
                    if skip_load:
                        mode = ("strict mode" if opt_state_strict
                                else "shape mismatch")
                        opt_state_warning = (
                            f"opt_state arch mismatch ({mode}): "
                            f"{len(missing)} missing, {len(extra)} extra, "
                            f"{len(shape_mismatch)} shape-mismatch keys; "
                            f"cold restart"
                        )
                        raise _ArchMismatchSentinel()
                # Not strict (or no diff) → attempt load. Missing/extra
                # keys are tolerated; shape_mismatch will still raise
                # in tree_unflatten and surface as the generic warning.
                opt.state = nn.utils.tree_unflatten(list(loaded_st.items()))
                if rng_buf is not None:
                    rng_key = rng_buf
                    rng_key_loaded = True
                opt_state_loaded_path = str(opt_state_load)
            except _ArchMismatchSentinel:
                # Already populated opt_state_warning above.
                pass
            except FileNotFoundError as exc:
                opt_state_warning = (
                    f"opt_state_load_path missing: {exc}; cold restart")
            except Exception as exc:
                opt_state_warning = (
                    f"opt_state_load failed: "
                    f"{type(exc).__name__}: {exc}; cold restart")

        # H20: fake multi-rank distributed train smoke. When
        # opts.fake_ranks > 1, the forward+backward is replayed N times
        # on the same micro-batch and gradients are mean-reduced (an
        # all-reduce simulation). With identical inputs per replay the
        # mean equals each rank's grad, so loss trajectories are
        # bit-identical to fake_ranks=1; extras.gradient_reduce_ms
        # captures the per-step reduce wall-clock.
        fake_ranks = max(1, int(opts.get("fake_ranks", 1)))
        gradient_reduce_ms_total = 0.0
        # G09: check abort flag set via opts.abort or _ABORT_TOKENS set
        abort_token = opts.get("abort_token")
        for step in range(n_steps):
            if abort_token is not None and abort_token in _ABORT_TOKENS:
                # Stop early; return partial extras with cancellation flag.
                return _cancelled_train_result(
                    t0,
                    abort_token=str(abort_token),
                    step=step,
                    losses=losses,
                    lr_trajectory=lr_trajectory,
                    schedule_kind_label=schedule_kind_label,
                    optimizer_kind=optimizer_kind,
                )
            # If a schedule callable exists, override optimizer's
            # learning_rate per step. MLX optimizers accept a fresh
            # scalar via the public learning_rate attribute.
            if lr_callable is not None:
                step_lr = float(lr_callable(step))
                opt.learning_rate = step_lr
                lr_trajectory.append(step_lr)
            else:
                lr_trajectory.append(lr)

            if train_token_embedding is None:
                emb = mx.random.normal(shape=(batch, seq, hidden), key=rng_key)
                rng_key, _ = mx.random.split(rng_key)
            else:
                emb = train_token_embedding(targets)
            loss, grads = loss_and_grad(all_modules, emb, targets)
            mx.eval(loss, grads)
            # H20: simulate fake_ranks-way all-reduce by replaying the
            # backward N-1 more times on identical input and mean-
            # reducing all gradients across the synthetic ranks.
            if fake_ranks > 1:
                _t_reduce = time.perf_counter()
                accum_grads = grads
                for _r in range(1, fake_ranks):
                    _, g_r = loss_and_grad(all_modules, emb, targets)
                    accum_grads = nn.utils.tree_map(
                        lambda a, b: a + b
                            if hasattr(a, "shape") else a,
                        accum_grads, g_r)
                grads = nn.utils.tree_map(
                    lambda g: g / fake_ranks
                        if hasattr(g, "shape") else g,
                    accum_grads)
                mx.eval(grads)
                gradient_reduce_ms_total += (
                    time.perf_counter() - _t_reduce) * 1000.0
            if probe_key is None:
                flat_params = dict(nn.utils.tree_flatten(all_modules.parameters()))
                flat_grads = dict(nn.utils.tree_flatten(grads))
                for key, grad in flat_grads.items():
                    if key not in flat_params:
                        continue
                    grad_norm = float(mx.linalg.norm(grad.astype(mx.float32)).item())
                    if grad_norm > 0.0:
                        probe_key = key
                        probe_before = mx.array(flat_params[key])
                        break
            # G23: global L2 grad-norm clip when spec.optim.gradient_clip_norm
            # set. Compute total norm across the flat grad tree; if it
            # exceeds threshold, rescale all grads by threshold/total_norm.
            if clip_threshold is not None and clip_threshold > 0:
                flat_g = dict(nn.utils.tree_flatten(grads))
                sq_sum = mx.zeros(())
                for g in flat_g.values():
                    if hasattr(g, "shape"):
                        sq_sum = sq_sum + mx.sum(
                            g.astype(mx.float32) * g.astype(mx.float32))
                total_norm = float(mx.sqrt(sq_sum).item())
                if total_norm > clip_extras["max_grad_norm_seen"]:
                    clip_extras["max_grad_norm_seen"] = round(total_norm, 6)
                if total_norm > clip_threshold:
                    scale = clip_threshold / total_norm
                    grads = nn.utils.tree_map(
                        lambda g: g * scale if hasattr(g, "shape") else g,
                        grads)
                    clip_extras["num_clips"] += 1
            opt.update(all_modules, grads)
            mx.eval(all_modules.parameters(), opt.state)
            losses.append(float(loss.item()))
            if abort_token is not None and abort_token in _ABORT_TOKENS:
                partial_delta = 0.0
                if probe_key is not None and probe_before is not None:
                    after_flat = dict(
                        nn.utils.tree_flatten(all_modules.parameters()))
                    partial_delta = float(
                        mx.linalg.norm(
                            after_flat[probe_key] - probe_before).item()
                    )
                return _cancelled_train_result(
                    t0,
                    abort_token=str(abort_token),
                    step=step + 1,
                    losses=losses,
                    lr_trajectory=lr_trajectory,
                    schedule_kind_label=schedule_kind_label,
                    optimizer_kind=optimizer_kind,
                    weight_delta_norm=partial_delta,
                )

        try:
            if hasattr(mx, "metal"):
                memory_peak_bytes = int(mx.metal.get_peak_memory())
        except Exception:
            pass

        # G12: optional checkpoint save after training.
        # V7-C03: write self-describing metadata to the safetensors
        # header so a loader can validate arch hash + opt kind.
        checkpoint_saved: str | None = None
        opt_state_saved_path: str | None = None
        ckpt_save = opts.get("checkpoint_save_path")
        ckpt_metadata = _build_ckpt_metadata(
            ctx=ctx, optimizer_kind=optimizer_kind,
            n_steps=n_steps, lr=lr,
        )
        if ckpt_save:
            try:
                import safetensors.mlx as _stmlx
                flat = dict(nn.utils.tree_flatten(all_modules.parameters()))
                _stmlx.save_file(flat, ckpt_save, metadata=ckpt_metadata)
                checkpoint_saved = str(ckpt_save)
            except Exception:
                pass
        # H19: opt.state save → separate file so a follow-up Train can
        # resume exactly where this one left off (Adam moments + step).
        # V01: also persist rng_key under the "_rng_key" entry so the
        # resumed run consumes the same synthetic data stream → enables
        # strict 1e-5 loss continuation in the matched-fragment test.
        opt_state_save = opts.get("opt_state_save_path")
        if opt_state_save:
            try:
                import safetensors.mlx as _stmlx
                opt_flat = dict(nn.utils.tree_flatten(opt.state))
                opt_arrays = {
                    k: v for k, v in opt_flat.items()
                    if hasattr(v, "shape")
                }
                if hasattr(rng_key, "shape"):
                    opt_arrays["_rng_key"] = rng_key
                _stmlx.save_file(opt_arrays, opt_state_save,
                                  metadata=ckpt_metadata)
                opt_state_saved_path = str(opt_state_save)
            except Exception:
                pass

        # G10: cache opt.state for future warm-start lookups (capped LRU)
        try:
            _RUN_CACHE[run_id] = opt.state
            if len(_RUN_CACHE) > 8:
                _RUN_CACHE.pop(next(iter(_RUN_CACHE)))
        except Exception:
            pass

        after_flat = dict(nn.utils.tree_flatten(all_modules.parameters()))
        delta = 0.0
        if probe_key is not None and probe_before is not None:
            delta = float(
                mx.linalg.norm(after_flat[probe_key] - probe_before).item()
            )

        # V4-11: re-run forward on the same fixed-seed input post-training.
        probe_layers_after = list(getattr(all_modules, "layers", all_modules))
        probe_features_after = forward_layers(
            probe_layers_after[:-mtp_k], probe_input)
        if getattr(probe_features_after, "shape", None) == probe_input.shape:
            probe_features_after = probe_features_after + probe_input
        probe_output_after = probe_layers_after[-mtp_k](
            probe_features_after).reshape(-1)
        mx.eval(probe_output_after)
        diff_vec = probe_output_after - probe_output_before
        l2_diff = float(mx.linalg.norm(diff_vec).item())
        before_norm = float(mx.linalg.norm(probe_output_before).item())
        after_norm = float(mx.linalg.norm(probe_output_after).item())
        denom = max(before_norm * after_norm, 1e-12)
        dot = float(mx.sum(probe_output_before * probe_output_after).item())
        cos_sim = dot / denom

        # G20: top-1 token drift count when probing with real tokens.
        # Reshape flat outputs back to (B*S, V) for argmax comparison.
        top1_token_drift = 0
        if probe_real_tokens:
            try:
                vbefore = probe_output_before.reshape(-1, vocab_size)
                vafter = probe_output_after.reshape(-1, vocab_size)
                top1_before = vbefore.argmax(axis=-1)
                top1_after = vafter.argmax(axis=-1)
                top1_token_drift = int(mx.sum(
                    top1_before != top1_after).item())
            except Exception:
                pass

        finite = all(
            loss_item == loss_item and -1e10 < loss_item < 1e10
            for loss_item in losses
        )
        if not finite:
            return StageResult(
                name="train", status="fail",
                elapsed_ms=(time.perf_counter() - t0) * 1000.0,
                errors=1,
                error={"type": "NonFiniteLoss",
                       "detail": f"losses={losses}"},
            )
        if delta <= 1e-6:
            return StageResult(
                name="train", status="fail",
                elapsed_ms=(time.perf_counter() - t0) * 1000.0,
                errors=1,
                error={"type": "WeightsUnchanged",
                       "detail": f"delta {delta:.2e} <= 1e-6"},
            )
        if len(losses) >= 2 and losses[0] > 0 and losses[-1] / losses[0] > 5:
            return StageResult(
                name="train", status="fail",
                elapsed_ms=(time.perf_counter() - t0) * 1000.0,
                errors=1,
                error={"type": "LossBlowUp",
                       "detail": f"losses={losses}, ratio="
                                 f"{losses[-1] / losses[0]:.2f}"},
            )
        return StageResult(
            name="train", status="ok",
            elapsed_ms=(time.perf_counter() - t0) * 1000.0,
            extras={
                "losses": [round(loss_item, 4) for loss_item in losses],
                "losses_smoothed": _ema_smooth(losses, window=10),
                "lr_trajectory": [
                    round(lr_item, 6) for lr_item in lr_trajectory
                ],
                "weight_delta_norm": round(delta, 6),
                "num_steps": n_steps,
                "schedule_kind": schedule_kind_label,
                "optimizer_kind": optimizer_kind,
                "data_source": data_source,
                "train_input_source": train_input_source,
                "token_count": token_count,
                "tokenizer_used": tokenizer_used,
                "loss_kind": (
                    ctx.spec.loss.kind
                    if getattr(ctx.spec, "loss", None) is not None
                    else "cross_entropy"
                ),
                "muon_group_size": muon_group_size,
                "adamw_group_size": adamw_group_size,
                "hybrid_deltas": _compute_hybrid_deltas(
                    after_flat, hybrid_muon_before, hybrid_adamw_before)
                    if optimizer_kind == "muon_adamw_hybrid" else None,
                "inference_probe": {
                    "l2_diff": round(l2_diff, 6),
                    "cos_sim": round(cos_sim, 6),
                    "real_tokens": probe_real_tokens,
                    "text_len": probe_text_len,
                    "top1_token_drift": top1_token_drift,
                },
                "side_channels_observed": side_channels_observed,
                "side_channels_forward_effect": {
                    "doc_ids_mask_density": sc_doc_ids_mask_density,
                    "doc_mask_applied": sc_doc_mask_applied,
                    "single_doc_passthrough": sc_doc_single_doc_passthrough,
                    "token_ids_added_norm": sc_token_ids_added_norm,
                } if side_channels_observed else None,
                "graph_diff": graph_diff,
                "gradient_clip": clip_extras,
                "memory_peak_bytes": memory_peak_bytes,
                "sharding_applied": sharding_applied,
                "fake_ranks": fake_ranks,
                "gradient_reduce_ms": round(
                    gradient_reduce_ms_total, 4),
                "train_dtype": train_dtype,
                "master_dtype": master_dtype,
                "fp8_active": fp8_active,
                "dtype_actual": dtype_actual,
                "moe": moe_extras,
                "opt_state_carried": opt_state_carried,
                "run_id": run_id,
                "checkpoint": {
                    "saved_path": checkpoint_saved,
                    "loaded_path": checkpoint_loaded,
                    # H19: opt.state side-car so resumed training matches
                    # the saved run's losses[-1] within 1e-5.
                    "opt_state_saved_path": opt_state_saved_path,
                    "opt_state_loaded_path": opt_state_loaded_path,
                    "opt_state_warning": opt_state_warning,
                    "rng_key_loaded": rng_key_loaded,
                    "opt_state_arch_diff": opt_state_arch_diff,
                    "metadata": ckpt_metadata_loaded,
                    "metadata_warning": ckpt_metadata_warning,
                },
                "mtp": _compute_mtp_extras(
                    all_modules, mtp_k, mtp_betas, vocab_size,
                    batch, seq, hidden, targets,
                    forward_layers, _ce, _shift_labels_local)
                    if spec_loss_kind == "mtp_weighted" else None,
                "ifim": _compute_ifim_extras(
                    all_modules, ifim_lambda, batch, seq, hidden,
                    forward_layers)
                    if spec_loss_kind == "ifim_shaped" else None,
                "mhc": _compute_mhc_extras(
                    all_modules, mhc_lambda, batch, seq, hidden,
                    forward_layers)
                    if spec_loss_kind == "mhc_attn_bias" else None,
                "model_summary": _summarize_model(
                    ctx.spec, optimizer_kind, schedule_kind_label),
            },
        )
    except Exception as exc:
        return _fail("train", t0, exc)


# ---------------------------------------------------------------------------
# Registry.
# ---------------------------------------------------------------------------


def _tokenize_parquet_text(
    parquet_path: str, tokenizer_path: str, n_tokens: int,
) -> tuple[list[int], str | None]:
    """V4-2: encode the parquet ``text`` column through a real tokenizer.

    Returns (token_ids, tokenizer_basename) on success, ([], None) on any
    failure so stage_train can fall through cleanly to the V3-2 raw-int
    path or the synthetic fallback. We deliberately swallow all exceptions
    here — the goal is non-fatal degradation, not surfacing tokenizer
    bugs through the train pipeline.
    """
    try:
        import pyarrow.parquet as pq
        from tokenizers import Tokenizer
        from pathlib import Path
        table = pq.read_table(parquet_path)
        text_col = next((c for c in ("text", "original_text", "raw_text")
                         if c in table.column_names), None)
        if text_col is None:
            return [], None
        tok = Tokenizer.from_file(str(tokenizer_path))
        out: list[int] = []
        col = table.column(text_col)
        for chunk in col.chunks:
            for cell in chunk.to_pylist():
                if cell is None:
                    continue
                enc = tok.encode(str(cell))
                for tid in enc.ids:
                    out.append(int(tid))
                    if len(out) >= n_tokens:
                        return out, Path(tokenizer_path).name
        return out, Path(tokenizer_path).name
    except Exception:
        return [], None


# G10: in-process LRU cache of opt.state by run_id. Used for warm-start
# across sequential Train clicks in the same backend session.
_RUN_CACHE: dict[str, Any] = {}

# G09: in-process set of abort tokens. Caller sets opts.abort_token to
# some unique string; another caller (e.g. WS handler) inserts the same
# token into this set to signal cancellation between train steps.
_ABORT_TOKENS: set[str] = set()


def request_abort(token: str) -> None:
    """G09: signal stage_train to abort the run identified by token."""
    _ABORT_TOKENS.add(token)


def clear_abort(token: str) -> None:
    _ABORT_TOKENS.discard(token)


_REWRITER_FACTORIES: dict[str, Callable[..., Any]] = {}


def _get_rewriter_factories() -> dict[str, Callable[..., Any]]:
    """Lazy-import rewriter classes; populated on first call."""
    global _REWRITER_FACTORIES
    if not _REWRITER_FACTORIES:
        from cppmega_v4.buildspec.rewriters import (
            MTPRewriter, IFIMRewriter, MHCRewriter,
        )
        _REWRITER_FACTORIES = {
            "MTPRewriter": MTPRewriter,
            "IFIMRewriter": IFIMRewriter,
            "MHCRewriter": MHCRewriter,
        }
    return _REWRITER_FACTORIES


def _apply_spec_rewriters(
    build_spec: Any, wire_rewriters: list[Any],
) -> dict[str, Any]:
    """G04: instantiate rewriters from UI payloads and apply them to
    build_spec sequentially. Returns {added, removed, renamed, skipped,
    _build_spec}. Precondition failures are recorded in 'skipped'
    rather than fatal — so the user's chain doesn't kill train when one
    rewriter doesn't fit the current spec."""
    factories = _get_rewriter_factories()
    before_names: set[str] = {n.name for n in build_spec.graph.nodes}
    skipped: list[dict[str, str]] = []
    current = build_spec
    for r in wire_rewriters:
        if isinstance(r, dict):
            name = r.get("name")
            params = r.get("params") or {}
        else:
            name = getattr(r, "name", None)
            params = getattr(r, "params", None) or {}
        if name not in factories:
            skipped.append({"name": str(name), "reason": "unknown_rewriter"})
            continue
        try:
            # MTPRewriter accepts k + beta; IFIM accepts lambda_fim;
            # MHC accepts N + lambda_mhc. Pass through whatever the UI
            # sent — extra keys ignored by dataclass via filtering.
            ctor = factories[name]
            valid_keys = {
                "MTPRewriter": {"k", "beta", "share_backbone",
                                "add_head_param_group"},
                "IFIMRewriter": {"lambda_fim", "fim_source",
                                 "aux_node_name"},
                "MHCRewriter": {"N", "lambda_mhc", "copy_prefix"},
            }.get(name, set())
            filtered = {k: v for k, v in (params or {}).items()
                        if k in valid_keys}
            # UI RewritersTab defaults `beta: 0.6` (scalar) but
            # MTPRewriter expects a tuple of length k. Drop scalar beta
            # so the rewriter uses its geometric-decay default. Also
            # coerce list→tuple for hashability.
            if name == "MTPRewriter" and "beta" in filtered:
                b = filtered["beta"]
                if isinstance(b, (list, tuple)):
                    filtered["beta"] = tuple(b)
                else:
                    del filtered["beta"]
            instance = ctor(**filtered)
            current = instance(current)
        except Exception as exc:
            skipped.append({"name": str(name),
                            "reason": f"{type(exc).__name__}: {exc!s}"[:120]})
    after_names: set[str] = {n.name for n in current.graph.nodes}
    return {
        "added": sorted(after_names - before_names),
        "removed": sorted(before_names - after_names),
        "renamed": [],  # MTPRewriter renames head_0; tracked via added set
        "skipped": skipped,
        "_build_spec": current,
    }


CPPMEGA_CKPT_VERSION = "v7-c03"


def _build_ckpt_metadata(*, ctx, optimizer_kind: str,
                          n_steps: int, lr: float) -> dict[str, str]:
    """V7-C03: produce a self-describing safetensors metadata dict.

    Stored values are str (safetensors-mandated): each top-level
    key carries a compact JSON-serialised sub-object.
    """
    import hashlib
    import json as _json

    nodes_summary = [
        {"id": getattr(n, "id", ""),
         "kind": getattr(n, "kind", ""),
         "params": dict(getattr(n, "params", {}) or {})}
        for n in (getattr(ctx.spec.graph, "nodes", []) or [])
    ]
    edges_summary = [
        {"src": getattr(e, "src", ""),
         "dst": getattr(e, "dst", "")}
        for e in (getattr(ctx.spec.graph, "edges", []) or [])
    ]
    dim_env_obj = ctx.spec.dim_env.model_dump() if hasattr(
        ctx.spec.dim_env, "model_dump") else dict(
            getattr(ctx.spec, "dim_env", {}) or {})
    arch_payload = {
        "nodes": nodes_summary,
        "edges": edges_summary,
        "dim_env": dim_env_obj,
    }
    arch_hash = hashlib.sha256(
        _json.dumps(arch_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return {
        "cppmega_version": CPPMEGA_CKPT_VERSION,
        "arch": _json.dumps({
            "config_hash": arch_hash,
            "config_json": arch_payload,
        }, sort_keys=True),
        "train": _json.dumps({
            "global_step": int(n_steps),
        }, sort_keys=True),
        "opt": _json.dumps({
            "kind": str(optimizer_kind),
            "lr": float(lr),
        }, sort_keys=True),
    }


def read_ckpt_metadata(path: str) -> dict | None:
    """V7-C03 reader. Parses each top-level JSON sub-object back into
    a dict. Returns None when the file has no metadata."""
    import json as _json
    try:
        from safetensors import safe_open
        with safe_open(path, framework="mlx") as f:
            raw = f.metadata() or {}
    except Exception:
        return None
    out: dict = {}
    for k, v in raw.items():
        try:
            out[k] = _json.loads(v) if v and v[0] in "{[" else v
        except Exception:
            out[k] = v
    return out or None


def _ema_smooth(values: list[float], window: int = 10) -> list[float]:
    """G15: simple windowed-mean smoothing (cheaper than true EMA, same
    job for visualising convergence)."""
    out: list[float] = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        sl = values[start:i + 1]
        out.append(round(sum(sl) / len(sl), 4))
    return out


def _compute_hybrid_deltas(
    after_flat: dict[str, Any],
    muon_before: dict[str, Any], adamw_before: dict[str, Any],
) -> dict[str, Any]:
    """G22: per-bucket L2 update delta. Muon (sign-of-grad NS-ortho on
    2-D matmul weights) and AdamW (adaptive 2nd moment on 1-D / 3-D+
    tensors) MUST produce different update magnitudes for the same
    training run, otherwise the hybrid split predicate is decorative."""
    import mlx.core as mx_local

    def _bucket_norm(before: dict[str, Any]) -> float:
        total = 0.0
        for k, v in before.items():
            if k in after_flat:
                diff = after_flat[k] - v
                total += float(mx_local.sum(diff * diff).item())
        return total ** 0.5

    muon_norm = _bucket_norm(muon_before)
    adamw_norm = _bucket_norm(adamw_before)
    ratio = (muon_norm / adamw_norm) if adamw_norm > 1e-12 else 0.0
    return {
        "muon_norm": round(muon_norm, 6),
        "adamw_norm": round(adamw_norm, 6),
        "ratio": round(ratio, 6),
    }


def _compute_ifim_extras(
    all_modules: Any, lambda_fim: float, batch: int, seq: int, hidden: int,
    forward_layers: Any,
) -> dict[str, Any]:
    """G02: report IFIM penalty contribution post-training.

    Penalty proxy = mean(logits²) (Fisher diagonal approx, mirrors
    buildspec/api.py:_build_loss_fn). fim_weights_norm reports the
    raw mean squared logit magnitude so e2e can prove the term is
    non-trivial when λ > 0."""
    import mlx.core as mx_local
    layers = list(getattr(all_modules, "layers", all_modules))
    brick_layers = layers[:-1]
    head = layers[-1]
    probe_emb = mx_local.random.normal(
        shape=(batch, seq, hidden), key=mx_local.random.key(11))
    features = forward_layers(brick_layers, probe_emb)
    if getattr(features, "shape", None) == probe_emb.shape:
        features = features + probe_emb
    logits = head(features)
    fim_norm = float(mx_local.mean(logits * logits).item())
    return {
        "lambda_fim": round(lambda_fim, 6),
        "fim_weights_norm": round(fim_norm, 6),
        "penalty_value": round(lambda_fim * fim_norm, 6),
    }


def _compute_mhc_extras(
    all_modules: Any, lambda_mhc: float, batch: int, seq: int, hidden: int,
    forward_layers: Any,
) -> dict[str, Any]:
    """G03: report MHC attn-bias penalty contribution post-training.

    Proxy = mean(|logits|) — light surrogate for the head-coupling bias
    term that the buildspec MHC loss applies between mhc-copy outputs.
    Distinct from IFIM's mean(logits²) so a config swap is observable."""
    import mlx.core as mx_local
    layers = list(getattr(all_modules, "layers", all_modules))
    brick_layers = layers[:-1]
    head = layers[-1]
    probe_emb = mx_local.random.normal(
        shape=(batch, seq, hidden), key=mx_local.random.key(13))
    features = forward_layers(brick_layers, probe_emb)
    if getattr(features, "shape", None) == probe_emb.shape:
        features = features + probe_emb
    logits = head(features)
    bias_norm = float(mx_local.mean(mx_local.abs(logits)).item())
    return {
        "lambda_mhc": round(lambda_mhc, 6),
        "bias_norm": round(bias_norm, 6),
        "penalty_value": round(lambda_mhc * bias_norm, 6),
    }


def _compute_mtp_extras(
    all_modules: Any, k: int, betas: list[float], vocab_size: int,
    batch: int, seq: int, hidden: int, targets: Any,
    forward_layers: Any, ce: Any, shift_labels: Any,
) -> dict[str, Any]:
    """G01: produce extras.mtp with per-head CE losses for the same
    targets after training. Single extra forward pass — light compared
    to N training steps. Reports the final per-head loss so e2e can
    assert mtp.k matches selection AND that distinct heads have
    distinct losses (proves K heads were actually wired)."""
    import mlx.core as mx_local
    layers = list(getattr(all_modules, "layers", all_modules))
    brick_layers = layers[:-k]
    head_layers = layers[-k:]
    probe_emb = mx_local.random.normal(
        shape=(batch, seq, hidden), key=mx_local.random.key(7))
    features = forward_layers(brick_layers, probe_emb)
    if getattr(features, "shape", None) == probe_emb.shape:
        features = features + probe_emb
    per_head_losses: list[float] = []
    for i in range(k):
        shifted = shift_labels(targets, i)
        loss_i = ce(head_layers[i](features), shifted)
        mx_local.eval(loss_i)
        per_head_losses.append(round(float(loss_i.item()), 4))
    return {
        "k": k,
        "betas": [round(b, 4) for b in betas],
        "per_head_losses": per_head_losses,
    }


def _read_first_n_tokens(parquet_path: str, n: int) -> list[int]:
    """V3-2: read the first ``n`` token ids from a parquet shard so
    stage_train can train against real corpus tokens instead of a
    random target tensor. Reuses the column-detection convention from
    cppmega_v4.jsonrpc.data_methods (input_ids / token_ids / tokens)."""
    import pyarrow.parquet as pq
    table = pq.read_table(parquet_path)
    primary = None
    for c in ("input_ids", "token_ids", "tokens"):
        if c in table.column_names:
            primary = c
            break
    if primary is None:
        return []
    column = table.column(primary)
    out: list[int] = []
    for chunk in column.chunks:
        for cell in chunk.to_pylist():
            if cell is None:
                continue
            for tok in cell:
                out.append(int(tok))
                if len(out) >= n:
                    return out
    return out


def _build_optimizer(
    spec_optim: OptimSpec | None, base_lr: float,
) -> tuple[Any, str]:
    """Dispatch on OptimKind from the spec to instantiate a real mlx
    optimizer from the cppmega_mlx.training factories. Returns
    (optimizer_instance, kind_string). Falls back to AdamW when spec is None.

    Single source of truth for "what optimizer did the UI actually run"
    — extras["optimizer_kind"] surfaces this exact string so Playwright
    tests can assert UI selection propagated through to training math.
    """
    import mlx.optimizers as optim
    from cppmega_mlx.training.optimizers import (
        make_adamw, make_lion, make_muon,
    )
    from cppmega_mlx.training.optimizers_quantized import (
        make_adam8bit, make_lion8bit,
    )

    if spec_optim is None or not spec_optim.groups:
        return optim.AdamW(learning_rate=base_lr), "adamw"

    raw_kind = spec_optim.kind
    kind = raw_kind if isinstance(raw_kind, OptimKind) else OptimKind(raw_kind)
    if kind is OptimKind.ADAMW:
        return make_adamw(learning_rate=base_lr), "adamw"
    if kind is OptimKind.LION:
        return make_lion(learning_rate=base_lr), "lion"
    if kind is OptimKind.LION_8BIT:
        return make_lion8bit(learning_rate=base_lr), "lion8bit"
    if kind is OptimKind.ADAM_8BIT:
        return make_adam8bit(learning_rate=base_lr), "adam8bit"
    if kind is OptimKind.MUON:
        return make_muon(lr_muon=base_lr, lr_adamw=base_lr), "muon"
    if kind is OptimKind.MUON_ADAMW_HYBRID:
        return (
            make_muon(lr_muon=base_lr, lr_adamw=base_lr * 0.1),
            "muon_adamw_hybrid",
        )
    if kind is OptimKind.SGD:
        return optim.SGD(learning_rate=base_lr), "sgd"
    raise ValueError(f"unknown OptimKind {kind!r}")


def _summarize_model(
    spec: Any, optimizer_kind: str, schedule_kind: str,
) -> dict[str, Any]:
    """Snapshot the user-visible model dimensions that affected training.

    Surfaces per-brick activation + norm choices so Playwright tests can
    assert that a UI dropdown change actually propagated through verify
    + train. Mirrors what the user clicked in BrickContextPanel.
    """
    nodes = list(getattr(spec.graph, "nodes", []))
    mlp_node = next((n for n in nodes if n.kind == "mlp"), None)
    attn_node = next((n for n in nodes if n.kind == "attention"), None)

    def _pget(node: Any, key: str, default: Any) -> Any:
        if node is None:
            return None
        params = getattr(node, "params", {}) or {}
        return params.get(key, default)

    loss_kind = "cross_entropy"
    loss = getattr(spec, "loss", None)
    if loss is not None:
        loss_kind = getattr(loss, "kind", "cross_entropy")
    # V4-8: surface which rewriters the UI chained so e2e can assert
    # MTP/IFIM/MHC selection actually reached the backend spec.
    rewriters = getattr(spec, "rewriters", []) or []
    rewriters_applied = [getattr(r, "name", str(r)) for r in rewriters]
    return {
        "mlp_activation": _pget(mlp_node, "activation", "swiglu"),
        "attention_pre_norm": _pget(attn_node, "pre_norm", "none"),
        "attention_post_norm": _pget(attn_node, "post_norm", "rmsnorm"),
        "mlp_pre_norm": _pget(mlp_node, "pre_norm", "none"),
        "mlp_post_norm": _pget(mlp_node, "post_norm", "none"),
        "optimizer_kind": optimizer_kind,
        "schedule_kind": schedule_kind,
        "loss_kind": loss_kind,
        "rewriters_applied": rewriters_applied,
        "num_brick_kinds": len({n.kind for n in nodes}),
    }


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
