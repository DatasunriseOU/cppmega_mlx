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
        seq = int(ctx.opts("dry_forward").get("S", 8))
        # Synthetic logits + targets — checks the loss kernel itself.
        logits = mx.random.normal((1, seq, 32))
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

        # Synthetic LM head — for MTP we create K of them. lm_head is the
        # first one (used by single-head paths and the inference probe).
        lm_heads = [nn.Linear(hidden, vocab_size, bias=False)
                    for _ in range(mtp_k)]
        lm_head = lm_heads[0]

        def forward_layers(layer_iter, input_embeds: mx.array) -> mx.array:
            x = input_embeds
            for mod in layer_iter:
                out = mod(x)
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
        rng_key = mx.random.key(0)
        data_source = "synthetic"
        token_count = 0
        tokenizer_used: str | None = None
        # V4-10: record which side-channels the caller supplied. Pure
        # observation for now — actual per-channel forward routing is
        # v5+ work. Surfacing via extras lets e2e prove UI toggles
        # reached the backend opts surface.
        side_channels_in = opts.get("side_channels") or {}
        side_channels_observed: list[str] = []
        if isinstance(side_channels_in, dict):
            for name, data in side_channels_in.items():
                if isinstance(data, (list, tuple)) and len(data) > 0:
                    side_channels_observed.append(str(name))
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
        all_modules = nn.Sequential(*modules, *lm_heads)
        opt, optimizer_kind = _build_optimizer(spec_optim, lr)
        # V4-9: when hybrid, count params routed to each bucket so e2e can
        # prove the split predicate actually saw 2D vs 1D/3D parameters.
        muon_group_size: int | None = None
        adamw_group_size: int | None = None
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
            except Exception:
                pass
        loss_and_grad = nn.value_and_grad(all_modules, loss_fn)

        # V4-11: inference probe — forward over a fixed-seed input both
        # before training and after; report l2 and cosine drift. Proves
        # the optimizer's update actually changed observable model output,
        # not just internal optimizer state. G01: skip the K-1 extra LM
        # heads — probe only the primary head so output shape stays sane
        # for mtp_k > 1.
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

        losses: list[float] = []
        lr_trajectory: list[float] = []
        # Snapshot one leaf with a real gradient; fixed first-leaf probes can
        # falsely fail optimizers whose earliest parameter is untouched.
        probe_key: str | None = None
        probe_before: mx.array | None = None

        for step in range(n_steps):
            # If a schedule callable exists, override optimizer's
            # learning_rate per step. MLX optimizers accept a fresh
            # scalar via the public learning_rate attribute.
            if lr_callable is not None:
                step_lr = float(lr_callable(step))
                opt.learning_rate = step_lr
                lr_trajectory.append(step_lr)
            else:
                lr_trajectory.append(lr)

            emb = mx.random.normal(shape=(batch, seq, hidden), key=rng_key)
            rng_key, _ = mx.random.split(rng_key)
            loss, grads = loss_and_grad(all_modules, emb, targets)
            mx.eval(loss, grads)
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
                "lr_trajectory": [
                    round(lr_item, 6) for lr_item in lr_trajectory
                ],
                "weight_delta_norm": round(delta, 6),
                "num_steps": n_steps,
                "schedule_kind": schedule_kind_label,
                "optimizer_kind": optimizer_kind,
                "data_source": data_source,
                "token_count": token_count,
                "tokenizer_used": tokenizer_used,
                "loss_kind": (
                    ctx.spec.loss.kind
                    if getattr(ctx.spec, "loss", None) is not None
                    else "cross_entropy"
                ),
                "muon_group_size": muon_group_size,
                "adamw_group_size": adamw_group_size,
                "inference_probe": {
                    "l2_diff": round(l2_diff, 6),
                    "cos_sim": round(cos_sim, 6),
                },
                "side_channels_observed": side_channels_observed,
                "graph_diff": graph_diff,
                "gradient_clip": clip_extras,
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
