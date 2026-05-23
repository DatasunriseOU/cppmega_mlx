"""Transport-agnostic JSON-RPC 2.0 dispatcher.

One :func:`dispatch` entry point handles both the HTTP and WebSocket
transports. Validates the envelope, routes the method, coerces params
through the matching Pydantic model, calls the handler, packs the
result back into a JsonRpcResponse, and folds exceptions into the
error envelope per VBPlan §5.3.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Mapping

from pydantic import BaseModel, ValidationError

from cppmega_v4.jsonrpc.cache import LRUCache
from cppmega_v4.jsonrpc.methods import (
    build_preset_specs,
    probe_run,
    suggest_adapters,
    suggest_sharding,
    verify,
)
from cppmega_v4.jsonrpc.tokenizer_methods import (
    EncodeVisualizeParams,
    encode_visualize,
    list_presets as tokenizer_list_presets,
)
from cppmega_v4.jsonrpc.data_methods import (
    PreviewParquetParams,
    preview_parquet,
)
from cppmega_v4.jsonrpc.catalog_methods import (
    catalog_explain,
    catalog_list_options,
)
from cppmega_v4.jsonrpc.suggest_groups_method import suggest_optim_groups
from cppmega_v4.jsonrpc.roundtrip_method import (
    RoundtripCheckParams, roundtrip_check,
)
from cppmega_v4.jsonrpc.ablation_method import (
    AblationRunParams, ablation_run,
)
from cppmega_v4.jsonrpc.ckpt_inspect_method import (
    CkptInspectParams, ckpt_inspect,
)
from cppmega_v4.jsonrpc.dtype_cost_method import (
    DtypeCostParams, dtype_cost_estimate,
)
from cppmega_v4.jsonrpc.gen_run_method import (
    GenRunParams, gen_run,
)
from cppmega_v4.jsonrpc.schema import (
    BuildPresetSpecsParams,
    CatalogExplainParams,
    CatalogListOptionsParams,
    ErrorCode,
    JsonRpcError,
    JsonRpcRequest,
    JsonRpcResponse,
    PipelineAbortParams,
    PipelineAbortResult,
    PipelineRunParams,
    PipelineRunResult,
    ProbeRunParams,
    SuggestAdaptersParams,
    SuggestOptimGroupsParams,
    SuggestShardingParams,
    VerifyParams,
)

_log = logging.getLogger(__name__)


_Handler = Callable[[BaseModel, LRUCache | None], BaseModel]


_ROUTES: Mapping[str, tuple[type[BaseModel], _Handler]] = {
    "verify": (
        VerifyParams,
        lambda p, c: verify(p, cache=c),
    ),
    "suggest_sharding": (
        SuggestShardingParams,
        lambda p, c: suggest_sharding(p, cache=c),
    ),
    "suggest_adapters": (
        SuggestAdaptersParams,
        lambda p, c: suggest_adapters(p, cache=c),
    ),
    "build_preset_specs": (
        BuildPresetSpecsParams,
        lambda p, c: build_preset_specs(p, cache=c),
    ),
    "probe.run": (
        ProbeRunParams,
        lambda p, c: probe_run(p, cache=c),
    ),
    "pipeline.run": (
        PipelineRunParams,
        lambda p, c: _pipeline_run(p),
    ),
    "pipeline.abort": (
        PipelineAbortParams,
        lambda p, c: _pipeline_abort(p),
    ),
    "pipeline.pause": (
        PipelineAbortParams,
        lambda p, c: _pipeline_pause(p),
    ),
    "pipeline.resume": (
        PipelineAbortParams,
        lambda p, c: _pipeline_resume(p),
    ),
    "tokenizer.encode_visualize": (
        EncodeVisualizeParams,
        lambda p, c: encode_visualize(p, cache=c),
    ),
    "data.preview_parquet": (
        PreviewParquetParams,
        lambda p, c: preview_parquet(p, cache=c),
    ),
    "catalog.explain": (
        CatalogExplainParams,
        lambda p, c: catalog_explain(p, cache=c),
    ),
    "catalog.list_options": (
        CatalogListOptionsParams,
        lambda p, c: catalog_list_options(p, cache=c),
    ),
    "suggest_optim_groups": (
        SuggestOptimGroupsParams,
        lambda p, c: suggest_optim_groups(p, cache=c),
    ),
    "data.roundtrip_check": (
        RoundtripCheckParams,
        lambda p, c: roundtrip_check(p, cache=c),
    ),
    "ablation.run": (
        AblationRunParams,
        lambda p, c: ablation_run(p, cache=c),
    ),
    "ckpt.inspect": (
        CkptInspectParams,
        lambda p, c: ckpt_inspect(p, cache=c),
    ),
    "dtype.cost_estimate": (
        DtypeCostParams,
        lambda p, c: dtype_cost_estimate(p, cache=c),
    ),
    "gen.run": (
        GenRunParams,
        lambda p, c: gen_run(p, cache=c),
    ),
}


def _pipeline_run(params: PipelineRunParams) -> PipelineRunResult:
    # Lazy import — runner package depends on jsonrpc.schema for VerifyParams;
    # binding the symbols here breaks the import cycle.
    from cppmega_v4.runner import Pipeline, run_pipeline
    pipeline = Pipeline.from_dict({
        "stages": params.pipeline.stages,
        "stage_options": params.pipeline.stage_options,
        "continue_on_failure": params.pipeline.continue_on_failure,
    })
    report = run_pipeline(params.spec, pipeline)
    return PipelineRunResult.model_validate(report.to_dict())


def _pipeline_abort(params: PipelineAbortParams) -> PipelineAbortResult:
    from cppmega_v4.runner.stages import request_abort
    request_abort(params.run_id)
    return PipelineAbortResult(run_id=params.run_id)


def _pipeline_pause(params: PipelineAbortParams) -> PipelineAbortResult:
    """V7-H06: mark a train run as paused; the loop waits between steps."""
    from cppmega_v4.runtime.job_control import pause
    pause(params.run_id)
    return PipelineAbortResult(run_id=params.run_id)


def _pipeline_resume(params: PipelineAbortParams) -> PipelineAbortResult:
    """V7-H06: clear the paused flag so the train loop proceeds."""
    from cppmega_v4.runtime.job_control import resume
    resume(params.run_id)
    return PipelineAbortResult(run_id=params.run_id)


def dispatch(
    payload: Mapping[str, Any] | JsonRpcRequest,
    *,
    cache: LRUCache | None = None,
) -> JsonRpcResponse:
    """Route one JSON-RPC envelope to its handler.

    Never raises; all errors are folded into ``JsonRpcResponse.error``.
    """
    try:
        request = (
            payload if isinstance(payload, JsonRpcRequest)
            else JsonRpcRequest.model_validate(payload)
        )
    except ValidationError as exc:
        return _error_response(
            request_id=_safe_id(payload),
            code=ErrorCode.INVALID_REQUEST,
            message="Invalid JSON-RPC envelope",
            data={"errors": exc.errors()},
        )
    except Exception as exc:
        return _error_response(
            request_id=_safe_id(payload),
            code=ErrorCode.PARSE_ERROR,
            message="Parse error",
            data={"type": type(exc).__name__, "detail": str(exc)},
        )

    if request.method == "backend.status":
        return JsonRpcResponse(id=request.id, result={"status": "ok"})

    if request.method == "tokenizer.list_presets":
        return JsonRpcResponse(
            id=request.id,
            result=tokenizer_list_presets().model_dump(mode="json"),
        )

    if request.method == "architectures.list_presets":
        from cppmega_v4.architectures import available_presets
        from cppmega_v4.jsonrpc.schema import ArchitecturesListPresetsResult
        return JsonRpcResponse(
            id=request.id,
            result=ArchitecturesListPresetsResult(
                presets=list(available_presets()),
            ).model_dump(mode="json"),
        )

    route = _ROUTES.get(request.method)
    if route is None:
        return _error_response(
            request_id=request.id,
            code=ErrorCode.METHOD_NOT_FOUND,
            message=f"Method {request.method!r} not found",
            data={"available": sorted(_ROUTES)},
        )

    params_model, handler = route
    try:
        params = params_model.model_validate(request.params)
    except ValidationError as exc:
        return _error_response(
            request_id=request.id,
            code=ErrorCode.INVALID_PARAMS,
            message="Invalid params",
            data={"type": "ValidationError", "errors": exc.errors()},
        )

    try:
        result = handler(params, cache)
    except ValueError as exc:
        return _error_response(
            request_id=request.id,
            code=ErrorCode.INVALID_PARAMS,
            message="Invalid params",
            data={"type": type(exc).__name__, "detail": str(exc)},
        )
    except Exception as exc:
        _log.exception("dispatch handler %s raised", request.method)
        return _error_response(
            request_id=request.id,
            code=ErrorCode.INTERNAL_ERROR,
            message="Internal error",
            data={"type": type(exc).__name__, "detail": str(exc)},
        )

    return JsonRpcResponse(id=request.id, result=result.model_dump(mode="json"))


def _safe_id(payload: Any) -> str | int | None:
    try:
        if isinstance(payload, Mapping):
            v = payload.get("id")
            if isinstance(v, (str, int)):
                return v
    except Exception:
        pass
    return None


def _error_response(
    *, request_id: str | int | None, code: int, message: str,
    data: Mapping[str, Any] | None = None,
) -> JsonRpcResponse:
    return JsonRpcResponse(
        id=request_id,
        error=JsonRpcError(code=code, message=message, data=dict(data) if data else None),
    )
