from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import operator
from threading import Lock
from typing import Any


_BACKEND_NAME = "tilelang"
_STRICT_BACKEND_NAME = "tilelang_strict"
_FUSION_PATTERN_NAMES = ("gemm_softmax", "qk_reduce_sm_scale")
_LOWERED_PATTERNS = frozenset[str]()
_QK_REDUCE_TARGETS = frozenset(
    {
        "fp8_sparse_mla_indexed_qk_reduce",
        "qk_reduce",
    }
)
_FUSION_HITS: dict[str, int] = {name: 0 for name in _FUSION_PATTERN_NAMES}
_REGISTERED_BACKENDS: dict[str, bool] = {}
_LOCK = Lock()


@dataclass(frozen=True)
class FXGraphInspection:
    op_trace: tuple[str, ...]
    pattern_hits: Mapping[str, int]
    lowering_status: Mapping[str, str]
    pending_lowerings: tuple[str, ...]
    lowered_patterns: tuple[str, ...]


@dataclass(frozen=True)
class TileLangLoweringAttempt:
    compiled: bool
    source: str
    artifact_name: str | None = None
    error: str | None = None
    runner: Any | None = None


def _target_name(target: Any) -> str:
    if isinstance(target, str):
        return target
    name = getattr(target, "__name__", None)
    if isinstance(name, str):
        return name
    qualified_name = getattr(target, "__qualname__", None)
    if isinstance(qualified_name, str):
        return qualified_name
    return str(target)


def _node_label(node: Any) -> str:
    return f"{node.op}:{_target_name(node.target)}"


def _is_transpose(node: Any) -> bool:
    return node.op == "call_method" and _target_name(node.target) in {
        "transpose",
        "transpose_",
    }


def _is_matmul(node: Any) -> bool:
    if node.op == "call_method":
        return _target_name(node.target) in {"matmul", "__matmul__"}
    if node.op != "call_function":
        return False
    return node.target in {operator.matmul} or _target_name(node.target) in {
        "matmul",
        "__matmul__",
    }


def _is_softmax(node: Any) -> bool:
    return node.op == "call_function" and _target_name(node.target) in {
        "softmax",
    }


def _is_mul(node: Any) -> bool:
    if node.op == "call_method":
        return _target_name(node.target) in {"mul", "__mul__"}
    if node.op != "call_function":
        return False
    return node.target in {operator.mul} or _target_name(node.target) in {
        "mul",
        "__mul__",
    }


def _is_qk_reduce(node: Any) -> bool:
    if node.op != "call_function":
        return False
    target_name = _target_name(node.target).lower().replace("::", ".")
    target_parts = tuple(part for part in target_name.split(".") if part)
    return any(
        target == target_name
        or target in target_parts
        or target_name.endswith(f".{target}")
        for target in _QK_REDUCE_TARGETS
    )


def _node_args(node: Any) -> tuple[Any, ...]:
    args = getattr(node, "args", ())
    return tuple(args) if isinstance(args, tuple) else tuple(args or ())


def _node_kwargs(node: Any) -> Mapping[str, Any]:
    kwargs = getattr(node, "kwargs", {})
    return kwargs if isinstance(kwargs, Mapping) else {}


def _has_transposed_rhs(matmul_node: Any) -> bool:
    args = _node_args(matmul_node)
    if len(args) < 2:
        return False
    rhs = args[1]
    return _is_transpose(rhs)


def _is_last_dim_softmax(node: Any) -> bool:
    args = _node_args(node)
    kwargs = _node_kwargs(node)
    dim = kwargs.get("dim")
    if dim is None and len(args) >= 2:
        dim = args[1]
    return dim in {-1, "dim=-1"}


def _matches_gemm_softmax(node: Any) -> bool:
    if not _is_softmax(node) or not _is_last_dim_softmax(node):
        return False
    args = _node_args(node)
    if not args:
        return False
    matmul_node = args[0]
    return _is_matmul(matmul_node) and _has_transposed_rhs(matmul_node)


def _matches_qk_reduce_sm_scale(node: Any) -> bool:
    if not _is_mul(node):
        return False
    args = _node_args(node)
    if len(args) < 2:
        return False
    lhs, rhs = args[:2]
    return _is_qk_reduce(lhs) or _is_qk_reduce(rhs)


def inspect_fx_graph(graph_module: Any) -> FXGraphInspection:
    nodes = tuple(graph_module.graph.nodes)
    hits = {name: 0 for name in _FUSION_PATTERN_NAMES}
    for node in nodes:
        if _matches_gemm_softmax(node):
            hits["gemm_softmax"] += 1
        if _matches_qk_reduce_sm_scale(node):
            hits["qk_reduce_sm_scale"] += 1
    lowering_status = {
        name: (
            "lowered_tilelang_emitter"
            if name in _LOWERED_PATTERNS and count > 0
            else "pending_tilelang_emitter"
            if count > 0
            else "not_matched"
        )
        for name, count in hits.items()
    }
    pending_lowerings = tuple(
        name
        for name in _FUSION_PATTERN_NAMES
        if hits[name] > 0 and lowering_status[name] == "pending_tilelang_emitter"
    )
    lowered_patterns = tuple(
        name
        for name in _FUSION_PATTERN_NAMES
        if hits[name] > 0 and lowering_status[name] == "lowered_tilelang_emitter"
    )
    return FXGraphInspection(
        op_trace=tuple(_node_label(node) for node in nodes),
        pattern_hits=hits,
        lowering_status=lowering_status,
        pending_lowerings=pending_lowerings,
        lowered_patterns=lowered_patterns,
    )


def reset_fusion_hits() -> None:
    with _LOCK:
        for name in _FUSION_HITS:
            _FUSION_HITS[name] = 0


def fusion_hits() -> dict[str, int]:
    with _LOCK:
        return dict(_FUSION_HITS)


def _record_inspection(report: FXGraphInspection) -> None:
    with _LOCK:
        for name, count in report.pattern_hits.items():
            if name in _FUSION_HITS:
                _FUSION_HITS[name] += int(count)


def _propagate_fx_shapes(graph_module: Any, example_inputs: Sequence[Any]) -> None:
    try:
        from torch.fx.passes.shape_prop import ShapeProp  # type: ignore
    except Exception:
        return
    try:
        ShapeProp(graph_module).propagate(*example_inputs)
    except Exception:
        return


def _artifact_compiled_without_extern_fallback(artifact: Any) -> bool:
    prim_funcs = tuple(getattr(artifact, "prim_funcs", ()) or ())
    if not prim_funcs and getattr(artifact, "prim_func", None) is not None:
        prim_funcs = (getattr(artifact, "prim_func"),)
    if not prim_funcs:
        return False
    source = str(getattr(artifact, "source", "") or "").lower()
    fallback_markers = (
        "extern slot",
        "tilelang.compile failed",
        "_materialize_subgraph failed",
    )
    return not any(marker in source for marker in fallback_markers)


def _attempt_tilelang_lowering(
    graph_module: Any,
    example_inputs: Sequence[Any],
) -> TileLangLoweringAttempt:
    try:
        from poc.torch_dynamo.custom_op_wrapper import wrap_as_custom_op
        from poc.torch_dynamo.fx_to_tilelang import FXToTileLang
    except Exception as exc:
        return TileLangLoweringAttempt(
            compiled=False,
            source="",
            error=f"{type(exc).__name__}: {exc}",
        )
    try:
        _propagate_fx_shapes(graph_module, example_inputs)
        lowerer = FXToTileLang(graph_module, list(example_inputs))
        artifact = lowerer.run()
        source = str(getattr(artifact, "source", "") or "")
        compiled = _artifact_compiled_without_extern_fallback(artifact)
        runner = wrap_as_custom_op(artifact, lowerer.fx_signature()) if compiled else None
        return TileLangLoweringAttempt(
            compiled=compiled,
            source=source,
            artifact_name=str(getattr(artifact, "name", "") or "") or None,
            runner=runner,
        )
    except Exception as exc:
        return TileLangLoweringAttempt(
            compiled=False,
            source="",
            error=f"{type(exc).__name__}: {exc}",
        )


def _lowering_attempt_detail(attempt: TileLangLoweringAttempt) -> str:
    if attempt.source:
        return attempt.source
    if attempt.error:
        return attempt.error
    return "no TileLang lowerer detail available"


def compile_fx_graph(
    graph_module: Any,
    example_inputs: Sequence[Any],
    *,
    require_lowered_patterns: bool = False,
) -> Any:
    report = inspect_fx_graph(graph_module)
    _record_inspection(report)
    if require_lowered_patterns and report.pending_lowerings:
        attempt = _attempt_tilelang_lowering(graph_module, example_inputs)
        if attempt.compiled and attempt.runner is not None:
            return attempt.runner
        pending = ", ".join(report.pending_lowerings)
        raise RuntimeError(
            "tilelang torch.compile backend matched pattern(s) without "
            f"compiled TileLang emitters: {pending}. "
            f"TileLang lowerer detail: {_lowering_attempt_detail(attempt)}"
        )
    return graph_module.forward


def tilelang_backend(graph_module: Any, example_inputs: Sequence[Any]) -> Any:
    return compile_fx_graph(graph_module, example_inputs)


def strict_tilelang_backend(graph_module: Any, example_inputs: Sequence[Any]) -> Any:
    return compile_fx_graph(
        graph_module,
        example_inputs,
        require_lowered_patterns=True,
    )


def torch_compile_backend_available() -> bool:
    try:
        import torch._dynamo as dynamo  # type: ignore
    except Exception:
        return False
    return callable(getattr(dynamo, "register_backend", None))


def _backend_for_mode(require_lowered_patterns: bool) -> Any:
    return strict_tilelang_backend if require_lowered_patterns else tilelang_backend


def register_tilelang_backend(
    name: str = _BACKEND_NAME,
    *,
    require_lowered_patterns: bool = False,
) -> str:
    with _LOCK:
        registered_mode = _REGISTERED_BACKENDS.get(name)
    if registered_mode is not None:
        if registered_mode != require_lowered_patterns:
            raise RuntimeError(
                f"torch.compile backend {name!r} is already registered with "
                f"require_lowered_patterns={registered_mode}"
            )
        return name
    try:
        import torch._dynamo as dynamo  # type: ignore
    except Exception as exc:  # pragma: no cover - host dependent.
        raise RuntimeError("torch._dynamo is unavailable") from exc
    register_backend = getattr(dynamo, "register_backend", None)
    if not callable(register_backend):
        raise RuntimeError("torch._dynamo.register_backend is unavailable")
    backend = _backend_for_mode(require_lowered_patterns)
    try:
        register_backend(backend, name=name)
    except Exception as exc:
        message = str(exc).lower()
        if "already" not in message and "duplicate" not in message:
            raise
    with _LOCK:
        _REGISTERED_BACKENDS[name] = require_lowered_patterns
    return name


def register_tilelang_strict_backend() -> str:
    return register_tilelang_backend(
        name=_STRICT_BACKEND_NAME,
        require_lowered_patterns=True,
    )


__all__ = [
    "FXGraphInspection",
    "TileLangLoweringAttempt",
    "compile_fx_graph",
    "fusion_hits",
    "inspect_fx_graph",
    "register_tilelang_backend",
    "register_tilelang_strict_backend",
    "reset_fusion_hits",
    "strict_tilelang_backend",
    "tilelang_backend",
    "torch_compile_backend_available",
]
