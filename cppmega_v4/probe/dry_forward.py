"""Dry forward pass — synthetic input_ids on the instantiated graph.

Last gate after static checks. Each brick already has a unit test that
its forward preserves (B, S, H) at small sizes; here we chain the
instantiated graph and run **one** forward at hidden=64, batch=1, seq=8
to catch residual brick-to-brick coupling that the resolver missed.

Returns a one-word verdict + an optional exception trace. Never raises.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import mlx.core as mx

from cppmega_v4.fusion.brick_graph import BrickGraph
from cppmega_v4.models.unified_superblock_v4 import BLOCK_BUILDERS


@dataclass(frozen=True)
class DryForwardResult:
    verdict: Literal["ok", "shape_mismatch", "exception"]
    detail: str = ""
    # V7-M0.3: when capture_logits=True the caller gets the final
    # activation tensor (shape (B, S, H)) back as a flat python list +
    # shape tuple. Stays None when capture is off — preserves the
    # zero-overhead default path used by the verify pipeline.
    output_logits: tuple[int, ...] | None = field(default=None)
    output_values: list[float] | None = field(default=None)


def dry_forward(
    graph: BrickGraph,
    *,
    hidden_size: int = 64,
    seq_len: int = 8,
    batch: int = 1,
    capture_logits: bool = False,
    seed: int | None = None,
) -> DryForwardResult:
    """Walk ``graph`` in declared order, forwarding a synthetic activation.

    Parallel-block topologies are handled by averaging branch outputs at
    the fan-in point — matches the convention BrickGraph uses for
    Tiny-Aya style ``GQA‖MLP``.
    """
    try:
        # Build modules fresh — caller may have passed instantiate=False.
        modules: dict[str, object] = {}
        for node in graph.nodes:
            builder = BLOCK_BUILDERS.get(node.kind)
            if builder is None:
                return DryForwardResult(
                    verdict="exception",
                    detail=f"unknown brick kind {node.kind!r}",
                )
            modules[node.name] = builder(hidden_size, dict(node.params))

        # V7-M0.3: deterministic synthetic input when a seed is given so
        # MLX-vs-CUDA parity harness can compare logits bit-for-bit.
        if seed is not None:
            _key = mx.random.key(int(seed))
            x0 = mx.random.normal(
                shape=(batch, seq_len, hidden_size), key=_key)
            token_ids = mx.random.randint(0, 1000, (batch, seq_len), key=_key)
            doc_ids = mx.random.randint(0, 5, (batch, seq_len), key=_key)
        else:
            _key = None
            x0 = mx.random.normal((batch, seq_len, hidden_size))
            token_ids = mx.random.randint(0, 1000, (batch, seq_len))
            doc_ids = mx.random.randint(0, 5, (batch, seq_len))

        def _call(mod: object, x: mx.array) -> mx.array:
            """Call a brick and coerce tuple/dict returns to a single array.

            Some bricks (mamba3, certain ssm wrappers) emit
            ``(activations, state)`` — take the first array-shaped element.
            """
            import inspect
            kwargs = {}
            if hasattr(mod, "__call__"):
                try:
                    sig_params = inspect.signature(mod.__call__).parameters
                    if "token_ids" in sig_params:
                        kwargs["token_ids"] = token_ids
                    if "doc_ids" in sig_params:
                        kwargs["doc_ids"] = doc_ids
                    elif "document_ids" in sig_params:
                        kwargs["document_ids"] = doc_ids
                except Exception:
                    pass
            out = mod(x, **kwargs)  # type: ignore[operator]
            if isinstance(out, tuple) or isinstance(out, list):
                for item in out:
                    if hasattr(item, "shape"):
                        return item
                raise TypeError(
                    f"brick {type(mod).__name__} returned tuple with no array",
                )
            if isinstance(out, dict):
                for v in out.values():
                    if hasattr(v, "shape"):
                        return v
                raise TypeError(
                    f"brick {type(mod).__name__} returned dict with no array",
                )
            if not hasattr(out, "shape"):
                raise TypeError(
                    f"brick {type(mod).__name__} returned {type(out).__name__} "
                    f"with no .shape attribute",
                )
            return out
        # Topo: pre-compute predecessors map; if a node has multiple
        # predecessors, mean-reduce their outputs before forwarding.
        outputs: dict[str, mx.array] = {}
        roots = [n.name for n in graph.nodes if not graph.predecessors(n.name)]
        for name in roots:
            node = graph.by_name(name)
            if node.kind == "embedding_table":
                if seed is not None:
                    x_root = mx.random.randint(0, 1000, (batch, seq_len), key=_key)
                else:
                    x_root = mx.random.randint(0, 1000, (batch, seq_len))
            else:
                x_root = x0
            outputs[name] = _call(modules[name], x_root)
        # Iterate remaining nodes in declared order (graph is already a
        # topological declaration thanks to from_block_specs).
        for node in graph.nodes:
            if node.name in outputs:
                continue
            preds = graph.predecessors(node.name)
            if not preds:
                if node.kind == "embedding_table":
                    if seed is not None:
                        x_root = mx.random.randint(0, 1000, (batch, seq_len), key=_key)
                    else:
                        x_root = mx.random.randint(0, 1000, (batch, seq_len))
                else:
                    x_root = x0
                outputs[node.name] = _call(modules[node.name], x_root)
                continue
            if len(preds) == 1:
                inp = outputs[preds[0]]
            else:
                stacked = mx.stack([outputs[p] for p in preds], axis=0)
                inp = mx.mean(stacked, axis=0)
            outputs[node.name] = _call(modules[node.name], inp)

        last_node = graph.nodes[-1]
        y = outputs[last_node.name]
        
        expected_shape = (batch, seq_len, hidden_size)
        if last_node.kind == "embedding_table":
            vocab_size = last_node.params.get("vocab_size", 65536)
            expected_shape = (batch, seq_len, vocab_size)
            
        if y.shape != expected_shape:
            return DryForwardResult(
                verdict="shape_mismatch",
                detail=f"final shape {y.shape} != expected {expected_shape}",
            )
        # V7-M0.3: optionally surface the final-block activations as
        # the "logits proxy" for parity diffing. The graph passed here
        # is the brick stack alone — no LM head — so this is the
        # hidden-state output before token projection.
        if capture_logits:
            try:
                mx.eval(y)
                flat = y.flatten().tolist()
            except Exception:
                flat = None
            return DryForwardResult(
                verdict="ok",
                output_logits=tuple(int(d) for d in y.shape),
                output_values=flat,
            )
        return DryForwardResult(verdict="ok")
    except Exception as exc:
        return DryForwardResult(
            verdict="exception",
            detail=f"{type(exc).__name__}: {exc}",
        )
