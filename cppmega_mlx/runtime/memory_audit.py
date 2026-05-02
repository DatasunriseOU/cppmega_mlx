"""MLX-native runtime memory audit.

Port of ``cppmega/tools/memory_dtype_audit.py`` from the CUDA/Megatron stack.
Reports param/grad/optimizer/device-cache memory bucketed by (category, dtype)
so we can verify exactly *where* memory goes and reconcile against the
calibrated 1.2B target for the ``local_gb10_quarter`` profile.

Usage:

    from cppmega_mlx.runtime.memory_audit import audit_memory, format_report
    report = audit_memory(model, optimizer=opt, tag="post-init")
    print(format_report(report))
    # JSON dump:
    import json
    json.dump(report, open("memory_audit.json", "w"), indent=2)

Categories follow the upstream torch tool so receipts cross-reference:

* ``mamba_scalar_bc`` — Mamba/Mamba3 1D/3D scalars (``A_log``, ``D``,
  ``dt_bias``, ``B_bias``, ``C_bias``, ``B_norm_weight``, ``C_norm_weight``,
  ``mimo_x``, ``mimo_z``, ``mimo_o``).
* ``scalar_fallback_embedding_or_output`` — token/position embeddings, lm head.
* ``scalar_fallback_non_2d`` — anything not 2D and not in the mamba leaf set.
* ``muon_matrix`` — 2D weight matrices (the bulk of an LM).

Device cache: pulls from ``mx.metal`` if available.
"""

from __future__ import annotations

import os
import time
from collections.abc import Iterable
from typing import Any

import mlx.core as mx
import mlx.nn as nn

_MAMBA_SCALAR_LEAVES = frozenset(
    {
        "A_log",
        "dt_bias",
        "D",
        "B_bias",
        "C_bias",
        "B_norm_weight",
        "C_norm_weight",
        "mimo_x",
        "mimo_z",
        "mimo_o",
    }
)
_EMBEDDING_LIKE_HINTS = ("embed", "embedding", "lm_head", "wte", "wpe")


def _gib(nbytes: int | float) -> float:
    return float(nbytes) / (1024**3)


def _array_nbytes(value: mx.array) -> int:
    return int(value.size * value.dtype.size)


def _flatten_named_arrays(
    tree: Any, prefix: str = ""
) -> Iterable[tuple[str, mx.array]]:
    if isinstance(tree, dict):
        for key, value in tree.items():
            full = f"{prefix}.{key}" if prefix else str(key)
            yield from _flatten_named_arrays(value, full)
    elif isinstance(tree, (list, tuple)):
        for idx, value in enumerate(tree):
            full = f"{prefix}.{idx}" if prefix else str(idx)
            yield from _flatten_named_arrays(value, full)
    elif isinstance(tree, mx.array):
        yield prefix, tree


def _param_category(name: str, param: mx.array) -> str:
    leaf = name.rsplit(".", 1)[-1]
    if leaf in _MAMBA_SCALAR_LEAVES:
        return "mamba_scalar_bc"
    name_lower = name.lower()
    if any(hint in name_lower for hint in _EMBEDDING_LIKE_HINTS):
        return "scalar_fallback_embedding_or_output"
    if param.ndim != 2:
        return "scalar_fallback_non_2d"
    return "muon_matrix"


def _new_bucket() -> dict[str, Any]:
    return {"count": 0, "numel": 0, "bytes": 0}


def _add(buckets: dict[str, dict[str, Any]], key: str, *, numel: int, nbytes: int) -> None:
    bucket = buckets.setdefault(key, _new_bucket())
    bucket["count"] += 1
    bucket["numel"] += int(numel)
    bucket["bytes"] += int(nbytes)


def _collect_params(model: nn.Module) -> dict[str, Any]:
    by_dtype: dict[str, dict[str, Any]] = {}
    by_category: dict[str, dict[str, Any]] = {}
    by_top: dict[str, dict[str, Any]] = {}
    leaves: list[dict[str, Any]] = []
    aliases: dict[int, list[str]] = {}
    seen_ids: set[int] = set()

    for name, param in _flatten_named_arrays(model.parameters()):
        pid = id(param)
        aliases.setdefault(pid, []).append(name)
        if pid in seen_ids:
            # Same underlying array exposed under another attribute name —
            # MLX returns both paths in the param tree but the storage is
            # shared. Skip in totals.
            continue
        seen_ids.add(pid)
        category = _param_category(name, param)
        nbytes = _array_nbytes(param)
        dtype_str = str(param.dtype).split(".")[-1]
        top = name.split(".", 1)[0]
        _add(by_dtype, dtype_str, numel=param.size, nbytes=nbytes)
        _add(by_category, f"{category}|{dtype_str}", numel=param.size, nbytes=nbytes)
        _add(by_top, top, numel=param.size, nbytes=nbytes)
        leaves.append(
            {
                "name": name,
                "category": category,
                "dtype": dtype_str,
                "shape": list(param.shape),
                "numel": int(param.size),
                "bytes": nbytes,
            }
        )

    leaves.sort(key=lambda entry: (-entry["bytes"], entry["name"]))

    aliased = sorted(
        (
            {
                "primary_name": names[0],
                "alias_names": names[1:],
                "alias_count": len(names) - 1,
            }
            for names in aliases.values()
            if len(names) > 1
        ),
        key=lambda entry: entry["primary_name"],
    )

    return {
        "model_params_by_dtype": by_dtype,
        "model_params_by_category_dtype": by_category,
        "model_params_by_top_module": by_top,
        "model_param_leaves": leaves,
        "model_param_count": sum(b["numel"] for b in by_dtype.values()),
        "model_param_bytes": sum(b["bytes"] for b in by_dtype.values()),
        "model_param_aliased_arrays": aliased,
        "model_param_unique_arrays": len(seen_ids),
    }


def _collect_optimizer(optimizer: Any | None) -> dict[str, Any]:
    if optimizer is None:
        return {"optimizer_state_by_dtype": {}, "optimizer_state_by_key_dtype": {}}

    state = getattr(optimizer, "state", None)
    if state is None:
        return {"optimizer_state_by_dtype": {}, "optimizer_state_by_key_dtype": {}}

    by_dtype: dict[str, dict[str, Any]] = {}
    by_key_dtype: dict[str, dict[str, Any]] = {}

    if isinstance(state, dict):
        candidates = state
    else:
        candidates = {"state": state}

    for state_key, sub in candidates.items():
        if isinstance(sub, mx.array):
            nbytes = _array_nbytes(sub)
            dtype_str = str(sub.dtype).split(".")[-1]
            _add(by_dtype, dtype_str, numel=sub.size, nbytes=nbytes)
            _add(by_key_dtype, f"{state_key}|{dtype_str}", numel=sub.size, nbytes=nbytes)
            continue
        for name, arr in _flatten_named_arrays(sub):
            nbytes = _array_nbytes(arr)
            dtype_str = str(arr.dtype).split(".")[-1]
            _add(by_dtype, dtype_str, numel=arr.size, nbytes=nbytes)
            _add(by_key_dtype, f"{state_key}|{dtype_str}", numel=arr.size, nbytes=nbytes)

    return {
        "optimizer_state_by_dtype": by_dtype,
        "optimizer_state_by_key_dtype": by_key_dtype,
    }


def _collect_device_metal() -> dict[str, Any]:
    out: dict[str, Any] = {}
    for fn_name in (
        "get_active_memory",
        "get_cache_memory",
        "get_peak_memory",
        "device_info",
    ):
        fn = getattr(mx, fn_name, None)
        if not callable(fn):
            metal = getattr(mx, "metal", None)
            fn = getattr(metal, fn_name, None) if metal is not None else None
        if not callable(fn):
            continue
        try:
            out[fn_name] = fn()
        except Exception as exc:  # pragma: no cover - hardware-dependent
            out[fn_name] = f"{type(exc).__name__}: {exc}"
    return out


def audit_memory(
    model: nn.Module,
    *,
    optimizer: Any | None = None,
    tag: str = "audit",
) -> dict[str, Any]:
    """Return a structured memory audit for ``model`` (and optional ``optimizer``)."""

    return {
        "tag": tag,
        "time_unix": time.time(),
        "pid": os.getpid(),
        "device_metal": _collect_device_metal(),
        **_collect_params(model),
        **_collect_optimizer(optimizer),
    }


def format_report(report: dict[str, Any], *, top_n: int = 30) -> str:
    """Pretty-print an audit report with Gib totals."""

    lines: list[str] = []
    lines.append(f"[memory_audit] tag={report['tag']}")
    total_p = report.get("model_param_count", 0)
    total_b = report.get("model_param_bytes", 0)
    lines.append(
        f"[memory_audit] total params: {total_p:,} ({total_p/1e9:.3f}B); "
        f"bytes: {total_b:,} ({_gib(total_b):.3f} GiB)"
    )

    metal = report.get("device_metal", {})
    if metal:
        lines.append("[memory_audit] device_metal:")
        for k, v in sorted(metal.items()):
            if isinstance(v, (int, float)):
                lines.append(f"  {k}: {v:,} ({_gib(v):.3f} GiB)")
            else:
                lines.append(f"  {k}: {v}")

    for section in (
        "model_params_by_dtype",
        "model_params_by_category_dtype",
        "model_params_by_top_module",
        "optimizer_state_by_dtype",
        "optimizer_state_by_key_dtype",
    ):
        rows = report.get(section, {})
        lines.append(f"[memory_audit] {section}")
        if not rows:
            lines.append("  <empty>")
            continue
        ranked = sorted(rows.items(), key=lambda item: -item[1]["bytes"])[:top_n]
        for key, value in ranked:
            lines.append(
                f"  {key:60s} count={value['count']:5d} "
                f"numel={value['numel']:14,} bytes={value['bytes']:14,} "
                f"gib={_gib(value['bytes']):.4f}"
            )

    leaves = report.get("model_param_leaves", [])
    if leaves:
        lines.append(f"[memory_audit] top {top_n} largest leaves")
        for leaf in leaves[:top_n]:
            lines.append(
                f"  {leaf['name']:60s} {leaf['category']:38s} {leaf['dtype']:8s} "
                f"shape={leaf['shape']!s:24s} bytes={leaf['bytes']:14,} "
                f"gib={_gib(leaf['bytes']):.4f}"
            )
    return "\n".join(lines)


__all__ = ["audit_memory", "format_report"]
