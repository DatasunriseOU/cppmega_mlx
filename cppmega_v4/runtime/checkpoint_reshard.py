"""V7-C05: tensor-axis re-shard for resume across topology change.

The V7-C02 shard format distributes whole tensors across files by key
(round-robin). For real FSDP-style resume the tensors themselves are
split along a parameter axis (e.g. column-parallel attention output
projection: ``W`` is N×D, shard N along axis-0 into N/M pieces per
rank).

This module provides the axis-wise re-shard primitive:

  resave_along_axis(state, axis_for_key, save_world, load_world, out_dir)
    -> writes load_world-many shards where each shard's tensor for key
       K is the rank-r slice of state[K] along axis_for_key[K].

  reshard_along_axis(state, axis_for_key, target_world)
    -> returns list of dicts (one per target rank) with axis-sliced
       tensors. Used by tests to assert identical model outputs after
       a save_world -> load_world topology change.

Compatibility check ``ensure_axis_divisible`` raises ValueError when
``state[k].shape[axis_for_key[k]]`` is not divisible by target_world,
matching V7-C05 AC#3 ("clear error if M incompatible").
"""

from __future__ import annotations

import json
import pathlib
from typing import Mapping

import mlx.core as mx
import safetensors.mlx as st


def ensure_axis_divisible(
    state: Mapping[str, mx.array],
    axis_for_key: Mapping[str, int],
    target_world: int,
) -> None:
    """Raise ValueError on the first key whose shard axis isn't divisible.

    Mirrors V7-C05 AC#3: n_heads-not-divisible-by-M style mismatches
    must be loud, not silent corruption."""
    if target_world <= 0:
        raise ValueError(f"target_world must be > 0, got {target_world}")
    for k, ax in axis_for_key.items():
        if k not in state:
            continue
        dim = int(state[k].shape[ax])
        if dim % target_world != 0:
            raise ValueError(
                f"key {k!r} dim {dim} along axis {ax} is not divisible "
                f"by target_world {target_world}")


def reshard_along_axis(
    state: Mapping[str, mx.array],
    axis_for_key: Mapping[str, int],
    target_world: int,
) -> list[dict[str, mx.array]]:
    """Split each axis-aware tensor into ``target_world`` equal pieces.

    Keys without an axis entry are replicated (every rank gets a copy)
    — matches how FSDP treats norm scales, bias, embeddings without
    sharding metadata."""
    ensure_axis_divisible(state, axis_for_key, target_world)
    out: list[dict[str, mx.array]] = [{} for _ in range(target_world)]
    for k, t in state.items():
        ax = axis_for_key.get(k)
        if ax is None:
            for r in range(target_world):
                out[r][k] = t
            continue
        chunks = mx.split(t, target_world, axis=ax)
        for r in range(target_world):
            out[r][k] = chunks[r]
    return out


def gather_full_from_axis_shards(
    rank_states: list[dict[str, mx.array]],
    axis_for_key: Mapping[str, int],
) -> dict[str, mx.array]:
    """Inverse of reshard_along_axis: concatenate per-rank pieces back
    into the full tensor. Used to verify save N -> load M -> full
    state round-trip."""
    if not rank_states:
        return {}
    keys = sorted(rank_states[0].keys())
    out: dict[str, mx.array] = {}
    for k in keys:
        ax = axis_for_key.get(k)
        pieces = [rs[k] for rs in rank_states]
        if ax is None:
            out[k] = pieces[0]
        else:
            out[k] = mx.concatenate(pieces, axis=ax)
    return out


def save_resharded(
    state: dict[str, mx.array],
    axis_for_key: Mapping[str, int],
    out_dir: str | pathlib.Path,
    *,
    target_world: int,
    prefix: str = "model",
) -> dict:
    """Write ``target_world`` axis-aware shards + an index manifest
    that records the per-key axis so a future resume can re-gather.
    """
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rank_states = reshard_along_axis(state, axis_for_key, target_world)
    weight_map: dict[str, str] = {}
    for r, rs in enumerate(rank_states):
        fname = f"{prefix}.axis-shard-rank{r:03d}-of-{target_world:03d}.safetensors"
        st.save_file(rs, str(out_dir / fname))
        for k in rs:
            # All ranks own all keys (after sharding); weight_map records
            # rank-0's filename as the canonical lookup target.
            weight_map.setdefault(k, fname)
    index = {
        "format": "axis-shard-v1",
        "target_world": int(target_world),
        "axis_for_key": {k: int(v) for k, v in axis_for_key.items()},
        "weight_map": weight_map,
        "shard_template": f"{prefix}.axis-shard-rank{{rank:03d}}-of-{target_world:03d}.safetensors",
    }
    (out_dir / f"{prefix}.axis-index.json").write_text(
        json.dumps(index, indent=2, sort_keys=True))
    return index


def load_resharded(
    index_path: str | pathlib.Path,
    *,
    load_world: int,
) -> list[dict[str, mx.array]]:
    """Read the axis-shard index, gather the full state, then re-split
    into ``load_world`` shards. Returns one dict per target rank.

    AC#1: the returned per-rank tensors, gathered back, must be bitwise
    equal to the original state — ``test_v7_c05_*`` pins this."""
    index_path = pathlib.Path(index_path)
    index = json.loads(index_path.read_text())
    base = index_path.parent
    if index.get("format") != "axis-shard-v1":
        raise ValueError(f"unsupported index format: {index.get('format')}")
    src_world = int(index["target_world"])
    axis_for_key = {k: int(v) for k, v in index["axis_for_key"].items()}
    template = index["shard_template"]
    src_states: list[dict[str, mx.array]] = []
    for r in range(src_world):
        path = base / template.format(rank=r)
        src_states.append(st.load_file(str(path)))
    full = gather_full_from_axis_shards(src_states, axis_for_key)
    return reshard_along_axis(full, axis_for_key, load_world)


__all__ = [
    "ensure_axis_divisible", "reshard_along_axis",
    "gather_full_from_axis_shards",
    "save_resharded", "load_resharded",
]
