"""V7-C02: sharded safetensors save with index.json manifest.

  save_sharded(state, out_dir, prefix='model', num_shards=N)
    Writes:
      out_dir/{prefix}.shard-00001-of-NNNNN.safetensors  (N files)
      out_dir/{prefix}.index.json
        { "metadata": {"total_size": int, "num_shards": int},
          "weight_map": {"param_key": "shard_file_basename"} }

  load_sharded(index_path) → dict[str, mx.array]
    Reads each shard listed in the index and merges into one dict.

  load_sharded_for_rank(index_path, rank, world_size) → dict
    Skips shards not owned by this rank (round-robin assignment).
"""

from __future__ import annotations

import json
import pathlib
from typing import Any

import mlx.core as mx
import safetensors.mlx as st


def _shard_name(prefix: str, idx: int, total: int) -> str:
    return f"{prefix}.shard-{idx + 1:05d}-of-{total:05d}.safetensors"


def save_sharded(state: dict[str, mx.array],
                  out_dir: str | pathlib.Path,
                  *, prefix: str = "model",
                  num_shards: int = 1,
                  metadata: dict[str, str] | None = None
                  ) -> dict[str, Any]:
    if num_shards <= 0:
        raise ValueError("num_shards must be > 0")
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    keys = sorted(state.keys())
    shards: list[dict[str, mx.array]] = [
        {} for _ in range(num_shards)]
    weight_map: dict[str, str] = {}
    total_size = 0
    for i, k in enumerate(keys):
        shard_idx = i % num_shards
        shards[shard_idx][k] = state[k]
        fname = _shard_name(prefix, shard_idx, num_shards)
        weight_map[k] = fname
        if hasattr(state[k], "shape"):
            total_size += int(state[k].size) * int(state[k].dtype.size)
    for i, shard in enumerate(shards):
        fname = _shard_name(prefix, i, num_shards)
        st.save_file(shard, str(out_dir / fname), metadata=metadata)
    index = {
        "metadata": {"total_size": int(total_size),
                      "num_shards": int(num_shards)},
        "weight_map": weight_map,
    }
    index_path = out_dir / f"{prefix}.index.json"
    index_path.write_text(json.dumps(index, indent=2, sort_keys=True))
    return index


def _shard_index_from_name(shard_name: str) -> int | None:
    """Parse shard index from '<prefix>.shard-NNNNN-of-MMMMM.safetensors'."""
    try:
        return int(shard_name.split("-")[-3]) - 1
    except (ValueError, IndexError):
        return None


def load_sharded(
    index_path: str | pathlib.Path,
    *, _opened: list[str] | None = None,
) -> dict[str, mx.array]:
    """Read every shard listed in the index manifest into a single dict.

    ``_opened`` is a test-only sink — when supplied, each shard's absolute
    path is appended exactly once. Lets tests assert which files were
    actually opened on disk (V7-C02 AC#2)."""
    index_path = pathlib.Path(index_path)
    index = json.loads(index_path.read_text())
    base = index_path.parent
    out: dict[str, mx.array] = {}
    seen_shards: set[str] = set()
    for shard_name in index["weight_map"].values():
        if shard_name in seen_shards:
            continue
        seen_shards.add(shard_name)
        path = str(base / shard_name)
        if _opened is not None:
            _opened.append(path)
        out.update(st.load_file(path))
    return out


def load_sharded_for_rank(
    index_path: str | pathlib.Path,
    *, rank: int, world_size: int,
    _opened: list[str] | None = None,
) -> dict[str, mx.array]:
    """Round-robin shard ownership: rank r reads shards i where
    i % world_size == r. Each owned shard is opened **exactly once**.

    ``_opened`` is the same test-only sink as ``load_sharded``."""
    if world_size <= 0 or not (0 <= rank < world_size):
        raise ValueError(
            f"invalid rank/world_size: rank={rank}, world_size={world_size}")
    index_path = pathlib.Path(index_path)
    index = json.loads(index_path.read_text())
    base = index_path.parent
    num_shards = int(index["metadata"]["num_shards"])
    owned = {i for i in range(num_shards) if i % world_size == rank}
    out: dict[str, mx.array] = {}
    # Deduplicate shard filenames before opening — the previous impl
    # iterated weight_map keys and re-opened the same file once per
    # tensor, which scaled O(num_params) instead of O(num_owned_shards).
    seen_shards: set[str] = set()
    for shard_name in index["weight_map"].values():
        if shard_name in seen_shards:
            continue
        idx = _shard_index_from_name(shard_name)
        if idx is None or idx not in owned:
            continue
        seen_shards.add(shard_name)
        path = str(base / shard_name)
        if _opened is not None:
            _opened.append(path)
        out.update(st.load_file(path))
    return out


def load_with_backward_compat(
    path: str | pathlib.Path,
    *, _opened: list[str] | None = None,
) -> dict[str, mx.array]:
    """V7-C02 AC#3: accept either a sharded index.json OR a legacy
    single-file safetensors checkpoint.

    If ``path`` ends in ``.index.json`` (or names an index.json file
    that exists), delegate to :func:`load_sharded`. Otherwise treat the
    path as a flat safetensors file and read it directly.
    """
    p = pathlib.Path(path)
    if p.is_dir():
        # Convenience: caller passed a checkpoint directory — pick the
        # canonical model.index.json.
        cand = p / "model.index.json"
        if cand.is_file():
            return load_sharded(cand, _opened=_opened)
        raise FileNotFoundError(f"no model.index.json under {p}")
    if p.name.endswith(".index.json"):
        return load_sharded(p, _opened=_opened)
    # Legacy single-file path.
    if _opened is not None:
        _opened.append(str(p))
    return st.load_file(str(p))


__all__ = [
    "save_sharded", "load_sharded", "load_sharded_for_rank",
    "load_with_backward_compat",
]
