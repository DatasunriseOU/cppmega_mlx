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


def load_sharded(index_path: str | pathlib.Path) -> dict[str, mx.array]:
    index_path = pathlib.Path(index_path)
    index = json.loads(index_path.read_text())
    base = index_path.parent
    out: dict[str, mx.array] = {}
    seen_shards: set[str] = set()
    for k, shard_name in index["weight_map"].items():
        if shard_name in seen_shards:
            continue
        seen_shards.add(shard_name)
        out.update(st.load_file(str(base / shard_name)))
    return out


def load_sharded_for_rank(index_path: str | pathlib.Path,
                           *, rank: int, world_size: int
                           ) -> dict[str, mx.array]:
    """Round-robin shard ownership: rank r reads shards i where
    i % world_size == r."""
    if world_size <= 0 or not (0 <= rank < world_size):
        raise ValueError("invalid rank/world_size")
    index_path = pathlib.Path(index_path)
    index = json.loads(index_path.read_text())
    base = index_path.parent
    num_shards = int(index["metadata"]["num_shards"])
    owned = {i for i in range(num_shards) if i % world_size == rank}
    out: dict[str, mx.array] = {}
    for k, shard_name in index["weight_map"].items():
        # Reverse-derive shard index from filename "*-NNNNN-of-MMMMM"
        try:
            idx = int(shard_name.split("-")[-3]) - 1
        except (ValueError, IndexError):
            continue
        if idx in owned:
            if shard_name not in {n for n in
                                  (index["weight_map"][kk]
                                   for kk in index["weight_map"])}:
                continue
            # Load each shard once.
            if shard_name not in out.get("__loaded__", set()):
                out.update(st.load_file(str(base / shard_name)))
    return out


__all__ = ["save_sharded", "load_sharded", "load_sharded_for_rank"]
