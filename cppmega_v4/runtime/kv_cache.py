"""V7-F02: KV-cache data structure for autoregressive generation.

Holds per-layer (key, value) tensors that grow by 1 along the
sequence axis on each decode step. The structural piece is here;
hybrid_lm attention integration is a follow-up sub-task.

  cache = KVCache(num_layers=L)
  for step in range(max_new):
      k, v = compute_new_kv(hidden)
      cache.append(layer_idx, k, v)
      keys, vals = cache.get(layer_idx)  # full (B, S_so_far, H)
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx


class KVCache:
    """Append-only key/value buffer per layer.

    Each append on layer L extends both buffers along axis=1 (seq).
    get(L) returns the full concatenated (B, S_so_far, H_k/H_v).
    """

    def __init__(self, num_layers: int) -> None:
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        self._num_layers = num_layers
        self._keys: list[mx.array | None] = [None] * num_layers
        self._values: list[mx.array | None] = [None] * num_layers

    @property
    def num_layers(self) -> int:
        return self._num_layers

    def length(self, layer: int) -> int:
        k = self._keys[layer]
        return 0 if k is None else int(k.shape[1])

    def append(self, layer: int,
               new_keys: mx.array, new_values: mx.array) -> None:
        if new_keys.shape[1] != new_values.shape[1]:
            raise ValueError(
                "key/value seq dims must match: "
                f"{new_keys.shape} vs {new_values.shape}"
            )
        if self._keys[layer] is None:
            self._keys[layer] = new_keys
            self._values[layer] = new_values
            return
        self._keys[layer] = mx.concatenate(
            [self._keys[layer], new_keys], axis=1)
        self._values[layer] = mx.concatenate(
            [self._values[layer], new_values], axis=1)

    def get(self, layer: int) -> tuple[mx.array, mx.array]:
        if self._keys[layer] is None:
            raise ValueError(f"layer {layer} is empty")
        return self._keys[layer], self._values[layer]

    def reset(self) -> None:
        self._keys = [None] * self._num_layers
        self._values = [None] * self._num_layers

    def total_bytes(self) -> int:
        total = 0
        for k, v in zip(self._keys, self._values):
            if k is not None:
                total += int(k.size) * int(k.dtype.size)
            if v is not None:
                total += int(v.size) * int(v.dtype.size)
        return total


__all__ = ["KVCache"]
