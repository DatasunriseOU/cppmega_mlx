"""MLX domain/role/confidence embeddings for cppmega world-code models."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import numpy as np


class DomainEmbedding(nn.Module):
    """Additive token embedding over the stable domain-routing sidecars."""

    COMPONENTS = ("domain", "role", "confidence")

    def __init__(
        self,
        *,
        hidden_size: int,
        num_domains: int = 64,
        num_roles: int = 128,
        num_confidences: int = 8,
        bottleneck_dim: int = 32,
    ) -> None:
        super().__init__()
        for name, value in (
            ("hidden_size", hidden_size),
            ("num_domains", num_domains),
            ("num_roles", num_roles),
            ("num_confidences", num_confidences),
            ("bottleneck_dim", bottleneck_dim),
        ):
            if int(value) <= 0:
                raise ValueError(f"{name} must be positive, got {value}")

        self.hidden_size = int(hidden_size)
        self.bottleneck_dim = int(bottleneck_dim)
        vocab_sizes = (int(num_domains), int(num_roles), int(num_confidences))
        offsets = (0, vocab_sizes[0], vocab_sizes[0] + vocab_sizes[1])
        self._comp_offsets = mx.array(offsets, dtype=mx.int64)
        self._comp_max = mx.array(
            [vocab_size - 1 for vocab_size in vocab_sizes],
            dtype=mx.int64,
        )
        self.stacked_emb = nn.Embedding(sum(vocab_sizes), self.bottleneck_dim)
        self.up_proj = nn.Linear(self.bottleneck_dim, self.hidden_size, bias=False)
        self.stacked_emb.weight = mx.zeros_like(self.stacked_emb.weight)
        self.component_scales = mx.full(
            (len(self.COMPONENTS),),
            1.0 / len(self.COMPONENTS),
            dtype=mx.float32,
        )

    def __call__(
        self,
        *,
        domain_ids: mx.array | None,
        role_ids: mx.array | None,
        confidence_ids: mx.array | None,
        target_dtype: mx.Dtype | None = None,
    ) -> mx.array:
        inputs = {
            "domain": domain_ids,
            "role": role_ids,
            "confidence": confidence_ids,
        }
        ref = next((value for value in inputs.values() if value is not None), None)
        if ref is None:
            raise ValueError(
                "[cppmega-domain] domain embedding is enabled but all domain "
                "sidecars are absent"
            )
        if not isinstance(ref, mx.array) or ref.ndim != 2:
            shape = getattr(ref, "shape", None)
            raise ValueError(f"domain sidecars must be shaped (B, S), got {shape}")

        expected_shape = tuple(ref.shape)
        ids_list: list[mx.array] = []
        present: list[float] = []
        for index, name in enumerate(self.COMPONENTS):
            tensor = inputs[name]
            if tensor is None:
                ids_list.append(mx.zeros(expected_shape, dtype=mx.int64))
                present.append(0.0)
                continue
            if not isinstance(tensor, mx.array) or tensor.ndim != 2:
                shape = getattr(tensor, "shape", None)
                raise ValueError(f"{name}_ids must be shaped (B, S), got {shape}")
            if tuple(tensor.shape) != expected_shape:
                raise ValueError(
                    f"{name}_ids shape {tuple(tensor.shape)} must match "
                    f"{expected_shape}"
                )
            ids = tensor.astype(mx.int64)
            max_id = int(self._comp_max[index].item())
            invalid = mx.any((ids < 0) | (ids > max_id))
            mx.eval(invalid)
            if bool(invalid.item()):
                values = np.asarray(ids)
                bad = values[(values < 0) | (values > max_id)][:8].tolist()
                raise ValueError(
                    f"[cppmega-domain] {name}_ids out of range [0,{max_id}]: "
                    f"offending values {bad}; refusing to clamp"
                )
            ids_list.append(ids + self._comp_offsets[index])
            present.append(1.0)

        stacked_ids = mx.stack(ids_list, axis=-1)
        embeddings = self.stacked_emb.weight[stacked_ids]
        present_mask = mx.array(present, dtype=self.component_scales.dtype)
        scales = self.component_scales * present_mask
        weighted = mx.sum(
            embeddings * mx.reshape(scales, (1, 1, -1, 1)),
            axis=2,
        )
        dtype = target_dtype or self.up_proj.weight.dtype
        if weighted.dtype != dtype:
            weighted = weighted.astype(dtype)
        weight = self.up_proj.weight
        if weight.dtype != dtype:
            weight = weight.astype(dtype)
        return weighted @ weight.T


CppMegaDomainEmbedding = DomainEmbedding

__all__ = ["CppMegaDomainEmbedding", "DomainEmbedding"]
