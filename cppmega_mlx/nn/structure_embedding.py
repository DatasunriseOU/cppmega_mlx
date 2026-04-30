"""MLX structure-aware token embedding for cppmega input enrichment."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


class StructureEmbedding(nn.Module):
    """Additive token-level embedding for cppmega structure metadata."""

    ALL_COMPONENTS = (
        "structure",
        "dep_level",
        "ast_depth",
        "sibling_index",
        "ast_node_type",
    )
    CORE_COMPONENTS = ("structure", "dep_level")

    def __init__(
        self,
        *,
        hidden_size: int,
        num_categories: int = 9,
        max_dep_level: int = 16,
        max_ast_depth: int = 64,
        max_sibling_index: int = 64,
        num_node_types: int = 256,
        active_components: str = "core",
        bottleneck_dim: int = 64,
    ) -> None:
        super().__init__()
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if num_categories <= 0:
            raise ValueError("num_categories must be positive")
        if max_dep_level <= 0:
            raise ValueError("max_dep_level must be positive")
        if max_ast_depth <= 0:
            raise ValueError("max_ast_depth must be positive")
        if max_sibling_index <= 0:
            raise ValueError("max_sibling_index must be positive")
        if num_node_types <= 0:
            raise ValueError("num_node_types must be positive")
        if bottleneck_dim <= 0:
            raise ValueError("bottleneck_dim must be positive")

        self.hidden_size = int(hidden_size)
        self.bottleneck_dim = int(bottleneck_dim)
        self.active_component_names = self._parse_components(active_components)

        vocab_sizes = {
            "structure": int(num_categories),
            "dep_level": int(max_dep_level),
            "ast_depth": int(max_ast_depth),
            "sibling_index": int(max_sibling_index),
            "ast_node_type": int(num_node_types),
        }

        offsets: list[int] = []
        clamp_max: list[int] = []
        total_vocab = 0
        for name in self.active_component_names:
            offsets.append(total_vocab)
            total_vocab += vocab_sizes[name]
            clamp_max.append(vocab_sizes[name] - 1)

        self._comp_offsets = mx.array(offsets, dtype=mx.int64)
        self._comp_clamp_max = mx.array(clamp_max, dtype=mx.int64)
        self.stacked_emb = nn.Embedding(total_vocab, self.bottleneck_dim)
        self.up_proj = nn.Linear(self.bottleneck_dim, self.hidden_size, bias=False)
        self.stacked_emb.weight = mx.zeros_like(self.stacked_emb.weight)
        self.up_proj.weight = mx.zeros_like(self.up_proj.weight)
        self.component_scales = mx.full(
            (len(self.active_component_names),),
            1.0 / len(self.active_component_names),
            dtype=mx.float32,
        )

    @classmethod
    def _parse_components(cls, spec: str) -> tuple[str, ...]:
        if spec == "all":
            return cls.ALL_COMPONENTS
        if spec == "core":
            return cls.CORE_COMPONENTS
        requested = {item.strip() for item in spec.split(",") if item.strip()}
        invalid = sorted(requested - set(cls.ALL_COMPONENTS))
        if invalid:
            raise ValueError(f"unknown structure components: {invalid!r}")
        if not requested:
            raise ValueError("active_components must select at least one component")
        return tuple(name for name in cls.ALL_COMPONENTS if name in requested)

    @staticmethod
    def _shape_of(tensor: mx.array | None) -> tuple[int, int] | None:
        if tensor is None:
            return None
        if len(tensor.shape) != 2:
            raise ValueError(f"structure inputs must have shape (batch, seq), got {tensor.shape}")
        return (int(tensor.shape[0]), int(tensor.shape[1]))

    def _collect_inputs(
        self,
        *,
        structure_ids: mx.array | None,
        dep_levels: mx.array | None,
        ast_depth_ids: mx.array | None,
        sibling_index_ids: mx.array | None,
        node_type_ids: mx.array | None,
    ) -> dict[str, mx.array | None]:
        inputs = {
            "structure": structure_ids,
            "dep_level": dep_levels,
            "ast_depth": ast_depth_ids,
            "sibling_index": sibling_index_ids,
            "ast_node_type": node_type_ids,
        }
        ref_shape = next(
            (shape for name in self.active_component_names if (shape := self._shape_of(inputs[name])) is not None),
            None,
        )
        if ref_shape is None:
            return inputs
        for name in self.active_component_names:
            shape = self._shape_of(inputs[name])
            if shape is not None and shape != ref_shape:
                raise ValueError(
                    f"structure input {name} shape {shape} does not match reference shape {ref_shape}"
                )
        return inputs

    def __call__(
        self,
        *,
        structure_ids: mx.array | None,
        dep_levels: mx.array | None,
        ast_depth_ids: mx.array | None = None,
        sibling_index_ids: mx.array | None = None,
        node_type_ids: mx.array | None = None,
        target_dtype: mx.Dtype | None = None,
    ) -> mx.array:
        inputs = self._collect_inputs(
            structure_ids=structure_ids,
            dep_levels=dep_levels,
            ast_depth_ids=ast_depth_ids,
            sibling_index_ids=sibling_index_ids,
            node_type_ids=node_type_ids,
        )
        ref = next((inputs[name] for name in self.active_component_names if inputs[name] is not None), None)
        dtype = target_dtype or self.up_proj.weight.dtype
        if ref is None:
            return mx.array(0.0, dtype=dtype)

        batch, seq = ref.shape
        ids_list: list[mx.array] = []
        present: list[float] = []
        for index, name in enumerate(self.active_component_names):
            tensor = inputs[name]
            if tensor is None:
                ids_list.append(mx.zeros((batch, seq), dtype=mx.int64))
                present.append(0.0)
                continue
            ids = tensor.astype(mx.int64)
            clamped = mx.clip(ids, mx.array(0, dtype=mx.int64), self._comp_clamp_max[index])
            ids_list.append(clamped + self._comp_offsets[index])
            present.append(1.0)

        stacked_ids = mx.stack(ids_list, axis=-1)
        embeddings = self.stacked_emb.weight[stacked_ids]
        present_mask = mx.array(present, dtype=self.component_scales.dtype)
        scales = self.component_scales * present_mask
        weighted = mx.sum(embeddings * mx.reshape(scales, (1, 1, -1, 1)), axis=2)
        if weighted.dtype != dtype:
            weighted = weighted.astype(dtype)
        weight = self.up_proj.weight
        if weight.dtype != dtype:
            weight = weight.astype(dtype)
        return weighted @ weight.T


CppMegaStructureEmbedding = StructureEmbedding

