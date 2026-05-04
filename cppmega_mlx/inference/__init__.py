"""MLX-native inference helpers."""

from cppmega_mlx.inference.engine import (
    ContiguousKVCache,
    ContiguousKVCacheConfig,
    PromptCacheEntry,
    clone_contiguous_kv_cache,
    kv_cache_position,
    make_contiguous_kv_cache,
    prefill_contiguous_kv_cache,
    rollback_contiguous_kv_cache,
    trim_contiguous_kv_cache,
)
from cppmega_mlx.inference.infilling import build_fim_prompt_ids
from cppmega_mlx.inference.generation import (
    GenerationChunk,
    build_prompt_cache,
    generate_tokens,
    generate_tokens_with_prompt_cache,
    generate_tokens_with_kv_cache,
    next_token_logits,
    stream_generate_tokens,
)
from cppmega_mlx.inference.quantization import (
    InferenceQuantizationConfig,
    make_quantized_kv_cache,
    quantize_kv_cache,
    quantize_linear_for_inference,
    should_start_kv_quantization,
    validate_kv_head_dim,
)
from cppmega_mlx.inference.sampling import sample_next_token
from cppmega_mlx.inference.serving import (
    ContinuousBatchScheduler,
    PAGED_ATTENTION_NOT_INTEGRATED_MESSAGE,
    PagedKVBlockManager,
    PagedKVBlockManagerConfig,
    SchedulerOutput,
    SequenceRequest,
    build_paged_block_table,
    require_model_integrated_paged_attention,
)
from cppmega_mlx.inference.speculative_decode import (
    speculative_acceptance,
    speculative_acceptance_batch,
    typical_acceptance,
    typical_acceptance_batch,
)

__all__ = [
    "ContiguousKVCache",
    "ContiguousKVCacheConfig",
    "ContinuousBatchScheduler",
    "GenerationChunk",
    "InferenceQuantizationConfig",
    "PAGED_ATTENTION_NOT_INTEGRATED_MESSAGE",
    "PagedKVBlockManager",
    "PagedKVBlockManagerConfig",
    "PromptCacheEntry",
    "SchedulerOutput",
    "SequenceRequest",
    "build_prompt_cache",
    "build_fim_prompt_ids",
    "build_paged_block_table",
    "clone_contiguous_kv_cache",
    "generate_tokens",
    "generate_tokens_with_prompt_cache",
    "generate_tokens_with_kv_cache",
    "kv_cache_position",
    "make_contiguous_kv_cache",
    "make_quantized_kv_cache",
    "next_token_logits",
    "prefill_contiguous_kv_cache",
    "quantize_kv_cache",
    "quantize_linear_for_inference",
    "require_model_integrated_paged_attention",
    "rollback_contiguous_kv_cache",
    "sample_next_token",
    "should_start_kv_quantization",
    "speculative_acceptance",
    "speculative_acceptance_batch",
    "stream_generate_tokens",
    "trim_contiguous_kv_cache",
    "typical_acceptance",
    "typical_acceptance_batch",
    "validate_kv_head_dim",
]
