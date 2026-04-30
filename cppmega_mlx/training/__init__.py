"""Training, loss, and checkpoint helpers."""

from cppmega_mlx.training.compiled import (
    CompiledPretrainingStep,
    PretrainingMetrics,
    PretrainingState,
    STABLE_BATCH_KEYS,
    normalize_compiled_batch,
)
from cppmega_mlx.training.eval import EvalMetrics, evaluate_batches
from cppmega_mlx.training.mlx_lm_adapter import (
    MLXLMAPIInfo,
    as_mlx_lm_loss_args,
    as_mlx_lm_token_mapping,
    describe_mlx_lm_trainer_apis,
)

__all__ = [
    "CompiledPretrainingStep",
    "EvalMetrics",
    "MLXLMAPIInfo",
    "as_mlx_lm_loss_args",
    "as_mlx_lm_token_mapping",
    "describe_mlx_lm_trainer_apis",
    "evaluate_batches",
    "normalize_compiled_batch",
    "PretrainingMetrics",
    "PretrainingState",
    "STABLE_BATCH_KEYS",
]
