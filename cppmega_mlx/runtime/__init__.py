"""Runtime helpers for local MLX training."""

from cppmega_mlx.runtime.kernel_policy import (
    KernelPath,
    clear_dispatch_log,
    get_dispatch_log,
    record_dispatch,
    selected_path,
)
from cppmega_mlx.runtime.memory import (
    DEFAULT_METAL_RATIO,
    DEFAULT_WIRED_RATIO,
    AppliedMemoryLimits,
    MemoryLimitPlan,
    apply_memory_limit_plan,
    device_total_memory_bytes,
    memory_limit_plan,
)
from cppmega_mlx.runtime.env import (
    RuntimeEnvironment,
    detect_runtime_environment,
)
from cppmega_mlx.runtime.seed import (
    capture_rng_state,
    mlx_rng_state_available,
    restore_rng_state,
    seed_all,
)

__all__ = [
    "AppliedMemoryLimits",
    "DEFAULT_METAL_RATIO",
    "DEFAULT_WIRED_RATIO",
    "KernelPath",
    "MemoryLimitPlan",
    "RuntimeEnvironment",
    "apply_memory_limit_plan",
    "capture_rng_state",
    "clear_dispatch_log",
    "detect_runtime_environment",
    "device_total_memory_bytes",
    "get_dispatch_log",
    "memory_limit_plan",
    "mlx_rng_state_available",
    "record_dispatch",
    "restore_rng_state",
    "seed_all",
    "selected_path",
]
