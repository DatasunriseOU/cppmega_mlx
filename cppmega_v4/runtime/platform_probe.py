"""Dynamic hardware/runtime capability detection for cppmega_v4.

This module probes the local platform to identify the available accelerator hardware
(MLX, CUDA, TPU, or fallback CPU) and returns the topologies and communication
backends supported by the active environment.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def probe_platform() -> dict[str, Any]:
    """Identify the active accelerator platform, supported topologies, and backends.

    Returns:
        dict carrying:
            - "active_device": str ("mlx", "cuda", "tpu", or "cpu")
            - "available_topologies": list[str]
            - "available_comm_backends": list[str]
    """
    has_mlx = False
    has_cuda = False
    has_tpu = False

    # 1. Check for MLX (Apple Silicon)
    try:
        import mlx.core as mx
        # Verify MLX is actually usable
        _ = mx.zeros((4,))
        has_mlx = True
    except ImportError:
        pass
    except Exception as exc:
        logger.warning("MLX imported but failed to initialize basic tensor: %s", exc)

    # 2. Check for CUDA/PyTorch
    if not has_mlx:
        try:
            import torch
            if torch.cuda.is_available():
                has_cuda = True
        except ImportError:
            pass

    # 3. Check for TPU/XLA
    if not has_mlx and not has_cuda:
        try:
            import torch_xla
            has_tpu = True
        except ImportError:
            pass
        except Exception:
            # Check environment variables as fallback
            import os
            if "TPU_NAME" in os.environ or "PJRT_DEVICE" in os.environ:
                has_tpu = True

    # Assign active device and appropriate capabilities
    if has_mlx:
        active_device = "mlx"
        available_topologies = ["m3_ultra_solo", "gb10_quarter"]
        available_comm_backends = ["ring", "jaccl"]
    elif has_cuda:
        active_device = "cuda"
        available_topologies = ["h100_8x", "h200_8x", "a100_8x", "b100_8x"]
        available_comm_backends = ["nccl", "mpi"]
    elif has_tpu:
        active_device = "tpu"
        available_topologies = ["tpu_v6e_8", "tpu_v5p_4"]
        available_comm_backends = ["pjrt"]
    else:
        active_device = "cpu"
        available_topologies = ["m3_ultra_solo", "gb10_quarter"]  # Safe fallback for simulation
        available_comm_backends = ["ring"]

    return {
        "active_device": active_device,
        "available_topologies": available_topologies,
        "available_comm_backends": available_comm_backends,
    }
