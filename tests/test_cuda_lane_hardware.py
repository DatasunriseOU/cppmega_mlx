from __future__ import annotations

import os

import pytest


@pytest.mark.cuda_hardware
@pytest.mark.cuda_required
def test_cuda_lane_executes_a_real_gpu_matmul() -> None:
    if os.environ.get("CPPMEGA_RUN_CUDA_TESTS") != "1":
        pytest.skip("CUDA lane opt-in is not enabled")

    import torch

    assert torch.cuda.is_available()
    assert torch.version.cuda
    device = torch.device("cuda")
    left = torch.ones((32, 32), device=device, dtype=torch.float16)
    right = torch.full((32, 32), 2, device=device, dtype=torch.float16)
    result = left @ right
    torch.cuda.synchronize()
    assert torch.all(result == 64)
