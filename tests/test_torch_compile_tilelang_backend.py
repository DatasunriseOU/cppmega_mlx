from __future__ import annotations

import importlib.util

import pytest


def qk_reduce(q, k, indices):  # type: ignore[no-untyped-def]
    del indices
    return q @ k.transpose(-1, -2)


def _torch_or_skip():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch unavailable")
    import torch

    if not hasattr(torch, "compile"):
        pytest.skip("torch.compile unavailable")
    if importlib.util.find_spec("torch._dynamo") is None:
        pytest.skip("torch._dynamo unavailable")
    return torch


def _export_qk_reduce_scale_graph(torch):  # type: ignore[no-untyped-def]
    import torch._dynamo as dynamo

    dynamo.allow_in_graph(qk_reduce)

    def qk_reduce_scale(q, k, indices, sm_scale):  # type: ignore[no-untyped-def]
        return qk_reduce(q, k, indices) * sm_scale

    q = torch.randn(16, 8, dtype=torch.float32)
    k = torch.randn(12, 8, dtype=torch.float32)
    indices = torch.arange(12)
    sm_scale = torch.tensor(0.125, dtype=torch.float32)
    try:
        exported = dynamo.export(qk_reduce_scale)(q, k, indices, sm_scale)
    except TypeError:
        exported = dynamo.export(qk_reduce_scale, q, k, indices, sm_scale)
    graph = (
        exported.graph_module
        if hasattr(exported, "graph_module")
        else exported[0]
    )
    return graph, qk_reduce_scale, (q, k, indices, sm_scale)


def test_tilelang_torch_compile_backend_detects_gemm_softmax_fx_pattern() -> None:
    torch = _torch_or_skip()
    from torch import nn
    from torch.fx import symbolic_trace

    from cppmega_mlx.runtime import torch_compile_backend

    class TinyQKSoftmax(nn.Module):
        def forward(self, q, k):  # type: ignore[no-untyped-def]
            return torch.softmax(q @ k.transpose(-1, -2), dim=-1)

    report = torch_compile_backend.inspect_fx_graph(
        symbolic_trace(TinyQKSoftmax().eval())
    )

    assert report.pattern_hits["gemm_softmax"] == 1
    assert report.op_trace == (
        "placeholder:q",
        "placeholder:k",
        "call_method:transpose",
        "call_function:matmul",
        "call_function:softmax",
        "output:output",
    )


def test_tilelang_torch_compile_backend_preserves_autograd_with_eager_fallback() -> None:
    torch = _torch_or_skip()
    from torch import nn

    from cppmega_mlx.runtime import torch_compile_backend

    if not torch_compile_backend.torch_compile_backend_available():
        pytest.skip("torch._dynamo.register_backend unavailable")

    torch_compile_backend.reset_fusion_hits()
    torch_compile_backend.register_tilelang_backend()

    class TinyQKSoftmax(nn.Module):
        def forward(self, q, k):  # type: ignore[no-untyped-def]
            return torch.softmax(q @ k.transpose(-1, -2), dim=-1)

    model = TinyQKSoftmax().eval()
    q = torch.randn(2, 4, 8, dtype=torch.float32, requires_grad=True)
    k = torch.randn(2, 4, 8, dtype=torch.float32, requires_grad=True)
    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k.detach().clone().requires_grad_(True)

    compiled = torch.compile(model, backend="tilelang", fullgraph=True)
    y = compiled(q, k)
    y_ref = model(q_ref, k_ref)

    torch.testing.assert_close(y, y_ref, rtol=1e-5, atol=1e-6)
    y.square().sum().backward()
    y_ref.square().sum().backward()
    torch.testing.assert_close(q.grad, q_ref.grad, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(k.grad, k_ref.grad, rtol=1e-5, atol=1e-6)

    hits = torch_compile_backend.fusion_hits()
    assert hits["gemm_softmax"] >= 1


def test_tilelang_torch_compile_backend_detects_qk_reduce_sm_scale() -> None:
    torch = _torch_or_skip()
    from torch import nn

    from cppmega_mlx.runtime import torch_compile_backend

    if not torch_compile_backend.torch_compile_backend_available():
        pytest.skip("torch._dynamo.register_backend unavailable")

    import torch._dynamo as dynamo

    dynamo.allow_in_graph(qk_reduce)
    torch_compile_backend.reset_fusion_hits()
    torch_compile_backend.register_tilelang_backend()

    class TinyQKReduceScale(nn.Module):
        def forward(self, q, k, indices, sm_scale):  # type: ignore[no-untyped-def]
            return qk_reduce(q, k, indices) * sm_scale

    model = TinyQKReduceScale().eval()
    q = torch.randn(2, 4, 8, dtype=torch.float32, requires_grad=True)
    k = torch.randn(2, 4, 8, dtype=torch.float32, requires_grad=True)
    indices = torch.arange(4)
    sm_scale = torch.tensor(0.125, dtype=torch.float32)
    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k.detach().clone().requires_grad_(True)

    compiled = torch.compile(model, backend="tilelang", fullgraph=True)
    y = compiled(q, k, indices, sm_scale)
    y_ref = model(q_ref, k_ref, indices, sm_scale)

    torch.testing.assert_close(y, y_ref, rtol=1e-5, atol=1e-6)
    y.square().sum().backward()
    y_ref.square().sum().backward()
    torch.testing.assert_close(q.grad, q_ref.grad, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(k.grad, k_ref.grad, rtol=1e-5, atol=1e-6)

    hits = torch_compile_backend.fusion_hits()
    assert hits["qk_reduce_sm_scale"] >= 1


def test_tilelang_torch_compile_backend_reports_gemm_softmax_lowered() -> None:
    torch = _torch_or_skip()
    from torch import nn
    from torch.fx import symbolic_trace

    from cppmega_mlx.runtime import torch_compile_backend

    class TinyQKSoftmax(nn.Module):
        def forward(self, q, k):  # type: ignore[no-untyped-def]
            return torch.softmax(q @ k.transpose(-1, -2), dim=-1)

    report = torch_compile_backend.inspect_fx_graph(
        symbolic_trace(TinyQKSoftmax().eval())
    )

    assert report.lowering_status["gemm_softmax"] == "lowered_tilelang_emitter"
    assert report.pending_lowerings == ()
    assert report.lowered_patterns == ("gemm_softmax",)


def test_tilelang_torch_compile_backend_reports_qk_reduce_sm_scale_lowered() -> None:
    torch = _torch_or_skip()

    from cppmega_mlx.runtime import torch_compile_backend

    graph, _fn, _inputs = _export_qk_reduce_scale_graph(torch)
    report = torch_compile_backend.inspect_fx_graph(graph)

    assert report.lowering_status["qk_reduce_sm_scale"] == "lowered_tilelang_emitter"
    assert report.pending_lowerings == ()
    assert report.lowered_patterns == ("qk_reduce_sm_scale",)


def test_tilelang_torch_compile_backend_strict_mode_lowers_batched_qk_softmax() -> None:
    torch = _torch_or_skip()
    from torch import nn
    from torch.fx import symbolic_trace

    from cppmega_mlx.runtime import torch_compile_backend

    class TinyQKSoftmax(nn.Module):
        def forward(self, q, k):  # type: ignore[no-untyped-def]
            return torch.softmax(q @ k.transpose(-1, -2), dim=-1)

    graph = symbolic_trace(TinyQKSoftmax().eval())
    q = torch.randn(2, 4, 8, dtype=torch.float32)
    k = torch.randn(2, 4, 8, dtype=torch.float32)

    compiled = torch_compile_backend.compile_fx_graph(
        graph,
        (q, k),
        require_lowered_patterns=True,
    )

    torch.testing.assert_close(compiled(q, k), graph(q, k), rtol=1e-2, atol=1e-2)


def test_tilelang_strict_compile_uses_compiled_gemm_softmax_lowering() -> None:
    torch = _torch_or_skip()
    from torch.fx import symbolic_trace

    from cppmega_mlx.runtime import torch_compile_backend

    def qk_softmax(q, k):  # type: ignore[no-untyped-def]
        return torch.softmax(torch.matmul(q, k.transpose(-1, -2)), dim=-1)

    graph = symbolic_trace(qk_softmax)
    q = torch.randn(16, 8, dtype=torch.float32)
    k = torch.randn(16, 8, dtype=torch.float32)

    compiled = torch_compile_backend.compile_fx_graph(
        graph,
        (q, k),
        require_lowered_patterns=True,
    )

    before_count = torch_compile_backend.runtime_fallback_count(compiled)
    assert before_count == 0
    artifact = getattr(compiled, "_tilelang_artifact", None)
    assert artifact is not None
    assert "tilelang.compile ok" in str(getattr(artifact, "source", ""))
    assert "metal_bridge=mx.fast.metal_kernel" in str(getattr(artifact, "source", ""))
    torch.testing.assert_close(compiled(q, k), qk_softmax(q, k), rtol=1e-2, atol=1e-2)
    assert torch_compile_backend.runtime_fallback_count(compiled) == before_count


def test_tilelang_strict_compile_uses_compiled_qk_reduce_sm_scale_lowering() -> None:
    torch = _torch_or_skip()

    from cppmega_mlx.runtime import torch_compile_backend

    graph, qk_reduce_scale, inputs = _export_qk_reduce_scale_graph(torch)
    q, k, indices, sm_scale = inputs

    compiled = torch_compile_backend.compile_fx_graph(
        graph,
        inputs,
        require_lowered_patterns=True,
    )

    before_count = torch_compile_backend.runtime_fallback_count(compiled)
    assert before_count == 0
    artifact = getattr(compiled, "_tilelang_artifact", None)
    assert artifact is not None
    assert "tilelang.compile ok" in str(getattr(artifact, "source", ""))
    assert "metal_bridge=mx.fast.metal_kernel" in str(getattr(artifact, "source", ""))
    torch.testing.assert_close(
        compiled(q, k, indices, sm_scale),
        qk_reduce_scale(q, k, indices, sm_scale),
        rtol=1e-2,
        atol=1e-2,
    )
    assert torch_compile_backend.runtime_fallback_count(compiled) == before_count


def test_tilelang_strict_compile_preserves_autograd_for_gemm_softmax() -> None:
    torch = _torch_or_skip()
    from torch.fx import symbolic_trace

    from cppmega_mlx.runtime import torch_compile_backend

    def qk_softmax(q, k):  # type: ignore[no-untyped-def]
        return torch.softmax(torch.matmul(q, k.transpose(-1, -2)), dim=-1)

    graph = symbolic_trace(qk_softmax)
    q = torch.randn(16, 8, dtype=torch.float32, requires_grad=True)
    k = torch.randn(16, 8, dtype=torch.float32, requires_grad=True)
    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k.detach().clone().requires_grad_(True)

    compiled = torch_compile_backend.compile_fx_graph(
        graph,
        (q, k),
        require_lowered_patterns=True,
    )

    assert torch_compile_backend.is_aot_autograd_runner(compiled)
    assert not torch_compile_backend.is_autograd_preserving_fallback(compiled)
    y = compiled(q, k)
    y_ref = qk_softmax(q_ref, k_ref)
    torch.testing.assert_close(y, y_ref, rtol=1e-5, atol=1e-6)
    y.square().sum().backward()
    y_ref.square().sum().backward()
    torch.testing.assert_close(q.grad, q_ref.grad, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(k.grad, k_ref.grad, rtol=1e-5, atol=1e-6)


def test_tilelang_runtime_fallback_count_reads_compiled_artifact_launcher() -> None:
    from cppmega_mlx.runtime import torch_compile_backend

    class Launcher:
        _tilelang_runtime_fallback_count = 3

    class Artifact:
        launcher = Launcher()

    class Compiled:
        _tilelang_artifact = Artifact()

    assert torch_compile_backend.runtime_fallback_count(Compiled()) == 3


def test_tilelang_strict_torch_compile_backend_lowers_batched_qk_softmax() -> None:
    torch = _torch_or_skip()
    from torch import nn

    from cppmega_mlx.runtime import torch_compile_backend

    if not torch_compile_backend.torch_compile_backend_available():
        pytest.skip("torch._dynamo.register_backend unavailable")

    class TinyQKSoftmax(nn.Module):
        def forward(self, q, k):  # type: ignore[no-untyped-def]
            return torch.softmax(q @ k.transpose(-1, -2), dim=-1)

    backend_name = torch_compile_backend.register_tilelang_strict_backend()
    model = TinyQKSoftmax().eval()
    q = torch.randn(2, 4, 8, dtype=torch.float32)
    k = torch.randn(2, 4, 8, dtype=torch.float32)

    compiled = torch.compile(model, backend=backend_name, fullgraph=True)
    torch.testing.assert_close(compiled(q, k), model(q, k), rtol=1e-2, atol=1e-2)


def test_register_tilelang_backend_accepts_custom_strict_backend_name() -> None:
    torch = _torch_or_skip()
    from torch import nn

    from cppmega_mlx.runtime import torch_compile_backend

    if not torch_compile_backend.torch_compile_backend_available():
        pytest.skip("torch._dynamo.register_backend unavailable")

    backend_name = torch_compile_backend.register_tilelang_backend(
        name="tilelang_test_strict",
        require_lowered_patterns=True,
    )

    class TinyAdd(nn.Module):
        def forward(self, lhs, rhs):  # type: ignore[no-untyped-def]
            return lhs + rhs

    lhs = torch.randn(2, 4, dtype=torch.float32)
    rhs = torch.randn(2, 4, dtype=torch.float32)
    compiled = torch.compile(TinyAdd().eval(), backend=backend_name, fullgraph=True)

    torch.testing.assert_close(compiled(lhs, rhs), lhs + rhs)
