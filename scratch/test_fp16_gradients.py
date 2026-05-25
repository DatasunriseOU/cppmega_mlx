import sys
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
from cppmega_v4.architectures.presets import build_preset_specs
from cppmega_v4.fusion.brick_graph import from_block_specs
from cppmega_v4.buildspec.api import BuiltSequentialModel

def test_fp16():
    specs = build_preset_specs("xlstm_7b", hidden_size=64)
    graph = from_block_specs(specs, hidden_size=64, instantiate=False)
    
    # 1. FP32 Test
    model_fp32 = BuiltSequentialModel(graph, hidden_size=64)
    x_fp32 = mx.random.uniform(-0.1, 0.1, (1, 8, 64))
    targets = mx.random.randint(0, 256, (8,))
    
    def loss_fn(model, x, targets):
        outputs = model(x)
        features = outputs[specs[-1]["name"]]
        head = mx.random.uniform(-0.1, 0.1, (64, 256), dtype=x.dtype)
        logits = features[0] @ head
        return mx.mean(nn.losses.cross_entropy(logits, targets))
        
    loss_and_grad_fp32 = nn.value_and_grad(model_fp32, loss_fn)
    loss_fp32, grads_fp32 = loss_and_grad_fp32(model_fp32, x_fp32, targets)
    mx.eval(loss_fp32, grads_fp32)
    
    print("FP32 Gradients sum:", sum(mx.sum(mx.abs(g)).item() for _, g in tree_flatten(grads_fp32) if hasattr(g, "shape")))
    
    # 2. FP16 Test
    model_fp16 = BuiltSequentialModel(graph, hidden_size=64)
    model_fp16.set_dtype(mx.float16)
    x_fp16 = x_fp32.astype(mx.float16)
    
    loss_and_grad_fp16 = nn.value_and_grad(model_fp16, loss_fn)
    loss_fp16, grads_fp16 = loss_and_grad_fp16(model_fp16, x_fp16, targets)
    mx.eval(loss_fp16, grads_fp16)
    
    print("FP16 Gradients sum:", sum(mx.sum(mx.abs(g)).item() for _, g in tree_flatten(grads_fp16) if hasattr(g, "shape")))

if __name__ == "__main__":
    test_fp16()
