import sys
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
from cppmega_v4.architectures.presets import build_preset_specs
from cppmega_v4.fusion.brick_graph import from_block_specs
from cppmega_v4.buildspec.api import BuiltSequentialModel

def debug_preset(preset_name: str):
    print(f"\n==========================================")
    print(f"Debugging Gradients for: {preset_name}")
    print(f"==========================================")
    specs = build_preset_specs(preset_name, hidden_size=64)
    print("Specs kinds:", [s["kind"] for s in specs])
    
    # Instantiate the unified superblock graph
    graph = from_block_specs(specs, hidden_size=64, instantiate=False)
    model = BuiltSequentialModel(graph, hidden_size=64)
    
    print("Model submodules:")
    for attr, val in model.__dict__.items():
        if attr.startswith("brick_"):
            print(f"  {attr}: {val.__class__.__name__}")
            
    # Generate random input tokens/hidden states
    mx.random.seed(42)
    x = mx.random.uniform(-0.1, 0.1, (1, 8, 64))
    targets = mx.random.randint(0, 256, (8,))
    
    def loss_fn(model, x, targets):
        outputs = model(x)
        # Get output of the last node
        last_node_name = specs[-1]["name"]
        features = outputs[last_node_name]
        print(f"    Last node output shape: {features.shape}")
        
        # Project to vocabulary size 256
        head = mx.random.uniform(-0.1, 0.1, (64, 256))
        logits = features[0] @ head # shape (8, 256)
        
        # Cross entropy loss
        loss = mx.mean(nn.losses.cross_entropy(logits, targets))
        return loss
        
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
    loss, grads = loss_and_grad_fn(model, x, targets)
    mx.eval(loss, grads)
    
    print("    Loss:", loss.item())
    
    # Sum of gradients for each parameter
    grad_sums = {}
    for name, grad in tree_flatten(grads):
        gsum = mx.sum(mx.abs(grad)).item()
        grad_sums[name] = gsum
    
    print("    Gradient sums per parameter:")
    for k, v in sorted(grad_sums.items()):
        if v > 0:
            print(f"      {k}: {v:.6f}")
        else:
            print(f"      {k}: 0.000000 (ZERO!)")

def main():
    debug_preset("llama3_8b")
    debug_preset("nemotron3")
    debug_preset("xlstm_7b")

if __name__ == "__main__":
    main()
