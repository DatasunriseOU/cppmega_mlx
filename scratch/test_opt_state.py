import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_map

model = nn.Sequential(nn.Linear(2, 2))
opt = optim.AdamW(learning_rate=1e-3)
params = model.parameters()
opt.init(params)
print("Initial state keys:", list(opt.state.keys()))
print("Initial step:", opt.state.get("step"))

# Run one step
grads = tree_map(lambda x: mx.zeros_like(x), params)
opt.update(model, grads)
print("After step 1, state keys:", list(opt.state.keys()))
print("After step 1, step:", opt.state.get("step"))

# Try setting state
saved_state = opt.state
opt.state = saved_state
print("After restoring, state keys:", list(opt.state.keys()))
