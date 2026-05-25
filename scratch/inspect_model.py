# scratch/inspect_model.py
import sys
import os

sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/.."))

from cppmega_v4.architectures.presets import PRESETS, build_preset_specs
from cppmega_v4.fusion.brick_graph import from_block_specs

def main():
    # Scale down llama3_8b with H=128 and 2 layers
    specs = build_preset_specs("llama3_8b", hidden_size=128, num_layers=2)
    
    print("\nSpecs nodes:")
    for idx, s in enumerate(specs):
        print(f"  [{idx}] Name: {s.get('name')}, Kind: {s.get('kind')}, Params: {s.get('params')}")
        
    # Instantiate modules
    graph = from_block_specs(specs, hidden_size=128, instantiate=True)
    modules = [n.module for n in graph.nodes]
    
    print("\nInstantiated module class names:")
    for idx, mod in enumerate(modules):
        if mod is None:
            print(f"  [{idx}] None")
        else:
            print(f"  [{idx}] {mod.__class__.__name__} (type: {type(mod)})")

if __name__ == "__main__":
    main()
