from cppmega_v4.architectures.presets import build_preset_specs
import json

for layers in [0, 4, 6, 7, 8]:
    try:
        specs = build_preset_specs("gemma3_27b", hidden_size=128, num_layers=layers)
        print(f"=== gemma3_27b (layers={layers}, specs count: {len(specs)}) ===")
        for s in specs:
            print("  ", s.get("kind"), s.get("name"))
    except Exception as e:
        print(f"Error for gemma3_27b with layers={layers}: {e}")

