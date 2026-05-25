import sys
import traceback
import mlx.core as mx
from mlx.utils import tree_flatten
from cppmega_v4.jsonrpc.schema import VerifyParams
from cppmega_v4.runner import Pipeline, run_pipeline
from cppmega_v4.architectures.presets import build_preset_specs

PRESETS_TO_TEST = [
    "xlstm_7b",
    "qwen3_next",
    "kimi_linear",
    "kimi_k2",
    "deepseek_v3",
    "deepseek_v4_flash",
    "gemma4",
    "mistral4",
    "ling26",
    "longcat",
    "nemotron3",
    "zaya1",
    "arcee_trinity",
    "gpt2_xl",
    "gemma_4_e2b",
]

def run_test_for_preset(preset_name: str):
    print(f"\n==========================================")
    print(f"Testing preset: {preset_name}")
    print(f"==========================================")
    try:
        specs = build_preset_specs(preset_name, hidden_size=64)
        print(f"Built {len(specs)} block specs: {[s['kind'] for s in specs]}")
        
        nodes = []
        for s in specs:
            nodes.append({
                "id": s["name"],
                "kind": s["kind"],
                "params": s.get("params") or {},
            })
            
        d = {
            "graph": {
                "nodes": nodes,
                "edges": [{"src": nodes[i]["id"], "dst": nodes[i+1]["id"]} for i in range(len(nodes)-1)],
            },
            "dim_env": {"B": 1, "S": 8, "H": 128, "nh": 2, "nkv": 1, "head_dim": 64},
            "loss": {"kind": "cross_entropy", "head_outputs": [nodes[-1]["id"]]},
            "optim": {"kind": "adamw",
                      "groups": [{"matcher": "all", "lr": 1e-3,
                                  "weight_decay": 0.01,
                                  "betas": [0.9, 0.95]}]},
            "sharding": {
                "topology": {"factory": "h100_8x", "kwargs": {}},
                "axis_assignments": [
                    {"axis_name": "dp", "kind": "fsdp2", "degree": 8},
                ],
                "compile_mode": "regional",
                "fp8_enabled": False,
                "comm_backend": "ring",
            }
        }
        
        spec = VerifyParams.model_validate(d)
        
        pipeline = Pipeline.from_dict({
            "stages": ["parse", "verify_build_spec", "build_model", "train"],
            "stage_options": {"train": {"num_steps": 4, "master_dtype": "fp16", "loss_scaler_init": 1.0}},
        })
        
        rep = run_pipeline(spec, pipeline)
        for s in rep.stages:
            print(f"  Stage {s.name}: {s.status}")
            if getattr(s, "error", None) is not None:
                print(f"    Error: {s.error}")
            if s.name == "train" and getattr(s, "extras", None) is not None:
                print(f"    Train Extras: {sorted(s.extras.keys())}")
                if "weight_delta_norm" in s.extras:
                    print(f"    Weight Delta Norm: {s.extras['weight_delta_norm']}")
    except Exception as e:
        print(f"Failed to process preset {preset_name}:")
        traceback.print_exc()

def main():
    for preset in PRESETS_TO_TEST:
        run_test_for_preset(preset)

if __name__ == "__main__":
    main()
