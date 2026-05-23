import json
import os
from tempfile import gettempdir
from cppmega_v4.runner import Pipeline, run_pipeline
from cppmega_v4.jsonrpc.schema import VerifyParams
from cppmega_v4.architectures import build_preset_specs

def main():
    with open("tests/fixtures/MATRIX.json", encoding="utf-8") as f:
        matrix = json.load(f)
    REAL_PARQUET = matrix["parquets"]["T2_gpt2_small__P1_minimal"]["path"]
    REAL_TOKENIZER = matrix["tokenizers"]["T2_gpt2_small"]["path"]
    SAVE = os.path.join(gettempdir(), "vbgui_h24_ckpt_long_repro.safetensors")
    
    if os.path.exists(SAVE):
        os.unlink(SAVE)
        
    specs = build_preset_specs("llama3_8b", hidden_size=16)
    graph = {
        "nodes": [
            {"id": s.get("name"), "kind": s["kind"], "params": s.get("params", {})}
            for s in specs
        ],
        "edges": [
            {"src": specs[i].get("name"), "dst": specs[i + 1].get("name")}
            for i in range(len(specs) - 1)
        ],
    }
    
    spec = VerifyParams.model_validate({
        "graph": graph,
        "dim_env": {"B": 1, "S": 64, "H": 16, "nh": 2, "nkv": 1,
                    "head_dim": 8, "num_experts": 4, "top_k": 2},
        "loss": {"kind": "cross_entropy", "head_outputs": [specs[-1].get("name")]},
        "optim": {
            "kind": "adamw",
            "mixed_precision": True,
            "groups": [
                {
                    "matcher": "all",
                    "lr": 0.0003,
                    "weight_decay": 0.01,
                    "betas": [0.9, 0.95],
                    "schedule": {
                        "kind": "linear_warmup",
                        "warmup_steps": 4,
                    },
                },
            ],
        },
        "sharding": {
            "topology": {"factory": "gb10_quarter", "kwargs": {}},
            "axis_assignments": [{"axis_name": "dp", "kind": "fsdp2", "degree": 8}],
            "compile_mode": "regional",
            "fp8_enabled": False,
        }
    })
    
    # 1. Run 10 steps to save checkpoint
    print("--- Running first pipeline (Train 10 steps and save) ---")
    pipeline_dict_1 = {
        "stages": ["parse", "verify_build_spec", "build_model", "train"],
        "stage_options": {
            "train": {
                "num_steps": 10,
                "checkpoint_save_path": SAVE,
                "parquet_path": REAL_PARQUET,
                "tokenizer_path": REAL_TOKENIZER,
            }
        }
    }
    r1 = run_pipeline(spec, Pipeline.from_dict(pipeline_dict_1))
    for stage in r1.stages:
        print(f"Stage {stage.name}: {stage.status}")
        if stage.status == "fail":
            print(f"Error: {stage.error}")
            
    # 2. Run 5 steps to resume from checkpoint
    print("\n--- Running second pipeline (Resume 5 steps from checkpoint) ---")
    pipeline_dict_2 = {
        "stages": ["parse", "verify_build_spec", "build_model", "train"],
        "stage_options": {
            "train": {
                "num_steps": 5,
                "checkpoint_load_path": SAVE,
                "parquet_path": REAL_PARQUET,
                "tokenizer_path": REAL_TOKENIZER,
            }
        }
    }
    r2 = run_pipeline(spec, Pipeline.from_dict(pipeline_dict_2))
    for stage in r2.stages:
        print(f"Stage {stage.name}: {stage.status}")
        if stage.status == "fail":
            print(f"Error: {stage.error}")

if __name__ == "__main__":
    main()
