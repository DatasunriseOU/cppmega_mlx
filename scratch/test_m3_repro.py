import sys
from cppmega_v4.runner import Pipeline, run_pipeline
from cppmega_v4.jsonrpc.schema import VerifyParams
from cppmega_v4.architectures import build_preset_specs

def main():
    specs = build_preset_specs("llama3_8b", hidden_size=128)
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
        "dim_env": {"B": 1, "S": 64, "H": 128, "nh": 2, "nkv": 1,
                    "head_dim": 64, "num_experts": 4, "top_k": 2},
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
            "topology": {"factory": "m3_ultra_solo", "kwargs": {}},
            "axis_assignments": [{"axis_name": "dp", "kind": "fsdp2", "degree": 1}],
            "compile_mode": "regional",
            "fp8_enabled": False,
        }
    })
    
    pipeline_dict = {
        "stages": [
            "parse",
            "verify_build_spec",
            "apply_rewrites",
            "resolve_shapes",
            "estimate_memory",
            "check_gotchas",
            "build_model",
            "dry_forward",
            "input_parity_check",
            "loss_smoke",
            "optimizer_smoke",
            "train"
        ],
        "stage_options": {
            "train": {
                "num_steps": 2,
            }
        }
    }
    
    r = run_pipeline(spec, Pipeline.from_dict(pipeline_dict))
    for stage in r.stages:
        print(f"Stage {stage.name}: {stage.status}")
        if stage.name in ("estimate_memory", "train"):
            print(f"Extras for {stage.name}: {stage.extras}")

if __name__ == "__main__":
    main()
