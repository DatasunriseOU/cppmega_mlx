from cppmega_v4.architectures.presets import build_preset_specs
from cppmega_v4.jsonrpc.schema import VerifyParams
from cppmega_v4.runner import Pipeline, run_pipeline
import traceback

presets = ["qwen3_next", "kimi_linear", "deepseek_v3", "mistral4", "gemma4"]

for preset in presets:
    print(f"\n=================== {preset} ===================")
    try:
        # Load preset specs using H=64 (approx. 1/32 scale)
        specs = build_preset_specs(preset, hidden_size=64)
        
        # Build graph dict
        nodes = []
        # Prepend input embedder
        nodes.append({"id": "input_embedder", "kind": "embedding_table", "params": {"vocab_size": 65536}})
        for i, s in enumerate(specs):
            nodes.append({"id": s["name"], "kind": s["kind"], "params": s.get("params") or {}})
        # Append output deembedder
        nodes.append({"id": "output_deembedder", "kind": "embedding_table", "params": {"vocab_size": 65536}})
        
        edges = []
        edges.append({"src": "input_embedder", "dst": specs[0]["name"]})
        for i in range(len(specs) - 1):
            edges.append({"src": specs[i]["name"], "dst": specs[i+1]["name"]})
        edges.append({"src": specs[-1]["name"], "dst": "output_deembedder"})
        
        dim_env = {
            "B": 1, "S": 64, "H": 64,
            "nh": 2, "nkv": 1, "head_dim": 32,
            "num_experts": 4, "top_k": 2,
        }
        
        spec_dict = {
            "graph": {
                "nodes": nodes,
                "edges": edges,
            },
            "dim_env": dim_env,
            "loss": {
                "kind": "cross_entropy",
                "head_outputs": ["output_deembedder"],
            },
            "optim": {
                "kind": "adamw",
                "groups": [{"matcher": "all", "lr": 1e-3, "weight_decay": 0.01, "betas": [0.9, 0.95]}],
            },
        }
        
        spec = VerifyParams.model_validate(spec_dict)
        
        pipeline = Pipeline.from_dict({
            "stages": [
                "parse", "verify_build_spec", "apply_rewrites", "resolve_shapes",
                "estimate_memory", "check_gotchas", "build_model", "dry_forward",
            ],
            "stage_options": {},
        })
        
        report = run_pipeline(spec, pipeline)
        print(f"Overall status: {report.overall_status}")
        for stage in report.stages:
            print(f"Stage {stage.name}: {stage.status} ({stage.elapsed_ms:.1f}ms)")
            if stage.status == "fail":
                print("ERROR DETAIL:", stage.error)
    except Exception as e:
        print(f"Exception for {preset}: {e}")
        traceback.print_exc()
