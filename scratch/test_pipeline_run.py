import sys
import traceback
from cppmega_v4.jsonrpc.schema import VerifyParams
from cppmega_v4.runner import Pipeline, run_pipeline

# Emulate frontend buildVerifyParams for llama3_8b
hidden = 128
dim_env = {
    "B": 1, "S": 64, "H": hidden,
    "nh": 2, "nkv": 1, "head_dim": 64,
    "num_experts": 4, "top_k": 2,
}

nodes = [
    {"id": "input_embedder", "kind": "embedding_table", "params": {"vocab_size": 65536}},
    {"id": "llama3_8b_attn", "kind": "attention", "params": {"num_heads": 4, "head_dim": 64}},
    {"id": "llama3_8b_mlp", "kind": "mlp", "params": {}},
    {"id": "output_deembedder", "kind": "embedding_table", "params": {"vocab_size": 65536}},
]

edges = [
    {"src": "input_embedder", "dst": "llama3_8b_attn"},
    {"src": "llama3_8b_attn", "dst": "llama3_8b_mlp"},
    {"src": "llama3_8b_mlp", "dst": "output_deembedder"},
]

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

try:
    spec = VerifyParams.model_validate(spec_dict)
    print("VerifyParams model validation succeeded!")
    
    pipeline = Pipeline.from_dict({
        "stages": [
            "parse", "verify_build_spec", "apply_rewrites", "resolve_shapes",
            "estimate_memory", "check_gotchas", "build_model", "dry_forward",
        ],
        "stage_options": {},
    })
    
    report = run_pipeline(spec, pipeline)
    print("Pipeline Report Status:", report.overall_status)
    for stage in report.stages:
        print(f"Stage {stage.name}: {stage.status} ({stage.elapsed_ms:.1f}ms)")
        if stage.status == "fail":
            print("ERROR DETAILS:", stage.error)
except Exception as e:
    print("An exception occurred during verification:")
    traceback.print_exc()
