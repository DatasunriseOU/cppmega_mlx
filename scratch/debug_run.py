import sys
import traceback
from cppmega_v4.parallelism.sharding_spec import CommBackend, ShardingSpec, AxisAssignment, ParallelismKind
from cppmega_v4.parallelism.topology import DeviceTopology, DeviceSpec, DeviceKind
from cppmega_v4.parallelism.gotcha_checker import check_gotchas
from cppmega_v4.buildspec.model_build_spec import ModelBuildSpec
from cppmega_v4.buildspec.loss_spec import LossSpec, LossKind
from cppmega_v4.buildspec.optim_spec import OptimSpec
from cppmega_v4.fusion.brick_graph import BrickGraph
from cppmega_v4.jsonrpc.schema import VerifyParams
from cppmega_v4.runner import Pipeline, run_pipeline
from cppmega_v4.architectures.presets import build_preset_specs as _build_preset_specs

def main():
    print("Loading llama3_8b preset...")
    specs = _build_preset_specs("llama3_8b", hidden_size=64)
    nodes = []
    for s in specs:
        nodes.append({
            "id": s["name"],
            "kind": s["kind"],
            "params": s.get("params") or {},
        })
    # Convert to schema format
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
        "stage_options": {"train": {"num_steps": 3, "master_dtype": "fp16"}},
    })
    
    print("Running pipeline...")
    try:
        rep = run_pipeline(spec, pipeline)
        for s in rep.stages:
            print(f"Stage {s.name}: {s.status}")
            if s.name == "train" and s.status == "ok":
                print("Train Extras Keys:", sorted(s.extras.keys()))
                print("Train Extras values of loss_scaler:", s.extras.get("loss_scaler"))
            if s.status == "error":
                print(f"Error details: {s.error}")
    except Exception as e:
        print("EXCEPTION RAISED:")
        traceback.print_exc()

if __name__ == "__main__":
    main()
