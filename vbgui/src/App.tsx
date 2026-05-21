import { useCallback, useReducer, useState } from "react";
import { ReactFlowProvider, type Edge, type Node } from "@xyflow/react";

import { FlowCanvas } from "@/components/FlowCanvas";
import { Palette } from "@/components/Palette";
import { Sidebar } from "@/components/Sidebar";
import { TopBar, type RunMode } from "@/components/TopBar";
import { BottomStrip } from "@/components/BottomStrip";

import {
  INITIAL_SPEC, specReducer, type TopologyFactory,
} from "@/state/spec";
import type { ShardingProposalView } from "@/components/sidebar/ShardingTab";

const PRESETS: readonly string[] = [
  "qwen3_next", "kimi_linear", "kimi_k2", "deepseek_v3",
  "deepseek_v4_flash", "gemma4", "mistral4", "ling26",
  "longcat", "nemotron3", "zaya1", "arcee_trinity",
];

const TOPOLOGIES: readonly TopologyFactory[] = [
  "h100_8x", "h200_8x", "a100_8x", "b100_8x",
  "gb10_quarter", "tpu_v6e_8", "tpu_v5p_4", "m3_ultra_solo",
];

export function App(): JSX.Element {
  const [nodes, setNodes] = useState<Node[]>([]);
  const [edges, setEdges] = useState<Edge[]>([]);
  const [projectName, setProjectName] = useState("untitled");
  const [proposals] = useState<ShardingProposalView[]>([]);
  const [spec, dispatch] = useReducer(specReducer, INITIAL_SPEC);

  const handleDropBrick = useCallback(
    (kind: string, position: { x: number; y: number }) => {
      setNodes((prev) => [
        ...prev,
        { id: `${kind}_${prev.length + 1}`,
          type: "brick", position, data: { kind } },
      ]);
    }, []);

  const handleConnect = useCallback(
    (p: { source: string; target: string }) => {
      setEdges((prev) => [
        ...prev,
        { id: `${p.source}->${p.target}`, source: p.source, target: p.target,
          data: { severity: "info" } },
      ]);
    }, []);

  const handlePresetDrop = useCallback((_name: string) => {
    // F-A.2/F-D will resolve preset → specs via build_preset_specs RPC and
    // fan out into Node[] additions. Stub for now.
  }, []);

  const handleRunPipeline = useCallback((_mode: RunMode) => {
    // F-D wires this to JSON-RPC pipeline.run; for now we only flag intent.
  }, []);

  return (
    <ReactFlowProvider>
      <div style={{ display: "flex", flexDirection: "column",
                    height: "100vh", margin: 0 }}>
        <TopBar
          state={spec}
          projectName={projectName}
          presets={PRESETS}
          topologies={TOPOLOGIES}
          onProjectNameChange={setProjectName}
          onPresetDrop={handlePresetDrop}
          onTopologyChange={(t) => dispatch({ type: "sharding.set",
            sharding: { ...spec.sharding, topology: t } })}
          onCompileModeChange={(m) => dispatch({ type: "sharding.set",
            sharding: { ...spec.sharding, compile_mode: m } })}
          onRunPipeline={handleRunPipeline}
        />
        <div style={{ flex: 1, display: "flex", minHeight: 0 }}>
          <Palette />
          <FlowCanvas
            nodes={nodes} edges={edges}
            onConnect={handleConnect}
            onDropBrick={handleDropBrick}
          />
          <Sidebar
            loss={spec.loss}
            optim={spec.optim}
            rewriters={spec.rewriters}
            sharding={spec.sharding}
            gotchas={spec.gotchas}
            proposals={proposals}
            onLossApply={(l) => dispatch({ type: "loss.set", loss: l })}
            onOptimApply={(o) => dispatch({ type: "optim.set", optim: o })}
            onRewriterAdd={(r) => dispatch({ type: "rewriters.add", rewriter: r })}
            onRewriterRemove={(i) => dispatch({ type: "rewriters.remove", index: i })}
            onRewriterReorder={(f, t) =>
              dispatch({ type: "rewriters.reorder", from: f, to: t })}
            onShardingChange={(s) =>
              dispatch({ type: "sharding.set", sharding: s })}
            onShardingAccept={(_idx) => { /* F-D wires accept → topology update */ }}
          />
        </div>
        <BottomStrip state={spec} fusedRegionCount={0} />
      </div>
    </ReactFlowProvider>
  );
}
