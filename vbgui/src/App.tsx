import { useCallback, useEffect, useReducer, useRef, useState } from "react";
import {
  ReactFlowProvider, type Edge, type Node,
} from "@xyflow/react";

import { FlowCanvas } from "@/components/FlowCanvas";
import { Palette } from "@/components/Palette";
import { Sidebar } from "@/components/Sidebar";
import { TopBar, type RunMode } from "@/components/TopBar";
import { BottomStrip } from "@/components/BottomStrip";
import { AppTabs, type AppTab } from "@/components/AppTabs";
import { RunResultModal, type RunReport } from "@/components/RunResultModal";
import { TokenizerPlayground } from "@/components/TokenizerPlayground";
import { DataInspector } from "@/components/DataInspector";
import { BrickContextPanel } from "@/components/BrickContextPanel";

import { useRpc } from "@/hooks/useRpc";
import { useVerifyAfter } from "@/hooks/useVerifyAfter";
import { usePresets } from "@/hooks/usePresets";

import {
  INITIAL_SPEC, specReducer, type TopologyFactory,
} from "@/state/spec";
import type { ShardingProposalView } from "@/components/sidebar/ShardingTab";

// PRESETS list is now fetched dynamically from the backend via
// architectures.list_presets — see usePresets() hook below. A fallback
// list lives in the hook for offline / first-paint cases.

const TOPOLOGIES: readonly TopologyFactory[] = [
  "h100_8x", "h200_8x", "a100_8x", "b100_8x",
  "gb10_quarter", "tpu_v6e_8", "tpu_v5p_4", "m3_ultra_solo",
];

// Mini-spec used for preset expansion in the GUI. Matches E2EMatrix.md §3.1.
const MINI_HIDDEN = 128;
const MINI_DEPTH = 2;
const MINI_DIM_ENV = {
  B: 1, S: 64, H: MINI_HIDDEN,
  nh: 2, nkv: 1, head_dim: 64,
  num_experts: 4, top_k: 2,
};

const SMOKE_STAGES = [
  "parse", "verify_build_spec", "apply_rewrites", "resolve_shapes",
  "estimate_memory", "check_gotchas", "build_model", "dry_forward",
];
const FULL_STAGES = [
  ...SMOKE_STAGES, "input_parity_check", "loss_smoke", "optimizer_smoke",
];
const TRAIN_STAGES = [...FULL_STAGES, "train"];

interface BrickSpec {
  kind: string;
  name?: string;
  params?: Record<string, unknown>;
}

function presetSpecsToNodes(specs: BrickSpec[]): { nodes: Node[]; edges: Edge[] } {
  const nodes: Node[] = [];
  const edges: Edge[] = [];
  let x = 60, y = 60;
  let lastName: string | null = null;
  for (const s of specs) {
    const name = s.name ?? `${s.kind}_${nodes.length}`;
    nodes.push({
      id: name,
      type: "brick",
      position: { x, y },
      data: { kind: s.kind, params: s.params ?? {} } as never,
    });
    if (lastName) {
      edges.push({
        id: `${lastName}->${name}`,
        source: lastName,
        target: name,
        data: { severity: "info" },
      });
    }
    lastName = name;
    x += 220;
    if (x > 980) { x = 60; y += 140; }
  }
  return { nodes, edges };
}

export function App(): JSX.Element {
  const [nodes, setNodes] = useState<Node[]>([]);
  const [edges, setEdges] = useState<Edge[]>([]);
  const [projectName, setProjectName] = useState("untitled");
  const [proposals, setProposals] = useState<ShardingProposalView[]>([]);
  const [spec, dispatch] = useReducer(specReducer, INITIAL_SPEC);
  const [activeTab, setActiveTab] = useState<AppTab>("canvas");
  const [runReport, setRunReport] = useState<RunReport | null>(null);
  const [runError, setRunError] = useState<string | null>(null);
  const [selectedBrickId, setSelectedBrickId] = useState<string | null>(null);

  const rpc = useRpc({
    baseUrl: (import.meta.env.VITE_BACKEND_URL as string | undefined)
              ?? "http://127.0.0.1:8765",
    enableWs: true,
    onBackendStatus: (s) => dispatch({ type: "backend.status", status: s }),
  });

  // Live preset list (62 entries from backend; falls back to bundled
  // snapshot when RPC is offline). E7-8: replaces the hardcoded 57-entry
  // list that was missing 5 architectures.
  const PRESETS = usePresets(rpc);

  // Keep one stable spec snapshot for the verify debouncer to read.
  const wireSpecRef = useRef({ nodes, edges, spec });
  useEffect(() => {
    wireSpecRef.current = { nodes, edges, spec };
  }, [nodes, edges, spec]);

  const runVerify = useCallback(async () => {
    const snap = wireSpecRef.current;
    if (snap.nodes.length === 0) return;
    const params = buildVerifyParams(snap.nodes, snap.edges, snap.spec);
    try {
      const r = await rpc.call<{
        memory_distributed?: { worst_rank?: { total_bytes?: number };
                               fits_on_topology?: boolean };
        memory_per_brick?: Record<string, { params_bytes: number;
                                            activations_bytes: number;
                                            kv_cache_bytes: number }>;
        gotchas?: { id: string; severity: "info" | "warning" | "error";
                    message: string; reference?: string }[];
        elapsed_ms: number;
        resolved?: { edges?: { src: string; dst: string;
                               matched: boolean;
                               severity: "info" | "warning" | "error" }[] };
      }>("verify", params);

      // Aggregate per-brick to a worst-rank-bytes proxy when no sharding.
      const total = sumPerBrick(r.memory_per_brick);
      const worst = r.memory_distributed?.worst_rank?.total_bytes ?? total;
      dispatch({ type: "memory.set", worst_rank_bytes: worst });
      dispatch({ type: "verify.complete",
                 elapsed_ms: r.elapsed_ms,
                 brick_count: snap.nodes.length });
      if (r.gotchas) {
        dispatch({ type: "gotchas.set", gotchas: r.gotchas });
      }
      if (r.resolved?.edges) {
        setEdges((prev) => recolorEdges(prev, r.resolved!.edges!));
      }
    } catch {
      // Backend down or invalid spec; leave state and let user retry.
    }
  }, [rpc]);

  const { schedule: scheduleVerify } = useVerifyAfter(
    wireSpecRef, runVerify, { debounceMs: 200 },
  );

  // Trigger verify only on inputs that the user controls. The verify
  // response writes back into spec AND edge severity — depending on
  // either of those in the effect would loop. Use structural keys.
  const nodesKey = nodes.map((n) => `${n.id}:${(n.data as { kind?: string })
                                                   ?.kind ?? ""}`).join("|");
  const edgesKey = edges.map((e) => `${e.source}>${e.target}`).join("|");
  const lossKey = `${spec.loss.kind}::${spec.loss.head_outputs.join(",")}`;
  const optimKey = `${spec.optim.kind}::${spec.optim.groups.length}`;
  const shardingKey =
    `${spec.sharding.topology}::${spec.sharding.compile_mode}` +
    `::${spec.sharding.axis_assignments.length}` +
    `::${spec.sharding.fp8_enabled ? 1 : 0}`;
  const rewriterKey = spec.rewriters.map((r) => r.name).join(",");
  useEffect(() => { scheduleVerify(); },
           [nodesKey, edgesKey, lossKey, optimKey, shardingKey,
            rewriterKey, scheduleVerify]);

  // ----- Handlers ----------------------------------------------------------

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

  const handlePresetDrop = useCallback(async (name: string) => {
    try {
      // num_layers omitted → one repeat unit. Each preset's unit is the
      // smallest topologically complete subgraph (some are 6+ bricks long
      // for sliding/global patterns), so asking for num_layers=MINI_DEPTH
      // would truncate to zero for those. Stick with the canonical unit.
      const r = await rpc.call<{ specs: BrickSpec[]; preset_name: string }>(
        "build_preset_specs",
        { preset_name: name, hidden_size: MINI_HIDDEN },
      );
      const { nodes: ns, edges: es } = presetSpecsToNodes(r.specs);
      setNodes(ns);
      setEdges(es);
      // Rebind loss.head_outputs to the last brick so verify_build_spec
      // accepts the freshly-loaded preset (which doesn't define a node
      // literally named "logits"). User can change this later via the
      // Loss tab.
      if (ns.length > 0) {
        dispatch({ type: "loss.set", loss: {
          ...spec.loss,
          head_outputs: [ns[ns.length - 1].id],
        }});
      }
    } catch (e) {
      setRunError(String(e));
    }
  }, [rpc, spec.loss]);

  const requestSuggestSharding = useCallback(async () => {
    const snap = wireSpecRef.current;
    if (snap.nodes.length === 0) return;
    try {
      const r = await rpc.call<{
        proposals: { strategy_name: string; fits: boolean;
                     estimated_per_rank_bytes: number; reason: string }[];
      }>("suggest_sharding", {
        graph: nodesToGraph(snap.nodes, snap.edges),
        dim_env: MINI_DIM_ENV,
        loss: { kind: snap.spec.loss.kind,
                head_outputs: snap.spec.loss.head_outputs },
        optim: { kind: snap.spec.optim.kind,
                 groups: snap.spec.optim.groups.map((g) => ({
                   matcher: g.matcher, lr: g.lr,
                   weight_decay: g.weight_decay,
                   betas: g.betas,
                 })) },
        topology: { factory: snap.spec.sharding.topology, kwargs: {} },
      });
      setProposals(r.proposals.map((p) => ({
        strategy_name: p.strategy_name, fits: p.fits,
        estimated_per_rank_bytes: p.estimated_per_rank_bytes,
        reason: p.reason,
      })));
    } catch { /* keep prior proposals on failure */ }
  }, [rpc]);

  useEffect(() => {
    void requestSuggestSharding();
  }, [requestSuggestSharding, spec.sharding.topology, nodes.length]);

  const handleRunPipeline = useCallback(async (mode: RunMode) => {
    const snap = wireSpecRef.current;
    if (snap.nodes.length === 0) {
      setRunError("canvas is empty — drop bricks or pick a preset first");
      setRunReport(null);
      return;
    }
    setRunError(null);
    setRunReport(null);
    const stages = mode === "smoke" ? SMOKE_STAGES
                 : mode === "full"  ? FULL_STAGES : TRAIN_STAGES;
    try {
      const r = await rpc.call<RunReport>("pipeline.run", {
        spec: buildVerifyParams(snap.nodes, snap.edges, snap.spec),
        pipeline: { stages, stage_options: {} },
      });
      setRunReport(r);
    } catch (e) {
      setRunError(String(e));
    }
  }, [rpc]);

  const handleShardingAccept = useCallback((idx: number) => {
    const chosen = proposals[idx];
    if (!chosen) return;
    // The proposal carries strategy + reason; backend already knows the
    // axis-assignments. We re-run verify so the new memory bar reflects.
    void scheduleVerify();
    setRunReport(null);
    setRunError(`sharding proposal "${chosen.strategy_name}" applied — re-verifying`);
    setTimeout(() => setRunError(null), 2000);
  }, [proposals, scheduleVerify]);

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
        <AppTabs active={activeTab} onChange={setActiveTab} />
        <div style={{ flex: 1, display: "flex", minHeight: 0 }}>
          {activeTab === "canvas" && (
            <>
              <Palette />
              <FlowCanvas
                nodes={nodes} edges={edges}
                onConnect={handleConnect}
                onDropBrick={handleDropBrick}
                onNodeClick={setSelectedBrickId}
              />
              {selectedBrickId && (() => {
                const selected = nodes.find((n) => n.id === selectedBrickId);
                if (!selected) return null;
                const data = selected.data as {
                  kind?: string; params?: Record<string, unknown>;
                };
                return (
                  <BrickContextPanel
                    rpc={rpc}
                    brickId={selectedBrickId}
                    brickKind={data.kind ?? "mlp"}
                    params={data.params ?? {}}
                    onApply={(newParams) => {
                      setNodes((prev) => prev.map((n) =>
                        n.id === selectedBrickId
                          ? { ...n, data: { ...(n.data as object),
                                            params: newParams } as never }
                          : n));
                    }}
                    onClose={() => setSelectedBrickId(null)}
                  />
                );
              })()}
              <Sidebar
                loss={spec.loss}
                optim={spec.optim}
                rewriters={spec.rewriters}
                sharding={spec.sharding}
                gotchas={spec.gotchas}
                proposals={proposals}
                rpc={rpc}
                graphNodes={nodes}
                graphEdges={edges}
                onLossApply={(l) => dispatch({ type: "loss.set", loss: l })}
                onOptimApply={(o) => dispatch({ type: "optim.set", optim: o })}
                onRewriterAdd={(r) =>
                  dispatch({ type: "rewriters.add", rewriter: r })}
                onRewriterRemove={(i) =>
                  dispatch({ type: "rewriters.remove", index: i })}
                onRewriterReorder={(f, t) =>
                  dispatch({ type: "rewriters.reorder", from: f, to: t })}
                onShardingChange={(s) =>
                  dispatch({ type: "sharding.set", sharding: s })}
                onShardingAccept={handleShardingAccept}
              />
            </>
          )}
          {activeTab === "tokenizer" && (
            <TokenizerPlayground rpc={rpc} />
          )}
          {activeTab === "data" && (
            <DataInspector rpc={rpc} />
          )}
        </div>
        <BottomStrip state={spec} fusedRegionCount={0} />
        <RunResultModal report={runReport} error={runError}
                        onClose={() => { setRunReport(null);
                                         setRunError(null); }} />
      </div>
    </ReactFlowProvider>
  );
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function nodesToGraph(nodes: Node[], edges: Edge[]) {
  return {
    nodes: nodes.map((n) => {
      const data = n.data as { kind?: string; params?: Record<string, unknown> };
      return { id: n.id, kind: data.kind ?? "mlp", params: data.params ?? {} };
    }),
    edges: edges.map((e) => ({ src: e.source, dst: e.target })),
  };
}

function buildVerifyParams(nodes: Node[], edges: Edge[],
                           spec: ReturnType<typeof specReducer>) {
  return {
    graph: nodesToGraph(nodes, edges),
    dim_env: MINI_DIM_ENV,
    loss: { kind: spec.loss.kind, head_outputs: spec.loss.head_outputs,
            params: spec.loss.params },
    optim: { kind: spec.optim.kind,
             groups: spec.optim.groups.map((g) => ({
               matcher: g.matcher, lr: g.lr,
               weight_decay: g.weight_decay, betas: g.betas,
             })) },
    rewriters: spec.rewriters.map((r) => ({ name: r.name, params: r.params })),
    sharding: {
      topology: { factory: spec.sharding.topology, kwargs: {} },
      axis_assignments: spec.sharding.axis_assignments,
      compile_mode: spec.sharding.compile_mode,
      fp8_enabled: spec.sharding.fp8_enabled,
    },
    training: true,
    side_channels: spec.side_channels,
    available_side_channels: ["doc_ids", "token_ids"],
  };
}

function sumPerBrick(per?: Record<string, { params_bytes: number;
                                            activations_bytes: number;
                                            kv_cache_bytes: number }>): number {
  if (!per) return 0;
  return Object.values(per).reduce(
    (acc, r) => acc + r.params_bytes + r.activations_bytes + r.kv_cache_bytes,
    0,
  );
}

function recolorEdges(prev: Edge[],
                      resolved: { src: string; dst: string;
                                  severity: "info" | "warning" | "error";
                                  matched: boolean }[]): Edge[] {
  const map = new Map<string, { severity: "info" | "warning" | "error" }>();
  for (const e of resolved) map.set(`${e.src}->${e.dst}`, { severity: e.severity });
  return prev.map((e) => {
    const m = map.get(`${e.source}->${e.target}`);
    if (!m) return e;
    return { ...e, data: { ...(e.data ?? {}), severity: m.severity } };
  });
}
