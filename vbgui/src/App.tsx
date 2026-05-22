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
  INITIAL_SPEC, specReducer, type SpecState, type TopologyFactory,
} from "@/state/spec";
import { migrate } from "@/state/migrations";
import { useHistory } from "@/hooks/useHistory";
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
  // V7-H03: bounded undo/redo. Snapshot is the (nodes, edges, spec)
  // triple captured on every Verify cycle (cheap proxy for any
  // user-meaningful mutation that landed). Undo/redo restore the
  // snapshot triple in one shot.
  const history = useHistory<{ nodes: Node[]; edges: Edge[];
                                spec: SpecState }>(50);
  const lastHistoryKeyRef = useRef<string>("");
  // V7-H03: when undo/redo applies a snapshot back, the resulting
  // state-change effect would otherwise push that snapshot AGAIN
  // and clear the redo stack. This ref lets the apply-path skip
  // the next push.
  const suppressNextHistoryPushRef = useRef<boolean>(false);
  const [activeTab, setActiveTab] = useState<AppTab>("canvas");
  const [runReport, setRunReport] = useState<RunReport | null>(null);
  const [runError, setRunError] = useState<string | null>(null);
  const [trainInFlight, setTrainInFlight] = useState(false);
  // V7-I03: synchronous lock. React's setTrainInFlight schedules an
  // async commit, so two button clicks within the same microtask
  // both read trainInFlight=false and both call rpc.call. The ref
  // is mutated synchronously inside handleRunPipeline before any
  // await, closing the ~10ms window.
  const trainInFlightLockRef = useRef<boolean>(false);
  const [trainRunId, setTrainRunId] = useState<string | null>(null);
  // H04: most recent successfully-completed Train run_id, used as
  // continue_from_run_id when the warm-start checkbox is on. Cleared
  // when an error/cancel terminates the next run.
  const [lastTrainRunId, setLastTrainRunId] = useState<string | null>(null);
  const [selectedBrickId, setSelectedBrickId] = useState<string | null>(null);
  const [inferenceLog, setInferenceLog] = useState<
    { brick: string; param: string; value: unknown;
      source: "user" | "auto"; reason: string }[]
  >([]);
  // V4-1: data source for stage_train. When set, handleRunPipeline
  // forwards via stage_options.train.parquet_path; UI shows indicator.
  const [trainParquetPath, setTrainParquetPath] =
    useState<string | null>(null);
  const [trainParquetShards, setTrainParquetShards] =
    useState<string[]>([]);
  const [trainTokenizerPath, setTrainTokenizerPath] =
    useState<string | null>(null);
  const [availableSideChannels, setAvailableSideChannels] =
    useState<string[]>(["doc_ids", "token_ids"]);
  const [trainSideChannels, setTrainSideChannels] = useState<string[]>([]);

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
  const wireSpecRef = useRef({ nodes, edges, spec, availableSideChannels });
  useEffect(() => {
    wireSpecRef.current = { nodes, edges, spec, availableSideChannels };
  }, [nodes, edges, spec, availableSideChannels]);

  useEffect(() => {
    const available = new Set(availableSideChannels);
    setTrainSideChannels((prev) => prev.filter((name) => available.has(name)));
  }, [availableSideChannels]);

  const runVerify = useCallback(async () => {
    const snap = wireSpecRef.current;
    if (snap.nodes.length === 0) return;
    const params = buildVerifyParams(
      snap.nodes, snap.edges, snap.spec, snap.availableSideChannels,
    );
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
        inference_log?: { brick: string; param: string; value: unknown;
                          source: "user" | "auto"; reason: string }[];
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
      if (r.inference_log) {
        setInferenceLog(r.inference_log);
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
  // H07: include params hash so the verify debouncer re-fires when the
  // user mutates brick.data.params via DimensionsTab Apply or
  // BrickContextPanel. Without this, the inferred-dimensions feedback
  // loop never closed because the table reflected stale verify state.
  const nodesKey = nodes.map((n) => {
    const d = n.data as { kind?: string; params?: Record<string, unknown> };
    return `${n.id}:${d.kind ?? ""}:${JSON.stringify(d.params ?? {})}`;
  }).join("|");
  const edgesKey = edges.map((e) => `${e.source}>${e.target}`).join("|");
  const lossKey = `${spec.loss.kind}::${spec.loss.head_outputs.join(",")}`;
  const optimKey = `${spec.optim.kind}::${spec.optim.groups.length}`;
  const shardingKey =
    `${spec.sharding.topology}::${spec.sharding.compile_mode}` +
    `::${spec.sharding.axis_assignments.length}` +
    `::${spec.sharding.fp8_enabled ? 1 : 0}`;
  const rewriterKey = spec.rewriters.map((r) => r.name).join(",");
  const sideChannelKey = JSON.stringify(spec.side_channels);
  const availableSideChannelKey = availableSideChannels.join(",");
  useEffect(() => { scheduleVerify(); },
           [nodesKey, edgesKey, lossKey, optimKey, shardingKey,
            rewriterKey, sideChannelKey, availableSideChannelKey,
            scheduleVerify]);

  // V7-H03: snapshot the (nodes, edges, spec) triple every time a
  // structural key changes — that's the same set of user-meaningful
  // mutations the verify debouncer reacts to.
  useEffect(() => {
    const key = `${nodesKey}|${edgesKey}|${lossKey}|${optimKey}|`
      + `${shardingKey}|${rewriterKey}|${sideChannelKey}`;
    if (key === lastHistoryKeyRef.current) return;
    lastHistoryKeyRef.current = key;
    if (suppressNextHistoryPushRef.current) {
      suppressNextHistoryPushRef.current = false;
      return;
    }
    history.push({ nodes, edges, spec });
  }, [nodesKey, edgesKey, lossKey, optimKey, shardingKey,
      rewriterKey, sideChannelKey, history, nodes, edges, spec]);

  const handleUndo = useCallback(() => {
    const prev = history.undo();
    if (prev) {
      suppressNextHistoryPushRef.current = true;
      setNodes(prev.nodes);
      setEdges(prev.edges);
      dispatch({ type: "spec.replace", spec: prev.spec });
    }
  }, [history]);
  const handleRedo = useCallback(() => {
    const nxt = history.redo();
    if (nxt) {
      suppressNextHistoryPushRef.current = true;
      setNodes(nxt.nodes);
      setEdges(nxt.edges);
      dispatch({ type: "spec.replace", spec: nxt.spec });
    }
  }, [history]);

  // V7-H03: Cmd/Ctrl+Z = undo, Shift+Cmd/Ctrl+Z = redo.
  useEffect(() => {
    function onKey(ev: KeyboardEvent) {
      const meta = ev.metaKey || ev.ctrlKey;
      if (!meta || ev.key.toLowerCase() !== "z") return;
      // Don't intercept inside text inputs.
      const tag = (ev.target as HTMLElement | null)?.tagName ?? "";
      if (tag === "INPUT" || tag === "TEXTAREA") return;
      ev.preventDefault();
      if (ev.shiftKey) handleRedo();
      else handleUndo();
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [handleUndo, handleRedo]);

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
                     estimated_per_rank_bytes: number; reason: string;
                     sharding?: { axis_assignments: {
                       axis_name: string; kind: string; degree: number;
                     }[] } }[];
      }>("suggest_sharding", {
        graph: nodesToGraph(snap.nodes, snap.edges),
        dim_env: MINI_DIM_ENV,
        loss: { kind: snap.spec.loss.kind,
                head_outputs: snap.spec.loss.head_outputs },
        optim: { kind: snap.spec.optim.kind,
                 gradient_clip_norm: snap.spec.optim.grad_clip_norm,
                 groups: snap.spec.optim.groups.map((g) => ({
                   matcher: g.matcher, lr: g.lr,
                   weight_decay: g.weight_decay,
                   betas: g.betas,
                   ns_steps: g.ns_steps,
                   schedule: g.schedule,
                 })) },
        topology: { factory: snap.spec.sharding.topology, kwargs: {} },
      });
      setProposals(r.proposals.map((p) => ({
        strategy_name: p.strategy_name, fits: p.fits,
        estimated_per_rank_bytes: p.estimated_per_rank_bytes,
        reason: p.reason,
        axis_assignments: p.sharding?.axis_assignments,
      })));
    } catch { /* keep prior proposals on failure */ }
  }, [rpc]);

  useEffect(() => {
    void requestSuggestSharding();
  }, [requestSuggestSharding, spec.sharding.topology, nodes.length]);

  const handleRunPipeline = useCallback(async (
    mode: RunMode,
    opts?: { num_steps?: number; warm_start?: boolean;
      checkpoint_save_path?: string;
      checkpoint_load_path?: string;
      inference_probe_text?: string;
      master_dtype?: "fp32" | "bf16" | "fp16" | "auto";
    },
  ) => {
    // V7-I03: synchronous lock check at the very top — before any
    // await, before the canvas-empty guard. Two rapid clicks both
    // pass through React's stale-prop trainInFlight=false but only
    // the first acquires the ref.
    if (mode === "train") {
      if (trainInFlightLockRef.current) return;
      trainInFlightLockRef.current = true;
    }
    const snap = wireSpecRef.current;
    if (snap.nodes.length === 0) {
      if (mode === "train") trainInFlightLockRef.current = false;
      setRunError("canvas is empty — drop bricks or pick a preset first");
      setRunReport(null);
      return;
    }
    setRunError(null);
    setRunReport(null);
    const stages = mode === "smoke" ? SMOKE_STAGES
                 : mode === "full"  ? FULL_STAGES : TRAIN_STAGES;
    // V3-6: TopBar exposes train_num_steps; thread it via stage_options.
    // V4-1: forward parquet_path + tokenizer_path picked in Data/Tokenizer tabs.
    const stage_options: Record<string, Record<string, unknown>> = {};
    let activeTrainRunId: string | null = null;
    if (mode === "train") {
      const trainOpts: Record<string, unknown> = {};
      activeTrainRunId = makeTrainRunId();
      trainOpts.run_id = activeTrainRunId;
      trainOpts.abort_token = activeTrainRunId;
      if (typeof opts?.num_steps === "number") {
        trainOpts.num_steps = opts.num_steps;
      }
      // H04: warm-start uses lastTrainRunId as continue_from_run_id so
      // the backend G10 LRU cache restores opt.state from prior run.
      if (opts?.warm_start && lastTrainRunId) {
        trainOpts.continue_from_run_id = lastTrainRunId;
      }
      // H05: forward checkpoint save/load paths to stage_train G12.
      if (opts?.checkpoint_save_path) {
        trainOpts.checkpoint_save_path = opts.checkpoint_save_path;
      }
      if (opts?.checkpoint_load_path) {
        trainOpts.checkpoint_load_path = opts.checkpoint_load_path;
      }
      // H08: forward inference probe text (G20). Backend pairs it with
      // tokenizer_path (also forwarded above) to encode real tokens for
      // the pre-vs-post forward divergence probe.
      if (opts?.inference_probe_text) {
        trainOpts.inference_probe_text = opts.inference_probe_text;
      }
      // H23: master_dtype override (fp32/bf16/fp16). "auto" defers to
      // spec.optim.mixed_precision (H02) — don't override the wire.
      if (opts?.master_dtype && opts.master_dtype !== "auto") {
        trainOpts.master_dtype = opts.master_dtype;
      }
      // H20: when the spec carries sharding axes, derive fake_ranks
      // from the product of their degrees so a Train run simulates a
      // mean-reduced multi-rank backward (extras.fake_ranks +
      // gradient_reduce_ms light up).
      const axes = snap.spec.sharding?.axis_assignments ?? [];
      if (axes.length > 0) {
        const totalDegree = axes.reduce(
          (acc, a) => acc * Math.max(1, (a as { degree?: number }).degree
            ?? 1), 1);
        if (totalDegree > 1) trainOpts.fake_ranks = totalDegree;
      }
      if (trainParquetPath) trainOpts.parquet_path = trainParquetPath;
      if (trainParquetShards.length > 1) {
        trainOpts.parquet_shards = trainParquetShards;
      }
      if (trainTokenizerPath) trainOpts.tokenizer_path = trainTokenizerPath;
      // Forward SideChannelsTab train selection as synthetic int lists for the
      // stage_train G17 math-effect smoke path.
      if (trainSideChannels.length > 0) {
        const sc: Record<string, number[]> = {};
        for (const name of trainSideChannels) {
          sc[name] = [0, 1, 2, 3, 4, 5, 6, 7];  // synthetic 8-token sample
        }
        trainOpts.side_channels = sc;
      }
      if (Object.keys(trainOpts).length > 0) stage_options.train = trainOpts;
      setTrainRunId(activeTrainRunId);
      setTrainInFlight(true);
    }
    try {
      const r = await rpc.call<RunReport>("pipeline.run", {
        spec: buildVerifyParams(
          snap.nodes, snap.edges, snap.spec, snap.availableSideChannels,
        ),
        pipeline: { stages, stage_options },
      });
      setRunReport(r);
      // H04: remember run_id of a successful Train so a follow-up
      // warm-start run can reference it.
      if (mode === "train" && activeTrainRunId) {
        const trainStage = r.stages?.find((s) => s.name === "train");
        if (trainStage?.status === "ok") {
          setLastTrainRunId(activeTrainRunId);
          // H11: surface the Metal peak alongside the verify estimate.
          const peak = (trainStage as unknown as
            { memory_peak_bytes?: number }).memory_peak_bytes;
          if (typeof peak === "number" && peak > 0) {
            dispatch({ type: "memory.actual_set",
                       actual_peak_bytes: peak });
          }
        }
      }
    } catch (e) {
      setRunError(String(e));
    } finally {
      if (mode === "train") {
        setTrainInFlight(false);
        setTrainRunId(null);
        trainInFlightLockRef.current = false;
      }
    }
  }, [rpc, trainParquetPath, trainSideChannels, trainTokenizerPath,
      lastTrainRunId]);

  const handleCancelTrain = useCallback(async () => {
    const runId = trainRunId;
    if (!runId) return;
    try {
      await rpc.call("pipeline.abort", { run_id: runId });
    } catch (e) {
      setRunError(String(e));
    }
  }, [rpc, trainRunId]);

  const handleShardingAccept = useCallback((idx: number) => {
    const chosen = proposals[idx];
    if (!chosen) return;
    // H01: actually mutate spec.sharding.axis_assignments with the
    // proposal's axes. Previous version only re-verified the OLD spec,
    // so accepting a proposal had zero observable effect downstream.
    if (chosen.axis_assignments && chosen.axis_assignments.length > 0) {
      dispatch({ type: "sharding.set",
                 sharding: { ...spec.sharding,
                             axis_assignments: chosen.axis_assignments } });
    }
    void scheduleVerify();
    setRunReport(null);
    setRunError(
      `sharding proposal "${chosen.strategy_name}" applied — re-verifying`);
    setTimeout(() => setRunError(null), 2000);
  }, [proposals, scheduleVerify, spec.sharding]);

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
          trainInFlight={trainInFlight}
          trainRunId={trainRunId}
          onCancelTrain={handleCancelTrain}
          onUndo={handleUndo}
          onRedo={handleRedo}
          canUndo={history.canUndo}
          canRedo={history.canRedo}
          onMixedPrecisionChange={(enabled) => dispatch({ type: "optim.set",
            optim: { ...spec.optim, mixed_precision: enabled } })}
          onFp8EnabledChange={(enabled) => dispatch({ type: "sharding.set",
            sharding: { ...spec.sharding, fp8_enabled: enabled } })}
          trainParquetPath={trainParquetPath}
          trainTokenizerPath={trainTokenizerPath}
          onSaveSpec={() => {
            // G11: serialise full SpecState + canvas (nodes/edges) to JSON
            const blob = new Blob([JSON.stringify({
              projectName, spec, nodes, edges,
            }, null, 2)], { type: "application/json" });
            const url = URL.createObjectURL(blob);
            const a = document.createElement("a");
            a.href = url;
            a.download = `${projectName}.spec.json`;
            a.click();
            URL.revokeObjectURL(url);
          }}
          onLoadSpec={(file: File) => {
            const reader = new FileReader();
            reader.onload = () => {
              try {
                const raw = JSON.parse(String(reader.result));
                // V7-H04: migrate-on-load so older saved bundles still
                // hydrate cleanly. Future-version specs throw a
                // FutureSchemaError that bubbles into the error modal.
                const obj = migrate(raw);
                if (obj.projectName) setProjectName(String(obj.projectName));
                if (obj.spec) dispatch({ type: "spec.replace",
                                          spec: obj.spec as never });
                if (obj.nodes) setNodes(obj.nodes as never[]);
                if (obj.edges) setEdges(obj.edges as never[]);
              } catch (e) {
                setRunError(`Load failed: ${String(e)}`);
              }
            };
            reader.readAsText(file);
          }}
          trainDisabled={
            (() => {
              // V3-8/V3-9: gate Train on gotcha severity. The verify RPC
              // returns gotchas with severity "error" for both validator
              // failures (verify=error, V3-9) and check_gotchas critical
              // findings (V3-8). Surfacing them via the disabled-reason
              // attribute lets Playwright assert on the exact gating cause.
              const errs = spec.gotchas.filter(g => g.severity === "error");
              if (errs.length === 0) return null;
              return { reason: `${errs.length} critical issue${
                errs.length > 1 ? "s" : ""}: ${errs[0].message}` };
            })()
          }
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
                sideChannels={spec.side_channels}
                availableSideChannels={availableSideChannels}
                selectedTrainSideChannels={trainSideChannels}
                sharding={spec.sharding}
                gotchas={spec.gotchas}
                proposals={proposals}
                rpc={rpc}
                graphNodes={nodes}
                graphEdges={edges}
                inferenceLog={inferenceLog}
                onHighlightBrick={setSelectedBrickId}
                onLossApply={(l) => dispatch({ type: "loss.set", loss: l })}
                onOptimApply={(o) => dispatch({ type: "optim.set", optim: o })}
                onRewriterAdd={(r) =>
                  dispatch({ type: "rewriters.add", rewriter: r })}
                onRewriterRemove={(i) =>
                  dispatch({ type: "rewriters.remove", index: i })}
                onRewriterReorder={(f, t) =>
                  dispatch({ type: "rewriters.reorder", from: f, to: t })}
                onRewriterApply={() => void scheduleVerify()}
                onSideChannelsApply={(s) =>
                  dispatch({ type: "side_channels.set", side_channels: s })}
                onTrainSideChannelsChange={setTrainSideChannels}
                onShardingChange={(s) =>
                  dispatch({ type: "sharding.set", sharding: s })}
                onShardingAccept={handleShardingAccept}
                onGotchaAutoFix={(id) => {
                  // V7-H01: dispatch recovery for known auto-fixable
                  // gotcha ids. Unknown ids no-op gracefully.
                  if (id === "fsdp2_whole_compile"
                      || id === "megatron_tp_whole_compile") {
                    dispatch({ type: "sharding.set",
                      sharding: { ...spec.sharding,
                                  compile_mode: "regional" } });
                  } else if (id === "bad_dtype_combo") {
                    dispatch({ type: "optim.set",
                      optim: { ...spec.optim,
                                mixed_precision: false } });
                    dispatch({ type: "sharding.set",
                      sharding: { ...spec.sharding,
                                  fp8_enabled: false } });
                  } else if (id === "unknown_brick") {
                    setNodes((prev) => prev.filter((n) => {
                      const k = (n.data as { kind?: string })?.kind;
                      return typeof k === "string" && k.length > 0;
                    }));
                  } else if (id === "missing_edge") {
                    setEdges((prev) => {
                      const ids = nodes.map((n) => n.id);
                      const next: Edge[] = [];
                      for (let i = 0; i < ids.length - 1; i++) {
                        const eid = `${ids[i]}->${ids[i + 1]}`;
                        if (!prev.some((e) => e.id === eid)) {
                          next.push({ id: eid,
                            source: ids[i], target: ids[i + 1],
                            data: { severity: "info" } });
                        }
                      }
                      return [...prev, ...next];
                    });
                  }
                  void scheduleVerify();
                }}
                onDimensionsApply={(entry) => {
                  // H07: write the auto-suggested value into the matched
                  // brick's params. Re-verify will then re-render the
                  // entry as source="user" (or drop it entirely if the
                  // inference no longer fires), closing the feedback loop.
                  setNodes((prev) => prev.map((n) => {
                    if (n.id !== entry.brick) return n;
                    const d = (n.data ?? {}) as {
                      kind?: string; params?: Record<string, unknown>;
                    };
                    return { ...n, data: {
                      ...d,
                      params: { ...(d.params ?? {}),
                                [entry.param]: entry.value },
                    } as never };
                  }));
                }}
              />
            </>
          )}
          {activeTab === "tokenizer" && (
            <TokenizerPlayground rpc={rpc}
              onUseForTrain={(t) => setTrainTokenizerPath(t)}
              trainTokenizerPath={trainTokenizerPath} />
          )}
          {activeTab === "data" && (
            <DataInspector rpc={rpc}
              onUseForTrain={(p, t, shards) => {
                setTrainParquetPath(p);
                setTrainParquetShards(shards ?? []);
                if (t !== null) setTrainTokenizerPath(t);
              }}
              onAvailableChannelsChange={setAvailableSideChannels}
              trainParquetPath={trainParquetPath} />
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

function makeTrainRunId(): string {
  if (typeof crypto !== "undefined" && "randomUUID" in crypto) {
    return `train-${crypto.randomUUID()}`;
  }
  return `train-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function buildVerifyParams(
  nodes: Node[],
  edges: Edge[],
  spec: SpecState,
  availableSideChannels: string[] = ["doc_ids", "token_ids"],
) {
  return {
    graph: nodesToGraph(nodes, edges),
    dim_env: MINI_DIM_ENV,
    loss: { kind: spec.loss.kind, head_outputs: spec.loss.head_outputs,
            params: spec.loss.params },
    optim: { kind: spec.optim.kind,
             gradient_clip_norm: spec.optim.grad_clip_norm,
             mixed_precision: spec.optim.mixed_precision,
             groups: spec.optim.groups.map((g) => ({
               matcher: g.matcher, lr: g.lr,
               weight_decay: g.weight_decay, betas: g.betas,
               // V3-5: thread schedule + ns_steps through to backend.
               // The wire payload dropped these previously, making the
               // OptimTab ScheduleEditor and Muon ns_steps decorative —
               // backend always saw constant lr and default ns_steps=5.
               ns_steps: g.ns_steps,
               schedule: g.schedule,
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
    data_materialization: spec.data_materialization,
    available_side_channels: availableSideChannels,
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
