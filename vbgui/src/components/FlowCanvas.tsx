import { useCallback, useMemo, useState, useEffect } from "react";
import {
  ReactFlow,
  Background,
  Controls,
  BaseEdge,
  EdgeLabelRenderer,
  getBezierPath,
  Handle,
  Position,
  type Edge,
  type Node,
  type NodeTypes,
  type EdgeTypes,
  type EdgeProps,
  type IsValidConnection,
  type OnNodesChange,
  type OnEdgesChange,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";

import { BrickNode } from "./BrickNode";
import { AdapterNode } from "./AdapterNode";
import { LossGhostNode } from "./LossGhostNode";
import { TokenizerVirtualNode, DetokenizerVirtualNode } from "./VirtualNodes";

export interface FlowCanvasProps {
  nodes: Node[];
  edges: Edge[];
  onConnect?: (params: { source: string; target: string }) => void;
  isValidConnection?: IsValidConnection;
  onDropBrick?: (
    kind: string,
    position: { x: number; y: number },
    params?: Record<string, unknown>,
  ) => void;
  /** Fires when the user clicks a brick node — opens BrickContextPanel
   *  (E7-5/E7-6). */
  onNodeClick?: (nodeId: string) => void;
  /** Callback to insert an adapter into the middle of an edge. */
  onInsertAdapter?: (kind: string, edge: { source: string; target: string }) => void;
  /** Callback to trigger ELK.js auto-layout graph alignment. */
  onAutoAlign?: () => void;
  /** React Flow node change callback to enable drag-and-drop movement. */
  onNodesChange?: OnNodesChange;
  /** React Flow edge change callback. */
  onEdgesChange?: OnEdgesChange;
}

// Beautiful custom glowing residual addition (+) node component
export function ResidualAddNode({ id }: { id: string }): JSX.Element {
  return (
    <div
      role="region"
      aria-label="residual addition convergence node"
      data-testid={`brick-node-${id}`}
      style={{
        width: 38,
        height: 38,
        borderRadius: "50%",
        background: "rgba(16, 185, 129, 0.12)", // emerald tint
        backdropFilter: "blur(8px)",
        border: "2px solid rgba(16, 185, 129, 0.6)", // emerald border
        boxShadow: "0 0 10px rgba(16, 185, 129, 0.4)", // emerald glow
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        color: "#10b981", // emerald text
        fontSize: 20,
        fontWeight: 800,
        fontFamily: "var(--vb-font-mono, monospace)",
        position: "relative",
      }}
    >
      <Handle
        type="target"
        position={Position.Left}
        style={{ background: "#10b981", width: 6, height: 6, border: "none" }}
      />
      <span>+</span>
      <Handle
        type="source"
        position={Position.Right}
        style={{ background: "#10b981", width: 6, height: 6, border: "none" }}
      />
    </div>
  );
}

const NODE_TYPES: NodeTypes = {
  brick: BrickNode as unknown as NodeTypes[string],
  adapter: AdapterNode as unknown as NodeTypes[string],
  loss_ghost: LossGhostNode as unknown as NodeTypes[string],
  tokenizer_virtual: TokenizerVirtualNode as unknown as NodeTypes[string],
  detokenizer_virtual: DetokenizerVirtualNode as unknown as NodeTypes[string],
  residual_add: ResidualAddNode as unknown as NodeTypes[string],
};

// 1. High-fidelity custom MidpointEdge component matching the visual builder mockup.
export function MidpointEdge({
  id,
  sourceX,
  sourceY,
  targetX,
  targetY,
  sourcePosition,
  targetPosition,
  style = {},
  markerEnd,
  data,
}: EdgeProps): JSX.Element {
  const [edgePath, labelX, labelY] = getBezierPath({
    sourceX,
    sourceY,
    sourcePosition,
    targetX,
    targetY,
    targetPosition,
  });

  const sev = (data as { severity?: string } | undefined)?.severity;
  const adapter = (data as { adapter?: boolean } | undefined)?.adapter;
  
  const debuggerMode = (data as { debuggerMode?: boolean } | undefined)?.debuggerMode;
  const direction = (data as { direction?: "forward" | "backward" } | undefined)?.direction;
  const isActiveFlow = (data as { isActiveFlow?: boolean } | undefined)?.isActiveFlow;

  let stroke = "#10b981"; // default emerald green
  let strokeDasharray: string | undefined = undefined;
  let animation: string | undefined = undefined;
  let opacity: number | undefined = undefined;

  if (debuggerMode) {
    if (isActiveFlow) {
      stroke = direction === "forward" ? "var(--vb-accent)" : "var(--vb-cat-moe)";
      strokeDasharray = "4 4";
      animation = `${direction === "forward" ? "vbFlowFwd" : "vbFlowBwd"} 0.8s linear infinite`;
      opacity = 1.0;
    } else {
      stroke = "var(--vb-border-strong)";
      opacity = 0.25;
    }
  } else {
    if (adapter) {
      stroke = "#9ca3af";
      strokeDasharray = "4 2";
    } else if (sev === "error") {
      stroke = "#dc2626";
    } else if (sev === "warning") {
      stroke = "#d97706";
    }
  }

  const finalStyle = {
    ...style,
    stroke,
    strokeWidth: debuggerMode && isActiveFlow ? 3.5 : 2.5,
    strokeDasharray,
    animation,
    opacity,
  };

  return (
    <>
      <style>{`
        @keyframes vbFlowFwd {
          from { stroke-dashoffset: 16; }
          to { stroke-dashoffset: 0; }
        }
        @keyframes vbFlowBwd {
          from { stroke-dashoffset: 0; }
          to { stroke-dashoffset: 16; }
        }
      `}</style>
      <BaseEdge path={edgePath} markerEnd={markerEnd} style={finalStyle} />
      <EdgeLabelRenderer>
        <div
          style={{
            position: "absolute",
            transform: `translate(-50%, -50%) translate(${labelX}px,${labelY}px)`,
            pointerEvents: "all",
            zIndex: 10,
          }}
        >
          <button
            className="nodrag nopan"
            data-testid={`edge-plus-${id}`}
            onClick={(e) => {
              e.stopPropagation();
              const onClickMidpoint = (data as { onClickMidpoint?: (e: React.MouseEvent) => void })?.onClickMidpoint;
              onClickMidpoint?.(e);
            }}
            style={{
              width: 26,
              height: 26,
              borderRadius: "50%",
              border: "1px solid rgba(255, 255, 255, 0.25)",
              background: "rgba(15, 23, 42, 0.9)",
              backdropFilter: "blur(4px)",
              color: "white",
              cursor: "pointer",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              fontSize: 16,
              fontWeight: "bold",
              boxShadow: "0 4px 10px rgba(0, 0, 0, 0.4)",
              transition: "all 0.15s cubic-bezier(0.4, 0, 0.2, 1)",
            }}
            onMouseOver={(ev) => {
              ev.currentTarget.style.transform = "scale(1.25)";
              ev.currentTarget.style.background = "#0891b2"; // cyan hover color
              ev.currentTarget.style.borderColor = "rgba(255, 255, 255, 0.4)";
            }}
            onMouseOut={(ev) => {
              ev.currentTarget.style.transform = "scale(1)";
              ev.currentTarget.style.background = "rgba(15, 23, 42, 0.9)";
              ev.currentTarget.style.borderColor = "rgba(255, 255, 255, 0.25)";
            }}
          >
            +
          </button>
        </div>
      </EdgeLabelRenderer>
    </>
  );
}

const EDGE_TYPES: EdgeTypes = {
  midpoint: MidpointEdge as unknown as EdgeTypes[string],
};

export function FlowCanvas({
  nodes, edges, onConnect, isValidConnection, onDropBrick, onNodeClick, onInsertAdapter, onAutoAlign, onNodesChange, onEdgesChange,
}: FlowCanvasProps): JSX.Element {
  const [edgeMenu, setEdgeMenu] = useState<{ edge: Edge; x: number; y: number } | null>(null);

  // Auto-close on escape key or clicking outside
  useEffect(() => {
    if (!edgeMenu) return;
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") setEdgeMenu(null);
    };
    const handleClickOutside = (e: MouseEvent) => {
      const target = e.target as HTMLElement;
      if (!target.closest('[data-testid="edge-radial-menu"]')) {
        setEdgeMenu(null);
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    const timer = setTimeout(() => {
      window.addEventListener("click", handleClickOutside);
    }, 50);

    return () => {
      window.removeEventListener("keydown", handleKeyDown);
      window.removeEventListener("click", handleClickOutside);
      clearTimeout(timer);
    };
  }, [edgeMenu]);

  const handleConnect = useCallback(
    (p: { source: string; target: string }) => onConnect?.(p),
    [onConnect],
  );

  const handleMidpointClick = useCallback(
    (event: React.MouseEvent, edge: Edge) => {
      const canvasEl = document.querySelector('[data-testid="flow-canvas"]');
      if (canvasEl) {
        const r = canvasEl.getBoundingClientRect();
        setEdgeMenu({
          edge,
          x: event.clientX - r.left,
          y: event.clientY - r.top,
        });
      } else {
        const r = event.currentTarget.getBoundingClientRect();
        setEdgeMenu({
          edge,
          x: event.clientX - r.left,
          y: event.clientY - r.top,
        });
      }
    },
    [],
  );

  const handleDrop = useCallback(
    (e: React.DragEvent<HTMLDivElement>) => {
      e.preventDefault();
      const brick = e.dataTransfer.getData("application/x-cppmega-brick");
      const adapter = e.dataTransfer.getData("application/x-cppmega-adapter");
      const transplantKind = e.dataTransfer.getData("application/x-cppmega-transplant-kind");
      const transplantParamsRaw = e.dataTransfer.getData("application/x-cppmega-transplant-params");

      const rect = e.currentTarget.getBoundingClientRect();
      const position = { x: e.clientX - rect.left, y: e.clientY - rect.top };

      if (transplantKind && onDropBrick) {
        let params: Record<string, unknown> = {};
        try {
          if (transplantParamsRaw) {
            params = JSON.parse(transplantParamsRaw);
          }
        } catch { /* ignore */ }
        onDropBrick(transplantKind, position, params);
      } else if ((brick || adapter) && onDropBrick) {
        onDropBrick(brick || adapter, position);
      }
    },
    [onDropBrick],
  );

  const handleDragOver = useCallback((e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.dataTransfer.dropEffect = "copy";
  }, []);

  // Map every edge to use the custom 'midpoint' type with a click callback registered in data
  const styledEdges = useMemo(
    () => edges.map((e) => ({
      ...e,
      type: "midpoint",
      data: {
        ...e.data,
        onClickMidpoint: (event: React.MouseEvent) => {
          handleMidpointClick(event, e);
        }
      }
    })),
    [edges, handleMidpointClick],
  );

  return (
    <div
      data-testid="flow-canvas"
      style={{ flex: 1, height: "100%", position: "relative" }}
      onDrop={handleDrop}
      onDragOver={handleDragOver}
    >
      <ReactFlow
        nodes={nodes}
        edges={styledEdges}
        nodeTypes={NODE_TYPES}
        edgeTypes={EDGE_TYPES}
        onConnect={handleConnect as never}
        onNodeClick={(_e, node) => onNodeClick?.(node.id)}
        onEdgeClick={handleMidpointClick}
        isValidConnection={isValidConnection}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        fitView
        proOptions={{ hideAttribution: true }}
      >
        <Background />
        <Controls />
      </ReactFlow>

      {onAutoAlign && (
        <button
          onClick={() => { onAutoAlign(); }}
          data-testid="auto-align-button"
          style={{
            position: "absolute",
            top: 15,
            right: 15,
            zIndex: 100,
            padding: "8px 16px",
            background: "rgba(15, 23, 42, 0.75)",
            backdropFilter: "blur(12px)",
            border: "1px solid rgba(255, 255, 255, 0.15)",
            borderRadius: "8px",
            color: "#22d3ee", // Premium cyan color
            fontSize: "12px",
            fontWeight: "bold",
            cursor: "pointer",
            display: "flex",
            alignItems: "center",
            gap: "6px",
            boxShadow: "0 4px 15px rgba(0, 0, 0, 0.25)",
            transition: "all 0.2s cubic-bezier(0.4, 0, 0.2, 1)",
          }}
          onMouseOver={(e) => {
            e.currentTarget.style.transform = "scale(1.05)";
            e.currentTarget.style.background = "rgba(15, 23, 42, 0.9)";
            e.currentTarget.style.borderColor = "rgba(34, 211, 238, 0.4)";
            e.currentTarget.style.boxShadow = "0 6px 20px rgba(34, 211, 238, 0.25)";
          }}
          onMouseOut={(e) => {
            e.currentTarget.style.transform = "scale(1)";
            e.currentTarget.style.background = "rgba(15, 23, 42, 0.75)";
            e.currentTarget.style.borderColor = "rgba(255, 255, 255, 0.15)";
            e.currentTarget.style.boxShadow = "0 4px 15px rgba(0, 0, 0, 0.25)";
          }}
        >
          🪄 Auto Align Graph
        </button>
      )}

      {edgeMenu && (
        <div
          data-testid="edge-radial-menu"
          style={{
            position: "absolute",
            left: edgeMenu.x,
            top: edgeMenu.y,
            transform: "translate(-50%, -50%)",
            zIndex: 1000,
            background: "rgba(15, 23, 42, 0.85)",
            backdropFilter: "blur(12px)",
            border: "1px solid rgba(255, 255, 255, 0.15)",
            borderRadius: "50%",
            width: 140,
            height: 140,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            boxShadow: "0 10px 25px -5px rgba(0, 0, 0, 0.6), 0 8px 10px -6px rgba(0, 0, 0, 0.6)",
            transition: "all 0.2s cubic-bezier(0.4, 0, 0.2, 1)",
          }}
        >
          <button
            onClick={() => setEdgeMenu(null)}
            style={{
              position: "absolute",
              width: 32,
              height: 32,
              borderRadius: "50%",
              border: "none",
              background: "rgba(255, 255, 255, 0.1)",
              color: "#94a3b8",
              cursor: "pointer",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              fontSize: 14,
              fontWeight: "bold",
              zIndex: 10,
              transition: "all 0.15s ease",
            }}
            onMouseOver={(e) => {
              e.currentTarget.style.background = "rgba(239, 68, 68, 0.2)";
              e.currentTarget.style.color = "#f87171";
            }}
            onMouseOut={(e) => {
              e.currentTarget.style.background = "rgba(255, 255, 255, 0.1)";
              e.currentTarget.style.color = "#94a3b8";
            }}
          >
            ✕
          </button>
          
          <button
            data-testid="radial-insert-linear_bridge"
            onClick={() => {
              onInsertAdapter?.("linear_bridge", edgeMenu.edge);
              setEdgeMenu(null);
            }}
            title="Insert linear_bridge"
            style={{
              position: "absolute",
              top: 10,
              left: "50%",
              transform: "translateX(-50%)",
              width: 36,
              height: 36,
              borderRadius: "50%",
              border: "none",
              background: "#0891b2",
              color: "white",
              cursor: "pointer",
              fontSize: 10,
              fontWeight: "bold",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              transition: "all 0.15s ease",
            }}
            onMouseOver={(e) => {
              e.currentTarget.style.transform = "translateX(-50%) scale(1.15)";
              e.currentTarget.style.background = "#06b6d4";
            }}
            onMouseOut={(e) => {
              e.currentTarget.style.transform = "translateX(-50%) scale(1)";
              e.currentTarget.style.background = "#0891b2";
            }}
          >
            LB
          </button>

          <button
            data-testid="radial-insert-transpose_bnsd"
            onClick={() => {
              onInsertAdapter?.("transpose_bnsd", edgeMenu.edge);
              setEdgeMenu(null);
            }}
            title="Insert transpose_bnsd"
            style={{
              position: "absolute",
              bottom: 12,
              left: 12,
              width: 36,
              height: 36,
              borderRadius: "50%",
              border: "none",
              background: "#4f46e5",
              color: "white",
              cursor: "pointer",
              fontSize: 10,
              fontWeight: "bold",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              transition: "all 0.15s ease",
            }}
            onMouseOver={(e) => {
              e.currentTarget.style.transform = "scale(1.15)";
              e.currentTarget.style.background = "#6366f1";
            }}
            onMouseOut={(e) => {
              e.currentTarget.style.transform = "scale(1)";
              e.currentTarget.style.background = "#4f46e5";
            }}
          >
            TR
          </button>

          <button
            data-testid="radial-insert-rmsnorm"
            onClick={() => {
              onInsertAdapter?.("rmsnorm", edgeMenu.edge);
              setEdgeMenu(null);
            }}
            title="Insert rmsnorm"
            style={{
              position: "absolute",
              bottom: 12,
              right: 12,
              width: 36,
              height: 36,
              borderRadius: "50%",
              border: "none",
              background: "#059669",
              color: "white",
              cursor: "pointer",
              fontSize: 10,
              fontWeight: "bold",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              transition: "all 0.15s ease",
            }}
            onMouseOver={(e) => {
              e.currentTarget.style.transform = "scale(1.15)";
              e.currentTarget.style.background = "#10b981";
            }}
            onMouseOut={(e) => {
              e.currentTarget.style.transform = "scale(1)";
              e.currentTarget.style.background = "#059669";
            }}
          >
            RN
          </button>

          {/* Edge Insert Radial Menu Legend (V4-R03) */}
          <div
            style={{
              position: "absolute",
              top: 150,
              left: "50%",
              transform: "translateX(-50%)",
              background: "rgba(15, 23, 42, 0.95)",
              backdropFilter: "blur(12px)",
              border: "1px solid rgba(255, 255, 255, 0.15)",
              borderRadius: "8px",
              padding: "10px 12px",
              width: 220,
              boxShadow: "0 10px 25px rgba(0, 0, 0, 0.5)",
              fontFamily: "system-ui, sans-serif",
              color: "#e2e8f0",
              fontSize: 10,
              display: "flex",
              flexDirection: "column",
              gap: 4,
            }}
          >
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", borderBottom: "1px solid rgba(255,255,255,0.1)", paddingBottom: 4, marginBottom: 2 }}>
              <span style={{ fontWeight: "bold", color: "#22d3ee" }}>⚡ Dynamic Adapter Splice</span>
              <span
                title="Dynamic Adapter Splicing: Right-click on any edge to splice in compatible sharding, transpose, and normalization adapters on the fly."
                style={{
                  background: "rgba(255,255,255,0.1)",
                  borderRadius: "50%",
                  width: 14,
                  height: 14,
                  display: "inline-flex",
                  alignItems: "center",
                  justifyContent: "center",
                  cursor: "help",
                  color: "#22d3ee",
                  fontWeight: "bold",
                }}
              >
                ?
              </span>
            </div>
            <div><strong style={{ color: "#38bdf8" }}>LB</strong>: Linear Bridge (sharding adapter)</div>
            <div><strong style={{ color: "#818cf8" }}>TR</strong>: Transpose BNSD (layout adapter)</div>
            <div><strong style={{ color: "#34d399" }}>RN</strong>: RMSNorm (normalization adapter)</div>
          </div>
        </div>
      )}
    </div>
  );
}
