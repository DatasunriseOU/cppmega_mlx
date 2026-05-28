import { useCallback, useMemo, useState, useEffect, useRef } from "react";
import {
  ReactFlow,
  Background,
  Controls,
  BaseEdge,
  EdgeLabelRenderer,
  getBezierPath,
  useReactFlow,
  useUpdateNodeInternals,
  MarkerType,
  type Edge,
  type Node,
  type Connection,
  type NodeTypes,
  type EdgeTypes,
  type EdgeProps,
  type IsValidConnection,
  type OnNodesChange,
  type OnEdgesChange,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";

import { FourSideHandles } from "./nodeHandles";
import { BrickNode } from "./BrickNode";
import { AdapterNode } from "./AdapterNode";
import { LossGhostNode } from "./LossGhostNode";
import { TokenizerVirtualNode, DetokenizerVirtualNode } from "./VirtualNodes";
import { BlockGroupNode } from "./BlockGroupNode";

export interface FlowCanvasProps {
  nodes: Node[];
  edges: Edge[];
  onConnect?: (params: { source: string; target: string }) => void;
  onReconnectEdge?: (oldEdge: Edge, newConnection: Connection) => void;
  onDeleteEdge?: (id: string) => void;
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
  /** Callback when an edge is context-clicked (right-click) or double-clicked to tap it. */
  onEdgeTap?: (edgeId: string) => void;
  /** Callback when the flow canvas container is resized. */
  onResize?: (width: number, height: number) => void;
}

// Beautiful custom glowing residual addition (+) node component
export function ResidualAddNode({ id }: { id: string; data?: any }): JSX.Element {
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
      <FourSideHandles style={{ background: "#10b981", width: 6, height: 6, border: "none" }} />
      <span>+</span>
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
  block_group: BlockGroupNode as unknown as NodeTypes[string],
};

// Build a rounded-corner orthogonal path through ELK bend points.
function roundedOrthPath(points: { x: number; y: number }[], r = 9): string {
  if (points.length < 2) return "";
  let d = `M ${points[0].x} ${points[0].y}`;
  for (let i = 1; i < points.length - 1; i++) {
    const p0 = points[i - 1], p1 = points[i], p2 = points[i + 1];
    const v1x = p1.x - p0.x, v1y = p1.y - p0.y;
    const v2x = p2.x - p1.x, v2y = p2.y - p1.y;
    const l1 = Math.hypot(v1x, v1y) || 1;
    const l2 = Math.hypot(v2x, v2y) || 1;
    const rr = Math.min(r, l1 / 2, l2 / 2);
    const ax = p1.x - (v1x / l1) * rr, ay = p1.y - (v1y / l1) * rr;
    const bx = p1.x + (v2x / l2) * rr, by = p1.y + (v2y / l2) * rr;
    d += ` L ${ax} ${ay} Q ${p1.x} ${p1.y} ${bx} ${by}`;
  }
  const last = points[points.length - 1];
  d += ` L ${last.x} ${last.y}`;
  return d;
}

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
  const elkBends = (data as { elkBends?: { x: number; y: number }[] } | undefined)?.elkBends;
  let edgePath: string;
  let labelX: number;
  let labelY: number;
  // Bends are computed once per auto-align. If the user has since dragged
  // either endpoint, the stored bend points no longer match the live handles
  // and the path would crook back to the old position. Detect that and fall
  // back to a clean bezier — the orthogonal route returns on the next align.
  const bendsFresh =
    !!elkBends && elkBends.length >= 2 &&
    Math.hypot(elkBends[0].x - sourceX, elkBends[0].y - sourceY) < 24 &&
    Math.hypot(elkBends[elkBends.length - 1].x - targetX, elkBends[elkBends.length - 1].y - targetY) < 24;
  if (bendsFresh && elkBends) {
    // Glue endpoints to the live handle positions, route through ELK's bends.
    const pts = elkBends.map((p) => ({ ...p }));
    pts[0] = { x: sourceX, y: sourceY };
    pts[pts.length - 1] = { x: targetX, y: targetY };
    edgePath = roundedOrthPath(pts);
    const mid = pts[Math.floor(pts.length / 2)];
    labelX = mid.x;
    labelY = mid.y;
  } else {
    [edgePath, labelX, labelY] = getBezierPath({
      sourceX,
      sourceY,
      sourcePosition,
      targetX,
      targetY,
      targetPosition,
    });
  }

  const sev = (data as { severity?: string } | undefined)?.severity;
  const adapter = (data as { adapter?: boolean } | undefined)?.adapter;
  
  const debuggerMode = (data as { debuggerMode?: boolean } | undefined)?.debuggerMode;
  const direction = (data as { direction?: "forward" | "backward" } | undefined)?.direction;
  const isActiveFlow = (data as { isActiveFlow?: boolean } | undefined)?.isActiveFlow;

  const isTapped = (data as { isTapped?: boolean } | undefined)?.isTapped;
  const hasHook = (data as { hasHook?: boolean } | undefined)?.hasHook;

  let stroke = "#10b981"; // default emerald green
  let strokeDasharray: string | undefined = undefined;
  let animation: string | undefined = undefined;
  let opacity: number | undefined = undefined;

  if (isTapped || hasHook) {
    stroke = "#a855f7"; // Neon purple accent
    strokeDasharray = "6 3";
    animation = "vbEdgePulse 1.2s linear infinite";
    opacity = 1.0;
  } else if (debuggerMode) {
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
        @keyframes vbEdgePulse {
          0% { stroke: #a855f7; filter: drop-shadow(0 0 2px #a855f7); stroke-dashoffset: 0; }
          50% { stroke: #22d3ee; filter: drop-shadow(0 0 8px #22d3ee); }
          100% { stroke: #a855f7; filter: drop-shadow(0 0 2px #a855f7); stroke-dashoffset: 18; }
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
  nodes, edges, onConnect, onReconnectEdge, onDeleteEdge, isValidConnection, onDropBrick, onNodeClick, onInsertAdapter, onAutoAlign, onNodesChange, onEdgesChange, onEdgeTap, onResize,
}: FlowCanvasProps): JSX.Element {
  const [edgeMenu, setEdgeMenu] = useState<{ edge: Edge; x: number; y: number } | null>(null);
  const { fitView } = useReactFlow();
  const updateNodeInternals = useUpdateNodeInternals();

  // Manual edge reconnection — drag an edge end onto any handle (or onto empty
  // canvas to delete it). The success ref distinguishes a valid drop from a
  // drop into the void (onReconnectEnd fires in both cases).
  const reconnectOk = useRef(true);
  const handleReconnectStart = useCallback(() => { reconnectOk.current = false; }, []);
  const handleReconnect = useCallback((oldEdge: Edge, conn: Connection) => {
    reconnectOk.current = true;
    onReconnectEdge?.(oldEdge, conn);
  }, [onReconnectEdge]);
  const handleReconnectEnd = useCallback((_e: unknown, edge: Edge) => {
    if (!reconnectOk.current) onDeleteEdge?.(edge.id);
    reconnectOk.current = true;
  }, [onDeleteEdge]);

  // Each node now exposes 8 handles (4 sides × source/target); the router
  // assigns which handle each edge uses via edge.sourceHandle/targetHandle and
  // also flips data.sourcePosition/targetPosition. React Flow caches handle
  // bounds and resolves edge endpoints from the referenced handle — both must
  // be re-measured when the router reassigns sides, or routes attach to the
  // stale side and bezier-cross node boxes. Re-measure on any change to node
  // handle sides OR the set of handle ids edges reference.
  const handleSidesSig = nodes
    .map((n) => `${n.id}:${(n.data as any)?.sourcePosition ?? ""}:${(n.data as any)?.targetPosition ?? ""}`)
    .join("|");
  const edgeHandleSig = edges
    .map((e) => `${e.id}:${e.sourceHandle ?? ""}:${e.targetHandle ?? ""}`)
    .join("|");
  useEffect(() => {
    nodes.forEach((n) => updateNodeInternals(n.id));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [handleSidesSig, edgeHandleSig, updateNodeInternals]);

  const wrapperRef = useRef<HTMLDivElement | null>(null);
  useEffect(() => {
    if (!wrapperRef.current || typeof ResizeObserver === "undefined" || !onResize) return;
    const observer = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const w = entry.contentRect.width;
        const h = entry.contentRect.height;
        if (w > 600) {
          onResize(w, h);
        }
      }
    });
    observer.observe(wrapperRef.current);
    return () => observer.disconnect();
  }, [onResize]);

  // Robust automatic centering & zoom fitting when nodes list changes (preset load or auto-layout alignment)
  useEffect(() => {
    if (nodes.length > 0) {
      const timer = setTimeout(() => {
        void fitView({ padding: 0.15, duration: 250 });
      }, 50);
      return () => clearTimeout(timer);
    }
  }, [nodes.length, fitView]);

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
    () => edges.map((e) => {
      const isTapped = e.data?.isTapped || e.data?.hasHook;
      const debuggerMode = e.data?.debuggerMode;
      const isActiveFlow = e.data?.isActiveFlow;
      const direction = e.data?.direction;
      const adapter = e.data?.adapter;
      const sev = e.data?.severity;

      let arrowColor = "#10b981"; // default emerald green
      if (isTapped) {
        arrowColor = "#a855f7"; // Neon purple accent
      } else if (debuggerMode) {
        if (isActiveFlow) {
          arrowColor = direction === "forward" ? "#22d3ee" : "#ec4899";
        } else {
          arrowColor = "rgba(100, 116, 139, 0.4)";
        }
      } else if (adapter) {
        arrowColor = "#9ca3af";
      } else if (sev === "error") {
        arrowColor = "#dc2626";
      } else if (sev === "warning") {
        arrowColor = "#d97706";
      }

      return {
        ...e,
        type: "midpoint",
        markerEnd: {
          type: MarkerType.ArrowClosed,
          width: 14,
          height: 14,
          color: arrowColor,
        },
        data: {
          ...e.data,
          onClickMidpoint: (event: React.MouseEvent) => {
            handleMidpointClick(event, e);
          }
        }
      };
    }),
    [edges, handleMidpointClick],
  );

  return (
    <div
      ref={wrapperRef}
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
        onReconnect={handleReconnect as never}
        onReconnectStart={handleReconnectStart}
        onReconnectEnd={handleReconnectEnd as never}
        reconnectRadius={12}
        onNodeClick={(_e, node) => onNodeClick?.(node.id)}
        onEdgeClick={handleMidpointClick}
        onEdgeContextMenu={(e, edge) => {
          e.preventDefault();
          onEdgeTap?.(edge.id);
        }}
        onEdgeDoubleClick={(_e, edge) => {
          onEdgeTap?.(edge.id);
        }}
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
