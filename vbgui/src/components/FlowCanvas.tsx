import { useCallback, useMemo } from "react";
import {
  ReactFlow,
  Background,
  Controls,
  type Edge,
  type Node,
  type NodeTypes,
  type IsValidConnection,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";

import { BrickNode } from "./BrickNode";
import { AdapterNode } from "./AdapterNode";

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
}

const NODE_TYPES: NodeTypes = {
  brick: BrickNode as unknown as NodeTypes[string],
  adapter: AdapterNode as unknown as NodeTypes[string],
};

export function FlowCanvas({
  nodes, edges, onConnect, isValidConnection, onDropBrick, onNodeClick,
}: FlowCanvasProps): JSX.Element {
  const handleConnect = useCallback(
    (p: { source: string; target: string }) => onConnect?.(p),
    [onConnect],
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

  const styledEdges = useMemo(
    () => edges.map(styleEdge),
    [edges],
  );

  return (
    <div
      data-testid="flow-canvas"
      style={{ flex: 1, height: "100%" }}
      onDrop={handleDrop}
      onDragOver={handleDragOver}
    >
      <ReactFlow
        nodes={nodes}
        edges={styledEdges}
        nodeTypes={NODE_TYPES}
        onConnect={handleConnect as never}
        onNodeClick={(_e, node) => onNodeClick?.(node.id)}
        isValidConnection={isValidConnection}
        fitView
      >
        <Background />
        <Controls />
      </ReactFlow>
    </div>
  );
}

function styleEdge(edge: Edge): Edge {
  const sev = (edge.data as { severity?: string } | undefined)?.severity;
  const adapter = (edge.data as { adapter?: boolean } | undefined)?.adapter;
  if (adapter) {
    return { ...edge, animated: false,
             style: { stroke: "#9ca3af", strokeDasharray: "4 2" } };
  }
  const stroke =
    sev === "error" ? "#dc2626" :
    sev === "warning" ? "#d97706" :
    "#10b981";
  return { ...edge, style: { ...(edge.style ?? {}), stroke, strokeWidth: 2 } };
}
