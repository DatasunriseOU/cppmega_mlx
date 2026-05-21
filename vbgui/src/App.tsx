import { useCallback, useState } from "react";
import { ReactFlowProvider, type Edge, type Node } from "@xyflow/react";
import { FlowCanvas } from "@/components/FlowCanvas";
import { Palette } from "@/components/Palette";

export function App(): JSX.Element {
  const [nodes, setNodes] = useState<Node[]>([]);
  const [edges, setEdges] = useState<Edge[]>([]);

  const handleDropBrick = useCallback(
    (kind: string, position: { x: number; y: number }) => {
      setNodes((prev) => [
        ...prev,
        {
          id: `${kind}_${prev.length + 1}`,
          type: "brick",
          position,
          data: { kind },
        },
      ]);
    }, []);

  const handleConnect = useCallback(
    (p: { source: string; target: string }) => {
      setEdges((prev) => [
        ...prev,
        { id: `${p.source}->${p.target}`,
          source: p.source, target: p.target,
          data: { severity: "info" } },
      ]);
    }, []);

  return (
    <ReactFlowProvider>
      <div style={{ display: "flex", height: "100vh", margin: 0 }}>
        <Palette />
        <FlowCanvas
          nodes={nodes}
          edges={edges}
          onConnect={handleConnect}
          onDropBrick={handleDropBrick}
        />
      </div>
    </ReactFlowProvider>
  );
}
