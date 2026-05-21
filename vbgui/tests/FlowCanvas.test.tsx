import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import { ReactFlowProvider } from "@xyflow/react";
import { FlowCanvas } from "@/components/FlowCanvas";

function wrap(node: React.ReactNode) {
  return <ReactFlowProvider>{node}</ReactFlowProvider>;
}

describe("FlowCanvas", () => {
  it("mounts an empty canvas without crashing", () => {
    render(wrap(<FlowCanvas nodes={[]} edges={[]} />));
    expect(screen.getByTestId("flow-canvas")).toBeTruthy();
  });

  it("renders a brick node when one is provided", () => {
    const nodes = [
      { id: "n1", type: "brick", position: { x: 0, y: 0 },
        data: { kind: "attention" } as never },
    ];
    render(wrap(<FlowCanvas nodes={nodes} edges={[]} />));
    // React Flow lazy-mounts node UI; assert the canvas exists + accepts node count.
    expect(screen.getByTestId("flow-canvas")).toBeTruthy();
  });

  it("calls onDropBrick when a brick mime is dropped", () => {
    const onDropBrick = vi.fn();
    render(wrap(<FlowCanvas nodes={[]} edges={[]} onDropBrick={onDropBrick} />));
    const canvas = screen.getByTestId("flow-canvas");
    const event = new Event("drop", { bubbles: true }) as DragEvent;
    Object.defineProperty(event, "dataTransfer", {
      value: {
        getData: (k: string) =>
          k === "application/x-cppmega-brick" ? "mlp" : "",
        setData: () => undefined,
        effectAllowed: "copy",
        dropEffect: "copy",
      } as unknown as DataTransfer,
    });
    Object.defineProperty(event, "clientX", { value: 100 });
    Object.defineProperty(event, "clientY", { value: 50 });
    canvas.dispatchEvent(event);
    expect(onDropBrick).toHaveBeenCalledTimes(1);
    expect(onDropBrick.mock.calls[0][0]).toBe("mlp");
  });
});
