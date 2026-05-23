import { describe, it, expect } from "vitest";
import { createElement } from "react";
import { render, screen } from "@testing-library/react";
import { ReactFlowProvider } from "@xyflow/react";
import { BrickNode, type BrickNodeData } from "@/components/BrickNode";
import { AdapterNode, type AdapterNodeData } from "@/components/AdapterNode";

function wrap(node: React.ReactNode) {
  return <ReactFlowProvider>{node}</ReactFlowProvider>;
}

const baseProps = {
  id: "n1",
  type: "brick",
  dragging: false,
  selected: false,
  selectable: true,
  deletable: true,
  draggable: true,
  isConnectable: true,
  zIndex: 0,
  xPos: 0,
  yPos: 0,
  positionAbsoluteX: 0,
  positionAbsoluteY: 0,
};

// Render component with mixed-in baseProps via createElement to bypass
// JSX strict spread typing — BrickNode/AdapterNode internally only read
// data and id, the other NodeProps fields are React Flow plumbing.
function renderBrick(data: BrickNodeData) {
  return render(wrap(createElement(
    BrickNode as never, { ...baseProps, data } as never,
  )));
}

function renderAdapter(data: AdapterNodeData) {
  return render(wrap(createElement(
    AdapterNode as never, { ...baseProps, data, type: "adapter" } as never,
  )));
}

describe("BrickNode", () => {
  it("renders the brick label and kind", () => {
    renderBrick({ kind: "attention" });
    expect(screen.getByText("Attention (vanilla)")).toBeTruthy();
    expect(screen.getByText("sdpa attention")).toBeTruthy();
  });

  it("shows resolved shape when provided", () => {
    renderBrick({ kind: "mlp", shape: [1, 4, 64] });
    expect(screen.getByTestId("brick-shape").textContent).toContain("[1, 4, 64]");
  });

  it("shows memory bar when provided", () => {
    renderBrick({ kind: "mlp", memory_mb: 12.5 });
    expect(screen.getByTestId("brick-memory-bar").textContent).toContain("12.5");
  });

  it("shows side-channel warning when missing", () => {
    renderBrick({ kind: "engram", side_channels_ok: false });
    expect(screen.getByTestId("brick-side-channel-warn")).toBeTruthy();
  });

  it("falls back to kind when meta is unknown", () => {
    renderBrick({ kind: "totally_made_up" });
    expect(screen.getAllByText("totally_made_up").length).toBeGreaterThan(0);
  });
});

describe("AdapterNode", () => {
  it("renders the adapter label", () => {
    renderAdapter({ kind: "residual" });
    expect(screen.getByText("Residual Add")).toBeTruthy();
  });

  it("falls back to kind for unknown adapter", () => {
    renderAdapter({ kind: "no_such_adapter" });
    expect(screen.getByText("no_such_adapter")).toBeTruthy();
  });
});
