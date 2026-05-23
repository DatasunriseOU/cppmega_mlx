/**
 * fhxg: LossGhostNode renders the spec.loss kind + params as a
 * dashed canvas node with the synthetic-loss label.
 */

import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { ReactFlowProvider } from "@xyflow/react";
import { LossGhostNode } from "@/components/LossGhostNode";

function renderNode(data: Record<string, unknown>) {
  return render(
    <ReactFlowProvider>
      <LossGhostNode
        id="g1"
        data={data as never}
        type="loss_ghost"
        selected={false}
        dragging={false}
        isConnectable={false}
        positionAbsoluteX={0}
        positionAbsoluteY={0}
        zIndex={0} />
    </ReactFlowProvider>,
  );
}

describe("LossGhostNode", () => {
  it("renders the loss kind chip", () => {
    renderNode({ kind: "ifim_shaped", params: { lambda_t: 0.1 } });
    expect(screen.getByTestId("loss-ghost-g1")).toBeDefined();
    expect(screen.getByTestId("loss-ghost-kind").textContent)
      .toContain("ifim_shaped");
    expect(screen.getByTestId("loss-ghost-params").textContent)
      .toContain("lambda_t=0.1");
  });

  it("renders without params block when params are empty", () => {
    renderNode({ kind: "cross_entropy" });
    expect(screen.getByTestId("loss-ghost-kind").textContent)
      .toContain("cross_entropy");
  });
});
