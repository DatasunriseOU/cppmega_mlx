// V7-H44: AutoGroupButton in diff-mode stages proposals + renders
// side-by-side comparison vs currentGroups; Accept-all commits.

import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent, waitFor }
  from "@testing-library/react";
import { AutoGroupButton } from "@/components/AutoGroupButton";
import type { RpcClient } from "@/lib/rpc";
import type { Node, Edge } from "@xyflow/react";

const NODES: Node[] = [
  { id: "attn", type: "brick", position: { x: 0, y: 0 },
    data: { kind: "attention", params: {} } as never },
  { id: "mlp", type: "brick", position: { x: 200, y: 0 },
    data: { kind: "mlp", params: {} } as never },
];
const EDGES: Edge[] = [];

const REPLY = {
  proposals: [
    { matcher: "attention", optim_kind: "adamw", lr: 5e-4,
      weight_decay: 0.01, betas: [0.9, 0.95] as [number, number],
      ns_steps: null, param_count: 100, rationale: "attention matrices" },
    { matcher: "mlp", optim_kind: "adamw", lr: 1e-3,
      weight_decay: 0.0, betas: [0.9, 0.95] as [number, number],
      ns_steps: null, param_count: 200, rationale: "mlp weights" },
  ],
  total_params: 300, uncovered_params: 0,
};

const CURRENT_GROUPS = [
  { matcher: "all", lr: 1e-4, weight_decay: 0.01 },
];

function makeRpc(reply: unknown): RpcClient {
  return { call: vi.fn(async () => reply as never) } as unknown as RpcClient;
}

describe("V7-H44 AutoGroupButton diff mode", () => {
  it("legacy path (no currentGroups) auto-applies on click", async () => {
    const rpc = makeRpc(REPLY);
    const onApply = vi.fn();
    render(<AutoGroupButton rpc={rpc} optimKind="adamw"
                             nodes={NODES} edges={EDGES}
                             onApply={onApply} />);
    fireEvent.click(screen.getByTestId("optim-auto-group"));
    await waitFor(() => expect(onApply).toHaveBeenCalledTimes(1));
    expect(onApply.mock.calls[0]![0]).toHaveLength(2);
    // No diff panel in legacy mode.
    expect(screen.queryByTestId("optim-auto-group-diff")).toBeNull();
  });

  it("diff mode (currentGroups supplied) stages proposals without applying",
     async () => {
    const rpc = makeRpc(REPLY);
    const onApply = vi.fn();
    render(<AutoGroupButton rpc={rpc} optimKind="adamw"
                             nodes={NODES} edges={EDGES}
                             currentGroups={CURRENT_GROUPS}
                             onApply={onApply} />);
    fireEvent.click(screen.getByTestId("optim-auto-group"));
    await waitFor(() => {
      expect(screen.getByTestId("optim-auto-group-diff")).toBeTruthy();
    });
    expect(onApply).not.toHaveBeenCalled();
    expect(screen.getByTestId("optim-auto-group-diff-summary").textContent)
      .toContain("2 group");
    expect(screen.getByTestId("optim-auto-group-diff-summary").textContent)
      .toContain("current has 1");
    expect(screen.getByTestId("optim-diff-row-attention")).toBeTruthy();
    expect(screen.getByTestId("optim-diff-row-mlp")).toBeTruthy();
  });

  it("Accept-all commits proposals + hides diff panel", async () => {
    const rpc = makeRpc(REPLY);
    const onApply = vi.fn();
    render(<AutoGroupButton rpc={rpc} optimKind="adamw"
                             nodes={NODES} edges={EDGES}
                             currentGroups={CURRENT_GROUPS}
                             onApply={onApply} />);
    fireEvent.click(screen.getByTestId("optim-auto-group"));
    await waitFor(() => {
      expect(screen.getByTestId("optim-auto-group-accept-all"))
        .toBeTruthy();
    });
    fireEvent.click(screen.getByTestId("optim-auto-group-accept-all"));
    expect(onApply).toHaveBeenCalledTimes(1);
    expect(onApply.mock.calls[0]![0]).toHaveLength(2);
    expect(screen.queryByTestId("optim-auto-group-diff")).toBeNull();
  });

  it("Discard drops proposals without committing", async () => {
    const rpc = makeRpc(REPLY);
    const onApply = vi.fn();
    render(<AutoGroupButton rpc={rpc} optimKind="adamw"
                             nodes={NODES} edges={EDGES}
                             currentGroups={CURRENT_GROUPS}
                             onApply={onApply} />);
    fireEvent.click(screen.getByTestId("optim-auto-group"));
    await waitFor(() => {
      expect(screen.getByTestId("optim-auto-group-discard")).toBeTruthy();
    });
    fireEvent.click(screen.getByTestId("optim-auto-group-discard"));
    expect(onApply).not.toHaveBeenCalled();
    expect(screen.queryByTestId("optim-auto-group-diff")).toBeNull();
  });
});
