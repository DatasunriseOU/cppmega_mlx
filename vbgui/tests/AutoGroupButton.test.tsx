import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { AutoGroupButton } from "@/components/AutoGroupButton";
import type { Node, Edge } from "@xyflow/react";

const NODES: Node[] = [
  { id: "attn", type: "brick", position: { x: 0, y: 0 }, data: { kind: "attention" } },
  { id: "mlp",  type: "brick", position: { x: 0, y: 0 }, data: { kind: "mlp" } },
];
const EDGES: Edge[] = [
  { id: "attn->mlp", source: "attn", target: "mlp" },
];

const FAKE_RESULT = {
  proposals: [
    { matcher: "regex:.*bias.*", optim_kind: "adamw",
      lr: 3e-4, weight_decay: 0.01, betas: [0.9, 0.95],
      ns_steps: null, param_count: 128, rationale: "1D biases" },
    { matcher: "regex:.*\\.weight$", optim_kind: "muon",
      lr: 2e-3, weight_decay: 0.01, betas: null,
      ns_steps: 5, param_count: 327680, rationale: "2D backbone" },
  ],
  total_params: 327808,
  uncovered_params: 0,
};

function fakeRpc(payload: unknown) {
  return { call: vi.fn(async () => payload) } as never;
}

describe("AutoGroupButton", () => {
  it("renders disabled when canvas empty", () => {
    render(<AutoGroupButton rpc={fakeRpc(FAKE_RESULT)}
                             optimKind="muon_adamw_hybrid"
                             nodes={[]} edges={[]}
                             onApply={() => {}} />);
    const btn = screen.getByTestId("optim-auto-group") as HTMLButtonElement;
    expect(btn.disabled).toBe(true);
  });

  it("calls suggest_optim_groups + fires onApply with mapped groups", async () => {
    const rpc = fakeRpc(FAKE_RESULT);
    const onApply = vi.fn();
    render(<AutoGroupButton rpc={rpc}
                             optimKind="muon_adamw_hybrid"
                             nodes={NODES} edges={EDGES}
                             onApply={onApply} />);
    fireEvent.click(screen.getByTestId("optim-auto-group"));
    await waitFor(() => expect(onApply).toHaveBeenCalled());
    const [groups, banner] = onApply.mock.calls[0];
    expect(groups.length).toBe(2);
    expect(groups[0].matcher).toContain("bias");
    expect(groups[1].matcher).toContain("weight");
    expect(banner).toContain("Auto-grouped 2 groups");
    expect(rpc.call).toHaveBeenCalledWith(
      "suggest_optim_groups",
      expect.objectContaining({
        optim_kind: "muon_adamw_hybrid",
        hidden_size: 128,
        graph: expect.any(Object),
      }),
    );
  });

  it("shows error on RPC failure", async () => {
    const rpc = { call: vi.fn(async () => { throw new Error("boom"); }) } as never;
    render(<AutoGroupButton rpc={rpc}
                             optimKind="adamw"
                             nodes={NODES} edges={EDGES}
                             onApply={() => {}} />);
    fireEvent.click(screen.getByTestId("optim-auto-group"));
    await waitFor(() => {
      expect(screen.getByTestId("optim-auto-group-error").textContent)
        .toContain("boom");
    });
  });

  it("button label shows Analysing… during pending RPC", async () => {
    const rpc = { call: vi.fn(() => new Promise(() => {})) } as never;
    render(<AutoGroupButton rpc={rpc}
                             optimKind="adamw"
                             nodes={NODES} edges={EDGES}
                             onApply={() => {}} />);
    fireEvent.click(screen.getByTestId("optim-auto-group"));
    await waitFor(() => {
      expect(screen.getByTestId("optim-auto-group").textContent)
        .toContain("Analysing");
    });
  });
});
