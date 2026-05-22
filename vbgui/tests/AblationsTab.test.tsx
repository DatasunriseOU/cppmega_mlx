import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { AblationsTab } from "@/components/sidebar/AblationsTab";
import type { RpcClient } from "@/lib/rpc";
import type { Node, Edge } from "@xyflow/react";
import type { OptimState, LossState } from "@/state/spec";

const NODES: Node[] = [
  { id: "mlp_0", type: "brick", position: { x: 0, y: 0 },
    data: { kind: "mlp" } },
];
const EDGES: Edge[] = [];
const OPTIM: OptimState = {
  kind: "adamw",
  groups: [{ matcher: "all", lr: 3e-4, weight_decay: 0.01,
              betas: [0.9, 0.95] }],
  grad_clip_norm: 1.0,
  mixed_precision: true,
};
const LOSS: LossState = {
  kind: "cross_entropy",
  head_outputs: ["mlp_0"],
  params: {},
};

const RESULT = {
  results: [
    { variant: "glu", status: "ok", losses: [5.4, 5.2],
      elapsed_ms: 10, weight_delta_norm: 0.5 },
    { variant: "swiglu", status: "ok", losses: [5.3, 5.0],
      elapsed_ms: 12, weight_delta_norm: 0.6 },
  ],
  ranked_by_final_loss: ["swiglu", "glu"],
  baseline_variant: "glu",
  elapsed_ms_total: 25,
};

function fakeRpc(payload: unknown): RpcClient {
  return { call: vi.fn(async () => payload) } as unknown as RpcClient;
}

describe("AblationsTab", () => {
  it("renders axis dropdown + variant chips + Run button", () => {
    render(<AblationsTab rpc={fakeRpc(RESULT)} nodes={NODES} edges={EDGES}
                          optim={OPTIM} loss={LOSS} />);
    expect(screen.getByTestId("ablation-axis")).toBeTruthy();
    expect(screen.getByTestId("ablation-run")).toBeTruthy();
    expect(screen.getByTestId("ablation-variant-glu")).toBeTruthy();
    expect(screen.getByTestId("ablation-variant-swiglu")).toBeTruthy();
  });

  it("switching axis swaps variant list", () => {
    render(<AblationsTab rpc={fakeRpc(RESULT)} nodes={NODES} edges={EDGES}
                          optim={OPTIM} loss={LOSS} />);
    fireEvent.change(screen.getByTestId("ablation-axis"),
                     { target: { value: "optimizer" } });
    expect(screen.getByTestId("ablation-variant-lion")).toBeTruthy();
    expect(screen.queryByTestId("ablation-variant-swiglu")).toBeNull();
  });

  it("Run with <2 variants surfaces error", async () => {
    render(<AblationsTab rpc={fakeRpc(RESULT)} nodes={NODES} edges={EDGES}
                          optim={OPTIM} loss={LOSS} />);
    // Untick one of the default variants
    fireEvent.click(screen.getByTestId("ablation-variant-swiglu"));
    fireEvent.click(screen.getByTestId("ablation-run"));
    await waitFor(() =>
      expect(screen.getByTestId("ablation-error").textContent)
        .toContain("at least 2"));
  });

  it("Run on empty canvas surfaces error", async () => {
    render(<AblationsTab rpc={fakeRpc(RESULT)} nodes={[]} edges={[]}
                          optim={OPTIM} loss={LOSS} />);
    fireEvent.click(screen.getByTestId("ablation-run"));
    await waitFor(() =>
      expect(screen.getByTestId("ablation-error").textContent)
        .toContain("empty"));
  });

  it("Run renders results table sorted ascending by final loss", async () => {
    const rpc = fakeRpc(RESULT);
    render(<AblationsTab rpc={rpc} nodes={NODES} edges={EDGES}
                          optim={OPTIM} loss={LOSS} />);
    fireEvent.click(screen.getByTestId("ablation-run"));
    await waitFor(() => screen.getByTestId("ablation-results"));
    // ranked_by_final_loss=[swiglu,glu] so swiglu row appears first.
    // Match exact variant rows (H14 adds child rows like ablation-row-X-expand
    // and ablation-row-X-extras, so use a $-anchored regex).
    const rows = screen.getAllByTestId(/^ablation-row-[a-z0-9_]+$/);
    expect(rows[0].getAttribute("data-testid")).toBe("ablation-row-swiglu");
    expect(rows[1].getAttribute("data-testid")).toBe("ablation-row-glu");
  });

  it("baseline gets ★ marker", async () => {
    render(<AblationsTab rpc={fakeRpc(RESULT)} nodes={NODES} edges={EDGES}
                          optim={OPTIM} loss={LOSS} />);
    fireEvent.click(screen.getByTestId("ablation-run"));
    await waitFor(() => screen.getByTestId("ablation-row-glu"));
    expect(screen.getByTestId("ablation-row-glu").textContent)
      .toContain("★");
  });

  it("H14: expand row reveals extras subtree with losses + per-key rows",
    async () => {
      const RESULT_WITH_EXTRAS = {
        results: [
          { variant: "glu", status: "ok", losses: [5.4, 5.2],
            elapsed_ms: 10, weight_delta_norm: 0.5,
            extras: {
              optimizer_kind: "adamw", schedule_kind: "constant",
              data_source: "synthetic",
              model_summary: { optimizer_kind: "adamw",
                                num_brick_kinds: 2 },
            } },
        ],
        ranked_by_final_loss: ["glu"],
        baseline_variant: "glu", elapsed_ms_total: 12,
      };
      render(<AblationsTab rpc={fakeRpc(RESULT_WITH_EXTRAS)}
                            nodes={NODES} edges={EDGES}
                            optim={OPTIM} loss={LOSS} />);
      fireEvent.click(screen.getByTestId("ablation-run"));
      await waitFor(() => screen.getByTestId("ablation-row-glu"));
      // Pre-expand: extras row absent.
      expect(screen.queryByTestId("ablation-row-glu-extras")).toBeNull();
      // Click expand → extras row appears with per-key testids.
      fireEvent.click(screen.getByTestId("ablation-row-glu-expand"));
      expect(screen.getByTestId("ablation-row-glu-extras")).toBeTruthy();
      expect(screen.getByTestId("ablation-row-glu-losses").textContent)
        .toContain("5.4000");
      expect(screen.getByTestId(
        "ablation-row-glu-extras-optimizer_kind").textContent)
        .toBe("adamw");
      expect(screen.getByTestId(
        "ablation-row-glu-extras-data_source").textContent)
        .toBe("synthetic");
    });

  it("num_steps input updates state", () => {
    render(<AblationsTab rpc={fakeRpc(RESULT)} nodes={NODES} edges={EDGES}
                          optim={OPTIM} loss={LOSS} />);
    fireEvent.change(screen.getByTestId("ablation-num-steps"),
                     { target: { value: "20" } });
    expect((screen.getByTestId("ablation-num-steps") as HTMLInputElement)
           .value).toBe("20");
  });
});
