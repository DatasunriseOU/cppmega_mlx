import { describe, it, expect, vi } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import { CompileTracePanel } from "@/components/CompileTracePanel";

function makeFakeRpc(responses: Record<string, unknown>) {
  const calls: { method: string; params: unknown }[] = [];
  return {
    calls,
    rpc: {
      call: vi.fn(async (m: string, p: unknown) => {
        calls.push({ method: m, params: p });
        const r = responses[m];
        if (r instanceof Error) throw r;
        return r;
      }),
    } as never,
  };
}

const SYNC = {
  necessary_syncs: [{ after_op: "c", reason: "output" }],
  redundant_syncs: [
    { after_op: "a", reason: "same backend" },
    { after_op: "b", reason: "same backend" },
  ],
  advice: [
    { op: "a", fix: "remove mx.eval", confidence: "high" },
    { op: "b", fix: "remove mx.eval", confidence: "high" },
  ],
  z3_solver_status: "sat",
  z3_elapsed_ms: 12.3,
};

const TRACE = {
  ops: [
    { name: "a", fused: true,  group: "region_00_path_c",
      materialised: false, dlpack_boundary: false, backend: "path_c" },
    { name: "b", fused: true,  group: "region_00_path_c",
      materialised: false, dlpack_boundary: false, backend: "path_c" },
    { name: "c", fused: false, group: "region_01_dlpack_handoff",
      materialised: true,  dlpack_boundary: true,  backend: "dlpack_handoff" },
  ],
  fused_groups: ["region_00_path_c"],
  dlpack_crossings: 1,
  materialised_ops: ["c"],
  compile_artifact_path: null,
  backend: "mlx",
};

describe("V8-R06 CompileTracePanel", () => {
  it("renders ops + chips + aggregate counters", async () => {
    const { rpc } = makeFakeRpc({
      "compile.trace": TRACE,
      "sync.check": SYNC,
    });
    render(<CompileTracePanel rpc={rpc} specPayload={{}} backend="mlx" />);
    await waitFor(() => {
      expect(screen.getByTestId("compile-trace")).toBeDefined();
    });
    await waitFor(() => {
      expect(screen.getByTestId("sync-check-badge")).toBeDefined();
    });
    expect(screen.getByTestId("sync-check-redundant-count").textContent)
      .toBe("2");
    expect(screen.getByTestId("compile-trace-fused-count").textContent)
      .toContain("1");
    expect(screen.getByTestId("compile-trace-dlpack-crossings").textContent)
      .toContain("1");
    expect(screen.getByTestId("compile-trace-materialised-count").textContent)
      .toContain("1");
    expect(screen.getByTestId("compile-trace-op-0")).toBeDefined();
    expect(screen.getByTestId("compile-trace-op-2-materialised"))
      .toBeDefined();
    expect(screen.getByTestId("compile-trace-op-2-dlpack")).toBeDefined();
  });

  it("renders error banner on RPC failure", async () => {
    const { rpc } = makeFakeRpc({
      "compile.trace": new Error("planner crashed") });
    render(<CompileTracePanel rpc={rpc} specPayload={{}} />);
    await waitFor(() => {
      expect(screen.getByTestId("compile-trace-error").textContent ?? "")
        .toContain("planner crashed");
    });
  });
});
