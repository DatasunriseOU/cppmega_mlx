/**
 * V8-R03 vitest: MemoryMatrixTab.
 *
 * Asserts:
 *  - issues memory.matrix RPC on mount, given a specPayload
 *  - renders one cell per (topology × precision) with bytes label
 *    + fits/over chip + data-fits attribute
 *  - re-fires the RPC when specPayload changes (debounce-equivalent)
 *  - renders an error banner when RPC throws
 */

import { describe, it, expect, vi } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import { MemoryMatrixTab }
  from "@/components/sidebar/MemoryMatrixTab";

function makeFakeRpc(responses: Record<string, unknown>) {
  const calls: { method: string; params: unknown }[] = [];
  return {
    calls,
    rpc: {
      call: vi.fn(async (method: string, params: unknown) => {
        calls.push({ method, params });
        const r = responses[method];
        if (r instanceof Error) throw r;
        return r;
      }),
    } as never,
  };
}

const MATRIX_OK = {
  cells: [
    { topology: "h100_8x", precision: "bf16",
      bytes: 1_200_000_000, device_hbm_bytes: 8 * 80 * 1024 ** 3,
      fits: true, headroom: 0.9,
      breakdown: { weights: 1_000_000_000, grads: 200_000_000,
                   optimizer: 0, activations: 0, kv_cache: 0,
                   edge_handoff: 0 } },
    { topology: "h100_8x", precision: "mxfp4",
      bytes: 300_000_000, device_hbm_bytes: 8 * 80 * 1024 ** 3,
      fits: true, headroom: 0.9,
      breakdown: { weights: 250_000_000, grads: 50_000_000,
                   optimizer: 0, activations: 0, kv_cache: 0,
                   edge_handoff: 0 } },
    { topology: "m3_ultra_solo", precision: "bf16",
      bytes: 200_000_000_000, device_hbm_bytes: 128 * 1024 ** 3,
      fits: false, headroom: 0.9,
      breakdown: { weights: 200_000_000_000, grads: 0,
                   optimizer: 0, activations: 0, kv_cache: 0,
                   edge_handoff: 0 } },
    { topology: "m3_ultra_solo", precision: "mxfp4",
      bytes: 50_000_000_000, device_hbm_bytes: 128 * 1024 ** 3,
      fits: true, headroom: 0.9,
      breakdown: { weights: 50_000_000_000, grads: 0,
                   optimizer: 0, activations: 0, kv_cache: 0,
                   edge_handoff: 0 } },
  ],
  topologies: ["h100_8x", "m3_ultra_solo"],
  precisions: ["bf16", "mxfp4"],
};

describe("V8-R03 MemoryMatrixTab", () => {
  it("renders the grid and per-cell fits chip", async () => {
    const { rpc, calls } = makeFakeRpc({ "memory.matrix": MATRIX_OK });
    render(
      <MemoryMatrixTab rpc={rpc}
                       specPayload={{ stub: 1 }}
                       topologies={["h100_8x", "m3_ultra_solo"]}
                       precisions={["bf16", "mxfp4"]} />,
    );

    await waitFor(() => {
      expect(calls.some((c) => c.method === "memory.matrix")).toBe(true);
    });
    await waitFor(() => {
      expect(screen.getByTestId("memory-matrix")).toBeDefined();
    });

    // 4 cells exist with proper testids
    expect(screen.getByTestId("memory-matrix-cell-h100_8x-bf16"))
      .toBeDefined();
    expect(screen.getByTestId("memory-matrix-cell-m3_ultra_solo-mxfp4"))
      .toBeDefined();

    // Per-cell fits chip
    const h100bf16 = screen.getByTestId("memory-matrix-cell-fits-h100_8x-bf16");
    expect(h100bf16.getAttribute("data-fits")).toBe("true");
    expect(h100bf16.textContent).toBe("fits");

    const m3bf16 = screen.getByTestId(
      "memory-matrix-cell-fits-m3_ultra_solo-bf16");
    expect(m3bf16.getAttribute("data-fits")).toBe("false");
    expect(m3bf16.textContent).toBe("over");
  });

  it("refetches when specPayload changes", async () => {
    const { rpc, calls } = makeFakeRpc({ "memory.matrix": MATRIX_OK });
    const { rerender } = render(
      <MemoryMatrixTab rpc={rpc} specPayload={{ a: 1 }} />,
    );
    await waitFor(() => { expect(calls.length).toBe(1); });
    rerender(<MemoryMatrixTab rpc={rpc} specPayload={{ a: 2 }} />);
    await waitFor(() => { expect(calls.length).toBe(2); });
  });

  it("renders an error banner if memory.matrix throws", async () => {
    const { rpc } = makeFakeRpc({
      "memory.matrix": new Error("backend boom"),
    });
    render(
      <MemoryMatrixTab rpc={rpc} specPayload={{}} />,
    );
    await waitFor(() => {
      expect(screen.getByTestId("memory-matrix-error").textContent ?? "")
        .toContain("backend boom");
    });
  });
});
