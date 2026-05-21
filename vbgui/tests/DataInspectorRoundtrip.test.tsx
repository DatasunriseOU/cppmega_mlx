import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { DataInspector } from "@/components/DataInspector";
import type { RpcClient } from "@/lib/rpc";

const PREVIEW = {
  rows: [
    { row_index: 0, tokens: ["a", "b"], channels: {} },
    { row_index: 1, tokens: ["c", "d"], channels: {} },
  ],
  token_column: "input_ids",
  available_channels: [],
  bytes_per_token_avg: 2.5,
  bytes_per_token_p95: 4,
  bytes_per_token_max: 8,
  total_rows: 2,
};

const ROUNDTRIP = {
  rows: [
    { row_idx: 0, matches: true, byte_diff: 0,
      decoded_preview: "ab", original_bytes: 2, decoded_bytes: 2 },
    { row_idx: 1, matches: false, byte_diff: 3,
      decoded_preview: "xy", original_bytes: 4, decoded_bytes: 2 },
  ],
  pass_rate: 0.5,
  tokenizer_capability: "exact",
  has_original_text: true,
};

function fakeRpc(): RpcClient {
  return {
    call: vi.fn(async (method: string) => {
      if (method === "data.preview_parquet") return PREVIEW;
      if (method === "data.roundtrip_check") return ROUNDTRIP;
      throw new Error("unexpected " + method);
    }),
  } as unknown as RpcClient;
}

describe("DataInspector roundtrip badge (E7-3-UI)", () => {
  it("tokenizer input + Check roundtrip button render", () => {
    render(<DataInspector rpc={fakeRpc()} />);
    expect(screen.getByTestId("data-tokenizer-path")).toBeTruthy();
    const btn = screen.getByTestId("data-roundtrip") as HTMLButtonElement;
    expect(btn.disabled).toBe(true);  // disabled until both paths filled
  });

  it("Check roundtrip button enables when both paths filled", async () => {
    render(<DataInspector rpc={fakeRpc()} />);
    fireEvent.change(screen.getByTestId("data-path"),
                     { target: { value: "/p.parquet" } });
    fireEvent.change(screen.getByTestId("data-tokenizer-path"),
                     { target: { value: "/tok.json" } });
    const btn = screen.getByTestId("data-roundtrip") as HTMLButtonElement;
    expect(btn.disabled).toBe(false);
  });

  it("Check roundtrip → per-row OK/FAIL badges appear", async () => {
    render(<DataInspector rpc={fakeRpc()} />);
    fireEvent.change(screen.getByTestId("data-path"),
                     { target: { value: "/p.parquet" } });
    fireEvent.click(screen.getByTestId("data-load"));
    await waitFor(() => screen.getByTestId("data-row-0"));
    fireEvent.change(screen.getByTestId("data-tokenizer-path"),
                     { target: { value: "/tok.json" } });
    fireEvent.click(screen.getByTestId("data-roundtrip"));
    await waitFor(() => screen.getByTestId("data-roundtrip-0"));
    expect(screen.getByTestId("data-roundtrip-0").textContent)
      .toContain("OK");
    expect(screen.getByTestId("data-roundtrip-1").textContent)
      .toContain("FAIL");
  });
});
