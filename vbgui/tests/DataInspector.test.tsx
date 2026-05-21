import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { DataInspector, type PreviewParquetResult } from "@/components/DataInspector";
import { RpcClient } from "@/lib/rpc";

function mockClient(response: object) {
  const fetchImpl = vi.fn(async () =>
    new Response(JSON.stringify({ jsonrpc: "2.0", id: 1, result: response }),
                 { status: 200,
                   headers: { "content-type": "application/json" } }),
  );
  return new RpcClient({ baseUrl: "http://test", fetchImpl: fetchImpl as never });
}

const SAMPLE: PreviewParquetResult = {
  rows: [
    { row_index: 0, tokens: [1, 2, 3],
      channels: { doc_ids: 0, loss_mask: [1, 1, 0] } },
    { row_index: 1, tokens: [4, 5],
      channels: { doc_ids: 1, loss_mask: [1, 1] } },
  ],
  token_column: "input_ids",
  available_channels: ["doc_ids", "loss_mask"],
  bytes_per_token_avg: 1.4,
  bytes_per_token_p95: 2.0,
  bytes_per_token_max: 2,
  total_rows: 50,
  elapsed_ms: 1.2,
};

describe("DataInspector", () => {
  it("renders path + load button", () => {
    render(<DataInspector rpc={mockClient(SAMPLE)} />);
    expect(screen.getByTestId("data-path")).toBeTruthy();
    expect(screen.getByTestId("data-load")).toBeTruthy();
  });

  it("Load populates rows + channel toggles + metrics", async () => {
    render(<DataInspector rpc={mockClient(SAMPLE)}
                          initialPath="/tmp/sh.parquet" />);
    fireEvent.click(screen.getByTestId("data-load"));
    await waitFor(() => {
      expect(screen.getByTestId("data-metrics")).toBeTruthy();
    });
    expect(screen.getByTestId("data-metrics").textContent)
      .toContain("50 rows");
    expect(screen.getByTestId("data-channel-toggle-doc_ids")).toBeTruthy();
    expect(screen.getByTestId("data-channel-toggle-loss_mask")).toBeTruthy();
    expect(screen.getByTestId("data-row-0")).toBeTruthy();
    expect(screen.getByTestId("data-row-1")).toBeTruthy();
  });

  it("toggling a channel hides its ribbon", async () => {
    render(<DataInspector rpc={mockClient(SAMPLE)} initialPath="/x" />);
    fireEvent.click(screen.getByTestId("data-load"));
    await waitFor(() =>
      screen.getByTestId("data-ribbon-0-doc_ids"));
    fireEvent.click(
      screen.getByTestId("data-channel-toggle-doc_ids")
            .querySelector("input")!);
    expect(screen.queryByTestId("data-ribbon-0-doc_ids")).toBeNull();
  });

  it("renders array channel as per-token strip", async () => {
    render(<DataInspector rpc={mockClient(SAMPLE)} initialPath="/x" />);
    fireEvent.click(screen.getByTestId("data-load"));
    await waitFor(() =>
      screen.getByTestId("data-ribbon-0-loss_mask"));
    const ribbon = screen.getByTestId("data-ribbon-0-loss_mask");
    expect(ribbon.textContent).toContain("1");
    expect(ribbon.textContent).toContain("0");
  });

  it("pagination Prev/Next call backend with new offsets", async () => {
    const fetchImpl = vi.fn(async () =>
      new Response(JSON.stringify({ jsonrpc: "2.0", id: 1, result: SAMPLE }),
                   { status: 200,
                     headers: { "content-type": "application/json" } }),
    );
    const client = new RpcClient({
      baseUrl: "http://x", fetchImpl: fetchImpl as never,
    });
    render(<DataInspector rpc={client} initialPath="/x" pageSize={16} />);
    fireEvent.click(screen.getByTestId("data-load"));
    await waitFor(() => screen.getByTestId("data-pagination"));
    fireEvent.click(screen.getByTestId("data-next"));
    await waitFor(() => {
      const last = fetchImpl.mock.calls.at(-1) as unknown as [string, RequestInit];
      const body = JSON.parse(last[1].body as string);
      expect(body.params.offset).toBe(16);
    });
  });

  it("renders error envelope when backend fails", async () => {
    const failing = new RpcClient({
      baseUrl: "http://x",
      fetchImpl: vi.fn(async () =>
        new Response(JSON.stringify({
          jsonrpc: "2.0", id: 1,
          error: { code: -32603, message: "boom" },
        }), { status: 200,
              headers: { "content-type": "application/json" } })) as never,
    });
    render(<DataInspector rpc={failing} initialPath="/x" />);
    fireEvent.click(screen.getByTestId("data-load"));
    await waitFor(() => {
      expect(screen.getByTestId("data-error")).toBeTruthy();
    });
  });
});
