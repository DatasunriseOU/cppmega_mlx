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
  side_channel_families: {
    universal: {
      family: "universal",
      status: "derived",
      columns: ["input_ids", "doc_ids"],
      missing_columns: ["target_ids"],
      dropped_columns: [],
      token_alignment: "yes",
      graph_remapping: "not_applicable",
      provenance: "derived",
      non_null_ratio: 1.0,
    },
    structure: {
      family: "structure",
      status: "missing",
      columns: [],
      missing_columns: ["token_structure_ids"],
      dropped_columns: [],
      token_alignment: "unknown",
      graph_remapping: "not_applicable",
      provenance: "missing",
      non_null_ratio: 0.0,
    },
  },
  edge_distributions: {},
  shards: [],
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

  it("renders side-channel family coverage diagnostics", async () => {
    render(<DataInspector rpc={mockClient(SAMPLE)} initialPath="/x" />);
    fireEvent.click(screen.getByTestId("data-load"));
    await waitFor(() =>
      expect(screen.getByTestId("data-family-coverage")).toBeTruthy());
    expect(screen.getByTestId("data-family-universal-status").textContent)
      .toContain("derived");
    expect(screen.getByTestId("data-family-structure-missing").textContent)
      .toContain("token_structure_ids");
  });

  it("renders real clang edge id distributions", async () => {
    const edgeSample: PreviewParquetResult = {
      ...SAMPLE,
      available_channels: ["call_edges", "type_edges"],
      rows: [
        { row_index: 0, tokens: [1, 2, 3],
          channels: { call_edges: [{ from: 5, to: 54 }], type_edges: [] } },
      ],
      edge_distributions: {
        call_edges: {
          column: "call_edges",
          edge_count: 3,
          row_count: 1,
          non_empty_rows: 1,
          min_node_id: 5,
          max_node_id: 54,
          distinct_node_count: 4,
          per_row_min: 3,
          per_row_avg: 3,
          per_row_max: 3,
          synthetic_0_to_7_only: false,
          sample_edges: [{ from: 5, to: 54 }],
        },
      },
    };
    render(<DataInspector rpc={mockClient(edgeSample)} initialPath="/x" />);
    fireEvent.click(screen.getByTestId("data-load"));
    await waitFor(() =>
      expect(screen.getByTestId("data-edge-distribution-call_edges")).toBeTruthy());
    const panel = screen.getByTestId("data-edge-distribution-call_edges");
    expect(panel.textContent).toContain("call_edges");
    expect(panel.textContent).toContain("max 54");
    expect(panel.textContent).toContain("real");
  });

  it("renders ordered shard list and passes all shards for training", async () => {
    const onUseForTrain = vi.fn();
    const shardSample: PreviewParquetResult = {
      ...SAMPLE,
      shards: [
        { index: 0, path: "/corpus/val_00000.parquet", byte_size: 128, row_count: 2 },
        { index: 1, path: "/corpus/val_00001.parquet", byte_size: 256, row_count: 3 },
      ],
    };
    render(<DataInspector rpc={mockClient(shardSample)}
                          initialPath="/corpus/val_00000.parquet"
                          onUseForTrain={onUseForTrain} />);
    fireEvent.click(screen.getByTestId("data-load"));
    await waitFor(() =>
      expect(screen.getByTestId("data-shards")).toBeTruthy());
    expect(screen.getByTestId("data-shard-0").textContent)
      .toContain("val_00000.parquet");
    expect(screen.getByTestId("data-shard-1").textContent)
      .toContain("3 rows");

    fireEvent.click(screen.getByTestId("data-use-for-train"));
    expect(onUseForTrain).toHaveBeenCalledWith(
      "/corpus/val_00000.parquet",
      null,
      ["/corpus/val_00000.parquet", "/corpus/val_00001.parquet"],
    );
  });

  it("notifies parent when loaded channels change", async () => {
    const onAvailableChannelsChange = vi.fn();
    render(<DataInspector rpc={mockClient(SAMPLE)} initialPath="/x"
                          onAvailableChannelsChange={onAvailableChannelsChange} />);
    fireEvent.click(screen.getByTestId("data-load"));
    await waitFor(() =>
      expect(onAvailableChannelsChange).toHaveBeenCalledWith([
        "doc_ids", "loss_mask",
      ]));
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

  // V4-1: Use-for-train button
  it("data-use-for-train disabled until parquet loaded", () => {
    render(<DataInspector rpc={mockClient(SAMPLE)} initialPath="/x"
                          onUseForTrain={() => {}} />);
    expect(screen.getByTestId("data-use-for-train")
      .hasAttribute("disabled")).toBe(true);
  });

  it("data-use-for-train fires onUseForTrain(path, null) after load",
    async () => {
      const onUseForTrain = vi.fn();
      render(<DataInspector rpc={mockClient(SAMPLE)}
                            initialPath="/tmp/sh.parquet"
                            onUseForTrain={onUseForTrain} />);
      fireEvent.click(screen.getByTestId("data-load"));
      await waitFor(() => {
        expect(screen.getByTestId("data-metrics")).toBeTruthy();
      });
      fireEvent.click(screen.getByTestId("data-use-for-train"));
      expect(onUseForTrain).toHaveBeenCalledWith("/tmp/sh.parquet", null);
    });

  it("data-use-for-train fires onUseForTrain(path, tokenizer) when both set",
    async () => {
      const onUseForTrain = vi.fn();
      render(<DataInspector rpc={mockClient(SAMPLE)}
                            initialPath="/p.parquet"
                            onUseForTrain={onUseForTrain} />);
      fireEvent.click(screen.getByTestId("data-load"));
      await waitFor(() =>
        expect(screen.getByTestId("data-metrics")).toBeTruthy());
      fireEvent.change(screen.getByTestId("data-tokenizer-path"),
        { target: { value: "/t.json" } });
      fireEvent.click(screen.getByTestId("data-use-for-train"));
      expect(onUseForTrain).toHaveBeenCalledWith("/p.parquet", "/t.json");
    });

  it("data-use-for-train shows ✓ Training when this path is active", async () => {
    render(<DataInspector rpc={mockClient(SAMPLE)} initialPath="/p"
                          trainParquetPath="/p"
                          onUseForTrain={() => {}} />);
    fireEvent.click(screen.getByTestId("data-load"));
    await waitFor(() =>
      expect(screen.getByTestId("data-metrics")).toBeTruthy());
    expect(screen.getByTestId("data-use-for-train").textContent)
      .toContain("Training");
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
