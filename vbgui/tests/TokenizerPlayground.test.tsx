import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { TokenizerPlayground } from "@/components/TokenizerPlayground";
import { RpcClient } from "@/lib/rpc";

function mockClient(response: object) {
  const fetchImpl = vi.fn(async () =>
    new Response(JSON.stringify({ jsonrpc: "2.0", id: 1, result: response }),
                 { status: 200,
                   headers: { "content-type": "application/json" } }),
  );
  return new RpcClient({ baseUrl: "http://test", fetchImpl: fetchImpl as never });
}

const SAMPLE_RESULT = {
  tokens: [
    { id: 1, text: "hi", start: 0, end: 2, is_special: false },
    { id: 2, text: "!",  start: 2, end: 3, is_special: false },
  ],
  token_count: 2,
  bytes_total: 3,
  bytes_per_token_avg: 1.5,
  bytes_per_token_max: 2,
  capabilities: { vocab_size: 65536, has_fim: true, has_space_nl: true,
                  decoder_kind: "custom" as const },
  elapsed_ms: 0.5,
};

describe("TokenizerPlayground", () => {
  it("renders an input + add-panel button", () => {
    render(<TokenizerPlayground rpc={mockClient(SAMPLE_RESULT)} />);
    expect(screen.getByTestId("tokenizer-input")).toBeTruthy();
    expect(screen.getByTestId("add-panel")).toBeTruthy();
  });

  it("renders initialSources as panels", () => {
    render(<TokenizerPlayground rpc={mockClient(SAMPLE_RESULT)}
                                initialSources={["a", "b"]} />);
    expect(screen.getByTestId("tokenizer-panel-0")).toBeTruthy();
    expect(screen.getByTestId("tokenizer-panel-1")).toBeTruthy();
  });

  it("Add tokenizer adds panels up to maxPanels", () => {
    render(<TokenizerPlayground rpc={mockClient(SAMPLE_RESULT)}
                                maxPanels={2} />);
    fireEvent.click(screen.getByTestId("add-panel"));
    fireEvent.click(screen.getByTestId("add-panel"));
    fireEvent.click(screen.getByTestId("add-panel"));   // capped
    expect(screen.getByTestId("tokenizer-panel-1")).toBeTruthy();
    expect(screen.queryByTestId("tokenizer-panel-2")).toBeNull();
  });

  it("Encode populates chips when backend responds", async () => {
    render(<TokenizerPlayground rpc={mockClient(SAMPLE_RESULT)}
                                initialSources={["x"]} />);
    fireEvent.click(screen.getByTestId("tokenizer-encode-0"));
    await waitFor(() => {
      expect(screen.getByTestId("tokenizer-chips-0")).toBeTruthy();
    });
    expect(screen.getByTestId("tokenizer-chip-0-0").textContent).toBe("hi");
    expect(screen.getByTestId("tokenizer-chip-0-1").textContent).toBe("!");
    expect(screen.getByTestId("tokenizer-metrics-0").textContent)
      .toContain("2 tokens");
  });

  it("Remove drops a panel", () => {
    render(<TokenizerPlayground rpc={mockClient(SAMPLE_RESULT)}
                                initialSources={["a", "b"]} />);
    fireEvent.click(screen.getByTestId("tokenizer-remove-0"));
    expect(screen.queryByTestId("tokenizer-panel-1")).toBeNull();
    expect(screen.getByTestId("tokenizer-panel-0")).toBeTruthy();
  });

  it("Renders error envelope when backend fails", async () => {
    const failing = new RpcClient({
      baseUrl: "http://x",
      fetchImpl: vi.fn(async () =>
        new Response(JSON.stringify({
          jsonrpc: "2.0", id: 1,
          error: { code: -32603, message: "boom" },
        }), { status: 200,
              headers: { "content-type": "application/json" } })) as never,
    });
    render(<TokenizerPlayground rpc={failing} initialSources={["x"]} />);
    fireEvent.click(screen.getByTestId("tokenizer-encode-0"));
    await waitFor(() => {
      expect(screen.getByTestId("tokenizer-error-0")).toBeTruthy();
    });
  });
});
