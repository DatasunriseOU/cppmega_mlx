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

  // V4-3: Use-for-train button
  it("tokenizer-use-for-train-{i} hidden when onUseForTrain absent", () => {
    render(<TokenizerPlayground rpc={mockClient(SAMPLE_RESULT)}
                                initialSources={["/x.json"]} />);
    expect(screen.queryByTestId("tokenizer-use-for-train-0")).toBeNull();
  });

  it("tokenizer-use-for-train-{i} disabled until source is set", () => {
    const onUseForTrain = vi.fn();
    render(<TokenizerPlayground rpc={mockClient(SAMPLE_RESULT)}
                                initialSources={[""]}
                                onUseForTrain={onUseForTrain} />);
    expect(screen.getByTestId("tokenizer-use-for-train-0")
      .hasAttribute("disabled")).toBe(true);
  });

  it("tokenizer-use-for-train-{i} fires onUseForTrain(source)", () => {
    const onUseForTrain = vi.fn();
    render(<TokenizerPlayground rpc={mockClient(SAMPLE_RESULT)}
                                initialSources={["/my/tok.json"]}
                                onUseForTrain={onUseForTrain} />);
    fireEvent.click(screen.getByTestId("tokenizer-use-for-train-0"));
    expect(onUseForTrain).toHaveBeenCalledWith("/my/tok.json");
  });

  it("active panel shows ✓ Train when trainTokenizerPath matches", () => {
    render(<TokenizerPlayground rpc={mockClient(SAMPLE_RESULT)}
                                initialSources={["/active.json", "/other.json"]}
                                onUseForTrain={() => {}}
                                trainTokenizerPath="/active.json" />);
    expect(screen.getByTestId("tokenizer-use-for-train-0").textContent)
      .toContain("✓ Train");
    expect(screen.getByTestId("tokenizer-use-for-train-1").textContent)
      .not.toContain("✓ Train");
  });

  it("V7-H46: roundtrip pill renders ok when byte_roundtrip=true", async () => {
    const withRoundtripOk = {
      ...SAMPLE_RESULT,
      capabilities: { ...SAMPLE_RESULT.capabilities,
                       byte_roundtrip: true },
    };
    render(<TokenizerPlayground rpc={mockClient(withRoundtripOk)}
                                initialSources={["/ok.json"]} />);
    fireEvent.click(screen.getByTestId("tokenizer-encode-0"));
    await waitFor(() => {
      expect(screen.getByTestId("tokenizer-roundtrip-0")).toBeTruthy();
    });
    const pill = screen.getByTestId("tokenizer-roundtrip-0");
    expect(pill.getAttribute("data-roundtrip")).toBe("ok");
    expect(pill.textContent).toContain("✓");
  });

  it("V7-H46: roundtrip pill renders fail when byte_roundtrip=false", async () => {
    const withRoundtripFail = {
      ...SAMPLE_RESULT,
      capabilities: { ...SAMPLE_RESULT.capabilities,
                       byte_roundtrip: false },
    };
    render(<TokenizerPlayground rpc={mockClient(withRoundtripFail)}
                                initialSources={["/bad.json"]} />);
    fireEvent.click(screen.getByTestId("tokenizer-encode-0"));
    await waitFor(() => {
      const pill = screen.getByTestId("tokenizer-roundtrip-0");
      expect(pill.getAttribute("data-roundtrip")).toBe("fail");
      expect(pill.textContent).toContain("✗");
    });
  });

  it("V7-H46: roundtrip pill hidden when backend omits byte_roundtrip",
  async () => {
    // SAMPLE_RESULT.capabilities has no byte_roundtrip → no pill.
    render(<TokenizerPlayground rpc={mockClient(SAMPLE_RESULT)}
                                initialSources={["/missing.json"]} />);
    fireEvent.click(screen.getByTestId("tokenizer-encode-0"));
    await waitFor(() => {
      expect(screen.getByTestId("tokenizer-metrics-0")).toBeTruthy();
    });
    expect(screen.queryByTestId("tokenizer-roundtrip-0")).toBeNull();
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
