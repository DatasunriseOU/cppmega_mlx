// V7-H46: TokenizerPlayground roundtrip-check button calls
// tokenizer.roundtrip_text RPC and renders badge.

import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent, waitFor }
  from "@testing-library/react";
import { TokenizerPlayground } from "@/components/TokenizerPlayground";
import type { RpcClient } from "@/lib/rpc";

function makeRpc(returns: Record<string, unknown>): RpcClient {
  return {
    call: vi.fn(async (method: string) => returns[method] as never),
  } as unknown as RpcClient;
}

const ENCODE = {
  tokens: [{ id: 1, text: "Hello", start: 0, end: 5, is_special: false }],
  token_count: 1, bytes_per_token_avg: 5.0, bytes_per_token_max: 5,
  capabilities: { vocab_size: 100, has_fim: false, byte_roundtrip: true },
  elapsed_ms: 1.0,
};

describe("V7-H46 TokenizerPlayground roundtrip-check", () => {
  it("Roundtrip-check button hidden before encode populates state.result",
     () => {
    const rpc = makeRpc({ "tokenizer.list_presets": { presets: [] } });
    render(<TokenizerPlayground rpc={rpc} initialSources={[""]} />);
    expect(screen.queryByTestId("tokenizer-roundtrip-run-0")).toBeNull();
  });

  it("calls tokenizer.roundtrip_text + renders badge with matches=yes",
     async () => {
    const rpc = makeRpc({
      "tokenizer.list_presets": { presets: [] },
      "tokenizer.encode_visualize": ENCODE,
      "tokenizer.roundtrip_text": {
        matches: true, decoded: "Hello, world!",
        original_bytes: 13, decoded_bytes: 13, byte_diff: 0,
        tokenizer_capability: "exact", elapsed_ms: 2.0,
      },
    });
    render(<TokenizerPlayground rpc={rpc}
            initialSources={["cppmega_mlx/tokenizer/tokenizer.json"]} />);
    // Need to trigger encode first to populate state.result.
    fireEvent.click(screen.getByTestId("tokenizer-encode-0"));
    await waitFor(() => screen.getByTestId("tokenizer-roundtrip-run-0"));
    fireEvent.click(screen.getByTestId("tokenizer-roundtrip-run-0"));
    await waitFor(() => {
      expect(rpc.call).toHaveBeenCalledWith(
        "tokenizer.roundtrip_text",
        expect.objectContaining({
          tokenizer_source: "cppmega_mlx/tokenizer/tokenizer.json",
        }));
    });
    const badge = await screen.findByTestId("tokenizer-roundtrip-badge-0");
    expect(badge.getAttribute("data-matches")).toBe("yes");
    expect(badge.getAttribute("data-capability")).toBe("exact");
    expect(badge.textContent).toContain("byte-exact");
  });

  it("renders mismatch + byte-diff when backend reports matches=false",
     async () => {
    const rpc = makeRpc({
      "tokenizer.list_presets": { presets: [] },
      "tokenizer.encode_visualize": ENCODE,
      "tokenizer.roundtrip_text": {
        matches: false, decoded: "Hel world",
        original_bytes: 13, decoded_bytes: 9, byte_diff: 4,
        tokenizer_capability: "approx", elapsed_ms: 2.0,
      },
    });
    render(<TokenizerPlayground rpc={rpc}
            initialSources={["nanochat/tokenizer.json"]} />);
    fireEvent.click(screen.getByTestId("tokenizer-encode-0"));
    await waitFor(() => screen.getByTestId("tokenizer-roundtrip-run-0"));
    fireEvent.click(screen.getByTestId("tokenizer-roundtrip-run-0"));
    const badge = await screen.findByTestId("tokenizer-roundtrip-badge-0");
    expect(badge.getAttribute("data-matches")).toBe("no");
    expect(badge.textContent).toContain("byte mismatch");
    expect(badge.textContent).toContain("Δ4B");
  });
});
