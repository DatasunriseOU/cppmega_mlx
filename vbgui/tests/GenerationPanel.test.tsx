import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { GenerationPanel } from "@/components/GenerationPanel";

function fakeRpc(handler: (m: string, p: unknown) => Promise<unknown>) {
  return { call: async <T,>(m: string, p: unknown) =>
                  (await handler(m, p)) as T } as never;
}

describe("V7 GenerationPanel (gen.run UI)", () => {
  it("renders the panel + Run button", () => {
    render(<GenerationPanel rpc={null} />);
    expect(screen.getByTestId("generation-panel")).toBeDefined();
    expect(screen.getByTestId("gen-run")).toBeDefined();
  });

  it("Run disabled when rpc is null", () => {
    render(<GenerationPanel rpc={null} />);
    expect((screen.getByTestId("gen-run") as HTMLButtonElement).disabled)
      .toBe(true);
  });

  it("Run fires gen.run RPC with parsed prompt + sampler params", async () => {
    const callSpy = vi.fn(async (_method: string, _params: unknown) => ({
      tokens: [9, 8, 7], finish_reason: "length",
      elapsed_ms: 1.2, strategy: "top_k", smoke: true,
    }));
    const rpc = fakeRpc(callSpy);
    render(<GenerationPanel rpc={rpc} />);
    fireEvent.change(screen.getByTestId("gen-prompt-tokens"),
                     { target: { value: "5, 4, 3" } });
    fireEvent.change(screen.getByTestId("gen-strategy"),
                     { target: { value: "top_k" } });
    fireEvent.change(screen.getByTestId("gen-max-new-tokens"),
                     { target: { value: "8" } });
    fireEvent.click(screen.getByTestId("gen-run"));
    await waitFor(() => {
      expect(screen.getByTestId("gen-result")).toBeDefined();
    });
    expect(callSpy).toHaveBeenCalledTimes(1);
    const args = callSpy.mock.calls[0]![1] as Record<string, unknown>;
    expect(args.prompt_tokens).toEqual([5, 4, 3]);
    expect(args.strategy).toBe("top_k");
    expect(args.max_new_tokens).toBe(8);
  });

  it("renders one chip per returned token", async () => {
    const rpc = fakeRpc(async () => ({
      tokens: [1, 2, 3], finish_reason: "length",
      elapsed_ms: 0.5, strategy: "greedy", smoke: true,
    }));
    render(<GenerationPanel rpc={rpc} />);
    fireEvent.click(screen.getByTestId("gen-run"));
    await waitFor(() => {
      expect(screen.getByTestId("gen-token-0")).toBeDefined();
    });
    expect(screen.getByTestId("gen-token-0").textContent).toBe("1");
    expect(screen.getByTestId("gen-token-1").textContent).toBe("2");
    expect(screen.getByTestId("gen-token-2").textContent).toBe("3");
    expect(screen.getByTestId("gen-finish-reason").textContent)
      .toContain("length");
  });

  it("surfaces error on RPC reject", async () => {
    const rpc = fakeRpc(async () => { throw new Error("backend down"); });
    render(<GenerationPanel rpc={rpc} />);
    fireEvent.click(screen.getByTestId("gen-run"));
    await waitFor(() => {
      expect(screen.getByTestId("gen-error").textContent)
        .toContain("backend down");
    });
  });
});
