import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, fireEvent, waitFor, act } from "@testing-library/react";
import { Tooltip } from "@/components/Tooltip";
import { _clearCatalogMemo } from "@/hooks/useCatalog";
import type { RpcClient } from "@/lib/rpc";

function fakeRpc(payload: unknown): RpcClient {
  return {
    call: vi.fn(async (_method: string, _params: unknown) => payload),
  } as unknown as RpcClient;
}

const LION_PAYLOAD = {
  entry: {
    category: "optimizer",
    name: "lion",
    summary: "Sign-based momentum, 50% less state than AdamW.",
    when_to_use: "Memory-constrained training. Works 100M-7B.",
    when_to_avoid: "Small batch (<64).",
    recommended_params: { lr: 1e-4 },
    paper_ref: "Chen 2023",
    paper_url: "https://arxiv.org/abs/2302.06675",
    gotchas: ["lr > 5e-4 → NaN"],
  },
  not_found_message: null,
};

describe("Tooltip", () => {
  beforeEach(() => {
    _clearCatalogMemo();
    vi.useFakeTimers();
  });
  afterEach(() => { vi.useRealTimers(); });

  it("does not show popup before delay elapses", () => {
    const rpc = fakeRpc(LION_PAYLOAD);
    render(<Tooltip rpc={rpc} category="optimizer" name="lion">
             <span>Kind</span>
           </Tooltip>);
    fireEvent.mouseEnter(screen.getByTestId("tooltip-optimizer-lion"));
    expect(screen.queryByTestId("tooltip-popup-optimizer-lion")).toBeNull();
  });

  it("shows popup after delay + fetches catalog", async () => {
    // Use real timers for this test — fake timers deadlock when the
    // RPC promise needs the microtask queue to flush.
    vi.useRealTimers();
    const rpc = fakeRpc(LION_PAYLOAD);
    render(<Tooltip rpc={rpc} category="optimizer" name="lion"
                    delayMs={20}>
             <span>Kind</span>
           </Tooltip>);
    fireEvent.mouseEnter(screen.getByTestId("tooltip-optimizer-lion"));
    // Wait for both the popup to appear AND the async fetch to resolve.
    await waitFor(() =>
      expect(screen.getByText(/Sign-based momentum/)).toBeTruthy()
    );
    expect(rpc.call).toHaveBeenCalledWith("catalog.explain",
                                          { category: "optimizer", name: "lion" });
  });

  it("hides popup on mouse leave + cancels pending fetch", async () => {
    const rpc = fakeRpc(LION_PAYLOAD);
    render(<Tooltip rpc={rpc} category="optimizer" name="lion"
                    delayMs={100}>
             <span>Kind</span>
           </Tooltip>);
    const root = screen.getByTestId("tooltip-optimizer-lion");
    fireEvent.mouseEnter(root);
    fireEvent.mouseLeave(root);
    await act(async () => { vi.advanceTimersByTime(200); });
    expect(screen.queryByTestId("tooltip-popup-optimizer-lion")).toBeNull();
  });

  it("renders info icon when onInfoClick provided + fires callback", () => {
    const rpc = fakeRpc(LION_PAYLOAD);
    const onInfoClick = vi.fn();
    render(<Tooltip rpc={rpc} category="optimizer" name="lion"
                    onInfoClick={onInfoClick}>
             <span>Kind</span>
           </Tooltip>);
    fireEvent.click(screen.getByTestId("tooltip-info-optimizer-lion"));
    expect(onInfoClick).toHaveBeenCalledTimes(1);
  });

  it("does not render info icon when callback omitted", () => {
    const rpc = fakeRpc(LION_PAYLOAD);
    render(<Tooltip rpc={rpc} category="optimizer" name="lion">
             <span>Kind</span>
           </Tooltip>);
    expect(screen.queryByTestId("tooltip-info-optimizer-lion")).toBeNull();
  });
});
