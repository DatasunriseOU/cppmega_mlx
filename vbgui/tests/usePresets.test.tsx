import { describe, it, expect, vi, beforeEach } from "vitest";
import { renderHook, waitFor } from "@testing-library/react";
import { usePresets, _clearPresetsCache } from "@/hooks/usePresets";
import type { RpcClient } from "@/lib/rpc";

function fakeRpc(payload: { presets: string[] } | Error): RpcClient {
  return {
    call: vi.fn(async () => {
      if (payload instanceof Error) throw payload;
      return payload;
    }),
  } as unknown as RpcClient;
}

beforeEach(() => _clearPresetsCache());

describe("usePresets", () => {
  it("returns fallback list immediately on mount", () => {
    const rpc = fakeRpc({ presets: ["a", "b"] });
    const { result } = renderHook(() => usePresets(rpc));
    expect(result.current.length).toBeGreaterThan(50);
    expect(result.current).toContain("llama3_8b");
  });

  it("replaces fallback with live RPC list", async () => {
    const rpc = fakeRpc({ presets: ["alpha", "beta", "gamma"] });
    const { result } = renderHook(() => usePresets(rpc));
    await waitFor(() => {
      expect(result.current).toEqual(["alpha", "beta", "gamma"]);
    });
    expect(rpc.call).toHaveBeenCalledWith(
      "architectures.list_presets", {});
  });

  it("keeps fallback when RPC fails", async () => {
    const rpc = fakeRpc(new Error("backend offline"));
    const { result } = renderHook(() => usePresets(rpc));
    // Wait a tick for the rejected promise
    await new Promise((r) => setTimeout(r, 10));
    expect(result.current.length).toBeGreaterThan(50);
  });

  it("module-level cache short-circuits second fetch", async () => {
    const rpc = fakeRpc({ presets: ["a", "b"] });
    const { result, unmount } = renderHook(() => usePresets(rpc));
    await waitFor(() => expect(result.current).toEqual(["a", "b"]));
    unmount();

    const rpc2 = fakeRpc({ presets: ["c", "d"] });
    const { result: r2 } = renderHook(() => usePresets(rpc2));
    expect(r2.current).toEqual(["a", "b"]);  // hit cache
    expect(rpc2.call).not.toHaveBeenCalled();
  });
});
