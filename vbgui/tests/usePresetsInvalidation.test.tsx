// V7-H47: usePresets refetches when invalidationKey changes.

import { describe, it, expect, vi, beforeEach } from "vitest";
import { renderHook, waitFor } from "@testing-library/react";
import { usePresets, _clearPresetsCache } from "@/hooks/usePresets";
import type { RpcClient } from "@/lib/rpc";

function makeRpc(values: string[]): RpcClient {
  return { call: vi.fn(async () => ({ presets: values } as never)) } as
    unknown as RpcClient;
}

describe("V7-H47 usePresets invalidation", () => {
  beforeEach(() => { _clearPresetsCache(); });

  it("fetches once on mount", async () => {
    const rpc = makeRpc(["a", "b"]);
    const { result } = renderHook(
      () => usePresets(rpc, null));
    await waitFor(() => expect(result.current).toEqual(["a", "b"]));
    expect(rpc.call).toHaveBeenCalledTimes(1);
  });

  it("refetches when invalidationKey changes", async () => {
    const rpc = makeRpc(["a"]);
    const { result, rerender } = renderHook(
      ({ key }: { key: string | null }) => usePresets(rpc, key),
      { initialProps: { key: "bid-1" } });
    await waitFor(() => expect(result.current).toEqual(["a"]));
    expect(rpc.call).toHaveBeenCalledTimes(1);

    // Swap the call to return a new list under the new key.
    (rpc.call as unknown as
      ReturnType<typeof vi.fn>).mockResolvedValueOnce({ presets: ["a", "c"] });
    rerender({ key: "bid-2" });
    await waitFor(() => expect(result.current).toEqual(["a", "c"]));
    expect(rpc.call).toHaveBeenCalledTimes(2);
  });

  it("does not refetch when invalidationKey is unchanged", async () => {
    const rpc = makeRpc(["a"]);
    const { result, rerender } = renderHook(
      ({ key }: { key: string | null }) => usePresets(rpc, key),
      { initialProps: { key: "bid-1" } });
    await waitFor(() => expect(result.current).toEqual(["a"]));
    rerender({ key: "bid-1" });
    expect(rpc.call).toHaveBeenCalledTimes(1);
  });
});
