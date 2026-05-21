import { describe, it, expect, vi, beforeEach } from "vitest";
import { renderHook, waitFor } from "@testing-library/react";
import { useRpc } from "@/hooks/useRpc";

class FakeWebSocket {
  static OPEN = 1;
  static CLOSED = 3;
  static instances: FakeWebSocket[] = [];
  url: string;
  readyState = 0;
  onopen: ((e: unknown) => void) | null = null;
  onclose: ((e: unknown) => void) | null = null;
  onerror: ((e: unknown) => void) | null = null;
  onmessage: ((e: { data: string }) => void) | null = null;
  constructor(url: string) {
    this.url = url;
    FakeWebSocket.instances.push(this);
  }
  close() { this.readyState = FakeWebSocket.CLOSED; this.onclose?.({}); }
  fakeOpen() { this.readyState = FakeWebSocket.OPEN; this.onopen?.({}); }
  fakeStatusFrame() {
    this.onmessage?.({ data: JSON.stringify({
      jsonrpc: "2.0", id: null,
      method: "backend.status",
      params: { status: "ok" },
    }) });
  }
}

beforeEach(() => {
  FakeWebSocket.instances.length = 0;
  (globalThis as unknown as { WebSocket: typeof FakeWebSocket }).WebSocket =
    FakeWebSocket;
});

describe("useRpc", () => {
  it("returns a stable RpcClient instance across re-renders", () => {
    const { result, rerender } = renderHook(() => useRpc());
    const first = result.current;
    rerender();
    expect(result.current).toBe(first);
  });

  it("does not open a WebSocket when enableWs is false", () => {
    renderHook(() => useRpc({ enableWs: false }));
    expect(FakeWebSocket.instances).toHaveLength(0);
  });

  it("opens a WebSocket and reports connected on open", async () => {
    const onBackendStatus = vi.fn();
    renderHook(() => useRpc({ enableWs: true, onBackendStatus }));
    await waitFor(() => expect(FakeWebSocket.instances.length).toBe(1));
    expect(onBackendStatus).toHaveBeenCalledWith("reconnecting");
    FakeWebSocket.instances[0].fakeOpen();
    expect(onBackendStatus).toHaveBeenCalledWith("connected");
  });

  it("forwards backend.status frames as connected", async () => {
    const onBackendStatus = vi.fn();
    renderHook(() => useRpc({ enableWs: true, onBackendStatus }));
    await waitFor(() => expect(FakeWebSocket.instances.length).toBe(1));
    FakeWebSocket.instances[0].fakeOpen();
    FakeWebSocket.instances[0].fakeStatusFrame();
    expect(onBackendStatus.mock.calls.at(-1)?.[0]).toBe("connected");
  });

  it("translates http baseUrl to ws://...", async () => {
    renderHook(() => useRpc({ enableWs: true,
                              baseUrl: "http://host:9000",
                              onBackendStatus: () => {} }));
    await waitFor(() => expect(FakeWebSocket.instances.length).toBe(1));
    expect(FakeWebSocket.instances[0].url).toBe("ws://host:9000/ws");
  });

  it("closes WebSocket on unmount", async () => {
    const { unmount } = renderHook(() =>
      useRpc({ enableWs: true, onBackendStatus: () => {} }));
    await waitFor(() => expect(FakeWebSocket.instances.length).toBe(1));
    unmount();
    expect(FakeWebSocket.instances[0].readyState).toBe(FakeWebSocket.CLOSED);
  });
});
