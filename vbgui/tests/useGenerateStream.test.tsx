// V7-H42: useGenerateStream opens /ws/gen/{job_id}, collects events,
// surfaces finish toast, reconnects on unexpected drop.

import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { renderHook, act, waitFor } from "@testing-library/react";
import { useGenerateStream } from "@/hooks/useGenerateStream";

class MockSocket {
  static instances: MockSocket[] = [];
  static reset(): void { MockSocket.instances = []; }
  url: string;
  onmessage: ((e: MessageEvent) => void) | null = null;
  onclose: ((e: CloseEvent) => void) | null = null;
  onerror: ((e: Event) => void) | null = null;
  closed = false;
  constructor(url: string) {
    this.url = url;
    MockSocket.instances.push(this);
  }
  close(): void {
    this.closed = true;
    this.onclose?.(new CloseEvent("close"));
  }
  push(frame: unknown): void {
    this.onmessage?.(new MessageEvent("message",
      { data: JSON.stringify(frame) }));
  }
  drop(): void {
    // Simulate unexpected disconnect (no finish frame).
    this.onclose?.(new CloseEvent("close"));
  }
}

describe("V7-H42 useGenerateStream", () => {
  beforeEach(() => {
    MockSocket.reset();
    (globalThis as unknown as { WebSocket: typeof WebSocket })
      .WebSocket = MockSocket as unknown as typeof WebSocket;
  });
  afterEach(() => {
    (globalThis as unknown as { WebSocket: typeof WebSocket | null })
      .WebSocket = null as unknown as typeof WebSocket;
  });

  it("opens /ws/gen/{job_id} URL with backend base", async () => {
    renderHook(() => useGenerateStream(
      "http://127.0.0.1:8765", "job-1", true));
    await waitFor(() => {
      expect(MockSocket.instances.length).toBe(1);
    });
    expect(MockSocket.instances[0]?.url)
      .toBe("ws://127.0.0.1:8765/ws/gen/job-1");
  });

  it("does not open socket while inactive", () => {
    renderHook(() => useGenerateStream("http://x", "job-2", false));
    expect(MockSocket.instances).toEqual([]);
  });

  it("collects token events and flips finishToast on finish frame",
     async () => {
    const { result } = renderHook(() => useGenerateStream(
      "http://x", "job-3", true));
    await waitFor(() => {
      expect(MockSocket.instances.length).toBe(1);
    });
    const sock = MockSocket.instances[0]!;
    act(() => {
      sock.push({ event: { step: 0, token_id: 42 } });
      sock.push({ event: { step: 1, token_id: 99 } });
      sock.push({ finish: "ok" });
    });
    await waitFor(() => {
      expect(result.current.finishToast).toBe(true);
    });
    expect(result.current.events.map((e) => e.token_id))
      .toEqual([42, 99]);
    expect(sock.closed).toBe(true);
  });

  it("reset clears events + finishToast + reconnectAttempts",
     async () => {
    const { result } = renderHook(() => useGenerateStream(
      "http://x", "job-4", true));
    await waitFor(() => expect(MockSocket.instances.length).toBe(1));
    act(() => {
      MockSocket.instances[0]!.push({ event: { step: 0, token_id: 1 } });
      MockSocket.instances[0]!.push({ finish: "ok" });
    });
    await waitFor(() => expect(result.current.finishToast).toBe(true));
    act(() => result.current.reset());
    expect(result.current.events).toEqual([]);
    expect(result.current.finishToast).toBe(false);
    expect(result.current.reconnectAttempts).toBe(0);
  });

  it("dismissToast clears the finish toast only", async () => {
    const { result } = renderHook(() => useGenerateStream(
      "http://x", "job-5", true));
    await waitFor(() => expect(MockSocket.instances.length).toBe(1));
    act(() => {
      MockSocket.instances[0]!.push({ event: { step: 0, token_id: 7 } });
      MockSocket.instances[0]!.push({ finish: "ok" });
    });
    await waitFor(() => expect(result.current.finishToast).toBe(true));
    act(() => result.current.dismissToast());
    expect(result.current.finishToast).toBe(false);
    // Events survive.
    expect(result.current.events).toHaveLength(1);
  });
});
