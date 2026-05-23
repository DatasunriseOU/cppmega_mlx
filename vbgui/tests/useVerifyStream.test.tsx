// V7-H37: canonicalJson matches Python json.dumps(sort_keys=True);
// useVerifyStream collects WS events + closes on finish frame.

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { renderHook, act, waitFor } from "@testing-library/react";
import { canonicalJson, useVerifyStream }
  from "@/hooks/useVerifyStream";

describe("V7-H37 canonicalJson", () => {
  it("sorts object keys deterministically (matches Python sort_keys)", () => {
    const out = canonicalJson({ b: 2, a: { d: 4, c: 3 } });
    expect(out).toBe('{"a":{"c":3,"d":4},"b":2}');
  });

  it("emits arrays in order without key-sort effect", () => {
    expect(canonicalJson([3, 1, 2])).toBe("[3,1,2]");
  });

  it("nests deeply with stable order", () => {
    expect(canonicalJson({ z: 1, a: [{ y: 1, x: 2 }] }))
      .toBe('{"a":[{"x":2,"y":1}],"z":1}');
  });

  it("handles primitives + null", () => {
    expect(canonicalJson("hi")).toBe('"hi"');
    expect(canonicalJson(42)).toBe("42");
    expect(canonicalJson(null)).toBe("null");
  });
});

// useVerifyStream WS lifecycle — driven by an in-test WS shim that
// captures the constructed URL + lets the test push synthetic frames.

class MockSocket {
  static instances: MockSocket[] = [];
  static reset(): void { MockSocket.instances = []; }
  url: string;
  onmessage: ((e: MessageEvent) => void) | null = null;
  onerror: ((e: Event) => void) | null = null;
  closed = false;
  constructor(url: string) {
    this.url = url;
    MockSocket.instances.push(this);
  }
  close(): void { this.closed = true; }
  push(frame: unknown): void {
    this.onmessage?.(new MessageEvent("message",
      { data: JSON.stringify(frame) }));
  }
}

describe("V7-H37 useVerifyStream", () => {
  beforeEach(() => {
    MockSocket.reset();
    (globalThis as unknown as { WebSocket: typeof WebSocket })
      .WebSocket = MockSocket as unknown as typeof WebSocket;
  });
  afterEach(() => {
    (globalThis as unknown as { WebSocket: typeof WebSocket | null })
      .WebSocket = null as unknown as typeof WebSocket;
  });

  it("opens /ws/verify/{hash} URL with backend base", async () => {
    renderHook(() => useVerifyStream(
      "http://127.0.0.1:8765", "abc123", true));
    await waitFor(() => {
      expect(MockSocket.instances.length).toBe(1);
    });
    expect(MockSocket.instances[0]?.url)
      .toBe("ws://127.0.0.1:8765/ws/verify/abc123");
  });

  it("doesn't open socket while inactive", () => {
    renderHook(() => useVerifyStream("http://x", "abc", false));
    expect(MockSocket.instances).toEqual([]);
  });

  it("collects events and flips finished on finish frame", async () => {
    const { result } = renderHook(() => useVerifyStream(
      "http://x", "abc", true));
    await waitFor(() => {
      expect(MockSocket.instances.length).toBe(1);
    });
    const sock = MockSocket.instances[0]!;
    act(() => {
      sock.push({ event: { phase: "start" }, spec_hash: "abc" });
      sock.push({ event: { phase: "resolve_shapes" }, spec_hash: "abc" });
      sock.push({ finish: "ok", spec_hash: "abc" });
    });
    await waitFor(() => {
      expect(result.current.finished).toBe(true);
    });
    expect(result.current.events.map((e) => e.phase))
      .toEqual(["start", "resolve_shapes"]);
    expect(sock.closed).toBe(true);
  });
});
