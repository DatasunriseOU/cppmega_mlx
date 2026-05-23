/**
 * fhxg: clicking Apply in the LossTab fires a transient toast that
 * names the chosen loss kind and the head_outputs target.
 */

import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { App } from "@/App";

class FakeWebSocket {
  static OPEN = 1; static CLOSED = 3;
  url: string; readyState = 0;
  onopen: ((e: unknown) => void) | null = null;
  onclose: ((e: unknown) => void) | null = null;
  onerror: ((e: unknown) => void) | null = null;
  onmessage: ((e: { data: string }) => void) | null = null;
  constructor(url: string) { this.url = url; }
  close() { this.readyState = FakeWebSocket.CLOSED; this.onclose?.({}); }
}

function recorder(responses: Record<string, unknown>) {
  const calls: { method: string; params: unknown }[] = [];
  const fetchFn = vi.fn(async (_url: unknown, init: unknown) => {
    const body = JSON.parse((init as RequestInit).body as string);
    calls.push({ method: body.method, params: body.params });
    const result = responses[body.method] ?? {};
    return new Response(JSON.stringify({
      jsonrpc: "2.0", id: body.id, result,
    }), { status: 200,
          headers: { "content-type": "application/json" } });
  });
  return { fetchFn, calls };
}

beforeEach(() => {
  (globalThis as unknown as { WebSocket: typeof FakeWebSocket }).WebSocket =
    FakeWebSocket;
});

describe("fhxg: loss Apply toast", () => {
  it("clicking LossTab Apply shows a transient toast", async () => {
    const { fetchFn } = recorder({
      build_preset_specs: {
        specs: [
          { kind: "attention", name: "a0", params: {} },
          { kind: "mlp", name: "a1", params: {} },
        ],
        preset_name: "llama3_8b",
        defaults: { lr: 3e-4, batch_size: 1024, schedule: "wsd",
                    warmup_steps: 2000, betas: [0.9, 0.95],
                    gradient_clip: 1.0, mixed_precision: true,
                    optimizer: "adamw",
                    source_paper_url: "https://arxiv.org/abs/2407.21783" },
      },
      verify: { memory_per_brick: {}, gotchas: [], elapsed_ms: 0.1,
                resolved: { edges: [] } },
      suggest_sharding: { proposals: [] },
    });
    (globalThis as unknown as { fetch: typeof fetch }).fetch =
      fetchFn as never;

    render(<App />);
    fireEvent.change(screen.getByTestId("preset-launcher"),
      { target: { value: "llama3_8b" } });

    // Switch to Loss tab + click Apply.
    fireEvent.click(screen.getByTestId("sidebar-tab-loss"));
    await waitFor(() => {
      expect(screen.getByTestId("loss-apply")).toBeDefined();
    });
    fireEvent.click(screen.getByTestId("loss-apply"));

    // Toast renders with the loss kind + the head_outputs target name.
    await waitFor(() => {
      const toast = screen.getByTestId("loss-apply-toast");
      expect(toast.textContent ?? "").toMatch(/Loss applied/);
    });
  });
});
