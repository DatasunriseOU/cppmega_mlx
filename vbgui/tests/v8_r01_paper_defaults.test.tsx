/**
 * V8-R01 vitest: dropping a preset auto-fills the Optim tab from the
 * paper-anchored defaults block returned by `build_preset_specs`.
 *
 * AC (mirrors VisualBuilderPlan-v8 §R01): selecting `llama3_8b` makes
 * the first optim group's lr equal 3e-4 and its schedule.kind equal
 * "wsd" without any further user input.
 */

import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { App } from "@/App";

class FakeWebSocket {
  static OPEN = 1;
  static CLOSED = 3;
  url: string;
  readyState = 0;
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

describe("V8-R01: build_preset_specs defaults auto-fill", () => {
  it("llama3_8b paper-defaults -> lr 3e-4 + wsd schedule", async () => {
    const { fetchFn, calls } = recorder({
      build_preset_specs: {
        specs: [
          { kind: "attention", name: "a0", params: { num_heads: 2 } },
          { kind: "mlp", name: "a1", params: {} },
        ],
        preset_name: "llama3_8b",
        defaults: {
          lr: 3e-4,
          batch_size: 1024,
          schedule: "wsd",
          warmup_steps: 2000,
          betas: [0.9, 0.95],
          gradient_clip: 1.0,
          mixed_precision: true,
          optimizer: "adamw",
          source_paper_url: "https://arxiv.org/abs/2407.21783",
        },
      },
      verify: { memory_per_brick: {}, gotchas: [], elapsed_ms: 0.1,
                resolved: { edges: [] } },
      suggest_sharding: { proposals: [] },
    });
    (globalThis as unknown as { fetch: typeof fetch }).fetch = fetchFn as never;

    render(<App />);
    fireEvent.change(screen.getByTestId("preset-launcher"),
      { target: { value: "llama3_8b" } });

    await waitFor(() => {
      expect(calls.some((c) => c.method === "build_preset_specs"))
        .toBe(true);
    });

    // Switch to the Optim tab so its inputs are mounted in the DOM.
    fireEvent.click(screen.getByTestId("sidebar-tab-optim"));
    await waitFor(() => {
      expect(screen.getByTestId("optim-kind")).toBeDefined();
    });

    // After the defaults block lands, the optim kind should be adamw,
    // mixed_precision on, grad_clip == 1.0, and the first group's lr
    // matches the paper-anchored 3e-4.
    await waitFor(() => {
      const kind = screen.getByTestId("optim-kind") as HTMLSelectElement;
      expect(kind.value).toBe("adamw");
    });
    const lr = screen.getByTestId("optim-group-0-lr") as HTMLInputElement;
    expect(Number(lr.value)).toBeCloseTo(3e-4, 7);
    const clip = screen.getByTestId("optim-clip") as HTMLInputElement;
    expect(Number(clip.value)).toBe(1.0);
    const mp = screen.getByTestId("optim-mp") as HTMLInputElement;
    expect(mp.checked).toBe(true);

    // Open the first group's schedule editor and check kind=wsd.
    fireEvent.click(screen.getByTestId("optim-group-0-schedule-toggle"));
    await waitFor(() => {
      const sk = screen.getByTestId("schedule-kind-0") as HTMLSelectElement;
      expect(sk.value).toBe("wsd");
    });
  });
});
