/**
 * F-I integration tests — the App wires preset launcher and run-pipeline
 * end-to-end through fetch/WS. We stub fetch with a recorder and assert
 * the wire-shaped payloads + UI side effects.
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

describe("App integration — preset launcher", () => {
  it("calling preset launcher hits build_preset_specs and populates canvas", async () => {
    const { fetchFn, calls } = recorder({
      build_preset_specs: {
        specs: [
          { kind: "attention", name: "a0", params: { num_heads: 2 } },
          { kind: "mlp",       name: "a1", params: {} },
        ],
        preset_name: "llama3_8b",
      },
      verify: { memory_per_brick: {}, gotchas: [], elapsed_ms: 0.1,
                resolved: { edges: [] } },
      suggest_sharding: { proposals: [] },
    });
    (globalThis as unknown as { fetch: typeof fetch }).fetch = fetchFn as never;

    render(<App />);
    fireEvent.change(screen.getByTestId("preset-launcher"),
      { target: { value: "qwen3_next" } });

    await waitFor(() => {
      expect(calls.some((c) => c.method === "build_preset_specs")).toBe(true);
    });
    const presetCall = calls.find((c) => c.method === "build_preset_specs")!;
    expect(presetCall.params).toMatchObject({
      preset_name: "qwen3_next",
      hidden_size: 128,
    });
  });
});

describe("App integration — run pipeline", () => {
  it("Smoke button calls pipeline.run with smoke stages and opens the modal", async () => {
    const { fetchFn, calls } = recorder({
      build_preset_specs: {
        specs: [
          { kind: "attention", name: "a0", params: {} },
          { kind: "mlp",       name: "a1", params: {} },
        ],
        preset_name: "llama3_8b",
      },
      verify: { memory_per_brick: {}, gotchas: [], elapsed_ms: 0.1,
                resolved: { edges: [] } },
      suggest_sharding: { proposals: [] },
      "pipeline.run": {
        stages: [
          { name: "parse",             status: "ok",   elapsed_ms: 0.5 },
          { name: "verify_build_spec", status: "ok",   elapsed_ms: 1.2 },
          { name: "build_model",       status: "ok",   elapsed_ms: 10.0 },
        ],
        overall_status: "ok",
        total_elapsed_ms: 11.7,
      },
    });
    (globalThis as unknown as { fetch: typeof fetch }).fetch = fetchFn as never;

    render(<App />);
    // First populate canvas via preset so pipeline.run is allowed
    fireEvent.change(screen.getByTestId("preset-launcher"),
      { target: { value: "llama3_8b" } });
    await waitFor(() => {
      expect(calls.some((c) => c.method === "build_preset_specs")).toBe(true);
    });

    fireEvent.click(screen.getByTestId("run-pipeline"));

    await waitFor(() => {
      expect(calls.some((c) => c.method === "pipeline.run")).toBe(true);
    });
    const pipelineCall = calls.find((c) => c.method === "pipeline.run")!;
    const pipelineParams = pipelineCall.params as {
      pipeline: { stages: string[] };
    };
    expect(pipelineParams.pipeline.stages).toContain("parse");
    expect(pipelineParams.pipeline.stages).toContain("dry_forward");
    expect(pipelineParams.pipeline.stages).not.toContain("train");

    await waitFor(() => {
      expect(screen.getByTestId("run-result-modal")).toBeTruthy();
    });
    expect(screen.getByTestId("run-result-overall").textContent)
      .toContain("ok");
  });

  it("Smoke on empty canvas shows error in modal, no pipeline.run call", async () => {
    const { fetchFn, calls } = recorder({
      verify: { memory_per_brick: {}, gotchas: [], elapsed_ms: 0,
                resolved: { edges: [] } },
      suggest_sharding: { proposals: [] },
    });
    (globalThis as unknown as { fetch: typeof fetch }).fetch = fetchFn as never;

    render(<App />);
    fireEvent.click(screen.getByTestId("run-pipeline"));

    await waitFor(() => {
      expect(screen.getByTestId("run-result-error").textContent)
        .toContain("empty");
    });
    expect(calls.find((c) => c.method === "pipeline.run")).toBeUndefined();
  });
});

describe("App integration — tab switching", () => {
  it("mounts TokenizerPlayground when Tokenizer tab is selected", async () => {
    const { fetchFn } = recorder({
      verify: { memory_per_brick: {}, gotchas: [], elapsed_ms: 0,
                resolved: { edges: [] } },
      suggest_sharding: { proposals: [] },
    });
    (globalThis as unknown as { fetch: typeof fetch }).fetch = fetchFn as never;

    render(<App />);
    fireEvent.click(screen.getByTestId("app-tab-tokenizer"));
    expect(screen.getByTestId("tokenizer-playground")).toBeTruthy();
  });

  it("mounts DataInspector when Data tab is selected", async () => {
    const { fetchFn } = recorder({
      verify: { memory_per_brick: {}, gotchas: [], elapsed_ms: 0,
                resolved: { edges: [] } },
      suggest_sharding: { proposals: [] },
    });
    (globalThis as unknown as { fetch: typeof fetch }).fetch = fetchFn as never;

    render(<App />);
    fireEvent.click(screen.getByTestId("app-tab-data"));
    expect(screen.getByTestId("data-inspector")).toBeTruthy();
  });
});

describe("App integration — side-channel data wiring", () => {
  it("threads DataInspector side-channel coverage into pipeline spec", async () => {
    const { fetchFn, calls } = recorder({
      build_preset_specs: {
        specs: [
          { kind: "attention", name: "a0", params: {} },
          { kind: "mlp",       name: "a1", params: {} },
        ],
        preset_name: "llama3_8b",
      },
      verify: { memory_per_brick: {}, gotchas: [], elapsed_ms: 0,
                resolved: { edges: [] } },
      suggest_sharding: { proposals: [] },
      "data.preview_parquet": {
        rows: [{ row_index: 0, tokens: [1, 2],
                 channels: { platform_ids: [1, 2] } }],
        token_column: "input_ids",
        available_channels: ["platform_ids", "token_structure_ids"],
        side_channel_families: {
          platform: {
            family: "platform", status: "present",
            columns: ["platform_ids"], missing_columns: [],
            dropped_columns: [], token_alignment: "yes",
            graph_remapping: "not_applicable",
            provenance: "original", non_null_ratio: 1.0,
          },
        },
        bytes_per_token_avg: 1.0,
        bytes_per_token_p95: 1.0,
        bytes_per_token_max: 1,
        total_rows: 1,
        elapsed_ms: 0.1,
      },
      "pipeline.run": {
        stages: [{ name: "train", status: "ok", elapsed_ms: 1 }],
        overall_status: "ok",
        total_elapsed_ms: 1,
      },
    });
    (globalThis as unknown as { fetch: typeof fetch }).fetch = fetchFn as never;

    render(<App />);
    fireEvent.change(screen.getByTestId("preset-launcher"),
      { target: { value: "llama3_8b" } });
    await waitFor(() => {
      expect(calls.some((c) => c.method === "build_preset_specs")).toBe(true);
    });

    fireEvent.click(screen.getByTestId("app-tab-data"));
    fireEvent.change(screen.getByTestId("data-path"),
      { target: { value: "/tmp/enriched.parquet" } });
    fireEvent.click(screen.getByTestId("data-load"));
    await waitFor(() => {
      expect(screen.getByTestId("data-family-platform-status").textContent)
        .toContain("present");
    });

    fireEvent.click(screen.getByTestId("run-pipeline-toggle"));
    fireEvent.click(screen.getByTestId("run-pipeline-train"));
    await waitFor(() => {
      expect(calls.some((c) => c.method === "pipeline.run")).toBe(true);
    });
    const pipelineCall = calls.find((c) => c.method === "pipeline.run")!;
    const params = pipelineCall.params as {
      spec: { available_side_channels: string[] };
    };
    expect(params.spec.available_side_channels).toEqual([
      "platform_ids", "token_structure_ids",
    ]);
  });

  it("threads DataInspector shard list into train stage options", async () => {
    const { fetchFn, calls } = recorder({
      build_preset_specs: {
        specs: [
          { kind: "attention", name: "a0", params: {} },
          { kind: "mlp",       name: "a1", params: {} },
        ],
        preset_name: "llama3_8b",
      },
      verify: { memory_per_brick: {}, gotchas: [], elapsed_ms: 0,
                resolved: { edges: [] } },
      suggest_sharding: { proposals: [] },
      "data.preview_parquet": {
        rows: [{ row_index: 0, tokens: [1, 2], channels: {} }],
        token_column: "input_ids",
        available_channels: [],
        side_channel_families: {},
        edge_distributions: {},
        shards: [
          { index: 0, path: "/tmp/corpus/val_00000.parquet", byte_size: 128, row_count: 1 },
          { index: 1, path: "/tmp/corpus/val_00001.parquet", byte_size: 256, row_count: 1 },
        ],
        bytes_per_token_avg: 1.0,
        bytes_per_token_p95: 1.0,
        bytes_per_token_max: 1,
        total_rows: 1,
        elapsed_ms: 0.1,
      },
      "pipeline.run": {
        stages: [{ name: "train", status: "ok", elapsed_ms: 1 }],
        overall_status: "ok",
        total_elapsed_ms: 1,
      },
    });
    (globalThis as unknown as { fetch: typeof fetch }).fetch = fetchFn as never;

    render(<App />);
    fireEvent.change(screen.getByTestId("preset-launcher"),
      { target: { value: "llama3_8b" } });
    await waitFor(() => {
      expect(calls.some((c) => c.method === "build_preset_specs")).toBe(true);
    });

    fireEvent.click(screen.getByTestId("app-tab-data"));
    fireEvent.change(screen.getByTestId("data-path"),
      { target: { value: "/tmp/corpus/val_00000.parquet" } });
    fireEvent.click(screen.getByTestId("data-load"));
    await waitFor(() => {
      expect(screen.getByTestId("data-shard-1").textContent)
        .toContain("val_00001.parquet");
    });
    fireEvent.click(screen.getByTestId("data-use-for-train"));

    fireEvent.click(screen.getByTestId("run-pipeline-toggle"));
    fireEvent.click(screen.getByTestId("run-pipeline-train"));
    await waitFor(() => {
      expect(calls.some((c) => c.method === "pipeline.run")).toBe(true);
    });
    const pipelineCall = calls.find((c) => c.method === "pipeline.run")!;
    const params = pipelineCall.params as {
      pipeline: { stage_options: { train: { parquet_shards?: string[] } } };
    };
    expect(params.pipeline.stage_options.train.parquet_shards).toEqual([
      "/tmp/corpus/val_00000.parquet",
      "/tmp/corpus/val_00001.parquet",
    ]);
  });
});
