import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { TopBar } from "@/components/TopBar";
import { BottomStrip } from "@/components/BottomStrip";
import { MemoryBar } from "@/components/MemoryBar";
import { INITIAL_SPEC, type TopologyFactory } from "@/state/spec";

const PRESETS = ["qwen3_next", "kimi_k2"] as const;
const TOPOS: readonly TopologyFactory[] = ["h100_8x", "m3_ultra_solo"];

function defaultTopProps(overrides: Partial<Parameters<typeof TopBar>[0]> = {}) {
  return {
    state: INITIAL_SPEC,
    projectName: "untitled",
    presets: PRESETS,
    topologies: TOPOS,
    onProjectNameChange: () => {},
    onPresetDrop: () => {},
    onTopologyChange: () => {},
    onCompileModeChange: () => {},
    onRunPipeline: () => {},
    ...overrides,
  } as Parameters<typeof TopBar>[0];
}

describe("TopBar", () => {
  it("renders project name + preset launcher + topology + compile + memory + run", () => {
    render(<TopBar {...defaultTopProps()} />);
    expect(screen.getByTestId("project-name")).toBeTruthy();
    expect(screen.getByTestId("preset-launcher")).toBeTruthy();
    expect(screen.getByTestId("topology-selector")).toBeTruthy();
    expect(screen.getByTestId("compile-mode")).toBeTruthy();
    expect(screen.getByTestId("memory-bar")).toBeTruthy();
    expect(screen.getByTestId("run-pipeline")).toBeTruthy();
  });

  // V4-1: train-data-source indicator
  it("train-data-source reads 'synthetic' when no parquet selected", () => {
    render(<TopBar {...defaultTopProps()} />);
    expect(screen.getByTestId("train-data-source").textContent)
      .toBe("synthetic");
  });

  it("train-data-source shows parquet basename when trainParquetPath set",
    () => {
      render(<TopBar {...defaultTopProps({
        trainParquetPath: "/tmp/fixtures/T2_gpt2_small__P1.parquet",
      })} />);
      const indicator = screen.getByTestId("train-data-source").textContent!;
      expect(indicator).toContain("parquet:");
      expect(indicator).toContain("T2_gpt2_small__P1.parquet");
      expect(indicator).not.toContain("/tmp/");
    });

  it("train-data-source appends tokenizer basename when both set", () => {
    render(<TopBar {...defaultTopProps({
      trainParquetPath: "/a/b/foo.parquet",
      trainTokenizerPath: "/x/y/cppmega_tokenizer.json",
    })} />);
    const t = screen.getByTestId("train-data-source").textContent!;
    expect(t).toContain("foo.parquet");
    expect(t).toContain("tok:");
    expect(t).toContain("cppmega_tokenizer.json");
  });

  it("preset launcher fires onPresetDrop when chosen", () => {
    const onPresetDrop = vi.fn();
    render(<TopBar {...defaultTopProps({ onPresetDrop })} />);
    fireEvent.change(screen.getByTestId("preset-launcher"),
      { target: { value: "kimi_k2" } });
    expect(onPresetDrop).toHaveBeenCalledWith("kimi_k2");
  });

  it("Smoke button calls onRunPipeline('smoke')", () => {
    const onRunPipeline = vi.fn();
    render(<TopBar {...defaultTopProps({ onRunPipeline })} />);
    fireEvent.click(screen.getByTestId("run-pipeline"));
    expect(onRunPipeline).toHaveBeenCalledWith("smoke");
  });

  it("toggle reveals Full + Train menu", () => {
    const onRunPipeline = vi.fn();
    render(<TopBar {...defaultTopProps({ onRunPipeline })} />);
    fireEvent.click(screen.getByTestId("run-pipeline-toggle"));
    fireEvent.click(screen.getByTestId("run-pipeline-full"));
    expect(onRunPipeline).toHaveBeenCalledWith("full");
    fireEvent.click(screen.getByTestId("run-pipeline-toggle"));
    fireEvent.click(screen.getByTestId("run-pipeline-train"));
    expect(onRunPipeline).toHaveBeenCalledWith("train",
      expect.objectContaining({ num_steps: expect.any(Number) }));
  });

  it("topology selector emits onTopologyChange", () => {
    const onTopologyChange = vi.fn();
    render(<TopBar {...defaultTopProps({ onTopologyChange })} />);
    fireEvent.change(screen.getByTestId("topology-selector"),
      { target: { value: "m3_ultra_solo" } });
    expect(onTopologyChange).toHaveBeenCalledWith("m3_ultra_solo");
  });

  it("compile mode selector emits onCompileModeChange", () => {
    const onCompileModeChange = vi.fn();
    render(<TopBar {...defaultTopProps({ onCompileModeChange })} />);
    fireEvent.change(screen.getByTestId("compile-mode"),
      { target: { value: "off" } });
    expect(onCompileModeChange).toHaveBeenCalledWith("off");
  });
});

describe("BottomStrip", () => {
  it("renders backend status + verify latency + brick count + help toggle", () => {
    render(<BottomStrip state={{
      ...INITIAL_SPEC, backend_status: "connected",
      last_verify_ms: 4.2, brick_count: 22,
    }} fusedRegionCount={3} />);
    expect(screen.getByTestId("backend-status").textContent)
      .toContain("Backend connected");
    expect(screen.getByTestId("verify-latency").textContent).toContain("4.2");
    expect(screen.getByTestId("brick-count").textContent)
      .toContain("22 bricks, 3 fused regions");
    expect(screen.getByTestId("help-toggle")).toBeTruthy();
  });
});

describe("MemoryBar", () => {
  it("fill width tracks fill ratio", () => {
    const s = { ...INITIAL_SPEC,
                worst_rank_bytes: 40 * 1024 ** 3,
                device_hbm_bytes: 80 * 1024 ** 3 };
    render(<MemoryBar state={s} />);
    const fill = screen.getByTestId("memory-bar-fill");
    expect(fill.getAttribute("style")).toContain("width: 50%");
  });
});
