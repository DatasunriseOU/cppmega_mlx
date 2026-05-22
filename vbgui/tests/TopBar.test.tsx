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

  // G11: Save/Load buttons
  it("G11: spec-save + spec-load hidden when callbacks not provided", () => {
    render(<TopBar {...defaultTopProps()} />);
    expect(screen.queryByTestId("spec-save")).toBeNull();
    expect(screen.queryByTestId("spec-load")).toBeNull();
  });

  it("G11: spec-save fires onSaveSpec callback", () => {
    const onSaveSpec = vi.fn();
    render(<TopBar {...defaultTopProps({ onSaveSpec })} />);
    fireEvent.click(screen.getByTestId("spec-save"));
    expect(onSaveSpec).toHaveBeenCalledTimes(1);
  });

  it("G11: spec-load-input fires onLoadSpec with file", () => {
    const onLoadSpec = vi.fn();
    render(<TopBar {...defaultTopProps({ onLoadSpec })} />);
    const file = new File(['{"spec":{}}'], "test.spec.json",
      { type: "application/json" });
    fireEvent.change(screen.getByTestId("spec-load-input"),
      { target: { files: [file] } });
    expect(onLoadSpec).toHaveBeenCalledWith(file);
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

  it("Cancel button is disabled without an in-flight train run", () => {
    render(<TopBar {...defaultTopProps()} />);
    expect(screen.getByTestId("run-pipeline-cancel"))
      .toHaveProperty("disabled", true);
  });

  it("Cancel button fires onCancelTrain for an in-flight train run", () => {
    const onCancelTrain = vi.fn();
    render(<TopBar {...defaultTopProps({
      trainInFlight: true,
      trainRunId: "train-1",
      onCancelTrain,
    })} />);
    fireEvent.click(screen.getByTestId("run-pipeline-cancel"));
    expect(onCancelTrain).toHaveBeenCalledTimes(1);
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

  it("H05: train-checkpoint-save/load-path inputs forward into opts",
    () => {
      const onRunPipeline = vi.fn();
      render(<TopBar {...defaultTopProps({ onRunPipeline })} />);
      fireEvent.click(screen.getByTestId("run-pipeline-toggle"));
      fireEvent.change(screen.getByTestId("train-checkpoint-save-path"),
        { target: { value: "/tmp/save.safetensors" } });
      fireEvent.change(screen.getByTestId("train-checkpoint-load-path"),
        { target: { value: "/tmp/load.safetensors" } });
      fireEvent.click(screen.getByTestId("run-pipeline-train"));
      expect(onRunPipeline).toHaveBeenLastCalledWith("train",
        expect.objectContaining({
          checkpoint_save_path: "/tmp/save.safetensors",
          checkpoint_load_path: "/tmp/load.safetensors",
        }));
    });

  it("H08: train-probe-text textarea forwards inference_probe_text",
    () => {
      const onRunPipeline = vi.fn();
      render(<TopBar {...defaultTopProps({ onRunPipeline })} />);
      fireEvent.click(screen.getByTestId("run-pipeline-toggle"));
      fireEvent.change(screen.getByTestId("train-probe-text"),
        { target: { value: "hello world from probe" } });
      fireEvent.click(screen.getByTestId("run-pipeline-train"));
      expect(onRunPipeline).toHaveBeenLastCalledWith("train",
        expect.objectContaining({
          inference_probe_text: "hello world from probe",
        }));
    });

  it("H08: empty probe text omits field", () => {
    const onRunPipeline = vi.fn();
    render(<TopBar {...defaultTopProps({ onRunPipeline })} />);
    fireEvent.click(screen.getByTestId("run-pipeline-toggle"));
    fireEvent.click(screen.getByTestId("run-pipeline-train"));
    const opts = onRunPipeline.mock.calls.at(-1)?.[1];
    expect(opts.inference_probe_text).toBeUndefined();
  });

  it("H05: empty checkpoint inputs omit fields", () => {
    const onRunPipeline = vi.fn();
    render(<TopBar {...defaultTopProps({ onRunPipeline })} />);
    fireEvent.click(screen.getByTestId("run-pipeline-toggle"));
    fireEvent.click(screen.getByTestId("run-pipeline-train"));
    const opts = onRunPipeline.mock.calls.at(-1)?.[1];
    expect(opts.checkpoint_save_path).toBeUndefined();
    expect(opts.checkpoint_load_path).toBeUndefined();
  });

  it("H04: train-warm-start checkbox forwards warm_start flag", () => {
    const onRunPipeline = vi.fn();
    render(<TopBar {...defaultTopProps({ onRunPipeline })} />);
    fireEvent.click(screen.getByTestId("run-pipeline-toggle"));
    // Default OFF → warm_start: false
    fireEvent.click(screen.getByTestId("run-pipeline-train"));
    expect(onRunPipeline).toHaveBeenLastCalledWith("train",
      expect.objectContaining({ warm_start: false }));
    // Toggle ON → warm_start: true
    fireEvent.click(screen.getByTestId("run-pipeline-toggle"));
    fireEvent.click(screen.getByTestId("train-warm-start"));
    fireEvent.click(screen.getByTestId("run-pipeline-train"));
    expect(onRunPipeline).toHaveBeenLastCalledWith("train",
      expect.objectContaining({ warm_start: true }));
  });

  it("does not render legacy train side-channel checkboxes", () => {
    render(<TopBar {...defaultTopProps()} />);
    fireEvent.click(screen.getByTestId("run-pipeline-toggle"));
    expect(screen.queryByTestId("train-side-channel-doc_ids")).toBeNull();
    expect(screen.queryByTestId("train-side-channel-token_ids")).toBeNull();
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

  it("H11: estimate testid always present, actual hidden until set",
    () => {
      const s = { ...INITIAL_SPEC,
                  worst_rank_bytes: 10 * 1024 ** 3,
                  device_hbm_bytes: 80 * 1024 ** 3 };
      render(<MemoryBar state={s} />);
      expect(screen.getByTestId("memory-bar-estimate").textContent)
        .toContain("est");
      expect(screen.queryByTestId("memory-bar-actual")).toBeNull();
    });

  it("H11: actual testid appears once actual_peak_bytes is set", () => {
    const s = { ...INITIAL_SPEC,
                worst_rank_bytes: 10 * 1024 ** 3,
                device_hbm_bytes: 80 * 1024 ** 3,
                actual_peak_bytes: 12 * 1024 ** 3 };
    render(<MemoryBar state={s} />);
    expect(screen.getByTestId("memory-bar-estimate").textContent)
      .toContain("10.00 GB");
    expect(screen.getByTestId("memory-bar-actual").textContent)
      .toContain("12.00 GB");
  });
});
