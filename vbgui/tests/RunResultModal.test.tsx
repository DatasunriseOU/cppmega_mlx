import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { RunResultModal, type RunReport } from "@/components/RunResultModal";

const REPORT_OK: RunReport = {
  overall_status: "ok",
  total_elapsed_ms: 12.3,
  stages: [
    { name: "parse",             status: "ok",      elapsed_ms: 0.4 },
    { name: "verify_build_spec", status: "ok",      elapsed_ms: 1.2 },
    { name: "build_model",       status: "ok",      elapsed_ms: 234.0 },
  ],
};

const REPORT_FAIL: RunReport = {
  overall_status: "fail",
  total_elapsed_ms: 50.1,
  stages: [
    { name: "parse",        status: "ok",   elapsed_ms: 0.5 },
    { name: "dry_forward",  status: "fail", elapsed_ms: 12,
      error: { type: "ShapeMismatch",
               detail: "brick 'attn' expected (1,8,4096), got (1,8,2048)" } },
    { name: "train",        status: "skipped", elapsed_ms: 0 },
  ],
};

describe("RunResultModal", () => {
  it("returns null when there's nothing to show", () => {
    const { container } = render(
      <RunResultModal report={null} onClose={() => {}} />,
    );
    expect(container.firstChild).toBeNull();
  });

  it("renders overall_status and total elapsed in title", () => {
    render(<RunResultModal report={REPORT_OK} onClose={() => {}} />);
    expect(screen.getByTestId("run-result-overall").textContent)
      .toContain("ok");
    expect(screen.getByTestId("run-result-overall").textContent)
      .toContain("12.3");
  });

  it("renders one row per stage", () => {
    render(<RunResultModal report={REPORT_OK} onClose={() => {}} />);
    for (const s of REPORT_OK.stages) {
      expect(screen.getByTestId(`run-result-stage-${s.name}`)).toBeTruthy();
    }
  });

  it("close button fires onClose", () => {
    const onClose = vi.fn();
    render(<RunResultModal report={REPORT_OK} onClose={onClose} />);
    fireEvent.click(screen.getByTestId("run-result-close"));
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it("clicking backdrop fires onClose", () => {
    const onClose = vi.fn();
    render(<RunResultModal report={REPORT_OK} onClose={onClose} />);
    fireEvent.click(screen.getByTestId("run-result-modal-backdrop"));
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it("clicking inside the modal does NOT close it", () => {
    const onClose = vi.fn();
    render(<RunResultModal report={REPORT_OK} onClose={onClose} />);
    fireEvent.click(screen.getByTestId("run-result-modal"));
    expect(onClose).not.toHaveBeenCalled();
  });

  it("expand reveals failed-stage detail", () => {
    render(<RunResultModal report={REPORT_FAIL} onClose={() => {}} />);
    fireEvent.click(screen.getByTestId("run-result-expand-dry_forward"));
    const detail = screen.getByTestId("run-result-detail-dry_forward");
    expect(detail.textContent).toContain("ShapeMismatch");
    expect(detail.textContent).toContain("expected");
  });

  it("renders error envelope when report is null but error provided", () => {
    render(<RunResultModal report={null} error="backend down"
                           onClose={() => {}} />);
    expect(screen.getByTestId("run-result-error").textContent)
      .toContain("backend down");
  });

  // ---------------- V3-4: extras surfaced under each stage row -----------

  const REPORT_WITH_EXTRAS: RunReport = {
    overall_status: "ok",
    total_elapsed_ms: 500.0,
    stages: [
      {
        name: "train",
        status: "ok",
        elapsed_ms: 480.0,
        losses: [3.4, 3.1, 2.9],
        lr_trajectory: [0.001, 0.001, 0.001],
        weight_delta_norm: 0.0123,
        num_steps: 3,
        schedule_kind: "constant",
        optimizer_kind: "lion",
        model_summary: {
          mlp_activation: "swiglu",
          attention_pre_norm: "layernorm",
          attention_post_norm: "rmsnorm",
          mlp_pre_norm: "none",
          mlp_post_norm: "none",
          optimizer_kind: "lion",
          schedule_kind: "constant",
          num_brick_kinds: 2,
        },
      },
    ],
  };

  it("expand reveals extras when stage has only extras (no error)", () => {
    render(<RunResultModal report={REPORT_WITH_EXTRAS} onClose={() => {}} />);
    fireEvent.click(screen.getByTestId("run-result-expand-train"));
    expect(screen.getByTestId("run-result-extras-row-train")).toBeTruthy();
    expect(screen.getByTestId("run-result-extras-train")).toBeTruthy();
  });

  it("primitives surface with run-result-extras-{stage}-{key} testids", () => {
    render(<RunResultModal report={REPORT_WITH_EXTRAS} onClose={() => {}} />);
    fireEvent.click(screen.getByTestId("run-result-expand-train"));
    expect(screen.getByTestId("run-result-extras-train-optimizer_kind")
      .textContent).toBe("lion");
    expect(screen.getByTestId("run-result-extras-train-schedule_kind")
      .textContent).toBe("constant");
    expect(screen.getByTestId("run-result-extras-train-weight_delta_norm")
      .textContent).toBe("0.0123");
    expect(screen.getByTestId("run-result-extras-train-num_steps")
      .textContent).toBe("3");
  });

  it("arrays render with run-result-extras-{stage}-{key}-{i} per item", () => {
    render(<RunResultModal report={REPORT_WITH_EXTRAS} onClose={() => {}} />);
    fireEvent.click(screen.getByTestId("run-result-expand-train"));
    expect(screen.getByTestId("run-result-extras-train-losses-0")
      .textContent).toBe("3.4");
    expect(screen.getByTestId("run-result-extras-train-losses-2")
      .textContent).toBe("2.9");
    expect(screen.getByTestId("run-result-extras-train-lr_trajectory-1")
      .textContent).toBe("0.001");
  });

  it("nested objects render with run-result-extras-{stage}-{key}-{sub}", () => {
    render(<RunResultModal report={REPORT_WITH_EXTRAS} onClose={() => {}} />);
    fireEvent.click(screen.getByTestId("run-result-expand-train"));
    expect(screen.getByTestId(
      "run-result-extras-train-model_summary-mlp_activation")
      .textContent).toBe("swiglu");
    expect(screen.getByTestId(
      "run-result-extras-train-model_summary-attention_pre_norm")
      .textContent).toBe("layernorm");
    expect(screen.getByTestId(
      "run-result-extras-train-model_summary-optimizer_kind")
      .textContent).toBe("lion");
  });

  it("expand button hidden when stage has no content (no error+no extras)", () => {
    render(<RunResultModal report={REPORT_OK} onClose={() => {}} />);
    // REPORT_OK stages have no error and no extras → no expand button.
    expect(screen.queryByTestId("run-result-expand-parse")).toBeNull();
    expect(screen.queryByTestId("run-result-expand-build_model")).toBeNull();
  });

  it("expand reveals BOTH error and extras when both present", () => {
    const mixed: RunReport = {
      overall_status: "fail", total_elapsed_ms: 1.0,
      stages: [{
        name: "train", status: "fail", elapsed_ms: 1.0,
        error: { type: "Boom", detail: "kaboom" },
        losses: [1, 2],
      }],
    };
    render(<RunResultModal report={mixed} onClose={() => {}} />);
    fireEvent.click(screen.getByTestId("run-result-expand-train"));
    expect(screen.getByTestId("run-result-detail-train").textContent)
      .toContain("Boom");
    expect(screen.getByTestId("run-result-extras-train-losses-0")
      .textContent).toBe("1");
  });
});
