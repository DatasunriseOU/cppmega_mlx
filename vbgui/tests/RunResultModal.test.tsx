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
});
