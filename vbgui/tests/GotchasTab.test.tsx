import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { GotchasTab } from "@/components/sidebar/GotchasTab";

describe("V7-L48/L49/L50 GotchasTab", () => {
  it("V7-L50: each severity card carries data-severity + tinted bg", () => {
    render(<GotchasTab gotchas={[
      { id: "g_err",  severity: "error",   message: "bad" },
      { id: "g_warn", severity: "warning", message: "soft" },
      { id: "g_info", severity: "info",    message: "fyi" },
    ]} />);
    expect(screen.getByTestId("gotcha-g_err").getAttribute("data-severity"))
      .toBe("error");
    expect(screen.getByTestId("gotcha-g_warn").getAttribute("data-severity"))
      .toBe("warning");
    expect(screen.getByTestId("gotcha-g_info").getAttribute("data-severity"))
      .toBe("info");
    // backgrounds must differ between severities.
    const bgErr = (screen.getByTestId("gotcha-g_err") as HTMLElement)
      .style.background;
    const bgWarn = (screen.getByTestId("gotcha-g_warn") as HTMLElement)
      .style.background;
    const bgInfo = (screen.getByTestId("gotcha-g_info") as HTMLElement)
      .style.background;
    expect(bgErr).not.toBe(bgWarn);
    expect(bgWarn).not.toBe(bgInfo);
    expect(bgErr).not.toBe(bgInfo);
  });

  it("V7-L50: severity pill renders with uppercase text", () => {
    render(<GotchasTab gotchas={[
      { id: "g1", severity: "warning", message: "x" },
    ]} />);
    expect(screen.getByTestId("gotcha-g1-severity").textContent)
      .toBe("warning");
  });

  it("V7-L49: source chip parses out the file basename from reference", () => {
    render(<GotchasTab gotchas={[
      { id: "g1", severity: "error", message: "x",
        reference: "cppmega_v4/buildspec/diagnostics.py:283" },
    ]} />);
    const src = screen.getByTestId("gotcha-g1-source");
    expect(src.textContent).toContain("diagnostics.py:283");
    expect(src.getAttribute("title")).toBe(
      "cppmega_v4/buildspec/diagnostics.py:283");
  });

  it("V7-L49: source chip handles URL references too", () => {
    render(<GotchasTab gotchas={[
      { id: "g1", severity: "error", message: "x",
        reference: "https://github.com/x/y/blob/main/foo.py#L42" },
    ]} />);
    expect(screen.getByTestId("gotcha-g1-source").textContent)
      .toContain("foo.py#L42");
  });

  it("V7-L49: no source chip when reference unset", () => {
    render(<GotchasTab gotchas={[
      { id: "g1", severity: "error", message: "x" },
    ]} />);
    expect(screen.queryByTestId("gotcha-g1-source")).toBeNull();
  });

  it("V7-L48: backend suggested_fix renders Apply button + label", () => {
    const onFix = vi.fn();
    render(<GotchasTab onAutoFix={onFix} gotchas={[
      { id: "totally_new_gotcha", severity: "warning",
        message: "novel issue", suggested_fix: "Restart with bf16" },
    ]} />);
    const btn = screen.getByTestId("gotcha-totally_new_gotcha-autofix");
    expect(btn.textContent).toContain("Restart with bf16");
    fireEvent.click(btn);
    expect(onFix).toHaveBeenCalledWith("totally_new_gotcha");
  });

  it("V7-L48: legacy AUTO_FIXABLE ids still work without suggested_fix", () => {
    const onFix = vi.fn();
    render(<GotchasTab onAutoFix={onFix} gotchas={[
      { id: "missing_edge", severity: "error", message: "no edge" },
    ]} />);
    const btn = screen.getByTestId("gotcha-missing_edge-autofix");
    expect(btn.textContent).toContain("Insert missing edge");
    fireEvent.click(btn);
    expect(onFix).toHaveBeenCalledWith("missing_edge");
  });

  it("V7-L48: suggested_fix wins over legacy FIX_LABELS for known ids", () => {
    render(<GotchasTab onAutoFix={() => {}} gotchas={[
      { id: "missing_edge", severity: "error",
        message: "n", suggested_fix: "Override: auto-insert linear bridge" },
    ]} />);
    expect(screen.getByTestId("gotcha-missing_edge-autofix").textContent)
      .toContain("auto-insert linear bridge");
  });

  it("V7-L48: when no onAutoFix prop, suggested_fix shows as hint not button", () => {
    render(<GotchasTab gotchas={[
      { id: "g1", severity: "warning", message: "x",
        suggested_fix: "do the thing" },
    ]} />);
    expect(screen.queryByTestId("gotcha-g1-autofix")).toBeNull();
    expect(screen.getByTestId("gotcha-g1-fix-hint").textContent)
      .toContain("do the thing");
  });

  it("empty list renders the no-gotchas hint", () => {
    render(<GotchasTab gotchas={[]} />);
    expect(screen.getByText(/no gotchas fired/i)).toBeDefined();
  });
});
