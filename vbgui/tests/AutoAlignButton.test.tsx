/**
 * Regression: the Auto-Align button used to wire `onClick={onAutoAlign}`
 * which leaks the MouseEvent into the first positional arg, breaking
 * the App.tsx handleAutoAlign(customNodes?: Node[]) signature. The
 * click would silently fail with "activeNodes.map is not a function"
 * inside a try/catch.
 *
 * This test asserts that clicking the button invokes the callback
 * with zero arguments (so the optional customNodes stays undefined).
 */

import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { ReactFlowProvider } from "@xyflow/react";
import { FlowCanvas } from "@/components/FlowCanvas";

describe("FlowCanvas Auto-Align button", () => {
  it("invokes onAutoAlign with zero arguments", () => {
    const onAutoAlign = vi.fn();
    render(
      <ReactFlowProvider>
        <FlowCanvas nodes={[]} edges={[]} onAutoAlign={onAutoAlign} />
      </ReactFlowProvider>,
    );
    fireEvent.click(screen.getByTestId("auto-align-button"));
    expect(onAutoAlign).toHaveBeenCalledTimes(1);
    // The fix wraps onClick in an arrow that calls onAutoAlign() with
    // no args — assert the call site never receives the MouseEvent.
    expect(onAutoAlign.mock.calls[0].length).toBe(0);
  });
});
