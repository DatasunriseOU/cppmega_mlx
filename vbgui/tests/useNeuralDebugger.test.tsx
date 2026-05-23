import { describe, it, expect, vi } from "vitest";
import { renderHook, act } from "@testing-library/react";
import { useNeuralDebugger } from "@/hooks/useNeuralDebugger";

describe("useNeuralDebugger hook", () => {
  it("initializes with default debugger mode false and correct step -1", () => {
    const { result } = renderHook(() => useNeuralDebugger());
    expect(result.current.debuggerMode).toBe(false);
    expect(result.current.activeStep).toBe(-1);
    expect(result.current.direction).toBe("forward");
    expect(result.current.prompt).toBe("The cat sat on the mat");
    expect(result.current.tokens.length).toBeGreaterThan(0);
    expect(result.current.isPlaying).toBe(false);
    expect(result.current.isWeightUpdated).toBe(false);
  });

  it("advances activeStep and transitions on stepForward", () => {
    const { result } = renderHook(() => useNeuralDebugger());
    
    // Step forward from -1 to 0
    act(() => {
      result.current.stepForward(3);
    });
    expect(result.current.activeStep).toBe(0);
    expect(result.current.direction).toBe("forward");

    // Advance all the way to maxStep
    act(() => {
      result.current.stepForward(3); // to 1
      result.current.stepForward(3); // to 2
      result.current.stepForward(3); // to 3 (maxStep)
    });
    expect(result.current.activeStep).toBe(3);

    // One more step forward transitions to backward pass!
    act(() => {
      result.current.stepForward(3);
    });
    expect(result.current.direction).toBe("backward");
    expect(result.current.activeStep).toBe(3);
  });

  it("resets state on resetDebugger", () => {
    const { result } = renderHook(() => useNeuralDebugger());
    
    act(() => {
      result.current.setDebuggerMode(true);
      result.current.stepForward(3);
      result.current.resetDebugger();
    });

    expect(result.current.activeStep).toBe(-1);
    expect(result.current.direction).toBe("forward");
    expect(result.current.isPlaying).toBe(false);
  });

  it("correctly simulates snappy full train step sequence", () => {
    vi.useFakeTimers();
    const { result } = renderHook(() => useNeuralDebugger());

    act(() => {
      result.current.animateFullTrainStep(3);
    });

    expect(result.current.activeStep).toBe(-1);
    expect(result.current.direction).toBe("forward");

    // Fast-forward fake timers
    act(() => {
      vi.advanceTimersByTime(2500); // 10 steps of 250ms
    });

    expect(result.current.isWeightUpdated).toBe(true);
    expect(result.current.direction).toBe("forward");
    
    vi.useRealTimers();
  });
});
