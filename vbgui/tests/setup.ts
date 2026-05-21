import "@testing-library/react";

// jsdom shims that React Flow's runtime needs but jsdom doesn't ship.
if (!("ResizeObserver" in globalThis)) {
  class ResizeObserverShim {
    observe(): void { /* noop */ }
    unobserve(): void { /* noop */ }
    disconnect(): void { /* noop */ }
  }
  (globalThis as unknown as { ResizeObserver: unknown }).ResizeObserver =
    ResizeObserverShim;
}

if (!("DOMMatrixReadOnly" in globalThis)) {
  class DOMMatrixReadOnlyShim {
    m22 = 1;
  }
  (globalThis as unknown as { DOMMatrixReadOnly: unknown }).DOMMatrixReadOnly =
    DOMMatrixReadOnlyShim;
}
