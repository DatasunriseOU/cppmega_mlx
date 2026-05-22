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

function hasUsableStorage(storage: Storage | undefined): storage is Storage {
  return storage != null &&
    typeof storage.getItem === "function" &&
    typeof storage.setItem === "function" &&
    typeof storage.removeItem === "function" &&
    typeof storage.clear === "function";
}

if (!hasUsableStorage(globalThis.localStorage)) {
  const values = new Map<string, string>();
  const storage: Storage = {
    get length() {
      return values.size;
    },
    clear() {
      values.clear();
    },
    getItem(key: string) {
      return values.get(key) ?? null;
    },
    key(index: number) {
      return Array.from(values.keys())[index] ?? null;
    },
    removeItem(key: string) {
      values.delete(key);
    },
    setItem(key: string, value: string) {
      values.set(key, value);
    },
  };
  Object.defineProperty(globalThis, "localStorage", {
    configurable: true,
    value: storage,
  });
  if (typeof window !== "undefined") {
    Object.defineProperty(window, "localStorage", {
      configurable: true,
      value: storage,
    });
  }
}
