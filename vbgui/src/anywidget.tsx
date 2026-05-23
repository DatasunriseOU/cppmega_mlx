// Anywidget ESM entry — mounts the React App into the host element.
//
// Anywidget API contract: this module's `default.render(model, el)`
// is invoked by the kernel-side AnyWidget. The model is an
// AnyModel<traitlets>; the el is the container <div> in the kernel
// frontend (Jupyter, JupyterLab, VS Code).
//
// We sync the canvas + spec via traitlets. The kernel side owns the
// authoritative state; we re-render whenever a traitlet changes.

import { StrictMode } from "react";
import { createRoot, type Root } from "react-dom/client";
import { App } from "./App";
import "@xyflow/react/dist/style.css";
import "./theme.css";

interface AnyModel {
  get<T = unknown>(key: string): T;
  set<T = unknown>(key: string, value: T): void;
  save_changes(): void;
  on(event: string, cb: () => void): void;
  off(event: string, cb: () => void): void;
}

interface RenderContext {
  model: AnyModel;
  el: HTMLElement;
}

const roots = new WeakMap<HTMLElement, Root>();

function render({ el }: RenderContext): () => void {
  let root = roots.get(el);
  if (!root) {
    root = createRoot(el);
    roots.set(el, root);
  }
  root.render(
    <StrictMode>
      <App />
    </StrictMode>,
  );
  // Cleanup function — anywidget invokes this when the widget unmounts.
  return () => {
    root?.unmount();
    roots.delete(el);
  };
}

export default { render };
