import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { Palette } from "@/components/Palette";
import { BRICKS, ADAPTERS } from "@/lib/bricks";

describe("Palette", () => {
  it("renders every brick and adapter as a draggable", () => {
    render(<Palette />);
    for (const b of BRICKS) {
      expect(screen.getByTestId(`palette-brick-${b.kind}`)).toBeTruthy();
    }
    for (const a of ADAPTERS) {
      expect(screen.getByTestId(`palette-adapter-${a.kind}`)).toBeTruthy();
    }
  });

  it("fires onDragStart with kind + class when a brick is dragged", () => {
    const onDragStart = vi.fn();
    render(<Palette onDragStart={onDragStart} />);
    const tile = screen.getByTestId("palette-brick-attention");
    const event = createDragEvent("dragstart");
    fireEvent(tile, event);
    expect(onDragStart).toHaveBeenCalledWith("attention", "brick");
    expect(event.dataTransfer.getData("application/x-cppmega-brick")).toBe("attention");
  });

  it("emits the adapter mime type when an adapter is dragged", () => {
    const onDragStart = vi.fn();
    render(<Palette onDragStart={onDragStart} />);
    const tile = screen.getByTestId("palette-adapter-residual");
    const event = createDragEvent("dragstart");
    fireEvent(tile, event);
    expect(event.dataTransfer.getData("application/x-cppmega-adapter")).toBe("residual");
    expect(onDragStart).toHaveBeenCalledWith("residual", "adapter");
  });
});

function createDragEvent(type: string): DragEvent & { dataTransfer: DataTransfer } {
  const event = new Event(type, { bubbles: true }) as DragEvent;
  const data: Record<string, string> = {};
  Object.defineProperty(event, "dataTransfer", {
    value: {
      setData(k: string, v: string) { data[k] = v; },
      getData(k: string) { return data[k] ?? ""; },
      effectAllowed: "",
      dropEffect: "",
    } as unknown as DataTransfer,
    writable: false,
  });
  return event as DragEvent & { dataTransfer: DataTransfer };
}
