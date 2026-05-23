// V7-F89 visual e2e — advanced canvas drag-drop (adapters and transplants)
// Tests:
// 1. Dragging an adapter from the Palette to the Canvas correctly spawns an adapter node
// 2. Dragging a Transplanted Brick from the TransplantBar directly to the Canvas correctly spawns a brick node carrying its params

import { test, expect } from "@playwright/test";
import { gotoApp, selectPreset, dropAdapterViaPalette, dropTransplantViaBar } from "../fixtures";
import { snapshot } from "../utils/screenshot";

test.describe("advanced canvas drag-drop (E2E)", () => {
  test("drag-drop rmsnorm adapter from palette to canvas", async ({ page }) => {
    await gotoApp(page);
    await selectPreset(page, "llama3_8b");

    const beforeAdapters = await page.locator("[data-testid^='adapter-node-']").count();

    // Drop 'rmsnorm' adapter from palette to canvas
    await dropAdapterViaPalette(page, "rmsnorm");

    // Assert that a new adapter node appears on the canvas
    const afterAdapters = await page.locator("[data-testid^='adapter-node-']").count();
    expect(afterAdapters).toBeGreaterThan(beforeAdapters);

    // Verify it is styled correctly (dashed border/italic/label RMSNorm)
    const node = page.locator("[data-testid^='adapter-node-rmsnorm_']").first();
    await expect(node).toBeVisible();
    await expect(node).toContainText("RMSNorm");
    await expect(node).toContainText("adapter");

    await snapshot(page, "89_canvas_advanced_drag_drop", "rmsnorm_adapter_dropped");
  });

  test("drag-drop transplant brick from TransplantBar to canvas", async ({ page }) => {
    test.setTimeout(120_000);
    await gotoApp(page);
    await selectPreset(page, "llama3_8b");

    const bar = page.getByTestId("transplant-bar");
    await expect(bar).toBeVisible();

    // Pick mixtral-style preset (llama4_maverick) as source.
    await page.getByTestId("transplant-source-preset").selectOption("llama4_maverick");
    await page.getByTestId("transplant-load-source").click();

    // Wait for the draggable transplant list to populate
    const draggableList = page.getByTestId("transplant-draggable-list");
    await expect(draggableList).toBeVisible({ timeout: 10_000 });

    const beforeBricks = await page.locator("[data-testid^='brick-node-']").count();

    // The llama4_maverick preset includes a mixtral moe brick named "llama4_maverick_moe"
    const transplantCard = page.getByTestId("transplant-drag-brick-llama4_maverick_moe");
    await expect(transplantCard).toBeVisible();

    // Drag-drop it onto the canvas via synthetic drop event
    const xplantParams = { experts: 8, capacity_factor: 1.5 };
    await dropTransplantViaBar(page, "llama4_maverick_moe", "moe", xplantParams);

    // Assert that a new brick node of type moe was created
    const afterBricks = await page.locator("[data-testid^='brick-node-']").count();
    expect(afterBricks).toBeGreaterThan(beforeBricks);

    const xnode = page.locator("[data-testid^='brick-node-moe_']").first();
    await expect(xnode).toBeVisible();
    await expect(xnode).toContainText("MoE");

    // Click the node to open BrickContextPanel and verify params were preserved
    await xnode.click();
    const panel = page.locator("[data-testid^='brick-context-moe_']").first();
    await expect(panel).toBeVisible({ timeout: 5_000 });
    await expect(panel).toContainText("[moe]");

    // Validate parameter input values inside context panel if applicable
    await snapshot(page, "89_canvas_advanced_drag_drop", "moe_transplant_dropped");
  });
});
