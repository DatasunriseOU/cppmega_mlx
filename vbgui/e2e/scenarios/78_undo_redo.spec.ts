// V7-H03: undo/redo through TopBar buttons + canvas state.
//
// Drop a preset → verify canvas has bricks → click Undo → assert
// fewer bricks (or none) → click Redo → assert bricks back.

import { test, expect } from "@playwright/test";
import { gotoApp, selectPreset } from "../fixtures";

test("V7-H03: Undo after preset drop reduces canvas; Redo restores",
  async ({ page }) => {
    test.setTimeout(60_000);
    await gotoApp(page);

    // Pre-state: canvas empty, both buttons disabled.
    await expect(page.getByTestId("top-bar-undo")).toBeDisabled();
    await expect(page.getByTestId("top-bar-redo")).toBeDisabled();

    await selectPreset(page, "llama3_8b");
    // After preset drop, canvas has bricks.
    const before = await page.locator(
      "[data-testid^='brick-node-']").count();
    expect(before).toBeGreaterThan(0);

    // Undo should now be enabled (we have >=2 history entries:
    // initial empty + post-preset).
    await expect.poll(async () =>
      await page.getByTestId("top-bar-undo").isEnabled(),
      { timeout: 5_000 }).toBe(true);

    await page.getByTestId("top-bar-undo").click();
    await expect.poll(async () =>
      await page.locator("[data-testid^='brick-node-']").count(),
      { timeout: 5_000 }).toBeLessThan(before);

    // Redo restores the brick count.
    await expect(page.getByTestId("top-bar-redo")).toBeEnabled();
    await page.getByTestId("top-bar-redo").click();
    await expect.poll(async () =>
      await page.locator("[data-testid^='brick-node-']").count(),
      { timeout: 5_000 }).toBe(before);
  });
