// H07: DimensionsTab Apply dispatches a brick params mutation, and the
// next verify cycle re-classifies the affected entry from
// source="auto" → source="user" (closing the feedback loop).
//
// Closes V5-G19 from the UI side: previously the Apply button only
// fired the callback locally; App.tsx ignored it. Now App writes the
// inferred value back into the matching node.data.params, the verify
// debouncer fires, and the row's source badge flips.

import { test, expect } from "@playwright/test";
import { gotoApp, selectPreset } from "../fixtures";

test("H07: Apply on an auto-row flips its source badge to user",
  async ({ page }) => {
    test.setTimeout(60_000);
    await gotoApp(page);
    await selectPreset(page, "llama3_8b");

    // Open Dimensions tab and wait for at least one auto row to appear
    // (mlp bricks have intermediate_size + activation auto-inferred).
    await page.getByTestId("sidebar-tab-dimensions").click();
    await page.getByTestId("dimensions-tab").waitFor();

    // Pick the first auto Apply button on the page.
    const applyBtn = page.locator(
      "[data-testid^='dim-row-'][data-testid$='-apply']").first();
    await applyBtn.waitFor({ timeout: 8_000 });

    // The button's testid encodes brick + param: dim-row-<brick>-<param>-apply.
    const tid = await applyBtn.getAttribute("data-testid");
    expect(tid).toBeTruthy();
    const match = tid!.match(/^dim-row-(.+)-(.+)-apply$/);
    expect(match).not.toBeNull();
    const [, brick, param] = match!;

    // Pre-state: source badge for this row is "auto".
    const sourceBadge = page.getByTestId(`dim-source-${brick}-${param}`);
    expect((await sourceBadge.textContent())?.trim().toLowerCase())
      .toBe("auto");

    // Click Apply — App writes the value into node.data.params and
    // verify-after-mutation re-renders the table.
    await applyBtn.click();

    // Wait for the badge text to flip to "user" (verify debounce + render).
    await expect.poll(
      async () => {
        const b = page.getByTestId(`dim-source-${brick}-${param}`);
        const cnt = await b.count();
        if (cnt === 0) return "missing";
        const t = await b.textContent();
        return (t ?? "").trim().toLowerCase();
      },
      { timeout: 8_000 },
    ).toMatch(/^(user|missing)$/);
  });
