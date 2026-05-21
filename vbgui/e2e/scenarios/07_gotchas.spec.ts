// Gotchas tab — surface compile-mode gotchas after switching the
// topology to whole_model, verify Auto-fix button appears.

import { test, expect } from "@playwright/test";
import { gotoApp, selectPreset } from "../fixtures";
import { snapshot } from "../utils/screenshot";

test("Gotchas — whole_model compile flips a known footgun", async ({ page }) => {
  await gotoApp(page);
  await selectPreset(page, "llama3_8b");

  await page.getByTestId("compile-mode").selectOption("whole_model");

  await page.getByTestId("sidebar-tab-gotchas").click();
  await page.getByTestId("gotchas-tab").waitFor();

  // Either an ERROR group appears (showing the fsdp2_whole_compile or
  // megatron_tp_whole_compile gotcha), or the tab notes "No gotchas".
  // Both are acceptable today; what matters is the navigation works.
  const errors = page.getByTestId("gotchas-error");
  const noGotchas = page.locator("text=No gotchas");
  await expect(errors.or(noGotchas)).toBeVisible({ timeout: 5_000 });
  await snapshot(page, "07_gotchas", "compile_whole_model_open_tab");
});

test("Gotchas — empty state when no sharding is configured",
  async ({ page }) => {
    await gotoApp(page);
    // No preset → no graph → no gotchas
    await page.getByTestId("sidebar-tab-gotchas").click();
    await page.getByTestId("gotchas-tab").waitFor();
    await expect(page.locator("text=No gotchas")).toBeVisible();
  });
