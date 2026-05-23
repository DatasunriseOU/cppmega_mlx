// V7-L48 / L49 / L50 visual e2e — GotchasTab severity colours, source
// chip, and backend-driven suggested_fix recovery flow. Trigger via
// dim-env mismatch (which lands a v7_f56b_dim_env_mismatch gotcha),
// switch to the Gotchas tab, inspect the rendered card.

import { test, expect } from "@playwright/test";
import { gotoApp, selectPreset } from "../fixtures";

test("L48/L49/L50: GotchasTab surfaces severity + source + suggested_fix",
async ({ page }) => {
  test.setTimeout(60_000);
  await gotoApp(page);
  await selectPreset(page, "llama3_8b");

  // Force the F56b mismatch (3 * 64 = 192 != H=128).
  await page.getByTestId("dim-env-nh").fill("3");
  await page.getByTestId("dim-env-apply").click();
  // verify roundtrip lands the gotcha.
  await expect(page.getByTestId("symbolic-dim-warn-badge"))
    .toBeVisible({ timeout: 8_000 });

  // Open Gotchas sidebar tab.
  await page.getByTestId("sidebar-tab-gotchas").click();
  const card = page.getByTestId("gotcha-v7_f56b_dim_env_mismatch");
  await expect(card).toBeVisible({ timeout: 5_000 });

  // L50: severity is exposed as a data attribute + pill text.
  await expect(card).toHaveAttribute("data-severity", "warning");
  await expect(
    page.getByTestId("gotcha-v7_f56b_dim_env_mismatch-severity")
  ).toContainText(/warning/i);

  // L49: source chip parses the file basename.
  const src = page.getByTestId("gotcha-v7_f56b_dim_env_mismatch-source");
  await expect(src).toBeVisible();
  await expect(src).toContainText("diagnostics.py");

  // L48: backend pushed suggested_fix → Apply button visible (the
  // host hasn't wired onAutoFix for this id yet, so the hint chip
  // renders instead — assert either path exists).
  const fixBtn = page.getByTestId(
    "gotcha-v7_f56b_dim_env_mismatch-autofix");
  const fixHint = page.getByTestId(
    "gotcha-v7_f56b_dim_env_mismatch-fix-hint");
  const btnVisible = await fixBtn.isVisible().catch(() => false);
  const hintVisible = await fixHint.isVisible().catch(() => false);
  expect(btnVisible || hintVisible).toBe(true);
  if (btnVisible) {
    await expect(fixBtn).toContainText("Snap H");
  }
  if (hintVisible) {
    await expect(fixHint).toContainText("Snap H");
  }
});
