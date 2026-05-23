// V7-Q07.3: GotchasTab adapter-suggestion panel walk.
//
// Loads a preset, navigates to the Gotchas tab, asserts that the
// "Suggest adapters" / fix-suggestion UI surfaces are reachable and
// don't crash. Closes the "no e2e for adapter UI" coverage gap from
// docs/UI-TO-TRAIN-AUDIT-2026-05-23.md Lane 2.

import { test, expect } from "@playwright/test";
import { gotoApp, selectPreset } from "../fixtures";

test("Q07.3: GotchasTab adapter-suggestion surface is reachable",
async ({ page }) => {
  await gotoApp(page);
  // Pick a preset that's known to surface dim_env / shape gotchas
  // so the suggestion panel has something to render.
  await selectPreset(page, "llama3_8b");

  // Open the Gotchas tab in the sidebar (testid contract preserved
  // from V6 spec §3.5).
  const gotchasTab = page.getByTestId("sidebar-tab-gotchas");
  await gotchasTab.waitFor({ timeout: 5_000 });
  await gotchasTab.click();

  // The Gotchas pane mounts. Even if there are zero gotchas right now,
  // the tab content container + suggest-adapters panel must render.
  const pane = page.getByTestId("gotchas-tab");
  await pane.waitFor({ timeout: 5_000 });
  await expect(pane).toBeVisible();

  const suggester = page.getByTestId("gotchas-suggest-adapters-panel");
  await suggester.waitFor({ timeout: 5_000 });
  await expect(suggester).toBeVisible();
  // Both producer + consumer inputs + Run button are reachable.
  await expect(page.getByTestId("gotchas-suggest-adapters-producer"))
    .toBeVisible();
  await expect(page.getByTestId("gotchas-suggest-adapters-consumer"))
    .toBeVisible();
  await expect(page.getByTestId("gotchas-suggest-adapters-run"))
    .toBeVisible();
});
