// V7-K7 visual e2e — after each successful train, RunHistoryPicker
// gains a new entry. The picker selector lets the architect pick
// which past run_id to continue from on the next warm-start.

import { test, expect } from "@playwright/test";
import {
  gotoApp, selectPreset, clickRunPipeline, closeModal,
} from "../fixtures";

test("K7: RunHistoryPicker grows after each train, selectable", async ({
  page,
}) => {
  test.setTimeout(180_000);
  await gotoApp(page);
  await selectPreset(page, "llama3_8b");

  // Initially empty.
  await expect(page.getByTestId("run-history-count"))
    .toContainText("0 runs");

  // First train.
  await clickRunPipeline(page, "train");
  await closeModal(page);

  await expect(page.getByTestId("run-history-count"))
    .toContainText("1 run in history", { timeout: 10_000 });

  // Second train.
  await clickRunPipeline(page, "train");
  await closeModal(page);
  await expect(page.getByTestId("run-history-count"))
    .toContainText("2 runs", { timeout: 10_000 });

  // Picker now offers (latest) + 2 named runs.
  const sel = page.getByTestId("run-history-select");
  await expect(sel.locator("option")).toHaveCount(3);

  // Select the older (second-in-the-list) run; the value is the
  // train-{timestamp}-{rand} id, so we read it via option.first().
  const optValues = await sel.evaluate((el) =>
    Array.from((el as HTMLSelectElement).options).map((o) => o.value));
  // optValues[0] = "" (latest), [1] = newest, [2] = oldest.
  await sel.selectOption(optValues[2]);
  // The select reflects the chosen value.
  await expect(sel).toHaveValue(optValues[2]);
});
