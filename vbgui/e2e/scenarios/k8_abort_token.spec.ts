// V7-K8 visual e2e — explicit abort_token in TrainOptionsPanel.
// Sets a custom abort token, runs train, asserts it completes (the
// token itself isn't aborted in this test — abort RPC e2e lives in
// 57_train_cancel.spec.ts; here we just verify the token field is
// surfaced + value persists across a run).

import { test, expect } from "@playwright/test";
import {
  gotoApp, selectPreset, clickRunPipeline, closeModal,
} from "../fixtures";

test("K8: abort_token override input persists + train ok", async ({
  page,
}) => {
  test.setTimeout(120_000);
  await gotoApp(page);
  await selectPreset(page, "llama3_8b");

  await page.getByTestId("train-options-toggle").click();
  const tokenInput = page.getByTestId("train-opt-abort_token");
  await tokenInput.fill("my-cancel-handle");
  await expect(tokenInput).toHaveValue("my-cancel-handle");

  await clickRunPipeline(page, "train");
  await page.getByTestId("run-result-expand-train").click();
  await expect(page.getByTestId("chart-svg")).toBeVisible({
    timeout: 30_000,
  });
  await closeModal(page);

  // Token value persists after train completes (the panel state isn't
  // cleared between runs).
  await expect(tokenInput).toHaveValue("my-cancel-handle");
});
