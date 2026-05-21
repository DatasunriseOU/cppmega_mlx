// V3-8 + V3-9: Train button gating.
//
// Train must be disabled when verify or check_gotchas produces a
// severity='error' gotcha. We exercise this by intercepting the
// /rpc verify response and injecting a synthetic error gotcha. The
// UI's gating logic (App.tsx → TopBar.trainDisabled) must surface
// the reason via data-testid='top-bar-train-disabled-reason'.

import { test, expect } from "@playwright/test";
import { gotoApp, selectPreset } from "../fixtures";

async function injectVerifyGotcha(
  page: import("@playwright/test").Page,
  gotcha: { id: string; severity: "info" | "warning" | "error";
            message: string },
): Promise<void> {
  await page.route("**/rpc", async (route, request) => {
    const body = JSON.parse(request.postData() ?? "{}");
    if (body.method !== "verify") {
      await route.continue();
      return;
    }
    const upstream = await page.request.fetch(request);
    const json = await upstream.json();
    json.result.gotchas = [...(json.result.gotchas ?? []), gotcha];
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify(json),
    });
  });
}

// ---------------------------------------------------------------------------
// V3-8: critical gotcha (check_gotchas) disables Train
// ---------------------------------------------------------------------------

test("V3-8: critical gotcha disables Train + surfaces reason", async ({
  page,
}) => {
  await injectVerifyGotcha(page, {
    id: "fake_critical",
    severity: "error",
    message: "synthetic critical: incompatible loss for brick",
  });
  await gotoApp(page);
  await selectPreset(page, "llama3_8b");

  // Wait for verify to complete; the gotcha route fires per verify.
  await page.waitForTimeout(500);
  await page.getByTestId("run-pipeline-toggle").click();
  const trainBtn = page.getByTestId("run-pipeline-train");
  await expect(trainBtn).toBeDisabled();
  await expect(page.getByTestId("top-bar-train-disabled-reason"))
    .toContainText("synthetic critical");
});

// ---------------------------------------------------------------------------
// V3-9: verify=error severity gotcha disables Train
// ---------------------------------------------------------------------------

test("V3-9: verify error severity disables Train", async ({ page }) => {
  await injectVerifyGotcha(page, {
    id: "fake_verify_error",
    severity: "error",
    message: "synthetic verify error: parallel-block requires pre_norm",
  });
  await gotoApp(page);
  await selectPreset(page, "llama3_8b");

  await page.waitForTimeout(500);
  await page.getByTestId("run-pipeline-toggle").click();
  await expect(page.getByTestId("run-pipeline-train")).toBeDisabled();
  await expect(page.getByTestId("top-bar-train-disabled-reason"))
    .toContainText("parallel-block requires pre_norm");
});

// ---------------------------------------------------------------------------
// Negative control: no error gotcha → Train remains enabled
// ---------------------------------------------------------------------------

test("Train stays enabled when no error-severity gotchas present",
  async ({ page }) => {
    await gotoApp(page);
    await selectPreset(page, "llama3_8b");
    await page.waitForTimeout(500);
    await page.getByTestId("run-pipeline-toggle").click();
    await expect(page.getByTestId("run-pipeline-train")).toBeEnabled();
    await expect(page.getByTestId("top-bar-train-disabled-reason"))
      .toHaveCount(0);
  });
