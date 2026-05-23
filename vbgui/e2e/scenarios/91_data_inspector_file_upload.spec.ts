// E-AUDIT-01: Playwright setInputFiles regression — drop a fixture
// parquet via the new DataInspector file picker and assert the path
// field auto-populates with /tmp/vbgui_uploads/<uuid>.parquet.

import { test, expect } from "@playwright/test";
import * as fs from "node:fs";
import * as path from "node:path";
import { gotoApp } from "../fixtures";

test("E-AUDIT-01: file upload via DataInspector picker", async ({ page }) => {
  test.setTimeout(30_000);
  await gotoApp(page);

  // Switch to the Data tab.
  await page.getByTestId("app-tab-data").click();
  const upload = page.getByTestId("data-inspector-file-upload");
  await expect(upload).toBeVisible();

  // Use any fixture parquet from tests/fixtures (the matrix builder
  // populates these — pick the first available .parquet by glob).
  const fixturesDir = path.resolve(
    __dirname, "..", "..", "..", "tests", "fixtures");
  let parquetPath: string | null = null;
  for (const f of fs.readdirSync(fixturesDir)) {
    if (f.endsWith(".parquet")) {
      parquetPath = path.join(fixturesDir, f);
      break;
    }
  }
  if (!parquetPath) {
    test.skip(true, "no .parquet fixture in tests/fixtures/");
    return;
  }
  await upload.setInputFiles(parquetPath);

  // Path field auto-populates with the backend-returned tmp path.
  const pathInput = page.getByTestId("data-path");
  await expect(pathInput).toHaveValue(
    /\/tmp\/vbgui_uploads\/[0-9a-f]+\.parquet/, { timeout: 5000 });
});
