// V7-F56b visual e2e: edit dim_env via the inline editor so
// nh*head_dim != H, assert the live mismatch indicator surfaces
// during typing AND the post-verify canvas badge surfaces.
//
// Honest closure note: today the warning shows up as a yellow banner
// (severity=warning, not fail) — the verify call still returns ok
// because the codebase intentionally supports decoupled Q dim via
// internal W_Q : R^H → R^{nh*head_dim} projection. The badge alerts
// the architect that they probably mis-typed dim_env.

import { test, expect } from "@playwright/test";
import { gotoApp, selectPreset } from "../fixtures";

test("F56b: dim_env mismatch surfaces live + post-verify badge", async ({
  page,
}) => {
  await gotoApp(page);
  await selectPreset(page, "llama3_8b");

  // Editor is visible above the canvas after preset drop.
  const editor = page.getByTestId("dim-env-editor");
  await expect(editor).toBeVisible();

  // Compatible default (H=128, nh=2, head_dim=64 — 2*64=128) → no
  // inline mismatch, no badge.
  await expect(page.getByTestId("dim-env-inline-mismatch"))
    .toHaveCount(0);
  await expect(page.getByTestId("symbolic-dim-warn-badge"))
    .toHaveCount(0);

  // Mutate nh to 3 — 3*64=192 ≠ 128 (the F56 honest finding shape).
  // The inline indicator fires *during* typing, before Apply.
  await page.getByTestId("dim-env-nh").fill("3");
  const inline = page.getByTestId("dim-env-inline-mismatch");
  await expect(inline).toBeVisible();
  await expect(inline).toContainText("192");
  await expect(inline).toContainText("128");

  // Apply → verify RPC roundtrip → gotcha lands → badge renders.
  await page.getByTestId("dim-env-apply").click();
  const badge = page.getByTestId("symbolic-dim-warn-badge");
  await expect(badge).toBeVisible({ timeout: 8_000 });
  const message = page.getByTestId("symbolic-dim-warn-message");
  await expect(message).toContainText("128");
  await expect(message).toContainText("192");

  // Restore consistency → badge clears after the next verify.
  await page.getByTestId("dim-env-nh").fill("2");
  await page.getByTestId("dim-env-apply").click();
  await expect(page.getByTestId("symbolic-dim-warn-badge"))
    .toHaveCount(0, { timeout: 8_000 });
});
