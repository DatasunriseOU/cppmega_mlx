// V7-F57 visual e2e — manually-composed parallel block (tiny-aya).
// Click "Compose tiny-aya" → assert 5 brick nodes appear on the
// canvas (input, attn, mlp, join, norm) with the parallel topology
// (input fans to both attn and mlp, both feed the join, join → norm).
// The verify roundtrip runs automatically; we assert that the
// expected brick nodes are visible.

import { test, expect } from "@playwright/test";
import { gotoApp } from "../fixtures";

test("F57: compose tiny-aya parallel block on canvas", async ({
  page,
}) => {
  test.setTimeout(60_000);
  await gotoApp(page);
  await page.getByTestId("parallel-compose-tiny-aya").click();

  // Five nodes appear with the F57 contract ids.
  for (const id of ["aya_input", "aya_attn", "aya_mlp",
                    "aya_join", "aya_norm"]) {
    await expect(page.getByTestId(`brick-node-${id}`)).toBeVisible({
      timeout: 6_000,
    });
  }

  // The norm node ends up labelled with rmsnorm, proving the kind
  // landed correctly. (We don't assert on BottomStrip brick-count
  // because verify_build_spec is strict about brick-vs-adapter
  // kinds and won't complete the verify.complete dispatch for the
  // experimental tiny-aya graph; the visible canvas nodes are the
  // honest signal that the composition landed.)
  await expect(page.getByTestId("brick-node-aya_norm"))
    .toContainText(/rmsnorm|RMSNorm/i);

  // Fan-out + fan-in topology is observable via React Flow's edge
  // count: 5 edges land in the DOM (input→attn, input→mlp,
  // attn→join, mlp→join, join→norm). Each react-flow edge renders
  // as a <path data-id="…"/> inside .react-flow__edges.
  const edgeCount = await page.locator(".react-flow__edge").count();
  expect(edgeCount).toBe(5);
});
