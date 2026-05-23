// Per-test helpers wrapping common GUI walks so scenarios stay short.

import { expect, type Locator, type Page } from "@playwright/test";

export const TIMEOUT_RUN = 20_000;

export async function gotoApp(page: Page): Promise<void> {
  await page.goto("/", { waitUntil: "load" });
  await expect(page.getByTestId("preset-launcher")).toBeVisible();
}

export async function selectPreset(page: Page, name: string): Promise<void> {
  await page.getByTestId("preset-launcher").selectOption(name);
  // canvas populates async (RPC → setNodes). Wait for at least one node.
  await expect.poll(async () =>
    await page.locator("[data-testid^='brick-node-']").count(),
    { timeout: 8_000 },
  ).toBeGreaterThan(0);
}

export async function clickTab(page: Page,
                               key: "canvas" | "tokenizer" | "data"): Promise<void> {
  await page.getByTestId(`app-tab-${key}`).click();
}

export async function loadTokenizerInPlayground(
  page: Page, sourcePath: string,
): Promise<void> {
  await clickTab(page, "tokenizer");
  await page.getByTestId("tokenizer-playground").waitFor();
  await page.getByTestId("add-panel").click();
  await page.getByTestId("tokenizer-source-0").fill(sourcePath);
  await page.getByTestId("tokenizer-encode-0").click();
  await page.getByTestId("tokenizer-metrics-0").waitFor();
}

export async function loadParquetInInspector(
  page: Page, parquetPath: string,
): Promise<void> {
  await clickTab(page, "data");
  await page.getByTestId("data-inspector").waitFor();
  await page.getByTestId("data-path").fill(parquetPath);
  await page.getByTestId("data-load").click();
  await page.getByTestId("data-metrics").waitFor();
}

export async function clickRunPipeline(
  page: Page, mode: "smoke" | "full" | "train",
): Promise<Locator> {
  await clickTab(page, "canvas");
  if (mode === "smoke") {
    await page.getByTestId("run-pipeline").click();
  } else {
    await page.getByTestId("run-pipeline-toggle").click();
    await page.getByTestId(`run-pipeline-${mode}`).click();
  }
  const modal = page.getByTestId("run-result-modal");
  await modal.waitFor({ timeout: TIMEOUT_RUN });
  return modal;
}

export async function assertOverallStatus(
  modal: Locator, expected: "ok" | "fail",
): Promise<void> {
  const overall = modal.getByTestId("run-result-overall");
  await expect(overall).toContainText(expected);
}

export async function closeModal(page: Page): Promise<void> {
  await page.getByTestId("run-result-close").click().catch(() => undefined);
  await page.getByTestId("run-result-modal").waitFor({ state: "detached",
                                                       timeout: 2_000 })
    .catch(() => undefined);
}

/** Drop a brick onto the canvas via synthetic dragstart/drop events. */
export async function dropBrickViaPalette(
  page: Page, kind: string,
): Promise<void> {
  const tile = page.getByTestId(`palette-brick-${kind}`);
  const canvas = page.getByTestId("flow-canvas");
  const before = await page.locator("[data-testid^='brick-node-']").count();
  await tile.hover();
  await page.mouse.down();
  const box = await canvas.boundingBox();
  if (!box) throw new Error("canvas has no bounding box");
  await page.mouse.move(box.x + 200, box.y + 200, { steps: 6 });
  await page.mouse.up();
  // Synthetic drop event (Playwright cannot natively dispatch HTML5
  // drag/drop, so fall back to the App's onDropBrick handler via JS).
  await canvas.evaluate((el, k) => {
    const dt = new DataTransfer();
    dt.setData("application/x-cppmega-brick", k);
    const ev = new DragEvent("drop", { bubbles: true, dataTransfer: dt,
                                       clientX: 200, clientY: 200 });
    el.dispatchEvent(ev);
  }, kind);
  await expect.poll(async () =>
    await page.locator("[data-testid^='brick-node-']").count(),
    { timeout: 4_000 },
  ).toBeGreaterThan(before);
}

/** Drop an adapter onto the canvas via synthetic dragstart/drop events. */
export async function dropAdapterViaPalette(
  page: Page, kind: string,
): Promise<void> {
  const tile = page.getByTestId(`palette-adapter-${kind}`);
  const canvas = page.getByTestId("flow-canvas");
  const before = await page.locator("[data-testid^='adapter-node-']").count();
  await tile.hover();
  await page.mouse.down();
  const box = await canvas.boundingBox();
  if (!box) throw new Error("canvas has no bounding box");
  await page.mouse.move(box.x + 200, box.y + 200, { steps: 6 });
  await page.mouse.up();
  // Synthetic drop event for adapter
  await canvas.evaluate((el, k) => {
    const dt = new DataTransfer();
    dt.setData("application/x-cppmega-adapter", k);
    const ev = new DragEvent("drop", { bubbles: true, dataTransfer: dt,
                                       clientX: 200, clientY: 200 });
    el.dispatchEvent(ev);
  }, kind);
  await expect.poll(async () =>
    await page.locator("[data-testid^='adapter-node-']").count(),
    { timeout: 4_000 },
  ).toBeGreaterThan(before);
}

/** Drop a transplant brick onto the canvas via synthetic dragstart/drop events. */
export async function dropTransplantViaBar(
  page: Page, name: string, kind: string, params: Record<string, unknown>,
): Promise<void> {
  const tile = page.getByTestId(`transplant-drag-brick-${name}`);
  const canvas = page.getByTestId("flow-canvas");
  const before = await page.locator("[data-testid^='brick-node-']").count();
  await tile.hover();
  await page.mouse.down();
  const box = await canvas.boundingBox();
  if (!box) throw new Error("canvas has no bounding box");
  await page.mouse.move(box.x + 300, box.y + 300, { steps: 6 });
  await page.mouse.up();
  // Synthetic drop event for transplant
  await canvas.evaluate((el, { k, p }) => {
    const dt = new DataTransfer();
    dt.setData("application/x-cppmega-transplant-kind", k);
    dt.setData("application/x-cppmega-transplant-params", JSON.stringify(p));
    const ev = new DragEvent("drop", { bubbles: true, dataTransfer: dt,
                                       clientX: 300, clientY: 300 });
    el.dispatchEvent(ev);
  }, { k: kind, p: params });
  await expect.poll(async () =>
    await page.locator("[data-testid^='brick-node-']").count(),
    { timeout: 4_000 },
  ).toBeGreaterThan(before);
}

