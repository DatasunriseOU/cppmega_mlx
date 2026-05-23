# Instructions

- Following Playwright test failed.
- Explain why, be concise, respect Playwright best practices.
- Provide a snippet of code with the fix, if possible.

# Test info

- Name: e2e/scenarios/11_ui_to_train.spec.ts >> UI optimizer change to Muon propagates and produces weight delta
- Location: e2e/scenarios/11_ui_to_train.spec.ts:175:1

# Error details

```
Error: page.goto: Protocol error (Page.navigate): Cannot navigate to invalid URL
Call log:
  - navigating to "/", waiting until "load"

```

# Test source

```ts
  1   | // Per-test helpers wrapping common GUI walks so scenarios stay short.
  2   | 
  3   | import { expect, type Locator, type Page } from "@playwright/test";
  4   | 
  5   | export const TIMEOUT_RUN = 20_000;
  6   | 
  7   | export async function gotoApp(page: Page): Promise<void> {
> 8   |   await page.goto("/", { waitUntil: "load" });
      |              ^ Error: page.goto: Protocol error (Page.navigate): Cannot navigate to invalid URL
  9   |   await expect(page.getByTestId("preset-launcher")).toBeVisible();
  10  | }
  11  | 
  12  | export async function selectPreset(page: Page, name: string): Promise<void> {
  13  |   await page.getByTestId("preset-launcher").selectOption(name);
  14  |   // canvas populates async (RPC → setNodes). Wait for at least one node.
  15  |   await expect.poll(async () =>
  16  |     await page.locator("[data-testid^='brick-node-']").count(),
  17  |     { timeout: 8_000 },
  18  |   ).toBeGreaterThan(0);
  19  | }
  20  | 
  21  | export async function clickTab(page: Page,
  22  |                                key: "canvas" | "tokenizer" | "data"): Promise<void> {
  23  |   await page.getByTestId(`app-tab-${key}`).click();
  24  | }
  25  | 
  26  | export async function loadTokenizerInPlayground(
  27  |   page: Page, sourcePath: string,
  28  | ): Promise<void> {
  29  |   await clickTab(page, "tokenizer");
  30  |   await page.getByTestId("tokenizer-playground").waitFor();
  31  |   await page.getByTestId("add-panel").click();
  32  |   await page.getByTestId("tokenizer-source-0").fill(sourcePath);
  33  |   await page.getByTestId("tokenizer-encode-0").click();
  34  |   await page.getByTestId("tokenizer-metrics-0").waitFor();
  35  | }
  36  | 
  37  | export async function loadParquetInInspector(
  38  |   page: Page, parquetPath: string,
  39  | ): Promise<void> {
  40  |   await clickTab(page, "data");
  41  |   await page.getByTestId("data-inspector").waitFor();
  42  |   await page.getByTestId("data-path").fill(parquetPath);
  43  |   await page.getByTestId("data-load").click();
  44  |   await page.getByTestId("data-metrics").waitFor();
  45  | }
  46  | 
  47  | export async function clickRunPipeline(
  48  |   page: Page, mode: "smoke" | "full" | "train",
  49  | ): Promise<Locator> {
  50  |   await clickTab(page, "canvas");
  51  |   if (mode === "smoke") {
  52  |     await page.getByTestId("run-pipeline").click();
  53  |   } else {
  54  |     await page.getByTestId("run-pipeline-toggle").click();
  55  |     await page.getByTestId(`run-pipeline-${mode}`).click();
  56  |   }
  57  |   const modal = page.getByTestId("run-result-modal");
  58  |   await modal.waitFor({ timeout: TIMEOUT_RUN });
  59  |   return modal;
  60  | }
  61  | 
  62  | export async function assertOverallStatus(
  63  |   modal: Locator, expected: "ok" | "fail",
  64  | ): Promise<void> {
  65  |   const overall = modal.getByTestId("run-result-overall");
  66  |   await expect(overall).toContainText(expected);
  67  | }
  68  | 
  69  | export async function closeModal(page: Page): Promise<void> {
  70  |   await page.getByTestId("run-result-close").click().catch(() => undefined);
  71  |   await page.getByTestId("run-result-modal").waitFor({ state: "detached",
  72  |                                                        timeout: 2_000 })
  73  |     .catch(() => undefined);
  74  | }
  75  | 
  76  | /** Drop a brick onto the canvas via synthetic dragstart/drop events. */
  77  | export async function dropBrickViaPalette(
  78  |   page: Page, kind: string,
  79  | ): Promise<void> {
  80  |   const tile = page.getByTestId(`palette-brick-${kind}`);
  81  |   const canvas = page.getByTestId("flow-canvas");
  82  |   const before = await page.locator("[data-testid^='brick-node-']").count();
  83  |   await tile.hover();
  84  |   await page.mouse.down();
  85  |   const box = await canvas.boundingBox();
  86  |   if (!box) throw new Error("canvas has no bounding box");
  87  |   await page.mouse.move(box.x + 200, box.y + 200, { steps: 6 });
  88  |   await page.mouse.up();
  89  |   // Synthetic drop event (Playwright cannot natively dispatch HTML5
  90  |   // drag/drop, so fall back to the App's onDropBrick handler via JS).
  91  |   await canvas.evaluate((el, k) => {
  92  |     const dt = new DataTransfer();
  93  |     dt.setData("application/x-cppmega-brick", k);
  94  |     const ev = new DragEvent("drop", { bubbles: true, dataTransfer: dt,
  95  |                                        clientX: 200, clientY: 200 });
  96  |     el.dispatchEvent(ev);
  97  |   }, kind);
  98  |   await expect.poll(async () =>
  99  |     await page.locator("[data-testid^='brick-node-']").count(),
  100 |     { timeout: 4_000 },
  101 |   ).toBeGreaterThan(before);
  102 | }
  103 | 
  104 | /** Drop an adapter onto the canvas via synthetic dragstart/drop events. */
  105 | export async function dropAdapterViaPalette(
  106 |   page: Page, kind: string,
  107 | ): Promise<void> {
  108 |   const tile = page.getByTestId(`palette-adapter-${kind}`);
```