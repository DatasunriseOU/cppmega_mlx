# Instructions

- Following Playwright test failed.
- Explain why, be concise, respect Playwright best practices.
- Provide a snippet of code with the fix, if possible.

# Test info

- Name: e2e/scenarios/92_neural_debugger.spec.ts >> Interactive Neural Debugger & Step-by-Step Training Simulator >> runs debugger step-by-step and full train simulation with zero console warnings
- Location: e2e/scenarios/92_neural_debugger.spec.ts:6:3

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
  14  |   
  15  |   // If the LLM gallery wizard modal shows up, submit it
  16  |   const generateBtn = page.getByTestId("llm-wizard-generate");
  17  |   try {
  18  |     await generateBtn.waitFor({ state: "visible", timeout: 2000 });
  19  |     await generateBtn.click();
  20  |   } catch {
  21  |     // Soft ignore if the modal doesn't appear
  22  |   }
  23  | 
  24  |   // canvas populates async (RPC → setNodes). Wait for at least one node.
  25  |   await expect.poll(async () =>
  26  |     await page.locator("[data-testid^='brick-node-']").count(),
  27  |     { timeout: 8_000 },
  28  |   ).toBeGreaterThan(0);
  29  | }
  30  | 
  31  | export async function clickTab(page: Page,
  32  |                                key: "canvas" | "tokenizer" | "data"): Promise<void> {
  33  |   await page.getByTestId(`app-tab-${key}`).click();
  34  | }
  35  | 
  36  | export async function loadTokenizerInPlayground(
  37  |   page: Page, sourcePath: string,
  38  | ): Promise<void> {
  39  |   await clickTab(page, "tokenizer");
  40  |   await page.getByTestId("tokenizer-playground").waitFor();
  41  |   await page.getByTestId("add-panel").click();
  42  |   await page.getByTestId("tokenizer-source-0").fill(sourcePath);
  43  |   await page.getByTestId("tokenizer-encode-0").click();
  44  |   await page.getByTestId("tokenizer-metrics-0").waitFor();
  45  | }
  46  | 
  47  | export async function loadParquetInInspector(
  48  |   page: Page, parquetPath: string,
  49  | ): Promise<void> {
  50  |   await clickTab(page, "data");
  51  |   await page.getByTestId("data-inspector").waitFor();
  52  |   await page.getByTestId("data-path").fill(parquetPath);
  53  |   await page.getByTestId("data-load").click();
  54  |   await page.getByTestId("data-metrics").waitFor();
  55  | }
  56  | 
  57  | export async function clickRunPipeline(
  58  |   page: Page, mode: "smoke" | "full" | "train",
  59  | ): Promise<Locator> {
  60  |   await clickTab(page, "canvas");
  61  |   if (mode === "smoke") {
  62  |     await page.getByTestId("run-pipeline").click();
  63  |   } else {
  64  |     await page.getByTestId("run-pipeline-toggle").click();
  65  |     await page.getByTestId(`run-pipeline-${mode}`).click();
  66  |   }
  67  |   const modal = page.getByTestId("run-result-modal");
  68  |   await modal.waitFor({ timeout: TIMEOUT_RUN });
  69  |   return modal;
  70  | }
  71  | 
  72  | export async function assertOverallStatus(
  73  |   modal: Locator, expected: "ok" | "fail",
  74  | ): Promise<void> {
  75  |   const overall = modal.getByTestId("run-result-overall");
  76  |   await expect(overall).toContainText(expected);
  77  | }
  78  | 
  79  | export async function closeModal(page: Page): Promise<void> {
  80  |   await page.getByTestId("run-result-close").click().catch(() => undefined);
  81  |   await page.getByTestId("run-result-modal").waitFor({ state: "detached",
  82  |                                                        timeout: 2_000 })
  83  |     .catch(() => undefined);
  84  | }
  85  | 
  86  | /** Drop a brick onto the canvas via synthetic dragstart/drop events. */
  87  | export async function dropBrickViaPalette(
  88  |   page: Page, kind: string,
  89  | ): Promise<void> {
  90  |   const tile = page.getByTestId(`palette-brick-${kind}`);
  91  |   const canvas = page.getByTestId("flow-canvas");
  92  |   const before = await page.locator("[data-testid^='brick-node-']").count();
  93  |   await tile.hover();
  94  |   await page.mouse.down();
  95  |   const box = await canvas.boundingBox();
  96  |   if (!box) throw new Error("canvas has no bounding box");
  97  |   await page.mouse.move(box.x + 200, box.y + 200, { steps: 6 });
  98  |   await page.mouse.up();
  99  |   // Synthetic drop event (Playwright cannot natively dispatch HTML5
  100 |   // drag/drop, so fall back to the App's onDropBrick handler via JS).
  101 |   await canvas.evaluate((el, k) => {
  102 |     const dt = new DataTransfer();
  103 |     dt.setData("application/x-cppmega-brick", k);
  104 |     const ev = new DragEvent("drop", { bubbles: true, dataTransfer: dt,
  105 |                                        clientX: 200, clientY: 200 });
  106 |     el.dispatchEvent(ev);
  107 |   }, kind);
  108 |   await expect.poll(async () =>
```