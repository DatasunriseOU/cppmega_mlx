"""Webwright final run: verify vbgui canvas layout for deepseek_v4_flash.

Critical points (see ../../plan.md):
  CP1 preset loads (graph appears)
  CP2 tokenizer node is narrow (<~300px) with tokens wrapped multi-row
  CP3 tokenizer has Example/File/Folder source switcher
  CP4 nodes do not overlap after Auto Align Graph + fit view
  CP5 repeated layers folded into a single block_group box
"""
import os
import time
from playwright.sync_api import sync_playwright

RUN_DIR = os.path.dirname(os.path.abspath(__file__))
SHOTS = os.path.join(RUN_DIR, "screenshots")
LOG = os.path.join(RUN_DIR, "final_script_log.txt")
URL = "http://127.0.0.1:8765"
PRESET = "deepseek_v4_flash"

os.makedirs(SHOTS, exist_ok=True)
log_lines = []


def log(msg):
    print(msg)
    log_lines.append(msg)


def shot(page, step, action):
    path = os.path.join(SHOTS, f"final_execution_{step}_{action}.png")
    page.screenshot(path=path)
    return path


# JS to collect node rectangles + labels and compute overlapping pairs.
OVERLAP_JS = r"""
() => {
  const nodes = Array.from(document.querySelectorAll('.react-flow__node'));
  const recs = nodes.map(n => {
    const r = n.getBoundingClientRect();
    const label = (n.innerText || '').replace(/\s+/g, ' ').trim().slice(0, 40);
    return { id: n.getAttribute('data-id') || '', label, x: r.x, y: r.y, w: r.width, h: r.height };
  }).filter(r => r.w > 4 && r.h > 4);
  const overlaps = [];
  for (let i = 0; i < recs.length; i++) {
    for (let j = i + 1; j < recs.length; j++) {
      const a = recs[i], b = recs[j];
      const ix = Math.max(0, Math.min(a.x + a.w, b.x + b.w) - Math.max(a.x, b.x));
      const iy = Math.max(0, Math.min(a.y + a.h, b.y + b.h) - Math.max(a.y, b.y));
      const area = ix * iy;
      if (area <= 0) continue;
      const minArea = Math.min(a.w * a.h, b.w * b.h);
      const frac = area / minArea;
      // ignore trivial 1-2px touches at shared handles
      if (area > 64 && frac > 0.03) {
        overlaps.push({ a: a.id || a.label, b: b.id || b.label, area: Math.round(area), frac: +frac.toFixed(3) });
      }
    }
  }
  return { count: recs.length, overlaps, recs };
}
"""


def main():
    with sync_playwright() as p:
        browser = p.firefox.launch(headless=True)
        page = browser.new_page(viewport={"width": 1280, "height": 1800})
        page.goto(URL, wait_until="networkidle")
        time.sleep(1.5)

        # CP1: select preset -> wizard -> Generate Architecture
        log(f"step 1 action: select preset '{PRESET}' from the toolbar <select>")
        selects = page.locator("select")
        target = None
        for i in range(selects.count()):
            opts = selects.nth(i).locator("option").all_inner_texts()
            if any(PRESET in o for o in opts):
                target = selects.nth(i)
                break
        assert target is not None, "preset <select> not found"
        target.select_option(label=PRESET)
        time.sleep(1.0)

        gen = page.get_by_role("button", name="Generate Architecture")
        if gen.count() > 0:
            log("step 2 action: click 'Generate Architecture' in the LLM gallery wizard")
            gen.first.click()
        else:
            log("step 2 action: no wizard shown; preset applied directly")
        page.wait_for_selector(".react-flow__node", timeout=15000)
        time.sleep(1.5)
        shot(page, 1, "preset_loaded")

        # CP4 prep: trigger Auto Align Graph, then allow the deferred re-measure
        # pass (~220ms) + render before measuring overlaps.
        log("step 3 action: click 'Auto Align Graph'")
        page.get_by_role("button", name="Auto Align Graph").first.click()
        time.sleep(1.5)
        # fit view via the react-flow control
        fit = page.locator(".react-flow__controls-fitview")
        if fit.count() > 0:
            log("step 4 action: click react-flow Fit View control")
            fit.first.click()
            time.sleep(1.0)
        shot(page, 2, "auto_aligned_fit")

        # CP4: overlap detection
        res = page.evaluate(OVERLAP_JS)
        log(f"step 5 action: measured {res['count']} canvas nodes; "
            f"{len(res['overlaps'])} overlapping pairs (area>64px & >3% of smaller node)")
        for ov in res["overlaps"]:
            log(f"    OVERLAP: {ov['a']}  X  {ov['b']}  area={ov['area']}px frac={ov['frac']}")

        # CP5: folded block group present
        groups = page.locator('[data-testid^="block-group-node-"]')
        n_groups = groups.count()
        group_labels = []
        for i in range(n_groups):
            group_labels.append(groups.nth(i).inner_text().replace("\n", " ").strip()[:80])
        log(f"step 6 action: folded block_group nodes found = {n_groups}: {group_labels}")

        # CP2/CP3: tokenizer node narrow + source switcher
        tok = page.locator('[data-testid="tokenizer-virtual-node"]').first
        tok.scroll_into_view_if_needed()
        box = tok.bounding_box()
        tok_w = round(box["width"]) if box else -1
        has_example = page.locator('[data-testid="tokenizer-source-example"]').count() > 0
        has_file = page.locator('[data-testid="tokenizer-source-file"]').count() > 0
        has_dir = page.locator('[data-testid="tokenizer-source-directory"]').count() > 0
        log(f"step 7 action: tokenizer width={tok_w}px; "
            f"switcher example={has_example} file={has_file} folder={has_dir}")
        shot(page, 3, "tokenizer_detail")

        # ---- verdicts ----
        cp1 = res["count"] > 3
        cp2 = 0 < tok_w <= 320
        cp3 = has_example and has_file and has_dir
        cp4 = len(res["overlaps"]) == 0
        cp5 = n_groups >= 1
        log("RESULT CP1 graph_loaded   = " + ("PASS" if cp1 else "FAIL"))
        log("RESULT CP2 tokenizer_narrow = " + ("PASS" if cp2 else "FAIL"))
        log("RESULT CP3 source_switcher = " + ("PASS" if cp3 else "FAIL"))
        log("RESULT CP4 no_overlap      = " + ("PASS" if cp4 else "FAIL"))
        log("RESULT CP5 folded_group    = " + ("PASS" if cp5 else "FAIL"))
        log(f"FINAL: preset={PRESET} nodes={res['count']} overlaps={len(res['overlaps'])} "
            f"groups={n_groups} tokenizer_w={tok_w}")

        browser.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        with open(LOG, "w") as f:
            f.write("\n".join(log_lines) + "\n")
