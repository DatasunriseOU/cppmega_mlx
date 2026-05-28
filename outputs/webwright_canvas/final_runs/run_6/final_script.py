"""Run 6: re-verify wrap using logical (flow-coord) positions.

Run 5's metric was on post-fitView screen coords which compress the layout,
breaking the row-cluster detection. React Flow writes each node's logical
position into its inline style.transform, so we parse that — independent of
the current zoom.
"""
import os, time, re
from playwright.sync_api import sync_playwright

RUN = os.path.dirname(os.path.abspath(__file__))
SHOTS = os.path.join(RUN, "screenshots"); os.makedirs(SHOTS, exist_ok=True)
LOG = os.path.join(RUN, "final_script_log.txt")
URL = "http://127.0.0.1:8765"
PRESET = "deepseek_v4_flash"
VIEW = {"width": 1440, "height": 900}

log_lines = []
def log(m): print(m); log_lines.append(m)
def shot(p, step, name): p.screenshot(path=os.path.join(SHOTS, f"final_execution_{step}_{name}.png"))

WRAP_METRICS = r"""
() => {
  const nodes = Array.from(document.querySelectorAll('.react-flow__node'));
  const list = [];
  const txRe = /translate\(\s*(-?\d+(?:\.\d+)?)px\s*,\s*(-?\d+(?:\.\d+)?)px\s*\)/;
  nodes.forEach(n => {
    const transform = (n.getAttribute('style') || '') + ' ' + (n.style ? n.style.transform || '' : '');
    const m = transform.match(txRe);
    if (!m) return;
    const x = parseFloat(m[1]), y = parseFloat(m[2]);
    const r = n.getBoundingClientRect();
    const t = n.querySelector('.react-flow__handle.target');
    const s = n.querySelector('.react-flow__handle.source');
    list.push({
      id: n.getAttribute('data-id'),
      x, y,
      w: parseFloat(n.style.width) || r.width,
      h: parseFloat(n.style.height) || r.height,
      tpos: t ? t.getAttribute('data-handlepos') || '' : '',
      spos: s ? s.getAttribute('data-handlepos') || '' : '',
    });
  });

  // Cluster into rows by logical y gap > 80px (V_GAP=110 in wrap code).
  const byY = [...list].sort((a, b) => a.y - b.y);
  const rows = [];
  for (const n of byY) {
    const last = rows[rows.length - 1];
    if (last) {
      const lastMaxY = Math.max(...last.map(p => p.y + p.h));
      if (n.y - lastMaxY > 80) rows.push([n]); else last.push(n);
    } else rows.push([n]);
  }
  const rowSummary = rows.map((row, i) => {
    const minX = Math.min(...row.map(n => n.x));
    const maxR = Math.max(...row.map(n => n.x + n.w));
    const inR = row.filter(n => n.tpos === 'right').length;
    const inL = row.filter(n => n.tpos === 'left').length;
    const outR = row.filter(n => n.spos === 'right').length;
    const outL = row.filter(n => n.spos === 'left').length;
    return { row: i, n: row.length, minX: Math.round(minX), maxR: Math.round(maxR),
             width: Math.round(maxR - minX), inL, inR, outL, outR };
  });

  const overlaps = [];
  for (let i = 0; i < list.length; i++) for (let j = i + 1; j < list.length; j++) {
    const a = list[i], b = list[j];
    const ix = Math.max(0, Math.min(a.x + a.w, b.x + b.w) - Math.max(a.x, b.x));
    const iy = Math.max(0, Math.min(a.y + a.h, b.y + b.h) - Math.max(a.y, b.y));
    const area = ix * iy;
    if (area > 64 && area / Math.min(a.w * a.h, b.w * b.h) > 0.03)
      overlaps.push({ a: a.id, b: b.id, area: Math.round(area) });
  }

  return { nodes: list.length, rows: rowSummary, overlaps };
}
"""


def main():
    with sync_playwright() as p:
        b = p.firefox.launch(headless=True)
        pg = b.new_page(viewport=VIEW)
        pg.goto(URL, wait_until="networkidle"); time.sleep(1.5)
        sels = pg.locator("select"); tgt = None
        for i in range(sels.count()):
            if any(PRESET in o for o in sels.nth(i).locator("option").all_inner_texts()):
                tgt = sels.nth(i); break
        log(f"step 1 action: load preset {PRESET} (viewport {VIEW['width']}x{VIEW['height']})")
        tgt.select_option(label=PRESET); time.sleep(1.0)
        g = pg.get_by_role("button", name="Generate Architecture")
        if g.count() > 0: g.first.click()
        pg.wait_for_selector(".react-flow__node", timeout=15000); time.sleep(1.0)

        log("step 2 action: Unpack All, then Auto Align Graph")
        pg.locator('[data-testid^="unpack-btn-"]').first.click(); time.sleep(0.4)
        pg.locator('[data-testid^="confirm-unpack-all-"]').first.click(); time.sleep(2.0)
        pg.get_by_role("button", name="Auto Align Graph").first.click(); time.sleep(3.0)
        pg.locator(".react-flow__controls-fitview").first.click(); time.sleep(0.8)
        shot(pg, 1, "unpacked_wrapped")

        m = pg.evaluate(WRAP_METRICS)
        log(f"  layout: nodes={m['nodes']}  rows={len(m['rows'])}  overlaps={len(m['overlaps'])}")
        for r in m['rows']:
            log(f"    row {r['row']}: n={r['n']}  x=[{r['minX']}..{r['maxR']}] width={r['width']} "
                f"in:L={r['inL']}/R={r['inR']} out:L={r['outL']}/R={r['outR']}")

        cp1 = len(m['rows']) >= 2
        cp4 = len(m['overlaps']) == 0
        # Even rows (0,2,4): expect input-Left, output-Right majority.
        # Odd rows (1,3,5):  expect input-Right, output-Left majority.
        def even_ok(r): return r['inL'] > r['inR'] and r['outR'] > r['outL']
        def odd_ok(r):  return r['inR'] > r['inL'] and r['outL'] > r['outR']
        cp2 = all((even_ok(r) if r['row'] % 2 == 0 else odd_ok(r)) for r in m['rows'])
        cp3 = all(r['width'] <= VIEW['width'] for r in m['rows'])
        log(f"RESULT CP1 wrapped_multirow = {'PASS' if cp1 else 'FAIL'} ({len(m['rows'])} rows)")
        log(f"RESULT CP2 row_directions   = {'PASS' if cp2 else 'FAIL'}")
        log(f"RESULT CP3 each_row_fits    = {'PASS' if cp3 else 'FAIL'}")
        log(f"RESULT CP4 no_overlap       = {'PASS' if cp4 else 'FAIL'}")
        b.close()


if __name__ == "__main__":
    try: main()
    finally:
        with open(LOG, "w") as f: f.write("\n".join(log_lines) + "\n")
