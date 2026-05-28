"""Edge-over-box violation detector across multiple unpack states.

For each scenario, samples every .react-flow__edge-path SVG at 80 points and
runs a Liang-Barsky segment-vs-rect test against every node box (excluding the
edge's own endpoint nodes, and skipping the first/last 4 samples near handles).
Reports the number of edges that pierce a node body — the metric that must
reach 0.
"""
import sys, time
from playwright.sync_api import sync_playwright

URL = "http://127.0.0.1:8765"
PRESET = "deepseek_v4_flash"

VIOLATION_JS = r"""
() => {
  // Use FLOW coordinates for BOTH nodes and edges. Edge path getPointAtLength
  // returns viewport-local (flow) coords; node transform translate gives flow
  // position and offsetWidth/Height give unscaled (flow) size. Mixing flow
  // edge coords with screen getBoundingClientRect rects gives false positives.
  const txRe = /translate(?:3d)?\(\s*(-?\d+(?:\.\d+)?)px(?:\s*,\s*(-?\d+(?:\.\d+)?)px)?/;
  const nodes = Array.from(document.querySelectorAll('.react-flow__node')).map(n => {
    const m = (n.style.transform || '').match(txRe);
    if (!m) return null;
    const x = parseFloat(m[1]); const y = m[2] ? parseFloat(m[2]) : 0;
    const w = n.offsetWidth; const h = n.offsetHeight;
    return { id: n.getAttribute('data-id'), x, y, w, h };
  }).filter(n => n && n.w > 4 && n.h > 4);

  const edges = [];
  Array.from(document.querySelectorAll('.react-flow__edge')).forEach(e => {
    const p = e.querySelector('.react-flow__edge-path'); if (!p) return;
    const aria = e.getAttribute('aria-label') || ''; const m = aria.match(/Edge from (.+?) to (.+)/);
    if (!m) return;
    let L = 0; try { L = p.getTotalLength(); } catch (e) { return; }
    if (!isFinite(L) || L < 8) return;
    const N = 80, pts = [];
    for (let i = 0; i <= N; i++) { const q = p.getPointAtLength(i / N * L); pts.push({ x: q.x, y: q.y }); }
    edges.push({ s: m[1], t: m[2], pts });
  });

  function segRect(p1, p2, r) {
    let x0 = p1.x, y0 = p1.y;
    const dx = p2.x - x0, dy = p2.y - y0;
    let t0 = 0, t1 = 1;
    const xmin = r.x + 3, xmax = r.x + r.w - 3, ymin = r.y + 3, ymax = r.y + r.h - 3;
    if (xmin >= xmax || ymin >= ymax) return false;
    const pp = [-dx, dx, -dy, dy], qq = [x0 - xmin, xmax - x0, y0 - ymin, ymax - y0];
    for (let i = 0; i < 4; i++) {
      if (pp[i] === 0) { if (qq[i] < 0) return false; }
      else { const t = qq[i] / pp[i];
        if (pp[i] < 0) { if (t > t1) return false; if (t > t0) t0 = t; }
        else { if (t < t0) return false; if (t < t1) t1 = t; } }
    }
    return t0 < t1;
  }

  const violations = [];
  for (const e of edges) {
    for (const n of nodes) {
      if (n.id === e.s || n.id === e.t) continue;
      let hit = false;
      for (let i = 4; i < e.pts.length - 1 - 4; i++) {
        if (segRect(e.pts[i], e.pts[i + 1], n)) { hit = true; break; }
      }
      if (hit) { violations.push(`${e.s.slice(0,28)} -> ${e.t.slice(0,28)}  OVER  ${n.id.slice(0,28)}`); break; }
    }
  }
  return { edges: edges.length, nodes: nodes.length, violations };
}
"""


def select_preset(pg):
    sels = pg.locator("select")
    for i in range(sels.count()):
        if any(PRESET in o for o in sels.nth(i).locator("option").all_inner_texts()):
            sels.nth(i).select_option(label=PRESET); return True
    return False


def align(pg):
    pg.get_by_role("button", name="Auto Align Graph").first.click(); time.sleep(3.5)
    pg.locator(".react-flow__controls-fitview").first.click(); time.sleep(1)


def report(pg, tag):
    m = pg.evaluate(VIOLATION_JS)
    print(f"  [{tag}] edges={m['edges']} nodes={m['nodes']}  EDGES-OVER-BOXES = {len(m['violations'])}")
    for v in m['violations'][:15]:
        print(f"      ! {v}")
    return len(m['violations'])


def run_scenario(pg, scenario, shot_path):
    pg.goto(URL, wait_until="networkidle"); time.sleep(2)
    bundle = pg.evaluate(r"""() => Array.from(document.scripts).map(s => s.src).find(s => s.includes('index-')) || ''""")
    select_preset(pg); time.sleep(1.5)
    g = pg.get_by_role("button", name="Generate Architecture")
    if g.count() > 0: g.first.click()
    pg.wait_for_selector(".react-flow__node", timeout=15000); time.sleep(2)
    # Fit view first so the block_group's unpack button isn't under another node.
    pg.locator(".react-flow__controls-fitview").first.click(); time.sleep(0.8)
    if scenario == "folded":
        pass
    elif scenario == "unpack1":
        pg.locator('[data-testid^="unpack-btn-"]').first.click(force=True); time.sleep(0.4)
        inp = pg.locator('[data-testid^="unpack-count-input-"]').first
        inp.click(force=True); inp.fill("1"); time.sleep(0.3)
        pg.locator('[data-testid^="confirm-unpack-n-"]').first.click(force=True); time.sleep(2.5)
    elif scenario == "unpackall":
        pg.locator('[data-testid^="unpack-btn-"]').first.click(force=True); time.sleep(0.4)
        pg.locator('[data-testid^="confirm-unpack-all-"]').first.click(force=True); time.sleep(2.5)
    align(pg)
    pg.screenshot(path=shot_path)
    n = report(pg, scenario)
    return bundle, n


def main():
    scenarios = sys.argv[1:] or ["folded", "unpack1", "unpackall"]
    with sync_playwright() as p:
        b = p.firefox.launch(headless=True)
        pg = b.new_page(viewport={"width": 2400, "height": 1400})
        total = 0
        bundle = ""
        for sc in scenarios:
            bundle, n = run_scenario(pg, sc, f"/tmp/route_{sc}.png")
            total += n
        print(f"\nbundle: {bundle}")
        print(f"TOTAL EDGES-OVER-BOXES across {scenarios} = {total}")
        b.close()


if __name__ == "__main__":
    main()
