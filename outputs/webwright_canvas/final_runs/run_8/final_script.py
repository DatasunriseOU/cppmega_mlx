"""Run 8: better crossing detector — interior segments only.

Run 7's filter "skip pairs that share any endpoint node" was wrong for residual
DAGs where add nodes are hubs (many edges meet there). It hid real crossings
between skip edges that legitimately share a hub. Here we test EVERY pair but
only on INTERIOR polyline segments (drop the first and last 3 segments of each
edge), so handles meeting at a node don't cause false positives.
"""
import os, time
from playwright.sync_api import sync_playwright

RUN = os.path.dirname(os.path.abspath(__file__))
SHOTS = os.path.join(RUN, "screenshots"); os.makedirs(SHOTS, exist_ok=True)
LOG = os.path.join(RUN, "final_script_log.txt")
URL = "http://127.0.0.1:8765"
PRESET = "deepseek_v4_flash"
VIEW = {"width": 1920, "height": 1200}

log_lines = []
def log(m): print(m); log_lines.append(m)
def shot(p, step, name): p.screenshot(path=os.path.join(SHOTS, f"final_execution_{step}_{name}.png"))

CROSSING_JS = r"""
() => {
  const edgeEls = Array.from(document.querySelectorAll('.react-flow__edge'));
  const edges = [];
  edgeEls.forEach(e => {
    const path = e.querySelector('.react-flow__edge-path');
    if (!path) return;
    const aria = e.getAttribute('aria-label') || '';
    const m = aria.match(/Edge from (.+?) to (.+)/);
    if (!m) return;
    let totalLen = 0;
    try { totalLen = path.getTotalLength(); } catch (err) { return; }
    if (!isFinite(totalLen) || totalLen < 8) return;
    const N = 60;
    const pts = [];
    for (let i = 0; i <= N; i++) {
      const p = path.getPointAtLength((i / N) * totalLen);
      pts.push({ x: p.x, y: p.y });
    }
    edges.push({ src: m[1], tgt: m[2], pts });
  });

  function ccw(ax, ay, bx, by, cx, cy) {
    return (cy - ay) * (bx - ax) > (by - ay) * (cx - ax);
  }
  function segInter(a, b, c, d) {
    return ccw(a.x, a.y, c.x, c.y, d.x, d.y) !== ccw(b.x, b.y, c.x, c.y, d.x, d.y)
        && ccw(a.x, a.y, b.x, b.y, c.x, c.y) !== ccw(a.x, a.y, b.x, b.y, d.x, d.y);
  }

  // Use interior segments only: drop first and last EDGE_SKIP segments of each
  // edge polyline, so meeting at a shared handle doesn't trigger a crossing.
  const EDGE_SKIP = 6;
  const crossings = [];
  for (let i = 0; i < edges.length; i++) {
    const ei = edges[i];
    for (let j = i + 1; j < edges.length; j++) {
      const ej = edges[j];
      if (ei.src === ej.src && ei.tgt === ej.tgt) continue; // duplicate
      let crossed = false;
      outer:
      for (let a = EDGE_SKIP; a < ei.pts.length - 1 - EDGE_SKIP; a++) {
        for (let b = EDGE_SKIP; b < ej.pts.length - 1 - EDGE_SKIP; b++) {
          if (segInter(ei.pts[a], ei.pts[a+1], ej.pts[b], ej.pts[b+1])) {
            crossed = true;
            break outer;
          }
        }
      }
      if (crossed) crossings.push({ e1: `${ei.src} -> ${ei.tgt}`, e2: `${ej.src} -> ${ej.tgt}` });
    }
  }
  return { edges: edges.length, crossings };
}
"""


def measure(page, tag):
    m = page.evaluate(CROSSING_JS)
    log(f"  [{tag}] {m['edges']} edges, crossings (interior) = {len(m['crossings'])}")
    for c in m['crossings'][:15]:
        log(f"    X  {c['e1'][:55]:55s}  ×  {c['e2'][:55]}")
    if len(m['crossings']) > 15:
        log(f"    ...+{len(m['crossings']) - 15} more")
    return m


def main():
    with sync_playwright() as p:
        b = p.firefox.launch(headless=True)
        pg = b.new_page(viewport=VIEW)
        pg.goto(URL, wait_until="networkidle"); time.sleep(2)
        bundle = pg.evaluate(r"""() => Array.from(document.scripts).map(s => s.src).find(s => s.includes('index-')) || ''""")
        log(f"bundle: {bundle}")
        sels = pg.locator("select"); tgt = None
        for i in range(sels.count()):
            if any(PRESET in o for o in sels.nth(i).locator("option").all_inner_texts()):
                tgt = sels.nth(i); break
        log(f"step 1 action: load preset {PRESET}")
        tgt.select_option(label=PRESET); time.sleep(1.5)
        g = pg.get_by_role("button", name="Generate Architecture")
        if g.count() > 0: g.first.click()
        pg.wait_for_selector(".react-flow__node", timeout=15000); time.sleep(2)

        log("step 2 action: Auto Align Graph (folded)")
        pg.get_by_role("button", name="Auto Align Graph").first.click(); time.sleep(3)
        pg.locator(".react-flow__controls-fitview").first.click(); time.sleep(1)
        shot(pg, 1, "folded")
        folded = measure(pg, "folded")

        log("step 3 action: Unpack All, then Auto Align Graph")
        pg.locator('[data-testid^="unpack-btn-"]').first.click(); time.sleep(0.5)
        pg.locator('[data-testid^="confirm-unpack-all-"]').first.click(); time.sleep(2.5)
        pg.get_by_role("button", name="Auto Align Graph").first.click(); time.sleep(4)
        pg.locator(".react-flow__controls-fitview").first.click(); time.sleep(1)
        shot(pg, 2, "unpacked")
        unpacked = measure(pg, "unpacked")

        log("RESULT folded   crossings = " + str(len(folded['crossings'])))
        log("RESULT unpacked crossings = " + str(len(unpacked['crossings'])))
        b.close()


if __name__ == "__main__":
    try: main()
    finally:
        with open(LOG, "w") as f: f.write("\n".join(log_lines) + "\n")
