"""Run 3: verify FREE-port handle-side freedom on deepseek_v4_flash.

Measures what actually matters:
- Per-node handle sides (count of inputs on left vs right, outputs left vs right).
  If ELK never flipped any, the free-port machinery is idle in practice.
- Per-edge path type: ELK orthogonal (L/Q commands from elkBends) vs bezier
  fallback (C command). Mismatches between port side and chosen handle side
  fall back to bezier — they should be the minority.
- Node overlaps (must stay 0). Crossings reported but only informational
  (straight-line approximation is unfair to orthogonal channel routing).
"""
import os, time
from playwright.sync_api import sync_playwright

RUN = os.path.dirname(os.path.abspath(__file__))
SHOTS = os.path.join(RUN, "screenshots"); os.makedirs(SHOTS, exist_ok=True)
LOG = os.path.join(RUN, "final_script_log.txt")
URL = "http://127.0.0.1:8765"; PRESET = "deepseek_v4_flash"
log_lines = []
def log(m): print(m); log_lines.append(m)
def shot(p, step, name):
    p.screenshot(path=os.path.join(SHOTS, f"final_execution_{step}_{name}.png"))

METRICS = r"""
() => {
  const nodes = Array.from(document.querySelectorAll('.react-flow__node'));
  const list = [];
  nodes.forEach(n => {
    const r = n.getBoundingClientRect();
    if (r.width < 4) return;
    list.push({ id: n.getAttribute('data-id'), x:r.x, y:r.y, w:r.width, h:r.height, el: n });
  });
  let inLeft=0, inRight=0, outLeft=0, outRight=0;
  const flipped = [];
  list.forEach(rec => {
    const tHandle = rec.el.querySelector('.react-flow__handle.target');
    const sHandle = rec.el.querySelector('.react-flow__handle.source');
    const tpos = tHandle ? tHandle.getAttribute('data-handlepos') || '' : '';
    const spos = sHandle ? sHandle.getAttribute('data-handlepos') || '' : '';
    if (tpos === 'right') { inRight++; flipped.push(rec.id + ':in=R'); }
    else if (tpos === 'left') inLeft++;
    if (spos === 'left')  { outLeft++; flipped.push(rec.id + ':out=L'); }
    else if (spos === 'right') outRight++;
  });

  const edgeEls = Array.from(document.querySelectorAll('.react-flow__edge-path'));
  let ortho = 0, bezier = 0;
  edgeEls.forEach(p => {
    const d = p.getAttribute('d') || '';
    if (d.indexOf('C') >= 0) bezier++;
    else if (d.indexOf('L') >= 0 || d.indexOf('Q') >= 0) ortho++;
  });

  const overlaps = [];
  for (let i=0; i<list.length; i++) for (let j=i+1; j<list.length; j++) {
    const a = list[i], b = list[j];
    const ix = Math.max(0, Math.min(a.x+a.w, b.x+b.w) - Math.max(a.x, b.x));
    const iy = Math.max(0, Math.min(a.y+a.h, b.y+b.h) - Math.max(a.y, b.y));
    const area = ix * iy;
    if (area > 64 && area / Math.min(a.w*a.h, b.w*b.h) > 0.03)
      overlaps.push({ a: a.id, b: b.id, area: Math.round(area) });
  }
  return { nodes: list.length, edgeCount: edgeEls.length,
           inLeft, inRight, outLeft, outRight, flipped, ortho, bezier, overlaps };
}
"""


def measure(page, tag):
    m = page.evaluate(METRICS)
    log(f"  [{tag}] nodes={m['nodes']} edges={m['edgeCount']} "
        f"handles in:L={m['inLeft']}/R={m['inRight']}  out:L={m['outLeft']}/R={m['outRight']}  "
        f"edges ortho={m['ortho']} bezier-fallback={m['bezier']}  overlaps={len(m['overlaps'])}")
    if m['flipped']:
        log(f"      flipped sides: {m['flipped']}")
    for ov in m['overlaps'][:6]:
        log(f"      OVERLAP {ov['a']} X {ov['b']} area={ov['area']}")
    return m


def main():
    with sync_playwright() as p:
        b = p.firefox.launch(headless=True)
        pg = b.new_page(viewport={"width": 1920, "height": 1200})
        pg.goto(URL, wait_until="networkidle"); time.sleep(1.5)
        sels = pg.locator("select"); tgt = None
        for i in range(sels.count()):
            if any(PRESET in o for o in sels.nth(i).locator("option").all_inner_texts()):
                tgt = sels.nth(i); break
        log(f"step 1 action: load preset {PRESET}")
        tgt.select_option(label=PRESET); time.sleep(1.0)
        g = pg.get_by_role("button", name="Generate Architecture")
        if g.count() > 0: g.first.click()
        pg.wait_for_selector(".react-flow__node", timeout=15000); time.sleep(1.0)

        log("step 2 action: Auto Align Graph (folded)")
        pg.get_by_role("button", name="Auto Align Graph").first.click(); time.sleep(2.0)
        pg.locator(".react-flow__controls-fitview").first.click(); time.sleep(0.8)
        shot(pg, 1, "folded_aligned"); folded = measure(pg, "folded")

        log("step 3 action: Unpack All -> Auto Align (expanded)")
        pg.locator('[data-testid^="unpack-btn-"]').first.click(); time.sleep(0.4)
        pg.locator('[data-testid^="confirm-unpack-all-"]').first.click(); time.sleep(2.0)
        pg.get_by_role("button", name="Auto Align Graph").first.click(); time.sleep(3.0)
        pg.locator(".react-flow__controls-fitview").first.click(); time.sleep(0.8)
        shot(pg, 2, "unpacked_aligned"); unpacked = measure(pg, "unpacked")

        log("RESULT no_overlap = " +
            ("PASS" if not folded['overlaps'] and not unpacked['overlaps'] else "FAIL"))
        log("RESULT bezier_fallback_is_minority = " +
            ("PASS" if unpacked['bezier'] <= unpacked['ortho'] else "FAIL"))
        b.close()


if __name__ == "__main__":
    try: main()
    finally:
        with open(LOG, "w") as f: f.write("\n".join(log_lines) + "\n")
