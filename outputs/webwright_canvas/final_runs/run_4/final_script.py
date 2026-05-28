"""Run 4: verify row-wrap (snake) + stale-bend fallback on drag.

CP1 unpacked deepseek_v4_flash wraps into >=2 rows when the graph is wider than
    the canvas (uses vertical space instead of running off the right edge).
CP2 odd rows flow right-to-left: their nodes have input handle on the right
    and output handle on the left.
CP3 row widths each <= canvas budget.
CP4 0 node overlaps in the wrapped layout.
CP5 after dragging a node, its connected edges' SVG path becomes bezier
    (contains 'C' command) — the stale ELK bends are correctly invalidated.
"""
import os, time
from playwright.sync_api import sync_playwright

RUN = os.path.dirname(os.path.abspath(__file__))
SHOTS = os.path.join(RUN, "screenshots"); os.makedirs(SHOTS, exist_ok=True)
LOG = os.path.join(RUN, "final_script_log.txt")
URL = "http://127.0.0.1:8765"
PRESET = "deepseek_v4_flash"
VIEW = {"width": 1440, "height": 900}  # narrower so wrap is triggered

log_lines = []
def log(m): print(m); log_lines.append(m)
def shot(p, step, name): p.screenshot(path=os.path.join(SHOTS, f"final_execution_{step}_{name}.png"))

WRAP_METRICS = r"""
() => {
  const nodes = Array.from(document.querySelectorAll('.react-flow__node'));
  const list = [];
  nodes.forEach(n => {
    const r = n.getBoundingClientRect();
    if (r.width < 4) return;
    const t = n.querySelector('.react-flow__handle.target');
    const s = n.querySelector('.react-flow__handle.source');
    list.push({
      id: n.getAttribute('data-id'),
      x: r.x, y: r.y, w: r.width, h: r.height,
      tpos: t ? t.getAttribute('data-handlepos') || '' : '',
      spos: s ? s.getAttribute('data-handlepos') || '' : '',
    });
  });

  // Cluster nodes into rows by y-band (tolerance = node height).
  const byY = [...list].sort((a, b) => a.y - b.y);
  const rows = [];
  for (const n of byY) {
    const last = rows[rows.length - 1];
    if (last && Math.abs(n.y - last[0].y) < Math.max(60, n.h * 0.6)) last.push(n);
    else rows.push([n]);
  }
  const rowSummary = rows.map((row, i) => {
    const minX = Math.min(...row.map(n => n.x));
    const maxR = Math.max(...row.map(n => n.x + n.w));
    const inR = row.filter(n => n.tpos === 'right').length;
    const inL = row.filter(n => n.tpos === 'left').length;
    const outR = row.filter(n => n.spos === 'right').length;
    const outL = row.filter(n => n.spos === 'left').length;
    return { row: i, n: row.length, minX, maxR, width: Math.round(maxR - minX),
             inL, inR, outL, outR };
  });

  // Overlaps
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

# Drag a brick node, then read edge path 'd' attributes connected to it.
DRAG_PROBE = r"""
(nodeId) => {
  const node = document.querySelector(`[data-id="${nodeId}"]`);
  if (!node) return { error: 'node-not-found' };
  // Find edges referencing this node.
  const edges = Array.from(document.querySelectorAll('.react-flow__edge'));
  const related = edges.filter(e => {
    const s = e.getAttribute('data-source') || ''; const t = e.getAttribute('data-target') || '';
    return s === nodeId || t === nodeId;
  });
  const summary = related.map(e => {
    const p = e.querySelector('.react-flow__edge-path');
    const d = p ? p.getAttribute('d') || '' : '';
    return { id: e.getAttribute('data-id'), hasC: d.indexOf('C') >= 0,
             hasL: d.indexOf('L') >= 0 || d.indexOf('Q') >= 0,
             preview: d.slice(0, 60) };
  });
  return { count: related.length, summary };
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
            log(f"    row {r['row']}: n={r['n']} width={r['width']}px "
                f"in:L={r['inL']}/R={r['inR']} out:L={r['outL']}/R={r['outR']}")

        cp1 = len(m['rows']) >= 2
        cp4 = len(m['overlaps']) == 0
        # CP2: odd rows should have input-Right majority and output-Left majority
        odd_rows = [r for r in m['rows'] if r['row'] % 2 == 1]
        cp2 = all(r['inR'] >= r['inL'] and r['outL'] >= r['outR'] for r in odd_rows) if odd_rows else False
        cp3 = all(r['width'] <= VIEW['width'] for r in m['rows'])
        log(f"RESULT CP1 wrapped_multirow = {'PASS' if cp1 else 'FAIL'} ({len(m['rows'])} rows)")
        log(f"RESULT CP2 odd_rows_R_to_L  = {'PASS' if cp2 else 'FAIL'}")
        log(f"RESULT CP3 each_row_fits    = {'PASS' if cp3 else 'FAIL'}")
        log(f"RESULT CP4 no_overlap       = {'PASS' if cp4 else 'FAIL'}")

        # CP5: drag a brick and verify edges fall back to bezier (C command)
        bricks = pg.locator('.react-flow__node[data-id^="rmsnorm_unpacked"]')
        if bricks.count() == 0:
            bricks = pg.locator('.react-flow__node[data-id^="lightning"]')
        target_id = bricks.first.get_attribute("data-id") if bricks.count() > 0 else None
        log(f"step 3 action: drag node {target_id} 180px down to detach from its ELK route")
        if target_id:
            box = bricks.first.bounding_box()
            pg.mouse.move(box['x'] + box['width']/2, box['y'] + box['height']/2)
            pg.mouse.down()
            pg.mouse.move(box['x'] + box['width']/2 + 30, box['y'] + box['height']/2 + 180, steps=10)
            pg.mouse.up(); time.sleep(0.6)
            shot(pg, 2, "after_drag")
            probe = pg.evaluate(DRAG_PROBE, target_id)
            log(f"  edges touching dragged node: {probe['count']}")
            for s in probe['summary']:
                log(f"    {s['id']}: hasC={s['hasC']} hasL/Q={s['hasL']} d='{s['preview']}...'")
            cp5 = probe['count'] > 0 and all(s['hasC'] for s in probe['summary'])
            log(f"RESULT CP5 drag_falls_back_to_bezier = {'PASS' if cp5 else 'FAIL'}")
        b.close()


if __name__ == "__main__":
    try: main()
    finally:
        with open(LOG, "w") as f: f.write("\n".join(log_lines) + "\n")
