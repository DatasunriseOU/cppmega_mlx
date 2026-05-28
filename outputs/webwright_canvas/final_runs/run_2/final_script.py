"""Webwright run 2: verify ELK layout on the UNPACKED deepseek_v4_flash graph.

This is the case the user flagged as a tangled mess (screenshot 5) and as
having crossing connector lines (screenshot 3). We measure BOTH node overlaps
and (approx) edge crossings in the folded AND fully-unpacked states.

CP4  no node overlaps (folded + unpacked)
CP6  edge crossings drastically reduced vs a snake layout (report counts;
     unpacked residual DAG should be a clean left-to-right layering)
CP7  unpack modal exposes +/- steppers and a numeric count input
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


# Collect node rects + approximate edge crossings. Each edge is approximated as
# a straight segment from the right-center of its source node to the left-center
# of its target node (handles are pinned Left-in / Right-out by the ELK layout).
METRICS_JS = r"""
() => {
  const nodeEls = Array.from(document.querySelectorAll('.react-flow__node'));
  const rects = {};
  const recList = [];
  nodeEls.forEach(n => {
    const id = n.getAttribute('data-id');
    const r = n.getBoundingClientRect();
    if (r.width < 4 || r.height < 4) return;
    const rec = { id, x: r.x, y: r.y, w: r.width, h: r.height,
                  label: (n.innerText||'').replace(/\s+/g,' ').trim().slice(0,30) };
    rects[id] = rec; recList.push(rec);
  });

  // node overlaps
  const overlaps = [];
  for (let i=0;i<recList.length;i++) for (let j=i+1;j<recList.length;j++){
    const a=recList[i], b=recList[j];
    const ix=Math.max(0,Math.min(a.x+a.w,b.x+b.w)-Math.max(a.x,b.x));
    const iy=Math.max(0,Math.min(a.y+a.h,b.y+b.h)-Math.max(a.y,b.y));
    const area=ix*iy;
    if(area>64 && area/Math.min(a.w*a.h,b.w*b.h)>0.03)
      overlaps.push({a:a.id,b:b.id,area:Math.round(area)});
  }

  // edges -> segments (source right-center -> target left-center)
  const edgeEls = Array.from(document.querySelectorAll('.react-flow__edge'));
  const segs = [];
  edgeEls.forEach(e=>{
    const id=e.getAttribute('data-id')||'';
    // data-id format: "reactflow__edge-<source><sourceHandle>-<target><targetHandle>" varies;
    // fall back to aria via testid attributes if present
    const s=e.getAttribute('data-source'), t=e.getAttribute('data-target');
    let src=s, tgt=t;
    if((!src||!tgt) && id.includes('->')){ const m=id.split('->'); src=m[0]; tgt=m[1]; }
    if(!rects[src]||!rects[tgt]) return;
    const A=rects[src], B=rects[tgt];
    segs.push({src,tgt,
      x1:A.x+A.w, y1:A.y+A.h/2,
      x2:B.x,      y2:B.y+B.h/2});
  });
  function ccw(ax,ay,bx,by,cx,cy){return (cy-ay)*(bx-ax)>(by-ay)*(cx-ax);}
  function inter(s1,s2){
    if(s1.src===s2.src||s1.src===s2.tgt||s1.tgt===s2.src||s1.tgt===s2.tgt) return false;
    return ccw(s1.x1,s1.y1,s2.x1,s2.y1,s2.x2,s2.y2)!==ccw(s1.x2,s1.y2,s2.x1,s2.y1,s2.x2,s2.y2)
        && ccw(s1.x1,s1.y1,s1.x2,s1.y2,s2.x1,s2.y1)!==ccw(s1.x1,s1.y1,s1.x2,s1.y2,s2.x2,s2.y2);
  }
  let crossings=0;
  for(let i=0;i<segs.length;i++) for(let j=i+1;j<segs.length;j++) if(inter(segs[i],segs[j])) crossings++;

  return { nodes: recList.length, edges: segs.length, overlaps, crossings };
}
"""


def measure(page, tag):
    m = page.evaluate(METRICS_JS)
    log(f"  [{tag}] nodes={m['nodes']} edges={m['edges']} "
        f"node_overlaps={len(m['overlaps'])} edge_crossings={m['crossings']}")
    for ov in m["overlaps"][:8]:
        log(f"      OVERLAP {ov['a']} X {ov['b']} area={ov['area']}")
    return m


def main():
    with sync_playwright() as p:
        browser = p.firefox.launch(headless=True)
        page = browser.new_page(viewport={"width": 1280, "height": 1800})
        page.goto(URL, wait_until="networkidle")
        time.sleep(1.5)

        # load preset
        selects = page.locator("select")
        target = None
        for i in range(selects.count()):
            if any(PRESET in o for o in selects.nth(i).locator("option").all_inner_texts()):
                target = selects.nth(i); break
        assert target is not None
        log(f"step 1 action: load preset {PRESET}")
        target.select_option(label=PRESET)
        time.sleep(1.0)
        gen = page.get_by_role("button", name="Generate Architecture")
        if gen.count() > 0:
            gen.first.click()
        page.wait_for_selector(".react-flow__node", timeout=15000)
        time.sleep(1.0)

        log("step 2 action: Auto Align Graph (folded), then measure")
        page.get_by_role("button", name="Auto Align Graph").first.click()
        time.sleep(2.0)  # ELK async + deferred remeasure pass
        page.locator(".react-flow__controls-fitview").first.click()
        time.sleep(0.8)
        shot(page, 1, "folded_aligned")
        folded = measure(page, "folded")

        # CP7: open unpack modal, assert stepper + numeric input exist
        ubtn = page.locator('[data-testid^="unpack-btn-"]').first
        ubtn.click()
        time.sleep(0.5)
        has_inc = page.locator('[data-testid^="unpack-inc-"]').count() > 0
        has_dec = page.locator('[data-testid^="unpack-dec-"]').count() > 0
        has_num = page.locator('[data-testid^="unpack-count-input-"]').count() > 0
        log(f"step 3 action: unpack modal -> stepper+ ={has_inc} stepper- ={has_dec} numeric_input={has_num}")
        shot(page, 2, "unpack_modal")

        # CP6/CP4 unpacked: Unpack All -> ELK relayout the expanded residual DAG
        log("step 4 action: Unpack All, then Auto Align Graph on expanded graph")
        page.locator('[data-testid^="confirm-unpack-all-"]').first.click()
        time.sleep(2.0)
        page.get_by_role("button", name="Auto Align Graph").first.click()
        time.sleep(2.5)
        page.locator(".react-flow__controls-fitview").first.click()
        time.sleep(0.8)
        shot(page, 3, "unpacked_aligned")
        unpacked = measure(page, "unpacked")

        # verdicts
        cp4 = len(folded["overlaps"]) == 0 and len(unpacked["overlaps"]) == 0
        cp7 = has_inc and has_dec and has_num
        log("RESULT CP4 no_overlap (folded+unpacked) = " + ("PASS" if cp4 else "FAIL"))
        log("RESULT CP7 unpack_stepper+numeric        = " + ("PASS" if cp7 else "FAIL"))
        log(f"FINAL: folded(nodes={folded['nodes']},xings={folded['crossings']},ovl={len(folded['overlaps'])}) "
            f"unpacked(nodes={unpacked['nodes']},xings={unpacked['crossings']},ovl={len(unpacked['overlaps'])})")
        browser.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        with open(LOG, "w") as f:
            f.write("\n".join(log_lines) + "\n")
