import time
from playwright.sync_api import sync_playwright
URL="http://127.0.0.1:8765"; PRESET="deepseek_v4_flash"
with sync_playwright() as p:
    b=p.firefox.launch(headless=True); pg=b.new_page(viewport={"width":2400,"height":1400})
    pg.goto(URL,wait_until="networkidle"); time.sleep(2)
    print("bundle:", pg.evaluate(r"""() => Array.from(document.scripts).map(s => s.src).find(s => s.includes('index-')) || ''"""))
    sels=pg.locator("select")
    for i in range(sels.count()):
        if any(PRESET in o for o in sels.nth(i).locator("option").all_inner_texts()):
            sels.nth(i).select_option(label=PRESET); break
    time.sleep(1.5)
    g=pg.get_by_role("button",name="Generate Architecture")
    if g.count()>0: g.first.click()
    pg.wait_for_selector(".react-flow__node",timeout=15000); time.sleep(2)
    pg.locator('[data-testid^="unpack-btn-"]').first.click(); time.sleep(0.4)
    inp = pg.locator('[data-testid^="unpack-count-input-"]').first
    inp.click(); inp.fill("1"); time.sleep(0.3)
    confirm = pg.locator('[data-testid^="confirm-unpack-n-"]').first
    confirm.click(); time.sleep(2.5)
    pg.get_by_role("button",name="Auto Align Graph").first.click(); time.sleep(3.5)
    pg.locator(".react-flow__controls-fitview").first.click(); time.sleep(1)
    pg.screenshot(path="/tmp/snap_unpack1.png")
    # measure edges-through-node-bodies: for each edge segment check if it
    # crosses any node's bounding rect (excluding nodes that are endpoints).
    res = pg.evaluate(r"""
() => {
  const txRe = /translate(?:3d)?\(\s*(-?\d+(?:\.\d+)?)px(?:\s*,\s*(-?\d+(?:\.\d+)?)px)?/;
  const nodes = Array.from(document.querySelectorAll('.react-flow__node')).map(n => {
    const m = (n.style?.transform || '').match(txRe);
    if(!m) return null;
    const r = n.getBoundingClientRect();
    return { id: n.getAttribute('data-id'),
             x: r.x, y: r.y, w: r.width, h: r.height };
  }).filter(Boolean);
  const nodeMap = {}; nodes.forEach(n => nodeMap[n.id] = n);

  const edges = [];
  Array.from(document.querySelectorAll('.react-flow__edge')).forEach(e => {
    const p = e.querySelector('.react-flow__edge-path'); if(!p) return;
    const aria = e.getAttribute('aria-label') || ''; const m = aria.match(/Edge from (.+?) to (.+)/);
    if(!m) return; let L=0; try{L=p.getTotalLength();}catch(e){} if(L<8) return;
    const N=80, pts=[]; for(let i=0;i<=N;i++){const q=p.getPointAtLength(i/N*L); pts.push({x:q.x,y:q.y});}
    edges.push({s:m[1], t:m[2], pts});
  });

  // segment vs box test (Liang-Barsky)
  function segRect(p1, p2, r) {
    let x0=p1.x, y0=p1.y, x1=p2.x, y1=p2.y;
    let dx=x1-x0, dy=y1-y0;
    let t0=0, t1=1;
    const xmin=r.x+2, xmax=r.x+r.w-2, ymin=r.y+2, ymax=r.y+r.h-2;
    const p=[-dx,dx,-dy,dy], q=[x0-xmin, xmax-x0, y0-ymin, ymax-y0];
    for(let i=0;i<4;i++){
      if(p[i]===0){ if(q[i]<0) return false; }
      else { const t=q[i]/p[i];
        if(p[i]<0){ if(t>t1) return false; if(t>t0) t0=t; }
        else { if(t<t0) return false; if(t<t1) t1=t; }}
    }
    return true;
  }

  const violations = [];
  for (const e of edges) {
    for (const n of nodes) {
      if (n.id === e.s || n.id === e.t) continue;
      // skip first/last 4 samples to avoid handle noise
      for (let i=4; i<e.pts.length-1-4; i++) {
        if (segRect(e.pts[i], e.pts[i+1], n)) {
          violations.push(`${e.s.slice(0,30)} -> ${e.t.slice(0,30)}  OVERLAPS  ${n.id.slice(0,30)}`);
          break;
        }
      }
    }
  }
  return { edges: edges.length, nodes: nodes.length, violations };
}
""")
    print(f"edges={res['edges']} nodes={res['nodes']}  EDGES OVER BOXES = {len(res['violations'])}")
    for v in res['violations'][:20]: print("  !", v)
    b.close()
