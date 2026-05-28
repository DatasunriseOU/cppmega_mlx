import time
from playwright.sync_api import sync_playwright
URL="http://127.0.0.1:8765"; PRESET="deepseek_v4_flash"
with sync_playwright() as p:
    b=p.firefox.launch(headless=True); pg=b.new_page(viewport={"width":2400,"height":1400})
    pg.goto(URL,wait_until="networkidle"); time.sleep(2)
    sels=pg.locator("select")
    for i in range(sels.count()):
        if any(PRESET in o for o in sels.nth(i).locator("option").all_inner_texts()):
            sels.nth(i).select_option(label=PRESET); break
    time.sleep(1.5)
    g=pg.get_by_role("button",name="Generate Architecture")
    if g.count()>0: g.first.click()
    pg.wait_for_selector(".react-flow__node",timeout=15000); time.sleep(2)
    # Unpack PARTIAL — set count to 2 (leaves 2 of 4 folded)
    pg.locator('[data-testid^="unpack-btn-"]').first.click(); time.sleep(0.4)
    inp = pg.locator('[data-testid^="unpack-count-input-"]').first
    inp.click(); inp.fill("2"); time.sleep(0.3)
    confirm = pg.locator('[data-testid^="confirm-unpack-n-"]').first
    confirm.click(); time.sleep(2.5)
    pg.get_by_role("button",name="Auto Align Graph").first.click(); time.sleep(3.5)
    pg.locator(".react-flow__controls-fitview").first.click(); time.sleep(1)
    pg.screenshot(path="/tmp/snap_partial2.png")
    # honest detector
    res = pg.evaluate(r"""
() => {
  const edges = [];
  Array.from(document.querySelectorAll('.react-flow__edge')).forEach(e => {
    const p = e.querySelector('.react-flow__edge-path'); if(!p) return;
    const aria = e.getAttribute('aria-label') || ''; const m = aria.match(/Edge from (.+?) to (.+)/);
    if(!m) return; let L=0; try{L=p.getTotalLength();}catch(e){} if(L<8) return;
    const N=60, pts=[]; for(let i=0;i<=N;i++){const q=p.getPointAtLength(i/N*L); pts.push({x:q.x,y:q.y});}
    edges.push({s:m[1],t:m[2],pts});
  });
  function ccw(a,b,c){return (c.y-a.y)*(b.x-a.x)>(b.y-a.y)*(c.x-a.x);}
  function inter(a,b,c,d){return ccw(a,c,d)!==ccw(b,c,d)&&ccw(a,b,c)!==ccw(a,b,d);}
  const xings = [];
  for(let i=0;i<edges.length;i++)for(let j=i+1;j<edges.length;j++){
    const ei=edges[i],ej=edges[j];
    if(ei.s===ej.s||ei.s===ej.t||ei.t===ej.s||ei.t===ej.t)continue;
    let c=false; outer:for(let a=0;a<ei.pts.length-1;a++)for(let b=0;b<ej.pts.length-1;b++)
      if(inter(ei.pts[a],ei.pts[a+1],ej.pts[b],ej.pts[b+1])){c=true;break outer;}
    if(c) xings.push(`${ei.s.slice(0,32)} -> ${ei.t.slice(0,32)}  ×  ${ej.s.slice(0,32)} -> ${ej.t.slice(0,32)}`);
  }
  const re = /translate(?:3d)?\(\s*(-?\d+(?:\.\d+)?)px(?:\s*,\s*(-?\d+(?:\.\d+)?)px)?/;
  const nodes = Array.from(document.querySelectorAll('.react-flow__node')).map(n => {
    const m = (n.style?.transform || '').match(re);
    if(!m) return null;
    return { id: n.getAttribute('data-id'), x: parseFloat(m[1]), y: m[2]? parseFloat(m[2]) : 0 };
  }).filter(Boolean).sort((a,b) => a.y - b.y || a.x - b.x);
  return { edges: edges.length, xings, nodes };
}
""")
    print(f"PARTIAL UNPACK (2 of 4): edges={res['edges']}  REAL crossings = {len(res['xings'])}")
    print("\n-- node positions --")
    last_y = None
    for n in res['nodes']:
        if last_y is not None and abs(n['y'] - last_y) > 80: print(f"  --- y gap to {n['y']:.0f} ---")
        print(f"  {n['id'][:40]:40s}  x={n['x']:7.0f}  y={n['y']:6.0f}")
        last_y = n['y']
    print("\n-- crossings --")
    for x in res['xings'][:20]: print(f"  X  {x}")
    b.close()
