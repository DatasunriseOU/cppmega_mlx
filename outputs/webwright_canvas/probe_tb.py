import time
from playwright.sync_api import sync_playwright
URL="http://127.0.0.1:8765"; PRESET="deepseek_v4_flash"
with sync_playwright() as p:
    b=p.firefox.launch(headless=True); pg=b.new_page(viewport={"width":2400,"height":1500})
    pg.goto(URL,wait_until="networkidle"); time.sleep(2)
    sels=pg.locator("select")
    for i in range(sels.count()):
        if any(PRESET in o for o in sels.nth(i).locator("option").all_inner_texts()):
            sels.nth(i).select_option(label=PRESET); break
    time.sleep(1.5)
    g=pg.get_by_role("button",name="Generate Architecture")
    if g.count()>0: g.first.click()
    pg.wait_for_selector(".react-flow__node",timeout=15000); time.sleep(2)
    pg.locator(".react-flow__controls-fitview").first.click(); time.sleep(0.8)
    pg.locator('[data-testid^="unpack-btn-"]').first.click(force=True); time.sleep(0.4)
    pg.locator('[data-testid^="confirm-unpack-all-"]').first.click(force=True); time.sleep(2.5)
    pg.get_by_role("button",name="Auto Align Graph").first.click(); time.sleep(4)
    pg.locator(".react-flow__controls-fitview").first.click(); time.sleep(1)
    pg.screenshot(path="/tmp/route_tb.png")
    res=pg.evaluate(r"""
() => {
  const txRe=/translate(?:3d)?\(\s*(-?\d+(?:\.\d+)?)px(?:\s*,\s*(-?\d+(?:\.\d+)?)px)?/;
  const rects={},addIds=new Set();
  document.querySelectorAll('.react-flow__node').forEach(n=>{
    const m=(n.style.transform||'').match(txRe); if(!m)return;
    const id=n.getAttribute('data-id');
    rects[id]={x:parseFloat(m[1]),y:m[2]?parseFloat(m[2]):0,w:n.offsetWidth,h:n.offsetHeight};
    if(n.className.includes('node-residual_add'))addIds.add(id);
  });
  const per={};
  document.querySelectorAll('.react-flow__edge').forEach(e=>{
    const al=e.getAttribute('aria-label')||'';const m=al.match(/Edge from (.+?) to (.+)/);if(!m)return;
    if(!addIds.has(m[2]))return;
    const p=e.querySelector('.react-flow__edge-path');if(!p)return;let L=0;try{L=p.getTotalLength();}catch(e){return;}
    const end=p.getPointAtLength(L);const r=rects[m[2]];if(!r)return;
    const dl=Math.abs(end.x-r.x),dr=Math.abs(end.x-(r.x+r.w)),dt=Math.abs(end.y-r.y),db=Math.abs(end.y-(r.y+r.h));
    const mn=Math.min(dl,dr,dt,db);const side=mn===dt?'top':mn===db?'bottom':mn===dl?'left':'right';
    per[m[2]]=per[m[2]]||{top:0,right:0,bottom:0,left:0};per[m[2]][side]++;
  });
  let tb=0,tot=0;
  for(const k in per){tot++;const v=per[k];if(v.top>0&&v.bottom>0)tb++;}
  return {count:addIds.size,withInputs:tot,topAndBottom:tb,sample:Object.entries(per).slice(0,5)};
}
""")
    print("residual_add nodes:",res['count'],"with inputs measured:",res['withInputs'])
    print("with BOTH top+bottom inputs:",res['topAndBottom'])
    for k,v in res['sample']: print("  ",k[:38],v)
    b.close()
