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
    pg.get_by_role("button",name="Auto Align Graph").first.click(); time.sleep(3.5)
    pg.locator(".react-flow__controls-fitview").first.click(); time.sleep(1)
    pg.screenshot(path="/tmp/snap_folded_v2.png")
    data = pg.evaluate(r"""
() => {
  const re = /translate(?:3d)?\(\s*(-?\d+(?:\.\d+)?)px(?:\s*,\s*(-?\d+(?:\.\d+)?)px)?/;
  const nodes = Array.from(document.querySelectorAll('.react-flow__node')).map(n => {
    const m = (n.style?.transform || '').match(re);
    if(!m) return null;
    const r = n.getBoundingClientRect();
    return { id: n.getAttribute('data-id'),
             x: parseFloat(m[1]), y: m[2]?parseFloat(m[2]):0,
             w: Math.round(r.width), h: Math.round(r.height) };
  }).filter(Boolean).sort((a,b)=>a.y-b.y||a.x-b.x);
  return nodes;
}
""")
    for n in data:
        print(f"  {n['id'][:35]:35s}  x=[{n['x']:6.0f}..{n['x']+n['w']:6.0f}]  y={n['y']:6.0f}")
    b.close()
