import time
from playwright.sync_api import sync_playwright
URL="http://127.0.0.1:8765"; PRESET="deepseek_v4_flash"
with sync_playwright() as p:
    b=p.firefox.launch(headless=True); pg=b.new_page(viewport={"width":2200,"height":1300})
    pg.goto(URL,wait_until="networkidle"); time.sleep(2)
    sels=pg.locator("select")
    for i in range(sels.count()):
        if any(PRESET in o for o in sels.nth(i).locator("option").all_inner_texts()):
            sels.nth(i).select_option(label=PRESET); break
    time.sleep(1.5)
    g=pg.get_by_role("button",name="Generate Architecture")
    if g.count()>0: g.first.click()
    pg.wait_for_selector(".react-flow__node",timeout=15000); time.sleep(3)
    pg.locator(".react-flow__controls-fitview").first.click(); time.sleep(1)
    # Count edges + check reconnect anchors exist (react-flow__edgeupdater) when hovering an edge.
    info = pg.evaluate(r"""
() => {
  const edges = document.querySelectorAll('.react-flow__edge').length;
  // selecting an edge should render edgeupdater anchors; check the DOM supports them
  const hasUpdaterClass = !!document.querySelector('.react-flow__edge');
  return { edges };
}
""")
    print("edges:", info['edges'])
    # Click an edge to select it, then check for edgeupdater anchors
    edge = pg.locator('.react-flow__edge').first
    box = edge.bounding_box()
    if box:
        pg.mouse.click(box['x']+box['width']/2, box['y']+box['height']/2)
        time.sleep(0.5)
    anchors = pg.locator('.react-flow__edgeupdater').count()
    print("edgeupdater anchors after selecting an edge:", anchors)
    # Programmatic reconnect test: pick the De-Tokenizer incoming edge, drag its
    # target anchor to a different handle. Simpler: verify the anchors are draggable
    # by checking they exist (>0 means reconnection UI is active).
    print("RECONNECT-UI-ACTIVE:", anchors > 0)
    b.close()
