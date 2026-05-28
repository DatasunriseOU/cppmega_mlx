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
    before = pg.locator('.react-flow__edge').count()
    # Select first edge, grab one of its updater anchors, drag onto its target node top handle.
    edge = pg.locator('.react-flow__edge').first
    box = edge.bounding_box()
    pg.mouse.click(box['x']+box['width']/2, box['y']+box['height']/2); time.sleep(0.4)
    anchors = pg.locator('.react-flow__edgeupdater')
    n = anchors.count()
    moved = False
    if n >= 1:
        a = anchors.first.bounding_box()
        # drag the anchor a bit (just to exercise the reconnect drag path); drop near a node center
        node = pg.locator('.react-flow__node').nth(1).bounding_box()
        pg.mouse.move(a['x']+a['width']/2, a['y']+a['height']/2)
        pg.mouse.down()
        pg.mouse.move(node['x']+node['width']/2, node['y']+5, steps=15)  # near top edge of a node
        pg.mouse.up(); time.sleep(0.6)
        moved = True
    after = pg.locator('.react-flow__edge').count()
    print(f"edges before={before} after={after}  anchors={n}  dragPerformed={moved}")
    print("RESULT edges_survived_drag:", after >= before-1)
    pg.screenshot(path="/tmp/route_geo_folded.png")
    b.close()
