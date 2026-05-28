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
    zin=pg.locator(".react-flow__controls-zoomin").first
    zin.click();time.sleep(0.3);zin.click();time.sleep(0.6)
    pg.screenshot(path="/tmp/route_folded_tb.png")
    print("saved")
    b.close()
