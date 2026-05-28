import time
from playwright.sync_api import sync_playwright

URL = "http://127.0.0.1:8765"
PRESET = "deepseek_v4_flash"

with sync_playwright() as p:
    b = p.firefox.launch(headless=True)
    pg = b.new_page(viewport={"width": 1920, "height": 1200})
    pg.goto(URL, wait_until="networkidle"); time.sleep(1.5)
    sels = pg.locator("select"); tgt = None
    for i in range(sels.count()):
        if any(PRESET in o for o in sels.nth(i).locator("option").all_inner_texts()):
            tgt = sels.nth(i); break
    tgt.select_option(label=PRESET); time.sleep(1.0)
    g = pg.get_by_role("button", name="Generate Architecture")
    if g.count() > 0: g.first.click()
    pg.wait_for_selector(".react-flow__node", timeout=15000); time.sleep(1.0)
    pg.get_by_role("button", name="Auto Align Graph").first.click(); time.sleep(2.0)
    # unpack all
    pg.locator('[data-testid^="unpack-btn-"]').first.click(); time.sleep(0.4)
    pg.locator('[data-testid^="confirm-unpack-all-"]').first.click(); time.sleep(2.0)
    pg.get_by_role("button", name="Auto Align Graph").first.click(); time.sleep(3.0)
    pg.locator(".react-flow__controls-fitview").first.click(); time.sleep(1.0)
    pg.screenshot(path="explore_unpacked_fit.png")
    print("saved explore_unpacked_fit.png at 1920x1200")
    b.close()
