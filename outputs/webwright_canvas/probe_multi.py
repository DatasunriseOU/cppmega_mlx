import time, sys
from playwright.sync_api import sync_playwright
URL="http://127.0.0.1:8765"
PRESETS = ["gemma4","llama3_8b","deepseek_v3","gpt_oss_20b","minimax_m2"]
VIOLATION_JS = open("probe_routing.py").read().split('VIOLATION_JS = r"""')[1].split('"""')[0]
def measure(pg):
    return pg.evaluate("()=>{"+VIOLATION_JS.split("() => {",1)[1])
with sync_playwright() as p:
    b=p.firefox.launch(headless=True); pg=b.new_page(viewport={"width":2400,"height":1400})
    grand=0
    for preset in PRESETS:
        for scen in ["folded","unpackall"]:
            pg.goto(URL,wait_until="networkidle"); time.sleep(1.5)
            sels=pg.locator("select"); ok=False
            for i in range(sels.count()):
                if any(preset==o.strip() for o in sels.nth(i).locator("option").all_inner_texts()):
                    sels.nth(i).select_option(label=preset); ok=True; break
            if not ok: print(f"  {preset}: NOT FOUND"); continue
            time.sleep(1.2)
            g=pg.get_by_role("button",name="Generate Architecture")
            if g.count()>0: g.first.click()
            try: pg.wait_for_selector(".react-flow__node",timeout=15000)
            except: print(f"  {preset}/{scen}: no nodes"); continue
            time.sleep(1.5)
            if scen=="unpackall":
                ub=pg.locator('[data-testid^="unpack-btn-"]')
                if ub.count()>0:
                    ub.first.click(); time.sleep(0.4)
                    ua=pg.locator('[data-testid^="confirm-unpack-all-"]')
                    if ua.count()>0: ua.first.click(); time.sleep(2.2)
            pg.get_by_role("button",name="Auto Align Graph").first.click(); time.sleep(4)
            m=measure(pg)
            grand+=len(m['violations'])
            print(f"  {preset:16s}/{scen:9s}: edges={m['edges']:3d} nodes={m['nodes']:3d}  OVER-BOXES={len(m['violations'])}")
            for v in m['violations'][:4]: print(f"        ! {v}")
    print(f"\nGRAND TOTAL edges-over-boxes = {grand}")
    b.close()
