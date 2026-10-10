import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import time
from playwright.sync_api import sync_playwright
from scripts.colab.browser_utils import launch_stealth_context, get_notebook_dom_state
from scripts.colab.config import WORKERS

def check_all():
    for wid in [1, 2, 3, 4, 5]:
        w = WORKERS[wid]
        try:
            with sync_playwright() as p:
                ctx = launch_stealth_context(p, w["profile_dir"], headless=True)
                page = ctx.pages[0] if ctx.pages else ctx.new_page()
                page.goto(w["notebook_url"], timeout=30000)
                time.sleep(4)
                st = get_notebook_dom_state(page, ["selfplay", "progresso", "zquoridor"])
                lines = [l.strip() for l in st["outText"].splitlines() if l.strip()]
                prog = [l for l in lines if "progresso:" in l]
                prog_str = prog[-1] if prog else (lines[-1] if lines else "No output")
                print(f"[{w['name']}] RUNNING: {st['running']} (VM: {st['statusText']}) -> {prog_str[:80]}", flush=True)
                ctx.close()
        except Exception as exc:
            print(f"[{w['name']}] Error checking: {exc}", flush=True)

if __name__ == "__main__":
    check_all()
