import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import time
from playwright.sync_api import sync_playwright
from scripts.colab.browser_utils import launch_stealth_context
from scripts.colab.config import WORKERS

def query_progress():
    print("=================================================================")
    print("STATUS ATUAL DA FROTA DE SELF-PLAY (5 COLABS)")
    print("=================================================================")
    with sync_playwright() as p:
        for wid in [2, 3, 4, 5]:
            port = 9000 + wid
            w = WORKERS[wid]
            try:
                b = p.chromium.connect_over_cdp(f"http://127.0.0.1:{port}")
                ctx = b.contexts[0]
                pg = ctx.pages[0]
                txt = pg.evaluate("""() => {
                    const streams = Array.from(document.querySelectorAll('.output-stream, .output_text, colab-output'));
                    return streams.map(s => s.innerText).join('\\n');
                }""")
                lines = [l.strip() for l in txt.splitlines() if l.strip()]
                prog = [l for l in lines if "progresso:" in l]
                val = prog[-1] if prog else (lines[-1] if lines else "No output")
                print(f"[{w['name']}] (CDP :{port}) -> {val}")
            except Exception as e:
                print(f"[{w['name']}] (CDP :{port}) Error: {e}")

        # Worker 1
        w1 = WORKERS[1]
        try:
            ctx1 = launch_stealth_context(p, w1["profile_dir"], headless=True)
            pg1 = ctx1.pages[0] if ctx1.pages else ctx1.new_page()
            pg1.goto(w1["notebook_url"], timeout=40000)
            time.sleep(5)
            txt1 = pg1.evaluate("""() => {
                const streams = Array.from(document.querySelectorAll('.output-stream, .output_text, colab-output'));
                return streams.map(s => s.innerText).join('\\n');
            }""")
            lines1 = [l.strip() for l in txt1.splitlines() if l.strip()]
            prog1 = [l for l in lines1 if "progresso:" in l]
            val1 = prog1[-1] if prog1 else (lines1[-1] if lines1 else "No output")
            print(f"[{w1['name']}] (Persistent Profile) -> {val1}")
            ctx1.close()
        except Exception as e:
            print(f"[{w1['name']}] Error: {e}")
    print("=================================================================")

if __name__ == "__main__":
    query_progress()
