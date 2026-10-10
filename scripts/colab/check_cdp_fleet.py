import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from playwright.sync_api import sync_playwright
from scripts.colab.config import WORKERS

def check_all_cdp():
    print("=" * 65)
    print("ALL 5 WORKERS LIVE CDP PROGRESS REPORT")
    print("=" * 65)
    with sync_playwright() as p:
        for wid in [1, 2, 3, 4, 5]:
            port = 9000 + wid
            w = WORKERS[wid]
            try:
                b = p.chromium.connect_over_cdp(f"http://127.0.0.1:{port}")
                ctx = b.contexts[0]
                pages = [pg for pg in ctx.pages if "colab.research.google.com" in pg.url]
                if not pages:
                    print(f"[{w['name']}] (Port {port}): No Colab page")
                    continue
                pg = pages[0]
                txt = pg.evaluate("""() => {
                    const streams = Array.from(document.querySelectorAll('.output-stream, .output_text, colab-output'));
                    return streams.map(s => s.innerText).join('\\n');
                }""")
                lines = [l.strip() for l in txt.splitlines() if l.strip()]
                prog = [l for l in lines if "progresso:" in l]
                val = prog[-1] if prog else (lines[-1] if lines else "No output")
                print(f"[{w['name']}] (Port {port}) -> {val}", flush=True)
            except Exception as e:
                print(f"[{w['name']}] (Port {port}) Error: {e}", flush=True)
    print("=" * 65)

if __name__ == "__main__":
    check_all_cdp()
