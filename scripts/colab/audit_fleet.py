import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import time
from playwright.sync_api import sync_playwright
from scripts.colab.browser_utils import launch_stealth_context
from scripts.colab.config import WORKERS

def query_fleet():
    print("=" * 65)
    print("ZQUORIDOR FLEET AUDIT")
    print("=" * 65)
    for wid in [1, 2, 3, 4, 5]:
        w = WORKERS[wid]
        try:
            with sync_playwright() as p:
                ctx = launch_stealth_context(p, w["profile_dir"], headless=True)
                page = ctx.pages[0] if ctx.pages else ctx.new_page()
                page.goto(w["notebook_url"], timeout=45000)
                time.sleep(6)
                
                info = page.evaluate("""() => {
                    const streams = Array.from(document.querySelectorAll('.output-stream, .output_text, colab-output'));
                    const txt = streams.map(s => s.innerText).join('\\n');
                    const toast = document.querySelector('colab-toast, .paper-toast');
                    const toastMsg = toast ? (toast.innerText || '').trim() : '';
                    
                    const btn = document.querySelector('colab-connect-button');
                    const sr = btn ? btn.shadowRoot : null;
                    const connBtn = sr ? sr.querySelector('#connect') : null;
                    const vmStatus = connBtn ? connBtn.innerText.trim() : (btn ? btn.innerText.trim() : '');
                    
                    return {
                        vmStatus,
                        toastMsg,
                        lines: txt.split('\\n').map(l => l.trim()).filter(l => l.length > 0)
                    };
                }""")
                
                lines = info.get("lines", [])
                prog_lines = [l for l in lines if any(k in l for k in ["progresso:", "selfplay", "Launching", "chunk"])]
                last_prog = prog_lines[-1] if prog_lines else (lines[-1] if lines else "No output")
                vm_stat = info.get("vmStatus", "")
                toast = info.get("toastMsg", "")
                toast_str = f" [Toast: {toast}]" if toast else ""
                
                print(f"[{w['name']}] (VM: {vm_stat}) -> {last_prog}{toast_str}", flush=True)
                ctx.close()
        except Exception as exc:
            print(f"[{w['name']}] Query error: {exc}", flush=True)
    print("=" * 65)

if __name__ == "__main__":
    query_fleet()
