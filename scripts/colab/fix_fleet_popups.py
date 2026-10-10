import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import time
from playwright.sync_api import sync_playwright

from scripts.colab.browser_utils import solve_recaptcha_playwright


def handle_oauth(ctx):
    oauth_pages = [pg for pg in ctx.pages if "accounts.google.com" in (pg.url or "")]
    if not oauth_pages:
        return False
    print(f"  Handling {len(oauth_pages)} OAuth popups...")
    # Close older duplicate tabs
    for p in oauth_pages[:-1]:
        try:
            p.close()
        except Exception:
            pass

    pg = oauth_pages[-1]
    try:
        checkboxes = pg.locator("input[type='checkbox']").all()
        for cb in checkboxes:
            if not cb.is_checked():
                cb.check()
                time.sleep(0.5)

        btn = pg.locator("#submit_approve_access, button:has-text('Allow'), button:has-text('Permitir'), button:has-text('Continuar'), button:has-text('Avançar')")
        if btn.count() > 0 and btn.first.is_visible():
            print("  Clicking OAuth approve button...")
            btn.first.click()
            time.sleep(3)
            return True
    except Exception as exc:
        print(f"  OAuth error: {exc}")
    return False


def solve_recaptcha(page):
    try:
        anchor_frame = next((f for f in page.frames if "anchor" in f.url and "k=6LfQttQU" in f.url), None)
        if anchor_frame:
            anchor = anchor_frame.locator("#recaptcha-anchor")
            if anchor.count() > 0:
                checked = anchor.get_attribute("aria-checked")
                if checked == "true":
                    return True
                print("  Clicking reCAPTCHA anchor via JS...")
                anchor.evaluate("el => el.click()")
                time.sleep(3)
                if anchor.get_attribute("aria-checked") == "true":
                    print("  reCAPTCHA checked successfully!")
                    return True
        bframe = next((f for f in page.frames if "bframe" in f.url), None)
        if bframe:
            print("  bframe present. Solving via audio/Whisper...")
            return solve_recaptcha_playwright(page)
    except Exception as exc:
        print(f"  reCAPTCHA error: {exc}")
    return False


def click_run(page):
    try:
        page.evaluate("""() => {
            const nb = globalThis.colab?.global?.notebook;
            const cell = nb?.cells ? nb.cells[0] : null;
            if (cell) {
                const elem = cell.getElement ? cell.getElement() : (cell.element_ || cell.dom_);
                const btn = elem ? elem.querySelector('colab-run-button') : null;
                const inner = btn && btn.shadowRoot ? btn.shadowRoot.querySelector('#run-button, button') : null;
                if (inner) inner.click();
                else if (typeof cell.manualExecute === 'function') cell.manualExecute();
            }
        }""")
    except Exception:
        pass


def main():
    print("=======================================================")
    print("RESOLVING FLEET POPUPS AND ENSURING EXECUTION")
    print("=======================================================")
    with sync_playwright() as p:
        for wid in [1, 2, 3, 4, 5]:
            port = 9000 + wid
            print(f"\n--- Checking Worker {wid} (Port {port}) ---")
            try:
                b = p.chromium.connect_over_cdp(f"http://127.0.0.1:{port}")
                ctx = b.contexts[0]
                
                # 1. Handle OAuth popup if any
                handle_oauth(ctx)
                
                # 2. Get Colab page
                page = next((pg for pg in ctx.pages if "colab.research.google.com" in (pg.url or "")), None)
                if not page:
                    print("  No colab page found!")
                    continue
                
                # 3. Handle reCAPTCHA if any
                solve_recaptcha(page)
                
                # 4. Check if cell 0 is running
                st = page.evaluate("""() => {
                    const nb = globalThis.colab?.global?.notebook;
                    const cell = nb?.cells ? nb.cells[0] : null;
                    return cell && cell.isRunning ? cell.isRunning() : false;
                }""")
                print(f"  Worker {wid} running: {st}")
                if not st:
                    print("  Triggering run on Cell 0...")
                    click_run(page)
                    time.sleep(3)
                    st2 = page.evaluate("""() => {
                        const nb = globalThis.colab?.global?.notebook;
                        const cell = nb?.cells ? nb.cells[0] : null;
                        return cell && cell.isRunning ? cell.isRunning() : false;
                    }""")
                    print(f"  Worker {wid} running after trigger: {st2}")
            except Exception as e:
                print(f"  Error on Worker {wid}: {e}")

if __name__ == "__main__":
    main()
