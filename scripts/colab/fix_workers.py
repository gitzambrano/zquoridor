import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from playwright.sync_api import sync_playwright
from scripts.colab.browser_utils import transcribe_audio_bytes, dismiss_modals

def inspect_and_fix_worker1():
    print("=== WORKER 1 (Port 9001) ===", flush=True)
    with sync_playwright() as p:
        b = p.chromium.connect_over_cdp("http://127.0.0.1:9001")
        colab_page = None
        for pg in b.contexts[0].pages:
            if "colab.research.google.com" in pg.url:
                colab_page = pg
                break
        if not colab_page:
            print("No colab page found!", flush=True)
            return

        dismiss_modals(colab_page)
        
        # Execute cell 0
        res = colab_page.evaluate("""() => {
            const nb = globalThis.colab?.global?.notebook;
            const cell = nb?.cells ? nb.cells[0] : null;
            if (!cell) return { ok: false, err: 'no_cell' };
            
            let clicked = false;
            const elem = cell.getElement ? cell.getElement() : (cell.element_ || cell.dom_);
            if (elem) {
                const btn = elem.querySelector('colab-run-button');
                if (btn && btn.shadowRoot) {
                    const inner = btn.shadowRoot.querySelector('#run-button, button');
                    if (inner) {
                        inner.click();
                        clicked = true;
                    }
                }
            }
            if (!clicked && typeof cell.manualExecute === 'function') {
                cell.manualExecute();
                clicked = true;
            }
            return {
                ok: true,
                clicked: clicked,
                running: cell.isRunning ? cell.isRunning() : false
            };
        }""")
        print(f"Trigger Cell 0: {res}", flush=True)
        time.sleep(5)
        dismiss_modals(colab_page)
        
        info = colab_page.evaluate("""() => {
            const nb = globalThis.colab?.global?.notebook;
            const cell = nb?.cells ? nb.cells[0] : null;
            let out = '';
            if (cell) {
                const elem = cell.getElement ? cell.getElement() : (cell.element_ || cell.dom_);
                const d = elem ? elem.querySelector('.output, colab-output, .output-stream, .output_text') : null;
                out = d ? d.innerText.slice(-300) : '';
            }
            return {
                running: cell && cell.isRunning ? cell.isRunning() : false,
                out: out
            };
        }""")
        print(f"Status after trigger: {info}", flush=True)

def solve_worker_recaptcha(wid: int, port: int):
    print(f"=== Solving reCAPTCHA for Worker {wid} (Port {port}) ===", flush=True)
    with sync_playwright() as p:
        b = p.chromium.connect_over_cdp(f"http://127.0.0.1:{port}")
        page = b.contexts[0].pages[0]

        # 1. Look for anchor frames
        anchor_frames = [f for f in page.frames if "k=6LfQttQU" in f.url and "anchor" in f.url]
        print(f"Found {len(anchor_frames)} recaptcha anchor frames", flush=True)
        if not anchor_frames:
            print("No recaptcha anchor frame found. Checking modals...", flush=True)
            dismiss_modals(page)
            return False

        # Target the last one (usually topmost) or check each
        target_frame = anchor_frames[-1]
        print(f"Targeting frame: {target_frame.name}", flush=True)

        anchor = target_frame.locator("#recaptcha-anchor")
        if anchor.count() == 0:
            print("No #recaptcha-anchor element found in frame!", flush=True)
            return False

        checked = anchor.get_attribute("aria-checked")
        print(f"Initial checked state: {checked}", flush=True)
        if checked == "true":
            print("Already checked!", flush=True)
            return True

        # Click via evaluate to bypass pointer interception
        print("Dispatching JS click on #recaptcha-anchor...", flush=True)
        anchor.evaluate("el => el.click()")
        time.sleep(3)

        checked = anchor.get_attribute("aria-checked")
        print(f"Checked state after click: {checked}", flush=True)
        if checked == "true":
            print("Solved immediately via checkbox click!", flush=True)
            time.sleep(2)
            dismiss_modals(page)
            return True

        # If not checked, look for audio challenge button in bframe
        bframes = [f for f in page.frames if "bframe" in f.url]
        print(f"Found {len(bframes)} bframes", flush=True)
        target_bframe = bframes[-1] if bframes else None

        if target_bframe:
            audio_btn = target_bframe.locator("#recaptcha-audio-button")
            if audio_btn.count() > 0 and audio_btn.first.is_visible():
                print("Clicking audio button...", flush=True)
                audio_btn.evaluate("el => el.click()")
                time.sleep(4)

            # Check download link
            dl_link = target_bframe.locator("a.rc-audiochallenge-tdownload-link, a[href*='payload'], a[href*='audio.mp3']")
            if dl_link.count() > 0:
                audio_url = dl_link.first.get_attribute("href")
                print(f"Downloading audio: {audio_url[:60]}...", flush=True)
                resp = page.request.get(audio_url)
                audio_bytes = resp.body()
                text = transcribe_audio_bytes(audio_bytes)
                print(f"Whisper transcribed text: '{text}'", flush=True)

                audio_input = target_bframe.locator("#audio-response")
                if text and audio_input.count() > 0:
                    audio_input.fill(text)
                    time.sleep(1)
                    verify_btn = target_bframe.locator("#recaptcha-verify-button")
                    if verify_btn.count() > 0:
                        print("Submitting audio challenge...", flush=True)
                        verify_btn.evaluate("el => el.click()")
                        time.sleep(4)

        checked = anchor.get_attribute("aria-checked")
        print(f"Final checked state: {checked}", flush=True)
        if checked == "true":
            time.sleep(2)
            dismiss_modals(page)
            return True
        return False

if __name__ == "__main__":
    inspect_and_fix_worker1()

