from pathlib import Path
import sys
import time
import io
import soundfile as sf
import scipy.signal
import whisper
from playwright.sync_api import sync_playwright

model = whisper.load_model("tiny")

def solve_audio(port: int):
    print(f"Connecting to port {port}...")
    with sync_playwright() as p:
        browser = p.chromium.connect_over_cdp(f"http://127.0.0.1:{port}")
        ctx = browser.contexts[0]
        page = next(pg for pg in ctx.pages if "colab" in (pg.url or ""))
        
        bframe = next((f for f in page.frames if "bframe" in f.url), None)
        if bframe:
            audio_btn = bframe.locator("#recaptcha-audio-button")
            if audio_btn.count() > 0 and audio_btn.first.is_visible():
                print("Clicking headphones audio button...")
                audio_btn.first.click()
                time.sleep(3)
        
        for round_idx in range(5):
            print(f"--- Audio Round {round_idx + 1} ---")
            bframe = next((f for f in page.frames if "bframe" in f.url), None)
            if not bframe:
                print("No bframe found! Checking if anchor is verified...")
                break
            
            dl = bframe.locator("a.rc-audiochallenge-tdownload-link, a[href*='payload'], a[href*='audio.mp3']")
            if dl.count() == 0:
                print("No audio download link found.")
                break
            audio_url = dl.first.get_attribute("href")
            print("Audio URL:", audio_url[:60] if audio_url else None)
            if not audio_url:
                break
                
            resp = page.request.get(audio_url)
            bio = io.BytesIO(resp.body())
            data, sr = sf.read(bio, dtype="float32")
            if len(data.shape) > 1:
                data = data.mean(axis=1)
            if sr != 16000:
                num_samples = int(len(data) * 16000 / sr)
                data = scipy.signal.resample(data, num_samples)
            res = model.transcribe(data, fp16=False)
            text = res.get("text", "").strip()
            print("Transcribed text:", text)
            inp = bframe.locator("#audio-response")
            inp.fill(text)
            time.sleep(1)
            btn = bframe.locator("#recaptcha-verify-button")
            btn.click()
            time.sleep(4)
            page.screenshot(path=f"worker_{port}_round_{round_idx+1}.png")

        # Now check if anchor is verified and click run
        print("Checking anchor and clicking run button...")
        try:
            # Handle Conectar sem GPU if present
            gpu_btn = page.locator("button:has-text('Conectar sem GPU'), md-text-button:has-text('Conectar sem GPU')")
            if gpu_btn.count() > 0:
                print("Clicking Conectar sem GPU...")
                gpu_btn.first.click()
                time.sleep(2)
        except Exception:
            pass

        # Click Run
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
        time.sleep(3)
        page.screenshot(path=f"worker_{port}_final.png")
        print("Done!")

if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 9005
    solve_audio(port)
