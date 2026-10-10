import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from playwright.sync_api import sync_playwright
from scripts.colab.browser_utils import transcribe_audio_bytes

def solve_worker_recaptcha(wid, port):
    print(f"=== Solving reCAPTCHA for Worker {wid} (Port {port}) ===", flush=True)
    with sync_playwright() as p:
        b = p.chromium.connect_over_cdp(f"http://127.0.0.1:{port}")
        page = b.contexts[0].pages[0]
        
        # Find the recaptcha frame with ar=1&k=6LfQttQU
        target_frame = None
        for frame in page.frames:
            if "k=6LfQttQU" in frame.url and "anchor" in frame.url:
                anchor = frame.locator("#recaptcha-anchor")
                if anchor.count() > 0 and anchor.first.is_visible():
                    target_frame = frame
                    break
        
        if not target_frame:
            # Try any anchor that is visible
            for frame in page.frames:
                if "anchor" in frame.url:
                    anchor = frame.locator("#recaptcha-anchor")
                    if anchor.count() > 0:
                        try:
                            if anchor.first.is_visible():
                                target_frame = frame
                                break
                        except Exception:
                            pass
        
        if not target_frame:
            print("No visible reCAPTCHA anchor found!", flush=True)
            return False
            
        print(f"Using frame: {target_frame.name} ({target_frame.url[:60]})", flush=True)
        anchor = target_frame.locator("#recaptcha-anchor")
        checked = anchor.get_attribute("aria-checked")
        print(f"Current anchor checked state: {checked}", flush=True)
        
        if checked != "true":
            print("Clicking reCAPTCHA anchor...", flush=True)
            anchor.click()
            time.sleep(3)
            checked = anchor.get_attribute("aria-checked")
            print(f"Post-click checked state: {checked}", flush=True)
            
        if checked == "true":
            print("reCAPTCHA already solved directly!", flush=True)
            return True
            
        # If not solved directly, check corresponding bframe
        bframe = None
        # Often the bframe name matches or corresponds
        for frame in page.frames:
            if "bframe" in frame.url:
                audio_btn = frame.locator("#recaptcha-audio-button")
                if audio_btn.count() > 0:
                    bframe = frame
                    break
                    
        if not bframe:
            print("No bframe with audio button found!", flush=True)
            return False
            
        print(f"Found bframe: {bframe.name}", flush=True)
        audio_btn = bframe.locator("#recaptcha-audio-button")
        if audio_btn.is_visible():
            print("Clicking audio challenge button...", flush=True)
            audio_btn.click()
            time.sleep(4)
            
        # Check download link
        dl_link = bframe.locator("a.rc-audiochallenge-tdownload-link, a[href*='payload'], a[href*='audio.mp3']")
        if dl_link.count() > 0:
            audio_url = dl_link.first.get_attribute("href")
            print(f"Downloading audio: {audio_url[:60]}...", flush=True)
            resp = page.request.get(audio_url)
            audio_bytes = resp.body()
            text = transcribe_audio_bytes(audio_bytes)
            print(f"Whisper transcribed text: '{text}'", flush=True)
            
            audio_input = bframe.locator("#audio-response")
            if text and audio_input.count() > 0:
                audio_input.fill(text)
                time.sleep(1)
                verify_btn = bframe.locator("#recaptcha-verify-button")
                if verify_btn.count() > 0:
                    print("Submitting audio answer...", flush=True)
                    verify_btn.click()
                    time.sleep(4)
                    
        checked = anchor.get_attribute("aria-checked")
        print(f"Final checked state: {checked}", flush=True)
        return checked == "true"

if __name__ == "__main__":
    wid = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 9004
    solve_worker_recaptcha(wid, port)
