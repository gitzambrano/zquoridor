"""Deploy the Claustrophobia search games cell to Colab workers and launch execution."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.colab.search_games_profile import build_worker_profile, render_bootloader
from scripts.colab.browser_utils import dismiss_modals, connect_runtime_if_needed, is_cdp_reachable


def deploy_worker(wid: int, headless: bool = True, force: bool = False) -> dict:
    profile = build_worker_profile(wid)
    code = render_bootloader(profile)
    port = profile["cdp_port"]

    print(f"\n==========================================", flush=True)
    print(f"Deploying Worker {wid} ({profile['account']})", flush=True)
    print(f"Mode: {profile.get('mode', 'match')}", flush=True)
    print(f"Drive output: {profile['drive_dir']}", flush=True)
    print(f"Seed: {profile['seed']}", flush=True)
    print(f"==========================================", flush=True)

    with sync_playwright() as p:
        if is_cdp_reachable(port):
            print(f"Worker {wid}: Connecting over existing CDP on port {port}...", flush=True)
            b = p.chromium.connect_over_cdp(f"http://127.0.0.1:{port}")
            ctx = b.contexts[0]
            owns_browser = False
        else:
            print(f"Worker {wid}: Launching persistent Chrome context...", flush=True)
            ctx = p.chromium.launch_persistent_context(
                user_data_dir=profile["profile_dir"],
                headless=headless,
                channel="chrome",
                args=[f"--remote-debugging-port={port}", "--remote-allow-origins=*"],
            )
            owns_browser = True

        page = None
        for pg in ctx.pages:
            if "colab.research.google.com" in pg.url:
                page = pg
                break
        if not page:
            page = ctx.pages[0] if ctx.pages else ctx.new_page()
            page.goto(profile["notebook_url"], wait_until="domcontentloaded", timeout=60000)

        page.wait_for_timeout(8000)
        connect_runtime_if_needed(page)
        dismiss_modals(page)

        # Inspect current state
        state = page.evaluate("""() => {
            const nb = globalThis.colab?.global?.notebook;
            if (!nb || !nb.cells || nb.cells.length === 0) return { error: 'no_cells' };
            const cell = nb.cells[0];
            const txt = cell.getText ? cell.getText() : '';
            return {
                isSearchGame: txt.includes('run_search_games.py'),
                isRunning: cell.isRunning ? cell.isRunning() : false,
                hasNewRevision: txt.includes('38f17b80d4dc1aaa1dc5fbfe65a2f5ac8cd3f07f'),
                textSnippet: txt.slice(0, 150)
            };
        }""")
        print(f"Worker {wid} initial state: {state}", flush=True)

        if not force and state.get("isSearchGame") and state.get("isRunning") and state.get("hasNewRevision"):
            print(f"Worker {wid} is ALREADY running the new version!", flush=True)
            if owns_browser:
                ctx.close()
            return {"worker_id": wid, "status": "already_running"}

        # Inject code, save notebook, and execute cell
        print(f"Worker {wid}: Injecting code, saving notebook, and executing cell...", flush=True)
        res = page.evaluate("""async (newCode) => {
            const nb = globalThis.colab?.global?.notebook;
            if (!nb || !nb.cells || nb.cells.length === 0) return { ok: false, error: 'no_cells' };
            const cell = nb.cells[0];

            // 1. Interrupt any running code
            if (cell.isRunning && cell.isRunning()) {
                try { cell.interrupt(); } catch (e) {}
                try { nb.interrupt(); } catch (e) {}
            }

            // 2. Set new code on all model layers
            if (typeof cell.setText === 'function') cell.setText(newCode);
            if (cell.model && typeof cell.model.setText === 'function') cell.model.setText(newCode);
            if (cell.model && cell.model.textModel && typeof cell.model.textModel.setValue === 'function') {
                cell.model.textModel.setValue(newCode);
            }
            if (typeof monaco !== 'undefined') {
                for (const m of monaco.editor.getModels()) {
                    m.setValue(newCode);
                }
            }
            if (cell.model && typeof cell.model.removeOutputs === 'function') {
                cell.model.removeOutputs();
            }

            // 3. Save notebook to Google Drive
            let saved = false;
            if (typeof nb.saveNotebook === 'function') {
                try {
                    await nb.saveNotebook();
                    saved = true;
                } catch (e) {}
            }

            // 4. Trigger execution via shadow DOM run-button
            let clicked = false;
            const elem = cell.getElement ? cell.getElement() : (cell.element_ || cell.dom_);
            if (elem) {
                const btn = elem.querySelector('colab-run-button');
                if (btn && btn.shadowRoot) {
                    const inner = btn.shadowRoot.querySelector('#run-button, button, [role="button"]');
                    if (inner) {
                        inner.click();
                        clicked = true;
                    }
                }
            }
            if (!clicked && typeof cell.manualExecute === 'function') {
                try { cell.manualExecute(); clicked = true; } catch (e) {}
            }

            return {
                ok: true,
                saved: saved,
                clicked: clicked,
                textLen: cell.getText ? cell.getText().length : 0
            };
        }""", code)
        print(f"Worker {wid} injection & trigger result: {res}", flush=True)

        time.sleep(4)
        dismiss_modals(page)
        time.sleep(4)
        dismiss_modals(page)

        # Verify execution status
        final_status = page.evaluate("""() => {
            const nb = globalThis.colab?.global?.notebook;
            const cell = nb ? nb.cells[0] : null;
            if (!cell) return { isRunning: false };
            const elem = cell.getElement ? cell.getElement() : (cell.element_ || cell.dom_);
            let outText = '';
            if (elem) {
                const outDiv = elem.querySelector('.output, colab-output, .output-stream, .output_text');
                outText = outDiv ? outDiv.innerText.slice(-600) : '';
            }
            return {
                isRunning: cell.isRunning ? cell.isRunning() : false,
                isPending: cell.isPending ? cell.isPending() : false,
                outText: outText,
                cellTextSnippet: (cell.getText ? cell.getText() : '').slice(-150)
            };
        }""")
        print(f"Worker {wid} final status: {final_status}", flush=True)

        screenshot_path = ROOT / f"artifacts/colab/deploy_worker_{wid}.png"
        screenshot_path.parent.mkdir(parents=True, exist_ok=True)
        page.screenshot(path=str(screenshot_path))
        print(f"Worker {wid} screenshot saved to {screenshot_path.name}", flush=True)

        if owns_browser:
            ctx.close()
        return {"worker_id": wid, **final_status}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    results = {}
    for wid in args.workers:
        try:
            results[wid] = deploy_worker(wid, headless=args.headless, force=args.force)
        except Exception as e:
            print(f"Worker {wid} error: {e}", flush=True)
            results[wid] = {"worker_id": wid, "error": str(e)}

    print("\n================ DEPLOYMENT SUMMARY ================")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
