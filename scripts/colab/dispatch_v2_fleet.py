"""Dispatch and monitor the complete zQuoridor 4.0 (858) collection fleet on Google Colab.

- Workers 1, 3, 4, 5: GPU T4 search games against Claustrophobia (output: claustro_search_games_v2).
- Worker 2: CPU pure C++ selfplay of zQuoridor 4.0 858 (output: selfplay_858_v1).
- Openings: 50% Center Rush sound + 40% Normal + 10% Weakness, temperature sampling on first 12 plies.
- Continuous keep-alive and real-time ledger monitoring.
"""
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

from scripts.colab.config import WORKERS
from scripts.colab.search_games_profile import build_worker_profile, render_bootloader, CONFIG as PROFILE_CONFIG
from scripts.colab.browser_utils import (
    dismiss_modals,
    connect_runtime_if_needed,
    is_cdp_reachable,
    transcribe_audio_bytes,
)


def solve_recaptcha_if_present(page) -> bool:
    """Handle any reCAPTCHA challenge modal if present."""
    try:
        anchor_frames = [f for f in page.frames if "k=6LfQttQU" in f.url and "anchor" in f.url]
        if not anchor_frames:
            return False
        target = anchor_frames[-1]
        anchor = target.locator("#recaptcha-anchor")
        if anchor.count() > 0:
            checked = anchor.get_attribute("aria-checked")
            if checked != "true":
                print("  [reCAPTCHA] Clicking anchor via JS...", flush=True)
                anchor.evaluate("el => el.click()")
                time.sleep(3)
                dismiss_modals(page)
                return True
    except Exception:
        pass
    return False


def setup_worker_session(p, wid: int, headless: bool = True):
    profile = build_worker_profile(wid)
    port = profile["cdp_port"]
    print(f"\n--- Initializing Worker {wid} ({profile['account']}) ---", flush=True)

    if is_cdp_reachable(port):
        print(f"  Worker {wid}: Connecting over existing CDP on port {port}...", flush=True)
        b = p.chromium.connect_over_cdp(f"http://127.0.0.1:{port}")
        ctx = b.contexts[0]
    else:
        print(f"  Worker {wid}: Launching persistent Chrome context on port {port}...", flush=True)
        ctx = p.chromium.launch_persistent_context(
            user_data_dir=profile["profile_dir"],
            headless=headless,
            channel="chrome",
            args=[f"--remote-debugging-port={port}", "--remote-allow-origins=*"],
        )

    page = None
    for pg in ctx.pages:
        if "colab.research.google.com" in pg.url:
            page = pg
            break
    if not page:
        page = ctx.pages[0] if ctx.pages else ctx.new_page()
        page.goto(profile["notebook_url"], wait_until="domcontentloaded", timeout=60000)

    time.sleep(6)
    connect_runtime_if_needed(page)
    dismiss_modals(page)
    solve_recaptcha_if_present(page)

    return {"profile": profile, "ctx": ctx, "page": page}


def deploy_cell(session: dict, force: bool = False) -> bool:
    wid = session["profile"]["worker_id"]
    page = session["page"]
    profile = session["profile"]
    code = render_bootloader(profile)
    mode = profile.get("mode", "match")

    print(f"  Worker {wid} [{mode}]: Verifying notebook cell...", flush=True)

    # Check if already running the target revision
    state = page.evaluate("""() => {
        const nb = globalThis.colab?.global?.notebook;
        if (!nb || !nb.cells || nb.cells.length === 0) return { error: 'no_cells' };
        const cell = nb.cells[0];
        const txt = cell.getText ? cell.getText() : '';
        return {
            isSearchGame: txt.includes('run_search_games.py'),
            isRunning: cell.isRunning ? cell.isRunning() : false,
            hasNewRevision: txt.includes('38f17b80d4dc1aaa1dc5fbfe65a2f5ac8cd3f07f'),
            hasTargetDrive: txt.includes('claustro_search_games_v2') || txt.includes('selfplay_858_v1'),
            textSnippet: txt.slice(0, 100).replace(/\\n/g, ' ')
        };
    }""")
    print(f"  Worker {wid} current cell: {state}", flush=True)

    if not force and state.get("isRunning") and state.get("hasNewRevision") and state.get("hasTargetDrive"):
        print(f"  Worker {wid} is ALREADY running the target revision ({mode})!", flush=True)
        return True

    print(f"  Worker {wid}: Injecting updated cell code ({mode}) and triggering execution...", flush=True)
    res = page.evaluate("""async (newCode) => {
        const nb = globalThis.colab?.global?.notebook;
        if (!nb || !nb.cells || nb.cells.length === 0) return { ok: false, error: 'no_cells' };
        const cell = nb.cells[0];

        // 1. Interrupt any running cell
        if (cell.isRunning && cell.isRunning()) {
            try { cell.interrupt(); } catch (e) {}
            try { nb.interrupt(); } catch (e) {}
        }

        // 2. Set new code on all layers
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

        // 3. Save notebook
        let saved = false;
        if (typeof nb.saveNotebook === 'function') {
            try { await nb.saveNotebook(); saved = true; } catch (e) {}
        }

        // 4. Trigger run
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

        return { ok: true, clicked: clicked, saved: saved };
    }""", code)
    print(f"  Worker {wid} trigger result: {res}", flush=True)

    time.sleep(5)
    dismiss_modals(page)
    solve_recaptcha_if_present(page)

    return True


def poll_worker_output(page) -> dict:
    return page.evaluate("""() => {
        const nb = globalThis.colab?.global?.notebook;
        const cell = nb?.cells ? nb.cells[0] : null;
        let out = '';
        if (cell) {
            const elem = cell.getElement ? cell.getElement() : (cell.element_ || cell.dom_);
            const d = elem ? elem.querySelector('.output, colab-output, .output-stream, .output_text') : null;
            out = d ? d.innerText.slice(-300) : '';
        }
        const btn = document.querySelector('colab-connect-button');
        const connText = btn ? (btn.shadowRoot ? btn.shadowRoot.textContent.replace(/\\s+/g, ' ').trim() : btn.textContent) : '';
        return {
            running: cell && cell.isRunning ? cell.isRunning() : false,
            conn: connText,
            out: out.replace(/\\n/g, ' ').trim()
        };
    }""")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--daemon", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    print("=================================================================", flush=True)
    print("ZQUORIDOR 4.0 (858) FLEET DISPATCHER & KEEP-ALIVE MONITOR", flush=True)
    print(f"Workers: {args.workers} | Revision: 38f17b80d4dc1aaa1dc5fbfe65a2f5ac8cd3f07f", flush=True)
    print(f"GPU Root: claustro_search_games_v2 | CPU Root: selfplay_858_v1", flush=True)
    print("=================================================================", flush=True)

    with sync_playwright() as p:
        sessions = {}
        for wid in args.workers:
            try:
                sess = setup_worker_session(p, wid, headless=args.headless)
                deploy_cell(sess, force=args.force)
                sessions[wid] = sess
            except Exception as e:
                print(f"ERROR initializing Worker {wid}: {e}", flush=True)

        print("\nAll workers deployed. Entering live monitor loop...\n", flush=True)

        cycle = 0
        while True:
            cycle += 1
            print(f"--- FLEET PROGRESS [Cycle {cycle} | {time.strftime('%H:%M:%S')}] ---", flush=True)
            for wid, sess in sessions.items():
                try:
                    page = sess["page"]
                    dismiss_modals(page)
                    solve_recaptcha_if_present(page)
                    info = poll_worker_output(page)
                    mode_tag = "CPU-Selfplay" if sess["profile"].get("is_cpu_selfplay") else "GPU-Claustro"
                    status_tag = "RUNNING" if info["running"] else "STOPPED"
                    out_snip = info["out"][-90:] if info["out"] else "(starting...)"
                    print(f"  [Worker {wid}] ({mode_tag}) [{status_tag}] -> {out_snip}", flush=True)

                    # Auto-restart if stopped unexpectedly
                    if not info["running"]:
                        print(f"  [Worker {wid}] Cell stopped. Re-triggering...", flush=True)
                        deploy_cell(sess, force=False)
                except Exception as e:
                    print(f"  [Worker {wid}] Monitor error: {e}", flush=True)

            if not args.daemon:
                break
            time.sleep(30)


if __name__ == "__main__":
    main()
