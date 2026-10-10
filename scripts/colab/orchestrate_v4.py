"""Robust orchestrator and daemon monitor for zQuoridor 4.0 (Network 858) collection across all 5 workers.

Key features:
- Automatically detects GPU quota limits; seamlessly falls back to CPU Selfplay.
- GPU Mode: zQuoridor 4.0 vs Claustrophobia (400ms -> 40ms, 12-ply diversified openings).
- CPU Mode: Pure C++ selfplay of 858 from main (400ms -> 40ms, 12-ply diversified openings).
- Target segregation: claustro_search_games_v2 (GPU) and selfplay_858_v1 (CPU).
  Preserves claustro_search_games_v1 untouched.
- Solves reCAPTCHA v2 challenges via direct anchor JS click.
- Handles Google Drive mount dialogues and OAuth consent popups automatically.
- Continuous keep-alive loop monitors running state and re-executes if interrupted.
"""
from __future__ import annotations

import json
from pathlib import Path
import shlex
import sys
import time
from typing import Any, Dict, Optional

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.colab.config import WORKERS
from scripts.colab.search_games_profile import (
    build_worker_profile,
    BOOTLOADER_TEMPLATE,
    CPU_SELFPLAY_BOOTLOADER,
    CONFIG as PROFILE_CONFIG,
)
from scripts.colab.browser_utils import is_cdp_reachable, solve_recaptcha_playwright


def solve_recaptcha_if_present(page: Any) -> bool:
    """Detect and solve reCAPTCHA v2 challenges via direct JS evaluation with audio fallback."""
    try:
        anchor_frames = [f for f in page.frames if "k=6LfQttQU" in f.url and "anchor" in f.url]
        if not anchor_frames:
            return False
        anchor = anchor_frames[-1].locator("#recaptcha-anchor")
        if anchor.count() > 0:
            checked = anchor.get_attribute("aria-checked")
            if checked == "true":
                return True
            print("    [reCAPTCHA] Clicking anchor via JS...", flush=True)
            anchor.evaluate("el => el.click()")
            time.sleep(3)
            try:
                if anchor.get_attribute("aria-checked") == "true":
                    print("    [reCAPTCHA] Checked successfully via JS.", flush=True)
                    return True
            except Exception:
                pass
        bframe = [f for f in page.frames if "bframe" in f.url]
        if bframe:
            print("    [reCAPTCHA] BFrame active. Solving via audio/Whisper...", flush=True)
            return solve_recaptcha_playwright(page)
    except Exception as exc:
        print(f"    [reCAPTCHA error] {exc}", flush=True)
        return False
    return False


def handle_dialogs(page: Any) -> str:
    """Handle all known Colab modals accurately without closing them erroneously.
    
    Returns action performed: 'gpu_quota_switched_to_cpu', 'drive_connected', 'run_anyway', or 'none'.
    """
    try:
        res = page.evaluate("""() => {
            const dialogs = Array.from(document.querySelectorAll('mwc-dialog, colab-dialog')).filter(d => d.open || d.offsetParent !== null);
            for (const d of dialogs) {
                const txt = (d.innerText || '');
                // 1. GPU quota dialog
                if (txt.includes('limites de uso do Colab') || txt.includes('GPU usage limits') || txt.includes('limite de uso') || txt.includes('usage limits')) {
                    for (const b of d.querySelectorAll('button, md-text-button, mwc-button, paper-button')) {
                        const btxt = (b.innerText || '').toLowerCase();
                        if (btxt.includes('conectar sem gpu') || btxt.includes('without gpu')) {
                            b.click();
                            return { action: 'gpu_quota_switched_to_cpu' };
                        }
                    }
                }
                // 2. Google Drive connection authorization dialog
                for (const b of d.querySelectorAll('button, md-text-button, mwc-button, paper-button')) {
                    const btxt = (b.innerText || '').toLowerCase();
                    if (btxt.includes('conectar ao google drive') || btxt.includes('connect to google drive')) {
                        b.click();
                        return { action: 'drive_connected' };
                    }
                }
                // 3. Notebook execution warning dialog (author confirmation)
                for (const b of d.querySelectorAll('button, md-text-button, mwc-button, paper-button')) {
                    const btxt = (b.innerText || '').toLowerCase();
                    if (btxt.includes('executar mesmo assim') || btxt.includes('executar de qualquer maneira') || btxt.includes('run anyway')) {
                        b.click();
                        return { action: 'run_anyway' };
                    }
                }
            }
            return { action: 'none' };
        }""")
        action = res.get("action", "none")
        if action != "none":
            print(f"    [MODAL] Handled: {action}", flush=True)
            time.sleep(2)
        return action
    except Exception:
        return "none"


def connect_runtime_if_needed(page: Any) -> bool:
    """Ensure runtime is connected if currently disconnected."""
    try:
        return page.evaluate("""() => {
            const btn = document.querySelector('colab-connect-button');
            if (btn && btn.shadowRoot) {
                const conn = btn.shadowRoot.querySelector('#connect');
                if (conn) {
                    const txt = conn.innerText || '';
                    if (txt.includes('Conectar') || txt.includes('Reconectar') || txt.includes('Connect') || txt.includes('Reconnect')) {
                        conn.click();
                        return true;
                    }
                }
            }
            return false;
        }""")
    except Exception:
        return False


def handle_oauth_popups(ctx: Any) -> bool:
    """Approve Google Drive OAuth consent popup if present in the browser context."""
    handled = False
    try:
        if len(ctx.pages) > 1:
            for pg in ctx.pages[1:]:
                url = pg.url or ""
                if "accounts.google.com" in url:
                    print("    [OAUTH] Authorizing Drive popup...", flush=True)
                    try:
                        for cb in pg.locator("input[type='checkbox']").all():
                            if not cb.is_checked():
                                cb.check()
                                time.sleep(1)
                    except Exception:
                        pass
                    btn = pg.locator("#submit_approve_access, button:has-text('Allow'), button:has-text('Permitir'), button:has-text('Continuar'), button:has-text('Avançar')")
                    if btn.count() > 0 and btn.first.is_visible():
                        btn.first.click()
                        handled = True
                        time.sleep(3)
    except Exception:
        pass
    return handled


def set_cell_code(page: Any, code: str) -> bool:
    """Inject Python source code into Cell 0 of the Colab notebook."""
    try:
        return page.evaluate("""async (newCode) => {
            const nb = globalThis.colab?.global?.notebook;
            if (!nb || !nb.cells || nb.cells.length === 0) return false;
            const cell = nb.cells[0];
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
            if (typeof nb.saveNotebook === 'function') {
                try { await nb.saveNotebook(); } catch (e) {}
            }
            return true;
        }""", code)
    except Exception:
        return False


def click_run_button(page: Any) -> bool:
    """Click the Run button for Cell 0."""
    try:
        return page.evaluate("""() => {
            const nb = globalThis.colab?.global?.notebook;
            const cell = nb?.cells ? nb.cells[0] : null;
            if (!cell) return false;
            const elem = cell.getElement ? cell.getElement() : (cell.element_ || cell.dom_);
            if (elem) {
                const btn = elem.querySelector('colab-run-button');
                if (btn && btn.shadowRoot) {
                    const inner = btn.shadowRoot.querySelector('#run-button, button');
                    if (inner) {
                        inner.click();
                        return true;
                    }
                }
            }
            if (typeof cell.manualExecute === 'function') {
                cell.manualExecute();
                return true;
            }
            return false;
        }""")
    except Exception:
        return False


def get_cell_execution_state(page: Any) -> Dict[str, Any]:
    """Inspect Cell 0 execution status, pending state, and current stdout/stderr tail."""
    try:
        return page.evaluate("""() => {
            const nb = globalThis.colab?.global?.notebook;
            const cell = nb?.cells ? nb.cells[0] : null;
            let running = false;
            let pending = false;
            let out = '';

            if (cell) {
                running = cell.isRunning ? cell.isRunning() : false;
                pending = cell.isPending ? cell.isPending() : false;

                const elem = cell.getElement ? cell.getElement() : (cell.element_ || cell.dom_);
                if (elem) {
                    const runBtn = elem.querySelector('colab-run-button');
                    const sr = runBtn ? runBtn.shadowRoot : null;
                    const wrap = sr ? sr.querySelector('.cell-execution') : null;
                    const cls = wrap ? wrap.className : '';
                    if (cls.includes('running')) running = true;
                    if (cls.includes('pending') || cls.includes('waiting')) pending = true;

                    const d = elem.querySelector('.output, colab-output, .output-stream, .output_text');
                    out = d ? (d.innerText || '') : '';
                }

                // Check model output streams if DOM text is still empty
                if (!out && cell.model && cell.model.outputs) {
                    for (const o of cell.model.outputs) {
                        if (o.text) {
                            out += Array.isArray(o.text) ? o.text.join('') : o.text;
                        } else if (o.traceback) {
                            out += o.traceback.join('\\n');
                        }
                    }
                }
            }

            return {
                running: running,
                pending: pending,
                out: out.trim().slice(-300)
            };
        }""")
    except Exception as exc:
        return {"running": False, "pending": False, "out": f"error: {exc}"}


def generate_code_for_worker(wid: int, mode: str) -> str:
    """Generate bootloader code tailored for either GPU or CPU execution."""
    profile = build_worker_profile(wid)
    seed = int(PROFILE_CONFIG["seed_stride"]) * wid
    rt_dir = profile['runtime_dir_literal'].strip("'")

    if mode == "cpu":
        drive_root = PROFILE_CONFIG.get("cpu_selfplay_drive_root", "/content/drive/MyDrive/zquoridor_data/selfplay_858_v1")
        drive_dir = f"{str(drive_root).rstrip('/')}/worker_{wid}"
        args = [
            "--mode", "zquoridor-selfplay",
            "--pairs", str(PROFILE_CONFIG["pairs"]),
            "--batch-games", str(PROFILE_CONFIG["batch_games"]),
            "--workers", str(PROFILE_CONFIG["workers"]),
            "--seed", str(seed),
            "--start-move-time-ms", str(PROFILE_CONFIG["start_move_time_ms"]),
            "--end-move-time-ms", str(PROFILE_CONFIG["end_move_time_ms"]),
            "--decay-start-ply", str(PROFILE_CONFIG["decay_start_ply"]),
            "--decay-end-ply", str(PROFILE_CONFIG["decay_end_ply"]),
            "--schedule-origin", str(PROFILE_CONFIG["schedule_origin"]),
            "--opening-temperature", str(PROFILE_CONFIG["opening_temperature"]),
            "--temperature-plies", str(PROFILE_CONFIG["temperature_plies"]),
            "--output", drive_dir,
            "--zq-executable", f"{rt_dir}/zquoridor_uci",
            "--no-auto-setup",
            "--resume",
            "--export-targets",
            "--no-export-final-run",
            "--compress-game-ledger",
        ]
        for book, weight in PROFILE_CONFIG["opening_weights"].items():
            args.extend(["--opening-weight", f"{book}={weight}"])
        if PROFILE_CONFIG["unique_openings_first"]:
            args.append("--unique-openings-first")
        profile["extra_args"] = shlex.join(args)
        profile["drive_dir"] = drive_dir
        return CPU_SELFPLAY_BOOTLOADER.format(**profile, cmd_args=profile["extra_args"])
    else:
        # GPU Claustrophobia mode
        return BOOTLOADER_TEMPLATE.format(**profile, cmd_args=profile["extra_args"])


def setup_worker(p: Any, wid: int) -> Dict[str, Any]:
    """Launch persistent Chrome context and prepare worker for execution."""
    w = WORKERS[wid]
    port = w["cdp_port"]
    print(f"\n=======================================================", flush=True)
    print(f"INITIALIZING WORKER {wid} ({w['account']}) - PORT {port}", flush=True)
    print(f"=======================================================", flush=True)

    if is_cdp_reachable(port):
        print(f"  Worker {wid}: Connecting to existing CDP session on port {port}...", flush=True)
        browser = p.chromium.connect_over_cdp(f"http://127.0.0.1:{port}")
        ctx = browser.contexts[0]
    else:
        print(f"  Worker {wid}: Launching persistent Chrome context...", flush=True)
        ctx = p.chromium.launch_persistent_context(
            user_data_dir=w["profile_dir"],
            headless=True,
            channel="chrome",
            args=[f"--remote-debugging-port={port}", "--remote-allow-origins=*"],
        )

    page = next((pg for pg in ctx.pages if "colab.research.google.com" in (pg.url or "")), None)
    if not page:
        page = ctx.pages[0] if ctx.pages else ctx.new_page()
        page.goto(w["notebook_url"], wait_until="domcontentloaded", timeout=60000)

    time.sleep(5)
    connect_runtime_if_needed(page)
    time.sleep(2)
    action = handle_dialogs(page)
    solve_recaptcha_if_present(page)
    handle_oauth_popups(ctx)

    # Check if cell is ALREADY running from previous session
    initial_st = get_cell_execution_state(page)
    if initial_st["running"]:
        print(f"  Worker {wid} is ALREADY actively running! Preserving active run...", flush=True)
        # Determine mode from keywords in profile or output
        is_cpu = ("selfplay_858_v1" in initial_st["out"]) or (wid in [2, 4, 5])
        mode = "cpu" if is_cpu else "gpu"
        mode_label = "CPU-Selfplay (858)" if is_cpu else "GPU-Claustro (858 vs Claustro)"
        return {
            "wid": wid,
            "account": w["account"],
            "mode": mode,
            "mode_label": mode_label,
            "ctx": ctx,
            "page": page,
            "last_seen_running": True,
        }

    # Determine mode: all accounts default to CPU Selfplay unless already active on GPU
    is_cpu = True
    mode = "cpu"
    mode_label = "CPU-Selfplay (858)"
    print(f"  Selected Mode: {mode_label}", flush=True)

    code = generate_code_for_worker(wid, mode)
    print("  Injecting code into Cell 0...", flush=True)
    set_cell_code(page, code)
    print("  Triggering run execution...", flush=True)
    click_run_button(page)

    # Initial stabilization loop
    for _ in range(4):
        time.sleep(3)
        connect_runtime_if_needed(page)
        act = handle_dialogs(page)
        solve_recaptcha_if_present(page)
        handle_oauth_popups(ctx)
        if act == "gpu_quota_switched_to_cpu" and mode == "gpu":
            print("  [GPU QUOTA] Dialog switched to CPU runtime. Re-injecting CPU code...", flush=True)
            mode = "cpu"
            mode_label = "CPU-Selfplay (858)"
            code = generate_code_for_worker(wid, mode)
            set_cell_code(page, code)
            click_run_button(page)
        elif not get_cell_execution_state(page)["running"]:
            click_run_button(page)

    st = get_cell_execution_state(page)
    print(f"  Worker {wid} ready: running={st['running']}, pending={st['pending']}", flush=True)

    return {
        "wid": wid,
        "account": w["account"],
        "mode": mode,
        "mode_label": mode_label,
        "ctx": ctx,
        "page": page,
        "last_seen_running": st["running"] or st["pending"],
    }


def main():
    print("=================================================================", flush=True)
    print("ORCHESTRATING ZQUORIDOR 4.0 (NETWORK 858) - ALL 5 WORKERS", flush=True)
    print("=================================================================", flush=True)

    workers_state: Dict[int, Dict[str, Any]] = {}

    with sync_playwright() as p:
        for wid in [1, 2, 3, 4, 5]:
            try:
                workers_state[wid] = setup_worker(p, wid)
            except Exception as exc:
                print(f"Worker {wid} initialization error: {exc}", flush=True)

        print("\n================ FLEET MONITOR LOOP STARTED ================\n", flush=True)
        cycle = 0
        while True:
            cycle += 1
            time.sleep(25)
            print(f"\n--- FLEET MONITOR [Cycle {cycle} | {time.strftime('%H:%M:%S')}] ---", flush=True)

            for wid in [1, 2, 3, 4, 5]:
                st_info = workers_state.get(wid)
                if not st_info:
                    continue
                page = st_info["page"]
                ctx = st_info["ctx"]

                try:
                    # 1. Handle background modals / popups
                    act = handle_dialogs(page)
                    solve_recaptcha_if_present(page)
                    handle_oauth_popups(ctx)

                    if act == "gpu_quota_switched_to_cpu" and st_info["mode"] == "gpu":
                        print(f"  [Worker {wid}] Quota limit reached -> Switching to CPU Selfplay...", flush=True)
                        st_info["mode"] = "cpu"
                        st_info["mode_label"] = "CPU-Selfplay (858)"
                        code = generate_code_for_worker(wid, "cpu")
                        set_cell_code(page, code)
                        click_run_button(page)
                        continue

                    # 2. Check execution state
                    st = get_cell_execution_state(page)
                    is_active = st["running"] or st["pending"]

                    if not is_active:
                        print(f"  [Worker {wid}] Inactive! Re-checking modals and re-clicking run...", flush=True)
                        connect_runtime_if_needed(page)
                        act = handle_dialogs(page)
                        solve_recaptcha_if_present(page)
                        handle_oauth_popups(ctx)
                        if act == "gpu_quota_switched_to_cpu" and st_info["mode"] == "gpu":
                            st_info["mode"] = "cpu"
                            st_info["mode_label"] = "CPU-Selfplay (858)"
                            code = generate_code_for_worker(wid, "cpu")
                            set_cell_code(page, code)
                        click_run_button(page)
                        time.sleep(3)
                        st = get_cell_execution_state(page)
                        is_active = st["running"] or st["pending"]

                    st_info["last_seen_running"] = is_active

                    # 4. Format clean output snippet
                    raw_out = st["out"].replace("\n", " ").strip()
                    out_snip = raw_out[-80:] if raw_out else "(compiling / initializing environment...)"
                    tag = "RUNNING" if is_active else "STOPPED"
                    print(f"  Worker {wid} [{st_info['mode_label']}] [{tag}] -> {out_snip}", flush=True)

                    # 5. Keep-alive interaction: mouse move on Colab banner
                    try:
                        page.mouse.move(50, 50)
                    except Exception:
                        pass

                except Exception as exc:
                    print(f"  Worker {wid} monitor check error: {exc}", flush=True)


if __name__ == "__main__":
    main()
