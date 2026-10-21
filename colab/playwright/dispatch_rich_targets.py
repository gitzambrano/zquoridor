#!/usr/bin/env python3
"""Dispatch Wall-Drop and Gumbel simulation jobs across all 5 Google Colab workers using version 4.10."""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_repo_root = _Path(__file__).resolve().parents[2]
if str(_repo_root) not in _sys.path:
    _sys.path.insert(0, str(_repo_root))


import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from colab.config import WORKERS
from colab.playwright.browser_utils import is_cdp_reachable, solve_recaptcha_playwright

BOOTLOADER_TEMPLATE = """from google.colab import drive
import os
import shutil
import subprocess
from pathlib import Path

if not Path('/content/drive/MyDrive').is_dir():
    try:
        drive.mount('/content/drive')
    except Exception as e:
        print('Initial drive mount failed, attempting force remount:', e, flush=True)
        drive.mount('/content/drive', force_remount=True)

checkout = Path('/content/zquoridor')
if not checkout.exists():
    print('Cloning repository...', flush=True)
    subprocess.run(['git', 'clone', 'https://github.com/gitzambrano/zquoridor.git', str(checkout)], check=True)
else:
    print('Updating repository to latest main...', flush=True)
    subprocess.run(['git', '-C', str(checkout), 'fetch', 'origin', 'main'], check=True)
    subprocess.run(['git', '-C', str(checkout), 'reset', '--hard', 'origin/main'], check=True)

os.chdir(checkout)

runtime_dir = Path('/content/zquoridor_runtime')
runtime_dir.mkdir(parents=True, exist_ok=True)
zq_bin = runtime_dir / 'zquoridor_uci'
print('Compiling native C++ engine (version 4.10) with AVX2 and FMA...', flush=True)
subprocess.run(['g++', '-O3', '-std=c++17', '-mavx2', '-mfma',
                '-I', str(checkout / 'src'),
                str(checkout / 'tools/external/zquoridor_uci.cpp'),
                '-o', str(zq_bin)], check=True)
zq_bin.chmod(0o755)

# Verify version 4.10 in UCI identity
chk = subprocess.run([str(zq_bin)], input='uci\\nquit\\n', capture_output=True, text=True, check=True, cwd=str(checkout))
ver_lines = [l.strip() for l in chk.stdout.splitlines() if 'id name' in l]
print('Engine verification:', ver_lines, flush=True)

# Clean stale temporary/backup files and ensure targets.npz is synchronized
output_dir = Path('{output_dir}')
if output_dir.is_dir():
    for stale_tmp in output_dir.glob('*.tmp*'):
        try:
            stale_tmp.unlink()
            print(f'Cleaned stale temporary file: {{stale_tmp.name}}', flush=True)
        except Exception:
            pass
    for stale_bak in output_dir.glob('*.bak*'):
        try:
            stale_bak.unlink()
            print(f'Cleaned old backup file: {{stale_bak.name}}', flush=True)
        except Exception:
            pass
    if (output_dir / 'positions.jsonl').is_file() and (output_dir / 'targets.jsonl').is_file():
        print('Verifying and synchronizing targets.npz from canonical JSONL records...', flush=True)
        subprocess.run(['python', 'tools/rebuild_npz_from_jsonl.py', '--dir', str(output_dir), '--no-backup'], check=True)

print('Launching: {job_label}', flush=True)
!{cli_command}
"""

JOB_SPECS = {
    1: {
        "job_label": "Wall Drops v4.10 (Medium Mazes: 3-8 walls) [Target: 500,000]",
        "output_dir": "/content/drive/MyDrive/zquoridor_data/wall_drops_v410/worker_1",
        "cli_command": "python -u tools/generate_wall_drops.py --count 500000 --min-walls 3 --max-walls 8 --pawn-walk-plies 4 --move-time-ms 20 --teacher zquoridor-4.10 --zq-executable /content/zquoridor_runtime/zquoridor_uci --output /content/drive/MyDrive/zquoridor_data/wall_drops_v410/worker_1 --seed 1000001",
    },
    2: {
        "job_label": "Wall Drops v4.10 (Extreme Mazes: 8-14 walls) [Target: 500,000]",
        "output_dir": "/content/drive/MyDrive/zquoridor_data/wall_drops_v410/worker_2",
        "cli_command": "python -u tools/generate_wall_drops.py --count 500000 --min-walls 8 --max-walls 14 --pawn-walk-plies 4 --move-time-ms 20 --teacher zquoridor-4.10 --zq-executable /content/zquoridor_runtime/zquoridor_uci --output /content/drive/MyDrive/zquoridor_data/wall_drops_v410/worker_2 --seed 2000002",
    },
    3: {
        "job_label": "Gumbel AlphaZero Sequential Halving v4.10 (Top-K=8, 15ms) [Target: 20,000]",
        "output_dir": "/content/drive/MyDrive/zquoridor_data/gumbel_sims_v410/worker_3",
        "cli_command": "python -u tools/generate_gumbel_sims.py --count 20000 --top-k 8 --c-scale 2.0 --move-time-ms 15 --teacher zquoridor-4.10-gumbel --zq-executable /content/zquoridor_runtime/zquoridor_uci --output /content/drive/MyDrive/zquoridor_data/gumbel_sims_v410/worker_3 --seed 3000003",
    },
    4: {
        "job_label": "Gumbel AlphaZero Sequential Halving v4.10 (Top-K=8, 15ms) [Target: 20,000]",
        "output_dir": "/content/drive/MyDrive/zquoridor_data/gumbel_sims_v410/worker_4",
        "cli_command": "python -u tools/generate_gumbel_sims.py --count 20000 --top-k 8 --c-scale 2.0 --move-time-ms 15 --teacher zquoridor-4.10-gumbel --zq-executable /content/zquoridor_runtime/zquoridor_uci --output /content/drive/MyDrive/zquoridor_data/gumbel_sims_v410/worker_4 --seed 4000004",
    },
    5: {
        "job_label": "Gumbel Deep Planning v4.10 (Top-K=16, 20ms) [Target: 20,000]",
        "output_dir": "/content/drive/MyDrive/zquoridor_data/gumbel_sims_v410/worker_5",
        "cli_command": "python -u tools/generate_gumbel_sims.py --count 20000 --top-k 16 --c-scale 2.5 --move-time-ms 20 --teacher zquoridor-4.10-gumbel --zq-executable /content/zquoridor_runtime/zquoridor_uci --output /content/drive/MyDrive/zquoridor_data/gumbel_sims_v410/worker_5 --seed 5000005",
    },
}


def handle_dialogs(page: Any) -> str:
    """Handle all known Colab modals accurately without closing them erroneously."""
    try:
        res = page.evaluate("""() => {
            const dialogs = Array.from(document.querySelectorAll('mwc-dialog, colab-dialog, paper-dialog, md-dialog, .modal-dialog')).filter(d => d.open || d.offsetParent !== null || (window.getComputedStyle(d).display !== 'none' && window.getComputedStyle(d).visibility !== 'hidden'));
            for (const d of dialogs) {
                const txt = (d.innerText || '');
                if (txt.includes('limites de uso do Colab') || txt.includes('GPU usage limits') || txt.includes('limite de uso') || txt.includes('usage limits')) {
                    for (const b of d.querySelectorAll('button, md-text-button, mwc-button, paper-button')) {
                        const btxt = (b.innerText || '').toLowerCase();
                        if (btxt.includes('conectar sem gpu') || btxt.includes('without gpu')) {
                            b.click();
                            return { action: 'gpu_quota_switched_to_cpu' };
                        }
                    }
                }
                if (txt.includes('Google Drive') || txt.includes('drive')) {
                    for (const b of d.querySelectorAll('button, md-text-button, mwc-button, paper-button, #ok')) {
                        const btxt = (b.innerText || '').toLowerCase();
                        if (btxt.includes('conectar') || btxt.includes('connect') || btxt.includes('permitir') || btxt.includes('allow') || b.id === 'ok') {
                            b.click();
                            return { action: 'drive_connected' };
                        }
                    }
                }
                for (const b of d.querySelectorAll('button, md-text-button, mwc-button, paper-button')) {
                    const btxt = (b.innerText || '').toLowerCase();
                    if (btxt.includes('executar mesmo assim') || btxt.includes('executar de qualquer maneira') || btxt.includes('run anyway')) {
                        b.click();
                        return { action: 'run_anyway' };
                    }
                }
                if (txt.includes('disconnected') || txt.includes('desconectado') || txt.includes('Falha ao executar') || txt.includes('Erro ao executar')) {
                    for (const b of d.querySelectorAll('button, md-text-button, mwc-button, paper-button')) {
                        const btxt = (b.innerText || '').toLowerCase();
                        if (btxt.includes('cancelar') || btxt.includes('ok') || btxt.includes('fechar') || btxt.includes('close')) {
                            b.click();
                            return { action: 'dismiss_error' };
                        }
                    }
                }
            }
            return { action: 'none' };
        }""")
        return res.get("action", "none")
    except Exception:
        return "none"


def handle_oauth(ctx: Any) -> bool:
    """Handle any Google OAuth consent popups (e.g. Google Drive access approval)."""
    try:
        oauth_pages = [pg for pg in ctx.pages if "accounts.google.com" in (pg.url or "")]
        if not oauth_pages:
            return False
        print(f"  [OAuth] Handling {len(oauth_pages)} OAuth consent popup(s)...", flush=True)
        for p in oauth_pages[:-1]:
            try:
                p.close()
            except Exception:
                pass
        pg = oauth_pages[-1]

        # Handle account selection tile if present
        acc_tiles = pg.locator("[data-identifier], [data-email]")
        if acc_tiles.count() > 0:
            print("  [OAuth] Account tile found, clicking...", flush=True)
            acc_tiles.first.click(timeout=5000)
            time.sleep(2)

        checkboxes = pg.locator("input[type='checkbox']").all()
        for cb in checkboxes:
            if not cb.is_checked():
                cb.check(timeout=5000)
                time.sleep(0.5)
        btn = pg.locator("#submit_approve_access, button:has-text('Allow'), button:has-text('Permitir'), button:has-text('Continuar'), button:has-text('Avançar')")
        if btn.count() > 0 and btn.first.is_visible():
            print("  [OAuth] Clicking approval button...", flush=True)
            btn.first.click(timeout=5000)
            time.sleep(3)
            return True
    except Exception as exc:
        print(f"  [OAuth ERROR] {exc}", flush=True)
    return False


def handle_recaptcha(page: Any) -> bool:
    """Detect and solve any active Google reCAPTCHA challenges."""
    try:
        has_iframe = page.evaluate("""() => {
            const frames = Array.from(document.querySelectorAll('iframe')).map(f => f.src);
            return frames.some(src => src.includes('recaptcha'));
        }""")
        if not has_iframe:
            return False

        anchor_frame = next((f for f in page.frames if "anchor" in f.url and ("recaptcha" in f.url or "k=" in f.url)), None)
        if anchor_frame:
            anchor = anchor_frame.locator("#recaptcha-anchor")
            if anchor.count() > 0:
                checked = anchor.get_attribute("aria-checked")
                if checked == "false":
                    print("  [reCAPTCHA] Checkbox is unchecked. Clicking anchor...", flush=True)
                    anchor.evaluate("el => el.click()")
                    time.sleep(3)
        bframe = next((f for f in page.frames if "bframe" in f.url), None)
        if bframe and bframe.locator("#recaptcha-audio-button").count() > 0:
            print("  [reCAPTCHA] Challenge detected. Solving via Whisper...", flush=True)
            return solve_recaptcha_playwright(page)
    except Exception as exc:
        print(f"  [reCAPTCHA ERROR] {exc}", flush=True)
    return False


def connect_runtime_if_needed(page: Any) -> bool:
    """Ensure runtime is connected if currently disconnected."""
    try:
        return page.evaluate("""() => {
            const nb = globalThis.colab?.global?.notebook;
            if (nb && typeof nb.connectKernel === 'function') {
                try { nb.connectKernel(); } catch (e) {}
            }
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


def get_cell_execution_state(page: Any) -> Dict[str, Any]:
    """Inspect execution status, pending state, error state, and current stdout tail."""
    try:
        return page.evaluate("""() => {
            let running = false;
            let pending = false;
            let hasError = false;
            let out = '';

            const nb = globalThis.colab?.global?.notebook;
            const cell = nb?.cells?.[0];
            const isBusy = cell ? !!cell.busy : false;
            const kernelBusy = nb?.kernel?.state === 'busy';
            const lastFailed = cell ? !!cell.lastExecutionFailed : false;

            const btn = document.querySelector('colab-run-button');
            const sr = btn ? btn.shadowRoot : null;
            const wrap = sr ? sr.querySelector('.cell-execution') : null;
            const cls = wrap ? wrap.className : '';

            // If it is busy, running, or animating, it is actively executing!
            if (isBusy || kernelBusy || cls.includes('running') || cls.includes('animating')) {
                running = true;
                hasError = false;
            } else if (cls.includes('pending') || cls.includes('waiting')) {
                pending = true;
                hasError = false;
            } else {
                running = false;
                if (cls.includes('error')) {
                    hasError = true;
                }
            }

            let out = '';
            const streams = Array.from(document.querySelectorAll('.output-stream, colab-output, .output_text'));
            if (streams.length > 0) {
                out = (streams[0].innerText || '').trim();
            }

            return {
                running: running,
                pending: pending,
                hasError: hasError,
                out: out.trim().slice(-400)
            };
        }""")
    except Exception as exc:
        return {"running": False, "pending": False, "hasError": True, "out": f"error: {exc}"}


def interrupt_cell(page: Any) -> bool:
    """Interrupt any currently executing cell in the Colab notebook."""
    try:
        return page.evaluate("""() => {
            const btn = document.querySelector('colab-run-button');
            const sr = btn ? btn.shadowRoot : null;
            const wrap = sr ? sr.querySelector('.cell-execution') : null;
            const cls = wrap ? wrap.className : '';
            if (cls.includes('running') || cls.includes('pending')) {
                const inner = sr ? sr.querySelector('#run-button') : null;
                if (inner) {
                    inner.click();
                    return true;
                }
            }
            const nb = globalThis.colab?.global?.notebook;
            const cell = nb?.cells ? nb.cells[0] : null;
            if (cell && typeof cell.interrupt === 'function') {
                try { cell.interrupt(); } catch (e) {}
            }
            if (nb && typeof nb.interrupt === 'function') {
                try { nb.interrupt(); } catch (e) {}
            }
            return false;
        }""")
    except Exception:
        return False


def set_cell_code(page: Any, code: str) -> bool:
    """Inject Python source code into Cell 0 of the Colab notebook and purge duplicate cells."""
    try:
        return page.evaluate("""(newCode) => {
            const nb = globalThis.colab?.global?.notebook;
            if (!nb) return false;
            if (!nb.cells || nb.cells.length === 0) {
                if (typeof nb.appendCodeCell === 'function') {
                    nb.appendCodeCell();
                } else {
                    const addBtn = document.querySelector('#toolbar-add-code, colab-toolbar-button#add-code, button[aria-label*="Código"], button[aria-label*="Code"]');
                    if (addBtn) addBtn.click();
                }
            }
            if (!nb.cells || nb.cells.length === 0) return false;

            // Delete any extra duplicate cells from previous runs
            while (nb.cells && nb.cells.length > 1) {
                if (typeof nb.deleteCell === 'function') {
                    nb.deleteCell(nb.cells[1]);
                } else {
                    break;
                }
            }

            const cell = nb.cells[0];
            if (cell) cell.lastExecutionFailed = false;
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
                try { nb.saveNotebook(); } catch (e) {}
            }
            return true;
        }""", code)
    except Exception:
        return False


def click_run_button(page: Any) -> bool:
    """Trigger execution for Cell 0 via cell focus and Control+Enter keyboard shortcut."""
    try:
        page.evaluate("""() => {
            const nb = globalThis.colab?.global?.notebook;
            const cell = nb?.cells?.[0];
            if (cell) {
                if (cell.element) cell.element.click();
                if (typeof cell.focus === 'function') cell.focus();
            }
        }""")
        time.sleep(0.5)
        page.keyboard.press("Control+Enter")
        return True
    except Exception:
        return False


def deploy_worker(p: Any, wid: int, force_relaunch: bool = True) -> Dict[str, Any]:
    """Connect to worker browser context, inject rich target code, and trigger execution."""
    w = WORKERS[wid]
    port = w["cdp_port"]
    spec = JOB_SPECS[wid]

    print(f"\n=======================================================", flush=True)
    print(f"DEPLOYING WORKER {wid} ({w['account']}) - PORT {port}", flush=True)
    print(f"Job: {spec['job_label']}", flush=True)
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
    elif page.url == "about:blank" or w["notebook_url"] not in (page.url or ""):
        page.goto(w["notebook_url"], wait_until="domcontentloaded", timeout=60000)

    time.sleep(3)
    handle_oauth(ctx)
    handle_recaptcha(page)
    handle_dialogs(page)
    connect_runtime_if_needed(page)
    time.sleep(2)
    handle_oauth(ctx)
    handle_dialogs(page)

    has_target_code = page.evaluate("""(cmd) => {
        const c = globalThis.colab?.global?.notebook?.cells?.[0];
        const t = c && c.getText ? c.getText() : '';
        return t.includes(cmd);
    }""", spec["cli_command"])

    st = get_cell_execution_state(page)
    is_active = (st["running"] or st["pending"]) and not st.get("hasError", False)

    if is_active and has_target_code and not force_relaunch:
        print(f"  Worker {wid} is already running the target job! Preserving current execution.", flush=True)
    else:
        if st["running"] or st["pending"]:
            print(f"  Worker {wid}: Stopping current execution to ensure latest git commits...", flush=True)
            interrupt_cell(page)
            for _ in range(6):
                time.sleep(2)
                handle_dialogs(page)
                st_now = get_cell_execution_state(page)
                if not (st_now["running"] or st_now["pending"]):
                    break

        code = BOOTLOADER_TEMPLATE.format(
            job_label=spec["job_label"],
            cli_command=spec["cli_command"],
            output_dir=spec["output_dir"],
        )
        print("  Injecting code into Cell 0 and purging duplicate cells...", flush=True)
        set_cell_code(page, code)
        time.sleep(2)

        connect_runtime_if_needed(page)
        time.sleep(1)
        handle_dialogs(page)
        print("  Triggering run execution...", flush=True)
        click_run_button(page)

    for _ in range(4):
        time.sleep(2)
        handle_oauth(ctx)
        handle_recaptcha(page)
        handle_dialogs(page)
        connect_runtime_if_needed(page)

    st = get_cell_execution_state(page)
    print(f"  Worker {wid} deployed: running={st['running']}, pending={st['pending']}, hasError={st.get('hasError')}", flush=True)

    return {
        "wid": wid,
        "account": w["account"],
        "job_label": spec["job_label"],
        "ctx": ctx,
        "page": page,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force-relaunch", dest="force_relaunch", action="store_true", default=False,
                        help="Preserve active worker execution instead of relaunching with version 4.10.")
    parser.add_argument("--workers", type=int, nargs="+", default=[1, 2, 3, 4, 5],
                        help="List of worker IDs to deploy and monitor.")
    args = parser.parse_args()

    print("=================================================================", flush=True)
    print("DISPATCHING RICH TARGETS (WALL DROPS + GUMBEL) - VERSION 4.10", flush=True)
    print("=================================================================", flush=True)

    workers_state: Dict[int, Dict[str, Any]] = {}

    with sync_playwright() as p:
        for wid in args.workers:
            try:
                workers_state[wid] = deploy_worker(p, wid, force_relaunch=args.force_relaunch)
            except Exception as exc:
                print(f"Worker {wid} initial deployment error: {exc}", flush=True)

        print("\n================ FLEET MONITOR LOOP STARTED ================\n", flush=True)
        cycle = 0
        while True:
            cycle += 1
            try:
                time.sleep(25)
                print(f"\n--- FLEET MONITOR [Cycle {cycle} | {time.strftime('%H:%M:%S')}] ---", flush=True)

                for wid in args.workers:
                    st_info = workers_state.get(wid)
                    page = st_info.get("page") if st_info else None
                    ctx = st_info.get("ctx") if st_info else None

                    # Self-healing: if context was closed or disconnected, recover worker
                    if not st_info or page is None or page.is_closed():
                        print(f"  Worker {wid} context lost/closed. Re-deploying and recovering...", flush=True)
                        try:
                            workers_state[wid] = deploy_worker(p, wid, force_relaunch=False)
                            st_info = workers_state.get(wid)
                            page = st_info.get("page") if st_info else None
                            ctx = st_info.get("ctx") if st_info else None
                        except Exception as e:
                            print(f"  Worker {wid} recovery error: {e}", flush=True)
                            continue

                    if not page:
                        continue

                    try:
                        if ctx:
                            handle_oauth(ctx)
                        handle_recaptcha(page)
                        handle_dialogs(page)
                        st = get_cell_execution_state(page)
                        is_active = (st["running"] or st["pending"]) and not st.get("hasError", False)

                        if not is_active:
                            if "already satisfied" in st.get("out", ""):
                                print(f"  Worker {wid} [{st_info['job_label']}] [COMPLETED] -> All targets satisfied.", flush=True)
                            else:
                                print(f"  Worker {wid} not active (running={st['running']}, pending={st.get('pending')}, hasError={st.get('hasError')}). Re-running...", flush=True)
                                connect_runtime_if_needed(page)
                                handle_dialogs(page)
                                spec = JOB_SPECS[wid]
                                code = BOOTLOADER_TEMPLATE.format(
                                    job_label=spec["job_label"],
                                    cli_command=spec["cli_command"],
                                    output_dir=spec["output_dir"],
                                )
                                set_cell_code(page, code)
                                time.sleep(2)
                                connect_runtime_if_needed(page)
                                time.sleep(1)
                                handle_dialogs(page)
                                click_run_button(page)
                                time.sleep(4)
                                if ctx:
                                    handle_oauth(ctx)
                                handle_dialogs(page)
                                st = get_cell_execution_state(page)
                                is_active = (st["running"] or st["pending"]) and not st.get("hasError", False)

                        raw_out = st["out"].replace("\n", " ").strip()
                        out_snip = raw_out[-100:] if raw_out else "(compiling / initializing environment...)"
                        tag = "RUNNING" if is_active else "STOPPED"
                        print(f"  Worker {wid} [{st_info['job_label']}] [{tag}] -> {out_snip}", flush=True)

                        try:
                            page.mouse.move(50, 50)
                        except Exception:
                            pass

                    except Exception as exc:
                        print(f"  Worker {wid} check error: {exc}", flush=True)
                        if "closed" in str(exc).lower() or "target" in str(exc).lower():
                            workers_state[wid] = None
            except Exception as cycle_exc:
                print(f"Cycle {cycle} error: {cycle_exc}. Retrying in 10s...", flush=True)
                time.sleep(10)


if __name__ == "__main__":
    from colab.control import invocation_lease
    with invocation_lease(globals().get("CONFIG", {}), "playwright"):
        main()
