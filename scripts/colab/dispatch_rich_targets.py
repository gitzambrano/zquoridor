#!/usr/bin/env python3
"""Dispatch Wall-Drop and Gumbel simulation jobs across all 5 Google Colab workers."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import time
from typing import Any, Dict

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.colab.config import WORKERS
from scripts.colab.browser_utils import is_cdp_reachable, solve_recaptcha_playwright

BOOTLOADER_TEMPLATE = """from google.colab import drive
import os
import shutil
import subprocess
from pathlib import Path

if not Path('/content/drive/MyDrive').is_dir():
    drive.mount('/content/drive')

checkout = Path('/content/zquoridor')
if not checkout.exists():
    print('Cloning repository...', flush=True)
    subprocess.run(['git', 'clone', 'https://github.com/gitzambrano/zquoridor.git', str(checkout)], check=True)
else:
    print('Updating repository to latest main...', flush=True)
    subprocess.run(['git', '-C', str(checkout), 'fetch', 'origin', 'main'], check=True)
    subprocess.run(['git', '-C', str(checkout), 'reset', '--hard', 'origin/main'], check=True)

runtime_dir = Path('/content/zquoridor_runtime')
runtime_dir.mkdir(parents=True, exist_ok=True)
zq_bin = runtime_dir / 'zquoridor_uci'
if not zq_bin.exists():
    print('Compiling native C++ engine with AVX2 and FMA...', flush=True)
    subprocess.run(['g++', '-O3', '-std=c++17', '-mavx2', '-mfma',
                    '-I', str(checkout / 'src'),
                    str(checkout / 'tools/external/zquoridor_uci.cpp'),
                    '-o', str(zq_bin)], check=True)
    zq_bin.chmod(0o755)

os.chdir(checkout)
print('Launching: {job_label}', flush=True)
!{cli_command}
"""

JOB_SPECS = {
    1: {
        "job_label": "Wall Drops (Medium Mazes: 3-8 walls)",
        "cli_command": "python -u tools/generate_wall_drops.py --count 10000 --min-walls 3 --max-walls 8 --pawn-walk-plies 4 --move-time-ms 20 --zq-executable /content/zquoridor_runtime/zquoridor_uci --output /content/drive/MyDrive/zquoridor_data/wall_drops_v1/worker_1 --seed 1000001",
    },
    2: {
        "job_label": "Wall Drops (Extreme Mazes: 8-14 walls)",
        "cli_command": "python -u tools/generate_wall_drops.py --count 10000 --min-walls 8 --max-walls 14 --pawn-walk-plies 4 --move-time-ms 20 --zq-executable /content/zquoridor_runtime/zquoridor_uci --output /content/drive/MyDrive/zquoridor_data/wall_drops_v1/worker_2 --seed 2000002",
    },
    3: {
        "job_label": "Gumbel AlphaZero Sequential Halving (Top-K=8, 15ms)",
        "cli_command": "python -u tools/generate_gumbel_sims.py --count 10000 --top-k 8 --c-scale 2.0 --move-time-ms 15 --zq-executable /content/zquoridor_runtime/zquoridor_uci --output /content/drive/MyDrive/zquoridor_data/gumbel_sims_v1/worker_3 --seed 3000003",
    },
    4: {
        "job_label": "Gumbel AlphaZero Sequential Halving (Top-K=8, 15ms)",
        "cli_command": "python -u tools/generate_gumbel_sims.py --count 10000 --top-k 8 --c-scale 2.0 --move-time-ms 15 --zq-executable /content/zquoridor_runtime/zquoridor_uci --output /content/drive/MyDrive/zquoridor_data/gumbel_sims_v1/worker_4 --seed 4000004",
    },
    5: {
        "job_label": "Gumbel Deep Planning (Top-K=16, 20ms)",
        "cli_command": "python -u tools/generate_gumbel_sims.py --count 10000 --top-k 16 --c-scale 2.5 --move-time-ms 20 --zq-executable /content/zquoridor_runtime/zquoridor_uci --output /content/drive/MyDrive/zquoridor_data/gumbel_sims_v1/worker_5 --seed 5000005",
    },
}


def handle_dialogs(page: Any) -> str:
    """Handle all known Colab modals accurately without closing them erroneously."""
    try:
        res = page.evaluate("""() => {
            const dialogs = Array.from(document.querySelectorAll('mwc-dialog, colab-dialog')).filter(d => d.open || d.offsetParent !== null);
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
                for (const b of d.querySelectorAll('button, md-text-button, mwc-button, paper-button')) {
                    const btxt = (b.innerText || '').toLowerCase();
                    if (btxt.includes('conectar ao google drive') || btxt.includes('connect to google drive')) {
                        b.click();
                        return { action: 'drive_connected' };
                    }
                }
                for (const b of d.querySelectorAll('button, md-text-button, mwc-button, paper-button')) {
                    const btxt = (b.innerText || '').toLowerCase();
                    if (btxt.includes('executar mesmo assim') || btxt.includes('executar de qualquer maneira') || btxt.includes('run anyway')) {
                        b.click();
                        return { action: 'run_anyway' };
                    }
                }
                // Dismiss disconnected / error notification
                if (txt.includes('disconnected') || txt.includes('desconectado') || txt.includes('Falha ao executar')) {
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
            }

            return {
                running: running,
                pending: pending,
                out: out.trim().slice(-300)
            };
        }""")
    except Exception as exc:
        return {"running": False, "pending": False, "out": f"error: {exc}"}


def deploy_worker(p: Any, wid: int) -> Dict[str, Any]:
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

    time.sleep(3)
    handle_dialogs(page)
    connect_runtime_if_needed(page)
    time.sleep(2)
    handle_dialogs(page)

    init_st = get_cell_execution_state(page)
    if init_st["running"] or init_st["pending"]:
        print(f"  Worker {wid} is already running/pending! Preserving current execution.", flush=True)
    else:
        code = BOOTLOADER_TEMPLATE.format(job_label=spec["job_label"], cli_command=spec["cli_command"])
        print("  Injecting code into Cell 0...", flush=True)
        set_cell_code(page, code)
        time.sleep(2)

        print("  Triggering run execution...", flush=True)
        click_run_button(page)

    for _ in range(3):
        time.sleep(2)
        handle_dialogs(page)
        connect_runtime_if_needed(page)

    st = get_cell_execution_state(page)
    print(f"  Worker {wid} deployed: running={st['running']}, pending={st['pending']}", flush=True)

    return {
        "wid": wid,
        "account": w["account"],
        "job_label": spec["job_label"],
        "ctx": ctx,
        "page": page,
    }


def main():
    print("=================================================================", flush=True)
    print("DISPATCHING RICH TARGETS (WALL DROPS + GUMBEL) - ALL 5 WORKERS", flush=True)
    print("=================================================================", flush=True)

    workers_state: Dict[int, Dict[str, Any]] = {}

    with sync_playwright() as p:
        for wid in [1, 2, 3, 4, 5]:
            try:
                workers_state[wid] = deploy_worker(p, wid)
            except Exception as exc:
                print(f"Worker {wid} deployment error: {exc}", flush=True)

        print("\n================ FLEET MONITOR LOOP STARTED ================\n", flush=True)
        cycle = 0
        while True:
            cycle += 1
            try:
                time.sleep(25)
                print(f"\n--- FLEET MONITOR [Cycle {cycle} | {time.strftime('%H:%M:%S')}] ---", flush=True)

                for wid in [1, 2, 3, 4, 5]:
                    st_info = workers_state.get(wid)
                    if not st_info:
                        continue
                    page = st_info["page"]

                    try:
                        handle_dialogs(page)
                        st = get_cell_execution_state(page)
                        is_active = st["running"] or st["pending"]

                        if not is_active:
                            connect_runtime_if_needed(page)
                            handle_dialogs(page)
                            click_run_button(page)
                            time.sleep(2)
                            st = get_cell_execution_state(page)
                            is_active = st["running"] or st["pending"]

                        raw_out = st["out"].replace("\n", " ").strip()
                        out_snip = raw_out[-80:] if raw_out else "(compiling / initializing environment...)"
                        tag = "RUNNING" if is_active else "STOPPED"
                        print(f"  Worker {wid} [{st_info['job_label']}] [{tag}] -> {out_snip}", flush=True)

                        try:
                            page.mouse.move(50, 50)
                        except Exception:
                            pass

                    except Exception as exc:
                        print(f"  Worker {wid} check error: {exc}", flush=True)
            except Exception as cycle_exc:
                print(f"Cycle {cycle} error: {cycle_exc}. Retrying in 10s...", flush=True)
                time.sleep(10)


if __name__ == "__main__":
    main()
