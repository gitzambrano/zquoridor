from pathlib import Path
import sys
import time
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

def solve_recaptcha_js(page):
    try:
        anchor_frames = [f for f in page.frames if "k=6LfQttQU" in f.url and "anchor" in f.url]
        if not anchor_frames:
            return False
        anchor = anchor_frames[-1].locator("#recaptcha-anchor")
        if anchor.count() > 0:
            checked = anchor.get_attribute("aria-checked")
            if checked != "true":
                print("  [reCAPTCHA] Found unchecked anchor, clicking via JS...", flush=True)
                anchor.evaluate("el => el.click()")
                time.sleep(3)
                return True
    except Exception as e:
        print(f"  [reCAPTCHA error] {e}", flush=True)
    return False

def handle_all_modals(page):
    res = page.evaluate("""() => {
        const dialogs = Array.from(document.querySelectorAll('mwc-dialog, colab-dialog')).filter(d => d.open || d.offsetParent !== null);
        for (const d of dialogs) {
            const txt = (d.innerText || '');
            // Check GPU quota
            if (txt.includes('limites de uso do Colab') || txt.includes('GPU usage limits') || txt.includes('limite de uso') || txt.includes('usage limits')) {
                for (const b of d.querySelectorAll('button, md-text-button, mwc-button, paper-button')) {
                    const btxt = (b.innerText || '').toLowerCase();
                    if (btxt.includes('conectar sem gpu') || btxt.includes('without gpu')) {
                        b.click();
                        return { action: 'gpu_quota_cpu_selected' };
                    }
                }
            }
            // Check Drive
            for (const b of d.querySelectorAll('button, md-text-button, mwc-button, paper-button')) {
                const btxt = (b.innerText || '').toLowerCase();
                if (btxt.includes('conectar ao google drive') || btxt.includes('connect to google drive')) {
                    b.click();
                    return { action: 'drive_connected' };
                }
            }
            // Check Run anyway
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
    if res.get("action") != "none":
        print(f"  [MODAL HANDLED] {res['action']}", flush=True)
        time.sleep(2)
        return res["action"]
    return None

def click_cell_run(page):
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

def set_cell_code(page, code):
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

def get_cell_status(page):
    return page.evaluate("""() => {
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
            out: out.replace(/\\n/g, ' ').trim()
        };
    }""")

def test_worker(wid: int, mode_choice: str = "cpu"):
    w = WORKERS[wid]
    port = w["cdp_port"]
    print(f"Testing Worker {wid} ({mode_choice.upper()}) on port {port}...", flush=True)
    with sync_playwright() as p:
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

        handle_all_modals(page)
        solve_recaptcha_js(page)

        profile = build_worker_profile(wid)
        if mode_choice == "cpu":
            seed = int(PROFILE_CONFIG["seed_stride"]) * wid
            rt_dir = profile['runtime_dir_literal'].strip("'")
            import shlex
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
                "--output", f"/content/drive/MyDrive/zquoridor_data/selfplay_858_v1/worker_{wid}",
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
            code = CPU_SELFPLAY_BOOTLOADER.format(**profile, cmd_args=profile["extra_args"])
        else:
            code = BOOTLOADER_TEMPLATE.format(**profile, cmd_args=profile["extra_args"])

        print("Setting code...", flush=True)
        set_cell_code(page, code)
        print("Clicking run...", flush=True)
        click_cell_run(page)

        for i in range(12):
            time.sleep(5)
            handle_all_modals(page)
            solve_recaptcha_js(page)
            # Check OAuth popups
            if len(ctx.pages) > 1:
                for pg in ctx.pages[1:]:
                    if "accounts.google.com" in (pg.url or ""):
                        print("  [OAUTH] Authorizing Drive popup...", flush=True)
                        btn = pg.locator("#submit_approve_access, button:has-text('Allow'), button:has-text('Permitir')")
                        if btn.count() > 0:
                            btn.first.click()
            st = get_cell_status(page)
            print(f"  Check {i+1} ({(i+1)*5}s): running={st['running']} out='{st['out'][-100:]}'", flush=True)
            if not st['running'] and not st['out']:
                print("  Re-clicking run button...", flush=True)
                click_cell_run(page)

        ctx.close()

if __name__ == "__main__":
    wid = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    m = sys.argv[2] if len(sys.argv) > 2 else "cpu"
    test_worker(wid, m)
