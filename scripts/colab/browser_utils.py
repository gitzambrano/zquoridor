"""Browser session utilities and safe Playwright helpers for Zquoridor Colab automation.

Provides CDP connection fallback, profile lock detection, modal handling,
and safe execution primitives without terminating active Chrome processes.
"""

import os
import socket
import time
from typing import Dict, Any, Optional, Tuple
import psutil


def is_cdp_reachable(port: int, host: str = "127.0.0.1", timeout: float = 1.0) -> bool:
    """Check whether a Chrome DevTools Protocol port is open and accepting connections."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except (socket.timeout, ConnectionRefusedError, OSError):
        return False


def is_profile_in_use(profile_dir: str) -> Tuple[bool, Optional[int]]:
    """Check if any running chrome.exe process is currently using the specified user data directory."""
    target_name = os.path.basename(os.path.normpath(profile_dir)).lower()
    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            p_name = proc.info.get("name") or ""
            if "chrome" in p_name.lower():
                cmdline = proc.info.get("cmdline") or []
                cmd_str = " ".join(cmdline).lower()
                if target_name in cmd_str and ("user-data-dir" in cmd_str or "profile" in cmd_str):
                    return True, proc.info.get("pid")
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return False, None


def dismiss_modals(page: Any) -> bool:
    """Detect and click common Colab warning and authorization confirmation dialogs."""
    try:
        return page.evaluate("""() => {
            const buttons = Array.from(document.querySelectorAll(
                'button, md-text-button, paper-button, .colab-dialog-button, colab-dialog button'
            ));
            const targets = [
                'Executar mesmo assim',
                'Executar de qualquer maneira',
                'Run anyway',
                'Conectar ao Google Drive',
                'Connect to Google Drive',
                'Continuar',
                'Continue',
                'OK',
                'Ok'
            ];
            for (const b of buttons) {
                const txt = (b.innerText || '').trim();
                for (const t of targets) {
                    if (txt === t || txt.includes(t)) {
                        b.click();
                        return true;
                    }
                }
            }
            return false;
        }""")
    except Exception:
        return False


def connect_runtime_if_needed(page: Any) -> bool:
    """Ensure the Colab runtime kernel is connected."""
    try:
        return page.evaluate("""() => {
            const btn = document.querySelector('colab-connect-button');
            if (btn && btn.shadowRoot) {
                const conn = btn.shadowRoot.querySelector('#connect');
                if (conn && (conn.innerText.includes('Conectar') || conn.innerText.includes('Reconectar') || conn.innerText.includes('Connect'))) {
                    conn.click();
                    return true;
                }
            }
            return false;
        }""")
    except Exception:
        return False


def get_notebook_dom_state(page: Any, target_keywords: list = None) -> Dict[str, Any]:
    """Inspect the Colab notebook DOM and extract runtime and cell execution state."""
    if target_keywords is None:
        target_keywords = ["run_colab_worker.py", "selfplay_15m"]

    return page.evaluate("""(keywords) => {
        const curUrl = window.location.href || '';
        const bodyText = document.body ? (document.body.innerText || '') : '';
        if (curUrl.includes('accounts.google.com') || curUrl.includes('signin') || bodyText.includes('Confirme que') || bodyText.includes('Confirm it')) {
            return {
                statusText: 'Auth Required (Google Login)',
                kConnected: false,
                running: false,
                pending: false,
                targetIndex: -1,
                outText: 'Google account requires verification (Confirm it is you).'
            };
        }

        const btn = document.querySelector('colab-connect-button');
        const sr = btn ? btn.shadowRoot : null;
        const connBtn = sr ? sr.querySelector('#connect') : null;
        const statusText = connBtn ? connBtn.innerText.trim() : (btn ? btn.innerText.trim() : 'no button');

        const k = typeof colab !== 'undefined' && colab.global && colab.global.notebook ? colab.global.notebook.kernel : null;
        const kConnected = k && k.isConnected ? k.isConnected() : false;

        const cells = typeof colab !== 'undefined' && colab.global && colab.global.notebook && colab.global.notebook.cells ? colab.global.notebook.cells : [];
        let targetCell = null;
        let targetIndex = -1;

        for (let i = cells.length - 1; i >= 0; i--) {
            const txt = cells[i].getText() || '';
            for (const kw of keywords) {
                if (txt.includes(kw)) {
                    targetCell = cells[i];
                    targetIndex = i;
                    break;
                }
            }
            if (targetCell) break;
        }

        if (!targetCell && cells.length > 0) {
            targetCell = cells[cells.length - 1];
            targetIndex = cells.length - 1;
        }

        const isRunningBtn = Array.from(document.querySelectorAll(
            'colab-run-button[title*="Interromper"], colab-run-button[aria-label*="Interromper"], colab-run-button.running'
        ));

        const nb = typeof colab !== 'undefined' && colab.global ? colab.global.notebook : null;
        let running = isRunningBtn.length > 0;
        if (nb && typeof nb.isExecuting === 'function') {
            running = running || nb.isExecuting();
        }
        let pending = false;
        let outText = '';

        if (targetCell) {
            running = running || (targetCell.isRunning ? targetCell.isRunning() : false);
            pending = targetCell.isPending ? targetCell.isPending() : false;
            const dom = targetCell.getElement ? targetCell.getElement() : (targetCell.element_ || targetCell.dom_);
            const outDiv = dom ? dom.querySelector('.output, colab-output, .output-stream, .output_text') : null;
            outText = outDiv ? outDiv.innerText.slice(-1200) : '';
        }

        return {
            statusText,
            kConnected,
            running,
            pending,
            targetIndex,
            outText
        };
    }""", target_keywords)


def trigger_cell_execution(page: Any, worker_id: int, bootloader_template: str) -> bool:
    """Inject bootloader code into the Colab target cell and trigger execution."""
    code = bootloader_template.format(worker_id=worker_id)
    try:
        res = page.evaluate("""(newCode) => {
            const nb = typeof colab !== 'undefined' && colab.global ? colab.global.notebook : null;
            if (!nb || !nb.cells || nb.cells.length === 0) return { success: false, reason: 'no_cells' };

            const cells = nb.cells;
            let targetCell = null;
            for (let i = cells.length - 1; i >= 0; i--) {
                const txt = cells[i].getText ? cells[i].getText() : '';
                if (txt.includes('run_colab_worker.py') || txt.includes('selfplay_15m') || txt.includes('zquoridor')) {
                    targetCell = cells[i];
                    break;
                }
            }
            if (!targetCell) {
                targetCell = cells[cells.length - 1];
            }

            if (targetCell.isRunning && targetCell.isRunning()) {
                return { success: true, alreadyRunning: true };
            }

            if (targetCell.model && targetCell.model.setText) {
                targetCell.model.setText(newCode);
            }
            if (targetCell.model && targetCell.model.removeOutputs) {
                targetCell.model.removeOutputs();
            }

            if (typeof monaco !== 'undefined') {
                for (const m of monaco.editor.getModels()) {
                    const val = m.getValue ? m.getValue() : '';
                    if (val.includes('run_colab_worker.py') || val.includes('selfplay_15m') || val.includes('zquoridor')) {
                        m.setValue(newCode);
                    }
                }
            }

            const elem = targetCell.getElement ? targetCell.getElement() : (targetCell.element_ || targetCell.dom_);
            if (elem && elem.scrollIntoView) elem.scrollIntoView();

            if (typeof targetCell.manualExecute === 'function') {
                targetCell.manualExecute();
                return { success: true, method: 'manualExecute' };
            } else if (elem) {
                const btn = elem.querySelector('colab-run-button');
                if (btn) {
                    if (btn.shadowRoot) {
                        const inner = btn.shadowRoot.querySelector('button, [role="button"]');
                        if (inner) inner.click();
                        else btn.click();
                    } else {
                        btn.click();
                    }
                    return { success: true, method: 'colab-run-button' };
                }
            }
            return { success: false, reason: 'no_run_method' };
        }""", code)
        time.sleep(3)
        dismiss_modals(page)
        return res.get("success", False) if isinstance(res, dict) else False
    except Exception:
        return False

