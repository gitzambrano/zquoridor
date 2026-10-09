#!/usr/bin/env python3
"""Resume the frozen Edge campaign, retrying transient Windows status-file locks."""
from __future__ import annotations

import ctypes
import os
import sys
import time
from ctypes import wintypes
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools import run_edge_acc_queue as queue


_save_once = queue.Queue.save


def save_with_windows_lock_retry(self) -> None:
    delay = 0.1
    for attempt in range(12):
        try:
            _save_once(self)
            return
        except PermissionError:
            if attempt == 11:
                raise
            print(f"Transient status-file lock; retrying save ({attempt + 1}/12).", flush=True)
            time.sleep(delay)
            delay = min(delay * 1.7, 2.0)


queue.Queue.save = save_with_windows_lock_retry


_campaign_wrapper_command = queue.campaign_wrapper_command


def campaign_wrapper_command_with_worker_cap(arguments: list[str]) -> list[str]:
    command = _campaign_wrapper_command(arguments)
    return [argument.replace("h2h_workers=14", "h2h_workers=12")
            if isinstance(argument, str) else argument for argument in command]


queue.campaign_wrapper_command = campaign_wrapper_command_with_worker_cap


def limit_process_affinity() -> None:
    if os.name != "nt":
        return
    cpu_count = os.cpu_count() or 12
    mask = (1 << min(12, cpu_count)) - 1
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    kernel.SetProcessAffinityMask.argtypes = [wintypes.HANDLE, ctypes.c_size_t]
    kernel.SetProcessAffinityMask.restype = wintypes.BOOL
    if not kernel.SetProcessAffinityMask(kernel.GetCurrentProcess(), mask):
        raise ctypes.WinError(ctypes.get_last_error())

if __name__ == "__main__":
    limit_process_affinity()
    raise SystemExit(queue.main())
