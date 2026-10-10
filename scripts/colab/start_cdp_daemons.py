"""Launch all 5 Colab Chrome instances headlessly with remote debugging ports 9001-9005."""
import socket
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.colab.config import WORKERS

CHROME_BIN = r"C:\Program Files\Google\Chrome\Application\chrome.exe"

def is_port_open(port: int) -> bool:
    s = socket.socket()
    s.settimeout(1.0)
    try:
        s.connect(("127.0.0.1", port))
        return True
    except Exception:
        return False
    finally:
        s.close()

def start_daemons():
    print("==================================================", flush=True)
    print("STARTING COLAB CHROME CDP DAEMONS (PORTS 9001-9005)", flush=True)
    print("==================================================", flush=True)

    for wid, w in WORKERS.items():
        port = w["cdp_port"]
        profile = w["profile_dir"]
        url = w["notebook_url"]

        if is_port_open(port):
            print(f"Worker {wid} (Port {port}): Already running and open.", flush=True)
            continue

        cmd = [
            CHROME_BIN,
            f"--remote-debugging-port={port}",
            f"--user-data-dir={profile}",
            "--no-first-run",
            "--no-default-browser-check",
            "--headless=new",
            url
        ]
        proc = subprocess.Popen(cmd)
        print(f"Worker {wid} (Port {port}): Started chrome.exe (PID {proc.pid})...", flush=True)

    time.sleep(5)
    print("\n--- Verifying CDP Port Connectivity ---", flush=True)
    for wid, w in WORKERS.items():
        port = w["cdp_port"]
        ok = is_port_open(port)
        status = "ONLINE" if ok else "OFFLINE"
        print(f"  Worker {wid} (Port {port}): {status}", flush=True)

if __name__ == "__main__":
    start_daemons()
