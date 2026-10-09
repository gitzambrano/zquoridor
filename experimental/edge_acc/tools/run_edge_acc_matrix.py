#!/usr/bin/env python3
"""Run the remaining Edge matrix with one sequential, bounded queue."""
from __future__ import annotations

import argparse
import ctypes
from ctypes import wintypes
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools import run_edge_acc_campaign as campaign
from tools import run_edge_acc_queue as legacy_queue

CONFIG = {
    "output": ROOT / "results/benchmarks/edge_acc_campaign_clock_v3",
    "previous_output": ROOT / "results/benchmarks/edge_acc_campaign_clock_v2",
    "cpu_limit": 8,
    "h2h_workers": 4,
    "ram_reserve_mib": 2048,
    "fixed_workers": True,
    "modest_start": False,
    "modest_start_free_mib": 4096,
    "h2h_ram_per_game_mib": 6144,
    "external_ram_per_game_mib": 3072,
    "titanium_ram_per_game_mib": 2048,
    "claustrophobia_workers": 4,
    "gpu_reserve_mib": 2048,
    "gpu_worker_mib": 256,
    "poll_seconds": 5,
    "report_interval_seconds": 300,
    "infrastructure_retries": 1,
    "shards": 4,
    "networks": ["504", "858"],
    "variants": ["delta_dense_only", "v3_node_dense_bfs", "v3"],
    "report_script": ROOT / "tools/report_edge_acc_campaign_clock_v2.py",
    "protocol": "edge-acc-native-clock-v3-forfeit",
}


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    for attempt in range(12):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 11:
                raise
            time.sleep(0.25)


def apply_limits() -> None:
    if not 1 <= CONFIG["h2h_workers"] <= 6 or not CONFIG["h2h_workers"] <= CONFIG["cpu_limit"] <= 12:
        raise RuntimeError("The CPU limit must not exceed 12 and the H2H game limit must not exceed 6")
    if not 1 <= CONFIG["claustrophobia_workers"] <= 6:
        raise RuntimeError("The Claustrophobia worker limit must not exceed 6")
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    if os.name == "nt":
        mask = (1 << min(CONFIG["cpu_limit"], os.cpu_count() or CONFIG["cpu_limit"])) - 1
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        kernel.SetProcessAffinityMask.argtypes = [wintypes.HANDLE, ctypes.c_size_t]
        if not kernel.SetProcessAffinityMask(kernel.GetCurrentProcess(), mask):
            raise ctypes.WinError(ctypes.get_last_error())


def task_plan() -> list[dict]:
    tasks = []
    for shard in range(CONFIG["shards"]):
        for network in CONFIG["networks"]:
            for variant in CONFIG["variants"]:
                tasks.append(dict(name=f"h2h_{network}_{variant}_{shard}", mode="h2h",
                                  network=network, variant=variant, shard=shard))
        tasks.append(dict(name=f"cross_hybrid858_main504_{shard}", mode="cross", shard=shard))
        for opponent in ("claustrophobia", "titanium"):
            for network in CONFIG["networks"]:
                for variant in ["off", *CONFIG["variants"]]:
                    tasks.append(dict(name=f"{opponent}_{network}_{variant}_{shard}", mode="external",
                                      network=network, variant=variant, opponent=opponent, shard=shard))
    return tasks


def source_identity() -> dict:
    previous = json.loads((CONFIG["previous_output"] / "queue/identity.json").read_text())
    expected = previous["sources"]
    engines = {"experimental_source": campaign.source_hash(ROOT),
               "main_source": campaign.source_hash(campaign.MAIN),
               "weights": {net: campaign.sha256(Path(campaign.CONFIG["networks"][net]["weights"]))
                           for net in CONFIG["networks"]}}
    for field, value in engines.items():
        if value != expected[field]:
            raise RuntimeError(f"The frozen engine identity changed: {field}")
    paths = [Path(__file__), ROOT / "tools/external/local_arena.py",
             ROOT / "tools/run_edge_acc_campaign.py", ROOT / "tools/run_cross_network_h2h.py"]
    return {"engines": engines, "protocol": CONFIG["protocol"],
            "sources": {str(path.relative_to(ROOT)): campaign.sha256(path) for path in paths},
            "cpu_limit": CONFIG["cpu_limit"], "h2h_workers": CONFIG["h2h_workers"],
            "claustrophobia_workers": CONFIG["claustrophobia_workers"], "plan": task_plan()}


def configure_campaign() -> None:
    campaign.OUT = Path(CONFIG["output"])
    campaign.CONFIG.update(output=str(campaign.OUT), h2h_workers=CONFIG["h2h_workers"],
                           gpu_workers=CONFIG["claustrophobia_workers"], clock_protocol=CONFIG["protocol"],
                           clock_protocol_validation=str(campaign.OUT / "clock_protocol_validation.json"))


def run_child(name: str, workers: int | None = None) -> None:
    apply_limits()
    configure_campaign()
    task = next(task for task in task_plan() if task["name"] == name)
    if workers is not None:
        if not 1 <= workers <= 6:
            raise RuntimeError("The selected game worker count must be between 1 and 6")
        campaign.CONFIG["h2h_workers"] = workers
        campaign.CONFIG["gpu_workers"] = workers
    if task["mode"] == "h2h":
        result = campaign.run_h2h_3plus2(task["network"], task["variant"], task["shard"], CONFIG["shards"])
    elif task["mode"] == "cross":
        from tools import run_cross_network_h2h as cross
        cross.OUT_ROOT = campaign.OUT
        cross.CONFIG["h2h_workers"] = workers or CONFIG["h2h_workers"]
        result = cross.run("3plus2", task["shard"], CONFIG["shards"])
    else:
        result = campaign.run_external_campaign(
            task["network"], task["variant"], task["opponent"], three_plus_two=True,
            shard_index=task["shard"], shard_count=CONFIG["shards"],
            participants=("off",) if task["variant"] == "off" else ("candidate",))
    print(json.dumps(result, indent=2), flush=True)


def report() -> None:
    if Path(CONFIG["report_script"]).is_file():
        result = subprocess.run([sys.executable, str(CONFIG["report_script"])], cwd=ROOT,
                                stdout=subprocess.DEVNULL, check=False)
        if result.returncode:
            print(f"The report process returned {result.returncode}. The game queue continues.", flush=True)


def live_validation() -> None:
    from tools import run_clock_protocol_validation as probe
    configure_campaign()
    adapters = campaign.native_clock_preflight()
    try:
        campaign.validate_clock_protocol_audit(adapters)
        return
    except RuntimeError:
        pass
    # The network probes read the original immutable weight manifest.
    campaign.OUT = Path(CONFIG["previous_output"])
    probe.CONFIG.update(output=Path(CONFIG["output"]) / "clock_protocol_validation.json", gpu_workers=2)
    probe.main()
    configure_campaign()
    campaign.validate_clock_protocol_audit(adapters)


def wait_for_gpu(status: dict, path: Path) -> None:
    required = CONFIG["gpu_reserve_mib"] + CONFIG["claustrophobia_workers"] * CONFIG["gpu_worker_mib"]
    while True:
        free = legacy_queue.query_gpu_free_mib()
        status["gpu"] = dict(free_mib=free, required_mib=required)
        atomic_json(path, status)
        if free >= required:
            return
        status["state"] = "waiting_for_resources"
        time.sleep(30)



def available_ram_mib() -> int:
    class MemoryStatus(ctypes.Structure):
        _fields_ = [("length", wintypes.DWORD), ("load", wintypes.DWORD),
                    ("total_physical", ctypes.c_ulonglong), ("available_physical", ctypes.c_ulonglong),
                    ("total_page", ctypes.c_ulonglong), ("available_page", ctypes.c_ulonglong),
                    ("total_virtual", ctypes.c_ulonglong), ("available_virtual", ctypes.c_ulonglong),
                    ("available_extended", ctypes.c_ulonglong)]
    value = MemoryStatus()
    value.length = ctypes.sizeof(value)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(value)):
        raise ctypes.WinError()
    return int(value.available_physical // (1024 * 1024))


def ram_per_game(task: dict) -> int:
    if task.get("opponent") == "titanium":
        return CONFIG["titanium_ram_per_game_mib"]
    if task["mode"] == "external":
        return CONFIG["external_ram_per_game_mib"]
    return CONFIG["h2h_ram_per_game_mib"]


def safe_workers(task: dict, free_mib: int) -> int:
    cap = CONFIG["claustrophobia_workers"] if task.get("opponent") == "claustrophobia" else CONFIG["h2h_workers"]
    if CONFIG["fixed_workers"]:
        return cap
    if CONFIG["modest_start"] and free_mib >= CONFIG["modest_start_free_mib"]:
        return min(cap, 1)
    return max(0, min(cap, (free_mib - CONFIG["ram_reserve_mib"]) // ram_per_game(task)))


def eligible_task(pending: list[dict], done: set[str], free_mib: int) -> dict | None:
    for task in sorted(pending, key=ram_per_game):
        if task["mode"] == "external" and task["variant"] != "off":
            baseline = f"{task['opponent']}_{task['network']}_off_{task['shard']}"
            if baseline not in done:
                continue
        if safe_workers(task, free_mib):
            return task
    return None


def wait_for_ram(task: dict, status: dict, path: Path) -> int:
    while True:
        free = available_ram_mib()
        workers = safe_workers(task, free)
        status["ram"] = dict(free_mib=free, reserve_mib=CONFIG["ram_reserve_mib"], required_mib=CONFIG["ram_reserve_mib"] + ram_per_game(task), selected_workers=workers, modest_start=CONFIG["modest_start"])
        if CONFIG["fixed_workers"]:
            status["ram"]["required_mib"] = 0
            status["ram"]["ram_guard_enabled"] = False
            status["ram"]["fixed_workers"] = True
        if CONFIG["modest_start"]:
            status["ram"]["required_mib"] = CONFIG["modest_start_free_mib"]
        if not workers:
            status.update(state="waiting_for_resources", current=task)
        atomic_json(path, status)
        if workers:
            return workers
        time.sleep(30)


def stop_owned_job(pid: int) -> None:
    # Capture descendants before stopping the runner so unfinished games cannot be recorded as ordinary forfeits.
    script = f"""$all = @(Get-CimInstance Win32_Process)
$owned = [System.Collections.Generic.HashSet[int]]::new()
[void]$owned.Add({pid})
do {{
    $changed = $false
    foreach ($process in $all) {{
        if ($owned.Contains([int]$process.ParentProcessId) -and $owned.Add([int]$process.ProcessId)) {{ $changed = $true }}
    }}
}} while ($changed)
Stop-Process -Id {pid} -Force -ErrorAction SilentlyContinue
foreach ($processId in $owned) {{ if ($processId -ne {pid}) {{ Stop-Process -Id $processId -Force -ErrorAction SilentlyContinue }} }}
"""
    subprocess.run(["powershell", "-NoProfile", "-Command", script], check=True,
                   creationflags=subprocess.CREATE_NO_WINDOW)

def execute() -> None:
    apply_limits()
    directory = Path(CONFIG["output"]) / "queue"
    directory.mkdir(parents=True, exist_ok=True)
    import msvcrt
    with (directory / "queue.lock").open("a+b") as lock:
        if lock.tell() == 0:
            lock.write(b"0")
            lock.flush()
        lock.seek(0)
        try:
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            raise RuntimeError("Another matrix supervisor owns the queue lock")
        identity = source_identity()
        identity_path = directory / "identity.json"
        if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
            raise RuntimeError("The matrix identity changed. Preserve the existing campaign")
        atomic_json(identity_path, identity)
        status_path = directory / "status.json"
        previous = json.loads(status_path.read_text()) if status_path.exists() else {}
        completed = previous.get("completed_steps", [])
        status = dict(pid=os.getpid(), state="preflight", current=None, completed_steps=completed,
                      total_steps=len(task_plan()), resource_limits=dict(cpu=CONFIG["cpu_limit"], h2h=CONFIG["h2h_workers"], claustrophobia=CONFIG["claustrophobia_workers"], ram_reserve_mib=CONFIG["ram_reserve_mib"], ram_guard_enabled=not CONFIG["fixed_workers"]),
                      source_identity=identity["engines"])
        current = previous.get("current") or {}
        if current.get("child_pid") and current.get("child_token") and legacy_queue.process_token(current["child_pid"]) == current["child_token"]:
            status.update(state="waiting_existing_job", current=current)
            while legacy_queue.process_token(current["child_pid"]) == current.get("child_token"):
                free = available_ram_mib()
                status["ram"] = dict(free_mib=free, reserve_mib=CONFIG["ram_reserve_mib"], selected_workers=current.get("workers", 1))
                status["updated_local"] = time.strftime("%Y-%m-%d %H:%M:%S")
                if not CONFIG["fixed_workers"] and free < CONFIG["ram_reserve_mib"]:
                    stop_owned_job(current["child_pid"])
                    key = ("titanium_ram_per_game_mib" if current.get("opponent") == "titanium"
                           else "external_ram_per_game_mib" if current["mode"] == "external" else "h2h_ram_per_game_mib")
                    learned = previous.setdefault("learned_ram_mib", {})
                    learned[key] = max(CONFIG[key], learned.get(key, 0)) + 2048
                    status["learned_ram_mib"] = learned
                    status.setdefault("resource_events", []).append(dict(task=current["name"], free_mib=free,
                        next_estimate_mib=learned[key], time_local=time.strftime("%Y-%m-%d %H:%M:%S")))
                    atomic_json(status_path, status)
                    break
                atomic_json(status_path, status)
                time.sleep(CONFIG["poll_seconds"])
        atomic_json(status_path, status)
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
        try:
            legacy_queue.verify_main_reference()
            for network in CONFIG["networks"]:
                for variant in CONFIG["variants"]:
                    campaign.validate_parity(network, variant)
            wait_for_gpu(status, status_path)
            live_validation()
            done = {row["name"] for row in completed}
            last_report = 0.0
            pending = [task for task in task_plan() if task["name"] not in done]
            learned = previous.get("learned_ram_mib", {})
            for key, value in learned.items():
                CONFIG[key] = max(CONFIG[key], value)
            status["learned_ram_mib"] = learned
            while pending:
                task = eligible_task(pending, done, available_ram_mib())
                if task is None:
                    status.update(state="waiting_for_resources", current=None)
                    status["ram"] = dict(free_mib=available_ram_mib(), reserve_mib=CONFIG["ram_reserve_mib"],
                                         required_mib=min(ram_per_game(t) for t in pending) + CONFIG["ram_reserve_mib"], selected_workers=0)
                    atomic_json(status_path, status)
                    if time.monotonic() - last_report >= CONFIG["report_interval_seconds"]:
                        report()
                        last_report = time.monotonic()
                    time.sleep(30)
                    continue
                resource_aborted = False
                if source_identity() != identity:
                    raise RuntimeError("An engine, weight, or protocol changed during the matrix")
                if task.get("opponent") == "claustrophobia":
                    wait_for_gpu(status, status_path)
                for attempt in range(CONFIG["infrastructure_retries"] + 1):
                    workers = wait_for_ram(task, status, status_path)
                    log = directory / (task["name"] + ".log")
                    with log.open("a", encoding="utf-8", buffering=1) as stream:
                        CONFIG["modest_start"] = False
                        command = [sys.executable, "-u", str(Path(__file__)), "--child", task["name"], "--workers", str(workers)]
                        child = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                                 creationflags=subprocess.BELOW_NORMAL_PRIORITY_CLASS)
                        status.update(state="running", current={**task, "child_pid": child.pid,
                                      "child_token": legacy_queue.process_token(child.pid), "log": str(log),
                                      "attempt": attempt + 1, "workers": workers})
                        while child.poll() is None:
                            status["ram"]["free_mib"] = available_ram_mib()
                            if not CONFIG["fixed_workers"] and status["ram"]["free_mib"] < CONFIG["ram_reserve_mib"]:
                                stop_owned_job(child.pid)
                                key = ("titanium_ram_per_game_mib" if task.get("opponent") == "titanium"
                                       else "external_ram_per_game_mib" if task["mode"] == "external" else "h2h_ram_per_game_mib")
                                CONFIG[key] += 2048
                                learned[key] = CONFIG[key]
                                status.setdefault("resource_events", []).append(dict(task=task["name"],
                                    free_mib=status["ram"]["free_mib"], next_estimate_mib=CONFIG[key],
                                    time_local=time.strftime("%Y-%m-%d %H:%M:%S")))
                                status.update(state="resource_deferred", error="The RAM reserve was breached. The owned job was stopped and its resource class requires more memory before resuming")
                                atomic_json(status_path, status)
                                resource_aborted = True
                                report()
                                break
                            status["updated_local"] = time.strftime("%Y-%m-%d %H:%M:%S")
                            atomic_json(status_path, status)
                            if time.monotonic() - last_report >= CONFIG["report_interval_seconds"]:
                                report()
                                last_report = time.monotonic()
                            time.sleep(CONFIG["poll_seconds"])
                    if resource_aborted:
                        break
                    if child.returncode == 0:
                        break
                    if attempt == CONFIG["infrastructure_retries"]:
                        raise RuntimeError(f"The matrix task failed: {task['name']}. Examine {log}")
                if resource_aborted:
                    continue
                pending.remove(task)
                done.add(task["name"])
                status.pop("error", None)
                completed.append(dict(name=task["name"], finished_local=time.strftime("%Y-%m-%d %H:%M:%S")))
                atomic_json(status_path, status)
                report()
            status.update(state="complete", current=None)
            atomic_json(status_path, status)
            report()
        except BaseException as error:
            status.update(state="failed", error=f"{type(error).__name__}: {error}")
            atomic_json(status_path, status)
            report()
            raise
        finally:
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--child")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--modest-start", action="store_true", help="Admit the first job with 4096 MiB free RAM; retain the runtime reserve and conservative admission after that job")
    args = parser.parse_args()
    CONFIG["modest_start"] = args.modest_start
    if args.child:
        run_child(args.child, args.workers)
    elif args.dry_run:
        print(json.dumps(source_identity(), indent=2))
    else:
        execute()


if __name__ == "__main__":
    main()
