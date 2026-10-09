#!/usr/bin/env python3
"""Run parity-gated EdgeAcc campaigns in a persistent sequential queue."""
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

CONFIG = {
    "output": campaign.OUT / "queue",
    "h2h_order": [
        ("504", "delta_dense_only"), ("858", "v3_node_dense_bfs"),
        ("504", "v3_node_dense_bfs"), ("858", "delta_dense_only"),
        ("504", "v3"), ("858", "v3"),
    ],
    "networks": ["504", "858"],
    "three_plus_two_finalists_per_network": 3,
    "three_plus_two_shards": 4,
    "poll_seconds": 5,
    "resource_policy": {
        "campaign_h2h_workers": 12,
        "campaign_claustrophobia_workers": 6,
        "claustrophobia_gpu_reserve_mib": 2048,
        "claustrophobia_mib_per_worker": 256,
        "gpu_wait_poll_seconds": 30,
        "child_priority": "below_normal",
    },
    "wait_pid": None,
    "main_reference_manifest": ROOT / "results/benchmarks/edge-parity-local/main_reference_parity.json",
    "network_weight_manifest": campaign.OUT / "network_weights" / "manifest.json",
    "network_weight_names": ["pre300", "v300", "v301"],
    "network_weight_sha256": {
        "pre300": "f05e8a987d2d646ef618058920da990bf3d63a00f2d78805429713dc0a1f88e8",
        "v300": "4fd62cfe6c6045d1c6abe822eb5936349a145f61eaa9f9f87065f16abde3e89d",
        "v301": "f29b4bdde846191a747166885b1523ece5198f067beaa86688e5462e18232792",
    },
    "network_h2h_pairs": [("v300", "pre300"), ("v301", "pre300"), ("v301", "v300")],
}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def process_token(pid: int) -> int | None:
    if os.name != "nt":
        return None
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.GetProcessTimes.argtypes = [wintypes.HANDLE, *([ctypes.POINTER(wintypes.FILETIME)] * 4)]
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = kernel.OpenProcess(0x1000, False, pid)
    if not handle:
        return None
    try:
        values = [wintypes.FILETIME() for _ in range(4)]
        if not kernel.GetProcessTimes(handle, *(ctypes.byref(value) for value in values)):
            return None
        return (values[0].dwHighDateTime << 32) | values[0].dwLowDateTime
    finally:
        kernel.CloseHandle(handle)


def waiting_current(previous: dict | None, pid: int,
                    token: int | None = None) -> dict:
    """Keep durable child identity while annotating an existing-process wait."""
    current = dict(previous or {})
    current.update(pid=pid, child_pid=pid,
                   child_token=token if token is not None else process_token(pid))
    return current


def snapshot() -> dict:
    clock_audit = Path(campaign.CONFIG["clock_protocol_validation"])
    network_weight_manifest_path = Path(CONFIG["network_weight_manifest"])
    network_weight_manifest = read_json(network_weight_manifest_path)
    if network_weight_manifest.get("schema") != "zquoridor.immutable-network-revalidation.v1":
        raise RuntimeError("immutable network weight manifest schema is invalid")
    network_weights = {}
    for name in CONFIG["network_weight_names"]:
        entry = network_weight_manifest.get("weights", {}).get(name, {})
        weight_path = Path(entry.get("path", ""))
        if (entry.get("sha256") != CONFIG["network_weight_sha256"][name]
                or not weight_path.is_file() or campaign.sha256(weight_path) != entry.get("sha256")
                or weight_path.stat().st_size != entry.get("size")):
            raise RuntimeError(f"immutable network weight is missing or stale: {name}")
        network_weights[name] = {"path": str(weight_path.resolve()), "sha256": entry["sha256"],
                                 "size": entry["size"], "revision": entry.get("revision")}
    network_script = ROOT / "tools/run_network_revalidation.py"
    if not network_script.is_file():
        raise RuntimeError(f"network revalidation runner is missing: {network_script}")
    return {
        "experimental_source": campaign.source_hash(ROOT),
        "main_source": campaign.source_hash(campaign.MAIN),
        "weights": {net: campaign.sha256(Path(campaign.CONFIG["networks"][net]["weights"]))
                    for net in CONFIG["networks"]},
        "clock_protocol_audit_sha256": campaign.sha256(clock_audit) if clock_audit.is_file() else None,
        "network_weight_manifest_sha256": campaign.sha256(network_weight_manifest_path),
        "network_weights": network_weights,
        "network_revalidation_runner_sha256": campaign.sha256(network_script),
        "resource_policy": CONFIG["resource_policy"],
        "protocol": {str(path.relative_to(ROOT)).replace("\\", "/"): campaign.sha256(path)
                     for path in [ROOT / "tools/run_edge_acc_campaign.py", ROOT / "tools/run_edge_acc_report.py",
                                  ROOT / "tools/run_edge_acc_queue.py", ROOT / "tools/external/local_arena.py",
                                  ROOT / "tools/build_claustrophobia_dynamic_bridge.py",
                                  ROOT / "tools/external/claustrophobia_benchmark_bridge.rs",
                                  ROOT / "tools/build_titanium_clock_bridge.py"]},
    }


def verify_main_reference() -> None:
    manifest = read_json(Path(CONFIG["main_reference_manifest"]))
    if manifest.get("status") != "PASS":
        raise RuntimeError("Main OFF reference parity did not pass")
    for row in manifest.get("header_hashes", []):
        if campaign.sha256(campaign.MAIN / "src" / row["file"]) != row["main_original_sha256"]:
            raise RuntimeError("Main headers changed after reference parity")
    if len(manifest.get("header_hashes", [])) != len(list((campaign.MAIN / "src").glob("*.hpp"))):
        raise RuntimeError("Main reference header inventory changed")


def h2h_screen(summary: dict) -> tuple[bool, str]:
    for category, pairs in campaign.CONFIG["h2h_pairs"].items():
        row = summary.get(category, {})
        if row.get("complete_pairs") != pairs or row.get("failed_games") != 0:
            raise RuntimeError("H2H screening is incomplete")
        upper = row.get("paired_bootstrap_95", {}).get("score_high_pct")
        if upper is None:
            raise RuntimeError("H2H screening lacks a paired confidence interval")
        if upper < 50.0:
            return False, f"{category} H2H loss; paired 95% upper bound below 50%"
    center_score = summary["center"]["score_pct"]
    if center_score < 50.0 or summary.get("score_pct", 0) < 50.0:
        return False, "H2H Center or overall point score below 50%"
    return True, "H2H screening passed"


def external_screen(summary: dict) -> tuple[bool, str]:
    deltas = summary.get("paired_delta_candidate_minus_off", {})
    for category, pairs in campaign.CONFIG["external_pairs"].items():
        row = deltas.get(category, {})
        interval = row.get("paired_bootstrap_95")
        if row.get("complete_paired_openings") != pairs or not isinstance(interval, list):
            raise RuntimeError("Claustrophobia screening lacks complete matched OFF comparisons")
        if interval[1] < 0:
            return False, f"{category} Claustrophobia loss; paired delta upper bound below zero"
    return True, "No clear paired Claustrophobia regression"


def campaign_task_plan() -> list[dict]:
    """Describe every requested restart job in deterministic execution order."""
    variants_by_network = {
        network: list(dict.fromkeys(variant for net, variant in CONFIG["h2h_order"] if net == network))
        for network in CONFIG["networks"]
    }
    tasks = []
    # The first screening pass isolates network-version regressions against
    # Claustrophobia before spending time on Edge variants or other opponents.
    for weight in CONFIG["network_weight_names"]:
        tasks.append({"stage": "network_external_200",
                      "name": f"network_200ms_{weight}_claustrophobia",
                      "script": "run_network_revalidation.py",
                      "args": ["external", "--weight", weight,
                               "--opponent", "claustrophobia",
                               "--time-control", "200ms"]})
    tasks.append({"stage": "network_claustrophobia_200_report",
                  "name": "network_external_200_claustrophobia_paired_report",
                  "script": "run_network_revalidation.py",
                  "args": ["report", "--mode", "external",
                           "--opponent", "claustrophobia",
                           "--time-control", "200ms"]})
    for left, right in CONFIG["network_h2h_pairs"]:
        tasks.append({"stage": "network_h2h_200", "name": f"network_h2h200_{left}_vs_{right}",
                      "script": "run_network_revalidation.py",
                      "args": ["h2h", "--left", left, "--right", right,
                               "--time-control", "200ms"]})
    for network, variant in CONFIG["h2h_order"]:
        tasks.append({"stage": "edge_h2h_200", "name": f"h2h200_{network}_{variant}",
                      "script": "run_edge_acc_campaign.py",
                      "args": ["h2h", "--network", network, "--variant", variant]})
    for opponent in ("claustrophobia", "titanium"):
        for network in CONFIG["networks"]:
            tasks.append({"stage": "edge_external_200", "name": f"200ms_{network}_off_{opponent}_off",
                          "script": "run_edge_acc_campaign.py",
                          "args": ["external", "--network", network, "--variant", "off",
                                   "--opponent", opponent, "--participants", "off"]})
            for variant in variants_by_network[network]:
                tasks.append({"stage": "edge_external_200", "name": f"200ms_{network}_{variant}_{opponent}_candidate",
                              "script": "run_edge_acc_campaign.py",
                              "args": ["external", "--network", network, "--variant", variant,
                                       "--opponent", opponent, "--participants", "candidate"]})
    for weight in CONFIG["network_weight_names"]:
        for opponent in ("claustrophobia", "titanium"):
            if opponent == "claustrophobia":
                continue  # already scheduled as the first three tasks above
            tasks.append({"stage": "network_external_200", "name": f"network_200ms_{weight}_{opponent}",
                          "script": "run_network_revalidation.py",
                          "args": ["external", "--weight", weight, "--opponent", opponent,
                                   "--time-control", "200ms"]})
    for left, right in CONFIG["network_h2h_pairs"]:
        for shard in range(CONFIG["three_plus_two_shards"]):
            tasks.append({"stage": "network_h2h_3plus2", "name": f"network_h2h3plus2_{left}_vs_{right}_shard_{shard}",
                          "script": "run_network_revalidation.py",
                          "args": ["h2h", "--left", left, "--right", right,
                                   "--time-control", "3plus2", "--shard-index", str(shard),
                                   "--shard-count", str(CONFIG["three_plus_two_shards"])]})
    for opponent in ("claustrophobia", "titanium"):
        for network in CONFIG["networks"]:
            for participant, variants in (("off", ["off"]), ("candidate", variants_by_network[network])):
                for variant in variants:
                    for shard in range(CONFIG["three_plus_two_shards"]):
                        tasks.append({"stage": "edge_external_3plus2",
                            "name": f"3plus2_{network}_{variant}_{opponent}_{participant}_shard_{shard}",
                            "script": "run_edge_acc_campaign.py",
                            "args": ["external-3plus2", "--network", network, "--variant", variant,
                                     "--opponent", opponent, "--participants", participant,
                                     "--shard-index", str(shard), "--shard-count",
                                     str(CONFIG["three_plus_two_shards"])]})
    for weight in CONFIG["network_weight_names"]:
        for opponent in ("claustrophobia", "titanium"):
            for shard in range(CONFIG["three_plus_two_shards"]):
                tasks.append({"stage": "network_external_3plus2",
                    "name": f"network_3plus2_{weight}_{opponent}_shard_{shard}",
                    "script": "run_network_revalidation.py",
                    "args": ["external", "--weight", weight, "--opponent", opponent,
                             "--time-control", "3plus2", "--shard-index", str(shard),
                             "--shard-count", str(CONFIG["three_plus_two_shards"])]})
    for network, variant in CONFIG["h2h_order"]:
        for shard in range(CONFIG["three_plus_two_shards"]):
            tasks.append({"stage": "edge_h2h_3plus2", "name": f"h2h3plus2_{network}_{variant}_shard_{shard}",
                          "script": "run_edge_acc_campaign.py",
                          "args": ["h2h-3plus2", "--network", network, "--variant", variant,
                                   "--shard-index", str(shard), "--shard-count",
                                   str(CONFIG["three_plus_two_shards"])]})
    return tasks


def campaign_plan_summary(plan: list[dict] | None = None) -> dict:
    plan = campaign_task_plan() if plan is None else plan
    jobs_by_stage = {}
    games_per_job = {
        "network_h2h_200": 400, "edge_h2h_200": 400,
        "edge_external_200": 300, "network_external_200": 300,
        "network_claustrophobia_200_report": 0,
        "network_h2h_3plus2": 100, "edge_external_3plus2": 100,
        "network_external_3plus2": 100, "edge_h2h_3plus2": 100,
    }
    games_by_stage = {}
    for task in plan:
        stage = task["stage"]
        jobs_by_stage[stage] = jobs_by_stage.get(stage, 0) + 1
        games_by_stage[stage] = games_by_stage.get(stage, 0) + games_per_job[stage]
    max_games = sum(games_by_stage.values())
    # The v3.01 / 504 OFF files are common with network revalidation only if
    # the downstream runner confirms exact executable, weight, opening,
    # opponent, referee, and native-clock provenance.
    potential_v301_reuse = jobs_by_stage.get("network_external_200", 0) // 3 * 300
    potential_v301_reuse += jobs_by_stage.get("network_external_3plus2", 0) // 3 * 100
    return {"task_count": len(plan), "jobs_by_stage": jobs_by_stage,
            "games_by_stage_max": games_by_stage, "games_total_max": max_games,
            "games_total_if_v301_off_reuse_is_exact": max_games - potential_v301_reuse,
            "potential_v301_off_games_reused": potential_v301_reuse}


def campaign_wrapper_command(arguments: list[str]) -> list[str]:
    """Invoke campaign with queue-only concurrency overrides, preserving its CLI."""
    code = (
        "import sys; from tools import run_edge_acc_campaign as c; "
        f"c.CONFIG.update(h2h_workers={CONFIG['resource_policy']['campaign_h2h_workers']}, "
        f"gpu_workers={CONFIG['resource_policy']['campaign_claustrophobia_workers']}); "
        "sys.argv=['run_edge_acc_campaign.py', *sys.argv[1:]]; "
        "raise SystemExit(c.main())"
    )
    return [sys.executable, "-u", "-c", code, *arguments]


def claustrophobia_campaign(arguments: list[str]) -> bool:
    """Whether this campaign task will allocate the six-worker GPU pool."""
    if not arguments or arguments[0] not in ("external", "external-3plus2"):
        return False
    try:
        opponent_index = arguments.index("--opponent") + 1
    except (ValueError, IndexError):
        return False
    return opponent_index < len(arguments) and arguments[opponent_index] == "claustrophobia"


def parse_gpu_free_mib(output: str) -> int:
    """Parse the first device's free-memory field from nvidia-smi CSV output."""
    for line in output.splitlines():
        value = line.strip().split()[0] if line.strip() else ""
        if value.isdecimal():
            return int(value)
    raise RuntimeError(f"Could not parse nvidia-smi free memory output: {output!r}")


def required_cla_gpu_mib() -> int:
    policy = CONFIG["resource_policy"]
    return (policy["claustrophobia_gpu_reserve_mib"] +
            policy["campaign_claustrophobia_workers"] *
            policy["claustrophobia_mib_per_worker"])


def query_gpu_free_mib() -> int:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--id=0", "--query-gpu=memory.free",
             "--format=csv,noheader,nounits"],
            check=True, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(f"Cannot verify GPU 0 free memory before Claustrophobia job: {exc}") from exc
    return parse_gpu_free_mib(result.stdout)


class Queue:
    def __init__(self, output: Path):
        self.output = output
        self.output.mkdir(parents=True, exist_ok=True)
        self.status_path = self.output / "status.json"
        self.identity_path = self.output / "identity.json"
        self.frozen = snapshot()
        identity = {"sources": self.frozen, "h2h_order": CONFIG["h2h_order"],
                    "networks": CONFIG["networks"], "finalists_per_network": CONFIG["three_plus_two_finalists_per_network"],
                    "shards": CONFIG["three_plus_two_shards"],
                    "protocol": campaign.CONFIG["clock_protocol"],
                    "policy": "network_and_edge_h2h200_external200_then_full_3plus2_native_clock",
                    "network_h2h_pairs": CONFIG["network_h2h_pairs"],
                    "network_weight_names": CONFIG["network_weight_names"],
                    "task_names": [task["name"] for task in campaign_task_plan()]}
        encoded = json.dumps(identity, sort_keys=True, indent=2)
        if self.identity_path.exists() and self.identity_path.read_text(encoding="utf-8") != encoded:
            raise RuntimeError("Queue identity differs from the existing campaign")
        self.identity_path.write_text(encoded, encoding="utf-8")
        self.status = {"pid": os.getpid(), "state": "starting", "current": None,
                       "completed_steps": [], "screening": {}, "finalists": {},
                       "sources": self.frozen}
        self.orphan = None
        if self.status_path.exists():
            previous = read_json(self.status_path)
            self.status["completed_steps"] = previous.get("completed_steps", [])
            current = previous.get("current") or {}
            if current.get("child_pid") and current.get("child_token"):
                self.orphan = current
        self.completed = {row["name"] for row in self.status["completed_steps"]}

    def save(self) -> None:
        self.status["updated_local"] = time.strftime("%Y-%m-%d %H:%M:%S")
        temporary = self.status_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.status, indent=2), encoding="utf-8")
        temporary.replace(self.status_path)

    def verify(self) -> None:
        if snapshot() != self.frozen:
            raise RuntimeError("Source or weight hashes changed during the queue")

    def wait_for_cla_gpu(self, name: str) -> None:
        policy = CONFIG["resource_policy"]
        required = required_cla_gpu_mib()
        while True:
            free = query_gpu_free_mib()  # fail closed if the device cannot be queried
            if free >= required:
                self.status["current"].pop("resource_wait", None)
                self.save()
                print(f"GPU memory gate passed for {name}: {free} MiB free, {required} MiB required.",
                      flush=True)
                return
            message = {"gpu": 0, "free_mib": free, "required_mib": required,
                       "next_check_seconds": policy["gpu_wait_poll_seconds"]}
            self.status["current"]["resource_wait"] = message
            self.save()
            print(f"Waiting to start {name}: GPU 0 has {free} MiB free; "
                  f"need {required} MiB. Rechecking in {policy['gpu_wait_poll_seconds']}s.",
                  flush=True)
            time.sleep(policy["gpu_wait_poll_seconds"])

    def run(self, name: str, arguments: list[str], script: str = "run_edge_acc_campaign.py") -> None:
        self.verify()
        if name in self.completed:
            return
        if (script in ("run_edge_acc_campaign.py", "run_network_revalidation.py")
                and claustrophobia_campaign(arguments)):
            self.status.update(state="waiting_for_resources",
                               current={"name": name, "script": script, "arguments": arguments})
            self.save()
            self.wait_for_cla_gpu(name)
            self.status.update(state="running", current=None)
        if script == "run_edge_acc_campaign.py":
            command = campaign_wrapper_command(arguments)
        else:
            command = [sys.executable, "-u", str(ROOT / "tools" / script), *arguments]
        log = self.output / f"{name}.log"
        self.status.update(state="running", current={"name": name, "command": command, "log": str(log)})
        self.save()
        print(f"START {name}", flush=True)
        with log.open("a", encoding="utf-8", buffering=1) as stream:
            stream.write(f"\nCOMMAND {json.dumps(command)}\n")
            below_normal = getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0x00004000)
            child = subprocess.Popen(command, cwd=ROOT, stdout=stream,
                                     stderr=subprocess.STDOUT, creationflags=below_normal)
            self.status["current"]["child_pid"] = child.pid
            self.status["current"]["child_token"] = process_token(child.pid)
            self.save()
            while child.poll() is None:
                time.sleep(CONFIG["poll_seconds"])
                self.save()
            if child.returncode:
                raise RuntimeError(f"Step {name} failed with exit code {child.returncode}; inspect {log}")
        self.status["completed_steps"].append({"name": name, "log": str(log), "finished_local": time.strftime("%Y-%m-%d %H:%M:%S")})
        self.completed.add(name)
        self.save()
        print(f"DONE {name}", flush=True)

    def report(self, name: str) -> None:
        self.run(f"report_{name}", [], "run_edge_acc_report.py")

    def record_200ms_screening(self) -> None:
        for network, variant in CONFIG["h2h_order"]:
            summary = read_json(campaign.OUT / f"h2h_{network}_{variant}" / "summary.json")
            passed, reason = h2h_screen(summary)
            key = f"{network}/{variant}"
            self.status["screening"][key] = {"h2h_pass": passed, "h2h_reason": reason}
            claustro_summary = read_json(
                campaign.OUT / "200ms" / network / variant / "claustrophobia" / "summary.json")
            claustro_passed, claustro_reason = external_screen(claustro_summary)
            self.status["screening"][key].update(
                claustro_pass=claustro_passed, claustro_reason=claustro_reason)
        for network in CONFIG["networks"]:
            self.status["finalists"][network] = [
                variant for net, variant in CONFIG["h2h_order"] if net == network]
        self.save()

    def execute(self, wait_pid: int | None) -> None:
        verify_main_reference()
        for network, variant in CONFIG["h2h_order"]:
            campaign.validate_parity(network, variant)
        if wait_pid:
            self.wait_pid(wait_pid)
        if self.orphan:
            self.wait_pid(self.orphan["child_pid"], self.orphan["child_token"])
        clock_adapters = campaign.native_clock_preflight()
        campaign.validate_clock_protocol_audit(clock_adapters)
        print(json.dumps({"native_clock_preflight": clock_adapters}, indent=2), flush=True)
        plan = campaign_task_plan()
        screening_recorded = False
        for task in plan:
            if task["stage"] == "network_h2h_3plus2" and not screening_recorded:
                self.record_200ms_screening()
                screening_recorded = True
            self.run(task["name"], task["args"], task["script"])
        if not screening_recorded:
            self.record_200ms_screening()
        self.report("final")
        self.status.update(state="complete", current=None)
        self.save()
        print("QUEUE COMPLETE", flush=True)

    def wait_pid(self, pid: int, expected_token: int | None = None) -> None:
        if os.name != "nt":
            raise RuntimeError("The existing-process wait currently supports Windows only")
        if expected_token is not None and process_token(pid) != expected_token:
            return
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = kernel.OpenProcess(0x1000, False, pid)
        if not handle:
            return
        try:
            self.status.update(
                state="waiting_existing_campaign",
                current=waiting_current(self.status.get("current"), pid, expected_token))
            self.save()
            code = wintypes.DWORD()
            while kernel.GetExitCodeProcess(handle, ctypes.byref(code)) and code.value == 259:
                time.sleep(CONFIG["poll_seconds"])
                self.save()
        finally:
            kernel.CloseHandle(handle)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait-pid", type=int, default=CONFIG["wait_pid"])
    parser.add_argument("--output", type=Path, default=CONFIG["output"])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        verify_main_reference()
        for network, variant in CONFIG["h2h_order"]:
            campaign.validate_parity(network, variant)
        clock_adapters = campaign.native_clock_preflight()
        campaign.validate_clock_protocol_audit(clock_adapters)
        plan = campaign_task_plan()
        print(json.dumps({"h2h": CONFIG["h2h_order"], "baseline_networks": CONFIG["networks"],
                          "variants_per_network": CONFIG["three_plus_two_finalists_per_network"],
                          "shards": CONFIG["three_plus_two_shards"],
                          "native_clock_preflight": clock_adapters,
                          "campaign_plan": campaign_plan_summary(plan),
                          "sources": snapshot()}, indent=2))
        return 0
    clock_audit_path = Path(campaign.CONFIG["clock_protocol_validation"])
    if not clock_audit_path.is_file():
        raise SystemExit(f"native-clock live validation is missing: {clock_audit_path}")
    clock_audit = read_json(clock_audit_path)
    if clock_audit.get("status") != "PASS" or clock_audit.get("protocol") != "native-clock-v2":
        raise SystemExit("native-clock live validation must pass before queue initialization")
    import msvcrt
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "queue.lock").open("a+b") as lock:
        if lock.tell() == 0:
            lock.write(b"0")
            lock.flush()
        lock.seek(0)
        try:
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            raise SystemExit("Another queue process owns the campaign lock")
        queue = Queue(args.output)
        try:
            # Keep the system awake for the durable job; the display may sleep.
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
            queue.execute(args.wait_pid)
        except BaseException as error:
            queue.status.update(state="failed", error=f"{type(error).__name__}: {error}")
            queue.save()
            raise
        finally:
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
