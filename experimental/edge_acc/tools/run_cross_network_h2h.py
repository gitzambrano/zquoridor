#!/usr/bin/env python3
"""Paired 200 ms H2H between the parity-approved 858 hybrid and 504 OFF."""
from __future__ import annotations

import argparse
import concurrent.futures
import ctypes
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
from ctypes import wintypes

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools import run_edge_acc_campaign as campaign
from tools.external import local_arena

OUT_ROOT = campaign.OUT
CONFIG = {
    "h2h_workers": 12,
    "shard_count": 4,
}
CANDIDATE_NETWORK = "858"
CANDIDATE_VARIANT = "v3_node_dense_bfs"
BASELINE_NETWORK = "504"
BASELINE_VARIANT = "off"


def digest(path: Path) -> str:
    return campaign.sha256(path)


def _shard_rows(rows: list[dict], shard_index: int, shard_count: int) -> list[dict]:
    begin = len(rows) * shard_index // shard_count
    end = len(rows) * (shard_index + 1) // shard_count
    return rows[begin:end]


def run(time_control: str, shard_index: int | None = None,
        shard_count: int = CONFIG["shard_count"]) -> dict:
    if shard_count < 1:
        raise ValueError("shard_count must be positive")
    if shard_index is not None and not 0 <= shard_index < shard_count:
        raise ValueError(f"shard_index must be in [0, {shard_count})")
    if time_control == "200ms" and shard_index is not None:
        raise ValueError("sharding is supported only for 3plus2")

    campaign.validate_parity(CANDIDATE_NETWORK, CANDIDATE_VARIANT)

    candidate_exe = campaign.build(CANDIDATE_NETWORK, CANDIDATE_VARIANT, ROOT, "exp")
    baseline_exe = campaign.build(BASELINE_NETWORK, BASELINE_VARIANT, campaign.MAIN, "main")
    candidate_weights = Path(campaign.CONFIG["networks"][CANDIDATE_NETWORK]["weights"]).resolve()
    baseline_weights = Path(campaign.CONFIG["networks"][BASELINE_NETWORK]["weights"]).resolve()
    if time_control == "200ms":
        books = campaign.prepare_books()["books"]
        initial_ms, increment_ms, move_ms = 0, 0, 200
        out = OUT_ROOT / "cross_network_h2h_200ms" / "hybrid858_vs_main504_off"
        tc_name = "200ms-movetime"
        clock_policy = "both engines receive go movetime 200 on every move"
    elif time_control == "3plus2":
        clock_adapters = campaign.native_clock_preflight()
        campaign.validate_clock_protocol_audit(clock_adapters)
        books = {label: [{"source_index": index, "moves": moves}
                         for index, moves in campaign.run_benchmark._read_openings(
                             Path(campaign.CONFIG[key]), 100, campaign.CONFIG["seed"])]
                 for label, key in (("center", "center_book"), ("normal", "normal_book"))}
        if shard_index is not None:
            for label in ("center", "normal"):
                books[label] = _shard_rows(books[label], shard_index, shard_count)
        initial_ms, increment_ms, move_ms = 180_000, 2_000, None
        out = OUT_ROOT / "cross_network_h2h_3plus2" / "hybrid858_vs_main504_off"
        if shard_index is not None:
            out /= f"shard_{shard_index:03d}_of_{shard_count:03d}"
        tc_name = "3plus2-native-clock"
        clock_policy = "native 180 s game clock plus 2 s increment for both engines"
    else:
        raise ValueError(f"unsupported time control: {time_control}")
    expected_pairs = {category: len(rows) for category, rows in books.items()}

    artifacts = {
        "candidate_executable": candidate_exe,
        "candidate_weights": candidate_weights,
        "baseline_executable": baseline_exe,
        "baseline_weights": baseline_weights,
        "center_book": Path(campaign.CONFIG["center_book"]),
        "normal_book": Path(campaign.CONFIG["normal_book"]),
        "referee": ROOT / "tools" / "external" / "local_arena.py",
        "parity_manifest": Path(campaign.CONFIG["parity"]["manifest"]),
    }
    manifest_config = {
        "protocol": f"cross-network-{time_control}-v1",
        "time_control": tc_name,
        "clock_policy": clock_policy,
        "candidate": {"network": CANDIDATE_NETWORK, "variant": CANDIDATE_VARIANT,
                      "nnue_sha256": digest(candidate_weights)},
        "baseline": {"network": BASELINE_NETWORK, "variant": BASELINE_VARIANT,
                     "nnue_sha256": digest(baseline_weights)},
        "seed": campaign.CONFIG["seed"],
        "move_time_ms": move_ms,
        "initial_ms": initial_ms or None,
        "increment_ms": increment_ms,
        "selected_openings": books,
        "parity_source_sha256": hashlib.sha256(
            json.dumps(campaign.source_hash_map(), sort_keys=True).encode()).hexdigest(),
    }
    if shard_index is not None:
        manifest_config["shard"] = [shard_index, shard_count]
    manifest = local_arena.make_manifest(manifest_config, artifacts)
    games_path, prior = local_arena.prepare_resume(out, manifest)
    latest = {(int(row["opening_index"]), int(row["zq_player"])): row for row in prior}
    jobs = []
    for category, opening_rows in (("center", books["center"]), ("normal", books["normal"])):
        offset = 0 if category == "center" else 100000
        for item in opening_rows:
            index, moves = offset + int(item["source_index"]), item["moves"]
            for side in (0, 1):
                old = latest.get((index, side))
                if old is None or old.get("status") != "ok":
                    jobs.append((category, index, moves, side))

    def play(job: tuple[str, int, list[str], int]) -> dict:
        category, index, moves, side = job
        row = local_arena.play_game(
            opponent="main_504_off", opening_index=index, opening=moves, zq_player=side,
            zq_factory=lambda: local_arena.UciPlayer(
                [str(candidate_exe), "--nnue", str(candidate_weights)], "hybrid_858"),
            opponent_factory=lambda: local_arena.UciPlayer(
                [str(baseline_exe), "--nnue", str(baseline_weights)], "main_504_off"),
            zq_budget=move_ms or initial_ms, opponent_budget=move_ms or initial_ms,
            move_timeout_s=20.0 if time_control == "200ms" else 240.0, max_plies=240,
            run_id=manifest["run_id"], clock_initial_ms=initial_ms,
            clock_increment_ms=increment_ms,
        )
        row["category"] = category
        return row

    random.Random(campaign.CONFIG["seed"]).shuffle(jobs)
    with games_path.open("a", encoding="utf-8", buffering=1) as stream:
        with concurrent.futures.ThreadPoolExecutor(
                max_workers=min(CONFIG["h2h_workers"], 12)) as pool:
            futures = [pool.submit(play, job) for job in jobs]
            for future in concurrent.futures.as_completed(futures):
                row = future.result()
                stream.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
                latest[int(row["opening_index"]), int(row["zq_player"])] = row

    rows = list(latest.values())
    summary = {"all": local_arena.summarize_pairs(
        rows, bootstrap=campaign.CONFIG["bootstrap"], seed=campaign.CONFIG["seed"])}
    for category in ("center", "normal"):
        category_rows = [row for row in rows if row.get("category") == category]
        summary[category] = local_arena.summarize_pairs(
            category_rows, bootstrap=campaign.CONFIG["bootstrap"], seed=campaign.CONFIG["seed"])
        if (summary[category].get("failed_games")
                or summary[category].get("complete_pairs") != expected_pairs[category]):
            raise RuntimeError(f"incomplete {category} cross-network H2H")
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def wait_for_process(pid: int, expected_token: int | None) -> None:
    """Wait for a Windows process to exit without relying on os.kill(pid, 0)."""
    if os.name != "nt":
        while True:
            try:
                os.kill(pid, 0)
            except OSError:
                return
            time.sleep(30)
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel.WaitForSingleObject.restype = wintypes.DWORD
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    from tools.run_edge_acc_queue import process_token
    while True:
        handle = kernel.OpenProcess(0x00100000 | 0x1000, False, pid)  # SYNCHRONIZE | QUERY_LIMITED_INFORMATION
        if not handle:
            error = ctypes.get_last_error()
            if error == 87:  # ERROR_INVALID_PARAMETER: process no longer exists
                return
            raise ctypes.WinError(error)
        try:
            result = kernel.WaitForSingleObject(handle, 30000)
        finally:
            kernel.CloseHandle(handle)
        if result == 0:  # WAIT_OBJECT_0
            return
        if result != 0x102:  # WAIT_TIMEOUT
            raise ctypes.WinError(ctypes.get_last_error())
        if expected_token is not None:
            token = process_token(pid)
            if token is not None and token != expected_token:
                return


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait-pid", type=int)
    parser.add_argument("--wait-token", type=int)
    parser.add_argument("--time-control", choices=("200ms", "3plus2"), default="200ms")
    parser.add_argument("--shard-index", type=int, default=None)
    parser.add_argument("--shard-count", type=int, default=CONFIG["shard_count"])
    parser.add_argument("--require-queue-complete", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.shard_count < 1:
        parser.error("--shard-count must be positive")
    if args.shard_index is not None and not 0 <= args.shard_index < args.shard_count:
        parser.error("--shard-index must be in [0, --shard-count)")
    if args.time_control == "200ms" and args.shard_index is not None:
        parser.error("--shard-index is supported only with --time-control 3plus2")
    if args.wait_pid:
        wait_for_process(args.wait_pid, args.wait_token)
    if args.require_queue_complete:
        status_path = queue_status_path = campaign.OUT / "queue" / "status.json"
        if not status_path.is_file():
            raise RuntimeError(f"main campaign queue status is missing: {queue_status_path}")
        queue_status = json.loads(status_path.read_text(encoding="utf-8"))
        if queue_status.get("state") != "complete":
            raise RuntimeError("main campaign did not complete; cross-network 3+2 was not started")
    if args.dry_run:
        campaign.validate_parity(CANDIDATE_NETWORK, CANDIDATE_VARIANT)
        candidate_weights = Path(campaign.CONFIG["networks"][CANDIDATE_NETWORK]["weights"]).resolve()
        baseline_weights = Path(campaign.CONFIG["networks"][BASELINE_NETWORK]["weights"]).resolve()
        print(json.dumps({"status": "PARITY_PASS", "candidate": "858 hybrid",
                          "baseline": "main 504 OFF", "time_control": args.time_control,
                          "shard": ([args.shard_index, args.shard_count]
                                    if args.shard_index is not None else None),
                          "candidate_nnue_sha256": digest(candidate_weights),
                          "baseline_nnue_sha256": digest(baseline_weights),
                          "games": 2 * (sum(campaign.CONFIG["h2h_pairs"].values())
                                        if args.time_control == "200ms" else
                                        (sum(100 * (args.shard_index + 1) // args.shard_count
                                             - 100 * args.shard_index // args.shard_count
                                             for _ in ("center", "normal"))
                                         if args.shard_index is not None else 200)),
                          "pairs_by_category": campaign.CONFIG["h2h_pairs"]
                          if args.time_control == "200ms" else
                          ({category: 100 * (args.shard_index + 1) // args.shard_count
                            - 100 * args.shard_index // args.shard_count
                            for category in ("center", "normal")}
                           if args.shard_index is not None else
                           {"center": 100, "normal": 100})}, indent=2))
        return 0
    print(json.dumps(run(args.time_control, args.shard_index, args.shard_count), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
