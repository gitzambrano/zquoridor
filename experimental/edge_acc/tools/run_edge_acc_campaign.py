#!/usr/bin/env python3
"""Prepare parity-gated, paired EdgeAcc campaigns on fixed books."""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import random
import statistics
import subprocess
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAIN = Path(r"C:\Projetos\Zquoridor")
OUT = ROOT / "results" / "benchmarks" / "edge_acc_campaign_clock_v2"
CONFIG = {
    "main_root": str(MAIN),
    "output": str(OUT),
    "seed": 20261006,
    "move_time_ms": 200,
    "h2h_pairs": {"center": 133, "normal": 67},
    "external_pairs": {"center": 100, "normal": 50},
    "h2h_workers": 8,
    "gpu_workers": 2,
    "claustrophobia_device": "gpu",
    # Retained for the player constructor/API; native clock mode has no fixed
    # simulation budget and stops when its deadline expires.
    "claustrophobia_max_sims": 4294967295,
    "claustrophobia_3plus2_max_sims": 4294967295,
    "clock_protocol": "edge-acc-native-clock-v2",
    "claustrophobia_bridge_protocol": "deadline-and-engine-clock-v2",
    "clock_protocol_validation": str(OUT / "clock_protocol_validation.json"),
    "clock_policy": {
        "200ms": "native per-move deadline: 200 ms",
        "3plus2": "native remaining-clock plus 2 s increment allocation",
    },
    "claustrophobia_search_policy": (
        "deadline-driven MCTS; warmup is not used to estimate search rate; stop after a complete evaluation wave and report actual simulations/overshoot"
    ),
    "bootstrap": 20000,
    "normal_book": str(ROOT / "tools" / "external" / "openings_normal_confirm_400.jsonl"),
    "center_book": str(ROOT / "tools" / "external" / "openings_center_rush_sound_5k.jsonl"),
    "networks": {
        "504": {
            "arch": "multipath_phase_bucketed",
            "parity_key": "n504",
            "weights": str(MAIN / "data" / "nnue" / "nnue_weights_int8.bin"),
            "features": 504,
            "contact": 0,
        },
        "858": {
            "arch": "multipath_phase_contact_bucketed",
            "parity_key": "contact858",
            "weights": str(MAIN / "results" / "experiments" / "contact_soup_tri_512" / "soup_tri_champion" / "student_int8.bin"),
            "features": 858,
            "contact": 1,
        },
    },
    # The unsafe canonical v2 remains buildable only for speed measurement.
    "variants": {
        "off": {},
        "v1": {"ZQ_EXP_EDGE_ACC_CACHE": 1},
        "v2": {"ZQ_EXP_EDGE_ACC_FULL_CANONICAL": 1},
        "v3": {"ZQ_EXP_EDGE_ACC_PATH_CONTEXT": 1},
        "v3_node": {"ZQ_EXP_EDGE_ACC_NODE_CONTEXT": 1},
        "v3_parent": {"ZQ_EXP_EDGE_ACC_NODE_CONTEXT": 1, "ZQ_EXP_EDGE_ACC_PARENT_CONTEXT": 1},
        "hybrid_sparse": {"ZQ_EXP_EDGE_ACC_CACHE": 1, "ZQ_EXP_EDGE_FEATURE_DELTA": 1},
        "hybrid_dense": {"ZQ_EXP_EDGE_ACC_CACHE": 1, "ZQ_EXP_EDGE_FEATURE_DELTA": 1, "ZQ_EXP_EDGE_DENSE_DELTA": 1},
        "delta_dense_only": {"ZQ_EXP_EDGE_FEATURE_DELTA": 1, "ZQ_EXP_EDGE_DENSE_DELTA": 1, "ZQ_EXP_EDGE_BFS_REPLAY": 1},
        "v3_node_dense": {"ZQ_EXP_EDGE_ACC_NODE_CONTEXT": 1, "ZQ_EXP_EDGE_FEATURE_DELTA": 1, "ZQ_EXP_EDGE_DENSE_DELTA": 1},
        "hybrid_dense_bfs": {"ZQ_EXP_EDGE_ACC_CACHE": 1, "ZQ_EXP_EDGE_FEATURE_DELTA": 1, "ZQ_EXP_EDGE_DENSE_DELTA": 1, "ZQ_EXP_EDGE_BFS_REPLAY": 1},
        "v3_node_dense_bfs": {"ZQ_EXP_EDGE_ACC_NODE_CONTEXT": 1, "ZQ_EXP_EDGE_FEATURE_DELTA": 1, "ZQ_EXP_EDGE_DENSE_DELTA": 1, "ZQ_EXP_EDGE_BFS_REPLAY": 1},
    },
    "parity": {
        "manifest": str(ROOT / "results" / "benchmarks" / "edge-parity-local" / "parity_manifest.json"),
        "alias_manifest": str(ROOT / "results" / "benchmarks" / "edge-parity-local" / "edge_acc_alias_tree_parity.json"),
        "natural_manifest": str(ROOT / "results" / "benchmarks" / "edge-parity-local" / "edge_acc_natural_tree_parity.json"),
        "required_networks": ["504", "858"],
        "alias_node_budgets": [1000, 4000],
        "natural_node_budgets": [100000, 300000],
        "natural_variants": ["v3", "v3_node", "v3_node_dense", "v3_node_dense_bfs", "delta_dense_only"],
    },
    "three_plus_two": {
        "h2h_pairs_per_shard": 100,
        "external_pairs_per_shard": 100,
        "shard_count": 4,
        "opponents": ["claustrophobia", "titanium"],
        "time_control": "3+2",
    },
}

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools import run_benchmark
from tools.external import local_arena


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_hash(source_root: Path) -> str:
    digest = hashlib.sha256()
    files = [source_root / "tools" / "external" / "zquoridor_uci.cpp"]
    files += sorted((source_root / "src").glob("*.hpp"))
    for path in files:
        digest.update(path.relative_to(source_root).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def network_flags(network: str) -> list[str]:
    if network == "504":
        return []
    spec = CONFIG["networks"][network]
    return [
        "-DZQ_NNUE_RACE_FEATURES=1", "-DZQ_NNUE_MULTIPATH_FEATURES=1",
        "-DZQ_NNUE_MARGIN_REGIME_FEATURES=0", "-DZQ_NNUE_PHASE_FEATURES=1",
        f"-DZQ_NNUE_CONTACT_FEATURES={spec['contact']}", "-DZQ_NNUE_HIDDEN=512",
        "-DZQ_NNUE_VALUE_BUCKETS=6", "-DZQ_NNUE_VALUE_DEPTH=2",
    ]


def build(network: str, variant: str, source_root: Path, label: str) -> Path:
    if variant not in CONFIG["variants"]:
        raise ValueError(f"unknown variant {variant}")
    exe = ROOT / "bin" / "edge_campaign" / f"zq_{network}_{label}_{variant}.exe"
    exe.parent.mkdir(parents=True, exist_ok=True)
    compiler = "g++"
    command = [compiler, "-O3", "-DNDEBUG", "-std=c++17", "-pthread",
               *network_flags(network)]
    command.extend(f"-D{name}={value}" for name, value in CONFIG["variants"][variant].items())
    command.extend(["-Isrc", str(source_root / "tools" / "external" / "zquoridor_uci.cpp"),
                    "-o", str(exe)])
    stamp = exe.with_suffix(".build.json")
    identity = {"command": command, "source_sha256": source_hash(source_root),
                "weights_sha256": sha256(Path(CONFIG["networks"][network]["weights"]))}
    if exe.is_file() and stamp.is_file() and json.loads(stamp.read_text(encoding="utf-8")) == identity:
        return exe.resolve()
    subprocess.run(command, cwd=source_root, check=True)
    stamp.write_text(json.dumps(identity, indent=2) + "\n", encoding="utf-8")
    return exe.resolve()


def selected_book(label: str) -> list[tuple[int, list[str]]]:
    key = "center_book" if label == "center" else "normal_book"
    count = CONFIG["h2h_pairs"][label]
    return run_benchmark._read_openings(Path(CONFIG[key]), count, CONFIG["seed"])


def materialize_book(label: str, *, pairs: int, tc: str) -> Path:
    source = Path(CONFIG["center_book"] if label == "center" else CONFIG["normal_book"])
    selected = run_benchmark._read_openings(source, pairs, CONFIG["seed"])
    target = OUT / "books" / tc / f"{label}_{pairs}_pairs.jsonl"
    target.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for source_index, moves in selected:
        lines.append(json.dumps({"opening_index": source_index, "moves": moves}, separators=(",", ":")))
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


def prepare_books() -> dict:
    books = {}
    for label in ("center", "normal"):
        selected = selected_book(label)
        books[label] = [{"source_index": index, "moves": moves} for index, moves in selected]
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "opening_selection.json"
    payload = {"seed": CONFIG["seed"], "books": books}
    encoded = json.dumps(payload, indent=2) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") != encoded:
        raise RuntimeError("opening selection already exists with different content")
    path.write_text(encoded, encoding="utf-8")
    return payload


def source_hash_map() -> dict[str, str]:
    return {str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path)
            for path in sorted((ROOT / "src").glob("*.hpp"))}


def normalize_source_hashes(source_hashes: dict[str, str]) -> dict[str, str]:
    """Normalize manifest path separators before a source hash comparison."""
    return {str(path).replace("\\", "/"): digest for path, digest in source_hashes.items()}


def validate_parity_entry(manifest: dict, network: str, variant: str,
                          expected_source: dict[str, str], weights_path: Path) -> dict:
    key = f"{CONFIG['networks'][network]['parity_key']}/{variant}"
    if normalize_source_hashes(manifest.get("source_sha256", {})) != normalize_source_hashes(expected_source):
        raise RuntimeError("parity source hashes do not match the current src/*.hpp files")
    recorded_weight = manifest.get("weights", {}).get(CONFIG["networks"][network]["parity_key"], {})
    if recorded_weight.get("sha256") != sha256(weights_path):
        raise RuntimeError(f"parity weight hash does not match {network}")
    builds = [row for row in manifest.get("builds", [])
              if row.get("network") == CONFIG["networks"][network]["parity_key"]
              and row.get("variant") == variant and row.get("kind") == "uci"]
    if not builds or builds[-1].get("returncode") != 0:
        raise RuntimeError(f"parity UCI build is missing or failed for {network}/{variant}")
    command = builds[-1].get("command", [])
    expected_macros = network_flags(network) + [
        f"-D{name}={value}" for name, value in CONFIG["variants"][variant].items()]
    actual_macros = [arg for arg in command if arg.startswith("-DZQ_")]
    if actual_macros != expected_macros:
        raise RuntimeError(f"parity compiler macro flags do not match campaign build for {network}/{variant}")
    if not all(flag in command for flag in ("-O3", "-DNDEBUG", "-std=c++17", "-pthread", "-Isrc")):
        raise RuntimeError(f"parity compiler base flags do not match campaign build for {network}/{variant}")
    large = manifest.get("large_tree", {}).get(key, {})
    comparison = large.get("comparison_to_off", {})
    if large.get("status") != "RAN" or comparison.get("status") != "PASS" or comparison.get("mismatches"):
        raise RuntimeError(f"large-tree parity gate failed for {network}/{variant}")
    if comparison.get("actual_off") != 10 or comparison.get("actual_candidate") != 10:
        raise RuntimeError(f"large-tree parity did not complete 10 checkpoints for {network}/{variant}")
    accumulator = manifest.get("accumulator", {}).get(key, {})
    if accumulator.get("status") != "PASS" or "EDGE_ACC_PARITY_OK" not in accumulator.get("stdout", ""):
        raise RuntimeError(f"direct accumulator parity gate failed for {network}/{variant}")
    fixed = manifest.get("fixed_node", {}).get(key, {})
    fixed_comparison = fixed.get("comparison_to_off", {})
    if fixed_comparison.get("status") != "PASS" or fixed_comparison.get("positions") != 80 or fixed_comparison.get("mismatches"):
        raise RuntimeError(f"fixed-node parity gate failed for {network}/{variant}")
    return {"large_tree": large, "accumulator": accumulator, "fixed_node": fixed}


def validate_parity(network: str, variant: str) -> dict:
    if variant == "v2":
        raise RuntimeError("v2 is a speed-only reference and cannot enter game campaigns")
    path = Path(CONFIG["parity"]["manifest"])
    if not path.is_file():
        raise RuntimeError(f"parity gate is closed: missing {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not manifest.get("finished_local"):
        raise RuntimeError("parity manifest is incomplete; wait for run_local_edge_parity.py to finish")
    sources = source_hash_map()
    approvals = {}
    for net in CONFIG["parity"]["required_networks"]:
        weights_path = Path(CONFIG["networks"][net]["weights"]).resolve()
        approvals[net] = validate_parity_entry(
            manifest, net, variant, sources, weights_path)
        alias_path = Path(CONFIG["parity"]["alias_manifest"])
        if not alias_path.is_file():
            raise RuntimeError(f"adversarial alias gate is closed: missing {alias_path}")
        alias_manifest = json.loads(alias_path.read_text(encoding="utf-8"))
        validate_alias_entry(alias_manifest, net, variant, sources, weights_path)
        if variant in CONFIG["parity"]["natural_variants"]:
            natural_path = Path(CONFIG["parity"]["natural_manifest"])
            if not natural_path.is_file():
                raise RuntimeError(f"natural sibling gate is closed: missing {natural_path}")
            natural_manifest = json.loads(natural_path.read_text(encoding="utf-8"))
            validate_natural_entry(natural_manifest, net, variant, sources, weights_path)
    return approvals[network]


def validate_alias_entry(manifest: dict, network: str, variant: str,
                         expected_source: dict[str, str], weights_path: Path) -> dict:
    """Require fresh, source-bound adversarial tree parity for one network."""
    if not manifest.get("finished_local"):
        raise RuntimeError("adversarial alias manifest is incomplete")
    if normalize_source_hashes(manifest.get("source_sha256", {})) != normalize_source_hashes(expected_source):
        raise RuntimeError("adversarial alias source hashes do not match current src/*.hpp files")
    recorded_weight = manifest.get("weights", {}).get(network, {})
    if recorded_weight.get("sha256") != sha256(weights_path):
        raise RuntimeError(f"adversarial alias weight hash does not match {network}")
    rows = [row for row in manifest.get("comparisons", [])
            if row.get("network") == network and row.get("variant") == variant
            and row.get("phase") in ("warm", "alias")]
    expected = {(phase, nodes) for phase in ("warm", "alias")
                for nodes in CONFIG["parity"]["alias_node_budgets"]}
    actual = {(row.get("phase"), row.get("nodes")) for row in rows}
    if actual != expected:
        raise RuntimeError(f"adversarial alias gate is incomplete for {network}/{variant}: {actual}")
    failures = [row for row in rows if row.get("status") != "PASS" or row.get("mismatches")]
    if failures:
        raise RuntimeError(f"adversarial alias gate failed for {network}/{variant}: {failures}")
    return {"status": "PASS", "comparisons": rows}


def validate_natural_entry(manifest: dict, network: str, variant: str,
                           expected_source: dict[str, str], weights_path: Path) -> dict:
    """Require natural H63/V63 sibling coverage at every fixed node budget."""
    if not manifest.get("finished_local"):
        raise RuntimeError("natural sibling manifest is incomplete")
    if normalize_source_hashes(manifest.get("source_sha256", {})) != normalize_source_hashes(expected_source):
        raise RuntimeError("natural sibling source hashes do not match current src/*.hpp files")
    recorded_weight = manifest.get("weights", {}).get(network, {})
    if recorded_weight.get("sha256") != sha256(weights_path):
        raise RuntimeError(f"natural sibling weight hash does not match {network}")
    rows = [row for row in manifest.get("comparisons", [])
            if row.get("network") == network and row.get("variant") == variant]
    expected = set(CONFIG["parity"]["natural_node_budgets"])
    actual = {row.get("nodes") for row in rows}
    if actual != expected:
        raise RuntimeError(f"natural sibling gate is incomplete for {network}/{variant}: {actual}")
    failures = [row for row in rows if row.get("status") != "PASS" or row.get("mismatches")]
    if failures:
        raise RuntimeError(f"natural sibling gate failed for {network}/{variant}: {failures}")
    result_rows = [row for row in manifest.get("results", [])
                   if row.get("network") == network and row.get("variant") == variant]
    coverage = {row.get("nodes"): (row.get("h63_n", 0), row.get("v63_n", 0))
                for row in result_rows}
    if set(coverage) != expected or any(h <= 0 or v <= 0 for h, v in coverage.values()):
        raise RuntimeError(f"natural sibling branches were not both visited for {network}/{variant}: {coverage}")
    return {"status": "PASS", "comparisons": rows, "branch_visits": coverage}

def h2h(network: str, variant: str) -> dict:
    approval = validate_parity(network, variant)
    exe = build(network, variant, ROOT, "exp")
    off_exe = build(network, "off", MAIN, "main")
    weights = Path(CONFIG["networks"][network]["weights"]).resolve()
    run_id = f"h2h_{network}_{variant}"
    out = OUT / run_id
    selected = prepare_books()["books"]
    artifacts = {"candidate_executable": exe, "main_executable": off_exe,
                 "candidate_nnue": weights, "center_book": Path(CONFIG["center_book"]),
                 "normal_book": Path(CONFIG["normal_book"]),
                 "referee": ROOT / "tools" / "external" / "local_arena.py"}
    manifest = local_arena.make_manifest({"protocol": CONFIG["clock_protocol"], "time_control": "200ms-movetime",
        "network": network,
        "variant": variant, "parity_source_sha256": hashlib.sha256(json.dumps(source_hash_map(), sort_keys=True).encode()).hexdigest(),
        "seed": CONFIG["seed"], "move_time_ms": CONFIG["move_time_ms"],
        "clock_policy": CONFIG["clock_policy"]["200ms"],
        "selected_openings": selected}, artifacts)
    games_path, prior = local_arena.prepare_resume(out, manifest)
    latest = {(int(row["opening_index"]), int(row["zq_player"])): row for row in prior}
    jobs = []
    # Stable disjoint index ranges preserve category identity in result files.
    for category, rows in (("center", selected["center"]), ("normal", selected["normal"])):
        offset = 0 if category == "center" else 100000
        for item in rows:
            index, moves = offset + int(item["source_index"]), item["moves"]
            for side in (0, 1):
                if (index, side) not in latest or latest[index, side].get("status") != "ok":
                    jobs.append((category, index, moves, side))

    def play(job: tuple[str, int, list[str], int]) -> dict:
        category, index, moves, side = job
        cmd = [str(exe), "--nnue", str(weights)]
        return local_arena.play_game(opponent="main", opening_index=index,
            opening=moves, zq_player=side,
            zq_factory=lambda: local_arena.UciPlayer(cmd, variant),
            opponent_factory=lambda: local_arena.UciPlayer(
                [str(off_exe), "--nnue", str(weights)], "main"),
            zq_budget=CONFIG["move_time_ms"], opponent_budget=CONFIG["move_time_ms"],
            move_timeout_s=30.0, max_plies=240, run_id=manifest["run_id"])

    random.Random(CONFIG["seed"]).shuffle(jobs)
    # Candidate and production baseline share the same weights, openings, and seed.
    with games_path.open("a", encoding="utf-8", buffering=1) as stream:
        with concurrent.futures.ThreadPoolExecutor(max_workers=CONFIG["h2h_workers"]) as pool:
            for future in concurrent.futures.as_completed([pool.submit(play, job) for job in jobs]):
                row = future.result()
                stream.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
                latest[int(row["opening_index"]), int(row["zq_player"])] = row
    rows = list(latest.values())
    summary = local_arena.summarize_pairs(rows, bootstrap=CONFIG["bootstrap"], seed=CONFIG["seed"])
    summary["center"] = local_arena.summarize_pairs([r for r in rows if int(r["opening_index"]) < 100000], bootstrap=CONFIG["bootstrap"], seed=CONFIG["seed"])
    summary["normal"] = local_arena.summarize_pairs([r for r in rows if int(r["opening_index"]) >= 100000], bootstrap=CONFIG["bootstrap"], seed=CONFIG["seed"])
    if any(summary[category].get("failed_games") or summary[category].get("complete_pairs") != CONFIG["h2h_pairs"][category] for category in ("center", "normal")):
        raise RuntimeError("H2H is incomplete; inspect games.jsonl before interpreting scores")
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def gpu_search_summary(rows: list[dict]) -> dict:
    groups = {}
    for row in rows:
        if row.get("status") != "ok":
            continue
        key = (row.get("participant"), row.get("category"))
        values = groups.setdefault(key, {"sims": [], "search_ms": [], "elapsed_ms": [], "sims_per_second": []})
        for timing in row.get("move_times", []):
            if timing.get("player") != "claustrophobia" or not timing.get("search_last"):
                continue
            try:
                search = json.loads(timing["search_last"])
                sims = int(search.get("sims", 0))
                search_ms = float(search.get("search_ms", 0.0))
                elapsed_ms = float(search.get("elapsed_ms", timing.get("elapsed_ms", 0.0)))
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
            values["sims"].append(sims)
            values["search_ms"].append(search_ms)
            values["elapsed_ms"].append(elapsed_ms)
            if elapsed_ms > 0:
                values["sims_per_second"].append(sims * 1000.0 / elapsed_ms)
    output = {}
    for (participant, category), values in groups.items():
        output[f"{participant}/{category}"] = {"moves": len(values["sims"])}
        for name, samples in values.items():
            if not samples:
                output[f"{participant}/{category}"][name] = None
                continue
            ordered = sorted(samples)
            output[f"{participant}/{category}"][name] = {
                "mean": statistics.fmean(samples),
                "p10": ordered[int(0.10 * (len(ordered) - 1))],
                "median": statistics.median(samples),
                "p90": ordered[int(0.90 * (len(ordered) - 1))],
            }
    return output


def paired_candidate_delta(rows: list[dict], category: str) -> dict:
    paired = {}
    for row in rows:
        if row.get("category") != category or row.get("status") != "ok":
            continue
        key = int(row["opening_index"])
        paired.setdefault(key, {}).setdefault(row["participant"], {})[int(row["zq_player"])] = float(row["result"])
    deltas = []
    for values in paired.values():
        if set(values) != {"candidate", "off"} or any(set(values[name]) != {0, 1} for name in ("candidate", "off")):
            continue
        deltas.append((sum(values["candidate"].values()) - sum(values["off"].values())) / 2.0)
    if not deltas:
        return {"complete_paired_openings": 0, "delta_score_pct": None, "paired_bootstrap_95": None}
    mean = sum(deltas) / len(deltas)
    rng = random.Random(CONFIG["seed"])
    samples = [sum(deltas[rng.randrange(len(deltas))] for _ in deltas) / len(deltas)
               for _ in range(CONFIG["bootstrap"])]
    samples.sort()
    low = samples[int(0.025 * len(samples))]
    high = samples[min(len(samples) - 1, int(0.975 * len(samples)))]
    return {"complete_paired_openings": len(deltas), "delta_score_pct": 100.0 * mean,
            "paired_bootstrap_95": [100.0 * low, 100.0 * high]}


class _PersistentClaustrophobiaLease:
    """Per-game facade over a worker-owned persistent model process."""

    def __init__(self, player) -> None:
        self._player = player

    def bestmove(self, *args, **kwargs):
        return self._player.bestmove(*args, **kwargs)

    def bestmove_clock(self, *args, **kwargs):
        return self._player.bestmove_clock(*args, **kwargs)

    def close(self) -> None:
        # local_arena closes per game; the worker owns this process until the
        # complete batch ends, so a lease must not close it.
        return None


class PersistentClaustrophobiaPool:
    """Load and warm one GPU model per search worker before timed game work."""

    def __init__(self, info: dict, *, workers: int, three_plus_two: bool) -> None:
        self._local = threading.local()
        self._players = []
        self._players_lock = threading.Lock()
        self._ready = threading.Barrier(workers + 1)
        max_sims = (CONFIG["claustrophobia_3plus2_max_sims"] if three_plus_two
                    else CONFIG["claustrophobia_max_sims"])
        self._player_args = (Path(info["benchmark_bridge"]), Path(info["checkpoint"]),
                             CONFIG["move_time_ms"], max_sims, 1.5,
                             CONFIG["claustrophobia_device"])
        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=workers, initializer=self._initialize_worker)
        try:
            # Each initializer waits at the barrier, forcing all workers to
            # finish model load/warmup before any timed game is submitted.
            self._warm_futures = [self.executor.submit(lambda: None) for _ in range(workers)]
            self._ready.wait(timeout=600.0)
            for future in self._warm_futures:
                future.result()
        except BaseException:
            self._ready.abort()
            self.close()
            raise

    def _initialize_worker(self) -> None:
        bridge, checkpoint, move_ms, max_sims, cpuct, device = self._player_args
        player = local_arena.ClaustrophobiaPlayer(
            bridge, checkpoint, move_time_ms=move_ms, max_sims=max_sims,
            cpuct=cpuct, device=device)
        self._local.player = player
        with self._players_lock:
            self._players.append(player)
        try:
            self._ready.wait(timeout=600.0)
        except threading.BrokenBarrierError as error:
            player.close()
            raise RuntimeError("Claustrophobia worker warmup barrier failed") from error

    def factory(self):
        player = getattr(self._local, "player", None)
        if player is None:
            raise RuntimeError("Claustrophobia game ran outside its warmed worker thread")
        return _PersistentClaustrophobiaLease(player)

    def close(self) -> None:
        executor = getattr(self, "executor", None)
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
            self.executor = None
        for player in self._players:
            player.close()
        self._players.clear()


def normalize_participants(value: str | tuple[str, ...] | list[str]) -> tuple[str, ...]:
    supplied = value.split(",") if isinstance(value, str) else list(value)
    selected = {part.strip() for part in supplied if part.strip()}
    allowed = {"off", "candidate"}
    if not selected or selected - allowed:
        raise ValueError("participants must be a non-empty subset of 'off,candidate'")
    return tuple(name for name in ("candidate", "off") if name in selected)


def external_output_dir(network: str, variant: str, opponent: str, *,
                        three_plus_two: bool, shard_index: int, shard_count: int,
                        participants: tuple[str, ...]) -> Path:
    tc = "3plus2" if three_plus_two else "200ms"
    mode_variant = "off" if participants == ("off",) else variant
    out = OUT / tc / network / mode_variant / opponent
    if three_plus_two:
        out /= f"shard_{shard_index:03d}_of_{shard_count:03d}"
    return out


def shared_off_rows(network: str, opponent: str, *, three_plus_two: bool,
                    shard_index: int, shard_count: int, candidate_manifest: dict,
                    expected_categories: dict, expected_off_executable: Path) -> list[dict]:
    """Load the separately-run OFF baseline only after validating its provenance."""
    baseline_dir = external_output_dir(network, "off", opponent,
        three_plus_two=three_plus_two, shard_index=shard_index,
        shard_count=shard_count, participants=("off",))
    manifest_path = baseline_dir / "manifest.json"
    games_path = baseline_dir / "games.jsonl"
    if not manifest_path.is_file() or not games_path.is_file():
        raise RuntimeError(f"shared OFF baseline is missing: {baseline_dir}")
    baseline_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    base_cfg = baseline_manifest.get("config", {})
    cand_cfg = candidate_manifest.get("config", {})
    required_identity = ("protocol", "network", "opponent", "time_control", "move_time_ms",
                         "clock_initial_ms", "clock_increment_ms", "clock_policy",
                         "claustrophobia_max_sims", "claustrophobia_search_policy",
                         "opponent_rng", "seed", "shard",
                         "selected_openings")
    if (base_cfg.get("participants") != ["off"] or not base_cfg.get("baseline_only")
            or any(base_cfg.get(key) != cand_cfg.get(key) for key in required_identity)):
        raise RuntimeError("shared OFF baseline identity does not match candidate campaign")
    if base_cfg.get("participant_source_sha256", {}).get("off") != source_hash(MAIN):
        raise RuntimeError("shared OFF baseline source hash is stale")
    base_artifacts = baseline_manifest.get("artifacts", {})
    cand_artifacts = candidate_manifest.get("artifacts", {})
    for name in ("nnue", "center_book", "normal_book", "referee", "titanium_executable",
                 "titanium_clock_bridge_build", "claustrophobia_checkpoint", "claustrophobia_bridge",
                 "claustrophobia_worker", "claustrophobia_clock_bridge_build"):
        if name in cand_artifacts and base_artifacts.get(name, {}).get("sha256") != cand_artifacts[name].get("sha256"):
            raise RuntimeError(f"shared OFF baseline artifact mismatch: {name}")
    expected_exe = Path(expected_off_executable).resolve()
    expected_entry = {"sha256": sha256(expected_exe), "size": expected_exe.stat().st_size}
    baseline_entry = base_artifacts.get("main_executable", {})
    if any(baseline_entry.get(key) != value for key, value in expected_entry.items()):
        raise RuntimeError("shared OFF baseline executable does not match the Main OFF build")
    rows = []
    for lineno, line in enumerate(games_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("run_id") != baseline_manifest.get("run_id") or row.get("participant") != "off":
            raise RuntimeError(f"invalid shared OFF baseline row at {games_path}:{lineno}")
        rows.append(row)
    expected = {(category, int(index) + (0 if category == "center" else 100000), side)
                for category, openings in expected_categories.items()
                for index, _moves in openings for side in (0, 1)}
    actual = {(row.get("category"), int(row["opening_index"]), int(row["zq_player"]))
              for row in rows if row.get("status") == "ok"}
    if actual != expected:
        raise RuntimeError(f"shared OFF baseline is incomplete or has unexpected games: "
                           f"{len(actual)}/{len(expected)}")
    return rows


def run_external_campaign(network: str, variant: str, opponent: str, *,
                          three_plus_two: bool = False,
                          shard_index: int = 0, shard_count: int = 1,
                          participants: tuple[str, ...] = ("candidate", "off")) -> dict:
    participants = normalize_participants(participants)
    if participants == ("off",) and variant != "off":
        raise ValueError("baseline-only external runs use --variant off")
    approval = validate_parity(network, variant) if "candidate" in participants else None
    if opponent == "claustrophobia":
        import torch
        if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
            raise RuntimeError("GPU benchmark requested but torch reports no CUDA device")
    candidate_exe = build(network, variant, ROOT, "exp") if "candidate" in participants else None
    main_exe = build(network, "off", MAIN, "main") if "off" in participants else None
    weights = Path(CONFIG["networks"][network]["weights"]).resolve()
    bots = setup_bots()
    tc = "3plus2" if three_plus_two else "200ms"
    categories = {}
    for category in ("center", "normal"):
        pairs = 100 if three_plus_two else CONFIG["external_pairs"][category]
        book_path = Path(CONFIG["center_book"] if category == "center" else CONFIG["normal_book"])
        rows = run_benchmark._read_openings(book_path, pairs, CONFIG["seed"])
        if three_plus_two:
            shard_size = (pairs + shard_count - 1) // shard_count
            start = shard_index * shard_size
            rows = rows[start:min(start + shard_size, pairs)]
        categories[category] = rows
    if any(not rows for rows in categories.values()):
        raise ValueError("the selected shard contains no openings")

    out = external_output_dir(network, variant, opponent,
        three_plus_two=three_plus_two, shard_index=shard_index,
        shard_count=shard_count, participants=participants)
    out.mkdir(parents=True, exist_ok=True)
    artifacts = {"nnue": weights, "center_book": Path(CONFIG["center_book"]),
                 "normal_book": Path(CONFIG["normal_book"]),
                 "referee": ROOT / "tools" / "external" / "local_arena.py"}
    if candidate_exe is not None:
        artifacts["candidate_executable"] = candidate_exe
    if main_exe is not None:
        artifacts["main_executable"] = main_exe
    if opponent == "titanium":
        artifacts["titanium_executable"] = Path(bots[opponent]["executable"])
        artifacts["titanium_clock_bridge_build"] = Path(bots[opponent]["clock_bridge_build"])
    else:
        artifacts["claustrophobia_checkpoint"] = Path(bots[opponent]["checkpoint"])
        artifacts["claustrophobia_bridge"] = Path(bots[opponent]["benchmark_bridge"])
        artifacts["claustrophobia_worker"] = Path(bots[opponent]["benchmark_bridge"]).parent / "zq_inference_worker.py"
        artifacts["claustrophobia_clock_bridge_build"] = Path(bots[opponent]["clock_bridge_build"])
    participant_sources = {}
    if candidate_exe is not None:
        participant_sources["candidate"] = source_hash(ROOT)
    if main_exe is not None:
        participant_sources["off"] = source_hash(MAIN)
    identity = {"protocol": CONFIG["clock_protocol"], "network": network, "variant": variant,
        "participants": list(participants), "baseline_only": participants == ("off",),
        "opponent": opponent, "time_control": tc, "move_time_ms": CONFIG["move_time_ms"],
        "clock_initial_ms": 180000 if three_plus_two else 0,
        "clock_increment_ms": 2000 if three_plus_two else 0,
        "clock_policy": CONFIG["clock_policy"][tc],
        "claustrophobia_max_sims": (CONFIG["claustrophobia_3plus2_max_sims"] if three_plus_two else CONFIG["claustrophobia_max_sims"]),
        "claustrophobia_search_policy": CONFIG["claustrophobia_search_policy"],
        "opponent_rng": "deterministic search; no RNG seed parameter or root noise",
        "seed": CONFIG["seed"], "shard": [shard_index, shard_count],
        "selected_openings": {k: [i for i, _ in v] for k, v in categories.items()},
        "participant_source_sha256": participant_sources,
        "parity_source_sha256": hashlib.sha256(json.dumps(source_hash_map(), sort_keys=True).encode()).hexdigest()}
    manifest = local_arena.make_manifest(identity, artifacts)
    shared_baseline_rows = None
    if participants == ("candidate",):
        # Fail before scheduling expensive games if the common OFF run is
        # absent, incomplete, or tied to different sources/assets/openings.
        expected_main_exe = build(network, "off", MAIN, "main")
        shared_baseline_rows = shared_off_rows(network, opponent,
            three_plus_two=three_plus_two, shard_index=shard_index,
            shard_count=shard_count, candidate_manifest=manifest,
            expected_categories=categories, expected_off_executable=expected_main_exe)
    games_path, prior = local_arena.prepare_resume(out, manifest)
    latest = {(row.get("participant"), row.get("category"), int(row["opening_index"]), int(row["zq_player"])): row for row in prior}
    jobs = []
    for participant in participants:
        for category, openings in categories.items():
            offset = 0 if category == "center" else 100000
            for source_index, moves in openings:
                index = offset + int(source_index)
                for side in (0, 1):
                    key = (participant, category, index, side)
                    if key not in latest or latest[key].get("status") != "ok":
                        jobs.append((participant, category, index, moves, side))

    workers = CONFIG["gpu_workers"] if opponent == "claustrophobia" else CONFIG["h2h_workers"]
    persistent_cla = (PersistentClaustrophobiaPool(
        bots["claustrophobia"], workers=workers, three_plus_two=three_plus_two)
        if opponent == "claustrophobia" and jobs else None)

    def play(job: tuple[str, str, int, list[str], int]) -> dict:
        participant, category, index, moves, side = job
        engine = candidate_exe if participant == "candidate" else main_exe
        command = [str(engine), "--nnue", str(weights)]
        zq_factory = lambda: local_arena.UciPlayer(command, participant)
        if opponent == "titanium":
            opponent_factory = lambda: local_arena.TitaniumPlayer(Path(bots[opponent]["executable"]))
            opponent_budget = CONFIG["move_time_ms"]
        else:
            if persistent_cla is None:
                raise RuntimeError("Claustrophobia worker pool was not initialized")
            opponent_factory = persistent_cla.factory
            opponent_budget = CONFIG["move_time_ms"]
        row = local_arena.play_game(opponent=opponent, opening_index=index, opening=moves,
            zq_player=side, zq_factory=zq_factory, opponent_factory=opponent_factory,
            zq_budget=CONFIG["move_time_ms"], opponent_budget=opponent_budget,
            move_timeout_s=240.0 if three_plus_two else 30.0, max_plies=240,
            run_id=manifest["run_id"], clock_initial_ms=180000 if three_plus_two else 0,
            clock_increment_ms=2000 if three_plus_two else 0)
        row.update({"participant": participant, "category": category})
        return row

    random.Random(CONFIG["seed"]).shuffle(jobs)
    print(f"Running {len(jobs)} games with one shared {workers}-worker pool; Center and Normal are interleaved.", flush=True)
    executor = (persistent_cla.executor if persistent_cla is not None
                else concurrent.futures.ThreadPoolExecutor(max_workers=workers))
    try:
        with games_path.open("a", encoding="utf-8", buffering=1) as stream:
            futures = [executor.submit(play, job) for job in jobs]
            for future in concurrent.futures.as_completed(futures):
                row = future.result()
                stream.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
                key = (row["participant"], row["category"], int(row["opening_index"]), int(row["zq_player"]))
                latest[key] = row
    finally:
        if persistent_cla is not None:
            persistent_cla.close()
        else:
            executor.shutdown(wait=True, cancel_futures=True)
    report = {"network": network, "variant": variant, "opponent": opponent,
              "time_control": tc, "participants": {}}
    for participant in participants:
        report["participants"][participant] = {}
        for category in ("center", "normal"):
            group = [r for key, r in latest.items() if key[0] == participant and key[1] == category]
            report["participants"][participant][category] = local_arena.summarize_pairs(
                group, bootstrap=CONFIG["bootstrap"], seed=CONFIG["seed"])
    report["gpu_search_metrics"] = gpu_search_summary(list(latest.values())) if opponent == "claustrophobia" else None
    if set(participants) == {"candidate", "off"}:
        report["paired_delta_candidate_minus_off"] = {
            category: paired_candidate_delta([r for r in latest.values()], category)
            for category in ("center", "normal")}
    elif participants == ("candidate",):
        paired_rows = list(latest.values()) + (shared_baseline_rows or [])
        report["paired_delta_candidate_minus_off"] = {
            category: paired_candidate_delta(paired_rows, category)
            for category in ("center", "normal")}
    else:
        report["paired_delta_candidate_minus_off"] = None
    expected_pairs = {category: len(rows) for category, rows in categories.items()}
    for participant in participants:
        for category, count in expected_pairs.items():
            summary = report["participants"][participant][category]
            if summary.get("failed_games") != 0 or summary.get("complete_pairs") != count:
                raise RuntimeError(f"incomplete external campaign: {participant}/{category} "
                                   f"pairs={summary.get('complete_pairs')} failed={summary.get('failed_games')}")
            if report["paired_delta_candidate_minus_off"] is not None:
                delta = report["paired_delta_candidate_minus_off"][category]
                if delta.get("complete_paired_openings") != count:
                    raise RuntimeError(f"missing paired delta data: {participant}/{category}")
    (out / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def native_clock_preflight() -> dict:
    """Build and verify isolated opponent adapters before any campaign starts."""
    bots = setup_bots()
    return {name: {"executable": str(info.get("executable", info.get("benchmark_bridge"))),
                   "executable_sha256": sha256(Path(info.get("executable", info.get("benchmark_bridge")))),
                   "clock_bridge_build": str(info["clock_bridge_build"]),
                   "clock_bridge_build_sha256": sha256(Path(info["clock_bridge_build"]))}
            for name, info in bots.items()}


def validate_clock_protocol_audit(adapters: dict, audit_path: Path | None = None) -> dict:
    """Require root's live-probe approval for the exact campaign code and bots."""
    path = Path(audit_path or CONFIG["clock_protocol_validation"])
    if not path.is_file():
        raise RuntimeError(f"native-clock live validation is missing: {path}")
    audit = json.loads(path.read_text(encoding="utf-8"))
    if audit.get("status") != "PASS":
        raise RuntimeError("native-clock live validation did not pass")
    expected_sources = {
        "arena": sha256(ROOT / "tools/external/local_arena.py"),
        "claustrophobia_builder": sha256(ROOT / "tools/build_claustrophobia_dynamic_bridge.py"),
        "claustrophobia_source": sha256(ROOT / "tools/external/claustrophobia_benchmark_bridge.rs"),
        "titanium_builder": sha256(ROOT / "tools/build_titanium_clock_bridge.py"),
    }
    expected_executables = {
        "claustrophobia": adapters["claustrophobia"]["executable_sha256"],
        "titanium": adapters["titanium"]["executable_sha256"],
    }
    for name, digest in expected_sources.items():
        if audit.get("source_sha256", {}).get(name) != digest:
            raise RuntimeError(f"native-clock live validation source hash mismatch: {name}")
    for name, digest in expected_executables.items():
        if audit.get("executable_sha256", {}).get(name) != digest:
            raise RuntimeError(f"native-clock live validation executable hash mismatch: {name}")
    if audit.get("protocol") != "native-clock-v2":
        raise RuntimeError("native-clock live validation protocol does not match this campaign")
    return audit


def run_h2h_3plus2(network: str, variant: str, shard_index: int, shard_count: int) -> dict:
    approval = validate_parity(network, variant)
    candidate_exe = build(network, variant, ROOT, "exp")
    main_exe = build(network, "off", MAIN, "main")
    weights = Path(CONFIG["networks"][network]["weights"]).resolve()
    categories = {}
    for category in ("center", "normal"):
        source = Path(CONFIG["center_book"] if category == "center" else CONFIG["normal_book"])
        rows = run_benchmark._read_openings(source, 100, CONFIG["seed"])
        size = (100 + shard_count - 1) // shard_count
        start = shard_index * size
        categories[category] = rows[start:min(start + size, 100)]
    out = OUT / "3plus2" / network / variant / f"h2h_shard_{shard_index:03d}_of_{shard_count:03d}"
    artifacts = {"candidate_executable": candidate_exe, "main_executable": main_exe,
                 "nnue": weights, "center_book": Path(CONFIG["center_book"]),
                 "normal_book": Path(CONFIG["normal_book"]),
                 "referee": ROOT / "tools" / "external" / "local_arena.py"}
    manifest = local_arena.make_manifest({"protocol": CONFIG["clock_protocol"], "time_control": "3plus2-native-clock",
        "clock_policy": "native remaining-clock plus 2 s increment allocation", "network": network,
        "variant": variant, "seed": CONFIG["seed"], "initial_ms": 180000, "increment_ms": 2000,
        "shard": [shard_index, shard_count],
        "openings": {k: [i for i, _ in v] for k, v in categories.items()},
        "parity_source_sha256": hashlib.sha256(json.dumps(source_hash_map(), sort_keys=True).encode()).hexdigest()}, artifacts)
    games_path, prior = local_arena.prepare_resume(out, manifest)
    latest = {(int(r["opening_index"]), int(r["zq_player"])): r for r in prior}
    jobs = []
    for category, openings in categories.items():
        offset = 0 if category == "center" else 100000
        for source_index, moves in openings:
            for side in (0, 1):
                idx = offset + source_index
                if (idx, side) not in latest or latest[idx, side].get("status") != "ok":
                    jobs.append((category, idx, moves, side))
    random.Random(CONFIG["seed"] + shard_index).shuffle(jobs)
    def play(job):
        category, index, moves, side = job
        row = local_arena.play_game(opponent="main", opening_index=index, opening=moves,
            zq_player=side,
            zq_factory=lambda: local_arena.UciPlayer([str(candidate_exe), "--nnue", str(weights)], variant),
            opponent_factory=lambda: local_arena.UciPlayer([str(main_exe), "--nnue", str(weights)], "main"),
            zq_budget=180000, opponent_budget=180000, move_timeout_s=240.0, max_plies=240,
            run_id=manifest["run_id"], clock_initial_ms=180000, clock_increment_ms=2000)
        row["category"] = category
        return row
    with games_path.open("a", encoding="utf-8", buffering=1) as stream:
        with concurrent.futures.ThreadPoolExecutor(max_workers=CONFIG["h2h_workers"]) as pool:
            for future in concurrent.futures.as_completed([pool.submit(play, job) for job in jobs]):
                row = future.result()
                stream.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
                latest[int(row["opening_index"]), int(row["zq_player"])] = row
    rows = list(latest.values())
    result = {"all": local_arena.summarize_pairs(rows, bootstrap=CONFIG["bootstrap"], seed=CONFIG["seed"])}
    for category in ("center", "normal"):
        result[category] = local_arena.summarize_pairs([r for r in rows if r.get("category") == category], bootstrap=CONFIG["bootstrap"], seed=CONFIG["seed"])
        if result[category].get("failed_games") or result[category].get("complete_pairs") != len(categories[category]):
            raise RuntimeError(f"incomplete 3+2 H2H shard: {category}")
    (out / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result

def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("phase", nargs="?", default="prepare", choices=("prepare", "h2h", "external", "h2h-3plus2", "external-3plus2"))
    ap.add_argument("--network", choices=tuple(CONFIG["networks"]))
    ap.add_argument("--variant", choices=tuple(CONFIG["variants"]))
    ap.add_argument("--opponent", choices=("claustrophobia", "titanium"))
    ap.add_argument("--participants", default="candidate,off",
                    help="external campaigns only: comma-separated off,candidate; default runs both")
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--shard-count", type=int, default=CONFIG["three_plus_two"]["shard_count"])
    return ap


def main() -> int:
    args = parser().parse_args()
    if args.phase == "prepare":
        prepare_books()
        built = {}
        for network in CONFIG["networks"]:
            built[network] = {}
            built[network]["off"] = str(build(network, "off", ROOT, "exp"))
            built[network]["main_off"] = str(build(network, "off", MAIN, "main"))
        bots = setup_bots()
        print(json.dumps({"executables": built, "bots": {k: {n: str(v) for n, v in d.items()} for k, d in bots.items()}}, indent=2))
        return 0
    if not args.network or not args.variant:
        raise SystemExit("--network and --variant are required")
    if args.phase == "h2h":
        print(json.dumps(h2h(args.network, args.variant), indent=2))
        return 0
    if args.phase == "external":
        if not args.opponent:
            raise SystemExit("--opponent is required for external screening")
        result = run_external_campaign(args.network, args.variant, args.opponent,
            participants=normalize_participants(args.participants))
    elif args.phase == "h2h-3plus2":
        result = run_h2h_3plus2(args.network, args.variant, args.shard_index, args.shard_count)
    else:
        if not args.opponent:
            raise SystemExit("--opponent is required for 3+2 external testing")
        result = run_external_campaign(args.network, args.variant, args.opponent,
            three_plus_two=True, shard_index=args.shard_index, shard_count=args.shard_count,
            participants=normalize_participants(args.participants))
    print(json.dumps(result, indent=2))
    return 0


def setup_bots() -> dict:
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError("GPU setup blocked: PyTorch reports no CUDA device")
    bots_root = Path(CONFIG["main_root"]).resolve()
    from tools.external.bot_setup import ensure_bot
    bots = {name: ensure_bot(name, root=bots_root, build=True)
            for name in ("claustrophobia", "titanium")}
    # Main's legacy bridge accepts history only. The experimental client sends
    # per-search clocks, so compile the matching bridge outside the shared bot.
    from tools.build_claustrophobia_dynamic_bridge import ensure_bridge
    bots["claustrophobia"]["benchmark_bridge"] = ensure_bridge(bots_root)
    from tools.build_titanium_clock_bridge import ensure_bridge as ensure_titanium_clock_bridge
    bots["titanium"]["executable"] = ensure_titanium_clock_bridge(bots_root)
    bots["titanium"]["clock_bridge_build"] = Path(bots["titanium"]["executable"]).parents[3] / "build.json"
    bots["claustrophobia"]["clock_bridge_build"] = Path(bots["claustrophobia"]["benchmark_bridge"]).parent / "build.json"
    for player_type in (local_arena.TitaniumPlayer, local_arena.ClaustrophobiaPlayer):
        if not callable(getattr(player_type, "bestmove_clock", None)):
            raise RuntimeError(f"{player_type.__name__} lacks native-clock support")
    titanium_build = json.loads(Path(bots["titanium"]["clock_bridge_build"]).read_text(encoding="utf-8"))
    if "inc" not in titanium_build.get("protocol", ""):
        raise RuntimeError("Titanium bridge build does not expose increment-aware native clock support")
    claustro_build = json.loads(Path(bots["claustrophobia"]["clock_bridge_build"]).read_text(encoding="utf-8"))
    if claustro_build.get("identity", {}).get("protocol") != CONFIG["claustrophobia_bridge_protocol"]:
        raise RuntimeError("Claustrophobia bridge build does not match the deadline-aware clock protocol")
    for name, info in bots.items():
        for key, path in info.items():
            if key != "checkout" and not Path(path).is_file():
                raise FileNotFoundError(f"{name} artifact is missing: {path}")
    return bots


if __name__ == "__main__":
    raise SystemExit(main())
