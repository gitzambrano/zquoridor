#!/usr/bin/env python3
"""Run complete promotion benchmark battery for a candidate against main and external bots.

Evaluates candidate against:
1. Main champion (head-to-head on normal and center-rush books)
2. Claustrophobia (normal and center-rush books)
3. Titanium (normal and center-rush books)

Measures and compares win rate, Elo (95% bootstrap CI), search depth, and nodes per second (NPS).
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import match_finalists, run_benchmark
from tools.external import local_arena

CONFIG: Dict[str, Any] = {
    "candidate_exe": str(ROOT / "results" / "experiments" / "contact-bucketed512-central-20261003" / "zquoridor.exe"),
    "candidate_nnue": str(ROOT / "results" / "experiments" / "contact-bucketed512-central-20261003" / "student_int8.bin"),
    "candidate_name": "contact_bucketed512",
    "baseline_exe": str(ROOT / "results" / "experiments" / "frozen-production-20261002" / "zquoridor.exe"),
    "baseline_nnue": str(ROOT / "data" / "nnue" / "nnue_weights_int8.bin"),
    "baseline_name": "production_champion",
    "suite_name": "candidate-contact-bucketed512-full-battery",
    "move_time_ms": 200,
    "h2h_pairs": 50,
    "claustro_pairs": 50,
    "titanium_pairs": 50,
    "workers": 4,
    "claustro_workers": 2,
    "seed": 20261003,
    "normal_openings": str(ROOT / "tools" / "external" / "openings_screen_v1.jsonl"),
    "centerrush_openings": str(ROOT / "tools" / "external" / "openings_center_rush_sound_5k.jsonl"),
    "claustrophobia_device": "cpu",
    "bootstrap": 20000,
    "skip_main": False,
    "skip_claustro": False,
    "skip_titanium": False,
    "dry_run": False,
}

_DEPTH_RE = re.compile(r"depth\s+(\d+)")
_NODES_RE = re.compile(r"nodes\s+(\d+)")
_TIME_RE = re.compile(r"time\s+(\d+)")


def parse_move_metrics(timing: dict) -> Tuple[Optional[int], Optional[int], Optional[float]]:
    """Extract depth, nodes, and nps from move timing record."""
    search_str = str(timing.get("search_last") or "")
    m_depth = _DEPTH_RE.search(search_str)
    m_nodes = _NODES_RE.search(search_str)
    m_time = _TIME_RE.search(search_str)

    depth = int(m_depth.group(1)) if m_depth else None
    nodes = int(m_nodes.group(1)) if m_nodes else None
    elapsed_ms = float(m_time.group(1)) if m_time else float(timing.get("elapsed_ms") or 0.0)

    nps = None
    if nodes is not None and elapsed_ms > 0:
        nps = nodes / (elapsed_ms / 1000.0)
    return depth, nodes, nps


def compute_metrics_from_games(games_file: Path) -> Dict[str, Any]:
    """Aggregate depth and NPS metrics by player from games.jsonl."""
    if not games_file.is_file():
        return {}

    metrics: Dict[str, Dict[str, List[float]]] = {}
    with games_file.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                g = json.loads(line)
            except Exception:
                continue

            for m in g.get("move_times", []):
                player = m.get("player", "unknown")
                if player not in metrics:
                    metrics[player] = {"depths": [], "nodes": [], "nps": []}

                d, n, nps = parse_move_metrics(m)
                if d is not None:
                    metrics[player]["depths"].append(float(d))
                if n is not None:
                    metrics[player]["nodes"].append(float(n))
                if nps is not None:
                    metrics[player]["nps"].append(float(nps))

    summary: Dict[str, Any] = {}
    for player, vals in metrics.items():
        summary[player] = {
            "move_count": len(vals["depths"]),
            "mean_depth": float(sum(vals["depths"]) / len(vals["depths"])) if vals["depths"] else 0.0,
            "mean_nodes": float(sum(vals["nodes"]) / len(vals["nodes"])) if vals["nodes"] else 0.0,
            "mean_nps": float(sum(vals["nps"]) / len(vals["nps"])) if vals["nps"] else 0.0,
        }
    return summary


def run_h2h_subsuite(
    name: str,
    openings_path: str,
    pairs: int,
    out_dir: Path,
    config: dict,
) -> dict:
    """Run head-to-head match between candidate and baseline."""
    print(f"\n{'=' * 70}")
    print(f"RUNNING H2H: {config['candidate_name']} vs {config['baseline_name']} ({name.upper()})")
    print(f"Pairs: {pairs} (Total games: {pairs * 2}) | Clock: {config['move_time_ms']} ms/move")
    print(f"{'=' * 70}")

    h2h_cfg = {
        "engine1_name": config["candidate_name"],
        "engine1_executable": config["candidate_exe"],
        "engine1_nnue": config["candidate_nnue"],
        "engine1_args": "",
        "engine2_name": config["baseline_name"],
        "engine2_executable": config["baseline_exe"],
        "engine2_nnue": config["baseline_nnue"],
        "engine2_args": "",
        "pairs": pairs,
        "move_time_ms": config["move_time_ms"],
        "workers": config["workers"],
        "seed": config["seed"],
        "openings": openings_path,
        "output": str(out_dir),
        "bootstrap": config["bootstrap"],
    }
    report = match_finalists.run(h2h_cfg)
    metrics = compute_metrics_from_games(out_dir / "games.jsonl")
    report["search_metrics"] = metrics
    return report


def run_external_subsuite(
    opponent: str,
    name: str,
    openings_path: str,
    pairs: int,
    workers: int,
    out_dir: Path,
    config: dict,
) -> dict:
    """Run benchmark match against an external reference bot."""
    print(f"\n{'=' * 70}")
    print(f"RUNNING EXTERNAL: {config['candidate_name']} vs {opponent.upper()} ({name.upper()})")
    print(f"Pairs: {pairs} (Total games: {pairs * 2}) | Clock: {config['move_time_ms']} ms/move")
    print(f"{'=' * 70}")

    bench_cfg = {
        "opponents": [opponent],
        "pairs": pairs,
        "workers": workers,
        "seed": config["seed"],
        "openings": openings_path,
        "output": str(out_dir),
        "resume": True,
        "retry_failed": True,
        "auto_setup": True,
        "zq_executable": config["candidate_exe"],
        "nnue": config["candidate_nnue"],
        "zq_args": [],
        "zq_move_time_ms": config["move_time_ms"],
        "titanium_move_time_ms": config["move_time_ms"],
        "claustrophobia_move_time_ms": config["move_time_ms"],
        "claustrophobia_max_sims": 4096,
        "claustrophobia_cpuct": 1.5,
        "claustrophobia_device": config["claustrophobia_device"],
        "startup_timeout_s": 120.0,
        "move_timeout_s": 30.0,
        "max_plies": 180,
        "bootstrap": config["bootstrap"],
        "required_opening_categories": [],
        "category_score_threshold": 60.0,
        "dry_run": False,
    }
    report = run_benchmark.run(bench_cfg)
    metrics = compute_metrics_from_games(out_dir / "games.jsonl")
    report["search_metrics"] = metrics
    return report


def print_comparison_table(results: Dict[str, dict]) -> None:
    """Display comprehensive benchmark comparison table including depth and NPS."""
    print("\n" + "=" * 96)
    print("PROMOTION BENCHMARK SUMMARY TABLE (STRENGTH + SEARCH SPEED & DEPTH)")
    print("=" * 96)
    header = f"{'Sub-suite':<24} | {'Games':<5} | {'Score %':<7} | {'Elo [95% CI]':<18} | {'Cand Depth':<10} | {'Cand NPS':<9} | {'Opp Depth':<9} | {'Opp NPS':<8}"
    print(header)
    print("-" * len(header))

    for sub_name, rep in results.items():
        summary = rep.get("summary") or rep.get("summaries", {})
        if "summary" in rep:
            s = rep["summary"]
        elif isinstance(summary, dict) and summary:
            first_key = next(iter(summary.keys()))
            s = summary[first_key]
        else:
            s = rep

        games = s.get("included_games", s.get("recorded_games", 0))
        score_pct = s.get("score_pct", 0.0)
        elo = s.get("elo", 0.0)
        boot = s.get("paired_bootstrap_95", {})
        e_low = boot.get("elo_low", elo)
        e_high = boot.get("elo_high", elo)
        elo_str = f"{elo:+.1f} [{e_low:+.0f}, {e_high:+.0f}]"

        m = rep.get("search_metrics", {})
        cand_m = m.get("zquoridor") or m.get("candidate") or {}
        opp_keys = [k for k in m.keys() if k not in ("zquoridor", "candidate")]
        opp_m = m.get(opp_keys[0]) if opp_keys else {}

        cand_d = f"{cand_m.get('mean_depth', 0.0):.1f}" if cand_m else "-"
        cand_nps = f"{cand_m.get('mean_nps', 0.0):,.0f}" if cand_m else "-"
        opp_d = f"{opp_m.get('mean_depth', 0.0):.1f}" if opp_m else "-"
        opp_nps = f"{opp_m.get('mean_nps', 0.0):,.0f}" if opp_m else "-"

        print(f"{sub_name:<24} | {games:<5} | {score_pct:>6.1f}% | {elo_str:<18} | {cand_d:>10} | {cand_nps:>9} | {opp_d:>9} | {opp_nps:>8}")
    print("=" * 96)


def run(config: dict) -> dict:
    base_out = ROOT / "results" / "benchmarks" / config["suite_name"]
    base_out.mkdir(parents=True, exist_ok=True)

    all_results: Dict[str, dict] = {}

    # Phase 1: H2H vs Main (Normal Book)
    if not config["skip_main"] and config["h2h_pairs"] > 0:
        res = run_h2h_subsuite(
            name="Main Normal",
            openings_path=config["normal_openings"],
            pairs=config["h2h_pairs"],
            out_dir=base_out / "vs_main_normal",
            config=config,
        )
        all_results["vs_main_normal"] = res

    # Phase 2: H2H vs Main (Center Rush Book)
    if not config["skip_main"] and config["h2h_pairs"] > 0:
        res = run_h2h_subsuite(
            name="Main Center Rush",
            openings_path=config["centerrush_openings"],
            pairs=config["h2h_pairs"],
            out_dir=base_out / "vs_main_centerrush",
            config=config,
        )
        all_results["vs_main_centerrush"] = res

    # Phase 3: vs Claustrophobia (Normal Book)
    if not config["skip_claustro"] and config["claustro_pairs"] > 0:
        res = run_external_subsuite(
            opponent="claustrophobia",
            name="Claustro Normal",
            openings_path=config["normal_openings"],
            pairs=config["claustro_pairs"],
            workers=config["claustro_workers"],
            out_dir=base_out / "vs_claustro_normal",
            config=config,
        )
        all_results["vs_claustro_normal"] = res

    # Phase 4: vs Claustrophobia (Center Rush Book)
    if not config["skip_claustro"] and config["claustro_pairs"] > 0:
        res = run_external_subsuite(
            opponent="claustrophobia",
            name="Claustro Center Rush",
            openings_path=config["centerrush_openings"],
            pairs=config["claustro_pairs"],
            workers=config["claustro_workers"],
            out_dir=base_out / "vs_claustro_centerrush",
            config=config,
        )
        all_results["vs_claustro_centerrush"] = res

    # Phase 5: vs Titanium (Normal Book)
    if not config["skip_titanium"] and config["titanium_pairs"] > 0:
        res = run_external_subsuite(
            opponent="titanium",
            name="Titanium Normal",
            openings_path=config["normal_openings"],
            pairs=config["titanium_pairs"],
            workers=config["workers"],
            out_dir=base_out / "vs_titanium_normal",
            config=config,
        )
        all_results["vs_titanium_normal"] = res

    # Phase 6: vs Titanium (Center Rush Book)
    if not config["skip_titanium"] and config["titanium_pairs"] > 0:
        res = run_external_subsuite(
            opponent="titanium",
            name="Titanium Center Rush",
            openings_path=config["centerrush_openings"],
            pairs=config["titanium_pairs"],
            workers=config["workers"],
            out_dir=base_out / "vs_titanium_centerrush",
            config=config,
        )
        all_results["vs_titanium_centerrush"] = res

    print_comparison_table(all_results)
    out_summary = base_out / "promotion_summary.json"
    out_summary.write_text(json.dumps(all_results, indent=2, default=str), encoding="utf-8")
    print(f"\nPromotion battery summary written to {out_summary}")
    return all_results


def parse_config(argv=None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    for key, value in CONFIG.items():
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            parser.add_argument(flag, action=argparse.BooleanOptionalAction, default=argparse.SUPPRESS)
        elif isinstance(value, int):
            parser.add_argument(flag, type=int, default=argparse.SUPPRESS)
        else:
            parser.add_argument(flag, type=type(value), default=argparse.SUPPRESS)
    config = copy.deepcopy(CONFIG)
    config.update(vars(parser.parse_args(argv)))
    return config


def main(argv=None) -> int:
    config = parse_config(argv)
    if config["dry_run"]:
        print(json.dumps(config, indent=2))
        return 0
    run(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
