#!/usr/bin/env python3
"""Same-position search telemetry: Zquoridor pure AB vs pinned Titanium.

The purpose is diagnostic, not Elo estimation. We first generate real game
histories with Zquoridor pure alpha-beta against Titanium, then probe both
engines from the exact same positions at several fixed move times. Search state
is reset before every probe so TT/history carry-over cannot contaminate the
comparison.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
EXTERNAL = ROOT / "tools" / "external"
sys.path.insert(0, str(EXTERNAL))

# Importing this module patches titanium_arena.UCIEngine so Titanium uses its
# native session protocol while Zquoridor keeps conventional UCI.
import titanium_arena_fixed  # noqa: F401,E402
import titanium_arena as arena  # noqa: E402

os.environ["TITANIUM_PONDERING"] = "0"

ZQ_INFO_RE = re.compile(r"^info depth\s+(\d+)\s+score\s+cp\s+(-?\d+)\s+nodes\s+(\d+)\s+time\s+(\d+)")


def wall_counts(history: Sequence[str]) -> Tuple[int, int]:
    left = [10, 10]
    for ply, mv in enumerate(history):
        if arena.is_wall(mv):
            left[ply & 1] -= 1
    return left[0], left[1]


def phase(history: Sequence[str]) -> str:
    w0, w1 = wall_counts(history)
    lo = min(w0, w1)
    if w0 == 0 and w1 == 0:
        return "both-empty"
    if lo == 0:
        return "one-empty"
    if lo <= 2:
        return "low-wall"
    if lo <= 5:
        return "mid-wall"
    return "wall-rich"


def parse_zq_info(lines: Sequence[str]) -> Dict[str, object]:
    for line in reversed(lines):
        m = ZQ_INFO_RE.match(line)
        if not m:
            continue
        depth, score, nodes, elapsed = map(int, m.groups())
        nps = nodes * 1000.0 / max(1, elapsed)
        return {
            "depth": depth,
            "score": score,
            "nodes": nodes,
            "elapsed_ms": elapsed,
            "nps": nps,
            "raw": line,
        }
    return {"depth": 0, "score": 0, "nodes": 0, "elapsed_ms": 0, "nps": 0.0, "raw": ""}


def parse_ti_info(lines: Sequence[str]) -> Dict[str, object]:
    for line in reversed(lines):
        if not line.startswith("info json "):
            continue
        try:
            d = json.loads(line[len("info json "):])
        except json.JSONDecodeError:
            continue
        d["raw"] = line
        return d
    return {}


def reset_zq(e) -> None:
    e._send("ucinewgame")
    e._send("isready")
    e._wait_for("readyok", 20.0)


def reset_ti(e) -> None:
    e._send("reset")
    e._wait_for("ready", 20.0)


def avg(xs: Iterable[float]) -> float:
    vals = list(xs)
    return statistics.fmean(vals) if vals else 0.0


def median(xs: Iterable[float]) -> float:
    vals = list(xs)
    return statistics.median(vals) if vals else 0.0


def choose_positions(games: Sequence[arena.GameResult], max_positions: int) -> List[List[str]]:
    # Sample every six plies from real games, excluding the terminal move. Then
    # round-robin phases so wall-rich opening positions do not drown out the
    # low-wall/endgame cases we specifically need to diagnose.
    buckets: Dict[str, List[List[str]]] = defaultdict(list)
    seen = set()
    targets = list(range(6, 121, 6))
    for g in games:
        moves = list(g.moves)
        for n in targets:
            if n >= len(moves):
                break
            hist = tuple(moves[:n])
            if hist in seen:
                continue
            seen.add(hist)
            buckets[phase(hist)].append(list(hist))

    order = ["wall-rich", "mid-wall", "low-wall", "one-empty", "both-empty"]
    selected: List[List[str]] = []
    while len(selected) < max_positions:
        progress = False
        for p in order:
            if buckets[p] and len(selected) < max_positions:
                selected.append(buckets[p].pop(0))
                progress = True
        if not progress:
            break
    return selected


def generate_games(openings: Sequence[Sequence[str]], zq_cmd: Sequence[str], ti_cmd: Sequence[str], movetime: int, max_plies: int) -> List[arena.GameResult]:
    out: List[arena.GameResult] = []
    for i, op in enumerate(openings):
        for zq_player in (0, 1):
            g = arena.play_game(i, op, zq_player, zq_cmd, ti_cmd, movetime, max_plies)
            out.append(g)
            print(
                f"game {len(out):2d}: opening={i} zq=P{zq_player} result={g.result:.1f} "
                f"plies={g.plies} first_zero={g.first_zero}",
                flush=True,
            )
    return out


def summarize(rows: Sequence[dict]) -> Tuple[dict, str]:
    summary: dict = {"probes": len(rows), "by_budget": {}, "by_phase_at_200ms": {}}
    lines = []
    lines.append("SAME-POSITION SEARCH TELEMETRY — ZQ PURE AB vs TITANIUM")
    lines.append("")
    lines.append(f"probes: {len(rows)}")
    lines.append("")
    lines.append("budget  positions  agree%  ZQdepth  TIdepth  depthGap  ZQnodes    TInodes   ZQnps      TInps     nodeRatio")

    for budget in sorted({int(r["budget_ms"]) for r in rows}):
        sub = [r for r in rows if r["budget_ms"] == budget]
        agrees = avg(1.0 if r["zq_move"] == r["ti_move"] else 0.0 for r in sub) * 100.0
        zqd = avg(r["zq"]["depth"] for r in sub)
        tid = avg(r["ti"].get("mainCompletedDepth", r["ti"].get("searchDepth", 0)) for r in sub)
        gap = tid - zqd
        zqn = avg(r["zq"]["nodes"] for r in sub)
        tin = avg(r["ti"].get("totalNodes", r["ti"].get("nodes", 0)) for r in sub)
        zqnps = avg(r["zq"]["nps"] for r in sub)
        tinps = avg(r["ti"].get("nps", 0) for r in sub)
        ratio = tin / zqn if zqn else 0.0
        summary["by_budget"][str(budget)] = {
            "positions": len(sub), "move_agreement_pct": agrees,
            "zq_depth_avg": zqd, "ti_depth_avg": tid, "depth_gap_ti_minus_zq": gap,
            "zq_nodes_avg": zqn, "ti_nodes_avg": tin,
            "zq_nps_avg": zqnps, "ti_nps_avg": tinps, "node_ratio_ti_over_zq": ratio,
        }
        lines.append(
            f"{budget:6d} {len(sub):10d} {agrees:7.1f} {zqd:8.2f} {tid:8.2f} {gap:9.2f} "
            f"{zqn:9.0f} {tin:10.0f} {zqnps:10.0f} {tinps:10.0f} {ratio:10.2f}"
        )

    lines.append("")
    lines.append("200ms by phase")
    lines.append("phase        n  agree%  ZQdepth  TIdepth  depthGap  ZQnodes    TInodes")
    sub200 = [r for r in rows if r["budget_ms"] == 200]
    for p in ["wall-rich", "mid-wall", "low-wall", "one-empty", "both-empty"]:
        sub = [r for r in sub200 if r["phase"] == p]
        if not sub:
            continue
        agrees = avg(1.0 if r["zq_move"] == r["ti_move"] else 0.0 for r in sub) * 100.0
        zqd = avg(r["zq"]["depth"] for r in sub)
        tid = avg(r["ti"].get("mainCompletedDepth", r["ti"].get("searchDepth", 0)) for r in sub)
        gap = tid - zqd
        zqn = avg(r["zq"]["nodes"] for r in sub)
        tin = avg(r["ti"].get("totalNodes", r["ti"].get("nodes", 0)) for r in sub)
        summary["by_phase_at_200ms"][p] = {
            "positions": len(sub), "move_agreement_pct": agrees,
            "zq_depth_avg": zqd, "ti_depth_avg": tid, "depth_gap_ti_minus_zq": gap,
            "zq_nodes_avg": zqn, "ti_nodes_avg": tin,
        }
        lines.append(f"{p:<12} {len(sub):2d} {agrees:7.1f} {zqd:8.2f} {tid:8.2f} {gap:9.2f} {zqn:9.0f} {tin:10.0f}")

    # Show the exact positions with the largest 200ms completed-depth deficit.
    ranked = sorted(
        sub200,
        key=lambda r: r["ti"].get("mainCompletedDepth", r["ti"].get("searchDepth", 0)) - r["zq"]["depth"],
        reverse=True,
    )[:12]
    lines.append("")
    lines.append("largest 200ms Titanium completed-depth advantages")
    for r in ranked:
        tid = r["ti"].get("mainCompletedDepth", r["ti"].get("searchDepth", 0))
        lines.append(
            f"gap={tid-r['zq']['depth']:+3.0f} phase={r['phase']:<10} walls={r['walls']} "
            f"ZQd={r['zq']['depth']} TId={tid} ZQ={r['zq_move']} TI={r['ti_move']} "
            f"history={' '.join(r['history'])}"
        )

    return summary, "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zq", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--titanium", required=True)
    ap.add_argument("--openings", required=True)
    ap.add_argument("--game-openings", type=int, default=6)
    ap.add_argument("--game-movetime", type=int, default=200)
    ap.add_argument("--game-max-plies", type=int, default=100)
    ap.add_argument("--max-positions", type=int, default=48)
    ap.add_argument("--budgets", nargs="+", type=int, default=[50, 200, 800])
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()

    openings = [json.loads(x)["moves"] for x in Path(args.openings).read_text().splitlines() if x.strip()]
    openings = openings[: args.game_openings]
    zq_cmd = [args.zq, "--nnue", args.weights, "--no-mcab"]
    ti_cmd = [args.titanium, "uci", "--threads", "1"]

    print("generating real pure-AB-vs-Titanium histories...", flush=True)
    games = generate_games(openings, zq_cmd, ti_cmd, args.game_movetime, args.game_max_plies)
    positions = choose_positions(games, args.max_positions)
    print(f"selected {len(positions)} unique positions", flush=True)
    phase_counts = defaultdict(int)
    for h in positions:
        phase_counts[phase(h)] += 1
    print("phase mix:", dict(phase_counts), flush=True)

    zq = arena.UCIEngine(zq_cmd, "zquoridor")
    ti = arena.UCIEngine(ti_cmd, "titanium")
    rows: List[dict] = []
    try:
        for pi, hist in enumerate(positions):
            w = wall_counts(hist)
            p = phase(hist)
            for budget in args.budgets:
                reset_zq(zq)
                zq_move, zq_wall_s, zq_info = zq.bestmove(hist, budget)
                reset_ti(ti)
                ti_move, ti_wall_s, ti_info = ti.bestmove(hist, budget)
                zi = parse_zq_info(zq_info)
                tj = parse_ti_info(ti_info)
                row = {
                    "position_index": pi,
                    "budget_ms": budget,
                    "plies": len(hist),
                    "phase": p,
                    "walls": list(w),
                    "history": hist,
                    "zq_move": zq_move,
                    "ti_move": ti_move,
                    "same_move": zq_move == ti_move,
                    "zq_wall_clock_ms": zq_wall_s * 1000.0,
                    "ti_wall_clock_ms": ti_wall_s * 1000.0,
                    "zq": zi,
                    "ti": tj,
                }
                rows.append(row)
                tid = tj.get("mainCompletedDepth", tj.get("searchDepth", 0))
                print(
                    f"probe {len(rows):3d}: pos={pi:02d} {p:<10} {budget:4d}ms "
                    f"ZQ d{zi['depth']} n{zi['nodes']} {zq_move} | "
                    f"TI d{tid} n{tj.get('totalNodes', tj.get('nodes', 0))} {ti_move}",
                    flush=True,
                )
    finally:
        zq.close()
        ti.close()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "search-telemetry.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r, separators=(",", ":")) + "\n")
    summary, text = summarize(rows)
    summary["generated_games"] = [
        {"opening_index": g.opening_index, "zq_player": g.zq_player, "result": g.result, "plies": g.plies,
         "first_zero": g.first_zero, "moves": g.moves}
        for g in games
    ]
    (out / "search-telemetry-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (out / "search-telemetry-summary.txt").write_text(text)
    print("\n" + text, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
