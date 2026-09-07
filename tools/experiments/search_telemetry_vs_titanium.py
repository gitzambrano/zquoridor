#!/usr/bin/env python3
"""Same-position telemetry: Zquoridor pure alpha-beta vs Titanium."""
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
sys.path.insert(0, str(ROOT / "tools" / "external"))
import titanium_arena_fixed  # noqa: F401,E402 - patches UCIEngine
import titanium_arena as arena  # noqa: E402

os.environ["TITANIUM_PONDERING"] = "0"
# Actual adapter format: "info depth D nodes N time T string ..."
ZQ_INFO_RE = re.compile(r"^info depth\s+(\d+)\s+nodes\s+(\d+)\s+time\s+(\d+)")


def mean(xs: Iterable[float]) -> float:
    xs = list(xs)
    return statistics.fmean(xs) if xs else 0.0


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


def parse_zq(lines: Sequence[str]) -> Dict[str, object]:
    for line in reversed(lines):
        m = ZQ_INFO_RE.match(line)
        if m:
            depth, nodes, elapsed = map(int, m.groups())
            return {
                "depth": depth,
                "nodes": nodes,
                "elapsed_ms": elapsed,
                "nps": nodes * 1000.0 / max(1, elapsed),
                "raw": line,
            }
    raise RuntimeError(f"no ZQ telemetry line found: {list(lines)[-5:]}")


def parse_ti(lines: Sequence[str]) -> Dict[str, object]:
    for line in reversed(lines):
        if line.startswith("info json "):
            try:
                return json.loads(line[len("info json "):])
            except json.JSONDecodeError:
                pass
    raise RuntimeError(f"no Titanium info json found: {list(lines)[-5:]}")


def reset_zq(e) -> None:
    e._send("ucinewgame")
    e._send("isready")
    e._wait_for("readyok", 20.0)


def reset_ti(e) -> None:
    e._send("reset")
    e._wait_for("ready", 20.0)


def generate_games(openings, zq_cmd, ti_cmd, movetime, max_plies):
    games = []
    for i, op in enumerate(openings):
        for color in (0, 1):
            g = arena.play_game(i, op, color, zq_cmd, ti_cmd, movetime, max_plies)
            games.append(g)
            print(
                f"game {len(games):02d}: op={i} zq=P{color} result={g.result:.1f} "
                f"plies={g.plies} first_zero={g.first_zero}", flush=True
            )
    return games


def choose_positions(games, cap):
    buckets = defaultdict(list)
    seen = set()
    for g in games:
        moves = list(g.moves)
        for n in range(6, 121, 6):
            if n >= len(moves):
                break
            h = tuple(moves[:n])
            if h in seen:
                continue
            seen.add(h)
            buckets[phase(h)].append(list(h))
    order = ["wall-rich", "mid-wall", "low-wall", "one-empty", "both-empty"]
    out = []
    while len(out) < cap:
        progress = False
        for p in order:
            if buckets[p] and len(out) < cap:
                out.append(buckets[p].pop(0))
                progress = True
        if not progress:
            break
    return out


def ti_depth(t: dict) -> int:
    return int(t.get("mainCompletedDepth", t.get("searchDepth", 0)) or 0)


def ti_nodes(t: dict) -> int:
    return int(t.get("totalNodes", t.get("nodes", 0)) or 0)


def summarize(rows):
    result = {"probes": len(rows), "by_budget": {}, "by_phase_at_200ms": {}}
    text = ["SAME-POSITION SEARCH TELEMETRY — ZQ PURE AB vs TITANIUM", ""]
    text.append("budget n agree% ZQdepth TIdepth gap(TI-ZQ) ZQnodes TInodes ZQnps TInps nodeRatio")
    for b in sorted({r["budget_ms"] for r in rows}):
        s = [r for r in rows if r["budget_ms"] == b]
        agree = 100 * mean(r["same_move"] for r in s)
        zd = mean(r["zq"]["depth"] for r in s)
        td = mean(ti_depth(r["ti"]) for r in s)
        zn = mean(r["zq"]["nodes"] for r in s)
        tn = mean(ti_nodes(r["ti"]) for r in s)
        znps = mean(r["zq"]["nps"] for r in s)
        tnps = mean(float(r["ti"].get("nps", 0) or 0) for r in s)
        d = {"positions": len(s), "move_agreement_pct": agree, "zq_depth_avg": zd,
             "ti_depth_avg": td, "depth_gap_ti_minus_zq": td-zd, "zq_nodes_avg": zn,
             "ti_nodes_avg": tn, "zq_nps_avg": znps, "ti_nps_avg": tnps,
             "node_ratio_ti_over_zq": tn/zn if zn else 0.0}
        result["by_budget"][str(b)] = d
        text.append(f"{b:6d} {len(s):2d} {agree:6.1f} {zd:7.2f} {td:7.2f} {td-zd:10.2f} "
                    f"{zn:7.0f} {tn:7.0f} {znps:7.0f} {tnps:7.0f} {d['node_ratio_ti_over_zq']:8.2f}")

    text += ["", "200ms by phase", "phase n agree% ZQdepth TIdepth gap ZQnodes TInodes"]
    s200 = [r for r in rows if r["budget_ms"] == 200]
    for p in ["wall-rich", "mid-wall", "low-wall", "one-empty", "both-empty"]:
        s = [r for r in s200 if r["phase"] == p]
        if not s:
            continue
        agree = 100 * mean(r["same_move"] for r in s)
        zd = mean(r["zq"]["depth"] for r in s)
        td = mean(ti_depth(r["ti"]) for r in s)
        zn = mean(r["zq"]["nodes"] for r in s)
        tn = mean(ti_nodes(r["ti"]) for r in s)
        result["by_phase_at_200ms"][p] = {"positions": len(s), "move_agreement_pct": agree,
            "zq_depth_avg": zd, "ti_depth_avg": td, "depth_gap_ti_minus_zq": td-zd,
            "zq_nodes_avg": zn, "ti_nodes_avg": tn}
        text.append(f"{p:<11} {len(s):2d} {agree:6.1f} {zd:7.2f} {td:7.2f} {td-zd:5.2f} {zn:7.0f} {tn:7.0f}")

    worst = sorted(s200, key=lambda r: ti_depth(r["ti"])-r["zq"]["depth"], reverse=True)[:12]
    text += ["", "largest 200ms completed-depth advantages"]
    for r in worst:
        gap = ti_depth(r["ti"]) - r["zq"]["depth"]
        text.append(f"gap={gap:+d} phase={r['phase']:<10} walls={r['walls']} "
                    f"ZQd={r['zq']['depth']} TId={ti_depth(r['ti'])} ZQ={r['zq_move']} TI={r['ti_move']} "
                    f"history={' '.join(r['history'])}")
    return result, "\n".join(text) + "\n"


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
    a = ap.parse_args()

    openings = [json.loads(x)["moves"] for x in Path(a.openings).read_text().splitlines() if x.strip()][:a.game_openings]
    zq_cmd = [a.zq, "--nnue", a.weights, "--no-mcab"]
    ti_cmd = [a.titanium, "uci", "--threads", "1"]
    print("Generating real pure-AB-vs-Titanium histories", flush=True)
    games = generate_games(openings, zq_cmd, ti_cmd, a.game_movetime, a.game_max_plies)
    positions = choose_positions(games, a.max_positions)
    mix = defaultdict(int)
    for h in positions:
        mix[phase(h)] += 1
    print(f"selected {len(positions)} positions; phase mix={dict(mix)}", flush=True)

    zq = arena.UCIEngine(zq_cmd, "zquoridor")
    ti = arena.UCIEngine(ti_cmd, "titanium")
    rows = []
    try:
        for pi, h in enumerate(positions):
            for budget in a.budgets:
                reset_zq(zq)
                zm, zsec, zlines = zq.bestmove(h, budget)
                reset_ti(ti)
                tm, tsec, tlines = ti.bestmove(h, budget)
                z, t = parse_zq(zlines), parse_ti(tlines)
                row = {"position_index": pi, "budget_ms": budget, "plies": len(h),
                    "phase": phase(h), "walls": list(wall_counts(h)), "history": h,
                    "zq_move": zm, "ti_move": tm, "same_move": zm == tm,
                    "zq_wall_clock_ms": zsec*1000, "ti_wall_clock_ms": tsec*1000,
                    "zq": z, "ti": t}
                rows.append(row)
                print(f"probe {len(rows):03d}: pos={pi:02d} {row['phase']:<10} {budget:4d}ms "
                      f"ZQ d{z['depth']} n{z['nodes']} {zm} | TI d{ti_depth(t)} n{ti_nodes(t)} {tm}", flush=True)
    finally:
        zq.close(); ti.close()

    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    with (out / "search-telemetry.jsonl").open("w") as f:
        for r in rows: f.write(json.dumps(r, separators=(",", ":")) + "\n")
    summary, report = summarize(rows)
    summary["generated_games"] = [{"opening_index": g.opening_index, "zq_player": g.zq_player,
        "result": g.result, "plies": g.plies, "first_zero": g.first_zero, "moves": g.moves} for g in games]
    (out / "search-telemetry-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (out / "search-telemetry-summary.txt").write_text(report)
    print("\n" + report, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
