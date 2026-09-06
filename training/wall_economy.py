#!/usr/bin/env python3
"""Wall-resource diagnostics and hard-example mining for V3 self-play data.

The key transition identity comes from mover-canonical samples. For sample t,
t+1 is the opponent's perspective after the played move. Therefore, inside one
game:

    mover_dist_after = next.opp_dist
    opponent_dist_after = next.own_dist

For a wall played at t we define immediate wall efficiency as

    (opponent_dist_after - opponent_dist_before)
      - (mover_dist_after - mover_dist_before)

Positive values lengthen the opponent more than ourselves. This is only a
local diagnostic, not a hand-written evaluation term.

The script also builds a focused replay from COMPLETE games that contain
resource-critical decisions. Training still uses the recorded MCAB visit
policy; no heuristic label is invented here.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

from read_selfplay import SAMPLE_DTYPE, game_start_mask

PAWN_OUT = 81
POLICY_OUT = 209
PROB_SCALE = 65535.0


def expand_inputs(items: list[str]) -> list[Path]:
    out: list[Path] = []
    for item in items:
        p = Path(item)
        if p.is_dir():
            out.extend(sorted(p.glob("*.bin")))
        elif any(ch in item for ch in "*?["):
            out.extend(Path(x) for x in sorted(glob.glob(item)))
        elif p.is_file():
            out.append(p)
    seen = set()
    uniq = []
    for p in out:
        rp = str(p.resolve())
        if rp not in seen:
            seen.add(rp)
            uniq.append(p)
    if not uniq:
        raise SystemExit("no .bin inputs found")
    return uniq


def load_v3(path: Path) -> np.ndarray:
    size = path.stat().st_size
    if size == 0 or size % SAMPLE_DTYPE.itemsize:
        raise SystemExit(f"{path}: expected V3 {SAMPLE_DTYPE.itemsize}-byte records")
    return np.memmap(path, dtype=SAMPLE_DTYPE, mode="r")


def game_ranges(arr: np.ndarray):
    starts = np.flatnonzero(game_start_mask(arr))
    if len(starts) == 0:
        return
    # A shard written by selfplay starts at a game boundary. Refuse silently
    # concatenated/corrupt input because transition metrics would be false.
    if int(starts[0]) != 0:
        raise ValueError("shard does not begin at a detected game boundary")
    ends = np.append(starts[1:], len(arr))
    for a, b in zip(starts, ends):
        yield int(a), int(b)


def wall_visit_mass(game: np.ndarray) -> np.ndarray:
    idx = game["policy_top_idx"].astype(np.int32)
    prob = game["policy_top_prob"].astype(np.float64) / PROB_SCALE
    valid = (idx >= 0) & (idx < POLICY_OUT)
    return np.sum(prob * (valid & (idx >= PAWN_OUT)), axis=1)


def played_wall(game: np.ndarray) -> np.ndarray:
    return game["policy_target"].astype(np.int32) >= PAWN_OUT


def transition_wall_metrics(game: np.ndarray) -> dict[str, np.ndarray]:
    n = len(game)
    valid = np.zeros(n, dtype=bool)
    d_opp = np.zeros(n, dtype=np.int16)
    d_own = np.zeros(n, dtype=np.int16)
    net = np.zeros(n, dtype=np.int16)
    if n < 2:
        return {"valid": valid, "delta_opp": d_opp, "delta_own": d_own, "net": net}

    is_wall = played_wall(game)
    cur_ow = game[:-1]["walls_left_own"].astype(np.int16)
    cur_pw = game[:-1]["walls_left_opp"].astype(np.int16)
    nxt_ow = game[1:]["walls_left_own"].astype(np.int16)
    nxt_pw = game[1:]["walls_left_opp"].astype(np.int16)

    # Perspective swaps every ply. next.own == current opponent;
    # next.opp == current mover. Validate the resource transition explicitly.
    expected_next_own = cur_pw
    expected_next_opp = cur_ow - is_wall[:-1].astype(np.int16)
    edge_ok = (nxt_ow == expected_next_own) & (nxt_pw == expected_next_opp)
    valid[:-1] = edge_ok

    before_opp = game[:-1]["opp_dist"].astype(np.int16)
    before_own = game[:-1]["own_dist"].astype(np.int16)
    after_opp = game[1:]["own_dist"].astype(np.int16)
    after_own = game[1:]["opp_dist"].astype(np.int16)
    d_opp[:-1] = after_opp - before_opp
    d_own[:-1] = after_own - before_own
    net[:-1] = d_opp[:-1] - d_own[:-1]
    return {"valid": valid, "delta_opp": d_opp, "delta_own": d_own, "net": net}


def critical_mask(game: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Positions worth replaying, without inventing a new policy label."""
    own_w = game["walls_left_own"].astype(np.int16)
    opp_w = game["walls_left_opp"].astype(np.int16)
    own_d = game["own_dist"].astype(np.int16)
    opp_d = game["opp_dist"].astype(np.int16)
    wmass = wall_visit_mass(game)
    is_wall = played_wall(game)
    tm = transition_wall_metrics(game)

    # Existing wandering family: ahead in the race but badly out-resourced.
    wall_poor_race = (
        (own_w <= 2)
        & ((opp_w - own_w) >= 4)
        & ((opp_d - own_d) >= 2)
        & (own_d > 0)
    )
    # Policy itself is deciding between pawn and wall: useful distillation data.
    wall_pawn_ambiguous = (wmass >= 0.15) & (wmass <= 0.85)
    # The policy still spends substantial probability on walls when nearly empty.
    low_resource_wall_interest = (own_w <= 3) & (wmass >= 0.10)
    # Observed wall that locally buys no net path-distance advantage. This is a
    # mining signal only; the target remains the search visit distribution.
    inefficient_wall = is_wall & tm["valid"] & (tm["net"] <= 0)
    depletion = is_wall & (own_w <= 2)

    mask = wall_poor_race | wall_pawn_ambiguous | low_resource_wall_interest | inefficient_wall | depletion
    detail = {
        **tm,
        "wall_mass": wmass,
        "played_wall": is_wall,
        "wall_poor_race": wall_poor_race,
        "ambiguous": wall_pawn_ambiguous,
        "low_resource_interest": low_resource_wall_interest,
        "inefficient_wall": inefficient_wall,
        "depletion": depletion,
    }
    return mask, detail


def _pct(x: int, n: int) -> float:
    return 100.0 * x / n if n else 0.0


def analyze(paths: list[Path]) -> tuple[dict, list[tuple[float, np.ndarray]]]:
    totals = {
        "games": 0, "positions": 0, "wall_moves": 0, "valid_wall_transitions": 0,
        "inefficient_walls": 0, "zero_gain_walls": 0, "critical_positions": 0,
        "wall_poor_race_positions": 0, "ambiguous_positions": 0,
        "low_resource_wall_interest_positions": 0, "depletion_walls": 0,
        "same_player_wall_chains": 0,
    }
    net_values: list[int] = []
    d_opp_values: list[int] = []
    d_own_values: list[int] = []
    wall_mass_values: list[float] = []
    milestones: dict[int, list[int]] = {5: [], 3: [], 1: [], 0: []}
    milestone_results: dict[int, list[int]] = {5: [], 3: [], 1: [], 0: []}
    walls_hist = {i: 0 for i in range(11)}
    games_for_focus: list[tuple[float, np.ndarray]] = []

    for path in paths:
        arr = load_v3(path)
        for a, b in game_ranges(arr):
            game = np.asarray(arr[a:b])
            totals["games"] += 1
            totals["positions"] += len(game)
            mask, d = critical_mask(game)
            is_wall = d["played_wall"]
            valid_wall = is_wall & d["valid"]
            totals["wall_moves"] += int(is_wall.sum())
            totals["valid_wall_transitions"] += int(valid_wall.sum())
            totals["critical_positions"] += int(mask.sum())
            totals["inefficient_walls"] += int((valid_wall & (d["net"] <= 0)).sum())
            totals["zero_gain_walls"] += int((valid_wall & (d["delta_opp"] <= 0)).sum())
            totals["wall_poor_race_positions"] += int(d["wall_poor_race"].sum())
            totals["ambiguous_positions"] += int(d["ambiguous"].sum())
            totals["low_resource_wall_interest_positions"] += int(d["low_resource_interest"].sum())
            totals["depletion_walls"] += int(d["depletion"].sum())

            vw = np.flatnonzero(valid_wall)
            net_values.extend(int(d["net"][i]) for i in vw)
            d_opp_values.extend(int(d["delta_opp"][i]) for i in vw)
            d_own_values.extend(int(d["delta_own"][i]) for i in vw)
            wall_mass_values.extend(float(x) for x in d["wall_mass"])
            for w in range(11):
                walls_hist[w] += int((game["walls_left_own"] == w).sum())

            # Same physical player acts every two plies. Count repeated wall usage.
            if len(game) >= 3:
                totals["same_player_wall_chains"] += int(np.sum(is_wall[:-2] & is_wall[2:]))

            own_w = game["walls_left_own"].astype(np.int16)
            for i in np.flatnonzero(is_wall):
                after = int(own_w[i]) - 1
                if after in milestones:
                    milestones[after].append(int(i))
                    milestone_results[after].append(int(game[i]["game_result"]))

            # Rank focus games: reward critical density and the most dangerous
            # resource patterns, not random selection.
            score = (
                float(mask.sum())
                + 3.0 * float(d["inefficient_wall"].sum())
                + 4.0 * float(d["depletion"].sum())
                + 2.0 * float(d["wall_poor_race"].sum())
            )
            if score > 0:
                games_for_focus.append((score, game.copy()))

    nv = np.asarray(net_values, dtype=np.float64)
    do = np.asarray(d_opp_values, dtype=np.float64)
    dy = np.asarray(d_own_values, dtype=np.float64)
    wm = np.asarray(wall_mass_values, dtype=np.float64)

    def stats(a: np.ndarray) -> dict:
        if len(a) == 0:
            return {"n": 0, "mean": None, "median": None, "p10": None, "p90": None}
        return {
            "n": int(len(a)), "mean": float(a.mean()), "median": float(np.median(a)),
            "p10": float(np.quantile(a, 0.10)), "p90": float(np.quantile(a, 0.90)),
        }

    result = dict(totals)
    result.update({
        "wall_move_fraction_pct": _pct(totals["wall_moves"], totals["positions"]),
        "inefficient_wall_fraction_pct": _pct(totals["inefficient_walls"], totals["valid_wall_transitions"]),
        "zero_opponent_gain_wall_fraction_pct": _pct(totals["zero_gain_walls"], totals["valid_wall_transitions"]),
        "wall_efficiency": stats(nv),
        "wall_delta_opponent_distance": stats(do),
        "wall_delta_own_distance": stats(dy),
        "wall_visit_mass": stats(wm),
        "positions_by_own_walls": walls_hist,
        "milestones": {},
    })
    for w in (5, 3, 1, 0):
        plies = np.asarray(milestones[w], dtype=np.float64)
        res = np.asarray(milestone_results[w], dtype=np.float64)
        result["milestones"][str(w)] = {
            "events": int(len(plies)),
            "mean_ply": float(plies.mean()) if len(plies) else None,
            "median_ply": float(np.median(plies)) if len(plies) else None,
            "eventual_win_pct": float(100.0 * np.mean(res > 0)) if len(res) else None,
        }
    return result, games_for_focus


def write_focus(games: list[tuple[float, np.ndarray]], output: Path, copies: int, max_positions: int) -> dict:
    games = sorted(games, key=lambda x: x[0], reverse=True)
    kept: list[np.ndarray] = []
    n = 0
    for _, game in games:
        if kept and n + len(game) > max_positions:
            continue
        kept.append(game)
        n += len(game)
        if n >= max_positions:
            break
    if not kept:
        raise SystemExit("no critical games selected for focus replay")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as f:
        for _ in range(max(1, copies)):
            for game in kept:
                game.tofile(f)
    return {
        "focus_games": len(kept),
        "positions_per_copy": n,
        "copies": max(1, copies),
        "output_positions": n * max(1, copies),
        "path": str(output),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("inputs", nargs="+", help="V3 .bin files, directories, or globs")
    p.add_argument("--json", dest="json_path")
    p.add_argument("--focus-out")
    p.add_argument("--focus-copies", type=int, default=2)
    p.add_argument("--focus-max-fraction", type=float, default=0.12,
                   help="maximum selected positions per copy as fraction of analyzed positions")
    args = p.parse_args()
    if not (0 < args.focus_max_fraction <= 1):
        raise SystemExit("--focus-max-fraction must be in (0,1]")
    paths = expand_inputs(args.inputs)
    result, focus_games = analyze(paths)
    if args.focus_out:
        cap = max(1, int(result["positions"] * args.focus_max_fraction))
        result["focus_replay"] = write_focus(
            focus_games, Path(args.focus_out), args.focus_copies, cap
        )
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.json_path:
        out = Path(args.json_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
