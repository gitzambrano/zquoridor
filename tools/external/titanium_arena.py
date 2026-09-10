#!/usr/bin/env python3
"""Paired external arena: Zquoridor UCI adapter vs Titanium UCI.

The referee deliberately knows only enough Quoridor semantics to determine a
winner and resource/tempo diagnostics. Both engines receive the full move
history and independently validate legality.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import math
import os
import selectors
import subprocess
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Sequence, Tuple


class UCIEngine:
    def __init__(self, argv: Sequence[str], name: str):
        self.argv = list(argv)
        self.name = name
        self.p = subprocess.Popen(
            self.argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        assert self.p.stdin and self.p.stdout
        self.sel = selectors.DefaultSelector()
        self.sel.register(self.p.stdout, selectors.EVENT_READ)
        self._send("uci")
        self._wait_for("uciok", 20.0)
        self._send("isready")
        self._wait_for("readyok", 20.0)
        self._send("ucinewgame")

    def _send(self, line: str) -> None:
        assert self.p.stdin
        self.p.stdin.write(line + "\n")
        self.p.stdin.flush()

    def _readline_timeout(self, timeout: float) -> str:
        events = self.sel.select(timeout)
        if not events:
            raise TimeoutError(f"{self.name}: timeout waiting for output")
        assert self.p.stdout
        line = self.p.stdout.readline()
        if line == "":
            err = ""
            if self.p.stderr:
                try:
                    err = self.p.stderr.read()
                except Exception:
                    pass
            raise RuntimeError(f"{self.name}: process exited rc={self.p.poll()} stderr={err[-2000:]}")
        return line.strip()

    def _wait_for(self, token: str, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            line = self._readline_timeout(max(0.01, deadline - time.monotonic()))
            if line == token or line.startswith(token + " "):
                return
        raise TimeoutError(f"{self.name}: did not return {token}")

    def bestmove(self, history: Sequence[str], movetime_ms: int) -> Tuple[str, float, List[str]]:
        pos = "position startpos"
        if history:
            pos += " moves " + " ".join(history)
        self._send(pos)
        t0 = time.monotonic()
        self._send(f"go movetime {movetime_ms}")
        info: List[str] = []
        timeout = max(10.0, movetime_ms / 1000.0 * 8.0 + 3.0)
        deadline = time.monotonic() + timeout
        while True:
            line = self._readline_timeout(max(0.01, deadline - time.monotonic()))
            if line.startswith("info "):
                info.append(line)
                if "error" in line.lower():
                    raise RuntimeError(f"{self.name}: {line}")
            elif line.startswith("bestmove "):
                mv = line.split()[1]
                return mv, time.monotonic() - t0, info

    def close(self) -> None:
        try:
            self._send("quit")
        except Exception:
            pass
        try:
            self.p.wait(timeout=1.0)
        except Exception:
            try:
                self.p.kill()
            except Exception:
                pass
        try:
            self.sel.close()
        except Exception:
            pass


@dataclass
class GameResult:
    opening_index: int
    zq_player: int
    result: float  # from Zquoridor perspective: 1, .5, 0
    winner: int  # 0/1, -1 draw
    plies: int
    zq_think_s: float
    titanium_think_s: float
    zq_walls_remaining: int
    titanium_walls_remaining: int
    zq_wall_moves: int
    zq_pawn_moves: int
    zq_backward_pawn_moves: int
    zq_lateral_pawn_moves: int
    zq_emptyhand_backward_moves: int
    both_emptyhand_zq_pawn_moves: int
    first_zero: str
    termination: str
    moves: List[str]


def is_wall(mv: str) -> bool:
    return len(mv) == 3 and mv[-1] in "hv"


def pawn_rank(mv: str) -> int:
    return int(mv[1])


def syntax_ok(mv: str) -> bool:
    if len(mv) == 2:
        return mv[0] in "abcdefghi" and mv[1] in "123456789"
    if len(mv) == 3:
        return mv[0] in "abcdefgh" and mv[1] in "12345678" and mv[2] in "hv"
    return False


def reached_goal(player: int, mv: str) -> bool:
    if is_wall(mv) or len(mv) != 2:
        return False
    return (player == 0 and mv[1] == "9") or (player == 1 and mv[1] == "1")


def play_game(
    opening_index: int,
    opening: Sequence[str],
    zq_player: int,
    zq_cmd: Sequence[str],
    titanium_cmd: Sequence[str],
    movetime_ms: int,
    max_plies: int,
) -> GameResult:
    zq = UCIEngine(zq_cmd, "zquoridor")
    ti = UCIEngine(titanium_cmd, "titanium")
    history = list(opening)
    walls_left = [10, 10]
    pawn_ranks = [1, 9]
    for ply, mv in enumerate(history):
        player = ply & 1
        if is_wall(mv):
            walls_left[player] -= 1
        else:
            pawn_ranks[player] = pawn_rank(mv)

    zq_think = ti_think = 0.0
    zq_wall_moves = zq_pawn_moves = 0
    zq_backward = zq_lateral = 0
    zq_emptyhand_backward = both_emptyhand_zq_pawns = 0
    first_zero: Optional[str] = None
    winner = -1
    termination = "max_plies"

    try:
        while len(history) < max_plies:
            player = len(history) & 1
            engine = zq if player == zq_player else ti
            mv, think_s, _ = engine.bestmove(history, movetime_ms)
            if mv == "(none)":
                # A no-move response is valid only after a terminal history.
                termination = "no_move"
                break
            if not syntax_ok(mv):
                raise RuntimeError(f"{engine.name}: invalid move syntax {mv!r}")

            before_rank = pawn_ranks[player]
            before_both_empty = walls_left[0] == 0 and walls_left[1] == 0
            history.append(mv)
            if engine is zq:
                zq_think += think_s
            else:
                ti_think += think_s

            if is_wall(mv):
                walls_left[player] -= 1
                if walls_left[player] < 0:
                    raise RuntimeError(f"player {player} used >10 walls")
                if walls_left[player] == 0 and first_zero is None:
                    first_zero = "zquoridor" if player == zq_player else "titanium"
                if player == zq_player:
                    zq_wall_moves += 1
            else:
                new_rank = pawn_rank(mv)
                pawn_ranks[player] = new_rank
                if player == zq_player:
                    zq_pawn_moves += 1
                    forward = (new_rank > before_rank) if player == 0 else (new_rank < before_rank)
                    backward = (new_rank < before_rank) if player == 0 else (new_rank > before_rank)
                    lateral = new_rank == before_rank
                    if backward:
                        zq_backward += 1
                    if lateral:
                        zq_lateral += 1
                    if before_both_empty:
                        both_emptyhand_zq_pawns += 1
                        if backward:
                            zq_emptyhand_backward += 1

            if reached_goal(player, mv):
                winner = player
                termination = "goal"
                break
    finally:
        zq.close()
        ti.close()

    result = 0.5 if winner < 0 else (1.0 if winner == zq_player else 0.0)
    return GameResult(
        opening_index=opening_index,
        zq_player=zq_player,
        result=result,
        winner=winner,
        plies=len(history),
        zq_think_s=zq_think,
        titanium_think_s=ti_think,
        zq_walls_remaining=walls_left[zq_player],
        titanium_walls_remaining=walls_left[1 - zq_player],
        zq_wall_moves=zq_wall_moves,
        zq_pawn_moves=zq_pawn_moves,
        zq_backward_pawn_moves=zq_backward,
        zq_lateral_pawn_moves=zq_lateral,
        zq_emptyhand_backward_moves=zq_emptyhand_backward,
        both_emptyhand_zq_pawn_moves=both_emptyhand_zq_pawns,
        first_zero=first_zero or "neither",
        termination=termination,
        moves=history,
    )


def elo_from_score(p: float) -> float:
    p = min(1.0 - 1e-9, max(1e-9, p))
    return 400.0 * math.log10(p / (1.0 - p))


def summarize(results: Sequence[GameResult]) -> dict:
    n = len(results)
    scores = [g.result for g in results]
    p = sum(scores) / n if n else 0.5
    wins = sum(g.result == 1.0 for g in results)
    draws = sum(g.result == 0.5 for g in results)
    losses = sum(g.result == 0.0 for g in results)
    if n > 1:
        mean = p
        sample_var = sum((x - mean) ** 2 for x in scores) / (n - 1)
        se = math.sqrt(sample_var / n)
    else:
        se = 0.0
    plo = max(1e-6, p - 1.96 * se)
    phi = min(1 - 1e-6, p + 1.96 * se)
    elo = elo_from_score(p)
    elo_lo, elo_hi = elo_from_score(plo), elo_from_score(phi)

    zq_pawns = sum(g.zq_pawn_moves for g in results)
    zq_back = sum(g.zq_backward_pawn_moves for g in results)
    zq_lat = sum(g.zq_lateral_pawn_moves for g in results)
    empty_pawns = sum(g.both_emptyhand_zq_pawn_moves for g in results)
    empty_back = sum(g.zq_emptyhand_backward_moves for g in results)

    def avg(vals):
        vals = list(vals)
        return sum(vals) / len(vals) if vals else 0.0

    by_color = {}
    for color in (0, 1):
        sub = [g for g in results if g.zq_player == color]
        by_color[str(color)] = {
            "games": len(sub),
            "score_pct": 100.0 * avg(g.result for g in sub),
            "wins": sum(g.result == 1.0 for g in sub),
            "draws": sum(g.result == 0.5 for g in sub),
            "losses": sum(g.result == 0.0 for g in sub),
        }

    first_zero = {}
    for who in ("zquoridor", "titanium", "neither"):
        sub = [g for g in results if g.first_zero == who]
        first_zero[who] = {
            "games": len(sub),
            "zq_score_pct": 100.0 * avg(g.result for g in sub),
        }

    return {
        "games": n,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "score_pct": 100.0 * p,
        "elo": elo,
        "elo_95_low": elo_lo,
        "elo_95_high": elo_hi,
        "avg_plies": avg(g.plies for g in results),
        "avg_plies_wins": avg(g.plies for g in results if g.result == 1.0),
        "avg_plies_losses": avg(g.plies for g in results if g.result == 0.0),
        "avg_zq_think_s": avg(g.zq_think_s for g in results),
        "avg_titanium_think_s": avg(g.titanium_think_s for g in results),
        "zq_wall_fraction_pct": 100.0 * sum(g.zq_wall_moves for g in results) / max(1, sum(g.zq_wall_moves + g.zq_pawn_moves for g in results)),
        "zq_backward_pawn_pct": 100.0 * zq_back / max(1, zq_pawns),
        "zq_lateral_pawn_pct": 100.0 * zq_lat / max(1, zq_pawns),
        "zq_both_emptyhand_backward_pct": 100.0 * empty_back / max(1, empty_pawns),
        "zq_both_emptyhand_pawn_moves": empty_pawns,
        "by_color": by_color,
        "by_first_zero_walls": first_zero,
        "terminations": {k: sum(g.termination == k for g in results) for k in sorted({g.termination for g in results})},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zq", required=True)
    ap.add_argument("--zq-arg", action="append", default=[])
    ap.add_argument("--titanium", required=True)
    ap.add_argument("--openings", required=True)
    ap.add_argument("--movetime", type=int, default=200)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit-openings", type=int, default=0)
    ap.add_argument("--max-plies", type=int, default=180)
    ap.add_argument("--json-out", default="")
    ap.add_argument("--games-out", default="")
    args = ap.parse_args()

    openings = []
    for line in Path(args.openings).read_text().splitlines():
        if line.strip():
            openings.append(json.loads(line)["moves"])
    if args.limit_openings > 0:
        openings = openings[: args.limit_openings]

    zq_cmd = [args.zq] + args.zq_arg
    titanium_cmd = [args.titanium, "uci"]
    tasks = []
    for i, op in enumerate(openings):
        tasks.append((i, op, 0))
        tasks.append((i, op, 1))

    results: List[GameResult] = []
    print(f"arena: {len(tasks)} games, {len(openings)} paired openings, movetime={args.movetime}ms, workers={args.workers}", flush=True)
    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [
            ex.submit(play_game, i, op, color, zq_cmd, titanium_cmd, args.movetime, args.max_plies)
            for i, op, color in tasks
        ]
        for done, fut in enumerate(cf.as_completed(futs), 1):
            g = fut.result()
            results.append(g)
            if done % 10 == 0 or done == len(futs):
                s = summarize(results)
                print(
                    f"[{done:3d}/{len(futs)}] W/D/L={s['wins']}/{s['draws']}/{s['losses']} "
                    f"score={s['score_pct']:.1f}% Elo={s['elo']:+.1f} "
                    f"95%=[{s['elo_95_low']:+.1f},{s['elo_95_high']:+.1f}]",
                    flush=True,
                )

    results.sort(key=lambda g: (g.opening_index, g.zq_player))
    summary = summarize(results)
    print("\nFINAL")
    print(json.dumps(summary, indent=2, sort_keys=True))

    if args.games_out:
        Path(args.games_out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.games_out, "w") as f:
            for g in results:
                f.write(json.dumps(asdict(g), separators=(",", ":")) + "\n")
    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
