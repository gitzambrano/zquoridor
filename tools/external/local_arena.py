#!/usr/bin/env python3
"""Run paired Quoridor games under a complete local referee."""
from __future__ import annotations

import hashlib
import json
import math
import os
import queue
import random
import signal
import subprocess
import sys
import threading
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Callable, Iterable, Sequence


class IllegalMove(ValueError):
    """A move is not legal in the current position."""


class EngineError(RuntimeError):
    """An engine process returned an error or stopped."""


class EngineTimeout(EngineError):
    """An engine process exceeded a response time limit."""


def _move_text(row: int, col: int) -> str:
    return chr(ord("a") + col) + str(row + 1)


class Referee:
    """Validate standard two-player Quoridor moves and determine the winner."""

    def __init__(self) -> None:
        self.pawns = [(0, 4), (8, 4)]
        self.walls_left = [10, 10]
        self.horizontal: set[tuple[int, int]] = set()
        self.vertical: set[tuple[int, int]] = set()
        self.side_to_move = 0
        self.winner: int | None = None

    @staticmethod
    def _inside(cell: tuple[int, int]) -> bool:
        return 0 <= cell[0] < 9 and 0 <= cell[1] < 9

    def _edge_open(self, first: tuple[int, int], second: tuple[int, int]) -> bool:
        r1, c1 = first
        r2, c2 = second
        if abs(r1 - r2) + abs(c1 - c2) != 1:
            return False
        if c1 == c2:
            row = min(r1, r2)
            return (row, c1) not in self.horizontal and (row, c1 - 1) not in self.horizontal
        col = min(c1, c2)
        return (r1, col) not in self.vertical and (r1 - 1, col) not in self.vertical

    def _neighbors(self, cell: tuple[int, int]) -> Iterable[tuple[int, int]]:
        row, col = cell
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            target = (row + dr, col + dc)
            if self._inside(target) and self._edge_open(cell, target):
                yield target

    def _pawn_destinations(self) -> set[tuple[int, int]]:
        own = self.pawns[self.side_to_move]
        other = self.pawns[1 - self.side_to_move]
        result: set[tuple[int, int]] = set()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            adjacent = (own[0] + dr, own[1] + dc)
            if not self._inside(adjacent) or not self._edge_open(own, adjacent):
                continue
            if adjacent != other:
                result.add(adjacent)
                continue
            beyond = (other[0] + dr, other[1] + dc)
            if self._inside(beyond) and self._edge_open(other, beyond):
                result.add(beyond)
                continue
            for turn_dr, turn_dc in ((dc, dr), (-dc, -dr)):
                diagonal = (other[0] + turn_dr, other[1] + turn_dc)
                if self._inside(diagonal) and self._edge_open(other, diagonal):
                    result.add(diagonal)
        return result

    def _has_goal_path(self, player: int) -> bool:
        goal_row = 8 if player == 0 else 0
        todo = deque([self.pawns[player]])
        seen = {self.pawns[player]}
        while todo:
            cell = todo.popleft()
            if cell[0] == goal_row:
                return True
            for target in self._neighbors(cell):
                if target not in seen:
                    seen.add(target)
                    todo.append(target)
        return False

    def _wall_geometry_ok(self, row: int, col: int, orientation: str) -> bool:
        slot = (row, col)
        if orientation == "h":
            if slot in self.vertical:
                return False
            return all((row, c) not in self.horizontal for c in (col - 1, col, col + 1))
        if slot in self.horizontal:
            return False
        return all((r, col) not in self.vertical for r in (row - 1, row, row + 1))

    def _wall_legal(self, row: int, col: int, orientation: str) -> bool:
        if not (0 <= row < 8 and 0 <= col < 8):
            return False
        if self.walls_left[self.side_to_move] <= 0:
            return False
        if not self._wall_geometry_ok(row, col, orientation):
            return False
        walls = self.horizontal if orientation == "h" else self.vertical
        walls.add((row, col))
        try:
            return self._has_goal_path(0) and self._has_goal_path(1)
        finally:
            walls.remove((row, col))

    def legal_moves(self) -> list[str]:
        if self.winner is not None:
            return []
        moves = sorted(_move_text(*cell) for cell in self._pawn_destinations())
        if self.walls_left[self.side_to_move] > 0:
            for orientation in ("h", "v"):
                for row in range(8):
                    for col in range(8):
                        if self._wall_legal(row, col, orientation):
                            moves.append(_move_text(row, col) + orientation)
        return moves

    def apply(self, move: str) -> None:
        text = move.strip().lower()
        if text not in self.legal_moves():
            raise IllegalMove(f"illegal move {move!r} for player {self.side_to_move}")
        player = self.side_to_move
        col = ord(text[0]) - ord("a")
        row = int(text[1]) - 1
        if len(text) == 2:
            self.pawns[player] = (row, col)
            if row == (8 if player == 0 else 0):
                self.winner = player
        else:
            walls = self.horizontal if text[2] == "h" else self.vertical
            walls.add((row, col))
            self.walls_left[player] -= 1
        self.side_to_move ^= 1


def _reader(stream, output: "queue.Queue[object]") -> None:
    try:
        for line in iter(stream.readline, ""):
            output.put(line.rstrip("\r\n"))
        output.put(None)
    except BaseException as exc:
        output.put(exc)


def _stderr_reader(stream, tail: deque[str]) -> None:
    try:
        for line in iter(stream.readline, ""):
            tail.append(line.rstrip("\r\n"))
    except BaseException as exc:
        tail.append(f"stderr reader error: {exc}")


def terminate_process_tree(process: subprocess.Popen[str]) -> None:
    """Stop one engine process and its child processes."""
    if process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    else:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    try:
        process.wait(timeout=2.0)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=2.0)


class LinePlayer:
    """Manage one persistent line-based engine process."""

    def __init__(self, argv: Sequence[str], name: str, *, startup_timeout_s: float,
                 env: dict[str, str] | None = None) -> None:
        self.argv = [str(item) for item in argv]
        self.name = name
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
        self.process = subprocess.Popen(
            self.argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=env,
            creationflags=creationflags,
            start_new_session=os.name != "nt",
        )
        assert self.process.stdin and self.process.stdout and self.process.stderr
        self._lines: "queue.Queue[object]" = queue.Queue()
        self._stderr: deque[str] = deque(maxlen=200)
        threading.Thread(target=_reader, args=(self.process.stdout, self._lines), daemon=True).start()
        threading.Thread(target=_stderr_reader, args=(self.process.stderr, self._stderr), daemon=True).start()
        self.startup_timeout_s = startup_timeout_s

    def _send(self, line: str) -> None:
        if self.process.poll() is not None:
            raise EngineError(f"{self.name}: process exited with code {self.process.returncode}")
        assert self.process.stdin
        try:
            self.process.stdin.write(line + "\n")
            self.process.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise EngineError(f"{self.name}: process input failed: {exc}") from exc

    def _read(self, timeout_s: float) -> str:
        try:
            item = self._lines.get(timeout=timeout_s)
        except queue.Empty as exc:
            terminate_process_tree(self.process)
            self._close_streams()
            raise EngineTimeout(
                f"{self.name}: timeout after {timeout_s:.3f} seconds; stderr={self._tail()!r}"
            ) from exc
        if item is None:
            terminate_process_tree(self.process)
            self._close_streams()
            raise EngineError(
                f"{self.name}: process exited with code {self.process.poll()}; stderr={self._tail()!r}"
            )
        if isinstance(item, BaseException):
            terminate_process_tree(self.process)
            self._close_streams()
            raise EngineError(f"{self.name}: output reader failed: {item}")
        return str(item).strip()

    def _tail(self) -> str:
        return "\n".join(self._stderr)[-4000:]

    def _close_streams(self) -> None:
        for stream in (self.process.stdin, self.process.stdout, self.process.stderr):
            if stream is not None and not stream.closed:
                stream.close()

    def _wait_token(self, token: str, timeout_s: float) -> None:
        deadline = time.monotonic() + timeout_s
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                terminate_process_tree(self.process)
                self._close_streams()
                raise EngineTimeout(f"{self.name}: timeout while waiting for {token}")
            line = self._read(remaining)
            if line == token or line.startswith(token + " "):
                return
            if "error" in line.lower():
                terminate_process_tree(self.process)
                self._close_streams()
                raise EngineError(f"{self.name}: {line}")

    def close(self) -> None:
        if self.process.poll() is None:
            try:
                self._send("quit")
                self.process.wait(timeout=1.0)
            except (EngineError, subprocess.TimeoutExpired):
                terminate_process_tree(self.process)
        self._close_streams()


class UciPlayer(LinePlayer):
    """Use the conventional UCI-style protocol of the Zquoridor adapter."""

    def __init__(self, argv: Sequence[str], name: str, *, startup_timeout_s: float = 20.0,
                 env: dict[str, str] | None = None) -> None:
        super().__init__(argv, name, startup_timeout_s=startup_timeout_s, env=env)
        self._send("uci")
        self._wait_token("uciok", startup_timeout_s)
        self._send("isready")
        self._wait_token("readyok", startup_timeout_s)
        self._send("ucinewgame")

    def bestmove(self, history: Sequence[str], *, budget: int,
                 timeout_s: float) -> tuple[str, float, list[str]]:
        command = "position startpos"
        if history:
            command += " moves " + " ".join(history)
        self._send(command)
        started = time.monotonic()
        self._send(f"go movetime {budget}")
        info: list[str] = []
        deadline = started + timeout_s
        while True:
            line = self._read(max(0.001, deadline - time.monotonic()))
            if "error" in line.lower():
                raise EngineError(f"{self.name}: {line}")
            if line.startswith("info "):
                info.append(line)
            elif line.startswith("bestmove "):
                return line.split()[1], time.monotonic() - started, info


class TitaniumPlayer(LinePlayer):
    """Use Titanium's documented native session protocol."""

    def __init__(self, executable: Path, *, startup_timeout_s: float = 20.0) -> None:
        env = os.environ.copy()
        env["TITANIUM_PONDERING"] = "0"
        super().__init__([str(executable), "uci"], "titanium",
                         startup_timeout_s=startup_timeout_s, env=env)
        self._send("reset")
        self._wait_token("ready", startup_timeout_s)

    def bestmove(self, history: Sequence[str], *, budget: int,
                 timeout_s: float) -> tuple[str, float, list[str]]:
        self._send("position" + (" " + " ".join(history) if history else ""))
        self._wait_token("ready", timeout_s)
        started = time.monotonic()
        self._send(f"go {budget / 1000.0:.6f}")
        deadline = started + timeout_s
        while True:
            line = self._read(max(0.001, deadline - time.monotonic()))
            if line.startswith("error ") or "error" in line.lower():
                raise EngineError(f"titanium: {line}")
            if line.startswith("bestmove "):
                return line.split()[1], time.monotonic() - started, []


class ClaustrophobiaPlayer(LinePlayer):
    """Use a persistent Claustrophobia model and a fixed move clock."""

    def __init__(self, bridge: Path, checkpoint: Path, *, move_time_ms: int, max_sims: int, cpuct: float,
                 device: str, startup_timeout_s: float = 120.0) -> None:
        environment = os.environ.copy()
        environment["ZQ_PYTHON"] = sys.executable
        super().__init__(
            [str(bridge), str(checkpoint), str(move_time_ms), str(cpuct), device, str(max_sims)],
            "claustrophobia",
            startup_timeout_s=startup_timeout_s,
            env=environment,
        )
        self._wait_token("ready", startup_timeout_s)

    def bestmove(self, history: Sequence[str], *, budget: int,
                 timeout_s: float) -> tuple[str, float, list[str]]:
        started = time.monotonic()
        self._send("position\t" + " ".join(history))
        line = self._read(timeout_s)
        try:
            result = json.loads(line)
        except json.JSONDecodeError as exc:
            raise EngineError(f"claustrophobia: invalid JSON output {line!r}") from exc
        if "error" in result:
            raise EngineError(f"claustrophobia: {result['error']}")
        if int(result.get("move_time_ms", -1)) != budget:
            raise EngineError("claustrophobia: the reported move clock does not match")
        return str(result["bestmove"]), time.monotonic() - started, [line]


def play_game(*, opponent: str, opening_index: int, opening: Sequence[str],
              zq_player: int, zq_factory: Callable[[], LinePlayer],
              opponent_factory: Callable[[], LinePlayer], zq_budget: int,
              opponent_budget: int, move_timeout_s: float, max_plies: int,
              run_id: str) -> dict:
    """Play one game and return an explicit success or failure record."""
    base = {
        "schema": "zquoridor.local_benchmark.game.v1",
        "run_id": run_id,
        "opponent": opponent,
        "opening_index": opening_index,
        "zq_player": zq_player,
    }
    zq: LinePlayer | None = None
    other: LinePlayer | None = None
    history: list[str] = []
    referee = Referee()
    think = {"zquoridor": 0.0, opponent: 0.0}
    move_times: list[dict[str, float | int | str]] = []
    state_visits = defaultdict(int)
    repeated_states = 0
    def count_state():
        nonlocal repeated_states
        key = (referee.side_to_move, tuple(referee.pawns), tuple(referee.walls_left),
               tuple(sorted(referee.horizontal)), tuple(sorted(referee.vertical)))
        if state_visits[key]:
            repeated_states += 1
        state_visits[key] += 1
    count_state()
    try:
        for move in opening:
            referee.apply(move)
            history.append(move)
            count_state()
        if referee.winner is not None:
            raise IllegalMove("the opening is already terminal")
        zq = zq_factory()
        other = opponent_factory()
        while referee.winner is None and len(history) < max_plies:
            side = referee.side_to_move
            player = zq if side == zq_player else other
            name = "zquoridor" if side == zq_player else opponent
            budget = zq_budget if side == zq_player else opponent_budget
            move, elapsed, _ = player.bestmove(history, budget=budget, timeout_s=move_timeout_s)
            think[name] += elapsed
            move_times.append({"player": name, "budget_ms": budget, "elapsed_ms": elapsed * 1000.0})
            referee.apply(move)
            history.append(move)
            count_state()
        if referee.winner is None:
            result = 0.5
            termination = "max_plies"
        else:
            result = 1.0 if referee.winner == zq_player else 0.0
            termination = "goal"
        return {
            **base,
            "status": "ok",
            "repeated_states": repeated_states,
            "result": result,
            "winner": referee.winner,
            "termination": termination,
            "plies": len(history),
            "moves": history,
            "think_s": think,
            "move_times": move_times,
        }
    except BaseException as exc:
        if isinstance(exc, KeyboardInterrupt):
            raise
        return {
            **base,
            "status": "failed",
            "termination": "error",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "plies": len(history),
            "moves": history,
            "think_s": think,
            "move_times": move_times,
        }
    finally:
        for player in (zq, other):
            if player is not None:
                player.close()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_value(value: object) -> object:
    if isinstance(value, Path):
        return str(value.resolve())
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def make_manifest(config: dict, artifacts: dict[str, Path]) -> dict:
    artifact_rows = {}
    for name, raw_path in sorted(artifacts.items()):
        path = Path(raw_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"benchmark artifact not found: {path}")
        artifact_rows[name] = {"path": str(path), "size": path.stat().st_size,
                               "sha256": _sha256(path)}
    identity = {"config": _json_value(config), "artifacts": artifact_rows}
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "schema": "zquoridor.local_benchmark.manifest.v1",
        "run_id": hashlib.sha256(encoded).hexdigest(),
        **identity,
    }


def prepare_resume(output_dir: Path, manifest: dict) -> tuple[Path, list[dict]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    games_path = output_dir / "games.jsonl"
    if manifest_path.exists():
        stored = json.loads(manifest_path.read_text(encoding="utf-8"))
        if stored.get("run_id") != manifest.get("run_id"):
            raise ValueError(
                "the output directory contains a different benchmark configuration; "
                "select another output directory"
            )
    elif games_path.exists() and games_path.stat().st_size:
        raise ValueError("the output directory has games but no benchmark manifest")
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    rows = []
    if games_path.exists():
        for lineno, line in enumerate(games_path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("run_id") != manifest.get("run_id"):
                raise ValueError(f"games.jsonl:{lineno}: the run id does not match the manifest")
            rows.append(row)
    return games_path, rows


def _percentile(values: list[float], quantile: float) -> float:
    values = sorted(values)
    point = quantile * (len(values) - 1)
    low = math.floor(point)
    high = math.ceil(point)
    if low == high:
        return values[low]
    return values[low] * (high - point) + values[high] * (point - low)


def _elo(score: float) -> float:
    score = min(1.0 - 1e-9, max(1e-9, score))
    return 400.0 * math.log10(score / (1.0 - score))


def summarize_pairs(rows: Sequence[dict], *, bootstrap: int, seed: int) -> dict:
    """Calculate an interval from complete pairs and exclude all other games."""
    if bootstrap < 1000:
        raise ValueError("bootstrap must be at least 1000")
    latest = {}
    for row in rows:
        key = (row.get("opponent"), int(row.get("opening_index", -1)),
               int(row.get("zq_player", -1)))
        latest[key] = row
    failed = sum(row.get("status") != "ok" for row in latest.values())
    grouped: dict[tuple[str, int], dict[int, dict]] = defaultdict(dict)
    for (opponent, opening_index, color), row in latest.items():
        if row.get("status") == "ok":
            grouped[(str(opponent), opening_index)][color] = row
    pair_points = []
    used_games = 0
    ok_games = sum(row.get("status") == "ok" for row in latest.values())
    for colors in grouped.values():
        if set(colors) == {0, 1}:
            pair_points.append(float(colors[0]["result"]) + float(colors[1]["result"]))
            used_games += 2
    report = {
        "schema": "zquoridor.local_benchmark.summary.v1",
        "recorded_games": len(latest),
        "failed_games": failed,
        "complete_pairs": len(pair_points),
        "included_games": used_games,
        "excluded_ok_games": ok_games - used_games,
    }
    included = [row for colors in grouped.values() if set(colors) == {0, 1} for row in colors.values()]
    report["max_plies_games"] = sum(row.get("termination") == "max_plies" for row in included)
    report["repeated_states"] = sum(row.get("repeated_states", 0) for row in included)
    report["mean_plies"] = sum(row.get("plies", 0) for row in included) / max(1, len(included))
    if not pair_points:
        return {**report, "score_pct": None, "elo": None, "paired_bootstrap_95": None}
    score = sum(pair_points) / (2.0 * len(pair_points))
    # A bootstrap can collapse to a point for a tiny or constant sample.
    # This bound remains conservative for independent bounded pair scores.
    radius = math.sqrt(math.log(40.0) / (2.0 * len(pair_points)))
    rng = random.Random(seed)
    samples = [
        sum(pair_points[rng.randrange(len(pair_points))] for _ in pair_points)
        / (2.0 * len(pair_points))
        for _ in range(bootstrap)
    ]
    low = _percentile(samples, 0.025)
    high = _percentile(samples, 0.975)
    return {
        **report,
        "score_pct": 100.0 * score,
        "elo": _elo(score),
        "decision_95": {"method": "paired Hoeffding bound",
                        "score_low_pct": 100 * max(0.0, score - radius),
                        "score_high_pct": 100 * min(1.0, score + radius)},
        "strength_claim_ready": len(pair_points) >= 100 and failed == 0,
        "paired_bootstrap_95": {
            "iterations": bootstrap,
            "seed": seed,
            "score_low_pct": 100.0 * low,
            "score_high_pct": 100.0 * high,
            "elo_low": _elo(low),
            "elo_high": _elo(high),
        },
    }
