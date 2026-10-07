#!/usr/bin/env python3
"""Collect complete games and the search targets that Claustrophobia used in each game."""
from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import gzip
import hashlib
import json
import math
import os
import random
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Iterator, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.external import local_arena
from tools import run_benchmark
from training.teachers.targets import (
    POLICY_DIM, claustrophobia_policy_to_zq, make_label, sample_id,
    zq_move_to_policy_index,
)

# Edit these defaults for a normal run. CLI options override one invocation.
CONFIG = {
    "mode": "match",
    "pairs": 1500,
    "workers": 1,
    "seed": 20261007,
    "opening_books": {
        "normal": str(ROOT / "tools/external/openings_irregular_bank.jsonl"),
        "center_rush": str(ROOT / "tools/external/openings_center_rush_sound_5k.jsonl"),
        "weakness": str(ROOT / "tools/external/weak_openings_mined.jsonl"),
    },
    "opening_weights": {},
    "unique_openings_first": False,
    "opening_temperature": 0.0,
    "temperature_plies": 14,
    "record_both_searches": False,
    "output": str(ROOT / "data/teaching/search_games"),
    "start_move_time_ms": 400,
    "end_move_time_ms": 50,
    "decay_start_ply": 14,
    "decay_end_ply": 80,
    "schedule_origin": "opening",
    "max_plies": 180,
    "move_timeout_s": 30.0,
    "startup_timeout_s": 120.0,
    "zq_executable": None,
    "nnue": str(ROOT / "data/nnue/nnue_weights_int8.bin"),
    "zq_args": [],
    "claustrophobia_bridge": None,
    "claustrophobia_checkpoint": None,
    "claustrophobia_cpuct": 1.5,
    "claustrophobia_device": "cpu",
    "validation_fraction": 0.1,
    "auto_setup": True,
    "resume": True,
    "export_targets": True,
    "batch_games": 250,
    "export_final_run": True,
    "compress_game_ledger": False,
    "dry_run": False,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("match", "zquoridor-selfplay", "claustrophobia-selfplay"))
    parser.add_argument("--opening-book", action="append", dest="opening_books", metavar="NAME=PATH",
                        help="Replace the default books. Repeat this option for each book.")
    parser.add_argument("--opening-weight", action="append", dest="opening_weights", metavar="NAME=WEIGHT",
                        help="Set a nonnegative book weight. Unspecified books use weight 1.")
    parser.add_argument("--opening-temperature", type=float)
    parser.add_argument("--temperature-plies", type=int)
    for key in ("pairs", "workers", "seed", "start_move_time_ms", "end_move_time_ms",
                "decay_start_ply", "decay_end_ply", "max_plies", "batch_games"):
        parser.add_argument("--" + key.replace("_", "-"), type=int)
    for key in ("move_timeout_s", "startup_timeout_s", "claustrophobia_cpuct", "validation_fraction"):
        parser.add_argument("--" + key.replace("_", "-"), type=float)
    for key in ("output", "zq_executable", "nnue", "claustrophobia_bridge", "claustrophobia_checkpoint"):
        parser.add_argument("--" + key.replace("_", "-"))
    parser.add_argument("--claustrophobia-device", choices=("cpu", "gpu"))
    parser.add_argument("--schedule-origin", choices=("opening", "game"))
    parser.add_argument("--zq-arg", dest="zq_args", action="append")
    for key in ("auto_setup", "resume", "export_targets", "export_final_run", "compress_game_ledger", "dry_run", "record_both_searches",
                "unique_openings_first"):
        parser.add_argument("--" + key.replace("_", "-"), action=argparse.BooleanOptionalAction, default=None)
    return parser


def resolve_config(args: argparse.Namespace) -> dict:
    config = {**CONFIG, "opening_books": dict(CONFIG["opening_books"]),
              "opening_weights": dict(CONFIG["opening_weights"]), "zq_args": list(CONFIG["zq_args"])}
    config.update({key: value for key, value in vars(args).items() if value is not None})
    if isinstance(config["opening_books"], list):
        books = {}
        for text in config["opening_books"]:
            name, separator, path = text.partition("=")
            if not separator or not name.strip() or not path.strip() or name in books:
                raise ValueError("each opening book must have a unique NAME=PATH")
            books[name] = path
        config["opening_books"] = books
    if isinstance(config["opening_weights"], list):
        weights = {}
        for text in config["opening_weights"]:
            name, separator, value = text.partition("=")
            if not separator or not name or name in weights:
                raise ValueError("each opening weight must have a unique NAME=WEIGHT")
            weights[name] = float(value)
        config["opening_weights"] = weights
    opening_quotas(config)
    if not math.isfinite(config["opening_temperature"]) or config["opening_temperature"] < 0:
        raise ValueError("opening_temperature must be finite and nonnegative")
    if config["temperature_plies"] < 0:
        raise ValueError("temperature_plies must be nonnegative")
    if config["record_both_searches"] and config["mode"] != "match":
        raise ValueError("record_both_searches requires match mode")
    for key in ("pairs", "workers", "start_move_time_ms", "end_move_time_ms", "max_plies", "batch_games"):
        if int(config[key]) <= 0:
            raise ValueError(f"{key} must be positive")
    if not 0 <= config["decay_start_ply"] < config["decay_end_ply"]:
        raise ValueError("decay_start_ply must be nonnegative and less than decay_end_ply")
    if config["start_move_time_ms"] < config["end_move_time_ms"]:
        raise ValueError("start_move_time_ms must be at least end_move_time_ms")
    for key in ("move_timeout_s", "startup_timeout_s", "claustrophobia_cpuct"):
        if not math.isfinite(float(config[key])) or float(config[key]) <= 0:
            raise ValueError(f"{key} must be finite and positive")
    if config["move_timeout_s"] * 1000 <= config["start_move_time_ms"]:
        raise ValueError("move_timeout_s must exceed the largest search budget")
    if not math.isfinite(config["validation_fraction"]) or not 0 <= config["validation_fraction"] < 1:
        raise ValueError("validation_fraction must be in [0,1)")
    if not config["opening_books"]:
        raise ValueError("at least one opening book is required")
    return config


def move_time_ms(ply: int, config: dict) -> int:
    """Return the rounded linear budget for a ply relative to the schedule origin."""
    fraction = min(1.0, max(0.0, (ply - config["decay_start_ply"]) /
                            (config["decay_end_ply"] - config["decay_start_ply"])))
    budget = config["start_move_time_ms"] + fraction * (
        config["end_move_time_ms"] - config["start_move_time_ms"])
    return int(math.floor(budget + 0.5))


def read_book(name: str, path: Path) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    if path.suffix.lower() == ".json":
        content = json.loads(path.read_text(encoding="utf-8"))
        rows = content if isinstance(content, list) else content.get("positions", [])
        records = enumerate(rows, 1)
    else:
        records = ((number, json.loads(line)) for number, line in
                   enumerate(path.read_text(encoding="utf-8").splitlines(), 1) if line.strip())
    for number, row in records:
        moves = row.get("history", row.get("moves"))
        if not isinstance(moves, list) or not all(isinstance(move, str) for move in moves):
            raise ValueError(f"{path}:{number}: expected a string history or moves list")
        category = str(row.get("category") or name)
        groups[category].append({"book": name, "category": category,
                                 "book_row": number, "source_id": str(row.get("id", number)),
                                 "opening": [move.strip().lower() for move in moves]})
    if not groups:
        raise ValueError(f"the opening book is empty: {path}")
    return groups


def opening_quotas(config: dict) -> dict[str, int]:
    """Allocate exact pair quotas by the largest remainder method."""
    names = sorted(config["opening_books"])
    explicit = config.get("opening_weights", {})
    if set(explicit) - set(names):
        raise ValueError("an opening weight refers to an unknown book")
    weights = {name: float(explicit.get(name, 1.0)) for name in names}
    if any(not math.isfinite(weight) or weight < 0 for weight in weights.values()):
        raise ValueError("opening weights must be finite and nonnegative")
    total = sum(weights.values())
    if not math.isfinite(total) or total <= 0:
        raise ValueError("the opening weights must have positive total mass")
    fractional = {name: int(config["pairs"]) * weight / total for name, weight in weights.items()}
    quotas = {name: math.floor(value) for name, value in fractional.items()}
    tie_order = list(names)
    random.Random(config["seed"]).shuffle(tie_order)
    ranking = sorted(tie_order, key=lambda name: -(fractional[name] - quotas[name]))
    for name in ranking[:int(config["pairs"]) - sum(quotas.values())]:
        quotas[name] += 1
    return quotas


def unique_book_cycle(groups: dict[str, list[dict]], categories: list[str],
                      rng: random.Random) -> Iterator[dict]:
    """Exhaust each category once before a new shuffled book cycle."""
    while True:
        available = list(categories)
        offsets: Counter = Counter()
        cursor = 0
        while available:
            cursor %= len(available)
            category = available[cursor]
            rows = groups[category]
            yield rows[offsets[category]]
            offsets[category] += 1
            if offsets[category] == len(rows):
                available.pop(cursor)
            else:
                cursor += 1
        for rows in groups.values():
            rng.shuffle(rows)
        rng.shuffle(categories)


def select_openings(config: dict) -> list[dict]:
    """Allocate weighted books and balanced categories, then cycle shuffled rows."""
    rng = random.Random(config["seed"])
    books = {name: read_book(name, Path(path).resolve())
             for name, path in sorted(config["opening_books"].items())}
    quotas = opening_quotas(config)
    names = [name for name, quota in quotas.items() for _ in range(quota)]
    rng.shuffle(names)
    categories = {}
    for name, groups in books.items():
        if config.get("unique_openings_first", False):
            seen: set[tuple[str, ...]] = set()
            for category in sorted(groups):
                unique_rows = []
                for row in groups[category]:
                    history = tuple(row["opening"])
                    if history not in seen:
                        seen.add(history)
                        unique_rows.append(row)
                groups[category] = unique_rows
            for category in [category for category, rows in groups.items() if not rows]:
                del groups[category]
        categories[name] = list(sorted(groups))
        rng.shuffle(categories[name])
        for rows in groups.values():
            rng.shuffle(rows)
    cycles = {name: unique_book_cycle(groups, categories[name], rng)
              for name, groups in books.items()} if config.get("unique_openings_first", False) else {}
    book_counts: Counter = Counter()
    group_counts: Counter = Counter()
    selected = []
    validated: set[tuple[str, ...]] = set()
    for pair_index in range(config["pairs"]):
        name = names[pair_index]
        if cycles:
            row = dict(next(cycles[name]))
        else:
            category = categories[name][book_counts[name] % len(categories[name])]
            book_counts[name] += 1
            rows = books[name][category]
            group = (name, category)
            row = dict(rows[group_counts[group] % len(rows)])
            group_counts[group] += 1
        history = tuple(row["opening"])
        if history not in validated:
            referee = local_arena.Referee()
            repetition = local_arena.RepetitionTracker()
            repetition.observe(referee)
            for move in history:
                referee.apply(move)
                if referee.winner is not None or repetition.observe(referee):
                    raise ValueError(f"{name}:{row['book_row']}: the opening is terminal")
            if len(history) >= config["max_plies"]:
                raise ValueError(f"{name}:{row['book_row']}: the opening reaches max_plies")
            validated.add(history)
        opening_group = sample_id(history)
        point = int(hashlib.sha256((str(config["seed"]) + opening_group).encode()).hexdigest()[:16], 16)
        row.update(pair_index=pair_index, opening_group=opening_group,
                   split="val" if point / 2**64 < config["validation_fraction"] else "train")
        selected.append(row)
    return selected


def global_action(move: str) -> int:
    """Map a raw board move to a global 209-action index."""
    return zq_move_to_policy_index(move, 0)


def canonical_to_global(index: int, side: int) -> int:
    """Convert a Zquoridor canonical action into the raw board frame."""
    if side == 0:
        return index
    if index < 81:
        row, col = divmod(index, 9)
        return (8 - row) * 9 + col
    base = 81 if index < 145 else 145
    row, col = divmod(index - base, 8)
    return base + (7 - row) * 8 + col


def parse_root(info: Sequence[str], referee: local_arena.Referee, move: str, budget: int) -> dict:
    """Validate the actual root visits and preserve the upstream solver metadata."""
    raw = next((json.loads(line) for line in reversed(info) if line.lstrip().startswith("{")), None)
    if not isinstance(raw, dict):
        raise local_arena.EngineError("Claustrophobia did not return a root JSON record")
    if raw.get("policy_frame") != "claustrophobia-canonical-209":
        raise local_arena.EngineError("Claustrophobia returned an unsupported action frame")
    side = referee.side_to_move
    if raw.get("side_to_move") != side or raw.get("root_value_perspective") != "side-to-move":
        raise local_arena.EngineError("Claustrophobia returned an inconsistent value perspective")
    if raw.get("move_time_ms") != budget or raw.get("bestmove") != move:
        raise local_arena.EngineError("Claustrophobia returned inconsistent search metadata")
    visits = raw.get("visit_counts")
    if not isinstance(visits, list) or len(visits) != POLICY_DIM or any(
            isinstance(n, bool) or not isinstance(n, int) or n < 0 for n in visits):
        raise local_arena.EngineError("Claustrophobia returned invalid root visits")
    value = float(raw.get("root_value", float("nan")))
    if not math.isfinite(value) or not -1 <= value <= 1:
        raise local_arena.EngineError("Claustrophobia returned an invalid root value")
    zq_visits = claustrophobia_policy_to_zq(visits, side)
    global_visits = [0] * POLICY_DIM
    for index, count in enumerate(zq_visits):
        global_visits[canonical_to_global(index, side)] = int(count)
    best = raw.get("best_action")
    if isinstance(best, bool) or not isinstance(best, int) or not 0 <= best < POLICY_DIM:
        raise local_arena.EngineError("Claustrophobia returned an invalid best_action")
    best_vector = [0] * POLICY_DIM
    best_vector[best] = 1
    zq_best = claustrophobia_policy_to_zq(best_vector, side).index(1)
    if canonical_to_global(zq_best, side) != global_action(move):
        raise local_arena.EngineError("Claustrophobia best_action does not match bestmove")
    legal = {global_action(text) for text in referee.legal_moves()}
    if any(count and index not in legal for index, count in enumerate(global_visits)):
        raise local_arena.EngineError("Claustrophobia returned visits on an illegal action")
    total = sum(global_visits)
    if raw.get("protocol") != "deadline-and-engine-clock-v2" or raw.get("mode") != "movetime":
        raise local_arena.EngineError("Claustrophobia did not use the deadline protocol")
    if raw.get("stop_reason") not in ("deadline", "solved"):
        raise local_arena.EngineError("Claustrophobia did not stop on its deadline or a proof")
    solved = raw.get("stop_reason") == "solved"
    if raw.get("sims") != raw.get("root_visits") or (
            total != raw.get("root_visits") and not solved):
        raise local_arena.EngineError("Claustrophobia returned inconsistent visit counts")
    return {"raw": raw, "visit_counts_global": global_visits, "root_value": value,
            "policy": [count / total for count in zq_visits] if total else None,
            "target_status": "excluded_solver_policy" if solved else "ok" if total else "excluded_zero_visits",
            "action_frame": "global-board-209", "root_value_perspective": "side-to-move"}


def parse_zq_root(info: Sequence[str], referee: local_arena.Referee, move: str, budget: int) -> dict:
    """Read the complete Zquoridor root counters without a sparse policy substitute."""
    payload = next((line.split("root_json ", 1)[1] for line in reversed(info)
                    if line.startswith("info string root_json ")), None)
    if payload is None:
        raise local_arena.EngineError("Zquoridor did not return complete root JSON. Use an adapter with --dump-root.")
    raw = json.loads(payload)
    if (raw.get("policy_frame") != "global-board-209" or
            raw.get("root_value_perspective") != "side-to-move" or
            raw.get("side_to_move") != referee.side_to_move):
        raise local_arena.EngineError("Zquoridor returned an inconsistent root perspective")
    if (raw.get("move_time_ms") != budget or raw.get("bestmove") != move or
            raw.get("best_action") != global_action(move)):
        raise local_arena.EngineError("Zquoridor returned inconsistent search metadata")
    visits = raw.get("visit_counts")
    if not isinstance(visits, list) or len(visits) != POLICY_DIM or any(
            isinstance(n, bool) or not isinstance(n, (int, float)) or not math.isfinite(n) or n < 0
            for n in visits):
        raise local_arena.EngineError("Zquoridor returned invalid root visits")
    legal = {global_action(text) for text in referee.legal_moves()}
    if any(count and index not in legal for index, count in enumerate(visits)):
        raise local_arena.EngineError("Zquoridor returned visits on an illegal action")
    total = sum(visits)
    value = raw.get("root_value")
    if value is None:
        if total:
            raise local_arena.EngineError("Zquoridor omitted a root value for positive visits")
    elif isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not -1 <= value <= 1:
        raise local_arena.EngineError("Zquoridor returned an invalid root value")
    if not isinstance(raw.get("root_visits"), (int, float)) or not math.isclose(
            raw["root_visits"], total, rel_tol=1e-7, abs_tol=1e-6):
        raise local_arena.EngineError("Zquoridor returned inconsistent visit counts")
    return {"raw": raw, "visit_counts_global": visits, "root_value": value,
            "action_frame": "global-board-209", "root_value_perspective": "side-to-move",
            "target_status": "comparison_only"}


def sampled_move(root: dict | None, bestmove: str, referee: local_arena.Referee,
                 temperature: float, rng: random.Random) -> tuple[str, str]:
    """Sample legal actual visit counts using a stable exponent calculation."""
    if temperature <= 0:
        return bestmove, "bestmove"
    if root is None:
        raise local_arena.EngineError("temperature requires the engine's complete root visits")
    if root["raw"].get("stop_reason") == "solved":
        return bestmove, "solver_bestmove"
    candidates = [(move, root["visit_counts_global"][global_action(move)])
                  for move in sorted(referee.legal_moves())
                  if root["visit_counts_global"][global_action(move)] > 0]
    if not candidates:
        return bestmove, "zero_visits_bestmove"
    max_log = max(math.log(count) for _, count in candidates)
    masses = [math.exp((math.log(count) - max_log) / temperature) for _, count in candidates]
    point = rng.random() * sum(masses)
    for (move, _), mass in zip(candidates, masses):
        point -= mass
        if point < 0:
            return move, "visit_temperature"
    return candidates[-1][0], "visit_temperature"


def play_game(task: dict, config: dict, run_id: str,
              factories: dict[str, Callable[[], object]]) -> dict:
    """Play one game without retries and retain every completed search response."""
    names = ("zquoridor", "claustrophobia")
    if config["mode"] == "match":
        names = names if task["zq_player"] == 0 else names[::-1]
    else:
        name = "zquoridor" if config["mode"] == "zquoridor-selfplay" else "claustrophobia"
        names = (name, name)
    game = {"schema": "zquoridor.search_game.v1", "run_id": run_id,
            **task, "players": list(names), "moves": [], "searches": [], "status": "failed"}
    players = []
    referee = local_arena.Referee()
    repetition = local_arena.RepetitionTracker()
    repetition.observe(referee)
    history = game["moves"]
    draw = False
    seed_text = f"{config.get('seed', 0)}:{task['pair_index']}:{task['zq_player']}:visit-temperature"
    rng = random.Random(int(hashlib.sha256(seed_text.encode()).hexdigest(), 16))
    try:
        for move in task["opening"]:
            referee.apply(move)
            history.append(move)
            if referee.winner is not None or repetition.observe(referee):
                raise local_arena.IllegalMove("the opening is terminal")
        for name in names:
            players.append(factories[name]())
        while referee.winner is None and not draw and len(history) < config["max_plies"]:
            side = referee.side_to_move
            name = names[side]
            schedule_ply = len(history) - (len(task["opening"]) if config["schedule_origin"] == "opening" else 0)
            budget = move_time_ms(schedule_ply, config)
            relative_ply = len(history) - len(task["opening"])
            temperature = (config.get("opening_temperature", 0.0)
                           if relative_ply < config.get("temperature_plies", 14) else 0.0)
            paired = config.get("record_both_searches", False)
            indices = [side, 1 - side] if paired else [side]
            current_searches = []
            for index in indices:
                engine_name = names[index]
                move, elapsed, info = players[index].bestmove(
                    list(history), budget=budget, timeout_s=config["move_timeout_s"])
                move = move.strip().lower()
                search = {"ply": len(history), "side_to_move": side, "player": engine_name,
                          "role": "mover" if index == side else "comparison",
                          "history": list(history), "move": move, "bestmove": move,
                          "budget_ms": budget, "schedule_ply": schedule_ply,
                          "elapsed_ms": float(elapsed) * 1000, "raw_info": list(info)}
                game["searches"].append(search)
                current_searches.append(search)
                if not referee.is_legal_move(move):
                    raise local_arena.IllegalMove(f"{engine_name}: illegal move {move!r}")
                search["best_action_global"] = global_action(move)
                if engine_name == "claustrophobia":
                    search["root"] = parse_root(info, referee, move, budget)
                elif paired or temperature > 0:
                    search["root"] = parse_zq_root(info, referee, move, budget)
            mover = current_searches[0]
            played, selection = sampled_move(mover.get("root"), mover["bestmove"], referee, temperature, rng)
            for search in current_searches:
                search.update(playedmove=played, played_action_global=global_action(played),
                              temperature=temperature, selection=selection)
            referee.apply(played)
            history.append(played)
            if referee.winner is None:
                draw = repetition.observe(referee)
        game.update(winner=referee.winner, plies=len(history),
                    repeated_states=repetition.repeated_states,
                    repetition_max_count=repetition.max_count)
        if referee.winner is not None or draw:
            game.update(status="ok", termination="goal" if referee.winner is not None else "repetition")
            for search in game["searches"]:
                search["outcome"] = (0.0 if referee.winner is None else
                                     1.0 if search["side_to_move"] == referee.winner else -1.0)
        else:
            game.update(status="truncated", termination="max_plies")
    except (Exception, KeyboardInterrupt) as error:
        game.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                    termination="error", error_type=type(error).__name__, error=str(error),
                    plies=len(history))
    finally:
        for player in players:
            with contextlib.suppress(Exception):
                player.close()
    return game


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, separators=(",", ":"), sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def ledger_game_id(path: Path) -> str:
    return path.name.removesuffix(".gz").removesuffix(".json")


def ledger_path(output: Path, task: dict, compressed: bool = False) -> Path:
    return output / "games" / (task_name(task) + (".json.gz" if compressed else ".json"))


def ledger_paths(output: Path) -> list[Path]:
    paths = sorted([*(output / "games").glob("*.json"), *(output / "games").glob("*.json.gz")])
    identifiers = [ledger_game_id(path) for path in paths]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("the game ledger contains duplicate compressed and raw identifiers")
    return paths


def read_game(path: Path) -> dict:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            return json.load(stream)
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_game(path: Path, game: dict) -> None:
    """Commit one raw or compressed game before a batch can include the game."""
    if path.suffix != ".gz":
        atomic_json(path, game)
        return
    temporary = path.with_suffix(".gz.tmp")
    with gzip.open(temporary, "wt", encoding="utf-8", newline="\n", compresslevel=6) as stream:
        json.dump(game, stream, separators=(",", ":"), sort_keys=True)
        stream.write("\n")
    with temporary.open("r+b") as stream:
        os.fsync(stream.fileno())
    temporary.replace(path)


@contextlib.contextmanager
def run_lock(output: Path) -> Iterator[None]:
    """Hold an operating system lock that survives no process termination."""
    output.mkdir(parents=True, exist_ok=True)
    stream = (output / ".run.lock").open("a+b")
    stream.seek(0)
    if not stream.read(1):
        stream.write(b"0")
        stream.flush()
    stream.seek(0)
    try:
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as error:
        stream.close()
        raise RuntimeError("another collector holds the output directory lock") from error
    try:
        yield
    finally:
        stream.seek(0)
        if os.name == "nt":
            msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        stream.close()


def task_name(task: dict) -> str:
    return f"{task['pair_index']:07d}_{task['zq_player']}"


def prepare_ledger(output: Path, manifest: dict, tasks: list[dict], resume: bool,
                   batched_games: set[str] | None = None, compress_game_ledger: bool = False) -> list[dict]:
    """Validate identity and mark interrupted claims before any further games."""
    manifest_path = output / "manifest.json"
    ledger = output / "games"
    if manifest_path.exists():
        if not resume:
            raise ValueError("the output contains a run; enable resume or use another directory")
        if json.loads(manifest_path.read_text(encoding="utf-8")).get("run_id") != manifest["run_id"]:
            raise ValueError("the output contains a different run identity")
    else:
        if any(path.name != ".run.lock" for path in output.iterdir()):
            raise ValueError("the output contains artifacts without a manifest")
        atomic_json(manifest_path, manifest)
    ledger.mkdir(exist_ok=True)
    pending = []
    for task in tasks:
        path = ledger_path(output, task, compress_game_ledger)
        claim = ledger / (task_name(task) + ".pending")
        if task_name(task) in (batched_games or set()):
            claim.unlink(missing_ok=True)
            continue
        if path.exists():
            row = read_game(path)
            if row.get("run_id") != manifest["run_id"]:
                raise ValueError(f"the game record has a different run identity: {path}")
            claim.unlink(missing_ok=True)
        elif claim.exists():
            row = json.loads(claim.read_text(encoding="utf-8"))
            if row.get("run_id") != manifest["run_id"]:
                raise ValueError(f"the interrupted claim has a different run identity: {claim}")
            atomic_game(path, {**row, "status": "interrupted", "termination": "process_interruption",
                               "moves": [], "searches": []})
            claim.unlink()
        else:
            pending.append(task)
    return pending


def training_positions(game: dict) -> Iterator[dict]:
    """Yield valid teacher roots only from complete goal or repetition games."""
    if game.get("status") != "ok":
        return
    for search in game.get("searches", []):
        root = search.get("root")
        if search.get("player") != "claustrophobia" or not root or root.get("target_status") != "ok":
            continue
        key = f"{game['run_id']}:{game['pair_index']}:{game['zq_player']}:{search['ply']}"
        yield {"schema": "zquoridor.position.v1",
               "id": hashlib.sha256(key.encode()).hexdigest()[:24],
               "source_position_id": sample_id(search["history"]),
               "source": "claustrophobia-in-game-search", "history": search["history"],
               "side_to_move": search["side_to_move"], "ply": search["ply"],
               "split": game["split"], "book": game["book"], "category": game["category"],
               "opening_index": game["pair_index"], "opening_group": game["opening_group"],
               "policy": root["policy"], "value": root["root_value"],
               "outcome": search["outcome"], "budget_ms": search["budget_ms"],
               "bestmove": search.get("bestmove", search["move"]),
               "playedmove": search.get("playedmove", search["move"]),
               "search_role": search.get("role", "mover"), "target_kind": "actual-root-visits",
               "policy_frame": "zquoridor-canonical-209",
               "metadata": {"run_id": game["run_id"], "game_id": task_name(game),
                            "search_role": search.get("role", "mover"),
                            "playedmove": search.get("playedmove", search["move"]),
                            "stop_reason": root["raw"].get("stop_reason"),
                            "value_perspective": "side-to-move",
                            "visit_count": sum(root["visit_counts_global"])}}


def export_run(output: Path, manifest: dict, config: dict, *,
               game_paths: Sequence[Path] | None = None, compressed: bool = False) -> dict:
    """Rebuild atomic exports from committed game records without another search."""
    paths = ledger_paths(output) if game_paths is None else list(game_paths)
    statuses: Counter = Counter()
    counts: Counter = Counter()
    position_count = 0
    for path in paths:
        game = read_game(path)
        statuses[game["status"]] += 1
        count = sum(1 for _ in training_positions(game))
        position_count += count
        counts[game["book"]] += count
    numpy = None
    if config["export_targets"] and position_count:
        import numpy
    with tempfile.TemporaryDirectory(prefix=".export_", dir=output) as temporary:
        temp = Path(temporary)
        arrays = {}
        if numpy is not None:
            for name, dtype, shape in (("id", "S24", (position_count,)),
                                       ("policy", "float32", (position_count, POLICY_DIM)),
                                       ("value", "float32", (position_count,)),
                                       ("game_result", "float32", (position_count,))):
                arrays[name] = numpy.lib.format.open_memmap(temp / (name + ".npy"),
                                                           mode="w+", dtype=dtype, shape=shape)
        index = 0
        names = [name + (".gz" if compressed else "")
                 for name in ("games.jsonl", "positions.jsonl", "labels.jsonl")]
        def open_text(name: str):
            if compressed:
                return gzip.open(temp / name, "wt", encoding="utf-8", newline="\n", compresslevel=6)
            return (temp / name).open("w", encoding="utf-8", newline="\n")
        with open_text(names[0]) as games, open_text(names[1]) as positions, open_text(names[2]) as labels:
            for path in paths:
                game = read_game(path)
                games.write(json.dumps(game, separators=(",", ":")) + "\n")
                for row in training_positions(game):
                    positions.write(json.dumps(row, separators=(",", ":")) + "\n")
                    label = make_label(row, teacher="claustrophobia", mode="in-game-search",
                                       policy=row["policy"], value=row["value"], bestmove=row["bestmove"],
                                       budget={"move_time_ms": row["budget_ms"]}, metadata=row["metadata"])
                    labels.write(json.dumps(label, separators=(",", ":")) + "\n")
                    if arrays:
                        arrays["id"][index] = row["id"].encode("ascii")
                        arrays["policy"][index] = row["policy"]
                        arrays["value"][index] = row["value"]
                        arrays["game_result"][index] = row["outcome"]
                    index += 1
        if arrays:
            save_arrays = numpy.savez_compressed if compressed else numpy.savez
            save_arrays(temp / "teacher_targets.npz", **arrays)
            for array in arrays.values():
                array.flush()
                array._mmap.close()
            arrays.clear()
            (temp / "teacher_targets.npz").replace(output / "teacher_targets.npz")
        else:
            (output / "teacher_targets.npz").unlink(missing_ok=True)
        for name in names:
            (temp / name).replace(output / name)
    summary = {"run_id": manifest["run_id"], "games": sum(statuses.values()),
               "statuses": dict(statuses), "training_positions": position_count,
               "positions_by_book": dict(counts), "retries": 0}
    atomic_json(output / "summary.json", summary)
    atomic_json(output / "teacher_targets.manifest.json",
                {"schema": "zquoridor.search_targets.manifest.v1", **summary,
                 "value_source": "actual in-game Claustrophobia root value",
                 "policy_source": "normalized actual in-game Claustrophobia root visits",
                 "policy_frame": "zquoridor-canonical-209",
                 "value_perspective": "side-to-move",
                 "targets_exported": bool(config["export_targets"] and position_count)})
    return summary


class BatchExporter:
    """Commit independent compressed batches and recover unbatched ledger records."""

    def __init__(self, output: Path, manifest: dict, config: dict) -> None:
        self.output = output
        self.manifest = manifest
        self.config = config
        self.directory = output / "batches"
        self.covered: set[str] = set()
        self.buffer: list[Path] = []
        self.buffer_ids: set[str] = set()
        self.statuses: Counter = Counter()
        self.positions_by_book: Counter = Counter()
        self.training_count = 0
        self.completed_batches = 0
        self.partial_batches = 0
        self.next_index = 0
        for path in sorted(self.directory.glob("batch_*/manifest.json")):
            batch = json.loads(path.read_text(encoding="utf-8"))
            if batch.get("schema") != "zquoridor.search_games.batch.v1" or batch.get("run_id") != manifest["run_id"]:
                raise ValueError(f"the saved batch has a different run identity: {path}")
            identifiers = batch["game_ids"]
            if len(identifiers) != len(set(identifiers)) or self.covered.intersection(identifiers):
                raise ValueError(f"the saved batch contains duplicate game identifiers: {path}")
            expected_id = hashlib.sha256(json.dumps({"run_id": manifest["run_id"],
                "game_ids": identifiers}, sort_keys=True).encode()).hexdigest()
            if (batch.get("batch_id") != expected_id or batch.get("game_count") != len(identifiers) or
                    sum(batch["summary"]["statuses"].values()) != len(identifiers) or
                    batch.get("completed") != (len(identifiers) == batch["configured_games"])):
                raise ValueError(f"the saved batch identity or counts do not match: {path}")
            for name, artifact in batch["artifacts"].items():
                file = path.parent / name
                if not file.is_file() or file.stat().st_size != artifact["size"]:
                    raise ValueError(f"a saved batch artifact is missing or incomplete: {file}")
            self._register(batch)
            self.next_index = max(self.next_index, int(batch["batch_index"]) + 1)

    def _register(self, batch: dict) -> None:
        self.covered.update(batch["game_ids"])
        self.statuses.update(batch["summary"]["statuses"])
        self.positions_by_book.update(batch["summary"]["positions_by_book"])
        self.training_count += int(batch["summary"]["training_positions"])
        self.completed_batches += int(batch["completed"])
        self.partial_batches += int(not batch["completed"])

    def add(self, path: Path) -> None:
        identifier = ledger_game_id(path)
        if identifier in self.covered or identifier in self.buffer_ids:
            return
        self.buffer.append(path)
        self.buffer_ids.add(identifier)
        if len(self.buffer) >= self.config.get("batch_games", 250):
            self.flush()

    def recover(self) -> None:
        """Read ledger filenames once and export only records absent from batches."""
        for path in ledger_paths(self.output):
            self.add(path)

    def flush(self) -> dict | None:
        if not self.buffer:
            return None
        self.directory.mkdir(parents=True, exist_ok=True)
        paths = sorted(self.buffer)
        identifiers = [ledger_game_id(path) for path in paths]
        for path in paths:
            if read_game(path).get("run_id") != self.manifest["run_id"]:
                raise ValueError(f"the unbatched game has a different run identity: {path}")
        identity = {"run_id": self.manifest["run_id"], "game_ids": identifiers}
        batch_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        with tempfile.TemporaryDirectory(prefix=".batch_export_", dir=self.directory) as temporary:
            stage = Path(temporary) / "payload"
            stage.mkdir()
            summary = export_run(stage, self.manifest, self.config, game_paths=paths, compressed=True)
            artifacts = {path.name: {"size": path.stat().st_size, "sha256": local_arena._sha256(path)}
                         for path in sorted(stage.iterdir()) if path.is_file()}
            batch = {"schema": "zquoridor.search_games.batch.v1", **identity,
                     "batch_id": batch_id, "batch_index": self.next_index,
                     "completed": len(paths) == self.config.get("batch_games", 250),
                     "configured_games": self.config.get("batch_games", 250),
                     "game_count": len(paths), "first_game_id": identifiers[0],
                     "last_game_id": identifiers[-1], "summary": summary, "artifacts": artifacts}
            atomic_json(stage / "manifest.json", batch)
            destination = self.directory / f"batch_{self.next_index:06d}"
            if destination.exists():
                raise ValueError(f"the batch destination already exists: {destination}")
            stage.replace(destination)
        self._register(batch)
        self.next_index += 1
        self.buffer.clear()
        self.buffer_ids.clear()
        self.write_summary()
        print(json.dumps({"saved_batch": str(destination), "games": len(paths),
                          "completed": batch["completed"], "batch_id": batch_id}), flush=True)
        return batch

    def write_summary(self) -> dict:
        summary = {"run_id": self.manifest["run_id"], "games": len(self.covered),
                   "statuses": dict(self.statuses), "training_positions": self.training_count,
                   "positions_by_book": dict(self.positions_by_book), "retries": 0,
                   "completed_batches": self.completed_batches, "partial_batches": self.partial_batches,
                   "batch_games": self.config.get("batch_games", 250)}
        atomic_json(self.output / "summary.json", summary)
        return summary


def engine_setup(config: dict) -> tuple[dict[str, Callable[[], object]], dict[str, Path]]:
    artifacts = {"collector": Path(__file__), "referee": ROOT / "tools/external/local_arena.py",
                 "target_helpers": ROOT / "training/teachers/targets.py",
                 "setup": ROOT / "tools/external/bot_setup.py"}
    factories = {}
    if config["mode"] != "claustrophobia-selfplay":
        executable = (Path(config["zq_executable"]).resolve() if config["zq_executable"] else
                      run_benchmark._build_zq(ROOT / "bin/local_benchmark" /
                                              ("zquoridor_uci.exe" if os.name == "nt" else "zquoridor_uci")))
        nnue = Path(config["nnue"]).resolve()
        artifacts.update(zq_executable=executable, nnue=nnue,
                         zq_adapter=ROOT / "tools/external/zquoridor_uci.cpp")
        artifacts.update({"header_" + path.name: path for path in (ROOT / "src").glob("*.hpp")})
        command = [str(executable), "--nnue", str(nnue), *config["zq_args"]]
        if config.get("record_both_searches") or config.get("opening_temperature", 0) > 0:
            if "--dump-root" not in command:
                command.append("--dump-root")
        factories["zquoridor"] = lambda: local_arena.UciPlayer(
            command, "zquoridor", startup_timeout_s=config["startup_timeout_s"])
    if config["mode"] != "zquoridor-selfplay":
        if config["claustrophobia_bridge"] and config["claustrophobia_checkpoint"]:
            info = {"benchmark_bridge": config["claustrophobia_bridge"],
                    "checkpoint": config["claustrophobia_checkpoint"]}
        else:
            info = run_benchmark._bot_info("claustrophobia", config["auto_setup"])
        bridge = Path(config["claustrophobia_bridge"] or info["benchmark_bridge"]).resolve()
        checkpoint = Path(config["claustrophobia_checkpoint"] or info["checkpoint"]).resolve()
        artifacts.update(claustrophobia_bridge=bridge, claustrophobia_checkpoint=checkpoint,
                         bridge_source=ROOT / "tools/external/claustrophobia_benchmark_bridge.rs",
                         inference_worker=ROOT / "training/teachers/claustrophobia_inference_worker.py")
        worker_copy = bridge.parent / "zq_inference_worker.py"
        if worker_copy.is_file():
            artifacts["inference_worker_copy"] = worker_copy
        factories["claustrophobia"] = lambda: local_arena.ClaustrophobiaPlayer(
            bridge, checkpoint, move_time_ms=config["start_move_time_ms"],
            max_sims=4294967295, cpuct=config["claustrophobia_cpuct"],
            device=config["claustrophobia_device"], startup_timeout_s=config["startup_timeout_s"])
    return factories, artifacts


def run(config: dict) -> dict:
    openings = select_openings(config)
    tasks = [{**opening, "zq_player": color} for opening in openings for color in (0, 1)]
    schedule = {key: config[key] for key in ("start_move_time_ms", "end_move_time_ms",
                                            "decay_start_ply", "decay_end_ply", "schedule_origin")}
    if config["dry_run"]:
        return {"dry_run": True, "planned_games": len(tasks), "pairs": len(openings),
                "mode": config["mode"], "workers": config["workers"], "schedule": schedule,
                "opening_quotas": opening_quotas(config),
                "opening_temperature": config.get("opening_temperature", 0.0),
                "temperature_plies": config.get("temperature_plies", 14),
                "record_both_searches": config.get("record_both_searches", False),
                "unique_openings_first": config.get("unique_openings_first", False),
                "batch_games": config.get("batch_games", 250),
                "export_final_run": config.get("export_final_run", True),
                "compress_game_ledger": config.get("compress_game_ledger", False),
                "pairs_by_book": dict(Counter(row["book"] for row in openings)),
                "unique_openings_by_book": {
                    name: len({row["opening_group"] for row in openings if row["book"] == name})
                    for name in config["opening_books"]},
                "unique_openings": len({row["opening_group"] for row in openings}),
                "repeated_opening_pairs": len(openings) - len({row["opening_group"] for row in openings}),
                "pairs_by_category": dict(Counter(f"{row['book']}:{row['category']}" for row in openings)),
                "selected_opening_digest": hashlib.sha256(json.dumps(openings, sort_keys=True).encode()).hexdigest(),
                "output": str(Path(config["output"]).resolve())}
    output = Path(config["output"]).resolve()
    with run_lock(output):
        factories, artifacts = engine_setup(config)
        artifacts.update({"book_" + name: Path(path) for name, path in config["opening_books"].items()})
        identity = {key: value for key, value in config.items()
                    if key not in ("output", "resume", "auto_setup", "dry_run", "export_targets", "export_final_run")}
        identity["selected_openings"] = openings
        identity["clock_semantics"] = "configured ply origin; linear decay; nearest millisecond; no retries"
        manifest = local_arena.make_manifest(identity, artifacts)
        manifest["schema"] = "zquoridor.search_games.manifest.v1"
        batches = BatchExporter(output, manifest, config)
        pending = prepare_ledger(output, manifest, tasks, config["resume"], batched_games=batches.covered,
                                 compress_game_ledger=config.get("compress_game_ledger", False))
        batches.recover()
        print(json.dumps({"run_id": manifest["run_id"], "pending_games": len(pending),
                          "workers": config["workers"], "schedule": schedule}), flush=True)

        def one(task: dict) -> dict:
            name = task_name(task)
            claim = output / "games" / (name + ".pending")
            atomic_json(claim, {"run_id": manifest["run_id"], **task})
            game = play_game(task, config, manifest["run_id"], factories)
            atomic_game(ledger_path(output, task, config.get("compress_game_ledger", False)), game)
            claim.unlink()
            return game

        try:
            if config["workers"] == 1:
                for task in pending:
                    game = one(task)
                    batches.add(ledger_path(output, game, config.get("compress_game_ledger", False)))
                    print(f"Game {task_name(game)}: {game['status']}", flush=True)
                    if game["status"] == "interrupted":
                        break
            else:
                # Submit only active workers so an interruption cannot launch queued games.
                with concurrent.futures.ThreadPoolExecutor(max_workers=config["workers"]) as executor:
                    remaining = iter(pending)
                    active = {executor.submit(one, task) for task in
                              [next(remaining, None) for _ in range(config["workers"])] if task is not None}
                    while active:
                        completed, active = concurrent.futures.wait(
                            active, return_when=concurrent.futures.FIRST_COMPLETED)
                        for future in completed:
                            game = future.result()
                            batches.add(ledger_path(output, game, config.get("compress_game_ledger", False)))
                            print(f"Game {task_name(game)}: {game['status']}", flush=True)
                            task = next(remaining, None)
                            if task is not None:
                                active.add(executor.submit(one, task))
        finally:
            batches.recover()
            batches.flush()
            summary = batches.write_summary()
            if config.get("export_final_run", True):
                if len(ledger_paths(output)) != len(batches.covered):
                    raise ValueError("final exports require every individual game ledger record; "
                                     "the compressed batches retain saved games; use --no-export-final-run")
                export_run(output, manifest, config)
                atomic_json(output / "summary.json", summary)
        return summary


def main(argv: Sequence[str] | None = None) -> int:
    try:
        config = resolve_config(build_parser().parse_args(argv))
        result = run(config)
        print(json.dumps(result, indent=2, sort_keys=True), flush=True)
        return 1 if any(result.get("statuses", {}).get(key) for key in ("failed", "interrupted", "truncated")) else 0
    except (OSError, ValueError, RuntimeError) as error:
        print(f"Search game collection failed: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Search game collection was interrupted. Resume excludes interrupted games.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
