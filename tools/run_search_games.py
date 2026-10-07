#!/usr/bin/env python3
"""Collect complete games and the search targets that Claustrophobia used in each game."""
from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
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
    "dry_run": False,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("match", "zquoridor-selfplay", "claustrophobia-selfplay"))
    parser.add_argument("--opening-book", action="append", dest="opening_books", metavar="NAME=PATH",
                        help="Replace the default books. Repeat this option for each book.")
    for key in ("pairs", "workers", "seed", "start_move_time_ms", "end_move_time_ms",
                "decay_start_ply", "decay_end_ply", "max_plies"):
        parser.add_argument("--" + key.replace("_", "-"), type=int)
    for key in ("move_timeout_s", "startup_timeout_s", "claustrophobia_cpuct", "validation_fraction"):
        parser.add_argument("--" + key.replace("_", "-"), type=float)
    for key in ("output", "zq_executable", "nnue", "claustrophobia_bridge", "claustrophobia_checkpoint"):
        parser.add_argument("--" + key.replace("_", "-"))
    parser.add_argument("--claustrophobia-device", choices=("cpu", "gpu"))
    parser.add_argument("--schedule-origin", choices=("opening", "game"))
    parser.add_argument("--zq-arg", dest="zq_args", action="append")
    for key in ("auto_setup", "resume", "export_targets", "dry_run"):
        parser.add_argument("--" + key.replace("_", "-"), action=argparse.BooleanOptionalAction, default=None)
    return parser


def resolve_config(args: argparse.Namespace) -> dict:
    config = {**CONFIG, "opening_books": dict(CONFIG["opening_books"]), "zq_args": list(CONFIG["zq_args"])}
    config.update({key: value for key, value in vars(args).items() if value is not None})
    if isinstance(config["opening_books"], list):
        books = {}
        for text in config["opening_books"]:
            name, separator, path = text.partition("=")
            if not separator or not name.strip() or not path.strip() or name in books:
                raise ValueError("each opening book must have a unique NAME=PATH")
            books[name] = path
        config["opening_books"] = books
    for key in ("pairs", "workers", "start_move_time_ms", "end_move_time_ms", "max_plies"):
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


def select_openings(config: dict) -> list[dict]:
    """Balance books and categories, then cycle shuffled rows as necessary."""
    rng = random.Random(config["seed"])
    books = {name: read_book(name, Path(path).resolve())
             for name, path in sorted(config["opening_books"].items())}
    names = list(books)
    rng.shuffle(names)
    categories = {}
    for name, groups in books.items():
        categories[name] = list(sorted(groups))
        rng.shuffle(categories[name])
        for rows in groups.values():
            rng.shuffle(rows)
    book_counts: Counter = Counter()
    group_counts: Counter = Counter()
    selected = []
    validated: set[tuple[str, ...]] = set()
    for pair_index in range(config["pairs"]):
        name = names[pair_index % len(names)]
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
            move, elapsed, info = players[side].bestmove(
                list(history), budget=budget, timeout_s=config["move_timeout_s"])
            move = move.strip().lower()
            search = {"ply": len(history), "side_to_move": side, "player": name,
                      "history": list(history), "move": move, "budget_ms": budget,
                      "schedule_ply": schedule_ply,
                      "elapsed_ms": float(elapsed) * 1000, "raw_info": list(info)}
            game["searches"].append(search)
            if not referee.is_legal_move(move):
                raise local_arena.IllegalMove(f"{name}: illegal move {move!r}")
            search["best_action_global"] = global_action(move)
            if name == "claustrophobia":
                search["root"] = parse_root(info, referee, move, budget)
            referee.apply(move)
            history.append(move)
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


def prepare_ledger(output: Path, manifest: dict, tasks: list[dict], resume: bool) -> list[dict]:
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
        path = ledger / (task_name(task) + ".json")
        claim = ledger / (task_name(task) + ".pending")
        if path.exists():
            row = json.loads(path.read_text(encoding="utf-8"))
            if row.get("run_id") != manifest["run_id"]:
                raise ValueError(f"the game record has a different run identity: {path}")
            claim.unlink(missing_ok=True)
        elif claim.exists():
            row = json.loads(claim.read_text(encoding="utf-8"))
            if row.get("run_id") != manifest["run_id"]:
                raise ValueError(f"the interrupted claim has a different run identity: {claim}")
            atomic_json(path, {**row, "status": "interrupted", "termination": "process_interruption",
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
        if not root or root.get("target_status") != "ok":
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
               "bestmove": search["move"], "target_kind": "actual-root-visits",
               "policy_frame": "zquoridor-canonical-209",
               "metadata": {"run_id": game["run_id"], "game_id": task_name(game),
                            "stop_reason": root["raw"].get("stop_reason"),
                            "value_perspective": "side-to-move",
                            "visit_count": sum(root["visit_counts_global"])}}


def export_run(output: Path, manifest: dict, config: dict) -> dict:
    """Rebuild atomic exports from committed game records without another search."""
    paths = sorted((output / "games").glob("*.json"))
    statuses: Counter = Counter()
    counts: Counter = Counter()
    position_count = 0
    for path in paths:
        game = json.loads(path.read_text(encoding="utf-8"))
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
        with (temp / "games.jsonl").open("w", encoding="utf-8", newline="\n") as games, \
                (temp / "positions.jsonl").open("w", encoding="utf-8", newline="\n") as positions, \
                (temp / "labels.jsonl").open("w", encoding="utf-8", newline="\n") as labels:
            for path in paths:
                game = json.loads(path.read_text(encoding="utf-8"))
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
            numpy.savez(temp / "teacher_targets.npz", **arrays)
            for array in arrays.values():
                array.flush()
                array._mmap.close()
            arrays.clear()
            (temp / "teacher_targets.npz").replace(output / "teacher_targets.npz")
        else:
            (output / "teacher_targets.npz").unlink(missing_ok=True)
        for name in ("games.jsonl", "positions.jsonl", "labels.jsonl"):
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
                "pairs_by_book": dict(Counter(row["book"] for row in openings)),
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
                    if key not in ("output", "resume", "auto_setup", "dry_run", "export_targets")}
        identity["selected_openings"] = openings
        identity["clock_semantics"] = "configured ply origin; linear decay; nearest millisecond; no retries"
        manifest = local_arena.make_manifest(identity, artifacts)
        manifest["schema"] = "zquoridor.search_games.manifest.v1"
        pending = prepare_ledger(output, manifest, tasks, config["resume"])
        print(json.dumps({"run_id": manifest["run_id"], "pending_games": len(pending),
                          "workers": config["workers"], "schedule": schedule}), flush=True)

        def one(task: dict) -> dict:
            name = task_name(task)
            claim = output / "games" / (name + ".pending")
            atomic_json(claim, {"run_id": manifest["run_id"], **task})
            game = play_game(task, config, manifest["run_id"], factories)
            atomic_json(output / "games" / (name + ".json"), game)
            claim.unlink()
            return game

        try:
            if config["workers"] == 1:
                for task in pending:
                    game = one(task)
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
                            print(f"Game {task_name(game)}: {game['status']}", flush=True)
                            task = next(remaining, None)
                            if task is not None:
                                active.add(executor.submit(one, task))
        finally:
            summary = export_run(output, manifest, config)
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
