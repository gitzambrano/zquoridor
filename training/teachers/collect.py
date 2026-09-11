#!/usr/bin/env python3
"""Collect architecture-neutral data from a Quoridor teacher engine."""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EXTERNAL = ROOT / "tools" / "external"
sys.path.insert(0, str(EXTERNAL))
sys.path.insert(0, str(HERE))

import titanium_arena_fixed as titanium_fixed  # noqa: E402
from common import (  # noqa: E402
    deepest_rich_snapshot,
    load_independent_openings,
    parse_budgets,
    reaches_goal,
    summarize_probes,
)

FROZEN_BENCHMARK_OPENINGS = (EXTERNAL / "openings_titanium.jsonl").resolve()


@dataclass(frozen=True)
class CollectorConfig:
    """Store one collection job configuration."""

    teacher: str
    protocol: str
    command: tuple[str, ...]
    budgets: tuple[int, ...]
    repeats: int
    max_plies: int
    out_dir: Path
    val_mod: int
    student: str | None = None
    student_protocol: str = "uci"
    student_command: tuple[str, ...] = ()
    student_movetime_ms: int = 200


def make_engine(name: str, protocol: str, command: Sequence[str]):
    """Start one engine process."""
    engine_name = "titanium" if protocol == "titanium" else name
    return titanium_fixed.arena.UCIEngine(list(command), engine_name)


def make_teacher_streams(config: CollectorConfig) -> list[tuple[int, int, object]]:
    """Start one independent engine stream for each budget and repeat."""
    streams: list[tuple[int, int, object]] = []
    try:
        for budget in config.budgets:
            for repeat in range(config.repeats):
                engine = make_engine(config.teacher, config.protocol, config.command)
                streams.append((budget, repeat, engine))
    except Exception:
        for _, _, engine in streams:
            engine.close()
        raise
    return streams


def probe_teacher(streams: Sequence[tuple[int, int, object]], history: Sequence[str]) -> list[dict]:
    """Query every independent teacher stream at one position."""
    probes: list[dict] = []
    for budget, repeat, engine in streams:
        move, think_s, info = engine.bestmove(history, budget)
        probes.append(
            {
                "movetime_ms": budget,
                "repeat": repeat,
                "bestmove": move,
                "think_s": think_s,
                "info": info,
                "rich": deepest_rich_snapshot(info),
            }
        )
    return probes


def probe_student(engine, history: Sequence[str], movetime_ms: int) -> dict:
    """Query the student once at the teacher position."""
    move, think_s, info = engine.bestmove(history, movetime_ms)
    return {
        "movetime_ms": movetime_ms,
        "bestmove": move,
        "think_s": think_s,
        "info": info,
        "rich": deepest_rich_snapshot(info),
    }


def representative_probe(probes: Sequence[dict], diagnostics: dict) -> dict:
    """Return one highest-budget probe that supports the selected move."""
    budget = int(diagnostics["highest_budget_ms"])
    move = diagnostics["selected_move"]
    candidates = [
        probe
        for probe in probes
        if int(probe["movetime_ms"]) == budget and probe.get("bestmove") == move
    ]
    if candidates:
        return candidates[0]
    return probes[-1]


def collect_game(index: int, opening: Sequence[str], config: CollectorConfig) -> dict:
    """Collect one teacher trajectory and write one shard."""
    split = "val" if index % config.val_mod == 0 else "train"
    history = list(opening)
    samples = 0
    disagreements = 0
    termination = "max_plies"
    shard = config.out_dir / split / f"opening_{index:06d}.jsonl"
    shard.parent.mkdir(parents=True, exist_ok=True)
    teacher_streams = make_teacher_streams(config)
    student_engine = None
    if config.student is not None:
        student_engine = make_engine(
            config.student,
            config.student_protocol,
            config.student_command,
        )
    try:
        with shard.open("w", encoding="utf-8") as fh:
            while len(history) < config.max_plies:
                pre_move_ply = len(history)
                probes = probe_teacher(teacher_streams, history)
                diagnostics = summarize_probes(probes, config.budgets)
                move = diagnostics["selected_move"]
                if move == "(none)":
                    termination = "no_move"
                    break

                student_probe = None
                if student_engine is not None:
                    student_probe = probe_student(
                        student_engine,
                        history,
                        config.student_movetime_ms,
                    )
                    student_move = student_probe["bestmove"]
                    student_agrees = student_move == move
                    diagnostics["student_bestmove"] = student_move
                    diagnostics["student_agrees"] = student_agrees
                    if not student_agrees:
                        disagreements += 1

                selected = representative_probe(probes, diagnostics)
                record = {
                    "schema": "zquoridor.teacher.raw.v2",
                    "teacher": config.teacher,
                    "protocol": config.protocol,
                    "opening_index": index,
                    "split": split,
                    "sample_index": samples,
                    "ply": pre_move_ply,
                    "side_to_move": pre_move_ply & 1,
                    "history": history,
                    "bestmove": move,
                    "movetime_ms": selected["movetime_ms"],
                    "think_s": selected["think_s"],
                    "info": selected["info"],
                    "rich": selected["rich"],
                    "diagnostics": diagnostics,
                    "probes": probes,
                }
                if student_probe is not None:
                    record["student"] = config.student
                    record["student_probe"] = student_probe
                fh.write(json.dumps(record, separators=(",", ":")) + "\n")
                history.append(move)
                samples += 1
                if reaches_goal(pre_move_ply, move):
                    termination = "goal"
                    break
    finally:
        for _, _, engine in teacher_streams:
            engine.close()
        if student_engine is not None:
            student_engine.close()

    return {
        "opening_index": index,
        "split": split,
        "opening_plies": len(opening),
        "final_plies": len(history),
        "samples": samples,
        "disagreements": disagreements,
        "termination": termination,
        "shard": str(shard.relative_to(config.out_dir)),
    }


def run_collection(config: CollectorConfig, openings: Sequence[Sequence[str]], workers: int) -> dict:
    """Collect all trajectories and write the corpus manifest."""
    config.out_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []
    with cf.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(collect_game, index, opening, config)
            for index, opening in enumerate(openings)
        ]
        for done, future in enumerate(cf.as_completed(futures), 1):
            result = future.result()
            results.append(result)
            print(
                f"teacher [{done}/{len(futures)}] opening={result['opening_index']} "
                f"split={result['split']} samples={result['samples']} "
                f"disagreements={result['disagreements']} "
                f"termination={result['termination']}",
                flush=True,
            )

    results.sort(key=lambda item: item["opening_index"])
    games_path = config.out_dir / "games.jsonl"
    with games_path.open("w", encoding="utf-8") as fh:
        for result in results:
            fh.write(json.dumps(result, separators=(",", ":")) + "\n")

    sample_count = sum(item["samples"] for item in results)
    disagreement_count = sum(item["disagreements"] for item in results)
    manifest = {
        "schema": "zquoridor.teacher.manifest.v2",
        "teacher": config.teacher,
        "protocol": config.protocol,
        "command": list(config.command),
        "benchmark_openings_used": False,
        "budgets_ms": list(config.budgets),
        "repeats": config.repeats,
        "workers": workers,
        "max_plies": config.max_plies,
        "val_mod": config.val_mod,
        "games": len(results),
        "train_games": sum(item["split"] == "train" for item in results),
        "val_games": sum(item["split"] == "val" for item in results),
        "samples": sample_count,
        "terminations": {
            key: sum(item["termination"] == key for item in results)
            for key in sorted({item["termination"] for item in results})
        },
        "target": "best move plus repeated multi-budget raw probes",
        "encoding": "architecture-neutral move history",
    }
    if config.student is not None:
        manifest["student"] = {
            "name": config.student,
            "protocol": config.student_protocol,
            "command": list(config.student_command),
            "movetime_ms": config.student_movetime_ms,
            "disagreements": disagreement_count,
            "disagreement_fraction": disagreement_count / max(1, sample_count),
        }
    (config.out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def _build_command(engine: str, engine_args: Sequence[str], protocol: str) -> tuple[str, ...]:
    """Build one executable command from CLI fields."""
    command = [engine, *engine_args]
    if protocol == "titanium":
        command.append("uci")
    return tuple(command)


def main(argv: Sequence[str] | None = None) -> int:
    """Parse CLI options and collect the teacher corpus."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--protocol", choices=("uci", "titanium"), default="uci")
    parser.add_argument("--engine", required=True)
    parser.add_argument("--engine-arg", action="append", default=[])
    parser.add_argument("--student")
    parser.add_argument("--student-protocol", choices=("uci", "titanium"), default="uci")
    parser.add_argument("--student-engine")
    parser.add_argument("--student-engine-arg", action="append", default=[])
    parser.add_argument("--student-movetime", type=int, default=200)
    parser.add_argument("--openings", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--budgets", default="200")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-plies", type=int, default=180)
    parser.add_argument("--val-mod", type=int, default=5)
    args = parser.parse_args(argv)

    try:
        budgets = parse_budgets(args.budgets)
        openings = load_independent_openings(Path(args.openings), FROZEN_BENCHMARK_OPENINGS)
    except (ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(str(exc)) from exc

    if args.repeats <= 0 or args.workers <= 0 or args.max_plies <= 0 or args.val_mod < 2:
        raise SystemExit(
            "repeats, workers, and max-plies must be positive; val-mod must be at least 2"
        )
    if args.student_movetime <= 0:
        raise SystemExit("student-movetime must be positive")
    if (args.student is None) != (args.student_engine is None):
        raise SystemExit("set both --student and --student-engine, or set neither")

    student_command: tuple[str, ...] = ()
    if args.student is not None:
        student_command = _build_command(
            args.student_engine,
            args.student_engine_arg,
            args.student_protocol,
        )

    config = CollectorConfig(
        teacher=args.teacher,
        protocol=args.protocol,
        command=_build_command(args.engine, args.engine_arg, args.protocol),
        budgets=tuple(budgets),
        repeats=args.repeats,
        max_plies=args.max_plies,
        out_dir=Path(args.out_dir),
        val_mod=args.val_mod,
        student=args.student,
        student_protocol=args.student_protocol,
        student_command=student_command,
        student_movetime_ms=args.student_movetime,
    )
    manifest = run_collection(config, openings, args.workers)
    print(json.dumps(manifest, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
