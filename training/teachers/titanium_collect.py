#!/usr/bin/env python3
"""Collect architecture-neutral policy-teacher data from pinned Titanium.

The output is raw JSONL: it deliberately does not depend on SAMPLE_DTYPE,
feature encoding, network dimensions, or a particular trainer.  A later
converter/trainer decides how to encode each position and target.

The frozen Titanium benchmark openings are rejected as teacher input so the
historical external benchmark remains independent from training data.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import sys
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[2]
EXTERNAL = ROOT / "tools" / "external"
sys.path.insert(0, str(EXTERNAL))

import titanium_arena_fixed as titanium_fixed  # noqa: E402

FROZEN_BENCHMARK_OPENINGS = (EXTERNAL / "openings_titanium.jsonl").resolve()


def load_openings(path: Path) -> list[list[str]]:
    if path.resolve() == FROZEN_BENCHMARK_OPENINGS:
        raise SystemExit(
            "refusing to use frozen benchmark openings as teacher data; "
            "provide an independent opening corpus"
        )
    openings: list[list[str]] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        obj = json.loads(line)
        moves = obj.get("moves")
        if not isinstance(moves, list) or not all(isinstance(m, str) for m in moves):
            raise SystemExit(f"{path}:{lineno}: expected JSON object with string list 'moves'")
        openings.append(moves)
    if not openings:
        raise SystemExit(f"no openings found in {path}")
    return openings


def collect_game(index: int, opening: Sequence[str], titanium: str,
                 movetime_ms: int, max_plies: int, out_dir: Path,
                 val_mod: int) -> dict:
    split = "val" if index % val_mod == 0 else "train"
    history = list(opening)
    engine = titanium_fixed.arena.UCIEngine([titanium, "uci"], "titanium")
    samples = 0
    termination = "max_plies"
    shard = out_dir / split / f"opening_{index:06d}.jsonl"
    shard.parent.mkdir(parents=True, exist_ok=True)
    try:
        with shard.open("w", encoding="utf-8") as fh:
            while len(history) < max_plies:
                move, think_s, info = engine.bestmove(history, movetime_ms)
                if move == "(none)":
                    termination = "no_move"
                    break
                rec = {
                    "schema": "zquoridor.titanium_teacher.raw.v1",
                    "opening_index": index,
                    "split": split,
                    "sample_index": samples,
                    "ply": len(history),
                    "side_to_move": len(history) & 1,
                    "history": history,
                    "bestmove": move,
                    "movetime_ms": movetime_ms,
                    "think_s": think_s,
                    "info": info,
                }
                fh.write(json.dumps(rec, separators=(",", ":")) + "\n")
                history.append(move)
                samples += 1
    finally:
        engine.close()
    return {
        "opening_index": index,
        "split": split,
        "opening_plies": len(opening),
        "final_plies": len(history),
        "samples": samples,
        "termination": termination,
        "shard": str(shard.relative_to(out_dir)),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--titanium", required=True)
    ap.add_argument("--openings", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--movetime", type=int, default=200)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--max-plies", type=int, default=180)
    ap.add_argument("--val-mod", type=int, default=5,
                    help="whole-opening validation split: opening_index %% val_mod == 0")
    args = ap.parse_args()

    if args.movetime <= 0 or args.workers <= 0 or args.max_plies <= 0 or args.val_mod < 2:
        raise SystemExit("movetime/workers/max-plies must be positive and val-mod >= 2")

    openings_path = Path(args.openings)
    openings = load_openings(openings_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [
            ex.submit(
                collect_game, i, opening, args.titanium, args.movetime,
                args.max_plies, out_dir, args.val_mod
            )
            for i, opening in enumerate(openings)
        ]
        for done, fut in enumerate(cf.as_completed(futures), 1):
            result = fut.result()
            results.append(result)
            print(
                f"teacher [{done}/{len(futures)}] opening={result['opening_index']} "
                f"split={result['split']} samples={result['samples']}",
                flush=True,
            )

    results.sort(key=lambda x: x["opening_index"])
    games_path = out_dir / "games.jsonl"
    with games_path.open("w", encoding="utf-8") as fh:
        for result in results:
            fh.write(json.dumps(result, separators=(",", ":")) + "\n")

    manifest = {
        "schema": "zquoridor.titanium_teacher.manifest.v1",
        "teacher": "Titanium",
        "teacher_binary": str(Path(args.titanium)),
        "openings": str(openings_path),
        "benchmark_openings_used": False,
        "movetime_ms": args.movetime,
        "workers": args.workers,
        "max_plies": args.max_plies,
        "val_mod": args.val_mod,
        "games": len(results),
        "train_games": sum(r["split"] == "train" for r in results),
        "val_games": sum(r["split"] == "val" for r in results),
        "samples": sum(r["samples"] for r in results),
        "target": "raw bestmove + raw info lines",
        "encoding": "architecture-neutral move history",
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
