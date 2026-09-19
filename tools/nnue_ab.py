#!/usr/bin/env python3
"""Paired A/B benchmark for two configurations of the same ZQuoridor UCI binary."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import run_benchmark
from tools.external import local_arena


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--exe", required=True)
    p.add_argument("--nnue", required=True)
    p.add_argument("--candidate-arg", action="append", default=[])
    p.add_argument("--baseline-arg", action="append", default=[])
    p.add_argument("--pairs", type=int, default=40)
    p.add_argument("--time", type=int, default=200)
    p.add_argument("--seed", type=int, default=20260919)
    p.add_argument("--openings", default="tools/external/openings_confirmation_v1.jsonl")
    p.add_argument("--output", required=True)
    p.add_argument("--bootstrap", type=int, default=20000)
    p.add_argument("--max-plies", type=int, default=240)
    p.add_argument("--move-timeout-s", type=float, default=30.0)
    return p


def main() -> int:
    a = parser().parse_args()
    if a.pairs <= 0 or a.time <= 0:
        raise SystemExit("pairs/time must be positive")

    exe = Path(a.exe).resolve()
    nnue = Path(a.nnue).resolve()
    openings_path = Path(a.openings).resolve()
    out = Path(a.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if not exe.is_file() or not nnue.is_file() or not openings_path.is_file():
        raise SystemExit("missing executable, NNUE, or openings file")

    openings = run_benchmark._read_openings(openings_path, a.pairs, a.seed)
    identity = {
        "schema": "zquoridor.nnue_ab.v1",
        "exe": str(exe),
        "nnue": str(nnue),
        "candidate_args": a.candidate_arg,
        "baseline_args": a.baseline_arg,
        "pairs": a.pairs,
        "time_ms": a.time,
        "seed": a.seed,
        "openings": str(openings_path),
        "max_plies": a.max_plies,
    }
    run_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    manifest_path = out / "manifest.json"
    games_path = out / "games.jsonl"

    if manifest_path.exists():
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        if old.get("run_id") != run_id:
            raise SystemExit("output directory belongs to a different A/B run")
    else:
        manifest_path.write_text(json.dumps({**identity, "run_id": run_id}, indent=2) + "\n", encoding="utf-8")

    existing = {}
    if games_path.exists():
        for line in games_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            existing[(int(row["opening_index"]), int(row["zq_player"]))] = row

    candidate_cmd = [str(exe), "--nnue", str(nnue), *a.candidate_arg]
    baseline_cmd = [str(exe), "--nnue", str(nnue), *a.baseline_arg]

    with games_path.open("a", encoding="utf-8") as sink:
        for opening_index, opening in openings:
            for side in (0, 1):
                key = (int(opening_index), side)
                if key in existing and existing[key].get("status") == "ok":
                    continue
                row = local_arena.play_game(
                    opponent="baseline",
                    opening_index=int(opening_index),
                    opening=opening,
                    zq_player=side,
                    zq_factory=lambda cmd=candidate_cmd: local_arena.UciPlayer(cmd, "candidate"),
                    opponent_factory=lambda cmd=baseline_cmd: local_arena.UciPlayer(cmd, "baseline"),
                    zq_budget=a.time,
                    opponent_budget=a.time,
                    move_timeout_s=a.move_timeout_s,
                    max_plies=a.max_plies,
                    run_id=run_id,
                )
                sink.write(json.dumps(row, separators=(",", ":")) + "\n")
                sink.flush()
                existing[key] = row

    rows = list(existing.values())
    summary = local_arena.summarize_pairs(rows, bootstrap=a.bootstrap, seed=a.seed ^ 0x9E3779B9)
    payload = {**identity, "run_id": run_id, "summary": summary}
    (out / "summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
