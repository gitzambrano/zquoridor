#!/usr/bin/env python3
"""Prove that the controlled LMR experiment actually executes in production AB leaves.

No --no-mcab. All fixtures are legal 0/8 positions reached by normal game moves.
The standard LMR depth-3 threshold cannot fire inside default 2-ply leaves;
this test rejects an experiment where the modified threshold also does nothing.
"""
import argparse
import json
import re
import subprocess
from pathlib import Path


def measure(engine: str, row: dict, args: list[str]) -> tuple[int, int, int]:
    cmd = ["position startpos moves " + " ".join(row["moves"]), "go movetime 200", "quit"]
    completed = subprocess.run([engine, *args], input="\n".join(cmd) + "\n",
                               capture_output=True, text=True, check=True, timeout=30)
    if "info string error" in completed.stdout or "bestmove " not in completed.stdout:
        raise AssertionError(completed.stdout + "\n" + completed.stderr)
    traces = [s for s in completed.stdout.splitlines() if "lmrReductions=" in s]
    if not traces:
        raise AssertionError("LMR instrumentation absent:\n" + completed.stdout)
    trace = traces[-1]
    def get(name):
        match = re.search(r"\b" + name + r"=(\d+)", trace)
        if not match:
            raise AssertionError(f"{name} absent: {trace}")
        return int(match.group(1))
    return get("lmrReductions"), get("abNodes"), get("abLeaves")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--engine", required=True)
    p.add_argument("--book", required=True)
    args = p.parse_args()
    rows = [json.loads(x) for x in Path(args.book).read_text().splitlines() if x.strip()]
    rows = rows[:12]
    if not rows or any(x["wall_stocks"] != [0, 8] for x in rows):
        raise AssertionError("Expected nonterminal 0/8 production gate fixtures")
    baseline = [measure(args.engine, row, []) for row in rows]
    modified = [measure(args.engine, row, [
        "--lmr-min-depth", "2", "--lmr-min-move", "2", "--lmr-divisor", "1.20"
    ]) for row in rows]
    assert all(c[0] == 0 for c in baseline), f"LMR baseline unexpectedly active: {baseline}"
    assert all(c[2] > 0 for c in baseline), f"production endgame AB leaves never invoked: {baseline}"
    assert sum(c[0] for c in modified) > 0, f"Experimental LMR never fired: {modified}"
    print("AB endgame production-gate check PASS", flush=True)
    print(f"baseline reductions=0, leaves={sum(x[2] for x in baseline)}, "
          f"AB nodes={sum(x[1] for x in baseline)}", flush=True)
    print(f"experimental reductions={sum(x[0] for x in modified)}, "
          f"leaves={sum(x[2] for x in modified)}, "
          f"AB nodes={sum(x[1] for x in modified)}", flush=True)


if __name__ == "__main__":
    main()
