#!/usr/bin/env python3
"""Aggregate a strictly complete 400-game paired search-only tournament."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools.external import local_arena

CONFIG = {
    "input": "results/search400-shards",
    "output": "results/search400-aggregate",
    "expected_books": ("normal", "center_rush"),
    "pairs_per_book": 100,
    "expected_shards": 10,
    "bootstrap": 10000,
    "seed": 20261117,
}


def run(cfg: dict) -> dict:
    input_dir = ROOT / cfg["input"]
    paths = sorted(input_dir.rglob("games.jsonl"))
    if len(paths) != cfg["expected_shards"]:
        raise RuntimeError(f"Expected {cfg['expected_shards']} game shards, found {len(paths)}")
    rows = []
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    target = cfg["pairs_per_book"] * 2 * len(cfg["expected_books"])
    if len(rows) != target:
        raise RuntimeError(f"Expected {target} games, found {len(rows)}")
    seen = set()
    for row in rows:
        if row.get("status") != "ok":
            raise RuntimeError(f"Failed game at opening {row.get('opening_index')}: {row.get('error')}")
        if row.get("ponder_calls", -1) != 0:
            raise RuntimeError("Pondering was enabled in a game")
        if row.get("book") not in cfg["expected_books"]:
            raise RuntimeError(f"Unexpected book: {row.get('book')}")
        key = (row["book"], row["opening_index"], row["zq_player"])
        if key in seen:
            raise RuntimeError(f"Duplicated color/opening: {key}")
        seen.add(key)
        if not all(move.get("clock_protocol") == "fixed-movetime-v2"
                   for move in row["move_times"]):
            raise RuntimeError("A game used a non-fixed-movetime protocol")
    report = local_arena.summarize_pairs(rows, bootstrap=cfg["bootstrap"], seed=cfg["seed"])
    by_book = {}
    for book in cfg["expected_books"]:
        subset = [row for row in rows if row["book"] == book]
        pairs = Counter(row["opening_index"] for row in subset)
        if len(subset) != 2 * cfg["pairs_per_book"] or len(pairs) != cfg["pairs_per_book"]:
            raise RuntimeError(f"{book}: invalid number of games or openings")
        if any(count != 2 for count in pairs.values()):
            raise RuntimeError(f"{book}: incomplete color-reversed opening")
        book_report = local_arena.summarize_pairs(
            subset, bootstrap=cfg["bootstrap"], seed=cfg["seed"] + len(by_book) + 1)
        if book_report["complete_pairs"] != cfg["pairs_per_book"] or book_report["failed_games"]:
            raise RuntimeError(f"{book}: failed pair summary")
        by_book[book] = book_report
    if report["complete_pairs"] != cfg["pairs_per_book"] * len(cfg["expected_books"]):
        raise RuntimeError("Incomplete overall pair summary")
    report["by_book"] = by_book
    report["source_shards"] = [str(path.relative_to(ROOT)) for path in paths]
    report["candidate"] = "4.10 search + identical Apex INT8 weights"
    report["baseline"] = "4.01 search + identical Apex INT8 weights"
    report["protocol"] = "paired 200ms fixed UCI move time; no pondering"
    output = ROOT / cfg["output"]
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("BOOK_SUMMARY " + json.dumps({
        book: {"games": 2 * cfg["pairs_per_book"], "score_pct": value["score_pct"],
               "elo": value["elo"], "paired_bootstrap_95": value.get("paired_bootstrap_95")}
        for book, value in by_book.items()
    }), flush=True)
    print("SUMMARY " + json.dumps({
        key: report.get(key)
        for key in ("included_games", "complete_pairs", "failed_games", "score_pct",
                    "elo", "paired_bootstrap_95", "strength_claim_ready")
    }), flush=True)
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default=CONFIG["input"])
    ap.add_argument("--output", default=CONFIG["output"])
    ap.add_argument("--seed", type=int, default=CONFIG["seed"])
    args = ap.parse_args()
    run({**CONFIG, **vars(args)})


if __name__ == "__main__":
    main()
