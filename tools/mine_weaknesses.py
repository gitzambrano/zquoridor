#!/usr/bin/env python3
"""Mine weak openings for Main 3.01 and Candidate networks against Claustrophobia and Titanium.

Outputs a curated dataset of empirically weak openings across both network architectures
for variation expansion and reinforcement learning rollouts.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

CONFIG = {
    "normal_openings": "tools/external/openings_normal_screen_100.jsonl",
    "cr_openings": "tools/external/openings_center_rush_sound_5k.jsonl",
    "output_json": "tools/external/weak_openings_mined.json",
    "max_score_threshold": 0.40,
    "min_losses_threshold": 2,
}


def load_openings(path: Path) -> list[dict[str, Any]]:
    """Load line-delimited JSON opening definitions."""
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def mine_weaknesses(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Aggregate benchmark results and isolate weak lines for Main 3.01 and Candidate."""
    normal_openings = load_openings(ROOT / config["normal_openings"])
    cr_openings = load_openings(ROOT / config["cr_openings"])

    benchmark_suites = [
        # Candidate (Soup_Tri_Contact, 858 features)
        ("Soup_Tri_Contact", "claustrophobia", "normal", ROOT / "results/benchmarks/candidate-contact-soup-battery/vs_claustro_normal/games.jsonl"),
        ("Soup_Tri_Contact", "claustrophobia", "center_rush", ROOT / "results/benchmarks/candidate-contact-soup-battery/vs_claustro_centerrush/games.jsonl"),
        ("Soup_Tri_Contact", "titanium", "normal", ROOT / "results/benchmarks/candidate-contact-soup-battery/vs_titanium_normal/games.jsonl"),
        ("Soup_Tri_Contact", "titanium", "center_rush", ROOT / "results/benchmarks/candidate-contact-soup-battery/vs_titanium_centerrush/games.jsonl"),
        ("Soup_Tri_Contact", "claustrophobia", "normal", ROOT / "results/benchmarks/candidate-contact-soup-full-800g/claustro_normal/games.jsonl"),
        # Main 3.01 Champion (Soup_Tri_Equal, 504 features)
        ("Main_3.01", "claustrophobia", "normal", ROOT / "results/benchmarks/souptri_claustro_normal/games.jsonl"),
        ("Main_3.01", "claustrophobia", "center_rush", ROOT / "results/benchmarks/souptri_claustro_centerrush/games.jsonl"),
        ("Main_3.01", "titanium", "normal", ROOT / "results/benchmarks/souptri_titanium_normal/games.jsonl"),
        ("Main_3.01", "titanium", "center_rush", ROOT / "results/benchmarks/souptri_titanium_centerrush/games.jsonl"),
    ]

    stats: dict[tuple[str, int], dict[str, Any]] = defaultdict(lambda: {
        "exp_results": [],
        "main_results": [],
        "exp_losses": 0,
        "main_losses": 0,
        "opponents": set(),
        "models": set(),
    })

    for model, opponent, book, path in benchmark_suites:
        if not path.is_file():
            continue
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    game = json.loads(line)
                except Exception:
                    continue
                op_idx = game.get("opening_index")
                if op_idx is None:
                    continue
                res = float(game.get("result", 0.0))
                key = (book, op_idx)
                stats[key]["opponents"].add(opponent)
                stats[key]["models"].add(model)
                if model == "Soup_Tri_Contact":
                    stats[key]["exp_results"].append(res)
                    if res == 0.0:
                        stats[key]["exp_losses"] += 1
                elif model == "Main_3.01":
                    stats[key]["main_results"].append(res)
                    if res == 0.0:
                        stats[key]["main_losses"] += 1

    curated_weak: list[dict[str, Any]] = []
    score_thresh = config["max_score_threshold"]
    loss_thresh = config["min_losses_threshold"]

    for (book, op_idx), data in stats.items():
        exp_games = len(data["exp_results"])
        main_games = len(data["main_results"])
        exp_score = sum(data["exp_results"]) / exp_games if exp_games > 0 else None
        main_score = sum(data["main_results"]) / main_games if main_games > 0 else None

        exp_is_weak = exp_score is not None and (exp_score <= score_thresh or data["exp_losses"] >= loss_thresh)
        main_is_weak = main_score is not None and (main_score <= score_thresh or data["main_losses"] >= loss_thresh)

        if not (exp_is_weak or main_is_weak):
            continue

        book_source = normal_openings if book == "normal" else cr_openings
        if op_idx >= len(book_source):
            continue

        raw = book_source[op_idx]
        moves = raw.get("moves") or raw.get("history") or []

        if exp_is_weak and main_is_weak:
            origin = "both"
        elif exp_is_weak:
            origin = "Soup_Tri_Contact"
        else:
            origin = "Main_3.01"

        total_games = exp_games + main_games
        total_losses = data["exp_losses"] + data["main_losses"]
        overall_score = (sum(data["exp_results"]) + sum(data["main_results"])) / total_games if total_games > 0 else 0.0

        curated_weak.append({
            "schema": "zquoridor.weakness_seed.v1",
            "book": book,
            "opening_index": op_idx,
            "category": raw.get("category", book),
            "id": raw.get("id", f"{book}_{op_idx:04d}"),
            "origin": origin,
            "overall_score": round(overall_score, 4),
            "total_games": total_games,
            "total_losses": total_losses,
            "exp_score": round(exp_score, 4) if exp_score is not None else None,
            "exp_games": exp_games,
            "exp_losses": data["exp_losses"],
            "main_score": round(main_score, 4) if main_score is not None else None,
            "main_games": main_games,
            "main_losses": data["main_losses"],
            "opponents": sorted(list(data["opponents"])),
            "moves": moves,
            "plies": len(moves),
        })

    # Sort primarily by origin ('both' first), then overall score ascending, then total losses descending
    origin_priority = {"both": 0, "Soup_Tri_Contact": 1, "Main_3.01": 2}
    curated_weak.sort(key=lambda x: (origin_priority.get(x["origin"], 3), x["overall_score"], -x["total_losses"]))

    return curated_weak


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=str, default=CONFIG["output_json"], help="Output JSON path")
    parser.add_argument("--score-threshold", type=float, default=CONFIG["max_score_threshold"], help="Max score threshold")
    parser.add_argument("--losses-threshold", type=int, default=CONFIG["min_losses_threshold"], help="Min losses threshold")
    args = parser.parse_args()

    cfg = dict(CONFIG)
    cfg["output_json"] = args.output
    cfg["max_score_threshold"] = args.score_threshold
    cfg["min_losses_threshold"] = args.losses_threshold

    weak_seeds = mine_weaknesses(cfg)
    out_path = ROOT / cfg["output_json"]
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(weak_seeds, f, indent=2)

    jsonl_path = out_path.with_suffix(".jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for seed in weak_seeds:
            f.write(json.dumps(seed) + "\n")

    both_count = sum(1 for w in weak_seeds if w["origin"] == "both")
    exp_count = sum(1 for w in weak_seeds if w["origin"] == "Soup_Tri_Contact")
    main_count = sum(1 for w in weak_seeds if w["origin"] == "Main_3.01")

    print(f"Mined {len(weak_seeds)} empirically weak openings:")
    print(f"  - Shared critical weaknesses (both networks): {both_count}")
    print(f"  - Experimental candidate only (Soup_Tri_Contact): {exp_count}")
    print(f"  - Production champion only (Main_3.01): {main_count}")
    print(f"Saved to: {out_path} and {jsonl_path}")


if __name__ == "__main__":
    main()
