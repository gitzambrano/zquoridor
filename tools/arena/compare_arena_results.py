"""Compare candidate and baseline arena benchmark results against common opponents.

Reads summary.json and games.jsonl from arena evaluations to produce paired
comparisons, delta Elo calculations, and markdown tables.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]

CONFIG = {
    "eval_dir": str(ROOT / "results" / "benchmarks" / "colab_eval"),
    "drive_dir": "/content/drive/MyDrive/zquoridor_data/arena_candidate_eval",
}


def build_parser() -> argparse.ArgumentParser:
    """Build command-line parser overriding CONFIG."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--eval-dir",
        type=str,
        default=CONFIG["eval_dir"],
        help="Base directory containing arena evaluation subdirectories.",
    )
    return parser


def load_summary(path: Path) -> dict | None:
    """Read summary.json if available."""
    target = path / "summary.json"
    if not target.is_file():
        return None
    try:
        return json.loads(target.read_text(encoding="utf-8"))
    except Exception:
        return None


def format_row(
    label: str,
    opponent: str,
    summary: dict | None,
) -> str:
    """Format single result row."""
    if not summary:
        return f"| {label:25s} | {opponent:15s} | Pending / No data |"
    games = summary.get("games", 0)
    score = summary.get("score_pct", 0.0)
    elo = summary.get("elo", 0.0)
    ci = summary.get("elo_ci_95", [0.0, 0.0])
    wins = summary.get("wins", 0)
    losses = summary.get("losses", 0)
    draws = summary.get("draws", 0)
    return (
        f"| {label:25s} | {opponent:15s} | {games:5d} | "
        f"{wins:3d}W - {draws:2d}D - {losses:3d}L ({score:5.1f}%) | "
        f"{elo:+6.1f} [{ci[0]:+6.1f}, {ci[1]:+6.1f}] |"
    )


def compare_battery(base_dir: Path) -> None:
    """Compare candidate and baseline across available benchmarks."""
    opponents = ["main", "titanium", "claustrophobia"]

    print("\n" + "=" * 80)
    print("ARENA EVALUATION BATTERY SUMMARY")
    print("=" * 80)
    print(f"Directory: {base_dir}\n")

    # Direct Head-to-Head
    h2h_sum = load_summary(base_dir / "main")
    print("### Direct Head-to-Head (Candidate vs Baseline)")
    print("| Match                     | Opponent        | Games | Score                   | Elo [95% CI]         |")
    print("| :------------------------ | :-------------- | ----: | :---------------------- | :------------------- |")
    print(format_row("Candidate (Direct H2H)", "Main Baseline", h2h_sum))
    print()

    # External Opponents
    print("### External Opponents Comparison (Same Hardware, Same Openings)")
    print("| Model                     | Opponent        | Games | Score                   | Elo [95% CI]         |")
    print("| :------------------------ | :-------------- | ----: | :---------------------- | :------------------- |")

    for opp in ["titanium", "claustrophobia"]:
        cand_sum = load_summary(base_dir / opp)
        base_sum = load_summary(base_dir / f"baseline_vs_{opp}")

        print(format_row(f"Candidate vs {opp.capitalize()}", opp.capitalize(), cand_sum))
        print(format_row(f"Baseline vs {opp.capitalize()}", opp.capitalize(), base_sum))

        if cand_sum and base_sum:
            cand_elo = cand_sum.get("elo", 0.0)
            base_elo = base_sum.get("elo", 0.0)
            delta_elo = cand_elo - base_elo
            cand_sc = cand_sum.get("score_pct", 0.0)
            base_sc = base_sum.get("score_pct", 0.0)
            delta_sc = cand_sc - base_sc
            print(
                f"| -> DELTA ({opp:12s}) | DIFF            |   --  | "
                f"Delta Score: {delta_sc:+5.1f}%     | Delta Elo: {delta_elo:+6.1f}     |"
            )
        print("|" + "-" * 27 + "|" + "-" * 17 + "|" + "-" * 7 + "|" + "-" * 25 + "|" + "-" * 22 + "|")
    print()


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    eval_dir = Path(args.eval_dir).resolve()
    drive_dir = Path(CONFIG["drive_dir"])
    target = drive_dir if drive_dir.is_dir() else eval_dir

    compare_battery(target)
    return 0


if __name__ == "__main__":
    sys.exit(main())
