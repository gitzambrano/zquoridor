#!/usr/bin/env python3
"""Select high-value samples from architecture-neutral teacher corpora."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Sequence

SUPPORTED_RAW_SCHEMAS = {
    "zquoridor.teacher.raw.v2",
    "zquoridor.teacher.raw.v3",
}


def iter_records(corpus: Path) -> Iterable[dict]:
    """Yield supported teacher records from one JSONL file or corpus directory."""
    if corpus.is_file():
        paths = [corpus]
    else:
        paths = sorted(corpus.glob("train/*.jsonl")) + sorted(corpus.glob("val/*.jsonl"))
    if not paths:
        raise ValueError(f"no teacher JSONL shards found in {corpus}")
    for path in paths:
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            record = json.loads(line)
            schema = record.get("schema")
            if schema not in SUPPORTED_RAW_SCHEMAS:
                raise ValueError(
                    f"{path}:{lineno}: unsupported teacher schema {schema!r}; "
                    f"supported={sorted(SUPPORTED_RAW_SCHEMAS)}"
                )
            yield record


def should_select(record: dict, mode: str, min_vote: float,
                  require_budget_agreement: bool) -> bool:
    """Return true when one sample satisfies the selection policy."""
    diagnostics = record.get("diagnostics", {})
    vote = float(diagnostics.get("highest_budget_vote_fraction", 0.0))
    budget_agreement = bool(diagnostics.get("budget_winners_agree", False))
    student_agrees = diagnostics.get("student_agrees")

    if mode == "disagreement":
        if student_agrees is not False or vote < min_vote:
            return False
        return budget_agreement or not require_budget_agreement
    if mode == "unstable":
        return vote < min_vote or not budget_agreement
    if mode == "all":
        return vote >= min_vote and (budget_agreement or not require_budget_agreement)
    raise ValueError(f"unknown selection mode: {mode}")


def main(argv: Sequence[str] | None = None) -> int:
    """Parse options and write selected samples plus a manifest."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--mode",
        choices=("disagreement", "unstable", "all"),
        default="disagreement",
    )
    parser.add_argument("--min-vote", type=float, default=1.0)
    parser.add_argument("--allow-budget-disagreement", action="store_true")
    args = parser.parse_args(argv)

    if not 0.0 <= args.min_vote <= 1.0:
        raise SystemExit("min-vote must be between 0 and 1")

    corpus = Path(args.corpus)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    selected = 0
    with out.open("w", encoding="utf-8") as fh:
        try:
            for record in iter_records(corpus):
                total += 1
                if not should_select(
                    record,
                    args.mode,
                    args.min_vote,
                    not args.allow_budget_disagreement,
                ):
                    continue
                selected += 1
                diagnostics = record["diagnostics"]
                selected_record = dict(record)
                selected_record["selection"] = {
                    "mode": args.mode,
                    "teacher_vote_fraction": diagnostics.get(
                        "highest_budget_vote_fraction"
                    ),
                    "budget_winners_agree": diagnostics.get(
                        "budget_winners_agree"
                    ),
                    "student_agrees": diagnostics.get("student_agrees"),
                }
                fh.write(json.dumps(selected_record, separators=(",", ":")) + "\n")
        except (ValueError, json.JSONDecodeError) as exc:
            raise SystemExit(str(exc)) from exc

    manifest = {
        "schema": "zquoridor.teacher.selection.v1",
        "source": str(corpus),
        "output": str(out),
        "mode": args.mode,
        "min_vote": args.min_vote,
        "require_budget_agreement": not args.allow_budget_disagreement,
        "samples_total": total,
        "samples_selected": selected,
        "selection_fraction": selected / max(1, total),
    }
    manifest_path = out.with_suffix(out.suffix + ".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
