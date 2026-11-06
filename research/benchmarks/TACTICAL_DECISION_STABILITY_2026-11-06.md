# Tactical decision stability and quiet-wall audit

Date: 6 November 2026. Production reference:
`main` at `bdffab86ad9e7f21b64b8841fa7f1620668f3fe9`.
Production INT8 network SHA-256:
`fe8473255fd219fab247b1a7d44be25384a6cfc5daa5c301def49109badb7f2b`.

## Purpose

Investigate why more simulations have not translated into proven Elo gain,
and identify tactical wall decisions for further independent annotation.
This study does not change production code or train a neural network.

Each selected legal position is searched with the frozen production engine
at 200 ms, 1,200 ms and 3,000 ms, plus 3,000 ms using
`--graph-leaf-mix 0.55`. The last variant uses **the same trained network**;
it is a search sensitivity check, not independent ground truth.

The audit reconstructs the board with the independent Python referee and
uses breadth-first search to measure each player's minimum path length
before and after a selected wall. A wall is called *path-distance-neutral*
only when both distances are unchanged. This does not by itself prove the
move is strategically quiet, good, or winning.

## Bounded production-model audit

GitHub Actions: https://github.com/gitzambrano/zquoridor/actions/runs/42242998450

All three parallel jobs completed successfully.

| Source | Samples | 200 ms vs 3 s best-move changes | 3 s graph-mix agreement | 3 s wall choices | Path-distance-neutral walls | Nonoverlapping histories |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Normal confirmation | 14 | 0 | 13/14 | 0 | 0 | 0/14 |
| Center Rush | 14 | 1 | 12/14 | 10 | 3 | 0/14 |
| Targeted weaknesses | 14 | 2 | 11/14 | 5 | 1 | 13/14 |

Nonoverlap requires no exact or prefix relation in either direction with
any frozen Normal confirmation or Center Rush opening. Every frozen benchmark
position must be excluded from training regardless of move stability.

Specific targeted diagnostic examples:
- Source index 1711: 200 ms selected pawn `c1`; 3 s selected wall `d6h`,
  but the alternative 3 s search again selected `c1`. **Not a robust target**.
- Source index 475: 200 ms selected pawn `d3`; 3 s selected wall `e7v`,
  but the alternative 3 s search again selected `d3`. **Not a robust target**.
- Source index 1229: all four searches selected `d4v`; this was a legal
  path-distance-neutral wall. It is an example to study, **not a strength label**.
- The Center Rush book contained three such neutral wall choices
  (indices 648, 3114, and 1715). They belong to the frozen benchmark and
  must never be reused as training targets.

## Independent source extension

A separate audit samples the irregular self-play seed bank, whose
origins are not the frozen evaluation books. Each history is nevertheless
checked against the full frozen benchmark histories for prefix overlap.
GitHub Actions:
https://github.com/gitzambrano/zquoridor/actions/runs/42243406946

The output files `positions.json` and `summary.json` are saved as
GitHub Actions artifacts, not committed to Git or blended into a trainer.

## Interpretation and next strength steps

The 42-position pilot is too small to establish the prevalence of quiet
walls or a confident training-target distribution. The pilot shows why
deep self-search moves cannot be blindly treated as teacher labels:
in several cases a minor search parameter change reversed the preferred
move.

Recommended order:
1. Finish the independent-seed eligibility audit and deduplicate positions.
2. Select candidate cases with repeated cross-configuration agreement.
   Require independent stronger labels or multiple genuinely independent
   evaluators before assigning training targets.
3. Build a balanced, benchmark-disjoint dataset containing difficult
   pawn-vs-wall decisions, legal quiet walls, immediate tactical walls,
   and negative examples. Preserve separate policy and value evidence.
4. Fine-tune a new NNUE candidate using the existing
   `training/train_nnue.py` entry point. Do not train on frozen Normal
   or Center Rush evaluation openings.
5. Use a cheap paired 200 ms internal screen, then paired external
   Normal and Center Rush games against frozen main, Claustrophobia,
   and Titanium, followed by real 3+2. No promotion from NPS or
   pilot self-play results alone.

**Promotion status: no new engine or network is approved.**
