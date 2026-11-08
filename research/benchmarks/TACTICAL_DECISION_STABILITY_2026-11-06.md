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
| Targeted weaknesses | 14 | 2 | 11/14 | 5 | 1 | **9/14** |
| Irregular self-play seeds | 20 | 4 | 17/20 | 0 | 0 | **17/20** |

Nonoverlap requires no exact or prefix relation in either direction with
**all four** frozen opening books: Normal screening (100), Normal confirmation
(400), Center Rush (5,000), and empirical Titanium (40).
The first audit checked only two books; its original 13/14 targeted
eligibility count was **invalid**. A separate verified four-book
recheck corrected the count to **9/14** without repeating inference:
https://github.com/gitzambrano/zquoridor/actions/runs/42244231474 .
Every frozen benchmark position must be excluded from training regardless
of move stability. All search results remain diagnostic, not targets.

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

This extra job finished successfully: 20 irregular positions, four
200 ms versus 3 s move changes, 17/20 agreement between the two long
search configurations, and **zero** long-search wall choices. The
four-book eligibility recheck found 17/20 nonoverlapping histories.
This book is useful as a negative/control sample, not as a rich quiet-wall
source.

The output files `positions.json` and `summary.json` are saved as
GitHub Actions artifacts, not committed to Git or blended into a trainer.
The `eligibility-recheck.json` artifact contains corrected per-position
overlap results for all 62 sampled histories.

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

## Final independent Titanium diagnostic and closure (8 November 2026)

Source: https://github.com/gitzambrano/zquoridor/actions/runs/42244788786

This GitHub Actions job **completed successfully**, using pinned Titanium
`1ac94755f69799f01b2e06869403e701bbf0bd51` without pondering.
It compared production zQuoridor at 200 ms with Titanium at 1,000 ms on
eight legal, benchmark-disjoint wall-rich targeted histories. Different
budgets prohibit any fair-strength or Elo interpretation.

| Index | zQuoridor (200 ms) | Titanium (1,000 ms) | Titanium path-distance-neutral wall? |
| --- | --- | --- | --- |
| 253 | d3 | d3 | N/A |
| 1113 | e3 | e3 | N/A |
| 2336 | f4 | c3h | No |
| 1310 | e2 | e2 | N/A |
| 924 | e2 | d1v | Yes |
| 2197 | e3 | f2 | N/A |
| 777 | f6 | f6 | N/A |
| 1863 | d2 | e7v | Yes |

Summary: four move disagreements in eight positions; Titanium selected
three legal walls, of which two were distance-neutral under the specific
shortest-path definition. zQuoridor selected no walls in this sample.
There were zero exact/prefix opening-history overlaps with the four
frozen evaluation books.

**Interpretation:** These eight positions are independent *diagnostic
opinions*, not proof that the Titanium move is objectively superior, and
not a training batch. The sample is selected for tactical-wall content.

### Closure and practical release decision

- **Freeze search-calibration exploration.** The combined compact-MCGS
  + incremental-PUCT experiment improved 200 ms search throughput by
  approximately 7%, and the adaptive 3+2 root probe showed around 8%
  more simulations/s. Yet the candidate with cPuct 1.05 scored 48.75%
  (80 paired-color games) against the frozen production binary, with a
  45% Center Rush subset. It **failed promotion**.
- **Preserve opt-in performance code** in
  `exp/mcgs-strength-20261104`, and preserve these tactical diagnostic
  scripts and artifacts in `exp/tactical-consensus-20261106`.
  Do not merge search/calibration changes into production based only
  on simulations/s.
- **Maintain current `main` at 4.10.** No tested successor has
  established an advantage on the required complete promotion battery.
- **Next focused development**, not a continuation of this broad audit:
  use the existing `training/train_nnue.py` pathway to build one
  candidate policy/value network emphasizing delayed wall consequences.
  Obtain independent targets at many additional, benchmark-disjoint
  positions; isolate real strategy from engine disagreement; then run
  a cheap screening H2H against frozen main, followed by Normal,
  Center Rush, Claustrophobia, Titanium, and native 3+2 promotion gates.
  Avoid claiming this small probe is sufficient to start training.

**Status: the performance/tactical audit is complete. No additional
GitHub Actions jobs are needed for this audit. Production unchanged.**
