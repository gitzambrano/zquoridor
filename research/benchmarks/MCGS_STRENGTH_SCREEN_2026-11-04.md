# MCGS strength investigation: controlled hot-path experiments

Date: 4 November 2026. Baseline: zQuoridor 4.10 `main` at
`bdffab86ad9e7f21b64b8841fa7f1620668f3fe9`.
Working branch: `exp/mcgs-strength-20261104`.

## Objective and constraints

Measure whether a smaller search-node layout and a cached PUCT value sum increase
search throughput and game strength. Preserve all production behavior on `main`.
Use the frozen 4.10 INT8 weights, fixed 200 ms per move, legal paired openings,
and no pondering. Treat the two-color opening pair as the comparison unit.

The implementation uses compile-time experiment flags. Both flags default to zero.

| Flag | Change |
| --- | --- |
| `ZQ_EXP_COMPACT_MCGS_NODE=1` | Allocate the fixed-capacity legal-move buffer only when a node needs actions. |
| `ZQ_EXP_PUCT_SUM_CACHE=1` | Maintain the sum of float edge W values at backup time. Use the sum for the parent PUCT value. |
| `ZQ_EXP_FAST_UCI_REPLAY=1` | Validate only new position-history moves and validate each action without enumerating every legal wall. |

The search changes passed the existing MCGS core, Lazy expansion, sparse policy,
and endgame-leaf correctness tests.

## Fixed work and 200 ms performance

Source: https://github.com/gitzambrano/zquoridor/actions/runs/41955026109

Each variant was compared against an independently compiled baseline in the same
GitHub Actions job. Fixed-budget searches had exactly 36,000 expansions and
93,381 simulations across nine fixed-seed fixtures. Two repeated full-pool
fingerprints matched exactly for each variant.

| Variant | Baseline simulations in six 200 ms searches | Candidate simulations | Gain | Baseline max tree MiB | Candidate max tree MiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| Compact | 73,941 | 73,954 | +0.02% | 48.83 | 23.45 |
| PUCT sum | 103,691 | 109,924 | +6.01% | 51.96 | 53.15 |
| Combined | 99,192 | 106,241 | +7.11% | 51.98 | 28.86 |

Different jobs may run on different CPU hardware and under different
contention. Do not compare their absolute throughput.

## Competitive screen: 200 ms per move

Source: https://github.com/gitzambrano/zquoridor/actions/runs/41955323429

The suite used four paired openings from the frozen Normal screening book
and four paired openings from the frozen Center Rush book. Each candidate
played 16 full games against the default frozen `main` executable.

| Variant | Score | Estimated Elo | Complete pairs | Failures |
| --- | ---: | ---: | ---: | ---: |
| Compact | 43.75% | -43.7 | 8 | 0 |
| PUCT sum | 50.00% | 0.0 | 8 | 0 |
| Combined | 50.00% | 0.0 | 8 | 0 |

The sample is too small for an Elo promotion claim. These games do not
demonstrate a competitive advantage. The original 200 ms screen attempt
(https://github.com/gitzambrano/zquoridor/actions/runs/41955288279)
had an output log path error. It was replaced with the complete successful run.

## Adaptive 3+2 root-budget screen

Source: https://github.com/gitzambrano/zquoridor/actions/runs/41955604907

Each comparison used one GitHub-hosted CPU and four deterministic legal root
positions. The runner used a 180,000 ms + 2,000 ms Fischer clock allocation,
Adaptive Time, and the same NNUE weights. It did not play full timed games.

| Variant | Main simulations/s | Candidate simulations/s | Gain | Main max tree MiB | Candidate max tree MiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| Compact | 67,127 | 69,155 | +3.0% | 1,187.2 | 753.0 |
| Combined | 60,577 | 65,539 | +8.2% | 729.4 | 737.3 |

Do not generalize the memory result across jobs. Peak memory depends on the
actual search trajectory and capacity reservations. The combined result
does not demonstrate reduced peak memory in the 3+2 probe.

## Incremental UCI history replay

Source: https://github.com/gitzambrano/zquoridor/actions/runs/41956204587

The opt-in parser passed the release UCI and native pondering tests.
A synchronized 128-position incremental replay measured 10.217 ms total
for the baseline and 5.610 ms for the fast parser: a 1.82x speedup.
Repeating an unchanged long position measured approximately 0.115 ms and
0.044 ms per command, respectively. The total time saved is small compared
with 200 ms of search. It is a protocol efficiency improvement, not a
measured increase in playing strength.

## Promotion decision

Do not change production `main`. Keep these variants isolated and retain the
full result artifacts. The measured throughput gains are useful, but the paired
game screen does not establish a strength gain. The next stage must address
move choice and tactical evaluation, including difficult wall placements.
Confirm any new candidate across normal and center openings and against both
Claustrophobia and Titanium before promotion.

## PUCT and graph-Q calibration pilot

Source: https://github.com/gitzambrano/zquoridor/actions/runs/41956357701

All three jobs completed successfully. Each calibration played four paired
Normal openings and four paired Center Rush openings at exactly 200 ms per move
against the frozen production binary. The test used seed 20261105, a different
seed from the first throughput-candidate screen. Candidates used both hot-path
optimization flags. Both engines used the same CPU target and original weights.

| Candidate | Normal score | Center Rush score | Overall score | Overall Elo estimate |
| --- | ---: | ---: | ---: | ---: |
| cPuct = 0.70 | 68.75% | 50.00% | 59.375% | +65.9 |
| cPuct = 1.05 | 75.00% | 50.00% | 62.50% | +88.7 |
| graphLeafMix = 0.55 | 62.50% | 50.00% | 56.25% | +43.7 |

These samples are directional. The bootstrap interval for cPuct 1.05
includes 50% and therefore does not establish a real advantage.

The highest pilot score, cPuct 1.05, was selected for an independent
confirmation. The confirmation separates 20 Normal pairs from the disjoint
400-opening confirmation book and 20 Center Rush pairs from new positions.
It uses seed 20261106 and the same true 200 ms move clock.
Confirmation: https://github.com/gitzambrano/zquoridor/actions/runs/41956913995

No search parameter is promoted from the pilot alone.

## Disjoint cPuct 1.05 confirmation: rejection

Source: https://github.com/gitzambrano/zquoridor/actions/runs/41956913995

Both confirmation jobs finished successfully. The candidate used the compact
node layout, the incremental PUCT sum, and cPuct = 1.05. The reference used
the unchanged production settings. Both loaded the exact same 4.10 INT8
checkpoint and used the same CPU target. No engine used pondering.
Each game used a real 200 ms per-move search budget. All opening pairs
contained both colors.

| Book | Pairs | Games | Candidate score | Elo point estimate | Paired bootstrap 95% score interval |
| --- | ---: | ---: | ---: | ---: | --- |
| Disjoint Normal confirmation | 20 | 40 | 52.50% | +17.4 | 40% to 65% |
| Center Rush, new seed | 20 | 40 | 45.00% | -34.9 | 35% to 55% |
| Combined descriptive score | 40 | 80 | 48.75% | -8.7 | Not computed from a combined bootstrap |

All 80 games completed without failures. No candidate won on time or
benefited from pondering. The Center Rush deficit makes the candidate fail
the strict promotion requirement regardless of the Normal point estimate.
The prior 16-game pilot score of 62.5% was not reproduced.

Decision: REJECT cPuct 1.05 for promotion. Retain compact-node and PUCT-sum
code as separate opt-in performance experiments, but treat strength
as unproven. Do not make further search-parameter promotions based on
small screening samples.

## Strength-directed research priority

The fixed 4.10 network has an updated policy head but unchanged value heads.
The next strength campaign should isolate the following two hypotheses.

1. The current value evaluator does not rank difficult quiet-wall plans
   consistently with deeper rollouts. Construct a tactic suite with
   verified delayed wall consequences, quiet walls, and center-rush
   defenses. Compare current NNUE values and root decisions against
   deeper reference search before changing heuristics.
2. The policy misallocates search to locally attractive pawn progress or
   immediate wall effects. Produce balanced deeper-search policy targets
   on hard openings. Train a candidate head against frozen value targets
   and use independent Normal and Center Rush books for validation.

Neither hypothesis is proved by this performance study. A network
promotion requires paired external measurements against frozen main,
Claustrophobia, and Titanium in both books and native 3+2. Keep all
training and evaluation datasets separate.

## Tactical disagreement diagnostic: targeted search budget

Source: https://github.com/gitzambrano/zquoridor/actions/runs/41958037309

Three jobs completed successfully. The unchanged 4.10 engine searched
12 fixed-seed legal openings per book at 200 ms and 1,200 ms per move.
The protocol emitted full root visit and prior arrays. This diagnostic
does not establish the correct move or measure Elo.

| Sample group | Positions | Different best moves | Deeper search moved from pawn to wall |
| --- | ---: | ---: | ---: |
| Frozen Normal confirmation | 12 | 1 (8.3%) | 0 |
| Frozen Center Rush | 12 | 1 (8.3%) | 0 |
| Targeted weakness seeds | 12 | 5 (41.7%) | 1 |

The targeted cases are more sensitive to search time than these small
general-purpose samples. The sample is not a representative random
distribution of all Quoridor positions.

In targeted weakness seed 1677, the 200 ms search chose pawn move
`e3`, while the 1,200 ms search chose wall `d6h`. The longer
search ranked that wall fourth by its own policy prior.
In seed 1266, the selected wall changed from `f6h` to `e4v`;
the longer search ranked the selected wall eleventh by prior.
In seed 605, the selected wall changed from `f7h` to `e6h`.
These differences identify concrete positions for later deeper evaluation.

The short search and longer search are the same model. Therefore, move
disagreement is not independent ground truth. Obtain stronger labels from
multiple independently configured evaluators or deeper search, and check
reliability before using them for training.

Keep the frozen Normal and Center Rush benchmark openings outside training
data. Use only appropriate non-benchmark weakness seeds, with exact
opening-overlap checks, to construct a training campaign. Reserve independent
held-out tactical positions to test any new policy or value head.

Decision: prioritize tactical wall evaluation and training-target quality.
Do not promote any search calibration variant.
