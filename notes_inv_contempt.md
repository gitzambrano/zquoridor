# Investigation: contempt / repetition wandering in near-endgames

Worktree `C:\Zq-inv-contempt`, branch `inv/contempt-wandering`, HEAD = main `2f0e4e3`.
Target symptom: the engine shuffles its pawn back and forth in near-endgame
positions with more than one path ("wandering").

## 1. Code reading (before any experiment)

### Three coexisting draw-score conventions

| Site | Rule | Sign seen by ROOT player |
| --- | --- | --- |
| `search.hpp:872` (negamax rep draw), `:757` (quiescence) | anchored to `ply % 2`: node mover of root parity sees `contempt` (-30), other side sees `-contempt` | -30 when root parity node repeats, +30 when opponent repeats |
| `search.hpp:923` (race-solver draw inside negamax) | always `contempt` from the NODE mover perspective, parity-blind | +30 when the node mover is the opponent (odd ply), -30 when it is the root side |
| `search.hpp:436` + `:442` (`chooseMove` empty-handed root branch) | child (opponent to move) draw scored `contempt`; root takes `-childScore` | always +30 |

So the root branch agrees with site 923 and disagrees with the repetition
anchoring of site 872. Within a pure-solver regime this cannot flip a move
choice: wins and losses are near +/-999000, draws only ever tie with other
draws, and all drawing children receive the same value either way. The sign
only changes reported scores, TT contents, and comparisons at the mixed
boundary where a solved draw (+/-30) competes with a heuristic eval of a
sibling move (hundreds of units).

### Where production actually hits these lines

- MCAB (`mcab.hpp:457`) delegates every `wallsLeft==(0,0)` root straight to
  `Negamax::chooseMove(root, cap, ms, stats, gameHistory)` -- the root
  branch above is the live production path for wall-less endings.
- The web GUI keeps ONE `Negamax` for the whole game, never clears the TT
  between moves (`clearTTPerMove` default off everywhere), and passes the
  full game history (`g_reptbl.push(g_state.hash)` each ply; draw claim at
  `count(hash) >= 3`).
- selfplay/arena clear the TT only between GAMES (`selfplay.hpp:689`,
  `arena.cpp:496`), not between moves.

### Candidate mechanisms for the wandering

- M1 (all-draw arbitrary choice): when the empty-handed root branch finds
  every child drawn (`ro.winner == -1`), all children score exactly +30 and
  `best` stays `rootMoves[0]` -- generation order of `pawnStepMoves`
  (N, S, E, W, jumps). The move is arbitrary and can be backward or
  sideways; next turn recomputes the same way. Multiple short paths make
  perpetual-chase draws MORE likely, which matches "more than one path".
- M2 (draw avoidance vs stale eval): with walls still on the board, search
  lines that advance pawns reach the `(0,0)` transition inside the horizon
  and return the solver DRAW score (about +/-30), while quiet sibling moves
  keep a heuristic/NNUE eval of hundreds. Alpha-beta then prefers moves
  that POSTPONE entering the solved region. The engine hoards its own
  walls (the transition needs BOTH stocks at zero) and shuffles.
- M3 (TT contamination): repetition scores are path-dependent but stored
  EXACT/LOWER/UPPER keyed by hash alone; the TT survives across real moves,
  so a score produced under an older root (different history, different
  parity anchor) is reused elsewhere. Can flip near-tie choices between
  equivalent pawn advances -> visible flapping.
- M4 (H5, plain eval ties): multiple equal-length paths give near-identical
  evals for several advances; ordering noise (history/killers/policy) picks
  different winners on consecutive searches -> wandering even with
  contempt 0.

## 2. Hypotheses to test

- H1: |contempt| too large; sweep {-60,-30,-15,-5,0} changes wandering rate.
- H2: sign inconsistency confirmed by reading; prove it cannot flip
  pure-solver choices, measure whether fixing it changes anything.
- H3: persistent TT vs clearTT-per-move changes wandering/flapping.
- H4: ply-parity anchor vs pre-existing repetitions (targeted unit test).
- H5: contempt 0 baseline; if wandering stays high, contempt is not the cause.

## 3. Results (2026-08-22)

Corpora (seed 20260822): MID = 220 positions, total walls 4..14,
`pathRobustness >= 2` for both players; EMPTY = 80 wall-less multi-path
positions. Games: color-swapped pairs per position, GUI draw rule
(count >= 3), history pushed like production. Metrics: back = share of
pawn moves that increase own BFS distance; prog/move = mean distance
reduction per pawn move; D = games ending in a claimed repetition draw;
TO = ply-cap timeouts. Raw CSV/logs in `investigation_data/`.

### H1 (contempt magnitude) -- REFUTED

NNUE, 150 ms/move, MID, 120 games each (persistent TT):

| contempt | back | prog/move | D | TO |
| --- | --- | --- | --- | --- |
| -60 | .121 | .511 | 4 | 2 |
| -30 | .125 | .508 | 6 | 2 |
| -15 | .119 | .525 | 2 | 8 |
| -5 | .121 | .515 | 5 | 2 |
| 0 | .131 | .494 | 7 | 4 |

Heuristic mode shows the same flatness (.152 to .177, no trend).
Fixed depth 5 confirms it in both modes. Head-to-head: c=-30 vs c=0
gives Elo(A-B) = -44 +/-76 (n.s.). Contempt value does not drive the
wandering.

### H3 (TT persistence) -- CONFIRMED as the main midgame driver

Same protocol, NNUE:

| tt policy | back | D | TO |
| --- | --- | --- | --- |
| persist (c=-30) | .125 | 6 | 2 |
| clear per move (c=-30) | .081 | 0 | 0 |

Heuristic mode is extreme: persistent TT produces 41/120 (34%)
repetition-drawn games at c=-30 (23..46 across values); clearing the TT
gives 0/120 and halves backward moves (.170 -> .080). BUT clearing costs
strength: duel default vs clear-per-move in NNUE = -71 +/-78 Elo for the
clearing side. Rejected as a production change; kept as an experiment
lever.

### Empty-handed endings -- losing-side optimal delay, not contempt

Winner/loser split (EMPTY corpus, solver regime, identical for every
contempt value): winner back-rate 0.006, loser back-rate 0.468.
Classification of 290 playout roots: 100 pure wins, 142 pure losses
(131 with >= 2 tied max-DTM children), 48 mixed, **0 all-draw roots**.
So M1 (arbitrary pick among all-drawn children) does not occur in
practice; the shuffling side is the LOSER choosing among exactly equal
max-DTM delays by generation order.

`setEndgameProgressTiebreak(true)` reorders only exactly-equal children:
loser back-rate 0.468 -> 0.275, loser progress/move 0.067 -> 0.451,
results bit-identical (80/80 wins each way, same ply counts). Larger
duel: Elo(A-B) = -12 +/-60 on MID (n.s.), 0.0 on EMPTY. Fires on
14/80 roots in the T3 test sample.

### H2 (sign inconsistency) -- CONFIRMED, provably harmless for choice

Three conventions coexist (rep-draw parity-anchored vs race-draw
parity-blind vs root branch negation). Within the solver regime draws
only tie with draws and wins/losses are ~1e5 apart, so no move-class
flip is possible; only TT contents and reported scores shift by <=60.
`setParityAnchoredRaceDraw(true)` pins the alternative convention; the
T1/T2 tests freeze both behaviors. No benefit measured from enabling it
(Elo -55 +/-76, n.s.) -- recommend NOT changing production.

### H4 / H5

H4 semantics (2-in-search vs 3-with-pre-root occurrences, markRoot
interplay) pinned exactly by T5 of `tests/test_contempt_repetition.cpp`;
no misbehavior found. H5: wandering with contempt=0 stays high in the
heuristic mode (back .177), confirming contempt is not the cause there
either; ties between equivalent advances are resolved by ordering noise
and amplified by TT reuse.

### Cost calibration

Depth-capped searches near the end: depth 8 needs >20M nodes (>30 s),
depth 5 completes in 0.2-0.8 s per move (~650k nps single thread).
Production-like time controls reach depth 5-6 here -- eval noise, not
search precision, decides most near-endgame moves.

