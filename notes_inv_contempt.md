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
