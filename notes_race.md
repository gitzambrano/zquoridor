# inv/race-fuzz -- bug hunt in src/endgame_race.hpp

Worktree `C:\Zq-racehunt`, branch `inv/race-fuzz`, base 3947e26.
Goal: adversarial verification of the empty-handed race solver and its
consumers; deliver a permanent oracle/fuzz regression test.

## 1. Audit of the four documented past corrections (read before touching)

1. **ETA gate removed from the decision pipeline** (header note + plano 4d-1).
   The "gap >= 3 tempos" margin assumed physical blocking costs at most one
   extra tempo beyond a jump. A random-topology test found a real
   counterexample (exact dtm 21 vs naive prediction 19). `raceETAGate`
   survives as an isolated, tested utility; it is NOT called by
   `resolveEmptyHandedEndgame`. Do not re-plug it without a new geometric
   proof.
2. **Disjoint gate rebased on reachable REGION** (header note + plano 4d-2).
   The original gate used shortest-path-mask disjointness. False security:
   a losing player may detour outside its own shortest-path set purely to
   block. Counterexample pinned in `testInfinitePursuitDraw` (synthetic
   topology, turn 1 = draw). Correct base: whole-region disjointness
   (`reachableRegionMask`); disjoint regions imply zero traversable edges
   between them, so no jump/block ever, on any route.
3. **Real-time budget for Service B + TT storage** (plano 4e-1).
   Per-topology DP rebuild measured ~790us/call; corrected Level-2 gate
   decides less often, so Service B became the common path (>50x nps drop,
   Elo -166 in an external arena before the fix). Fix: per-chooseMove
   chrono budget (~3% of move time) covering gate+DP, budget-exhausted
   nodes fall through to heuristic search at zero extra cost, plus
   TT EXACT depth=127 storage keyed by s.hash for exact-position reuse.
4. **Move-choice bug at the real root** (plano 4e-2).
   The solver returns a VALUE only. When the real game root is already
   empty-handed there is no parent node to compare children, and chooseMove
   read a TT placeholder best move -> arbitrary moves with correct scores;
   lost most games while nodes/s looked healthy. Fixed by comparing all
   pawn children by exact solver values inside chooseMove. Later extended
   by `endgameProgressTiebreak` (default ON since 2026-08-23): reorders
   ONLY exactly-equal children by root-side progress.

Also documented: succOff hole bug for p0==p1 states inside the CSR builder
(fixed), openDir/neighborCell precompute (~242us of the miss path),
multi-slot cache experiment (NSLOTS=1024).

## 2. Static-analysis findings to verify numerically

- F1 (gate robustness): `raceDisjointGate` computes
  `pl = off + 2*(rawDist-1)` with `shortestPathLen` possibly returning -1
  (unreachable) or 0 (pawn already on goal row). Outside the real-play
  invariant ("both pawns always keep a path after each legal wall") this
  yields pl < 1 and a negative/zero dtmOut or a WRONG winner, while the DP
  would answer correctly. Production hook never sees such states (winner()
  checked first; wall legality preserves the invariant), but
  `resolveEmptyHandedEndgame` is a public inline utility called directly by
  tests/tools. Candidate minimal fix: only let the gate decide when both
  rawDist >= 1.
- F2 (DP core): solveFor retrograde BFS dtm semantics look sound
  (queue monotone in dtm; universal branch takes last-entered successor =
  max dtm). To be proven empirically against an independent oracle.
- F3 (jump semantics): DP graph builder mirrors `pawnStepMoves` exactly
  (straight jump precedence, diagonals only when straight blocked, edge
  into opponent required open). Verified by reading; oracle re-derived
  independently.
- F4 (cache): slot key = exact wallsH/wallsV compare after hash mod 1024;
  DP solves ALL (p0,p1,t) states, so pawn/turn are query-time indices ->
  sound by construction; thread_local slots match per-thread engines in
  selfplay. Collisions fall back to rebuild. Empirical stress planned.
- F5 (budget globals): `g_raceExactBudgetUs/UsedUs` are plain global
  doubles; selfplay runs N threads x own Negamax but shares these ->
  cross-thread budget interference. Perf-only (fallback is safe), to be
  confirmed + documented.
- F6 (TT probe without depth check in race hook): any matching EXACT entry
  (including heuristic full-window entries left by earlier moves' searches)
  short-circuits the solver. Sound (value approximates truth), accepted
  tradeoff; document only.
- F7 (root branch ignores repetition history): internal negamax nodes rank
  repetition above the solver; the empty-handed ROOT branch ignores
  gameHistory entirely. Inconsistent priority, practical nit; document.

## 3. Plan

- P1 independent oracle (naive win-set fixpoint + separate simple retrograde
  dtm, successor generation written from scratch) vs
  `resolveEmptyHandedEndgame` AND vs `raceExactDTM` (splits gate bugs from
  DP bugs) over thousands of topologies (playout-born + synthetic +
  adversarial pawn configs).
- P2 root-choice optimality vs oracle children values (catches value-right
  move-wrong class).
- P3 budget exhaustion fallback (deterministic via g_raceExactUsedUs
  override + statistical tiny-budget chooseMove).
- P4 cache soundness under collisions/alternation.
- P5 gate boundary stress + degenerate rawDist probes (F1).
- P6 toggle combos (endgameProgressTiebreak x parityAnchoredRaceDraw):
  chosen child value must equal oracle-best under every combo.
- P7 long-game differential: engine-vs-engine from hands-empty positions,
  winner AND mate length must equal oracle prediction (engine plays
  optimally there, so game length == dtm exactly).

## 3b. Plan closure -- session 2 results per phase

All phases are closed. Counts come from the final run of
`bin/test_endgame_race_fuzz.exe` (-O2 build, about 12 s wall on this
machine) and one full run of `bin/bench_race_differential.exe`.

| Phase | Verdict | Evidence and counts |
|---|---|---|
| P1 oracle agreement | PASS | Phase A: 8468 comparisons of the production pipeline (gate + Service B) AND bare raceExactDTM vs the oracle, zero divergence. Corpus: 586 topologies = 110 playout-born + 470 synthetic (4 to 34 walls) + constructed partition families with and without a gap + 32 hunted drawn-root cases. Pawn configs mix random pairs, head-on adjacencies, dead-end cells, same-column pairs. The gate itself decided 610 and was refused 7858 times; every decided case matched. |
| P2 root-choice optimality | PASS | Phase B: 8468 roots (2765 won / 5671 lost / 32 drawn for the root side), 33872 chooseMove calls across all four toggle combos; every chosen move achieved the oracle-best child value and every reported root score equaled it. This pins the historical "value right, move wrong" class. |
| P3 budget fallback | PASS | Phase C: forced g_raceExactUsedUs=2e18 over 24 roots returns finite sane scores; tiny-budget chooseMove (1 to 4 ms) returned a legal move in 60/60 tries. |
| P4 cache soundness | PASS | Phase D: 960 interleaved production queries against remembered oracle answers under constant eviction pressure (1600 throwaway topologies over a 1024-slot table); zero divergence; misses grew 716 -> 2860 so evictions really happened. Slot keys compare wallsH/wallsV exactly after the hash mod, so a collision falls back to a rebuild by construction. |
| P5 gate boundary + degenerates | PASS | Constructed partition-with-gap topologies sit right at the gate boundary. E1/E1b/E2/E3 pin the degenerate cases (rawDist 0 behind a partition, sealed pocket, two pockets). E4 adds 600 randomized degenerate probes over real corpus topologies (goal-row pawn: rawDist 0; freshly carved 4-wall pocket: rawDist -1), all answered exactly, gate refused on every rawDist-0 input. |
| P6 toggle combos | PASS | Covered inside phase B: TB x parity in all 4 combinations, chosen value == oracle-best in every combo, score conventions match test_contempt_repetition (root-perspective contempt for draws). |
| P7 long-game differential | PASS | Phase F (in-test, deterministic): 14 engine-vs-engine games from hands-empty starts; every decisive game ended with the predicted winner in EXACTLY the predicted DTM plies; synthetic pursuit draws never finished (6 games). Heavy scale-up benchmarks/bench_race_differential.cpp: 360 decisive + 6 drawn games on 4 workers, 360/360 exact lengths, zero contradictions, zero illegal moves, longest decisive 31 plies. |

### Findings F5/F6/F7 -- closed as documented-only

- F5 (budget globals): CONFIRMED by reading. g_raceExactBudgetUs and
  g_raceExactUsedUs are plain global doubles (endgame_race.hpp). Selfplay
  spawns cfg.numThreads workers (tools/selfplay/selfplay.hpp), each running
  its own Negamax and calling chooseMove, which resets UsedUs and sets
  Budget per move (search.hpp). Concurrent moves therefore erase each
  other's accounting: one thread's reset reopens another thread's budget,
  and one thread's solver time can exhaust another thread's budget
  mid-move. Both effects only shift work between the exact solver and the
  ordinary heuristic search. Correctness is untouched: the budget-exhausted
  path is the pre-feature heuristic node, and the empty-handed ROOT branch
  runs before any budget logic. Aligned doubles do not tear on x86-64 in
  practice; a torn read would still only misprice one decision. Perf-only;
  fix (thread_local or per-engine fields) belongs to a perf pass, not this
  correctness hunt.
- F6 (TT probe without depth check in the race hook): accepted tradeoff.
  Any matching EXACT entry short-circuits the solver even when stored at a
  shallow depth by an earlier unrelated search. The stored score is either
  an exact race value or a heuristic full-window score that approximates
  the truth; both keep the parent comparison sound within normal search
  semantics. Documented; no action.
- F7 (root branch ignores repetition history): confirmed inconsistency.
  Internal nodes rank a real repetition above the solver result; the
  empty-handed root branch never consults gameHistory. Practical impact is
  limited because a position reached twice with both hands empty is rare
  and the solver value stays exact. Documented; candidate future fix is to
  pass gameHistory into the root branch and rank repetitions first there.

### Build-script drift found while wiring the suite

build_tests.sh still compiled tests/lazy_acc_parity.cpp, a file deleted by
commit b48b59b, so the Linux suite aborted at entry 9 under `set -e`. The
same script also lacked the three newer tests (wall_qextension, policy_ab,
contempt_repetition). Fixed: both build_tests scripts now list the same 16
entries and end with test_endgame_race_fuzz.

## 4. Log

### 2026-08-23 -- session 2: red->green re-verified from scratch

Rebuilt the harness at the pre-fix commit 858214c in a scratch directory
(outside the repo) and at HEAD c1c32ef:

- Pre-fix: 5 FAILURES, all in the degenerate-input probes -- E1b
  `prod=(0,-1)` vs oracle `(0,0)` (pawn on goal row behind a partition),
  E2 `prod=(0,-3)/(0,-2)` vs oracle/dp `(-1,0)` (sealed pocket flips the
  winner to a NEGATIVE dtm), E3 `prod=(0,-3)` and even `(1,-3)` depending
  on turn vs true draw. Phases A-D identical to HEAD and green there,
  which isolates the gate as the sole diverging component.
- Honesty note: at 858214c the pinned E2 expectation itself was wrong
  (`oc.winner == 1`; the oracle always answered draw `-1`). Commit c1c32ef
  corrected the pin together with the gate guard. The gate bug evidence
  stands on its own: production fabricated winners with negative dtm
  against BOTH the bare DP and the oracle.
### 2026-08-23 -- session 2: P1-P7 closed, permanent coverage wired

- Phase E4 added to test_endgame_race_fuzz: 600 randomized degenerate
  probes over real corpus topologies (rawDist-0 goal-row pawns and fresh
  sealed pockets), all exact, gate refusal verified on every rawDist-0
  input. This generalizes the F1 regression pins beyond the three
  hand-written families.
- Phase F added to test_endgame_race_fuzz: compact deterministic long-game
  differential (14 playout-born decisive games + the synthetic pursuit
  pairs). Every decisive game realized the oracle winner AND length
  exactly; draws never finished.
- Fate of benchmarks/bench_race_differential.cpp: FINISHED and committed,
  not deleted. The in-test phase F keeps a fast deterministic floor in the
  suite (whole fuzz binary runs in about 12 s); the bench scales the same
  check to 366 games on 4 workers (about 4 min) and stays standalone like
  tools/arena, since it costs minutes and spawns threads. Repairs applied:
  aggregate built before the draw stage used it, makeState() did not exist
  (local Zobrist-correct builder added), worker count capped at 4 per the
  load constraints of this investigation.
- build_tests.bat gained entry [16/16]; build_tests.sh was resynced (it
  still built the deleted lazy_acc_parity.cpp and lacked three newer
  tests). Full suite green on this tree; runtimes of the pre-existing
  heavy tests recorded separately.
- F5/F6/F7 finalized as documented-only (section above).

Final suite status at commit time: all 16 test binaries compile under
-O2 -std=c++17 and pass; nnue_verify needs explicit weight arguments and
is unchanged from main.

### 2026-08-23 -- session 2: camping survey (priority-zero symptom)

User symptom: "sometimes the engine in endgame seems not to understand it
has to go to the end of the board and STAYS ON THE FIRST ROW". New tool
`benchmarks/bench_camping_survey.cpp` (standalone, deterministic seed
20260823). Camping = the chosen pawn move keeps the root-side pawn inside
its own back two rows while some legal move advances that pawn a row toward
its goal. Every camping instance is classified by the independent oracle.

Empty-handed corpus: 2040 roots from 340 frozen-topology families (wall-
biased playouts plus a 5-ply random race walk per family).

- TB-on (production): camping 722/2040 = 35.39%. Split: WON side 321,
  LOST side 401, DRAWN side 0 (no drawn states exist in this corpus).
- Value check against oracle children: ZERO mismatches over all 2040
  roots x 2 modes (4080 chooseMove calls). Every camping instance is
  value-optimal: 401 are maximum-delay defenses while LOST (correct
  play), 321 are minimal-DTM geometry while WON (a sideways/backward
  step can be part of the unique fastest route).
- Tiebreak interplay: OFF raises total camping to 784/2040 = 38.43%
  (LOST side 463). endgameProgressTiebreak ON removes 62 lost-side
  camping moves (about 13% relative) at zero optimality cost.
- NNUE pass (weights read-only from C:/Zquoridor/data/nnue): root-move
  agreement 2040/2040. The MCAB shell needs no separate run here:
  src/mcab.hpp delegates an empty-handed root straight to
  Negamax::chooseMove, which both passes exercise.

Quasi-endgame corpus (no solver at the root; at least one side <= 2 walls,
not both zero): 220 states, 120 ms heuristic searches. The side ahead by
2+ raw plies is the root side in 13 of them; that side camps in 7.
Follow-up probes with 500 ms and 2000 ms budgets:

- 5 of the 7 stop camping with more time: 3 switch to a wall move,
  2 switch to a forward move.
- The stable wall choices are mostly rational trades: the chosen wall
  adds +4 raw plies to the opponent path in one case, +1 in another,
  and +0 in one case (own path unchanged too). The +0 case is ordinary
  heuristic noise in a won-looking position, not a logic defect.

Verdict: no bug behind the symptom so far. In the solved regime the
"first-row camping" is provably optimal play (delay when lost, forced
pace otherwise). In the quasi regime it is wall-building or stalling
while clearly ahead, and deeper searches mostly replace it. P7 below
adds the dynamic confirmation (realized game length must equal the
predicted DTM exactly, which caps cumulative suboptimality at zero).
