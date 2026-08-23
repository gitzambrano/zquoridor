# inv/ab-policy -- running notes (session 3)

Mission: reduce MCAB/MCTS dependence; use NNUE policy head cheaply inside
alpha-beta. Worktree C:\Zq-inv-abpolicy, branch inv/ab-policy @ 2f0e4e3.
This file is updated before every commit so a dead session can resume.

## Step 0 audit (inherited from session 1+2, uncommitted)

Modified files:
- src/mcab.hpp (+113): direction E params (abPrefilterDepth/TopK/TimeFrac,
  default 0/0/0.25 = off), SFINAE hooks mcabSeedPolicyHistory (B) +
  mcabRankRootMoves (E), filterRootTopK() root-edge filter with prior
  renormalization, calls in chooseMoveMCAB. VERDICT: looks complete and
  follows existing patterns (same SFINAE style as mcabPathCache).
- src/search.hpp (+243): stats.policyNodes counter; toggles
  setPolicyHistorySeedEnabled/Scale (B), setPolicyLmr*/ColdDelta (C),
  setPolicyLmp*/BaseMass/MinCount (D); seedPolicyHistoryFromRoot();
  rankRootMoves(); tryMove lambda extended with polLogit/polMaxLogit/polHave;
  policy-LMR hot/cold modulation; wall-stage LMP by cumulative softmax mass
  with depth-scaled threshold. All default OFF -> bit-identical production.
  VERDICT: complete, matches searchLeaf/searchShallow preamble patterns
  (deadline/stopped/g_raceExactBudgetUs=1e18, nnueAccStack[0] + childAcc
  pointer arithmetic same as chooseMove).
- src/search_tuning.hpp (+63): CLI knobs --policy-history[-scale],
  --policy-lmr[-hot/-cold], --policy-lmp[-base/-min-count] (+ no- forms).
  Delta values applied only when feature flag On. Usage text added.
  NOTE: usage/help strings in Portuguese (existing file convention is PT);
  acceptable, matches neighbors.
- tools/arena/arena.cpp (+51): --[e1|e2]-ab-prefilter-{depth,topk,timefrac}
  + global forms wired into McabParams per game. Comments partly Portuguese
  (matches file). VERDICT: complete.

Untracked:
- benchmarks/map_compute.cpp: compute-map instrument (unit costs, MCAB shape
  at 40/200ms, pure AB at minDepth {3,99,0}). Looks complete; NOT yet in
  build_bench.bat.
- tests/test_policy_ab.cpp: correctness gate (defaults bit-exact both eval
  modes on strided subset; B/C/D variants vs reference with >=85% score /
  >=90% decisive / 0 illegal over ~64 positions; stress all-on mindepth1;
  second LMP threshold 0.15). Looks complete; NOT yet in build_tests.bat.

Audit concerns to verify by build+run:
1. rankRootMoves uses an EMPTY RepetitionTable (markRoot only) instead of
   game history -> a repetition move can be ranked normally. Only affects
   which children stay active at the MCAB root; tree phase still uses
   localRepTbl. Accepted limitation, documented here.
2. filterRootTopK drops moves absent from `ranked` (time-truncated ranking)
   -> root may keep fewer than topK edges. Intended per comment.
3. LMP cut loop walks i from wn-1 down to policyLmpMinCount+1 accumulating
   mass; cut semantics look right; guards alpha/beta mate ranges present.
4. Build scripts lack entries for the two new files -> add
   [8] map_compute to build_bench.bat and [13] test_policy_ab to
   build_tests.bat.

Plan (mandatory incremental commits):
C1 wip preserve (this one) -> C2 audit fixes/build entries -> C3 tests pass
-> C4 compute map results -> C5 budget curve F results -> C6+ per-variant
perf/Elo -> final status.md update + report.

## Results

(pending)

## Step 0 audit outcome (session 3 fixes, commit 2)

1. TEST BUG (fixed): tests/test_policy_ab.cpp configureVariant() hard-wired
   evalMode=NNUE, so the "defaults-heur" block compared a Heuristic engine
   against an NNUE engine -> spurious FAIL. configureVariant now takes an
   explicit nnue flag.
2. ENGINE BUG (fixed): search.hpp gated LMR modulation and wall LMP on one
   shared flag `polActive = policyPtr && (policyLmrEnabled || policyLmpEnabled)`,
   so enabling ONLY --policy-lmp also armed policy-LMR hot/cold modulation
   inside tryMove (evidence: C and D both showed identical 0.52x node ratio).
   Now polLmrActive/polLmpActive are separate; each feature only its own
   toggle. Also reused expBuf[i] in the wall loop instead of re-calling
   policyLogitForMove per candidate.
3. ENGINE CRASH (fixed during fix 2, caught by test): first split dropped the
   policyPtr null check for LMP -> SIGSEGV at nodes where depth <
   policyOrderingMinDepth (no policy array). polLmpActive = policyPtr &&
   policyLmpEnabled now.
4. Build scripts: added map_compute.exe to build_bench.bat [8/8] and
   test_policy_ab.exe to build_tests.bat [13/13]; renumbered steps.

test_policy_ab results AFTER fixes (-O2, weights loaded):
- defaults bit-exact BOTH eval modes (22 positions each)
- history-seed(B): score-agree 98.4%, decisive 100%, illegal 0,
  nodes ref/var 1.04x (B slightly CHEAPER at equal depth)
- policy-lmr(C): agree ok but nodes ref/var 0.52x -> C DOUBLES
  nodes-to-equal-depth with hotDelta 2.5 (policy-hot exemption fires too
  often). Strength must come from arena to justify this cost.
- policy-lmp(D) base 0.05: 100% agreement, nodes 1.01x (barely prunes)
- lmp-base0.15(D): 100% agreement, nodes 1.03x
- stress all-on mindepth1: agree ok, nodes 0.82x (C dominates inflation)

Full regression suite after fixes: rules_sanity, staging, move_ordering,
endgame_race, lmr_pvs, repetition_diff, leaf_smoke, mcab_core,
mcab_dispatch, mcab_phase9 -- ALL PASS.


### Compute map (map_compute.exe, -O3 native AVX2, corpus=16 positions)

Unit costs: policyPass 790ns, nnueEvalInt 856ns, legalMoves 6268ns,
makeChildAccPair 67ns, buildRootAcc 544ns, searchLeaf(d5) 67.2us.

MCAB production shape (fresh tree, no noise): at 40ms/move about 3940
sims+expansions per move; policy passes take approximately 7.6% of wall
time, leaf evals 8.1%. At 200ms/move about 17.0k sims per move; policy
6.6%, leaf evals 7.2%. CONCLUSION: the NNUE policy head is NOT the
hybrid's cost center -- the MCTS loop itself is (about 84%): per-expansion
legalMoves (6.3us) times thousands of expansions, plus PUCT over up to 131
children. Any per-node policy inside AB costs about one leaf eval today.

Pure AB reference on same corpus: 15.0k nodes/move at 40ms (nps 643k),
92.4k nodes/move at 200ms (nps 819k), avg reached depth 19-20 (corpus has
late/race positions where the exact solver shortcuts).

Direction A note (root-only policy ordering) is MOOT as a saving: an ungated
per-node policy pass costs about one leaf eval today (790 vs 856ns), not the
5.8x of the pre-vectorization era.

### Experiment F part 1: MCAB node-budget curve vs PURE AB at 40ms

Setup: ref1 HEAD==5fe8e19 vs ref2 HEAD, both sides identical binary; E1 =
MCAB with nodeBudget N; E2 = pure AB (--e2-no-mcab). NNUE mode, invert
colors on, 400 games/pairing, 4 threads, seed 8589. Elo for E1-E2 with
95% margin from run_arena.py:

| N (nodes) | Elo +/- 95% CI    | W/D/L       |
|-----------|-------------------|-------------|
| 2000      | +19.1 +-33.1      | 200/22/178  |
| 5000      | +39.3 +-33.4      | 213/19/168  |
| 10000     | +57.9 +-33.6      | 223/20/157  |
| 20000     | +54.3 +-33.8      | 223/16/161  |

HEADLINE so far: even at the production selfplay TC (40ms) the hybrid
beats pure AB decisively. The knee is near 10k nodes (matches full 20k
within noise). Raw outputs: exp_f_40_n{2000,5000,10000,20000}.txt.

- Keep all four inherited pieces (A not implemented; directions B/C/D/E are,
  F needs no code beyond McabParams.nodeBudget which already exists).

### Experiment F part 2 + variants (final, session 3)

Full budget curve, Elo(E1)-Elo(E2) with 95% margin:

| TC    | N=2000        | N=5000       | N=10000      | N=20000(prod) |
|-------|---------------|--------------|--------------|---------------|
| 40ms  | +19.1 +-33.1  | +39.3 +-33.4 | +57.9 +-33.6 | +54.3 +-33.8  |
| 200ms | -92.5 +-40.0  | +10.4 +-38.3 | +60.8 +-39.2 | +97.4 +-40.1  |

Reading: at 40ms the hybrid is time-bound at about 3.9k sims/move, so
budgets >= 10k never bind and are equal; 2k/5k truncate sims and lose.
At 200ms a small budget makes the engine STOP EARLY WITH TIME LEFT
(2k sims take about 40ms of the 200ms), hence the -92.5. The node
budget is a ceiling for runaway nodes, not a speed knob.

Variant matches (all NNUE, colors swapped; E1-vs-E2 Elo):

| Match                                        | 40ms          | 200ms         |
|----------------------------------------------|---------------|---------------|
| AB+B vs pure AB                              | +10.4 +-34.0  | -1.2 +-39.1   |
| AB+C vs pure AB                              | -1.7 +-33.9   | -5.8 +-39.3   |
| AB+D(base .15) vs pure AB                    | not run       | -11.6 +-39.1  |
| AB+B+C+D(.15) vs pure AB                     | not run       | +8.1 +-39.3   |
| MCAB+B vs production MCAB                    | -16.5 +-32.8  | not run       |
| MCAB+B+C+D vs production MCAB                | not run       | -17.4 +-38.5  |
| MCAB+E(d2,k8,f.25) vs production MCAB        | not run       | -96.2 +-39.6  |
| MCAB+E(d3,k12,f.15) vs production MCAB       | not run       | -398.4 +-65.0 |

VERDICT: no cheap policy trick beats production; direction E is harmful.
Recommended configuration = production unchanged. All toggles stay in the
tree default-off, pinned by tests/test_policy_ab.cpp.

Pareto (equal TC, Elo over pure AB): pure AB < AB+tricks (~0) <
reduced MCAB (5k@200ms: +10.4; 10k: +60.8) < full MCAB (+97.4).
At 40ms: full MCAB +54.3, knee already at 10k (never binds).

### Fixes made during experiments (session 3)
- mcab.hpp: pre-ranking now carves its time OUT of the move budget
  (was additive on top; would have given side E +25% wall time).
- bench_policy_ab.cpp block 1 resets ordering state and seeds history
  explicitly per position (searchShallow does neither on its own).
