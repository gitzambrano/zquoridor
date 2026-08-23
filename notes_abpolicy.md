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

## Decisions

- Keep all four inherited pieces (A not implemented; directions B/C/D/E are,
  F needs no code beyond McabParams.nodeBudget which already exists).
