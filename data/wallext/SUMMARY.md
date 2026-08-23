# Wall-quiescence extension experiments -- results summary

All matches: games start from the low-wall corpus (`tools/wallext/corpus.hpp`,
360 positions, reserve totals 2..10), colors swapped per pair, single-thread
processes, `wallext_arena.exe` pure alpha-beta `chooseMove` (depth cap 8,
time control 100 ms/move). Elo margin is 95%, same formula as
`tools/arena/run_arena.py`. Machine was under background self-play load for
every run, so nps figures are indicative only; node counts are exact.

## Heuristic mode (previous session, validated)

Screen: 300 games/pairing (logs `screen_heuristic_*.log`).

| variant | W-D-L vs baseline | Elo(B-A) | +-95% |
|---------|-------------------|----------|-------|
| b1t2    | 146-7-147         | -1.2     | 38.9  |
| b1t4    | 156-7-137         | +22.0    | 38.9  |
| b1t6    | 148-7-145         | +3.5     | 38.9  |
| b2t2    | 148-6-146         | +2.3     | 38.9  |
| b2t4    | 150-4-146         | +4.6     | 39.1  |
| b2t6    | 148-4-148         | -0.0     | 39.1  |
| b3t2    | 155-1-144         | +12.7    | 39.3  |
| b3t4    | 145-2-153         | -9.3     | 39.2  |
| b3t6    | 155-4-141         | +16.2    | 39.1  |
| off     | 138-2-160         | -25.5    | 39.3  |

Top-up to 700 games (`topup_heuristic_*.log`, first 300 replayed the screen
seeds):

| variant | W-D-L vs baseline | Elo(B-A) | +-95% |
|---------|-------------------|----------|-------|
| b1t4    | 352-9-339         | +6.5     | 25.6  |
| b3t2    | 363-9-328         | +17.4    | 25.6  |
| b3t6    | 337-13-350        | -6.5     | 25.5  |
| off     | 295-11-394        | -49.5    | 25.8  |

Reading: every rule variant is inside noise of baseline; quiescence-off is
clearly worse, which shows the harness detects real effects.

## Perf grid (this session, exact node counts)

`perf_grid_stride24.log`: 7 corpus positions (reserve totals 2..5), fresh
engine per position, full window, LMR on.

| config   | nodes d8 | ratio | nodes d10 | ratio |
|----------|----------|-------|-----------|-------|
| baseline | 3359278  | 1.000 | 12462913  | 1.000 |
| b=1,t=2  | 3289482  | 0.979 | 14038824  | 1.126 |
| b=1,t=4  | 3336433  | 0.993 | 13836477  | 1.110 |
| b=1,t=6  | 3336433  | 0.993 | 13836477  | 1.110 |
| b=2,t=2  | 3322550  | 0.989 | 13942508  | 1.119 |
| b=2,t=4  | 3322221  | 0.989 | 13664669  | 1.096 |
| b=2,t=6  | 3322221  | 0.989 | 13664669  | 1.096 |
| b=3,t=2  | 3322550  | 0.989 | 13942508  | 1.119 |
| b=3,t=4  | 3322253  | 0.989 | 13664675  | 1.096 |
| b=3,t=6  | 3322253  | 0.989 | 13664675  | 1.096 |

Notes: threshold 4 and 6 coincide because every measured position has
total <= 5. Node cost of the rule is small (+~11% worst case at d10,
slightly negative at d8): wall quiescence rarely reaches qply >= 2 even in
low-wall positions.

## NNUE mode (this session)

Screen: 300 games/pairing (`nnue300_nnue_*.log`).

| variant | W-D-L vs baseline | Elo(B-A) | +-95% |
|---------|-------------------|----------|-------|
| b1t2    | 142-8-150         | -9.3     | 38.8  |
| b1t4    | 135-5-160         | -15.1    | 38.8  |
| b1t6    | 153-4-143         | +11.6    | 39.1  |
| b2t2    | 136-4-160         | -15.1    | 39.0  |
| b2t4    | 148-8-144         | +4.6     | 38.8  |
| b2t6    | 147-5-148         | -1.2     | 39.0  |
| b3t2    | 147-4-149         | -1.2     | 38.9  |
| b3t4    | 151-6-143         | +9.3     | 38.9  |
| b3t6    | 148-6-146         | -0.0     | 39.1  |
| off     | 123-9-168         | -52.5    | 39.2  |

Confirm stage: top-3 point estimates re-run on independent corpus slices
(start offsets 200/220/240 vs 17k used by the screen), 400 games
(`confirm_nnue_*.log`).

| variant | W-D-L vs baseline | Elo(B-A) | +-95% |
|---------|-------------------|----------|-------|
| b1t6    | 201-5-194         | +6.1     | 33.8  |
| b3t4    | 196-5-199         | -2.6     | 33.8  |
| b2t4    | 201-4-195         | +5.2     | 33.9  |

Pooled screen+confirm per variant (700 games each): b1t6 +8.4 +-25.6,
b2t4 +4.9 +-25.7, b3t4 +2.5 +-25.7 -- none significant.

## Correctness (this session)

`test_wall_qextension` full run (`test_wq_run2.log`, ~105 min at -O2 under
load): part A exact default-identity 135/135 heuristic + 8/8 NNUE
(score AND node count) against the regenerated reference; part B
agreement 73.5%/74.5%/100% with zero illegal moves; part C stack stress
(caps 6+6, threshold 20) clean. The standard suite (rules_sanity,
search_staging, move_ordering, endgame_race, lmr_pvs, repetition_diff)
passed on the featured headers before any experiment ran.

## Conclusion

The low-walls extension bonus does not produce a measurable strength gain
in either eval mode at 100 ms/move from low-wall positions, despite costing
only ~11% extra nodes at depth 10. Recommendation: keep production defaults
(bonus 0 / threshold 0); the knobs stay available for future re-testing at
other time controls.
