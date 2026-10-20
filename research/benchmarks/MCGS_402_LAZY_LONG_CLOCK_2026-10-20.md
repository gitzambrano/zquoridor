# 4.02 long-clock Lazy experiment — 20 October 2026

## Scope
Controlled local native searches. No new GitHub Actions; production `main` remains unchanged. Locally compiled Linux C++17 `-O3 -march=native -mavx2 -mfma`; original source was the archived 4.01 source snapshot with the known 4.02 Lazy+Sparse transformations, **not a byte-for-byte checkout of current main**. Same 858-feature 512-hidden NNUE and Adaptive Time, no pondering.

## Fixed clock results (original INT8 policy)

| Native clock scenario | Eager and Lazy agree | Eager mean simulations | Lazy mean simulations |
|---|---:|---:|---:|
| Fresh 1+1, 4 roots | 4/4 | 137035 | 167634 (+22.3%) |
| Fresh 3+2, 4 roots | 3/4 | 293854 | 362680 (+23.4%) |
| 3+2, repeated-root reuse, 2 positions x 2 searches | 4/4 | 245982 | 284118 (+15.5%) |
| 5+3, repeated-root reuse, 2 positions x 2 searches | 4/4 | 346111 | 398196 (+15.1%) |
| 5+3 fresh center-rush, 2 roots, corrected prep timer | 2/2 | 475988 | 520640 (+9.4%) |

**Divergent root:** fresh 3+2 center position `center_1153` Eager `e5v` versus Lazy `f4v`. The experiment does not establish which move has higher value or any Elo difference.

## Root preparation was not charged to the move's clock
In `chooseMoveMCAB`, the simulation clock started **after** potentially expensive reroot/compaction, graph-depth normalization, transposition index rebuilding and root preparation. The experimental fix begins the tree phase clock before setup, after an optional pre-ranking pass already deducted separately. All game clocks use the same rule, without time-control-specific parameters.

Reproduced local *same-root 5+3*:
- Eager second search: **17016 ms -> 13219 ms** (adaptive target 13219 ms), same `e2`.
- Lazy second search: **13927 ms -> 13219 ms**, same `e2`.

Reproduced local **child reroot after playing `e2`** in native 3+2, with `treeHit=1`:
- Eager second search: **8932 ms -> 8115 ms**, same `e8`.
- Lazy second search: **9127 ms -> 8115 ms**, same `e8`.

Additional fixed 200ms two-search UCI smoke checks for Eager and Lazy passed with identical selected moves before/after and correct reuse. These are deterministic illustrative local timings, not a full stress test or a guarantee that realized time equals a later-changing adaptive target on every root.

## Cross-check on operator-selected Apex Policy25 INT8 checkpoint
In a separate local test, with exactly the same patched executable and only weights replaced (Apex SHA-256 `fe8473255fd219fab247b1a7d44be25384a6cfc5daa5c301def49109badb7f2b`):
- 3+2 center `center_1153`: both played `f4v`; Eager 286721 sims, Lazy 345738 (+20.6%).
- 3+2 normal `normal_33`: both played `d1`; Eager 273841 sims, Lazy 282835 (+3.3%).
- 5+3 center `center_1153`: both played `f4v`; Eager 477549 sims, Lazy 540858 (+13.3%).
- The time fix behaved correctly in the child-reroot examples. One Apex fresh-search result still ended after its *reported final adaptive target*: a changing target/classification can explain this without violating the hard search limit. A distinct hysteresis experiment would require a strength gate.

## Experimental implementation and status

- `src/mcab.hpp`: genuinely count tree preprocessing against hard/adaptive budget.
- `tools/external/zquoridor_uci.cpp`: **opt-in** `--unified-lazy-search` flag keeps Lazy regardless of whether the controller sends `go movetime` or native `go wtime/btime`, without protocol-specific time thresholds. The production default is unchanged.
- No production promotion, build-system change or WASM build. Benchmarks used fresh and repeated roots; real 1+1 / 3+2 / 5+3 competitive Elo still unmeasured. Operator must approve production behavior change after correct-clock paired games.

The raw local files and trial scripts are packaged in the conversation artifact `zquoridor_402_lazy_long_clock_local_results.zip`.
