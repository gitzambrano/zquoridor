# Status

Design decisions, rationale, important findings, and open items for Zquoridor. `readme.md` describes what the code does now; this file explains *why* it's built that way, what was tried and rejected, and what's left to do. See `CLAUDE.md` for the convention this file follows.

---

## Module & function reference

Code-level detail (function/class names, file-by-file) that `readme.md` deliberately leaves out in favor of describing features. Useful when working on the code directly.

**`rules.hpp`** — `State` holds both pawn cells, two 64-bit wall bitboards (H/V), side to move, and a Zobrist hash. `legalMoves`/`pawnStepMoves`/`legalWallMoves` generate moves; wall legality requires both players to keep a path to goal, checked by a cheap geometric pre-filter followed by the rollback union-find in `dsu.hpp`. Four BFS variants share one engine: `hasPathToGoal`, `shortestPathLen`, `shortestPathTouchSlots`, `pathRobustness`. `evalSimple` is the heuristic evaluator. `moveToPolicyIndex`/`mirrorMoveForPerspective` map a `Move` to/from the NNUE policy head's 209-way output index in canonical (mirrored) perspective.

**`dsu.hpp`** — union-find with rollback, used to check wall legality incrementally.

**`cat.hpp`** — Corridor Attention Table: per-cell heat computed once per node (2 BFS), feeding wall-move ordering.

**`search.hpp` (`Negamax`)** — `chooseMove` is the iterative-deepening entry point (time or depth budget). Core loop is negamax with alpha-beta, a Zobrist-keyed transposition table, killer moves + history heuristic, `orderPawnMoves`/`orderWallMoves`. LMR+PVS and RFP+LMP apply at shallow depth; wall quiescence extends search past nominal depth for critical-looking walls. `RepetitionTable` handles 3-fold repetition with contempt. The endgame-race hook detects `wallsLeft[0]==0 && wallsLeft[1]==0` and defers to `endgame_race.hpp`. Policy-assisted ordering adds the NNUE policy logit as an extra ordering term. Runtime toggles: `setEvalMode(Heuristic|NNUE)`, `setLmrPvsEnabled`, `setRfpEnabled`, `setLmpEnabled`, `setQuiescenceEnabled`, `setPolicyOrderingEnabled`, `setPolicyOrderingMinDepth`.

**`endgame_race.hpp`** — `raceDisjointGate` is the cheap gate (skips search when both players' reachable regions are disjoint); `raceExactDTM` is the exact retrograde DP over the 81×81×2 state space, cached per wall topology, real-time-budgeted.

**`nnue.hpp`** — `buildAccumulatorQuant(state, perspective)` builds an int8 accumulator from scratch; incremental update functions keep it in sync per move without rebuilding. `forwardValueWLQuant`/`forwardValueAuxQuant` are the two value heads; `forwardPolicyQuant` is the policy head. `nnueEvalInt` is the leaf-eval entry point search calls. `policyLogitForMove` reads one move's logit out of a `forwardPolicyQuant` output. `AccPair` is the per-ply, per-perspective accumulator pair search maintains on its stack (`nnueAccStack`).

**`selfplay.hpp`/`selfplay_main.cpp`** — multi-threaded self-play generator; `SelfPlayConfig` holds all settings (see `readme.md` for the CLI flags it maps to).

**`teste/arena.cpp`/`run_arena.py`** — plays two engine builds head-to-head. `run_arena.py` compiles both refs into `teste/bin/arena.exe` and drives parallel worker processes; `arena.cpp` can also be built/run standalone via `build/build_arena.sh`.

**`training/*.py`** — `train_nnue.py`/`train_nnue_numpy.py` train the network from `.bin` self-play data (flags cover data/checkpoint selection, optimizer, early stopping, memory budgeting, per-head loss weights). `quantize_nnue.py` does standalone post-training quantization. `read_selfplay.py` is the numpy reader for the self-play binary format. `parity_check.py` is the Python half of the `nnue_verify` C++/Python parity check. `run_selfplay.py` orchestrates `bin/selfplay`.

**`gui_web/`** — `engine_wasm.cpp` is an `extern "C"` shell exposing `rules.hpp`/`search.hpp` to JS. `index.html`/`app.js` are the GUI. `quoridor.html` is a standalone single-file bundle from `build_standalone.py`. `quoridor.js`/`.wasm` are the compiled output.

### Tests (`teste/`)

| File                                               | Checks                                                                                         |
| -------------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| `test_rules_sanity.cpp`                          | move generation, wall-legality pre-filter + DSU, perft                                         |
| `test_search_staging.cpp`                        | staged move generation vs. a monolithic reference                                              |
| `test_move_ordering.cpp`                         | Corridor Attention Table shape and its use in`orderWallMoves`                                |
| `test_endgame_race.cpp`                          | endgame solver vs. exact DP, cache reuse,`chooseMove` end-to-end from a frozen-topology root |
| `test_lmr_pvs.cpp`                               | LMR+PVS+RFP+LMP vs. a full-window reference (agreement thresholds, never an illegal move)      |
| `nnue_verify.cpp` + `training/parity_check.py` | C++ vs. Python numerical parity (float32 and int8)                                             |
| `nnue_incremental_check.cpp`                     | incremental accumulator vs. rebuild-from-scratch                                               |
| `nnue_sign_check.cpp`                            | NNUE eval sign/perspective sanity vs.`evalSimple`                                            |
| `bench_quiescence_toggle.cpp`                    | nodes/s and node count with/without quiescence                                                 |
| `bench_lmr_pvs.cpp`                              | nodes-to-depth and head-to-head games with/without LMR+PVS+RFP+LMP                             |
| `bench_wall_touch_bonus.cpp`                     | nodes-to-depth with/without CAT-based wall ordering                                            |
| `bench_race_regression.cpp`                      | endgame-race solver performance regression                                                     |
| `tune_spsa.cpp`                                  | SPSA tuner for`evalSimple` weights                                                           |

---

## Documentation conventions

- `readme.md` presents the engine as NNUE-only by design choice — it deliberately doesn't mention `evalSimple`/heuristic mode or the `setEvalMode(Heuristic|NNUE)` runtime switch, even though both exist in `search.hpp` and are used internally (e.g. as the self-play/training scaffold target for the auxiliary head, and available via `--heuristic` on `selfplay`/`arena` even though that flag isn't documented in the readme either). If the heuristic mode ever becomes reader-relevant again (e.g. NNUE becomes optional rather than default), reconsider this.

## Search (`search.hpp`)

- **Move ordering / CAT**: wall ordering originally used a simpler binary bonus ("does this wall touch the witness shortest-path or not"), which only saw one route. Replaced with the Corridor Attention Table (`cat.hpp`, Prioridade 1) — continuous per-cell heat, computed once per node (2 BFS) instead of per wall candidate — which also credits walls that close low-cost detours outside that one route. Measured (`bench_wall_touch_bonus.cpp`, 40 fixed positions, 200ms/move): ~11.7× fewer nodes to equivalent depth vs. the binary bonus.
- **BFS caching** (Prioridades 6/6b): the same BFS (`shortestPathLen`/`pathRobustness`/`shortestPathTouchSlots`) was being recomputed multiple times per node (ordering, quiescence, eval) even for identical (wall topology, pawn, player) tuples. `PlayerPathCache` (per node) + `PlayerPathCacheTable` (~48MB, keyed on wall topology, not full position — shared across sibling nodes/transpositions) eliminated this. Measured (`bench_fixed_depth.cpp`, same node count before/after): ~57% (per-node cache) + ~5% (cross-node cache) speedup.
- **LMR + PVS** (Prioridades 3/8): never reduces the TT move, killers, or a CAT-hot wall. Toggle: `setLmrPvsEnabled(false)`.
- **RFP + LMP** (Prioridades 3b/3c): never applied at the root or to TT/killer/hot-wall moves. **Known pitfall, fixed**: LMR combined with LMP without an extra guard (`reducedByLmr` in `negamax`) was catastrophically worse than either alone (0-10 in isolated head-to-head) — LMP's aggressive pruning was contaminating LMR's reduced-verification search, preventing the safety re-search LMR depends on. See Prioridade 3c for the full analysis. Validated in `test_lmr_pvs.cpp` / `bench_lmr_pvs.cpp`: nodes-to-same-depth ~0.19–0.22×, head-to-head 6-3-1 in favor of enabled (small sample, not SPRT — see Prioridade 14).
- **Wall quiescence**: extension bounded by `QS_MAX_EXTRA_PLIES`; trigger thresholds (`QS_CRITICAL_BFS_DELTA`, `QS_CRITICAL_ROBUSTNESS_DROP_TO`) are still initial/untuned values inherited from the design doc, not empirically calibrated. Toggle: `setQuiescenceEnabled(false)`.
- **Contempt**: `CONTEMPT = -30`, applied in both `negamax` and quiescence, to bias the engine away from proposing draws in neutral/favorable positions while still allowing a draw as a defensive resource.
- **Endgame race solver** (`endgame_race.hpp`, Prioridade 4): **read the header comment in that file and Sections 4d/4e of `plano-additional.md` before touching it.** Four rounds of corrections after first integration:
  1. A cheaper ETA gate (Level 1 of the plan) was tested and discarded — decided wrongly on a real counterexample (physical blocking can cost more time than predicted).
  2. First version of the region gate tested only shortest-path disjunction (doesn't guarantee no interaction) — corrected to full reachable-region disjunction.
  3. The 1-slot per-topology cache alone was insufficient (real hit rate ~0.5% while placing the last walls) — added the real-time budget (`g_raceExactBudgetUs`, ~3% of the total move budget) so a cache-miss storm falls back to heuristic search instead of degrading nodes/s below baseline.
  4. **Most critical**: a move-*choice* bug (not a value bug) — when the real game root itself (not an internal node) was already at `wallsLeft==(0,0)`, the shortcut returned only the value, and `chooseMove` read a placeholder move from the TT instead of the move achieving the optimal DTM. The engine knew who won but played arbitrary moves — lost most games despite healthy nodes/s. Fixed by comparing root candidates via 1-ply exact value before the iterative-deepening loop. Validated via `teste/arena.cpp` (two real git refs in one binary): ~113/502 wins (Elo≈−132) before fix → ~46.5% score in 100 games after (within statistical noise of that sample).
- **Policy-assisted move ordering** (`prompt_policy_ordering.md`, 2026-08): adds the NNUE policy head's logit as an extra term in `orderPawnMoves`/`orderWallMoves`.
  - **Performance finding**: `forwardPolicyQuant` costs ~5.8× more than the leaf eval (`forwardValueWLQuant`) — `209×256=53,504` MACs vs. `~9,200`. First implementation called it unconditionally on every internal node; since shallow/near-horizon nodes dominate node count by orders of magnitude in an alpha-beta tree, this dropped nodes/s by ~3.3× and cost ~68 Elo in a 10k-game arena test.
  - **Fix**: `setPolicyOrderingMinDepth(int)` (default 3) — the forward pass only runs at nodes with remaining depth `>= policyOrderingMinDepth`, restricting the cost to the (few) nodes near the top of the tree where a cutoff has the highest ROI per call. Not yet independently re-benchmarked for Elo after the fix (only nodes/s recovery was confirmed); default 3 is an untuned initial guess.
  - Exposed identically across `search.hpp` (`setPolicyOrderingEnabled`/`setPolicyOrderingMinDepth`), `arena.cpp` (`--policy-order[-min-depth]`, `--e1-`/`--e2-` variants, `EN_POLICY_ORDER[ING]_DEFAULT` constants), `selfplay.hpp`/`selfplay_main.cpp` (`SelfPlayConfig::policyOrderingEnabled`/`policyOrderingMinDepth`, same flags), and both Python wrappers (`run_arena.py`, `run_selfplay.py`).
  - Default is **off** everywhere; not yet validated as a net strength gain — the person is running their own nodes/s + Elo benchmarks to decide.
  - `arena.cpp`'s cross-ref compatibility (`trySetPolicyOrdering`/`trySetPolicyOrderingMinDepth`, SFINAE) means pointing a git ref that predates this feature at `--policy-order` compiles and runs fine — the flag is silently a no-op on that side, with a warning printed at startup (`hasPolicyOrdering<Eng>` detector).

## NNUE (`nnue.hpp`)

- **Canonical perspective / symmetry bug (2026-08, fixed)**: `featOwnPawn`/`featWallH`/`featWallV` originally used raw board coordinates and only swapped which pawn was "mine" without row reflection — the initial symmetric position produced different evals for each perspective. Fixed by reflecting rows (`r → WS-1-r` walls, `r → N-1-r` pawns) when `perspective==1`; columns are never reflected (board symmetric on that axis). Caught by `nnue_sign_check.cpp`.
- **Remaining-walls buckets (features `[332,354)`, 2026-08, added)**: `nnue_sign_check.cpp` found that two positions identical in pawns/walls but different wall *counts* produced identical eval, because the network had no way to see wall counts. `NUM_FEATURES` went from 332 to 354. Weights trained with `NUM_FEATURES=332` are incompatible with the current build — `loadFromFile` checks file size and rejects mismatched formats with an explicit error.
- **QAT, not post-hoc quantization**: `QA=255`/`QB=64` are fixed *before* training; a `WeightClipper` (Stockfish `nnue-pytorch` style) clamps weights to int8/int16 range every optimizer step. These constants must stay in sync across `nnue.hpp`, `train_nnue.py` (`--qa`/`--qb`), and `quantize_nnue.py`, or `nnue_verify` parity breaks.
- **Auxiliary head**: exists as a training scaffold (imitates `evalSimple`) while self-play is generated from the heuristic engine. Planned removal once self-play is generated from the network itself.

## Build / tooling

- **`build_tests.sh`/`.bat` has a known pre-existing bug**: it looks for `lazy_acc_parity.cpp` under `teste/`, but the file lives in `src/`. Compile it manually (`g++ -O2 -std=c++17 -pthread -Isrc -o bin/lazy_acc_parity src/lazy_acc_parity.cpp`) until the script is fixed.
- **`build_arena_common.py` GCC quirk**: compiling `arena.cpp` twice against byte-identical headers written within the same second can make GCC's multiple-include handling conflate the two copies (headers appear "already included" even from different paths), silently skipping the second engine's namespace. `build_arena_common.py` already works around this — don't hand-roll a `g++` invocation for engine1==engine2 self-tests without checking that script first.
- **`gui_web` build script drift**: `build_wasm.bat` exports more functions (`_qr_load_nnue_weights`, `_qr_set_eval_heuristic`, `_qr_eval_mode_is_nnue`) than `build_wasm.sh`'s `EXPORTED_FUNCTIONS` list. Keep both in sync when adding a new WASM export.

## Uncalibrated / placeholder values

Several thresholds are still initial guesses inherited from the design doc or another engine, not empirically tuned:

- `QS_CRITICAL_BFS_DELTA`, `QS_CRITICAL_ROBUSTNESS_DROP_TO` (wall quiescence trigger)
- `RFP_MARGIN_*`, `LMP_COUNT_*`
- `robustnessWeight` (evalSimple)
- `POLICY_ORDER_SCALE` (400) and `policyOrderingMinDepth` (3) — see Policy-assisted move ordering above

`teste/tune_spsa.cpp` exists for SPSA-tuning `evalSimple` weights (checkpoints to `spsa_checkpoint.txt`, run from repo root) but hasn't been run against the thresholds above.

## Open items / future plans

- Re-run an Elo benchmark for policy-assisted move ordering now that the min-depth fix is in, and tune `policyOrderingMinDepth`/`POLICY_ORDER_SCALE` empirically.
- Remove the NNUE auxiliary head once self-play data comes from the network itself instead of the heuristic engine.
- WASM: engine move currently runs synchronously on the main thread, blocking the tab for the configured think time (200ms–4s). Migrating to a Web Worker is possible without changing `engine_wasm.cpp` — only how JS loads the module — not yet done.
- SPSA-tune the uncalibrated thresholds listed above.
- Real strength claims should come from `run_arena.py` games, not nodes/s alone — nodes/s changes (faster or slower) don't by themselves imply better or worse play.

## Changelog (dated entries)

- **2026-08**: NNUE symmetry bug fixed (canonical perspective row reflection).
- **2026-08**: NNUE remaining-walls-bucket features added (`NUM_FEATURES` 332→354); older weight files rejected on load.
- **2026-08**: Policy-assisted move ordering implemented (`prompt_policy_ordering.md`), found to regress nodes/s ~3.3× unconditionally, fixed with a depth floor (`setPolicyOrderingMinDepth`, default 3), wired through `arena.cpp`/`selfplay`/both Python orchestrators.
