# Status

Design decisions, rationale, important findings, and open items for Zquoridor. `readme.md` describes what the code does now; this file explains *why* it's built that way, what was tried and rejected, and what's left to do. See `CLAUDE.md` for the project guidelines.

---

## Future Plans & Roadmap

- **External Arena Integration**:
  - Connect Zquoridor to [Claustrophobia Engine Arena](https://claustrophobia.dev/).
  - Integrate engine with `path-clash-bot-arena` for automated match testing against other Quoridor bots.
- **Generation 5 NNUE Dataset & Training**:
  - Generate a massive Gen 5 self-play dataset using the fast Monte Carlo policy-temperature sampling mode (`--mc-mode`).
  - Train and quantize the Gen 5 NNUE network (`train_nnue.py` + `quantize_nnue.py`) with updated policy-loss weights and dataset blending.
- **Genetic Algorithm (GA) & SPSA Engine Parameter Tuning**:
  - Implement and run GA / SPSA parameter optimization for uncalibrated search thresholds (`RFP_MARGIN_*`, `LMP_COUNT_*`, `QS_CRITICAL_*`, `robustnessWeight`, `POLICY_ORDER_SCALE`, `catScoreScale`).
- **Hybrid Search (MCαβ)** — implemented and off by default (`tools/common/mcab.hpp`); see the design note below. What is left is *merit*, not plumbing:
  - ~~Run the real strength match across several `mcabNodeBudget`/`mcabLeafDepth` pairs~~ — done at 200ms/move; the hybrid wins by ~26 Elo, but only at `leafDepth=0`.
  - ~~Tune the MCαβ parameters once a working point is known~~ — swept by hand around `leafDepth=0`; only `fpuReduction=0.0` improved. `tune_spsa --mcab-tuning` has never been run to convergence.
  - Open: is the +46.9 ±23.5 Elo edge worth shipping? It costs ~9× the nodes/s (359k → 41k) and only shows up with `leafDepth=0`, i.e. as plain PUCT MCTS, not as MCαβ. Measure at other time controls before deciding — 200ms is the only one tested, and the trade almost certainly inverts as time shrinks.
  - Open: sweep `nodeBudget` (fixed at 20000 throughout, never the binding constraint at 200ms — time was) and re-check whether `adaptiveLeafDepth` changes the picture now that it actually works.
  - Decide whether self-play data generation should switch to `--mcab` (root Dirichlet noise is already wired for that use).
- **WASM Web Worker UI**:
  - Move the WebAssembly engine execution in `gui_web/` to a background Web Worker so deep engine searches do not block the browser UI thread.

---

## Module & function reference

Code-level detail (function/class names, file-by-file) for developer reference.

**`rules.hpp`** — `State` holds pawn positions (0..80), 64-bit wall bitboards (H/V), remaining walls per player, side to move, and Zobrist hash. Move generation (`legalMoves`, `pawnStepMoves`, `legalWallMoves`) uses a fast geometric pre-filter followed by rollback union-find in `dsu.hpp` for path validation. Four BFS variants share one engine: `hasPathToGoal`, `shortestPathLen`, `shortestPathTouchSlots`, `pathRobustness`. `evalSimple` is the heuristic evaluator. `moveToPolicyIndex`/`mirrorMoveForPerspective` convert moves to/from canonical policy index (0..208).

**`dsu.hpp`** — Union-find with rollback for fast incremental wall legality checking.

**`cat.hpp`** — Corridor Attention Table (CAT): per-cell heat computed once per node (2 BFS), feeding wall-move ordering.

**`search.hpp` (`Negamax`)** — Iterative deepening search with negamax alpha-beta, Zobrist transposition table, killer moves + history heuristic, `orderPawnMoves`/`orderWallMoves`. Pruning heuristics: LMR+PVS, RFP+LMP at shallow depths, wall quiescence extension, 3-fold repetition with `CONTEMPT = -30`. Endgame race solver hook (`wallsLeft==(0,0)`). Policy-assisted move ordering using NNUE policy logits (gated by depth floor `policyOrderingMinDepth`). Runtime toggles for all heuristics.

**`endgame_race.hpp`** — Disjoint-region gate (`raceDisjointGate`) and exact retrograde DP solver (`raceExactDTM`) over 81×81×2 state space, cached per wall topology with real-time budget.

**`nnue.hpp`** — 354 → 256 accumulator (SCReLU) → two heads: WL outcome head (`256→32→1`) and Policy head (`256→209`). `buildAccumulatorQuant` / `updateAccumulatorForMoveQuant` maintain fixed-point quantized accumulators incrementally. `nnueEvalInt` provides leaf evaluation for search; `nnueWinProbQuant` computes absolute win probability for self-play. `AccPair` manages per-ply per-perspective accumulators on the search stack (`nnueAccStack`).

**`tools/selfplay/` (`selfplay.hpp`, `selfplay_main.cpp`, `run_selfplay.py`)** — Multi-threaded self-play dataset generator and Python orchestrator. Supports traditional epsilon-greedy and fast Monte Carlo policy-temperature sampling (`--mc-mode`). Allocates memory on stack frames (`std::array`) to prevent MinGW TLS destructor heap corruption on thread exit.

**`tools/common/mcab.hpp`** — MCαβ hybrid: a best-first PUCT tree whose leaves are evaluated by a *real shallow alpha-beta search* (`Negamax::searchLeaf`) rather than by a random rollout, with minimax-hard backup (Huang, "Pruning Game Tree by Rollouts", AAAI 2015; the same shape Scorpio uses). `MCABSearch<...>` is a template over the engine type; `McabRunner`/`chooseMoveAuto` pick between the hybrid and pure AB **at compile time** via the `hasMcabSupport` SFINAE trait. Disabled by default everywhere — `params.enabled=false` delegates straight to `Negamax::chooseMove`. Lives in `tools/`, *not* `src/`, on purpose: `run_arena.py` checks out `src/` per git ref, so a header in `src/` could not be used to compile an engine from a ref that predates the feature.

**`tools/arena/` (`arena.cpp`, `run_arena.py`)** — Head-to-head engine strength tester with Elo estimation and confidence intervals. `--e1-mcab`/`--e2-mcab`/`--mcab` select the hybrid per engine; a ref older than the feature compiles fine, prints a one-shot warning, and plays pure AB.

**`tools/spsa/` (`tune_spsa.cpp`, `run_spsa.py`, `plot_spsa.py`)** — Multi-mode SPSA parameter tuner & visualizer.

**`tools/qtp/` (`qtp_main.cpp`)** — QTP text protocol interface.

**`training/*.py`** — PyTorch NNUE training pipeline (`train_nnue.py`), QAT quantization (`quantize_nnue.py`), dataset reader (`read_selfplay.py`), and parity verifier (`parity_check.py`).

**`gui_web/`** — `engine_wasm.cpp` C-API shell, compiled to `zquoridor.js`/`.wasm` for browser play (`index.html`, `app.js`).

---

## Design Decisions & Findings

### Search (`search.hpp`)

- **Move ordering / CAT**: Replaced single-path wall touch bonus with Corridor Attention Table (`cat.hpp`). Continuous per-cell heat computed once per node (2 BFS) yielded ~11.7× node count reduction to equivalent depth.
- **BFS Caching**: Node-level `PlayerPathCache` + global `PlayerPathCacheTable` (~48MB, keyed on wall topology) speed up search by ~57% + ~5% without extra BFS calls.
- **LMR + PVS + RFP + LMP**: LMR combined with LMP requires explicit guard (`reducedByLmr`) to prevent invalid pruning. Nodes-to-depth reduced ~0.19–0.22×.
- **Endgame race solver**: Activated when `wallsLeft[0]==0 && wallsLeft[1]==0`. Uses exact DP with a real-time budget (~3% of move budget). Includes root-move selection fix for accurate DTM move choice.
- **Policy-assisted move ordering**: NNUE policy logits boost move ordering in search. Gated by `policyOrderingMinDepth` (default 3) because policy evaluation (`209×256` MACs) costs ~5.8× more than leaf value evaluation.

### Hybrid MCαβ (`tools/common/mcab.hpp`)

- **Pure AB must not pay for it.** The hybrid is strictly additive: `benchmarks/bench_mcab.cpp` runs the exact fixed workload of `bench_fixed_depth.cpp` inside a binary that *includes* `mcab.hpp`. Node count is bit-identical (8,293,935); throughput sits inside CPU noise (1.32–1.37 M nodes/s across runs of both binaries).
- **`searchLeaf` must reset `stopped`/`deadline` per call.** `deadline` is a default-constructed `time_point` (the epoch), so a freshly built engine aborts on its very first time check, and a stale `stopped=true` from an earlier timed-out `chooseMove` sticks. Either way every leaf returned 0 → Q=0.5 across the whole tree, silently, with nothing crashing. Locked in by the regression block in `tests/test_search_leaf_smoke.cpp` (which clears the TT first — otherwise a TT hit satisfies "searched something" in one node).
- **The leaf time cap has to be propagated into `searchLeaf`.** The simulation loop only checks the clock *between* simulations, and one depth-4 leaf costs ~40ms in a midgame position: a 60ms budget measured ~110ms. `evaluateLeaf` now passes the remaining time down, and discards leaves that were truncated (counted in `McabStats::leafTruncated`) instead of feeding a partial score into Q.
- **Adaptive leaf depth keys on the parent's `totalN`, not the edge's `N`.** A leaf is evaluated exactly once, at creation, when the edge leading to it still has `N==0` — keying on the edge made the feature silently inert. This is the reading of Section 9 of the plan ("scales with N of the parent node"), and the log₄ step (+1 ply per 4× visits) is uncalibrated.
- **Equivalence mode isolates each child (TT cleared per child).** The real path shares the TT across neighbouring leaves on purpose; equivalence mode exists only to be compared against a yardstick that scores each move with a fresh engine. With a shared TT, siblings searched later inherit fail-soft bounds and score systematically higher — measured up to +105 on a single move, enough to swap two close moves and make the benchmark cry regression where there is none.
- **Phase 8 result: alpha-beta leaves are what makes the hybrid lose; the tree is what makes it win.** Measured at 200ms/move, local vs local, the only difference being `--e2-mcab` (Elo of pure AB relative to the hybrid, so negative = hybrid ahead):

  | `leafDepth` | MCTS nodes in 200ms | games | Elo (AB vs hybrid) |
  | --- | --- | --- | --- |
  | 3 | 24 | — | not played (tree too small to mean anything) |
  | 2 | 158 | 100 | **+338 ±89** (hybrid crushed) |
  | 1 | 981 | 100 | **+76 ±65** |
  | 0 | 3434 | 100 | **−56 ±64** |
  | 0 | 3434 | 400 | **−30 ±33** |
  | 0 | 3434 | 1000 | **−26.2 ±21.0** (44.2% / 51.8% / 4.8% draws — margin no longer crosses zero) |

  Quoridor's branching factor is ~130, so a tree of 24–158 nodes cannot visit the root's children even once each — the hybrid is then just a badly chosen shallow search. Every ply of leaf search costs roughly an order of magnitude of tree size, and the trade is not worth it at this time control. At `leafDepth=0` (NNUE value + wall quiescence, essentially no alpha-beta below the leaf) the hybrid beats pure AB by ~26 Elo — a margin that only became statistically separated from zero at 1000 games. Honest reading: what wins here is plain PUCT MCTS over the policy/value net, not the MCαβ idea of alpha-beta rollouts. Also worth noting the same trend in time: at `leafDepth=1`, pure AB is +76 ±65 at 200ms but only +38 ±81 at 1000ms — the hybrid scales better with time, as expected from a tree that keeps growing.
- **Parameter sweep around the `leafDepth=0` working point.** Hybrid vs hybrid, 300 games each at 200ms, both sides `--mcab --mcab-leaf-depth 0`, one knob changed on Engine 2 (Elo of the *default* config relative to the variant, so negative = variant better). Playing the configs against each other rather than each against pure AB removes most of the common-mode variance:

  | variant | Elo (default vs variant) |
  | --- | --- |
  | `cPuct` 2.5 (vs 1.5) | +91.2 ±36.7 |
  | `fpuReduction` 0.3 (vs 0.1) | +72.1 ±37.5 |
  | `scoreScale` 350 (vs 200) | +46.5 ±37.9 |
  | `rootSelectMode` visits-then-q | +45.4 ±37.3 |
  | `cPuct` 0.8 | +31.7 ±37.1 (inside the margin) |
  | `scoreScale` 120 | ±0.0 ±36.9 |
  | `fpuReduction` 0.0 | **−35.1 ±37.5** → **−24.4 ±22.9** over 800 games |

  **Combined config vs pure AB: −46.9 ±23.5 Elo over 800 games** (41.2% / 54.9% / 5.4% draws), hybrid ahead — this is the number that matters, since the 1000-game `leafDepth=0` result above was measured with the old `fpuReduction=0.1` and the `fpuReduction` result was measured hybrid-vs-hybrid. The two gains turned out to be near-additive (26.2 + 24.4 predicted, 46.9 measured). Cost: 359k nodes/s → 41k.

  Only `fpuReduction=0.0` improved on the plan's defaults, and only after 800 games did it separate from zero. `cPuct=1.5`, `scoreScale=200` and `rootSelectMode=MaxVisits` survive the sweep — they were guesses that happen to be near a local optimum, not measured values, and every deviation tried was worse. Defaults now in force: `leafDepth=0`, `fpuReduction=0.0` (in `mcab.hpp`, `arena.cpp` and `selfplay_main.cpp`, and as the GA `init` in `tune_spsa.cpp`). Every test and benchmark that exercises real alpha-beta leaves sets `leafDepth` explicitly, so none of them depend on the changed default.
- **The hybrid needs a real node budget to be worth anything.** At 60ms/move with `leafDepth=4`, the whole budget buys 1–3 MCTS nodes; at `nodeBudget=500`/`leafDepth=3` a move costs ~4,600 AB nodes per MCTS node and ~1.4MB of tree. Any strength claim must come from `run_arena.py` at a time control where the tree actually grows.

### NNUE Architecture (`nnue.hpp`)

- **Canonical perspective**: Rows reflected for perspective 1 (`r → N-1-r` pawns, `r → WS-1-r` walls) so the net evaluates from a single canonical frame.
- **Input Features (354)**: 81+81 pawn cells, 64+64 wall slots, 21+21 one-hot BFS distance buckets, 11+11 one-hot remaining-walls buckets.
- **Two Heads**: Outcome (WL) head (`256→32→1`) and Policy head (`256→209`). Legacy auxiliary head removed in 2026-08.
- **QAT (Quantization-Aware Training)**: Fixed `QA=255`, `QB=64` maintained during PyTorch training and export.
- **Thread Safety**: `selfplay.hpp` uses stack arrays (`std::array<..., 209>`) instead of dynamic TLS vectors to avoid MinGW-w64 thread-exit heap corruption (`STATUS_HEAP_CORRUPTION`).

---

## Evaluation: what each stage uses

Five different things all get called "the evaluation" depending on where you look, and they don't share a sign convention, perspective, or range. This trips people up when comparing a number from one stage against another, so here's the complete map:

| Stage | Symbol / entry point | What it computes | Perspective / sign | Range |
| --- | --- | --- | --- | --- |
| Search | `nnueEvalInt` (`nnue.hpp`), consumes `forwardValueWLQuant` | NNUE WL head raw logit, scaled ×`NNUE_EVAL_SCALE` (200) to an int on the same rough scale as `evalSimple` | mover-relative: positive = side to move is ahead | ~[-30000, 30000]; mate/certain-win positions saturate near the extremes |
| Heuristic fallback | `evalSimple` (`rules.hpp`), used only in `EvalMode::Heuristic` or via `--heuristic` on selfplay/arena/WASM | hand-tuned material + positional heuristic, uncalibrated weights | mover-relative | ~[-600, 600] in typical midgame positions |
| Self-play `.bin` | `TrainingSample::evalNNUE` (`selfplay.hpp`, `tools/arena/arena.cpp`) | NNUE WL head's own opinion of the position, sigmoid'd to a win probability, computed via `buildAccumulatorQuant`+`nnueWinProbQuant` **before** the move is chosen (independent of whether the move itself comes from random play, shallow search, or full search) | **absolute color**, not mover-relative: 0 = Black certain win, `EV_SCALE` (65535) = White certain win — chosen this way so the field can be read without needing `mover` alongside it just to know "how White is doing" | `uint16_t` 0..65535 (`EV_SCALE`) |
| Training target | `wl_target` in `to_chunk_tensors` (`train_nnue.py`) | blend of the real game outcome and the recorded `evalNNUE`: `wl_target = k · game_result_prob + (1-k) · ev_prob`, where `ev_prob` is `evalNNUE` reprojected from absolute-White back to mover-relative via the `mover` field, and `k` is chosen **per data source** in `DATA_SOURCES_DEFAULT`/`--data-sources` (`"k"`, default `1.0`) | mover-relative probability (same frame as `game_result_prob = (game_result+1)/2`) | float `[0, 1]` |
| HTML/WASM display | `formatEval`/`evalToWhitePercent` (`gui_web/app.js`, mirrored into `gui_web/zquoridor.html` and root `index.html`) | sigmoid of the raw search score (White-perspective, same units as the Search row) / 200, i.e. the inverse of the scaling `nnueEvalInt` applies going the other way | **absolute color**, always: 0% = Black certain win, 100% = White certain win, 50% = balanced. This is the one place a person actually reads a number, and it's deliberately *not* mover-relative — the percentage doesn't flip meaning depending on whose turn it is | `0–100%`, rounded to an integer |

Practical implications of this table:

- **`k=1.0` is required for any `.bin` recorded before this change** (same 32-byte `TrainingSample` layout, but that 2-byte slot used to hold a heuristic `evalSimple` score, not a probability). `k=1.0` makes `wl_target` reduce to `game_result_prob` exactly, so whatever garbage sits in that slot for old data is multiplied by zero and never touched — no format auto-detection needed, it's a source-by-source choice the person training makes (see `DATA_SOURCES_DEFAULT` comment in `train_nnue.py`).
- **`k<1.0` is a bootstrapping knob, not a free lunch.** It lets specific positions within a game "vote" with the network's own opinion at self-play time instead of only inheriting the final game result uniformly across every position of that game — the intuition is some positions are more responsible for the eventual win/loss than others, and this lets that signal in. Too aggressive a `k` (too close to 0) risks the new network just reinforcing whatever biases the *old* network (the one self-play used) already had, rather than learning from real outcomes. No default value below 1.0 is prescribed here — it's meant to be tuned by trial.
- `evalNNUE` is computed **once per position, before the move is played**, regardless of what selected the move (random exploration, shallow eval, or full search) — same "always cheap, always computed" pattern the old `searchScore`/`evalSimple` field followed, just with a forward pass through the (already-loaded) NNUE instead. If NNUE weights aren't loaded in that self-play run (heuristic-only mode), the forward pass over zeroed weights naturally degrades to logit 0 → 50% → the exact middle of `EV_SCALE`, with no special-casing needed.

---

## Uncalibrated Parameters

The following parameters are functional initial values awaiting GA / SPSA tuning:

- `QS_CRITICAL_BFS_DELTA`, `QS_CRITICAL_ROBUSTNESS_DROP_TO`
- `RFP_MARGIN_*`, `LMP_COUNT_*`
- `robustnessWeight` (`evalSimple`)
- `POLICY_ORDER_SCALE` (400) and `policyOrderingMinDepth` (3)
- `mcab::McabParams`: `leafDepth` (0) and `fpuReduction` (0.0) are **measured** at 200ms/move (see the sweep in the MCαβ note). `cPuct` (1.5), `scoreScale` (200, borrowed from `NNUE_EVAL_SCALE`) and `rootSelectMode` (MaxVisits) survived a sweep — every deviation tried was neutral or worse — but were never optimised, only defended. Still uncalibrated and untested: `nodeBudget` (20000), `leafDepthMax` (8), the log₄ growth rate of `effectiveLeafDepth`, and the Dirichlet noise `alpha`/`epsilon`. All measurements are at one time control (200ms) on one machine. `tune_spsa --mcab-tuning` exposes five of these as GA genes.

---

## Changelog

- **2026-08**: Fixed MinGW thread heap corruption in `selfplay.hpp` by replacing `thread_local std::vector` with stack-allocated `std::array`.
- **2026-08**: Introduced Monte Carlo policy-temperature sampling mode (`--mc-mode`) in self-play generator for fast opening generation.
- **2026-08**: Updated `TrainingSample` to 32 bytes packed carrying `evalNNUE` (network win probability) and CAT totals.
- **2026-08**: Removed legacy auxiliary NNUE head from model and quantization pipeline.
- **2026-08**: Added 11+11 remaining-walls bucket features (332 → 354 features).
- **2026-08**: Fixed NNUE perspective symmetry bug (row reflection for side 1).
- **2026-08**: Policy-assisted move ordering integrated and set to default ON with depth floor (`policyOrderingMinDepth = 3`).
- **2026-08**: Added SPSA hybrid mode (`--mode hybrid`), CSV logging, and `plot_spsa.py` visualization.
- **2026-08-13**: Hybrid MCαβ search (`tools/common/mcab.hpp`) implemented end to end and wired into arena, self-play, and the GA tuner — **off by default in all three**, so pure alpha-beta remains the shipped behaviour. Compile-time SFINAE dispatch (`hasMcabSupport`) keeps `arena.exe` compiling against git refs that predate the feature. New: `tests/test_mcab_core.cpp`, `tests/test_mcab_dispatch.cpp`, `tests/test_mcab_phase9.cpp`, `tests/test_search_leaf_smoke.cpp`, `benchmarks/bench_mcab.cpp`, `benchmarks/bench_mcab_equivalence.cpp` (all registered in `build/build_tests.*` and `build/build_bench.*`).
- **2026-08-13**: Fixed `Negamax::searchLeaf` not resetting `stopped`/`deadline`, which made every MCαβ leaf evaluate to 0 (Q=0.5) without any visible failure. Added `Negamax::searchWasStopped()` so the hybrid can discard time-truncated leaves.
- **2026-08-13**: Phase 8 (strength) run at 200ms/move. Hybrid loses badly with alpha-beta leaves (`leafDepth=2`: −338 Elo) and reaches parity/slight edge only with `leafDepth=0` (−30 ±33 Elo over 400 games, hybrid ahead). Default stays off; see the MCαβ design note for the full table.
- **2026-08-13**: `leafDepth=0` confirmed over 1000 games: **−26.2 ±21.0 Elo**, hybrid ahead, margin no longer crossing zero. Parameter sweep around that working point (hybrid vs hybrid, one knob at a time) found only `fpuReduction=0.0` better than the plan's defaults (−24.4 ±22.9 over 800 games); `cPuct=1.5`, `scoreScale=200` and `rootSelectMode=MaxVisits` beat every alternative tried. Defaults moved to `leafDepth=0`/`fpuReduction=0.0` in `mcab.hpp`, `arena.cpp`, `selfplay_main.cpp`, and as GA `init` in `tune_spsa.cpp` (whose `mcabLeafDepth` range was `1..20`, which could never reach the working point — now `0..20`). `run_arena.py` gained `--mcab-cpuct`/`--mcab-fpu`/`--mcab-score-scale`/`--mcab-root-select` in global and `--e1-`/`--e2-` forms, so configs can be played against each other instead of each against pure AB. Pure AB unchanged: `bench_fixed_depth` and `bench_mcab` block A still agree at 8,293,935 nodes, and all 9 tests plus `nnue_verify` pass.
