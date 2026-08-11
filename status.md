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
- **Hybrid Search (MCTS + Alpha-Beta / Negamax)**:
  - Research and prototype a hybrid search combining Monte Carlo Tree Search (MCTS) rollouts/priors with Negamax alpha-beta pruning.
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

**`selfplay.hpp` / `selfplay_main.cpp`** — Multi-threaded self-play dataset generator. Supports traditional epsilon-greedy and fast Monte Carlo policy-temperature sampling (`--mc-mode`). Allocates memory on stack frames (`std::array`) to prevent MinGW TLS destructor heap corruption on thread exit.

**`teste/arena.cpp` / `run_arena.py`** — Head-to-head engine strength tester with Elo estimation and confidence intervals.

**`training/*.py`** — PyTorch NNUE training pipeline (`train_nnue.py`), QAT quantization (`quantize_nnue.py`), dataset reader (`read_selfplay.py`), parity verifier (`parity_check.py`), and orchestrator (`run_selfplay.py`).

**`gui_web/`** — `engine_wasm.cpp` C-API shell, compiled to `zquoridor.js`/`.wasm` for browser play (`index.html`, `app.js`).

---

## Design Decisions & Findings

### Search (`search.hpp`)

- **Move ordering / CAT**: Replaced single-path wall touch bonus with Corridor Attention Table (`cat.hpp`). Continuous per-cell heat computed once per node (2 BFS) yielded ~11.7× node count reduction to equivalent depth.
- **BFS Caching**: Node-level `PlayerPathCache` + global `PlayerPathCacheTable` (~48MB, keyed on wall topology) speed up search by ~57% + ~5% without extra BFS calls.
- **LMR + PVS + RFP + LMP**: LMR combined with LMP requires explicit guard (`reducedByLmr`) to prevent invalid pruning. Nodes-to-depth reduced ~0.19–0.22×.
- **Endgame race solver**: Activated when `wallsLeft[0]==0 && wallsLeft[1]==0`. Uses exact DP with a real-time budget (~3% of move budget). Includes root-move selection fix for accurate DTM move choice.
- **Policy-assisted move ordering**: NNUE policy logits boost move ordering in search. Gated by `policyOrderingMinDepth` (default 3) because policy evaluation (`209×256` MACs) costs ~5.8× more than leaf value evaluation.

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
| Self-play `.bin` | `TrainingSample::evalNNUE` (`selfplay.hpp`, `teste/arena.cpp`) | NNUE WL head's own opinion of the position, sigmoid'd to a win probability, computed via `buildAccumulatorQuant`+`nnueWinProbQuant` **before** the move is chosen (independent of whether the move itself comes from random play, shallow search, or full search) | **absolute color**, not mover-relative: 0 = Black certain win, `EV_SCALE` (65535) = White certain win — chosen this way so the field can be read without needing `mover` alongside it just to know "how White is doing" | `uint16_t` 0..65535 (`EV_SCALE`) |
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
