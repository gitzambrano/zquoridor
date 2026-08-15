# Status

Design decisions, rationale, important findings, and open items for Zquoridor. `readme.md` describes what the code does now; this file explains *why* it's built that way, what was tried and rejected, and what's left to do. See `CLAUDE.md` for the project guidelines.

---

## Future Plans & Roadmap

### Where production stands

The shipped search is the hybrid PUCT tree in `src/mcab.hpp`. It is on by default in arena, self-play, and the WASM GUI. The working point is `leafDepth=0`, `fpuReduction=0.0`, `cPuct=1.5`, `scoreScale=200`, and `rootSelectMode=MaxVisits`. At 200ms per move, over 800 games, that config is **+46.9 ±23.5 Elo** ahead of pure alpha-beta. The cost is roughly a nine-fold drop in alpha-beta nodes per second (about 359k down to 41k). Phase 8's honest reading still holds: what wins at this time control is the tree sitting on the policy/value net, not alpha-beta rollouts at the leaves. Pure alpha-beta is unchanged and bit-identical; `--no-mcab` selects it.

The network in front of that search is **Generation 5** (`data/nnue/nnue_weights.bin` and the quantized `_int8.bin` companion), trained on the Gen 5 epsilon and montecarlo mix described in `train_nnue.py`. The hybrid is the production search in front of that net, not a second experiment waiting to ship.

Two caveats matter for everything below. First, 200ms is still the only time control that has been measured, so the default is only known to be justified there. Second, alpha-beta nodes per second is the wrong success metric for this search. Quoridor's branching factor is around 130, and a 200ms tree was only about 3434 MCTS nodes with quiescence leaves, which is not enough for the root to visit every child even once. The numbers to watch are simulations per second and Elo.

The MCTS path now shares Negamax's `PlayerPathCacheTable` (`xdistCache`) when building NNUE accumulators: `buildAccPairRoot` and `makeChildAccPair` receive `Negamax::pathCache()` through SFINAE (`mcab::mcabPathCache`), so a git ref that predates the getter still compiles and still rebuilds BFS from scratch.

At `leafDepth=0`, leaves are `nnueEvalInt` on that accumulator. The tree no longer calls `searchLeaf` or wall quiescence for the production working point. Alpha-beta nodes per second at this setting will sit near zero by design; the metric that should go up is simulations per second. Whether skipping quiescence holds the +46.9 Elo still needs an arena match against the previous (QS-leaf) binary.

### What to do next

Work through the list in order. Each item is one change. Measure it with `run_arena.py` against current production at 200ms, with enough games to separate the result from zero, before starting the next. Combining items makes it impossible to tell which one moved Elo.

1. **Done in code (2026-08-14): `leafDepth=0` leaves call `nnueEvalInt`, not `searchLeaf`.** Quiescence at every new MCTS node was the main source of the counted alpha-beta nodes and did not grow the PUCT tree. The fast path is locked by `testFastLeafSkipsSearchLeaf` (`SearchStats.nodes == 0` while `leafSearches > 0`). Equivalence mode still uses `searchLeaf`, so `bench_mcab_equivalence` is unchanged. What remains is the Elo gate: play this against the previous binary (QS leaves) at 200ms. If Elo holds or rises, this stays; if it falls, revert the fast path.

2. **Done (2026-08-14): the MCTS path shares Negamax's BFS cache.** `buildAccPairRoot` and `makeChildAccPair` used to receive `nullptr` on every call. They now take `engine.pathCache()` via `mcab::mcabPathCache`, which is the same `PlayerPathCacheTable` quiescence already uses. Toy engines and older refs without the getter keep compiling and keep the old rebuild. This does not need an Elo gate: the cached value is a pure function of wall topology and pawn cell, so the search is identical and only cheaper.

3. **Try mean backup instead of minimax-hard.** `BackupMode::AvgBlend` exists as an enum and has never been implemented. Huang's minimax-hard backup is a good fit for deep alpha-beta leaves; production leaves are noisy NNUE values, so a single lucky child can dominate its parent. The standard MCTS mean (`W += v`, `Q = W/N`) should sit behind the existing enum, off by default, and be measured hybrid-versus-hybrid.

4. **Measure the time-control curve, then pick a switch.** Play the hybrid against `--no-mcab` at 50, 100, 500, and 1000ms, plus a budget typical of the browser GUI. The expected shape is that alpha-beta wins when the clock is short and the hybrid pulls away as the tree has time to grow. From that curve, a single rule is enough — for example alpha-beta below roughly 80–100ms, hybrid at 150–200ms and above — so the GUI is not running the 200ms config on a 50ms clock. `nodeBudget=20000` is not the binding constraint at 200ms; time is. Leave `adaptiveLeafDepth` off until this curve exists, because extra leaf plies already lost badly at 200ms.

5. **Progressive widening.** Production expands every legal move at every node, which in Quoridor is about 130 children, four heap vectors, and a full PUCT scan on every descent ply. A better shape is to unprune the top *k* moves by policy first (`k ≈ c · N^α`, starting around 8–16 at the root) and widen as visits grow. Keep it behind a knob, default off, until Elo says otherwise. This is the main strength lever on the list: more visits land on moves the net actually ranks, and PUCT gets cheaper as a side effect. Lazily legalizing the tail of the move list can wait until the policy-top-*k* version has been measured.

6. **Run the genetic algorithm on the MCTS working point.** `tune_spsa --mcab-tuning` already exists and has never been run to convergence. Pin `leafDepth` at 0; the tuner's current range is `0..20`, which will spend generations rediscovering that depth 2 dies. The genes that belong in the first run are `cPuct`, `scoreScale`, and `fpuReduction`, plus the widening constants once item 5 has landed. Dirichlet `alpha` and `epsilon` should be scored on self-play diversity, not on arena Elo. The old alpha-beta leftovers (`RFP_MARGIN_*`, `LMP_COUNT_*`, `QS_CRITICAL_*`, `POLICY_ORDER_SCALE`, `catScoreScale`) are secondary while production leaves skip alpha-beta. They still matter for `--no-mcab` and for `leafDepth>0`. `robustnessWeight` only touches `evalSimple`.

7. **Train the next net on root visit distributions.** Hybrid self-play is already on, with Dirichlet noise at the root, but there has been no A/B on the network that comes out of it, and `--mc-mode` still generates openings without a tree. Once items 1–5 have made simulations cheap, generate a shard whose policy target is the visit distribution at the root, train on it, and play the new net against Gen 5. That is Generation 6, not another pass over Gen 5.

8. **Reshape the tree only if a profiler still points here after 1–5.** The candidates are a structure-of-arrays edge arena instead of four `vector`s per node, stopping the `State` copy on every descent ply (`beforeState = node.state`), and checking the clock every 16–64 simulations instead of every one. These are throughput changes. They have no Elo theory of their own.

### Later, and not on the critical path

The WASM GUI should eventually move engine work in `gui_web/` onto a Web Worker so a search does not freeze the page. That is independent of playing strength, but the time-control switch in item 4 should land first; otherwise the worker will simply run the wrong algorithm at GUI clocks.

Connecting to [Claustrophobia](https://claustrophobia.dev/) and `path-clash-bot-arena` is the way to publish results against other bots. Strength claims there should wait until items 1–6 have landed, otherwise the published engine is still the slow quiescence-leaf hybrid.

### What not to do

Raising `leafDepth` to 1–4 at 50–200ms has already been measured: the tree becomes smaller than the branching factor, and pure alpha-beta won by 76 to 338 Elo. Keeping quiescence in the leaf in order to inflate alpha-beta nodes per second does not grow the PUCT tree. Threads and virtual loss, before widening and a cheap leaf exist, would only parallelise expand-all-130. Generation 5 data generation is finished and should not return to the roadmap. While the shipped search is MCTS with `leafDepth=0`, the main GA run should not be a sweep of alpha-beta-only thresholds.

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

**`src/search_tuning.hpp`** — Single configuration surface for every runtime knob of `search.hpp`. `tuning::SearchTuning` is an override struct in which an empty field (`UNSET_INT`/`UNSET_I64`/`UNSET_REAL`/`Tri::Unset`) means "keep the production value"; the sentinels are `INT_MIN`/`INT64_MIN`/`-1e300` rather than mcab's `-1`, because `-1` is a legal value here (production `contempt` is `-30`). `applySearchTuning(engine, t)` calls only the setters whose fields were filled, so an empty struct is a complete no-op. `parseSearchTuningArg()` is the shared CLI parser, taking a `""`/`e1-`/`e2-` prefix so the same knob exists in a global and a per-engine form, and `describeSearchTuning()` produces the banner line listing only what differs from production. Every setter goes through a `QR_TUNING_TRY_SETTER` SFINAE pair, so a git ref lacking a setter silently ignores that knob instead of failing to compile — which is why `arena.cpp` includes this header (like `mcab.hpp`) by relative path from HEAD, never through `-Isrc`.

**`src/mcab.hpp`** — MCαβ hybrid: a best-first PUCT tree whose leaves are evaluated by a *real shallow alpha-beta search* (`Negamax::searchLeaf`) rather than by a random rollout, with minimax-hard backup (Huang, "Pruning Game Tree by Rollouts", AAAI 2015; the same shape Scorpio uses). `MCABSearch<...>` is a template over the engine type; `McabRunner`/`chooseMoveAuto` pick between the hybrid and pure AB **at compile time** via the `hasMcabSupport` SFINAE trait. **On by default** in arena and self-play since 2026-08-13 (`params.enabled=true`); `enabled=false` delegates straight to `Negamax::chooseMove`, which is what `--no-mcab` selects. `mcab::McabParams` is the single source of truth for production values — the config blocks at the top of `arena.cpp`, `selfplay_main.cpp` and `run_arena.py` are pure override lists whose fields start empty (`mcab::UNSET_INT`/`UNSET_REAL`/`Tri::Unset`/`None`) and document the production value in each field's comment. Precedence: `--e1-`/`--e2-` flag > global flag > override block > `McabParams`. It lives in `src/`, but `arena.cpp` includes it by **relative path** (`../../src/mcab.hpp`) instead of through `-Isrc`: `run_arena.py` repoints `src/` per git ref, and the relative include guarantees the module always comes from HEAD even when the engine around it is an older ref that predates the feature.

**`tools/arena/` (`arena.cpp`, `run_arena.py`)** — Head-to-head engine strength tester with Elo estimation and confidence intervals. The hybrid is on by default; `--no-mcab`/`--e1-no-mcab`/`--e2-no-mcab` select pure alpha-beta, and `--mcab`/`--e1-mcab`/`--e2-mcab` force it on explicitly. In `run_arena.py` these are tri-state (`None` = don't pass a flag, let the binary decide). A ref older than the feature compiles fine, prints a one-shot warning, and plays pure AB. The config block at the top of `run_arena.py` exposes **every** engine knob (MCab set + `search_tuning.hpp` set) as an `E1_`/`E2_` pair, empty by default, with the production value in the comment; two tables (`MCAB_VALUE_KNOBS`, `MCAB_FLAG_KNOBS`) generate the argparse flags and the per-engine argument lists from those constants, so adding a knob is one line. `run_selfplay.py` mirrors this without the per-engine prefix.

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

### Hybrid MCTS + alpha-beta (`src/mcab.hpp`)

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
- **The hybrid needs a real node budget to be worth anything.** At 60ms/move with `leafDepth=4`, the whole budget buys 1–3 MCTS nodes; at `nodeBudget=500`/`leafDepth=3` a move costs ~4,600 AB nodes per MCTS node and ~1.4MB of tree. Any strength claim must come from `run_arena.py` at a time control where the tree actually grows. Production leaves at `leafDepth=0` now call `nnueEvalInt` instead of wall quiescence, and the MCTS path shares Negamax's BFS-distance cache. What is left — mean backup, a time-control switch, progressive widening, and a real GA run — is the roadmap above.

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

These are the knobs that are still waiting on a real GA/SPSA run for the engine as it ships today. Phase 8 already measured some of them; those should stay pinned so generations are not spent rediscovering the same defaults. The corresponding work item is roadmap step 6.

The first run should be `--mcab-tuning` with `leafDepth` held at 0. `cPuct` (1.5), `scoreScale` (200), and `fpuReduction` (0.0) survived a one-at-a-time sweep at 200ms but have never been optimised together. `nodeBudget` (20000) is not binding at 200ms — time is — so it should only be swept after the time-control curve in roadmap step 4. Dirichlet `alpha` and `epsilon` belong on a self-play diversity metric, not on arena Elo. Progressive-widening `k` and `α` do not exist yet; they become genes when roadmap step 5 lands.

Two results from the sweep should stay pinned until the time-control curve says otherwise: `leafDepth` (0) and `rootSelectMode` (MaxVisits). The tuner still exposes `mcabLeafDepth` as a `0..20` gene; if that range is left open, the population will rediscover that depth 2 dies. `leafDepthMax` (8) and the log₄ `adaptiveLeafDepth` schedule are untested, and extra leaf plies already lost at 200ms, so they should not enter the first run either.

What remains are the alpha-beta leftovers, which are secondary while production leaves skip alpha-beta, but still matter for `--no-mcab` and for quiescence if step 1 keeps it: `QS_CRITICAL_BFS_DELTA`, `QS_CRITICAL_ROBUSTNESS_DROP_TO`, `RFP_MARGIN_*`, `LMP_COUNT_*`, `POLICY_ORDER_SCALE` (400), `policyOrderingMinDepth` (3), and `catScoreScale`. `robustnessWeight` only affects `evalSimple`.

---

## Changelog

- **2026-08-15**: Arena NPS accounting now counts expanded MCTS tree nodes when the hybrid search is active. Previously it always read `SearchStats.nodes`, which stays zero for the production `leafDepth=0` NNUE fast path, causing progress and final reports to display `0 nps` despite the search running normally. Pure alpha-beta and older refs without MCAB keep their existing node counter.
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
- **2026-08-13**: **The hybrid MCTS + alpha-beta search is now the default** in `arena` and `selfplay` (`mcab::McabParams::enabled = true`), on the strength of +46.9 ±23.5 Elo at 200ms/move. Pure alpha-beta is selected with `--no-mcab` (per-engine `--e1-no-mcab`/`--e2-no-mcab` in arena) and is unchanged bit-for-bit — node parity re-verified at 8,293,935. The config blocks at the top of `arena.cpp`, `selfplay_main.cpp` and `run_arena.py` were rewritten as pure override lists: every field starts empty and falls back to `mcab::McabParams`, with the production value documented in each field's comment, so the defaults now live in exactly one place. `run_arena.py`'s mcab flags became tri-state. Caveats recorded in the roadmap: 200ms is the only time control measured, and the WASM/browser build still runs pure alpha-beta.
- **2026-08-13**: `leafDepth=0` confirmed over 1000 games: **−26.2 ±21.0 Elo**, hybrid ahead, margin no longer crossing zero. Parameter sweep around that working point (hybrid vs hybrid, one knob at a time) found only `fpuReduction=0.0` better than the plan's defaults (−24.4 ±22.9 over 800 games); `cPuct=1.5`, `scoreScale=200` and `rootSelectMode=MaxVisits` beat every alternative tried. Defaults moved to `leafDepth=0`/`fpuReduction=0.0` in `mcab.hpp`, `arena.cpp`, `selfplay_main.cpp`, and as GA `init` in `tune_spsa.cpp` (whose `mcabLeafDepth` range was `1..20`, which could never reach the working point — now `0..20`). `run_arena.py` gained `--mcab-cpuct`/`--mcab-fpu`/`--mcab-score-scale`/`--mcab-root-select` in global and `--e1-`/`--e2-` forms, so configs can be played against each other instead of each against pure AB. Pure AB unchanged: `bench_fixed_depth` and `bench_mcab` block A still agree at 8,293,935 nodes, and all 9 tests plus `nnue_verify` pass.
- **2026-08-13**: **Every engine setting is now editable from the Python orchestrators**, not only from the command line. `run_arena.py` and `run_selfplay.py` carry a documented config block where each knob is a variable that is empty (`None`) by default, meaning "use the production value", with that value written in the comment; arena's block is per engine (`E1_`/`E2_` pairs) so a candidate config plays directly against production. Precedence is `--e1-/--e2- flag > global flag > constant in the Python file > production`. Coverage: the full MCab set (nodes, leaf-depth, leaf-depth-max, adaptive-leaf-depth, cpuct, fpu, score-scale, root-select, tree-reuse, clear-tt-per-move, max-tree-depth, equiv-mode) plus, new in this change, every runtime knob of `search.hpp` — contempt, policy-order-scale, cat-score-scale, lmr-min-depth, lmr-min-move-index, lmr-divisor, cat-hot-cm, cat-cold-cm, wall-bfs-order-max-ply, qs-critical-bfs-delta, quiescence, lmr-pvs. These are the same parameters `tune_spsa.cpp` optimizes, and until now a proposed candidate could only be measured by editing `search.hpp` and recompiling.
- **2026-08-13**: New `src/search_tuning.hpp`: `tuning::SearchTuning` (empty-means-production override struct), `applySearchTuning()` (touches only filled fields, so an empty struct is a complete no-op), `parseSearchTuningArg()` (shared CLI parsing, with a `""`/`e1-`/`e2-` prefix) and `describeSearchTuning()` (banner line that prints only what differs from production). Every setter is applied through a SFINAE helper, so `arena.exe` still compiles against git refs that lack any of them — the same mechanism `mcab.hpp` uses, and the reason both headers are included by relative path (`../../src/...`) instead of `-Isrc`, which `run_arena.py` repoints per ref.
- **2026-08-13**: Fixed: `--no-policy-order` (self-play) and `--e1-no-policy-order`/`--e2-no-policy-order` (arena) had no effect. Both call sites only called `setPolicyOrderingEnabled(true)` inside an `if (enabled)`, which was correct while the search default was OFF; after the default flipped to ON in 2026-08, the "off" branch silently left policy ordering enabled. The setter is now always called with the resolved value.
- **2026-08-13**: `selfplay.exe` gained `--mcab-root-select`, `--mcab-clear-tt-per-move` and `--mcab-root-noise` (the explicit "on" counterpart of `--mcab-no-root-noise`), closing the gap with `arena.exe`'s knob set. Both banners now print `root-select` and `tree-reuse`/`clear-tt-per-move`, which were previously invisible in the logs. Added `mcab::rootSelectName()` as the inverse of `resolveRootSelect()`.
- **2026-08-13**: Fixed three counting bugs in `run_arena.py` that made every match report a score computed from a different set of games than the one announced. (1) `--no-invert` and `--invert-colors` both declared a `default=` for the same dest; argparse keeps the first action's default and ignores the rest, so the effective default was `False` and the `INVERT_COLORS = True` constant never applied — matches ran without color inversion. Default now comes from `parser.set_defaults(invert_colors=INVERT_COLORS)`. (2) In the non-inverted path a worker could get an odd `--games`, which `arena.exe` rounds up (`totalGames++`), so it played one game more than Python counted — a 100-game match actually played 112, with the extra games landing outside the denominator. Worker game counts are now always even. (3) The final scoreboard was built from `PROGRESS_JSON` deltas, and each worker's tail (`worker_games % worker_report` games) is only ever reported in its `RESULT_JSON`, which was read only when `--report-games 0` — those games silently vanished from the score. `RESULT_JSON` (cumulative per worker) is now the authoritative source for the final report; `PROGRESS_JSON` only drives the live display. Verified with two identical engines: 10 requested → 10 played, 5–5.
- **2026-08-14**: The hybrid MCTS path now passes Negamax's `PlayerPathCacheTable` (`xdistCache`) into `buildAccPairRoot` / `makeChildAccPair` instead of `nullptr`. Detection is SFINAE (`mcab::mcabPathCache` / `Negamax::pathCache()`), so older refs and the toy engines in `test_mcab_dispatch.cpp` still compile and still rebuild BFS. Search values are unchanged; only the repeated distance BFS on the tree path is avoided. Locked by `testMctsPathCacheWired` in `tests/test_mcab_core.cpp`.
- **2026-08-14**: Production `leafDepth=0` leaves evaluate with `nnueEvalInt` on the incremental accumulator and no longer enter `searchLeaf` / wall quiescence. `leafDepth>0` and equivalence mode are unchanged. Locked by `testFastLeafSkipsSearchLeaf`. Elo vs the old QS leaf is still unmeasured.
