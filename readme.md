# Zquoridor

A Quoridor engine for the 9×9, 2-player variant (10 walls per side).

Negamax + alpha-beta search with a trained neural network (NNUE) for position evaluation and move ordering. Includes a self-play data generator, an NNUE training pipeline, a strength-testing arena, a test/benchmark suite, and a web GUI.

🎮 [Play online](https://gitzambrano.github.io/zquoridor/) — *WebAssembly build.*

---

## Features

- **Search**: negamax with alpha-beta pruning and a transposition table.
- **Move ordering**: killer moves + history heuristic, plus an NNUE policy head mixed with Corridor Attention Table (CAT, a wall-move heat map) as an extra ordering signal.
- **Pruning/reduction**: late move reductions with principal variation search (LMR+PVS), reverse futility pruning, and late move pruning (RFP+LMP) at shallow depth. All toggleable at runtime, no recompile needed.
- **Wall quiescence**: extends the search a few plies past nominal depth when the best move found is a critical-looking wall.
- **Exact endgame solver**: once both players run out of walls, the position is a frozen-board pawn race — solved exactly instead of searched.
- **Draw handling**: threefold repetition detection with a contempt factor against neutral/favorable draws.
- **NNUE evaluation**: quantized network trained on the engine's own self-play games; policy head can also assist move ordering (see above).
- **Self-play + training pipeline**: multi-threaded self-play generator, PyTorch (or numpy-only fallback) training, automatic int8 quantization.
- **Arena**: plays two builds head-to-head and reports an Elo estimate.
- **Web GUI**: WebAssembly build, playable in a browser.

---

## Build

No build system — every binary is a single `g++` invocation. Requires a C++17 compiler; training scripts need Python 3 (PyTorch optional, numpy fallback available).

```bash
chmod +x build/*.sh   # once, Linux/macOS
```

Windows: equivalent `build/*.bat` scripts (MinGW-w64 `g++` on `PATH`). ARM/Termux: `build/build_termux.sh` (drops the x86-only `-mavx2 -mfma` flags).

| Script                            | Builds                                                       |
| --------------------------------- | ------------------------------------------------------------ |
| `build_tests.sh` / `.bat`     | correctness test suite                                       |
| `build_bench.sh` / `.bat`     | performance benchmarks                                       |
| `build_selfplay.sh` / `.bat`  | self-play data generator                                     |
| `build_arena.sh` / `.bat`     | strength-testing arena                                       |
| `build_tune_spsa.sh` / `.bat` | SPSA parameter tuner (contempt / policy-order scale / CAT scale, plus a discrete policyOrderingMinDepth sweep) |
| `build_wasm.sh` / `.bat`      | WebAssembly build (needs`emsdk` activated)                 |
| `build_all.sh` / `.bat`       | all of the above except WASM (add`wasm` arg to include it) |

Binaries go to `bin/` (arena to `teste/bin/`), both git-ignored.

## Run

```bash
# tests — each binary is self-contained, no arguments
bin/test_rules_sanity
bin/test_search_staging
bin/test_move_ordering
bin/test_endgame_race
bin/test_lmr_pvs
bin/nnue_verify data/nnue/nnue_weights.bin data/nnue/nnue_weights_int8.bin

# self-play data generation
bin/selfplay --games 2000 --chunk-games 2000 --threads 8 --time-ms 100 \
    --out "data/selfplay/selfplay_{shard:03d}.bin"

# NNUE training (from training/)
python3 train_nnue.py --data ../data/selfplay/*.bin --out ../data/nnue/nnue_weights.bin --plot-dir ../data/nnue/plots
python3 quantize_nnue.py ../data/nnue/nnue_weights.bin ../data/nnue/nnue_weights_int8.bin

# strength test: two builds head-to-head (by git ref or working tree)
python3 teste/run_arena.py --ref1 "" --ref2 main --games 200 --time 500 --threads 14

# web GUI
cd gui_web && python3 -m http.server 8000   # open http://localhost:8000/index.html
```

### Self-play flags

| Flag                                   | Default                             | Meaning                                                           |
| -------------------------------------- | ----------------------------------- | ----------------------------------------------------------------- |
| `--games N`                          | 1000                                | games to play                                                     |
| `--out PATH`                         | required                            | output file/template (`{shard:03d}` shards the output)          |
| `--depth N`                          | 40                                  | max search depth                                                  |
| `--time-ms N`                        | 100                                 | time budget per move                                              |
| `--threads N`                        | all cores                           | games in parallel                                                 |
| `--opening-plies N`, `--epsilon F` | 6, 0.25                             | random-move window at game start                                  |
| `--epsilon-midgame F`                | 0.02                                | chance of playing 2nd/3rd-best move mid/endgame, for data variety |
| `--chunk-games N`                    | 2000                                | games per output shard                                            |
| `--nnue-weights PATH`                | `data/nnue/nnue_weights_int8.bin` | weights file                                                      |
| `--policy-order`                     | **on**                              | use the NNUE policy head for move ordering (default since 2026-08) |
| `--no-policy-order`                  | —                                    | disable it                                                         |
| `--policy-order-min-depth N`         | 3                                   | min remaining depth at which policy-assisted ordering applies     |

Full list: `bin/selfplay --help`. `training/run_selfplay.py` wraps the same binary (config block at the top of the script, or CLI flags). The policy-assisted ordering default above (on) is shared by `selfplay`, `arena`, `tune_spsa`, and the WASM build — it's a class-level default in `search.hpp`, so any new tool built on `Negamax` gets it too unless it opts out.

---

## Project layout

| Path                                            | Contents                                                            |
| ----------------------------------------------- | ------------------------------------------------------------------- |
| `src/rules.hpp`                               | board state, move generation, path search                           |
| `src/dsu.hpp`                                 | union-find with rollback (wall legality)                            |
| `src/cat.hpp`                                 | Corridor Attention Table (wall-move ordering heat map)              |
| `src/search.hpp`                              | negamax, alpha-beta, transposition table, move ordering, quiescence |
| `src/endgame_race.hpp`                        | exact empty-hands endgame solver                                    |
| `src/nnue.hpp`                                | neural network: accumulator, forward passes                         |
| `src/main.cpp`                                | performance benchmark entry point                                   |
| `src/selfplay.hpp`, `src/selfplay_main.cpp` | multi-threaded self-play generator                                  |
| `teste/`                                      | correctness tests, benchmarks, strength arena                       |
| `training/`                                   | NNUE training pipeline (Python)                                     |
| `data/`                                       | trained weights, self-play shards, arena output                     |
| `gui_web/`                                    | browser GUI (HTML/JS + WebAssembly)                                 |
| `build/`                                      | build scripts                                                       |
| `bin/`                                        | build output (git-ignored)                                          |

All test files and the WebAssembly shell include the engine headers directly (`-Isrc`) — no header is duplicated.

## Tests & benchmarks

| File                                               | Checks                                                                           |
| -------------------------------------------------- | -------------------------------------------------------------------------------- |
| `test_rules_sanity.cpp`                          | move generation, wall-legality pre-filter + DSU, perft                           |
| `test_search_staging.cpp`                        | staged move generation vs. a monolithic reference                                |
| `test_move_ordering.cpp`                         | Corridor Attention Table shape and use in wall ordering                          |
| `test_endgame_race.cpp`                          | endgame solver vs. exact DP, cache reuse, end-to-end from a frozen-topology root |
| `test_lmr_pvs.cpp`                               | LMR+PVS+RFP+LMP vs. a full-window reference                                      |
| `nnue_verify.cpp` + `training/parity_check.py` | C++ vs. Python numerical parity (float32 and int8)                               |
| `nnue_incremental_check.cpp`                     | incremental accumulator vs. rebuild-from-scratch                                 |
| `nnue_sign_check.cpp`                            | NNUE eval sign/perspective sanity                                                |
| `bench_quiescence_toggle.cpp`                    | nodes/s and node count with/without quiescence                                   |
| `bench_lmr_pvs.cpp`                              | nodes-to-depth and head-to-head games with/without LMR+PVS+RFP+LMP               |
| `bench_wall_touch_bonus.cpp`                     | nodes-to-depth with/without CAT-based wall ordering                              |
| `bench_race_regression.cpp`                      | endgame-race solver performance regression                                       |
| `tune_spsa.cpp`                                  | SPSA parameter tuner (`--mode spsa\|sweep-mindepth\|hybrid`) + `plot_spsa.py` for convergence plots |

## Arena (strength testing)

`teste/arena.cpp`, usually run via `teste/run_arena.py`, plays two engine builds head-to-head and reports an Elo estimate with a confidence interval. Each side is configured independently — weight file, policy-assisted ordering on/off — useful for comparing engine versions or configurations.

## SPSA tuning

`teste/tune_spsa.cpp`, usually run via `teste/run_spsa.py`, tunes `contempt`/`policyOrderScale`/`catScoreScale` against each other via self-play. Three modes:

- `--mode spsa` (default): classic SPSA (Spall 1998) over the continuous parameters above. Each iteration averages `--games-per-iter` antithetic match-ups (default 4) to keep the gradient estimate from drowning in noise; `--threads N` parallelizes those games.
- `--mode sweep-mindepth`: round-robin tournament over discrete `policyOrderingMinDepth` candidates (continuous params held fixed) — SPSA doesn't suit a small-range integer well.
- `--mode hybrid`: one OS thread per `policyOrderingMinDepth` candidate, each running a full independent continuous SPSA with that depth fixed — tunes the continuous parameters *for every candidate depth* in parallel instead of tuning once and sweeping depth afterward.

Every run logs a per-iteration CSV (`--history`, default `spsa_history.csv`); plot it with `python3 teste/plot_spsa.py` (`--glob` to overlay all `_depth{D}` files from a hybrid run).

## Web GUI

Mobile-friendly board: wall placement by tap or drag, remaining-walls indicator, live eval display, new-game side/strength picker. `quoridor.js`/`.wasm` are pre-built and committed; rebuild (`build_wasm.sh`/`.bat`) only after changing `rules.hpp`/`search.hpp`/`engine_wasm.cpp`.

---

## NNUE architecture

`354 → 256` accumulator (SCReLU activation) → three independent heads:

| Head         | Shape          | Predicts                                          |
| ------------ | -------------- | ------------------------------------------------- |
| Outcome (WL) | `256→32→1` | game result; used as the search's leaf evaluation |
| Auxiliary    | `256→32→1` | training aid                                      |
| Policy       | `256→209`   | move played; used for move ordering               |

### Input features (354)

| Group                                 |    Size | Encoding              |
| ------------------------------------- | ------: | --------------------- |
| Own / opponent pawn position          | 81 + 81 | one-hot cell          |
| Horizontal / vertical walls           | 64 + 64 | bit per slot          |
| Own / opponent shortest-path distance | 21 + 21 | bucketed BFS distance |
| Own / opponent remaining walls        | 11 + 11 | bucketed count        |

### Notes

- **Perspective**: every position is encoded from the side to move's point of view, with the board mirrored as needed — the network has no fixed white/black concept.
- **Quantization**: trained quantization-aware (QAT) — the same fixed-point rounding used at inference is simulated during training. Weight files store `[QA:int32][QB:int32]` followed by int8 weight blocks; files trained under a different feature set are rejected on load.
- **Incremental accumulator**: updated per move (1 feature swap for a pawn move, a few for a wall move) rather than rebuilt from scratch.
