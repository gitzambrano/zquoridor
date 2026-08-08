# Zquoridor

🎮 **[Play Zquoridor online in your browser!](https://gitzambrano.github.io/zquoridor/)** *(Web GUI with C++ engine compiled to WebAssembly)*

Zquoridor is a 2‑player Quoridor engine, played on a 9×9 board with 10 walls per player. The 4‑player variant is out of scope.

The engine searches moves using negamax with alpha‑beta pruning, evaluating positions via a neural network (NNUE with incremental accumulator) or a heuristic evaluation function (`evalSimple`).

The project is a sister engine to **Zchezz** (a chess engine), mirroring its conventions and architecture.

The strength goal is to beat the overwhelming majority of human players (~99%) and subsequently benchmark against the strongest publicly available Quoridor engines.

---

## 1. Directory Structure

```
zquoridor/
  src/                        # engine, production
    rules.hpp                 # board state, move generation,
                               #   Zobrist, path BFS, heuristic evaluation
    dsu.hpp                   # union‑find with rollback (wall legality checking)
    search.hpp                # negamax, alpha‑beta, transposition table,
                               #   move ordering, wall quiescence
    endgame_race.hpp           # exact empty‑hands endgame solver (plano‑additional.md
                               #   Priority 4) – reachable‑region gate + exact retrograde
                               #   DP cached by wall topology
    nnue.hpp                   # neural network: incremental accumulator, float32 and int8
    main.cpp                  # performance benchmark (self‑play + NNUE cost)
    selfplay.hpp               # training data generation via multi‑threaded self‑play
    selfplay_main.cpp         # self‑play command line interface

  teste/                      # correctness tests and benchmarks
    test_rules_sanity.cpp        # rules regression (pre‑filter + DSU)
    test_search_staging.cpp       # staged move generation vs. monolithic reference
    test_move_ordering.cpp         # wall ordering (path‑touch bonus)
    test_endgame_race.cpp            # empty‑hands endgame solver – correctness (gates vs.
                                        #   exact DP) and performance regression (cache)
    nnue_verify.cpp                 # C++ vs. Python numerical parity (float32 and int8)
    bench_quiescence_toggle.cpp      # nodes/s and total nodes with/without quiescence on fixed positions

  training/                   # NNUE PyTorch/Python training pipeline
    read_selfplay.py             # reading self‑play .bin files via numpy
    train_nnue.py                  # PyTorch training (AdamW, weight decay with annealing,
                                    #   LR schedule with warmup, early stopping)
    train_nnue_numpy.py             # same model, fallback without PyTorch
    quantize_nnue.py                 # post‑training int8/int16 quantization
    parity_check.py                   # Python side of parity check with nnue_verify

  data/
    nnue/
      nnue_weights.bin              # trained weights, float32
      nnue_weights_int8.bin          # same weights quantized – loaded by the engine

  gui_web/                    # human vs. engine GUI (HTML/JS + WASM)
    engine_wasm.cpp               # extern "C" shell around rules.hpp and search.hpp
    build_standalone.py             # packages everything into a single standalone quoridor.html
    index.html / app.js              # GUI, mobile‑first
    quoridor.js / quoridor.wasm       # compiled build output
    quoridor.html                      # standalone single‑file bundle

  build/                      # build scripts
    build_bench.bat / .sh         # main.cpp + bench_quiescence_toggle.cpp
    build_tests.bat / .sh          # the 5 test files in teste/
    build_selfplay.bat / .sh        # selfplay_main.cpp
    build_wasm.bat / .sh             # engine_wasm.cpp → quoridor.js/.wasm
    build_termux.sh                   # ARM/Android build via Termux
    build_all.bat / .sh                # runs bench + tests + selfplay; "wasm" argument includes WASM

  bin/                        # build outputs (git‑ignored)
```

All files in `teste/` include `src/` headers via `-Isrc` (passed by all build scripts) — no `.hpp` is duplicated. `gui_web/engine_wasm.cpp` includes `../src/rules.hpp` and `../src/search.hpp` directly.

---

## 2. How to Build and Run

### 2.1 Windows (`build/*.bat`, requires MinGW-w64 `g++` on PATH)

| Script | Outputs in`bin/` | Flags |
| ---------------------- | ------------------------------------------------------------------------------------------------------- | ------------------------------------------- |
| `build_bench.bat` | `bench.exe`, `bench_quiescence_toggle.exe` | `-O3 -march=native -mavx2 -mfma` |
| `build_tests.bat` | `test_rules_sanity.exe`, `test_search_staging.exe`, `test_move_ordering.exe`, `nnue_verify.exe` | `-O2` |
| `build_selfplay.bat` | `selfplay.exe` | `-O3 -march=native -mavx2 -mfma -pthread` |
| `build_wasm.bat` | `gui_web/quoridor.js` + `.wasm` | requires`emsdk_env.bat` activated first |
| `build_all.bat` | native targets above;`build_all.bat wasm` includes WASM | — |

### 2.2 Linux/macOS (`build/*.sh`)

Same targets and flags, without `.exe` extension:

```bash
chmod +x build/*.sh   # only once
build/build_bench.sh
build/build_tests.sh
build/build_selfplay.sh
build/build_wasm.sh      # requires emsdk activated in the shell
build/build_all.sh       # bench+tests+selfplay; "build_all.sh wasm" includes WASM
```

### 2.3 Termux (Android/ARM)

ARM does not have AVX2/FMA (those are x86 extensions) — the normal scripts fail
on those two flags. Use `build_termux.sh`: same optimization levels,
without `-mavx2 -mfma`; `-march=native` alone already enables NEON. Uses
`clang++` with fallback to `g++`.

```bash
pkg update && pkg install clang python
chmod +x build/build_termux.sh
build/build_termux.sh
```

WASM is not practical on Termux (due to emsdk requirements).

### 2.4 Web GUI (human vs. engine)

`gui_web/quoridor.js` + `quoridor.wasm` + `quoridor.html` are already
compiled in the repository — there is no need to run `build_wasm` before
testing; only recompile after modifying `rules.hpp`/`search.hpp`/`engine_wasm.cpp`.

```bash
cd gui_web && python3 -m http.server 8000
# open http://localhost:8000/index.html
# (quoridor.html also works standalone, without a server)
```

Features:

- Wall placement via selection mode (tap Horizontal/Vertical; legal slots
  light up on the board) or drag-and-drop.
- Remaining walls bar per player.
- Engine evaluation displayed in the move history (engine reads its own
  position after each move).
- New-game modal: choose side and engine strength (time per move) before
  starting.

To recompile after modifying the engine:

```bash
source /path/to/emsdk/emsdk_env.sh   # or emsdk_env.bat on Windows
build/build_wasm.sh                    # or build_wasm.bat
```

Known limitation: the engine move runs synchronously on the main thread and
blocks the tab for the configured time (200 ms to 4 s). Migrating to a
Web Worker is possible without changing `engine_wasm.cpp` — only the way
the JS loads the module — but is not yet implemented.

### 2.5 Self-play (training data generation)

```bash
build/build_selfplay.sh   # or .bat
bin/selfplay --games 2000 --out data/selfplay_001.bin
```

Plays games of the heuristic engine against itself and writes the result
directly in the binary format read by training (Section 2.6) — there is no
Python-side preprocessing step. The binary file IS the dataset: a single
`numpy.fromfile(path, dtype=SAMPLE_DTYPE)` (see `training/read_selfplay.py`)
yields a structured array ready to be converted to tensors.

| Flag                    | Default         | What it does                                                                                                                                                                   |
| ----------------------- | --------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `--games N`           | 1000            | number of games to play                                                                                                                                                        |
| `--out PATH`          | (required)      | output binary file path                                                                                                                                                        |
| `--depth N`           | 40              | maximum search depth                                                                                                                                                           |
| `--time-ms N`         | 100             | time budget per move in ms                                                                                                                                                     |
| `--threads N`         | available cores | games played in parallel                                                                                                                                                       |
| `--opening-plies N`   | 6               | how many opening plies of each game receive random noise                                                                                                                       |
| `--epsilon F`         | 0.25            | probability of playing a random move within the opening window above (prevents every game from starting identically)                                                           |
| `--epsilon-midgame F` | 0.02            | probability of random noise in the mid/endgame (choosing between the 2nd and 3rd best move of the shallow search, rather than fully random, preserving quality)               |
| `--chunk-games N`     | 2000            | splits the output into several binary files with at most N games each (automatic sharding)                                                                                    |
| `--max-plies N`       | 300             | safety cutoff per game; games exceeding this are discarded                                                                                                                     |
| `--seed N`            | 1               | random generator seed                                                                                                                                                          |

The self-play generator supports **automatic sharding** via `--chunk-games N`. This automatically generates several sequential `.bin` files (e.g., `selfplay_000.bin`, `selfplay_001.bin`, etc.), preventing single huge files from exhausting RAM during training. Training (Section 2.6) already accepts multiple files via `--data` and does not load everything into RAM at once.

### 2.6 NNUE Training (Python)

```bash
pip install torch numpy --break-system-packages   # or just numpy for the fallback
cd training
python3 train_nnue.py --data ../data/selfplay_*.bin \
    --out ../data/nnue/nnue_weights.bin --plot-dir ../data/plots
# without PyTorch:
python3 train_nnue_numpy.py --data ../data/selfplay_040k.bin --epochs 60 \
    --out ../data/nnue/nnue_weights.bin
# quantize already-trained weights without retraining:
python3 quantize_nnue.py ../data/nnue/nnue_weights.bin ../data/nnue/nnue_weights_int8.bin
```

`train_nnue.py` (PyTorch) and `train_nnue_numpy.py` (fallback without PyTorch)
accept virtually the same flags — the only differences are that the numpy
version has no `--device` or `--vram-budget-gb` (there is no GPU on that
path) and `--batch-size` is always a fixed integer, never `"auto"`.

**Data and checkpoint**

| Flag                 | Default    | What it does                                                                                                                              |
| -------------------- | ---------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| `--data PATH`      | (required) | self-play `.bin` file(s); the flag may be repeated, passed as a comma-separated list, a directory, or a glob                              |
| `--val-data PATH`  | none       | file(s) used only for validation (same rules as `--data`); if omitted, validation data is taken from a fraction of `--data`              |
| `--val-split F`    | 0.1        | fraction of `--data` held out for validation when `--val-data` is not passed                                                             |
| `--init-from PATH` | none       | existing `.bin` weights to continue a training run instead of starting from scratch                                                      |
| `--seed N`         | 0          | random generator seed                                                                                                                     |

**Optimization**

| Flag                                           | Default                              | What it does                                                                                         |
| ---------------------------------------------- | ------------------------------------ | ---------------------------------------------------------------------------------------------------- |
| `--epochs N`                                 | 60                                   | number of epochs                                                                                     |
| `--batch-size N\|auto`                        | `auto`                             | batch size; `auto` calculates from `--vram-budget-gb` (only in `train_nnue.py`)                   |
| `--lr F`                                     | 1e-3                                 | initial learning rate                                                                                |
| `--lr-min F`                                 | 1e-5                                 | learning rate floor for decaying schedules                                                           |
| `--lr-schedule none\|step\|exponential\|cosine` | `cosine`                           | how the learning rate changes over training                                                          |
| `--warmup-epochs N`                          | 2                                    | warm-up epochs at the start, before the normal schedule begins                                       |
| `--step-size N`                              | 10                                   | epochs per step, only used with `--lr-schedule=step`                                                |
| `--step-gamma F`                             | 0.5                                  | reduction factor per step, only with `--lr-schedule=step`                                           |
| `--exp-gamma F`                              | 0.97                                 | decay factor per epoch, only with `--lr-schedule=exponential`                                       |
| `--device cpu\|cuda`                          | `cuda` if available, else `cpu`    | only in `train_nnue.py`                                                                             |

**Weight decay with annealing** — stronger regularization at the start of
training, loosening toward the end; applied only to weight matrices, never to
biases (AdamW with decoupled weight decay):

| Flag                                          | Default    | What it does                                                        |
| --------------------------------------------- | ---------- | ------------------------------------------------------------------- |
| `--weight-decay F`                          | 1e-4       | initial weight decay value                                          |
| `--weight-decay-min F`                      | 0.0        | value weight decay converges to by the end of training              |
| `--wd-schedule none\|constant\|linear\|cosine` | `cosine` | how weight decay changes between the initial value and the minimum  |

**Early stopping** — stops training when the monitored metric stops
improving; by default restores the best epoch's weights (not the last
epoch's) in the final export:

| Flag                                                                   | Default      | What it does                                                                          |
| ---------------------------------------------------------------------- | ------------ | ------------------------------------------------------------------------------------- |
| `--early-stop` / `--no-early-stop`                                 | on           | enables/disables early stopping                                                       |
| `--patience N`                                                       | 8            | number of epochs without improvement before stopping                                  |
| `--min-delta F`                                                      | 1e-4         | minimum improvement to count as "improved"                                            |
| `--monitor val_loss\|val_outcome\|val_score\|val_policy\|val_policy_acc` | `val_loss` | which validation metric is monitored                                               |
| `--no-restore-best`                                                  | off          | exports the last epoch's weights instead of the best                                  |
| `--ckpt-dir PATH`                                                    | none         | directory to save `best.bin` (updated on every improvement) and `last.bin`           |

**Memory budget (RAM/VRAM)** — avoids running out of memory with large
datasets without having to calculate batch/chunk sizes manually:

| Flag                       | Default  | What it does                                                                                                              |
| -------------------------- | -------- | ------------------------------------------------------------------------------------------------------------------------- |
| `--vram-budget-gb F`     | 6.0      | VRAM budget used to calculate `--batch-size=auto`; only in `train_nnue.py`                                               |
| `--ram-budget-gb F`      | 32.0     | RAM budget used to calculate `--chunk-size=auto`                                                                         |
| `--ram-chunk-fraction F` | 0.25     | fraction of the RAM budget reserved for the shuffle buffer                                                                |
| `--chunk-size N\|auto`    | `auto` | number of samples kept in memory at a time when reading `.bin` files; `auto` calculates from `--ram-budget-gb`          |

**Loss weights and quantization (QAT)**

| Flag            | Default | What it does                                                                          |
| --------------- | ------- | ------------------------------------------------------------------------------------- |
| `--w-score F`   | 0.3     | weight of the auxiliary head (imitates `evalSimple`) in the total loss               |
| `--w-outcome F` | 1.0     | weight of the WL head (actual game result) in the total loss                         |
| `--w-policy F`  | 1.0     | weight of the policy head in the total loss                                           |
| `--qa N`        | 255     | quantization factor QA; must match `nnue.hpp` and `quantize_nnue.py`                |
| `--qb N`        | 64      | quantization factor QB; must match `nnue.hpp` and `quantize_nnue.py`                |

**Checkpointing, Interruption (Ctrl+C), and Quantization flow**

- **Updating the final `data/nnue/` folder**: The files `data/nnue/nnue_weights.bin` and `data/nnue/nnue_weights_int8.bin` are **only written when training completes fully** (either by reaching the `--epochs` limit or via *early stopping*). On completion the script automatically selects the **best epoch** weights (lowest `val_loss`), exports the float32 file, and runs automatic int8 quantization.
- **Interruption with Ctrl+C**: If training is interrupted mid-run (e.g., Ctrl+C at epoch 10), an emergency checkpoint is saved in `data/checkpoints/` (`train_state_*.pt`, `last_*.bin`, `train_config_*.json`). The `data/nnue/` folder is **not modified on Ctrl+C** to preserve the last validated model.
- **Resuming**: When running `python3 train_nnue.py` again, it automatically detects the highest-epoch `train_state_*` in `data/checkpoints/`, restores weights + AdamW state + RNGs, recalculates the learning-rate curve for that epoch, and restarts the interrupted epoch from scratch.
- **Using `quantize_nnue.py`**:
  - Without arguments (`python3 quantize_nnue.py`): quantizes the default file `data/nnue/nnue_weights.bin` -> `data/nnue/nnue_weights_int8.bin`.
  - With arguments (`python3 quantize_nnue.py <input.bin> <output_int8.bin>`): allows manually quantizing any specific float32 checkpoint (e.g., a `.bin` from the `data/checkpoints/` folder).

---

## 3. Implementation Status

This section describes what is already implemented in the engine, in order: first
how a position is represented and how legal moves are found, then how the
search decides which move to play, then the state of the neural network, then
the tests that ensure none of it broke.

### Rules and move generation (`rules.hpp`)

- **Position representation** (`State`): stores where both pawns are,
  which walls have been placed (as two bitboards, one for horizontal and
  one for vertical walls), whose turn it is, and a position hash (Zobrist)
  used to detect repeated positions quickly.
- **Move generation** (`legalMoves`, `pawnStepMoves`, `legalWallMoves`):
  enumerates all legal pawn moves and wall placements in the current
  position. Placing a wall is only legal if at least one path to the goal
  remains for both players — this check is the most expensive part of move
  generation, so before doing the full check (which requires a board-wide
  search) a cheap geometric pre-filter discards most obviously illegal
  walls, and the final check uses a union-find with rollback (`dsu.hpp`)
  that avoids repeating this work from scratch for each candidate wall.
- **Board path search** (`hasPathToGoal`, `shortestPathLen`,
  `shortestPathTouchSlots`, `pathRobustness`): four variants of the same
  breadth-first search (BFS) over board cells, each answering a different
  question — does any path to the goal row exist? what is the shortest
  path? which walls, if placed, would cut that shortest path? and how
  fragile is that path, i.e., how many low-cost detours exist if a new
  wall blocks it exactly? This last question (path robustness) is used
  both in position evaluation and to decide when the search needs to
  "look deeper" before trusting the result (quiescence, see below).

### Search (`search.hpp`)

- **Negamax with alpha-beta pruning**: the search algorithm itself — explores
  sequences of future moves, evaluates positions at the end of each
  sequence with `evalSimple`, and discards branches that are proven not to
  be chosen, without needing to finish exploring them.
- **Transposition table**: a cache of already-analyzed positions (by
  Zobrist hash), storing depth, result, and the best move found, to avoid
  re-analyzing from scratch a position the search has already visited via a
  different sequence of moves.
- **Move ordering** (killer moves, history heuristic, `orderWallMoves`):
  the order in which candidate moves are tested greatly affects pruning
  efficiency — testing the most promising moves first cuts more branches,
  sooner. The engine prioritizes moves that previously caused a cutoff in
  similar positions (killer moves), moves that have historically performed
  well (history heuristic), and for walls specifically, how much each wall
  disrupts the opponent's shortest path: near the root, an exact BFS delta
  (expensive — one BFS per candidate); at any depth, the **Corridor
  Attention Table** (`cat.hpp`, plano-additional.md, Priority 1) — a
  per-cell "heat" computed once per node (2 BFS, not per wall candidate)
  measuring how much each cell deviates from the opponent's optimal path.
  It replaced a simpler binary bonus ("does this wall touch the witness
  path or not") that only saw one route — the continuous heat also credits
  walls that close low-cost detours outside that specific route. Ad-hoc
  benchmark (`bench_wall_touch_bonus.cpp`, 40 fixed positions, 200 ms/move):
  ~11.7× fewer nodes to equivalent depth compared to ordering without this
  signal.
- **Per-node and cross-node BFS cache** (plano-additional.md, Priorities 6
  and 6b): the same distance BFS (`shortestPathLen`/`pathRobustness`/
  `shortestPathTouchSlots`, unified into a single engine in `rules.hpp`)
  was being recomputed several times per node (ordering, quiescence,
  evaluation) even for the same (wall topology, pawn, player) tuple.
  A `PlayerPathCache` computed once per node eliminates local duplication;
  a `PlayerPathCacheTable` (~48 MB, keyed by
  `wallsH/wallsV/pawnCell/player`, not the full position) also eliminates
  duplication across sibling nodes/transpositions that share the same wall
  topology. Measured speedup (`bench_fixed_depth.cpp`, same node count
  before/after — speed only, identical search): ~57% (per-node cache) +
  ~5% additional (cross-node cache).
- **LMR + PVS** (Late Move Reduction + Principal Variation Search,
  plano-additional.md Priorities 3 and 8): late moves in the ordering are
  searched at reduced depth and null window first, re-verifying at full
  depth/window only if the reduced result suggests it may be worthwhile —
  reduces nodes-to-same-depth without sacrificing finding the correct line
  when it exists. Never reduces the TT move, killers, or a "hot" wall in
  the CAT heat. Toggle: `Negamax::setLmrPvsEnabled(false)`.
- **RFP + LMP** (Reverse Futility Pruning + Late Move Pruning,
  plano-additional.md Priorities 3b and 3c): at shallow depth, RFP cuts
  the node without generating any move when the static eval is already far
  above beta; LMP permanently discards the tail of quiet moves after
  already having tried several without success. Never applied at the root,
  nor to the TT/killer/hot-wall move. Toggles:
  `Negamax::setRfpEnabled(false)`/`setLmpEnabled(false)`.
  **Documented pitfall:** LMR combined with LMP without an extra guard
  (`reducedByLmr` in `negamax`) formed a combination catastrophically worse
  than either alone (0-10 in isolated head-to-head games) — the LMR
  reduced-verification search was contaminated by LMP's aggressive pruning,
  never triggering the safety re-search that LMR depends on. Fixed; see
  Priority 3c of the plan for the full analysis. Validated in
  `test_lmr_pvs.cpp` (never an illegal move, score agreement ≥85% overall
  and ≥90% on decisive positions against a full-window reference with none
  of the four heuristics) and `bench_lmr_pvs.cpp` (nodes-to-same-depth
  ~0.19–0.22×; head-to-head 6-3-1 in favor of enabled, small sample —
  not SPRT, see Priority 14 of the plan).
- **Wall quiescence**: near the end of a search, if the best move found is
  a wall that significantly worsens the opponent's path, the engine extends
  the search by a few more plies before accepting that result, rather than
  stopping there and potentially erring by not having looked at the
  immediate consequence of the move (the classic "horizon effect" in game
  engines). The extension is bounded (`QS_MAX_EXTRA_PLIES`) and the trigger
  for considering a wall "critical" still uses initial, untuned values
  (`QS_CRITICAL_BFS_DELTA`, `QS_CRITICAL_ROBUSTNESS_DROP_TO`). Can be
  disabled at runtime with `Negamax::setQuiescenceEnabled(false)`, without
  recompiling — useful for comparing the engine with and without this
  extension in a benchmark, or for ruling it out as the cause of a bug
  during debugging.
- **Draw detection & Contempt**: 3-fold repetition draw detection
  (three occurrences of the same board position) implemented with a
  position history via `RepetitionTable`. A `CONTEMPT = -30` factor is
  added in the search (`negamax` and quiescence) to make the engine
  actively avoid draws (penalty of -30 for the player proposing the
  repetition) in neutral or favorable positions, while still allowing a
  draw as a defensive resource in very unfavorable ones.
- **Exact empty-hands endgame solver** (`endgame_race.hpp`,
  plano-additional.md Priority 4): when both players run out of walls,
  the wall topology is frozen forever and the game becomes a pure pawn
  race — the hook in `negamax` detects this condition
  (`wallsLeft[0]==0 && wallsLeft[1]==0`) and resolves it with mathematical
  certainty instead of continuing the heuristic search. Two layers: a
  cheap gate (`raceDisjointGate`) that decides without search when the
  **entire reachable regions** of both players (not just each player's
  shortest path — see note below) are disjoint, and an exact retrograde DP
  (`raceExactDTM`) over the 81×81×2=13,122 states `(pos0, pos1, turn)`
  for that fixed topology, which also detects draws by infinite pursuit.
  The DP is cached by wall topology (only recomputed when `wallsH`/`wallsV`
  change from the last call), and the cost of the exact solver is bounded
  by a **real-time budget** per move (`g_raceExactBudgetUs`, ~3% of the
  total `chooseMove` budget, measured via `chrono` on each expensive call)
  — in the worst case (positions requiring many different topologies in
  sequence, where the 1-slot cache does not help) the overflow falls back
  to the regular heuristic instead of continuing to pay the expensive
  rebuild cost, so nodes/s never becomes worse than the baseline without
  the feature.
  Several rounds of corrections happened after the first integration —
  anyone modifying this code should read the large comment at the top of
  `endgame_race.hpp` and Sections 4d/4e of `plano-additional.md` first:
  (1) a cheaper ETA gate (Level 1 of the plan) was tested and **discarded**
  from the decision pipeline because it decided wrongly on a real
  counterexample (physical blocking can cost more time than predicted);
  (2) the first version of the region gate tested only **shortest-path**
  disjunction, which does not guarantee the absence of interaction —
  corrected to test full reachable-region disjunction; (3) the 1-slot
  per-topology cache alone was insufficient (real hit rate measured at
  ~0.5% during the decision of where to place the last walls — hence the
  time budget above); (4) **most critically**: a MOVE-CHOICE bug (not a
  value bug) in `chooseMove` — when the real game ROOT itself (not an
  internal node) is already at `wallsLeft==(0,0)`, the shortcut returns
  only the value (correct leaf-node behavior) and `chooseMove` was reading
  a placeholder from the TT as the "best move" instead of the move that
  actually achieves the optimal DTM — the engine knew who won but played
  arbitrary moves to get there, losing most games despite healthy nodes/s.
  Fixed by comparing root candidates via 1-ply exact value before the
  iterative-deepening loop. Validated in a real arena (`teste/arena.cpp`,
  two real git refs in the same binary): from ~113/502 wins (Elo≈−132)
  before fix (4) to ~46.5% score in 100 games after — within the
  statistical noise of that sample.

### NNUE (`nnue.hpp`)

The neural network is **100% wired and integrated into search** (`src/search.hpp` and `src/nnue.hpp`).
Search maintains an incremental accumulator stack (`nnueAccStack`, an `AccPair` per ply) across the search tree.
Runtime switching between heuristic evaluation and quantized neural network evaluation is enabled via `Negamax::setEvalMode(EvalMode::Heuristic | EvalMode::NNUE)`.
This feature is exposed across all binaries (Self-play, Arena, Benchmarks, and WebAssembly shell).
Architecture details for 354 features and 3 heads are in Section 4.

### Tests

- `test_rules_sanity.cpp`: rules regression and the geometric pre-filter +
  union-find used to check wall legality.
- `test_search_staging.cpp`: compares staged move generation (the one used
  in production) against a simpler, more direct reference implementation,
  to ensure the optimized version did not change the result.
- `test_move_ordering.cpp`: validates the Corridor Attention Table
  (`cat.hpp`) in isolation (per-cell heat shape) and its use in
  `orderWallMoves` (favors the correct moves and does not change their
  legality).
- `test_endgame_race.cpp`: regression for the empty-hands endgame solver
  (`endgame_race.hpp`) — confirms that whenever the cheap gate decides,
  the result matches exactly the exact retrograde DP (fixed positions and
  random topologies), covers the infinite-pursuit draw case, includes a
  dedicated performance test (many consecutive calls with the same wall
  topology must reuse the cache), and tests `Negamax::chooseMove`
  end-to-end from a root already at `wallsLeft==(0,0)` — catches the
  move-choice bug (not a value bug) described in the "Search" section
  above (fails against the version without the fix, passes with it).
- `test_lmr_pvs.cpp`: validates LMR+PVS+RFP+LMP (plano-additional.md,
  Priorities 3/3b/3c/8) against a full-window reference with none of the
  four — does not require "0 divergences" (they are heuristics by design,
  unlike the staging test above), but requires never returning an illegal
  move, score agreement ≥85% overall and ≥90% on decisive positions
  (|score| high — that is where a mis-calibrated reduction/pruning would
  miss a fine tactical line), and reports nodes-to-same-depth as evidence
  that the change actually helps.
- `nnue_verify.cpp`: confirms that the C++ network implementation produces
  the same numbers as the Python implementation, both in float32 and the
  int8-quantized version.
- `bench_quiescence_toggle.cpp`: measures nodes per second and total node
  count with quiescence on and off, over a fixed set of positions at a
  fixed depth (not a time budget, which varies too much between runs and
  masks the measured effect).
- `bench_lmr_pvs.cpp`: same spirit as the above, but for LMR+PVS+RFP+LMP
  — nodes/s and average depth on fixed positions (fixed time budget), and
  head-to-head engine-vs-engine games (heuristics on vs. off, alternating
  colors) for a more direct answer to "does it play better".

---

## 4. NNUE Architecture

`354 → 256 (accumulator, SCReLU activation)` followed by 3 independent heads:

| Head         | Shape               | Target                           | Role                                                                                                                    |
| ------------ | ------------------- | -------------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| WL (Outcome) | `256→32→1`      | actual game result (+1/-1), BCE  | consumed by search via`forwardValueWLQuant`; logit scaled by `NNUE_EVAL_SCALE=200` for `evalSimple` compatibility |
| Auxiliary    | `256→32→1`      | `evalSimple` at move time, MSE | training scaffold while self-play comes from heuristic; to be removed once self-play comes from net itself              |
| Policy       | `256→209` logits | move played, CrossEntropy        | move ordering in search; not victory probability                                                                        |

### Input Features (354)

| Group                    | Indices        | Count | Encoding                                  |
| ------------------------ | -------------- | ----: | ----------------------------------------- |
| Own Pawn                 | `[0, 81)`    |    81 | one-hot per cell                          |
| Opponent Pawn            | `[81, 162)`  |    81 | one-hot per cell                          |
| Horizontal Wall          | `[162, 226)` |    64 | bit per 8×8 slot                         |
| Vertical Wall            | `[226, 290)` |    64 | bit per 8×8 slot                         |
| Own BFS Distance         | `[290, 311)` |    21 | one-hot bucket (0–19 exact, 20 = "≥20") |
| Opponent BFS Distance    | `[311, 332)` |    21 | one-hot bucket                            |
| Own Remaining Walls      | `[332, 343)` |    11 | one-hot bucket (0–10, no saturation)     |
| Opponent Remaining Walls | `[343, 354)` |    11 | one-hot bucket                            |

Remaining walls buckets (features `[332,354)`) were added in 2026-08 — `nnue_sign_check.cpp` confirmed that two positions with identical pawns/walls but different wall counts had identical `eval` prior to this addition because the network couldn't see this information. Weights trained with `NUM_FEATURES=332` are **incompatible** with the current architecture; `loadFromFile` checks file size before loading and rejects mismatched formats with an explicit error message.

### Canonical Perspective

The network has no fixed concept of white/black: all inputs are relative to perspective (`buildAccumulatorQuant(state, perspective)`), always evaluated from the side to move — negamax handles sign flip between plies.

Perspective is canonicalized by row reflection (`r → WS-1-r` for walls, `r → N-1-r` for pawns) when `perspective==1`. Symmetry bug fix (2026-08): previously `featOwnPawn`/`featWallH`/`featWallV` used raw board coordinates and only swapped which pawn was "mine" without row reflection — initial symmetric position produced different evals for both perspectives. Columns are never reflected (board is symmetric on that axis).

### Quantization (QAT)

Quantization-aware training (QAT), not post-hoc — `QA=255`/`QB=64` are fixed constants set before training (`nnue.hpp`, `train_nnue.py`, and `quantize_nnue.py` must match). A `WeightClipper` in the style of Stockfish's `nnue-pytorch` clamps weights to int8/int16 ranges during every optimizer step.

Quantized file layout: header `[QA:int32][QB:int32]`, followed by weight blocks. `NNUEWeightsQuant` pre-allocates zeroed vectors in its constructor for safe fallback if loading fails.

### Incremental Update

The accumulator (`AccumulatorQuant`, accumulated in `int32`) is updated incrementally:

- **Pawn move**: 1 feature removal + 1 feature addition, plus conditional update of own distance bucket (0 or 1 BFS, cached in `ownDistBucket`/`oppDistBucket`).
- **Wall move**: 1 wall feature addition, conditional update of both distance buckets (max 2 BFS), and conditional update of remaining walls bucket for the moving player.

Search maintains an `AccPair` stack (one accumulator per perspective per ply).

---
