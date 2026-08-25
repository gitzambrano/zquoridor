# Zquoridor

A high-performance Quoridor engine for the 9×9, 2-player variant (10 walls per side).

**Hybrid MCTS with alpha-beta**: a best-first PUCT tree guided by a Quantization-Aware Trained neural network (NNUE), whose leaves are resolved by a real alpha-beta search rather than a random rollout. The same NNUE provides position evaluation and policy-assisted move ordering. Pure alpha-beta remains fully supported and is one flag away (`--no-mcab`). Includes a multi-threaded self-play generator, NNUE training pipeline, strength-testing arena, test/benchmark suite, and WebAssembly browser GUI.

🎮 [Play online](https://gitzambrano.github.io/zquoridor/) — *WebAssembly build.*

---

## Features

- **Alpha-Beta Search**: Negamax search with transposition table, iterative deepening, and Principal Variation Search (PVS).
- **Policy-Assisted & CAT Move Ordering**: Killer moves, history heuristic, Corridor Attention Table (CAT wall heat map), and NNUE policy head logits (gated by depth floor `policyOrderingMinDepth`).
- **Pruning & Reductions**: Late Move Reductions (LMR), Reverse Futility Pruning (RFP), and Late Move Pruning (LMP) at shallow depths. All heuristics include runtime toggles.
- **Wall Quiescence Search**: Extends search past nominal depth for critical-looking wall placements.
- **Hybrid MCTS + Alpha-Beta (default search)**: Best-first PUCT tree whose leaves are evaluated by a real alpha-beta search instead of a random rollout, with minimax-hard backup (`src/mcab.hpp`). **On by default** in arena and self-play since 2026-08-13, worth **+46.9 ±23.5 Elo** over pure alpha-beta at 200 ms/move. Pure alpha-beta is preserved bit-for-bit behind `--no-mcab` and loses nothing when the hybrid is enabled. Requires NNUE (the PUCT priors come from the policy head).
- **Exact Endgame Solver**: Exact retrograde DP pawn-race solver over 81×81×2 states when both players run out of walls (`wallsLeft==(0,0)`), with a real-time budget.
- **NNUE Evaluation & Policy**: 354-feature network (pawn cells, wall bitboards, bucketed BFS distances, and remaining walls) with SCReLU activation, outputting win probability and move ordering logits.
- **Fast Monte Carlo Self-Play Generator**: Multi-threaded C++ self-play generator with standard epsilon-greedy and AlphaZero-style Monte Carlo policy-temperature sampling (`--mc-mode`) for rapid opening generation. Stack-allocated to ensure zero heap corruption.
- **Training & Quantization Pipeline**: PyTorch training script with dataset blending (`train_nnue.py`), automated int8 quantization (`quantize_nnue.py`), and C++/Python numerical parity verification (`nnue_verify`).
- **Strength Arena**: Automated head-to-head match runner (`tools/arena/run_arena.py`) with Elo estimation and confidence intervals.
- **SPSA Parameter Tuning**: Multi-mode SPSA tuner (`tools/spsa/tune_spsa.cpp`) supporting continuous parameter tuning (`--mode spsa`), discrete depth sweeps (`--mode sweep-mindepth`), and multi-depth parallel tuning (`--mode hybrid`) with matplotlib visualization (`tools/spsa/plot_spsa.py`).
- **WebAssembly & Browser GUI**: Mobile and desktop browser UI on a canvas board. Wall placement needs no mode button: the board reads the pointer position, shows a preview in the groove under it, and a drag across the other axis flips the orientation. Magnetic snapping to the nearest legal slot, animated pawn moves, and an engine that searches in a Web Worker so the interface stays responsive. A player strip above and below the board carries the name, clock, remaining walls and distance to goal. Nine board themes, 6 pawn styles, sound packs and presets. Includes an analysis tab running in a Web Worker: engine lines with evaluations, an evaluation graph, position navigation with takeback, hints and an automatic blunder check; a position editor with validity feedback; QFEN positions and QGN game text for import/export (clipboard, files, drag-and-drop and share links); and PNG/SVG board image export.

---

## Build

Single `g++` compilation per binary (C++17). Training scripts require Python 3 (PyTorch optional, numpy fallback available).

```bash
chmod +x build/*.sh   # once, Linux/macOS
```

Windows: equivalent `build/*.bat` scripts (MinGW-w64 `g++` on `PATH`). ARM/Termux: `build/build_termux.sh`.

| Script                            | Target                                                       |
| --------------------------------- | ------------------------------------------------------------ |
| `build_tests.sh` / `.bat`     | Correctness test suite                                       |
| `build_bench.sh` / `.bat`     | Performance benchmarks                                       |
| `build_selfplay.sh` / `.bat`  | Self-play data generator                                     |
| `build_arena.sh` / `.bat`     | Strength-testing arena                                       |
| `build_tune_spsa.sh` / `.bat` | SPSA parameter tuner (`spsa`, `sweep-mindepth`, `hybrid`)    |
| `build_wasm.sh` / `.bat`      | WebAssembly build (requires `emsdk`)                         |
| `build_all.sh` / `.bat`       | All targets except WASM (add `wasm` argument to include it)  |

Binaries are placed in `bin/`, git-ignored.

---

## Run

### Tests & Benchmarks
```bash
bin/test_rules_sanity
bin/test_search_staging
bin/test_move_ordering
bin/test_endgame_race
bin/test_lmr_pvs
bin/nnue_verify data/nnue/nnue_weights.bin data/nnue/nnue_weights_int8.bin
```

### Self-Play Data Generation
```bash
# Epsilon-greedy mode (legacy)
bin/selfplay --games 20000 --chunk-games 2000 --threads 12 --time-ms 100 \
    --out "data/selfplay/gen5-epsilon/selfplay_{shard:03d}.bin"

# Monte Carlo policy-temperature mode (fast openings)
bin/selfplay --games 20000 --chunk-games 2000 --threads 12 --time-ms 100 --mc-mode \
    --out "data/selfplay/gen5-montecarlo/selfplay_{shard:03d}.bin"

# Or via Python orchestrator
python3 tools/selfplay/run_selfplay.py
```

### NNUE Training & Quantization
```bash
cd training
python3 train_nnue.py --data ../data/selfplay/*.bin --out ../data/nnue/nnue_weights.bin --plot-dir ../data/plots
python3 quantize_nnue.py ../data/nnue/nnue_weights.bin ../data/nnue/nnue_weights_int8.bin
```

### Strength Arena (Head-to-Head)
```bash
python3 tools/arena/run_arena.py --ref1 "" --ref2 main --games 200 --time 500 --threads 14
```

### Hybrid MCTS + Alpha-Beta (default search)

This is the production search and it is **on by default** in `arena` and `selfplay`. It
requires NNUE — the PUCT priors come from the policy head — and falls back to pure
alpha-beta with a warning if the weights are missing or `--heuristic` is set.

```bash
# Pure alpha-beta on both sides (this is how you measure the AB baseline)
python3 tools/arena/run_arena.py --ref1 "" --ref2= --no-mcab --games 400 --time 200

# Hybrid vs pure alpha-beta
python3 tools/arena/run_arena.py --ref1 "" --ref2= --e1-no-mcab --games 800 --time 200
```

**Parameters.** Every knob has three places to set it, strongest first:

1. a command-line flag — `--mcab-nodes`, `--mcab-leaf-depth`, `--mcab-cpuct`, `--mcab-fpu`,
   `--mcab-score-scale`, `--mcab-root-select`, each also in `--e1-`/`--e2-` per-engine form;
2. the **override block** at the top of `tools/arena/arena.cpp`, `tools/selfplay/selfplay_main.cpp`
   and `tools/arena/run_arena.py` — every field starts *empty*, and each field's comment
   states the production value it falls back to;
3. the production defaults themselves, which live in one place: `mcab::McabParams` in
   `src/mcab.hpp`.

Production values: `leafDepth=0`, `fpuReduction=0.0`, `cPuct=1.5`, `scoreScale=200`,
`nodeBudget=20000`, `rootSelectMode=visits`.

**Validity range — read before changing the time control.** The +46.9 ±23.5 Elo was measured
at **200 ms/move**. The hybrid runs at roughly a ninth of pure alpha-beta's nodes/s (41k vs
359k), so the trade is expected to invert at much shorter time controls; it has not been
measured anywhere but 200 ms, on one machine. `leafDepth` is the parameter that matters —
`leafDepth=2` is ~340 Elo *worse* than `leafDepth=0`. Note also that `leafDepth=0` means there
is essentially no alpha-beta below the leaf, so what wins here is plain PUCT MCTS over the
policy/value net rather than the alpha-beta rollouts the design is named for. Playing at a very
different time control? Re-measure, and use `--mcab-leaf-depth` / `--no-mcab` accordingly.

Self-play adds root Dirichlet noise on top (on by default there, off in arena — noise buys
opening diversity in training data and would only add variance to a strength match). The GA
tuner still opts in explicitly via `bin/tune_spsa --mcab-tuning`, which tunes
`mcabCPuct`/`mcabLeafDepth`/`mcabFpuReduction`/`mcabScoreScale`/`mcabNodeBudget`. See
`bin/selfplay --help` and `bin/tune_spsa --help` for the full flag list. A `--ref` older than
the feature compiles fine and plays pure alpha-beta, with a one-line warning.

### Web GUI
```bash
cd gui_web && python3 -m http.server 8000   # open http://localhost:8000/index.html
```

---

## Key Self-Play Flags

| Flag | Default | Meaning |
| --- | --- | --- |
| `--games N` | 1000 | Total games to generate |
| `--out PATH` | required | Output template (`{shard:03d}` for sharding) |
| `--depth N` | 40 | Max search depth |
| `--time-ms N` | 100 | Move time budget in ms |
| `--threads N` | all cores | Parallel worker threads |
| `--mc-mode` | off | Enable fast AlphaZero-style Monte Carlo policy-temperature sampling |
| `--mc-obvious-plies N` | 3 | Initial plies with low fixed temperature |
| `--mc-temp-opening F` | 1.35 | Starting temperature for decay phase |
| `--mc-temp-end F` | 0.12 | Ending temperature for decay phase |
| `--nnue-weights PATH` | `data/nnue/nnue_weights_int8.bin` | Quantized NNUE weights file |
| `--policy-order` | **on** | Enable NNUE policy-assisted move ordering |
| `--policy-order-min-depth N` | 3 | Depth floor for policy-assisted ordering |

Full flag list: `bin/selfplay --help`.

---

## Repository Structure

| Directory | Description |
| --- | --- |
| `src/rules.hpp` | Board representation, move generation, BFS path algorithms |
| `src/dsu.hpp` | Rollback disjoint-set union for fast wall legality checking |
| `src/cat.hpp` | Corridor Attention Table (CAT) heat map calculation |
| `src/search.hpp` | Negamax, alpha-beta, transposition table, move ordering, quiescence |
| `src/endgame_race.hpp` | Exact retrograde DP pawn-race endgame solver |
| `src/nnue.hpp` | 354-feature NNUE network, incremental accumulator, inference |
| `src/mcab.hpp` | Hybrid MCTS + alpha-beta search (PUCT tree, alpha-beta leaves) — the default search |
| `src/search_tuning.hpp` | Single override surface for every runtime knob of `search.hpp` (empty field = production value) |
| `tools/` | CLI tools & orchestrators (`selfplay`, `arena`, `spsa`, `qtp`, `path_clash_bot_arena`) |
| `benchmarks/` | Performance benchmarks (`main.cpp`, `bench_*.cpp`) |
| `tests/` | Correctness and NNUE parity test suite |
| `training/` | PyTorch training, quantization, dataset readers |
| `gui_web/` | WebAssembly build and HTML/JS frontend |

---

## NNUE Architecture

`354 → 256` accumulator (SCReLU activation) → two heads:
- **Outcome (WL)**: `256→32→1` (game result; used as search leaf evaluation).
- **Policy**: `256→209` (canonical move probability logits; used for move ordering & Monte Carlo sampling).

### Input Features (354)
- **Pawn positions**: 81 (own) + 81 (opponent) one-hot cells.
- **Wall slots**: 64 (horizontal) + 64 (vertical) slot bitboards.
- **BFS distances**: 21 (own) + 21 (opponent) one-hot distance buckets.
- **Remaining walls**: 11 (own) + 11 (opponent) one-hot wall count buckets.

All features are canonical (perspective-relative with row reflection for side 1). Quantization uses fixed `QA=255`, `QB=64` (QAT).
