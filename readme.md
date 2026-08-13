# Zquoridor

A high-performance Quoridor engine for the 9×9, 2-player variant (10 walls per side).

Negamax + alpha-beta search powered by a Quantization-Aware Trained (QAT) neural network (NNUE) for position evaluation and policy-assisted move ordering. Includes a multi-threaded self-play generator, NNUE training pipeline, strength-testing arena, test/benchmark suite, and WebAssembly browser GUI.

🎮 [Play online](https://gitzambrano.github.io/zquoridor/) — *WebAssembly build.*

---

## Features

- **Alpha-Beta Search**: Negamax search with transposition table, iterative deepening, and Principal Variation Search (PVS).
- **Policy-Assisted & CAT Move Ordering**: Killer moves, history heuristic, Corridor Attention Table (CAT wall heat map), and NNUE policy head logits (gated by depth floor `policyOrderingMinDepth`).
- **Pruning & Reductions**: Late Move Reductions (LMR), Reverse Futility Pruning (RFP), and Late Move Pruning (LMP) at shallow depths. All heuristics include runtime toggles.
- **Wall Quiescence Search**: Extends search past nominal depth for critical-looking wall placements.
- **Optional MCαβ Hybrid Search**: Best-first PUCT tree whose leaves are evaluated by a real shallow alpha-beta search instead of a random rollout, with minimax-hard backup (`tools/common/mcab.hpp`). **Off by default** — arena, self-play, and the tuner all run pure alpha-beta unless the hybrid is explicitly enabled, and enabling it costs the pure path nothing.
- **Exact Endgame Solver**: Exact retrograde DP pawn-race solver over 81×81×2 states when both players run out of walls (`wallsLeft==(0,0)`), with a real-time budget.
- **NNUE Evaluation & Policy**: 354-feature network (pawn cells, wall bitboards, bucketed BFS distances, and remaining walls) with SCReLU activation, outputting win probability and move ordering logits.
- **Fast Monte Carlo Self-Play Generator**: Multi-threaded C++ self-play generator with standard epsilon-greedy and AlphaZero-style Monte Carlo policy-temperature sampling (`--mc-mode`) for rapid opening generation. Stack-allocated to ensure zero heap corruption.
- **Training & Quantization Pipeline**: PyTorch training script with dataset blending (`train_nnue.py`), automated int8 quantization (`quantize_nnue.py`), and C++/Python numerical parity verification (`nnue_verify`).
- **Strength Arena**: Automated head-to-head match runner (`tools/arena/run_arena.py`) with Elo estimation and confidence intervals.
- **SPSA Parameter Tuning**: Multi-mode SPSA tuner (`tools/spsa/tune_spsa.cpp`) supporting continuous parameter tuning (`--mode spsa`), discrete depth sweeps (`--mode sweep-mindepth`), and multi-depth parallel tuning (`--mode hybrid`) with matplotlib visualization (`tools/spsa/plot_spsa.py`).
- **WebAssembly & Browser GUI**: Mobile-friendly browser UI compiled via Emscripten with WebAssembly engine backend.

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

### MCαβ Hybrid (optional, off by default)

Requires NNUE (the PUCT priors come from the policy head). Add `--e1-mcab` / `--e2-mcab` to
enable it per engine, or `--mcab` for both; `--mcab-nodes` and `--mcab-leaf-depth` set the
budget. A `--ref` older than the feature compiles fine and plays pure alpha-beta, with a
one-line warning.

```bash
python3 tools/arena/run_arena.py --ref1 "" --ref2 main --e1-mcab --mcab-nodes 2000 --mcab-leaf-depth 3 --games 200 --time 500
```

The search knobs `--mcab-cpuct`, `--mcab-fpu`, `--mcab-score-scale` and `--mcab-root-select`
also come in `--e1-`/`--e2-` forms, so one config can be played directly against another
(same tree budget, one parameter changed) instead of measuring each against pure alpha-beta:

```bash
python3 tools/arena/run_arena.py --ref1 "" --ref2= --mcab --mcab-leaf-depth 0 --e2-mcab-cpuct 2.5 --games 400 --time 200
```

Defaults are the measured working point: `leafDepth=0` (NNUE value + wall quiescence at the
leaf) and `fpuReduction=0.0`. At 200 ms/move that config beats pure alpha-beta by ~26 Elo over
1000 games; deeper leaves lose badly (`leafDepth=2` is ~340 Elo *worse*). Note this makes the
hybrid effectively plain PUCT MCTS over the policy/value net rather than alpha-beta rollouts,
and that it runs at roughly a tenth of pure AB's nodes/s. See the MCαβ design note in
`status.md` for the full table.

Self-play (`bin/selfplay --mcab ...`, with root Dirichlet noise on by default there) and the
GA tuner (`bin/tune_spsa --mcab-tuning ...`, which also tunes `mcabCPuct`/`mcabLeafDepth`/
`mcabFpuReduction`/`mcabScoreScale`/`mcabNodeBudget`) take the same switch. See
`bin/selfplay --help` and `bin/tune_spsa --help` for the full flag list.

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
| `tools/common/mcab.hpp` | Optional MCαβ hybrid search (PUCT tree + alpha-beta leaves), off by default |
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
