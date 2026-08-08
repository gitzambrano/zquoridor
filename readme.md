# Zquoridor

Play online against Zquoridor in your browser: https://gitzambrano.github.io/zquoridor/
Web GUI with C++ engine compiled to WebAssembly.

Zquoridor is a 2-player Quoridor engine, played on a 9x9 board with 10 walls per player. The 4-player variant is out of scope.

The engine searches moves using negamax with alpha-beta pruning, evaluating positions via a neural network (NNUE with incremental accumulator) or a heuristic evaluation function (evalSimple).
The project is a sister engine to Zchezz (chess engine), mirroring its conventions and architecture.

The playing strength goal is to beat the overwhelming majority of human players (~99%) and subsequently benchmark against the strongest public Quoridor engines available.

## 1. Directory Structure

- src/: engine source code (rules.hpp, search.hpp, endgame_race.hpp, nnue.hpp, dsu.hpp, cat.hpp, main.cpp, selfplay.hpp, selfplay_main.cpp)
- teste/: correctness tests and performance benchmarks (test_rules_sanity.cpp, test_search_staging.cpp, test_move_ordering.cpp, test_endgame_race.cpp, nnue_verify.cpp, bench_quiescence_toggle.cpp)
- training/: Python NNUE training pipeline (train_nnue.py, train_nnue_numpy.py, quantize_nnue.py, read_selfplay.py, parity_check.py)
- data/: weights and selfplay data (data/nnue/nnue_weights_int8.bin)
- gui_web/: web interface and WASM files (engine_wasm.cpp, build_standalone.py, app.js, style.html, zquoridor.js, zquoridor.wasm, zquoridor.html)
- build/: build scripts for Windows (.bat), Linux/macOS (.sh), Termux

## 2. How to Build and Run

### 2.1 Windows (build/*.bat)
- build_bench.bat: compiles performance benchmarks (-O3 -march=native -mavx2 -mfma)
- build_tests.bat: compiles correctness tests (-O2)
- build_selfplay.bat: compiles selfplay binary
- build_wasm.bat: compiles WebAssembly engine (requires emsdk)
- build_all.bat: compiles all targets

### 2.2 Linux / macOS (build/*.sh)
- build/build_bench.sh
- build/build_tests.sh
- build/build_selfplay.sh
- build/build_wasm.sh
- build/build_all.sh

### 2.3 Termux (Android/ARM)
- build/build_termux.sh (compiles without AVX2/FMA flags)

### 2.4 Web GUI
- cd gui_web && python3 -m http.server 8000
- open http://localhost:8000/index.html (or open zquoridor.html standalone)

### 2.5 Self-play
- bin/selfplay --games 2000 --out data/selfplay_001.bin --threads 12 --time-ms 100

### 2.6 NNUE Training
- python3 training/train_nnue.py --data data/selfplay_*.bin --out data/nnue/nnue_weights.bin
- python3 training/quantize_nnue.py data/nnue/nnue_weights.bin data/nnue/nnue_weights_int8.bin

## 3. Implementation Status
- Position & Rules (rules.hpp): State representation, Zobrist hashing, DSU rollback wall legality, BFS pathfinding.
- Search (search.hpp): Negamax, Alpha-Beta, Transposition Table, Move Ordering (CAT heat table), LMR+PVS, RFP+LMP, Wall Quiescence, Repetition Table with Contempt = -30.
- Empty-Hands Endgame Solver (endgame_race.hpp): Exact retrograde DP for 0-wall pawn race.
- NNUE (nnue.hpp): Fully integrated into search, incremental accumulator maintenance across search stack.

## 4. NNUE Architecture
- Input features: 354 features (81 own pawn, 81 opp pawn, 64 H wall, 64 V wall, 21 own dist bucket, 21 opp dist bucket, 11 own walls left, 11 opp walls left).
- Structure: 354 -> 256 accumulator (SCReLU) -> 3 independent heads (Outcome WL 256->32->1, Auxiliary 256->32->1, Policy 256->209).
- Canonical Perspective: Relative to side to move with row reflection on perspective = 1.
- Quantization: QAT (QA=255, QB=64), WeightClipper enforced during optimization.
- Incremental update: AccPair maintained per ply during negamax search.
