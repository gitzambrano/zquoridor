# Additional Plan — zquoridor

**Source:** Comparative analysis between `zquoridor` and `titaniummachine1/titanium-engine` (Rust, same domain — Quoridor) plus classic alpha-beta engine techniques (Stockfish and similar).

## Current Baseline Status

- **Search:** Negamax + alpha-beta pruning, TT (2M entries), iterative deepening with aspiration windows, log-log LMR + null-window PVS, Reverse Futility Pruning, Late Move Pruning, killer moves + history heuristic, CAT (Corridor Attention Table) for wang ordering/pruning, node BFS cache + LRU distance cache per wang topology, perft regression tests.
- **Empty-Hands Endgame:** Exact pawn race solver (3-level retrograde DP) integrated into `chooseMove` (including when the root itself has `wallsLeft == (0,0)`).
- **NNUE:** 354 sparse features (81+81 own/opponent pawn position, 64+64 H/V wall slots, 21+21 BFS distance buckets one-hot, 11+11 remaining walls buckets), incremental accumulator with float32 and quantized int8/int16, dual heads WL + auxiliary (imitates `evalSimple`) + policy (209 outputs). Integrated directly into search (`Negamax::evalMode::NNUE `, `search.hpp`i). Canonical perspective (row reflection for perspective 1).
- **Builds:**�`build_bench``, `build_selfplay``, `build_tests``, `build_arena``, `build_wasm`` (batch / shell scripts).
- **NNUE Training:** `training/train_nnue.py` (PyTorch) with automatic checkpoint resume, early stopping, LR/weight-decay schedules with warmup, RAM/VRAM auto-budgeting.

:# Priority 1 — NNUE: Remaining Walls Feature + Perspective Mirroring Fix

### 1a. Remaining Walls (`wallsLeft`) Feature
`NUM_FEATURES` extended from 332 to 354 ((22 = 2x11 one-hot buckets for `wallsLeft` from 0 to `WALLS_PER_PLAYER=10`per side). `buildAccumulator`/`buildAccumulatorQuant` and incremental `updateAccumulatorForMove(Quant)` updated. Placing a wall decrements `wallsLeft` of side to move and updates feature in accumulator.

### 1b. Perspective Mirroring Fix
Row index mirrored (`r -> WS-1-r` for walls, `r -> N-1-r` for pawns) when perspective is 1. Ensures symmetric starting positions yield identical evaluation for both sides.

### 1c. QAQ (Quantization-Aware Training) & Weight Clipping
QAT pipeline using fixed `QA=255` and `QB=64`. `WeightClipper` active in PyTorch optimizer loop clamping weights within int8/int16 range each step.

:# Priority 2 — WebAssembly NNUE Preload Fix
WASM standalone bundle (`build_standalone.py`) modified to embed `zquoridor.data` (250 KB NNUE int8 weights) as base64 and inject it via `Module["getPreloadedPackage"]`, avoiding runtime HTTP fetch errors when opened via `file://`.
