# Status & Technical Reference

Technical reference, current engine status, design decisions, and future roadmap for Zquoridor.

---

## 1. Current Production Status

### Active Production Architecture
- **Production Search**: Hybrid PUCT MCTS (`src/mcab.hpp`) enabled by default across all tools (Arena, Selfplay, WASM GUI).
  - **Performance**: **+46.9 ±23.5 Elo** over pure alpha-beta at 200ms/move (800 games).
  - **Leaf Evaluation**: Fast `nnueEvalInt` direct evaluation at `leafDepth=0` (skips alpha-beta and quiescence rollouts at leaves, maximizing simulations/second).
  - **Backup Mode**: `AvgBlend` (standard MCTS average value $Q = W/N$), replacing legacy minimax-hard backup.
  - **Tree Reuse**: Subtree preservation across successive game plies (`treeReuse=true`).
  - **BFS Cache Integration**: Shared `PlayerPathCacheTable` during incremental NNUE accumulator updates.
- **Production Neural Network**: **Generation 5 NNUE** (`data/nnue/nnue_weights_int8.bin`).
  - Architecture: `354 -> 256` accumulator (SCReLU) with 2 output heads:
    - **WL Head**: `256 -> 32 -> 1` (win/loss evaluation).
    - **Policy Head**: `256 -> 209` (move priors and search ordering).
  - Quantization: Quantization-Aware Training (QAT) with fixed scales $Q_A=255$, $Q_B=64$.

### Production Parameters (`mcab::McabParams`)
| Parameter | Production Value | Description |
| --- | --- | --- |
| `enabled` | `true` | Hybrid MCTS default active (`--no-mcab` for pure AB) |
| `nodeBudget` | `20000` | Maximum MCTS node capacity per move |
| `leafDepth` | `0` | Alpha-beta plies at leaves (0 = direct NNUE evaluation) |
| `cPuct` | `1.5` | PUCT exploration constant |
| `fpuReduction` | `0.0` | First Play Urgency reduction |
| `scoreScale` | `200.0` | Sigmoid score scaling constant (`NNUE_EVAL_SCALE`) |
| `rootSelectMode` | `MaxVisits` | Move selection criterion at root |
| `backupMode` | `AvgBlend` | Value propagation strategy |
| `treeReuse` | `true` | Reuse search subtree between moves |
| `progressiveWidening` | `false` | Experimental lazy candidate expansion (disabled) |

### Key Experimental Finding: Progressive Widening
- **Implementation**: Lazy candidate enumeration (`pawnStepMoves` + local slot overlap), policy-based sorting, and on-demand BFS validation (`isWallMoveLegal`) as visits grow ($k = k_0 + c \cdot N^\alpha$).
- **Result**: Tested at 150ms/move over 1000 games (`initial=16, c=2.0, alpha=0.5`): **-6.9 ±20.7 Elo** vs. standard expansion with ~19% lower node throughput (10,345 vs 12,796 nps).
- **Status**: Kept in codebase as an experimental option (`--mcab-progressive-widening`), disabled in production.

---

## 2. Future Roadmap

### 1. Genetic Algorithm (GA / SPSA) Parameter Optimization
- Run automated tuning (`tools/spsa/tune_spsa.cpp`) across MCAB search parameters (`cPuct`, `fpuReduction`, `scoreScale`, `widening` parameters).
- Investigate `leafDepth` tuning specifically tailored for various time controls.

### 2. Time-Control Curve Analysis
- Profile Hybrid MCTS vs. pure Alpha-Beta across multiple time budgets (50ms, 100ms, 200ms, 500ms, 1000ms).
- Calibrate dynamic search selection or adaptive parameters based on available clock time (e.g. fast pure AB for ultra-bullet, deep MCTS for longer controls).

### 3. Generation 6 NNUE Self-Play & Training
- Generate new self-play datasets using the production MCTS visit distributions at the root as policy targets.
- Train, quantize, and evaluate **Generation 6 NNUE** against the Gen 5 baseline.

### 4. Web Worker Integration for WASM GUI
- Offload browser engine execution (`gui_web/`) to dedicated background Web Workers to maintain UI responsiveness at high search budgets.

---

## 3. Module & Architectural Reference

- **`src/rules.hpp`**: Board representation (`State`), bitboard wall structures (64-bit H/V), move generation (`pawnStepMoves`, `legalWallMoves`, `legalMoves`), four BFS metrics (`hasPathToGoal`, `shortestPathLen`, `shortestPathTouchSlots`, `pathRobustness`), and heuristic fallback `evalSimple`.
- **`src/dsu.hpp`**: Disjoint Set Union with rollback for incremental wall legality validation.
- **`src/cat.hpp`**: Corridor Attention Table (CAT) computing per-cell heat maps to guide wall ordering.
- **`src/search.hpp`**: Classical Negamax alpha-beta engine with TT, killers, history heuristic, LMR+PVS, RFP+LMP, wall quiescence, and depth-gated policy ordering.
- **`src/endgame_race.hpp`**: Retrograde dynamic programming solver for wall-less pawn race endgames (81×81×2 state space).
- **`src/nnue.hpp`**: Incremental accumulator management and quantized forward passes for dual-head NNUE (Value + Policy).
- **`src/mcab.hpp`**: Production hybrid PUCT MCTS engine with SFINAE compile-time compatibility traits for external/legacy engine integration.
- **`src/search_tuning.hpp`**: Unified parameter override structs and CLI parsing for search heuristic tuning.
- **`tools/arena/`**: Dual-engine head-to-head benchmarking tool with confidence-interval Elo calculation (`run_arena.py`, `arena.cpp`).
- **`tools/selfplay/`**: Multithreaded self-play data generation with thread-safe stack allocations.
- **`tools/spsa/`**: SPSA / Genetic Algorithm tuner (`tune_spsa.cpp`).
- **`training/`**: PyTorch NNUE training scripts (`train_nnue.py`, `quantize_nnue.py`, `parity_check.py`).

---

## 4. Evaluation Conventions & Scales

| Stage | Function / Field | Range | Frame / Perspective |
| --- | --- | --- | --- |
| **Search Score** | `nnueEvalInt` | ~[-30000, 30000] | Mover-relative (positive = side to move is winning) |
| **MCTS Q Value** | `scoreToQ` | [0.0, 1.0] | Mover-relative win probability |
| **Heuristic Score** | `evalSimple` | ~[-600, 600] | Mover-relative |
| **Self-Play Dataset** | `TrainingSample::evalNNUE` | 0 .. 65535 (`EV_SCALE`) | **Absolute White perspective** (65535 = White win, 0 = Black win) |
| **GUI / Web Display** | `formatEval` | 0% .. 100% | **Absolute White perspective** (50% = equal) |

---

## 5. Recent Changelog

- **2026-08-15**: Tested progressive widening at 150ms/move (1000 games): -6.9 ±20.7 Elo; kept experimental and default disabled.
- **2026-08-15**: Promoted `BackupMode::AvgBlend` to production default across engine and tools.
- **2026-08-15**: Fixed arena NPS reporting to track expanded MCTS nodes during hybrid search.
- **2026-08-14**: Integrated `nnueEvalInt` direct leaf evaluation at `leafDepth=0` and connected shared `PlayerPathCacheTable` to MCTS accumulator building.
- **2026-08-13**: Promoted Hybrid MCTS (`mcab.hpp`) to production default (+46.9 ±23.5 Elo over pure alpha-beta at 200ms).
