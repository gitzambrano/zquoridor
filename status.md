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

- **2026-08-22b** (branch `perf/speed-elo-100`): Second round on top of the first: **fast exp/sigmoid for MCAB inference paths** (`mcabFastExp` in `mcab.hpp`: scoreToQ + policy softmaxes; degree-5 poly of 2^f with exponent-bit scaling, ~1e-6 relative error; saturates like std::exp so MCAB_WIN_SCORE still maps to Q=0/1). NOT used by selfplay data recording (`nnueWinProbQuant` stays exact) nor any parity-checked path. Also replaced a per-expansion heap `std::vector` for policy logits with a stack buffer in `expandNode`. MCAB nodes/move ~14.6k -> ~18.5k (+29%). **Arena vs main: +194.6 (margem 23.9)**, 1000 games at 150ms/move (721W-213L-66D); production NPS 12.75k -> 28.8k. Post-optimization profile is flat (nothing above ~3%) -- remaining wins would be structural (per-node vector storage in MCABNode pool, AccPair copy elimination).
- **2026-08-22** (branch `perf/speed-elo-100`): Speed optimization round, **+156.8 Elo (margem 22.8) over main** (arena, 1000 games, 150ms/move; production NPS 12.9k -> 27.1k). Changes, all validated bit-exact or behavior-identical by tests:
  - **NNUE forward passes rewritten for auto-vectorization** (`nnue.hpp`) -- THE big win (~60% of MCAB wall time was `forwardPolicyQuant`): int64 scalar accumulation with a per-element branch replaced by branchless int32 accumulation (overflow-proof bound: max |sum| = 255*127*256 ~= 8.3M << INT32_MAX, so the integer is EXACTLY the same and the final double division stays bit-identical). Same treatment for the WL head 256x32 loop and `screluQuant` (int64 -> uint32 division). Verified by `nnue_verify`, `nnue_incremental_check` (|diff| = 0) and full suite.
  - **`PlayerPathCache` slimmed ~740B -> ~190B** (`rules.hpp`): dist as uint8_t, parent as int8_t (-1 sentinel), reached as an 81-bit mask + `cacheReached()`. `PlayerPathCacheTable` shrinks ~48MB -> ~12MB with 4x less copy traffic on get/put.
  - **Quiescence NNUE early stand-pat exit** (`search.hpp`): in NNUE mode the stand-pat no longer needs the BFS caches, so fail-high nodes return BEFORE paying 2 BFS. Heuristic mode unchanged.
  - **TT prefetch** in `tryMove` after `applyMove`.
  - **`RepetitionTable` bounded scan** (`rules.hpp`): `push(hash, irreversible)` marks wall moves; `isRepetitionDraw` scans only back to the newest wall entry INCLUSIVE (the post-wall position itself CAN recur via pawn cycles -- caught by the new differential test `tests/test_repetition_diff.cpp`, which compares old-vs-new semantics over random push/pop sequences). Performance-neutral in benchmarks but bounds worst-case scans in very long games. Callers updated: search.hpp/selfplay.hpp/arena.cpp (+ SFINAE `reptblPushCompat` in arena.cpp so old refs still compile).
  - **Tried and rejected**: per-cell direction-mask BFS (precomputed blocked/goal bits instead of per-neighbor `edgeBlocked`) -- measured -18% nps because setup cost dominates when caches make most BFS calls short. Do not re-attempt naively.
  - **New tooling**: `benchmarks/bench_repthist.cpp` (deterministic fixed-depth bench with realistic game history, heuristic+NNUE modes), `benchmarks/profile_harness.cpp` / `profile_mcab.cpp` (Windows sampling profilers: suspend main thread, bucket RIPs, resolve via nm; MUST cache module base/ImageBase before starting the sampler thread -- calling GetModuleHandleA from inside can deadlock on the loader lock while the main thread is suspended).
- **2026-08-15**: Tested progressive widening at 150ms/move (1000 games): -6.9 ±20.7 Elo; kept experimental and default disabled.
- **2026-08-15**: Promoted `BackupMode::AvgBlend` to production default across engine and tools.
- **2026-08-15**: Fixed arena NPS reporting to track expanded MCTS nodes during hybrid search.
- **2026-08-14**: Integrated `nnueEvalInt` direct leaf evaluation at `leafDepth=0` and connected shared `PlayerPathCacheTable` to MCTS accumulator building.
- **2026-08-13**: Promoted Hybrid MCTS (`mcab.hpp`) to production default (+46.9 ±23.5 Elo over pure alpha-beta at 200ms).
