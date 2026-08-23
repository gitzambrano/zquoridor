# Status & Technical Reference

Technical reference and roadmap for Zquoridor. History here is minimal:
only durable lessons that a future contributor (human or LLM) must not
relearn by experiment.

---

## 1. Production Status

- **Search**: hybrid PUCT MCTS (`src/mcab.hpp`), default in all tools.
  Leaves are direct `nnueEvalInt` (`leafDepth=0`). Backup `AvgBlend`,
  tree reuse on, node budget 20000/move (not time-binding: p99 is ~18.6k
  nodes at 150ms).
- **Network**: Gen 5 NNUE (`data/nnue/nnue_weights_int8.bin`),
  `354 -> 256` SCReLU accumulator, WL head `256->32->1`, policy head
  `256->209`. QAT with fixed `QA=255`, `QB=64`.
- **Performance baseline (2026-08-22)**: +194.6 ±23.9 Elo over the
  pre-optimization engine at 150ms/move; production NPS ~29k. See
  History Notes.

---

## 2. Future Plans (priority order)

1. **Self-play generation, Gen 6**: regenerate datasets with the current
   engine (~3.3x more MCTS nodes per move than the data the Gen 5 net saw),
   root visit distribution as policy target. Retrain, quantize, arena-test
   vs Gen 5.
2. **SPSA tuning** (`tools/spsa/tune_spsa.cpp`): MCAB knobs (`cPuct`,
   `fpuReduction`, `scoreScale`) and search tuning set from
   `search_tuning.hpp`. Run AFTER Gen 6 data exists.
3. **Structural speed work** (profile is flat now; diminishing returns):
   fixed-size storage for per-edge vectors inside `MCABNode`
   (~4 vectors x up to 131 moves, heap-allocated per expansion), and
   eliminating the ~1KB `AccPair` copy per traversed edge.
4. **Time-control curve**: measure hybrid vs pure AB at 500ms/1000ms;
   the 40ms and 200ms points are done (inv/ab-policy round: the hybrid
   wins at both, see History Notes).
5. **GUI worker offload** (`gui_web/`) -- tracked by the GUI section.

---

## 3. History Notes (durable lessons only)

- **inv/ab-policy round (2026-08-23, branch `inv/ab-policy`)**: asked
  whether the engine should rely less on the MCTS side of MCAB. Answer:
  no. Measured on the same binary with runtime knobs (branch has the
  toggles, all default OFF):
  - Node-budget curve vs pure alpha-beta (400/300 games per point,
    NNUE): at 40ms/move the hybrid wins by +54.3 ±33.8 Elo at the
    production 20000-node budget and never hits that budget (about 3.9k
    simulations fit in 40ms; fully time-bound). At 200ms/move it wins
    by +97.4 ±40.0. Budgets below about 10k nodes LOSE strength at
    200ms (-92.5 ±40.0 at 2k) because the engine stops early with time
    left. The node budget is a ceiling, not a tuning knob for speed.
  - Compute map (`benchmarks/map_compute.cpp`): policy passes are only
    about 7 percent of hybrid wall time; leaf evals about 8 percent;
    the MCTS loop itself (per-expansion move generation plus PUCT over
    up to 131 children) is about 84 percent. A quantized policy pass
    costs 790ns, roughly one leaf eval, so per-node policy inside
    alpha-beta is affordable since the vectorized inference work.
  - Cheap policy-inside-AB tricks are all null results at both 40ms
    and 200ms (each 300 to 400 games, Elo margin about +/-34 to +/-39):
    policy-seeded history (B) +10.4/-1.2, policy-scaled LMR (C)
    -1.7/-5.8, policy-mass LMP (D, base 0.15) -11.6 at 200ms. Policy
    LMR doubles nodes-to-fixed-depth (+99 percent at depth 8); the
    others cost or save under 10 percent of nodes.
  - Two-stage root (E: rank root children with shallow AB, keep top-k
    for the MCTS) is REJECTED: -96.2 ±39.6 (depth 2, top 8) and
    -398.4 ±65.0 (depth 3, top 12, truncated ranking) versus
    production MCAB. Cutting the root starves PUCT exploration, and a
    time-truncated ranking drops mostly wall candidates.
  The toggles stay in `search.hpp`/`mcab.hpp` default-off, pinned by
  `tests/test_policy_ab.cpp` (defaults bit-exact in both eval modes).
- **Speed round 2026-08-22 (+194.6 Elo total)**: NNUE forward passes
  vectorized (branchless int32 accumulation, bit-exact integer math),
  `PlayerPathCache` slimmed 740B -> 190B, quiescence NNUE stand-pat exits
  before BFS, TT prefetch, fast sigmoid/exp for MCAB inference
  (`mcabFastExp`; selfplay recording stays exact). Details are in git
  history (`perf/speed-elo-100`, merged).
- **Rejected -- do NOT re-attempt without new evidence**:
  - Direction-mask BFS (precomputed blocked/goal bits): -18% nps; setup
    cost dominates because caches make most BFS calls short.
  - `leafDepth >= 1`: catastrophic both before AND after the 2026-08-22
    speedups (retested at 406 games: about -250 Elo). Direct NNUE leaves win.
  - Progressive widening: -6.9 ±20.7 Elo (measured pre-speedup; revisit
    only alongside SPSA).
  - Minimax-hard backup, FPU > 0: lost to `AvgBlend` / FPU 0.0.
- **Profiler pitfalls (Windows/MinGW)**: use `benchmarks/profile_mcab.cpp`
  as template; cache module base/ImageBase BEFORE starting the sampler
  thread (GetModuleHandleA can deadlock on the loader lock against a
  suspended main thread); link with `--disable-dynamicbase
  --image-base=0x140000000` so nm addresses match runtime RIPs.
- **Repetition semantics**: wall moves split the repetition horizon (walls
  never disappear); `push(hash, irreversible)` exploits this. The subtle
  case (post-wall position recurring via pawn cycles) is pinned by
  `tests/test_repetition_diff.cpp`.

---

## 4. Module Reference

- **`src/rules.hpp`**: `State`, bitboard walls, move generation, four BFS
  variants sharing one engine, `evalSimple`, BFS node cache
  (`PlayerPathCache`) + cross-node cache (`PlayerPathCacheTable`, ~12MB),
  `RepetitionTable`.
- **`src/dsu.hpp`**: rollback DSU for cheap wall-legality proofs.
- **`src/cat.hpp`**: Corridor Attention Table (wall ordering heat).
- **`src/search.hpp`**: pure alpha-beta (TT, killers/history, LMR+PVS,
  wall quiescence, policy ordering); every heuristic has a runtime toggle.
- **`src/endgame_race.hpp`**: exact solver for wall-less pawn races;
  read its header comment before touching it.
- **`src/nnue.hpp`**: incremental quantized accumulators, forward passes,
  feature definitions.
- **`src/mcab.hpp`**: production hybrid PUCT MCTS; `McabParams` holds all
  production values.
- **`tools/arena/`**, **`tools/selfplay/`**, **`tools/spsa/`**,
  **`training/`**: strength testing, dataset generation, parameter
  tuning, NNUE training.

## 5. Evaluation Conventions

| Stage | Function | Range | Perspective |
| --- | --- | --- | --- |
| Search score | `nnueEvalInt` | ~[-30000, 30000] | mover-relative |
| MCTS Q | `scoreToQ` | [0, 1] | mover-relative win probability |
| Heuristic | `evalSimple` | ~[-600, 600] | mover-relative |
| Dataset field | `TrainingSample::evalNNUE` | 0..65535 | absolute White |
| GUI display | `formatEval` | 0%..100% | absolute White |
