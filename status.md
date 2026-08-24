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

- **Race-solver audit round (2026-08-23, branch `inv/race-fuzz`)**:
  adversarial verification of `src/endgame_race.hpp` against an
  independent brute-force oracle (`tests/race_oracle.hpp`: from-scratch
  successor generation, naive win-set fixpoint, separate DTM relaxation).
  Findings and outcomes:
  - **F1, fixed**: `raceDisjointGate` computed its ply arithmetic from raw
    BFS distances without guarding degenerate values. For direct callers of
    the public inline utilities a pawn already on its goal row (rawDist 0)
    produced dtm -1, and a sealed pawn (rawDist -1) flipped the winner with
    a negative dtm. No real game reaches such states (wall legality keeps
    both paths alive), so the production hook is unchanged; the gate now
    refuses when either rawDist < 1 and Service B answers instead.
  - **F5/F6/F7, documented-only**: the race budget globals are process-wide,
    so selfplay threads interfere on solver accounting (perf-only; the
    fallback path is the ordinary heuristic node); the race-hook TT probe
    accepts any EXACT entry regardless of depth (sound approximation);
    the empty-handed root branch ignores repetition history while internal
    nodes rank it first (rare in practice). See `notes_race.md` (branch
    log) for details.
  - **Camping symptom verdict** ("engine stays on the first row"): no bug.
    Survey over 2040 hands-empty roots x tiebreak on/off: about 35% of
    chosen moves keep the root-side pawn on its back two rows, and ALL of
    them achieve the oracle-best child value. Camping while lost is the
    correct maximum-delay defense; camping while won is minimal-DTM
    geometry. `endgameProgressTiebreak` ON cuts lost-side camping by about
    13% relative at zero optimality cost. In the quasi-endgame (no solver)
    camping-while-ahead appears at short budgets but mostly turns into
    wall trades that add +1 to +4 opponent plies under deeper search.
  - **Permanent coverage**: `tests/test_endgame_race_fuzz.cpp` wired into
    both build_tests scripts as entry [16/16] (about 12 s at -O2): 8468
    oracle comparisons, 33872 root-optimality checks across all four
    toggle combos, budget-fallback and cache-eviction stress, degenerate
    probes including 600 randomized ones, and a compact engine-vs-engine
    differential where every decisive game must end in exactly the
    predicted DTM. `benchmarks/bench_race_differential.cpp` scales that to
    366 games standalone (360/360 exact lengths).
  - **Drift fix**: `build/build_tests.sh` still compiled the deleted
    `lazy_acc_parity.cpp` (the Linux suite aborted there) and lacked three
    newer tests; both build_tests scripts now list the same 16 entries.

- **Production adoption round (2026-08-23, branch `prod/findings-2026-08`)**:
  merged `inv/qsendgame-ext`, `inv/ab-policy`, and `inv/contempt-wandering`
  into one integration line, then flipped ONE production default:
  `endgameProgressTiebreak` is now ON (see the setter comment in
  `src/search.hpp`). Everything else ships unchanged: MCAB stays the
  production search with its 20000-node budget; contempt stays -30;
  wall-quiescence caps stay at the old constant values. Validation for
  the default flip: full correctness suite green on the integration tree
  (staging 0 divergences, LMR/PVS agreement 97/100 and 58/58 decisive,
  contempt/repetition T1-T5, mcab core/dispatch/phase9), plus a 600-game
  arena vs `main` at 150 ms/move, MCAB + NNUE both sides:
  **+2.9 +-26.9 Elo (47.2% vs 46.3%, 39 draws) -- statistically neutral**,
  which is the expected result: the tie-break only reorders EXACTLY equal
  solver values, so the payoff is behavioral (loser backward moves
  47% -> ~27% in wall-less endings), never a value change. Rejected for
  production after measurement (see entries below): TT-clear per move,
  parity-anchored race draws, low-wall quiescence bonus, policy-history /
  policy-LMR / policy-LMP tricks, two-stage AB pre-ranking. All rejected
  features remain available as runtime knobs for future retests.
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
- **Wall-quiescence extension sweep (2026-08-23, inv/qsendgame-ext)**: the
  compile-time `QS_MAX_EXTRA_PLIES` bound became instance params
  (`qsMaxExtraPlies`, plus an endgame rule `qsLowWallsBonus` when
  `qsLowWallsThreshold >= wallsLeft[0]+wallsLeft[1]`). Defaults are
  bit-identical to the old constant (frozen reference values in
  `tests/wall_qextension_reference.inc`; setters resize `nnueAccStack`
  and quiescence clamps itself to free stack space, so raised caps can
  never write past the accumulator stack). The endgame rule was measured
  over bonus {1,2,3} x threshold {2,4,6} from low-wall corpus positions at
  100 ms/move: no significant Elo change vs defaults in heuristic or NNUE
  mode (best pooled result +8.4 +-25.6 at 700 games), despite only ~11%
  extra nodes to depth 10. Defaults stay production; knobs remain exposed
  (`--qs-low-walls-bonus/--qs-low-walls-threshold`). Full data:
  `data/wallext/SUMMARY.md` (untracked logs alongside).
- **Silent printf truncation (2026-08-23)**: `%u` applied to an
  `(unsigned long long)` value prints only the low 32 bits -- this is how
  the first frozen wall-extension reference file stored truncated wall
  bitboards and became unreplayable. Generators that emit state
  coordinates must round-trip every row through reconstruction and assert
  equality (see `tools/wallext/gen_reference.cpp`).
- **Contempt / wandering investigation 2026-08-22** (`inv/contempt-wandering`
  worktree; full data in that branch, `investigation_data/`): the reported
  pawn "wandering" in near-endgames is NOT caused by contempt. Sweep of
  `setContempt` over {-60,-30,-15,-5,0} moved the backward-move rate by
  less than measurement noise in both eval modes and both time controls;
  head-to-head -30 vs 0 gave -44 +/-76 Elo (n.s.). Findings, by share of
  the symptom: (1) In wall-less endings the LOSING side makes ~47%
  backward moves even with exact solver values -- among tied max-DTM
  losses the empty-handed root branch falls back to move-generation
  order. This is game-theoretically optimal resistance, but it looks
  like shuffling; `setEndgameProgressTiebreak(true)` reorders only
  exactly-equal children and cuts it to ~27% with identical results
  (Elo -12 +/-60 vs default). All-draw roots never occurred (0/290).
  (2) The persistent TT carries path-dependent repetition scores across
  real moves; clearing the TT per move removes nearly all of it
  (heuristic mode: repetition-drawn games 41/120 -> 0/120; NNUE mode:
  backward moves 0.125 -> 0.081) but costs about -71 +/-78 Elo from lost
  TT reuse -- rejected for production. (3) Confirmed sign inconsistency:
  race-solver draws use `contempt` from the node mover regardless of ply,
  while repetition draws anchor to root parity; provably unable to flip a
  move choice inside the solver regime (wins/losses are ~1e5 apart), so
  left as-is behind `setParityAnchoredRaceDraw`. Depth >=7 fixed-depth
  searches do not finish near-endgame positions within sane budgets
  (>20M nodes at depth 8); production time controls reach depth 5-6 there.
  New test: `tests/test_contempt_repetition.cpp`; bench:
  `benchmarks/bench_contempt_wandering.cpp`.

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
- **`tools/wallext/`**: wall-quiescence extension experiment kit
  (deterministic low-wall corpus, frozen-reference generator with
  round-trip proof, nodes-to-depth bench, single-process pairwise arena,
  match orchestrator `run_wallext.ps1`).

## 5. Evaluation Conventions

| Stage | Function | Range | Perspective |
| --- | --- | --- | --- |
| Search score | `nnueEvalInt` | ~[-30000, 30000] | mover-relative |
| MCTS Q | `scoreToQ` | [0, 1] | mover-relative win probability |
| Heuristic | `evalSimple` | ~[-600, 600] | mover-relative |
| Dataset field | `TrainingSample::evalNNUE` | 0..65535 | absolute White |
| GUI display | `formatEval` | 0%..100% | absolute White |
