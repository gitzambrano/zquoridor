# Status & Technical Reference

Technical reference and roadmap for Zquoridor. History here is minimal:
only durable lessons that a future contributor (human or LLM) must not
relearn by experiment.

---

## 1. Production Status

- **Search**: hybrid PUCT MCTS (`src/mcab.hpp`), default in all tools.
  Leaves are direct `nnueEvalInt` (`leafDepth=0`), except in a wall-poor
  endgame: `endgameMoverWallThreshold=0` gives alpha-beta leaves of 2 plies
  once the side to move has no walls left. Backup `AvgBlend`,
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
6. **MCTS visit-share %** in analysis PV rows needs a
   `rootNodeForInspection` export channel.

---

## 3. History Notes (durable lessons only)

- **MCTS endgame wandering (2026-08-24, branch
  `claude/zquoridor-mcts-bug-hf6uqv`)**: a user reported that the engine
  spends its walls fast, then shuffles its pawn sideways forever in a won
  race. The position was rebuilt exactly from the reported game and lives
  in `benchmarks/repro_wander.cpp`. Player 0 is the engine with 0 walls and
  5 steps to the goal. Player 1 holds 8 walls and needs 13 steps. All four
  DIST counters match the report. At production settings the engine answers
  f6, g6, f6, g6 and holds its own distance at 5 for six moves while the
  opponent closes from 13 to 6.

  The measured cause is the WL head, not the tree. `benchmarks/diag_wander.cpp`
  prints a win probability of 0.137, 0.153, 0.191 and 0.281 for pawn
  distances 5, 4, 3 and 2 in that position, and it only separates the moves
  at distance 1. With the wall stocks swapped, 8 against 0, the same head
  reports 0.997 to 0.999 for every distance. The head therefore reads the
  remaining-wall counts and is almost blind to the pawn race. The policy
  head stays correct and gives the advancing move a prior of 0.73.

  The hybrid amplifies the blind spot. Leaves are the raw net value at
  `leafDepth = 0`, so no leaf ever calls `winner()`. `AvgBlend` averages
  20000 near-equal leaves, the Q values of the root moves land within 0.02
  of each other, their order is inverted against the truth, and `MaxVisits`
  then picks a shuffle. Instrumentation showed the tree reaches ply 17 to 21
  in that position and still creates ZERO terminal nodes, so an MCTS solver
  would have nothing to prove. Pure alpha-beta wanders in the same position
  too, only less: it scores +2 against +1 over seven moves.

  Two candidate fixes were measured and both failed:
  1. `BackupMode::MinimaxHard` plays the repro position perfectly at every
     setting tried, even at a node budget of 2000. It is still NOT the fix.
     It lost the arena 0 wins to 56 in 60 games, approximately -585 Elo at
     200ms, and it also lost the 40-position corpus of
     `benchmarks/bench_mcab_endgame_progress.cpp` (mean progress 5.650
     against 6.625, 5 wandering positions against 0). This repeats the
     earlier rejection under "Rejected" below, without the FPU confound.
  2. An endgame alpha-beta leaf gated on the COMBINED wall stock at or below
     8 also fixes the position, but it fires through most of the game and
     cuts the node rate by 56%. It measured -88.7 +/- 86.7 Elo over 60
     games at 200ms.

  What shipped is `McabParams::endgameMoverWallThreshold`, ON in production
  at threshold 0 since 2026-08-24. It gates the alpha-beta leaf on the wall
  stock of the SIDE TO MOVE at the root. At threshold 0 the rule fires only
  after that side spends its last wall, which is the reported condition, and
  it costs approximately 6% of the node rate. It plays the repro position
  perfectly. It measured +17.4 +/- 37.8 Elo over 300 games at 200ms.

  Read that number as neutral, not as a gain. The interval spans
  approximately -20 to +55, therefore 300 games do not exclude a small
  loss. The rule is on because it removes a reported, reproducible failure
  at no measured cost. Re-measure with more games before you quote it as an
  improvement.

  Two consequences of the default change. Self-play data generation now
  uses alpha-beta leaves in wall-poor endgames, so `.bin` files recorded
  after this date differ from earlier ones in that regime. The WASM and GUI
  build inherits the new default through `McabParams{}`, but only after
  `build/build_wasm.sh` regenerates the bundled HTML.

  `tests/test_mcab_endgame_leaf.cpp` pins the exact threshold, the gate and
  the fixed behavior. Arena flags are `--e1-mcab-endgame-mover-walls` and
  `--e1-mcab-endgame-leaf-depth`.

  Durable lesson: a position where the engine plays badly is not proof that
  the search is at fault. Print the value head across the moves in question
  before you change the tree. Here the tree was doing exactly what a correct
  PUCT does with a flat, wrong value function.

  Open item: retrain the WL head on wall-poor endgames. The head cannot
  currently separate a won race from a lost one once a side runs out of
  walls, and no search rule fixes that at the source.

- **Arch-aware Linux builds (2026-08-23)**: `build_bench.sh`,
  `build_selfplay.sh` and `build_qtp.sh` add `-mavx2 -mfma` only on x86-64
  (`uname -m`); every other architecture compiles with `-march=native`
  alone, which enables NEON on ARM (AArch64 and AArch32 alike). Before
  this change, a non-x86 `g++` rejected the two flags and the scripts
  failed outright. Deployment note that came out of the first AArch64
  server install: a weaker engine there usually means missing NNUE
  weights (the QTP binary then runs `evalSimple` and prints a stderr
  warning) or a lower time budget, not the ISA itself.
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
- **QFEN lessons (2026-08-23, pinned by `tests/test_notation.cpp`)**:
  - Wall token rank is the slot row + 1 (the wall's south-west cell). An
    off-by-one here survives casual testing because every token still parses;
    it only shows up as import/export disagreement on real positions.
  - The checkable budget invariant from a QFEN alone is
    `placed + wallsLeft0 + wallsLeft1 <= 20`, NOT `placed <= wallsLeft sum`.
    A finished game legitimately exports hands 0/0 with all 20 walls placed.
    Fewer than 20 total is accepted (the editor may drop walls).
  - Small enclosures are impossible with legal walls (a 1x1 box needs two
    crossing walls; a 2x1 box needs colinear-adjacent ones). The smallest
    legal sealed region is 2x2 -- the shape the path-rejection test uses.
- **parity_check.py drift (found and fixed 2026-08-24)**: the Python half
  of the `nnue_verify` cross-check had drifted twice from `nnue.hpp` --
  first the walls-left buckets (it still read 332 of the real 354
  features), then the perspective mirroring of pawn/wall features added
  to `buildAccumulator`. Because it has no file-size check (unlike
  `quantize_nnue.py`), both drifts produced confident-looking garbage
  instead of an error. It now reads 354 features, sets the walls-left
  one-hots, and mirrors pawns/wall slots per perspective exactly like
  the C++ side. Verification rule after the fix: the int8 block must
  match `nnue_verify` digit-for-digit (integer math is deterministic),
  and the float32 block must match within summation-order noise
  (approximately 1e-6). Lesson: when `nnue.hpp` gains a feature or a
  transform, update `train_nnue.py`, `quantize_nnue.py` AND
  `parity_check.py` in the same commit.
- **PowerShell file round-trips corrupt UTF-8**: `Get-Content | Set-Content`
  mangles accented characters in this repo's scripts and adds CRLF/BOM. Patch
  committed files with Python byte I/O or the Edit tool only.
- **GUI bugs the browser test caught that unit tests could not (2026-08-23)**:
  - Tap-to-move was dead since the P2 interaction work: the board pointer
    handler passed a `{r,c}` cell object into code comparing against a
    numeric display index. Silent no-op, no exception -- only a click-driven
    Playwright test catches it.
  - Naming any `EXPORTED_RUNTIME_METHODS` in emcc turns it into an
    allowlist: `HEAPU8` silently vanished from `Module`, and every string
    export crashed only when first used. Export what you touch.
  - A `$` helper bound to `getElementById` does not take descendant
    selectors; `$('#ioFmt .on')` returned null at event time. Keep one
    lookup discipline per codebase.
  - Manual board flip was instantly undone: `doFlip` toggled `B.flipped`,
    then `syncFromEngine` recomputed it from `humanSide`. The flip now lives
    in `S.flipped` (settings) and `syncFromEngine` derives the display
    orientation from it -- derived state must have exactly one writer.
  - Deleting the last entry of the Recent Games sheet re-called
    `showRecentGames()`, which early-returns on an empty list and left the
    stale row visible. Re-rendered surfaces need an explicit empty state.
  - The analysis worker request must carry only the plies up to the review
    cursor (`pliesUpToCursor()`); replaying the full recorded line analysed
    a different position than the main-thread fallback when the user had
    navigated back.
  - Walls painted vertically mirrored since P2: `setData` stored engine-space
    wall slots while the paint and SVG-export paths index them in display
    space (pawns were converted at the same boundary, so only walls were
    wrong). A human wall clicked near their goal appeared near the engine's.
    Engine-state assertions cannot see it -- only a canvas `getImageData`
    probe at the display anchor can. Fix: `setData` mirrors wall slots like
    pawns; `QBoard.wallH/wallV` are display-space by convention.
  - `syncFromEngine` called `setData` (which renders) before
    `buildLegalSets`/`refreshHud`, so the legal dots and the side-to-move
    ring appeared only after the NEXT render. Build all board state first,
    then paint once.
  - Importing an already-finished game (QGN file/hash or editor apply)
    said "Game imported - your move" even though the position was won --
    the quiet end-check never announces. Imports now run the loud
    `checkEnd()`, so the result banner, toast and Recent entry appear.
  - `QBoard.fit` never assigned `this.S`, and `paintStatic` destructures
    `{S}` from the instance: the rounded base fill and every frame style
    drew with NaN coordinates -- silently ignored by canvas draw ops, so
    the board looked fine while NO frame style ever rendered, and only the
    beveled frame threw (createLinearGradient validates its arguments).
    Per-element painting (cells, pawns, walls) hid the hole; a canvas-hash
    sweep over every dressing option caught it in one pass.
  - The `crown` pawn style shared the `pillar` branch, so the
    distinct-shapes mapping (pillar -> crown for side 1) was a visual
    no-op -- an accessibility feature that did nothing for exactly the
    users who picked those styles. Crown now has its own crenellated
    silhouette.
- **Testing discipline for this GUI**: browser tests must drive the GUI
  entry points (`newGame()`, not `__w.newGame()`); calling C-level exports
  directly skips JS-side resets (humanSide, gameOver, clocks, level marks)
  and poisons every later assertion in the run.

---

## 4. Web GUI: NNUE weight loading

`gui_web/app.js` now calls `qr_load_nnue_weights("/data/nnue/nnue_weights_int8.bin")`
at boot. Before this change it never called the loader, so the published page
ran `evalSimple` with the hybrid MCTS off, because the hybrid requires NNUE.
That is a large strength loss, and it is the engine that produced the
reported pawn wandering: those games came from the heuristic evaluation with
pure alpha-beta, not from the MCTS.

Verified on the built bundle in headless Chrome: `qr_eval_mode_is_nnue()`
returns 1, `qr_mcab_active()` returns 1, and the console reports the load.
When the load fails the engine stays in the heuristic mode and a banner names
the path it tried, because a tooltip on a hidden checkbox is not a report.

Still open: the bundled `zquoridor.wasm` was compiled before
`McabParams::endgameMoverWallThreshold` existed, so the endgame wandering fix
reaches the native binaries only. One rebuild carries it into the page:

```bash
build/build_wasm.sh          # needs an active emsdk
cd gui_web && python3 build_standalone.py
```

Note for anyone rebuilding: `gui_web/zquoridor.wasm` is gitignored, so a
stale copy from another branch survives a `git checkout` and silently pairs a
mismatched binary with `zquoridor.js`. The symptom is a bundle that boots but
returns nonsense (`qr_walls_left` gave `undefined`). Delete the file, or
recover the matching one with `gui_web/extract_wasm_from_bundle.py`, before
running `build_standalone.py`.

---

## 5. Module Reference

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

## 6. Web GUI (`gui_web/`)

Premium interface per `gui-premium.md`. Phases P0-P5 (tokens, canvas board
`board.js` QBoard, wall/pawn interaction, HUD, layouts, play tab) landed
2026-08-22. P6 + P7 landed 2026-08-23; P8, P8b, P8c, P9 and P10 completed
2026-08-23 (same day), making the plan fully implemented:

- **P6 engine surface** (`engine_wasm.cpp`, exports synced in
  `wasm_args.rsp` and both `build_wasm.*` scripts -- keep the three lists
  identical):
  - Full game history in C (`g_histStates`/`g_histMoves` + cursor):
    `qr_goto_ply`, `qr_truncate_history`, `qr_ply_*` getters. Every live
    mutation records a ply; the JS snapshot-diffing of the old GUI is gone.
    `qr_goto_ply` rebuilds the repetition table from real history and resets
    the MCTS tree (tree reuse assumes one continuous line).
  - Scratch position shared by analysis/editor/blunder-check
    (`qr_scratch_from_live/from_ply/reset`, `qr_scr_*` getters). Analysis can
    never mutate the live game.
  - Multi-line analysis `qr_analyze(maxDepth, timeMs, lines)`: line 1 is a
    full search at the scratch root (the same move the game engine would
    play); lines 2+ come from independent searches of child positions after
    candidates ordered by a depth-2 pass. Honest but only line 1 is proven
    best -- documented here because the UI does not say it. PVs are rebuilt
    from the TT by the new public `Negamax::extractPv` (`src/search.hpp`).
    Runs on its own `Negamax g_anEngine` so the game TT stays warm.
  - Editor ops on scratch (`qr_edit_set_pawn/wall/walls_left/turn`,
    `qr_edit_validity` bitmask, `qr_edit_apply`). Wall placement refuses only
    physical conflicts (occupied/crossing/colinear); path problems go to the
    bitmask so the user can build freely and read what is wrong.
  - QFEN import/export (plan section 16.1): canonical export sorts walls by
    orientation/row/column, token rank = slot row + 1 (south-west cell);
    import validates before applying and reports the exact failing token via
    `qr_last_error`. Pinned by `tests/test_notation.cpp` (round-trip over
    random playouts + diagnostics table), which includes
    `gui_web/engine_wasm.cpp` directly to reach the anonymous namespace.
  - `_malloc`/`_free` and the `HEAPU8` view had to join the export lists:
    naming any `EXPORTED_RUNTIME_METHODS` turns it into an allowlist, and the
    JS string bridge reads QFEN bytes through `Module.HEAPU8`.
- **P7 analysis tab** (`app.js` section 14): ENGINE ON/OFF, depth select with
  infinite mode (iterative deepening loop capped at depth 22), 1-5 PV rows
  with eval chip / moves / opponent-distance delta badge / line preview on
  the board (walls translucent, pawn steps numbered), eval graph canvas with
  scrub-to-jump and blunder dots, ply navigation wired to `qr_goto_ply` plus
  `,` `.` `Home` `End` keys and an `A` engine toggle, move log rendered from
  recorded plies with click-to-jump, takeback that rolls back to the human's
  previous turn, hint (single-line analyze drawn as ghost/dot for 4 s),
  blunder check with cancel + progress + per-side accuracy card (win-prob
  drops 0.06/0.13/0.25 -> `?!` `?` `??`; `!` when the played move matches the
  engine best with a decent score).
- **P8 serialization & editor**: full QGN exporter (PGN-shaped headers,
  `{[%ev ...]}` comments from stored analysis scores, blunder symbols,
  result); game importer that strips headers/comments/numbering and applies
  tokens one by one through the legality-checked C surface, stopping at the
  first bad token with its index; dialect normalization for
  orientation-first (`Ha5`, `V f3`), coordinate-pair (`c6-d6`) and bare move
  lists; shape-based routing (`routeImport`) between QFEN and QGN. Text I/O
  modal (format toggle, copy/paste/load/download, diagnostics), drag-and-drop
  onto the board, file picker, `#qfen=`/`#qgn=` URL hash load plus Copy link,
  autosave of the live game to `zq.game` (debounced) with a 10 s Resume chip,
  and a 20-game `zq.recent` ring with Load/Copy/Delete sheet. Editor tab: tool
  palette, wall-budget steppers, side-to-move switch, live validity strip fed
  by `qr_edit_validity`, Apply gated on validity, Copy/Paste QFEN; while the
  pane is open the board renders the SCRATCH position.
- **P8b personalization**: settings schema v1 per plan section 11 with merge
  migration and corrupt-blob reset; presets Classic / Premium Dark / High
  Contrast / Minimal (touching any option flips the chip to Custom); 8 board
  themes including Carrara Marble (deterministic procedural veins drawn once
  into the static layer, mulberry32 seed) and Noir (value-separated pawns;
  light variant redefines wall tokens -- found missing by the contrast gate);
  board dressing switches (frame none/hairline/gilded/beveled, wall finish
  flat/beveled/glossy/etched, cell separation grooves/flat/inlaid,
  coordinates off/edges/all, board scale slider); accent colour presets plus
  custom picker with derived `--gold2`/dim/glow; 6 pawn styles (adds
  pawnChess and beacon), pawn size and shadow, distinct-shapes toggle
  (automatic in noir); sound packs wood/modern/marble/silent as data tables
  with per-event toggles, volume slider and Test button; haptics levels;
  motion full/reduced/off with speed multiplier over the duration tokens;
  density and text-size switches; handedness mirroring dock/toasts.
- **P8c image export**: PNG via `QBoard.renderExport` into a detached fixed
  size canvas (transparent-background, coordinates and footer-wordmark
  options) and SVG via `QBoard.toSVG`.
- **P9 accessibility**: `tools/gui/contrast_check.py` gate (wall/cell and
  wall/groove >= 3:1 for all 8x2 theme combinations, text >= 4.5:1) runs
  inside `build_standalone.py` and fails the bundle on regression; SR live
  region announcements after every ply; keyboard help overlay (`?`), full
  §14 key map including arrow-key pawn movement with Shift diagonals and
  straight-jump continuation; focus-visible outlines kept.
- **P10 standalone**: Google Fonts downloaded at bundle time into the
  committed `gui_web/fonts_cache/` and inlined as base64 WOFF2, so the
  single-file bundle needs no network; graceful fallback to the `<link>` if
  the cache is empty and the network is unavailable.
- **Analysis worker** (`worker.js`, plan section 12): a second WASM module
  instance receives self-contained requests (root QFEN optional plus the
  recorded packed plies), replays them onto its scratch and returns line
  data, keeping the main thread free during analysis. The main-thread
   slicing path remains as the silent fallback (file:// bundles cannot spawn
   the worker). Play-mode engine turns still run synchronously on the main
   thread -- tracked below as remaining work.
- **Post-commit hardening + classic board pass (2026-08-23)**: after the P0-P10
  commit, user reports ("engine plays for both sides", "walls cannot be
  placed", board looked worse than the old GUI) led to a turn-machine and
  input hardening round plus a board visual rework:
  - Turn ownership is now structural: `scheduleEngineTurn(delay)` is the
    single owner of the engine timer (a new schedule cancels any stale one),
    and `engineTurn` refuses to move unless it is really the engine's side --
    the engine can never move for the human even with a duplicated timer.
    Clicking while the engine thinks gives a throttled toast instead of
    queueing input.
  - Wall gestures never fail silently: `snapGhost` always snaps to the
    nearest anchor and reports ok/assisted/bad with a reason; presses near a
    cell center route to the pawn handler, not the wall gesture; `#board`
    has `touch-action:none` so touch drags commit instead of scrolling.
  - The default obsidian look moved back toward the old board: flat uniform
    cells with thin light gridlines, hairline frame, slim ivory walls
    (`--wall:#ece7da`, was gold), subtle goal markers and softer
    highlights. All dressing options keep working.
  - The mcab play budget is node-based (~20k nodes), so engine replies are
    usually faster than the level's nominal time -- tests must not assume
    `engineThinking` stays true for the level duration.
- **Dock press-and-drag was dead (found and fixed 2026-08-24)**: the M1
  gesture (press a wall on the dock, drag onto the board, release) never
  placed anything. `setPointerCapture` retargets every pointermove to the
  dock element, but the dock handler early-returned once `_dragging` was
  set -- so the ghost froze off-board and the release had nothing to
  commit. The board drag (press directly on a groove) was unaffected,
  which is why earlier suites passed. Fix: keep forwarding snapGhost on
  every move while the drag is engaged (`app.js`, dock pointermove).
  Pinned by the new `gui_web/test_deep_click.py`, an 80-check click
  suite covering pawn destinations, wall refusal reasons, keyboard arms,
  confirm chips, settings persistence, worker-off analysis fallback,
  editor validity gating, QFEN/QGN diagnostics, resume chip and recent
  games. Geometry note for assist tests: pressing dead-center on an
  illegal anchor leaves all neighbours at exactly 1 U, outside the
  0.9 U assist radius -- press at approximately 0.48 U toward the legal
  neighbour instead.
- **Per-feature sweep** (`gui_web/test_gui_features.py`, 2026-08-23): one
  pass over EVERY settings control and dressing option (8 board themes,
  6 pawn styles, frame/finish/surface/coords/scale, overlay toggles, UI
  theme, sound controls, multi-PV + line preview + infinite depth + graph
  scrub, SR live region, h/v/Esc keys, level modal, PNG/SVG content,
  `#qgn=` cold load). It found the `this.S` hole and the crown/pillar
  duplication above plus a paths-overlay settings lag (fixed: applySettings
  now calls `togglePaths`). Canvas-hash comparisons need one caveat: a
  value equal to the current default (coords `edges`, pawn `disc`) is a
  no-op by definition -- start the sweep away from defaults.
- **Full game simulation** (`gui_web/test_gameplay_sim.py`, 2026-08-23):
  plays a complete game like a person -- pawn clicks, arm+click walls,
  drag walls, keyboard walls, hint, flip mid-game, takeback+redo, review
  round-trip, level cycle -- asserting turn ownership, ply parity, wall
  conservation and ghost/dots state after EVERY engine reply; then runs
  the full analysis pass mid-game (3 PVs, previews, scrub, blunder check,
  accuracy card), resumes the same game to completion (proving analysis
  leaves the game state intact), and hands the QGN to a second page via
  `#qgn=`. First run reached ply 38 with zero desyncs: the turn machine,
  wall placement and analysis isolation hold under real play.
  Script-writing lessons that cost time:
  - A "nudge" click on the board while the engine thinks can land AFTER
    the reply arrives and then COMMIT a real move -- race, not bug. Nudge
    clicks in tests must target elements outside the board.
  - `B.dots` are display cells. `engPawnToDisp` is self-inverse, so a
    double conversion cancels only when another `cell_pt` follows; an
    inline `cellCenter(engPawnToDisp(d))` lands on the mirrored rank.
  - The return-to-game chip lives in the analysis pane; reviewing from
    the play tab goes back with the `End` key. The level chip opens the
    NEW GAME modal -- mid-game level changes are the `S` key.

- **GUI v2 rebuild (2026-08-25)**: the interface was rebuilt after a user
  review. The complaints were a dark board that did not read as a board,
  pawns that jumped between cells, a takeback button that did nothing, and
  two colored streaks across the middle of the board. The rebuild keeps the
  engine plumbing untouched. `engine_wasm.cpp` and the WASM exports did not
  change.
  - **Look**: a warm wooden board inside the dark Zchezz chrome. The board
    tokens moved to a `wood` default: tan cells, a walnut frame, dark walnut
    walls. `walnut` became a dark board with ivory walls, because its old
    values failed the contrast gate at 1.40 against a required 3.0. Nine
    board themes now exist.
  - **Layout**: one player strip above the board and one below it. Each
    strip carries the name on its outer edge, then the clock, the wall pips
    with a count, and the distance to goal. The old dual HUD bar, the
    desktop HUD rail and the wall dock are gone. The useful buttons sit in
    one Zchezz-style row under the board. There is one breakpoint at 900px.
  - **Wall placement without a mode**: `boardHit()` in `app.js` classifies a
    pointer position as a cell body or as a groove, and names the
    orientation of that groove. A hover over a groove paints a preview
    through the new `QBoard.setHover`. A press starts the drag, and a drag
    of more than `0.45*C` across the other axis flips the orientation. The
    `H` and `V` buttons only force an orientation. They are no longer
    required. The forced orientation expires after 6 seconds.
  - **Anchor geometry**: `anchorFor(o, px, py)` uses floor along the wall's
    own long axis and round across the groove. A first version subtracted 1
    from the vertical row. That version resolved the exact center of a
    groove crossing to the anchor above the correct one, and
    `test_deep_click.py` caught it through 7 failed wall checks.
  - **Pawn animation**: `QBoard.animateMove` tweens a pawn over 200ms and
    arcs a jump. The tween starts before the state sync paints, therefore
    the piece slides from the cell it left. A `requestAnimationFrame` loop
    runs only while an animation is active.
  - **Engine off the main thread**: `worker.js` accepts a new `bestmove`
    command. The worker replays the recorded line into its own live game and
    runs `qr_engine_move`, which is the hybrid search. The analysis path
    cannot serve this: `qr_analyze` runs pure alpha-beta on the scratch
    position, so it would silently change how the engine plays. Measured at
    the Titan level, the worst main-thread stall during an 8 second search
    fell to 5ms. A game from a custom position still searches locally,
    because the worker has no replay root for it.
  - **Takeback**: `#btnTakeback` was never wired to anything. The button is
    wired now, and six further defects around it are fixed. The guard
    refuses a takeback during review. `clockHist` gives the time back.
    Entries in `AN.scores`, `AN.annots` and `levelMarks` past the new end
    are dropped, because those keys are ply indexes and would otherwise
    attach to different moves. The eval bar resets. `engineGen` invalidates
    a search that is still running. `W.scratchFromLive()` restores the
    scratch position that the search loop overwrites.
  - **Path overlay**: the old version walked greedily in engine coordinates
    and handed engine cells to a renderer that expects display cells.
    Therefore both lines came out mirrored and crossed in the middle of the
    board. Those were the streaks in the user report. `shortestPath()` is
    now a real BFS, `recomputePaths()` runs on every state change, and the
    cells are converted with `engPawnToDisp`.
  - **Clock**: the default is 5+3 instead of none, so that the strip has a
    clock to show. `addIncrement()` credits the increment, which the schema
    declared but no code applied.
  - **Modal buttons**: `.btn` forces a 36px width. Every button inside a
    modal was clamped to that width, so the labeled buttons overlapped and
    swallowed each other's clicks. `.modal .btn` now sizes to its label.
  - Tests: all five browser suites pass. `test_deep_click.py` loads the
    bundled `zquoridor.html`, therefore `build_standalone.py` must run
    before that suite reports on current code.

- **GUI v2 review pass (2026-08-25)**: a second user review asked for a
  denser HUD and a desktop layout closer to a chess interface.
  - Each player strip is now one line: name, clock, wall pips with a count,
    then the distance to goal as a bare number behind a hairline. The engine
    level left the strip, because the header chip already names it.
  - Desktop moves the status line and the button row into the side rail, so
    the board keeps the full height of its column. `layoutReflow()` moves the
    same nodes across the 900px breakpoint. There is one copy of each
    control, never two.
  - The move log follows the same reflow. On a phone it fills the space under
    the button row, which was empty before. It is selectable text in both
    layouts.
  - The evaluation bar is a vertical bar on the board's left edge with a gold
    balance line and a numeric readout in a 30px gutter. The gutter is
    desktop only. On a phone the bar is a 4px hairline and the number stands
    down, because every pixel of width is board there.
  - `QBoard.fit()` measured `clientWidth`, which counts the zone padding, and
    it ignored the evaluation bar's share of the row. The board was therefore
    sized larger than its box and spilled over its neighbours at
    width-limited desktop sizes. It now measures the real content box.
  - The player strips align to the board, not to the column. The evaluation
    bar and its gutter sit inside the board zone, so a centred strip was
    visibly off by half of that width.
  - `.btn` forces a 36px width. The two wall buttons carry a live count, so
    they size to glyph plus number. The corner badge was too small to read.
  - `packedToTok()` read the low byte of a packed move for both kinds of
    move. A pawn move carries its destination in bits 16 to 23, so every pawn
    move in an analysis line printed as `a1`, and every wall printed as a
    bare `H`. The decoder now matches `plyNotation`.
  - The move count is a reading, not a control. Recent games is reached
    through the header menu, which already lists it.

- **GUI v2 element audit (2026-08-25)**: a per-element measurement pass, after
  a review found faults that a per-category read had missed.
  - **Levels are named after the chess pieces**, in order of material value:
    Pawn 50 ms, Knight 150 ms, Bishop 400 ms, Rook 1 s, Queen 2.5 s, King 8 s.
    The default is Rook, which holds the budget the old default held. Settings
    schema 2 maps each old key to the piece with the same budget. `curLevel()`
    falls back to the default, because an unknown key used to reach the search
    and throw on a missing time budget.
  - **One evaluation bar**, 12 px. It is vertical beside the board on a wide
    screen and horizontal under the board on a phone, and it is the same
    element in both. The mobile race strip is gone: it read as a second
    evaluation bar. The race meter keeps its labelled place in the Play panel.
    `setEval()` is the single writer and picks the axis; the analysis setter
    and the takeback reset both call it.
  - The wall count moved next to the player name. Beside the distance the two
    numbers read as one.
  - Every element in the board column now aligns to the board itself: both
    player strips, the status row, the button row, the move log and the
    horizontal evaluation bar, at every breakpoint.
  - Faults the measurement pass found: the vertical bar was 35 px taller than
    the board; the two wall buttons had different widths; the horizontal bar
    was indented twice; the button row overflowed a 390 px screen by 4 px
    because the settings gear was duplicated from the header; the level chip
    was a 26 px tap target; and the move counter was still a focusable button
    after it stopped being a control.

- **GUI v2 analysis and readability pass (2026-08-25)**:
  - The vertical evaluation bar is 24 px. Its readout is a small percentage in
    one fixed place under the bar, not a label that slides with the fill. The
    percentage is player 0's share, which is what the bar's geometry shows, and
    it matches the absolute-colour display convention in section 5.
  - The wall pips are 4 by 14 px with a 3 px gap. The board coordinates dropped
    to `0.32 * C` at 72 % alpha, so they label the board without competing with
    it.
  - **Evaluation left the play move log.** The play log lists moves only. The
    Analysis tab owns evaluation and has its own list, `renderAnMoveLog()`,
    with a score for every ply from `AN.scores`. A ply with no score shows a
    dash, and the list says how many are missing and that Blunder check fills
    them.
  - **Pawn drag was never implemented.** Tap to select and tap the destination
    worked, but pressing the pawn and pulling it did nothing. `QBoard.setDragPawn`
    holds the piece under the pointer and the release commits over a legal
    destination. A press that does not move is still the first tap.
  - The evaluation graph hides itself below 4 plies, because an empty plot area
    reads as a broken box.
  - `gui_web/` has a new end-to-end check that drives every feature through the
    real interface: both ways to move a pawn, all four ways to place a wall,
    takeback, hint, flip, paths, the evaluation bar, three analysis lines, the
    blunder check filling every ply, navigation, the editor, the settings modal
    and the level list. 30 checks, all passing.

- **GUI v2 graph, wall matrix and analysis export (2026-08-25)**:
  - **The evaluation graph auto-scales.** A whole game plotted over 0 to 1 is a
    flat line in the middle for every position that is not already decided,
    which shows nothing. The band is centred on 0.5 and opens only as far as
    the data needs, with a floor of `+-8 %` so that noise is not magnified into
    a swing, and a corner label names the band. A quiet game now reads at
    `+-8 %` instead of flat.
  - `--p0-soft` and `--p1-soft` did not exist in the rebuilt stylesheet. The
    graph read them for its area fill, an empty string leaves `fillStyle`
    unchanged, and the area silently never painted. Both tokens are defined,
    and `drawGraph()` falls back to the solid player colour.
  - **Two flexbox faults that swallowed clicks.** `#anGraph` could be shrunk by
    the flex column while its canvas still painted 72px, and `#anLines` could
    be shrunk below its content so its rows overflowed under the graph. In both
    cases the block that painted last covered the controls under it. Only
    `.card.grow` may take the leftover height now; every other child of a pane
    keeps its content height.
  - The Analysis tab has an EXPORT card: Copy QFEN, Copy game, Save .qgn and
    Copy link. QFEN exports the position the cursor is on, so any position in a
    game under review can be taken out. Verified: at the live end it gives
    `e2 e8 10 10 - 0 2`, one ply back `e2 e9 10 10 - 1 1`.
  - New `gui_web/test_wall_matrix.py`: 57 checks over wall placement with a
    mouse and with touch. Drag direction sets the orientation, hovering a
    groove previews that orientation, a cell body previews nothing, and every
    corner and edge anchor accepts both orientations by drag and by press. The
    asserted invariant is that the wall lands on the slot the ghost showed.

- **GUI v2 responsiveness and preference round (2026-08-26)**:
  - **A human move painted itself only after the engine replied.** `playPawn`
    applied the move to the engine but left the board arrays at the old cell.
    The slide tween therefore ended on a snap-back, and the piece reached its
    destination only when the engine reply ran `syncAll()`. A placed wall had
    the same delay. `playPawn`, `commitGhost` and the arrow-key move now call
    `syncFromEngine()` immediately after they apply the move, while the tween
    covers the repaint.
  - The default sound pack is quiet and rounded. Each tone starts with a short
    attack ramp (`atk`) and can pass through a lowpass filter (`lp`), the
    square and sawtooth voices are gone, and the default volume dropped from
    0.6 to 0.5.
  - The shortest-path overlay is opt-in only. Settings schema 3 clears a stored
    `paths` value once, and the path button shows its own on state.
  - The gold ring around the pawn that is to move is gone. The legal-move dots
    already show whose turn it is.
  - Board coordinates are half their previous size: `0.16 * C` on the edges,
    `0.12 * C` in study mode, and `0.15 * C` in the SVG export.
  - Wood-theme walls are pale rails. The tokens are now `--wall:#ecebe5`,
    `--wall-hi:#faf9f4` and `--wall-edge:#a6a49a`. `QBoard.wallDrawRect()`
    paints each beam at approximately 1.6 times its slot thickness, so the
    wall reads against the wood bed. Hit tests keep `wallRect()`.
  - The desktop gap between the evaluation bar and the board grew from 8 px to
    18 px. The board fitter reads the computed gap, so the board size adjusts
    by itself.
  - The settings modal is tabbed: Appearance, Board, Sound, Play & data. Each
    tab shows a one-line summary, rows are at least 36 px tall, and segment
    labels are full words. `wireSettingControls()` tolerates the controls of
    the tabs that are not open. The header menu groups its items under GAME,
    SHARE & FILES and EXTRAS labels.
  - The browser test suites learned the tab navigation. All seven suites pass:
    `test_gui_features`, `test_deep_click`, `test_features_e2e`,
    `test_wall_matrix`, `test_gameplay_sim`, `test_browser_full` and
    `test_standalone`.

### 2026-08-26: presentation pass

Design plan in `docs/superpowers/specs/2026-08-26-gui-refactor-plan.md`. All
seven browser suites pass after the change.

- **Settings modal opened empty (bug).** `$('btnSettings').onclick =
  modalSettings` passed the `MouseEvent` as the `tab` argument, and
  `modalSettings(tab)` starts with `if (tab) settingsTab = tab`. A
  `MouseEvent` is truthy, so `settingsTab` became the event object and every
  later lookup (`cards[...]`, `SETTINGS_TAB_HINT[...]`, the `on` class) missed.
  The first open rendered 45 characters of text; a click on any tab passed a
  real string and the same modal then rendered 459. Every modal opener is now
  wrapped in an arrow function. `modalNewGame()` takes no argument, so it never
  showed the defect.
- **Type and space tokens.** `:root` gained `--space-1` to `--space-8` and
  `--fs-xs` to `--fs-xl`. The type steps use `clamp()`, so they answer to both
  the viewport and the root font-size percentage that the text-size setting
  writes. 58 hardcoded `rem` sizes moved onto the scale. The body base went
  from `.7rem` to `--fs-md`, which measures 13.68 px at 1280 px wide against
  11.2 px before. `.btn` moved from a fixed `width:36px` to `min-width`, and
  `#controls` from `height` to `min-height`, because the larger labels
  overflowed the button row by 26 px.
- **Coordinates.** They were 8.6 px at 72 percent alpha, in a `--coord` that
  measured 2.26:1 against `--frame` in the default wood theme. Six of the nine
  board themes were below 4.5:1. The size is now `max(10, 0.235 * C)` at 92
  percent alpha, which measures 12.7 px on a 600 px board, and the six failing
  `--coord` values moved on lightness only, so the hue and the saturation are
  unchanged. `tools/gui/contrast_check.py` gained a `coord/frame` pair at 4.5,
  so the build fails if this regresses.
- **Evaluation bar.** `#boardZone > #evalWrap` carried `margin-bottom:15px` to
  reserve room for the `#evalNum` label. `#boardZone` centres its children, so
  the margin was centred with the strip and pushed the bar 7.5 px above the
  board's top edge. The label moved inside the strip, the margin is gone, and
  the bar now shares the board's top and bottom edge exactly.
- **Wall depth.** `board.js` hardcoded the shadow, the glossy highlight and the
  two etched inset lines, so a light board got a shadow built for a dark board.
  They are now `--wall-shadow`, `--wall-gloss`, `--wall-etch-dark` and
  `--wall-etch-light`.
- **Pawn.** The pawn never varied with engine strength. `LEVELS` maps a level
  to a search time budget only. The reported base was the contact shadow: a
  hard edged ellipse of radius `R*.92`, drawn at `R*.60` under the centre,
  which reads as a plinth. It is now a radial gradient that fades to zero, and
  the `disc` and `beacon` styles gained a specular highlight. The `pillar` and
  `pawnChess` styles still draw a flared foot, because that is their design.
- **Modal on desktop.** `#overlay` used `align-items:flex-end` at every width,
  so the sheet stuck to the bottom edge of a desktop window. Above 900 px it is
  now a centred dialog with four corners and a `fadeZoom` enter.

---

## 7. Evaluation Conventions

| Stage | Function | Range | Perspective |
| --- | --- | --- | --- |
| Search score | `nnueEvalInt` | ~[-30000, 30000] | mover-relative |
| MCTS Q | `scoreToQ` | [0, 1] | mover-relative win probability |
| Heuristic | `evalSimple` | ~[-600, 600] | mover-relative |
| Dataset field | `TrainingSample::evalNNUE` | 0..65535 | absolute White |
| GUI display | `formatEval` | 0%..100% | absolute White |
