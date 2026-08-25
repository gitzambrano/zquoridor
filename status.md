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
4. **Time-control curve**: measure hybrid vs pure AB at 50ms/500ms/1000ms;
   the hybrid's advantage was only validated near 150-200ms.
5. **GUI worker for play turns** (`gui_web/`): analysis already runs in a
   worker; the play-mode `engineMove` (up to Titan 8 s) still blocks the
   main thread. Replay-based request shape is ready in `worker.js`.
6. **MCTS visit-share %** in analysis PV rows needs a
   `rootNodeForInspection` export channel.

---

## 3. History Notes (durable lessons only)

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

## 4b. Web GUI (`gui_web/`)

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

---

## 5. Evaluation Conventions

| Stage | Function | Range | Perspective |
| --- | --- | --- | --- |
| Search score | `nnueEvalInt` | ~[-30000, 30000] | mover-relative |
| MCTS Q | `scoreToQ` | [0, 1] | mover-relative win probability |
| Heuristic | `evalSimple` | ~[-600, 600] | mover-relative |
| Dataset field | `TrainingSample::evalNNUE` | 0..65535 | absolute White |
| GUI display | `formatEval` | 0%..100% | absolute White |
