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

### 4. GUI v4 — Premium Interface (FUNCTIONAL COMPLETE, 2026-08-22)
- Full rewrite of the web GUI (`gui_web/`) to match the **Zchezz premium standard** (see `C:\Zchezz` sister project): dark gold-accented identity (Cinzel + JetBrains Mono), light/dark UI themes, six board themes, four pawn styles.
- Spec/contract: `gui_web/GUI_PLAN.md` — the single source of truth for the four workstreams:
  - **W1** `engine_wasm.cpp`: game object with history/cursor, navigation (`qr_goto_ply`/`qr_undo`/`qr_redo`), Multi-PV analysis without playing (`qr_analyze`, up to 5 lines via `rootNodeForInspection()` on the MCTS tree), position editor (`qr_edit_*` scratch buffer with validity bitmask), QFEN + game-text serialization.
  - **W2** `style.html`: static markup shell (Play / Analysis / Editor panels + settings/about/text modals), responsive desktop 2-column ≥900px / mobile single-column, `100dvh`, accessibility bar (focus-visible, contrast ≥4.5:1, `prefers-reduced-motion`).
  - **W3** `board.js`: canvas `QBoard` class — DPR-aware, animated pawn slides/wall fades, wall drag ghost with orientation flip gesture, legal-move dots, shortest-path overlays, lichess-style arrows, eval-driven goal glow, keyboard input, pointer events only (mouse+touch unified).
  - **W4** `app.js`: boot/settings persistence (`localStorage['zq.settings']`), play mode vs engine (depth/time/game-clock modes with live countdown clocks and flag-fall), hints, takebacks, move log with per-ply eval, vertical eval bar, synthesized WebAudio sounds, analysis mode with continuous sliced Multi-PV engine loop, blunder check with accuracy summary, eval graph, keyboard shortcuts, toast error surface.
- Feature parity targets from Zchezz: game panel (side selection, flip, clocks, move log, sounds, status line), analysis panel (engine toggle at any depth or ∞, eval bar, Multi-PV lines with visits, move navigation, eval graph with click-to-jump, one-click full-game blunder annotation, QFEN/game-text import-export), settings (themes, piece styles, toggles).
- **Beyond-Zchezz features** (inspired by claustrophobia.dev, the Quoridor reference site): named strength levels **Pebble→Titan** (50 ms–8 s time budgets over the same hybrid engine; `Custom…` reveals raw mode/depth/time selects; `S` cycles levels), a **race meter** (dual-fill bar showing each side's share of the remaining path — at-a-glance "who wins the race"), and **haptic feedback** on mobile (`navigator.vibrate` on move/wall/game-end/illegal, toggleable in settings).
- **Remaining**: real-device touch testing (desktop/mobile verified headless only); optional future items: rated puzzles, opening explorer.
### 5. Web Worker Integration for WASM GUI — DONE (2026-08-21)
- **Implemented** in `gui_web/app.js` (`SECTION 13b`): a background Web Worker runs its own WASM instance and performs every blocking search there, so long time-budget analyses never freeze the UI (mobile included).
- Architecture: no game state in the worker — the main thread sends a QFEN (`cmd:"position"`, coalesced so consecutive slices at the same cursor reuse the worker's MCTS/TT tree) plus single-shot `go` requests (`qr_analyze`, ≤250 ms per slice); results return as plain `{lines, nodes, depth}` objects. Worker boot embeds the emscripten factory via `ZquoridorModule.toString()` into a blob URL (`_scriptName` re-declared because the factory closes over it); standalone builds hand the worker the wasm/data bytes directly (file:// safe), dev builds use `locateFile`.
- Routed through the worker: continuous Multi-PV analysis loop, blunder check (both evals per ply), hints, and **play-mode engine moves** (best line's first move is applied on the main module; falls back to `qr_engine_move` if unusable). Every path degrades silently to main-thread slicing when workers are unavailable; a watchdog unblocks callers if a reply is lost.
- Verified end-to-end with headless Chrome + puppeteer-core over both `http://` (dev files) and `file://` (standalone bundle): worker boots, round-trip analysis returns a 12-ply PV in ~240 ms, and a full human-move → engine-reply exchange completes through the UI path in ~300 ms with zero page errors.
- Known limitation: the worker has no repetition context (QFEN root only), so engine moves computed there don't factor repetition history into contempt/3-fold avoidance — acceptable for play quality; the main-thread fallback retains full context.

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

## 6. Recent Changelog

- **2026-08-22**: GUI v4 UX audit round — every reported problem reproduced programmatically (headless Chrome measurements) and fixed:
  - **Wall count invisible** (root cause: CSS styled `.pwalls .pip`, JS emitted bare `'on'` class → pips rendered 546×0 px). Pips now carry `pip on/spent` classes, player-tinted gradients; explicit numeric counts added to both player bars (`#pl-walls-num-*`) and a wall toolbar.
  - **Wall placement moved under the board** (`#pl-wallbar`): large H/V toggle buttons + live "N left" counter for the human side + usage hint. Old hidden side-column buttons removed. Verified end-to-end with real PointerEvents: arm V → tap board → wall placed, counter drops.
  - **Board proportions** (claustrophobia.dev-style): corridor gap reduced from 23% to ~10% of cell (`board.js` `_resize` gap formula), cells now dominate visually; coordinate margin slightly tightened.
  - **Eval bar redesigned as ONE vertical bar**: 0–100% scale with faint ticks at 25/50/75 and a number chip that rides the gold/red boundary (clamped 6–94% so it never slides off). Both play and analysis panels. Eval value verified present across navigation/undo/live states (was reported disappearing).
  - **Sounds resynthesized**: shared lowpass bus + soft-attack envelopes + noise-burst `thud()` layer — wooden tap for pawn moves, heavy thunk for walls, warm resolved chord for game end; no more raw square/sawtooth beeps.
  - **Verified**: gap/cell = 0.096; pip height 16 px; wallbar below board; V-wall flow via real pointer events works; eval chip always shows a value; zero page errors on desktop + mobile viewports.
- **2026-08-22**: GUI v4 polish + hardening round (parallel subagent reviews of `board.js` and `app.js` against `GUI_PLAN.md`, then fixes):
  - **Features**: named strength levels Pebble→Titan with `Custom…` escape hatch (`#pl-sel-level`; `cycleStrength()`, shortcut `S`); race meter `#pl-race` (dual-fill, eased transitions, updated in `renderRaceMeter()`); haptic feedback via `navigator.vibrate` hooked into the sound helpers + `#cfg-haptics` toggle; player bars renamed to "You"/"Engine" (fixes mobile truncation); mobile stacked-layout fix (`#pl-board-col { flex:0 0 auto }` — removed the dead band under the board).
  - **Bug fixes (app.js)**: stale engine move could land in a new game — generation now captured at dispatch time and threaded through the worker callback (`engineMoveFromWorker(dispatchGen, engineSide, …)`) plus an `atLiveEnd()` re-check; worker time cap was hard-coded at 250 ms so named strength budgets never took effect — `ewAnalyzeAt(…, capMs)` passes a per-request cap (analysis slices stay at 250 ms); flag fall is now a real terminal state (`flagFallWinner` consulted by `gameOver()`/`renderStatus()`, cleared on `newGame`) instead of a toast-only event; jump sound detection implemented (Chebyshev distance vs. the mover's ply-two-back destination); input frozen during the blunder check (`bcRunning` guards in `humanPawnMove`/`humanWallMove`/`analysisPlay`, navigation blocked in `gotoPly`); eval graph no longer polluted by live analysis/browsing (`recordPosEval` gated on `bcRunning`); editor pawn handler no-ops for wall/erase tools.
  - **Bug fixes (board.js)**: double rAF loop race when a done-callback chains `animateMove` (loop keeps its handle until it actually ends); persistent last-move wall outline `_drawLastWallTint` closes the spec gap where wall moves had no highlight once animations were off; keyboard selection cursor only materializes on arrow keys and resets on `setPosition`; hover state torn down on pointer-up (previously suppressed later identical hovers); second-finger touches ignored during an active drag.
  - **Verification**: headless Chrome E2E over http (dev files) and file:// (standalone bundle), desktop 1440×900 and mobile 390×844 viewports: zero page errors across play/analysis(Multi-PV 3)/editor/settings; worker-driven engine exchange ~300 ms at Silver 250 ms; race meter, level select, advanced-group reveal, and mobile gap (10 px) all confirmed programmatically.
- **2026-08-21**: GUI v4 premium rewrite started (`gui_web/GUI_PLAN.md`): engine_wasm.cpp rewritten as game object with history/cursor, Multi-PV analysis (`qr_analyze` + `rootNodeForInspection()`), position editor and QFEN/game-text serialization; new canvas `board.js`; `style.html` shell (Play/Analysis/Editor) in Zchezz-style dark-gold identity; `app.js` rewritten around the new exports. In progress: build-script export lists, standalone bundler update, WASM rebuild, Web Worker offload.
- **2026-08-21**: Web Worker offload implemented (see Roadmap §5) — background search for analysis, blunder check, hints and play-mode engine moves, with silent main-thread fallback. Build scripts (`build_wasm.bat`/`.sh`) now carry the full 75-export list plus `UTF8ToString`/`stringToNewUTF8` runtime methods and `ENVIRONMENT=web,worker` (both scripts verified identical); `build_standalone.py` inlines `board.js` between loader and app. Fixed an early-fire bug: `drawEvalGraph` ran from a resize observer before WASM boot finished (guarded with `has('qr_history_len')`). Verified with headless Chrome over http and file://: clean boot, NNUE + hybrid MCTS active, worker round-trip 12-ply PV ~240 ms, full human→engine UI exchange ~300 ms, zero page errors.

- **2026-08-15**: Tested progressive widening at 150ms/move (1000 games): -6.9 ±20.7 Elo; kept experimental and default disabled.
- **2026-08-15**: Promoted `BackupMode::AvgBlend` to production default across engine and tools.
- **2026-08-15**: Fixed arena NPS reporting to track expanded MCTS nodes during hybrid search.
- **2026-08-14**: Integrated `nnueEvalInt` direct leaf evaluation at `leafDepth=0` and connected shared `PlayerPathCacheTable` to MCTS accumulator building.
- **2026-08-13**: Promoted Hybrid MCTS (`mcab.hpp`) to production default (+46.9 ±23.5 Elo over pure alpha-beta at 200ms).
