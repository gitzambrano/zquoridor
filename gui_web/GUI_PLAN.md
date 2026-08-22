# Zquoridor GUI v4 — Specification / Contract

This is the single source of truth for the new web GUI. Four workstreams own
disjoint files and code against the contracts below. **Do not edit files you
do not own.** If a contract here is wrong or impossible, say so in your final
report instead of silently deviating.

## File ownership

| Owner | Files |
| --- | --- |
| W1 (engine) | `gui_web/engine_wasm.cpp`, `build/build_wasm.bat`, `build/build_wasm.sh`, `gui_web/build_standalone.py`, additive const accessors in `src/mcab.hpp` |
| W2 (shell)  | `gui_web/style.html` (markup + all CSS) |
| W3 (board)  | `gui_web/board.js` |
| W4 (app)    | `gui_web/app.js` |

Build output (`zquoridor.js/.wasm/.data/.html`, root `index.html`) is generated,
never hand-edited.

## Vocabulary / coordinates

- Board 9x9. `cell = r*9 + c`, engine row 0 = TOP. Wall slots 8x8, `slot = r*8 + c`,
  `r` = corridor between engine rows r and r+1, `c` = corridor between cols c and c+1.
- Player 0 starts at `cellIdx(0,4)` (e9 in notation) and must reach engine row 8.
- Player 1 starts at `cellIdx(8,4)` (e1) and must reach engine row 0.
- Algebraic notation (fixed board frame, never flipped): column `a..i` left-to-right,
  rank `1..9` bottom-to-top, i.e. `rank = 9 - engine_row`.
  - Pawn move: `e5`
  - Wall move: `<col><rank><h|v>` where rank = `8 - r` (south side of the corridor).
    e.g. orientation=0, r=3, c=2 gives `c5h`.
- Eval sign: engine scores are **mover-relative** (positive = side to move is better).
  Display converts to player-0 perspective. `evalToWhitePercent(v) = 100/(1+exp(-v/200))`.

## QFEN (position serialization)

Single-line, space separated, 6 fields:

```
<pawn0> <pawn1> <wallsLeft0> <wallsLeft1> <turn> <walls>
```

- `pawn0`, `pawn1`: algebraic cell (`e9`, `e1`).
- `wallsLeft0/1`: integers 0..10.
- `turn`: `0` or `1`.
- `walls`: comma-separated wall notations (`c5h,f4v`) sorted ascending by
  `(orientation, r, c)`, or `-` when the board is empty.

Initial position: `e9 e1 10 10 0 -`

Round-trip must be exact. The parser is tolerant of extra whitespace and of the
walls field being omitted entirely (treated as `-`).

## Game text (PGN-analogue)

Plain text, one token per ply, numbered pairs:

```
1. e8 e2 2. e7 c5h 3. ...
```

Optional leading header lines of the form `[Key "value"]` are accepted and
ignored on import, except `[QFEN "..."]`, which sets the start position.

---

# W1 — WASM engine layer (`engine_wasm.cpp`)

Rewrite as a game object with a **move history and a cursor**, so navigation,
undo/redo, analysis-without-playing, and a position editor all work.

Internal model:
- `rootState` — start position of the current game, normally `initialState()`.
- `history` — vector<Move>, all plies from rootState.
- `cursor` — int, 0..history.size(); the displayed position is rootState with
  the first `cursor` moves applied.
- `cur` — cached State at `cursor`, `curMoves` its legal moves, `repTable`
  rebuilt for the plies up to `cursor`.

Every existing exported function keeps its name and meaning, now operating on
the position at `cursor`. New functions are listed below.

## Exports (complete list — keep `EXPORTED_FUNCTIONS` in both build scripts in sync)

### Lifecycle / engine config (existing)
```
void qr_new_game()
int  qr_load_nnue_weights(const char* path)
void qr_set_eval_heuristic()
int  qr_eval_mode_is_nnue()
void qr_set_mcab_enabled(int on)
int  qr_mcab_active()
```

### Position query (existing, now cursor-relative)
```
int qr_turn(), qr_winner(), qr_is_draw()
int qr_pawn(int p), qr_walls_left(int p)
int qr_wall_h_bit(int slot), qr_wall_v_bit(int slot)
int qr_dist_to_goal(int p)
```

### Position query (new)
```
int qr_static_eval()              // eval of current position, mover-relative int score
                                  // (NNUE via nnueEvalInt in NNUE mode, else evalSimple)
int qr_path_len(int p)            // same as qr_dist_to_goal, kept for symmetry
int qr_path_cell(int p, int i)    // i-th cell of one shortest path for player p
                                  // (i=0 is the pawn cell). -1 when i is past the end.
int qr_is_wall_legal(int o,int r,int c)  // legality for the side to move
int qr_wall_owner(int o,int r,int c)     // 0/1 = who placed it, -1 unknown/absent
                                         // (derived by replaying history; -1 for
                                         //  walls that came from a QFEN or the editor)
```

`qr_path_cell` builds a shortest path by BFS from the pawn to the goal row and
walks predecessors. Cache it per (position, player) so calling it 20 times in a
render loop is cheap.

### Legal moves (existing)
```
int qr_legal_moves_count()
int qr_legal_move_is_wall(int i), qr_legal_move_a(int i), qr_legal_move_b(int i), qr_legal_move_c(int i)
```

### Playing (existing, with one semantic change)
```
int qr_apply_pawn_move(int destCell)
int qr_apply_wall_move(int o,int r,int c)
int qr_engine_move(int maxDepth,int timeMs)
int qr_last_move_is_wall(), qr_last_move_a(), qr_last_move_b(), qr_last_move_c()
int qr_last_move_eval()
```
Applying a move when `cursor < history.size()` **truncates** the future first
(standard GUI behaviour), then appends.

`qr_engine_move` also records its search stats, readable through the `qr_an_*`
accessors below (`qr_an_line_count()` is >= 1 after it returns), so the GUI can
show what the engine was thinking without running a second search.

### History / navigation (new)
```
int  qr_history_len()          // plies recorded
int  qr_history_cursor()
int  qr_goto_ply(int ply)      // clamps to [0,len]; returns the resulting cursor
int  qr_undo()                 // goto(cursor-1); returns the new cursor
int  qr_redo()                 // goto(cursor+1); returns the new cursor
void qr_truncate_here()        // drop plies after the cursor
int  qr_hist_move_is_wall(int ply)   // the move leading from `ply` to `ply+1`
int  qr_hist_move_a(int ply), qr_hist_move_b(int ply), qr_hist_move_c(int ply)
int  qr_hist_mover(int ply)          // 0 or 1
```

### Analysis (new)
```
int qr_analyze(int maxDepth, int timeMs, int multipv)
```
Searches the **current position without applying anything**. Returns the number
of lines produced. Results are read with:
```
int qr_an_line_count()
int qr_an_line_score(int i)     // mover-relative int score for line i
int qr_an_line_visits(int i)    // MCTS visits (0 when running pure alpha-beta)
int qr_an_line_len(int i)       // number of plies in the PV
int qr_an_line_move_is_wall(int i,int j)
int qr_an_line_move_a(int i,int j), qr_an_line_move_b(int i,int j), qr_an_line_move_c(int i,int j)
int qr_an_nodes()               // nodes searched in the last qr_analyze
int qr_an_depth()               // reached depth
int qr_an_is_mcab()             // 1 if the last analysis used the hybrid
```
Line 0 is always the best move. Lines are sorted best-first.

Implementation:
- **Hybrid path (NNUE + mcab enabled)** — run `MCABSearch::chooseMoveMCAB` on the
  current position, then read the root node via `rootNodeForInspection()`: sort
  root edges by visit count `N` descending and take the top `multipv`. Score per
  line is `W/N` converted back to engine score units (invert `mcab::scoreToQ`,
  using `params.scoreScale`). The PV for a line is built by walking the child
  subtree picking the max-`N` edge at each level, up to 12 plies or until a node
  has no expanded child. This needs read-only access to the node pool: add to
  `MCABSearch` (additive, const, no behaviour change)
  ```cpp
  const NodeT* nodeAtForInspection(int idx) const;  // nullptr when out of range
  ```
  and expose it through `McabRunner` guarded by `if constexpr (supported)`.
  Keep tests and benchmarks compiling.
- **Alpha-beta path** — root split: for each root move, apply it, call
  `searchShallow`/`searchLeaf` at the requested depth, negate, sort, keep the top
  `multipv`. The PV for the best line is walked from the transposition table
  (probe, take best move, apply, repeat; max 12 plies, stop on a cycle). If a TT
  walk proves impractical, a 1-ply PV is acceptable — say so in your report.

`multipv` is clamped to `[1,5]`. `timeMs` is a hard budget: the function must
return within roughly that budget, because it blocks the main thread and the app
calls it in small slices.

### Position editor (new)

An independent scratch `State` that never touches the live game until commit.
```
void qr_edit_begin()                       // copy the current position into the buffer
void qr_edit_clear()                       // empty board, pawns at start cells, 10/10, turn 0
int  qr_edit_set_pawn(int p,int cell)      // 1 ok; 0 rejected (occupied by the other pawn)
int  qr_edit_toggle_wall(int o,int r,int c)// 1 placed, 0 removed, -1 rejected (overlap/crossing)
void qr_edit_set_walls_left(int p,int n)   // clamped 0..10
void qr_edit_set_turn(int t)
int  qr_edit_pawn(int p), qr_edit_walls_left(int p), qr_edit_turn()
int  qr_edit_wall_h_bit(int slot), qr_edit_wall_v_bit(int slot)
int  qr_edit_validate()                    // 0 = legal; otherwise a bitmask:
                                           //   1 = player 0 has no path to goal
                                           //   2 = player 1 has no path to goal
                                           //   4 = pawns on the same cell
                                           //   8 = more than 20 walls placed
                                           //  16 = a pawn already stands on its goal row
int  qr_edit_commit()                      // 0 when validate() != 0; otherwise adopts
                                           // the buffer as the new root, clears history,
                                           // resets cursor/TT/MCTS tree, returns 1
```
`qr_edit_toggle_wall` deliberately allows placements that block a player —
`qr_edit_validate` reports that so the UI can show it, and `qr_edit_commit`
refuses it.

### Serialization (new)

Strings are returned as `const char*` into a `static std::string` held by the
module (JS reads it with `UTF8ToString`); the pointer stays valid until the next
call to the same function.
```
const char* qr_get_qfen()          // QFEN of the position at the cursor
int         qr_set_qfen(const char* s)   // 1 ok, 0 parse error or illegal position.
                                         // On success: new root, history cleared.
const char* qr_get_game_text()     // numbered move list of the whole game, prefixed
                                   // with [QFEN "..."] when the root is not initial
int         qr_set_game_text(const char* s)  // 1 ok, 0 parse error
const char* qr_edit_get_qfen()     // QFEN of the edit buffer
int         qr_edit_set_qfen(const char* s)  // load into the edit buffer
const char* qr_move_notation(int isWall,int a,int b,int c)  // "e5" or "c5h"
```

Also export `EXPORTED_RUNTIME_METHODS=["ccall","cwrap","UTF8ToString","stringToNewUTF8"]`.

## W1 build-script work

- `build/build_wasm.bat` and `build/build_wasm.sh` must list every export above,
  identically, and keep `-msimd128 -O3`.
- `gui_web/build_standalone.py` must inline **`zquoridor.js`, then `board.js`,
  then `app.js`**, in that order, instead of only `app.js`, and rewrite the
  `<script src=...>` tags in `style.html`. Make the tag-stripping tolerant:
  match any `<script src="X.js"></script>` where X is one of
  `zquoridor|board|app`, in any order, and inject at the position of the first.
- Verify: run the emsdk build. `C:\emsdk\emsdk_env.bat` activates it. On a clean
  build both `zquoridor.js` and `zquoridor.wasm` regenerate and
  `python build_standalone.py` succeeds.
- The C++ must also compile without emscripten as a sanity check:
  `g++ -O2 -std=c++17 -Isrc -fsyntax-only gui_web/engine_wasm.cpp`
- If you touch `src/mcab.hpp`, re-run `build\build_tests.bat` plus
  `bin\test_mcab_core.exe`, `bin\test_mcab_dispatch.exe`,
  `bin\test_mcab_phase9.exe`, and report their output.

---

# W2 — Shell and design system (`style.html`)

One file: `<head>` + all CSS + all markup + the three `<script src>` tags at the
end (`zquoridor.js`, `board.js`, `app.js`, in that order).

## Design language

Mirror the Zchezz identity (`C:\Zchezz\index.html`, read its `<style>` block)
but adapted to Quoridor — dark, gold-accented, serif display face over a
monospace UI:

- Fonts: `Cinzel` (display, headings, logo) and `JetBrains Mono` (everything
  else), from Google Fonts with real fallbacks (`serif` / `monospace`).
- Base tokens (dark, default):
  ```
  --bg:#0d0d12  --surf:#16161f  --surf2:#1c1c28  --bor:#252538
  --gold:#c8a84b --gold2:#e6c96e --txt:#e0e0f0 --muted:#565672
  --green:#4a9c6a --red:#c0394a --blue:#4a7cc0
  ```
- Add a **light theme** under `[data-ui-theme="light"]` on `<html>` that
  redefines the same tokens. Every colour must come from a token — no hardcoded
  hex outside the token blocks.
- Board colours belong to W3 (canvas) and are passed as JS objects; CSS only
  styles board-adjacent chrome. Still declare the six board palettes as CSS
  custom properties on `[data-board-theme="..."]` so the settings previews can
  render swatches without touching JS.
- Player identity: player 0 = light/gold (`--p0`), player 1 = crimson (`--p1`).
  Define `--p0`, `--p0-deep`, `--p0-light`, `--p1`, `--p1-deep`, `--p1-light`.

## Layout

`#app` is a column: `#header`, then exactly one visible `.panel`.

Desktop (>= 900px): each panel is a two-column grid — a board column (board,
player bars, eval bar) and a fixed ~320px side column. Mobile (< 900px): single
column, side column below the board, board sized to `min(92vw, ...)`. Nothing
may scroll the page horizontally. Use `100dvh`.

## Required element IDs (the JS contract)

### Header
```
#header  #logo  #tabs
#tab-play  #tab-analysis  #tab-editor      (class .tab, .tab.active)
#btn-settings  #btn-about
```

### Play panel — `#play-panel` (`.panel`)
```
#pl-board-col
  .pbar#pl-bar-top     > .pdot#pl-dot-top  .pname#pl-name-top
                         .pwalls#pl-walls-top   (wall pips container)
                         .pdist#pl-dist-top     .pclock#pl-clock-top
  #pl-board-wrap       > canvas#pl-board  +  #pl-evalbar (vertical)
                         #pl-evalbar-fill  #pl-evalbar-num
  .pbar#pl-bar-bot     > (same ids with -bot)
#pl-side-col
  #pl-status      (status text)   #pl-status-dot
  #pl-controls    buttons: #pl-btn-new #pl-btn-flip #pl-btn-side #pl-btn-undo
                  #pl-btn-redo #pl-btn-hint #pl-btn-takeback
                  selects: #pl-sel-mode (depth|time|game) #pl-sel-depth #pl-sel-time
                           #pl-sel-tc  (time control, hidden unless mode=game)
                  #pl-wall-h  #pl-wall-v   (wall orientation toggle buttons)
  #pl-movelog-wrap > #pl-movelog
  #pl-engine-info
```

### Analysis panel — `#analysis-panel` (`.panel`)
```
#an-toolbar
  #an-eng-btn  #an-first #an-prev #an-next #an-last  #an-flip
  #an-sel-depth  #an-sel-time  #an-pv-plus #an-pv-minus #an-pv-count
  #an-btn-qfen  #an-btn-game  #an-btn-blunder  #an-btn-arrows
#an-content
  #an-board-col > canvas#an-board  #an-evalbar #an-evalbar-fill #an-evalbar-num
                  #an-lines  (contains .eng-line#an-line1..5 with
                              .line-rank, .line-score#an-ls1..5,
                              .line-moves#an-lm1..5, .line-visits#an-lv1..5)
                  #an-status
  #an-side-col  > #an-movelog-wrap > #an-movelog
                  #an-graph-wrap  > canvas#an-graph
                  #an-summary   (accuracy / blunder counts)
```

### Editor panel — `#editor-panel` (`.panel`)
```
#ed-toolbar
  #ed-tool-p0 #ed-tool-p1 #ed-tool-wallh #ed-tool-wallv #ed-tool-erase
    (class .ed-tool, .ed-tool.active)
  #ed-btn-clear #ed-btn-reset #ed-btn-flip
#ed-content
  #ed-board-col > canvas#ed-board
  #ed-side-col
    #ed-turn-p0 #ed-turn-p1        (side-to-move radio pair)
    #ed-walls0-dec #ed-walls0-val #ed-walls0-inc
    #ed-walls1-dec #ed-walls1-val #ed-walls1-inc
    #ed-qfen        (textarea)
    #ed-btn-qfen-load #ed-btn-qfen-copy
    #ed-validity    (message area; .ok / .bad modifier classes)
    #ed-btn-play  #ed-btn-analyze
```

### Modals
```
#settings-modal (#settings-box)
  #cfg-board-themes   (six .cfg-opt with data-theme, each holding a swatch)
  #cfg-pawn-styles    (four .cfg-opt with data-style: disc|pin|chess|glyph)
  #cfg-ui-theme       (.cfg-opt data-ui=dark|light)
  toggles (checkbox inputs inside .cfg-sw):
    #cfg-highlight #cfg-paths #cfg-coords #cfg-dots #cfg-anim
    #cfg-sound #cfg-evalbar #cfg-movelog-eval #cfg-nnue #cfg-mcab
  #cfg-sound-vol (select 0..5)
  #cfg-done
#about-modal (#about-box) with #about-close
#text-modal  (#text-modal-title, textarea#text-modal-area,
              #text-modal-load, #text-modal-copy, #text-modal-close)
#toast       (transient message strip)
#loading-overlay (#loading-text)
```

Give every button a `title` and an `aria-label`. Buttons use `.btn`, with
`.btn.primary` and `.btn.gld` variants. `.panel { display:none }`,
`.panel.active { display:flex }`. Move-log rows in `#pl-movelog` / `#an-movelog`
use classes `.ml-row`, `.ml-num`, `.ml-ply` (with `.ml-ply.current`,
`.ml-ply.p0`, `.ml-ply.p1`), `.ml-eval`, and glyph classes `.g-best`, `.g-good`,
`.g-inacc`, `.g-mistake`, `.g-blunder`.

Markup must be static — no JS-generated structure for the shell. W4 only fills
text and attributes and appends move rows.

## W2 quality bar

- No layout jumps when panels switch; no horizontal scrollbars at 360px width.
- Respect `prefers-reduced-motion`.
- Focus-visible outlines on every interactive element.
- Contrast: body text at least 4.5:1 against its background in both UI themes.

---

# W3 — Board renderer (`board.js`)

A single canvas-based, theme-able, animated Quoridor board. No engine access and
no DOM outside its own canvas — pure view plus input. Exposes one global class.

```js
class QBoard {
  constructor(canvas, opts = {})     // opts: { mode, theme, pawnStyle, flipped }
  destroy()

  // --- configuration ---
  setMode(mode)          // 'play' | 'analysis' | 'edit'
  setTheme(name)         // one of the six board themes
  setPawnStyle(name)     // 'disc' | 'pin' | 'chess' | 'glyph'
  setFlipped(bool)       // flip mirrors ROWS only; columns never change,
                         // so H walls never become V walls
  setOptions({ showCoords, showPaths, showDots, highlightLast, animate })
  setInteractive(bool)
  setEditTool(tool)      // 'p0' | 'p1' | 'wallh' | 'wallv' | 'erase'
  setWallOrientation(o)  // 0 = H, 1 = V — the orientation armed for placement

  // --- data in ---
  setPosition({ pawns:[c0,c1], wallsH:Uint8Array(64), wallsV:Uint8Array(64),
                wallOwner:{h:Int8Array(64), v:Int8Array(64)},
                turn, wallsLeft:[n0,n1], winner })
  setLegal({ pawnCells:[int], wallH:[slot], wallV:[slot] })
  setOverlays({ lastMove, hint, arrows, ghostWall, paths })
     // lastMove/hint: {isWall,a,b,c} or null
     // arrows: [{ from:cell, to:cell, color, width }] for pawn moves and
     //         [{ wall:{o,r,c}, color }] for wall suggestions
     // paths: [[cells...], [cells...]] shortest path per player, or null
  resize()               // re-measure the container and redraw
  render()

  // --- animation ---
  animateMove(move, mover, done)   // slides a pawn or fades a wall in

  // --- callbacks (assign directly) ---
  onPawnMove = (cell) => {}
  onWallPlace = (o, r, c) => {}
  onWallRemove = (o, r, c) => {}     // edit mode only
  onPawnPlace = (player, cell) => {} // edit mode only
  onHoverWall = (o, r, c, legal) => {}   // null args when leaving
  onWallOrientationChange = (o) => {}    // fired when the user flips with a gesture
}
```

## Rendering requirements

- Device-pixel-ratio aware, crisp at any DPR, `touch-action:none`.
- Cell grid with a subtle bevel / inner shadow, per-theme wood grain or flat fill.
- Coordinates a..i and 1..9 drawn in the margin when `showCoords`.
- Pawns: four styles, drawn procedurally (no image assets), with a soft drop
  shadow and a ring in the player colour. The `glyph` style renders a Unicode
  meeple or pawn character.
- Walls: rounded 3D slabs spanning two cells plus the gap, wood/ivory tone with a
  top highlight and a cast shadow. Tinted by owner when `wallOwner` is known.
- Legal pawn destinations: translucent dots, or a ring when the target is a
  jump/diagonal square.
- Wall ghost: follows the pointer while placing; green when legal, red when not,
  snapped to the nearest slot. Dragging from anywhere on the board works — the
  slot is chosen by nearest-corridor distance, and dragging past the halfway
  point of a cell in the perpendicular direction flips the orientation and fires
  `onWallOrientationChange`.
- Last-move highlight: pawn source and destination squares tinted; a wall pulses
  once.
- Shortest-path overlay: a thin translucent polyline per player from the pawn to
  the goal row, drawn under the pawns.
- Arrows: lichess-style tapered arrows with a rounded head, drawn over
  everything, semi-transparent.
- Win state: the winner's goal row glows.
- Animations run on `requestAnimationFrame`, and are skipped when
  `animate === false` or `prefers-reduced-motion` is set. Never leave a rAF loop
  running when nothing is animating.

## Board themes (define all six in `board.js` and export the table as `QBoard.THEMES`)

`wood` (default), `classic`, `emerald`, `ocean`, `coral`, `night`. Each theme is
an object with keys `cellLight`, `cellDark`, `boardBg`, `gridLine`, `wall`,
`wallEdge`, `wallShadow`, `coord`, `dot`, `lastMove`, `pathP0`, `pathP1`, `p0`,
`p0Edge`, `p1`, `p1Edge`, `goalGlow`.

## Input

- Click a legal destination cell gives `onPawnMove`.
- Click/tap and drag near a corridor gives the wall ghost; releasing on a legal
  slot gives `onWallPlace`. Releasing on an illegal slot cancels silently.
- Keyboard: when the canvas has focus, arrow keys move a selection cursor, Enter
  commits a pawn move, `H`/`V` arm the wall orientation, Escape cancels.
- Edit mode: left click applies the active tool, right click erases.
- Pointer events only (`pointerdown/move/up`), so mouse and touch share one path.

---

# W4 — Application (`app.js`)

Owns everything else: WASM boot, game loop, analysis driver, editor logic,
settings, persistence, sounds, keyboard shortcuts, move log, eval graph.

## Boot

```js
ZquoridorModule().then((Module) => { ... })
```
Keep this exact call shape — `build_standalone.py` rewrites it to inject the
embedded wasm bytes. Show `#loading-overlay` until ready. After boot:
1. try `qr_load_nnue_weights("/data/nnue/nnue_weights_int8.bin")`; on success set
   the engine-info line to `NNUE`, on failure to `Heuristic` and disable the
   `#cfg-nnue` and `#cfg-mcab` toggles with an explanatory `title`.
2. restore settings from `localStorage['zq.settings']`.
3. `newGame()`.

## Play mode

- Human versus engine, side selectable; the board auto-flips so the human is at
  the bottom. Engine moves run in a `setTimeout(..., 0)` slice with a `gameGen`
  guard, so switching sides or starting a new game cancels a pending engine move.
- Strength: `#pl-sel-mode` chooses depth-capped or time-capped search;
  `#pl-sel-depth` (2..12) or `#pl-sel-time` (100ms..10s). The `game` mode uses
  `#pl-sel-tc` time controls (`base|increment`, e.g. `300000|3000`) and drives
  the two `.pclock` displays with a 100ms ticker; flag-fall ends the game.
- Wall placement uses the board ghost/drag flow; `#pl-wall-h` and `#pl-wall-v`
  arm the orientation and stay in sync with `onWallOrientationChange`.
- `#pl-btn-hint` runs a short `qr_analyze` and shows the best move as an arrow or
  wall ghost without playing it.
- `#pl-btn-undo` takes back one ply; `#pl-btn-takeback` undoes two, back to the
  human's turn.
- Move log: one row per move pair, `1. e8  e2`, current ply highlighted; clicking
  a ply navigates via `qr_goto_ply` and switches to browsing while keeping the
  game intact. Per-ply eval is shown when `#cfg-movelog-eval` is on.
- Eval bar: vertical, player-0 share on top, animated transition, numeric
  percentage below. Driven by `qr_static_eval` after every ply and by the live
  analysis score while analysis is running.
- Sounds (WebAudio, synthesized, no asset files): pawn move (short click), wall
  place (wooden thunk), jump (soft blip), game end (chord), illegal (low buzz).
  Volume 0..5 from settings; 0 disables.
- The status line reports whose turn it is, near-goal states (`player 1 is 3
  steps from goal`), draw by threefold repetition, and the winner.

## Analysis mode

- Its own board instance bound to the same underlying game; navigation is shared
  with play mode through the WASM cursor.
- `#an-eng-btn` toggles a continuous analysis loop: repeatedly call
  `qr_analyze(depth, sliceMs, multipv)` in `setTimeout` slices of at most 250ms
  so the UI stays responsive, updating lines, eval bar and arrows after each
  slice. Stop on navigation, tab switch, or when the toggle is turned off.
- Multi-PV 1..5 via `#an-pv-plus` / `#an-pv-minus`; hidden line rows get
  `display:none`.
- Each line shows rank, score (both a `+0.42`-style figure and a win%), the visit
  count when the hybrid is active, and the PV in algebraic notation. Hovering a
  line draws its first move as an arrow or ghost on the board; clicking a line
  plays its first move as a temporary variation.
- `#an-btn-arrows` toggles best-move arrows.
- `#an-btn-blunder` ("Analyze game") walks every ply with a short search, records
  the eval before and after, classifies each move, and writes glyphs into the
  move log. Classification by win-probability drop in percentage points: under 2
  is best/good, 2 to 5 is an inaccuracy `?!`, 5 to 12 is a mistake `?`, above 12
  is a blunder `??`; a move matching the engine's top choice gets `!`. Show
  progress and allow cancelling. Fill `#an-summary` with per-player counts and an
  accuracy percentage.
- Eval graph: canvas line/area chart of win% over plies, player 0 above the
  midline, blunders marked with dots, current ply marked, click to navigate,
  responsive to container width, redrawn on resize and theme change.
- `#an-btn-qfen` and `#an-btn-game` open `#text-modal` for loading and copying
  QFEN and game text respectively.

## Editor mode

- Its own board instance in `edit` mode, bound to the WASM edit buffer.
- The tool palette drives `setEditTool`. Placing a pawn moves that player's pawn;
  wall tools toggle; the eraser removes walls and nothing else.
- Wall-count steppers and the side-to-move pair write straight into the buffer.
- `#ed-validity` shows the decoded `qr_edit_validate()` bitmask in plain English
  and disables `#ed-btn-play` and `#ed-btn-analyze` while invalid.
- `#ed-qfen` binds both ways: editing the board rewrites the textarea, and
  `#ed-btn-qfen-load` parses the textarea into the buffer.
- `#ed-btn-play` commits and switches to the play panel; `#ed-btn-analyze`
  commits and switches to analysis.

## Settings and persistence

Persist to `localStorage['zq.settings']` as one JSON object and apply on load:
board theme, pawn style, UI theme, all the `#cfg-*` toggles, sound volume, search
mode/depth/time/time-control, human side, flip state. Provide sane defaults for a
first visit and never throw on a corrupt or absent value.

## Keyboard shortcuts

Left/Right arrow for previous/next ply, Home/End for first/last, `f` flip, `n`
new game, `h` hint, `a` toggle the analysis engine, `1`/`2`/`3` switch tabs, `e`
toggle the editor, Escape closes a modal or cancels wall placement, `?` shows the
shortcut list in the about modal.

## Robustness

- Every WASM call goes through a thin wrapper that logs errors and surfaces them
  through `#toast` instead of throwing into the void.
- Guard against re-entrancy: a `busy` flag blocks input while the engine thinks.
- The app must work from `file://` with no server.
