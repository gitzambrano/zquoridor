# Zquoridor — GUI Premium: Enhancement Plan

> Detailed design + implementation contract for the next generation of `gui_web/`.
> Written in English per `CLAUDE.md`. Companion documents: `readme.md` (feature-level),
> `status.md` (technical reference / changelog).
>
> **Baseline today**: `gui_web/style.html` (377 lines, DOM grid board, mobile-only single
> column, 2 piece themes, 4 buttons, no clocks, no analysis) + `gui_web/app.js` (638 lines).
> **Target**: a premium, dual-form-factor interface at the Zchezz standard, with Quoridor's
> own interaction problem — *wall placement* — solved to the point of being effortless.

---

## 0. Scope & Design Principles

### 0.1 Non-negotiable product requirements

| # | Requirement | How it drives the design |
| - | ----------- | ------------------------ |
| R1 | **Works on mobile and desktop** | One codebase, two layouts (`>=900px` = 2-column, `<900px` = stacked). Pointer Events only — mouse, pen and touch share one path. `100dvh`, `env(safe-area-inset-*)`. |
| R2 | **Wall placement must be easy and intuitive** | Four independent input methods (§6), magnetic snapping with legality assist, orientation-flip gesture, optional confirm step, finger-occlusion offset. |
| R3 | **Wall buttons must be within easy reach** | The wall dock lives at the **bottom** of the mobile layout, inside the thumb arc, with H on the left edge and V on the right edge so either thumb can arm a wall without a hand shift. Never in the header. |
| R4 | **Clock, player names and remaining walls always glanceable** | Two persistent HUD cards flanking the board (opponent above, you below), never collapsed, never scrolled away. Clock in `JetBrains Mono` tabular numerals at >=1.35rem; walls as *both* a numeral and 10 pips; both re-render on every state change. |
| R5 | **Board beautiful but functional** | Depth via bevel/shadow, not via texture noise. Contrast between cell, groove and wall is the first constraint; decoration is applied only after that contrast passes §10. |

### 0.2 Principles

1. **The board is the app.** Every layout gives the board the largest square it can have; chrome shrinks first.
2. **No mode you can get stuck in.** Every armed state (wall pending, edit pending) shows an explicit escape affordance and is cleared by `Esc`, by an outside tap, or by 6 s of inactivity.
3. **Never punish a near-miss.** A pointer near an illegal wall snaps to the nearest legal one when one exists within the assist radius; only a genuinely intentional illegal drop shows the red ghost.
4. **Same information, two densities.** Mobile hides *presentation*, never *state*: the eval bar becomes a 6 px strip, the move log becomes a sheet — but nothing that R4 covers is ever hidden.
5. **Every action has three channels**: visual (motion), audio (synth), haptic (`navigator.vibrate`). All three individually toggleable.
6. **Degrade silently.** No worker → main-thread slicing. No WASM → static board with an error toast. No `vibrate` → no-op.

---

## 1. Design Language

### 1.1 Typography

| Role | Family | Weight / size | Usage |
| ---- | ------ | ------------- | ----- |
| Display | `Cinzel`, serif | 600–700, `.62–1.05rem`, `letter-spacing .06em` | Logo, panel titles, level names, modal headings, status line |
| UI / data | `JetBrains Mono`, monospace | 300–600, `.58–.86rem` | Everything else: buttons, move log, clocks, evals, coordinates |
| Numerals | `JetBrains Mono` + `font-variant-numeric: tabular-nums` | — | Clocks, wall counts, distances, eval, node counts |

Fonts are `<link>`ed from Google Fonts in the dev build and **inlined as base64 WOFF2** by
`build_standalone.py` for the `file://` bundle (the standalone HTML must never require network).
Fallback stack: `Cinzel, 'Times New Roman', serif` / `'JetBrains Mono', ui-monospace, Consolas, monospace`.

### 1.2 Core tokens — dark UI theme (default)

```css
:root {
  /* surfaces */
  --bg:      #0d0d12;   /* page */
  --surf:    #16161f;   /* cards, header, panels */
  --surf2:   #1c1c28;   /* raised: buttons, rows */
  --surf3:   #23233a;   /* hover / pressed */
  --bor:     #252538;   /* hairlines */
  --bor2:    #33334a;   /* emphasized borders */
  /* identity */
  --gold:    #c8a84b;   /* primary accent, titles, active borders */
  --gold2:   #e6c96e;   /* highlight, hover text */
  --gold-dim: rgba(200,168,75,.16);
  --gold-glow: rgba(200,168,75,.35);
  /* text */
  --txt:     #e0e0f0;
  --txt2:    #9a9ab8;
  --muted:   #565672;
  /* semantics */
  --green:   #4a9c6a;  --green-dim: rgba(74,156,106,.18);
  --red:     #c0394a;  --red-dim:   rgba(192,57,74,.18);
  --amber:   #d99b3c;
  --blue:    #4a7bc0;
  /* players (overridden per board theme) */
  --p0: #f0e6d2; --p0-deep: #b9a87e; --p0-light: #fdf8ec; --p0-soft: rgba(240,230,210,.16);
  --p1: #cf4155; --p1-deep: #7d1f2c; --p1-light: #e8697a; --p1-soft: rgba(207,65,85,.18);
  /* geometry */
  --r-sm: 6px; --r-md: 10px; --r-lg: 14px; --r-xl: 18px;
  --sp: 4px;              /* spacing unit; all gaps are multiples */
  --dur-fast: 110ms; --dur: 180ms; --dur-slow: 320ms;
  --ease: cubic-bezier(.22,.61,.36,1);
}
```

### 1.3 Light UI theme (`[data-ui="light"]`)

```css
--bg:#f2efe6; --surf:#ffffff; --surf2:#eae6d9; --surf3:#ded8c6;
--bor:#d6cfbc; --bor2:#bdb49c;
--gold:#8a6d22; --gold2:#a8842c; --gold-dim:rgba(138,109,34,.14); --gold-glow:rgba(138,109,34,.28);
--txt:#1c1b18; --txt2:#55524a; --muted:#8b8779;
--green:#2f7a4c; --red:#a52c3c;
--p0:#c9a227; --p0-deep:#8a6c12; --p0-light:#e8cf74;   /* a light board needs a darker "light" pawn */
--p1:#a52c3c; --p1-deep:#6d1a25; --p1-light:#cf5a68;
```

`data-ui` is `dark | light | auto`; `auto` follows `prefers-color-scheme` live through a
`matchMedia` listener.

### 1.4 Elevation & surfaces

| Level | Use | CSS |
| ----- | --- | --- |
| E0 | page | `background: var(--bg)` |
| E1 | panel / HUD card | `linear-gradient(180deg,var(--surf2),var(--surf))`, `1px solid var(--bor)`, `inset 0 1px 0 rgba(255,255,255,.03)` |
| E2 | button, row | `var(--surf2)` + `1px solid var(--bor)`; hover → `--surf3`, `border-color: var(--bor2)` |
| E3 | armed / active | `border-color: var(--gold)`, `box-shadow: 0 0 0 1px var(--gold), inset 0 0 12px var(--gold-dim)` |
| E4 | board | `box-shadow: 0 10px 34px rgba(0,0,0,.55), 0 0 0 1px var(--bor2)` + 2 px gold hairline frame |
| E5 | modal / sheet | `box-shadow: 0 24px 60px rgba(0,0,0,.62)`, overlay `rgba(0,0,0,.62)` + `backdrop-filter: blur(3px)` |

### 1.5 Motion

| Token | ms | Curve | Used by |
| ----- | -- | ----- | ------- |
| `--dur-fast` | 110 | `--ease` | button press, ghost snap, hover |
| `--dur` | 180 | `--ease` | pawn slide, panel switch, wall fade-in |
| `--dur-slow` | 320 | `--ease` | modal in/out, board flip, goal glow pulse |

Everything is gated by `@media (prefers-reduced-motion: reduce)` → durations collapse to `0ms`,
pulses and the eval-driven glow become static.

### 1.6 Iconography

Inline SVG only (no icon font, no external asset), 20×20 viewBox, `stroke: currentColor`,
`stroke-width: 1.6`, round caps. Set: `wall-h`, `wall-v`, `flip`, `undo`, `redo`, `first`, `last`,
`hint`, `engine`, `settings`, `info`, `copy`, `paste`, `graph`, `edit`, `play`, `sound-on`,
`sound-off`, `check`, `close`, `menu`, `path`, `clock`.

---

## 2. Board Themes & Pawn Styles

### 2.1 Board themes (6) — `data-board="…"`

Each theme sets the seven board variables. Cells are two-tone (`--cell-a` on `(r+c)` even) at a
very low delta — enough to read the grid without a checkerboard look, which Quoridor does not have.

| # | Key | Name | `--cell-a` / `--cell-b` | `--groove` | `--frame` | `--wall` → `--wall-edge` | Notes |
| - | --- | ---- | ----------------------- | ---------- | --------- | ------------------------ | ----- |
| 1 | `obsidian` | **Obsidian & Gold** *(default)* | `#1b1b26` / `#20202d` | `#0b0b12` | `#2a2a3c` + gold hairline | `#e2cd96` → `#8d6f31` | House identity; walls read as gilded beams |
| 2 | `walnut` | **Walnut** | `#5a4432` / `#4e3b2b` | `#2a1e16` | `#3a2a1e` | `#efe2c2` → `#9a8256` | Warm wooden board, ivory walls |
| 3 | `ivory` | **Ivory Classic** | `#efe7d6` / `#e6dcc7` | `#c3b699` | `#b6a684` | `#6f512f` → `#3d2c18` | Light board, dark walls (the physical game) |
| 4 | `slate` | **Slate Blue** | `#1e2a3a` / `#233245` | `#101a26` | `#2c3c52` | `#d3ddea` → `#7d8ea3` | Coolest, highest cell/wall contrast |
| 5 | `emerald` | **Emerald Felt** | `#16302a` / `#1a3a32` | `#0b1f1a` | `#22463c` | `#ecdfc0` → `#93815a` | Casino-table feel |
| 6 | `parchment` | **Parchment** | `#faf6ec` / `#f1ead9` | `#b3a68b` | `#a2937a` | `#3f3a2e` → `#201d16` | Max-contrast light theme, a11y-first |

Board theme is **independent** of UI theme (a light board on a dark UI is allowed and looks good;
the preview swatches in Settings render the real combination).

### 2.2 Pawn styles (4) — `data-pawn="…"`

All drawn in canvas, all scaled from cell size `C`:

| Key | Name | Rendering |
| --- | ---- | --------- |
| `disc` | **Disc** *(default)* | `0.62C` circle, radial gradient `--pX-light` @35%/30% → `--pX`, 1 px `--pX-deep` rim, elliptical contact shadow at `+0.06C` |
| `pillar` | **Pillar** | Cylinder: bottom ellipse (`0.56C × 0.20C`), body `0.56C × 0.46C` with a vertical light band, top ellipse in `--pX-light`; the most "3D" option |
| `crown` | **Crown** | Chess-pawn silhouette (bulb + collar + flared base) filled with the player gradient, 1.2 px `--pX-deep` outline — the Zchezz nod |
| `rune` | **Rune** | Hexagon `0.60C` with an engraved glyph (`▲` for the goal-up player, `▼` for the other), inner shadow; flattest and most legible on small screens |

The **side-to-move** pawn always carries the same treatment regardless of style: a 2 px `--gold`
ring at `r = 0.40C` plus a `--gold-glow` outer shadow, breathing at 1.6 s (static under reduced motion).

### 2.3 Wall rendering

A wall spans two cells plus the groove between them. Drawn as a beam:

```
thickness  T = groove width G  (see §6.1)
length     L = 2C + G
body       linear-gradient across T: --wall-edge 0%, --wall 22%, --wall 78%, --wall-edge 100%
cap        1px --wall-edge stroke, corner radius 2px
shadow     0 2px 6px rgba(0,0,0,.55) offset along the perpendicular axis
seam       a 1px --wall-edge line at the midpoint (shows it is a 2-cell piece)
```

The **last wall placed** gets a `--gold` 1.5 px outline that decays over 1.2 s.
Walls placed by the engine fade in from `opacity 0, scale .82` over `--dur`.

---

## 3. Layout — Mobile (`< 900px`, portrait-first)

All of R4's data lives in **one dual HUD bar directly above the board** — a single eye fixation,
immediately adjacent to what the player is looking at. Two stacked cards (one above, one below the
board) were rejected: they cost ~130 px of the vertical budget, which is exactly the scarce axis on
a phone, and they split the same comparison (my walls vs. yours, my clock vs. yours) across the
screen. The bar is 78 px total and both players are side by side, so the comparison is free.

Vertical budget on a 390 × 844 device (`100dvh` minus safe areas):

```
+----------------------------------------------+  <- env(safe-area-inset-top)
| HEADER                                  44px |
|  * ZQUORIDOR      [ Knight v ]       (o) (i) |
+----------------------------------------------+
| DUAL HUD BAR                            78px |
|  o You            |  o Zquoridor . Knight    |  <- 64px: two halves
|  (t) 5:02  *      |  (t) 4:31                |
|  ||||||||..  8  6 |  6  ||||||....  dist 8   |
|  ~~~~~~~~~~~~~~~~~|~~~~~~~~~~~~~~~~~~~~~~~~  |  <- 14px: race meter,
|  ==================------------  6 : 8  d+2  |     spans the full width
+----------------------------------------------+
|                                              |
|                                              |
|              BOARD (square)                  |
|         min(100vw-16, remaining)             |
|                                     ~412px   |
|  | eval strip 6px on the left edge           |
|                                              |
|                                              |
+----------------------------------------------+
| STATUS STRIP                            20px |
|  Your move - 2 legal jumps        [ Moves ]  |
+----------------------------------------------+
| WALL DOCK                               78px |
| +--------+ +----+ +----+ +----+ +--------+   |
| |  ====  | | <- | | ?  | | ^v | |   ||   |   |
| |   H  6 | |undo| |hint| |flip| |  V  6  |   |  <- live wall badges
| +--------+ +----+ +----+ +----+ +--------+   |
+----------------------------------------------+
| TAB BAR                                 52px |
|   > Play    |    ~ Analysis   |   * Editor   |
+----------------------------------------------+  <- env(safe-area-inset-bottom)
```

Rules:

* The board is `flex: 1 1 auto; min-height: 0` and always renders as a **square**: side =
  `min(containerWidth, containerHeight)`. Everything else is `flex: 0 0 auto`.
* **Half order is fixed: you on the left, opponent on the right** — regardless of colour, of who
  moves first, or of board flip. A HUD half that swaps places between games destroys the muscle
  memory that makes it glanceable. The pawn colour dot carries the identity; the position never moves.
* The active half is marked by a 2 px `--gold` underline plus the pulsing dot next to its clock;
  the inactive half drops to 78 % opacity (its numbers stay at full contrast — only the labels dim).
* Your own remaining wall count appears **twice**: in the HUD half above the board *and* as a live
  badge on each wall dock button below it. While placing a wall the thumb and the eye are at the
  bottom, and that is precisely when the count matters most.
* If the remaining height drops below `300px` (small phones in landscape, keyboard open),
  the layout switches to §3.3 landscape mode instead of squeezing.
* **Reach zones** (right-handed thumb arc measured from the bottom-right corner, ~150 px radius):
  the V wall button, the flip and hint buttons sit inside it; H sits in the mirrored left arc.
  Nothing destructive (New Game, Resign) is placed in either arc — those live in the header menu.
* The move log is **not** on the main screen on mobile. It is a bottom sheet raised by swiping up
  from the tab bar or by the `Moves` chip on the status strip (§5.8).

### 3.1 Header (mobile)

| Slot | Element | Spec |
| ---- | ------- | ---- |
| left | `ZQUORIDOR` | `Cinzel` 700 `.72rem`, `--gold`, letter-spacing `.14em`; tap = About modal |
| center | Level chip `#lvlChip` | `Cinzel` `.62rem` on `--surf2`, 1 px `--bor`, `r-sm`, 28 px tall; tap opens the Level sheet (§5.7) |
| right | `#btnSettings`, `#btnMenu` | 36 × 36 hit targets, `--txt2`, hover/active `--gold2` |

`#btnMenu` dropdown (right aligned, `--surf` E5): New Game · Flip board · Copy QFEN · Paste QFEN ·
Copy game text · Resign · Analyse this game · About.

### 3.2 Wall dock (mobile) — detail

```
outer: height 78px, padding 6px 8px calc(6px + safe-area-bottom),
       display: grid, grid-template-columns: 1.35fr .8fr .8fr .8fr 1.35fr, gap 8px
```

| id | Label | Size | Idle | Armed / active | Disabled |
| -- | ----- | ---- | ---- | -------------- | -------- |
| `#wallH` | swatch + "H" | >= 96 × 64 | E2, swatch `--wall-dim` 26×7 px | E3, swatch `--gold2`, label `--gold2`, 1.6 s breathing ring | walls = 0 → `opacity .32`, `pointer-events:none`, label "0 left" |
| `#wallV` | swatch + "V" | >= 96 × 64 | E2, swatch 7×26 px | same as H | same |
| `#btnUndo` | "Undo" | 64 × 64 | E2 | pressed `scale(.94)` | no history → `.32` |
| `#btnHint` | "Hint" | 64 × 64 | E2 | while computing: gold spinner ring | engine turn → `.32` |
| `#btnFlip` | "Flip" | 64 × 64 | E2 | — | never |

* Both wall buttons carry a **live count badge** in the top-right corner (`--gold` pill, `.54rem`)
  duplicating the HUD number — because while placing, the thumb (and the eye) is on the dock.
* Long-press (>= 400 ms) on `#wallH`/`#wallV` = "sticky arm": stays armed after a placement, for
  consecutive wall placement in the Editor and Analysis. In Play it is a no-op (one move per turn).
* The dock is `touch-action: none`, so a drag started on a button is never stolen by page scroll.

### 3.3 Mobile landscape (`< 900px` and `orientation: landscape`)

Height is now the scarce axis in the other direction, so the HUD moves **beside** the board, like
on desktop. Two columns: board on the left (square, full height), a 200–260 px right rail holding,
top to bottom: opponent HUD card (compact 44 px), vertical race meter + your HUD card, then the wall
dock **stacked vertically** (H above V, each 64 px tall, full rail width — still bottom-right, i.e.
inside the thumb arc), then the tab bar as three icons. The header collapses to 34 px.

---

## 4. Layout — Desktop (`>= 900px`)

Desktop is **three columns**: a dedicated HUD rail, the board, and the side panel. On a wide screen
the scarce axis is vertical, so nothing that can be put beside the board is allowed above or below
it — the board keeps the full column height and the HUD gets a permanent home of its own, at the
board's own eye level rather than at the top of the screen.

```
+-----------------------------------------------------------------------------------+
| HEADER 52px                                                                       |
| ZQUORIDOR  NNUE . MCab  [Pebble|Sprite|Squire|Knight|Sage|Titan|Custom]     (o)(i) |
+----------------------+------------------------------+-----------------------------+
| HUD RAIL  148px      | BOARD COLUMN     flex 1      | SIDE PANEL  380px (340-460) |
| +------------------+ |                              | +-------------------------+ |
| | o Zquoridor      | |                              | | > Play | ~ Analysis | * | |
| |   . Sage         | |   +----------------------+   | +-------------------------+ |
| |  (t) 4:31        | | e |                      |   | | STATUS LINE (Cinzel)    | |
| |  ||||||....   6  | | v |                      |   | | "Your move - 2 jumps"   | |
| |  dist        8   | | a |    BOARD (square)    |   | +-------------------------+ |
| +------------------+ | l |  min(col-w, col-h)   |   | | MOVE LOG      flex 1    | |
|                      |   |                      |   | | 1. e2 +0.12  e8  -0.05  | |
|  RACE METER          | b |                      |   | | 2. e3 +0.20  Vc6 -0.31  | |
|  (vertical, 16px)    | a |                      |   | | ...                     | |
|      #               | r |                      |   | +-------------------------+ |
|      #  6 : 8        |   +----------------------+   | | EVAL GRAPH        72px  | |
|      .  d+2          |                              | +-------------------------+ |
|      .               |  +- WALL DOCK ------- 56px   | | NAV  |< < > >|    72px  | |
|                      |  | [== Horizontal H]         | | New . Hint . Takeback   | |
| +------------------+ |  | [|| Vertical  V ]         | +-------------------------+ |
| | o You            | |  | [Undo][Flip][Paths]       |                             |
| |  (t) 5:02   *    | |  +---------------------------+                             |
| |  ||||||||..   8  | |                              |                             |
| |  dist        6   | |                              |                             |
| +------------------+ |                              |                             |
+----------------------+------------------------------+-----------------------------+
```

* **HUD rail** `#hudRail`, 148 px, `E1`, sits between the header and the bottom of the column.
  Opponent card is anchored to the **top**, your card to the **bottom**, the vertical race meter
  fills the space between them. That vertical arrangement mirrors the board: the opponent is the
  side you are running away from (top of the board), you are the bottom — so the rail reads as a
  legend for the board next to it.
* The rail card is the same component as the mobile HUD half (§5.1), laid out in 4 stacked lines
  (name / clock / pips + count / dist) instead of 2 rows — it has 148 px of width, so the clock
  goes up to `1.5rem` and `dist` keeps its label.
* **Race meter vertical** on desktop: fill grows from each end toward the middle, your colour from
  the bottom, the opponent's from the top, `--gold` divider at the meeting point, numbers rotated
  upright in the middle. Tap/click still toggles the path overlay.
* The **eval bar** is a vertical 14 px strip immediately left of the board (between rail and board),
  full board height, `--gold` 1 px centre line at 50 %, numeric label floating at the fill boundary.
* The wall dock stays **directly under the board**, horizontally centred: mouse travel from board to
  dock is one short move, and keyboard users never need it (`H` / `V`).
* `>= 1400px`: the side panel gets a second column for Multi-PV lines when Analysis is active
  (panel widens to 460 px); the board grows to fill.
* `900–1099px`: rail narrows to 120 px (the `dist` label drops, the number stays), side panel 340 px.
* `< 900px`: the rail dissolves back into the mobile dual HUD bar (§3) — same DOM nodes, same
  component, only the container's `flex-direction` and the card's internal grid change. There is one
  HUD implementation, never two.

---

## 5. Component Specifications

### 5.1 Player HUD card — `#hudTop` / `#hudBottom` (R4)

**One component, two layouts.** `.hud-card` holds the same five elements in both form factors; only
its internal grid changes with a container query / breakpoint class:

```
compact  (mobile half of the dual bar, ~186 x 64px, 2 rows)
+-------------------------------------+
| o You                    (t) 5:02 * |
| ||||||||..  8            dist 6     |
+-------------------------------------+

rail  (desktop HUD rail card, 148 x 96px, 4 lines)
+------------------+
| o Zquoridor      |
|   . Sage         |
| (t) 4:31         |
| ||||||....    6  |
| dist          8  |
+------------------+
```

| Element | Spec |
| ------- | ---- |
| Colour dot | 10 px, radial gradient `--pX-light → --pX`, 3 px `rgba(255,255,255,.05)` halo |
| Name | `JetBrains Mono` 600 `.80rem` `--txt`, ellipsis; the engine name is suffixed with `· <Level>` in `--txt2` |
| **Clock** | `JetBrains Mono` 600 **`1.35rem`** (mobile) / `1.5rem` (desktop), `tabular-nums`, `--txt`. `< 30 s` → `--amber` + 1 Hz pulse; `< 10 s` → `--red` + a tick each second; flagged → `--red`, struck through, card border `--red` |
| **Wall pips** | 10 bars `4 × 12px`, `gap 2px`. Remaining = `linear-gradient(180deg,--pX-light,--pX)`; spent = `rgba(255,255,255,.10)`. Pips deplete **right-to-left**; a pip that has just been spent flashes `--gold` for 400 ms |
| **Wall numeral** | `.95rem` 700 `--txt`, always next to the pips (pips are the pattern, the numeral is the truth) |
| `dist` | `shortestPathLen` for that player, `.88rem` 700 with a `.54rem` uppercase `--txt2` label. `--green` when it is the lower of the two, `--red` when higher |
| Turn state | The side to move: `border-color: var(--gold)`, `box-shadow: 0 0 0 1px var(--gold)`, `translateY(-1px)`; a small dot next to the clock pulses while the engine thinks |
| Result state | winner → gold border + trophy after the name; loser → dimmed to 70 %; draw → `--muted` border + handshake |

The 64 px compact height comes from name+clock on row 1 and pips+dist on row 2 at `line-height: 1.1`.
**The card never collapses, never scrolls, and is never covered by a sheet** — the move-log sheet
peaks at `60dvh` and is anchored below the board, never over the HUD.

### 5.2 Race meter — `#raceMeter` (R4, beyond-Zchezz)

Horizontal (mobile, 14 px, the bottom edge of the dual HUD bar) or vertical (desktop, 16 px, the
middle of the HUD rail). It answers "who wins the race right now?" from raw distances, with no
engine involved — it is true even when the engine is off:

```
share0 = d1 / (d0 + d1)      // your share grows as the opponent's path grows
=============------------      6 : 8   d+2
|- p0 fill -||- p1 fill -|     ^ labels, tabular-nums
```

* Fill widths animate over `--dur` on every state change; the boundary carries a 2 px `--gold`
  divider so the meeting point is unmistakable.
* Delta chip: `--green` when you lead the race, `--red` when behind, `--muted` at 0.
* Tapping the meter toggles the **shortest-path overlay** (same as the `Paths` button).
* If either distance is unreachable (rules guarantee it cannot happen) the bar renders `--red`
  hatching and the status line shows a rules error — a cheap invariant check in the UI.
* It is **not** the eval bar (§5.2b): the race meter is a raw-geometry fact, the eval bar is the
  network's opinion. Keeping them separate — and visually different (segmented dual fill vs. smooth
  gradient strip) — is deliberate: they disagree exactly in the positions that are interesting.

### 5.2b Eval bar — `#evalBar` (Zchezz parity)

A smooth strip that always shows *who is winning and by how much*, present in Play (subtle) and
Analysis (prominent). Vertical, 14 px, along the board's left edge on desktop; on mobile it becomes
a 6 px strip on the same edge, expanding to 12 px with numbers while the Analysis tab is open.

```
fill  share_p0 = 50 + 50 * tanh(score / 400)          // score = mover-relative NNUE cp, de-mirrored
      (or the WL head's win probability directly when the worker returns one)
colour  lerp( --p1 , --p0 )  over the same share  ->  the bar's own hue tells you who leads:
        share < 35%  : solid --p1 side dominates, cap tinted --p1-light
        35-65%       : gradient through a neutral --muted midpoint (a genuinely unclear position
                       looks unclear — no false precision)
        share > 65%  : solid --p0 side dominates, cap tinted --p0-light
centre  1px --gold line at 50%, always drawn, so the lead direction is readable at a glance
label   the numeric eval floats at the fill boundary, `.58rem` tabular-nums, auto-contrast colour;
        format: +1.4 / -0.8 (pawn-equivalent), or "W in 6" / "L in 4" when the search proves a win
```

* Animation: the boundary eases over `--dur` — never snaps — so a swing reads as motion, which is
  what makes a blunder *feel* like one.
* In Play mode the bar is drawn at 45 % opacity with no number unless "eval hints" is on (default
  on); turning it off hides the bar, the number and the goal glow together, in one setting, for
  players who want no engine assistance.
* The same fill share drives the **goal-edge glow** (§5.3), so board and bar never contradict.
* Sign convention: internally everything is mover-relative (per `CLAUDE.md`'s evaluation table);
  the bar de-mirrors to **absolute colour** exactly once, at render, like `evalToWhitePercent()`
  does today. P0's share fills from P0's own goal edge, so the bar's geometry matches the board's.

### 5.3 Board canvas — `#board` (`QBoard` class in `board.js`)

* One `<canvas>`, DPR-aware (`canvas.width = cssPx * devicePixelRatio`, context scaled once).
* Repaint is layered and dirty-tracked: a **static layer** (frame, cells, grooves, coordinates)
  drawn to an offscreen canvas and blitted; a **dynamic layer** (walls, pawns, ghosts, dots,
  overlays, arrows) redrawn per frame while an animation runs, otherwise only on state change.
* Coordinates: files `a–i` along the bottom edge, ranks `1–9` along the left edge, `.52rem`
  `--txt2` at 60 % opacity, drawn inside the frame margin (`0.42C`), auto-hidden when `C < 26px`.
  Toggleable in Settings.
* Goal edges: a 3 px bar along row 9 tinted `--p0` and along row 1 tinted `--p1` at 45 % opacity;
  the bar of the side that is winning per the current eval glows (`--dur-slow` pulse, amplitude
  scaled by `|eval|` and capped so it never distracts; off under reduced motion or "eval hints" off).
* Last move: the destination cell gets a `--gold-dim` fill + 1.5 px `--gold` outline for pawn
  moves; a wall gets the outline treatment from §2.3.
* Legal-move dots: `0.20C` circles at `--pX` 42 % opacity; a *jump* destination is a ring (not a
  filled dot) plus a faint arc from the pawn over the jumped pawn.
* Shortest-path overlay (toggle): a rounded polyline `0.10C` wide in `--pX` at 30 % opacity from
  each pawn to its goal, with a small chevron every other segment.
* Analysis arrows: lichess-style, `--gold2` for PV1 at 82 % opacity, then `--blue`, `--green`,
  `--amber`, `--muted` for PV2–5, width `0.13C`, arrowhead `0.30C`; a wall move inside a PV is
  drawn as a translucent ghost wall in the same colour instead of an arrow.

### 5.4 Side panel tabs — `#tabPlay` / `#tabAnalysis` / `#tabEditor`

Segmented control: `--surf2` track, active pill `--surf3` + 2 px `--gold` bottom border + `--gold2`
label. Mobile: fixed bottom tab bar, 52 px, 20 px icon + `.56rem` label. Desktop: the panel header.
Switching tabs never mutates the game; the Editor works on a scratch buffer (`qr_edit_*`) and only
commits on **Apply**.

### 5.5 Play tab

| Control | id | Placement | Behaviour |
| ------- | -- | --------- | --------- |
| Status line | `#status` | Panel top / mobile: a 20 px strip above the wall dock | `Cinzel` `.68rem` `--gold2`. "Your move", "Zquoridor is thinking… 12.4k nodes", "You won by reaching row 9 in 34 moves", "Illegal: that wall traps a player" |
| New Game | `#btnNew` | Panel actions row / mobile: `#btnMenu` | Opens the New Game modal (§5.10) |
| Hint | `#btnHint` | Dock (mobile) + actions row | Short analysis in the worker; draws the top move as a gold arrow / ghost wall for 4 s; never plays it |
| Takeback | `#btnTakeback` | Actions row / dock Undo | Rolls back to your previous turn (two plies vs the engine), animated in reverse |
| Resign | `#btnResign` | `#btnMenu` only | Confirm modal — never a one-tap action |
| Paths | `#btnPaths` | Dock (desktop) / race-meter tap (mobile) | Toggles the shortest-path overlay |

### 5.6 Analysis tab

Direct port of the Zchezz analysis panel (`an-*` ids), adapted to Quoridor's move space. Layout:

```
+---------------------------------------------------------------+
| [ ~ ENGINE ON ]   Depth [ 12 v ]   Lines [ 3 v ]      #anBar   |  48px
+---------------------------------------------------------------+
| d14 . 41.2k nodes . 18.3k nps . mcab tree 20k visits  #anInfo  |  18px
+---------------------------------------------------------------+
| +0.42 | e2 Hc6 e3 Vf5 e4 ...                  62%  12.4k   L1  |
| +0.18 | Hd3 e8 e2 Hf6 ...                     21%   4.2k   L2  |  each row
| -0.05 | Vb4 e2 e7 ...                         11%   2.1k   L3  |  38-56px
| ...                                                       L4-5 |
+---------------------------------------------------------------+
| EVAL GRAPH  (click / drag to jump)                      72px   |
|   /\    ___                                                    |
|  /  \__/   \____                                               |
+---------------------------------------------------------------+
| |<   <   >   >|     ply 24/38      [ Blunder check ]    72px   |
+---------------------------------------------------------------+
```

| Element | id | Spec |
| ------- | -- | ---- |
| **Engine toggle** | `#anEngBtn` | Wide E2 → E3 while running, label `ENGINE ON` in `Cinzel` `--gold2` with a 1.6 s breathing dot. Analysis never plays a move — it only reads the position at the current cursor |
| **Depth / time select** | `#anDepth` | `Depth 8 / 12 / 16 / 20 / ∞` or `Time 0.5s / 1s / 3s / ∞`. `∞` = the continuous sliced loop (≤ 250 ms per slice in the worker), so the UI stays responsive at any budget |
| **Lines select** | `#anPvCount` | `1 / 2 / 3 / 4 / 5` — Multi-PV width, read from `rootNodeForInspection()` on the MCTS tree (§12) |
| **Info line** | `#anInfo` | depth · nodes · nps · tree visits, `.58rem` `--txt2`, refreshed at ≤ 8 Hz so numbers stay readable rather than flickering |
| **PV row** | `#anLine1..5` | `[eval chip] [moves] [share] [visits]`. Eval chip: `--green` background tint when good for the side to move, `--red` when bad, `--muted` near 0, `tabular-nums`, same formatter as the eval bar. Moves in `JetBrains Mono` `.68rem`, wrapping to at most 2 lines, ellipsis after that. **Share %** = that move's visit share of the root, drawn as a thin `--gold` underline the width of the share — the MCTS-native equivalent of Zchezz's line ordering, and the honest answer to "how sure is it?" |
| PV row states | — | hover / press = preview on the board (arrows + ghost walls, colours per §5.3); click = enter a **preview cursor** stepping through that line, with a `Return to game` chip pinned above the panel; `1–5` keys do the same |
| **Eval graph** | `#anGraph` | 72 px canvas, x = ply, y = win probability. Area fill split at 50 % between `--p0-soft` and `--p1-soft`, 1.5 px line in `--gold2`, current ply a `--gold` vertical marker, blunders marked with a `--red` dot at their ply. Click or drag scrubs the move cursor; touch supported with a 44 px tall hit area |
| **Navigation** | `#navFirst/#navPrev/#navNext/#navLast` | 44 × 40 each, plus a `ply 24/38` readout; keyboard `,` `.` `Home` `End`. Backed by `qr_goto_ply` / `qr_undo` / `qr_redo` |
| **Blunder check** | `#anBlunderBtn` | Evaluates both candidate and played move per ply through the worker, with a determinate progress bar and a `Cancel`. Annotates the move log with `?!` / `?` / `??` at win-probability drops of 0.06 / 0.13 / 0.25, and `!` / `!!` for a found-only-move. Ends with an accuracy card: `You 84.2 % · Zquoridor 96.1 % · 2 blunders, 1 mistake, 3 inaccuracies` |

Quoridor-specific additions the chess original has no need for:

* A PV containing a wall move renders that wall as a translucent ghost on the board instead of an
  arrow, in the line's colour — so a "line" is legible even when it is three walls and no steps.
* Each PV row carries a small `Δdist` badge (`6→9` for the opponent) — the concrete consequence of
  the line in the game's own currency, next to the abstract eval.
* Analysis of an in-progress game is non-destructive: the engine reads the cursor position, the game
  keeps its own state, and `Return to game` restores the live cursor. Switching tabs never mutates.
* Mobile: the panel becomes the third tab's full-screen sheet; the PV rows collapse to
  `[eval] [first 3 moves] [share]` with the full line on tap; the eval graph keeps its 72 px.

### 5.7 Level selector — Pebble → Titan

| Level | Budget | Chip colour | Sheet description |
| ----- | ------ | ----------- | ----------------- |
| Pebble | 50 ms | `--muted` | "Learning the rules with you" |
| Sprite | 150 ms | `--txt2` | "Quick and careless" |
| Squire | 400 ms | `--green` | "Solid club player" |
| Knight | 1 s | `--blue` | "Punishes loose walls" |
| Sage | 2.5 s | `--amber` | "Sees the whole race" |
| Titan | 8 s | `--gold` | "Full strength — good luck" |
| Custom… | — | `--gold2` | Reveals the raw `mode` (depth / time / clock), `depth` and `time-ms` selects |

`S` cycles levels. The header chip shows the current name in `Cinzel`. Changing level mid-game is
allowed and logged in the move list as a `— level → Sage —` separator row.

### 5.8 Move log — `#moveLog`

Desktop: a flex-1 scroller in the side panel. Row = `<ply#> <P0 move> <eval> <P1 move> <eval>`,
`.72rem` monospace, alternating row tint `rgba(255,255,255,.02)`, current ply highlighted with
`--surf3` + a 2 px `--gold` left border, auto-scrolled into view.
Mobile: a **bottom sheet** `#moveSheet` — drag handle, snap points `0 / 42dvh / 74dvh`, opened by
swipe-up from the tab bar or the "Moves" chip; it dims the board but never covers the HUD cards.
Notation: pawn `e5`, wall `Hc6` / `Vf3`, jump `e5^`, with `!` / `?` annotations after blunder check.

### 5.9 Editor tab

Palette (one row of E2 toggles): `Pawn 0`, `Pawn 1`, `Wall H`, `Wall V`, `Erase`, plus a
`Walls left` stepper per side and a `Side to move` segmented pair. Below the board, a **validity
strip** fed by the `qr_edit_*` validity bitmask: `Legal position` (green) or the specific failure
(`Player 1 has no path to goal`, `Overlapping walls`, `Wall budget exceeded`). `Apply` stays
disabled until valid. `Copy QFEN` / `Paste QFEN` live here and in `#btnMenu`.

### 5.10 Modals & sheets

All share: E5 card, `max-width: 420px`, `r-xl`, `Cinzel` heading `.92rem` `--gold`, close button
top-right, `Esc` closes, focus trapped, overlay click closes (except confirm dialogs), enter/exit
`scale(.96) → 1` + fade over `--dur-slow`. On mobile they render as bottom sheets with a
drag-to-dismiss handle instead of centred cards.

1. **New Game** — Play as (First / Second / Random, with a pawn preview) · Level (the six chips +
   Custom) · Time control (Unlimited / 5 min / 5+3 / 10 min / Custom) · Board theme swatches ·
   Start (full-width `--gold` gradient CTA, 48 px).
2. **Settings** — grouped list: *Appearance* (UI theme, board theme, pawn style, coordinates,
   animations, text size) · *Board* (shortest paths, legal dots, last-move highlight, eval glow) ·
   *Input* (confirm walls, touch offset Off/Small/Large, sticky wall arm, haptics) ·
   *Sound* (master toggle + volume slider) · *Engine* (level, worker on/off).
3. **About** — version, engine description (NNUE 354→256→{WL, Policy}, hybrid MCTS + alpha-beta),
   a rules summary with a mini animated board, credits, repository link.
4. **Text I/O** — one textarea (`.58rem` monospace) with Copy / Paste / Load; used for both QFEN
   and full game text; parse errors surface as a toast plus a red border on the textarea.
5. **Confirm** (Resign / New Game over an unfinished game / Editor Apply) — title, one body line,
   `Cancel` (E2) + the destructive action (`--red` fill).

### 5.11 Toasts — `#toasts`

Bottom-centre stack (above the tab bar / dock), `--surf` E5, 1 px left border in the semantic
colour, `.68rem`, auto-dismiss 2.6 s (errors 4 s), max 3 stacked, newest on top, swipe to dismiss.
Types: `info` (`--gold`), `ok` (`--green`), `warn` (`--amber`), `err` (`--red`).

---

## 6. Wall Placement — the Interaction Spec (R2, R3)

This is the most important section: in Quoridor, wall placement is where a GUI is won or lost.

### 6.1 Geometry

Let `C` = cell side in CSS px and `G` = groove width. **`G = clamp(0.20 * C, 8px, 14px)`** — the
groove must be a real touch target, not a hairline. `U = C + G`. Board side `S = 9C + 8G`
(plus a frame margin `M = 0.42C` when coordinates are on).

* Cell `(r, c)` top-left = `(M + c*U, M + r*U)`.
* **Wall anchor** `(r, c)`, `r, c` in `[0,7]`, is the groove intersection south-east of cell
  `(r,c)`: centre = `(M + (c+1)*U − G/2, M + (r+1)*U − G/2)`.
* Horizontal wall `H(r,c)` = rect `x ∈ [M + c*U, M + c*U + 2C + G]`, `y ∈ [anchorY − G/2, anchorY + G/2]`.
* Vertical wall `V(r,c)` = the transpose.
* Every wall is identified by the same `(orientation, r, c)` triple the engine uses; display-flip
  conversion happens only at the boundary (`dispWallRowToEngine`).

### 6.2 Input methods (all four always available)

**M1 — Drag from the dock** *(primary on mobile)*
`pointerdown` on `#wallH` / `#wallV` arms that orientation, captures the pointer and enters
`DRAGGING`. A ghost wall follows the pointer; `pointerup` over a legal slot commits, anywhere else
cancels with a soft fade. Because the drag starts off-board, it is never ambiguous with a pawn move.

**M2 — Arm then tap** *(accessible / low dexterity)*
A `click` without drag on `#wallH` / `#wallV` enters `ARMED`. The board dims non-groove areas by
8 %, every legal slot for that orientation is painted `--gold` at 18 % opacity, and the dock button
goes E3. A tap near a groove commits (or previews, per §6.5). `Esc`, a second tap on the dock
button, or a tap on a cell centre cancels.

**M3 — Direct board gesture** *(fastest for experts, mouse and touch)*
`pointerdown` **on the board within `0.55*U` of an anchor** with no orientation armed enters
`DRAGGING` with a *provisional* orientation:
* mouse/pen: the orientation of the groove segment actually under the cursor (horizontal groove →
  H, vertical groove → V, exact intersection → the last used orientation);
* touch: the same, with the anchor chosen by rounding.

Then **the drag vector decides**: moving `>= 0.45*C` predominantly horizontally forces `H`,
predominantly vertically forces `V`. This is the flip gesture — you never have to release and press
another button to change your mind.

**M4 — Keyboard** *(desktop, full a11y)*
`H` / `V` arm; arrow keys move the ghost anchor one step; `Enter` / `Space` commits; `R` flips the
orientation; `Esc` cancels. The ghost anchor starts at the last used anchor, otherwise `(3,3)`.

### 6.3 Snapping and legality assist

On every `pointermove` while `DRAGGING`:

```
1. p  = pointer position, corrected by the touch offset (§6.4)
2. a  = nearest anchor = ( round(p.y/U) - 1 , round(p.x/U) - 1 ), clamped to [0,7]^2
3. if dist(p, centre(a)) > 0.85*U  -> no ghost (pointer far from any groove); after 700 ms show
                                      the hint chip "drag onto a groove"
4. cand = (orientation, a)
5. if cand is ILLEGAL:
       search the 8 neighbouring anchors, keep the LEGAL one whose centre is nearest to p and
       within 0.60*U  ->  snap there   (the magnetic legality assist)
6. if still illegal -> ghost renders "bad" at a, and the status line names the reason
```

**Legality reasons** are surfaced verbatim, never as a generic "illegal":
`overlaps an existing wall` · `crosses another wall` · `would leave <player> with no path` ·
`no walls left` · `outside the board`.

The legal-slot set is computed **once per turn** from `qr_legal_move_*` and cached in a
`Uint8Array(2 * 64)` bitmap — hit-testing during a drag is then O(1) and never calls into WASM.

### 6.4 Touch occlusion (the finger problem)

A fingertip covers about two cells. Setting **Touch offset**: `Off | Small (0.6*U) | Large (1.0*U)`,
default **Small** on coarse pointers and `Off` on fine pointers.
When enabled, from the moment a drag enters the board the ghost is evaluated at
`p = pointer + (0, −offset)` and a 1 px `--gold` leader line is drawn from the real pointer position
to the ghost — so the mapping is explicit and learnable rather than magical.

### 6.5 Confirm mode

Setting **Confirm walls**, default **on** for coarse pointers and **off** for fine pointers.

* Off → `pointerup` over a legal ghost commits immediately.
* On → `pointerup` freezes the ghost at 100 % opacity with a `--gold` outline and shows a small
  two-button chip floating just outside the wall's far end (auto-flipped to stay on screen):
  a `check` (44 × 44, `--green` fill) and a `close` (44 × 44, `--surf2`). Tapping elsewhere on the
  board moves the pending ghost; `Enter` / `Esc` also work. The chip has no timeout.

Committing a wall in Play mode with confirm off is still recoverable through **Undo** in the dock —
undo being always one tap away is what makes fast placement safe.

### 6.6 Ghost styles

| State | Fill | Outline | Extra |
| ----- | ---- | ------- | ----- |
| `ok` | `--wall` at 55 % | 2 px `--green` | soft 6 px `--green` glow |
| `assisted` (snapped by §6.3 step 5) | `--wall` at 55 % | 2 px `--gold` | the anchor you actually pointed at keeps a faint `--muted` outline, so the assist is visible rather than silent |
| `bad` | `--wall` at 20 % | 2 px `--red` | a small cross glyph at the wall centre + the reason in the status line |
| `pending` (confirm mode) | `--wall` at 100 % | 1.5 px `--gold` | the confirm chip |

### 6.7 State machine

```
IDLE ---(dock press M1 / dock click M2 / H,V key M4)---> ARMED
IDLE ---(board press near anchor M3)-------------------> DRAGGING (provisional orientation)
ARMED ---(pointerdown on board)------------------------> DRAGGING
ARMED ---(Esc / dock re-click / cell tap / 6s idle)----> IDLE
DRAGGING ---(pointermove)---> DRAGGING   [snap, flip-by-vector, legality]
DRAGGING ---(pointerup, legal, confirm off)------------> COMMIT ---> IDLE
DRAGGING ---(pointerup, legal, confirm on)-------------> PENDING
DRAGGING ---(pointerup, illegal / off-board)-----------> IDLE   [shake + err toast]
PENDING ---(check / Enter)-----------------------------> COMMIT ---> IDLE
PENDING ---(close / Esc / outside tap on a cell)-------> IDLE
COMMIT: apply -> animate -> sound + haptic -> engine turn
```

`pointercancel` (incoming call, notification, system gesture) is treated exactly as `Esc`,
never as a commit.

### 6.8 Failure feedback

An illegal drop: the ghost does a 180 ms 3 px horizontal shake, plays the illegal buzz, fires
`vibrate([12, 40, 12])`, and the status line plus a `warn` toast carry the reason from §6.3.
No modal, ever.

---

## 7. Pawn Movement

* **Tap to move**: tapping your pawn selects it (gold ring + legal dots); tapping a dot moves;
  tapping elsewhere deselects. On a fresh turn the legal dots are shown *automatically* at 55 %
  opacity (setting "Always show legal moves", default on) so one tap on a destination is enough.
* **Drag to move**: press the pawn and drag; it follows at 85 % opacity with a shadow, the target
  dot enlarges on hover, release commits. Snap radius `0.55*C`.
* **Jumps**: straight jumps and both diagonal deflections come from `qr_legal_move_*`; diagonals
  render as rings with the arc hint (§5.3). No special interaction — they are just destinations.
* **Keyboard**: arrows move the pawn one step when legal; when the straight jump is the legal
  continuation, the arrow performs it. `Shift+Arrow` picks between the two diagonal deflections.
* **Animation**: `--dur` ease slide along the axis; a jump animates as a shallow parabola over
  `--dur-slow * 0.8` with a small scale-up at the apex.

---

## 8. Feedback Channels

| Event | Visual | Sound (WebAudio synth) | Haptic |
| ----- | ------ | ---------------------- | ------ |
| Pawn move (you) | slide + last-move highlight | 180 Hz sine blip, 40 ms, fast decay | `vibrate(10)` |
| Pawn move (engine) | slide + highlight | same at 220 Hz, −4 dB | `vibrate(10)` |
| Wall placed | fade + scale in, gold outline decay | 90 Hz sine thump + filtered noise burst, 90 ms | `vibrate(18)` |
| Illegal | ghost shake, red | 140 Hz square, 70 ms, detuned | `vibrate([12,40,12])` |
| Wall armed | dock E3, legal slots lit | 900 Hz tick, 18 ms, −12 dB | `vibrate(6)` |
| Hint ready | gold arrow draw-on | 660 Hz triangle, 60 ms | — |
| Low time (< 10 s) | clock red pulse | 1 kHz tick each second, −16 dB | — |
| Win | goal edge gold burst, trophy on the card | rising arpeggio 523 / 659 / 784 Hz | `vibrate([20,60,20,60,40])` |
| Loss | card dim, muted board | falling 392 / 294 Hz | `vibrate(60)` |
| Flag fall | clock strike-through | single 160 Hz, 400 ms | `vibrate(80)` |

Audio graph: one `AudioContext` created **on the first user gesture** (iOS requirement), one master
`GainNode` bound to the volume slider, per-event oscillator / noise nodes created and discarded.
No audio files anywhere — that keeps the standalone HTML self-contained.

---

## 9. Responsive Rules

| Breakpoint | Layout |
| ---------- | ------ |
| `< 380px` | Mobile stacked; dual HUD bar 68 px, dock 72 px, `dist` label hidden (number stays), coordinates off |
| `380–899px` portrait | Mobile stacked as §3 — dual HUD bar **above** the board, dock below |
| `< 900px` landscape | §3.3 — board left, HUD + dock in a right rail (HUD **beside** the board) |
| `900–1099px` | Desktop 3-col: HUD rail 120 px, board, panel 340 px |
| `1100–1399px` | Desktop 3-col: HUD rail 148 px, board, panel 380 px |
| `>= 1400px` | Desktop 3-col: HUD rail 148 px, board, panel 460 px with Multi-PV in its own sub-column |

The HUD is **one component in two layouts** (§5.1): `.hud-card.compact` inside the mobile dual bar,
`.hud-card.rail` inside the desktop rail. The breakpoint swaps a class and the container's
`flex-direction`; it never swaps implementations, never re-mounts, and never loses clock state.

Additional rules:

* `height: 100dvh` everywhere; `overscroll-behavior: none` on `html, body`; the only scrollable
  regions are the move log, the Multi-PV list and modal bodies.
* `padding-bottom: env(safe-area-inset-bottom)` on the tab bar, `padding-top: env(safe-area-inset-top)`
  on the header, `viewport-fit=cover` in the meta viewport.
* `touch-action: none` on the board and the dock; `manipulation` elsewhere (kills the 300 ms delay
  without killing scroll in the log).
* Board resizing is observed with a `ResizeObserver` (debounced by one animation frame) and only
  re-blits the static layer; nothing else re-lays out.
* Orientation change: re-fit and re-render; the armed / pending wall state survives the change.

---

## 10. Accessibility

* **Contrast**: all text >= 4.5:1 against its surface; the wall/cell and wall/groove pairs of every
  board theme >= 3:1. `tools/gui/contrast_check.py` asserts this over the token table and runs in
  the build, failing it on regression.
* **Touch targets** >= 44 × 44 px for every interactive element; wall dock buttons are 64 px tall.
* **Focus**: `:focus-visible` = 2 px `--gold2` outline with 2 px offset, never removed. Tab order:
  header → HUD → board (a single tab stop, arrow-key driven) → dock → tabs → panel.
* **Screen readers**: the canvas is `role="application"` with an `aria-label` plus a visually hidden
  live region `#srBoard` announcing every move ("You: pawn to e5" / "Zquoridor: horizontal wall at
  c6; you have 6 walls left; your distance to goal is 8").
* **Reduced motion**: §1.5.
* **Colour independence**: turn is signalled by border *and* pulsing dot *and* status line; walls
  remaining by pips *and* numeral; the race by fill *and* numbers; blunders by symbol *and* colour.
* **Text scaling**: all sizes in `rem` with a `--fs` root multiplier in Settings
  (`Normal / Large / Larger` = 1 / 1.12 / 1.25); the board simply gets smaller.

---

## 11. Settings Schema — `localStorage['zq.settings']`

```jsonc
{
  "v": 1,
  "preset":  "premiumDark",  // classic | premiumDark | highContrast | minimal | custom
  "ui":      "dark",         // dark | light | auto
  "accent":  "gold",         // 8 presets or "#rrggbb"
  "fs":      1.0,            // 1 | 1.12 | 1.25
  "density": "comfortable",  // comfortable | compact

  "board":   "obsidian",     // §2.1 + §17.1 keys (8)
  "frame":   "hairline",     // none | hairline | gilded | beveled
  "wallFinish": "beveled",   // flat | beveled | glossy | etched
  "cellSep": "grooves",      // grooves | flat | inlaid
  "coords":  "edges",        // off | edges | all
  "boardScale": 1.0,         // 0.88 - 1.0

  "pawn":    "disc",         // §2.2 + §17.3 keys (6)
  "pawnSize":"regular",      // small | regular | large
  "pawnShadow":"soft",       // off | soft | deep
  "distinctShapes": false,

  "paths":   false,
  "dots":    true,
  "lastMove":true,
  "evalGlow":true,
  "evalBar": true,

  "sound":   true,
  "soundPack": "wood",       // wood | modern | marble | silent
  "volume":  0.6,
  "soundEvents": { "moves": true, "walls": true, "illegal": true,
                   "clock": true, "end": true, "ui": true },
  "haptics": "full",         // off | light | full
  "anim":    "full",         // full | reduced | off
  "animSpeed": 1.0,          // 0.5 | 1 | 1.5

  "confirmWalls": null,      // null = auto by pointer type
  "touchOffset":  null,      // null = auto ("small" on coarse pointers)
  "stickyArm":    false,
  "handedness":   "right",   // right | left | auto

  "level":   "knight",       // pebble|sprite|squire|knight|sage|titan|custom
  "custom":  { "mode": "time", "depth": 12, "timeMs": 1000 },
  "clock":   { "mode": "none", "baseMs": 300000, "incMs": 0 },
  "side":    0,
  "flipped": false,
  "worker":  true,

  "autosave": true           // §16.6; zq.game and zq.recent live in their own keys
}
```

Unknown keys are preserved; a version bump migrates by merging over the defaults. A corrupt blob is
discarded with an `info` toast ("Settings reset").

---

## 12. Engine Integration

* All blocking search runs in the Web Worker (already implemented, `app.js` SECTION 13b). The
  premium GUI keeps that contract: the main thread owns game state, the worker receives a QFEN plus
  a single-shot `go`, and replies are `{lines, nodes, depth}`.
* Level → engine params: `Pebble 50 ms … Titan 8000 ms`, all through the same hybrid MCTS + alpha-beta.
  No artificial blunder injection at any level — a weak level is a *short* search, which produces
  human-plausible mistakes rather than random ones.
* The status line shows live `nodes` and `depth` from the worker at <= 8 Hz.
* Known limitation carried over: worker searches have no repetition history (QFEN root only).
  The premium GUI additionally sends the **last 8 position hashes** with the `position` command so
  contempt / 3-fold behave correctly; a worker build that does not accept the field ignores it.

---

## 13. Implementation Plan

| Phase | Deliverable | Files | Depends on |
| ----- | ----------- | ----- | ---------- |
| **P0** | Token layer + fonts + light/dark + theme/pawn data attributes wired to Settings | `style.html` | — |
| **P1** | `QBoard` canvas: static layer, walls, pawns, DPR, resize, themes, coordinates | `board.js` (new) | P0 |
| **P2** | Interaction: pawn tap/drag, wall M1–M4, snapping, assist, confirm, ghosts (§6) | `board.js`, `app.js` | P1 |
| **P3** | HUD cards, clocks + flag fall, race meter, wall dock, status line (§5.1–5.2, R3/R4) | `style.html`, `app.js` | P1 |
| **P4** | Layout shells: mobile stacked, mobile landscape, desktop 2-col, tab bar, move sheet | `style.html` | P3 |
| **P5** | Play tab complete: new game modal, levels, hint, takeback, resign, sounds, haptics | `app.js` | P4 |
| **P6** | Engine surface: `qr_goto_ply/undo/redo/analyze/edit_*`, QFEN + game text | `engine_wasm.cpp` | — (parallel) |
| **P7** | Analysis tab: engine loop, Multi-PV, eval bar, eval graph, blunder check | `app.js` | P6, P5 |
| **P8** | Editor tab + Text I/O modal + **QFEN/QGN import-export, dialects, drag-drop, URL hash, autosave & recent games (§16)** | `app.js`, `engine_wasm.cpp` | P6 |
| **P8b** | **Premium personalization (§17)**: 8 board themes with live previews, board dressing, 6 pawn styles, 4 sound packs with per-event toggles, haptics, motion, density, handedness, presets, settings import/export | `style.html`, `board.js`, `app.js` | P5 |
| **P8c** | **Image export**: PNG/SVG board render, footer wordmark, transparent/coords options (§16.5) | `board.js`, `app.js` | P1 |
| **P9** | A11y pass, contrast script, reduced motion, SR announcements, keyboard map | all + `tools/gui/contrast_check.py` | P7, P8b |
| **P10** | Standalone bundling (font inlining), device testing, docs | `build_standalone.py`, `readme.md`, `status.md` | P9 |

`app.js` section map (numbered banner comments, Zchezz style):
`1` constants · `2` WASM bindings · `3` notation · `4` settings · `5` state · `6` board bridge ·
`7` HUD/clocks · `8` race meter · `9` wall input · `10` move log · `11` play mode · `12` analysis ·
`13` editor · `13b` worker · `14` sound/haptics · `15` modals/toasts · `16` keyboard · `17` boot.

---

## 14. Keyboard Map (desktop)

| Key | Action |
| --- | ------ |
| `Arrows` | Move the pawn (or move the wall ghost while armed) |
| `Shift+Arrows` | Choose between diagonal deflections |
| `H` / `V` | Arm a horizontal / vertical wall |
| `R` | Flip the ghost orientation |
| `Enter` / `Space` | Commit the ghost / confirm a pending wall |
| `Esc` | Cancel an arm, a pending wall, a modal, or a preview line |
| `,` `.` | Previous / next ply — `Home` `End` first / last |
| `F` | Flip board · `P` toggle paths · `M` mute |
| `N` | New game · `S` cycle level · `T` takeback |
| `A` | Toggle the analysis engine · `1–5` preview Multi-PV line *n* |
| `?` | Keyboard help overlay |

---

## 15. Acceptance Criteria

**R2 / R3 — walls**

1. From a cold start on a phone, a first-time user places a legal wall in under 6 s using only the
   dock, without reading instructions (user test, 5 participants).
2. Each of M1–M4 places a wall at anchor `(3,3)` in both orientations.
3. Dragging over an illegal slot that has a legal neighbour within `0.60*U` snaps to it and marks
   the ghost `assisted`.
4. A path-blocking wall attempt shows the exact reason, never a generic error.
5. A drag interrupted by `pointercancel` places nothing.
6. Both wall buttons lie fully inside a 150 px radius of their nearest bottom corner on a 390 px
   viewport (measured by the layout test).

**R4 — glanceability**

7. Clock, name and wall count for both players are visible without scrolling at every breakpoint,
   in every tab, and while any sheet is at its maximum snap point.
8. A wall count change is reflected in the pips, the numeral **and** the dock badge within one frame.
9. A clock below 10 s is unmistakable at arm's length (colour + pulse + tick).
9b. Both HUD halves sit **above** the board on mobile portrait and **beside** it on desktop and
    mobile landscape, from the same DOM nodes; resizing across `900px` mid-game preserves the
    running clocks, the wall counts and any armed wall state.

**Analysis parity (Zchezz)**

9c. Engine toggle, depth/lines selects, up to 5 Multi-PV rows with eval + visit share, eval graph
    with click-to-jump, move navigation and one-click blunder annotation all work at every
    breakpoint, and the analysis engine never mutates the live game.
9d. The eval bar's hue tracks the leader continuously (`--p1` → `--muted` → `--p0`), animates
    rather than snaps, and agrees with the goal-edge glow at all times.

**R1 / R5 — general**

10. 60 fps board interaction on a 2019-class phone during a wall drag with the paths overlay on.
11. The `file://` standalone bundle works with no network (fonts inlined, no CDN request in the log).
12. The contrast script passes for all 6 board themes × 2 UI themes.
13. No page error in headless Chrome across: new game → 20 plies → analysis on → blunder check →
    editor apply → navigate to ply 0 → resume.
14. A full keyboard playthrough with no pointer input.

**Serialization (§16)**

15. `parse(serialize(x)) === x` for 1000 random legal positions (QFEN) and for 200 complete
    self-play games (QGN), asserted in the browser and in `tests/test_notation.cpp`.
16. Every dialect row in §16.3 imports correctly from a fixture file.
17. An invalid QFEN/QGN never corrupts the live game: it is rejected with the offending token named,
    and the board is untouched.
18. A game exported after a blunder check re-imports with its evals, clocks and annotations intact
    (the eval graph and move log rebuild identically).
19. Copy, download, drag-drop, file picker and URL hash all work in the `file://` standalone bundle
    (clipboard falls back where blocked, with the toast saying so).
20. Autosave survives a reload mid-game and offers `Resume`; a full `zq.recent` ring buffer evicts
    the oldest without error.

**Personalization (§17)**

21. All 8 board themes × 6 pawn styles × 2 UI themes render without a single hardcoded colour in
    the canvas code (grep asserts every fill/stroke reads a token).
22. Changing any setting applies live, with no reload and no loss of game state, and survives a
    reload.
23. Each of the 4 sound packs plays every event; per-event toggles silence exactly their event; the
    first sound after boot never throws under the autoplay policy.
24. Left-handed mode mirrors the dock, the confirm chip and the toasts — and leaves the board
    geometry untouched.
25. The 5 presets each produce a coherent, contrast-passing configuration; touching one option flips
    the chip to `Custom`; `Reset all` returns to `Premium Dark`.

---

## 16. Serialization, Import & Export

Quoridor has no universal PGN/FEN, so Zquoridor defines two formats and **reads every dialect it
plausibly meets**. Everything below round-trips: `parse(serialize(x)) === x` is a unit test
(`tests/test_notation.cpp` for the engine side, a browser assertion for the GUI side).

### 16.1 QFEN — position (the FEN equivalent)

One line, six space-separated fields, ASCII only:

```
QFEN := <pawn0> <pawn1> <walls0> <walls1> <wallList> <turn>[ <plyCount>]

pawn0/1   cell in algebraic form, file a-i + rank 1-9          e.g.  e1  e9
walls0/1  walls still in hand, 0-10                            e.g.  10  8
wallList  '-' if empty, else comma-free concatenation of wall tokens sorted
          by orientation then row then column; a token is
          <file><rank><orientation>, anchored at the wall's south-west cell,
          orientation h|v                                      e.g.  c6h f3v
turn      0 or 1 (side to move)
plyCount  optional, default 0 (used for the move counter and 50-move-style stats)

example   e2 e9 8 6 c6h e4h f3v 0 14
```

Rules: the parser is whitespace-tolerant, case-insensitive on the orientation letter, accepts the
wall list either space-separated or slash-separated, and **validates before applying** — a QFEN that
would leave a player without a path, exceeds a wall budget, or overlaps walls is rejected with the
exact failing token, never silently repaired. `qr_edit_*` provides the same validity bitmask the
Editor uses (§5.9), so import and manual editing share one code path.

### 16.2 QGN — game (the PGN equivalent)

PGN-shaped on purpose, so anyone who has seen a chess file knows what to do with it:

```
[Event "Casual game"]
[Site "Zquoridor Web"]
[Date "2026.08.22"]
[Player0 "Gustavo"]
[Player1 "Zquoridor NNUE (Sage)"]
[Result "1-0"]
[TimeControl "300+3"]
[Walls "10"]
[QFEN "e1 e9 10 10 - 0"]        ; omitted when the game starts from the initial position
[Engine "Zquoridor NNUE-3 / MCab 20k"]
[Annotator "Zquoridor blunder check"]

1. e2 e8 2. e3 c6h {locks the left corridor} 3. f3v? e7 4. e4 e6 ...
   34. e9 1-0
```

* **Move tokens**: pawn = destination cell (`e2`); wall = `<file><rank><h|v>` (`c6h`); a jump needs
  no marker (the destination is unambiguous) except for a diagonal deflection, which is written
  `e5/` or `e5\` when both deflections are legal and the destination alone would be ambiguous.
* **Annotations**: `!` `!!` `?` `??` `!?` `?!` after the move; `{...}` comments; `$n` NAGs accepted
  on import and mapped to the symbol set. The blunder check writes these directly.
* **Evals**: optional `{[%ev +0.42]}` and `{[%clk 0:04:31]}` tags inside comments, PGN-style, so a
  file exported after analysis re-populates the move log's eval column and the eval graph on import.
* **Result**: `1-0`, `0-1`, `1/2-1/2` (repetition/agreed), `*` (unfinished).

### 16.3 Foreign dialects accepted on import

| Dialect | Example | Handling |
| ------- | ------- | -------- |
| Glendenning / community notation | `e2`, `a3h`, `c6v` | Native — this *is* §16.2's move token |
| Orientation-first variants | `ha3`, `Hc6`, `V f3` | Normalised on parse |
| Move-pair numbering with dots or parentheses | `1. e2 e8`, `1) e2 e8` | Numbers are skipped, not trusted |
| Coordinate-pair wall form | `c6-d6/c7-d7` | Converted to the anchor form when unambiguous |
| Bare move list, newline separated | one move per line | Accepted |
| A QFEN pasted where a game was expected (and vice versa) | — | Auto-detected by shape, then routed to the right importer — the user never has to pick |

Anything unparseable produces a **precise diagnostic**: `line 3, token 5: "z9h" — file 'z' is
outside a-i`, shown in the Text I/O modal with the offending token highlighted in `--red`, plus an
`Import anyway (up to the error)` button that loads the valid prefix.

### 16.4 Where import/export lives

| Surface | Action |
| ------- | ------ |
| `#btnMenu` (mobile) / panel actions (desktop) | Copy QFEN · Paste QFEN · Copy game (QGN) · Paste game · Save `.qgn` · Open `.qgn` |
| **Text I/O modal** (§5.10) | The full surface: a textarea prefilled with the current export, format toggle `QFEN / QGN`, `Copy`, `Paste`, `Load`, `Download`, diagnostics area |
| **Drag & drop** | Dropping a `.qgn` / `.txt` / `.qfen` file anywhere on the board loads it; the board shows a `--gold` dashed drop target during the drag |
| **File picker** | `<input type="file">` behind the `Open` button — works in the `file://` standalone bundle, where drag & drop and the clipboard may be restricted |
| **URL hash** | `#qfen=<url-encoded>` and `#qgn=<base64>` load on boot; `Copy link` builds one. Under `file://` the hash still works, so a position can be shared as a bookmark |
| **Web Share** | On mobile, `navigator.share({title, text})` when available; falls back to clipboard + toast |
| **Clipboard** | `navigator.clipboard` with a hidden-textarea + `execCommand` fallback for `file://` and old WebViews; every copy confirms with an `ok` toast naming what was copied |

### 16.5 Image export (premium touch)

* **Copy board as image** / **Download PNG**: `canvas.toBlob()` of a dedicated off-screen render at
  a fixed 1600 × 1600 (independent of the on-screen size), with the current board theme, pawn style,
  walls, last-move highlight, and an optional footer strip carrying the two player names, the clock
  and a small `ZQUORIDOR` wordmark in `Cinzel`.
* **Download SVG**: the same scene emitted as vector, for print or for the docs.
* Both honour a `Transparent background` checkbox and a `Include coordinates` checkbox.
* Everything is local — no upload, no service, works offline in the standalone bundle.

### 16.6 Autosave & recent games

* The live game is written to `localStorage['zq.game']` (QGN string) after every ply, debounced by
  250 ms. On boot, if it exists and is unfinished, a `Resume game?` chip appears in the status strip
  for 10 s — never a modal, never an automatic restore that hides a fresh start.
* Finished games push onto `localStorage['zq.recent']`, a ring buffer of the last **20** QGN strings
  with a header summary. The About/`#btnMenu` → `Recent games` sheet lists them with result, level,
  date and move count; each row offers `Load`, `Analyse`, `Copy`, `Delete`.
* A storage-quota failure degrades to "session only" with one `warn` toast, never an exception.

---

## 17. Personalization — the Premium Layer

Everything here is a *setting*, persisted in §11, applied live with no reload, and included in the
standalone bundle. This is the section that makes the GUI feel expensive.

### 17.1 Board themes — 8 total

The six of §2.1 plus two "signature" themes that carry extra rendering, not just extra hex values:

| # | Key | Name | What is different |
| - | --- | ---- | ----------------- |
| 7 | `marble` | **Carrara Marble** | Cells carry a procedurally generated vein texture (deterministic from a seed, drawn once into the static layer — zero per-frame cost); walls are polished basalt with a specular highlight band |
| 8 | `noir` | **Noir** | Pure monochrome: `#0a0a0a` → `#f2f2f2`, players separated by *value* and pawn *shape* rather than hue — the accessibility flagship, and the one that photographs best |

Each theme in Settings renders as a live 3 × 3 mini-board preview (real `QBoard` at `C = 14px`)
showing two pawns and one wall, in the currently selected pawn style — so the choice is made on the
real combination, never on a colour dot.

### 17.2 Board dressing (independent of theme)

| Option | Values | Default |
| ------ | ------ | ------- |
| Frame style | `None` · `Hairline` · `Gilded` (2 px gold + inner shadow) · `Beveled` (raised wooden edge) | Hairline |
| Wall finish | `Flat` · `Beveled` (§2.3) · `Glossy` (specular band) · `Etched` (inner shadow + seam) | Beveled |
| Cell separation | `Grooves` (recessed, default) · `Flat` (single surface, walls float) · `Inlaid` (thin gold hairline grid) | Grooves |
| Coordinates | `Off` · `Edges` · `Every cell` (faint, for study) | Edges |
| Board scale | `88% – 100%` slider — shrinks the board to leave breathing room on huge screens | 100 % |
| Accent colour | 8 presets + a custom picker driving `--gold`/`--gold2` (with an automatic contrast-corrected `--gold2`) | Gold |

### 17.3 Pawn styles — 6 total

The four of §2.2 plus:

| Key | Name | Rendering |
| --- | ---- | --------- |
| `pawnChess` | **Ivory & Ebony** | Full chess-pawn 3/4 view with a cast shadow, the closest thing to the physical set |
| `beacon` | **Beacon** | A ring with a rotating light arc pointing at that player's goal — the didactic style for new players; the arc is static under reduced motion |

Plus, independent of style: **Pawn size** `Small / Regular / Large` and **Shadow** `Off / Soft /
Deep`, and a `Distinct shapes` toggle that forces the two players to differ in silhouette (not only
in colour) for colour-blind play — automatic in `noir`.

### 17.4 Sound — packs, not a single buzzer

| Pack | Character | Notes |
| ---- | --------- | ----- |
| `wood` *(default)* | Physical set: a soft click for pawns, a dry wooden knock for walls | The §8 synth table |
| `modern` | Clean UI tones, sine + short reverb tail | Higher, brighter, quieter |
| `marble` | Heavier: low thud with a longer decay for walls, glassy tick for pawns | Pairs with the marble theme |
| `silent` | Nothing but the flag-fall and the win/loss cues | For playing in company |

All packs are **synthesised** (no audio files — the standalone bundle stays self-contained), each
described by a small parameter table (waveform, base frequency, envelope, filter, gain) so adding a
pack is data, not code. Controls:

* Master `Sound` toggle + `Volume` slider (0–100 %, logarithmic, live-previewed on release).
* Per-event toggles: `Moves` · `Walls` · `Illegal` · `Clock ticks` · `Game end` · `UI` — because the
  low-time tick is exactly the sound some players want and others hate.
* `Test` button next to each pack that plays the wall knock at the current volume.
* Autoplay-policy safe: the `AudioContext` is created on the first gesture and a `Sound is off until
  you tap` hint appears once if a sound was requested before that.

### 17.5 Haptics

`Off` · `Light` · `Full` (the §8 table). Auto-disabled when `navigator.vibrate` is missing, with the
setting greyed and labelled `Not supported on this device` rather than silently ignored.

### 17.6 Motion & density

* `Animations`: `Full` · `Reduced` (positional changes only) · `Off`, seeded from
  `prefers-reduced-motion` on first boot.
* `Animation speed`: `0.5× / 1× / 1.5×` multiplier over `--dur*`.
* `Density`: `Comfortable` (default) · `Compact` (−12 % on all chrome, +board), for small laptops.
* `Text size`: `Normal / Large / Larger` (§10).

### 17.7 Handedness & reach (feeds R3)

`Handedness`: `Right` (default) · `Left` · `Auto`. Left mirrors the wall dock (V on the left, H on
the right), the confirm chip's preferred side, the move-log sheet's drag handle offset and the
toast stack alignment. It does **not** mirror the board — that would change the game's geometry.

### 17.8 Settings UX

* Grouped, scrollable list: *Appearance · Board · Pieces · Sound · Haptics · Motion · Input ·
  Engine · Data*. Each group is a card (E1) with a `Cinzel` header.
* Every control previews live against the real board behind the sheet — on desktop the modal is
  offset to the panel side so the board stays visible while you change themes.
* **Presets** at the top: `Classic` · `Premium Dark` (default) · `High Contrast` · `Minimal` ·
  `Custom` — one tap sets ~15 options coherently; touching any option switches the chip to `Custom`.
* *Data* group: `Export settings` / `Import settings` (JSON), `Clear recent games`, `Reset all` —
  each destructive item behind a confirm dialog.

---

## 18. Out of Scope / Future

Rated puzzles · opening explorer · online play · the 4-player variant · a game database ·
account/cloud sync · engine-vs-engine spectator mode. Each would build on the `engine_wasm.cpp`
surface from P6 without changing the layout contract above.
