# Web GUI refactor plan

Date: 2026-08-26
Status: proposed, waits for approval

## Context

The web GUI works, and it carries a large feature set: nine board skins,
a light theme and a dark theme, an analysis pane, an editor, text import
and export, and a Playwright suite that covers the main flows. The
problem is not the feature set. The problem is that the presentation
layer has no shared scale for type or space, one real defect hides the
settings content, and several visual decisions sit outside the token
system.

This plan fixes seven reported items, and it adds the token layer that
keeps the fixes consistent.

## Measured baseline

Every number below comes from the running page at
`http://127.0.0.1:8123/index.html`.

Desktop, viewport 1280 by 720:

| Element | Measurement |
| --- | --- |
| Board canvas | 532 by 532 px |
| Eval bar | 24 by 532 px, top offset -7.5 px against the board |
| Side panel | 332.8 by 640 px, which is 108 px taller than the board |
| Settings modal, first open | 128.3 px tall, 45 characters of text |
| Settings modal, after a tab click | 698.3 px tall, 459 characters of text |

Mobile, viewport 375 by 812:

| Element | Measurement |
| --- | --- |
| Board canvas | 353 by 353 px |
| Eval bar | 353 by 12 px, below the board |
| Side panel | hidden, it becomes a fixed drawer |
| Settings modal | 375.3 by 128.3 px, anchored to the bottom |

## Item 1: board coordinates

`board.js:461-478` draws the coordinates on the canvas. The font is
`Math.max(7, 0.16 * C)`, where `C` is the cell side. The alpha is 0.72.
The color is `--coord`, which is `#8a6a45` in the default wood theme,
against a frame of `#4b3722`.

On a 532 px board the cell side is approximately 53 px, so the label is
approximately 8.5 px. On a 353 px mobile board the formula returns 5.6
px, so the 7 px floor applies. A label of 7 px at 72 percent alpha, in a
brown that sits close to the frame brown, is the cause of the report.

The fix raises the coefficient and the floor, raises the alpha, and
gives `--coord` a luminance that separates it from the frame. The
coordinates also move from the static layer to the dynamic layer, so a
wall or a pawn never paints over them.

Target: approximately 11 px on desktop and never below 10 px, alpha 0.9,
and a contrast ratio of at least 4.5 to 1 against the frame.

## Item 2: the settings modal

This is a real defect with a known cause.

`app.js:2194` reads:

```js
$('btnSettings').onclick = modalSettings;
```

The handler receives the `MouseEvent` as the first argument. The
function signature is `modalSettings(tab)`, and its first statement is
`if (tab) settingsTab = tab;` (`app.js:1219`). A `MouseEvent` is truthy,
so `settingsTab` becomes the event object. Every lookup that follows
then misses: no tab receives the `on` class, `SETTINGS_TAB_HINT[...]`
returns undefined, and `cards[...]` returns undefined. The modal opens
with a title, four inert tabs, and no content.

The proof is in the baseline table. The first open renders 45
characters. A click on any tab passes a real string, and the same modal
then renders 459 characters.

The fix is one line:

```js
$('btnSettings').onclick = () => modalSettings();
```

`btnNew` does not have this defect, because `modalNewGame()` takes no
argument. The plan still wraps it, so the pattern is uniform.

A second, separate point: `#overlay` uses `align-items:flex-end`
(`style.html:535`), so the modal is a bottom sheet on every viewport.
That pattern is correct on mobile and wrong on desktop. The plan
centers the sheet above 900 px and keeps the bottom sheet below it.

## Item 3: the vertical bar and the board

Two different elements can match the description, and they fail for two
different reasons.

The eval bar already tracks the board. `app.js:2745-2749` sets
`ew.style.height = B.cssSide + 'px'`, so the measured height difference
is zero. The visible defect is a vertical offset of 7.5 px, which comes
from the 15 px bottom margin at `style.html:255` that reserves room for
the `#evalNum` label.

The side panel does not track the board at all. `#sidePanel` takes
`height:100%` of `#main` (`style.html:438-441`), while the board takes a
JS computed side that fits inside `#boardZone`. The two use different
sizing algorithms, so they agree only by accident. The measured
difference is 108 px.

The user selected the eval bar on 2026-08-26. The plan therefore aligns
the eval bar to the board's top edge and bottom edge, by moving the
`#evalNum` label out of the reserved 15 px margin. The label moves
inside the strip, or it moves below the board's baseline.

The side panel keeps its current sizing. It stays out of scope.

## Item 4: wall color

The three wall tokens are `--wall`, `--wall-hi` and `--wall-edge`, and
every theme redefines them (`style.html:29-33` and the theme blocks that
follow). That part of the system is sound.

The defect is that the wall's depth cues sit outside the token layer.
`board.js:641-643` hardcodes `rgba(0,0,0,.45)` for the shadow. The
glossy finish hardcodes `rgba(255,255,255,.55)`. The etched finish
hardcodes two more literals. A theme therefore cannot tune its own
depth, and a light board gets a shadow built for a dark board.

The plan promotes those literals to tokens, and it retunes the three
wall tokens per theme for a contrast ratio of at least 3 to 1 against
the cell bed, without raising saturation.

## Item 5: the pawn

The report says that the pawn changes with engine strength. It does
not. `LEVELS` (`app.js:12-19`) maps a level to a search time budget
only, and no code in `board.js` or `app.js` reads the level to change
the pawn's shape, color, or size.

The pawn shape is a user setting, `S.pawn`, whose default is `disc`
(`app.js:155`). The `disc` shape is already a plain circle with a radial
gradient whose light source sits at the upper left
(`board.js:779-784`), which is what the report asks for.

The reported base has two candidate causes. The first is a stored
setting of `pillar` or `pawnChess`, both of which draw a flared foot by
design (`board.js:803-812` and `board.js:848-861`). The second is the
contact shadow, an ellipse drawn under the pawn at
`board.js:772-778`, which can read as a pedestal.

The plan does not remove the alternative shapes, because they are a
deliberate feature. The plan makes `disc` the guaranteed default,
retunes the contact shadow so it reads as a shadow and not as a plate,
and strengthens the sphere's light.

## Item 6: wall placement

The requested behavior already exists.

The hover preview is at `board.js:521-527`. It draws the wall at
`globalAlpha = 0.35` when the pointer sits over a legal groove.

The orientation follows the drag. `app.js:883-886` flips the
orientation once the drag exceeds 45 percent of a cell: a wider
horizontal pull selects a horizontal wall, and a wider vertical pull
selects a vertical wall. The H and V buttons only force an orientation,
and the force expires after 6000 ms (`app.js:724`).

This item therefore needs verification and tuning, not new code. The
plan measures the 45 percent threshold and the 0.35 alpha against real
use, and it treats `test_wall_matrix.py` as the acceptance gate.

## Item 7: mobile and desktop

The input layer is already pointer based. `app.js:2201-2205` binds
`pointerdown`, `pointermove`, `pointerup`, `pointercancel` and
`pointerleave`, so mouse, touch and pen share one path. A coarse
pointer enables a touch offset and a confirm step
(`app.js:199-206`).

The layout has two breakpoints, at 900 px and at 900 px with landscape
orientation (`style.html:408-468`). The gap is the type scale and the
spacing scale, which do not respond to viewport at all.

## The token layer

This is the part that makes the other items hold.

The stylesheet already has tokens for color, radius and duration
(`style.html:13-42`). It has none for space or type. Every gap and
padding is a hardcoded pixel value, and every font size is a raw `rem`
with no scale and no `clamp()`.

The plan adds two token groups to `:root`:

1. A spacing scale, `--space-1` through `--space-8`, and a rewrite of
   the hardcoded gaps and paddings onto it.
2. A type scale built with `clamp()`, so the text responds to viewport
   width instead of depending only on the JS root font-size override at
   `app.js:1397`.

The existing font-size setting keeps working, because `clamp()` values
in `rem` still scale with the root percentage.

## Constraints the refactor must respect

`build_standalone.py` fails the build on two exact string matches. A
refactor must keep both.

1. The literal `ZquoridorModule().then((Module) => {` in `app.js`
   (`build_standalone.py:107-116`).
2. The script tag sequence in `style.html`, in this order: `board.js`,
   then `zquoridor.js`, then `app.js` (`build_standalone.py:119-125`).

`tools/gui/contrast_check.py` runs as a gate inside the bundler. A color
change that fails it stops the build, which is the wanted behavior.

`index.html` and `gui_web/zquoridor.html` are generated from the same
string. Never edit either by hand.

## Order of work

1. Fix the settings modal defect. One line, immediate.
2. Add the spacing tokens and the type tokens.
3. Fix the coordinates.
4. Fix the eval bar alignment, and the side panel after the decision
   below.
5. Retune the wall tokens and promote the hardcoded depth literals.
6. Retune the pawn default and its contact shadow.
7. Verify wall placement and tune the thresholds.
8. Sweep both viewports.

## Verification

Playwright is not installed in the current environment. The first step
of verification is:

```
pip install playwright
playwright install chromium
```

Then, from `gui_web/`, with a server on port 8123 already running for
the last two:

```
python test_gui_features.py
python test_deep_click.py
python test_browser_full.py
python test_gameplay_sim.py
python dev_server.py          # in another shell, for the two below
python test_features_e2e.py
python test_wall_matrix.py
```

After the source edits, rebuild the bundle and confirm that the gate
passes:

```
python build_standalone.py
```

A visual check of both viewports closes the work.

## Open questions

1. Playwright is not installed. The plan assumes that installing it is
   acceptable. Without it, no automated regression gate exists for this
   refactor.

## Decisions

- 2026-08-26: "The vertical bar" means the eval bar. The side panel
  keeps its current sizing and stays out of scope.
