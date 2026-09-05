#!/usr/bin/env python3
"""Apply the September 2026 GUI micro-polish pass.

This is intentionally a deterministic source-to-source patcher so the change can
be applied and tested on an isolated branch without hand-editing generated HTML.
It fails fast if an expected source anchor has drifted.
"""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
STYLE = ROOT / "gui_web" / "style.html"
APP = ROOT / "gui_web" / "app.js"
BOARD = ROOT / "gui_web" / "board.js"


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected exactly one match, found {count}")
    return text.replace(old, new, 1)


def sub_once(text: str, pattern: str, repl: str, label: str, flags=0) -> str:
    out, count = re.subn(pattern, repl, text, count=1, flags=flags)
    if count != 1:
        raise RuntimeError(f"{label}: expected exactly one regex match, found {count}")
    return out


# ---------------------------------------------------------------------------
# style.html: typography, progressive-disclosure styling and tiny chrome polish
# ---------------------------------------------------------------------------
style = STYLE.read_text(encoding="utf-8")

style = replace_once(
    style,
    "font-family:'JetBrains Mono',ui-monospace,Consolas,monospace;",
    "font-family:ui-sans-serif,-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;",
    "modern UI font",
)
style = replace_once(
    style,
    ".num{font-variant-numeric:tabular-nums}",
    ".num{font-family:'JetBrains Mono',ui-monospace,Consolas,monospace;font-variant-numeric:tabular-nums}",
    "technical numeric font",
)
style = replace_once(
    style,
    '.pbar .dist::before{content:"STEPS";',
    '.pbar .dist::before{content:"PATH";',
    "PATH label",
)
style = style.replace('title="Steps to goal"', 'title="Shortest path to goal"')
style = replace_once(
    style,
    "#boardZone > #evalWrap #evalStrip{width:24px;height:100%}",
    "#boardZone > #evalWrap #evalStrip{width:20px;height:100%}",
    "desktop eval thickness",
)
style = replace_once(
    style,
    "#underBoard > #evalWrap{width:100%;height:12px}\n#underBoard > #evalWrap #evalStrip{width:100%;height:12px}",
    "#underBoard > #evalWrap{width:100%;height:10px}\n#underBoard > #evalWrap #evalStrip{width:100%;height:10px}",
    "portrait eval thickness",
)

POLISH_CSS = r'''

/* ======================== micro-polish 2026-09 ========================= */
/* UI prose is intentionally sans-serif; data keeps the technical mono face. */
html[data-ui-font="technical"] body,
html[data-ui-font="technical"] button,
html[data-ui-font="technical"] input,
html[data-ui-font="technical"] select{
  font-family:'JetBrains Mono',ui-monospace,Consolas,monospace;
}
html[data-ui-font="modern"] body{
  font-family:ui-sans-serif,-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;
}
.clock,.wn,.dist b,#evalNum,#moveLog,.pvScore,.anInfo,.num,
input[type="number"],input[type="text"],textarea{
  font-family:'JetBrains Mono',ui-monospace,Consolas,monospace;
}

/* Advanced choices exist, but do not dominate the first settings scan. */
.advSettings{
  margin-top:var(--space-3);border-top:1px solid var(--bor);padding-top:var(--space-2);
}
.advSettings summary{
  list-style:none;cursor:pointer;color:var(--txt2);font-size:var(--fs-sm);
  font-weight:600;letter-spacing:.02em;padding:var(--space-2) 0;user-select:none;
}
.advSettings summary::-webkit-details-marker{display:none}
.advSettings summary::after{content:'›';float:right;color:var(--muted);font-size:1.15em;
  transform:rotate(90deg);transition:transform var(--dur-fast)}
.advSettings[open] summary::after{transform:rotate(-90deg)}
.advSettings .advBody{padding-top:var(--space-1)}

/* A barely-visible material layer prevents flat themes from looking synthetic.
   It is deliberately weak enough to leave pawns, walls and coordinates clean. */
#boardWrap::after{
  content:"";position:absolute;inset:0;border-radius:inherit;pointer-events:none;z-index:8;
  background:
    repeating-linear-gradient(12deg,rgba(255,255,255,.22) 0 1px,transparent 1px 9px),
    repeating-linear-gradient(101deg,rgba(0,0,0,.20) 0 1px,transparent 1px 13px);
  mix-blend-mode:soft-light;opacity:.025;
}
html[data-board-texture="off"] #boardWrap::after{display:none}
html[data-board-texture="subtle"] #boardWrap::after{opacity:.025}
html[data-board-texture="natural"] #boardWrap::after{opacity:.055}

/* Contrast is intentionally subtle: a preference, not a second theme system. */
html[data-board-contrast="soft"] #board{filter:contrast(.96) saturate(.99)}
html[data-board-contrast="standard"] #board{filter:none}
html[data-board-contrast="strong"] #board{filter:contrast(1.07) saturate(1.015)}

/* Goal-row strength remains neutral and never becomes a player-coloured band. */
html[data-goal-rows="off"] #boardWrap{--goal-wash:rgba(120,122,132,0)}
html[data-goal-rows="subtle"] #boardWrap{--goal-wash:rgba(120,122,132,.12)}
html[data-goal-rows="clear"] #boardWrap{--goal-wash:rgba(120,122,132,.21)}

/* The evaluation strip belongs to the board, but should not compete with it. */
#evalStrip{border-radius:3px}
#boardZone > #evalWrap #evalNum{font-size:12px;bottom:-18px}
'''
style = replace_once(style, "</style>", POLISH_CSS + "\n</style>", "micro-polish CSS")
STYLE.write_text(style, encoding="utf-8")


# ---------------------------------------------------------------------------
# app.js: settings schema, progressive disclosure and data attributes
# ---------------------------------------------------------------------------
app = APP.read_text(encoding="utf-8")
app = replace_once(app, "  v: 3,", "  v: 4,", "settings schema v4")
app = replace_once(
    app,
    "  accent: 'gold', fs: 1, density: 'comfortable',",
    "  accent: 'gold', fs: 1, density: 'comfortable', uiFont: 'modern',",
    "ui font default",
)
app = replace_once(
    app,
    "  frame: 'beveled', wallFinish: 'beveled', cellSep: 'grooves',\n  coords: 'edges', boardScale: 1,",
    "  frame: 'beveled', wallFinish: 'beveled', cellSep: 'grooves',\n  boardTexture: 'subtle', boardContrast: 'standard', wallProfile: 'standard', wallPreview: 'normal',\n  goalRows: 'subtle', moveMarkers: 'ring', lastMoveStyle: 'subtle',\n  coords: 'edges', boardScale: 1,",
    "board microsetting defaults",
)
app = replace_once(
    app,
    "    if ((parsed.v || 1) < 3) parsed.paths = false;\n    if (!LEVELS[parsed.level]) delete parsed.level;",
    "    if ((parsed.v || 1) < 3) parsed.paths = false;\n    if ((parsed.v || 1) < 4) {\n      if (parsed.lastMoveStyle == null) parsed.lastMoveStyle = parsed.lastMove === false ? 'off' : 'subtle';\n      if (parsed.moveMarkers == null) parsed.moveMarkers = 'ring';\n    }\n    if (!LEVELS[parsed.level]) delete parsed.level;",
    "settings v4 migration",
)

# Keep presets coherent rather than letting the new controls inherit random old state.
app = replace_once(
    app,
    "    pawnShadow: 'deep', soundPack: 'wood', anim: 'full', fs: 1,",
    "    pawnShadow: 'deep', soundPack: 'wood', anim: 'full', fs: 1, uiFont: 'modern',\n    boardTexture: 'subtle', boardContrast: 'standard', wallProfile: 'standard', wallPreview: 'normal',\n    goalRows: 'subtle', moveMarkers: 'ring', lastMoveStyle: 'subtle',",
    "classic preset polish",
)
app = replace_once(
    app,
    "    coords: 'all', fs: 1.12, cellSep: 'inlaid', wallFinish: 'beveled', anim: 'reduced',",
    "    coords: 'all', fs: 1.12, cellSep: 'inlaid', wallFinish: 'beveled', anim: 'reduced',\n    boardTexture: 'subtle', boardContrast: 'strong', wallProfile: 'standard', wallPreview: 'strong',\n    goalRows: 'clear', moveMarkers: 'ring', lastMoveStyle: 'clear', uiFont: 'modern',",
    "high contrast preset polish",
)
app = replace_once(
    app,
    "    evalGlow: false, sound: false, haptics: 'off', anim: 'reduced', density: 'compact',",
    "    evalGlow: false, sound: false, haptics: 'off', anim: 'reduced', density: 'compact',\n    boardTexture: 'off', boardContrast: 'soft', wallProfile: 'slim', wallPreview: 'subtle',\n    goalRows: 'off', moveMarkers: 'minimal', lastMoveStyle: 'off', uiFont: 'modern',",
    "minimal preset polish",
)

# Appearance tab: keep the first scan short and move tuning under disclosure.
old_look = r'''        <div class="row"><label>Text size</label>${seg('fs', [1, 1.12, 1.25], ['Normal', 'Large', 'Larger'])}</div>
        <div class="row"><label>Density</label>${seg('density', ['comfortable', 'compact'], ['Comfortable', 'Compact'])}</div>
      </div>
      <div class="card"><h4>MOTION</h4>
        <div class="row"><label>Animations</label>${seg('anim', ['full', 'reduced', 'off'], ['Full', 'Reduced', 'Off'])}</div>
        <div class="row"><label>Anim speed</label>${seg('animSpeed', [0.5, 1, 1.5], ['0.5x', '1x', '1.5x'])}</div>
      </div>`,'''
new_look = r'''        <div class="row"><label>Text size</label>${seg('fs', [1, 1.12, 1.25], ['Normal', 'Large', 'Larger'])}</div>
        <details class="advSettings"><summary>Advanced appearance</summary><div class="advBody">
          <div class="row"><label>Interface font</label>${seg('uiFont', ['modern', 'technical'], ['Modern', 'Technical'])}</div>
          <div class="row"><label>Density</label>${seg('density', ['comfortable', 'compact'], ['Comfortable', 'Compact'])}</div>
          <div class="row"><label>Animations</label>${seg('anim', ['full', 'reduced', 'off'], ['Full', 'Reduced', 'Off'])}</div>
          <div class="row"><label>Anim speed</label>${seg('animSpeed', [0.5, 1, 1.5], ['0.5x', '1x', '1.5x'])}</div>
        </div></details>
      </div>`,'''
app = replace_once(app, old_look, new_look, "appearance progressive disclosure")

old_board = r'''        <div class="row"><label>Pawn size</label>${seg('pawnSize', ['small', 'regular', 'large'], ['S', 'M', 'L'])}</div>
        <div class="row"><label>Pawn shadow</label>${seg('pawnShadow', ['off', 'soft', 'deep'], ['Off', 'Soft', 'Deep'])}</div>
        <div class="row"><label>Distinct shapes</label>${seg('distinctShapes', [true, false], ['On', 'Off'])}</div>
        <div class="row"><label>Frame</label>${seg('frame', ['none', 'hairline', 'gilded', 'beveled'], ['None', 'Hairline', 'Gilded', 'Beveled'])}</div>
        <div class="row"><label>Wall finish</label>${seg('wallFinish', ['flat', 'beveled', 'glossy', 'etched'], ['Flat', 'Beveled', 'Glossy', 'Etched'])}</div>
        <div class="row"><label>Cell surface</label>${seg('cellSep', ['grooves', 'flat', 'inlaid'], ['Grooves', 'Flat', 'Inlaid'])}</div>
        <div class="row"><label>Board scale</label>${seg('boardScale', [0.88, 0.94, 1], ['88%', '94%', '100%'])}</div>
      </div>
      <div class="card"><h4>MARKS &amp; OVERLAYS</h4>
        <div class="row"><label>Coordinates</label>${seg('coords', ['off', 'edges', 'all'], ['Off', 'Edges', 'All cells'])}</div>
        <div class="row"><label>Legal moves (dots)</label>${seg('dots', [true, false], ['On', 'Off'])}</div>
        <div class="row"><label>Shortest paths</label>${seg('paths', [false, true], ['Off', 'On'])}</div>
        <div class="row"><label>Highlight last move</label>${seg('lastMove', [true, false], ['On', 'Off'])}</div>
        <div class="row"><label>Evaluation bar</label>${seg('evalBar', [true, false], ['On', 'Off'])}</div>
      </div>`,'''
new_board = r'''        <details class="advSettings"><summary>Advanced board appearance</summary><div class="advBody">
          <div class="row"><label>Board texture</label>${seg('boardTexture', ['off', 'subtle', 'natural'], ['Off', 'Subtle', 'Natural'])}</div>
          <div class="row"><label>Board contrast</label>${seg('boardContrast', ['soft', 'standard', 'strong'], ['Soft', 'Standard', 'Strong'])}</div>
          <div class="row"><label>Pawn size</label>${seg('pawnSize', ['small', 'regular', 'large'], ['S', 'M', 'L'])}</div>
          <div class="row"><label>Pawn shadow</label>${seg('pawnShadow', ['off', 'soft', 'deep'], ['Off', 'Soft', 'Deep'])}</div>
          <div class="row"><label>Distinct shapes</label>${seg('distinctShapes', [true, false], ['On', 'Off'])}</div>
          <div class="row"><label>Frame</label>${seg('frame', ['none', 'hairline', 'gilded', 'beveled'], ['None', 'Hairline', 'Gilded', 'Beveled'])}</div>
          <div class="row"><label>Wall finish</label>${seg('wallFinish', ['flat', 'beveled', 'glossy', 'etched'], ['Flat', 'Soft relief', 'Glossy', 'Etched'])}</div>
          <div class="row"><label>Wall profile</label>${seg('wallProfile', ['slim', 'standard', 'bold'], ['Slim', 'Standard', 'Bold'])}</div>
          <div class="row"><label>Wall preview</label>${seg('wallPreview', ['subtle', 'normal', 'strong'], ['Subtle', 'Normal', 'Strong'])}</div>
          <div class="row"><label>Cell surface</label>${seg('cellSep', ['grooves', 'flat', 'inlaid'], ['Grooves', 'Flat', 'Inlaid'])}</div>
          <div class="row"><label>Board scale</label>${seg('boardScale', [0.88, 0.94, 1], ['88%', '94%', '100%'])}</div>
        </div></details>
      </div>
      <div class="card"><h4>MARKS &amp; OVERLAYS</h4>
        <div class="row"><label>Coordinates</label>${seg('coords', ['off', 'edges', 'all'], ['Off', 'Edges', 'All cells'])}</div>
        <div class="row"><label>Shortest paths</label>${seg('paths', [false, true], ['Off', 'On'])}</div>
        <div class="row"><label>Evaluation bar</label>${seg('evalBar', [true, false], ['On', 'Off'])}</div>
        <details class="advSettings"><summary>Advanced markers</summary><div class="advBody">
          <div class="row"><label>Legal moves</label>${seg('dots', [true, false], ['On', 'Off'])}</div>
          <div class="row"><label>Move markers</label>${seg('moveMarkers', ['ring', 'dot', 'minimal'], ['Ring', 'Dot', 'Minimal'])}</div>
          <div class="row"><label>Goal rows</label>${seg('goalRows', ['off', 'subtle', 'clear'], ['Off', 'Subtle', 'Clear'])}</div>
          <div class="row"><label>Last move</label>${seg('lastMoveStyle', ['off', 'subtle', 'clear'], ['Off', 'Subtle', 'Clear'])}</div>
        </div></details>
      </div>`,'''
app = replace_once(app, old_board, new_board, "board progressive disclosure")

app = replace_once(
    app,
    "  ds.coords = S.coords;\n  ds.frame = S.frame; ds.wallFinish = S.wallFinish; ds.cellSep = S.cellSep;",
    "  ds.coords = S.coords;\n  ds.uiFont = S.uiFont || 'modern';\n  ds.boardTexture = S.boardTexture || 'subtle'; ds.boardContrast = S.boardContrast || 'standard';\n  ds.wallProfile = S.wallProfile || 'standard'; ds.wallPreview = S.wallPreview || 'normal';\n  ds.goalRows = S.goalRows || 'subtle'; ds.moveMarkers = S.moveMarkers || 'ring';\n  ds.lastMoveStyle = S.lastMoveStyle || 'subtle';\n  S.lastMove = ds.lastMoveStyle !== 'off';\n  ds.frame = S.frame; ds.wallFinish = S.wallFinish; ds.cellSep = S.cellSep;",
    "apply microsetting datasets",
)
APP.write_text(app, encoding="utf-8")


# ---------------------------------------------------------------------------
# board.js: tiny renderer refinements only; hit testing stays untouched
# ---------------------------------------------------------------------------
board = BOARD.read_text(encoding="utf-8")
board = replace_once(
    board,
    "    if ((ds.board || '') === 'marble') {",
    "    if ((ds.board || '') === 'marble' && (ds.boardTexture || 'subtle') !== 'off') {",
    "marble texture setting",
)
board = replace_once(
    board,
    "      const bandX = bx / 2, bandY = (bx + bw + S) / 2;",
    "      // Snap the two margin centres to half pixels. At common DPRs this\n      // keeps the small mono glyphs visually centred instead of one edge\n      // looking a fraction heavier.\n      const bandX = Math.round(bx / 2) + .5;\n      const bandY = Math.round((bx + bw + S) / 2) + .5;",
    "coordinate pixel snapping",
)
board = replace_once(
    board,
    "      g.font = `600 ${Math.max(10, 0.235 * C)}px 'JetBrains Mono', monospace`;",
    "      g.font = `500 ${Math.max(10.5, 0.228 * C)}px 'JetBrains Mono', monospace`;",
    "coordinate typography",
)

old_last = r'''        g.save();
        g.strokeStyle = this.css('--gold'); g.lineWidth = 1.5; g.globalAlpha = .9;
        this.rr(g, rc.x - 1.75, rc.y - 1.75, rc.w + 3.5, rc.h + 3.5, 3.5); g.stroke();
        g.restore();'''
new_last = r'''        g.save();
        const lm = this.ds().lastMoveStyle || 'subtle';
        g.strokeStyle = this.css('--gold');
        g.lineWidth = lm === 'clear' ? 1.75 : 1.15;
        g.globalAlpha = lm === 'clear' ? .92 : .58;
        const pad = lm === 'clear' ? 1.9 : 1.45;
        this.rr(g, rc.x - pad, rc.y - pad, rc.w + 2 * pad, rc.h + 2 * pad, 3.5); g.stroke();
        g.restore();'''
board = replace_once(board, old_last, new_last, "last move strength")

board = replace_once(
    board,
    "      g.save(); g.globalAlpha = 0.35;\n      this.drawWall(g, h.o, h.r, h.c, true);",
    "      const wp = this.ds().wallPreview || 'normal';\n      g.save(); g.globalAlpha = wp === 'strong' ? .52 : wp === 'subtle' ? .24 : .35;\n      this.drawWall(g, h.o, h.r, h.c, true);",
    "wall hover preview strength",
)

old_markers = r'''        g.save();
        g.globalAlpha = .38;
        g.beginPath(); g.arc(ctr.x, ctr.y, 0.16 * C, 0, 7);
        if (jump) {
          g.lineWidth = Math.max(2, 0.045 * C);
          g.strokeStyle = this.dotColor(); g.stroke();
        } else {
          g.fillStyle = this.dotColor(); g.fill();
        }
        g.restore();'''
new_markers = r'''        g.save();
        const marker = this.ds().moveMarkers || 'ring';
        const col = this.dotColor();
        if (marker === 'dot' && !jump) {
          g.globalAlpha = .34;
          g.beginPath(); g.arc(ctr.x, ctr.y, 0.12 * C, 0, 7);
          g.fillStyle = col; g.fill();
        } else {
          const minimal = marker === 'minimal';
          g.globalAlpha = minimal ? .28 : .44;
          g.beginPath(); g.arc(ctr.x, ctr.y, (minimal ? .105 : .155) * C, 0, 7);
          g.lineWidth = Math.max(minimal ? 1.2 : 1.6, (minimal ? .026 : .036) * C);
          g.strokeStyle = col; g.stroke();
        }
        g.restore();'''
board = replace_once(board, old_markers, new_markers, "move marker styles")

old_wall_geom = r'''    const inf = Math.max(1.5, this.G * 0.36);
    // Along its length: half a corridor past each end. The slot stops at the
    // cell edges, so two walls in line stopped short of the crossing between
    // them and left a gap. Half a corridor each takes both to the exact centre
    // of that crossing, where they meet and read as one continuous rail. The
    // ends still land inside the play area: at the outermost slot the
    // extension reaches the frame edge and no further.
    const ext = this.G / 2;'''
new_wall_geom = r'''    const profile = this.ds().wallProfile || 'standard';
    const profileScale = profile === 'slim' ? .27 : profile === 'bold' ? .41 : .33;
    const inf = Math.max(1.4, this.G * profileScale);
    // Extend a fraction beyond the mathematical crossing centre. The overlap
    // is visual only (wallRect still owns hit testing) and removes antialias
    // hairlines where collinear rails or legal T/L junctions meet.
    const ext = this.G * .54;'''
board = replace_once(board, old_wall_geom, new_wall_geom, "wall profile and joints")

# Make the default relief more matte without flattening the material completely.
board = replace_once(board, "      g.shadowBlur = 5;\n      g.shadowOffsetY = 2;", "      g.shadowBlur = 4;\n      g.shadowOffsetY = 1.5;", "wall shadow polish")
board = replace_once(board, "      g.fillStyle = 'rgba(255,255,255,.28)';", "      g.fillStyle = 'rgba(255,255,255,.18)';", "wall highlight polish")
board = replace_once(board, "      g.fillStyle = 'rgba(0,0,0,.20)';", "      g.fillStyle = 'rgba(0,0,0,.16)';", "wall shade polish")
board = replace_once(board, "'rgba(255,255,255,.09)' : 'rgba(0,0,0,.06)'", "'rgba(255,255,255,.06)' : 'rgba(0,0,0,.045)'", "wall grain polish")
BOARD.write_text(board, encoding="utf-8")

print("Applied GUI micro-polish to style.html, app.js and board.js")
