from pathlib import Path
import re


def replace_once(text, old, new, label):
    n = text.count(old)
    if n != 1:
        raise SystemExit(f"{label}: expected 1 exact match, found {n}")
    return text.replace(old, new, 1)


def regex_once(text, pattern, repl, label, flags=0):
    out, n = re.subn(pattern, repl, text, count=1, flags=flags)
    if n != 1:
        raise SystemExit(f"{label}: expected 1 regex match, found {n}")
    return out


STYLE = Path("gui_web/style.html")
BOARD = Path("gui_web/board.js")
TEST = Path("gui_web/test_micro_polish_semantics.py")

# ---------------------------------------------------------------------------
# style.html: material texture, board contrast and goal-row strength are no
# longer DOM post-processing effects. QBoard owns them so screen + PNG + SVG
# share the same semantics. Keep the already-final 12 px eval readout.
# ---------------------------------------------------------------------------
style = STYLE.read_text(encoding="utf-8")
old_fx = r'''/* A barely-visible material layer prevents flat themes from looking synthetic.
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
'''
new_fx = r'''/* Board material, cell/groove contrast and goal-row strength are painted by
   QBoard itself. Keeping them out of DOM filters/pseudo-elements means they
   never post-process pawns or walls, and exports use the same rendering path. */
'''
style = replace_once(style, old_fx, new_fx, "renderer-owned board effects CSS")
if '#boardZone > #evalWrap #evalNum{font-size:12px;bottom:-18px}' not in style:
    raise SystemExit("eval readout: final 12px override missing")
STYLE.write_text(style, encoding="utf-8")

# ---------------------------------------------------------------------------
# board.js: shared contrast + deterministic material primitives.
# ---------------------------------------------------------------------------
board = BOARD.read_text(encoding="utf-8")
helper_anchor = """}\n\n// Ease in out, cubic. Used by the pawn slide.\n"""
helpers = r''' }

// Board contrast belongs to the substrate only. Transform the three board
// colours before painting instead of filtering the completed canvas, so pawns,
// walls, marks and text keep their authored colours.
function qrBoardContrastColor(color, mode) {
  if (!mode || mode === 'standard') return color;
  const m = /^#([0-9a-f]{6})$/i.exec(String(color || '').trim());
  if (!m) return color;
  const n = parseInt(m[1], 16);
  let rgb = [(n >> 16) & 255, (n >> 8) & 255, n & 255];
  const mean = (rgb[0] + rgb[1] + rgb[2]) / 3;
  const sat = mode === 'strong' ? 1.015 : .99;
  const contrast = mode === 'strong' ? 1.07 : .96;
  rgb = rgb.map(v => mean + (v - mean) * sat)
           .map(v => 128 + (v - 128) * contrast)
           .map(v => Math.max(0, Math.min(255, Math.round(v))));
  return '#' + rgb.map(v => v.toString(16).padStart(2, '0')).join('');
}

function qrTextureStrength(mode) {
  return mode === 'off' ? 0 : mode === 'natural' ? 1 : .52;
}

// One deterministic primitive generator feeds both Canvas and SVG. Textures
// therefore describe the material (wood grain, marble veins, paper fibre,
// mineral streaks) rather than a generic overlay laid over the whole UI.
function qrTexturePrimitives(theme, mode, bx, bw, C) {
  const s = qrTextureStrength(mode);
  if (!s) return [];
  const seeds = {
    wood: 0x574f4f44, walnut: 0x57414c4e, ivory: 0x49564f52,
    marble: 0x4d415242, parchment: 0x50415243, obsidian: 0x4f425349,
    slate: 0x534c4154, emerald: 0x454d4552, noir: 0x4e4f4952,
  };
  const rnd = qrMulberry32(seeds[theme] || 0x5a51554f);
  const out = [];
  const tone = bias => rnd() < (bias == null ? .5 : bias) ? 'dark' : 'light';
  const line = (x1, y1, x2, y2, width, t, alpha) =>
    out.push({ kind: 'line', x1, y1, x2, y2, width, tone: t, alpha });
  const dot = (x, y, radius, t, alpha) =>
    out.push({ kind: 'dot', x, y, radius, tone: t, alpha });

  if (theme === 'wood' || theme === 'walnut') {
    const count = theme === 'walnut' ? 34 : 28;
    const base = theme === 'walnut' ? .050 : .040;
    for (let i = 0; i < count; i++) {
      const y = bx + rnd() * bw;
      const drift = (rnd() - .5) * bw * .035;
      const a = base * s * (.70 + .30 * rnd());
      const t = tone(.58);
      line(bx - 2, y, bx + bw + 2, y + drift, .45 + rnd() * .75, t, a);
      if (rnd() > .70) {
        const off = (rnd() - .5) * C * .18;
        line(bx, y + off, bx + bw, y + drift + off,
             .35 + rnd() * .45, t, a * .55);
      }
    }
  } else if (theme === 'marble') {
    for (let v = 0; v < 18; v++) {
      let x = bx + rnd() * bw, y = bx - 4;
      const pts = [{ x, y }];
      while (y < bx + bw + 4) {
        y += C * (.42 + rnd() * .62);
        x += (rnd() - .5) * C * 1.15;
        pts.push({ x, y });
      }
      out.push({ kind: 'poly', points: pts, width: .55 + rnd() * 1.45,
                 tone: tone(.72), alpha: (.070 + rnd() * .035) * s });
    }
  } else if (theme === 'parchment' || theme === 'ivory') {
    const dots = theme === 'parchment' ? 105 : 62;
    const da = (theme === 'parchment' ? .030 : .018) * s;
    for (let i = 0; i < dots; i++)
      dot(bx + rnd() * bw, bx + rnd() * bw, .35 + rnd() * .85,
          tone(theme === 'parchment' ? .70 : .55), da * (.55 + .45 * rnd()));
    const fibres = theme === 'parchment' ? 22 : 10;
    for (let i = 0; i < fibres; i++) {
      const x = bx + rnd() * bw, y = bx + rnd() * bw;
      line(x, y, x + C * (.35 + rnd() * .65), y + (rnd() - .5) * C * .10,
           .35 + rnd() * .35, tone(.66), da * 1.25);
    }
  } else if (theme === 'slate' || theme === 'obsidian') {
    const count = theme === 'slate' ? 42 : 30;
    const a = (theme === 'slate' ? .034 : .026) * s;
    for (let i = 0; i < count; i++) {
      const x = bx + rnd() * bw, y = bx + rnd() * bw;
      const len = C * (.30 + rnd() * .85);
      line(x, y, x + len, y - len * (.18 + rnd() * .25),
           .45 + rnd() * .65, tone(.60), a * (.65 + .35 * rnd()));
    }
  } else if (theme === 'emerald') {
    for (let i = 0; i < 28; i++) {
      const x = bx + rnd() * bw, y = bx + rnd() * bw;
      line(x, y, x + (rnd() - .5) * C * .20, y + C * (.45 + rnd() * .85),
           .40 + rnd() * .55, tone(.62), .025 * s * (.6 + .4 * rnd()));
    }
  } else if (theme === 'noir') {
    for (let i = 0; i < 52; i++)
      dot(bx + rnd() * bw, bx + rnd() * bw, .30 + rnd() * .70,
          tone(.48), .016 * s * (.5 + .5 * rnd()));
  }
  return out;
}

// Ease in out, cubic. Used by the pawn slide.
'''
# Preserve the exact closing brace from qrMulberry32; helpers begins with it.
if helper_anchor not in board:
    raise SystemExit("helper insertion anchor missing")
board = board.replace(helper_anchor, helpers, 1)

# Insert the Canvas material painter immediately before paintStatic.
paint_anchor = """  paintStatic(g) {\n"""
paint_method = r'''  paintMaterialTexture(g, bx, bw) {
    const ds = this.ds();
    const parts = qrTexturePrimitives(ds.board || 'wood', ds.boardTexture || 'subtle',
                                      bx, bw, this.C);
    if (!parts.length) return;
    g.save();
    this.rr(g, bx, bx, bw, bw, 4); g.clip();
    g.lineCap = 'round'; g.lineJoin = 'round';
    for (const p of parts) {
      const col = p.tone === 'light' ? '#ffffff' : '#000000';
      g.globalAlpha = p.alpha;
      if (p.kind === 'dot') {
        g.fillStyle = col;
        g.beginPath(); g.arc(p.x, p.y, p.radius, 0, 7); g.fill();
      } else if (p.kind === 'poly') {
        if (!p.points.length) continue;
        g.strokeStyle = col; g.lineWidth = p.width;
        g.beginPath(); g.moveTo(p.points[0].x, p.points[0].y);
        for (let i = 1; i < p.points.length; i++) g.lineTo(p.points[i].x, p.points[i].y);
        g.stroke();
      } else {
        g.strokeStyle = col; g.lineWidth = p.width;
        g.beginPath(); g.moveTo(p.x1, p.y1); g.lineTo(p.x2, p.y2); g.stroke();
      }
    }
    g.restore();
  }

  paintStatic(g) {
'''
board = replace_once(board, paint_anchor, paint_method, "material texture painter insertion")

# Contrast is applied only to substrate colours.
board = replace_once(
    board,
    """    g.fillStyle = this.css('--groove');\n    this.rr(g, bx, bx, bw, bw, 4); g.fill();\n""",
    """    const boardContrast = ds.boardContrast || 'standard';\n    const groove = qrBoardContrastColor(this.css('--groove'), boardContrast);\n    g.fillStyle = groove;\n    this.rr(g, bx, bx, bw, bw, 4); g.fill();\n""",
    "groove contrast",
)
board = replace_once(
    board,
    """    const ca = this.css('--cell-a'), cb = this.css('--cell-b');\n""",
    """    const ca = qrBoardContrastColor(this.css('--cell-a'), boardContrast),\n          cb = qrBoardContrastColor(this.css('--cell-b'), boardContrast);\n""",
    "cell contrast",
)

# Put material under goal overlays, marks, walls and pawns.
board = replace_once(
    board,
    """    // Goal rows: a light neutral wash over rank 1 and rank 9, the two rows a\n""",
    """    this.paintMaterialTexture(g, bx, bw);\n\n    // Goal rows: a light neutral wash over rank 1 and rank 9, the two rows a\n""",
    "material texture call",
)

old_goal = r'''    // Goal rows: a light neutral wash over rank 1 and rank 9, the two rows a
    // game can end on. Drawn over the cells so it follows their shape, and
    // under everything else so a pawn or a wall still reads normally.
    {
      const wash = this.css('--goal-wash');
      if (wash) {
        g.fillStyle = wash;
        for (const r of [0, 8]) for (let c = 0; c < 9; c++) {
          const p = this.cellXY(r, c);
          if (cellSep === 'grooves') { this.rr(g, p.x, p.y, C, C, rad); g.fill(); }
          else g.fillRect(p.x - G / 2, p.y - G / 2, C + G, C + G);
        }
      }
    }
'''
new_goal = r'''    // Goal rows are renderer-owned too: Off really means no wash and no edge.
    const goalMode = ds.goalRows || 'subtle';
    const goalAlpha = goalMode === 'clear' ? .21 : goalMode === 'off' ? 0 : .12;
    if (goalAlpha > 0) {
      g.fillStyle = `rgba(120,122,132,${goalAlpha})`;
      for (const r of [0, 8]) for (let c = 0; c < 9; c++) {
        const p = this.cellXY(r, c);
        if (cellSep === 'grooves') { this.rr(g, p.x, p.y, C, C, rad); g.fill(); }
        else g.fillRect(p.x - G / 2, p.y - G / 2, C + G, C + G);
      }
    }
'''
board = replace_once(board, old_goal, new_goal, "goal-row wash semantics")

# Remove the old marble-only painter; the shared material generator replaces it.
old_marble = r'''    // Carrara marble signature theme: deterministic veins baked into the
    // static layer, so they cost nothing per frame.
    if ((ds.board || '') === 'marble' && (ds.boardTexture || 'subtle') !== 'off') {
      const rnd = qrMulberry32(0xCAFE);
      g.save();
      g.beginPath(); g.rect(bx, bx, bw, bw); g.clip();
      for (let v = 0; v < 26; v++) {
        g.strokeStyle = rnd() > .5 ? 'rgba(120,125,140,.16)' : 'rgba(70,74,88,.12)';
        g.lineWidth = .6 + rnd() * 1.8;
        let x = M + rnd() * bw, y = M - 4;
        g.beginPath(); g.moveTo(x, y);
        while (y < S - M + 4) {
          y += 14 + rnd() * 30;
          x += (rnd() - .5) * 46;
          g.quadraticCurveTo(x + (rnd() - .5) * 24, y - 12, x, y);
        }
        g.stroke();
      }
      g.restore();
    }

'''
board = replace_once(board, old_marble, "", "old marble texture removal")

old_edges = r'''    // Goal edges: a quiet 2px line just inside the top and bottom play edges.
    g.globalAlpha = .30;
    g.fillStyle = this.flipped ? this.css('--p0') : this.css('--p1');
    g.fillRect(bx, bx, bw, 2);
    g.fillStyle = this.flipped ? this.css('--p1') : this.css('--p0');
    g.fillRect(bx, bx + bw - 2, bw, 2);
    g.globalAlpha = 1;
'''
new_edges = r'''    // Goal edges follow the same control. Off removes every goal-row cue.
    if (goalMode !== 'off') {
      g.globalAlpha = goalMode === 'clear' ? .40 : .30;
      g.fillStyle = this.flipped ? this.css('--p0') : this.css('--p1');
      g.fillRect(bx, bx, bw, 2);
      g.fillStyle = this.flipped ? this.css('--p1') : this.css('--p0');
      g.fillRect(bx, bx + bw - 2, bw, 2);
      g.globalAlpha = 1;
    }
'''
board = replace_once(board, old_edges, new_edges, "goal-edge off semantics")

# Last pawn move: a thin halo around the body, not a filled tile.
old_last_pawn = r'''      if (this.lastMove.type === 'pawn') {
        // No mark on a pawn's square. The filled gold tile read as a second
        // piece on the board and fought with the pawn standing on it. A wall
        // still takes its gold rim below, because a wall has no other way to
        // say it was the move just played.
      } else {
'''
new_last_pawn = r'''      if (this.lastMove.type === 'pawn') {
        const lm = this.ds().lastMoveStyle || 'subtle';
        if (lm !== 'off') {
          const ctr = this.cellCenter(this.lastMove.r, this.lastMove.c);
          const sizeMul = { small: .85, large: 1.15 }[this.ds().pawnSize] || 1;
          const haloR = (.30 * sizeMul + (lm === 'clear' ? .070 : .058)) * C;
          g.save();
          g.beginPath(); g.arc(ctr.x, ctr.y, haloR, 0, 7);
          g.strokeStyle = this.css('--gold');
          g.lineWidth = lm === 'clear' ? Math.max(1.5, .026 * C) : Math.max(1.0, .018 * C);
          g.globalAlpha = lm === 'clear' ? .88 : .52;
          g.stroke(); g.restore();
        }
      } else {
'''
board = replace_once(board, old_last_pawn, new_last_pawn, "last pawn halo")

# Active ghost obeys Wall Preview too; state colour/outline remain authoritative.
old_ghost_alpha = """    const st = gh.state;\n    const alpha = st === 'bad' ? .20 : st === 'pending' ? 1 : .60;\n"""
new_ghost_alpha = """    const st = gh.state;\n    const wp = this.ds().wallPreview || 'normal';\n    const alpha = wp === 'subtle'\n      ? (st === 'bad' ? .14 : st === 'pending' ? .90 : .45)\n      : wp === 'strong'\n        ? (st === 'bad' ? .28 : st === 'pending' ? 1 : .75)\n        : (st === 'bad' ? .20 : st === 'pending' ? 1 : .60);\n"""
board = replace_once(board, old_ghost_alpha, new_ghost_alpha, "active wall preview strength")

# SVG: use the same board geometry, contrast transform, texture primitives,
# goal-row semantics and wall-profile geometry as the Canvas renderer.
svg_pattern = re.compile(
    r"    const frame = cssOf\('--frame'\), groove = cssOf\('--groove'\);\n"
    r".*?"
    r"    for \(let pl = 0; pl < 2; pl\+\+\) \{\n",
    re.S,
)
svg_repl = r'''    const ds = document.documentElement.dataset;
    const contrast = ds.boardContrast || 'standard';
    const frame = cssOf('--frame');
    const groove = qrBoardContrastColor(cssOf('--groove'), contrast);
    const ca = qrBoardContrastColor(cssOf('--cell-a'), contrast);
    const cb = qrBoardContrastColor(cssOf('--cell-b'), contrast);
    const wall = cssOf('--wall'), edge = cssOf('--wall-edge');
    const gold = cssOf('--gold'), coord = cssOf('--coord');
    const coordsMode = opts.coords === false ? 'off' : (ds.coords || 'edges');
    const k = coordsMode !== 'off' ? .46 : .20;
    const G = Math.max(4, Math.min(10, .145 * (size / 10.6)));
    const C = (size - 8 * G) / (9 + 2 * k), M = k * C, U = C + G;
    const bx = M - G / 2, bw = 9 * U;
    let b = `<svg xmlns="http://www.w3.org/2000/svg" width="${size}" height="${size}" viewBox="0 0 ${size} ${size}">`;
    b += `<defs><clipPath id="zqPlayClip"><rect x="${bx.toFixed(1)}" y="${bx.toFixed(1)}" width="${bw.toFixed(1)}" height="${bw.toFixed(1)}" rx="4"/></clipPath></defs>`;
    if (!opts.transparent) b += `<rect width="${size}" height="${size}" fill="${esc(cssOf('--bg'))}"/>`;
    b += `<rect width="${size}" height="${size}" rx="14" fill="${esc(frame)}"/>`;
    b += `<rect x="${bx.toFixed(1)}" y="${bx.toFixed(1)}" width="${bw.toFixed(1)}" height="${bw.toFixed(1)}" rx="4" fill="${esc(groove)}"/>`;
    for (let r = 0; r < 9; r++) for (let c = 0; c < 9; c++)
      b += `<rect x="${(M + c * U).toFixed(1)}" y="${(M + r * U).toFixed(1)}" width="${C.toFixed(1)}" height="${C.toFixed(1)}" rx="${(C * .10).toFixed(1)}" fill="${esc(((r + c) & 1) ? cb : ca)}"/>`;

    const textureMode = ds.boardTexture || 'subtle';
    const texture = qrTexturePrimitives(ds.board || 'wood', textureMode, bx, bw, C);
    if (texture.length) {
      b += `<g data-zq-texture="${esc((ds.board || 'wood') + ':' + textureMode)}" clip-path="url(#zqPlayClip)" stroke-linecap="round" stroke-linejoin="round">`;
      for (const p of texture) {
        const col = p.tone === 'light' ? '#ffffff' : '#000000';
        if (p.kind === 'dot') {
          b += `<circle cx="${p.x.toFixed(2)}" cy="${p.y.toFixed(2)}" r="${p.radius.toFixed(2)}" fill="${col}" fill-opacity="${p.alpha.toFixed(4)}"/>`;
        } else if (p.kind === 'poly') {
          const pts = p.points.map(q => `${q.x.toFixed(2)},${q.y.toFixed(2)}`).join(' ');
          b += `<polyline points="${pts}" fill="none" stroke="${col}" stroke-opacity="${p.alpha.toFixed(4)}" stroke-width="${p.width.toFixed(2)}"/>`;
        } else {
          b += `<line x1="${p.x1.toFixed(2)}" y1="${p.y1.toFixed(2)}" x2="${p.x2.toFixed(2)}" y2="${p.y2.toFixed(2)}" stroke="${col}" stroke-opacity="${p.alpha.toFixed(4)}" stroke-width="${p.width.toFixed(2)}"/>`;
        }
      }
      b += `</g>`;
    }

    const goalMode = ds.goalRows || 'subtle';
    if (goalMode !== 'off') {
      const goalAlpha = goalMode === 'clear' ? .21 : .12;
      b += `<g data-zq-goal-rows="${goalMode}">`;
      for (const r of [0, 8]) for (let c = 0; c < 9; c++)
        b += `<rect x="${(M + c * U).toFixed(1)}" y="${(M + r * U).toFixed(1)}" width="${C.toFixed(1)}" height="${C.toFixed(1)}" rx="${(C * .10).toFixed(1)}" fill="#787a84" fill-opacity="${goalAlpha}"/>`;
      const top = this.flipped ? cssOf('--p0') : cssOf('--p1');
      const bottom = this.flipped ? cssOf('--p1') : cssOf('--p0');
      const edgeAlpha = goalMode === 'clear' ? .40 : .30;
      b += `<rect x="${bx.toFixed(1)}" y="${bx.toFixed(1)}" width="${bw.toFixed(1)}" height="2" fill="${esc(top)}" fill-opacity="${edgeAlpha}"/>`;
      b += `<rect x="${bx.toFixed(1)}" y="${(bx + bw - 2).toFixed(1)}" width="${bw.toFixed(1)}" height="2" fill="${esc(bottom)}" fill-opacity="${edgeAlpha}"/>`;
      b += `</g>`;
    }

    const profile = ds.wallProfile || 'standard';
    const profileScale = profile === 'slim' ? .27 : profile === 'bold' ? .41 : .33;
    const inf = Math.max(1.4, G * profileScale), ext = G * .54;
    b += `<g data-zq-wall-profile="${profile}">`;
    for (let r = 0; r < 8; r++) for (let c = 0; c < 8; c++) {
      if (this.wallH[r * 8 + c]) {
        const x = M + c * U - ext, y = M + (r + 1) * U - G - inf;
        b += `<rect x="${x.toFixed(1)}" y="${y.toFixed(1)}" width="${(2 * C + G + 2 * ext).toFixed(1)}" height="${(G + 2 * inf).toFixed(1)}" rx="2" fill="${esc(wall)}" stroke="${esc(edge)}"/>`;
      }
      if (this.wallV[r * 8 + c]) {
        const x = M + (c + 1) * U - G - inf, y = M + r * U - ext;
        b += `<rect x="${x.toFixed(1)}" y="${y.toFixed(1)}" width="${(G + 2 * inf).toFixed(1)}" height="${(2 * C + G + 2 * ext).toFixed(1)}" rx="2" fill="${esc(wall)}" stroke="${esc(edge)}"/>`;
      }
    }
    b += `</g>`;
    for (let pl = 0; pl < 2; pl++) {
'''
board, n = svg_pattern.subn(svg_repl, board, count=1)
if n != 1:
    raise SystemExit(f"SVG renderer block: expected 1 match, found {n}")

# SVG pawn size and coordinates follow the on-screen settings/geometry.
board = replace_once(
    board,
    """      const R = .30 * C;\n""",
    """      const pawnScale = { small: .85, large: 1.15 }[ds.pawnSize] || 1;\n      const R = .30 * C * pawnScale;\n""",
    "SVG pawn size",
)
old_svg_coords = r'''    if (opts.coords !== false) {
      b += `<g fill="${esc(coord)}" font-family="monospace" font-size="${(C * .15).toFixed(0)}">`;
      for (let c = 0; c < 9; c++) {
        const x = M + c * U + C / 2;
        b += `<text x="${x.toFixed(1)}" y="${size - M / 2}" text-anchor="middle">${'abcdefghi'[c]}</text>`;
      }
      for (let r = 0; r < 9; r++) {
        const y = M + r * U + C / 2;
        b += `<text x="${M / 2}" y="${y.toFixed(1)}" text-anchor="middle">${this.flipped ? r + 1 : 9 - r}</text>`;
      }
      b += `</g>`;
    }
'''
new_svg_coords = r'''    if (opts.coords !== false) {
      const bandX = bx / 2, bandY = (bx + bw + size) / 2;
      b += `<g fill="${esc(coord)}" font-family="'JetBrains Mono', monospace" font-size="${Math.max(10.5, C * .228).toFixed(1)}" font-weight="500" dominant-baseline="middle">`;
      for (let c = 0; c < 9; c++) {
        const x = M + c * U + C / 2;
        b += `<text x="${x.toFixed(1)}" y="${bandY.toFixed(1)}" text-anchor="middle">${'abcdefghi'[c]}</text>`;
      }
      for (let r = 0; r < 9; r++) {
        const y = M + r * U + C / 2;
        b += `<text x="${bandX.toFixed(1)}" y="${y.toFixed(1)}" text-anchor="middle">${this.flipped ? r + 1 : 9 - r}</text>`;
      }
      b += `</g>`;
    }
'''
board = replace_once(board, old_svg_coords, new_svg_coords, "SVG coordinate parity")
BOARD.write_text(board, encoding="utf-8")

# ---------------------------------------------------------------------------
# Targeted browser regression: verify semantic ownership, canvas behaviour,
# SVG/PNG propagation, active ghost strength and the 12 px eval readout.
# ---------------------------------------------------------------------------
TEST.write_text(r'''"""Targeted regression for the semantic micro-polish pass.
Run from gui_web/: python test_micro_polish_semantics.py
"""
import os
import subprocess
import sys
import time

from playwright.sync_api import sync_playwright

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    srv = subprocess.Popen([sys.executable, "dev_server.py", "8213"], cwd=HERE,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(1.0)
    failed = []
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch()
            page = browser.new_page(viewport={"width": 1280, "height": 900})
            errors = []
            page.on("pageerror", lambda e: errors.append(str(e).split("\n")[0]))
            page.goto("http://127.0.0.1:8213/style.html")
            page.wait_for_timeout(2600)

            def check(name, cond):
                print(("ok: " if cond else "FAIL: ") + name)
                if not cond:
                    failed.append(name)

            def board_hash():
                return page.evaluate("document.getElementById('board').toDataURL()")

            check("boot", "Loading" not in (page.text_content("#status") or ""))
            check("eval readout final size is 12px",
                  page.evaluate("getComputedStyle(document.getElementById('evalNum')).fontSize") == "12px")

            # Freeze optional dynamic overlays so static-effect hashes are stable.
            page.evaluate("setOpt('paths', false); setOpt('dots', false); setOpt('lastMoveStyle', 'off'); setOpt('board', 'wood')")
            page.wait_for_timeout(120)

            # Contrast must repaint the Canvas without a CSS filter on the completed board.
            page.evaluate("setOpt('boardTexture', 'off'); setOpt('goalRows', 'off'); setOpt('boardContrast', 'standard')")
            page.wait_for_timeout(100)
            c0 = board_hash()
            page.evaluate("setOpt('boardContrast', 'strong')")
            page.wait_for_timeout(100)
            c1 = board_hash()
            css_filter = page.evaluate("getComputedStyle(document.getElementById('board')).filter")
            check("board contrast repaints substrate", c0 != c1)
            check("board contrast is not a whole-canvas CSS filter", css_filter == "none")

            # Texture now lives inside QBoard; Off/Subtle/Natural must be distinct.
            page.evaluate("setOpt('boardContrast', 'standard'); setOpt('boardTexture', 'off')")
            page.wait_for_timeout(100)
            t0 = board_hash()
            page.evaluate("setOpt('boardTexture', 'subtle')")
            page.wait_for_timeout(100)
            t1 = board_hash()
            page.evaluate("setOpt('boardTexture', 'natural')")
            page.wait_for_timeout(100)
            t2 = board_hash()
            pseudo = page.evaluate("getComputedStyle(document.getElementById('boardWrap'),'::after').content")
            check("texture off/subtle/natural all differ", len({t0, t1, t2}) == 3)
            check("texture is not a boardWrap pseudo overlay", pseudo in ("none", "normal"))

            # Off means no goal wash AND no coloured goal-edge marker.
            page.evaluate("setOpt('boardTexture', 'off'); setOpt('goalRows', 'off')")
            page.wait_for_timeout(100)
            g0 = board_hash()
            svg_off = page.evaluate("window.__qb.toSVG({coords:true})")
            page.evaluate("setOpt('goalRows', 'subtle')")
            page.wait_for_timeout(100)
            g1 = board_hash()
            svg_subtle = page.evaluate("window.__qb.toSVG({coords:true})")
            check("goal rows repaint", g0 != g1)
            check("goal rows Off removes SVG wash and edges", 'data-zq-goal-rows=' not in svg_off)
            check("goal rows Subtle exports", 'data-zq-goal-rows="subtle"' in svg_subtle)

            # A pawn last move gets a fine halo; Clear is stronger; Off removes it.
            page.evaluate("setOpt('goalRows', 'off'); setOpt('lastMoveStyle', 'off')")
            page.evaluate("window.__qb.lastMove=null; window.__qb.render()")
            lm0 = board_hash()
            page.evaluate("setOpt('lastMoveStyle', 'subtle'); window.__qb.lastMove={type:'pawn',r:8,c:4}; window.__qb.render()")
            lm1 = board_hash()
            page.evaluate("setOpt('lastMoveStyle', 'clear'); window.__qb.lastMove={type:'pawn',r:8,c:4}; window.__qb.render()")
            lm2 = board_hash()
            check("pawn last-move halo appears", lm0 != lm1)
            check("pawn last-move Clear differs from Subtle", lm1 != lm2)

            # Wall Preview now controls active ghosts as well as passive hover.
            ghost_hashes = []
            for mode in ("subtle", "normal", "strong"):
                page.evaluate(f"setOpt('wallPreview','{mode}'); window.__qb.ghost={{o:0,r:3,c:3,state:'ok'}}; window.__qb.render()")
                page.wait_for_timeout(50)
                ghost_hashes.append(board_hash())
            page.evaluate("window.__qb.ghost=null; window.__qb.render()")
            check("active ghost obeys all three Wall Preview strengths", len(set(ghost_hashes)) == 3)

            # PNG uses QBoard renderer; SVG consumes the same deterministic primitives.
            page.evaluate("setOpt('boardTexture','off'); setOpt('boardContrast','standard'); setOpt('goalRows','off'); setOpt('wallProfile','slim')")
            png0 = page.evaluate("window.__qb.renderExport({size:640,coords:true}).toDataURL()")
            page.evaluate("setOpt('boardTexture','natural'); setOpt('boardContrast','strong'); setOpt('goalRows','clear'); setOpt('wallProfile','bold')")
            png1 = page.evaluate("window.__qb.renderExport({size:640,coords:true}).toDataURL()")
            svg = page.evaluate("window.__qb.toSVG({size:640,coords:true})")
            check("PNG export carries renderer-owned effects", png0 != png1)
            check("SVG export carries material texture", 'data-zq-texture="wood:natural"' in svg)
            check("SVG export carries clear goal rows", 'data-zq-goal-rows="clear"' in svg)
            check("SVG export carries wall profile", 'data-zq-wall-profile="bold"' in svg)
            check("SVG export still has coordinates", '<text' in svg and 'abcdefghi' not in svg)
            check("zero page errors", not errors)

            browser.close()
    finally:
        srv.terminate()
        try:
            srv.wait(timeout=2)
        except subprocess.TimeoutExpired:
            srv.kill()

    print(f"\n{len(failed)} failed")
    if failed:
        for f in failed:
            print(" -", f)
        return 1
    print("RESULT: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
''', encoding="utf-8")

print("Applied semantic GUI polish to style.html + board.js and wrote targeted test")
