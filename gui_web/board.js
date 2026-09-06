// board.js -- QBoard: the canvas board for Zquoridor.
// One <canvas>, DPR aware, two layers. The static layer (frame, bed, cells,
// grooves, coordinates) is painted once into an offscreen canvas and blitted.
// The dynamic layer (walls, pawns, dots, ghosts, paths, overlays) redraws on
// every state change.
// Every colour comes from a CSS custom property read at draw time, so a theme
// switch needs no canvas palette. Neutral highlights and shadows use rgba.
// A requestAnimationFrame loop runs only while an animation is in flight.
'use strict';

// Deterministic PRNG for the marble vein texture (mulberry32). The same seed
// gives the same veins, so the static layer cache stays valid.
function qrMulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
 }

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
function qrEase(t) {
  return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
}

const QR_MOVE_MS = 200;    // pawn slide duration
const QR_WALL_MS = 180;    // wall fade and scale in
const QR_WALL_RIM_MS = 900;  // gold rim decay after a wall lands

// Silhouette that side 1 takes when distinct shapes are on. Every entry maps
// to a different shape, so the two sides never share a silhouette.
// The default piece is a ball for BOTH sides. Distinct shapes marks side 1
// with a ring cut into the same sphere instead of swapping the silhouette for
// a crown, so the board never shows a piece that is not a pawn.
const QR_PAWN_ALT = {
  disc: 'discRing', discRing: 'disc', pillar: 'crown', crown: 'disc',
  rune: 'pawnChess', pawnChess: 'rune', beacon: 'pillar'
};

class QBoard {
  constructor(canvas, opts = {}) {
    this.cv = canvas;
    this.ctx = canvas.getContext('2d');
    this.onChange = opts.onChange || (() => {});
    this.fixedSide = opts.fixedSide || 0;   // export renders and mini previews
    // Game state mirrored from the engine, in display orientation. Player 0
    // sits at the bottom and moves up, unless the board is flipped.
    this.flipped = false;
    this.pawn = [];
    this.wallH = new Uint8Array(64);
    this.wallV = new Uint8Array(64);
    this.lastMove = null;
    this.dots = [];
    this.selected = -1;
    this.paths = null;
    this.ghost = null;
    this.ghostFrom = null;
    // Hover hit test result. Set by setHover, drawn as a faint wall preview.
    this.hover = null;
    // Analysis line preview: { walls:[{o,r,c}] engine coords, pawns:[engCell]
    // in step order, color }. Drawn under the pieces.
    this.linePreview = null;
    this.turn = 0;
    this.themeDirty = true;
    this._sideApplied = -1;
    // Animation state. Both are null when nothing animates.
    this._pawnAnim = [null, null];
    this._wallAnim = null;
    this._raf = 0;
    if (!this.fixedSide) new ResizeObserver(() => this.fit()).observe(canvas.parentElement);
    this.fit();
  }

  css(name) {
    // Detached canvases (export renders) have no parent element. Fall back to
    // the document root so token lookups keep working.
    const el = this.cv.parentElement || document.documentElement;
    return getComputedStyle(el).getPropertyValue(name).trim();
  }
  ds() { return document.documentElement.dataset; }

  fit() {
    let side;
    if (this.fixedSide) {
      side = this.fixedSide;
      this.cv.width = Math.round(side * (window.devicePixelRatio || 1));
      this.cv.height = Math.round(side * (window.devicePixelRatio || 1));
      this.cv.style.width = side + 'px';
      this.cv.style.height = side + 'px';
    } else {
      // Measure the layout zone, not the wrapper. The wrapper wraps the
      // canvas, so it would collapse to its own content size.
      const zone = this.cv.closest('#boardZone') ||
                   this.cv.parentElement.parentElement ||
                   this.cv.parentElement;
      // clientWidth counts the zone's own padding, and the evaluation strip
      // takes a share of the row. Subtract both, or the board is sized larger
      // than the space it actually has and spills over its neighbours.
      const cs = getComputedStyle(zone);
      const px = (v) => parseFloat(v) || 0;
      let zw = (zone.clientWidth || 320) - px(cs.paddingLeft) - px(cs.paddingRight);
      const zh = (zone.clientHeight || 320) - px(cs.paddingTop) - px(cs.paddingBottom);
      const gap = px(cs.columnGap) || px(cs.gap);
      const wrap0 = this.cv.parentElement;
      for (const ch of zone.children) {
        if (ch === wrap0) continue;
        const chs = getComputedStyle(ch);
        if (chs.position === 'absolute' || chs.position === 'fixed') continue;
        if (chs.display === 'none') continue;
        zw -= ch.getBoundingClientRect().width + gap;
      }
      const scale = Math.max(0.8, Math.min(1, parseFloat(this.ds().boardScale) || 1));
      // The floor must stay below what a phone in landscape can give, or the
      // board overflows its zone and covers the player strips.
      side = Math.max(150, Math.floor(Math.min(zw, zh) * scale) - 6);
      if (this._sideApplied === side) { this.render(); return; }
      const wrap = this.cv.parentElement;
      wrap.style.width = side + 'px';
      wrap.style.height = side + 'px';
    }
    this._sideApplied = side;
    const dpr = Math.min(window.devicePixelRatio || 1, 3);
    this.cssSide = side;
    this.S = side;   // paintStatic destructures S
    if (!this.fixedSide) {
      this.cv.width = Math.round(side * dpr);
      this.cv.height = Math.round(side * dpr);
      this.cv.style.width = side + 'px';
      this.cv.style.height = side + 'px';
    }
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    // Geometry: S = 2M + 9C + 8G, with M a fraction of C.
    // Solve for C: C = (S - 8G) / (9 + 2k).
    this.G = Math.max(4, Math.min(10, 0.145 * (side / 10.6)));
    const coordsMode = this.ds().coords || 'edges';
    const k = coordsMode !== 'off' ? 0.46 : 0.20;
    this.C = (side - 8 * this.G) / (9 + 2 * k);
    this.M = k * this.C;
    this.U = this.C + this.G;
    this.themeDirty = true;
    this.render();
  }

  cellXY(r, c) { return { x: this.M + c * this.U, y: this.M + r * this.U }; }
  cellCenter(r, c) {
    const p = this.cellXY(r, c);
    return { x: p.x + this.C / 2, y: p.y + this.C / 2 };
  }
  anchorCenter(r, c) {   // groove crossing south east of display cell (r,c)
    return { x: this.M + (c + 1) * this.U - this.G / 2,
             y: this.M + (r + 1) * this.U - this.G / 2 };
  }

  // ---- coordinate conversions -------------------------------------------
  // The human player (side 0) sits at the bottom and moves up, so display row
  // = 8 - engine row unless the board is flipped.
  engPawnToDisp(cell) {
    const r = Math.floor(cell / 9), c = cell % 9;
    return this.flipped ? r * 9 + c : (8 - r) * 9 + c;
  }
  dispPawnToEng(r, c) { return this.flipped ? r * 9 + c : (8 - r) * 9 + c; }
  engWallToDisp(o, r, c) { return this.flipped ? [o, r, c] : [o, 7 - r, c]; }
  dispWallToEng(o, r, c) { return this.flipped ? [o, r, c] : [o, 7 - r, c]; }
  // Absolute algebraic name (file plus engine rank), independent of the
  // display orientation. Rank 1 is always player 0's home row.
  engAlgName(engCell) {
    return 'abcdefghi'[engCell % 9] + (Math.floor(engCell / 9) + 1);
  }
  algName(dispCell) {
    return this.engAlgName(this.dispPawnToEng(Math.floor(dispCell / 9), dispCell % 9));
  }

  setData(pawnEng, wallsHEng, wallsVEng, lastMove) {
    this.pawn = [this.engPawnToDisp(pawnEng[0]), this.engPawnToDisp(pawnEng[1])];
    // Wall slots mirror like the pawns. The board arrays hold display
    // coordinates, so paint and export can index them directly. The map is a
    // bijection, so every display slot is rewritten and no stale bit survives.
    for (let r = 0; r < 8; r++) for (let c = 0; c < 8; c++) {
      const s = r * 8 + c, d = this.flipped ? s : (7 - r) * 8 + c;
      this.wallH[d] = wallsHEng[s] | 0;
      this.wallV[d] = wallsVEng[s] | 0;
    }
    this.lastMove = lastMove || null;
    this.render();
  }

  setTurn(t) { this.turn = t; }

  setPaths(paths) { this.paths = paths || null; this.render(); }

  setHover(h) {
    const a = this.hover, b = h || null;
    const same = (!a && !b) || (a && b && a.kind === b.kind && a.o === b.o &&
                                a.r === b.r && a.c === b.c && a.cell === b.cell);
    this.hover = b;
    if (!same) this.render();
  }

  render() {
    const ctx = this.ctx, S = this.cssSide;
    ctx.clearRect(0, 0, S, S);
    this.drawStatic(ctx);
    this.drawDynamic(ctx);
    if (this.onChange) this.onChange();
  }

  // ---- animation ---------------------------------------------------------

  get animating() {
    return !!(this._pawnAnim[0] || this._pawnAnim[1] || this._wallAnim);
  }

  _now() {
    return (typeof performance !== 'undefined' && performance.now)
      ? performance.now() : Date.now();
  }

  // Start the rAF loop. The loop stops itself as soon as the queue empties.
  _startLoop() {
    if (this._raf) return;
    const step = () => {
      this._raf = 0;
      const alive = this._advance();
      this.render();
      if (alive) this._raf = requestAnimationFrame(step);
    };
    this._raf = requestAnimationFrame(step);
  }

  // Advance every tween. Returns true while at least one is still running.
  _advance() {
    const now = this._now();
    for (let pl = 0; pl < 2; pl++) {
      const a = this._pawnAnim[pl];
      if (!a) continue;
      a.t = Math.min(1, (now - a.t0) / a.dur);
      if (a.t >= 1) {
        this._pawnAnim[pl] = null;
        if (a.done) a.done();
      }
    }
    const w = this._wallAnim;
    if (w) {
      w.t = (now - w.t0);
      if (w.t >= QR_WALL_MS + QR_WALL_RIM_MS) this._wallAnim = null;
    }
    return this.animating;
  }

  // Slide a pawn from one display cell to another. Returns a Promise that
  // resolves when the tween finishes.
  animateMove(player, fromDispCell, toDispCell, opts = {}) {
    const pl = player | 0;
    if (!Number.isInteger(fromDispCell) || !Number.isInteger(toDispCell) ||
        fromDispCell === toDispCell) {
      return Promise.resolve();
    }
    const dur = Math.max(1, opts.duration || QR_MOVE_MS);
    const fr = Math.floor(fromDispCell / 9), fc = fromDispCell % 9;
    const tr = Math.floor(toDispCell / 9), tc = toDispCell % 9;
    // A jump covers more than one cell on either axis. It arcs.
    const jump = Math.max(Math.abs(tr - fr), Math.abs(tc - fc)) > 1;
    // Cancel any tween still running for this pawn.
    const prev = this._pawnAnim[pl];
    if (prev && prev.done) prev.done();
    return new Promise(resolve => {
      this._pawnAnim[pl] = {
        from: fromDispCell, to: toDispCell, jump: jump,
        t0: this._now(), dur: dur, t: 0, done: resolve
      };
      this._startLoop();
    });
  }

  // Fade and scale a wall in, then let a gold rim decay.
  animateWall(o, r, c) {
    this._wallAnim = { o: o | 0, r: r | 0, c: c | 0, t0: this._now(), t: 0 };
    this._startLoop();
    return Promise.resolve();
  }

  // Holds a pawn under the pointer while the reader drags it. Screen
  // coordinates in CSS pixels, or null to release it back to its cell.
  setDragPawn(pl, x, y) {
    this._dragPawn = (pl == null) ? null : { pl: pl | 0, x: x, y: y };
    this.render();
  }

  // Interpolated screen position of an animating pawn, or null.
  _pawnPos(pl) {
    const d = this._dragPawn;
    if (d && d.pl === pl) return { x: d.x, y: d.y, scale: 1.06 };
    const a = this._pawnAnim[pl];
    if (!a) return null;
    const e = qrEase(Math.max(0, Math.min(1, a.t)));
    const from = this.cellCenter(Math.floor(a.from / 9), a.from % 9);
    const to = this.cellCenter(Math.floor(a.to / 9), a.to % 9);
    const x = from.x + (to.x - from.x) * e;
    let y = from.y + (to.y - from.y) * e;
    let scale = 1;
    if (a.jump) {
      const arc = Math.sin(Math.PI * e);   // 0 at both ends, 1 at the apex
      y -= 0.22 * this.C * arc;
      scale = 1 + 0.08 * arc;
    }
    return { x: x, y: y, scale: scale };
  }

  // ---- static layer ------------------------------------------------------

  drawStatic(ctx) {
    const S = this.cssSide;
    if (!this._static || this.themeDirty || this._static.side !== S) {
      const off = document.createElement('canvas');
      off.width = this.cv.width; off.height = this.cv.height;
      const octx = off.getContext('2d');
      octx.setTransform(this.ctx.getTransform());
      this.paintStatic(octx);
      this._static = { canvas: off, side: S };
      this.themeDirty = false;
    }
    ctx.drawImage(this._static.canvas, 0, 0, S, S);
  }

  rr(g, x, y, w, h, r) {
    r = Math.min(r, w / 2, h / 2);
    g.beginPath();
    g.moveTo(x + r, y);
    g.arcTo(x + w, y, x + w, y + h, r);
    g.arcTo(x + w, y + h, x, y + h, r);
    g.arcTo(x, y + h, x, y, r);
    g.arcTo(x, y, x + w, y, r);
    g.closePath();
  }

  // Frame styles. Each one paints a different border around the play area.
  //   none      bare page ground, no plate and no rim
  //   hairline  frame plate with a single light rim
  //   gilded    frame plate with a gold band and a gold bed rail
  //   beveled   frame plate with a raised bevel and an inner ridge
  paintFrame(g, style, bx, bw) {
    const S = this.cssSide;
    if (style === 'none') {
      g.fillStyle = this.css('--bg');
      this.rr(g, 0, 0, S, S, 14); g.fill();
      return;
    }
    g.fillStyle = this.css('--frame');
    this.rr(g, 0, 0, S, S, 14); g.fill();
    g.strokeStyle = this.css('--frame-hi'); g.lineWidth = 1;
    this.rr(g, 0.5, 0.5, S - 1, S - 1, 14); g.stroke();
    if (style === 'gilded') {
      g.strokeStyle = this.css('--gold'); g.lineWidth = 2;
      this.rr(g, 2.5, 2.5, S - 5, S - 5, 12); g.stroke();
      g.strokeStyle = this.css('--gold-dim'); g.lineWidth = 1.5;
      this.rr(g, bx - 3.5, bx - 3.5, bw + 7, bw + 7, 6); g.stroke();
    } else if (style === 'beveled') {
      const be = g.createLinearGradient(0, 0, 0, S);
      be.addColorStop(0, 'rgba(255,255,255,.12)');
      be.addColorStop(.06, 'rgba(0,0,0,.22)');
      be.addColorStop(.94, 'rgba(0,0,0,.18)');
      be.addColorStop(1, 'rgba(255,255,255,.08)');
      g.fillStyle = be;
      this.rr(g, 0, 0, S, S, 14); g.fill();
      g.strokeStyle = 'rgba(0,0,0,.45)'; g.lineWidth = 2;
      this.rr(g, bx - 3, bx - 3, bw + 6, bw + 6, 6); g.stroke();
      g.strokeStyle = 'rgba(255,255,255,.14)'; g.lineWidth = 1;
      this.rr(g, bx - 5.5, bx - 5.5, bw + 11, bw + 11, 7); g.stroke();
    }
  }

  paintMaterialTexture(g, bx, bw) {
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
    const { C, G, U, M, S } = this;
    const ds = this.ds();
    const frameStyle = ds.frame || 'beveled';
    const cellSep = ds.cellSep || 'grooves';
    const bx = M - G / 2, bw = 9 * U;
    // Board frame and bed. The frame carries the coordinate margin. The bed
    // colour shows through every groove.
    this.paintFrame(g, frameStyle, bx, bw);

    const boardContrast = ds.boardContrast || 'standard';
    const groove = qrBoardContrastColor(this.css('--groove'), boardContrast);
    g.fillStyle = groove;
    this.rr(g, bx, bx, bw, bw, 4); g.fill();
    if (frameStyle !== 'none') {
      // Inner bevel of the frame around the play area.
      g.strokeStyle = 'rgba(0,0,0,.35)'; g.lineWidth = 1;
      this.rr(g, bx - 0.5, bx - 0.5, bw + 1, bw + 1, 4); g.stroke();
    }

    // Cells. Two warm tones with a very low delta. The cell separator style
    // decides what fills the gap between them.
    //   grooves  rounded cells, bed colour in the gap, groove centre lines
    //   flat     square cells that grow over the gap, one seamless surface
    //   inlaid   flat surface with a gold hairline inlaid around each cell
    const ca = qrBoardContrastColor(this.css('--cell-a'), boardContrast),
          cb = qrBoardContrastColor(this.css('--cell-b'), boardContrast);
    const rad = 0.10 * C;
    for (let r = 0; r < 9; r++) for (let c = 0; c < 9; c++) {
      const p = this.cellXY(r, c);
      g.fillStyle = ((r + c) & 1) ? cb : ca;
      if (cellSep === 'grooves') {
        this.rr(g, p.x, p.y, C, C, rad); g.fill();
        g.lineWidth = 1;
        g.strokeStyle = 'rgba(255,255,255,.18)';
        g.beginPath();
        g.moveTo(p.x + rad, p.y + 0.5);
        g.lineTo(p.x + C - rad, p.y + 0.5);
        g.stroke();
        g.strokeStyle = 'rgba(0,0,0,.10)';
        g.beginPath();
        g.moveTo(p.x + rad, p.y + C - 0.5);
        g.lineTo(p.x + C - rad, p.y + C - 0.5);
        g.stroke();
      } else {
        g.fillRect(p.x - G / 2, p.y - G / 2, C + G, C + G);
        if (cellSep === 'inlaid') {
          g.strokeStyle = this.css('--gold');
          g.globalAlpha = .32; g.lineWidth = 1;
          g.strokeRect(p.x + 0.5, p.y + 0.5, C - 1, C - 1);
          g.globalAlpha = 1;
        }
      }
    }
    this.paintMaterialTexture(g, bx, bw);

    // Goal rows are renderer-owned too: Off really means no wash and no edge.
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
    if (cellSep === 'grooves') {
      // Groove centre lines, so the grid still reads when the cells are too
      // small for the gap alone to carry it. On a large board the gap is
      // already obvious, and the line then reads as an artefact, so it fades
      // out as the cell grows.
      const lineAlpha = Math.max(0, Math.min(.20, .20 * (34 - C) / 18));
      if (lineAlpha > .01) {
      g.strokeStyle = `rgba(0,0,0,${lineAlpha.toFixed(3)})`; g.lineWidth = 1;
      for (let i = 1; i < 9; i++) {
        const t = Math.round(M + i * U - G / 2) + 0.5;
        g.beginPath(); g.moveTo(bx, t); g.lineTo(bx + bw, t); g.stroke();
        g.beginPath(); g.moveTo(t, bx); g.lineTo(t, bx + bw); g.stroke();
      }
      }
    }

    // Goal edges follow the same control. Off removes every goal-row cue.
    if (goalMode !== 'off') {
      g.globalAlpha = goalMode === 'clear' ? .40 : .30;
      g.fillStyle = this.flipped ? this.css('--p0') : this.css('--p1');
      g.fillRect(bx, bx, bw, 2);
      g.fillStyle = this.flipped ? this.css('--p1') : this.css('--p0');
      g.fillRect(bx, bx + bw - 2, bw, 2);
      g.globalAlpha = 1;
    }

    // Coordinates, drawn on the frame margin.
    const coordsMode = ds.coords || 'edges';
    if (coordsMode !== 'off' && C >= 24) {
      // The coordinates label the board, they do not compete with it, so they
      // stay small and quiet. They must still be readable: the size holds a
      // 10px floor, and --coord clears 4.5:1 against --frame in every theme,
      // which tools/gui/contrast_check.py enforces at build time.
      g.save();
      g.fillStyle = this.css('--coord');
      g.globalAlpha = .92;
      g.font = `500 ${Math.max(10.5, 0.228 * C)}px 'JetBrains Mono', monospace`;
      g.textAlign = 'center'; g.textBaseline = 'middle';
      // Centre of the margin band, not M/2. The play area starts at bx, so the
      // left band is [0, bx] and the bottom band is [bx + bw, S]. Both are
      // M - G/2 wide, and M/2 sits G/4 short of their middle, which pushed
      // every label a little way onto the board.
      // Snap the two margin centres to half pixels. At common DPRs this
      // keeps the small mono glyphs visually centred instead of one edge
      // looking a fraction heavier.
      const bandX = Math.round(bx / 2) + .5;
      const bandY = Math.round((bx + bw + S) / 2) + .5;
      for (let c = 0; c < 9; c++) {
        g.fillText('abcdefghi'[c], this.cellCenter(0, c).x, bandY);
      }
      for (let r = 0; r < 9; r++) {
        // Absolute rank: display row r is engine row (flipped ? r : 8 - r).
        g.fillText(String(this.flipped ? r + 1 : 9 - r), bandX, this.cellCenter(r, 0).y);
      }
      g.restore();
      // Study mode: a faint file and rank in every cell corner.
      if (coordsMode === 'all' && C >= 40) {
        g.globalAlpha = .55;
        g.font = `${Math.max(9, 0.135 * C)}px 'JetBrains Mono', monospace`;
        g.textAlign = 'left'; g.textBaseline = 'top';
        for (let r = 0; r < 9; r++) for (let c = 0; c < 9; c++) {
          const p = this.cellXY(r, c);
          const engR = this.flipped ? r : 8 - r;
          g.fillText('abcdefghi'[c] + (engR + 1), p.x + 3, p.y + 2);
        }
        g.globalAlpha = 1;
      }
    }
  }

  // ---- dynamic layer -----------------------------------------------------

  drawDynamic(g) {
    const { C } = this;
    if (!(C > 0)) return;
    if (this.paths) for (const line of this.paths) this.drawPathLine(g, line);
    if (this.linePreview) this.drawLinePreview(g);

    const wa = this._wallAnim;
    for (let r = 0; r < 8; r++) for (let c = 0; c < 8; c++) {
      if (this.wallH[r * 8 + c]) this.drawWallAnimated(g, 0, r, c, wa);
      if (this.wallV[r * 8 + c]) this.drawWallAnimated(g, 1, r, c, wa);
    }

    if (this.lastMove) {
      if (this.lastMove.type === 'pawn') {
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
        const [o, r, c] = this.engWallToDisp(this.lastMove.o, this.lastMove.r, this.lastMove.c);
        // Frame the rail from outside. Stroking the beam itself put a gold
        // line through a piece only a few pixels wide, which read as a stripe
        // painted on the wall rather than as a mark on the move.
        const rc = this.wallDrawRect(o, r, c);
        g.save();
        const lm = this.ds().lastMoveStyle || 'subtle';
        g.strokeStyle = this.css('--gold');
        g.lineWidth = lm === 'clear' ? 1.75 : 1.15;
        g.globalAlpha = lm === 'clear' ? .92 : .58;
        const pad = lm === 'clear' ? 1.9 : 1.45;
        this.rr(g, rc.x - pad, rc.y - pad, rc.w + 2 * pad, rc.h + 2 * pad, 3.5); g.stroke();
        g.restore();
      }
    }

    // Hover ghost: the quiet wall preview that replaces the wall mode button.
    if (!this.ghost && this.hover && this.hover.kind === 'groove') {
      const h = this.hover;
      const wp = this.ds().wallPreview || 'normal';
      g.save(); g.globalAlpha = wp === 'strong' ? .52 : wp === 'subtle' ? .24 : .35;
      this.drawWall(g, h.o, h.r, h.c, true);
      g.restore();
    }
    if (this.ghost) this.drawGhost(g);

    const havePawns = Array.isArray(this.pawn) && this.pawn.length === 2 &&
                      Number.isInteger(this.pawn[0]) && Number.isInteger(this.pawn[1]);
    if (this.dots.length && !this.ghost && havePawns) {
      const me = this.cellCenter(Math.floor(this.pawn[this.turn] / 9), this.pawn[this.turn] % 9);
      for (const d of this.dots) {
        const ctr = this.cellCenter(Math.floor(d / 9), d % 9);
        const jump = Math.abs(ctr.x - me.x) + Math.abs(ctr.y - me.y) > this.U * 1.5;
        g.save();
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
        g.restore();
      }
    }

    if (havePawns) {
      this.drawPawn(g, this.pawn[0], 0, this._pawnPos(0));
      this.drawPawn(g, this.pawn[1], 1, this._pawnPos(1));
    }

    if (this.selected >= 0) {
      const ctr = this.cellCenter(Math.floor(this.selected / 9), this.selected % 9);
      g.beginPath(); g.arc(ctr.x, ctr.y, 0.40 * C, 0, 7);
      g.strokeStyle = this.css('--gold'); g.lineWidth = 2;
      g.setLineDash([4, 3]); g.stroke(); g.setLineDash([]);
    }
  }

  // The dots take the colour of the side to move.
  dotColor() { return this.css('--p' + this.turn); }

  // Analysis PV preview: every wall of the line as a translucent beam (wall
  // slots are order independent) plus numbered rings on the pawn steps.
  drawLinePreview(g) {
    const lp = this.linePreview;
    const col = lp.color || this.css('--blue');
    g.save();
    for (const w of lp.walls || []) {
      const [wo, wr, wc] = this.engWallToDisp(w.o, w.r, w.c);
      g.globalAlpha = .38;
      this.drawWall(g, wo, wr, wc, false, col);
    }
    g.globalAlpha = .9;
    let step = 1;
    for (const pc of lp.pawns || []) {
      const d = this.engPawnToDisp(pc);
      const ctr = this.cellCenter(Math.floor(d / 9), d % 9);
      g.beginPath(); g.arc(ctr.x, ctr.y, .24 * this.C, 0, 7);
      g.strokeStyle = col; g.lineWidth = 2; g.stroke();
      g.fillStyle = col;
      g.font = `600 ${Math.max(9, Math.floor(.34 * this.C))}px 'JetBrains Mono', monospace`;
      g.textAlign = 'center'; g.textBaseline = 'middle';
      g.fillText(String(step++), ctr.x, ctr.y);
    }
    g.restore();
  }

  drawPathLine(g, line) {
    if (!line || !line.cells || line.cells.length < 2) return;
    const col = line.color ||
      this.css('--p' + (line.player === 1 ? 1 : 0));
    g.save();
    g.strokeStyle = col;
    g.globalAlpha = .28;
    g.lineWidth = 0.09 * this.C;
    g.lineCap = 'round'; g.lineJoin = 'round';
    g.beginPath();
    line.cells.forEach((cell, i) => {
      const ctr = this.cellCenter(Math.floor(cell / 9), cell % 9);
      if (i) g.lineTo(ctr.x, ctr.y); else g.moveTo(ctr.x, ctr.y);
    });
    g.stroke(); g.restore();
  }

  wallRect(o, r, c) {
    const a = this.anchorCenter(r, c), T = this.G, L = 2 * this.C + this.G;
    return o === 0
      ? { x: a.x - this.C - this.G / 2, y: a.y - T / 2, w: L, h: T }
      : { x: a.x - T / 2, y: a.y - this.C - this.G / 2, w: T, h: L };
  }

  // Painted rect: the beam is drawn fatter than its layout slot so it reads as
  // a solid rail across the corridor. The slot itself (wallRect) stays the
  // geometry of record for hit tests, ghosts and chip positioning.
  wallDrawRect(o, r, c) {
    const rc = this.wallRect(o, r, c);
    // Across the corridor: a little fatter than the slot, so the rail seats in
    // its channel instead of looking laid on top of the board.
    const profile = this.ds().wallProfile || 'standard';
    const profileScale = profile === 'slim' ? .27 : profile === 'bold' ? .41 : .33;
    const inf = Math.max(1.4, this.G * profileScale);
    // Extend a fraction beyond the mathematical crossing centre. The overlap
    // is visual only (wallRect still owns hit testing) and removes antialias
    // hairlines where collinear rails or legal T/L junctions meet.
    const ext = this.G * .54;
    return o === 0
      ? { x: rc.x - ext, y: rc.y - inf, w: rc.w + 2 * ext, h: rc.h + 2 * inf }
      : { x: rc.x - inf, y: rc.y - ext, w: rc.w + 2 * inf, h: rc.h + 2 * ext };
  }

  // Wall beam: dark walnut, a rim, a drop shadow and a seam at the midpoint.
  // The wall finish decides how the beam body is filled.
  //   flat      one solid tone, square corners, no shadow
  //   beveled   the default bevel gradient across the thickness
  //   glossy    the bevel plus a bright highlight band along the top
  //   etched    a dark core, so the beam reads as engraved into the board
  drawWall(g, o, r, c, ghostStyle, outlineOverride) {
    const rc = this.wallDrawRect(o, r, c);
    const wall = this.css('--wall'), edge = this.css('--wall-edge');
    const hi = this.css('--wall-hi');
    // Depth cues come from the theme, not from literals. A light board needs
    // a different shadow than a dark board.
    const shadow = this.css('--wall-shadow'), gloss = this.css('--wall-gloss');
    const finish = this.ds().wallFinish || 'beveled';
    const rad = finish === 'flat' ? 0 : 2;
    g.save();
    if (finish !== 'flat') {
      g.shadowColor = shadow;
      g.shadowBlur = 4;
      g.shadowOffsetY = 1.5;
    }
    if (finish === 'flat') {
      g.fillStyle = wall;
    } else {
      const grad = o === 0
        ? g.createLinearGradient(0, rc.y, 0, rc.y + rc.h)
        : g.createLinearGradient(rc.x, 0, rc.x + rc.w, 0);
      if (finish === 'glossy') {
        grad.addColorStop(0, edge);
        grad.addColorStop(.16, hi);
        grad.addColorStop(.34, gloss);
        grad.addColorStop(.52, hi);
        grad.addColorStop(.78, wall);
        grad.addColorStop(1, edge);
      } else if (finish === 'etched') {
        grad.addColorStop(0, hi);
        grad.addColorStop(.22, edge);
        grad.addColorStop(.78, edge);
        grad.addColorStop(1, wall);
      } else {
        grad.addColorStop(0, edge);
        grad.addColorStop(.30, hi);
        grad.addColorStop(.70, wall);
        grad.addColorStop(1, edge);
      }
      g.fillStyle = grad;
    }
    this.rr(g, rc.x, rc.y, rc.w, rc.h, rad); g.fill();
    g.shadowColor = 'rgba(0,0,0,0)'; g.shadowBlur = 0; g.shadowOffsetY = 0;
    if (finish !== 'flat') {
      // Moulded surface. A plain gradient reads as a flat painted strip, so
      // the beam gets a lit top edge, a shaded bottom edge and a few hairlines
      // of grain along its length. All three are clipped to the beam, so the
      // rail keeps its rounded corners.
      g.save();
      this.rr(g, rc.x, rc.y, rc.w, rc.h, rad); g.clip();
      const along = o === 0;                       // beam runs left to right
      const thick = along ? rc.h : rc.w;
      // lit edge and shaded edge, one thin band each
      g.fillStyle = 'rgba(255,255,255,.18)';
      if (along) g.fillRect(rc.x, rc.y, rc.w, Math.max(1, thick * .16));
      else g.fillRect(rc.x, rc.y, Math.max(1, thick * .16), rc.h);
      g.fillStyle = 'rgba(0,0,0,.16)';
      if (along) g.fillRect(rc.x, rc.y + rc.h - Math.max(1, thick * .20), rc.w, Math.max(1, thick * .20));
      else g.fillRect(rc.x + rc.w - Math.max(1, thick * .20), rc.y, Math.max(1, thick * .20), rc.h);
      // grain: evenly spaced hairlines, alternating light and dark
      g.lineWidth = 1;
      // Two hairlines, not three, and faint: at full strength the beam reads
      // as a striped bar instead of a moulded one.
      const lines = 2;
      for (let i = 1; i <= lines; i++) {
        const f = i / (lines + 1);
        g.strokeStyle = (i % 2) ? 'rgba(255,255,255,.06)' : 'rgba(0,0,0,.045)';
        g.beginPath();
        if (along) {
          const y = Math.round(rc.y + f * rc.h) + .5;
          g.moveTo(rc.x + 1, y); g.lineTo(rc.x + rc.w - 1, y);
        } else {
          const x = Math.round(rc.x + f * rc.w) + .5;
          g.moveTo(x, rc.y + 1); g.lineTo(x, rc.y + rc.h - 1);
        }
        g.stroke();
      }
      g.restore();
    }
    g.lineWidth = outlineOverride ? 1.5 : 1;
    g.strokeStyle = outlineOverride || edge;
    g.stroke();
    if (finish === 'etched') {
      // Engraved inset: a dark line inside the rim with a light line under it.
      g.globalAlpha = .55; g.lineWidth = 1;
      g.strokeStyle = this.css('--wall-etch-dark');
      this.rr(g, rc.x + 1.5, rc.y + 1.5, rc.w - 3, rc.h - 3, 1); g.stroke();
      g.strokeStyle = this.css('--wall-etch-light');
      this.rr(g, rc.x + 2.5, rc.y + 2.5, rc.w - 3, rc.h - 3, 1); g.stroke();
      g.globalAlpha = 1;
    }
    // No seam at the midpoint. The two halves of a beam are one piece to the
    // player, and a line across the middle made the rail read as two shorter
    // walls that happen to touch.
    g.restore();
  }

  // A wall that is the target of animateWall fades and scales in, then keeps
  // a gold rim that decays.
  drawWallAnimated(g, o, r, c, wa) {
    if (!wa || wa.o !== o || wa.r !== r || wa.c !== c) {
      this.drawWall(g, o, r, c, false);
      return;
    }
    const t = Math.max(0, Math.min(1, wa.t / QR_WALL_MS));
    const rc = this.wallDrawRect(o, r, c);
    const cx = rc.x + rc.w / 2, cy = rc.y + rc.h / 2;
    const s = 0.86 + 0.14 * qrEase(t);
    g.save();
    g.globalAlpha = 0.25 + 0.75 * t;
    g.translate(cx, cy); g.scale(s, s); g.translate(-cx, -cy);
    this.drawWall(g, o, r, c, false);
    g.restore();
    const rim = (wa.t - QR_WALL_MS) / QR_WALL_RIM_MS;
    if (t >= 1 && rim < 1) {
      g.save();
      g.globalAlpha = 1 - Math.max(0, rim);
      g.strokeStyle = this.css('--gold'); g.lineWidth = 1.5;
      this.rr(g, rc.x, rc.y, rc.w, rc.h, 2); g.stroke();
      g.restore();
    }
  }

  drawGhost(g) {
    const gh = this.ghost, o = gh.o, r = gh.r, c = gh.c;
    const rc = this.wallDrawRect(o, r, c);
    const st = gh.state;
    const wp = this.ds().wallPreview || 'normal';
    const alpha = wp === 'subtle'
      ? (st === 'bad' ? .14 : st === 'pending' ? .90 : .45)
      : wp === 'strong'
        ? (st === 'bad' ? .28 : st === 'pending' ? 1 : .75)
        : (st === 'bad' ? .20 : st === 'pending' ? 1 : .60);
    const col = st === 'ok' ? this.css('--green')
              : st === 'assisted' ? this.css('--gold')
              : st === 'pending' ? this.css('--gold')
              : this.css('--red');
    g.save(); g.globalAlpha = alpha;
    this.drawWall(g, o, r, c, true);
    g.restore();
    g.save();
    g.strokeStyle = col; g.lineWidth = 2;
    this.rr(g, rc.x, rc.y, rc.w, rc.h, 2); g.stroke();
    if (st === 'bad') {
      const cx = rc.x + rc.w / 2, cy = rc.y + rc.h / 2, k = this.G * .8;
      g.beginPath();
      g.moveTo(cx - k, cy - k); g.lineTo(cx + k, cy + k);
      g.moveTo(cx + k, cy - k); g.lineTo(cx - k, cy + k);
      g.stroke();
    }
    if (gh.from && st === 'assisted') {
      const fc = this.wallRect(o, gh.from.r, gh.from.c);
      g.strokeStyle = this.css('--muted'); g.lineWidth = 1;
      g.setLineDash([3, 3]);
      this.rr(g, fc.x, fc.y, fc.w, fc.h, 2); g.stroke();
      g.setLineDash([]);
    }
    g.restore();
  }

  // pos is null for a resting pawn, or {x,y,scale} while it animates.
  drawPawn(g, dispCell, pl, pos) {
    const { C } = this;
    const r = Math.floor(dispCell / 9), c = dispCell % 9;
    const rest = this.cellCenter(r, c);
    const ctr = pos ? { x: pos.x, y: pos.y } : rest;
    const mul = pos ? pos.scale : 1;
    const px = this.css('--p' + pl), pli = this.css('--p' + pl + '-light'),
          pd = this.css('--p' + pl + '-deep');
    const sizeMul = { small: .85, large: 1.15 }[this.ds().pawnSize] || 1;
    const R = 0.30 * C * sizeMul * mul;
    g.save();
    // Elliptical contact shadow, kept on the ground while the pawn arcs.
    // The falloff is a radial gradient, not a solid ellipse. A hard edged
    // ellipse this close under the body reads as a plinth, which is what made
    // the pawn look like it stands on a base.
    const shadow = this.ds().pawnShadow || 'soft';
    if (shadow !== 'off') {
      const a = shadow === 'deep' ? .40 : .26;
      const sy = ctr.y + R * .74;
      g.save();
      g.translate(ctr.x, sy);
      g.scale(1, .30);
      const sh = g.createRadialGradient(0, 0, 0, 0, 0, R * 1.15);
      sh.addColorStop(0, 'rgba(0,0,0,' + a + ')');
      sh.addColorStop(.50, 'rgba(0,0,0,' + (a * .55).toFixed(3) + ')');
      sh.addColorStop(1, 'rgba(0,0,0,0)');
      g.fillStyle = sh;
      g.beginPath(); g.arc(0, 0, R * 1.15, 0, 7); g.fill();
      g.restore();
    }
    const grd = g.createRadialGradient(
      ctr.x - R * .30, ctr.y - R * .35, R * .12, ctr.x, ctr.y, R * 1.05);
    grd.addColorStop(0, pli);
    grd.addColorStop(.55, px);
    grd.addColorStop(1, pd);
    g.fillStyle = grd; g.strokeStyle = pd; g.lineWidth = 1;
    let style = this.ds().pawn || 'disc';
    // Distinct shapes: the two sides get different silhouettes, so the board
    // stays readable without colour.
    const dsv = this.ds().distinctShapes;
    if ((dsv === '1' || dsv === 'true') && pl === 1) {
      style = QR_PAWN_ALT[style] || 'disc';
    }
    this.drawPawnShape(g, style, ctr, R, pl, grd);
    // Specular highlight. It sits well inside the silhouette, so it needs no
    // clip, and it is what sells the body as a sphere lit from the upper left.
    if (style === 'disc' || style === 'discRing' || style === 'beacon') {
      const hx = ctr.x - R * .34, hy = ctr.y - R * .38;
      const spec = g.createRadialGradient(hx, hy, 0, hx, hy, R * .46);
      spec.addColorStop(0, 'rgba(255,255,255,.55)');
      spec.addColorStop(.6, 'rgba(255,255,255,.13)');
      spec.addColorStop(1, 'rgba(255,255,255,0)');
      g.fillStyle = spec;
      g.beginPath(); g.arc(hx, hy, R * .46, 0, 7); g.fill();
    }
    // No side-to-move ring here: the legal-move dots already say whose turn
    // it is, and a permanent halo around the pawn reads as noise.
    g.restore();
  }

  // One pawn silhouette, centred on ctr with radius R. The caller sets the
  // fill and the stroke, so every style shares the same body gradient.
  // grd is that gradient, which the beacon style strokes with.
  drawPawnShape(g, style, ctr, R, pl, grd) {
    const { C } = this;
    if (style === 'pillar') {
      // Domed head over a flared foot.
      g.beginPath();
      g.arc(ctr.x, ctr.y - R * .25, R * .58, 0, 7);
      g.moveTo(ctr.x - R * .34, ctr.y + R * .05);
      g.quadraticCurveTo(ctr.x, ctr.y + R * .18, ctr.x + R * .34, ctr.y + R * .05);
      g.lineTo(ctr.x + R * .52, ctr.y + R * .62);
      g.quadraticCurveTo(ctr.x, ctr.y + R * .40, ctr.x - R * .52, ctr.y + R * .62);
      g.closePath(); g.fill(); g.stroke();
      return;
    }
    if (style === 'crown') {
      // Crenellated crown with a tall centre spike.
      g.beginPath();
      g.moveTo(ctr.x - R * .58, ctr.y + R * .62);
      g.lineTo(ctr.x - R * .58, ctr.y - R * .02);
      g.lineTo(ctr.x - R * .28, ctr.y + R * .16);
      g.lineTo(ctr.x, ctr.y - R * .48);
      g.lineTo(ctr.x + R * .28, ctr.y + R * .16);
      g.lineTo(ctr.x + R * .58, ctr.y - R * .02);
      g.lineTo(ctr.x + R * .58, ctr.y + R * .62);
      g.closePath(); g.fill(); g.stroke();
      return;
    }
    if (style === 'rune') {
      // Hexagonal stone with an engraved chevron that points at the goal.
      const k = R * .98;
      g.beginPath();
      g.moveTo(ctr.x, ctr.y - k);
      g.lineTo(ctr.x + k * .87, ctr.y - k * .5);
      g.lineTo(ctr.x + k * .87, ctr.y + k * .5);
      g.lineTo(ctr.x, ctr.y + k);
      g.lineTo(ctr.x - k * .87, ctr.y + k * .5);
      g.lineTo(ctr.x - k * .87, ctr.y - k * .5);
      g.closePath(); g.fill(); g.stroke();
      g.strokeStyle = this.css('--bg');
      g.lineWidth = Math.max(1.5, C * .05);
      const dir = pl === 0 ? -1 : 1;
      g.beginPath();
      g.moveTo(ctr.x - R * .32, ctr.y + dir * R * .22);
      g.lineTo(ctr.x, ctr.y - dir * R * .22);
      g.lineTo(ctr.x + R * .32, ctr.y + dir * R * .22);
      g.stroke();
      return;
    }
    if (style === 'pawnChess') {
      // Chess pawn: bulb, collar and flared base.
      g.beginPath();
      g.arc(ctr.x, ctr.y - R * .36, R * .46, 0, 7);
      g.fill(); g.stroke();
      g.beginPath();
      g.moveTo(ctr.x - R * .26, ctr.y - R * .02);
      g.quadraticCurveTo(ctr.x - R * .42, ctr.y - R * .18, ctr.x - R * .30, ctr.y - R * .32);
      g.lineTo(ctr.x + R * .30, ctr.y - R * .32);
      g.quadraticCurveTo(ctr.x + R * .42, ctr.y - R * .18, ctr.x + R * .26, ctr.y - R * .02);
      g.lineTo(ctr.x + R * .56, ctr.y + R * .64);
      g.quadraticCurveTo(ctr.x, ctr.y + R * .38, ctr.x - R * .56, ctr.y + R * .64);
      g.closePath(); g.fill(); g.stroke();
      return;
    }
    if (style === 'beacon') {
      // Open ring with a gold arc that points at that player's goal edge.
      const dir = pl === 0 ? 1 : -1;
      const lw = Math.max(3, R * .38);
      g.beginPath(); g.arc(ctr.x, ctr.y, R * .78, 0, 7);
      g.lineWidth = lw; g.strokeStyle = grd; g.stroke();
      g.beginPath();
      g.arc(ctr.x, ctr.y, R * .78,
            dir > 0 ? Math.PI * .25 : Math.PI * 1.25,
            dir > 0 ? Math.PI * .75 : Math.PI * 1.75);
      g.lineWidth = lw; g.strokeStyle = this.css('--gold2'); g.stroke();
      return;
    }
    // disc: the default polished counter.
    g.beginPath(); g.arc(ctr.x, ctr.y, R, 0, 7); g.fill(); g.stroke();
    if (style === 'discRing') {
      // Same ball, one incised ring. It reads without colour, which is what
      // distinct shapes is for, and it is still a pawn.
      g.save();
      g.globalAlpha = .55; g.lineWidth = Math.max(1, R * .10);
      g.strokeStyle = 'rgba(0,0,0,.75)';
      g.beginPath(); g.arc(ctr.x, ctr.y, R * .56, 0, 7); g.stroke();
      g.globalAlpha = .38; g.strokeStyle = 'rgba(255,255,255,.85)';
      g.beginPath(); g.arc(ctr.x, ctr.y, R * .56 - g.lineWidth * .7, 0, 7); g.stroke();
      g.restore();
    }
  }

  // ---- pointer geometry --------------------------------------------------

  pointToCell(px, py) {
    const c = Math.floor((px - this.M) / this.U), r = Math.floor((py - this.M) / this.U);
    return (r >= 0 && r < 9 && c >= 0 && c < 9) ? { r, c } : null;
  }

  nearestAnchor(px, py) {
    const r = Math.round(py / this.U) - 1, c = Math.round(px / this.U) - 1;
    if (r < 0 || r > 7 || c < 0 || c > 7) return null;
    const ctr = this.anchorCenter(r, c);
    return { r, c, dist: Math.hypot(px - ctr.x, py - ctr.y) };
  }

  // Contextual hit test. Decides between a cell body and a wall groove.
  // opts.lastO breaks the tie at a groove crossing.
  hitTest(px, py, opts = {}) {
    const { C, G, U, M } = this;
    if (!(C > 0)) return { kind: 'none' };
    const bx = px - M, by = py - M;
    const lo = -G, hi = 9 * U - G / 2;
    if (bx < lo || by < lo || bx > hi || by > hi) return { kind: 'none' };

    const fx = ((bx % U) + U) % U;
    const fy = ((by % U) + U) % U;
    const inV = fx > C;    // inside a vertical groove
    const inH = fy > C;    // inside a horizontal groove
    // Distance to the nearest groove on each axis.
    const dx = inV ? 0 : Math.min(C - fx, fx);
    const dy = inH ? 0 : Math.min(C - fy, fy);

    let o = -1, conf = 'over';
    if (inV && inH) {
      o = (opts.lastO === 0 || opts.lastO === 1) ? opts.lastO : 1;
    } else if (inV) {
      o = 1;
    } else if (inH) {
      o = 0;
    } else {
      const near = Math.min(dx, dy);
      if (near <= 0.30 * C) { o = dx < dy ? 1 : 0; conf = 'near'; }
    }

    if (o < 0) {
      const cc = Math.min(8, Math.max(0, Math.floor(bx / U)));
      const rr = Math.min(8, Math.max(0, Math.floor(by / U)));
      return { kind: 'cell', cell: rr * 9 + cc, r: rr, c: cc };
    }
    const a = this.anchorFor(o, px, py);
    return { kind: 'groove', o: o, r: a.r, c: a.c, conf: conf };
  }

  // Anchor rounding. Along the wall's own long axis use floor, so the cell
  // you are inside is the anchor cell. Across the groove use round.
  anchorFor(o, px, py) {
    const { C, U, M } = this;
    const cl = v => Math.min(7, Math.max(0, v));
    if (o === 0) {
      return { r: cl(Math.round((py - M - C) / U)), c: cl(Math.floor((px - M) / U)) };
    }
    return { r: cl(Math.floor((py - M) / U) - 1), c: cl(Math.round((px - M - C) / U)) };
  }

  // ---- image export ------------------------------------------------------
  // Renders the current position into a detached canvas at a fixed size,
  // independent of the on screen board, with an optional footer strip that
  // carries both player names and a small wordmark. opts:
  //   { size=1600, transparent=false, coords=true, footer=null } where
  //   footer is { name0, name1 }.
  renderExport(opts = {}) {
    const size = opts.size || 1600;
    const cv = document.createElement('canvas');
    cv.width = size; cv.height = size;
    const ex = new QBoard(cv, { fixedSide: size });
    ex.flipped = this.flipped;
    ex.pawn = this.pawn.slice();
    ex.wallH = Uint8Array.from(this.wallH);
    ex.wallV = Uint8Array.from(this.wallV);
    ex.lastMove = null;
    ex.turn = this.turn;
    const savedCoords = document.documentElement.dataset.coords;
    if (!opts.coords) document.documentElement.dataset.coords = 'off';
    ex.themeDirty = true;
    ex.fit();
    const g = ex.ctx;
    g.save(); g.setTransform(1, 0, 0, 1, 0, 0);
    g.clearRect(0, 0, cv.width, cv.height);
    if (!opts.transparent) {
      g.fillStyle = this.css('--bg');   // page background, not the frame
      g.fillRect(0, 0, cv.width, cv.height);
    }
    g.restore();
    ex.render();
    document.documentElement.dataset.coords = savedCoords || 'edges';
    if (opts.footer) {
      const fh = Math.round(size * .055);
      const out = document.createElement('canvas');
      out.width = cv.width; out.height = cv.height + fh * 2;
      const og = out.getContext('2d');
      if (!opts.transparent) {
        og.fillStyle = this.css('--surf');
        og.fillRect(0, 0, out.width, out.height);
      }
      og.drawImage(cv, 0, fh * 2);
      og.fillStyle = this.css('--gold');
      og.font = `700 ${Math.round(fh * .52)}px Cinzel, 'Times New Roman', serif`;
      og.textAlign = 'center'; og.textBaseline = 'middle';
      og.fillText('Z Q U O R I D O R', out.width / 2, fh);
      og.fillStyle = this.css('--txt2');
      og.font = `${Math.round(fh * .40)}px 'JetBrains Mono', monospace`;
      og.fillText(`${opts.footer.name0}  2014  ${opts.footer.name1}`,
                  out.width / 2, fh * 3);
      return out;
    }
    return cv;
  }

  // Vector twin of the export renderer: the same scene emitted as SVG for
  // print or docs. Honours the same opts contract as renderExport.
  toSVG(opts = {}) {
    const size = opts.size || 1600;
    const tmp = document.createElement('div');
    tmp.style.display = 'none';
    document.body.appendChild(tmp);
    const cssOf = name => getComputedStyle(tmp).getPropertyValue(name).trim() ||
                          getComputedStyle(document.documentElement).getPropertyValue(name).trim();
    const esc = s => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;');
    const ds = document.documentElement.dataset;
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
      if (!Number.isInteger(this.pawn[pl])) continue;
      const d = this.pawn[pl];
      const cx = M + (d % 9) * U + C / 2, cy = M + Math.floor(d / 9) * U + C / 2;
      const pawnScale = { small: .85, large: 1.15 }[ds.pawnSize] || 1;
      const R = .30 * C * pawnScale;
      b += `<circle cx="${cx.toFixed(1)}" cy="${cy.toFixed(1)}" r="${R.toFixed(1)}" fill="${esc(cssOf('--p' + pl))}" stroke="${esc(cssOf('--p' + pl + '-deep'))}"/>`;
    }
    if (opts.coords !== false) {
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
    if (opts.footer) {
      const fh = size * .055;
      b = b.replace(`height="${size}"`, `height="${size + fh * 2}"`)
           .replace(`viewBox="0 0 ${size} ${size}"`, `viewBox="0 0 ${size} ${size + fh * 2}"`);
      b += `<text x="${size / 2}" y="${fh * .55}" text-anchor="middle" fill="${esc(gold)}" font-family="Cinzel, serif" font-weight="700" font-size="${(fh * .52).toFixed(0)}">Z Q U O R I D O R</text>`;
      b += `<text x="${size / 2}" y="${size + fh * 1.45}" text-anchor="middle" fill="${esc(cssOf('--txt2'))}" font-family="monospace" font-size="${(fh * .40).toFixed(0)}">${esc(opts.footer.name0)} 2014 ${esc(opts.footer.name1)}</text>`;
    }
    b += `</svg>`;
    tmp.remove();
    return b;
  }
}

window.QBoard = QBoard;
