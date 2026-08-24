// board.js -- QBoard: premium canvas board for Zquoridor (plan gui-premium.md,
// sections 2, 5.3, 6.1, 17). One <canvas>, DPR-aware, layered repaint: static
// layer (frame, cells, grooves, coordinates) is cached offscreen and blitted;
// the dynamic layer (walls, pawns, ghosts, overlays) redraws on state change.
// All colours come from CSS custom properties read at draw time, so theme
// switching needs no canvas-specific palette. Board dressing (frame style,
// wall finish, cell separation, coordinates mode, scale) and the extra pawn
// styles follow plan section 17.
'use strict';

// Deterministic PRNG for the marble vein texture (mulberry32): same seed,
// same veins, so the static layer cache stays valid.
function qrMulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

class QBoard {
  constructor(canvas, opts = {}) {
    this.cv = canvas;
    this.ctx = canvas.getContext('2d');
    this.onChange = opts.onChange || (() => {});
    this.fixedSide = opts.fixedSide || 0;   // export renders / mini previews
    // game state mirrored from the engine (display orientation: player 0 at
    // the bottom, moving up, unless flipped)
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
    // Analysis line preview (plan 5.3): { walls:[{o,r,c}] engine coords,
    // pawns:[engCell] in step order, color }. Drawn translucent under the
    // pieces; set by the analysis PV rows, cleared by clearGhost/Esc.
    this.linePreview = null;
    this.turn = 0;
    this.themeDirty = true;
    this._sideApplied = -1;
    if (!this.fixedSide) new ResizeObserver(() => this.fit()).observe(canvas.parentElement);
    this.fit();
  }

  css(name) {
    // Detached canvases (export renders) have no parent element; fall back
    // to the document root so token lookups keep working.
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
      // measure the LAYOUT ZONE, not the wrapper: the wrapper wraps the canvas,
      // which would otherwise collapse to its own content size (chicken/egg)
      const zone = this.cv.closest('#boardZone') ||
                   this.cv.parentElement.parentElement ||
                   this.cv.parentElement;
      const zw = zone.clientWidth || 320, zh = zone.clientHeight || 320;
      const scale = Math.max(0.8, Math.min(1, parseFloat(this.ds().boardScale) || 1));
      side = Math.max(220, Math.floor(Math.min(zw, zh) * scale) - 6);
      if (this._sideApplied === side) { this.render(); return; }
      const wrap = this.cv.parentElement;
      wrap.style.width = side + 'px';
      wrap.style.height = side + 'px';
    }
    this._sideApplied = side;
    const dpr = Math.min(window.devicePixelRatio || 1, 3);
    this.cssSide = side;
    this.S = side;   // paintStatic destructures {S}; leaving it unset made
                     // every frame style draw with NaN coordinates and the
                     // beveled frame throw on createLinearGradient
    if (!this.fixedSide) {
      this.cv.width = Math.round(side * dpr);
      this.cv.height = Math.round(side * dpr);
      this.cv.style.width = side + 'px';
      this.cv.style.height = side + 'px';
    }
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    // geometry per plan 6.1: S = 9C + 8G. G stays slim (classic look): it is
    // the wall thickness and the breathing room between cells.
    this.G = Math.max(4, Math.min(9, 0.14 * (side / 10.6)));
    const coordsMode = this.ds().coords || 'edges';
    this.M = coordsMode !== 'off' ? 0.42 * ((side - 8 * this.G) / 9) : 0;
    this.C = (side - 2 * this.M - 8 * this.G) / 9;
    this.U = this.C + this.G;
    this.themeDirty = true;
    this.render();
  }

  cellXY(r, c) { return { x: this.M + c * this.U, y: this.M + r * this.U }; }
  cellCenter(r, c) {
    const p = this.cellXY(r, c);
    return { x: p.x + this.C / 2, y: p.y + this.C / 2 };
  }
  anchorCenter(r, c) {   // groove intersection south-east of display cell (r,c)
    return { x: this.M + (c + 1) * this.U - this.G / 2,
             y: this.M + (r + 1) * this.U - this.G / 2 };
  }

  // ---- coordinate conversions -------------------------------------------
  // The human player (side 0) sits at the BOTTOM moving up: display row
  // = 8 - engine row unless the board is flipped.
  engPawnToDisp(cell) {
    const r = Math.floor(cell / 9), c = cell % 9;
    return this.flipped ? r * 9 + c : (8 - r) * 9 + c;
  }
  dispPawnToEng(r, c) { return this.flipped ? r * 9 + c : (8 - r) * 9 + c; }
  engWallToDisp(o, r, c) { return this.flipped ? [o, r, c] : [o, 7 - r, c]; }
  dispWallToEng(o, r, c) { return this.flipped ? [o, r, c] : [o, 7 - r, c]; }
  // Absolute algebraic name (file + engine rank), independent of the display
  // orientation: rank 1 is always player 0's home row, matching the QFEN.
  engAlgName(engCell) {
    return 'abcdefghi'[engCell % 9] + (Math.floor(engCell / 9) + 1);
  }
  algName(dispCell) {
    return this.engAlgName(this.dispPawnToEng(Math.floor(dispCell / 9), dispCell % 9));
  }

  setData(pawnEng, wallsHEng, wallsVEng, lastMove) {
    this.pawn = [this.engPawnToDisp(pawnEng[0]), this.engPawnToDisp(pawnEng[1])];
    // Wall slots mirror like the pawns: the board arrays hold display
    // coordinates (engine row r lands at display row 7-r), so the paint and
    // export paths can index them directly. The map is a bijection, so every
    // display slot is rewritten and no stale bit survives a position change.
    for (let r = 0; r < 8; r++) for (let c = 0; c < 8; c++) {
      const s = r * 8 + c, d = this.flipped ? s : (7 - r) * 8 + c;
      this.wallH[d] = wallsHEng[s] | 0;
      this.wallV[d] = wallsVEng[s] | 0;
    }
    this.lastMove = lastMove || null;
    this.render();
  }

  render() {
    const ctx = this.ctx, S = this.cssSide;
    ctx.clearRect(0, 0, S, S);
    this.drawStatic(ctx);
    this.drawDynamic(ctx);
    if (this.onChange) this.onChange();
  }

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

  paintStatic(g) {
    const { C, G, U, M, S } = this;
    const ds = this.ds();
    const frame = this.css('--frame'), groove = this.css('--groove');
    const ca = this.css('--cell-a'), cb = this.css('--cell-b');
    const frameStyle = ds.frame || 'hairline';
    const cellSep = ds.cellSep || 'grooves';
    // Classic base: one flat surface in the groove color, cells drawn on top
    // with a thin light edge -- the pre-premium look (flat, quiet).
    g.fillStyle = frameStyle === 'none' ? ca : groove;
    this.rr(g, 0, 0, S, S, 10); g.fill();
    if (frameStyle !== 'none') {
      g.strokeStyle = this.css('--bor2'); g.lineWidth = 1;
      this.rr(g, .5, .5, S - 1, S - 1, 10); g.stroke();
    }
    if (frameStyle === 'gilded') {
      g.strokeStyle = this.css('--gold'); g.lineWidth = 2;
      this.rr(g, 1.5, 1.5, S - 3, S - 3, 9); g.stroke();
    } else if (frameStyle === 'beveled') {
      const be = g.createLinearGradient(0, 0, 0, S);
      be.addColorStop(0, 'rgba(255,255,255,.08)');
      be.addColorStop(.05, 'rgba(0,0,0,.18)');
      be.addColorStop(.95, 'rgba(0,0,0,.15)');
      be.addColorStop(1, 'rgba(255,255,255,.05)');
      g.fillStyle = be; this.rr(g, 0, 0, S, S, 10); g.fill();
    }
    // cells: flat single tone + thin light grid line (classic). The 'inlaid'
    // dressing swaps the line for gold; 'flat' drops the lines entirely.
    const twoTone = false;   // classic: uniform cells
    for (let r = 0; r < 9; r++) for (let c = 0; c < 9; c++) {
      const p = this.cellXY(r, c);
      g.fillStyle = twoTone ? (((r + c) & 1) ? cb : ca) : ca;
      if (cellSep === 'grooves' || cellSep === 'inlaid') {
        g.fillRect(p.x - G / 2 + .5, p.y - G / 2 + .5, C + G - 1, C + G - 1);
        g.strokeStyle = cellSep === 'inlaid'
          ? this.css('--gold') : 'rgba(255,255,255,.13)';
        g.globalAlpha = cellSep === 'inlaid' ? .35 : 1;
        g.lineWidth = 1;
        g.strokeRect(p.x + .5, p.y + .5, C - 1, C - 1);
        g.globalAlpha = 1;
      } else {
        g.fillRect(p.x, p.y, C + .5, C + .5);
      }
    }
    // Carrara Marble signature theme: deterministic procedural veins drawn
    // once into the static layer -- zero per-frame cost (plan 17.1 #7).
    if ((ds.board || '') === 'marble') {
      const rnd = qrMulberry32(0xCAFE);
      g.save();
      g.beginPath(); g.rect(M - G / 2, M - G / 2, 9 * U, 9 * U); g.clip();
      for (let v = 0; v < 26; v++) {
        g.strokeStyle = rnd() > .5 ? 'rgba(120,125,140,.16)' : 'rgba(70,74,88,.12)';
        g.lineWidth = .6 + rnd() * 1.8;
        let x = M + rnd() * 9 * U, y = M - 4;
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
    // goal markers: a quiet 2px line just inside the top and bottom play
    // edges (subtle orientation cue, classic-quiet).
    const p0c = this.css('--p0'), p1c = this.css('--p1');
    g.globalAlpha = .22;
    g.fillStyle = this.flipped ? p0c : p1c;
    g.fillRect(M - G / 2, M - G / 2, 9 * U, 2);
    g.fillStyle = this.flipped ? p1c : p0c;
    g.fillRect(M - G / 2, S - M - G / 2 - 2, 9 * U, 2);
    g.globalAlpha = 1;
    const coordsMode = ds.coords || 'edges';
    if (coordsMode !== 'off' && C >= 26) {
      g.fillStyle = this.css('--txt2'); g.globalAlpha = .6;
      g.font = `${Math.max(9, .52 * C)}px 'JetBrains Mono', monospace`;
      g.textAlign = 'center'; g.textBaseline = 'middle';
      for (let c = 0; c < 9; c++) {
        const x = this.cellCenter(0, c).x;
        g.fillText('abcdefghi'[c], x, S - M / 2 + 1);
        g.fillText('abcdefghi'[c], x, M / 2 - 1);
      }
      g.textAlign = 'left';
      for (let r = 0; r < 9; r++) {
        const y = this.cellCenter(r, 0).y;
        // absolute rank: display row r is engine row (flipped ? r : 8-r)
        g.fillText(String(this.flipped ? r + 1 : 9 - r), M / 2 - 3, y);
      }
      // "every cell" study mode: faint file+rank in each cell corner
      if (coordsMode === 'all' && C >= 40) {
        g.globalAlpha = .28;
        g.font = `${Math.max(7, .26 * C)}px 'JetBrains Mono', monospace`;
        g.textAlign = 'left'; g.textBaseline = 'top';
        for (let r = 0; r < 9; r++) for (let c = 0; c < 9; c++) {
          const p = this.cellXY(r, c);
          const engR = this.flipped ? r : 8 - r;
          g.fillText('abcdefghi'[c] + (engR + 1), p.x + 3, p.y + 2);
        }
      }
      g.globalAlpha = 1;
    }
  }

  drawDynamic(g) {
    const { C } = this;
    if (this.paths) for (const line of this.paths) this.drawPathLine(g, line);
    if (this.linePreview) this.drawLinePreview(g);
    for (let r = 0; r < 8; r++) for (let c = 0; c < 8; c++) {
      if (this.wallH[r * 8 + c]) this.drawWall(g, 0, r, c, false);
      if (this.wallV[r * 8 + c]) this.drawWall(g, 1, r, c, false);
    }
    if (this.lastMove) {
      if (this.lastMove.type === 'pawn') {
        const p = this.cellXY(this.lastMove.r, this.lastMove.c);
        g.fillStyle = this.css('--gold-dim');
        g.globalAlpha = .8;
        g.fillRect(p.x, p.y, C, C);
        g.globalAlpha = 1;
        g.strokeStyle = this.css('--gold'); g.lineWidth = 1;
        g.strokeRect(p.x + .5, p.y + .5, C - 1, C - 1);
      } else {
        const [o, r, c] = this.engWallToDisp(this.lastMove.o, this.lastMove.r, this.lastMove.c);
        g.save(); g.globalAlpha = .9;
        this.drawWall(g, o, r, c, false, this.css('--gold'));
        g.restore();
      }
    }
    if (this.ghost) this.drawGhost(g);
    const havePawns = Array.isArray(this.pawn) && this.pawn.length === 2 &&
                      Number.isInteger(this.pawn[0]) && Number.isInteger(this.pawn[1]);
    if (this.dots.length && !this.ghost && havePawns) for (const d of this.dots) {
      const ctr = this.cellCenter(Math.floor(d / 9), d % 9);
      const me = this.cellCenter(Math.floor(this.pawn[this.turn] / 9), this.pawn[this.turn] % 9);
      const jump = Math.abs(ctr.x - me.x) + Math.abs(ctr.y - me.y) > this.U * 1.5;
      g.beginPath();
      if (jump) {
        g.arc(ctr.x, ctr.y, .21 * C, 0, 7); g.lineWidth = Math.max(2, .07 * C);
        g.strokeStyle = this.dotColor(); g.stroke();
      } else {
        g.arc(ctr.x, ctr.y, .16 * C, 0, 7);
        g.fillStyle = this.dotColor(); g.globalAlpha = .55; g.fill(); g.globalAlpha = 1;
      }
    }
    if (havePawns) {
      this.drawPawn(g, this.pawn[0], 0);
      this.drawPawn(g, this.pawn[1], 1);
    }
    if (this.selected >= 0) {
      const ctr = this.cellCenter(Math.floor(this.selected / 9), this.selected % 9);
      g.beginPath(); g.arc(ctr.x, ctr.y, .40 * C, 0, 7);
      g.strokeStyle = this.css('--gold'); g.lineWidth = 2;
      g.setLineDash([4, 3]); g.stroke(); g.setLineDash([]);
    }
  }

  dotColor() { return this.css('--gold2'); }

  setTurn(t) { this.turn = t; }

  // Analysis PV preview: every wall of the line as a translucent beam (wall
  // slots are order-independent) plus numbered rings on the pawn destinations.
  drawLinePreview(g) {
    const lp = this.linePreview;
    const col = lp.color || this.css('--blue');
    g.save();
    for (const w of lp.walls || []) {
      const [do_, dr, dc] = this.engWallToDisp(w.o, w.r, w.c);
      g.globalAlpha = .38;
      this.drawWall(g, do_, dr, dc, false, col);
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
    if (!line.cells || line.cells.length < 2) return;
    g.save();
    g.strokeStyle = line.color || this.css('--p0');
    g.globalAlpha = .32; g.lineWidth = .10 * this.C;
    g.lineCap = 'round'; g.lineJoin = 'round';
    g.beginPath();
    line.cells.forEach((cell, i) => {
      const ctr = this.cellCenter(Math.floor(cell / 9), cell % 9);
      i ? g.lineTo(ctr.x, ctr.y) : g.moveTo(ctr.x, ctr.y);
    });
    g.stroke(); g.restore();
  }

  wallRect(o, r, c) {
    const a = this.anchorCenter(r, c), T = this.G, L = 2 * this.C + this.G;
    return o === 0
      ? { x: a.x - this.C - this.G / 2, y: a.y - T / 2, w: L, h: T }
      : { x: a.x - T / 2, y: a.y - this.C - this.G / 2, w: T, h: L };
  }

  drawWall(g, o, r, c, ghostStyle, outlineOverride) {
    const rc = this.wallRect(o, r, c);
    const wall = this.css('--wall'), edge = this.css('--wall-edge');
    const finish = this.ds().wallFinish || 'beveled';
    g.save();
    if (finish !== 'flat') {
      g.fillStyle = 'rgba(0,0,0,.32)';
      if (o === 0) { this.rr(g, rc.x + 1, rc.y + 2, rc.w, rc.h, 3); g.fill(); }
      else { this.rr(g, rc.x + 2, rc.y + 1, rc.w, rc.h, 3); g.fill(); }
    }
    if (finish === 'glossy') {
      const grad = o === 0
        ? g.createLinearGradient(0, rc.y, 0, rc.y + rc.h)
        : g.createLinearGradient(rc.x, 0, rc.x + rc.w, 0);
      grad.addColorStop(0, edge); grad.addColorStop(.18, wall);
      grad.addColorStop(.42, 'rgba(255,255,255,.55)');
      grad.addColorStop(.60, wall); grad.addColorStop(1, edge);
      g.fillStyle = grad;
    } else if (finish === 'flat') {
      g.fillStyle = wall;
    } else {
      const grad = o === 0
        ? g.createLinearGradient(0, rc.y, 0, rc.y + rc.h)
        : g.createLinearGradient(rc.x, 0, rc.x + rc.w, 0);
      grad.addColorStop(0, edge); grad.addColorStop(.22, wall);
      grad.addColorStop(.78, wall); grad.addColorStop(1, edge);
      g.fillStyle = grad;
    }
    this.rr(g, rc.x, rc.y, rc.w, rc.h, finish === 'flat' ? 0 : 3); g.fill();
    if (finish !== 'flat') {
      g.lineWidth = outlineOverride ? 1.8 : 1;
      g.strokeStyle = outlineOverride || edge;
      g.stroke();
    } else if (outlineOverride) {
      g.lineWidth = 1.8; g.strokeStyle = outlineOverride;
      this.rr(g, rc.x, rc.y, rc.w, rc.h, 0); g.stroke();
    }
    if (finish === 'beveled' || finish === 'etched') {
      g.strokeStyle = edge; g.globalAlpha = finish === 'etched' ? .95 : .7;
      g.lineWidth = 1;
      g.beginPath();
      if (o === 0) { const mx = rc.x + rc.w / 2; g.moveTo(mx, rc.y + 1); g.lineTo(mx, rc.y + rc.h - 1); }
      else { const my = rc.y + rc.h / 2; g.moveTo(rc.x + 1, my); g.lineTo(rc.x + rc.w - 1, my); }
      g.stroke();
    }
    if (finish === 'etched') {
      // inner shadow along the top/left of the beam
      g.globalAlpha = .35;
      g.strokeStyle = 'rgba(0,0,0,.8)'; g.lineWidth = 1;
      this.rr(g, rc.x + 1, rc.y + 1, rc.w - 2, rc.h - 2, 2); g.stroke();
    }
    g.globalAlpha = 1;
    g.restore();
  }

  drawGhost(g) {
    const gh = this.ghost, o = gh.o, r = gh.r, c = gh.c;
    const rc = this.wallRect(o, r, c);
    const st = gh.state;
    const alpha = st === 'bad' ? .20 : st === 'pending' ? 1 : .55;
    const col = st === 'ok' ? this.css('--green')
              : st === 'assisted' ? this.css('--gold')
              : st === 'pending' ? this.css('--gold')
              : this.css('--red');
    g.save(); g.globalAlpha = alpha;
    this.drawWall(g, o, r, c, true);
    g.restore();
    g.save();
    g.strokeStyle = col; g.lineWidth = 2;
    if (st === 'ok') { g.shadowColor = col; g.shadowBlur = 6; }
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

  drawPawn(g, dispCell, pl) {
    const { C } = this;
    const r = Math.floor(dispCell / 9), c = dispCell % 9;
    const ctr = this.cellCenter(r, c);
    const px = this.css('--p' + pl), pli = this.css('--p' + pl + '-light'),
          pd = this.css('--p' + pl + '-deep');
    const sizeMul = { small: .85, large: 1.15 }[this.ds().pawnSize] || 1;
    const R = .31 * C * sizeMul;
    g.save();
    const shadow = this.ds().pawnShadow || 'soft';
    if (shadow !== 'off') {
      g.fillStyle = 'rgba(0,0,0,' + (shadow === 'deep' ? '.5' : '.35') + ')';
      g.beginPath();
      g.ellipse(ctr.x, ctr.y + R * .55, R * .95, R * .38, 0, 0, 7); g.fill();
    }
    const grd = g.createRadialGradient(ctr.x - R * .35, ctr.y - R * .45, R * .15, ctr.x, ctr.y, R * 1.05);
    grd.addColorStop(0, pli); grd.addColorStop(.55, px); grd.addColorStop(1, pd);
    g.fillStyle = grd; g.strokeStyle = pd; g.lineWidth = 1;
    let style = this.ds().pawn || 'disc';
    // Distinct shapes (plan 17.3): force the two players to differ in
    // silhouette for colour-independent play; automatic in the noir theme.
    const distinct = this.ds().distinctShapes === '1' || this.ds().board === 'noir';
    if (distinct && pl === 1 && (style === 'disc' || style === 'pillar')) style = 'crown';
    if (distinct && pl === 0 && style === 'crown') style = 'disc';
    if (style === 'pawnChess') {
      // full chess-pawn silhouette: bulb, collar, flared base
      g.beginPath();
      g.arc(ctr.x, ctr.y - R * .34, R * .46, 0, 7);
      g.moveTo(ctr.x - R * .26, ctr.y - R * .02);
      g.quadraticCurveTo(ctr.x - R * .40, ctr.y - R * .16, ctr.x - R * .30, ctr.y - R * .30);
      g.moveTo(ctr.x + R * .26, ctr.y - R * .02);
      g.quadraticCurveTo(ctr.x + R * .40, ctr.y - R * .16, ctr.x + R * .30, ctr.y - R * .30);
      g.moveTo(ctr.x - R * .26, ctr.y - R * .02);
      g.lineTo(ctr.x - R * .52, ctr.y + R * .58);
      g.quadraticCurveTo(ctr.x, ctr.y + R * .34, ctr.x + R * .52, ctr.y + R * .58);
      g.lineTo(ctr.x + R * .26, ctr.y - R * .02);
      g.closePath(); g.fill(); g.stroke();
    } else if (style === 'beacon') {
      // ring with a light arc pointing at that player's goal (didactic)
      const dir = pl === 0 ? 1 : -1;   // display-space: p0 goal is down when not flipped
      g.beginPath(); g.arc(ctr.x, ctr.y, R * .78, 0, 7);
      g.lineWidth = Math.max(3, R * .38); g.strokeStyle = grd; g.stroke();
      g.beginPath();
      g.arc(ctr.x, ctr.y, R * .78, dir > 0 ? Math.PI * .25 : Math.PI * 1.25,
            dir > 0 ? Math.PI * .75 : Math.PI * 1.75);
      g.lineWidth = Math.max(3, R * .38);
      g.strokeStyle = this.css('--gold2'); g.stroke();
    } else if (style === 'pillar') {
      g.beginPath();
      g.arc(ctr.x, ctr.y - R * .25, R * .58, 0, 7);
      g.moveTo(ctr.x - R * .34, ctr.y + R * .05);
      g.quadraticCurveTo(ctr.x, ctr.y + R * .18, ctr.x + R * .34, ctr.y + R * .05);
      g.lineTo(ctr.x + R * .52, ctr.y + R * .62);
      g.quadraticCurveTo(ctr.x, ctr.y + R * .40, ctr.x - R * .52, ctr.y + R * .62);
      g.closePath(); g.fill(); g.stroke();
    } else if (style === 'crown') {
      // crenellated crown: must differ from pillar in silhouette, or the
      // distinct-shapes mapping (pillar -> crown for side 1) does nothing
      g.beginPath();
      g.moveTo(ctr.x - R * .55, ctr.y + R * .62);
      g.lineTo(ctr.x - R * .55, ctr.y - R * .02);
      g.lineTo(ctr.x - R * .27, ctr.y + R * .14);
      g.lineTo(ctr.x, ctr.y - R * .40);
      g.lineTo(ctr.x + R * .27, ctr.y + R * .14);
      g.lineTo(ctr.x + R * .55, ctr.y - R * .02);
      g.lineTo(ctr.x + R * .55, ctr.y + R * .62);
      g.closePath(); g.fill(); g.stroke();
    } else if (style === 'rune') {
      const k = R * .95;
      g.beginPath();
      g.moveTo(ctr.x, ctr.y - k);
      g.lineTo(ctr.x + k * .87, ctr.y - k * .5);
      g.lineTo(ctr.x + k * .87, ctr.y + k * .5);
      g.lineTo(ctr.x, ctr.y + k);
      g.lineTo(ctr.x - k * .87, ctr.y + k * .5);
      g.lineTo(ctr.x - k * .87, ctr.y - k * .5);
      g.closePath(); g.fill(); g.stroke();
      g.strokeStyle = this.css('--bg'); g.lineWidth = Math.max(1.5, C * .05);
      const dir = pl === 0 ? -1 : 1;
      g.beginPath();
      g.moveTo(ctr.x - R * .3, ctr.y + dir * R * .22);
      g.lineTo(ctr.x, ctr.y - dir * R * .22);
      g.lineTo(ctr.x + R * .3, ctr.y + dir * R * .22);
      g.stroke();
    } else {
      g.beginPath(); g.arc(ctr.x, ctr.y, R, 0, 7); g.fill(); g.stroke();
    }
    if (this.turn === pl) {
      g.beginPath(); g.arc(ctr.x, ctr.y, .40 * C, 0, 7);
      g.strokeStyle = this.css('--gold'); g.lineWidth = 1.5;
      g.shadowColor = this.css('--gold-glow'); g.shadowBlur = 5;
      g.stroke();
    }
    g.restore();
  }

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

  // ---- image export (plan section 16.5) ----------------------------------
  // Renders the current position into a detached canvas at a fixed size,
  // independent of the on-screen board, with an optional footer strip
  // carrying both player names and a small wordmark. opts:
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
      og.fillText(`${opts.footer.name0}  \u2014  ${opts.footer.name1}`,
                  out.width / 2, fh * 3);
      return out;
    }
    return cv;
  }

  // Vector twin of the export renderer: same scene emitted as SVG for print
  // or docs. Honours the same opts contract as renderExport.
  toSVG(opts = {}) {
    const size = opts.size || 1600;
    const tmp = document.createElement('div');
    tmp.style.display = 'none';
    document.body.appendChild(tmp);
    const cssOf = name => getComputedStyle(tmp).getPropertyValue(name).trim() ||
                          getComputedStyle(document.documentElement).getPropertyValue(name).trim();
    const esc = s => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;');
    const frame = cssOf('--frame'), groove = cssOf('--groove');
    const ca = cssOf('--cell-a'), cb = cssOf('--cell-b'), wall = cssOf('--wall');
    const edge = cssOf('--wall-edge'), gold = cssOf('--gold'), txt2 = cssOf('--txt2');
    const G = size * 0.0115, M = size * 0.0195;
    const C = (size - 2 * M - 8 * G) / 9, U = C + G;
    let b = `<svg xmlns="http://www.w3.org/2000/svg" width="${size}" height="${size}" viewBox="0 0 ${size} ${size}">`;
    if (!opts.transparent) b += `<rect width="${size}" height="${size}" fill="${esc(cssOf('--bg'))}"/>`;
    b += `<rect width="${size}" height="${size}" rx="26" fill="${esc(frame)}"/>`;
    for (let r = 0; r < 9; r++) for (let c = 0; c < 9; c++)
      b += `<rect x="${(M + c * U).toFixed(1)}" y="${(M + r * U).toFixed(1)}" width="${C.toFixed(1)}" height="${C.toFixed(1)}" fill="${esc(((r + c) & 1) ? cb : ca)}"/>`;
    for (let i = 0; i <= 9; i++) {
      const t = M + i * U - G / 2;
      b += `<line x1="${M - G / 2}" y1="${t.toFixed(1)}" x2="${size - M + G / 2}" y2="${t.toFixed(1)}" stroke="${esc(groove)}" stroke-width="${G.toFixed(1)}"/>`;
      b += `<line x1="${t.toFixed(1)}" y1="${M - G / 2}" x2="${t.toFixed(1)}" y2="${size - M + G / 2}" stroke="${esc(groove)}" stroke-width="${G.toFixed(1)}"/>`;
    }
    for (let r = 0; r < 8; r++) for (let c = 0; c < 8; c++) {
      if (this.wallH[r * 8 + c]) {
        const x = M + c * U, y = M + (r + 1) * U - G / 2;
        b += `<rect x="${x.toFixed(1)}" y="${y.toFixed(1)}" width="${(2 * C + G).toFixed(1)}" height="${G.toFixed(1)}" rx="4" fill="${esc(wall)}" stroke="${esc(edge)}"/>`;
      }
      if (this.wallV[r * 8 + c]) {
        const x = M + (c + 1) * U - G / 2, y = M + r * U;
        b += `<rect x="${x.toFixed(1)}" y="${y.toFixed(1)}" width="${G.toFixed(1)}" height="${(2 * C + G).toFixed(1)}" rx="4" fill="${esc(wall)}" stroke="${esc(edge)}"/>`;
      }
    }
    for (let pl = 0; pl < 2; pl++) {
      if (!Number.isInteger(this.pawn[pl])) continue;
      const d = this.pawn[pl];
      const cx = M + (d % 9) * U + C / 2, cy = M + Math.floor(d / 9) * U + C / 2;
      const R = .31 * C;
      b += `<circle cx="${cx.toFixed(1)}" cy="${cy.toFixed(1)}" r="${R.toFixed(1)}" fill="${esc(cssOf('--p' + pl))}" stroke="${esc(cssOf('--p' + pl + '-deep'))}"/>`;
    }
    if (opts.coords !== false) {
      b += `<g fill="${esc(txt2)}" font-family="monospace" font-size="${(C * .16).toFixed(0)}" opacity=".6">`;
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
    if (opts.footer) {
      const fh = size * .055;
      b = b.replace(`height="${size}"`, `height="${size + fh * 2}"`)
           .replace(`viewBox="0 0 ${size} ${size}"`, `viewBox="0 0 ${size} ${size + fh * 2}"`);
      b += `<text x="${size / 2}" y="${fh * .55}" text-anchor="middle" fill="${esc(gold)}" font-family="Cinzel, serif" font-weight="700" font-size="${(fh * .52).toFixed(0)}">Z Q U O R I D O R</text>`;
      b += `<text x="${size / 2}" y="${size + fh * 1.45}" text-anchor="middle" fill="${esc(txt2)}" font-family="monospace" font-size="${(fh * .40).toFixed(0)}">${esc(opts.footer.name0)} \u2014 ${esc(opts.footer.name1)}</text>`;
    }
    b += `</svg>`;
    tmp.remove();
    return b;
  }
}

window.QBoard = QBoard;
