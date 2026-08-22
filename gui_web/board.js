// board.js -- QBoard: premium canvas board for Zquoridor (plan gui-premium.md,
// sections 2, 5.3, 6.1). One <canvas>, DPR-aware, layered repaint: static layer
// (frame, cells, grooves, coordinates) is cached offscreen and blitted; the
// dynamic layer (walls, pawns, ghosts, overlays) redraws on state change.
// All colours come from CSS custom properties read at draw time, so theme
// switching needs no canvas-specific palette.
'use strict';

class QBoard {
  constructor(canvas, opts = {}) {
    this.cv = canvas;
    this.ctx = canvas.getContext('2d');
    this.onChange = opts.onChange || (() => {});
    // game state mirrored from the engine (display orientation: player 0 at
    // the bottom, moving up, unless flipped)
    this.flipped = false;
    this.pawn = [/*disp cells*/];
    this.wallH = new Uint8Array(64);
    this.wallV = new Uint8Array(64);
    this.lastMove = null;          // {type:'pawn',r,c} | {type:'wall',o,r,c}
    this.dots = [];                // legal pawn destinations (display cells)
    this.selected = -1;
    this.paths = null;             // [{cells:[...]}] shortest-path overlay
    this.ghost = null;             // {o,r,c,state:'ok'|'assisted'|'bad'|'pending'}
    this.ghostFrom = null;         // anchor the pointer actually hit (assist)
    this.themeDirty = true;
    this._resizeObserved = false;
    new ResizeObserver(() => this.fit()).observe(canvas.parentElement);
    this.fit();
  }

  css(name) {
    return getComputedStyle(this.cv.parentElement).getPropertyValue(name).trim();
  }

  fit() {
    const box = this.cv.parentElement.getBoundingClientRect();
    const side = Math.max(200, Math.floor(Math.min(box.width, box.height)));
    const dpr = Math.min(window.devicePixelRatio || 1, 3);
    this.cssSide = side;
    this.cv.width = Math.round(side * dpr);
    this.cv.height = Math.round(side * dpr);
    this.cv.style.width = side + 'px';
    this.cv.style.height = side + 'px';
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    // geometry per plan 6.1
    this.G = Math.max(8, Math.min(14, 0.20 * (side / 10.42)));
    // solve C from S = 9C + 8G (+ margins handled by padding inside S)
    const hasCoords = document.documentElement.dataset.coords !== 'off';
    this.M = hasCoords ? 0.42 * ((side - 8 * this.G) / 9) : 0;
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
  engPawnToDisp(cell) {
    const r = Math.floor(cell / 9), c = cell % 9;
    return this.flipped ? (8 - r) * 9 + c : r * 9 + c;
  }
  dispPawnToEng(r, c) { return this.flipped ? (8 - r) * 9 + c : r * 9 + c; }
  engWallToDisp(o, r, c) { return this.flipped ? [o, 7 - r, c] : [o, r, c]; }
  dispWallToEng(o, r, c) { return this.flipped ? [o, 7 - r, c] : [o, r, c]; }
  algName(dispCell) {
    const r = Math.floor(dispCell / 9), c = dispCell % 9;
    return 'abcdefghi'[c] + (this.flipped ? r + 1 : 9 - r);
  }

  setData(pawnEng, wallsHEng, wallsVEng, lastMove) {
    this.pawn = [this.engPawnToDisp(pawnEng[0]), this.engPawnToDisp(pawnEng[1])];
    for (let s = 0; s < 64; s++) {
      this.wallH[s] = wallsHEng[s] | 0;
      this.wallV[s] = wallsVEng[s] | 0;
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

  // static layer: frame, cells, grooves, coordinates (cached offscreen)
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

  paintStatic(g) {
    const { C, G, U, M, S } = this;
    const frame = this.css('--frame'), groove = this.css('--groove');
    const ca = this.css('--cell-a'), cb = this.css('--cell-b');
    // frame slab
    g.fillStyle = frame;
    this.rr(g, 0, 0, S, S, 12); g.fill();
    // cells
    for (let r = 0; r < 9; r++) for (let c = 0; c < 9; c++) {
      const p = this.cellXY(r, c);
      g.fillStyle = ((r + c) & 1) ? cb : ca;
      g.fillRect(p.x, p.y, C + 0.5, C + 0.5);
    }
    // grooves
    g.strokeStyle = groove; g.lineWidth = G; g.lineCap = 'butt';
    g.beginPath();
    for (let i = 0; i <= 9; i++) {
      const t = M + i * U - G / 2;
      g.moveTo(M - G / 2, Math.min(t, M + 8 * U + G / 2));
      g.lineTo(S - M + G / 2, t);
      g.moveTo(t, M - G / 2);
      g.lineTo(t, S - M + G / 2);
    }
    g.stroke();
    // goal edges
    const p0 = this.css('--p0'), p1 = this.css('--p1');
    g.globalAlpha = .45;
    g.fillStyle = this.flipped ? p0 : p1;
    g.fillRect(M - G / 2, M - G / 2 - 4, 9 * U, 3);
    g.fillStyle = this.flipped ? p1 : p0;
    g.fillRect(M - G / 2, S - M - G / 2 + 1, 9 * U, 3);
    g.globalAlpha = 1;
    // coordinates
    if (document.documentElement.dataset.coords !== 'off' && C >= 26) {
      g.fillStyle = this.css('--txt2'); g.globalAlpha = .6;
      g.font = `${Math.max(9, .52 * C)}px 'JetBrains Mono', monospace`;
      g.textAlign = 'center'; g.textBaseline = 'middle';
      for (let c = 0; c < 9; c++) {
        const x = this.cellCenter(0, c).x;
        const file = 'abcdefghi'[c];
        g.fillText(file, x, S - M / 2 + 1);
        g.fillText(file, x, M / 2 - 1);
      }
      g.textAlign = 'left';
      for (let r = 0; r < 9; r++) {
        const y = this.cellCenter(r, 0).y;
        const rank = String(this.flipped ? r + 1 : 9 - r);
        g.fillText(rank, M / 2 - (M > C * .3 ? 4 : 2), y);
      }
      g.globalAlpha = 1;
    }
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

  drawDynamic(g) {
    const { C, G } = this;
    // path overlay under everything else
    if (this.paths) for (const line of this.paths) this.drawPathLine(g, line);
    // walls
    for (let r = 0; r < 8; r++) for (let c = 0; c < 8; c++) {
      if (this.wallH[r * 8 + c]) this.drawWall(g, 0, r, c, false);
      if (this.wallV[r * 8 + c]) this.drawWall(g, 1, r, c, false);
    }
    // last move highlight
    if (this.lastMove) {
      if (this.lastMove.type === 'pawn') {
        const p = this.cellXY(this.lastMove.r, this.lastMove.c);
        g.fillStyle = this.css('--gold-dim');
        g.fillRect(p.x, p.y, C, C);
        g.strokeStyle = this.css('--gold'); g.lineWidth = 1.5;
        g.strokeRect(p.x + .75, p.y + .75, C - 1.5, C - 1.5);
      } else {
        const [o, r, c] = this.engWallToDisp(this.lastMove.o, this.lastMove.r, this.lastMove.c);
        g.save(); g.globalAlpha = .9;
        this.drawWall(g, o, r, c, false, this.css('--gold'));
        g.restore();
      }
    }
    // ghost wall
    if (this.ghost) this.drawGhost(g);
    // legal dots
    if (this.dots.length && !this.ghost) for (const d of this.dots) {
      const ctr = this.cellCenter(Math.floor(d / 9), d % 9);
      const jump = Math.abs(Math.floor(d / 9) - Math.floor(this.pawn[this.turn ?? 0] / 9)) +
                   Math.abs((d % 9) - (this.pawn[this.turn ?? 0] % 9)) > 1;
      g.beginPath();
      if (jump) { g.arc(ctr.x, ctr.y, .21 * C, 0, 7); g.lineWidth = Math.max(2, .07 * C); g.strokeStyle = this.dotColor(); g.stroke(); }
      else { g.arc(ctr.x, ctr.y, .16 * C, 0, 7); g.fillStyle = this.dotColor(); g.globalAlpha = .55; g.fill(); g.globalAlpha = 1; }
    }
    // pawns
    this.drawPawn(g, this.pawn[0], 0);
    this.drawPawn(g, this.pawn[1], 1);
    // selected ring
    if (this.selected >= 0) {
      const ctr = this.cellCenter(Math.floor(this.selected / 9), this.selected % 9);
      g.beginPath(); g.arc(ctr.x, ctr.y, .40 * C, 0, 7);
      g.strokeStyle = this.css('--gold'); g.lineWidth = 2;
      g.setLineDash([4, 3]); g.stroke(); g.setLineDash([]);
    }
  }

  dotColor() { return this.css('--gold2'); }

  setTurn(t) { this.turn = t; }

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
    g.save();
    // shadow along perpendicular axis
    g.fillStyle = 'rgba(0,0,0,.45)';
    if (o === 0) this.rr(g, rc.x + 1, rc.y + 2.5, rc.w, rc.h, 2), g.fill();
    else this.rr(g, rc.x + 2.5, rc.y + 1, rc.w, rc.h, 2), g.fill();
    // body gradient across thickness
    const grad = o === 0
      ? g.createLinearGradient(0, rc.y, 0, rc.y + rc.h)
      : g.createLinearGradient(rc.x, 0, rc.x + rc.w, 0);
    grad.addColorStop(0, edge); grad.addColorStop(.22, wall);
    grad.addColorStop(.78, wall); grad.addColorStop(1, edge);
    g.fillStyle = grad;
    this.rr(g, rc.x, rc.y, rc.w, rc.h, 2); g.fill();
    g.lineWidth = outlineOverride ? 1.8 : 1;
    g.strokeStyle = outlineOverride || edge;
    g.stroke();
    // seam at midpoint
    g.strokeStyle = edge; g.globalAlpha = .7; g.lineWidth = 1;
    g.beginPath();
    if (o === 0) { const mx = rc.x + rc.w / 2; g.moveTo(mx, rc.y + 1); g.lineTo(mx, rc.y + rc.h - 1); }
    else { const my = rc.y + rc.h / 2; g.moveTo(rc.x + 1, my); g.lineTo(rc.x + rc.w - 1, my); }
    g.stroke(); g.globalAlpha = 1;
    g.restore();
  }

  drawGhost(g) {
    const gh = this.ghost, [o, r, c] = [gh.o, gh.r, gh.c];
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
    if (st === 'bad') {  // cross glyph
      const cx = rc.x + rc.w / 2, cy = rc.y + rc.h / 2, k = this.G * .8;
      g.beginPath();
      g.moveTo(cx - k, cy - k); g.lineTo(cx + k, cy + k);
      g.moveTo(cx + k, cy - k); g.lineTo(cx - k, cy + k);
      g.stroke();
    }
    // assist trace: faint outline where the pointer actually was
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
    const px = this.css('--p' + pl), pl2 = this.css('--p' + pl + '-light'),
          pd = this.css('--p' + pl + '-deep');
    const R = .31 * C;
    // contact shadow
    g.save();
    const shadow = document.documentElement.dataset.pawnShadow || 'soft';
    if (shadow !== 'off') {
      g.fillStyle = 'rgba(0,0,0,' + (shadow === 'deep' ? '.5' : '.35') + ')';
      g.beginPath();
      g.ellipse(ctr.x, ctr.y + R * .55, R * .95, R * .38, 0, 0, 7); g.fill();
    }
    // body: radial gradient
    const grd = g.createRadialGradient(ctr.x - R * .35, ctr.y - R * .45, R * .15, ctr.x, ctr.y, R * 1.05);
    grd.addColorStop(0, pl2); grd.addColorStop(.55, px); grd.addColorStop(1, pd);
    const style = document.documentElement.dataset.pawn || 'disc';
    g.fillStyle = grd; g.strokeStyle = pd; g.lineWidth = 1;
    if (style === 'crown' || style === 'pillar') {
      // simple chess-pawn silhouette
      g.beginPath();
      g.arc(ctr.x, ctr.y - R * .25, R * .58, 0, 7);           // bulb
      g.moveTo(ctr.x - R * .34, ctr.y + R * .05);
      g.quadraticCurveTo(ctr.x, ctr.y + R * .18, ctr.x + R * .34, ctr.y + R * .05);
      g.lineTo(ctr.x + R * .52, ctr.y + R * .62);
      g.quadraticCurveTo(ctr.x, ctr.y + R * .40, ctr.x - R * .52, ctr.y + R * .62);
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
      // engraved glyph toward that player's goal
      g.strokeStyle = this.css('--bg'); g.lineWidth = Math.max(1.5, C * .05);
      const dir = pl === 0 ? -1 : 1;
      g.beginPath();
      g.moveTo(ctr.x - R * .3, ctr.y + dir * R * .22);
      g.lineTo(ctr.x, ctr.y - dir * R * .22);
      g.lineTo(ctr.x + R * .3, ctr.y + dir * R * .22);
      g.stroke();
    } else {  // disc (default)
      g.beginPath(); g.arc(ctr.x, ctr.y, R, 0, 7); g.fill(); g.stroke();
    }
    // side-to-move gold breathing ring
    if (this.turn === pl) {
      g.beginPath(); g.arc(ctr.x, ctr.y, .40 * C, 0, 7);
      g.strokeStyle = this.css('--gold'); g.lineWidth = 2;
      g.shadowColor = this.css('--gold-glow'); g.shadowBlur = 8;
      g.stroke();
    }
    g.restore();
  }

  // ---- hit testing -------------------------------------------------------
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
}

window.QBoard = QBoard;
