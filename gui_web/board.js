/* =====================================================================
 * board.js -- Zquoridor GUI v4, W3: canvas board renderer.
 *
 * A single canvas-based, theme-able, animated Quoridor board. Pure view
 * plus input: it never touches the engine and never creates DOM outside
 * its own canvas. Exposes exactly one global, `QBoard`.
 *
 * Plain ES2020, no modules, no imports, no external libraries, no image
 * assets -- everything is drawn procedurally so the file can be inlined
 * verbatim into a single-file build.
 *
 * Coordinates (see GUI_PLAN.md "Vocabulary"):
 *   cell = er*9 + ec, engine row 0 = TOP, player 0 starts at (0,4).
 *   wall slot = er*8 + ec.  Orientation 0 (H) sits in the corridor
 *   between engine rows er and er+1 and spans columns ec, ec+1.
 *   Orientation 1 (V) sits in the corridor right of column ec and spans
 *   engine rows er, er+1.
 *
 *   `flipped` mirrors ROWS ONLY. Columns are never mirrored, so a 180
 *   turn never swaps H for V. Display row dr = flipped ? 8-er : er,
 *   display wall row dr = flipped ? 7-er : er, display column = ec.
 * ===================================================================== */

(function (global) {
    'use strict';

    var N = 9;        // cells per side
    var WS = 8;       // wall slots per side, per orientation

    /* ---------------------------------------------------------------
     * Board themes. All six required palettes, fully filled in.
     * Required keys (the contract): cellLight, cellDark, boardBg,
     * gridLine, wall, wallEdge, wallShadow, coord, dot, lastMove,
     * pathP0, pathP1, p0, p0Edge, p1, p1Edge, goalGlow.
     * Extra (additive, purely cosmetic, safe to ignore): frame, grain,
     * arrow.
     * --------------------------------------------------------------- */
    var THEMES = {
        wood: {
            // Rich dark walnut: espresso frame, mid-brown cells with a
            // subtle two-tone alternation, near-black chunky slabs.
            cellLight: '#b8874f', cellDark: '#9d6f3c', boardBg: '#33200e',
            frame: '#1d1003', gridLine: 'rgba(22,12,3,0.45)', grain: 0.7,
            wall: '#33200e', wallEdge: '#140901', wallShadow: 'rgba(6,3,0,0.65)',
            coord: '#eedbb6', dot: 'rgba(255,236,190,0.5)', lastMove: 'rgba(244,210,96,0.32)',
            pathP0: '#fff4d4', pathP1: '#ff9d84',
            p0: '#fbf4e2', p0Edge: '#8a6f35', p1: '#c53a48', p1Edge: '#550e18',
            goalGlow: 'rgba(238,208,118,0.62)', arrow: 'rgba(246,196,76,0.82)'
        },
        classic: {
            cellLight: '#f0f0f4', cellDark: '#b9bcc8', boardBg: '#3a3d4a',
            frame: '#2a2c36', gridLine: 'rgba(50,52,64,0.26)', grain: 0,
            wall: '#6d7284', wallEdge: '#414554', wallShadow: 'rgba(18,19,26,0.5)',
            coord: '#e6e8f0', dot: 'rgba(40,42,54,0.38)', lastMove: 'rgba(120,150,230,0.38)',
            pathP0: '#ffffff', pathP1: '#e2564f',
            p0: '#fdfdff', p0Edge: '#8a8fa4', p1: '#c0394a', p1Edge: '#5c1620',
            goalGlow: 'rgba(150,180,255,0.5)', arrow: 'rgba(120,160,255,0.72)'
        },
        emerald: {
            cellLight: '#dcecd6', cellDark: '#9dc0a0', boardBg: '#1d3a2a',
            frame: '#132a1e', gridLine: 'rgba(19,42,30,0.28)', grain: 0.2,
            wall: '#3f7a56', wallEdge: '#204434', wallShadow: 'rgba(8,24,16,0.55)',
            coord: '#d8f0dc', dot: 'rgba(16,40,26,0.4)', lastMove: 'rgba(230,200,90,0.4)',
            pathP0: '#f4fff0', pathP1: '#f0a04c',
            p0: '#f4fbef', p0Edge: '#7e9a80', p1: '#c4562e', p1Edge: '#5c2110',
            goalGlow: 'rgba(120,230,160,0.5)', arrow: 'rgba(240,205,90,0.72)'
        },
        ocean: {
            cellLight: '#dbe9f4', cellDark: '#8fb2cc', boardBg: '#12293d',
            frame: '#0b1c2c', gridLine: 'rgba(11,28,44,0.28)', grain: 0.14,
            wall: '#356a92', wallEdge: '#183c56', wallShadow: 'rgba(4,16,28,0.55)',
            coord: '#d6ecff', dot: 'rgba(10,32,50,0.4)', lastMove: 'rgba(110,200,230,0.4)',
            pathP0: '#f2fbff', pathP1: '#ffb057',
            p0: '#f0f8ff', p0Edge: '#7295b0', p1: '#d4643a', p1Edge: '#63240f',
            goalGlow: 'rgba(90,200,240,0.5)', arrow: 'rgba(90,200,240,0.75)'
        },
        coral: {
            cellLight: '#fbe4d6', cellDark: '#e0a48c', boardBg: '#4a2320',
            frame: '#361714', gridLine: 'rgba(54,23,20,0.26)', grain: 0.3,
            wall: '#a4553e', wallEdge: '#6a3021', wallShadow: 'rgba(32,12,8,0.55)',
            coord: '#ffe4d4', dot: 'rgba(60,24,16,0.4)', lastMove: 'rgba(250,200,120,0.42)',
            pathP0: '#fff6ec', pathP1: '#5f9ad8',
            p0: '#fff3e6', p0Edge: '#b08063', p1: '#2f5f94', p1Edge: '#122c4a',
            goalGlow: 'rgba(255,180,120,0.5)', arrow: 'rgba(255,160,90,0.75)'
        },
        night: {
            cellLight: '#2a2b3a', cellDark: '#1b1c27', boardBg: '#0b0b12',
            frame: '#05050a', gridLine: 'rgba(120,125,170,0.14)', grain: 0,
            wall: '#4b4f6b', wallEdge: '#282b3d', wallShadow: 'rgba(0,0,0,0.65)',
            coord: '#8b90b4', dot: 'rgba(200,208,255,0.3)', lastMove: 'rgba(200,168,75,0.34)',
            pathP0: '#e6c96e', pathP1: '#7aa2e8',
            p0: '#e6c96e', p0Edge: '#7a6524', p1: '#4a7cc0', p1Edge: '#1b3457',
            goalGlow: 'rgba(200,168,75,0.45)', arrow: 'rgba(230,201,110,0.75)'
        }
    };

    var PAWN_STYLES = ['disc', 'pin', 'chess', 'glyph'];

    /* ---------------------------- utils ---------------------------- */

    function clamp(v, lo, hi) { return v < lo ? lo : (v > hi ? hi : v); }

    function num(v, dflt) { return (typeof v === 'number' && isFinite(v)) ? v : dflt; }

    // Parse '#rgb' / '#rrggbb' / 'rgb(...)' / 'rgba(...)' into [r,g,b,a].
    function parseColor(c) {
        if (typeof c !== 'string') return [128, 128, 128, 1];
        var s = c.trim();
        if (s.charAt(0) === '#') {
            var h = s.slice(1);
            if (h.length === 3) h = h[0] + h[0] + h[1] + h[1] + h[2] + h[2];
            if (h.length === 8) {
                return [parseInt(h.slice(0, 2), 16), parseInt(h.slice(2, 4), 16),
                        parseInt(h.slice(4, 6), 16), parseInt(h.slice(6, 8), 16) / 255];
            }
            if (h.length !== 6) return [128, 128, 128, 1];
            return [parseInt(h.slice(0, 2), 16), parseInt(h.slice(2, 4), 16), parseInt(h.slice(4, 6), 16), 1];
        }
        var m = s.match(/rgba?\(([^)]+)\)/i);
        if (m) {
            var p = m[1].split(',').map(function (x) { return parseFloat(x); });
            return [p[0] | 0, p[1] | 0, p[2] | 0, p.length > 3 && isFinite(p[3]) ? p[3] : 1];
        }
        return [128, 128, 128, 1];
    }

    function rgba(c, a) {
        var p = parseColor(c);
        var alpha = (typeof a === 'number') ? a * p[3] : p[3];
        return 'rgba(' + p[0] + ',' + p[1] + ',' + p[2] + ',' + (Math.round(alpha * 1000) / 1000) + ')';
    }

    // amt > 0 lightens toward white, amt < 0 darkens toward black.
    function shade(c, amt) {
        var p = parseColor(c);
        var f = amt < 0 ? 0 : 255, t = amt < 0 ? -amt : amt;
        return 'rgba(' + Math.round((f - p[0]) * t + p[0]) + ',' +
            Math.round((f - p[1]) * t + p[1]) + ',' +
            Math.round((f - p[2]) * t + p[2]) + ',' + p[3] + ')';
    }

    function roundRectPath(ctx, x, y, w, h, r) {
        var rr = Math.min(r, Math.abs(w) / 2, Math.abs(h) / 2);
        if (!(rr > 0)) { ctx.beginPath(); ctx.rect(x, y, w, h); return; }
        ctx.beginPath();
        ctx.moveTo(x + rr, y);
        ctx.lineTo(x + w - rr, y);
        ctx.quadraticCurveTo(x + w, y, x + w, y + rr);
        ctx.lineTo(x + w, y + h - rr);
        ctx.quadraticCurveTo(x + w, y + h, x + w - rr, y + h);
        ctx.lineTo(x + rr, y + h);
        ctx.quadraticCurveTo(x, y + h, x, y + h - rr);
        ctx.lineTo(x, y + rr);
        ctx.quadraticCurveTo(x, y, x + rr, y);
        ctx.closePath();
    }

    // Deterministic PRNG so the wood grain never shimmers between frames.
    function mulberry32(seed) {
        var a = seed >>> 0;
        return function () {
            a |= 0; a = (a + 0x6D2B79F5) | 0;
            var t = Math.imul(a ^ (a >>> 15), 1 | a);
            t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
            return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
        };
    }

    function easeOutCubic(t) { var u = 1 - t; return 1 - u * u * u; }
    function easeInOutCubic(t) { return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2; }

    function prefersReducedMotion() {
        try {
            return !!(global.matchMedia && global.matchMedia('(prefers-reduced-motion: reduce)').matches);
        } catch (e) { return false; }
    }

    function bitAt(arr, i) {
        if (!arr) return 0;
        if (i < 0 || i >= 64) return 0;
        var v = arr[i];
        return v ? 1 : 0;
    }

    function ownerAt(arr, i) {
        if (!arr) return -1;
        if (i < 0 || i >= 64) return -1;
        var v = arr[i];
        return (v === 0 || v === 1) ? v : -1;
    }

    function toSet(list) {
        var s = new Set();
        if (!list) return s;
        for (var i = 0; i < list.length; i++) {
            var v = list[i];
            if (typeof v === 'number' && isFinite(v)) s.add(v | 0);
        }
        return s;
    }

    /* ============================== QBoard ============================== */

    function QBoard(canvas, opts) {
        opts = opts || {};
        if (!canvas || typeof canvas.getContext !== 'function') {
            throw new Error('QBoard: a canvas element is required');
        }
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');

        this.mode = (opts.mode === 'analysis' || opts.mode === 'edit') ? opts.mode : 'play';
        this.themeName = THEMES[opts.theme] ? opts.theme : 'wood';
        this.theme = THEMES[this.themeName];
        this.pawnStyle = PAWN_STYLES.indexOf(opts.pawnStyle) >= 0 ? opts.pawnStyle : 'disc';
        this.flipped = !!opts.flipped;
        this.interactive = opts.interactive !== false;
        this.editTool = 'wallh';
        this.wallOrientation = 0;

        this.options = {
            showCoords: true,
            showPaths: false,
            showDots: true,
            highlightLast: true,
            animate: true
        };
        if (opts.options) this.setOptions(opts.options);

        // --- data ---
        this.position = {
            pawns: [4, 76],                 // e9 / e1
            wallsH: new Uint8Array(64),
            wallsV: new Uint8Array(64),
            wallOwner: null,
            turn: 0,
            wallsLeft: [10, 10],
            winner: -1
        };
        this.legal = { pawn: new Set(), wallH: new Set(), wallV: new Set() };
        this.overlays = { lastMove: null, hint: null, arrows: [], ghostWall: null, paths: null };

        // --- callbacks (assigned directly by the integrator) ---
        this.onPawnMove = null;
        this.onWallPlace = null;
        this.onWallRemove = null;
        this.onPawnPlace = null;
        this.onHoverWall = null;
        this.onWallOrientationChange = null;

        // --- internals ---
        this._g = null;               // geometry
        this._dpr = 1;
        this._side = 0;
        this._base = null;            // cached static layer (board + cells + coords)
        this._baseKey = '';
        this._raf = 0;
        this._anim = null;            // { kind, t0, dur, done, ... }
        this._pulse = null;           // { t0, dur, o, r, c }
        this._lastPulseKey = '';
        this._drag = null;            // active pointer gesture
        this._ghost = null;           // { o, r, c, legal } -- internal ghost
        this._hoverKey = '';
        this._cursor = null;          // keyboard selection cursor { r, c }
        this._destroyed = false;

        // canvas setup
        try {
            canvas.style.touchAction = 'none';
            canvas.style.display = 'block';
            canvas.style.userSelect = 'none';
            canvas.style.webkitUserSelect = 'none';
            canvas.style.outline = 'none';
            if (!canvas.hasAttribute('tabindex')) canvas.setAttribute('tabindex', '0');
        } catch (e) { /* non-DOM environment: ignore */ }

        var self = this;
        this._h = {
            down: function (e) { self._onPointerDown(e); },
            move: function (e) { self._onPointerMove(e); },
            up: function (e) { self._onPointerUp(e); },
            cancel: function (e) { self._onPointerCancel(e); },
            leave: function (e) { self._onPointerLeave(e); },
            ctx: function (e) { if (self.mode === 'edit') e.preventDefault(); },
            key: function (e) { self._onKeyDown(e); },
            win: function () { self.resize(); }
        };
        canvas.addEventListener('pointerdown', this._h.down);
        canvas.addEventListener('pointermove', this._h.move);
        canvas.addEventListener('pointerup', this._h.up);
        canvas.addEventListener('pointercancel', this._h.cancel);
        canvas.addEventListener('pointerleave', this._h.leave);
        canvas.addEventListener('contextmenu', this._h.ctx);
        canvas.addEventListener('keydown', this._h.key);
        if (global.addEventListener) global.addEventListener('resize', this._h.win);

        this._ro = null;
        if (global.ResizeObserver && canvas.parentElement) {
            try {
                this._ro = new global.ResizeObserver(function () { self.resize(); });
                this._ro.observe(canvas.parentElement);
            } catch (e) { this._ro = null; }
        }

        this.resize();
    }

    QBoard.THEMES = THEMES;
    QBoard.PAWN_STYLES = PAWN_STYLES;
    QBoard.N = N;
    QBoard.WS = WS;

    /* --------------------------- configuration --------------------------- */

    QBoard.prototype.setMode = function (mode) {
        if (mode !== 'play' && mode !== 'analysis' && mode !== 'edit') return;
        if (this.mode === mode) return;
        this.mode = mode;
        this._cancelDrag();
        this.render();
    };

    QBoard.prototype.setTheme = function (name) {
        if (!THEMES[name] || this.themeName === name) return;
        this.themeName = name;
        this.theme = THEMES[name];
        this._base = null;
        this.render();
    };

    QBoard.prototype.setPawnStyle = function (name) {
        if (PAWN_STYLES.indexOf(name) < 0 || this.pawnStyle === name) return;
        this.pawnStyle = name;
        this.render();
    };

    QBoard.prototype.setFlipped = function (v) {
        v = !!v;
        if (this.flipped === v) return;
        this.flipped = v;
        this._base = null;
        this._cancelDrag();
        this.render();
    };

    QBoard.prototype.setOptions = function (o) {
        if (!o) return;
        var keys = ['showCoords', 'showPaths', 'showDots', 'highlightLast', 'animate'];
        var baseDirty = false;
        for (var i = 0; i < keys.length; i++) {
            var k = keys[i];
            if (Object.prototype.hasOwnProperty.call(o, k)) {
                var nv = !!o[k];
                if (k === 'showCoords' && nv !== this.options.showCoords) baseDirty = true;
                this.options[k] = nv;
            }
        }
        if (baseDirty) { this._base = null; this.resize(); }
        else if (this._g) this.render();
    };

    QBoard.prototype.setInteractive = function (v) {
        this.interactive = !!v;
        if (!this.interactive) this._cancelDrag();
        this.render();
    };

    QBoard.prototype.setEditTool = function (tool) {
        var ok = ['p0', 'p1', 'wallh', 'wallv', 'erase'];
        if (ok.indexOf(tool) < 0) return;
        this.editTool = tool;
        if (tool === 'wallh') this.wallOrientation = 0;
        else if (tool === 'wallv') this.wallOrientation = 1;
        this._ghost = null;
        this.render();
    };

    QBoard.prototype.setWallOrientation = function (o) {
        o = (o === 1 || o === '1' || o === 'v') ? 1 : 0;
        if (this.wallOrientation === o) return;
        this.wallOrientation = o;
        if (this._ghost) this._ghost = null;
        this.render();
    };

    /* ------------------------------ data in ------------------------------ */

    QBoard.prototype.setPosition = function (p) {
        if (!p) return;
        var pos = this.position;
        if (p.pawns && p.pawns.length >= 2) {
            pos.pawns = [clamp(p.pawns[0] | 0, 0, 80), clamp(p.pawns[1] | 0, 0, 80)];
        }
        if (p.wallsH) pos.wallsH = p.wallsH;
        if (p.wallsV) pos.wallsV = p.wallsV;
        if (Object.prototype.hasOwnProperty.call(p, 'wallOwner')) pos.wallOwner = p.wallOwner || null;
        if (typeof p.turn === 'number') pos.turn = p.turn ? 1 : 0;
        if (p.wallsLeft && p.wallsLeft.length >= 2) {
            pos.wallsLeft = [clamp(p.wallsLeft[0] | 0, 0, 10), clamp(p.wallsLeft[1] | 0, 0, 10)];
        }
        if (typeof p.winner === 'number') pos.winner = p.winner;
        // A new position invalidates any keyboard selection outline.
        this._cursor = null;
        this.render();
    };

    QBoard.prototype.setLegal = function (l) {
        l = l || {};
        this.legal.pawn = toSet(l.pawnCells);
        this.legal.wallH = toSet(l.wallH);
        this.legal.wallV = toSet(l.wallV);
        if (this._ghost) this._ghost.legal = this._wallLegal(this._ghost.o, this._ghost.r, this._ghost.c);
        this.render();
    };

    QBoard.prototype.setOverlays = function (o) {
        o = o || {};
        var ov = this.overlays;
        if (Object.prototype.hasOwnProperty.call(o, 'lastMove')) ov.lastMove = o.lastMove || null;
        if (Object.prototype.hasOwnProperty.call(o, 'hint')) ov.hint = o.hint || null;
        if (Object.prototype.hasOwnProperty.call(o, 'arrows')) ov.arrows = Array.isArray(o.arrows) ? o.arrows : [];
        if (Object.prototype.hasOwnProperty.call(o, 'ghostWall')) ov.ghostWall = o.ghostWall || null;
        if (Object.prototype.hasOwnProperty.call(o, 'paths')) ov.paths = Array.isArray(o.paths) ? o.paths : null;

        // A wall arriving as the last move pulses once.
        var lm = ov.lastMove;
        if (lm && lm.isWall) {
            var key = 'w' + lm.a + ',' + lm.b + ',' + lm.c;
            if (key !== this._lastPulseKey) {
                this._lastPulseKey = key;
                if (this._animationsEnabled()) {
                    this._pulse = { t0: 0, dur: 620, o: lm.a | 0, r: lm.b | 0, c: lm.c | 0 };
                    this._startLoop();
                }
            }
        } else {
            this._lastPulseKey = '';
            this._pulse = null;
        }
        this.render();
    };

    /* ------------------------------ geometry ----------------------------- */

    QBoard.prototype._measureSide = function () {
        var cv = this.canvas;
        var w = 0, h = 0;
        var p = cv.parentElement;
        if (p) { w = p.clientWidth || 0; h = p.clientHeight || 0; }
        if (!w) { var r = cv.getBoundingClientRect ? cv.getBoundingClientRect() : null; if (r) { w = r.width; h = r.height; } }
        if (!w) w = cv.width || 0;
        var side = w;
        if (h > 24 && h < side) side = h;
        return Math.max(0, Math.floor(side));
    };

    QBoard.prototype.resize = function () {
        if (this._destroyed) return;
        var side = this._measureSide();
        var dpr = Math.max(1, global.devicePixelRatio || 1);
        if (side <= 0) {
            this._side = 0; this._g = null;
            try {
                if (this.canvas.width !== 0) { this.canvas.width = 0; this.canvas.height = 0; }
            } catch (e) { /* ignore */ }
            return;
        }
        var changed = Math.abs(side - this._side) >= 1 || dpr !== this._dpr;
        this._side = side;
        this._dpr = dpr;

        if (changed) {
            this.canvas.width = Math.max(1, Math.round(side * dpr));
            this.canvas.height = Math.max(1, Math.round(side * dpr));
            this.canvas.style.width = side + 'px';
            this.canvas.style.height = side + 'px';
            this._base = null;
        }
        this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

        var m = this.options.showCoords ? Math.max(10, Math.round(side * 0.045)) : Math.max(4, Math.round(side * 0.014));
        var inner = side - 2 * m;
        // Thin corridors, big cells (claustrophobia.dev proportions):
        // gap ≈ 10% of cell  =>  gap = inner * r / (N + (N-1)*r), r = 0.10
        var rr = 0.10;
        var gap = Math.max(2, Math.round(inner * rr / (N + (N - 1) * rr)));
        var cell = (inner - gap * (N - 1)) / N;
        if (cell < 6) { cell = Math.max(2, inner / N); gap = Math.max(1, inner * 0.01); }
        this._g = {
            side: side, m: m, inner: inner, gap: gap, cell: cell, unit: cell + gap,
            r: Math.max(2, cell * 0.10)
        };
        this.render();
    };

    // top-left pixel of a display grid position
    QBoard.prototype._cellRect = function (er, ec) {
        var g = this._g; if (!g) return null;
        var dr = this.flipped ? (N - 1 - er) : er;
        return { x: g.m + ec * g.unit, y: g.m + dr * g.unit, w: g.cell, h: g.cell };
    };

    QBoard.prototype._cellCenter = function (cellIdx) {
        var g = this._g; if (!g) return null;
        var er = (cellIdx / N) | 0, ec = cellIdx % N;
        var r = this._cellRect(er, ec);
        return { x: r.x + g.cell / 2, y: r.y + g.cell / 2 };
    };

    // The pixel rectangle occupied by a wall slab.
    QBoard.prototype._slotRect = function (o, er, ec) {
        var g = this._g; if (!g) return null;
        var dr = this.flipped ? (WS - 1 - er) : er;
        if (o === 0) {
            return { x: g.m + ec * g.unit, y: g.m + dr * g.unit + g.cell, w: 2 * g.cell + g.gap, h: g.gap };
        }
        return { x: g.m + ec * g.unit + g.cell, y: g.m + dr * g.unit, w: g.gap, h: 2 * g.cell + g.gap };
    };

    // Both orientations of a slot are centred on the same corridor
    // intersection, which is what makes "snap to nearest corridor" a
    // single rounding for H and V alike.
    QBoard.prototype._slotToPoint = function (er, ec) {
        var g = this._g; if (!g) return null;
        var dr = this.flipped ? (WS - 1 - er) : er;
        return { x: g.m + ec * g.unit + g.cell + g.gap / 2, y: g.m + dr * g.unit + g.cell + g.gap / 2 };
    };

    QBoard.prototype._pointToSlot = function (x, y) {
        var g = this._g; if (!g) return null;
        var dc = clamp(Math.round((x - g.m - g.cell - g.gap / 2) / g.unit), 0, WS - 1);
        var dr = clamp(Math.round((y - g.m - g.cell - g.gap / 2) / g.unit), 0, WS - 1);
        var er = this.flipped ? (WS - 1 - dr) : dr;
        return { r: er, c: dc };
    };

    QBoard.prototype._pointToCell = function (x, y) {
        var g = this._g; if (!g) return -1;
        var dc = Math.floor((x - g.m) / g.unit);
        var dr = Math.floor((y - g.m) / g.unit);
        if (dc < 0 || dc > N - 1 || dr < 0 || dr > N - 1) return -1;
        var er = this.flipped ? (N - 1 - dr) : dr;
        return er * N + dc;
    };

    QBoard.prototype._eventPos = function (e) {
        var rect = this.canvas.getBoundingClientRect();
        return { x: e.clientX - rect.left, y: e.clientY - rect.top };
    };

    QBoard.prototype._inBoard = function (x, y, slack) {
        var g = this._g; if (!g) return false;
        var s = slack || 0;
        return x >= g.m - s && x <= g.m + g.inner + s && y >= g.m - s && y <= g.m + g.inner + s;
    };

    /* ------------------------------ legality ----------------------------- */

    QBoard.prototype._wallBit = function (o, er, ec) {
        var slot = er * WS + ec;
        return o === 0 ? bitAt(this.position.wallsH, slot) : bitAt(this.position.wallsV, slot);
    };

    QBoard.prototype._wallLegal = function (o, er, ec) {
        if (er < 0 || er >= WS || ec < 0 || ec >= WS) return false;
        var slot = er * WS + ec;
        if (this.mode === 'edit') {
            // The editor validates for itself; the board only refuses an
            // exact duplicate so the ghost does not lie about a no-op.
            return !this._wallBit(o, er, ec);
        }
        var set = o === 0 ? this.legal.wallH : this.legal.wallV;
        return set.has(slot);
    };

    // Existing wall whose slab contains (x,y) -- used by the eraser.
    QBoard.prototype._wallHitAt = function (x, y) {
        var g = this._g; if (!g) return null;
        var pad = Math.max(2, g.gap * 0.6);
        for (var o = 0; o < 2; o++) {
            for (var r = 0; r < WS; r++) {
                for (var c = 0; c < WS; c++) {
                    if (!this._wallBit(o, r, c)) continue;
                    var q = this._slotRect(o, r, c);
                    if (x >= q.x - pad && x <= q.x + q.w + pad && y >= q.y - pad && y <= q.y + q.h + pad) {
                        return { o: o, r: r, c: c };
                    }
                }
            }
        }
        return null;
    };

    /* -------------------------------- input ------------------------------ */

    QBoard.prototype._emit = function (name) {
        var fn = this[name];
        if (typeof fn !== 'function') return;
        var args = Array.prototype.slice.call(arguments, 1);
        try { fn.apply(this, args); } catch (e) {
            if (global.console && console.error) console.error('QBoard: ' + name + ' handler threw', e);
        }
    };

    QBoard.prototype._onPointerDown = function (e) {
        if (!this.interactive || !this._g || this._destroyed) return;
        var p = this._eventPos(e);
        try { this.canvas.focus({ preventScroll: true }); } catch (err) { try { this.canvas.focus(); } catch (e2) { } }

        if (this.mode === 'edit') {
            e.preventDefault();
            this._editApply(p, e.button === 2 || (e.pointerType === 'mouse' && e.buttons === 2));
            return;
        }

        if (!this._inBoard(p.x, p.y, this._g.cell * 0.6)) return;
        e.preventDefault();
        try { this.canvas.setPointerCapture(e.pointerId); } catch (err) { /* ignore */ }

        var cell = this._pointToCell(p.x, p.y);
        this._drag = {
            id: e.pointerId,
            x0: p.x, y0: p.y,
            cell: cell,
            pawnCandidate: cell >= 0 && this.legal.pawn.has(cell),
            moved: false,
            wallMode: false,
            baseO: this.wallOrientation
        };
        this._cursor = null;
        this.render();
    };

    QBoard.prototype._onPointerMove = function (e) {
        if (!this._g || this._destroyed) return;
        var p = this._eventPos(e);

        if (this.mode === 'edit') {
            if (this.interactive && (this.editTool === 'wallh' || this.editTool === 'wallv')) {
                var so = this.editTool === 'wallv' ? 1 : 0;
                var s = this._pointToSlot(p.x, p.y);
                if (s && this._inBoard(p.x, p.y, this._g.cell * 0.4)) {
                    this._setGhost(so, s.r, s.c);
                } else { this._clearGhost(); }
            }
            return;
        }

        var d = this._drag;
        // While a drag is active, other pointers (second finger) must not
        // fight over the hover state.
        if (d && d.id !== e.pointerId) return;
        if (!d || d.id !== e.pointerId) {
            // plain hover: report the corridor under the pointer
            if (this.interactive && this._inBoard(p.x, p.y, 0)) {
                var hs = this._pointToSlot(p.x, p.y);
                if (hs) {
                    var lg = this._wallLegal(this.wallOrientation, hs.r, hs.c);
                    var k = this.wallOrientation + ':' + hs.r + ':' + hs.c + ':' + lg;
                    if (k !== this._hoverKey) {
                        this._hoverKey = k;
                        this._emit('onHoverWall', this.wallOrientation, hs.r, hs.c, lg);
                    }
                }
            } else if (this._hoverKey) {
                this._hoverKey = '';
                this._emit('onHoverWall', null, null, null, null);
            }
            return;
        }

        e.preventDefault();
        var dx = p.x - d.x0, dy = p.y - d.y0;
        var dist = Math.sqrt(dx * dx + dy * dy);
        var thresh = this._g.cell * 0.30;
        if (!d.moved && dist > thresh) { d.moved = true; d.wallMode = true; }
        if (!d.wallMode) return;

        // A drag that runs past the halfway point of a cell in the
        // direction perpendicular to the armed wall flips the orientation.
        var half = this._g.cell * 0.5;
        var want = this.wallOrientation;
        if (Math.abs(dx) > half && Math.abs(dx) > Math.abs(dy) * 1.35) want = 0;       // sweeping sideways -> H slab
        else if (Math.abs(dy) > half && Math.abs(dy) > Math.abs(dx) * 1.35) want = 1;  // sweeping up/down  -> V slab
        if (want !== this.wallOrientation) {
            this.wallOrientation = want;
            this._emit('onWallOrientationChange', want);
        }

        var slot = this._pointToSlot(p.x, p.y);
        if (slot && this._inBoard(p.x, p.y, this._g.cell * 0.7)) this._setGhost(this.wallOrientation, slot.r, slot.c);
        else this._clearGhost();
    };

    QBoard.prototype._onPointerUp = function (e) {
        if (this.mode === 'edit') { this._clearGhostSoft(); return; }
        var d = this._drag;
        if (!d || d.id !== e.pointerId) return;
        this._drag = null;
        try { this.canvas.releasePointerCapture(e.pointerId); } catch (err) { /* ignore */ }
        var g = this._ghost;
        this._ghost = null;
        // The drag is over: stop reporting the ghost slot as "hovered" or a
        // later identical hover will be suppressed by the key comparison.
        this._hoverKey = '';
        this._emit('onHoverWall', null, null, null, null);

        if (!this.interactive) { this.render(); return; }

        if (d.wallMode) {
            if (g && g.legal) this._emit('onWallPlace', g.o, g.r, g.c);
            this.render();
            return;
        }

        // A tap. A legal pawn destination wins; otherwise the tap drops a
        // wall of the armed orientation on the nearest corridor.
        if (d.pawnCandidate) { this._emit('onPawnMove', d.cell); this.render(); return; }
        var p = this._eventPos(e);
        if (this._inBoard(p.x, p.y, 0)) {
            var s = this._pointToSlot(p.x, p.y);
            if (s && this._wallLegal(this.wallOrientation, s.r, s.c)) {
                this._emit('onWallPlace', this.wallOrientation, s.r, s.c);
            }
        }
        this.render();
    };

    QBoard.prototype._onPointerCancel = function () { this._cancelDrag(); };

    QBoard.prototype._onPointerLeave = function () {
        if (this._hoverKey) { this._hoverKey = ''; this._emit('onHoverWall', null, null, null, null); }
        if (!this._drag) this._clearGhostSoft();
    };

    QBoard.prototype._cancelDrag = function () {
        this._drag = null;
        this._ghost = null;
        if (this._hoverKey) { this._hoverKey = ''; this._emit('onHoverWall', null, null, null, null); }
        if (this._g) this.render();
    };

    QBoard.prototype._setGhost = function (o, r, c) {
        var legal = this._wallLegal(o, r, c);
        var cur = this._ghost;
        if (cur && cur.o === o && cur.r === r && cur.c === c && cur.legal === legal) return;
        this._ghost = { o: o, r: r, c: c, legal: legal };
        var k = o + ':' + r + ':' + c + ':' + legal;
        if (k !== this._hoverKey) { this._hoverKey = k; this._emit('onHoverWall', o, r, c, legal); }
        this.render();
    };

    QBoard.prototype._clearGhost = function () {
        if (!this._ghost) return;
        this._ghost = null;
        this._hoverKey = '';
        this._emit('onHoverWall', null, null, null, null);
        this.render();
    };

    QBoard.prototype._clearGhostSoft = function () {
        if (this._ghost) { this._ghost = null; this.render(); }
    };

    QBoard.prototype._editApply = function (p, erase) {
        if (!this._inBoard(p.x, p.y, this._g.cell * 0.5)) return;
        var tool = erase ? 'erase' : this.editTool;
        if (tool === 'erase') {
            var hit = this._wallHitAt(p.x, p.y);
            if (hit) this._emit('onWallRemove', hit.o, hit.r, hit.c);
            return;
        }
        if (tool === 'p0' || tool === 'p1') {
            var cell = this._pointToCell(p.x, p.y);
            if (cell >= 0) this._emit('onPawnPlace', tool === 'p1' ? 1 : 0, cell);
            return;
        }
        var o = tool === 'wallv' ? 1 : 0;
        var s = this._pointToSlot(p.x, p.y);
        if (!s) return;
        if (this._wallBit(o, s.r, s.c)) this._emit('onWallRemove', o, s.r, s.c);
        else this._emit('onWallPlace', o, s.r, s.c);
    };

    QBoard.prototype._onKeyDown = function (e) {
        if (!this.interactive || !this._g) return;
        var k = e.key;
        var moved = false;
        // Only arrow keys materialize the selection cursor -- H/V/Enter on
        // a fresh focus should not leave an outline sitting on the board.
        var isArrow = k === 'ArrowUp' || k === 'ArrowDown' || k === 'ArrowLeft' || k === 'ArrowRight';
        if (!this._cursor && (isArrow || k === 'Enter' || k === ' ')) {
            var own = this.position.pawns[this.position.turn ? 1 : 0] | 0;
            this._cursor = { r: (own / N) | 0, c: own % N };
            if (isArrow) { e.preventDefault(); this.render(); return; }
        }
        var cur = this._cursor;
        var dr = this.flipped ? (N - 1 - cur.r) : cur.r, dc = cur.c;
        if (k === 'ArrowUp') { dr = clamp(dr - 1, 0, N - 1); moved = true; }
        else if (k === 'ArrowDown') { dr = clamp(dr + 1, 0, N - 1); moved = true; }
        else if (k === 'ArrowLeft') { dc = clamp(dc - 1, 0, N - 1); moved = true; }
        else if (k === 'ArrowRight') { dc = clamp(dc + 1, 0, N - 1); moved = true; }
        if (moved) {
            e.preventDefault();
            cur.r = this.flipped ? (N - 1 - dr) : dr;
            cur.c = dc;
            this.render();
            return;
        }
        if (k === 'Enter' || k === ' ') {
            e.preventDefault();
            var cell = cur.r * N + cur.c;
            if (this.legal.pawn.has(cell)) this._emit('onPawnMove', cell);
            return;
        }
        if (k === 'h' || k === 'H') {
            e.preventDefault();
            if (this.wallOrientation !== 0) { this.wallOrientation = 0; this._emit('onWallOrientationChange', 0); this.render(); }
            return;
        }
        if (k === 'v' || k === 'V') {
            e.preventDefault();
            if (this.wallOrientation !== 1) { this.wallOrientation = 1; this._emit('onWallOrientationChange', 1); this.render(); }
            return;
        }
        if (k === 'Escape') { this._cursor = null; this._cancelDrag(); }
    };

    /* ------------------------------ animation ---------------------------- */

    QBoard.prototype._animationsEnabled = function () {
        return this.options.animate && !prefersReducedMotion() && !!this._g;
    };

    QBoard.prototype.animateMove = function (move, mover, done) {
        var cb = typeof done === 'function' ? done : null;
        var finish = function () { if (cb) { try { cb(); } catch (e) { if (global.console) console.error(e); } } };
        if (!move || !this._animationsEnabled()) {
            if (cb) global.setTimeout(finish, 0);
            return;
        }
        this._finishAnim();          // flush any in-flight animation first
        mover = mover ? 1 : 0;
        if (move.isWall) {
            this._anim = { kind: 'wall', o: move.a | 0, r: move.b | 0, c: move.c | 0,
                           mover: mover, t0: 0, dur: 260, done: finish };
        } else {
            var from = this.position.pawns[mover] | 0;
            var to = move.a | 0;
            if (from === to) { global.setTimeout(finish, 0); return; }
            this._anim = { kind: 'pawn', player: mover, from: from, to: to, t0: 0, dur: 220, done: finish };
        }
        this._startLoop();
    };

    QBoard.prototype._finishAnim = function () {
        var a = this._anim;
        this._anim = null;
        if (a && a.done) a.done();
    };

    QBoard.prototype._startLoop = function () {
        if (this._raf || this._destroyed) return;
        var self = this;
        var step = function (ts) {
            if (self._destroyed) { self._raf = 0; return; }
            var busy = false;
            var a = self._anim;
            if (a) {
                if (!a.t0) a.t0 = ts;
                if (ts - a.t0 >= a.dur) { self._anim = null; if (a.done) a.done(); }
                else busy = true;
            }
            var p = self._pulse;
            if (p) {
                if (!p.t0) p.t0 = ts;
                if (ts - p.t0 >= p.dur) self._pulse = null; else busy = true;
            }
            self._nowTs = ts;
            self._draw();
            // NOTE: _raf keeps its handle until the loop actually ends. A
            // done() callback that chains animateMove() must NOT spawn a
            // second loop here -- the running one picks up the new
            // animation through the busy re-check below.
            if (busy) { self._raf = global.requestAnimationFrame(step); }
            else { self._raf = 0; }
        };
        this._raf = global.requestAnimationFrame(step);
    };

    /* ------------------------------ rendering ---------------------------- */

    QBoard.prototype.render = function () {
        if (this._destroyed) return;
        this._draw();
    };

    QBoard.prototype._draw = function () {
        var g = this._g, ctx = this.ctx;
        if (!g || !ctx || g.side <= 0) return;
        var t = this.theme;

        ctx.setTransform(this._dpr, 0, 0, this._dpr, 0, 0);
        ctx.clearRect(0, 0, g.side, g.side);

        this._drawBase(ctx);
        this._drawGoalGlow(ctx);
        if (this.options.highlightLast) this._drawLastMoveCells(ctx);
        if (this.mode === 'edit') this._drawCorridorPips(ctx);
        if (this.options.showPaths) this._drawPaths(ctx);
        if (this.options.showDots) this._drawLegalDots(ctx);
        this._drawWalls(ctx);
        if (this.options.highlightLast) this._drawLastWallTint(ctx);
        this._drawPawns(ctx);
        this._drawGhost(ctx);
        this._drawCursor(ctx);
        this._drawArrows(ctx);
    };

    /* -- static layer: frame, cells, bevel, grain, grid lines, coords -- */

    QBoard.prototype._drawBase = function (ctx) {
        var g = this._g;
        var key = this.themeName + '|' + g.side + '|' + this._dpr + '|' + this.flipped + '|' + this.options.showCoords;
        if (!this._base || this._baseKey !== key) {
            this._base = this._buildBase();
            this._baseKey = key;
        }
        if (this._base) ctx.drawImage(this._base, 0, 0, g.side, g.side);
    };

    QBoard.prototype._buildBase = function () {
        var g = this._g, t = this.theme, dpr = this._dpr;
        var cv;
        try {
            cv = (global.document && global.document.createElement) ? global.document.createElement('canvas') : null;
        } catch (e) { cv = null; }
        if (!cv) return null;
        cv.width = Math.max(1, Math.round(g.side * dpr));
        cv.height = Math.max(1, Math.round(g.side * dpr));
        var c = cv.getContext('2d');
        if (!c) return null;
        c.setTransform(dpr, 0, 0, dpr, 0, 0);

        // outer frame
        var fr = Math.max(3, g.side * 0.016);
        var grd = c.createLinearGradient(0, 0, g.side, g.side);
        grd.addColorStop(0, shade(t.boardBg, 0.12));
        grd.addColorStop(0.5, t.boardBg);
        grd.addColorStop(1, shade(t.frame || t.boardBg, -0.12));
        roundRectPath(c, 0.5, 0.5, g.side - 1, g.side - 1, fr);
        c.fillStyle = grd; c.fill();
        c.strokeStyle = rgba(t.frame || t.boardBg, 0.9); c.lineWidth = Math.max(1, g.side * 0.004); c.stroke();

        // inset well behind the playing field
        var pad = Math.max(2, g.gap * 0.55);
        roundRectPath(c, g.m - pad, g.m - pad, g.inner + pad * 2, g.inner + pad * 2, Math.max(2, g.r * 1.2));
        c.fillStyle = shade(t.boardBg, -0.22); c.fill();
        c.save();
        c.clip();
        c.strokeStyle = 'rgba(0,0,0,0.35)';
        c.lineWidth = Math.max(1.5, g.gap * 0.5);
        roundRectPath(c, g.m - pad, g.m - pad, g.inner + pad * 2, g.inner + pad * 2, Math.max(2, g.r * 1.2));
        c.stroke();
        c.restore();

        // cells
        for (var er = 0; er < N; er++) {
            for (var ec = 0; ec < N; ec++) {
                this._paintCell(c, er, ec);
            }
        }
        if (this.options.showCoords) this._paintCoords(c);
        return cv;
    };

    QBoard.prototype._paintCell = function (c, er, ec) {
        var g = this._g, t = this.theme;
        var q = this._cellRect(er, ec);
        var light = ((er + ec) % 2) === 0;
        var base = light ? t.cellLight : t.cellDark;

        var lg = c.createLinearGradient(q.x, q.y, q.x, q.y + q.h);
        lg.addColorStop(0, shade(base, 0.07));
        lg.addColorStop(1, shade(base, -0.07));
        roundRectPath(c, q.x, q.y, q.w, q.h, g.r);
        c.fillStyle = lg;
        c.fill();

        // wood grain: a few deterministic streaks per cell
        var grain = num(t.grain, 0);
        if (grain > 0) {
            c.save();
            roundRectPath(c, q.x, q.y, q.w, q.h, g.r);
            c.clip();
            var rnd = mulberry32((er * 31 + ec * 7 + 1) * 2654435761);
            var lines = 5;
            c.lineWidth = Math.max(0.6, q.h * 0.035);
            for (var i = 0; i < lines; i++) {
                var yy = q.y + q.h * ((i + rnd() * 0.8) / lines);
                var a = 0.05 + rnd() * 0.09;
                c.strokeStyle = (rnd() > 0.45 ? 'rgba(90,55,20,' : 'rgba(255,235,200,') + (a * grain).toFixed(3) + ')';
                c.beginPath();
                c.moveTo(q.x - 2, yy);
                c.bezierCurveTo(q.x + q.w * 0.33, yy + (rnd() - 0.5) * q.h * 0.18,
                                q.x + q.w * 0.66, yy + (rnd() - 0.5) * q.h * 0.18,
                                q.x + q.w + 2, yy + (rnd() - 0.5) * q.h * 0.1);
                c.stroke();
            }
            c.restore();
        }

        // bevel: light top-left, dark bottom-right, plus an inner shadow
        c.save();
        roundRectPath(c, q.x, q.y, q.w, q.h, g.r);
        c.clip();
        var bw = Math.max(1, q.h * 0.06);
        c.lineWidth = bw;
        c.strokeStyle = 'rgba(255,255,255,0.22)';
        c.beginPath();
        c.moveTo(q.x + g.r * 0.5, q.y + bw * 0.5);
        c.lineTo(q.x + q.w - g.r * 0.5, q.y + bw * 0.5);
        c.stroke();
        c.beginPath();
        c.moveTo(q.x + bw * 0.5, q.y + g.r * 0.5);
        c.lineTo(q.x + bw * 0.5, q.y + q.h - g.r * 0.5);
        c.stroke();
        c.strokeStyle = 'rgba(0,0,0,0.20)';
        c.beginPath();
        c.moveTo(q.x + g.r * 0.5, q.y + q.h - bw * 0.5);
        c.lineTo(q.x + q.w - g.r * 0.5, q.y + q.h - bw * 0.5);
        c.stroke();
        c.beginPath();
        c.moveTo(q.x + q.w - bw * 0.5, q.y + g.r * 0.5);
        c.lineTo(q.x + q.w - bw * 0.5, q.y + q.h - g.r * 0.5);
        c.stroke();
        c.restore();

        roundRectPath(c, q.x + 0.5, q.y + 0.5, q.w - 1, q.h - 1, g.r);
        c.strokeStyle = this.theme.gridLine;
        c.lineWidth = 1;
        c.stroke();
    };

    QBoard.prototype._paintCoords = function (c) {
        var g = this._g, t = this.theme;
        var fs = Math.max(8, Math.round(g.m * 0.52));
        c.font = '600 ' + fs + 'px "JetBrains Mono", ui-monospace, monospace';
        c.fillStyle = t.coord;
        c.textAlign = 'center';
        c.textBaseline = 'middle';
        var i;
        // files a..i, never mirrored, below the board
        for (i = 0; i < N; i++) {
            var q = this._cellRect(0, i);
            c.fillText(String.fromCharCode(97 + i), q.x + g.cell / 2, g.m + g.inner + g.m * 0.5);
        }
        // ranks 1..9 (rank = 9 - engine row) in the left margin
        for (i = 0; i < N; i++) {
            var qq = this._cellRect(i, 0);
            c.fillText(String(N - i), g.m * 0.5, qq.y + g.cell / 2);
        }
    };

    /* ------------------------------- overlays ---------------------------- */

    QBoard.prototype._drawGoalGlow = function (ctx) {
        var w = this.position.winner;
        if (w !== 0 && w !== 1) return;
        var g = this._g, t = this.theme;
        var goalRow = w === 0 ? N - 1 : 0;
        var q0 = this._cellRect(goalRow, 0);
        var grd = ctx.createLinearGradient(0, q0.y, 0, q0.y + g.cell);
        var glow = t.goalGlow;
        grd.addColorStop(0, rgba(glow, 0.15));
        grd.addColorStop(0.5, rgba(glow, 1));
        grd.addColorStop(1, rgba(glow, 0.15));
        ctx.save();
        ctx.fillStyle = grd;
        roundRectPath(ctx, q0.x - g.gap * 0.4, q0.y - g.gap * 0.4, g.inner + g.gap * 0.8, g.cell + g.gap * 0.8, g.r);
        ctx.fill();
        ctx.restore();
    };

    QBoard.prototype._drawLastMoveCells = function (ctx) {
        var lm = this.overlays.lastMove;
        if (!lm || lm.isWall) return;
        var g = this._g, t = this.theme;
        var cells = [];
        if (typeof lm.from === 'number') cells.push(lm.from | 0);
        cells.push(lm.a | 0);
        ctx.save();
        for (var i = 0; i < cells.length; i++) {
            var cIdx = clamp(cells[i], 0, 80);
            var q = this._cellRect((cIdx / N) | 0, cIdx % N);
            ctx.fillStyle = rgba(t.lastMove, i === cells.length - 1 ? 1 : 0.6);
            roundRectPath(ctx, q.x, q.y, q.w, q.h, g.r);
            ctx.fill();
        }
        ctx.restore();
    };

    QBoard.prototype._drawLastWallTint = function (ctx) {
        // Persistent marker for a wall last move, so it stays identifiable
        // after the one-shot pulse ends (and when animations are off).
        var lm = this.overlays.lastMove;
        if (!lm || !lm.isWall) return;
        var g = this._g, t = this.theme;
        var q = this._wallRect(lm.a | 0, lm.b | 0, lm.c | 0);
        ctx.save();
        roundRectPath(ctx, q.x - 2, q.y - 2, q.w + 4, q.h + 4, Math.max(3, g.r));
        ctx.strokeStyle = rgba(t.lastMove, 0.9);
        ctx.lineWidth = 2.4;
        ctx.stroke();
        ctx.restore();
    };

    QBoard.prototype._drawCorridorPips = function (ctx) {
        var g = this._g;
        ctx.save();
        ctx.fillStyle = 'rgba(255,255,255,0.16)';
        for (var r = 0; r < WS; r++) {
            for (var c = 0; c < WS; c++) {
                var p = this._slotToPoint(r, c);
                ctx.beginPath();
                ctx.arc(p.x, p.y, Math.max(0.8, g.gap * 0.22), 0, Math.PI * 2);
                ctx.fill();
            }
        }
        ctx.restore();
    };

    QBoard.prototype._drawPaths = function (ctx) {
        var paths = this.overlays.paths;
        if (!paths) return;
        var g = this._g, t = this.theme;
        ctx.save();
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';
        for (var p = 0; p < paths.length && p < 2; p++) {
            var path = paths[p];
            if (!path || path.length < 2) continue;
            var col = p === 0 ? t.pathP0 : t.pathP1;
            var w = Math.max(1.5, g.cell * 0.13);

            // soft halo, then the line itself
            ctx.strokeStyle = rgba(col, 0.16);
            ctx.lineWidth = w * 2.1;
            this._tracePath(ctx, path);
            ctx.stroke();

            ctx.setLineDash([w * 1.35, w * 1.15]);
            ctx.strokeStyle = rgba(col, 0.85);
            ctx.lineWidth = w;
            this._tracePath(ctx, path);
            ctx.stroke();
            ctx.setLineDash([]);

            // goal end cap
            var last = this._cellCenter(clamp(path[path.length - 1] | 0, 0, 80));
            ctx.fillStyle = rgba(col, 0.9);
            ctx.beginPath();
            ctx.arc(last.x, last.y, w * 0.85, 0, Math.PI * 2);
            ctx.fill();
        }
        ctx.restore();
    };

    QBoard.prototype._tracePath = function (ctx, path) {
        ctx.beginPath();
        for (var i = 0; i < path.length; i++) {
            var v = path[i];
            if (typeof v !== 'number' || v < 0 || v > 80) continue;
            var p = this._cellCenter(v | 0);
            if (i === 0) ctx.moveTo(p.x, p.y); else ctx.lineTo(p.x, p.y);
        }
    };

    QBoard.prototype._drawLegalDots = function (ctx) {
        if (this.mode === 'edit') return;
        var set = this.legal.pawn;
        if (!set || set.size === 0) return;
        var g = this._g, t = this.theme;
        var own = this.position.pawns[this.position.turn ? 1 : 0] | 0;
        var or_ = (own / N) | 0, oc = own % N;
        var self = this;
        ctx.save();
        set.forEach(function (cell) {
            if (cell < 0 || cell > 80) return;
            var er = (cell / N) | 0, ec = cell % N;
            var center = self._cellCenter(cell);
            var step = Math.abs(er - or_) + Math.abs(ec - oc);
            var straightJump = (Math.abs(er - or_) === 2 && ec === oc) || (Math.abs(ec - oc) === 2 && er === or_);
            var diagonal = (Math.abs(er - or_) === 1 && Math.abs(ec - oc) === 1);
            if (straightJump || diagonal || step > 1) {
                // jump / diagonal target: a ring
                ctx.strokeStyle = rgba(t.dot, 1.0);
                ctx.lineWidth = Math.max(2, g.cell * 0.085);
                ctx.beginPath();
                ctx.arc(center.x, center.y, g.cell * 0.30, 0, Math.PI * 2);
                ctx.stroke();
                ctx.strokeStyle = 'rgba(255,255,255,0.18)';
                ctx.lineWidth = Math.max(1, g.cell * 0.03);
                ctx.beginPath();
                ctx.arc(center.x, center.y, g.cell * 0.30, 0, Math.PI * 2);
                ctx.stroke();
            } else {
                ctx.fillStyle = rgba(t.dot, 1.0);
                ctx.beginPath();
                ctx.arc(center.x, center.y, g.cell * 0.155, 0, Math.PI * 2);
                ctx.fill();
                ctx.fillStyle = 'rgba(255,255,255,0.16)';
                ctx.beginPath();
                ctx.arc(center.x, center.y - g.cell * 0.03, g.cell * 0.10, 0, Math.PI * 2);
                ctx.fill();
            }
        });
        ctx.restore();
    };

    /* --------------------------------- walls ----------------------------- */

    QBoard.prototype._drawWalls = function (ctx) {
        var g = this._g;
        var pos = this.position;
        var ownerH = pos.wallOwner ? pos.wallOwner.h : null;
        var ownerV = pos.wallOwner ? pos.wallOwner.v : null;
        var pulse = this._pulse;
        var pt = 0;
        if (pulse && this._nowTs) {
            pt = clamp((this._nowTs - (pulse.t0 || this._nowTs)) / pulse.dur, 0, 1);
        }

        for (var o = 0; o < 2; o++) {
            var bits = o === 0 ? pos.wallsH : pos.wallsV;
            var own = o === 0 ? ownerH : ownerV;
            for (var slot = 0; slot < 64; slot++) {
                if (!bitAt(bits, slot)) continue;
                var r = (slot / WS) | 0, c = slot % WS;
                var glow = 0;
                if (pulse && pulse.o === o && pulse.r === r && pulse.c === c) {
                    glow = Math.sin(pt * Math.PI) * (1 - pt * 0.35);
                }
                this._paintWall(ctx, o, r, c, ownerAt(own, slot), 1, glow);
            }
        }

        // wall currently fading in from animateMove()
        var a = this._anim;
        if (a && a.kind === 'wall' && this._nowTs) {
            var k = clamp((this._nowTs - (a.t0 || this._nowTs)) / a.dur, 0, 1);
            this._paintWall(ctx, a.o, a.r, a.c, a.mover, easeOutCubic(k), 0, (1 - easeOutCubic(k)) * g.cell * 0.35);
        }
    };

    // Slab footprint: the corridor rect grown so the wall overlaps onto
    // both neighbouring cells -- walls are chunky pieces resting ON the
    // board, not sticks inside the gap.
    QBoard.prototype._wallRect = function (o, r, c) {
        var g = this._g;
        var q = this._slotRect(o, r, c);
        if (!q) return null;
        var ex = Math.max(2.5, g.cell * 0.19);          // growth per side
        if (o === 0) return { x: q.x + ex * 0.25, y: q.y - ex, w: q.w - ex * 0.5, h: q.h + ex * 2 };
        return { x: q.x - ex, y: q.y + ex * 0.25, w: q.w + ex * 2, h: q.h - ex * 0.5 };
    };

    QBoard.prototype._paintWall = function (ctx, o, r, c, owner, alpha, glow, lift) {
        var g = this._g, t = this.theme;
        var q = this._wallRect(o, r, c);
        if (!q) return;
        lift = lift || 0;
        var y = q.y - lift;
        var rad = Math.max(2, Math.min(q.w, q.h) * 0.28);
        var elev = Math.max(2, Math.min(q.w, q.h) * 0.34) + lift;

        ctx.save();
        ctx.globalAlpha = clamp(alpha, 0, 1);

        // cast shadow onto the board
        ctx.save();
        ctx.shadowColor = t.wallShadow;
        ctx.shadowBlur = Math.max(2, g.gap * (1.1 + lift / Math.max(1, g.gap)));
        ctx.shadowOffsetX = elev * 0.45;
        ctx.shadowOffsetY = elev * 0.9;
        ctx.fillStyle = 'rgba(0,0,0,0.9)';
        roundRectPath(ctx, q.x, y, q.w, q.h, rad);
        ctx.fill();
        ctx.restore();

        // slab body: gradient across the short axis
        var grd = o === 0
            ? ctx.createLinearGradient(q.x, y, q.x, y + q.h)
            : ctx.createLinearGradient(q.x, y, q.x + q.w, y);
        grd.addColorStop(0, shade(t.wall, 0.30));
        grd.addColorStop(0.42, t.wall);
        grd.addColorStop(1, shade(t.wallEdge, -0.10));
        roundRectPath(ctx, q.x, y, q.w, q.h, rad);
        ctx.fillStyle = grd;
        ctx.fill();

        // owner tint
        if (owner === 0 || owner === 1) {
            ctx.save();
            roundRectPath(ctx, q.x, y, q.w, q.h, rad);
            ctx.clip();
            ctx.fillStyle = rgba(owner === 0 ? t.p0 : t.p1, 0.28);
            ctx.fillRect(q.x, y, q.w, q.h);
            // coloured end caps make ownership readable even at small sizes
            ctx.fillStyle = rgba(owner === 0 ? t.p0 : t.p1, 0.9);
            var capL = o === 0 ? q.w * 0.10 : q.w;
            var capT = o === 0 ? q.h : q.h * 0.10;
            ctx.fillRect(q.x, y, capL, capT);
            if (o === 0) ctx.fillRect(q.x + q.w - capL, y, capL, capT);
            else ctx.fillRect(q.x, y + q.h - capT, capL, capT);
            ctx.restore();
        }

        // top highlight and the darker "side" edge
        ctx.save();
        roundRectPath(ctx, q.x, y, q.w, q.h, rad);
        ctx.clip();
        var hi = Math.max(0.8, (o === 0 ? q.h : q.w) * 0.26);
        ctx.fillStyle = 'rgba(255,255,255,0.30)';
        if (o === 0) ctx.fillRect(q.x, y, q.w, hi);
        else ctx.fillRect(q.x, y, hi, q.h);
        ctx.fillStyle = 'rgba(0,0,0,0.28)';
        if (o === 0) ctx.fillRect(q.x, y + q.h - hi * 0.8, q.w, hi * 0.8);
        else ctx.fillRect(q.x + q.w - hi * 0.8, y, hi * 0.8, q.h);
        ctx.restore();

        roundRectPath(ctx, q.x + 0.4, y + 0.4, q.w - 0.8, q.h - 0.8, rad);
        ctx.strokeStyle = rgba(t.wallEdge, 0.85);
        ctx.lineWidth = Math.max(0.7, g.gap * 0.12);
        ctx.stroke();

        if (glow > 0) {
            ctx.save();
            ctx.shadowColor = rgba(t.goalGlow, 1);
            ctx.shadowBlur = g.gap * 3 * glow;
            ctx.strokeStyle = rgba(t.goalGlow, 0.9 * glow);
            ctx.lineWidth = Math.max(1.2, g.gap * 0.3);
            roundRectPath(ctx, q.x, y, q.w, q.h, rad);
            ctx.stroke();
            ctx.restore();
        }
        ctx.restore();
    };

    QBoard.prototype._drawGhost = function (ctx) {
        var gh = this._ghost || this.overlays.ghostWall;
        if (!gh) return;
        var o = (gh.o !== undefined ? gh.o : gh.a) | 0;
        var r = (gh.r !== undefined ? gh.r : gh.b) | 0;
        var c = gh.c | 0;
        if (r < 0 || r >= WS || c < 0 || c >= WS) return;
        var legal = (gh.legal === undefined) ? this._wallLegal(o, r, c) : !!gh.legal;
        var g = this._g;
        var q = this._wallRect(o, r, c);
        if (!q) return;
        var rad = Math.max(2, Math.min(q.w, q.h) * 0.28);
        var col = legal ? '#4a9c6a' : '#c0394a';

        ctx.save();
        ctx.shadowColor = rgba(col, 0.65);
        ctx.shadowBlur = g.gap * 2.2;
        roundRectPath(ctx, q.x, q.y, q.w, q.h, rad);
        ctx.fillStyle = rgba(col, 0.55);
        ctx.fill();
        ctx.shadowBlur = 0;
        roundRectPath(ctx, q.x, q.y, q.w, q.h, rad);
        ctx.strokeStyle = rgba(col, 0.95);
        ctx.lineWidth = Math.max(1.2, g.gap * 0.26);
        ctx.stroke();

        // a light seam along the top so the ghost still reads as a slab
        ctx.save();
        roundRectPath(ctx, q.x, q.y, q.w, q.h, rad);
        ctx.clip();
        ctx.fillStyle = 'rgba(255,255,255,0.28)';
        var hi = Math.max(0.8, (o === 0 ? q.h : q.w) * 0.24);
        if (o === 0) ctx.fillRect(q.x, q.y, q.w, hi); else ctx.fillRect(q.x, q.y, hi, q.h);
        ctx.restore();
        ctx.restore();
    };

    QBoard.prototype._drawCursor = function (ctx) {
        if (!this._cursor) return;
        var g = this._g, t = this.theme;
        var q = this._cellRect(this._cursor.r, this._cursor.c);
        ctx.save();
        ctx.strokeStyle = rgba(t.goalGlow, 0.95);
        ctx.lineWidth = Math.max(1.5, g.cell * 0.06);
        roundRectPath(ctx, q.x + 1, q.y + 1, q.w - 2, q.h - 2, g.r);
        ctx.stroke();
        ctx.restore();
    };

    /* --------------------------------- pawns ----------------------------- */

    QBoard.prototype._drawPawns = function (ctx) {
        var g = this._g;
        var a = this._anim;
        for (var p = 0; p < 2; p++) {
            var cell = this.position.pawns[p] | 0;
            var pt;
            if (a && a.kind === 'pawn' && a.player === p && this._nowTs) {
                var k = easeInOutCubic(clamp((this._nowTs - (a.t0 || this._nowTs)) / a.dur, 0, 1));
                var from = this._cellCenter(clamp(a.from, 0, 80));
                var to = this._cellCenter(clamp(a.to, 0, 80));
                pt = { x: from.x + (to.x - from.x) * k, y: from.y + (to.y - from.y) * k };
                // a small hop makes the slide read as a step
                pt.y -= Math.sin(k * Math.PI) * g.cell * 0.14;
            } else {
                pt = this._cellCenter(clamp(cell, 0, 80));
            }
            this._paintPawn(ctx, pt.x, pt.y, g.cell * 0.34, p);
        }
    };

    QBoard.prototype._paintPawn = function (ctx, x, y, R, player) {
        var t = this.theme;
        var body = player === 0 ? t.p0 : t.p1;
        var edge = player === 0 ? t.p0Edge : t.p1Edge;
        ctx.save();

        // ground shadow, shared by every style
        ctx.save();
        ctx.fillStyle = 'rgba(0,0,0,0.30)';
        ctx.filter = 'none';
        ctx.beginPath();
        ctx.ellipse(x + R * 0.10, y + R * 0.78, R * 0.92, R * 0.34, 0, 0, Math.PI * 2);
        ctx.fill();
        ctx.restore();

        var style = this.pawnStyle;
        if (style === 'pin') this._pawnPin(ctx, x, y, R, body, edge);
        else if (style === 'chess') this._pawnChess(ctx, x, y, R, body, edge);
        else if (style === 'glyph') this._pawnGlyph(ctx, x, y, R, body, edge);
        else this._pawnDisc(ctx, x, y, R, body, edge);

        ctx.restore();
    };

    QBoard.prototype._pawnDisc = function (ctx, x, y, R, body, edge) {
        // side wall of the cylinder
        ctx.beginPath();
        ctx.ellipse(x, y + R * 0.20, R, R * 0.92, 0, 0, Math.PI * 2);
        ctx.fillStyle = shade(edge, -0.10);
        ctx.fill();

        var grd = ctx.createRadialGradient(x - R * 0.35, y - R * 0.42, R * 0.08, x, y, R * 1.12);
        grd.addColorStop(0, shade(body, 0.35));
        grd.addColorStop(0.55, body);
        grd.addColorStop(1, shade(body, -0.28));
        ctx.beginPath();
        ctx.arc(x, y, R, 0, Math.PI * 2);
        ctx.fillStyle = grd;
        ctx.fill();

        ctx.beginPath();
        ctx.arc(x, y, R * 0.995, 0, Math.PI * 2);
        ctx.strokeStyle = edge;
        ctx.lineWidth = Math.max(1.2, R * 0.17);
        ctx.stroke();

        ctx.beginPath();
        ctx.arc(x, y, R * 0.55, 0, Math.PI * 2);
        ctx.strokeStyle = rgba(edge, 0.45);
        ctx.lineWidth = Math.max(0.7, R * 0.07);
        ctx.stroke();

        // specular
        ctx.beginPath();
        ctx.ellipse(x - R * 0.30, y - R * 0.38, R * 0.30, R * 0.18, -0.6, 0, Math.PI * 2);
        ctx.fillStyle = 'rgba(255,255,255,0.42)';
        ctx.fill();
    };

    QBoard.prototype._pawnPin = function (ctx, x, y, R, body, edge) {
        var baseY = y + R * 0.86;
        ctx.save();
        // base disc
        ctx.beginPath();
        ctx.ellipse(x, baseY, R * 0.82, R * 0.28, 0, 0, Math.PI * 2);
        ctx.fillStyle = shade(edge, 0.05);
        ctx.fill();

        // tapered body
        var grd = ctx.createLinearGradient(x - R, y, x + R, y);
        grd.addColorStop(0, shade(body, -0.24));
        grd.addColorStop(0.34, shade(body, 0.30));
        grd.addColorStop(1, shade(body, -0.30));
        ctx.beginPath();
        ctx.moveTo(x - R * 0.72, baseY);
        ctx.bezierCurveTo(x - R * 0.74, y + R * 0.20, x - R * 0.40, y - R * 0.02, x - R * 0.26, y - R * 0.34);
        ctx.lineTo(x + R * 0.26, y - R * 0.34);
        ctx.bezierCurveTo(x + R * 0.40, y - R * 0.02, x + R * 0.74, y + R * 0.20, x + R * 0.72, baseY);
        ctx.closePath();
        ctx.fillStyle = grd;
        ctx.fill();
        ctx.strokeStyle = edge;
        ctx.lineWidth = Math.max(1, R * 0.12);
        ctx.stroke();

        // head sphere
        var hy = y - R * 0.56, hr = R * 0.44;
        var hg = ctx.createRadialGradient(x - hr * 0.4, hy - hr * 0.45, hr * 0.1, x, hy, hr * 1.25);
        hg.addColorStop(0, shade(body, 0.45));
        hg.addColorStop(0.6, body);
        hg.addColorStop(1, shade(body, -0.3));
        ctx.beginPath();
        ctx.arc(x, hy, hr, 0, Math.PI * 2);
        ctx.fillStyle = hg;
        ctx.fill();
        ctx.strokeStyle = edge;
        ctx.lineWidth = Math.max(1, R * 0.12);
        ctx.stroke();

        // collar ring in the player colour
        ctx.beginPath();
        ctx.ellipse(x, y - R * 0.28, R * 0.34, R * 0.11, 0, 0, Math.PI * 2);
        ctx.fillStyle = rgba(edge, 0.85);
        ctx.fill();

        ctx.beginPath();
        ctx.ellipse(x - hr * 0.32, hy - hr * 0.36, hr * 0.30, hr * 0.20, -0.6, 0, Math.PI * 2);
        ctx.fillStyle = 'rgba(255,255,255,0.5)';
        ctx.fill();
        ctx.restore();
    };

    QBoard.prototype._pawnChess = function (ctx, x, y, R, body, edge) {
        var s = R / 1.0;
        var baseY = y + s * 0.92;
        var grd = ctx.createLinearGradient(x - s, y, x + s, y);
        grd.addColorStop(0, shade(body, -0.28));
        grd.addColorStop(0.32, shade(body, 0.34));
        grd.addColorStop(1, shade(body, -0.34));

        ctx.beginPath();
        // foot
        ctx.moveTo(x - s * 0.86, baseY);
        ctx.lineTo(x + s * 0.86, baseY);
        ctx.lineTo(x + s * 0.70, baseY - s * 0.20);
        ctx.lineTo(x - s * 0.70, baseY - s * 0.20);
        ctx.closePath();
        ctx.fillStyle = grd;
        ctx.fill();
        ctx.strokeStyle = edge;
        ctx.lineWidth = Math.max(1, s * 0.11);
        ctx.stroke();

        // stem, flaring to the foot and to the collar
        ctx.beginPath();
        ctx.moveTo(x - s * 0.62, baseY - s * 0.20);
        ctx.bezierCurveTo(x - s * 0.44, baseY - s * 0.62, x - s * 0.26, y + s * 0.10, x - s * 0.24, y - s * 0.06);
        ctx.lineTo(x + s * 0.24, y - s * 0.06);
        ctx.bezierCurveTo(x + s * 0.26, y + s * 0.10, x + s * 0.44, baseY - s * 0.62, x + s * 0.62, baseY - s * 0.20);
        ctx.closePath();
        ctx.fillStyle = grd;
        ctx.fill();
        ctx.strokeStyle = edge;
        ctx.stroke();

        // collar
        ctx.beginPath();
        ctx.ellipse(x, y - s * 0.10, s * 0.50, s * 0.16, 0, 0, Math.PI * 2);
        ctx.fillStyle = shade(body, 0.16);
        ctx.fill();
        ctx.strokeStyle = edge;
        ctx.lineWidth = Math.max(1, s * 0.10);
        ctx.stroke();

        // head
        var hy = y - s * 0.52, hr = s * 0.40;
        var hg = ctx.createRadialGradient(x - hr * 0.4, hy - hr * 0.45, hr * 0.1, x, hy, hr * 1.3);
        hg.addColorStop(0, shade(body, 0.48));
        hg.addColorStop(0.6, body);
        hg.addColorStop(1, shade(body, -0.32));
        ctx.beginPath();
        ctx.arc(x, hy, hr, 0, Math.PI * 2);
        ctx.fillStyle = hg;
        ctx.fill();
        ctx.strokeStyle = edge;
        ctx.stroke();

        ctx.beginPath();
        ctx.ellipse(x - hr * 0.3, hy - hr * 0.38, hr * 0.30, hr * 0.19, -0.6, 0, Math.PI * 2);
        ctx.fillStyle = 'rgba(255,255,255,0.5)';
        ctx.fill();
    };

    QBoard.prototype._pawnGlyph = function (ctx, x, y, R, body, edge) {
        var size = R * 2.35;
        ctx.save();
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.font = size + 'px "Segoe UI Symbol", "Noto Sans Symbols 2", "DejaVu Sans", serif';
        // ring in the player colour behind the glyph
        ctx.beginPath();
        ctx.arc(x, y, R * 1.02, 0, Math.PI * 2);
        ctx.fillStyle = rgba(body, 0.16);
        ctx.fill();
        ctx.strokeStyle = rgba(edge, 0.9);
        ctx.lineWidth = Math.max(1.2, R * 0.14);
        ctx.stroke();

        ctx.lineJoin = 'round';
        ctx.lineWidth = Math.max(1.5, R * 0.24);
        ctx.strokeStyle = edge;
        ctx.strokeText('♟', x, y + R * 0.06);
        ctx.fillStyle = body;
        ctx.fillText('♟', x, y + R * 0.06);
        ctx.restore();
    };

    /* -------------------------------- arrows ----------------------------- */

    QBoard.prototype._drawArrows = function (ctx) {
        var list = this.overlays.arrows || [];
        var hint = this.overlays.hint;
        var all = list.slice();
        if (hint) {
            if (hint.isWall) all.push({ wall: { o: hint.a | 0, r: hint.b | 0, c: hint.c | 0 }, color: this.theme.arrow, hint: true });
            else if (typeof hint.from === 'number') all.push({ from: hint.from | 0, to: hint.a | 0, color: this.theme.arrow, hint: true });
        }
        // A wall last move gets its persistent outline from
        // _drawLastWallTint (drawn right after the walls).
        for (var i = 0; i < all.length; i++) {
            var a = all[i];
            if (!a) continue;
            if (a.wall) this._drawWallArrow(ctx, a);
            else if (typeof a.from === 'number' && typeof a.to === 'number') this._drawPawnArrow(ctx, a);
        }
    };

    QBoard.prototype._drawPawnArrow = function (ctx, a) {
        var g = this._g;
        var from = clamp(a.from | 0, 0, 80), to = clamp(a.to | 0, 0, 80);
        if (from === to) return;
        var p0 = this._cellCenter(from), p1 = this._cellCenter(to);
        var dx = p1.x - p0.x, dy = p1.y - p0.y;
        var len = Math.sqrt(dx * dx + dy * dy);
        if (len < 1) return;
        var ux = dx / len, uy = dy / len;

        var wmul = num(a.width, 1);
        var wTail = g.cell * 0.20 * wmul;
        var wNeck = g.cell * 0.145 * wmul;
        var headL = g.cell * 0.42 * wmul;
        var headW = g.cell * 0.40 * wmul;

        var startPad = g.cell * 0.30;
        var endPad = g.cell * 0.16;
        var sx = p0.x + ux * startPad, sy = p0.y + uy * startPad;
        var ex = p1.x - ux * endPad, ey = p1.y - uy * endPad;
        var shaftLen = Math.sqrt((ex - sx) * (ex - sx) + (ey - sy) * (ey - sy));
        if (shaftLen <= headL * 0.7) { headL = Math.max(shaftLen * 0.6, 1); headW = headL * 0.95; }
        var nx = ex - ux * headL, ny = ey - uy * headL;   // neck
        var px = -uy, py = ux;                            // perpendicular

        var col = a.color || this.theme.arrow;
        ctx.save();
        ctx.globalAlpha = num(a.alpha, 0.82);
        ctx.fillStyle = col;
        ctx.lineJoin = 'round';
        ctx.lineCap = 'round';
        ctx.strokeStyle = col;
        ctx.lineWidth = Math.max(1, g.cell * 0.05);

        ctx.beginPath();
        ctx.moveTo(sx + px * wTail, sy + py * wTail);
        ctx.lineTo(nx + px * wNeck, ny + py * wNeck);
        ctx.lineTo(nx + px * headW / 2, ny + py * headW / 2);
        ctx.lineTo(ex, ey);
        ctx.lineTo(nx - px * headW / 2, ny - py * headW / 2);
        ctx.lineTo(nx - px * wNeck, ny - py * wNeck);
        ctx.lineTo(sx - px * wTail, sy - py * wTail);
        ctx.closePath();
        ctx.fill();
        ctx.stroke();   // round join/cap gives the head a soft tip

        ctx.restore();
    };

    QBoard.prototype._drawWallArrow = function (ctx, a) {
        var g = this._g;
        var w = a.wall;
        var o = w.o | 0, r = w.r | 0, c = w.c | 0;
        if (r < 0 || r >= WS || c < 0 || c >= WS) return;
        var q = this._slotRect(o, r, c);
        if (!q) return;
        var col = a.color || this.theme.arrow;
        var rad = Math.max(1.5, g.gap * 0.38);
        ctx.save();
        ctx.globalAlpha = num(a.alpha, 0.8);
        ctx.shadowColor = rgba(col, 0.9);
        ctx.shadowBlur = g.gap * 2.4;
        roundRectPath(ctx, q.x, q.y, q.w, q.h, rad);
        ctx.fillStyle = rgba(col, 0.5);
        ctx.fill();
        ctx.shadowBlur = 0;
        ctx.strokeStyle = col;
        ctx.lineWidth = Math.max(1.2, g.gap * 0.3);
        ctx.setLineDash([g.gap * 1.5, g.gap * 0.9]);
        roundRectPath(ctx, q.x, q.y, q.w, q.h, rad);
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.restore();
    };

    /* -------------------------------- teardown --------------------------- */

    QBoard.prototype.destroy = function () {
        if (this._destroyed) return;
        this._destroyed = true;
        if (this._raf) { global.cancelAnimationFrame(this._raf); this._raf = 0; }
        this._finishAnim();
        this._pulse = null;
        var cv = this.canvas, h = this._h;
        try {
            cv.removeEventListener('pointerdown', h.down);
            cv.removeEventListener('pointermove', h.move);
            cv.removeEventListener('pointerup', h.up);
            cv.removeEventListener('pointercancel', h.cancel);
            cv.removeEventListener('pointerleave', h.leave);
            cv.removeEventListener('contextmenu', h.ctx);
            cv.removeEventListener('keydown', h.key);
        } catch (e) { /* ignore */ }
        if (global.removeEventListener) global.removeEventListener('resize', h.win);
        if (this._ro) { try { this._ro.disconnect(); } catch (e) { } this._ro = null; }
        this._base = null;
        this.onPawnMove = this.onWallPlace = this.onWallRemove = null;
        this.onPawnPlace = this.onHoverWall = this.onWallOrientationChange = null;
    };

    global.QBoard = QBoard;

})(typeof window !== 'undefined' ? window : this);
