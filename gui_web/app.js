// app.js -- Zquoridor premium GUI logic (plan gui-premium.md).
// Sections: 1 constants - 2 wasm bindings - 3 settings - 4 state -
// 5 hud/clocks/race/eval - 6 board bridge - 7 wall input - 8 pawn input -
// 9 play flow - 10 sound/haptics - 11 modals/toasts - 12 keyboard - 13 boot
// plus 14 analysis (plan section 5.6, phase P7).
'use strict';

// ===================== 1. constants ====================================
// The levels are named after the chess pieces, in order of material value,
// to match the sister project. A level is a search time budget. There is no
// blunder injection at any level: a weak level is a short search.
const LEVELS = {
  pawn:   { ms: 50,   label: 'Pawn',   color: 'var(--muted)', desc: '50 ms per move' },
  knight: { ms: 150,  label: 'Knight', color: 'var(--txt2)',  desc: '150 ms per move' },
  bishop: { ms: 400,  label: 'Bishop', color: 'var(--green)', desc: '400 ms per move' },
  rook:   { ms: 1000, label: 'Rook',   color: 'var(--blue)',  desc: '1 s per move' },
  queen:  { ms: 2500, label: 'Queen',  color: 'var(--amber)', desc: '2.5 s per move' },
  king:   { ms: 8000, label: 'King',   color: 'var(--gold)',  desc: '8 s per move' },
};
// Level keys before the chess naming. A stored blob from that schema keeps its
// search budget, which is what the reader chose, not its old name.
const LEVEL_V1 = { pebble: 'pawn', sprite: 'knight', squire: 'bishop',
                   knight: 'rook', sage: 'queen', titan: 'king' };
// An unknown level key must never reach the search. It falls back to the
// production default instead of throwing on a missing time budget.
function curLevel() { return LEVELS[S.level] || LEVELS.rook; }
const BOARD_THEMES = ['wood', 'walnut', 'ivory', 'marble', 'parchment', 'obsidian', 'slate', 'emerald', 'noir'];
const PAWN_STYLES = ['disc', 'pillar', 'crown', 'rune', 'pawnChess', 'beacon'];
const FILES = 'abcdefghi';

// Absolute notation (plan 5.8 / 16.2): pawn e5, wall Hc6/Vf3. Rank 1 is
// player 0's home row, matching the QFEN and the board coordinates.
function engCellAlg(cell) { return FILES[cell % 9] + (Math.floor(cell / 9) + 1); }
// Packed layout is isWall<<24 | a<<16 | b<<8 | c. A pawn move carries its
// destination cell in a, not in c, and a wall needs its file and rank as well
// as its orientation. The previous version read the low byte for both, so
// every pawn move in a line printed as "a1" and every wall as a bare "H".
function packedToTok(packed) {
  const isWall = (packed >> 24) & 1;
  const a = (packed >> 16) & 255, b = (packed >> 8) & 255, c = packed & 255;
  if (!isWall) return engCellAlg(a);
  return (a === 0 ? 'H' : 'V') + FILES[c] + (b + 1);
}
function plyNotation(i) {
  const isWall = W.plyIsWall(i);
  if (!isWall) return engCellAlg(W.plyA(i));
  return (W.plyA(i) === 0 ? 'H' : 'V') + FILES[W.plyC(i)] + (W.plyB(i) + 1);
}
function packPly(i) {
  return (W.plyIsWall(i) ? (1 << 24) : 0) | (W.plyA(i) << 16) | (W.plyB(i) << 8) | W.plyC(i);
}
function unpackMove(packed) {
  return { isWall: !!((packed >> 24) & 1), a: (packed >> 16) & 255, b: (packed >> 8) & 255, c: packed & 255 };
}

// ===================== 2. wasm bindings ================================
let W = null;   // wrapped exports
function bindEngine(m) {
  const c = name => {
    if (typeof m[name] !== 'function') throw new Error('export missing: ' + name);
    return (...args) => m[name](...args);
  };
  // String helpers for the QFEN surface (ASCII only, so byte-wise decoding
  // is safe even across memory growth -- HEAPU8 is re-read every call).
  function readCStr(buf) {
    const u8 = m.HEAPU8; let s = '';
    for (let i = 0; i < 640; i++) { const b = u8[buf + i]; if (!b) break; s += String.fromCharCode(b); }
    return s;
  }
  function withCStr(str, fn) {
    const bytes = new TextEncoder().encode(String(str) + '\0');
    const p = m._malloc(bytes.length);
    try { m.HEAPU8.set(bytes, p); return fn(p); } finally { m._free(p); }
  }
  function callStrOut(name) {
    const buf = m._malloc(640);
    try { if (m[name](buf, 640) < 0) return null; return readCStr(buf); }
    finally { m._free(buf); }
  }
  W = {
    newGame: c('_qr_new_game'),
    turn: c('_qr_turn'),
    winner: c('_qr_winner'),
    pawn: c('_qr_pawn'),
    wallsLeft: c('_qr_walls_left'),
    wallHBit: c('_qr_wall_h_bit'),
    wallVBit: c('_qr_wall_v_bit'),
    dist: c('_qr_dist_to_goal'),
    moveCount: c('_qr_legal_moves_count'),
    mvIsWall: c('_qr_legal_move_is_wall'),
    mvA: c('_qr_legal_move_a'),
    mvB: c('_qr_legal_move_b'),
    mvC: c('_qr_legal_move_c'),
    applyPawn: c('_qr_apply_pawn_move'),
    applyWall: c('_qr_apply_wall_move'),
    engineMove: c('_qr_engine_move'),
    lastIsWall: c('_qr_last_move_is_wall'),
    lastA: c('_qr_last_move_a'),
    lastB: c('_qr_last_move_b'),
    lastC: c('_qr_last_move_c'),
    lastEval: c('_qr_last_move_eval'),
    isDraw: c('_qr_is_draw'),
    loadNnue: c('_qr_load_nnue_weights'),
    // P6: history navigation
    plyCount: c('_qr_ply_count'),
    cursor: c('_qr_cursor'),
    plyIsWall: c('_qr_ply_is_wall'),
    plyA: c('_qr_ply_a'),
    plyB: c('_qr_ply_b'),
    plyC: c('_qr_ply_c'),
    gotoPly: c('_qr_goto_ply'),
    truncateHistory: c('_qr_truncate_history'),
    // P6: scratch position (analysis / editor)
    scratchReset: c('_qr_scratch_reset'),
    scratchFromLive: c('_qr_scratch_from_live'),
    scratchFromPly: c('_qr_scratch_from_ply'),
    scrApplyPawn: c('_qr_scr_apply_pawn'),
    scrApplyWall: c('_qr_scr_apply_wall'),
    scrTurn: c('_qr_scr_turn'),
    scrPawn: c('_qr_scr_pawn'),
    scrWallsLeft: c('_qr_scr_walls_left'),
    scrWallHBit: c('_qr_scr_wall_h_bit'),
    scrWallVBit: c('_qr_scr_wall_v_bit'),
    scrDist: c('_qr_scr_dist'),
    // P6: multi-line analysis
    analyze: c('_qr_analyze'),
    anLineCount: c('_qr_an_line_count'),
    anLineScore: c('_qr_an_line_score'),
    anLineLen: c('_qr_an_line_len'),
    anLineMove: c('_qr_an_line_move'),
    anNodes: c('_qr_an_nodes'),
    anDepth: c('_qr_an_depth'),
    // P6: editor + QFEN (editor UI lands in P8; bindings ready)
    editSetPawn: c('_qr_edit_set_pawn'),
    editSetWall: c('_qr_edit_set_wall'),
    editSetWallsLeft: c('_qr_edit_set_walls_left'),
    editSetTurn: c('_qr_edit_set_turn'),
    editValidity: c('_qr_edit_validity'),
    editApply: c('_qr_edit_apply'),
    qfenExportStr: () => callStrOut('_qr_qfen_export'),
    qfenExportScratchStr: () => callStrOut('_qr_qfen_export_scratch'),
    qfenImportStr: s => withCStr(s, p => m._qr_qfen_import_scratch(p)),
    lastErrStr: () => callStrOut('_qr_last_error'),
  };
  window.__w = W;   // test/debug handle (browser smoke tests)
}

// ===================== 3. settings =====================================
// Full schema (plan section 11). Unknown keys survive a merge; a corrupt
// blob resets to defaults with a toast. `preset` tracks which coherent
// preset is active; touching any option flips it to 'custom'.
const DEFAULTS = {
  v: 4,
  preset: 'premiumDark',
  ui: 'dark', board: 'wood', pawn: 'disc',
  accent: 'gold', fs: 1, density: 'comfortable', uiFont: 'modern',
  // 'beveled' by default: the hairline frame is a flat band with no depth,
  // and the bevel gradient is what makes the board read as an object.
  frame: 'beveled', wallFinish: 'beveled', cellSep: 'grooves',
  boardTexture: 'subtle', boardContrast: 'standard', wallProfile: 'standard', wallPreview: 'normal',
  goalRows: 'subtle', moveMarkers: 'ring', lastMoveStyle: 'subtle',
  coords: 'edges', boardScale: 1,
  pawnSize: 'regular', pawnShadow: 'soft', distinctShapes: false,
  paths: false, dots: true, lastMove: true, evalGlow: true, evalBar: true,
  sound: true, soundPack: 'wood', volume: .5,
  soundEvents: { moves: true, walls: true, illegal: true, clock: true, end: true, ui: true },
  haptics: 'full',
  anim: 'full', animSpeed: 1,
  confirmWalls: null, touchOffset: null, stickyArm: false, handedness: 'right',
  level: 'rook',
  custom: { mode: 'time', depth: 12, timeMs: 1000 },
  clockMode: '5+3', baseMin: 5, incSec: 3, side: 0,
  flipped: false,
  worker: true, autosave: true,
};
let S = { ...DEFAULTS, soundEvents: { ...DEFAULTS.soundEvents } };
function loadSettings() {
  try {
    const raw = localStorage.getItem('zq.settings');
    if (!raw) return;
    const parsed = JSON.parse(raw);
    // Schema 1 named the levels after ranks and objects. Map the stored key to
    // the chess piece that holds the same time budget.
    if ((parsed.v || 1) < 2 && LEVEL_V1[parsed.level]) parsed.level = LEVEL_V1[parsed.level];
    // Schema 3 makes the shortest-path overlay strictly opt-in. Older blobs
    // that still carry paths=true lose it once, so the board starts clean.
    if ((parsed.v || 1) < 3) parsed.paths = false;
    if ((parsed.v || 1) < 4) {
      if (parsed.lastMoveStyle == null) parsed.lastMoveStyle = parsed.lastMove === false ? 'off' : 'subtle';
      if (parsed.moveMarkers == null) parsed.moveMarkers = 'ring';
    }
    if (!LEVELS[parsed.level]) delete parsed.level;
    S = {
      ...DEFAULTS, ...parsed, v: DEFAULTS.v,
      custom: { ...DEFAULTS.custom, ...(parsed.custom || {}) },
      soundEvents: { ...DEFAULTS.soundEvents, ...(parsed.soundEvents || {}) },
    };
  } catch (e) {
    S = JSON.parse(JSON.stringify(DEFAULTS));
    setTimeout(() => toast('info', 'Settings reset'), 800);
  }
}
function saveSettings() {
  try { localStorage.setItem('zq.settings', JSON.stringify(S)); }
  catch (e) { toast('warn', 'Settings are session only (storage full)'); }
}
const coarse = matchMedia('(pointer:coarse)').matches;
function touchOffsetPx() {
  if (S.touchOffset === 'off') return 0;
  if (S.touchOffset === 'small') return .6 * B.U;
  if (S.touchOffset === 'large') return 1.0 * B.U;
  return coarse ? .6 * B.U : 0;                       // auto
}
function confirmOn() { return S.confirmWalls === null ? coarse : S.confirmWalls; }

// ---- presets (plan 17.8): each sets ~15 options coherently -------------
const PRESETS = {
  classic: {
    ui: 'light', board: 'ivory', pawn: 'disc', accent: 'gold', frame: 'beveled',
    wallFinish: 'flat', cellSep: 'grooves', coords: 'edges', density: 'comfortable',
    pawnShadow: 'deep', soundPack: 'wood', anim: 'full', fs: 1, uiFont: 'modern',
    boardTexture: 'subtle', boardContrast: 'standard', wallProfile: 'standard', wallPreview: 'normal',
    goalRows: 'subtle', moveMarkers: 'ring', lastMoveStyle: 'subtle',
  },
  premiumDark: {},
  highContrast: {
    ui: 'dark', board: 'noir', pawn: 'rune', distinctShapes: true, accent: 'gold',
    coords: 'all', fs: 1.12, cellSep: 'inlaid', wallFinish: 'beveled', anim: 'reduced',
    boardTexture: 'subtle', boardContrast: 'strong', wallProfile: 'standard', wallPreview: 'strong',
    goalRows: 'clear', moveMarkers: 'ring', lastMoveStyle: 'clear', uiFont: 'modern',
  },
  minimal: {
    ui: 'dark', board: 'slate', pawn: 'rune', frame: 'none', wallFinish: 'flat',
    cellSep: 'flat', coords: 'off', paths: false, dots: false, evalBar: false,
    evalGlow: false, sound: false, haptics: 'off', anim: 'reduced', density: 'compact',
    boardTexture: 'off', boardContrast: 'soft', wallProfile: 'slim', wallPreview: 'subtle',
    goalRows: 'off', moveMarkers: 'minimal', lastMoveStyle: 'off', uiFont: 'modern',
  },
};
function applyPreset(name) {
  if (name === 'custom' || !PRESETS[name]) return;
  S = { ...S, ...PRESETS[name], preset: name };
  saveSettings(); applySettings();
}

// ---- accent colour (plan 17.2): presets or a custom hex ----------------
const ACCENTS = {
  gold: ['#c8a84b', '#e6c96e'], crimson: ['#c0394a', '#e8697a'],
  jade: ['#3f9d6d', '#6fd0a0'], sapphire: ['#4a7bc0', '#7fa8e0'],
  amethyst: ['#8a5fc0', '#b08ae0'], copper: ['#c07840', '#e09a68'],
  teal: ['#3f9da0', '#6fd0d2'], slate: ['#7a8699', '#a5b1c4'],
};
function applyAccent() {
  let pair = ACCENTS[S.accent];
  if (!pair && typeof S.accent === 'string' && /^#[0-9a-fA-F]{6}$/.test(S.accent)) {
    // derive a lighter companion for --gold2
    const n = parseInt(S.accent.slice(1), 16);
    const lift = ch => Math.min(255, Math.round(ch + (255 - ch) * .35));
    pair = [S.accent,
      '#' + [lift(n >> 16), lift((n >> 8) & 255), lift(n & 255)]
        .map(v => v.toString(16).padStart(2, '0')).join('')];
  }
  if (!pair) pair = ACCENTS.gold;
  const root = document.documentElement.style;
  root.setProperty('--gold', pair[0]);
  root.setProperty('--gold2', pair[1]);
  const rgb = pair[0].match(/^#(..)(..)(..)$/).map(v => parseInt(v, 16));
  root.setProperty('--gold-dim', `rgba(${rgb[1]},${rgb[2]},${rgb[3]},.16)`);
  root.setProperty('--gold-glow', `rgba(${rgb[1]},${rgb[2]},${rgb[3]},.35)`);
}

// ===================== 4. state ========================================
let B = null;                    // QBoard
let gameOver = false;
let humanSide = 0;
let engineThinking = false;
let legalPawn = new Set();       // display cells
let legalWall = new Uint8Array(128);  // [o*64 + r*8+c] display coords
let lastMoveInfo = null;

// ===================== 5. hud / clocks / race / eval ===================
const $ = id => document.getElementById(id);

// One player strip component, used twice. The top strip always carries the
// opponent and the bottom strip always carries you, so every number keeps one
// fixed place on the screen. The name sits on the outer edge of its strip:
// at the very top for the top player, at the very bottom for the bottom one.
function strip(pos) { return $(pos === 'top' ? 'hudTop' : 'hudBottom'); }
// The strips carry .win and .lose styling -- a green border on the winner --
// and newGame clears both. Nothing ever added them, so no ending marked who
// had won: not a goal reached, not a resignation, not a flag. Pass -1 for a
// draw, which marks neither.
function markResult(winnerSide) {
  const seat = { top: 1 - humanSide, bottom: humanSide };
  for (const pos of ['top', 'bottom']) {
    const el = strip(pos);
    if (!el) continue;
    el.classList.remove('win', 'lose');
    if (winnerSide === -1) continue;
    el.classList.add(seat[pos] === winnerSide ? 'win' : 'lose');
  }
}
function pips(el, n) {
  if (!el) return;
  if (el.childElementCount !== 10) el.innerHTML = '<i></i>'.repeat(10);
  [...el.children].forEach((p, i) => p.classList.toggle('on', i < n));
}
function fmtClock(ms) {
  if (ms == null) return '--:--';
  const s = Math.max(0, Math.ceil(ms / 1000));
  return Math.floor(s / 60) + ':' + String(s % 60).padStart(2, '0');
}
let clockMs = [null, null], clockTimer = null, lastTickAt = 0;
// Clock readings per ply count, so a takeback can give the time back. Index k
// holds the reading reached after k plies.
let clockHist = [];
function startClock() {
  stopClock();
  clockHist = [];
  if (S.clockMode === 'none') { renderClocks(); return; }
  clockMs = [S.baseMin * 60000, S.baseMin * 60000];
  lastTickAt = performance.now();
  clockTimer = setInterval(clockTick, 200);
  renderClocks();
}
function stopClock() { if (clockTimer) clearInterval(clockTimer); clockTimer = null; }
// Fischer increment, credited to the side that has just moved.
function addIncrement(side) {
  if (S.clockMode === 'none' || !S.incSec || gameOver) return;
  clockMs[side] += S.incSec * 1000;
  renderClocks();
}
// Starts the ticker again for a game that runs again, for example after a
// takeback of a move that lost on time. lastTickAt must be reset here or the
// first tick charges the whole idle interval to the side to move.
function resumeClock() {
  if (S.clockMode === 'none' || gameOver || clockTimer) return;
  lastTickAt = performance.now();
  clockTimer = setInterval(clockTick, 200);
}
function clockTick() {
  const now = performance.now();
  const dt = now - lastTickAt; lastTickAt = now;
  const t = W.turn();
  clockMs[t] -= dt;
  if (clockMs[t] <= 0) { clockMs[0] = Math.max(0, clockMs[0]); flagFall(t); }
  renderClocks();
}
function flagFall(side) {
  stopClock(); gameOver = true; engineThinking = false;
  setStatus((side === humanSide ? 'Your clock ran out' : 'Zquoridor flagged you'));
  const el = strip(side === humanSide ? 'bottom' : 'top');
  if (el) el.classList.add('flag');
  markResult(1 - side);   // whoever still had time on the clock
  pushRecent('flag');
  sound('end'); haptic([80]);
}
function renderClocks() {
  const none = S.clockMode === 'none';
  for (const pos of ['top', 'bottom']) {
    const el = strip(pos);
    if (!el) continue;
    const pl = Number(el.dataset.player || 0);
    const c = el.querySelector('.clock');
    if (!c) continue;
    c.textContent = fmtClock(none ? null : clockMs[pl]);
    c.classList.toggle('off', none);
    c.classList.toggle('low', !none && clockMs[pl] < 30000 && clockMs[pl] >= 10000);
    c.classList.toggle('crit', !none && clockMs[pl] < 10000);
  }
}
function refreshHud() {
  const wl = [W.wallsLeft(0), W.wallsLeft(1)];
  const d = [W.dist(0), W.dist(1)];
  const t = W.turn();
  B.setTurn(t);
  const seat = { top: 1 - humanSide, bottom: humanSide };
  for (const pos of ['top', 'bottom']) {
    const pl = seat[pos], el = strip(pos);
    if (!el) continue;
    el.dataset.player = pl;
    const nm = el.querySelector('.nm');
    if (nm) nm.textContent = pl === humanSide ? 'You' : 'Zquoridor';
    const wn = el.querySelector('.wn');
    if (wn) wn.textContent = wl[pl];
    pips(el.querySelector('.pips'), wl[pl]);
    const dEl = el.querySelector('.dist');
    if (dEl) {
      dEl.innerHTML = '<b>' + d[pl] + '</b>';
      dEl.classList.toggle('lead', d[pl] < d[1 - pl]);
      dEl.classList.toggle('behind', d[pl] > d[1 - pl]);
    }
    const toMove = t === pl && !gameOver;
    el.classList.toggle('active', toMove);
    el.classList.toggle('thinking', toMove && pl !== humanSide && engineThinking);
  }
  // The H and V buttons are only the orientation fallback, but they still
  // carry the wall count because that is where the thumb is while placing.
  const myWalls = wl[humanSide];
  const bh = $('badgeH'), bv = $('badgeV');
  if (bh) bh.textContent = myWalls;
  if (bv) bv.textContent = myWalls;
  const noWalls = myWalls <= 0 || t !== humanSide || gameOver || engineThinking || !atLiveEnd();
  $('wallH').classList.toggle('off', noWalls);
  $('wallV').classList.toggle('off', noWalls);
  const noBack = W.plyCount() === 0 || engineThinking || !atLiveEnd();
  for (const id of ['btnTakeback', 'btnUndo']) {
    const el = $(id);
    if (el) { el.classList.toggle('off', noBack); el.disabled = noBack; }
  }
  const dYou = d[humanSide], dOpp = d[1 - humanSide];
  setRace(dOpp / Math.max(1, dYou + dOpp) * 100, dYou, dOpp);
  renderClocks();
}
// The race meter is horizontal in every layout. Your share grows as the
// opponent's path grows.
function setRace(share, dYou, dOpp) {
  const txt = dYou + ' : ' + dOpp;
  for (const trio of [['raceFill0', 'raceDiv', 'raceLbl']]) {
    const fe = $(trio[0]), de = $(trio[1]), le = $(trio[2]);
    if (fe) { fe.style.width = share.toFixed(1) + '%'; fe.style.setProperty('--share', (share / 100).toFixed(3)); }
    if (de) de.style.left = share.toFixed(1) + '%';
    if (le) le.textContent = txt;
  }
}
function setEval(score) {   // score mover-relative from the engine's last move
  const abs = humanSide === 0 ? score : -score;      // de-mirror to colour 0 view
  const share = 50 + 50 * Math.tanh(abs / 400);
  const fill = $('evalFill'), num = $('evalNum'), bar = $('evalStrip');
  if (!fill || !bar) return;
  const vertical = matchMedia('(min-width:900px)').matches;
  if (vertical) { fill.style.width = ''; fill.style.height = share.toFixed(1) + '%'; }
  else { fill.style.height = ''; fill.style.width = share.toFixed(1) + '%'; }
  // The vertical bar hides this label, so the tooltip carries the number there.
  bar.title = 'Evaluation ' + Math.round(share) + '%';
  if (!num) return;
  num.textContent = Math.round(share) + '%';
  num.style.top = '';
}

// ===================== 6. board bridge =================================
// Last-move info straight from the recorded ply (no snapshot diffing).
function plyToLastMove(i) {
  if (i < 0 || i >= W.plyCount()) return null;
  if (!W.plyIsWall(i)) {
    const disp = B.engPawnToDisp(W.plyA(i));
    return { type: 'pawn', r: Math.floor(disp / 9), c: disp % 9 };
  }
  return { type: 'wall', o: W.plyA(i), r: W.plyB(i), c: W.plyC(i) };
}
function syncFromEngine() {
  const pw = [W.pawn(0), W.pawn(1)];
  const wh = [], wv = [];
  for (let s = 0; s < 64; s++) { wh.push(W.wallHBit(s)); wv.push(W.wallVBit(s)); }
  // Manual flip persists as a setting; the default orientation puts the
  // human (side 0) at the bottom.
  B.flipped = ((humanSide === 1) !== !!S.flipped);
  lastMoveInfo = plyToLastMove(W.cursor() - 1);
  B.linePreview = null;
  B.selected = -1;
  // Dots and the side-to-move marker must be current BEFORE setData paints,
  // or they only appear on the render after this one.
  buildLegalSets();
  refreshHud();
  B.setData(pw, wh, wv, lastMoveInfo);
  recomputePaths();
  renderMoveLog();
  renderAnMoveLog();
  updateNav();
  drawGraph();
  if (atLiveEnd()) clockHist[W.plyCount()] = clockMs.slice();
  if (typeof AN !== 'undefined' && AN.on) anRestart();
}
function buildLegalSets() {
  legalPawn.clear(); legalWall.fill(0);
  const n = W.moveCount();
  for (let i = 0; i < n; i++) {
    if (!W.mvIsWall(i)) legalPawn.add(B.engPawnToDisp(W.mvA(i)));
    else {
      const o = W.mvA(i), r = W.mvB(i), c = W.mvC(i);
      const [do_, dr, dc] = B.engWallToDisp(o, r, c);
      legalWall[do_ * 64 + dr * 8 + dc] = 1;
    }
  }
  B.dots = (S.dots && W.turn() === humanSide && !gameOver)
    ? [...legalPawn] : [];
}

// ===================== 9. play flow ====================================
// The status line has two writers: the game flow, and the wall drag. The
// engine's reply lands on a timer, so a routine "Your move" could arrive a few
// milliseconds after the user had already pressed on an illegal groove, and
// wipe the reason the ghost was red. While a drag owns the line, only the drag
// writes it. Anything that ends the game still writes, because that outranks a
// drag that is about to be cancelled anyway.
function setStatus(t, fromDrag) {
  if (!fromDrag && wallState === 'DRAGGING' && !gameOver) return;
  $('status').textContent = t;
}

// Single-owner engine timer: scheduling a new engine turn cancels any stale
// one, and engineTurn refuses to act unless it really is the engine's move.
// This makes "the engine playing for both sides" structurally impossible.
let engineTimer = null;
function scheduleEngineTurn(delay) {
  if (engineTimer) clearTimeout(engineTimer);
  engineTimer = setTimeout(() => { engineTimer = null; engineTurn(); }, delay);
}
function afterHumanMove(anim) {
  updateMovesChip();
  addIncrement(humanSide);
  engineThinking = false;
  refreshHud();
  // The engine only starts to think once the piece has finished sliding, so
  // the move you just played is always visible before the screen changes again.
  const done = (anim && typeof anim.then === 'function') ? anim : Promise.resolve();
  done.then(() => {
    checkEnd();
    if (gameOver) { refreshHud(); return; }
    engineThinking = true;
    refreshHud();
    setStatus('Zquoridor is thinking...');
    scheduleEngineTurn(40);
  });
}
// Generation counter for engine searches. A new game, a takeback or a jump
// through the history bumps it, and any reply that belongs to an older
// generation is dropped instead of being played into the new position.
let engineGen = 0;
function engineTurn() {
  if (!engineThinking) return;   // stale timer already superseded
  if (gameOver || W.winner() !== -1 || W.isDraw()) {
    engineThinking = false; syncAll(); return;
  }
  if (W.turn() === humanSide) {   // never let the engine move for the human
    engineThinking = false;
    syncAll();
    setStatus('Your move');
    return;
  }
  const lv = curLevel();
  const gen = ++engineGen;
  const pl = W.turn();
  const from = B.engPawnToDisp(W.pawn(pl));

  const finish = (packed, score, applied) => {
    if (gen !== engineGen) return;
    engineThinking = false;
    if (packed == null) { syncAll(); setStatus('Your move'); return; }
    const m = unpackMove(packed);
    if (!applied) {
      const ok = m.isWall ? W.applyWall(m.a, m.b, m.c) : W.applyPawn(m.a);
      if (!ok) { syncAll(); setStatus('Your move'); return; }
    }
    let anim = null;
    if (m.isWall) {
      const dw = B.engWallToDisp(m.a, m.b, m.c);
      anim = B.animateWall ? B.animateWall(dw[0], dw[1], dw[2]) : null;
      sound('wall'); haptic(18);
    } else {
      anim = B.animateMove ? B.animateMove(pl, from, B.engPawnToDisp(W.pawn(pl))) : null;
      sound('move'); haptic(10);
    }
    setEval(-score);
    addIncrement(pl);
    syncAll();
    const done = (anim && typeof anim.then === 'function') ? anim : Promise.resolve();
    done.then(() => {
      checkEnd();
      if (!gameOver) setStatus('Your move');
      refreshHud();
    });
  };

  // Off-thread search when the worker is available. The worker replays the
  // recorded line into its own live game and runs the same hybrid search, so
  // the engine plays exactly as it does here while this thread keeps painting.
  // A game that started from a custom position has no replay root, so it stays
  // local.
  const offThread = !g_startedFromCustom() && ANW.bestMove(
    { moves: allPliesPacked(), depth: 24, timeMs: lv.ms },
    res => {
      if (gen !== engineGen) return;
      if (!res || res.move == null) { engineLocalMove(gen, finish); return; }
      finish(res.move, res.score, false);
    });
  if (offThread) return;
  engineLocalMove(gen, finish);
}
// Fallback search on this thread. It blocks for the whole time budget.
function engineLocalMove(gen, finish) {
  if (gen !== engineGen) return;
  const lv = curLevel();
  const ok = W.engineMove(24, lv.ms);
  finish(ok ? packPly(W.plyCount() - 1) : null, W.lastEval(), true);
}
function checkEnd() {
  const w = W.winner();
  if (w !== -1) {
    gameOver = true;
    const youWon = w === humanSide;
    setStatus(youWon ? 'You won - goal reached' : 'Zquoridor won');
    toast(youWon ? 'ok' : 'err', youWon ? 'Victory' : 'Defeat');
    markResult(w);
    sound('end'); haptic(youWon ? [20, 60, 20, 60, 40] : [60]);
    pushRecent('played');
    refreshHud(); return true;
  }
  if (W.isDraw()) { gameOver = true; markResult(-1); setStatus('Draw by repetition'); pushRecent('draw'); return true; }
  return false;
}
function syncAll() {
  syncFromEngine();
  checkEndQuiet();
  scheduleAutosave();
  announceState();
}
// Screen reader announcement (plan section 10): one short sentence per move
// into the live region.
function announceState() {
  const el = $('srBoard');
  if (!el || !W) return;
  const n = W.plyCount();
  if (!n) { el.textContent = 'New game. Player 0 to move.'; return; }
  const who = (n - 1) % 2 === 0 ? 'Player 0' : 'Player 1';
  const tok = plyNotation(n - 1);
  const walls = W.wallsLeft((n - 1) % 2);
  el.textContent = `${who}: ${tok}. ${walls} walls left. ` +
    `Distances ${W.dist(0)} and ${W.dist(1)}.`;
}
function checkEndQuiet() { const w = W.winner(); return w !== -1 || W.isDraw(); }
function newGame() {
  W.newGame(); gameOver = false; engineThinking = false;
  engineGen++;                       // drop any reply from the previous game
  for (const pos of ['top', 'bottom']) {
    const el = strip(pos);
    if (el) el.classList.remove('flag', 'win', 'lose');
  }
  humanSide = S.side;
  AN.scores = {}; AN.annots = {}; AN.curDepth = 1;
  levelMarks = [];
  g_startedFromCustomFlag = false; g_rootQfen = '-';
  clearGhost();
  // Seat the evaluation bar at even. Without this the fill keeps its 0% start
  // and the bare track reads as "one side wins outright", which is a claim the
  // engine never made. The opening position is even, so 50% is honest.
  setEval(0);
  setStatus(humanSide === 0 ? 'Your move' : 'Zquoridor starts');
  syncAll();
  startClock();
  if (W.turn() !== humanSide) { engineThinking = true; refreshHud(); scheduleEngineTurn(150); }
}
function updateMovesChip() { $('movesChip').textContent = 'Moves ' + Math.ceil(W.plyCount() / 2); }

// ---- navigation over the recorded game (plan 5.6 nav row) -------------
function atLiveEnd() { return W.cursor() >= W.plyCount(); }
function navGo(ply) {
  if (AN.bcRun) { toast('warn', 'Blunder check running - cancel it first'); return; }
  if (engineThinking) { toast('warn', 'Wait for the engine move'); return; }
  ply = Math.max(0, Math.min(W.plyCount(), ply));
  if (ply === W.cursor()) return;
  W.gotoPly(ply);
  gameOver = false;   // reviewing an earlier ply leaves any result display
  clearGhost();
  syncAll();
  const end = atLiveEnd();
  if (end && (W.winner() !== -1 || W.isDraw())) {
    checkEnd();       // restore the result state of a finished game
    return;
  }
  $('btnReturn').style.display = end ? 'none' : 'flex';
  setStatus(end ? (W.turn() === humanSide ? 'Your move' : 'Zquoridor to move')
                : 'Reviewing ply ' + W.cursor() + ' - Return to game');
}
function updateNav() {
  const cur = W.cursor(), n = W.plyCount();
  $('navPly').textContent = cur + ' / ' + n;
  const end = atLiveEnd();
  $('btnReturn').style.display = end ? 'none' : 'flex';
  // Dim what cannot move. All four stayed lit at both ends, so at ply 0 the
  // back pair looked available and did nothing, and the same at the last ply
  // for the forward pair. .btn.off is the same treatment the wall buttons use
  // when a player is out of walls.
  const off = (id, cond) => { const e = $(id); if (e) e.classList.toggle('off', cond); };
  off('navFirst', cur <= 0);
  off('navPrev', cur <= 0);
  off('navNext', cur >= n);
  off('navLast', cur >= n);
}
// Rolls back to the human's previous turn: the latest position before the
// cursor where it is the human to move, then truncates the future.
function takeback() {
  if (AN.bcRun) { toast('warn', 'Blunder check running - cancel it first'); return; }
  if (engineThinking) { toast('warn', 'Wait for the engine move'); return; }
  if (!atLiveEnd()) { toast('info', 'Return to the game before you take a move back'); return; }
  const n = W.plyCount();
  if (!n) { toast('info', 'Nothing to take back'); return; }
  let target = 0;
  for (let p = Math.min(W.cursor(), n) - 1; p >= 0; p--) {
    W.scratchFromPly(p);
    if (W.scrTurn() === humanSide) { target = p; break; }
  }
  W.truncateHistory(target);
  // The scratch position is shared with the editor and with the analysis
  // fallback. The search loop above left it on a probe position, so put it
  // back on the live game.
  W.scratchFromLive();
  engineGen++;             // any search still running belongs to the old line
  // Analysis data is keyed by ply index. Entries past the new end describe
  // moves that no longer exist and would otherwise be shown against the
  // different moves played next.
  for (const k of Object.keys(AN.scores)) if (Number(k) >= target) delete AN.scores[k];
  for (const k of Object.keys(AN.annots)) if (Number(k) >= target) delete AN.annots[k];
  if (Array.isArray(levelMarks)) {
    levelMarks = levelMarks.filter(m => (m && m.afterPly != null ? m.afterPly : 0) < target);
  }
  // Give the time back, and start the clock again for a game that had ended.
  if (clockHist[target]) clockMs = clockHist[target].slice();
  clockHist.length = Math.min(clockHist.length, target + 1);
  gameOver = false;
  for (const pos of ['top', 'bottom']) {
    const el = strip(pos);
    if (el) el.classList.remove('flag', 'win', 'lose');
  }
  clearGhost();
  setEval(0);
  resumeClock();
  setStatus('Takeback - your move');
  sound('arm');
  syncAll();
}
// ===================== 7. wall input (plan section 6) ==================
// One gesture, no mode. Moving over the board hit-tests the pointer: over a
// groove it shows a wall preview in that orientation, over a cell body it
// shows nothing. Pressing a groove starts the drag, dragging across the other
// axis flips the orientation, and a release over a legal slot places the wall.
// The H and V buttons only force an orientation for people who want an
// explicit control. They are never required.
let wallState = 'IDLE';    // IDLE | DRAGGING | PENDING
let armedO = 0;            // last used orientation
let forcedO = null;        // orientation forced by the H / V buttons or keys
let dragPtr = null;
let dragFrom = null;
let forcedTimer = null;
// Pawn drag: press your own pawn and pull it onto a legal destination.
// Tap to select and tap the destination keeps working; a press that does not
// move is exactly that first tap.
let pawnDrag = null;

function humanCanAct() {
  return !gameOver && !engineThinking && atLiveEnd() && W.turn() === humanSide;
}
function releaseForced() {
  forcedO = null;
  if (forcedTimer) { clearTimeout(forcedTimer); forcedTimer = null; }
  $('wallH').classList.remove('armed');
  $('wallV').classList.remove('armed');
  hideLegalSlots();
}
function clearGhost() {
  wallState = 'IDLE'; dragPtr = null; dragFrom = null; kbWall = null;
  B.ghost = null; B.ghostFrom = null;
  pawnDrag = null;
  if (B.setDragPawn) B.setDragPawn(null);
  if (B.setHover) B.setHover(null);
  $('confirmChip').style.display = 'none';
  releaseForced();
}
function armWall(o) {
  if ($(o === 0 ? 'wallH' : 'wallV').classList.contains('off')) return;
  const was = forcedO;
  clearGhost();
  if (was === o) { B.render(); return; }      // second press releases it
  forcedO = o; armedO = o; gestureO = o;
  $(o === 0 ? 'wallH' : 'wallV').classList.add('armed');
  showLegalSlots(o);
  // Arming shows the wall right away, translucent, so the button answers with
  // the thing it is about to place instead of only a lit border. It lands
  // where the pointer last was, or at the middle of the board before the
  // pointer has ever entered it.
  previewArmed(o);
  // No mode you can get stuck in: the forced orientation expires by itself.
  forcedTimer = setTimeout(() => { releaseForced(); B.render(); }, 6000);
  sound('arm'); haptic(6);
  B.render();
}
// Translucent preview for the armed orientation. B.setHover draws the wall at
// 0.35 alpha, which is the same preview the board shows on hover, so arming
// and hovering answer with one visual language.
// Anchor the keyboard is pointing at while an orientation is armed. Arrow keys
// move this instead of the pawn, and Enter commits it, which is what makes a
// wall placeable without a pointer at all.
let kbWall = null;
function previewArmed(o) {
  if (!B.setHover || !humanCanAct()) return;
  const p = lastPt || { x: B.cssSide / 2, y: B.cssSide / 2 };
  const a = anchorFor(o, p.x, p.y);
  let r = a.r, c = a.c;
  if (!legalWall[o * 64 + r * 8 + c]) {
    // Nearest legal slot of this orientation, so the preview is never a wall
    // the user cannot actually place.
    let best = null;
    for (let rr = 0; rr < 8; rr++) for (let cc = 0; cc < 8; cc++) {
      if (!legalWall[o * 64 + rr * 8 + cc]) continue;
      const ctr = B.anchorCenter(rr, cc);
      const d = Math.hypot(p.x - ctr.x, p.y - ctr.y);
      if (!best || d < best.d) best = { r: rr, c: cc, d };
    }
    if (!best) return;
    r = best.r; c = best.c;
  }
  kbWall = { o, r, c };
  B.setHover({ o, r, c });
}
// Repaint the keyboard preview after the cursor moves or the orientation flips.
function kbWallShow() {
  if (!kbWall || !B.setHover) return;
  const ok = !!legalWall[kbWall.o * 64 + kbWall.r * 8 + kbWall.c];
  B.setHover(ok ? { o: kbWall.o, r: kbWall.r, c: kbWall.c } : null);
  setStatus(ok ? '' : illegalReason(...B.dispWallToEng(kbWall.o, kbWall.r, kbWall.c)), true);
  B.render();
}
let slotLayer = null;
function showLegalSlots(o) {
  hideLegalSlots();
  slotLayer = document.createElement('canvas');
  const dpr = Math.min(devicePixelRatio || 1, 3);
  slotLayer.width = B.cv.width; slotLayer.height = B.cv.height;
  slotLayer.style.cssText = `position:absolute;left:0;top:0;width:${B.cssSide}px;height:${B.cssSide}px;pointer-events:none`;
  const g = slotLayer.getContext('2d'); g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.fillStyle = getComputedStyle(document.documentElement).getPropertyValue('--gold').trim();
  for (let r = 0; r < 8; r++) for (let c = 0; c < 8; c++) {
    if (legalWall[o * 64 + r * 8 + c]) {
      const rc = B.wallRect(o, r, c);
      g.globalAlpha = .18; g.fillRect(rc.x, rc.y, rc.w, rc.h);
    }
  }
  $('boardWrap').appendChild(slotLayer);
}
function hideLegalSlots() { if (slotLayer) { slotLayer.remove(); slotLayer = null; } }

function boardPoint(ev) {
  const rect = B.cv.getBoundingClientRect();
  return { x: ev.clientX - rect.left, y: ev.clientY - rect.top };
}

// ---- sweep direction ---------------------------------------------------
// The direction the pointer travels asks for an orientation: a sideways sweep
// asks for a horizontal wall, a vertical sweep for a vertical one. It reads
// pointermove deltas only, so it behaves the same with the button down and
// with it up, and the same for a finger as for a mouse.
//
// The accumulators decay instead of resetting. A decaying sum answers within a
// few samples but ignores the one stray pixel that a finger produces when it
// lifts, which a raw last-delta test does not.
let gestureO = 0;
let lastPt = null;
let travelX = 0, travelY = 0;
function trackTravel(pt) {
  if (lastPt) {
    // One sample is capped at half a cell. Without the cap a single jump --
    // the pointer entering the board, or a synthetic event that teleports --
    // outweighs every sample of the sweep that follows it, and the wall comes
    // out perpendicular to the gesture the user actually made.
    const cap = .5 * B.C;
    const dx = Math.min(cap, Math.abs(pt.x - lastPt.x));
    const dy = Math.min(cap, Math.abs(pt.y - lastPt.y));
    travelX = travelX * .72 + dx;
    travelY = travelY * .72 + dy;
    // Below the threshold the pointer is resting, and a resting pointer must
    // not flip the orientation under the user.
    if (Math.max(travelX, travelY) > .06 * B.C) gestureO = travelX > travelY ? 0 : 1;
  }
  lastPt = { x: pt.x, y: pt.y };
}
// A press starts a fresh gesture: what the pointer did on its way to the board
// is not part of the drag the user is making now.
function resetTravel(pt) {
  travelX = 0; travelY = 0;
  lastPt = pt ? { x: pt.x, y: pt.y } : null;
}
// The board is a lattice of cells of side C separated by grooves of width G,
// repeating every U = C + G. Groove k has its centre at M + (k+1)*U - G/2, so
// the nearest groove on an axis and the distance to it come straight from the
// pointer position.
function grooveNear(p, n) {
  const U = B.U, G = B.G, M = B.M;
  let k = Math.round((p - M + G / 2) / U) - 1;
  k = Math.max(0, Math.min(n - 1, k));
  return { k, d: Math.abs(p - (M + (k + 1) * U - G / 2)) };
}
// Anchor for one orientation. Along the wall's own long axis the anchor is the
// cell the pointer is literally inside (floor); across the groove it snaps to
// the nearest corridor (round). That asymmetry is what makes placement feel
// right, because the wall always grows away from the cell you pointed at.
function anchorFor(o, px, py) {
  const M = B.M, U = B.U, C = B.C;
  let r, c;
  if (o === 0) { c = Math.floor((px - M) / U); r = Math.round((py - M - C) / U); }
  else { r = Math.floor((py - M) / U); c = Math.round((px - M - C) / U); }
  return { r: Math.max(0, Math.min(7, r)), c: Math.max(0, Math.min(7, c)) };
}
function cellAt(px, py) {
  const raw = B.pointToCell(px, py);
  if (raw == null || raw === -1 || raw === false) return null;
  if (typeof raw === 'number') return { r: Math.floor(raw / 9), c: raw % 9, idx: raw };
  return { r: raw.r, c: raw.c, idx: raw.r * 9 + raw.c };
}
// Decides what the pointer is over: a cell body, or a groove and which one.
function boardHit(px, py) {
  if (!B.U) return { kind: 'none' };
  const gv = grooveNear(px, 8), gh = grooveNear(py, 8);
  const cell = cellAt(px, py);
  // The groove the sweep is asking for reaches further than the other one, so
  // a deliberate sideways sweep catches a horizontal groove even when the
  // pointer sits slightly nearer a vertical one. gestureO 0 = horizontal.
  const reachOn = B.G / 2 + .22 * B.C;    // orientation the sweep asks for
  const reachOff = B.G / 2 + .10 * B.C;   // the other one
  const onV = gv.d <= (gestureO === 1 ? reachOn : reachOff);
  const onH = gh.d <= (gestureO === 0 ? reachOn : reachOff);
  if (!onV && !onH) return cell ? { kind: 'cell', cell: cell.idx } : { kind: 'none' };
  // A click meant for the pawn must never place a wall. Outside the groove
  // proper, a cell the pawn can legally reach wins over the groove: the old
  // reach of .26*C ran roughly a quarter of a cell deep on all four sides, so
  // aiming at a destination square regularly armed a wall instead.
  const insideGroove = Math.min(gv.d, gh.d) <= B.G / 2;
  if (!insideGroove && cell && legalPawn.has(cell.idx)) {
    return { kind: 'cell', cell: cell.idx };
  }
  // At a crossing both orientations fit, so the sweep decides. Where only one
  // groove is in range, geometry decides: the other wall would land somewhere
  // the pointer never was.
  const o = (onV && onH) ? gestureO : (onV ? 1 : 0);
  const a = anchorFor(o, px, py);
  return { kind: 'groove', o, r: a.r, c: a.c };
}
// The hover preview is what replaces the wall mode button: the board itself
// tells you where the wall would go before you commit to anything.
function updateHover(px, py) {
  if (!B.setHover) return;
  if (!humanCanAct() || currentPane === 'edPane') { B.setHover(null); B.render(); return; }
  const hit = boardHit(px, py);
  if (hit.kind !== 'groove') { B.setHover(null); B.render(); return; }
  const o = forcedO != null ? forcedO : hit.o;
  const a = forcedO != null ? anchorFor(o, px, py) : { r: hit.r, c: hit.c };
  const legal = !!legalWall[o * 64 + a.r * 8 + a.c];
  B.setHover(legal ? { o, r: a.r, c: a.c } : null);
  B.render();
}
function snapGhost(px, py) {
  const off = touchOffsetPx();
  const p = { x: px, y: py - off };
  const o = armedO;
  const a = anchorFor(o, p.x, p.y);
  let r = a.r, c = a.c;
  const eng = B.dispWallToEng(o, r, c);
  let st = legalWall[o * 64 + r * 8 + c] ? 'ok' : null;
  if (!st) {   // magnetic assist: nearest LEGAL anchor around the pointer
    let best = null;
    for (let dr = -1; dr <= 1; dr++) for (let dc = -1; dc <= 1; dc++) {
      const rr2 = r + dr, cc2 = c + dc;
      if (rr2 < 0 || rr2 > 7 || cc2 < 0 || cc2 > 7) continue;
      if (!legalWall[o * 64 + rr2 * 8 + cc2]) continue;
      const ctr = B.anchorCenter(rr2, cc2);
      const dist = Math.hypot(p.x - ctr.x, p.y - ctr.y);
      if (dist <= .90 * B.U && (!best || dist < best.dist)) best = { r: rr2, c: cc2, dist };
    }
    if (best) { r = best.r; c = best.c; st = 'assisted'; }
  }
  if (!st) st = 'bad';
  B.ghost = { o, r, c, state: st, from: st === 'assisted' ? a : null };
  if (st === 'bad') setStatus(illegalReason(eng[0], eng[1], eng[2]), true);
  else setStatus(st === 'assisted' ? 'Snapped to the nearest legal slot' : '', true);
  if (B.setHover) B.setHover(null);
  B.render();
}
function illegalReason(o, r, c) {
  if (W.wallsLeft(humanSide) <= 0) return 'No walls left';
  const sameOcc = (o === 0 ? W.wallHBit(r * 8 + c) : W.wallVBit(r * 8 + c));
  if (sameOcc) return 'Overlaps an existing wall';
  const cross = (o === 0 ? W.wallVBit(r * 8 + c) : W.wallHBit(r * 8 + c));
  if (cross) return 'Crosses another wall';
  return 'Would leave a player with no path to goal';
}

let lastNudge = 0;
function thinkNudge() {
  const now = performance.now();
  if (now - lastNudge < 1500) return;
  lastNudge = now;
  toast('info', 'Zquoridor is thinking...');
}

function onBoardPointerDown(ev) {
  if (currentPane === 'edPane') { edBoardDown(ev); return; }
  if (gameOver) return;
  if (engineThinking) { thinkNudge(); return; }
  if (!atLiveEnd()) { toast('info', 'Reviewing an earlier ply - press Return to game'); return; }
  const pt = boardPoint(ev);
  const hit = boardHit(pt.x, pt.y);
  resetTravel(pt);
  // With no orientation forced, a press on a cell body belongs to the pawn.
  if (forcedO == null && hit.kind !== 'groove') {
    if (hit.kind !== 'cell') return;
    pawnDown(hit.cell, pt);
    const mine = B.engPawnToDisp(W.pawn(humanSide));
    if (hit.cell === mine && humanCanAct()) {
      pawnDrag = { ptr: ev.pointerId, from: hit.cell, moved: false };
      try { B.cv.setPointerCapture(ev.pointerId); } catch (e) { /* no capture: taps still work */ }
      ev.preventDefault();
    }
    return;
  }
  armedO = forcedO != null ? forcedO : hit.o;
  wallState = 'DRAGGING'; dragPtr = ev.pointerId; dragFrom = pt;
  try { B.cv.setPointerCapture(ev.pointerId); } catch (e) { /* no capture: still works */ }
  snapGhost(pt.x, pt.y);
  ev.preventDefault();
}
function onBoardPointerMove(ev) {
  const pt = boardPoint(ev);
  trackTravel(pt);
  if (pawnDrag && ev.pointerId === pawnDrag.ptr) {
    pawnDrag.moved = true;
    B.setDragPawn(humanSide, pt.x, pt.y);
    ev.preventDefault();
    return;
  }
  if (wallState !== 'DRAGGING' || ev.pointerId !== dragPtr) { updateHover(pt.x, pt.y); return; }
  // Pull along the wall you want. This is the same sweep test that decides the
  // orientation on hover, so the gesture does not change meaning when the
  // button goes down. The old rule compared the drag against its own origin
  // and needed .45 of a cell of travel before it would flip, which made a
  // change of mind mid-drag feel stuck.
  if (forcedO == null && dragFrom) armedO = gestureO;
  snapGhost(pt.x, pt.y);
  ev.preventDefault();
}
function onBoardPointerUp(ev) {
  if (pawnDrag && ev.pointerId === pawnDrag.ptr) {
    const moved = pawnDrag.moved;
    pawnDrag = null;
    B.setDragPawn(null);
    if (!moved) return;              // a press without movement was a tap
    const hit = boardHit(boardPoint(ev).x, boardPoint(ev).y);
    if (hit.kind === 'cell' && legalPawn.has(hit.cell) && humanCanAct()) playPawn(hit.cell);
    else B.render();
    return;
  }
  if (wallState !== 'DRAGGING' || ev.pointerId !== dragPtr) return;
  const gh = B.ghost;
  if (gh && (gh.state === 'ok' || gh.state === 'assisted')) {
    if (confirmOn()) { gh.state = 'pending'; wallState = 'PENDING'; showConfirmChip(gh); B.render(); return; }
    commitGhost(gh);
  } else {
    if (gh && gh.state === 'bad') { shake(); toast('warn', $('status').textContent); }
    clearGhost(); B.render();
  }
}
function commitGhost(gh) {
  if (!gh || !humanCanAct()) {
    clearGhost(); B.render(); return;   // stale ghost: never apply out of turn
  }
  const eng = B.dispWallToEng(gh.o, gh.r, gh.c);
  if (W.applyWall(eng[0], eng[1], eng[2])) {
    clearGhost();
    // Mirror the placed wall before the fade-in starts, so it appears now
    // instead of after the engine's reply repaints the board.
    syncFromEngine();
    const anim = B.animateWall ? B.animateWall(gh.o, gh.r, gh.c) : null;
    sound('wall'); haptic(18);
    afterHumanMove(anim);
  } else { clearGhost(); B.render(); }
}
function showConfirmChip(gh) {
  const rc = B.wallRect(gh.o, gh.r, gh.c), wrap = $('boardWrap');
  const chip = $('confirmChip');
  chip.style.display = 'flex';
  const cx = Math.min(rc.x + rc.w, wrap.clientWidth - 100);
  const cy = Math.max(6, rc.y - 54);
  chip.style.left = cx + 'px'; chip.style.top = cy + 'px';
}
$('ccOk').addEventListener('click', () => { if (B.ghost) commitGhost(B.ghost); });
$('ccNo').addEventListener('click', () => { clearGhost(); B.render(); });

// H and V: a press and drag carries the ghost onto the board, a plain press
// forces the orientation for the next placement.
for (const pair of [['wallH', 0], ['wallV', 1]]) {
  const el = $(pair[0]), o = pair[1];
  el.addEventListener('pointerdown', ev => {
    try { el.setPointerCapture(ev.pointerId); } catch (e) { /* ignore */ }
    el._downAt = performance.now(); el._dragging = false;
  });
  el.addEventListener('pointermove', ev => {
    if (!el._downAt) return;
    // Pointer capture retargets every move to the button, so this handler is
    // the only channel that can carry the ghost across the board.
    if (!el._dragging &&
        (Math.abs(ev.movementX) + Math.abs(ev.movementY) > 3 ||
         performance.now() - el._downAt > 150)) el._dragging = true;
    if (el._dragging && wallState !== 'DRAGGING') {
      if ($(pair[0]).classList.contains('off')) return;
      armedO = o; wallState = 'DRAGGING'; dragPtr = ev.pointerId; dragFrom = null;
    }
    if (wallState === 'DRAGGING' && ev.pointerId === dragPtr) {
      const br = B.cv.getBoundingClientRect();
      snapGhost(ev.clientX - br.left, ev.clientY - br.top);
    }
  });
  el.addEventListener('pointerup', ev => {
    if (wallState === 'DRAGGING' && ev.pointerId === dragPtr) { onBoardPointerUp(ev); }
    else if (performance.now() - el._downAt < 300) armWall(o);
    el._downAt = null; el._dragging = false;
  });
}

// ===================== 8. pawn input ===================================
function pawnDown(dispCell, pt) {
  void pt;
  const myPawnDisp = B.engPawnToDisp(W.pawn(humanSide));
  const isMyTurn = W.turn() === humanSide && !engineThinking;
  if (!atLiveEnd()) { toast('info', 'Reviewing an earlier ply - press Return to game'); return; }
  if (dispCell === myPawnDisp && isMyTurn) { B.selected = dispCell; B.dots = [...legalPawn]; B.render(); return; }
  if (isMyTurn && legalPawn.has(dispCell)) { playPawn(dispCell); return; }
  if (B.selected >= 0) { B.selected = -1; buildLegalSets(); B.render(); }
}
// The tween starts before the state sync paints, so the piece slides from the
// square it left instead of appearing at the new one.
function playPawn(dispCell) {
  const engCell = B.dispPawnToEng(Math.floor(dispCell / 9), dispCell % 9);
  const from = B.engPawnToDisp(W.pawn(humanSide));
  if (!W.applyPawn(engCell)) return;
  const anim = B.animateMove ? B.animateMove(humanSide, from, dispCell) : null;
  // Mirror the played move onto the board while the slide runs. Without this
  // the tween ends on a board that still holds the old cell, so the piece
  // snaps back and only jumps forward when the engine's reply repaints it.
  syncFromEngine();
  sound('move'); haptic(10);
  afterHumanMove(anim);
}

// ===================== 10. sound & haptics ==============================
// Synth packs (plan 17.4): each event maps to a small parameter table per
// pack, so adding a pack is data, not code. No audio files anywhere.
// The default pack is deliberately quiet and rounded: short tones with a soft
// attack and a lowpass filter, closer to a fine mechanism than to an arcade.
const SOUND_PACKS = {
  wood: {
    move:   [{ f: 340, dur: 45, type: 'sine', g: -16, lp: 1600, atk: 4 }],
    wall:   [{ f: 128, dur: 75, type: 'sine', g: -13, lp: 800, atk: 6 },
             { f: 510, dur: 26, type: 'triangle', g: -24, lp: 1800, atk: 2, at: 14 }],
    illegal:[{ f: 165, dur: 85, type: 'sine', g: -18, lp: 650, atk: 6 }],
    arm:    [{ f: 1150, dur: 20, type: 'sine', g: -22, atk: 2 }],
    hint:   [{ f: 880, dur: 55, type: 'sine', g: -20, atk: 8 }],
    clock:  [{ f: 1500, dur: 15, type: 'sine', g: -24, atk: 2 }],
    end:    [{ f: 523, dur: 170, type: 'sine', g: -14, atk: 10 },
             { f: 659, dur: 170, type: 'sine', g: -14, atk: 10, at: 120 },
             { f: 784, dur: 280, type: 'sine', g: -13, atk: 10, at: 240 }],
    loss:   [{ f: 392, dur: 240, type: 'sine', g: -16, atk: 12 },
             { f: 294, dur: 360, type: 'sine', g: -16, atk: 12, at: 200 }],
  },
  modern: {
    move:   [{ f: 520, dur: 35, type: 'sine', g: -18, lp: 2400, atk: 3 }],
    wall:   [{ f: 240, dur: 60, type: 'sine', g: -16, lp: 1200, atk: 5 },
             { f: 480, dur: 45, type: 'sine', g: -22, lp: 2000, atk: 3, at: 30 }],
    illegal:[{ f: 220, dur: 80, type: 'sine', g: -22, lp: 900, atk: 5 }],
    arm:    [{ f: 880, dur: 15, type: 'sine', g: -22, atk: 2 }],
    hint:   [{ f: 740, dur: 50, type: 'sine', g: -20, atk: 6 }],
    clock:  [{ f: 1200, dur: 18, type: 'sine', g: -22, atk: 2 }],
    end:    [{ f: 660, dur: 110, type: 'sine', g: -16, atk: 8 },
             { f: 880, dur: 150, type: 'sine', g: -16, atk: 8, at: 100 }],
    loss:   [{ f: 330, dur: 170, type: 'sine', g: -18, atk: 10 },
             { f: 250, dur: 270, type: 'sine', g: -18, atk: 10, at: 150 }],
  },
  marble: {
    move:   [{ f: 320, dur: 55, type: 'sine', g: -12 }],
    wall:   [{ f: 70, dur: 160, type: 'sine', g: -7 }, { f: 1100, dur: 25, type: 'triangle', g: -20 }],
    illegal:[{ f: 110, dur: 90, type: 'square', g: -14 }],
    arm:    [{ f: 1300, dur: 22, type: 'triangle', g: -18 }],
    hint:   [{ f: 990, dur: 70, type: 'triangle', g: -16 }],
    clock:  [{ f: 1500, dur: 20, type: 'triangle', g: -20 }],
    end:    [{ f: 440, dur: 150, type: 'sine', g: -9 }, { f: 554, dur: 150, type: 'sine', g: -9, at: 150 }, { f: 659, dur: 260, type: 'sine', g: -9, at: 300 }],
    loss:   [{ f: 294, dur: 240, type: 'sine', g: -11 }, { f: 220, dur: 380, type: 'sine', g: -11, at: 230 }],
  },
  silent: {
    move: [], wall: [], illegal: [], arm: [], hint: [],
    clock: [], end: [{ f: 523, dur: 120, type: 'triangle', g: -8 }], loss: [{ f: 392, dur: 300, type: 'triangle', g: -10 }],
  },
};
let AC = null;
let acArmed = false;
function ac() {
  if (!AC && acArmed) try { AC = new (window.AudioContext || webkitAudioContext)(); } catch (e) {}
  return AC;
}
function tone(freq, dur, type, gainDb) { toneAt({ f: freq, dur, type, g: gainDb }); }
function toneAt(p) {
  const a = ac(); if (!a) return;
  const o = a.createOscillator(), g = a.createGain();
  o.type = p.type || 'sine'; o.frequency.value = p.f;
  const vol = Math.pow(10, ((p.g || -10) / 20)) * S.volume;
  const t0 = a.currentTime + (p.at || 0) / 1000;
  // Soft attack, then the usual exponential decay. The ramp keeps even a short
  // click free of the sharp onset that reads as retro hardware.
  const atk = Math.max(.002, Math.min((p.atk || 3) / 1000, p.dur / 4000));
  g.gain.setValueAtTime(.0001, t0);
  g.gain.exponentialRampToValueAtTime(Math.max(.0002, vol), t0 + atk);
  g.gain.exponentialRampToValueAtTime(.0001, t0 + p.dur / 1000);
  // Optional lowpass rounds the tone; without it the voice connects straight
  // to the gain.
  let tail = o;
  if (p.lp) {
    const f = a.createBiquadFilter();
    f.type = 'lowpass'; f.frequency.value = p.lp; f.Q.value = .7;
    o.connect(f); tail = f;
  }
  tail.connect(g); g.connect(a.destination);
  o.start(t0); o.stop(t0 + p.dur / 1000 + .05);
}
// first user gesture unlocks the AudioContext (autoplay policy)
addEventListener('pointerdown', () => { acArmed = true; ac(); }, { once: true, capture: true });
function sound(kind) {
  if (!S.sound || kind === 'silent') return;
  const gateMap = { move: 'moves', wall: 'walls', illegal: 'illegal', arm: 'ui',
                    hint: 'ui', clock: 'clock', end: 'end', loss: 'end' };
  const gate = gateMap[kind] || kind;
  if (!S.soundEvents[gate]) return;
  const pack = SOUND_PACKS[S.soundPack] || SOUND_PACKS.wood;
  const evts = pack[kind === 'end' ? (gameOver && W && W.winner() !== -1 && W.winner() !== humanSide ? 'loss' : 'end') : kind];
  for (const p of evts || []) toneAt(p);
}
function haptic(pat) {
  if (S.haptics === 'off' || !navigator.vibrate) return;
  try { navigator.vibrate(S.haptics === 'light' ? Math.min(...[pat].flat()) : pat); } catch (e) {}
}
function shake() { sound('illegal'); haptic([12, 40, 12]); }

// ===================== 11. modals & toasts =============================
function toast(kind, msg) {
  const t = document.createElement('div');
  t.className = 'toast ' + kind; t.textContent = msg;
  $('toasts').appendChild(t);
  while ($('toasts').childElementCount > 3) $('toasts').firstChild.remove();
  setTimeout(() => t.remove(), kind === 'err' ? 4000 : 2600);
}
function openModal(html) { $('modalBox').innerHTML = html; $('overlay').classList.add('open'); }
function closeModal() { $('overlay').classList.remove('open'); }
// One handler for both ways out: the backdrop, and any element carrying
// data-close. The test is hasAttribute, not dataset.close: the markup writes a
// bare `data-close`, whose value is the empty string, and an empty string is
// falsy, so the close button never fired. closest() lets the button hold an
// icon or a span without losing the click.
$('overlay').addEventListener('click', e => {
  if (e.target.id === 'overlay' || e.target.closest('[data-close]')) closeModal();
});

// Small helpers shared by the settings / IO / editor surfaces.
function confirmModal(title, body, onYes) {
  openModal(`<h3>${title} <span class="x" data-close>&#10005;</span></h3>
    <p style="color:var(--txt2);font-size:var(--fs-sm);margin-bottom:12px">${body}</p>
    <div class="row" style="gap:8px">
      <button class="btn" id="cfNo" style="flex:1">Cancel</button>
      <button class="btn danger" id="cfYes" style="flex:1">Confirm</button>
    </div>`);
  $('cfNo').onclick = closeModal;
  $('cfYes').onclick = () => { closeModal(); onYes(); };
}
function downloadText(name, text, mime) {
  const blob = new Blob([text], { type: mime || 'text/plain' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url; a.download = name;
  document.body.appendChild(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 4000);
  toast('ok', 'Downloaded ' + name);
}
function pickFile(accept, cb) {
  const inp = document.createElement('input');
  inp.type = 'file'; inp.accept = accept || '.qgn,.qfen,.txt';
  inp.onchange = () => {
    const f = inp.files[0];
    if (!f) return;
    f.text().then(t => cb(t)).catch(() => toast('err', 'Could not read file'));
  };
  inp.click();
}
async function copyText(text, what) {
  try {
    await navigator.clipboard.writeText(text);
    toast('ok', what + ' copied');
  } catch (e) {
    // clipboard API blocked (file:// in some browsers): hidden-textarea fallback
    try {
      const ta = document.createElement('textarea');
      ta.value = text; ta.style.position = 'fixed'; ta.style.opacity = '0';
      document.body.appendChild(ta); ta.select();
      document.execCommand('copy'); ta.remove();
      toast('ok', what + ' copied');
    } catch (e2) { toast('err', 'Clipboard blocked - use Copy inside Text I/O'); }
  }
}

function modalNewGame() {
  const lvls = Object.entries(LEVELS).map(([k, v]) =>
    `<button class="lvl ${S.level === k ? 'on' : ''}" data-lvl="${k}">
       <span class="ldot" style="background:${v.color}"></span><b>${v.label}</b><span>${v.desc}</span></button>`).join('');
  const sides = ['First', 'Second', 'Random'];
  const clocks = ['none', '5+0', '5+3', '10+0'];
  openModal(`<h3>NEW GAME <span class="x" data-close>&#10005;</span></h3>
    <div class="levels">${lvls}</div>
    <div class="row"><label>Play as</label><div class="seg" id="segSide">
      ${sides.map((s, i) => `<button data-side="${i}" class="${(S.side === i % 2 && i < 2) || (i === 2 && S.side > 1) ? 'on' : ''}">${s}</button>`).join('')}
    </div></div>
    <div class="row"><label>Clock</label><div class="seg" id="segClock">
      ${clocks.map(c => `<button data-clock="${c}" class="${S.clockMode === c ? 'on' : ''}">${c === 'none' ? 'None' : c}</button>`).join('')}
    </div></div>
    <div class="row"><label>Board</label><div class="swatches">
      ${BOARD_THEMES.map(t => `<button class="swatch ${S.board === t ? 'on' : ''}" data-board="${t}"
        style="background:var(--cell-a)" title="${t}">&#9823;</button>`).join('')}
    </div></div>
    <button class="btn gold" id="btnStart" style="width:100%;margin-top:10px;padding:13px">START GAME</button>`);
  $('modalBox').querySelectorAll('[data-lvl]').forEach(b => b.onclick = () => {
    setLevelMidGame(b.dataset.lvl);
    $('modalBox').querySelectorAll('[data-lvl]').forEach(x => x.classList.toggle('on', x.dataset.lvl === S.level));
  });
  $('modalBox').querySelectorAll('#segSide button').forEach(b => b.onclick = () => {
    $('modalBox').querySelectorAll('#segSide button').forEach(x => x.classList.remove('on'));
    b.classList.add('on');
    S.side = b.dataset.side === '2' ? Math.floor(Math.random() * 2) : +b.dataset.side; saveSettings();
  });
  $('modalBox').querySelectorAll('#segClock button').forEach(b => b.onclick = () => {
    $('modalBox').querySelectorAll('#segClock button').forEach(x => x.classList.remove('on'));
    b.classList.add('on'); setClockFromLabel(b.dataset.clock); saveSettings();
  });
  $('modalBox').querySelectorAll('.swatch').forEach(b => b.onclick = () => {
    S.board = b.dataset.board; saveSettings();
    document.documentElement.dataset.board = S.board;
    $('modalBox').querySelectorAll('.swatch').forEach(x => x.classList.toggle('on', x.dataset.board === S.board));
    B.themeDirty = true; B.render();
  });
  $('btnStart').onclick = () => { closeModal(); newGame(); };
}

function setClockFromLabel(l) {
  if (l === 'none') { S.clockMode = 'none'; return; }
  S.clockMode = l;
  const [m, inc] = l.split('+');
  S.baseMin = +m; S.incSec = +inc;
}

function applyLevelChip() {
  const lv = curLevel();
  $('lvlName').textContent = lv.label;
  const dot = $('lvlChip').querySelector('.dot');
  if (dot) dot.style.background = lv.color;
}
// The settings surface is tabbed: four short groups instead of one long form.
// Each tab carries a one-line summary, so the reader knows what is inside
// before clicking. Every control keeps its id, so the wiring below is shared.
let settingsTab = 'look';
const SETTINGS_TABS = [
  ['look', 'Appearance'],
  ['board', 'Board'],
  ['sound', 'Sound'],
  ['play', 'Play & data'],
];
const SETTINGS_TAB_HINT = {
  look: 'Theme, colours and motion.',
  board: 'Board look, marks and overlays.',
  sound: 'Volume, sound pack and vibration.',
  play: 'Input feel, engine worker and saved data.',
};
function modalSettings(tab) {
  if (tab) settingsTab = tab;
  const seg = (key, opts, labels) =>
    `<div class="seg" data-set="${key}">${opts.map((o, i) =>
      `<button data-v="${o}" class="${String(S[key]) === String(o) ? 'on' : ''}">${labels[i]}</button>`).join('')}</div>`;
  const presetNames = [['classic', 'Classic'], ['premiumDark', 'Premium Dark'], ['highContrast', 'High Contrast'], ['minimal', 'Minimal']];
  const cards = {
    look: `
      <div class="card"><h4>PRESET</h4>
        <div class="row"><div class="seg" id="presetSeg">
          ${presetNames.map(([k, l]) => `<button data-p="${k}" class="${S.preset === k ? 'on' : ''}">${l}</button>`).join('')}
          <button class="${S.preset === 'custom' ? 'on' : ''}" disabled>Custom</button>
        </div></div>
      </div>
      <div class="card"><h4>THEME</h4>
        <div class="row"><label>Interface</label>${seg('ui', ['dark', 'light', 'auto'], ['Dark', 'Light', 'Auto'])}</div>
        <div class="row"><label>Accent</label><div class="swatches" id="setAccents">
          ${Object.keys(ACCENTS).map(a => `<button class="swatch ${S.accent === a ? 'on' : ''}" data-a="${a}" style="background:${ACCENTS[a][0]}"></button>`).join('')}
          <input type="color" id="accentPick" value="#c8a84b" style="width:34px;height:26px;background:none;border:none" title="Custom accent">
        </div></div>
        <div class="row"><label>Text size</label>${seg('fs', [1, 1.12, 1.25], ['Normal', 'Large', 'Larger'])}</div>
        <details class="advSettings"><summary>Advanced appearance</summary><div class="advBody">
          <div class="row"><label>Interface font</label>${seg('uiFont', ['modern', 'technical'], ['Modern', 'Technical'])}</div>
          <div class="row"><label>Density</label>${seg('density', ['comfortable', 'compact'], ['Comfortable', 'Compact'])}</div>
          <div class="row"><label>Animations</label>${seg('anim', ['full', 'reduced', 'off'], ['Full', 'Reduced', 'Off'])}</div>
          <div class="row"><label>Anim speed</label>${seg('animSpeed', [0.5, 1, 1.5], ['0.5x', '1x', '1.5x'])}</div>
        </div></details>
      </div>`,
    board: `
      <div class="card"><h4>SURFACE</h4>
        <div class="row"><label>Board</label><div class="swatches" id="setBoards">
          ${BOARD_THEMES.map(t => `<button class="swatch ${S.board === t ? 'on' : ''}" data-b="${t}">
            <canvas width="30" height="24" data-mini="${t}"></canvas></button>`).join('')}</div></div>
        <div class="row"><label>Pawn style</label><div class="swatches" id="setPawns">
          ${PAWN_STYLES.map(t => `<button class="swatch ${S.pawn === t ? 'on' : ''}" data-b="${t}" title="${t}">&#9823;</button>`).join('')}</div></div>
        <details class="advSettings"><summary>Advanced board appearance</summary><div class="advBody">
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
      </div>`,
    sound: `
      <div class="card"><h4>SOUND</h4>
        <div class="row"><label>Sound</label>${seg('sound', [true, false], ['On', 'Off'])}</div>
        <div class="row"><label>Pack</label>
          <select class="sel" id="packSel">${Object.keys(SOUND_PACKS).map(p => `<option value="${p}" ${S.soundPack === p ? 'selected' : ''}>${p}</option>`).join('')}</select>
          <button class="btn" id="packTest" style="padding:6px 10px">Test</button></div>
        <div class="row"><label>Volume</label><input type="range" id="volSlider" min="0" max="100" value="${Math.round(S.volume * 100)}" style="flex:1.2"></div>
        <div class="row"><label>Events</label><div class="seg" style="flex-wrap:wrap" id="evSeg">
          ${Object.keys(S.soundEvents).map(k => `<button data-ev="${k}" class="${S.soundEvents[k] ? 'on' : ''}" style="font-size:var(--fs-xs);padding:5px 7px">${k}</button>`).join('')}</div></div>
      </div>
      <div class="card"><h4>HAPTICS</h4>
        <div class="row"><label>Vibration</label>${seg('haptics', ['full', 'light', 'off'],
          navigator.vibrate ? ['Full', 'Light', 'Off'] : ['N/A', 'N/A', 'Off'])}</div>
      </div>`,
    play: `
      <div class="card"><h4>INPUT FEEL</h4>
        <div class="row"><label>Confirm walls</label>${seg('confirmWalls', [null, true, false], ['Auto', 'On', 'Off'])}</div>
        <div class="row"><label>Touch offset</label>${seg('touchOffset', [null, 'small', 'large', 'off'], ['Auto', 'Small', 'Large', 'Off'])}</div>
        <div class="row"><label>Sticky wall arm</label>${seg('stickyArm', [true, false], ['On', 'Off'])}</div>
        <div class="row"><label>Handedness</label>${seg('handedness', ['right', 'left', 'auto'], ['Right', 'Left', 'Auto'])}</div>
      </div>
      <div class="card"><h4>ENGINE &amp; DATA</h4>
        <div class="row"><label>Worker analysis</label>${seg('worker', [true, false], ['On', 'Off'])}</div>
        <div class="row"><label>Autosave game</label>${seg('autosave', [true, false], ['On', 'Off'])}</div>
        <div class="row" style="gap:6px;flex-wrap:wrap">
          <button class="btn" id="btnSetExport">Export settings</button>
          <button class="btn" id="btnSetImport">Import settings</button>
          <button class="btn danger" id="btnResetAll">Reset all</button>
        </div>
      </div>`,
  };
  openModal(`<h3>SETTINGS <span class="x" data-close>&#10005;</span></h3>
    <div class="mTabs">${SETTINGS_TABS.map(([k, l]) =>
      `<button data-mtab="${k}" class="${settingsTab === k ? 'on' : ''}">${l}</button>`).join('')}</div>
    <p class="mHint">${SETTINGS_TAB_HINT[settingsTab] || ''}</p>
    ${cards[settingsTab] || ''}`);
  $('modalBox').querySelectorAll('[data-mtab]').forEach(b =>
    b.onclick = () => modalSettings(b.dataset.mtab));
  // mini live previews for board themes: a real QBoard at tiny scale
  $('modalBox').querySelectorAll('canvas[data-mini]').forEach(cv => {
    const theme = cv.dataset.mini;
    const saved = document.documentElement.dataset.board;
    document.documentElement.dataset.board = theme;
    const mb = new QBoard(cv, { fixedSide: 44 });
    mb.flipped = false;
    mb.pawn = [mb.engPawnToDisp(40), mb.engPawnToDisp(4)];
    mb.wallH[3 * 8 + 4] = 1;   // one wall so the preview shows beams too
    mb.turn = -1;
    mb.themeDirty = true; mb.fit();
    document.documentElement.dataset.board = saved;
  });
  wireSettingControls();
}
function setOpt(key, val) {
  S[key] = val;
  // Schema-v4 style controls and the legacy boolean stay bidirectionally
  // compatible so old stored settings/tests/integrations keep their meaning.
  if (key === 'lastMoveStyle') S.lastMove = val !== 'off';
  if (key === 'lastMove') S.lastMoveStyle = val ? 'subtle' : 'off';
  S.preset = 'custom';
  saveSettings(); applySettings();
}
function wireSettingControls() {
  const box = $('modalBox');
  box.querySelectorAll('[data-set]').forEach(sg => {
    sg.querySelectorAll('button').forEach(b => b.onclick = () => {
      let v = b.dataset.v;
      if (v === 'true') v = true; else if (v === 'false') v = false; else if (v === 'null') v = null;
      else if (!isNaN(parseFloat(v)) && /^-?[\d.]+$/.test(v)) v = parseFloat(v);
      setOpt(sg.dataset.set, v);
      sg.querySelectorAll('button').forEach(x => x.classList.toggle('on', x === b));
    });
  });
  box.querySelectorAll('#presetSeg button[data-p]').forEach(b => b.onclick = () => {
    applyPreset(b.dataset.p); modalSettings();   // re-render with new values
  });
  box.querySelectorAll('#setBoards .swatch').forEach(b => b.onclick = () => {
    setOpt('board', b.dataset.b);
    box.querySelectorAll('#setBoards .swatch').forEach(x => x.classList.toggle('on', x === b));
  });
  box.querySelectorAll('#setPawns .swatch').forEach(b => b.onclick = () => {
    setOpt('pawn', b.dataset.b);
    box.querySelectorAll('#setPawns .swatch').forEach(x => x.classList.toggle('on', x === b));
  });
  box.querySelectorAll('#setAccents .swatch').forEach(b => b.onclick = () => {
    setOpt('accent', b.dataset.a);
    box.querySelectorAll('#setAccents .swatch').forEach(x => x.classList.toggle('on', x === b));
  });
  // Controls live on one tab each, so every lookup below may be empty
  // depending on the tab that is open.
  const accentPick = $('accentPick');
  if (accentPick) accentPick.oninput = e => { setOpt('accent', e.target.value); };
  const packSel = $('packSel');
  if (packSel) packSel.onchange = e => { setOpt('soundPack', e.target.value); sound('wall'); };
  const packTest = $('packTest');
  if (packTest) packTest.onclick = () => { acArmed = true; sound('wall'); };
  const volSlider = $('volSlider');
  if (volSlider) {
    volSlider.onchange = e => { setOpt('volume', +e.target.value / 100); };
    volSlider.oninput = e => { S.volume = +e.target.value / 100; };
  }
  box.querySelectorAll('#evSeg button').forEach(b => b.onclick = () => {
    const k = b.dataset.ev;
    S.soundEvents[k] = !S.soundEvents[k];
    saveSettings();
    b.classList.toggle('on', S.soundEvents[k]);
  });
  const exp = $('btnSetExport');
  if (exp) exp.onclick = () => downloadText('zq-settings.json', JSON.stringify(S, null, 2), 'application/json');
  const imp = $('btnSetImport');
  if (imp) imp.onclick = () => pickFile('.json', txt => {
    try {
      const parsed = JSON.parse(txt);
      localStorage.setItem('zq.settings', JSON.stringify({ ...DEFAULTS, ...parsed }));
      loadSettings(); applySettings(); closeModal();
      toast('ok', 'Settings imported');
    } catch (e) { toast('err', 'Invalid settings file'); }
  });
  const reset = $('btnResetAll');
  if (reset) reset.onclick = () => confirmModal('Reset all settings?', 'This restores every option to its default.',
    () => { localStorage.removeItem('zq.settings'); S = JSON.parse(JSON.stringify(DEFAULTS)); applySettings(); modalSettings(); toast('ok', 'Settings reset'); });
}

function applySettings() {
  const h = document.documentElement, ds = h.dataset;
  ds.ui = S.ui === 'auto' ? (matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark') : S.ui;
  ds.board = S.board; ds.pawn = S.pawn;
  ds.coords = S.coords;
  ds.uiFont = S.uiFont || 'modern';
  ds.boardTexture = S.boardTexture || 'subtle'; ds.boardContrast = S.boardContrast || 'standard';
  ds.wallProfile = S.wallProfile || 'standard'; ds.wallPreview = S.wallPreview || 'normal';
  ds.goalRows = S.goalRows || 'subtle'; ds.moveMarkers = S.moveMarkers || 'ring';
  ds.lastMoveStyle = S.lastMoveStyle || 'subtle';
  S.lastMove = ds.lastMoveStyle !== 'off';
  ds.frame = S.frame; ds.wallFinish = S.wallFinish; ds.cellSep = S.cellSep;
  ds.boardScale = String(S.boardScale);
  ds.pawnShadow = S.pawnShadow; ds.pawnSize = S.pawnSize;
  ds.distinctShapes = S.distinctShapes ? '1' : '0';
  ds.density = S.density; ds.anim = S.anim === 'reduced' ? 'reduced' : '';
  if (S.anim === 'off') h.setAttribute('data-anim', 'off'); else h.removeAttribute('data-anim');
  h.style.fontSize = (S.fs * 100) + '%';
  h.style.setProperty('--dur', Math.round(180 / (S.animSpeed || 1)) + 'ms');
  h.style.setProperty('--dur-fast', Math.round(110 / (S.animSpeed || 1)) + 'ms');
  h.style.setProperty('--dur-slow', Math.round(320 / (S.animSpeed || 1)) + 'ms');
  const hand = S.handedness === 'auto' ? 'right' : S.handedness;
  ds.handed = hand;
  applyAccent();
  applyLevelChip();
  $('evalStrip').style.display = S.evalBar ? '' : 'none';
  if (B) {
    B.themeDirty = true;
    B.fit(); buildLegalSets();
    B.lastMove = S.lastMove ? plyToLastMove(W.cursor() - 1) : null;
    togglePaths();   // rebuild (or clear) the path-hints overlay right away
    refreshHud(); B.render();
  }
}
// auto UI theme follows the OS live; attached ONCE (applySettings runs often)
matchMedia('(prefers-color-scheme: light)')
  .addEventListener?.('change', () => { if (S.ui === 'auto') applySettings(); });

// ===================== 11b. header menu (plan section 3.1) =============
// Entries of length one are group headers; they render as quiet labels and
// keep related actions together, so the list scans at a glance.
const MENU_ITEMS = [
  ['#GAME'],
  ['New game', () => { modalNewGame(); }],
  ['Flip board', doFlip],
  ['Resign', resignConfirm],
  ['#SHARE & FILES'],
  ['Copy link', () => copyText(shareLink(), 'Link')],
  ['Copy QFEN', () => copyText(W.qfenExportStr(), 'QFEN')],
  ['Paste QFEN', () => openTextIO('qfen')],
  ['Copy game', () => copyText(qgnExport(), 'Game text')],
  ['Paste game', () => openTextIO('qgn')],
  ['Open .qgn file', () => pickFile('.qgn,.txt,.qfen', t => routeImport(t))],
  ['Save .qgn file', () => downloadText(gameFileName(), qgnExport(), 'application/x-zquoridor-game')],
  ['#EXTRAS'],
  ['Recent games', showRecentGames],
  ['Export image', exportImageModal],
];
function buildMenu() {
  const drop = $('menuDrop');
  drop.innerHTML = MENU_ITEMS.map(it => {
    if (it === null) return '<hr>';
    if (it.length === 1) return `<div class="menuHead">${it[0]}</div>`;
    return `<button class="${it[0] === 'Resign' ? 'danger' : ''}" role="menuitem">${it[0]}</button>`;
  }).join('');
  const btns = drop.querySelectorAll('button');
  let bi = 0;
  MENU_ITEMS.forEach(it => {
    if (it && it.length > 1) { btns[bi].onclick = () => { closeMenu(); it[1](); }; bi++; }
  });
}
function closeMenu() { $('menuDrop').classList.remove('open'); }
$('btnMenu').onclick = ev => {
  ev.stopPropagation();
  const d = $('menuDrop');
  if (!d.classList.contains('open')) buildMenu();
  d.classList.toggle('open');
};
document.addEventListener('click', ev => {
  if (!$('menuDrop').contains(ev.target)) closeMenu();
});
function resignConfirm() {
  if (gameOver) { toast('info', 'No game in progress'); return; }
  confirmModal('RESIGN?', 'The current game will be recorded as a loss.', () => {
    gameOver = true; engineThinking = false;
    pushRecent('resign');
    markResult(1 - humanSide);
    setStatus('You resigned');
    toast('err', 'You resigned');
    sound('loss'); haptic([60]);
    refreshHud();
  });
}

// ===================== 11c. QGN + import/export (plan section 16) =======
// QGN is PGN-shaped: headers plus numbered move tokens. Annotations come
// from the blunder check; evals ride inside {[%ev ...]} comments so an
// exported game rebuilds the move log and eval graph identically.
function resultString() {
  const w = W.winner();
  if (w === 0) return '1-0';
  if (w === 1) return '0-1';
  if (W.isDraw()) return '1/2-1/2';
  return '*';
}
function qgnExport() {
  const today = new Date();
  const date = `${today.getFullYear()}.${String(today.getMonth() + 1).padStart(2, '0')}.${String(today.getDate()).padStart(2, '0')}`;
  const startQfen = 'e5 e5 10 10 - 0';   // canonical initial position
  const lines = [];
  lines.push('[Event "Casual game"]');
  lines.push('[Site "Zquoridor Web"]');
  lines.push(`[Date "${date}"]`);
  lines.push(`[Player0 "Player 0"]`);
  lines.push(`[Player1 "Zquoridor ${curLevel().label}"]`);
  lines.push(`[Result "${resultString()}"]`);
  if (S.clockMode !== 'none') lines.push(`[TimeControl "${S.baseMin}+${S.incSec}"]`);
  lines.push(`[Walls "10"]`);
  // Only include a QFEN header when the game did not start from the initial
  // position (editor-applied or imported roots).
  if (g_startedFromCustom()) lines.push(`[QFEN "${g_rootQfen}"]`);
  lines.push(`[Engine "Zquoridor WASM"]`);
  if (Object.keys(AN.annots).length) lines.push(`[Annotator "Zquoridor blunder check"]`);
  lines.push('');
  let movetext = '';
  for (let i = 0; i < W.plyCount(); i++) {
    if (i % 2 === 0) movetext += `${i / 2 + 1}. `;
    movetext += plyNotation(i);
    const ann = AN.annots[i] ? AN.annots[i].sym : '';
    if (ann) movetext += ann;
    const sc = AN.scores[i];
    if (sc != null) movetext += ` {[%ev ${fmtScore(absScoreAt(sc, i))}]}`;
    movetext += i % 2 === 0 ? ' ' : '\n';
  }
  lines.push(movetext.trim() || '*');
  lines.push(resultString());
  return lines.join('\n');
}
function gameFileName() {
  const d = new Date();
  return `zquoridor-${d.getFullYear()}${String(d.getMonth() + 1).padStart(2, '0')}${String(d.getDate()).padStart(2, '0')}.qgn`;
}
function shareLink() {
  return location.origin + location.pathname +
    (W.plyCount() ? '#qgn=' + btoa(unescape(encodeURIComponent(qgnExport()))) :
      '#qfen=' + encodeURIComponent(W.qfenExportStr()));
}

// ---- dialect normalization (plan section 16.3) --------------------------
// Normalizes one move token to the engine's expectation and returns
// {kind:'pawn'|'wall', o,r,c} or throws {msg}.
function parseMoveToken(tok) {
  let t = tok.replace(/[.,;!?]+$/g, '');
  if (/^[a-i][1-9][\\/]?$/.test(t)) {                    // pawn e2, jump e5/
    return { kind: 'pawn', cell: cellIdxFromAlg(t.slice(0, 2)) };
  }
  // orientation-first: ha3 / Hc6 / V f3
  let m = /^([hv])\s*([a-i])([1-8])$/i.exec(t.replace(/\s+/g, ''));
  if (m) return wallTok(m[2], m[3], m[1]);
  // orientation-last: c6h / Vf3 already handled by wallTok below
  m = /^([a-i])([1-8])\s*([hv])$/i.exec(t);
  if (m) return wallTok(m[1], m[2], m[3]);
  // coordinate-pair form: c6-d6 -> H wall whose span starts at file c,
  // sitting on the row boundary of rank 6 (slot row = rank-1)
  m = /^([a-i])([1-9])-([a-i])([1-9])(?:\/([a-i])([1-9])-([a-i])([1-9]))?$/.exec(t);
  if (m) {
    const f1 = cellCol(m[1]), f2 = cellCol(m[3]);
    const rk = parseInt(m[2], 10);
    if (Math.abs(f1 - f2) !== 1 || rk < 2 || rk > 8)
      throw { msg: `"${tok}" - pair cells must be adjacent files on ranks 2-8` };
    return { kind: 'wall', o: 0, r: rk - 1, c: Math.min(f1, f2) };
  }
  throw { msg: `"${tok}" - not a recognized move token` };
}
function wallTok(fch, rank, orient) {
  const o = orient.toLowerCase() === 'h' ? 0 : 1;
  const r = parseInt(rank, 10) - 1;
  const c = 'abcdefghi'.indexOf(fch.toLowerCase());
  return { kind: 'wall', o, r: r - 0 + (o === 0 ? 0 : 0), c };
}
function cellIdxFromAlg(a) { return (parseInt(a[1], 10) - 1) * 9 + ('abcdefghi'.indexOf(a[0])); }
function cellCol(fch) { return 'abcdefghi'.indexOf(fch.toLowerCase()); }

// Routes raw text to the right importer by shape (plan 16.3): a leading
// '[' header or numbered moves means QGN; a leading pair of pawn cells plus
// numeric wall-count fields means QFEN.
function routeImport(text, opts = {}) {
  const t = text.trim();
  if (!t) { toast('warn', 'Nothing to import'); return false; }
  const looksQgn = /^\[Event\b/im.test(t) || /^\[\w+\s+"/m.test(t);
  const looksQfen = /^[a-i][1-9]\s+[a-i][1-9]\s+\d{1,2}\s+\d{1,2}(\s|$)/m.test(t);
  if (looksQgn && !opts.forceQfen && !(looksQfen && /^\[\w/.test(t) === false)) {
    return importQGN(t);
  }
  return importQFENIntoGame(t);
}

// QFEN into the live game (fresh root), with precise diagnostics.
function importQFENIntoGame(qfen, silentFail) {
  const code = W.qfenImportStr(qfen);
  if (code !== 0) {
    if (!silentFail) {
      toast('err', 'QFEN rejected: ' + (W.lastErrStr() || 'invalid'));
      openTextIO('qfen', qfen, W.lastErrStr());
    }
    return false;
  }
  commitScratchAsRoot(qfen);
  return true;
}
function commitScratchAsRoot(rootQfen) {
  W.editApply();                     // scratch -> live game (validated)
  gameOver = false; engineThinking = false;
  AN.scores = {}; AN.annots = {}; AN.curDepth = 1;
  g_startedFromCustomFlag = true;
  g_rootQfen = rootQfen || W.qfenExportStr();
  clearGhost();
  humanSide = W.turn();              // play as the side to move
  syncAll();
  // a loaded position may already be won: announce it instead of a move
  if (!checkEnd()) setStatus('Position loaded - your move');
  saveSettings();
}

// Full game import: headers optional, moves applied one by one through the
// legality-checked C surface. Returns true on full success.
function importQGN(text) {
  const diag = [];
  // extract header QFEN if present
  const qf = /\[QFEN\s+"([^"]+)"\]/i.exec(text);
  // strip header lines, then comments, before tokenizing moves
  const noHeaders = text.replace(/^\s*\[[^\]]*\]\s*$/gm, ' ');
  const noComments = noHeaders.replace(/\{[^}]*\}/g, ' ');
  const tokens = noComments.split(/[\s\n]+/)
    .map(s => s.replace(/^\d+\.+/, ''))          // 1. / 1) numbering
    .filter(s => s && !/^\d+\)?$/.test(s))
    .filter(s => !/^(1-0|0-1|1\/2-1\/2|\*)$/.test(s))
    .filter(s => !/^[?!]+$/.test(s))
    .filter(s => !/^\$/.test(s));
  // A QFEN pasted where a game was expected arrives here too; detect and reroute
  if (!qf && tokens.length >= 6 && /^[a-i][1-9]$/.test(tokens[0]) &&
      /^[01]$/.test(tokens[tokens.length - 1] === '*' ? '' : tokens[5] || '')) {
    return importQFENIntoGame(text);
  }
  let base = qf ? qf[1] : null;
  if (base) {
    const code = W.qfenImportStr(base);
    if (code !== 0) { importFailed('header QFEN: ' + (W.lastErrStr() || 'invalid'), text); return false; }
    W.editApply();
    g_startedFromCustomFlag = true;
    g_rootQfen = base;   // worker replays need the true root
  } else {
    W.newGame();
    g_startedFromCustomFlag = false; g_rootQfen = '-';
  }
  gameOver = false; engineThinking = false;
  AN.scores = {}; AN.annots = {};
  let applied = 0;
  for (const tok of tokens) {
    let mv;
    try { mv = parseMoveToken(tok); }
    catch (e) { return importFailed(`token ${applied + 1}: ${e.msg}`, text, applied); }
    let ok;
    if (mv.kind === 'pawn') ok = W.applyPawn(mv.cell);
    else {
      if (mv.r < 0 || mv.r > 7 || mv.c < 0 || mv.c > 7)
        return importFailed(`token ${applied + 1}: "${tok}" anchor outside the board`, text, applied);
      ok = W.applyWall(mv.o, mv.r, mv.c);
    }
    if (!ok) return importFailed(`token ${applied + 1}: "${tok}" is illegal here`, text, applied);
    applied++;
    if (W.winner() !== -1) break;
  }
  g_lastImportApplied = applied;
  humanSide = W.turn();
  clearGhost(); syncAll();
  // a finished game must announce its result, not invite a move
  if (!checkEnd()) {
    setStatus(applied < tokens.length
      ? `Loaded up to ply ${applied} of ${tokens.length}`
      : 'Game imported - your move');
  }
  toast('ok', `Imported ${applied} plies`);
  return true;
}
function importFailed(msg, text, appliedPlies) {
  g_lastImportApplied = appliedPlies || 0;
  toast('err', 'Import failed: ' + msg);
  openTextIO('qgn', text, msg, appliedPlies > 0);
  if (appliedPlies > 0) syncAll();
  return false;
}

// ---- Text I/O modal (plan section 5.10 #4) -----------------------------
function openTextIO(format, presetText, errMsg, allowPartial) {
  const fmt = format || 'qgn';
  const body = presetText != null ? presetText : (fmt === 'qfen' ? W.qfenExportStr() : qgnExport());
  openModal(`<h3>TEXT I/O <span class="x" data-close>&#10005;</span></h3>
    <div class="row"><label>Format</label><div class="seg" id="ioFmt">
      <button data-f="qgn" class="${fmt === 'qgn' ? 'on' : ''}">QGN</button>
      <button data-f="qfen" class="${fmt === 'qfen' ? 'on' : ''}">QFEN</button>
    </div></div>
    <textarea id="ioArea" class="ioArea ${errMsg ? 'err' : ''}" spellcheck="false"></textarea>
    <div id="ioDiag">${errMsg ? escapeHtml(errMsg) : ''}</div>
    <div class="row" style="gap:6px;flex-wrap:wrap;margin-top:8px">
      <button class="btn" id="ioCopy">Copy</button>
      <button class="btn" id="ioPaste">Paste</button>
      <button class="btn gold" id="ioLoad" style="flex:1">LOAD</button>
      <button class="btn" id="ioDownload">Download</button>
    </div>
    <div class="row" style="margin-top:6px" id="ioPartialRow" style="display:${allowPartial ? 'flex' : 'none'}">
      <button class="btn" id="ioPartial" style="flex:1">Import anyway (up to the error)</button>
    </div>`);
  const area = $('ioArea');
  area.value = body;
  $('modalBox').querySelectorAll('#ioFmt button').forEach(b => b.onclick = () => {
    const f = b.dataset.f;
    $('modalBox').querySelectorAll('#ioFmt button').forEach(x => x.classList.toggle('on', x === b));
    area.value = f === 'qfen' ? W.qfenExportStr() : qgnExport();
    area.classList.remove('err');
    $('ioDiag').textContent = '';
  });
  $('ioCopy').onclick = async () => {
    try { await navigator.clipboard.writeText(area.value); toast('ok', 'Copied'); }
    catch (e) { area.select(); document.execCommand('copy'); toast('ok', 'Copied'); }
  };
  $('ioPaste').onclick = async () => {
    try { area.value = await navigator.clipboard.readText(); area.classList.remove('err'); $('ioDiag').textContent = ''; }
    catch (e) { toast('err', 'Clipboard read blocked - paste manually'); area.focus(); }
  };
  $('ioDownload').onclick = () => downloadText(
    $('ioFmt').querySelector('.on').dataset.f === 'qfen' ? 'position.qfen' : gameFileName(),
    area.value);
  $('ioLoad').onclick = () => {
    const f = $('ioFmt').querySelector('.on').dataset.f;
    const txt = area.value;
    const ok = f === 'qfen' ? importQFENIntoGame(txt) : importQGN(txt);
    if (ok) closeModal();
  };
  $('ioPartial').onclick = () => {
    // The valid prefix is already on the board (importFailed applied it);
    // this button just acknowledges it and returns to the game.
    closeModal();
    syncAll();
    toast('info', `Loaded the valid prefix (${g_lastImportApplied} plies)`);
  };
}
function escapeHtml(s) {
  return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

// ---- drag & drop + boot-time hash load ---------------------------------
function wireDropTarget() {
  const zone = $('boardZone');
  const ov = $('dropOverlay');
  let depth = 0;
  addEventListener('dragenter', ev => { ev.preventDefault(); depth++; ov.classList.add('on'); });
  addEventListener('dragover', ev => ev.preventDefault());
  addEventListener('dragleave', () => { if (--depth <= 0) { depth = 0; ov.classList.remove('on'); } });
  addEventListener('drop', ev => {
    ev.preventDefault(); depth = 0; ov.classList.remove('on');
    const f = ev.dataTransfer.files && ev.dataTransfer.files[0];
    if (f) { f.text().then(t => routeImport(t)).catch(() => toast('err', 'Could not read file')); return; }
    const txt = ev.dataTransfer.getData('text/plain');
    if (txt) routeImport(txt);
  });
}
function bootHashLoad() {
  const mQ = /[#&]qfen=([^&]+)/.exec(location.hash);
  const mG = /[#&]qgn=([^&]+)/.exec(location.hash);
  try {
    if (mG) {
      const txt = decodeURIComponent(escape(atob(decodeURIComponent(mG[1]))));
      setTimeout(() => importQGN(txt), 300);
      return true;
    }
    if (mQ) {
      setTimeout(() => importQFENIntoGame(decodeURIComponent(mQ[1])), 300);
      return true;
    }
  } catch (e) { toast('err', 'Bad share link'); }
  return false;
}

// ---- autosave & recent games (plan section 16.6) -----------------------
let autosaveTimer = null;
let g_startedFromCustomFlag = false;
let g_rootQfen = '-';
let g_lastImportApplied = 0;
function g_startedFromCustom() { return g_startedFromCustomFlag; }
function scheduleAutosave() {
  if (!S.autosave || !W || !W.plyCount()) return;
  if (autosaveTimer) clearTimeout(autosaveTimer);
  autosaveTimer = setTimeout(() => {
    try { localStorage.setItem('zq.game', qgnExport()); } catch (e) {}
  }, 250);
}
function checkAutosaveOnBoot() {
  let saved = null;
  try { saved = localStorage.getItem('zq.game'); } catch (e) {}
  if (!saved) return;
  const finished = /\[Result "(1-0|0-1|1\/2-1\/2)"\]/.test(saved);
  if (finished) return;
  const chip = $('resumeChip');
  chip.style.display = '';
  chip.textContent = 'Resume game?';
  const hide = () => { chip.style.display = 'none'; };
  chip.onclick = () => { hide(); importQGN(saved); };
  setTimeout(hide, 10000);
}
function recentList() {
  try { return JSON.parse(localStorage.getItem('zq.recent') || '[]'); }
  catch (e) { return []; }
}
function pushRecent(how) {
  if (!W || !W.plyCount()) return;
  const list = recentList();
  list.unshift({ when: Date.now(), how, qgn: qgnExport() });
  while (list.length > 20) list.pop();
  try { localStorage.setItem('zq.recent', JSON.stringify(list)); }
  catch (e) { /* quota: session only */ }
}
function showRecentGames() {
  const list = recentList();
  if (!list.length) { toast('info', 'No recent games yet'); return; }
  const rows = list.map((g, i) => {
    const res = (/\[Result "([^"]+)"\]/.exec(g.qgn) || [, '*'])[1];
    const d = new Date(g.when);
    const when = `${d.getDate()}/${d.getMonth() + 1}`;
    const nMoves = ((g.qgn.match(/\b\d+\./g) || []).length);
    return `<div class="mlRow" style="align-items:center">
      <span style="flex:1;font-size:var(--fs-xs)">#${i + 1} ${when} ${res} (${nMoves}m)</span>
      <button class="btn" data-rload="${i}" style="padding:4px 8px;font-size:var(--fs-xs)">Load</button>
      <button class="btn" data-rcopy="${i}" style="padding:4px 8px;font-size:var(--fs-xs)">Copy</button>
      <button class="btn danger" data-rdel="${i}" style="padding:4px 8px;font-size:var(--fs-xs)">&#10005;</button>
    </div>`;
  }).join('');
  openModal(`<h3>RECENT GAMES <span class="x" data-close>&#10005;</span></h3>${rows}`);
  $('modalBox').querySelectorAll('[data-rload]').forEach(b => b.onclick = () => {
    importQGN(recentList()[+b.dataset.rload].qgn); closeModal();
  });
  $('modalBox').querySelectorAll('[data-rcopy]').forEach(b => b.onclick = () =>
    copyText(recentList()[+b.dataset.rcopy].qgn, 'Game'));
  $('modalBox').querySelectorAll('[data-rdel]').forEach(b => b.onclick = () => {
    const l = recentList();
    l.splice(+b.dataset.rdel, 1);
    try { localStorage.setItem('zq.recent', JSON.stringify(l)); } catch (e) {}
    if (l.length) showRecentGames();
    else { closeModal(); toast('info', 'Recent games cleared'); }
  });
}

// ---- image export modal (plan section 16.5) ----------------------------
function exportImageModal() {
  openModal(`<h3>EXPORT IMAGE <span class="x" data-close>&#10005;</span></h3>
    <div class="row"><label>Transparent background</label><input type="checkbox" id="exTrans"></div>
    <div class="row"><label>Include coordinates</label><input type="checkbox" id="exCoords" checked></div>
    <div class="row"><label>Footer with names</label><input type="checkbox" id="exFooter" checked></div>
    <div class="row" style="gap:8px;margin-top:10px">
      <button class="btn gold" id="exPng" style="flex:1">PNG</button>
      <button class="btn" id="exSvg" style="flex:1">SVG</button>
    </div>`);
  const opts = () => ({
    transparent: $('exTrans').checked,
    coords: $('exCoords').checked,
    footer: $('exFooter').checked ? { name0: 'Player 0', name1: 'Zquoridor ' + curLevel().label } : null,
  });
  $('exPng').onclick = () => {
    const cv = B.renderExport(opts());
    cv.toBlob(bl => {
      const url = URL.createObjectURL(bl);
      const a = document.createElement('a');
      a.href = url; a.download = 'zquoridor-board.png';
      document.body.appendChild(a); a.click(); a.remove();
      setTimeout(() => URL.revokeObjectURL(url), 4000);
    }, 'image/png');
    closeModal(); toast('ok', 'PNG exported');
  };
  $('exSvg').onclick = () => {
    downloadText('zquoridor-board.svg', B.toSVG(opts()), 'image/svg+xml');
    closeModal();
  };
}

// ===================== 11d. analysis worker facade (plan section 12) ====
// The worker owns a second WASM instance; requests replay the recorded line
// onto its scratch and run the same qr_analyze. Any failure (no Worker,
// spawn error, worker error, setting off) falls back to main-thread slicing
// -- plan principle "degrade silently".
const ANW = {
  wk: null, ready: false, failed: false,
  pending: new Map(), nextId: 1,
  ok() {
    return S.worker && typeof Worker !== 'undefined' && !this.failed;
  },
  ensure() {
    if (this.wk || this.failed) return;
    try {
      this.wk = new Worker('worker.js');
      this.wk.onmessage = ev => {
        const m = ev.data;
        if (m.type === 'ready') { this.ready = true; return; }
        if (m.type === 'fatal') { this.failed = true; return; }
        const cb = this.pending.get(m.id);
        if (cb) { this.pending.delete(m.id); cb(m); }
      };
      this.wk.onerror = () => { this.failed = true; };
    } catch (e) { this.failed = true; }
  },
  // Asks the worker for the engine's own move. It replays the recorded line
  // into its own live game and runs qr_engine_move, which is the hybrid search
  // the main thread would run. qr_analyze cannot serve this: it is pure
  // alpha-beta on the scratch position and would change how the engine plays.
  bestMove(req, cb) {
    this.ensure();
    if (!this.ok() || !this.ready) return false;
    const id = this.nextId++;
    this.pending.set(id, res => cb(res && res.type !== 'error' ? res : null));
    this.wk.postMessage({ id, cmd: 'bestmove', moves: req.moves,
                          depth: req.depth, timeMs: req.timeMs });
    return true;
  },
  analyze(req, cb) {
    this.ensure();
    if (!this.ok() || !this.ready) return false;
    const id = this.nextId++;
    this.pending.set(id, res => {
      // a per-request error (bad root, replay hiccup) answers null once but
      // keeps the worker alive -- only onerror/fatal kill it permanently
      if (res.type === 'error') { cb(null); return; }
      cb(res);
    });
    // send the root QFEN only when the game started from a custom position
    this.wk.postMessage({ id, cmd: 'analyze',
      qfen: g_startedFromCustom() ? g_rootQfen : null,
      moves: req.moves, depth: req.depth, timeMs: req.timeMs, lines: req.lines });
    return true;
  },
};
function allPliesPacked() {
  const out = [];
  for (let i = 0; i < W.plyCount(); i++) out.push(packPly(i));
  return out;
}

// ===================== 13b. editor (plan section 5.9) ==================
// The editor works on the scratch position (P6 C surface). While the
// Editor pane is open the board renders the SCRATCH and pointer input goes
// to the palette tools; Apply commits a validated position as a new game.
let currentPane = 'playPane';
const ED = { tool: 'pawn0' };

function switchPane(pane) {
  currentPane = pane;
  document.querySelectorAll('.tab[data-pane]').forEach(x =>
    x.classList.toggle('on', x.dataset.pane === pane));
  document.querySelectorAll('.pane').forEach(p =>
    p.classList.toggle('on', p.id === pane));
  const mobile = !matchMedia('(min-width:900px)').matches;
  if (mobile) $('sidePanel').classList.toggle('open', pane !== 'playPane');
  if (pane === 'edPane') edSyncFromLive();
}
function setEditorTool(t) {
  ED.tool = t;
  for (const [id, k] of [['etPawn0', 'pawn0'], ['etPawn1', 'pawn1'],
                         ['etWallH', 'wallH'], ['etWallV', 'wallV'], ['etErase', 'erase']])
    $(id).classList.toggle('on', k === t);
}
function edSyncFromLive() {
  W.scratchFromLive();
  edRefresh();
  renderScratchToBoard();
}
function edRefresh() {
  $('edW0').textContent = W.scrWallsLeft(0);
  $('edW1').textContent = W.scrWallsLeft(1);
  const v = W.editValidity();
  const msgs = [];
  if (v & 1) msgs.push('Both pawns share one cell');
  if (v & 2) msgs.push('Player 0 has no path to goal');
  if (v & 4) msgs.push('Player 1 has no path to goal');
  if (v & 8) msgs.push('A pawn is already on its goal row');
  if (v & 16) msgs.push('More walls than the 20 that exist');
  const el = $('edValidity');
  el.classList.toggle('ok', v === 0);
  el.classList.toggle('bad', v !== 0);
  el.textContent = v === 0 ? 'Legal position' : msgs.join(' \u00b7 ');
  $('btnEdApply').disabled = v !== 0;
  $('btnEdApply').style.opacity = v !== 0 ? .45 : 1;
}
function renderScratchToBoard() {
  const pw = [W.scrPawn(0), W.scrPawn(1)];
  const wh = [], wv = [];
  for (let s = 0; s < 64; s++) { wh.push(W.scrWallHBit(s)); wv.push(W.scrWallVBit(s)); }
  B.flipped = false;   // editor works in absolute coordinates
  B.lastMove = null;
  B.dots = []; B.selected = -1;
  B.setData(pw, wh, wv, null);
  refreshHudFromScratch();
}
// The editor paints in absolute coordinates, so player 0 always sits in the
// bottom strip and player 1 in the top one.
function refreshHudFromScratch() {
  const wl = [W.scrWallsLeft(0), W.scrWallsLeft(1)];
  const d = [W.scrDist(0), W.scrDist(1)];
  for (const seat of [['bottom', 0], ['top', 1]]) {
    const el = strip(seat[0]), pl = seat[1];
    if (!el) continue;
    el.dataset.player = pl;
    const wn = el.querySelector('.wn');
    if (wn) wn.textContent = wl[pl];
    pips(el.querySelector('.pips'), wl[pl]);
    const dEl = el.querySelector('.dist');
    if (dEl && d[pl] >= 0) dEl.innerHTML = '<b>' + d[pl] + '</b>';
  }
}
function edBoardDown(ev) {
  ev.preventDefault();
  const pt = boardPoint(ev);
  const cell = B.pointToCell(pt.x, pt.y);
  const anchor = B.nearestAnchor(pt.x, pt.y);
  const eng = cell ? B.dispPawnToEng(cell.r, cell.c) : -1;
  let toastMsg = null;
  if (ED.tool === 'pawn0' || ED.tool === 'pawn1') {
    if (!cell) return;
    const pl = ED.tool === 'pawn0' ? 0 : 1;
    const rc = W.editSetPawn(pl, eng);
    if (rc === -3) toastMsg = 'That cell is occupied by the other pawn';
    else if (rc === -2) toastMsg = 'Cell outside the board';
  } else if (ED.tool === 'wallH' || ED.tool === 'wallV') {
    if (!anchor) return;
    const o = ED.tool === 'wallH' ? 0 : 1;
    const [, er, ec] = B.dispWallToEng(o, anchor.r, anchor.c);
    if (W.editSetWall(o, er, ec, 1) === -1) toastMsg = 'Wall slot conflicts with an existing wall';
  } else if (ED.tool === 'erase') {
    if (anchor) {
      for (const o of [0, 1]) {
        const [eo2, er2, ec2] = B.dispWallToEng(o, anchor.r, anchor.c);
        const bit = o === 0 ? W.scrWallHBit(er2 * 8 + ec2) : W.scrWallVBit(er2 * 8 + ec2);
        if (bit) { W.editSetWall(o, er2, ec2, 0); break; }
      }
    } else {
      // nothing under the pointer: give a wall back to the side to move
      const t = W.scrTurn();
      W.editSetWallsLeft(t, Math.min(10, W.scrWallsLeft(t) + 1));
    }
  }
  if (toastMsg) { toast('warn', toastMsg); shake(); }
  edRefresh();
  renderScratchToBoard();
}
function wireEditor() {
  setEditorTool('pawn0');
  for (const [id, k] of [['etPawn0', 'pawn0'], ['etPawn1', 'pawn1'],
                         ['etWallH', 'wallH'], ['etWallV', 'wallV'], ['etErase', 'erase']])
    $(id).onclick = () => setEditorTool(k);
  document.querySelectorAll('[data-st]').forEach(b => b.onclick = () => {
    const m = /^w([01])([+-])(\d+)$/.exec(b.dataset.st);
    if (!m) return;
    const pl = +m[1];
    const delta = m[2] === '+' ? +m[3] : -+m[3];
    W.editSetWallsLeft(pl, W.scrWallsLeft(pl) + delta);
    edRefresh(); renderScratchToBoard();
  });
  $('edTurn').querySelectorAll('button').forEach(b => b.onclick = () => {
    W.editSetTurn(+b.dataset.t);
    $('edTurn').querySelectorAll('button').forEach(x => x.classList.toggle('on', x === b));
    edRefresh(); renderScratchToBoard();
  });
  $('btnEdApply').onclick = () => {
    if (W.editValidity() !== 0) return;
    commitScratchAsRoot(W.qfenExportScratchStr());
    closeModal();
    toast('ok', 'Position applied - new game');
    setStatus(humanSide === 0 ? 'Your move' : 'Engine to move');
    startClock();
    if (W.turn() !== humanSide && !gameOver) {
      engineThinking = true; refreshHud(); scheduleEngineTurn(150);
    }
  };
  $('btnEdClear').onclick = () => {
    W.scratchReset();
    W.editSetWallsLeft(0, 10); W.editSetWallsLeft(1, 10); W.editSetTurn(0);
    edRefresh(); renderScratchToBoard();
  };
  $('btnEdCopy').onclick = () => copyText(W.qfenExportScratchStr(), 'QFEN');
  $('btnEdPaste').onclick = async () => {
    try {
      const txt = await navigator.clipboard.readText();
      const code = W.qfenImportStr(txt);
      if (code !== 0) { toast('err', 'QFEN rejected: ' + (W.lastErrStr() || '')); shake(); }
    } catch (e) { toast('err', 'Clipboard read blocked - use Text I/O'); }
    edRefresh(); renderScratchToBoard();
  };
}

// ===================== 12. keyboard ====================================
addEventListener('keydown', e => {
  const tag = (e.target && e.target.tagName) || '';
  if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return;
  if (e.key === 'Escape') {
    B.linePreview = null;
    clearGhost(); B.render(); closeModal(); return;
  }
  if (e.key === 'h' || e.key === 'H') { armWall(0); return; }
  if (e.key === 'v' || e.key === 'V') { armWall(1); return; }
  if (e.key === 'r' || e.key === 'R') {
    armedO = 1 - armedO;
    if (B.ghost) { B.ghost.o = armedO; B.render(); }
    if (kbWall) { kbWall.o = armedO; if (forcedO != null) forcedO = armedO; kbWallShow(); }
    return;
  }
  if (e.key === 'f' || e.key === 'F') { doFlip(); return; }
  if (e.key === 'Enter' || e.key === ' ') {
    if (wallState === 'PENDING') { commitGhost(B.ghost); return; }
    // An armed orientation plus a keyboard anchor is a complete wall.
    if (forcedO != null && kbWall && humanCanAct() &&
        legalWall[kbWall.o * 64 + kbWall.r * 8 + kbWall.c]) {
      e.preventDefault();
      commitGhost({ o: kbWall.o, r: kbWall.r, c: kbWall.c, state: 'ok' });
      return;
    }
    if (e.key === 'Enter') return;
  }
  // plan section 14: , . Home End navigation, A engine toggle
  if (e.key === ',') { navGo(W.cursor() - 1); return; }
  if (e.key === '.') { navGo(W.cursor() + 1); return; }
  if (e.key === 'Home') { navGo(0); return; }
  if (e.key === 'End') { navGo(W.plyCount()); return; }
  if (e.key === 'a' || e.key === 'A') { anToggle(); return; }
  // plan section 14: N new game, S cycle level, T takeback, ? help
  if (e.key === 'n' || e.key === 'N') { modalNewGame(); return; }
  if (e.key === 't' || e.key === 'T') { takeback(); return; }
  if (e.key === 's' || e.key === 'S') { cycleLevel(); return; }
  if (e.key === '?') { showKeyboardHelp(); return; }
  if (e.key === 'm' || e.key === 'M') {
    S.sound = !S.sound; saveSettings();
    toast('info', S.sound ? 'Sound on' : 'Sound off');
    return;
  }
  // plan section 7: arrow keys move the pawn; a straight jump continues the
  // arrow; Shift+Arrow picks a diagonal deflection.
  if (e.key.startsWith('Arrow')) {
    if (!atLiveEnd() || gameOver || engineThinking || W.turn() !== humanSide) return;
    // With an orientation armed the arrows belong to the wall, not the pawn.
    // Without this branch a keyboard-only player could arm a wall, watch the
    // preview appear, and have no way to move or commit it: the arrows walked
    // the pawn instead, so walls were unreachable without a pointer.
    if (forcedO != null && kbWall) {
      const d = { ArrowUp: [-1, 0], ArrowDown: [1, 0],
                  ArrowLeft: [0, -1], ArrowRight: [0, 1] }[e.key];
      if (d) {
        e.preventDefault();
        kbWall.r = Math.max(0, Math.min(7, kbWall.r + d[0]));
        kbWall.c = Math.max(0, Math.min(7, kbWall.c + d[1]));
        kbWallShow();
        return;
      }
    }
    const cur = B.engPawnToDisp(W.pawn(humanSide));
    const row = Math.floor(cur / 9), col = cur % 9;
    let targets = [];
    if (e.shiftKey) {
      const diag = { ArrowUp: [-10, -8], ArrowDown: [8, 10],
                     ArrowLeft: [-10, 8], ArrowRight: [-8, 10] }[e.key] || [];
      targets = diag.filter(dd => legalPawn.has(cur + dd));
    } else {
      const d = { ArrowUp: -9, ArrowDown: 9, ArrowLeft: -1, ArrowRight: 1 }[e.key];
      if (d != null) {
        const sameRowOk = Math.abs(d) !== 1 ||
          (d === -1 ? col > 0 : col < 8);
        if (sameRowOk && legalPawn.has(cur + d)) targets.push(cur + d);
        else {
          const jr = Math.floor((cur + 2 * d) / 9);
          const jc = (cur + 2 * d) % 9;
          const jumpRowOk = Math.abs(d) === 1 ? jr === row : true;
          const jumpColOk = Math.abs(d) !== 1 ? jc === col :
            (d === -1 ? jc >= 0 : jc <= 8);
          if (jumpRowOk && jumpColOk && legalPawn.has(cur + 2 * d))
            targets.push(cur + 2 * d);
        }
      }
    }
    if (targets.length) {
      const disp = targets[0];
      const eng = B.dispPawnToEng(Math.floor(disp / 9), disp % 9);
      if (W.applyPawn(eng)) {
        B.selected = -1; B.dots = [];
        const anim = B.animateMove ? B.animateMove(humanSide, cur, disp) : null;
        syncFromEngine();   // paint the move at once, as in playPawn
        sound('move'); haptic(10);
        afterHumanMove(anim);
      }
      e.preventDefault();
    }
  }
});
function cycleLevel() {
  const keys = Object.keys(LEVELS);
  const idx = keys.indexOf(S.level);
  setLevelMidGame(keys[(idx + 1) % keys.length]);
}
function setLevelMidGame(key) {
  const was = S.level;
  S.level = key; saveSettings(); applyLevelChip();
  if (W && W.plyCount() > 0 && key !== was) {
    levelMarks.push({ afterPly: W.plyCount(), label: LEVELS[key].label });
    renderMoveLog();
  }
  toast('info', 'Level: ' + LEVELS[key].label);
}
let levelMarks = [];
function showKeyboardHelp() {
  const rows = [
    ['Arrows', 'Move the pawn / wall ghost'],
    ['H / V', 'Arm horizontal / vertical wall'],
    ['R', 'Flip wall orientation'],
    ['Enter', 'Commit pending wall'],
    ['Esc', 'Cancel arm, preview or dialog'],
    [', . Home End', 'Previous / next ply, first / last'],
    ['F', 'Flip board'],
    ['P', 'Path hints'],
    ['M', 'Mute'],
    ['N', 'New game'],
    ['S', 'Cycle level'],
    ['T', 'Takeback'],
    ['A', 'Analysis engine on/off'],
    ['?', 'This help'],
  ];
  openModal(`<h3>KEYBOARD <span class="x" data-close>&#10005;</span></h3>
    ${rows.map(r => `<div class="kbdRow"><kbd>${r[0]}</kbd><span>${r[1]}</span></div>`).join('')}`);
}

// ===================== misc handlers ===================================
function doFlip() {
  S.flipped = !S.flipped;   // persists; syncFromEngine derives B.flipped
  saveSettings();
  syncFromEngine();
  B.render();
}
$('btnFlip').onclick = doFlip;
for (const id of ['btnTakeback', 'btnUndo']) { const el = $(id); if (el) el.onclick = takeback; }
$('btnHint').onclick = showHint;
// Wrap every modal opener in an arrow function. A direct assignment passes
// the MouseEvent as the first argument, and modalSettings(tab) treats any
// truthy first argument as a tab name. That made the first open of the
// settings modal render an empty body.
$('btnNew').onclick = () => modalNewGame();
$('lvlChip').onclick = () => modalNewGame();
$('btnSettings').onclick = () => modalSettings();
$('logo').onclick = () => openModal(`<h3>ABOUT <span class="x" data-close>&#10005;</span></h3>
  <p style="line-height:1.7;color:var(--txt2)">Zquoridor plays with an NNUE evaluation network
  (354 inputs, hybrid PUCT MCTS over alpha-beta) trained on self-play.
  Place walls to slow your opponent; reach the far row to win.</p>`);
document.querySelectorAll('.tab[data-pane]').forEach(t => t.onclick = () => switchPane(t.dataset.pane));
const bcvs = $('board');
bcvs.addEventListener('pointerdown', onBoardPointerDown);
bcvs.addEventListener('pointermove', onBoardPointerMove);
bcvs.addEventListener('pointerup', onBoardPointerUp);
bcvs.addEventListener('pointercancel', () => { clearGhost(); B.render(); });
bcvs.addEventListener('pointerleave', () => { if (B.setHover) { B.setHover(null); B.render(); } });
$('raceMeter').onclick = () => { S.paths = !S.paths; togglePaths(); saveSettings(); };
$('btnPaths').onclick = () => { S.paths = !S.paths; togglePaths(); saveSettings(); };
// Shortest paths for the overlay, in DISPLAY coordinates. The previous
// version walked greedily in engine coordinates and handed engine cells to a
// renderer that expects display cells, so both lines came out mirrored and
// crossed in the middle of the board.
function edgeOpen(a, b) {
  const ar = Math.floor(a / 9), ac = a % 9, br = Math.floor(b / 9), bc = b % 9;
  if (ar === br) {          // horizontal step, blocked by vertical walls
    const cmin = Math.min(ac, bc), r = ar;
    const bit1 = r > 0 ? W.wallVBit((r - 1) * 8 + cmin) : 0;
    const bit2 = r < 8 ? W.wallVBit(r * 8 + cmin) : 0;
    return !(bit1 || bit2);
  }
  const rmin = Math.min(ar, br), c = ac;   // vertical step, blocked by H walls
  const bit1 = c > 0 ? W.wallHBit(rmin * 8 + c - 1) : 0;
  const bit2 = c < 8 ? W.wallHBit(rmin * 8 + c) : 0;
  return !(bit1 || bit2);
}
function shortestPath(pl) {
  const start = W.pawn(pl), goalRow = pl === 0 ? 8 : 0;
  const prev = new Int16Array(81).fill(-1);
  const seen = new Uint8Array(81);
  const q = [start];
  seen[start] = 1;
  let end = -1;
  for (let i = 0; i < q.length && end < 0; i++) {
    const cur = q[i];
    if (Math.floor(cur / 9) === goalRow) { end = cur; break; }
    const r = Math.floor(cur / 9), c = cur % 9;
    const steps = [[-1, 0], [1, 0], [0, -1], [0, 1]];
    for (const st of steps) {
      const nr = r + st[0], nc = c + st[1];
      if (nr < 0 || nr > 8 || nc < 0 || nc > 8) continue;
      const nx = nr * 9 + nc;
      if (seen[nx] || !edgeOpen(cur, nx)) continue;
      seen[nx] = 1; prev[nx] = cur; q.push(nx);
    }
  }
  if (end < 0) return null;
  const cells = [];
  for (let cur = end; cur !== -1; cur = prev[cur]) cells.push(B.engPawnToDisp(cur));
  cells.reverse();
  return { cells, player: pl,
           color: getComputedStyle(document.documentElement).getPropertyValue('--p' + pl).trim() };
}
// Recomputed on every state change, so the lines can never go stale.
function recomputePaths() {
  if (!B) return;
  const data = S.paths ? [shortestPath(0), shortestPath(1)].filter(Boolean) : null;
  if (B.setPaths) B.setPaths(data); else B.paths = data;
}
function togglePaths() {
  recomputePaths();
  const bp = $('btnPaths');
  if (bp) bp.classList.toggle('on', !!S.paths);   // the icon shows its own state
  B.render();
}

// ===================== 10b. move log (plan section 5.8) =================
function renderMoveLog() {
  const log = $('moveLog');
  if (!log) return;
  const n = W.plyCount();
  const cur = W.cursor();
  let html = '';
  for (let i = 0; i < n; i += 2) {
    const marks = levelMarks.filter(mk => mk.afterPly === i)
      .map(mk => `<div class="mlSep">\u2014 level \u2192 ${mk.label} \u2014</div>`).join('');
    if (marks) html += marks;
    const ann0 = AN.annots[i] ? AN.annots[i].sym : '';
    const ann1 = AN.annots[i + 1] ? AN.annots[i + 1].sym : '';
    const cell = (ply, txt, cls) =>
      `<span class="mlMv ${cls}" data-ply="${ply}">${txt}</span>`;
    html += `<div class="mlRow"><span class="mlNum">${i / 2 + 1}.</span>` +
      cell(i, plyNotation(i) + ann0, cur === i ? 'cur' : '') +
      (i + 1 < n ? cell(i + 1, plyNotation(i + 1) + ann1, cur === i + 1 ? 'cur' : '') : '<span></span>') +
      `</div>`;
    // a level change at the live end renders after the last row
    const tail = levelMarks.filter(mk => mk.afterPly === i + 2 || (i + 2 >= n && mk.afterPly > i && mk.afterPly <= n))
      .filter(mk => mk.afterPly !== i)
      .map(mk => `<div class="mlSep">\u2014 level \u2192 ${mk.label} \u2014</div>`).join('');
    if (tail) html += tail;
  }
  // Designed empty state. A bare sentence in a tall empty column reads as a
  // rendering failure, so the placeholder is centred and labelled.
  log.innerHTML = html ||
    '<div class="mlEmpty"><span>No moves yet</span>'
    + '<small>Move a pawn or place a wall to begin</small></div>';
  log.querySelectorAll('.mlMv').forEach(el =>
    el.onclick = () => navGo(+el.dataset.ply));
  const curEl = log.querySelector('.mlMv.cur');
  if (curEl) curEl.scrollIntoView({ block: 'nearest' });
}

// Move list for the analysis tab. Every ply shows its score, taken from
// AN.scores, which the engine fills at the cursor and the blunder check fills
// for the whole game. A ply with no score yet shows a dash.
function renderAnMoveLog() {
  const log = $('anMoveLog');
  if (!log) return;
  const n = W.plyCount(), cur = W.cursor();
  if (!n) { log.innerHTML = '<div class="alHint">No moves yet.</div>'; return; }
  const evCell = (ply) => {
    const sc = AN.scores[ply];
    if (sc == null) return '<span class="alEv">–</span>';
    const abs = absScoreAt(sc, ply);
    const cls = abs > 40 ? 'good' : abs < -40 ? 'bad' : '';
    const ann = AN.annots[ply] ? AN.annots[ply].sym : '';
    return `<span class="alEv ${cls}">${fmtScore(abs)}${ann}</span>`;
  };
  const mvCell = (ply) =>
    `<span class="alMv ${cur === ply ? 'cur' : ''}" data-ply="${ply}">${plyNotation(ply)}</span>`;
  let html = '';
  for (let i = 0; i < n; i += 2) {
    html += `<div class="alRow"><span class="alNum">${i / 2 + 1}.</span>` +
      mvCell(i) + evCell(i) +
      (i + 1 < n ? mvCell(i + 1) + evCell(i + 1) : '<span></span><span></span>') +
      '</div>';
  }
  const missing = [];
  for (let i = 0; i < n; i++) if (AN.scores[i] == null) missing.push(i);
  if (missing.length) {
    html += `<div class="alHint">${missing.length} of ${n} plies have no score yet. ` +
            'Run Blunder check to score the whole game.</div>';
  }
  log.innerHTML = html;
  log.querySelectorAll('.alMv').forEach(el => el.onclick = () => navGo(+el.dataset.ply));
  const cel = log.querySelector('.alMv.cur');
  if (cel) cel.scrollIntoView({ block: 'nearest' });
}

// ===================== 14. analysis (plan section 5.6) ==================
const AN = {
  on: false, busy: false, sid: 0, timer: null,
  curDepth: 1,                 // current depth of the infinite loop
  scores: {},                  // ply -> best score, mover-relative
  annots: {},                  // ply -> {sym, drop}
  bcRun: false, bcCancel: false,
};
const AN_INF_CAP = 22;

const wpOf = s => 1 / (1 + Math.exp(-s / 180));   // cp -> win probability
function fmtScore(sc) {
  if (sc >= 900000) return '#';
  if (sc <= -900000) return '-#';
  const v = sc / 100;
  return (v >= 0 ? '+' : '') + v.toFixed(1);
}
// De-mirror a mover-relative score at ply p to the absolute side-0 view.
function absScoreAt(score, ply) { return (ply % 2) === 0 ? score : -score; }

function anToggle() {
  AN.on = !AN.on;
  const b = $('anEngBtn');
  b.textContent = AN.on ? 'ENGINE ON' : 'ENGINE OFF';
  b.classList.toggle('gold', AN.on);
  if (AN.on) anKick();
  else {
    AN.sid++; AN.busy = false;
    if (AN.timer) clearTimeout(AN.timer), AN.timer = null;
    // Say why the list is empty. A bordered box with nothing in it reads as
    // a failure, the same way the empty move log did.
    $('anLines').innerHTML =
      '<div class="mlEmpty"><span>Engine off</span>'
      + '<small>Turn the engine on to see the lines it considers</small></div>';
    $('anInfo').textContent = 'Engine off';
  }
}
function anRestart() {
  if (!AN.on) return;
  AN.sid++; AN.busy = false;
  if (AN.timer) { clearTimeout(AN.timer); AN.timer = null; }
  anKick();
}
function anKick() {
  if (!AN.on || AN.busy || AN.bcRun) return;
  const selDepth = +$('anDepth').value;
  const lines = +$('anPvCount').value;
  const isInf = selDepth === 0;
  const d = isInf ? AN.curDepth : selDepth;
  const sid = ++AN.sid;
  AN.busy = true;

  const finish = (linesData, nodes, depthReached, ms) => {
    AN.busy = false;
    if (sid !== AN.sid || !AN.on) return;
    anRenderData(linesData, nodes, depthReached, ms, d);
    // feed the eval graph with the root score at the current cursor
    if (linesData.length > 0) {
      AN.scores[W.cursor()] = linesData[0].score;
      renderAnMoveLog();
      drawGraph();
    }
    if (isInf && linesData.length > 0) {
      AN.curDepth = Math.min(d + 1, AN_INF_CAP);
      if (AN.curDepth > d)
        AN.timer = setTimeout(anKick, d >= 14 ? 450 : d >= 10 ? 220 : 100);
    }
  };
  if (ANW.analyze({ moves: pliesUpToCursor(), depth: Math.max(2, d),
                    timeMs: isInf ? 500 : 1100, lines }, res => {
        if (!res) {   // worker failed: fall through to the local path once
          if (sid !== AN.sid || !AN.on) { AN.busy = false; return; }
          anRunLocal(sid, d, isInf, lines, finish);
          return;
        }
        finish(res.lines, res.nodes, res.depth, res.ms);
      })) {
    return;
  }
  anRunLocal(sid, d, isInf, lines, finish);
}
// Recorded plies up to the review cursor (the position being analysed).
function pliesUpToCursor() {
  const out = [];
  const n = Math.min(W.cursor(), W.plyCount());
  for (let i = 0; i < n; i++) out.push(packPly(i));
  return out;
}
function anRunLocal(sid, d, isInf, lines, finish) {
  setTimeout(() => {
    if (sid !== AN.sid || !AN.on) { AN.busy = false; return; }
    W.scratchFromLive();
    const t0 = performance.now();
    const got = W.analyze(Math.max(2, d), isInf ? 500 : 1100, lines);
    const ms = Math.max(1, performance.now() - t0);
    AN.busy = false;
    if (sid !== AN.sid || !AN.on) return;
    const linesData = [];
    for (let i = 0; i < got; i++) {
      const pv = [];
      for (let j = 0; j < W.anLineLen(i); j++) pv.push(W.anLineMove(i, j));
      linesData.push({ score: W.anLineScore(i), pv });
    }
    finish(linesData, W.anNodes(), W.anDepth(), ms);
  }, 30);
}
function pvText(i, maxMoves) {
  const len = W.anLineLen(i);
  const out = [];
  for (let j = 0; j < Math.min(len, maxMoves); j++) out.push(packedToTok(W.anLineMove(i, j)));
  if (len > maxMoves) out.push('\u2026');
  return out.join(' ');
}
// Concrete consequence of the line's first move in race currency:
// the opponent's shortest path before -> after that move.
function distDeltaFor(firstPacked) {
  W.scratchFromLive();
  const opp = 1 - W.scrTurn();
  let before = W.scrDist(opp);
  const m = unpackMove(firstPacked);
  const ok = m.isWall ? W.scrApplyWall(m.a, m.b, m.c) : W.scrApplyPawn(m.a);
  if (!ok) return '';
  const after = W.scrDist(opp);
  return before + '\u2192' + after;
}
function anRenderData(linesData, nodes, depthReached, ms, depth) {
  const box = $('anLines');
  const nps = Math.round(nodes / (ms / 1000));
  $('anInfo').textContent =
    `d${depth} \u00b7 ${(nodes / 1000).toFixed(1)}k nodes \u00b7 ${(nps / 1000).toFixed(1)}k nps` +
    ($('anDepth').value === '0' ? ' \u00b7 \u221e' : '');
  const shown = Math.min(linesData.length, +$('anPvCount').value);
  let html = '';
  for (let i = 0; i < shown; i++) {
    const ln = linesData[i];
    const abs = absScoreAt(ln.score, W.cursor());
    const cls = abs > 40 ? 'good' : abs < -40 ? 'bad' : '';
    const dd = distDeltaFor(ln.pv[0]);
    const toks = ln.pv.slice(0, 12).map(packedToTok);
    if (ln.pv.length > 12) toks.push('\u2026');
    html += `<div class="pvRow" data-line="${i}">
      <span class="pvChip ${cls}">${fmtScore(abs)}</span>
      <span class="pvMoves">${toks.join(' ')}</span>
      <span class="pvDist num" title="Opponent distance to goal">${dd}</span>
      <span class="pvTag">L${i + 1}</span></div>`;
  }
  box.innerHTML = html;
  const colors = ['--gold2', '--blue', '--green', '--amber', '--muted'];
  box.querySelectorAll('.pvRow').forEach(row => {
    const i = +row.dataset.line;
    row.onclick = () => showLinePreview(linesData[i], getComputedStyle(document.documentElement).getPropertyValue(colors[i]).trim());
    row.onmouseleave = () => { B.linePreview = null; B.render(); };
  });
  // eval bar follows the analysis while it is running
  if (linesData.length > 0) anSetEval(linesData[0].score);
}
function showLinePreview(line, color) {
  const walls = [], pawns = [];
  for (const packed of line.pv) {
    const m = unpackMove(packed);
    if (m.isWall) walls.push({ o: m.a, r: m.b, c: m.c });
    else pawns.push(m.a);
  }
  B.linePreview = { walls, pawns, color };
  B.render();
}
function anSetEval(moverScore) {
  const abs = absScoreAt(moverScore, W.cursor());   // side-0 view
  setEval(humanSide === 0 ? abs : -abs);
}

// ---- eval graph --------------------------------------------------------
function drawGraph() {
  const gw = $('anGraph');
  if (gw) gw.style.display = W.plyCount() >= 4 ? '' : 'none';
  const cv = $('anGraph');
  if (!cv || !cv.clientWidth) return;
  const dpr = Math.min(devicePixelRatio || 1, 3);
  const w = cv.clientWidth, h = 72;
  if (cv.width !== w * dpr || cv.height !== h * dpr) {
    cv.width = w * dpr; cv.height = h * dpr;
  }
  const g = cv.getContext('2d');
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.clearRect(0, 0, w, h);
  const n = W.plyCount();
  const cs = getComputedStyle(document.documentElement);
  const gold2 = cs.getPropertyValue('--gold2').trim();
  const red = cs.getPropertyValue('--red').trim();
  // A missing token leaves fillStyle unchanged and the area silently does not
  // paint, so fall back to the solid player colour at low alpha.
  const soft = (name, base) => cs.getPropertyValue(name).trim() ||
    cs.getPropertyValue(base).trim() || 'rgba(255,255,255,.15)';
  const p0s = soft('--p0-soft', '--p0');
  const p1s = soft('--p1-soft', '--p1');
  // sample points: ply -> wp (side-0 view); missing plies carry the last value
  const pts = [];
  let lastWp = .5;
  for (let i = 0; i <= n; i++) {
    if (i > 0 && AN.scores[i - 1] != null)
      lastWp = wpOf(absScoreAt(AN.scores[i - 1], i - 1));
    pts.push(lastWp);
  }
  const X = i => n ? (i / n) * (w - 6) + 3 : 3;
  // The curve is auto-scaled. A whole game inside 0 to 1 is a flat line in the
  // middle for every position that is not already decided, which shows
  // nothing. The band is centred on 0.5 and opens only as far as the data
  // needs, with a floor so that noise is not magnified into a swing. The
  // corner label names the band, so a zoomed wiggle is never read as a rout.
  let dev = 0;
  for (const wpv of pts) dev = Math.max(dev, Math.abs(wpv - .5));
  const span = Math.min(.5, Math.max(.08, dev * 1.18));
  const Y = wpv => {
    const t = (wpv - (.5 - span)) / (2 * span);
    return h - 4 - Math.max(0, Math.min(1, t)) * (h - 8);
  };
  // area fill split at 50%
  g.beginPath();
  g.moveTo(X(0), Y(.5));
  pts.forEach((wpv, i) => g.lineTo(X(i), Y(wpv)));
  g.lineTo(X(n), Y(.5));
  g.closePath();
  g.fillStyle = p0s; g.fill();
  g.save();
  g.beginPath();
  g.rect(0, 0, w, Y(.5));
  g.clip();
  g.fillStyle = p1s;
  g.beginPath();
  g.moveTo(X(0), Y(.5));
  pts.forEach((wpv, i) => g.lineTo(X(i), Y(wpv)));
  g.lineTo(X(n), Y(.5));
  g.closePath(); g.fill();
  g.restore();
  // centre line + curve
  g.strokeStyle = cs.getPropertyValue('--bor2').trim();
  g.lineWidth = 1;
  g.beginPath(); g.moveTo(3, Y(.5)); g.lineTo(w - 3, Y(.5)); g.stroke();
  g.strokeStyle = gold2; g.lineWidth = 1.5;
  g.beginPath();
  pts.forEach((wpv, i) => i ? g.lineTo(X(i), Y(wpv)) : g.moveTo(X(i), Y(wpv)));
  g.stroke();
  // blunder dots
  for (let i = 0; i < n; i++) {
    if (AN.annots[i] && AN.annots[i].sym === '??') {
      const wpv = wpOf(absScoreAt(AN.scores[i], i));
      g.beginPath(); g.arc(X(i + 1), Y(wpv), 3, 0, 7);
      g.fillStyle = red; g.fill();
    }
  }
  // cursor marker
  const cur = W.cursor();
  g.strokeStyle = cs.getPropertyValue('--gold').trim();
  g.lineWidth = 1.5;
  g.beginPath(); g.moveTo(X(cur), 2); g.lineTo(X(cur), h - 2); g.stroke();
  // scale label
  g.fillStyle = cs.getPropertyValue('--muted').trim();
  g.font = "9px 'JetBrains Mono', monospace";
  g.textAlign = 'right'; g.textBaseline = 'top';
  g.fillText('±' + Math.round(span * 100) + '%', w - 4, 3);
}
function graphScrub(ev) {
  const rect = $('anGraph').getBoundingClientRect();
  const x = ev.clientX - rect.left;
  const n = W.plyCount();
  navGo(Math.round(x / rect.width * n));
}

// ---- blunder check (plan section 5.6) ----------------------------------
function blunderCheck() {
  if (AN.bcRun) { AN.bcCancel = true; return; }
  const n = W.plyCount();
  if (!n) { toast('info', 'No moves to analyse'); return; }
  if (engineThinking) { toast('warn', 'Wait for the engine move'); return; }
  const depth = Math.max(4, +$('anDepth').value || 10);
  const perMoveMs = 260;
  AN.bcRun = true; AN.bcCancel = false;
  const btn = $('anBlunderBtn');
  btn.textContent = 'Cancel check'; btn.classList.add('danger');
  $('bcBox').style.display = 'block';
  $('bcSummary').style.display = 'none';

  function step(i) {
    if (AN.bcCancel || i >= n) { bcFinish(n); return; }
    W.scratchFromPly(i);
    let sym = '', drop = 0;
    const nb = W.analyze(depth, perMoveMs, 1);
    if (nb > 0) {
      const bestS = W.anLineScore(0);
      const m = unpackMove(packPly(i));
      const applied = m.isWall ? W.scrApplyWall(m.a, m.b, m.c) : W.scrApplyPawn(m.a);
      if (applied) {
        const nc = W.analyze(depth, perMoveMs, 1);
        if (nc > 0) {
          const playedS = -W.anLineScore(0);   // back to the mover-i view
          drop = wpOf(bestS) - wpOf(playedS);
          sym = drop >= .25 ? '??' : drop >= .13 ? '?' : drop >= .06 ? '?!' : '';
          if (!sym && playedS >= bestS - 15 && bestS > 60) sym = '!';
        }
      }
      AN.scores[i] = bestS;
      renderAnMoveLog();
    }
    AN.annots[i] = { sym, drop };
    $('bcFill').style.width = ((i + 1) / n * 100).toFixed(1) + '%';
    $('bcLabel').textContent = 'Evaluating ply ' + (i + 1) + ' / ' + n;
    setTimeout(() => step(i + 1), 15);
  }
  function bcFinish(total) {
    AN.bcRun = false;
    btn.textContent = 'Blunder check'; btn.classList.remove('danger');
    $('bcBox').style.display = 'none';
    renderMoveLog(); drawGraph();
    if (total && !AN.bcCancel) {
      const acc = [0, 0], cnt = [0, 0], blu = [0, 0];
      for (let k = 0; k < total; k++) {
        const a = AN.annots[k]; if (!a) continue;
        const sd = k % 2;
        acc[sd] += 1 - (a.drop || 0); cnt[sd]++;
        if (a.sym === '??') blu[sd]++;
      }
      const pc = v => cnt[v] ? (acc[v] / cnt[v] * 100).toFixed(1) + '%' : '\u2013';
      const el = $('bcSummary');
      el.style.display = 'block';
      el.innerHTML = `<b>You</b> ${pc(humanSide)} \u00b7 ${blu[humanSide]} blunders<br>` +
        `<b>Zquoridor</b> ${pc(1 - humanSide)} \u00b7 ${blu[1 - humanSide]} blunders`;
    } else {
      toast('info', 'Blunder check cancelled');
    }
    if (AN.on) anRestart();
  }
  setTimeout(() => step(0), 30);
}

// ---- hint (plan section 5.5) -------------------------------------------
let hintTimer = null;
function showHint() {
  if (gameOver || engineThinking || W.turn() !== humanSide) return;
  if (!atLiveEnd()) return;
  W.scratchFromLive();
  if (W.analyze(10, 700, 1) < 1) return;
  const m = unpackMove(W.anLineMove(0, 0));
  if (m.isWall) {
    const [do_, dr, dc] = B.engWallToDisp(m.a, m.b, m.c);
    clearGhost();
    B.ghost = { o: do_, r: dr, c: dc, state: 'pending' };
    wallState = 'IDLE';   // hint ghost is display-only; no confirm chip
    $('confirmChip').style.display = 'none';
  } else {
    B.selected = B.engPawnToDisp(m.a);
    B.dots = [];
  }
  tone(660, 60, 'triangle', -14);
  B.render();
  if (hintTimer) clearTimeout(hintTimer);
  hintTimer = setTimeout(() => {
    clearGhost(); B.selected = -1; buildLegalSets(); B.render();
  }, 4000);
}

// ===================== 12b. layout reflow ==============================
let g_wideLayout = null;
function layoutReflow() {
  const wide = matchMedia('(min-width:900px)').matches;
  if (wide === g_wideLayout) return;
  g_wideLayout = wide;
  const slot = $('controlsSlot'), under = $('underBoard');
  if (!slot || !under) return;
  // The single evaluation bar is vertical beside the board on a wide screen
  // and horizontal under the board on a phone. It is the same element.
  const zone = $('boardZone'), ew = $('evalWrap');
  if (ew && zone && under) {
    if (wide) { if (ew.parentElement !== zone) zone.insertBefore(ew, zone.firstChild); }
    else if (ew.parentElement !== under) under.insertBefore(ew, under.firstChild);
  }
  const target = wide ? slot : under;
  for (const id of ['statusRow', 'controls']) {
    const el = $(id);
    if (el && el.parentElement !== target) target.appendChild(el);
  }
  // The move log follows the same rule. On a phone it fills the space under
  // the button row, which is otherwise dead. On a wide screen it returns to
  // its card in the side rail.
  // ...but "phone" here means the one-column layout, not merely a narrow
  // window. Phone landscape is narrow AND two-column: the side rail is a real
  // static column there. Reading the rail's own computed position says which
  // layout is live without restating the media queries in JS. Without this the
  // log went under the board, where that block has no height in landscape, and
  // the rail showed a MOVE LOG card with nothing in it.
  const log = $('moveLog'), home = $('moveLogHome'), panel = $('sidePanel');
  const railIsColumn = !!panel && getComputedStyle(panel).display !== 'none' &&
                       getComputedStyle(panel).position === 'static';
  const logTarget = (wide || railIsColumn) ? home : under;
  if (log && logTarget && log.parentElement !== logTarget) logTarget.appendChild(log);
}

// ===================== 13. boot ========================================
function boot() {
  loadSettings();
  B = new QBoard($('board'));
  window.__qb = B;   // test/debug handle
  // The player strips must be exactly as wide as the board. On a phone in
  // landscape the board is limited by height, not by width, so without this
  // the strips stretch across the whole column and the board floats inside
  // them.
  B.onChange = () => {
    // The player strips and the race meter align to the board itself, not to
    // the column. The evaluation bar and its gutter sit inside the board zone,
    // so the board is not centred on the column and a centred strip would be
    // visibly off by half of that width.
    const col = $('boardCol'), wrap = B.cv.parentElement;
    if (!col || !wrap) return;
    const left = Math.max(0, Math.round(wrap.getBoundingClientRect().left -
                                        col.getBoundingClientRect().left));
    const aligned = ['hudTop', 'hudBottom', 'underBoard'];
    // The horizontal evaluation bar is part of the board block, so it takes
    // the same width. The vertical one lives beside the board and keeps its
    // own geometry.
    const ew = $('evalWrap');
    if (ew && ew.parentElement && ew.parentElement.id === 'underBoard') {
      // Pinned to the board's width, not to its parent's. The parent may be
      // wider than the board (see the button row below), and the bar reports
      // on the board, so it matches the board.
      ew.style.height = '';
      ew.style.width = B.cssSide + 'px';
      // Centred, not flush left: its parent can be wider than the board (see
      // the button row below), and both parent and board are centred on the
      // column, so centring is what keeps the bar under the board.
      ew.style.marginLeft = 'auto';
      ew.style.marginRight = 'auto';
    } else if (ew) {
      // Vertical: the bar reports on the board, so it is exactly as tall.
      ew.style.width = ''; ew.style.marginLeft = ''; ew.style.maxWidth = '';
      ew.style.height = B.cssSide + 'px';
    }
    for (const id of aligned) {
      const el = $(id);
      if (!el) continue;
      el.style.width = B.cssSide + 'px';
      el.style.maxWidth = '100%';
      el.style.marginLeft = left + 'px';
      el.style.marginRight = 'auto';
    }
    // The button row sits under the board on a phone. In landscape the board
    // is limited by height and can be much narrower than the column it sits
    // in, and the row then overflowed the board's width into a scroll with no
    // scrollbar, which left the wall buttons out of reach with nothing to say
    // so. Give that block the width its buttons need, up to the column.
    const under = $('underBoard'), ctrls = $('controls');
    if (under && ctrls && ctrls.parentElement === under) {
      const kids = [...ctrls.children].filter(e => getComputedStyle(e).display !== 'none');
      const gap = parseFloat(getComputedStyle(ctrls).gap) || 0;
      const need = kids.reduce((s, e) => s + e.getBoundingClientRect().width, 0) +
                   gap * Math.max(0, kids.length - 1);
      if (need > B.cssSide) {
        under.style.width = Math.min(col.clientWidth, Math.ceil(need)) + 'px';
        under.style.marginLeft = 'auto';
      }
    }
  };
  applySettings();
  W.newGame();
  humanSide = S.side;
  syncAll();
  setEval(0);   // seat the bar at even, same reason as in newGame()
  startClock();
  setStatus(humanSide === 0 ? 'Your move' : 'Zquoridor starts');
  // Layout reflow across the breakpoint. On a wide screen the status line and
  // the button row belong beside the board, in the side rail, so the board
  // keeps the full height of its column. On a phone they stay under the
  // board, inside the thumb arc. The same nodes move; there is one copy of
  // each control.
  layoutReflow();
  // deferred re-fits: the first fit() may run before final layout; re-measure
  // once the flex layout settles, and on any viewport resize.
  [200, 600, 1500].forEach(t => setTimeout(() => { layoutReflow(); B.fit(); }, t));
  window.addEventListener('resize', () => { layoutReflow(); B.fit(); drawGraph(); });

  // analysis controls (plan section 5.6)
  $('anEngBtn').onclick = anToggle;
  $('anDepth').onchange = anRestart;
  $('anPvCount').onchange = anRestart;
  $('anBlunderBtn').onclick = blunderCheck;
  // Export from the Analysis tab. QFEN exports the position the cursor is on,
  // which is the one on the board, so a reader can take any position out of a
  // game under review. The game text always covers the whole game.
  $('anCopyQfen').onclick = () => copyText(W.qfenExportStr(), 'QFEN');
  $('anCopyGame').onclick = () => copyText(qgnExport(), 'Game text');
  $('anSaveGame').onclick = () =>
    downloadText(gameFileName(), qgnExport(), 'application/x-zquoridor-game');
  $('anCopyLink').onclick = () => copyText(shareLink(), 'Link');
  $('btnReturn').onclick = () => navGo(W.plyCount());
  $('navFirst').onclick = () => navGo(0);
  $('navPrev').onclick = () => navGo(W.cursor() - 1);
  $('navNext').onclick = () => navGo(W.cursor() + 1);
  $('navLast').onclick = () => navGo(W.plyCount());
  const gEl = $('anGraph');
  let scrubbing = false;
  gEl.addEventListener('pointerdown', ev => { scrubbing = true; graphScrub(ev); });
  gEl.addEventListener('pointermove', ev => { if (scrubbing) graphScrub(ev); });
  addEventListener('pointerup', () => { scrubbing = false; });

  // P8 wiring: editor, drop target, autosave resume, share links
  wireEditor();
  wireDropTarget();
  if (!bootHashLoad()) checkAutosaveOnBoot();
  // The move count is a reading only. The move log is adjacent in both
  // layouts, and Recent games is a header menu item.
  // prewarm the analysis worker while the user plays
  try { ANW.ensure(); } catch (e) {}
  updateMovesChip();
}
ZquoridorModule().then((Module) => {
  try {
    bindEngine(Module);
    let nnueOk = false;
    try { nnueOk = !!Module._qr_load_nnue_weights('/data/nnue/nnue_weights_int8.bin'); } catch (e3) {}
    boot();
    if (!nnueOk) toast('warn', 'NNUE weights unavailable - heuristic mode');
  } catch (e) {
    document.title="BOOTERR "+(e&&e.stack?e.stack.split("\n")[1]:""+e);
  }
}).catch(e => { var st=document.getElementById("status"); if(st) st.textContent="WASM load failed: "+e; });
