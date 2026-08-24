// app.js -- Zquoridor premium GUI logic (plan gui-premium.md).
// Sections: 1 constants - 2 wasm bindings - 3 settings - 4 state -
// 5 hud/clocks/race/eval - 6 board bridge - 7 wall input - 8 pawn input -
// 9 play flow - 10 sound/haptics - 11 modals/toasts - 12 keyboard - 13 boot
// plus 14 analysis (plan section 5.6, phase P7).
'use strict';

// ===================== 1. constants ====================================
const LEVELS = {
  pebble: { ms: 50,   label: 'Pebble', color: 'var(--muted)', desc: 'Learning the rules with you' },
  sprite: { ms: 150,  label: 'Sprite', color: 'var(--txt2)',  desc: 'Quick and careless' },
  squire: { ms: 400,  label: 'Squire', color: 'var(--green)', desc: 'Solid club player' },
  knight: { ms: 1000, label: 'Knight', color: 'var(--blue)',  desc: 'Punishes loose walls' },
  sage:   { ms: 2500, label: 'Sage',   color: 'var(--amber)', desc: 'Sees the whole race' },
  titan:  { ms: 8000, label: 'Titan',  color: 'var(--gold)',  desc: 'Full strength' },
};
const BOARD_THEMES = ['obsidian', 'walnut', 'ivory', 'slate', 'emerald', 'parchment', 'marble', 'noir'];
const PAWN_STYLES = ['disc', 'pillar', 'crown', 'rune', 'pawnChess', 'beacon'];
const FILES = 'abcdefghi';

// Absolute notation (plan 5.8 / 16.2): pawn e5, wall Hc6/Vf3. Rank 1 is
// player 0's home row, matching the QFEN and the board coordinates.
function engCellAlg(cell) { return FILES[cell % 9] + (Math.floor(cell / 9) + 1); }
function packedToTok(packed) {
  const isWall = (packed >> 24) & 1;
  if (!isWall) return engCellAlg(packed & 255);
  return ((packed >> 16) & 255) === 0 ? 'H' : 'V';
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
  v: 1,
  preset: 'premiumDark',
  ui: 'dark', board: 'obsidian', pawn: 'disc',
  accent: 'gold', fs: 1, density: 'comfortable',
  frame: 'hairline', wallFinish: 'beveled', cellSep: 'grooves',
  coords: 'edges', boardScale: 1,
  pawnSize: 'regular', pawnShadow: 'soft', distinctShapes: false,
  paths: false, dots: true, lastMove: true, evalGlow: true, evalBar: true,
  sound: true, soundPack: 'wood', volume: .6,
  soundEvents: { moves: true, walls: true, illegal: true, clock: true, end: true, ui: true },
  haptics: 'full',
  anim: 'full', animSpeed: 1,
  confirmWalls: null, touchOffset: null, stickyArm: false, handedness: 'right',
  level: 'knight',
  custom: { mode: 'time', depth: 12, timeMs: 1000 },
  clockMode: 'none', baseMin: 5, incSec: 3, side: 0,
  flipped: false,
  worker: true, autosave: true,
};
let S = { ...DEFAULTS, soundEvents: { ...DEFAULTS.soundEvents } };
function loadSettings() {
  try {
    const raw = localStorage.getItem('zq.settings');
    if (!raw) return;
    const parsed = JSON.parse(raw);
    S = {
      ...DEFAULTS, ...parsed,
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
    pawnShadow: 'deep', soundPack: 'wood', anim: 'full', fs: 1,
  },
  premiumDark: {},
  highContrast: {
    ui: 'dark', board: 'noir', pawn: 'rune', distinctShapes: true, accent: 'gold',
    coords: 'all', fs: 1.12, cellSep: 'inlaid', wallFinish: 'beveled', anim: 'reduced',
  },
  minimal: {
    ui: 'dark', board: 'slate', pawn: 'rune', frame: 'none', wallFinish: 'flat',
    cellSep: 'flat', coords: 'off', paths: false, dots: false, evalBar: false,
    evalGlow: false, sound: false, haptics: 'off', anim: 'reduced', density: 'compact',
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
function pips(el, n) {
  if (el.childElementCount !== 10) el.innerHTML = '<i></i>'.repeat(10);
  [...el.children].forEach((p, i) => p.classList.toggle('on', i < n));
}
function fmtClock(ms) {
  if (ms == null) return '--:--';
  const s = Math.max(0, Math.ceil(ms / 1000));
  return Math.floor(s / 60) + ':' + String(s % 60).padStart(2, '0');
}
let clockMs = [null, null], clockTimer = null, lastTickAt = 0;
function startClock() {
  stopClock();
  if (S.clockMode === 'none') { renderClocks(); return; }
  clockMs = [S.baseMin * 60000, S.baseMin * 60000];
  lastTickAt = performance.now();
  clockTimer = setInterval(clockTick, 200);
  renderClocks();
}
function stopClock() { if (clockTimer) clearInterval(clockTimer); clockTimer = null; }
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
  pushRecent('flag');
  sound('end'); haptic([80]);
}
function renderClocks() {
  const pairs = [['clock0', 'mClock0'], ['clock1', 'mClock1']];
  for (let pl = 0; pl < 2; pl++) {
    const txt = fmtClock(S.clockMode === 'none' ? null : clockMs[pl]);
    for (const id of pairs[pl]) {
      const el = $(id); el.textContent = txt;
      el.classList.toggle('low', S.clockMode !== 'none' && clockMs[pl] < 30000 && clockMs[pl] >= 10000);
      el.classList.toggle('crit', S.clockMode !== 'none' && clockMs[pl] < 10000);
    }
  }
}
function refreshHud() {
  const wl = [W.wallsLeft(0), W.wallsLeft(1)];
  const d = [W.dist(0), W.dist(1)];
  const t = W.turn();
  B.setTurn(t);
  // desktop rail: opponent top, you bottom
  $('walls1').textContent = wl[1 - humanSide]; pips($('pips1'), wl[1 - humanSide]);
  $('walls0').textContent = wl[humanSide];     pips($('pips0'), wl[humanSide]);
  // mobile bar
  $('mWalls1').textContent = wl[1 - humanSide]; pips($('mPips1'), wl[1 - humanSide]);
  $('mWalls0').textContent = wl[humanSide];     pips($('mPips0'), wl[humanSide]);
  const dYou = d[humanSide], dOpp = d[1 - humanSide];
  setDist('dist0', 'mDist0', dYou, dYou <= dOpp);
  setDist('dist1', 'mDist1', dOpp, dOpp < dYou);
  // turn highlight
  const youTurn = t === humanSide && !gameOver;
  $('hudYou').classList.toggle('turn', youTurn); $('mHudYou').classList.toggle('turn', youTurn);
  $('hudOpp').classList.toggle('turn', !youTurn && !gameOver);
  $('mHudOpp').classList.toggle('turn', !youTurn && !gameOver);
  $('hudYou').classList.toggle('dim', !youTurn); $('mHudYou').classList.toggle('dim', !youTurn);
  $('hudOpp').classList.toggle('dim', youTurn || gameOver); $('mHudOpp').classList.toggle('dim', youTurn || gameOver);
  // dock badges
  const myWalls = wl[humanSide];
  $('badgeH').textContent = myWalls; $('badgeV').textContent = myWalls;
  const noWalls = myWalls <= 0 || t !== humanSide || gameOver || engineThinking;
  $('wallH').classList.toggle('off', noWalls); $('wallV').classList.toggle('off', noWalls);
  $('btnUndo').classList.toggle('off', W.plyCount() === 0 || engineThinking || !atLiveEnd());
  // race meter
  const total = Math.max(1, dYou + dOpp);
  const share = dOpp / total * 100;
  $('raceFill0').style.setProperty('--share', (dOpp / total).toFixed(3));
  $('raceFill0').style.width = share.toFixed(1) + '%';
  $('raceDiv').style.left = share.toFixed(1) + '%';
  $('raceLbl').textContent = dYou + ' : ' + dOpp;
  // mobile meter
  $('rf0M').style.width = share.toFixed(1) + '%';
  $('rdvM').style.left = share.toFixed(1) + '%';
  $('rlM').textContent = dYou + ' : ' + dOpp;
  renderClocks();
}
function setDist(idM, idMob, v, lead) {
  for (const id of [idM, idMob]) {
    const el = $(id);
    el.innerHTML = '<small>DIST</small>' + v;
    el.classList.toggle('lead', lead); el.classList.toggle('behind', !lead);
  }
}
function setEval(score) {   // score mover-relative from the engine's last move
  const abs = humanSide === 0 ? score : -score;      // de-mirror to colour 0 view
  const share = 50 + 50 * Math.tanh(abs / 400);
  $('evalFill').style.height = share.toFixed(1) + '%';
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
  // Dots and the side-to-move marker must be current BEFORE setData paints,
  // or they only appear on the render after this one.
  buildLegalSets();
  refreshHud();
  B.setData(pw, wh, wv, lastMoveInfo);
  renderMoveLog();
  updateNav();
  drawGraph();
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
function setStatus(t) { $('status').textContent = t; }

// Single-owner engine timer: scheduling a new engine turn cancels any stale
// one, and engineTurn refuses to act unless it really is the engine's move.
// This makes "the engine playing for both sides" structurally impossible.
let engineTimer = null;
function scheduleEngineTurn(delay) {
  if (engineTimer) clearTimeout(engineTimer);
  engineTimer = setTimeout(() => { engineTimer = null; engineTurn(); }, delay);
}
function afterHumanMove() {
  updateMovesChip();
  checkEnd();
  if (!gameOver) {
    engineThinking = true;
    refreshHud();
    setStatus('Zquoridor is thinking...');
    scheduleEngineTurn(120);
  }
}
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
  const lv = LEVELS[S.level];
  const ok = W.engineMove(24, lv.ms);
  engineThinking = false;
  if (!ok) { syncAll(); return; }
  setEval(-W.lastEval());          // eval is from the engine's perspective
  sound('move'); haptic(10);
  syncAll();
  checkEnd();
  if (!gameOver) setStatus('Your move');
}
function checkEnd() {
  const w = W.winner();
  if (w !== -1) {
    gameOver = true;
    const youWon = w === humanSide;
    setStatus(youWon ? 'You won - goal reached' : 'Zquoridor won');
    toast(youWon ? 'ok' : 'err', youWon ? 'Victory' : 'Defeat');
    sound('end'); haptic(youWon ? [20, 60, 20, 60, 40] : [60]);
    pushRecent('played');
    refreshHud(); return true;
  }
  if (W.isDraw()) { gameOver = true; setStatus('Draw by repetition'); pushRecent('draw'); return true; }
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
  humanSide = S.side;
  AN.scores = {}; AN.annots = {}; AN.curDepth = 1;
  levelMarks = [];
  g_startedFromCustomFlag = false; g_rootQfen = '-';
  clearGhost();
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
  $('navPly').textContent = W.cursor() + ' / ' + W.plyCount();
  const end = atLiveEnd();
  $('btnReturn').style.display = end ? 'none' : 'flex';
}
// Rolls back to the human's previous turn: the latest position before the
// cursor where it is the human to move, then truncates the future.
function takeback() {
  if (AN.bcRun) { toast('warn', 'Blunder check running - cancel it first'); return; }
  if (engineThinking) { toast('warn', 'Wait for the engine move'); return; }
  const n = W.plyCount();
  if (!n) { toast('info', 'Nothing to take back'); return; }
  let target = 0;
  for (let p = Math.min(W.cursor(), n) - 1; p >= 0; p--) {
    W.scratchFromPly(p);
    if (W.scrTurn() === humanSide) { target = p; break; }
  }
  W.truncateHistory(target);
  gameOver = false;
  clearGhost();
  setStatus('Takeback - your move');
  sound('arm');
  syncAll();
}

// ===================== 7. wall input (plan section 6) ==================
// states: IDLE -> ARMED -> DRAGGING -> (PENDING) -> IDLE/COMMIT
let wallState = 'IDLE', armedO = 0;
let dragPtr = null;

function clearGhost() {
  wallState = 'IDLE'; dragPtr = null;
  B.ghost = null; B.ghostFrom = null;
  $('wallH').classList.remove('armed'); $('wallV').classList.remove('armed');
  $('confirmChip').style.display = 'none';
  hideLegalSlots();
}
function armWall(o) {
  if ($('wallH').classList.contains('off')) return;
  if (wallState === 'ARMED' && armedO === o) { clearGhost(); return; }
  clearGhost();
  armedO = o; wallState = 'ARMED';
  $(o === 0 ? 'wallH' : 'wallV').classList.add('armed');
  showLegalSlots(o);
  sound('arm'); haptic(6);
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
function snapGhost(px, py) {
  const off = touchOffsetPx();
  const p = { x: px, y: py - off };
  // Always snap to the NEAREST anchor inside the board: a tap far from any
  // groove still produces a concrete ghost (ok / assisted / bad with the
  // reason), never a silent no-op.
  let a = B.nearestAnchor(p.x, p.y);
  if (!a) { B.ghost = null; setStatus('Tap between cells to place the wall'); B.render(); return; }
  let [o, r, c] = [armedO, a.r, a.c];
  const [eo, er, ec] = B.dispWallToEng(o, r, c);
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
  if (st === 'bad') setStatus(illegalReason(eo, er, ec));
  else setStatus(st === 'assisted' ? 'Snapped to the nearest legal slot' : '');
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
  const cell = B.pointToCell(pt.x, pt.y);
  if (wallState === 'ARMED') { wallState = 'DRAGGING'; dragPtr = ev.pointerId; snapGhost(pt.x, pt.y); ev.preventDefault(); return; }
  if (cell) {
    const near = B.nearestAnchor(pt.x, pt.y);
    // A press on a CELL CENTER belongs to the pawn; only presses hugging a
    // groove intersection start the direct wall gesture (plan section 6.2).
    let cellCenterPress = false;
    if (cell) {
      const cc2 = B.cellCenter(cell.r, cell.c);
      cellCenterPress = Math.hypot(pt.x - cc2.x, pt.y - cc2.y) < .34 * B.U;
    }
    if (!near || cellCenterPress || near.dist > .55 * B.U || !nearAnchorHasLegal(near)) {
      pawnDown(cell.r * 9 + cell.c, pt);   // pawnDown wants a display index
      return;
    }
    // direct gesture: provisional orientation from the groove under the cursor
    const fx = Math.abs((pt.x % B.U) - B.U / 2), fy = Math.abs((pt.y % B.U) - B.U / 2);
    armedO = (fy < fx) ? 0 : (fx < fy ? 1 : lastArmO());
    clearGhost(); armedO = armedO; wallState = 'DRAGGING'; dragPtr = ev.pointerId;
    snapGhost(pt.x, pt.y); ev.preventDefault();
  }
}
function nearAnchorHasLegal(a) {
  for (const o of [0, 1]) if (legalWall[o * 64 + a.r * 8 + a.c]) return true;
  return false;
}
function lastArmO() { return armedO; }
function onBoardPointerMove(ev) {
  if (wallState !== 'DRAGGING' || ev.pointerId !== dragPtr) return;
  const pt = boardPoint(ev); snapGhost(pt.x, pt.y); ev.preventDefault();
}
function onBoardPointerUp(ev) {
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
  if (!gh || engineThinking || gameOver || W.turn() !== humanSide) {
    clearGhost(); B.render(); return;   // stale ghost: never apply out of turn
  }
  const [eo, er, ec] = B.dispWallToEng(gh.o, gh.r, gh.c);
  if (W.applyWall(eo, er, ec)) {
    clearGhost();
    sound('wall'); haptic(18);
    afterHumanMove();
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

// dock events: press-and-drag (M1) vs click-to-arm (M2)
for (const [id, o] of [['wallH', 0], ['wallV', 1]]) {
  const el = $(id);
  el.addEventListener('pointerdown', ev => {
    el.setPointerCapture(ev.pointerId);
    el._downAt = performance.now(); el._dragging = false;
  });
  el.addEventListener('pointermove', ev => {
    if (el._dragging || !el._downAt) return;
    if (Math.abs(ev.movementX) + Math.abs(ev.movementY) > 3 || performance.now() - el._downAt > 150) el._dragging = true;
    if (el._dragging && wallState !== 'DRAGGING') { armWall(o); wallState = 'DRAGGING'; dragPtr = ev.pointerId; }
    if (wallState === 'DRAGGING' && ev.pointerId === dragPtr) {
      // ghost over the board while the pointer is still on the dock is not
      // visible; forward to board coordinates anyway so entering the board flows
      const br = B.cv.getBoundingClientRect();
      snapGhost(ev.clientX - br.left, ev.clientY - br.top);
    }
  });
  el.addEventListener('pointerup', ev => {
    if (wallState === 'DRAGGING' && ev.pointerId === dragPtr) { onBoardPointerUp(ev); }
    else if (performance.now() - el._downAt < 300) armWall(o);   // click = arm
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
  if (isMyTurn && legalPawn.has(dispCell)) {
    const engCell = B.dispPawnToEng(Math.floor(dispCell / 9), dispCell % 9);
    if (W.applyPawn(engCell)) {
      B.selected = -1; B.dots = [];
      sound('move'); haptic(10);
      afterHumanMove();
    }
    return;
  }
  if (B.selected >= 0) { B.selected = -1; buildLegalSets(); B.render(); }
}

// ===================== 10. sound & haptics ==============================
// Synth packs (plan 17.4): each event maps to a small parameter table per
// pack, so adding a pack is data, not code. No audio files anywhere.
const SOUND_PACKS = {
  wood: {
    move:   [{ f: 180, dur: 40, type: 'sine', g: -10 }],
    wall:   [{ f: 90, dur: 90, type: 'sine', g: -8 }, { f: 320, dur: 30, type: 'triangle', g: -16 }],
    illegal:[{ f: 140, dur: 70, type: 'square', g: -12 }],
    arm:    [{ f: 900, dur: 18, type: 'sine', g: -14 }],
    hint:   [{ f: 660, dur: 60, type: 'triangle', g: -14 }],
    clock:  [{ f: 1000, dur: 25, type: 'sine', g: -16 }],
    end:    [{ f: 523, dur: 120, type: 'triangle', g: -8 }, { f: 659, dur: 120, type: 'triangle', g: -8, at: 120 }, { f: 784, dur: 180, type: 'triangle', g: -8, at: 240 }],
    loss:   [{ f: 392, dur: 200, type: 'triangle', g: -10 }, { f: 294, dur: 300, type: 'triangle', g: -10, at: 200 }],
  },
  modern: {
    move:   [{ f: 520, dur: 35, type: 'sine', g: -14 }],
    wall:   [{ f: 240, dur: 60, type: 'sine', g: -12 }, { f: 480, dur: 45, type: 'sine', g: -18, at: 30 }],
    illegal:[{ f: 220, dur: 80, type: 'sawtooth', g: -20 }],
    arm:    [{ f: 880, dur: 15, type: 'sine', g: -18 }],
    hint:   [{ f: 740, dur: 50, type: 'sine', g: -16 }],
    clock:  [{ f: 1200, dur: 18, type: 'sine', g: -20 }],
    end:    [{ f: 660, dur: 100, type: 'sine', g: -12 }, { f: 880, dur: 140, type: 'sine', g: -12, at: 100 }],
    loss:   [{ f: 330, dur: 160, type: 'sine', g: -14 }, { f: 250, dur: 260, type: 'sine', g: -14, at: 150 }],
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
  g.gain.setValueAtTime(vol, t0);
  g.gain.exponentialRampToValueAtTime(.0001, t0 + p.dur / 1000);
  o.connect(g); g.connect(a.destination);
  o.start(t0); o.stop(t0 + p.dur / 1000 + .02);
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
$('overlay').addEventListener('click', e => { if (e.target.id === 'overlay') closeModal(); });
$('overlay').addEventListener('click', e => { if (e.target.dataset.close) closeModal(); });

// Small helpers shared by the settings / IO / editor surfaces.
function confirmModal(title, body, onYes) {
  openModal(`<h3>${title} <span class="x" data-close>&#10005;</span></h3>
    <p style="color:var(--txt2);font-size:.68rem;margin-bottom:12px">${body}</p>
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
  const lv = LEVELS[S.level];
  $('lvlName').textContent = lv.label;
  $('lvlChip').querySelector('.dot').style.background = lv.color;
  $('oppLvl').textContent = '\u00b7 ' + lv.label;
  $('mOppLvl').textContent = '\u00b7 ' + lv.label;
}

function modalSettings() {
  const seg = (key, opts, labels) =>
    `<div class="seg" data-set="${key}">${opts.map((o, i) =>
      `<button data-v="${o}" class="${String(S[key]) === String(o) ? 'on' : ''}">${labels[i]}</button>`).join('')}</div>`;
  const presetNames = [['classic', 'Classic'], ['premiumDark', 'Premium Dark'], ['highContrast', 'High Contrast'], ['minimal', 'Minimal']];
  openModal(`<h3>SETTINGS <span class="x" data-close>&#10005;</span></h3>
    <div class="card"><h4>PRESET</h4>
      <div class="row"><div class="seg" id="presetSeg">
        ${presetNames.map(([k, l]) => `<button data-p="${k}" class="${S.preset === k ? 'on' : ''}">${l}</button>`).join('')}
        <button class="${S.preset === 'custom' ? 'on' : ''}" disabled>Custom</button>
      </div></div>
    </div>
    <div class="card"><h4>APPEARANCE</h4>
      <div class="row"><label>UI theme</label>${seg('ui', ['dark', 'light', 'auto'], ['Dark', 'Light', 'Auto'])}</div>
      <div class="row"><label>Board</label><div class="swatches" id="setBoards">
        ${BOARD_THEMES.map(t => `<button class="swatch ${S.board === t ? 'on' : ''}" data-b="${t}">
          <canvas width="30" height="24" data-mini="${t}"></canvas></button>`).join('')}</div></div>
      <div class="row"><label>Pawn style</label><div class="swatches" id="setPawns">
        ${PAWN_STYLES.map(t => `<button class="swatch ${S.pawn === t ? 'on' : ''}" data-b="${t}" title="${t}">&#9823;</button>`).join('')}</div></div>
      <div class="row"><label>Pawn size</label>${seg('pawnSize', ['small', 'regular', 'large'], ['S', 'M', 'L'])}</div>
      <div class="row"><label>Pawn shadow</label>${seg('pawnShadow', ['off', 'soft', 'deep'], ['Off', 'Soft', 'Deep'])}</div>
      <div class="row"><label>Distinct shapes</label>${seg('distinctShapes', [true, false], ['On', 'Off'])}</div>
      <div class="row"><label>Accent</label><div class="swatches" id="setAccents">
        ${Object.keys(ACCENTS).map(a => `<button class="swatch ${S.accent === a ? 'on' : ''}" data-a="${a}" style="background:${ACCENTS[a][0]}"></button>`).join('')}
        <input type="color" id="accentPick" value="#c8a84b" style="width:34px;height:26px;background:none;border:none" title="Custom accent">
      </div></div>
      <div class="row"><label>Text size</label>${seg('fs', [1, 1.12, 1.25], ['Normal', 'Large', 'Larger'])}</div>
      <div class="row"><label>Density</label>${seg('density', ['comfortable', 'compact'], ['Comfortable', 'Compact'])}</div>
    </div>
    <div class="card"><h4>BOARD</h4>
      <div class="row"><label>Frame</label>${seg('frame', ['none', 'hairline', 'gilded', 'beveled'], ['None', 'Hair', 'Gild', 'Bevel'])}</div>
      <div class="row"><label>Wall finish</label>${seg('wallFinish', ['flat', 'beveled', 'glossy', 'etched'], ['Flat', 'Bev', 'Gloss', 'Etch'])}</div>
      <div class="row"><label>Cell surface</label>${seg('cellSep', ['grooves', 'flat', 'inlaid'], ['Groove', 'Flat', 'Inlay'])}</div>
      <div class="row"><label>Coordinates</label>${seg('coords', ['off', 'edges', 'all'], ['Off', 'Edges', 'All'])}</div>
      <div class="row"><label>Board scale</label>${seg('boardScale', [0.88, 0.94, 1], ['88%', '94%', '100%'])}</div>
      <div class="row"><label>Legal dots</label>${seg('dots', [true, false], ['On', 'Off'])}</div>
      <div class="row"><label>Path hints</label>${seg('paths', [false, true], ['Off', 'On'])}</div>
      <div class="row"><label>Last move mark</label>${seg('lastMove', [true, false], ['On', 'Off'])}</div>
      <div class="row"><label>Eval bar</label>${seg('evalBar', [true, false], ['On', 'Off'])}</div>
    </div>
    <div class="card"><h4>SOUND &amp; HAPTICS</h4>
      <div class="row"><label>Sound pack</label>
        <select class="sel" id="packSel">${Object.keys(SOUND_PACKS).map(p => `<option value="${p}" ${S.soundPack === p ? 'selected' : ''}>${p}</option>`).join('')}</select>
        <button class="btn" id="packTest" style="padding:6px 10px">Test</button></div>
      <div class="row"><label>Sound</label>${seg('sound', [true, false], ['On', 'Off'])}</div>
      <div class="row"><label>Volume</label><input type="range" id="volSlider" min="0" max="100" value="${Math.round(S.volume * 100)}" style="flex:1.2"></div>
      <div class="row"><label>Events</label><div class="seg" style="flex-wrap:wrap" id="evSeg">
        ${Object.keys(S.soundEvents).map(k => `<button data-ev="${k}" class="${S.soundEvents[k] ? 'on' : ''}" style="font-size:.52rem;padding:5px 7px">${k}</button>`).join('')}</div></div>
      <div class="row"><label>Haptics</label>${seg('haptics', ['full', 'light', 'off'],
        navigator.vibrate ? ['Full', 'Light', 'Off'] : ['N/A', 'N/A', 'Off'])}</div>
    </div>
    <div class="card"><h4>MOTION &amp; INPUT</h4>
      <div class="row"><label>Animations</label>${seg('anim', ['full', 'reduced', 'off'], ['Full', 'Reduced', 'Off'])}</div>
      <div class="row"><label>Anim speed</label>${seg('animSpeed', [0.5, 1, 1.5], ['0.5x', '1x', '1.5x'])}</div>
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
    </div>`);
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
  $('accentPick').oninput = e => { setOpt('accent', e.target.value); };
  $('packSel').onchange = e => { setOpt('soundPack', e.target.value); sound('wall'); };
  $('packTest').onclick = () => { acArmed = true; sound('wall'); };
  $('volSlider').onchange = e => { setOpt('volume', +e.target.value / 100); };
  $('volSlider').oninput = e => { S.volume = +e.target.value / 100; };
  box.querySelectorAll('#evSeg button').forEach(b => b.onclick = () => {
    const k = b.dataset.ev;
    S.soundEvents[k] = !S.soundEvents[k];
    saveSettings();
    b.classList.toggle('on', S.soundEvents[k]);
  });
  $('btnSetExport').onclick = () => downloadText('zq-settings.json', JSON.stringify(S, null, 2), 'application/json');
  $('btnSetImport').onclick = () => pickFile('.json', txt => {
    try {
      const parsed = JSON.parse(txt);
      localStorage.setItem('zq.settings', JSON.stringify({ ...DEFAULTS, ...parsed }));
      loadSettings(); applySettings(); closeModal();
      toast('ok', 'Settings imported');
    } catch (e) { toast('err', 'Invalid settings file'); }
  });
  $('btnResetAll').onclick = () => confirmModal('Reset all settings?', 'This restores every option to its default.',
    () => { localStorage.removeItem('zq.settings'); S = JSON.parse(JSON.stringify(DEFAULTS)); applySettings(); modalSettings(); toast('ok', 'Settings reset'); });
}

function applySettings() {
  const h = document.documentElement, ds = h.dataset;
  ds.ui = S.ui === 'auto' ? (matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark') : S.ui;
  ds.board = S.board; ds.pawn = S.pawn;
  ds.coords = S.coords;
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
const MENU_ITEMS = [
  ['New game', () => { modalNewGame(); }],
  ['Flip board', doFlip],
  null,
  ['Copy QFEN', () => copyText(W.qfenExportStr(), 'QFEN')],
  ['Paste QFEN', () => openTextIO('qfen')],
  ['Copy game', () => copyText(qgnExport(), 'Game text')],
  ['Paste game', () => openTextIO('qgn')],
  ['Open .qgn file', () => pickFile('.qgn,.txt,.qfen', t => routeImport(t))],
  ['Save .qgn file', () => downloadText(gameFileName(), qgnExport(), 'application/x-zquoridor-game')],
  ['Copy link', () => copyText(shareLink(), 'Link')],
  ['Recent games', showRecentGames],
  ['Export image', exportImageModal],
  null,
  ['Resign', resignConfirm],
];
function buildMenu() {
  const drop = $('menuDrop');
  drop.innerHTML = MENU_ITEMS.map(it => it === null ? '<hr>' :
    `<button class="${it[0] === 'Resign' ? 'danger' : ''}" role="menuitem">${it[0]}</button>`).join('');
  const btns = drop.querySelectorAll('button');
  let bi = 0;
  MENU_ITEMS.forEach(it => { if (it) { btns[bi].onclick = () => { closeMenu(); it[1](); }; bi++; } });
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
  lines.push(`[Player1 "Zquoridor ${LEVELS[S.level].label}"]`);
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
      <span style="flex:1;font-size:.62rem">#${i + 1} ${when} ${res} (${nMoves}m)</span>
      <button class="btn" data-rload="${i}" style="padding:4px 8px;font-size:.56rem">Load</button>
      <button class="btn" data-rcopy="${i}" style="padding:4px 8px;font-size:.56rem">Copy</button>
      <button class="btn danger" data-rdel="${i}" style="padding:4px 8px;font-size:.56rem">&#10005;</button>
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
    footer: $('exFooter').checked ? { name0: 'Player 0', name1: 'Zquoridor ' + LEVELS[S.level].label } : null,
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
function refreshHudFromScratch() {
  const wl = [W.scrWallsLeft(0), W.scrWallsLeft(1)];
  pips($('pips0'), wl[0]); pips($('pips1'), wl[1]);
  pips($('mPips0'), wl[0]); pips($('mPips1'), wl[1]);
  $('walls0').textContent = wl[0]; $('walls1').textContent = wl[1];
  $('mWalls0').textContent = wl[0]; $('mWalls1').textContent = wl[1];
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
  if (e.key === 'r' || e.key === 'R') { armedO = 1 - armedO; if (B.ghost) { B.ghost.o = armedO; B.render(); } return; }
  if (e.key === 'f' || e.key === 'F') { doFlip(); return; }
  if (e.key === 'Enter' && wallState === 'PENDING') { commitGhost(B.ghost); return; }
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
        sound('move'); haptic(10);
        afterHumanMove();
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
$('btnUndo').onclick = takeback;
$('btnHint').onclick = showHint;
$('btnNew').onclick = modalNewGame;
$('lvlChip').onclick = modalNewGame;
$('btnSettings').onclick = modalSettings;
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
$('raceMeter').onclick = () => { S.paths = !S.paths; togglePaths(); saveSettings(); };
$('btnPaths').onclick = () => { S.paths = !S.paths; togglePaths(); saveSettings(); };
function togglePaths() {
  if (!S.paths) { B.paths = null; B.render(); return; }
  const mk = pl => {
    const cells = []; let cell = W.pawn(pl); cells.push(cell);
    let guard = 0;
    while (guard++ < 40) {
      const r = Math.floor(cell / 9), goal = pl === 0 ? 8 : 0;
      if (r === goal) break;
      // greedy step: pick the neighbour (open edge) closest to goal
      let bestC = null, bestD = 99;
      for (const [dr, dc] of [[-1, 0], [1, 0], [0, -1], [0, 1]]) {
        const nr = r + dr, nc = c0(cell) + dc;
        if (nr < 0 || nr > 8 || nc < 0 || nc > 8) continue;
        const ncCell = nr * 9 + nc;
        const dd = Math.abs(nr - goal);
        if (dd < bestD && edgeOpen(cell, ncCell)) { bestD = dd; bestC = ncCell; }
      }
      if (bestC == null) break;
      cells.push(bestC); cell = bestC;
    }
    return { cells, color: getComputedStyle(document.documentElement).getPropertyValue('--p' + pl).trim() };
  };
  function c0(cell) { return cell % 9; }
  function edgeOpen(a, b) {
    const ar = Math.floor(a / 9), ac = a % 9, br = Math.floor(b / 9), bc2 = b % 9;
    if (ar === br) {  // horizontal step blocked by V walls at col min(ac,bc)
      const cmin = Math.min(ac, bc2), r = ar;
      const bit1 = r > 0 ? W.wallVBit((r - 1) * 8 + cmin) : 0;
      const bit2 = r < 8 ? W.wallVBit(r * 8 + cmin) : 0;
      return !(bit1 || bit2);
    }
    const rmin = Math.min(ar, br), c = ac;
    const bit1 = c > 0 ? W.wallHBit(rmin * 8 + c - 1) : 0;
    const bit2 = c < 8 ? W.wallHBit(rmin * 8 + c) : 0;
    return !(bit1 || bit2);
  }
  B.paths = [mk(0), mk(1)]; B.render();
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
  log.innerHTML = html || 'No moves yet.';
  log.querySelectorAll('.mlMv').forEach(el =>
    el.onclick = () => navGo(+el.dataset.ply));
  const curEl = log.querySelector('.mlMv.cur');
  if (curEl) curEl.scrollIntoView({ block: 'nearest' });
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
    $('anLines').innerHTML = '';
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
  const human = humanSide === 0 ? abs : -abs;
  const share = 50 + 50 * Math.tanh(human / 400);
  $('evalFill').style.height = share.toFixed(1) + '%';
}

// ---- eval graph --------------------------------------------------------
function drawGraph() {
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
  const p0s = cs.getPropertyValue('--p0-soft').trim();
  const p1s = cs.getPropertyValue('--p1-soft').trim();
  // sample points: ply -> wp (side-0 view); missing plies carry the last value
  const pts = [];
  let lastWp = .5;
  for (let i = 0; i <= n; i++) {
    if (i > 0 && AN.scores[i - 1] != null)
      lastWp = wpOf(absScoreAt(AN.scores[i - 1], i - 1));
    pts.push(lastWp);
  }
  const X = i => n ? (i / n) * (w - 6) + 3 : 3;
  const Y = wpv => h - 4 - wpv * (h - 8);
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

// ===================== 13. boot ========================================
function boot() {
  loadSettings();
  B = new QBoard($('board'));
  window.__qb = B;   // test/debug handle
  B.onChange = () => { $('evalStrip').style.height = (B.cssSide - 4) + 'px'; };
  applySettings();
  W.newGame();
  humanSide = S.side;
  syncAll();
  startClock();
  setStatus(humanSide === 0 ? 'Your move' : 'Zquoridor starts');
  // deferred re-fits: the first fit() may run before final layout; re-measure
  // once the flex layout settles, and on any viewport resize.
  [200, 600, 1500].forEach(t => setTimeout(() => B.fit(), t));
  window.addEventListener('resize', () => { B.fit(); drawGraph(); });

  // analysis controls (plan section 5.6)
  $('anEngBtn').onclick = anToggle;
  $('anDepth').onchange = anRestart;
  $('anPvCount').onchange = anRestart;
  $('anBlunderBtn').onclick = blunderCheck;
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
  $('movesChip').onclick = showRecentGames;
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
