// app.js -- Zquoridor premium GUI logic (plan gui-premium.md).
// Sections: 1 constants - 2 wasm bindings - 3 settings - 4 state -
// 5 hud/clocks/race/eval - 6 board bridge - 7 wall input - 8 pawn input -
// 9 play flow - 10 sound/haptics - 11 modals/toasts - 12 keyboard - 13 boot.
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
const BOARD_THEMES = ['obsidian', 'walnut', 'ivory', 'slate', 'emerald', 'parchment'];
const PAWN_STYLES = ['disc', 'pillar', 'crown', 'rune'];
const FILES = 'abcdefghi';

// ===================== 2. wasm bindings ================================
let W = null;   // wrapped exports
function bindEngine(m) {
  const c = name => {
    if (typeof m[name] !== 'function') throw new Error('export missing: ' + name);
    return (...args) => m[name](...args);
  };
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
  };
}

// ===================== 3. settings =====================================
const DEFAULTS = {
  ui: 'dark', board: 'obsidian', pawn: 'disc',
  coords: 'edges', dots: true, paths: false,
  level: 'knight', clockMode: 'none', baseMin: 5, incSec: 3, side: 0,
  confirmWalls: null, touchOffset: null, sound: true, volume: .6,
  haptics: 'full',
};
let S = { ...DEFAULTS };
function loadSettings() {
  try {
    const raw = localStorage.getItem('zq.settings');
    if (raw) S = { ...DEFAULTS, ...JSON.parse(raw) };
  } catch (e) { /* corrupt blob -> defaults */ }
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

// ===================== 4. state ========================================
let B = null;                    // QBoard
let history = [];                // [{notation, side}]
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
  $('btnUndo').classList.toggle('off', history.length === 0 || engineThinking);
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
function syncFromEngine() {
  const pw = [W.pawn(0), W.pawn(1)];
  const wh = [], wv = [];
  for (let s = 0; s < 64; s++) { wh.push(W.wallHBit(s)); wv.push(W.wallVBit(s)); }
  B.flipped = (humanSide === 1) !== (location.hash.includes('noflip'));
  B.setData(pw, wh, wv, lastMoveInfo);
  buildLegalSets();
  refreshHud();
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
let pendingEngine = null;
function afterHumanMove() {
  history.push(true);
  updateMovesChip();
  checkEnd();
  if (!gameOver) { engineThinking = true; refreshHud(); setStatus('Zquoridor is thinking...');
    setTimeout(engineTurn, 120); }
}
function engineTurn() {
  const lv = LEVELS[S.level];
  const pre = snapState();
  const ok = W.engineMove(24, lv.ms);
  engineThinking = false;
  if (!ok) { syncAll(); return; }
  history._snapBefore = pre;
  rebuildLastFromEngine();
  setEval(-W.lastEval());          // eval is from the engine's perspective
  history.push(false);
  updateMovesChip();
  sound('move'); haptic(10);
  syncAll();
  checkEnd();
  if (!gameOver) setStatus('Your move');
}
function rebuildLastFromEngine() {
  const cur = snapState();
  lastMoveInfo = diffMoves(history._snapBefore, cur);
  history._snapBefore = cur;
}
function snapState() {
  const wh = [], wv = [];
  for (let s = 0; s < 64; s++) { wh.push(W.wallHBit(s)); wv.push(W.wallVBit(s)); }
  return { pawn: [W.pawn(0), W.pawn(1)], wh, wv };
}
function diffMoves(a, b) {
  if (!a) return null;
  for (let s = 0; s < 64; s++) {
    if (a.wh[s] !== b.wh[s]) { const [o, r, c] = B.engWallToDisp(0, Math.floor(s / 8), s % 8); void o; return { type: 'wall', o: 0, r: Math.floor(s / 8), c: s % 8 }; }
    if (a.wv[s] !== b.wv[s]) return { type: 'wall', o: 1, r: Math.floor(s / 8), c: s % 8 };
  }
  for (let pl = 0; pl < 2; pl++) if (a.pawn[pl] !== b.pawn[pl]) {
    const disp = B.engPawnToDisp(b.pawn[pl]);
    return { type: 'pawn', r: Math.floor(disp / 9), c: disp % 9, from: B.engPawnToDisp(a.pawn[pl]) };
  }
  return null;
}
function checkEnd() {
  const w = W.winner();
  if (w !== -1) {
    gameOver = true;
    const youWon = w === humanSide;
    setStatus(youWon ? 'You won - goal reached' : 'Zquoridor won');
    toast(youWon ? 'ok' : 'err', youWon ? 'Victory' : 'Defeat');
    sound('end'); haptic(youWon ? [20, 60, 20, 60, 40] : [60]);
    refreshHud(); return true;
  }
  if (W.isDraw()) { gameOver = true; setStatus('Draw by repetition'); return true; }
  return false;
}
function syncAll() { syncFromEngine(); if (!checkEndQuiet()) {} }
function checkEndQuiet() { const w = W.winner(); return w !== -1 || W.isDraw(); }
function newGame() {
  W.newGame(); history = []; gameOver = false; engineThinking = false;
  humanSide = S.side;
  lastMoveInfo = null; history._snapBefore = snapState();
  clearGhost(); startClock();
  setStatus(humanSide === 0 ? 'Your move' : 'Zquoridor starts');
  syncAll();
  if (W.turn() !== humanSide) { engineThinking = true; refreshHud(); setTimeout(engineTurn, 150); }
}
function updateMovesChip() { $('movesChip').textContent = 'Moves ' + Math.ceil(history.length / 2); }

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
function legalSlotBitmap(o) {
  const on = document.createElement('canvas');
  // paint legal anchors as gold at low opacity via a dedicated overlay canvas
  return null; // overlay drawn directly in showLegalSlots
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
  let a = B.nearestAnchor(p.x, p.y);
  if (!a || a.dist > .85 * B.U) { B.ghost = null; setStatus('Drag onto a groove between cells'); return; }
  let [o, r, c] = [armedO, a.r, a.c];
  const [eo, er, ec] = B.dispWallToEng(o, r, c);
  let st = legalWall[o * 64 + r * 8 + c] ? 'ok' : null;
  if (!st) {   // magnetic assist: nearest legal anchor within .60U
    let best = null;
    for (let dr = -1; dr <= 1; dr++) for (let dc = -1; dc <= 1; dc++) {
      const rr2 = r + dr, cc2 = c + dc;
      if (rr2 < 0 || rr2 > 7 || cc2 < 0 || cc2 > 7) continue;
      if (!legalWall[o * 64 + rr2 * 8 + cc2]) continue;
      const ctr = B.anchorCenter(rr2, cc2);
      const dist = Math.hypot(p.x - ctr.x, p.y - ctr.y);
      if (dist <= .60 * B.U && (!best || dist < best.dist)) best = { r: rr2, c: cc2, dist };
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

function onBoardPointerDown(ev) {
  if (gameOver || engineThinking) return;
  const pt = boardPoint(ev);
  const cell = B.pointToCell(pt.x, pt.y);
  if (wallState === 'ARMED') { wallState = 'DRAGGING'; dragPtr = ev.pointerId; snapGhost(pt.x, pt.y); ev.preventDefault(); return; }
  if (cell) {
    const near = B.nearestAnchor(pt.x, pt.y);
    if (!near || near.dist > .55 * B.U || !nearAnchorHasLegal(near)) { pawnDown(cell, pt); return; }
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
  const [eo, er, ec] = B.dispWallToEng(gh.o, gh.r, gh.c);
  if (W.applyWall(eo, er, ec)) {
    lastMoveInfo = { type: 'wall', o: eo, r: er, c: ec };
    history._snapBefore = snapState();
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
  const myPawnDisp = B.engPawnToDisp(W.pawn(humanSide));
  const isMyTurn = W.turn() === humanSide && !engineThinking;
  if (dispCell === myPawnDisp && isMyTurn) { B.selected = dispCell; B.dots = [...legalPawn]; B.render(); return; }
  if (isMyTurn && legalPawn.has(dispCell)) {
    const engCell = B.flipped ? (8 - Math.floor(dispCell / 9)) * 9 + dispCell % 9
                              : Math.floor(dispCell / 9) * 9 + dispCell % 9;
    if (W.applyPawn(engCell)) {
      lastMoveInfo = { type: 'pawn', r: Math.floor(dispCell / 9), c: dispCell % 9 };
      history._snapBefore = snapState();
      B.selected = -1; B.dots = [];
      sound('move'); haptic(10);
      afterHumanMove();
    }
    return;
  }
  if (B.selected >= 0) { B.selected = -1; buildLegalSets(); B.render(); }
}

// ===================== 10. sound & haptics =============================
let AC = null;
function ac() { if (!AC) try { AC = new (window.AudioContext || webkitAudioContext)(); } catch (e) {} return AC; }
function tone(freq, dur, type, gainDb) {
  if (!S.sound) return;
  const a = ac(); if (!a) return;
  const o = a.createOscillator(), g = a.createGain();
  o.type = type || 'sine'; o.frequency.value = freq;
  const vol = Math.pow(10, ((gainDb || -10) / 20)) * S.volume;
  g.gain.setValueAtTime(vol, a.currentTime);
  g.gain.exponentialRampToValueAtTime(.0001, a.currentTime + dur / 1000);
  o.connect(g); g.connect(a.destination);
  o.start(); o.stop(a.currentTime + dur / 1000 + .02);
}
function sound(kind) {
  if (kind === 'move') tone(180, 40, 'sine', -10);
  else if (kind === 'wall') { tone(90, 90, 'sine', -8); tone(320, 30, 'triangle', -16); }
  else if (kind === 'arm') tone(900, 18, 'sine', -14);
  else if (kind === 'illegal') tone(140, 70, 'square', -12);
  else if (kind === 'end') { tone(523, 120, 'triangle', -8); setTimeout(() => tone(659, 120, 'triangle', -8), 120); setTimeout(() => tone(784, 180, 'triangle', -8), 240); }
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
    S.level = b.dataset.lvl; saveSettings(); applyLevelChip();
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
$('overlay').addEventListener('click', e => { if (e.target.dataset.close) closeModal(); });

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
      `<button data-v="${o}" class="${S[key] === o ? 'on' : ''}">${labels[i]}</button>`).join('')}</div>`;
  openModal(`<h3>SETTINGS <span class="x" data-close>&#10005;</span></h3>
    <div class="card"><h4>APPEARANCE</h4>
      <div class="row"><label>UI theme</label>${seg('ui', ['dark', 'light'], ['Dark', 'Light'])}</div>
      <div class="row"><label>Board</label><div class="swatches" id="setBoards">
        ${BOARD_THEMES.map(t => `<button class="swatch ${S.board === t ? 'on' : ''}" data-b="${t}" style="background:var(--frame)">${t.slice(0, 2)}</button>`).join('')}</div></div>
      <div class="row"><label>Pawn style</label>${seg('pawn', PAWN_STYLES, PAWN_STYLES.map(p => p[0].toUpperCase() + p.slice(1)))}</div>
      <div class="row"><label>Coordinates</label>${seg('coords', ['off', 'edges'], ['Off', 'Edges'])}</div>
    </div>
    <div class="card"><h4>BOARD</h4>
      <div class="row"><label>Legal dots</label>${seg('dots', [true, false], ['On', 'Off'])}</div>
      <div class="row"><label>Path hints</label>${seg('paths', [false, true], ['Off', 'On'])}</div>
    </div>
    <div class="card"><h4>INPUT</h4>
      <div class="row"><label>Confirm walls</label>${seg('confirmWalls', [null, true, false], ['Auto', 'On', 'Off'])}</div>
      <div class="row"><label>Touch offset</label>${seg('touchOffset', [null, 'small', 'large', 'off'], ['Auto', 'Small', 'Large', 'Off'])}</div>
      <div class="row"><label>Haptics</label>${seg('haptics', ['full', 'light', 'off'], ['Full', 'Light', 'Off'])}</div>
    </div>
    <div class="card"><h4>SOUND</h4>
      <div class="row"><label>Sound</label>${seg('sound', [true, false], ['On', 'Off'])}</div>
    </div>`);
  $('modalBox').querySelectorAll('[data-set]').forEach(sg => {
    sg.querySelectorAll('button').forEach(b => b.onclick = () => {
      let v = b.dataset.v;
      if (v === 'true') v = true; else if (v === 'false') v = false; else if (v === 'null') v = null;
      S[sg.dataset.set] = v; saveSettings(); applySettings();
      sg.querySelectorAll('button').forEach(x => x.classList.toggle('on', x === b));
    });
  });
  $('setBoards').querySelectorAll('.swatch').forEach(b => b.onclick = () => {
    S.board = b.dataset.b; saveSettings(); applySettings();
    $('setBoards').querySelectorAll('.swatch').forEach(x => x.classList.toggle('on', x === b));
  });
}

function applySettings() {
  const h = document.documentElement;
  h.dataset.ui = S.ui; h.dataset.board = S.board; h.dataset.pawn = S.pawn;
  h.dataset.coords = S.coords; h.dataset.pawnShadow = 'soft';
  applyLevelChip();
  if (B) { B.themeDirty = true; B.fit(); buildLegalSets(); refreshHud(); B.render(); }
}

// ===================== 12. keyboard ====================================
addEventListener('keydown', e => {
  if (e.key === 'Escape') { clearGhost(); B.render(); closeModal(); return; }
  if (e.key === 'h' || e.key === 'H') { armWall(0); return; }
  if (e.key === 'v' || e.key === 'V') { armWall(1); return; }
  if (e.key === 'r' || e.key === 'R') { armedO = 1 - armedO; if (B.ghost) { B.ghost.o = armedO; B.render(); } return; }
  if (e.key === 'f' || e.key === 'F') { doFlip(); return; }
  if (e.key === 'Enter' && wallState === 'PENDING') { commitGhost(B.ghost); return; }
});

// ===================== misc handlers ===================================
function doFlip() { B.flipped = !B.flipped; syncFromEngine(); B.render(); }
$('btnFlip').onclick = doFlip;
$('btnUndo').onclick = () => { toast('info', 'Takeback lands with the analysis iteration'); };
$('btnHint').onclick = () => { toast('info', 'Hint lands with the analysis iteration'); };
$('btnNew').onclick = modalNewGame;
$('lvlChip').onclick = modalNewGame;
$('btnSettings').onclick = modalSettings;
$('logo').onclick = () => openModal(`<h3>ABOUT <span class="x" data-close>&#10005;</span></h3>
  <p style="line-height:1.7;color:var(--txt2)">Zquoridor plays with an NNUE evaluation network
  (354 inputs, hybrid PUCT MCTS over alpha-beta) trained on self-play.
  Place walls to slow your opponent; reach the far row to win.</p>`);
document.querySelectorAll('.tab[data-pane]').forEach(t => t.onclick = () => {
  document.querySelectorAll('.tab[data-pane]').forEach(x => x.classList.toggle('on', x === t));
  document.querySelectorAll('.pane').forEach(p => p.classList.toggle('on', p.id === t.dataset.pane));
});
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

// ===================== 13. boot ========================================
function boot() {
  loadSettings();
  B = new QBoard($('board'));
  B.onChange = () => { $('evalStrip').style.height = (B.cssSide - 4) + 'px'; };
  applySettings();
  W.newGame();
  humanSide = S.side;
  history._snapBefore = snapState();
  syncAll();
  startClock();
  setStatus(humanSide === 0 ? 'Your move' : 'Zquoridor starts');
  // deferred re-fits: first fit() can run before final layout; also refit on resize
  [200, 600, 1500].forEach(t => setTimeout(() => B.fit(), t));
  window.addEventListener('resize', () => B.fit());
  // deferred re-fits: the first fit() may run before final layout; re-measure
  // once the flex layout settles, and on any viewport resize.
  [200,600,1500].forEach(function(t){ setTimeout(function(){ B.fit(); }, t); });
  window.addEventListener('resize', function(){ B.fit(); });

  // demo hook for visual testing: ?demo places a few moves programmatically
  if (location.search.includes('demo')) runDemo();
}
function runDemo() {
  const seq = [
    () => W.applyPawn(B.flipped ? (8 - 1) * 9 + 4 : 1 * 9 + 4),
    () => W.engineMove(8, 120),
    () => W.applyWall(0, 3, 3),
    () => W.engineMove(8, 120),
    () => W.applyPawn(B.flipped ? (8 - 2) * 9 + 4 : 2 * 9 + 4),
  ];
  let i = 0;
  const step = () => {
    if (i >= seq.length) { rebuildLastFromEngine2(); return; }
    seq[i++](); history.push(i % 2 === 1);
    rebuildLastFromEngine2();
    syncAll(); setEval(-W.lastEval());
    updateMovesChip();
    setTimeout(step, 350);
  };
  step();
}
function rebuildLastFromEngine2() {
  const cur = snapState();
  lastMoveInfo = diffMoves(history._snapBefore, cur);
  history._snapBefore = cur;
  syncFromEngine();
}
ZquoridorModule().then(inst => {
  try {
    bindEngine(inst);
    let nnueOk = false;
    try { nnueOk = !!inst._qr_load_nnue_weights('/data/nnue/nnue_weights_int8.bin'); } catch (e3) {}
    boot();
    if (!nnueOk) toast('warn', 'NNUE weights unavailable - heuristic mode');
  } catch (e) {
    document.title="BOOTERR "+(e&&e.stack?e.stack.split("\n")[1]:""+e);
  }
}).catch(e => { var st=document.getElementById("status"); if(st) st.textContent="WASM load failed: "+e; });
