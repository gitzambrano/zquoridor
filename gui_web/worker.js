// worker.js -- background engine and analysis worker for the Zquoridor GUI.
//
// The worker owns one WASM module instance. Game requests keep a persistent
// live position so MCAB can reuse its tree between moves. After an engine move,
// short ponder slices search the opponent-to-move root while the human thinks.
// Each slice yields back to the worker event loop, so an incoming move waits at
// most one short slice and the browser UI remains on the main thread.
//
// Protocol:
//   main -> worker : { id, cmd: 'analyze', qfen?, moves?, depth, timeMs, lines }
//                  | { id, cmd: 'bestmove', moves?, depth, timeMs }
//   worker -> main : { type: 'ready' } | { id, type: 'result', ... }
//                  | { id, type: 'error', msg }
'use strict';

importScripts('zquoridor.js');

let M = null;
let liveMoves = [];
let ponderTimer = null;
let ponderEpoch = 0;
let ponderSpentMs = 0;

const PONDER_CHUNK_MS = 24;
const PONDER_MAX_MS = 1200;

function readCStr(buf) {
  const u8 = M.HEAPU8; let s = '';
  for (let i = 0; i < 640; i++) {
    const b = u8[buf + i];
    if (!b) break;
    s += String.fromCharCode(b);
  }
  return s;
}

function lastErr() {
  const buf = M._malloc(640);
  try { M._qr_last_error(buf, 640); return readCStr(buf); }
  finally { M._free(buf); }
}

function withCStr(str, fn) {
  const bytes = new TextEncoder().encode(String(str) + '\0');
  const p = M._malloc(bytes.length);
  try { M.HEAPU8.set(bytes, p); return fn(p); }
  finally { M._free(p); }
}

function importQfenRoot(qfen) {
  const bytes = new TextEncoder().encode(String(qfen) + '\0');
  const p = M._malloc(bytes.length);
  let code;
  try { M.HEAPU8.set(bytes, p); code = M._qr_qfen_import_scratch(p); }
  finally { M._free(p); }
  return code === 0 ? null : ('root QFEN rejected: ' + lastErr());
}

function replayIntoScratch(qfen, moves) {
  M._qr_scratch_reset();
  if (qfen) {
    const err = importQfenRoot(qfen);
    if (err) return err;
  }
  for (const packed of moves || []) {
    const isWall = (packed >> 24) & 1;
    const a = (packed >> 16) & 255;
    const b = (packed >> 8) & 255;
    const c = packed & 255;
    const ok = isWall ? M._qr_scr_apply_wall(a, b, c) : M._qr_scr_apply_pawn(a);
    if (!ok) return 'replay failed at a recorded move';
  }
  return null;
}

function applyPackedLive(packed) {
  const isWall = (packed >> 24) & 1;
  const a = (packed >> 16) & 255;
  const b = (packed >> 8) & 255;
  const c = packed & 255;
  return isWall ? M._qr_apply_wall_move(a, b, c) : M._qr_apply_pawn_move(a);
}

// Keep the live game when the requested history extends the current one.
// A takeback, a new game, or any divergent history resets the module and tree.
function syncIntoLive(moves) {
  const target = Array.from(moves || [], x => x | 0);
  const extendsCurrent =
    liveMoves.length <= target.length &&
    liveMoves.every((move, i) => move === target[i]);

  if (!extendsCurrent) {
    M._qr_new_game();
    liveMoves = [];
  }

  for (let i = liveMoves.length; i < target.length; i++) {
    if (!applyPackedLive(target[i])) return 'replay failed at a recorded move';
    liveMoves.push(target[i]);
  }
  return null;
}

function cancelPonder() {
  ponderEpoch++;
  ponderSpentMs = 0;
  if (ponderTimer !== null) {
    clearTimeout(ponderTimer);
    ponderTimer = null;
  }
}

function startPonder() {
  if (!M || typeof M._qr_engine_ponder !== 'function') return;
  if (typeof M._qr_mcab_active === 'function' && !M._qr_mcab_active()) return;

  const epoch = ++ponderEpoch;
  ponderSpentMs = 0;

  const step = () => {
    ponderTimer = null;
    if (!M || epoch !== ponderEpoch || ponderSpentMs >= PONDER_MAX_MS) return;

    const slice = Math.max(1, Math.min(PONDER_CHUNK_MS,
      Math.ceil(PONDER_MAX_MS - ponderSpentMs)));
    const t0 = performance.now();
    try {
      M._qr_engine_ponder(24, slice);
    } catch (e) {
      return;
    }
    ponderSpentMs += performance.now() - t0;

    if (epoch === ponderEpoch && ponderSpentMs < PONDER_MAX_MS) {
      ponderTimer = setTimeout(step, 0);
    }
  };

  ponderTimer = setTimeout(step, 0);
}

function handleBestMove(req) {
  const err = syncIntoLive(req.moves);
  if (err) {
    postMessage({ id: req.id, type: 'error', msg: err });
    return;
  }

  const t0 = performance.now();
  const ok = M._qr_engine_move(Math.max(1, req.depth | 0),
                               Math.max(20, req.timeMs | 0));
  const ms = performance.now() - t0;
  if (!ok) {
    postMessage({ id: req.id, type: 'error', msg: 'engine returned no move' });
    return;
  }

  const isWall = M._qr_last_move_is_wall();
  const packed = (isWall ? (1 << 24) : 0) |
                 (M._qr_last_move_a() << 16) |
                 (M._qr_last_move_b() << 8) |
                 M._qr_last_move_c();
  liveMoves.push(packed);
  postMessage({ id: req.id, type: 'result',
                move: packed, score: M._qr_last_move_eval(), ms });

  // The live module now sits at the human-to-move root. Search it in bounded
  // slices until a new request arrives or the per-turn ponder cap is reached.
  startPonder();
}

self.onmessage = ev => {
  const req = ev.data;
  if (!M) {
    postMessage({ id: req.id, type: 'error', msg: 'module not ready' });
    return;
  }

  try {
    if (req.cmd === 'init') return;

    // Any real request has priority over background pondering. The current
    // slice may finish first, but it is bounded by PONDER_CHUNK_MS.
    cancelPonder();

    if (req.cmd === 'bestmove') {
      handleBestMove(req);
      return;
    }

    const err = replayIntoScratch(req.qfen, req.moves);
    if (err) {
      postMessage({ id: req.id, type: 'error', msg: err });
      return;
    }

    const t0 = performance.now();
    const got = M._qr_analyze(req.depth | 0, Math.max(50, req.timeMs | 0),
                              Math.max(1, Math.min(5, req.lines | 0)));
    const ms = performance.now() - t0;
    const lines = [];
    for (let i = 0; i < got; i++) {
      const pv = [];
      const len = M._qr_an_line_len(i);
      for (let j = 0; j < len; j++) pv.push(M._qr_an_line_move(i, j));
      lines.push({ score: M._qr_an_line_score(i), pv });
    }
    postMessage({ id: req.id, type: 'result', lines,
                  nodes: M._qr_an_nodes(), depth: M._qr_an_depth(), ms });
  } catch (e) {
    postMessage({ id: req.id, type: 'error', msg: String(e && e.message || e) });
  }
};

function bootModule(opts) {
  ZquoridorModule(opts || {}).then(m => {
    M = m;
    let nnue = false;
    try {
      nnue = !!withCStr('/data/nnue/nnue_weights_int8.bin',
                        p => m._qr_load_nnue_weights(p));
    } catch (e) {}

    // Initialize the live game explicitly. qr_new_game() also builds the
    // legal-move list, repetition history, scratch state and clears the
    // persistent MCAB tree. Without it, the first bestmove replay can see an
    // empty legal-move list even though liveMoves itself is empty.
    m._qr_new_game();
    liveMoves = [];
    postMessage({ type: 'ready', nnue });
  }).catch(e => postMessage({ type: 'fatal', msg: String(e) }));
}

if (typeof __STANDALONE_WORKER__ !== 'undefined' && __STANDALONE_WORKER__) {
  self.addEventListener('message', function _onInit(ev) {
    if (ev.data && ev.data.cmd === 'init') {
      self.removeEventListener('message', _onInit);
      const args = { wasmBinary: ev.data.wasmBinary };
      if (ev.data.dataBytes) args.getPreloadedPackage = () => ev.data.dataBytes;
      bootModule(args);
    }
  });
} else {
  bootModule();
}
