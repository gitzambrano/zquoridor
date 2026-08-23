// worker.js -- analysis worker for the Zquoridor premium GUI (plan sections
// 12 / 5.6). Owns a second WASM module instance so multi-line analysis and
// the infinite loop never block the main thread.
//
// Protocol (all requests self-contained -- the worker owns no game state):
//   main -> worker : { id, cmd: 'analyze', qfen?, moves?, depth, timeMs, lines }
//   worker -> main : { type: 'ready' } | { id, type: 'result', ... }
//                  | { id, type: 'error', msg }
//
// A request replays the game line onto the scratch position (QFEN root
// optional, then packed moves exactly like qr_ply_* exports them) and runs
// the same qr_analyze the main thread would run. Results are plain data.
'use strict';

importScripts('zquoridor.js');

let M = null;
const F = {};

function readCStr(buf) {
  const u8 = M.HEAPU8; let s = '';
  for (let i = 0; i < 640; i++) { const b = u8[buf + i]; if (!b) break; s += String.fromCharCode(b); }
  return s;
}
function lastErr() {
  const buf = M._malloc(640);
  try { M._qr_last_error(buf, 640); return readCStr(buf); } finally { M._free(buf); }
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
    const a = (packed >> 16) & 255, b = (packed >> 8) & 255, c = packed & 255;
    const ok = isWall ? M._qr_scr_apply_wall(a, b, c) : M._qr_scr_apply_pawn(a);
    if (!ok) return 'replay failed at a recorded move';
  }
  return null;
}

self.onmessage = ev => {
  const req = ev.data;
  if (!M) { postMessage({ id: req.id, type: 'error', msg: 'module not ready' }); return; }
  try {
    const err = replayIntoScratch(req.qfen, req.moves);
    if (err) { postMessage({ id: req.id, type: 'error', msg: err }); return; }
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

ZquoridorModule().then(m => {
  M = m;
  let nnue = false;
  try { nnue = !!m._qr_load_nnue_weights('/data/nnue/nnue_weights_int8.bin'); } catch (e) {}
  postMessage({ type: 'ready', nnue });
}).catch(e => postMessage({ type: 'fatal', msg: String(e) }));
