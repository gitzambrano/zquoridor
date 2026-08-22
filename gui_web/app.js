// =====================================================================
// Zquoridor web GUI -- application layer (W4).
//
// Owns: WASM boot, play loop, analysis driver, position editor, settings
// and persistence, sounds, keyboard shortcuts, move log, eval bar, eval
// graph, blunder check, QFEN / game-text modals, toasts.
//
// It does NOT own: the markup and CSS (`style.html`, W2), the board
// renderer (`board.js`, W3) or the engine bindings (`engine_wasm.cpp`, W1).
// Everything this file touches in those three is a documented contract in
// `gui_web/GUI_PLAN.md`; the code here degrades gracefully when a piece of
// the contract is missing so that a partial build is diagnosable instead
// of a blank page.
//
// Plain ES2020, no modules, no build step: the file is loaded as a plain
// <script src> and is also inlined verbatim into the single-file HTML
// build, so it must keep working from file:// with no server.
// =====================================================================

'use strict';

// =====================================================================
// SECTION 1 -- constants and tiny helpers
// =====================================================================

var N = 9;            // board is 9x9 cells
var WS = 8;           // wall slots are 8x8 corridors
var MAX_WALLS = 10;   // walls per player
var PV_MAX = 5;       // multi-PV hard cap (engine clamps to [1,5] too)
var PV_PLIES = 12;    // how many PV plies we are willing to render

function $(id) { return document.getElementById(id); }

function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }

function on(id, ev, fn, opts) {
    var el = $(id);
    if (!el) return null;
    el.addEventListener(ev, fn, opts);
    return el;
}

function setText(id, txt) {
    var el = $(id);
    if (el) el.textContent = txt;
}

function setDisabled(id, dis) {
    var el = $(id);
    if (el) {
        el.disabled = !!dis;
        el.classList.toggle('disabled', !!dis);
    }
}

function setShown(id, shown) {
    var el = $(id);
    if (el) el.style.display = shown ? '' : 'none';
}

function esc(s) {
    return String(s).replace(/[&<>"']/g, function (c) {
        return ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c];
    });
}

// ---- toast -----------------------------------------------------------
// Single transient strip; every soft failure in the app funnels here so a
// broken contract is visible to the user instead of only to the console.
var _toastTimer = null;
function toast(msg, kind) {
    try { console.log('[zq] ' + msg); } catch (e) { /* no console: ignore */ }
    var el = $('toast');
    if (!el) return;
    el.textContent = msg;
    el.classList.remove('ok', 'bad', 'warn');
    if (kind) el.classList.add(kind);
    el.classList.add('show');
    el.style.display = '';
    if (_toastTimer) clearTimeout(_toastTimer);
    _toastTimer = setTimeout(function () {
        el.classList.remove('show');
        el.style.display = 'none';
    }, kind === 'bad' ? 5200 : 2600);
}

// Copy that also works from file:// where the async clipboard API is
// unavailable (insecure context): falls back to a hidden textarea.
function copyText(txt) {
    var done = function () { toast('Copied to clipboard', 'ok'); };
    try {
        if (navigator.clipboard && window.isSecureContext) {
            navigator.clipboard.writeText(txt).then(done, function () { legacyCopy(txt, done); });
            return;
        }
    } catch (e) { /* fall through */ }
    legacyCopy(txt, done);
}

function legacyCopy(txt, done) {
    try {
        var ta = document.createElement('textarea');
        ta.value = txt;
        ta.style.cssText = 'position:fixed;top:0;left:0;opacity:0;font-size:16px;';
        document.body.appendChild(ta);
        ta.focus(); ta.select();
        document.execCommand('copy');
        document.body.removeChild(ta);
        done();
    } catch (e) {
        toast('Could not copy -- select the text manually', 'warn');
    }
}

// =====================================================================
// SECTION 2 -- algebraic notation
//
// Preserved verbatim in behaviour from the v3 GUI. Notation always lives
// in the FIXED board frame, never in the visually mirrored frame, so a
// flipped board never renames a move: column a..i left to right, rank
// 1..9 bottom to top, i.e. engine_row 0 (top, player 0's start side) is
// rank 9 and engine_row 8 is rank 1.
// =====================================================================

function colLetter(c) { return String.fromCharCode(97 + c); }

function pawnNotation(cell) {
    var er = Math.floor(cell / N), ec = cell % N;
    return colLetter(ec) + (N - er);
}

// `r` is the index of the corridor between engine rows r and r+1 (0..7).
// The SOUTH side of that corridor (engine row r+1, further down) is the
// anchor rank of the notation: (N-1-r) = 8-r.
function wallNotation(orientation, r, c) {
    return colLetter(c) + (N - 1 - r) + (orientation === 0 ? 'h' : 'v');
}

// mv: { isWall, a, b, c } -- for a pawn move only `a` (destination cell)
// is meaningful; for a wall a=orientation, b=r, c=c.
function moveNotation(mv) {
    if (!mv) return '';
    return mv.isWall ? wallNotation(mv.a, mv.b, mv.c) : pawnNotation(mv.a);
}

// Parse one notation token back into a move descriptor. Used by the
// game-text importer fallback and by nothing else -- the engine owns the
// authoritative parser (`qr_set_game_text`).
function parseNotation(tok) {
    if (!tok) return null;
    var m = /^([a-i])([1-9])([hv])$/i.exec(tok);
    if (m) {
        var c = m[1].toLowerCase().charCodeAt(0) - 97;
        var r = (N - 1) - parseInt(m[2], 10);
        if (r < 0 || r >= WS || c < 0 || c >= WS) return null;
        return { isWall: true, a: (m[3].toLowerCase() === 'h' ? 0 : 1), b: r, c: c };
    }
    m = /^([a-i])([1-9])$/i.exec(tok);
    if (m) {
        var cc = m[1].toLowerCase().charCodeAt(0) - 97;
        var er = N - parseInt(m[2], 10);
        return { isWall: false, a: er * N + cc, b: 0, c: 0 };
    }
    return null;
}

// =====================================================================
// SECTION 3 -- evaluation display
//
// Converts a raw engine score into a display percentage. Engine scores
// are MOVER-RELATIVE (positive = side to move is better); the display is
// always absolute-colour (player 0 = "white"): 0% = player 1 certain win,
// 100% = player 0 certain win, 50% = balanced.
//
// The sigmoid uses the same 200 scale factor that nnue.hpp uses to turn
// the WL head's raw logit into evalSimple-comparable units
// (NNUE_EVAL_SCALE). Here we walk that backwards (score -> logit ->
// probability), which is exact while the NNUE is active and a reasonable
// approximation in heuristic mode (evalSimple is not a real logit, but it
// lives in the same order of magnitude). Mate / certain-win scores are
// huge and saturate near 0%/100% by themselves, so they need no special
// case. The full table of which stage means what is in CLAUDE.md, section
// "Evaluation: what each stage uses" -- do not reinvent it here.
// =====================================================================

var EVAL_DISPLAY_SCALE = 200;

// Raw player-0-perspective score -> 0..100 percentage (player 0's share).
function evalToWhitePercent(v) {
    var pct = 100 / (1 + Math.exp(-v / EVAL_DISPLAY_SCALE));
    return clamp(pct, 0, 100);
}

// Mover-relative score -> that mover's own win percentage.
function moverWinPercent(score) { return evalToWhitePercent(score); }

// Mover-relative score + who the mover is -> player-0-perspective score.
function toP0Score(score, mover) { return mover === 0 ? score : -score; }

// Pawn-ish "+0.42" figure, in the same units chess GUIs use (score/100).
function formatScore(v) {
    if (v === null || v === undefined || !isFinite(v)) return '--';
    if (Math.abs(v) >= 20000) return v > 0 ? '+WIN' : '-WIN';
    return (v >= 0 ? '+' : '−') + (Math.abs(v) / 100).toFixed(2);
}

function formatPercent(p) {
    if (p === null || p === undefined || !isFinite(p)) return '--';
    return Math.round(p) + '%';
}

// =====================================================================
// SECTION 4 -- guarded WASM binding layer
//
// Every engine entry point is wrapped so that (a) an export the engine
// agent has not shipped yet is a SOFT failure -- the feature switches off
// and says so through the toast -- and (b) a throw inside the module
// never escapes into an event handler and blanks the page.
// =====================================================================

var Module = null;             // the emscripten module instance
var Q = {};                    // name -> safe callable
var wasmAvailable = {};        // name -> bool
var _reportedMissing = {};     // so a missing export toasts at most once
var _reportedError = {};

function _wasmMissing(name) {
    if (!_reportedMissing[name]) {
        _reportedMissing[name] = true;
        toast('Engine export missing: ' + name + ' -- that feature is disabled', 'warn');
    }
}

function _wasmError(name, err) {
    if (!_reportedError[name]) {
        _reportedError[name] = true;
        toast('Engine call failed: ' + name + ' (' + err + ')', 'bad');
    }
    try { console.error('[zq] ' + name, err); } catch (e) { /* ignore */ }
}

// Builds one guarded callable. `fallback` is what the app sees when the
// call cannot happen at all, chosen per function so the UI still renders.
function bindWasm(mod, name, ret, args, fallback) {
    var raw = null;
    try {
        if (typeof mod['_' + name] === 'function') raw = mod.cwrap(name, ret, args);
    } catch (e) {
        raw = null;
    }
    wasmAvailable[name] = !!raw;
    return function () {
        if (!raw) { _wasmMissing(name); return fallback; }
        try {
            var r = raw.apply(null, arguments);
            return (r === undefined || r === null) ? fallback : r;
        } catch (e) {
            _wasmError(name, e);
            return fallback;
        }
    };
}

function has(name) { return !!wasmAvailable[name]; }

// The complete export list from GUI_PLAN.md "W1 -- WASM engine layer".
function setupWasmBindings(mod) {
    var n = 'number', s = 'string';
    var b = function (name, ret, args, fb) { Q[name] = bindWasm(mod, name, ret, args, fb); };

    // -- lifecycle / engine config
    b('qr_new_game', null, [], undefined);
    b('qr_load_nnue_weights', n, [s], 0);
    b('qr_set_eval_heuristic', null, [], undefined);
    b('qr_eval_mode_is_nnue', n, [], 0);
    b('qr_set_mcab_enabled', null, [n], undefined);
    b('qr_mcab_active', n, [], 0);

    // -- position query
    b('qr_turn', n, [], 0);
    b('qr_winner', n, [], -1);
    b('qr_is_draw', n, [], 0);
    b('qr_pawn', n, [n], 0);
    b('qr_walls_left', n, [n], 0);
    b('qr_wall_h_bit', n, [n], 0);
    b('qr_wall_v_bit', n, [n], 0);
    b('qr_dist_to_goal', n, [n], 0);
    b('qr_static_eval', n, [], 0);
    b('qr_path_len', n, [n], 0);
    b('qr_path_cell', n, [n, n], -1);
    b('qr_is_wall_legal', n, [n, n, n], 0);
    b('qr_wall_owner', n, [n, n, n], -1);

    // -- legal moves
    b('qr_legal_moves_count', n, [], 0);
    b('qr_legal_move_is_wall', n, [n], 0);
    b('qr_legal_move_a', n, [n], 0);
    b('qr_legal_move_b', n, [n], 0);
    b('qr_legal_move_c', n, [n], 0);

    // -- playing
    b('qr_apply_pawn_move', n, [n], 0);
    b('qr_apply_wall_move', n, [n, n, n], 0);
    b('qr_engine_move', n, [n, n], 0);
    b('qr_last_move_is_wall', n, [], 0);
    b('qr_last_move_a', n, [], 0);
    b('qr_last_move_b', n, [], 0);
    b('qr_last_move_c', n, [], 0);
    b('qr_last_move_eval', n, [], 0);

    // -- history / navigation
    b('qr_history_len', n, [], 0);
    b('qr_history_cursor', n, [], 0);
    b('qr_goto_ply', n, [n], 0);
    b('qr_undo', n, [], 0);
    b('qr_redo', n, [], 0);
    b('qr_truncate_here', null, [], undefined);
    b('qr_hist_move_is_wall', n, [n], 0);
    b('qr_hist_move_a', n, [n], 0);
    b('qr_hist_move_b', n, [n], 0);
    b('qr_hist_move_c', n, [n], 0);
    b('qr_hist_mover', n, [n], 0);

    // -- analysis
    b('qr_analyze', n, [n, n, n], 0);
    b('qr_an_line_count', n, [], 0);
    b('qr_an_line_score', n, [n], 0);
    b('qr_an_line_visits', n, [n], 0);
    b('qr_an_line_len', n, [n], 0);
    b('qr_an_line_move_is_wall', n, [n, n], 0);
    b('qr_an_line_move_a', n, [n, n], 0);
    b('qr_an_line_move_b', n, [n, n], 0);
    b('qr_an_line_move_c', n, [n, n], 0);
    b('qr_an_nodes', n, [], 0);
    b('qr_an_depth', n, [], 0);
    b('qr_an_is_mcab', n, [], 0);

    // -- position editor
    b('qr_edit_begin', null, [], undefined);
    b('qr_edit_clear', null, [], undefined);
    b('qr_edit_set_pawn', n, [n, n], 0);
    b('qr_edit_toggle_wall', n, [n, n, n], -1);
    b('qr_edit_set_walls_left', null, [n, n], undefined);
    b('qr_edit_set_turn', null, [n], undefined);
    b('qr_edit_pawn', n, [n], 0);
    b('qr_edit_walls_left', n, [n], 0);
    b('qr_edit_turn', n, [], 0);
    b('qr_edit_wall_h_bit', n, [n], 0);
    b('qr_edit_wall_v_bit', n, [n], 0);
    b('qr_edit_validate', n, [], 0);
    b('qr_edit_commit', n, [], 0);

    // -- serialization
    b('qr_get_qfen', s, [], '');
    b('qr_set_qfen', n, [s], 0);
    b('qr_get_game_text', s, [], '');
    b('qr_set_game_text', n, [s], 0);
    b('qr_edit_get_qfen', s, [], '');
    b('qr_edit_set_qfen', n, [s], 0);
    b('qr_move_notation', s, [n, n, n, n], '');

    var missing = Object.keys(wasmAvailable).filter(function (k) { return !wasmAvailable[k]; });
    if (missing.length) {
        try { console.warn('[zq] missing engine exports:', missing.join(', ')); } catch (e) { /* ignore */ }
    }
    return missing;
}

// =====================================================================
// SECTION 5 -- settings and persistence
//
// One JSON blob in localStorage['zq.settings']. Reading never throws and
// never trusts a value: every field is validated against its default.
// =====================================================================

var SETTINGS_KEY = 'zq.settings';

var DEFAULT_SETTINGS = {
    boardTheme: 'wood',
    pawnStyle: 'disc',
    uiTheme: 'dark',
    highlight: true,
    paths: false,
    coords: true,
    dots: true,
    anim: true,
    sound: true,
    haptics: true,
    evalbar: true,
    movelogEval: true,
    nnue: true,
    mcab: true,
    soundVol: 3,
    strength: '4',                    // index into STRENGTH_LEVELS, or 'custom'
    searchMode: 'time',
    searchDepth: 6,
    searchTimeMs: 500,
    timeControl: '300000|3000',
    humanSide: 1,
    flipped: true,
    anDepth: 8,
    anTimeMs: 500,
    anMultiPv: 1,
    anArrows: true
};

// Named strength levels (claustrophobia-style): each is a per-move time
// budget handed to the same hybrid engine. 'custom' reveals the raw
// mode/depth/time selectors.
var STRENGTH_LEVELS = [
    { name: 'Pebble',   ms: 50 },
    { name: 'Bronze',   ms: 120 },
    { name: 'Silver',   ms: 250 },
    { name: 'Gold',     ms: 500 },
    { name: 'Platinum', ms: 1000 },
    { name: 'Diamond',  ms: 2000 },
    { name: 'Master',   ms: 4000 },
    { name: 'Titan',    ms: 8000 }
];

function strengthBudget() {
    var i = parseInt(settings.strength, 10);
    if (!(i >= 0) || i >= STRENGTH_LEVELS.length) return null;
    return STRENGTH_LEVELS[i].ms;
}

var BOARD_THEMES = ['wood', 'classic', 'emerald', 'ocean', 'coral', 'night'];
var PAWN_STYLES = ['disc', 'pin', 'chess', 'glyph'];

var settings = Object.assign({}, DEFAULT_SETTINGS);

function loadSettings() {
    var raw = null;
    try { raw = window.localStorage.getItem(SETTINGS_KEY); } catch (e) { raw = null; }
    var obj = null;
    if (raw) { try { obj = JSON.parse(raw); } catch (e) { obj = null; } }
    settings = Object.assign({}, DEFAULT_SETTINGS);
    if (!obj || typeof obj !== 'object') return;
    Object.keys(DEFAULT_SETTINGS).forEach(function (k) {
        var d = DEFAULT_SETTINGS[k], v = obj[k];
        if (v === undefined || v === null) return;
        if (typeof d === 'boolean') { settings[k] = !!v; return; }
        if (typeof d === 'number') { if (typeof v === 'number' && isFinite(v)) settings[k] = v; return; }
        if (typeof v === 'string') settings[k] = v;
    });
    // Enumerations get an extra sanity pass so a hand-edited value cannot
    // hand the board renderer a theme it has never heard of.
    if (BOARD_THEMES.indexOf(settings.boardTheme) < 0) settings.boardTheme = DEFAULT_SETTINGS.boardTheme;
    if (PAWN_STYLES.indexOf(settings.pawnStyle) < 0) settings.pawnStyle = DEFAULT_SETTINGS.pawnStyle;
    if (settings.uiTheme !== 'dark' && settings.uiTheme !== 'light') settings.uiTheme = 'dark';
    if (['depth', 'time', 'game'].indexOf(settings.searchMode) < 0) settings.searchMode = 'time';
    settings.humanSide = (settings.humanSide === 0) ? 0 : 1;
    if (settings.strength !== 'custom') {
        var si = parseInt(settings.strength, 10);
        if (!(si >= 0) || si >= STRENGTH_LEVELS.length) settings.strength = DEFAULT_SETTINGS.strength;
    }
    settings.soundVol = clamp(Math.round(settings.soundVol), 0, 5);
    settings.anMultiPv = clamp(Math.round(settings.anMultiPv), 1, PV_MAX);
    settings.searchDepth = clamp(Math.round(settings.searchDepth), 2, 12);
    settings.searchTimeMs = clamp(Math.round(settings.searchTimeMs), 50, 20000);
    settings.anDepth = clamp(Math.round(settings.anDepth), 1, 20);
    settings.anTimeMs = clamp(Math.round(settings.anTimeMs), 50, 20000);
}

function saveSettings() {
    try { window.localStorage.setItem(SETTINGS_KEY, JSON.stringify(settings)); }
    catch (e) { /* private mode / disabled storage: settings just do not persist */ }
}

// =====================================================================
// SECTION 6 -- sounds (WebAudio, synthesized -- no asset files)
// =====================================================================

var _audioCtx = null;
var _soundScale = 0.6;

function _ac() {
    if (!_audioCtx) {
        var C = window.AudioContext || window.webkitAudioContext;
        if (!C) return null;
        _audioCtx = new C();
    }
    return _audioCtx;
}

// Browsers keep the context suspended until a user gesture; call this
// from real input handlers so the first sound is not swallowed.
function audioUnlock() {
    try { var ac = _ac(); if (ac && ac.state === 'suspended') ac.resume(); } catch (e) { /* ignore */ }
}

// One shared lowpass bus so nothing sounds raw or buzzy; every voice goes
// through it, which warms square/saw sources and softens attacks.
function _bus(ac) {
    if (!_busNode) {
        _busNode = ac.createBiquadFilter();
        _busNode.type = 'lowpass';
        _busNode.frequency.value = 2600;
        _busNode.Q.value = 0.4;
        _busNode.connect(ac.destination);
    }
    return _busNode;
}
var _busNode = null;

function beep(freq, dur, type, vol, decay, freqEnd) {
    if (!settings.sound || settings.soundVol <= 0) return;
    try {
        var ac = _ac();
        if (!ac) return;
        var play = function () {
            var o = ac.createOscillator(), g = ac.createGain();
            o.connect(g); g.connect(_bus(ac));
            o.type = type || 'sine';
            o.frequency.setValueAtTime(freq, ac.currentTime);
            if (freqEnd) o.frequency.exponentialRampToValueAtTime(Math.max(1, freqEnd), ac.currentTime + dur);
            var v = (vol || 0.15) * _soundScale;
            g.gain.setValueAtTime(0.0001, ac.currentTime);
            g.gain.linearRampToValueAtTime(v, ac.currentTime + 0.006);   // soft attack
            g.gain.exponentialRampToValueAtTime(0.0008, ac.currentTime + (decay || dur));
            o.start(ac.currentTime);
            o.stop(ac.currentTime + dur + 0.05);
        };
        if (ac.state === 'suspended') ac.resume().then(play, function () { });
        else play();
    } catch (e) { /* audio is decorative: never let it break a move */ }
}

// Short filtered noise burst -- the "contact" ingredient that turns a beep
// into something percussive (a knock, a thud).
function thud(vol, cutoff, dur) {
    if (!settings.sound || settings.soundVol <= 0) return;
    try {
        var ac = _ac();
        if (!ac) return;
        var play = function () {
            var n = Math.floor(ac.sampleRate * (dur || 0.06));
            var buf = ac.createBuffer(1, n, ac.sampleRate);
            var d = buf.getChannelData(0);
            for (var i = 0; i < n; i++) d[i] = (Math.random() * 2 - 1) * (1 - i / n);
            var src = ac.createBufferSource(); src.buffer = buf;
            var f = ac.createBiquadFilter(); f.type = 'lowpass'; f.frequency.value = cutoff || 900;
            var g = ac.createGain();
            var v = (vol || 0.2) * _soundScale;
            g.gain.setValueAtTime(v, ac.currentTime);
            g.gain.exponentialRampToValueAtTime(0.0008, ac.currentTime + (dur || 0.06));
            src.connect(f); f.connect(g); g.connect(_bus(ac));
            src.start(ac.currentTime);
        };
        if (ac.state === 'suspended') ac.resume().then(play, function () { });
        else play();
    } catch (e) { /* ignore */ }
}

function applySoundVolume() {
    var scales = [0, 0.15, 0.35, 0.6, 1.0, 1.6];
    _soundScale = scales[clamp(settings.soundVol, 0, 5)] || 0.6;
}

// Wooden tap: pitched body + contact noise, both decaying fast.
function sndPawn() {
    beep(210, 0.09, 'sine', 0.16, 0.07, 150);
    thud(0.10, 1400, 0.03);
    haptic(12);
}
function sndJump() {
    beep(240, 0.07, 'sine', 0.13, 0.05, 200);
    setTimeout(function () { beep(320, 0.08, 'sine', 0.12, 0.06, 250); }, 55);
    haptic(18);
}
// Heavy wooden thunk: low swept body plus a darker noise slap.
function sndWall() {
    beep(120, 0.16, 'sine', 0.22, 0.14, 62);
    thud(0.22, 500, 0.05);
    setTimeout(function () { thud(0.12, 300, 0.04); }, 30);
    setTimeout(function () { haptic([16, 24, 16]); }, 20);
}
// Warm resolved chord, slow release -- a finish, not an alarm.
function sndEnd() {
    beep(262, 0.5, 'sine', 0.10, 0.45);
    setTimeout(function () { beep(330, 0.5, 'sine', 0.09, 0.45); }, 60);
    setTimeout(function () { beep(392, 0.7, 'sine', 0.09, 0.65); }, 120);
    setTimeout(function () { beep(523, 0.8, 'sine', 0.06, 0.75); }, 180);
    haptic([40, 60, 40, 60, 90]);
}
function sndIllegal() { beep(85, 0.12, 'triangle', 0.12, 0.11, 70); thud(0.06, 300, 0.05); haptic(50); }

// Haptic feedback: short vibration bursts on supported devices (mobile),
// gated by the settings toggle so it can be turned off entirely.
function haptic(pattern) {
    if (!settings.haptics || !navigator.vibrate) return;
    try { navigator.vibrate(pattern); } catch (e) { /* never break a move */ }
}

// =====================================================================
// SECTION 7 -- reading the engine position
// =====================================================================

// A shortest path per player, from qr_path_cell. Returns null when the
// export is unavailable so the renderer just skips the overlay.
function readPath(p) {
    if (!has('qr_path_cell')) return null;
    var out = [];
    for (var i = 0; i < 200; i++) {
        var c = Q.qr_path_cell(p, i);
        if (c < 0) break;
        out.push(c);
    }
    return out.length ? out : null;
}

function readPosition() {
    var wallsH = new Uint8Array(WS * WS);
    var wallsV = new Uint8Array(WS * WS);
    var ownH = new Int8Array(WS * WS);
    var ownV = new Int8Array(WS * WS);
    for (var i = 0; i < WS * WS; i++) {
        wallsH[i] = Q.qr_wall_h_bit(i) === 1 ? 1 : 0;
        wallsV[i] = Q.qr_wall_v_bit(i) === 1 ? 1 : 0;
        var r = Math.floor(i / WS), c = i % WS;
        ownH[i] = wallsH[i] ? Q.qr_wall_owner(0, r, c) : -1;
        ownV[i] = wallsV[i] ? Q.qr_wall_owner(1, r, c) : -1;
    }
    return {
        pawns: [Q.qr_pawn(0), Q.qr_pawn(1)],
        wallsH: wallsH,
        wallsV: wallsV,
        wallOwner: { h: ownH, v: ownV },
        turn: Q.qr_turn(),
        wallsLeft: [Q.qr_walls_left(0), Q.qr_walls_left(1)],
        winner: Q.qr_winner()
    };
}

function readLegal() {
    var n = Q.qr_legal_moves_count();
    var pawnCells = [], wallH = [], wallV = [];
    for (var i = 0; i < n; i++) {
        var isWall = Q.qr_legal_move_is_wall(i) === 1;
        var a = Q.qr_legal_move_a(i), b = Q.qr_legal_move_b(i), c = Q.qr_legal_move_c(i);
        if (!isWall) pawnCells.push(a);
        else if (a === 0) wallH.push(b * WS + c);
        else wallV.push(b * WS + c);
    }
    return { pawnCells: pawnCells, wallH: wallH, wallV: wallV };
}

function histMove(ply) {
    return {
        isWall: Q.qr_hist_move_is_wall(ply) === 1,
        a: Q.qr_hist_move_a(ply),
        b: Q.qr_hist_move_b(ply),
        c: Q.qr_hist_move_c(ply),
        mover: Q.qr_hist_mover(ply)
    };
}

function lastMoveOfCursor() {
    var cur = Q.qr_history_cursor();
    if (cur <= 0) return null;
    var m = histMove(cur - 1);
    return { isWall: m.isWall, a: m.a, b: m.b, c: m.c };
}

function gameOver() {
    return flagFallWinner >= 0 || Q.qr_winner() !== -1 || Q.qr_is_draw() === 1;
}

// True only when the cursor sits at the live end of the game -- browsing
// the history must not let a click play a move by accident.
function atLiveEnd() {
    return Q.qr_history_cursor() >= Q.qr_history_len();
}

// =====================================================================
// SECTION 8 -- per-ply annotations (blunder check + eval graph)
//
// posEval[k]  = player-0 win percentage of the position after k plies.
// plyInfo[k]  = { annot, drop } for the move played at ply k.
// =====================================================================

var posEval = [];
var plyInfo = [];

function resetAnnotations() { posEval = []; plyInfo = []; }

function recordPosEval(cursor, p0pct) {
    if (cursor < 0) return;
    posEval[cursor] = p0pct;
}

// =====================================================================
// SECTION 9 -- board instances
// =====================================================================

var boards = { play: null, an: null, ed: null };
var boardsOk = false;

function makeBoard(canvasId, mode) {
    var cv = $(canvasId);
    if (!cv) return null;
    if (typeof window.QBoard !== 'function') return null;
    try {
        return new window.QBoard(cv, {
            mode: mode,
            theme: settings.boardTheme,
            pawnStyle: settings.pawnStyle,
            flipped: settings.flipped
        });
    } catch (e) {
        try { console.error('[zq] QBoard construction failed', e); } catch (e2) { /* ignore */ }
        return null;
    }
}

function boardCall(board, method, args) {
    if (!board || typeof board[method] !== 'function') return undefined;
    try { return board[method].apply(board, args || []); }
    catch (e) {
        try { console.error('[zq] QBoard.' + method + ' failed', e); } catch (e2) { /* ignore */ }
        return undefined;
    }
}

function applyBoardOptions(board) {
    boardCall(board, 'setTheme', [settings.boardTheme]);
    boardCall(board, 'setPawnStyle', [settings.pawnStyle]);
    boardCall(board, 'setOptions', [{
        showCoords: settings.coords,
        showPaths: settings.paths,
        showDots: settings.dots,
        highlightLast: settings.highlight,
        animate: settings.anim
    }]);
}

function applyBoardOptionsAll() {
    ['play', 'an', 'ed'].forEach(function (k) { applyBoardOptions(boards[k]); });
    renderAll();
}

// =====================================================================
// SECTION 10 -- app-level state
// =====================================================================

var activeTab = 'play';
var busy = false;             // engine is thinking / a move is being applied
var gameGen = 0;              // bumped by new game / side switch; cancels a
// Flag-fall terminal state (clock mode): winner index, -1 otherwise.
var flagFallWinner = -1;
                              // pending engine move from a previous game
var engineInfoText = 'Heuristic';
var nnueLoaded = false;
var hintOverlay = null;       // { isWall, a, b, c } shown by the hint button
var anArrows = [];            // arrows drawn on the analysis board
var wallOrientation = 0;      // 0 = H, 1 = V, armed for placement

// =====================================================================
// SECTION 11 -- rendering
// =====================================================================

function renderAll() {
    if (!Module) return;
    var pos = readPosition();
    var legal = readLegal();
    var over = gameOver();
    var humanTurn = atLiveEnd() && !over && pos.turn === settings.humanSide;
    var paths = settings.paths ? [readPath(0), readPath(1)] : null;
    renderWallCount();

    // --- play board
    if (boards.play) {
        boardCall(boards.play, 'setFlipped', [settings.flipped]);
        boardCall(boards.play, 'setPosition', [pos]);
        boardCall(boards.play, 'setLegal', [(humanTurn && !busy) ? legal : { pawnCells: [], wallH: [], wallV: [] }]);
        boardCall(boards.play, 'setInteractive', [humanTurn && !busy]);
        boardCall(boards.play, 'setWallOrientation', [wallOrientation]);
        boardCall(boards.play, 'setOverlays', [{
            lastMove: settings.highlight ? lastMoveOfCursor() : null,
            hint: hintOverlay,
            arrows: hintOverlay ? hintArrows(hintOverlay) : [],
            ghostWall: null,
            paths: paths
        }]);
        boardCall(boards.play, 'render', []);
    }

    // --- analysis board
    if (boards.an) {
        boardCall(boards.an, 'setFlipped', [settings.flipped]);
        boardCall(boards.an, 'setPosition', [pos]);
        boardCall(boards.an, 'setLegal', [over ? { pawnCells: [], wallH: [], wallV: [] } : legal]);
        boardCall(boards.an, 'setInteractive', [!over]);
        boardCall(boards.an, 'setOverlays', [{
            lastMove: settings.highlight ? lastMoveOfCursor() : null,
            hint: null,
            arrows: settings.anArrows ? anArrows : [],
            ghostWall: null,
            paths: paths
        }]);
        boardCall(boards.an, 'render', []);
    }

    renderPlayerBars(pos, over);
    renderStatus(pos, over);
    renderMoveLog('pl-movelog');
    renderMoveLog('an-movelog');
    renderEvalBars();
    renderNavButtons();
    drawEvalGraph();
}

function hintArrows(mv) {
    if (!mv) return [];
    if (mv.isWall) return [{ wall: { o: mv.a, r: mv.b, c: mv.c }, color: '#c8a84b' }];
    return [{ from: Q.qr_pawn(Q.qr_turn()), to: mv.a, color: '#c8a84b', width: 3 }];
}

// ---- player bars -----------------------------------------------------
// Bottom bar is always the human's side in play mode (the board flips to
// match), so the two bars follow the flip state, not the player index.
function bottomPlayer() { return settings.flipped ? 0 : 1; }

function renderPlayerBars(pos, over) {
    var bot = bottomPlayer(), top = 1 - bot;
    renderOneBar('top', top, pos, over);
    renderOneBar('bot', bot, pos, over);
    renderRaceMeter();
}

function renderOneBar(which, player, pos, over) {
    var name = (player === settings.humanSide) ? 'You' : 'Engine';
    setText('pl-name-' + which, name);
    var dot = $('pl-dot-' + which);
    if (dot) {
        dot.classList.remove('p0', 'p1');
        dot.classList.add('p' + player);
    }
    var bar = $('pl-bar-' + which);
    if (bar) {
        bar.classList.toggle('active', !over && pos.turn === player);
        bar.classList.toggle('thinking', !over && pos.turn === player && pos.turn !== settings.humanSide && busy);
        bar.classList.toggle('winner', pos.winner === player);
    }
    renderWallPips('pl-walls-' + which, pos.wallsLeft[player], player);
    setText('pl-walls-num-' + which, String(pos.wallsLeft[player]));
    var d = Q.qr_dist_to_goal(player);
    setText('pl-dist-' + which, d + ' to goal');
    renderClock(which, player);
}

// Race meter: each side's share of the two remaining distances. If your
// distance is small relative to the opponent's, your colour fills most of
// the track -- an at-a-glance "who is winning the race" read.
function renderRaceMeter() {
    var f0 = $('pl-race-p0'), f1 = $('pl-race-p1');
    if (!f0 || !f1) return;
    var d0 = Q.qr_dist_to_goal(0), d1 = Q.qr_dist_to_goal(1);
    if (!isFinite(d0) || !isFinite(d1) || d0 + d1 <= 0) { d0 = 1; d1 = 1; }
    var pct0 = 100 * d1 / (d0 + d1);   // gold share grows as opponent gets farther
    var pct1 = 100 * d0 / (d0 + d1);
    f0.style.width = Math.max(4, Math.min(96, pct0)) + '%';
    f1.style.width = Math.max(4, Math.min(96, pct1)) + '%';
}

function renderWallPips(id, left, player) {
    var el = $(id);
    if (!el) return;
    // Rebuild only when the shape changed -- this runs on every render.
    if (el.childElementCount !== MAX_WALLS) {
        el.innerHTML = '';
        for (var i = 0; i < MAX_WALLS; i++) el.appendChild(document.createElement('i'));
    }
    for (var k = 0; k < MAX_WALLS; k++) {
        // NOTE: classes must match the stylesheet (.pwalls .pip / .spent);
        // a bare 'on' renders as an unstyled, invisible element.
        el.children[k].className = k < left ? ('pip on p' + player) : 'pip spent';
    }
    el.setAttribute('title', left + ' of ' + MAX_WALLS + ' walls left');
}

// ---- status line -----------------------------------------------------
function renderStatus(pos, over) {
    var txt;
    var dotCls = 'p' + pos.turn;
    if (flagFallWinner >= 0) {
        txt = (flagFallWinner === settings.humanSide ? 'You win on time!' : 'You lose on time') +
              ' (player ' + flagFallWinner + ')';
        dotCls = 'p' + flagFallWinner + ' over';
    } else if (pos.winner !== -1) {
        txt = (pos.winner === settings.humanSide ? 'You win!' : 'Zquoridor wins') +
              ' (player ' + pos.winner + ')';
        dotCls = 'p' + pos.winner + ' over';
    } else if (Q.qr_is_draw() === 1) {
        txt = 'Draw by threefold repetition';
        dotCls = 'draw';
    } else if (!atLiveEnd()) {
        txt = 'Browsing move ' + Q.qr_history_cursor() + ' of ' + Q.qr_history_len();
        dotCls = 'browse';
    } else if (busy) {
        txt = 'Zquoridor is thinking…';
        dotCls = 'p' + pos.turn + ' thinking';
    } else {
        txt = (pos.turn === settings.humanSide ? 'Your move' : 'Zquoridor to move');
        var d0 = Q.qr_dist_to_goal(0), d1 = Q.qr_dist_to_goal(1);
        var near = (d0 <= 3) ? 0 : (d1 <= 3 ? 1 : -1);
        if (near >= 0) {
            txt += ' — player ' + near + ' is ' + (near === 0 ? d0 : d1) + ' steps from goal';
        }
    }
    setText('pl-status', txt);
    var dot = $('pl-status-dot');
    if (dot) dot.className = 'sdot ' + dotCls;
    setText('pl-engine-info', engineInfoText);
}

// ---- move log --------------------------------------------------------
var ANNOT_CLASS = { '!': 'g-best', '': 'g-good', '?!': 'g-inacc', '?': 'g-mistake', '??': 'g-blunder' };

function renderMoveLog(elId) {
    var el = $(elId);
    if (!el) return;
    var len = Q.qr_history_len();
    var cursor = Q.qr_history_cursor();
    if (!len) {
        el.innerHTML = '<div class="ml-empty">No moves yet.</div>';
        return;
    }
    var html = '';
    for (var i = 0; i < len; i += 2) {
        html += '<div class="ml-row"><span class="ml-num">' + (i / 2 + 1) + '.</span>';
        html += plySpan(i, cursor);
        html += (i + 1 < len) ? plySpan(i + 1, cursor) : '<span class="ml-ply empty"></span>';
        html += '</div>';
    }
    el.innerHTML = html;

    // Click a ply -> navigate. Delegated per render so no listener leaks.
    el.querySelectorAll('.ml-ply[data-ply]').forEach(function (sp) {
        sp.addEventListener('click', function () {
            gotoPly(parseInt(sp.getAttribute('data-ply'), 10) + 1);
        });
    });
    var cur = el.querySelector('.ml-ply.current');
    if (cur && cur.scrollIntoView) cur.scrollIntoView({ block: 'nearest' });
}

function plySpan(ply, cursor) {
    var m = histMove(ply);
    var txt = moveNotation(m);
    var info = plyInfo[ply];
    var glyph = '';
    if (info && info.annot !== undefined && info.annot !== null && info.annot !== '') {
        glyph = '<span class="' + (ANNOT_CLASS[info.annot] || 'g-good') + '">' + esc(info.annot) + '</span>';
    }
    var evalHtml = '';
    if (settings.movelogEval && posEval[ply + 1] !== undefined) {
        evalHtml = '<span class="ml-eval">' + formatPercent(posEval[ply + 1]) + '</span>';
    }
    // cursor == ply+1 means "the position right after this move is shown".
    var cls = 'ml-ply p' + m.mover + (cursor === ply + 1 ? ' current' : '');
    return '<span class="' + cls + '" data-ply="' + ply + '" title="ply ' + (ply + 1) + '">' +
           esc(txt) + glyph + '</span>' + evalHtml;
}

// ---- eval bars -------------------------------------------------------
// Player 0's share is drawn on TOP of the vertical bar, per the plan.
var _lastEvalP0 = 50;

function currentP0Percent() {
    if (!has('qr_static_eval')) return _lastEvalP0;
    var s = Q.qr_static_eval();
    return evalToWhitePercent(toP0Score(s, Q.qr_turn()));
}

function renderEvalBars(overridePct) {
    var pct = (overridePct === undefined || overridePct === null) ? currentP0Percent() : overridePct;
    _lastEvalP0 = pct;
    // Only the systematic full-game scan may write graph points; live
    // analysis and browsing must not overwrite recorded evaluations.
    if (bcRunning) recordPosEval(Q.qr_history_cursor(), pct);
    setEvalBar('pl', pct);
    setEvalBar('an', pct);
}

function setEvalBar(prefix, pct) {
    var wrap = $(prefix + '-evalbar');
    if (wrap) wrap.style.display = settings.evalbar ? '' : 'none';
    var p = clamp(pct, 0, 100);
    var fill = $(prefix + '-evalbar-fill');
    if (fill) fill.style.height = p.toFixed(1) + '%';
    // The number rides the gold/red boundary; clamp keeps it on the bar at
    // the extremes instead of sliding off the ends.
    var chip = $(prefix + '-evalbar-num');
    if (chip) {
        chip.style.bottom = clamp(p, 6, 94).toFixed(1) + '%';
        chip.textContent = formatPercent(pct);
    }
}

// ---- nav buttons -----------------------------------------------------
function renderNavButtons() {
    var cur = Q.qr_history_cursor(), len = Q.qr_history_len();
    setDisabled('pl-btn-undo', cur <= 0);
    setDisabled('pl-btn-redo', cur >= len);
    setDisabled('pl-btn-takeback', cur < 2);
    setDisabled('an-first', cur <= 0);
    setDisabled('an-prev', cur <= 0);
    setDisabled('an-next', cur >= len);
    setDisabled('an-last', cur >= len);
    setDisabled('pl-btn-hint', busy || gameOver() || !atLiveEnd());
}

// =====================================================================
// SECTION 12 -- play mode
// =====================================================================

function newGame() {
    gameGen++;
    busy = false;
    flagFallWinner = -1;
    hintOverlay = null;
    anArrows = [];
    resetAnnotations();
    stopAnalysis();
    cancelBlunderCheck();
    Q.qr_new_game();
    settings.flipped = (settings.humanSide === 0);
    saveSettings();
    if (settings.searchMode === 'game') initClocks(); else stopClock();
    renderAll();
    maybeEngineMove();
}

// Reads the strength selectors into { depth, timeMs } for qr_engine_move.
// Named levels override everything; 'custom' exposes the raw selects.
function searchParams() {
    var budget = strengthBudget();
    if (budget !== null) return { depth: 40, timeMs: budget };
    var mode = settings.searchMode;
    if (mode === 'depth') return { depth: clamp(settings.searchDepth, 2, 12), timeMs: 0 };
    if (mode === 'game') {
        // Simple, defensible budget: assume ~30 moves left plus most of the
        // increment, hard-capped at a quarter of the remaining clock.
        var side = Q.qr_turn();
        var left = side === 0 ? clockMs[0] : clockMs[1];
        var budget = Math.round(left / 30 + clockInc * 0.75);
        budget = clamp(budget, 60, Math.max(60, Math.round(left * 0.25)));
        return { depth: 40, timeMs: budget };
    }
    return { depth: 40, timeMs: clamp(settings.searchTimeMs, 50, 20000) };
}

// The engine move runs with a generation guard, so starting a new game or
// switching sides while the engine is thinking discards the stale result
// instead of applying it to the new game. The search itself happens in the
// engine worker when available (the UI stays live during long thinks);
// the returned best move is then applied on the main-thread module.
function maybeEngineMove() {
    if (busy || !atLiveEnd() || gameOver()) return;
    if (Q.qr_turn() === settings.humanSide) return;
    busy = true;
    var myGen = gameGen;
    var engineSide = Q.qr_turn();
    renderNavButtons();
    renderStatus(readPosition(), false);

    // Worker path. The generation is captured NOW and threaded through the
    // callback -- reading gameGen inside the callback would accept stale
    // replies into a newer game.
    var dispatched = function () {
        var pp = searchParams();
        var t = pp.timeMs > 0 ? pp.timeMs : 5000;   // depth mode: generous cap, depth still bounds it
        t = Math.max(60, Math.min(t, 20000));
        return ewAnalyzeAt(Q.qr_get_qfen(), pp.depth, t, 1, function (res) {
            engineMoveFromWorker(myGen, engineSide, res);
        }, t);
    }();
    if (dispatched) return;

    // Main-thread fallback: one blocking qr_engine_move in a timer slice.
    setTimeout(function () {
        if (myGen !== gameGen) { busy = false; return; }
        var p = searchParams();
        Q.qr_engine_move(p.depth, p.timeMs);
        busy = false;
        if (myGen !== gameGen) return;
        var mv = {
            isWall: Q.qr_last_move_is_wall() === 1,
            a: Q.qr_last_move_a(), b: Q.qr_last_move_b(), c: Q.qr_last_move_c()
        };
        afterMoveApplied(mv, engineSide, true);
    }, 20);
}

function engineMoveFromWorker(dispatchGen, engineSide, res) {
    // Stale reply (new game / side switch / navigation since dispatch):
    // drop it without touching state.
    if (dispatchGen !== gameGen || !atLiveEnd() || gameOver()) {
        busy = false;
        renderAll();
        return;
    }
    var mv = null;
    if (res && res.lines && res.lines.length && res.lines[0].pv.length) {
        var cand = res.lines[0].pv[0];
        // Apply on the main module; a stale/illegal answer falls back.
        try {
            var ok = cand.isWall ? Q.qr_apply_wall_move(cand.a, cand.b, cand.c) === 1
                                 : Q.qr_apply_pawn_move(cand.a) === 1;
            if (ok) mv = { isWall: !!cand.isWall, a: cand.a, b: cand.b || 0, c: cand.c || 0 };
        } catch (e) { mv = null; }
    }
    if (!mv && !gameOver()) {
        // Worker produced nothing usable -- search here with a bounded
        // budget so a dead worker cannot turn into a second long freeze.
        var p = searchParams();
        var fbT = Math.min(p.timeMs > 0 ? p.timeMs : 5000, 3000);
        Q.qr_engine_move(Math.max(2, Math.min(p.depth, 12)), fbT);
        mv = {
            isWall: Q.qr_last_move_is_wall() === 1,
            a: Q.qr_last_move_a(), b: Q.qr_last_move_b(), c: Q.qr_last_move_c()
        };
    }
    busy = false;
    if (dispatchGen !== gameGen) return;
    afterMoveApplied(mv, engineSide, true);
}

// Shared tail for every applied move: sound, clocks, eval, redraw and the
// possible engine reply.
function afterMoveApplied(mv, mover, fromEngine) {
    hintOverlay = null;
    anArrows = [];
    if (mv && mv.isWall) sndWall();
    else if (mv) {
        // A pawn move that covers more than one step is a jump (over the
        // opponent or diagonal past them). The mover's previous cell is the
        // destination of their own previous ply, two plies back.
        var isJump = false;
        var cur = Q.qr_history_cursor();
        if (!mv.isWall && cur >= 2 && Q.qr_hist_move_is_wall(cur - 2) === 0 &&
            Q.qr_hist_mover(cur - 2) === mover) {
            var prev = Q.qr_hist_move_a(cur - 2);
            var dr = Math.abs(Math.floor(prev / 9) - Math.floor(mv.a / 9));
            var dc = Math.abs((prev % 9) - (mv.a % 9));
            isJump = dr + dc > 1;
        }
        if (isJump) sndJump(); else sndPawn();
    }
    if (settings.searchMode === 'game' && !gameOver()) {
        stopClock();
        addIncrement(mover);
        startClock(1 - mover);
    }
    renderAll();
    if (gameOver()) {
        stopClock();
        sndEnd();
        toast(flagFallWinner >= 0 ? ('Player ' + flagFallWinner + ' wins on time')
             : (Q.qr_winner() === -1 ? 'Draw by threefold repetition'
                                     : 'Player ' + Q.qr_winner() + ' wins'), 'ok');
        return;
    }
    if (!fromEngine) maybeEngineMove();
    else if (Q.qr_turn() !== settings.humanSide) maybeEngineMove();
}

function humanPawnMove(cell) {
    audioUnlock();
    if (busy || bcRunning || gameOver() || !atLiveEnd()) { sndIllegal(); return; }
    if (Q.qr_turn() !== settings.humanSide) return;
    var mover = Q.qr_turn();
    if (Q.qr_apply_pawn_move(cell) !== 1) { sndIllegal(); toast('Illegal move', 'warn'); return; }
    afterMoveApplied({ isWall: false, a: cell, b: 0, c: 0 }, mover, false);
}

function humanWallMove(o, r, c) {
    audioUnlock();
    if (busy || bcRunning || gameOver() || !atLiveEnd()) { sndIllegal(); return; }
    if (Q.qr_turn() !== settings.humanSide) return;
    var mover = Q.qr_turn();
    if (Q.qr_apply_wall_move(o, r, c) !== 1) { sndIllegal(); toast('Illegal wall', 'warn'); return; }
    afterMoveApplied({ isWall: true, a: o, b: r, c: c }, mover, false);
}

// In analysis the same board plays a "temporary variation" -- it uses the
// engine's own truncate-then-append semantics, so it is a real branch.
function analysisPlay(mv) {
    if (gameOver() || bcRunning) return;
    var ok = mv.isWall ? Q.qr_apply_wall_move(mv.a, mv.b, mv.c) : Q.qr_apply_pawn_move(mv.a);
    if (ok !== 1) { sndIllegal(); toast('Illegal move', 'warn'); return; }
    if (mv.isWall) sndWall(); else sndPawn();
    anArrows = [];
    renderAll();
    if (analysisOn) restartAnalysis();
}

function gotoPly(ply) {
    // During the full-game blunder check the cursor is owned by the
    // scanner -- navigation would desync its ply bookkeeping.
    if (bcRunning) { toast('Game analysis in progress', 'warn'); return; }
    stopAnalysisSearchOnly();
    Q.qr_goto_ply(clamp(ply, 0, Q.qr_history_len()));
    hintOverlay = null;
    anArrows = [];
    renderAll();
    if (analysisOn) restartAnalysis();
}

function doUndo() { gotoPly(Q.qr_history_cursor() - 1); }
function doRedo() { gotoPly(Q.qr_history_cursor() + 1); }

// Takeback: two plies, so the human is on move again.
function doTakeback() {
    gameGen++;             // cancel any engine move still queued
    busy = false;
    var target = Q.qr_history_cursor() - 2;
    Q.qr_goto_ply(clamp(target, 0, Q.qr_history_len()));
    Q.qr_truncate_here();
    hintOverlay = null;
    renderAll();
    maybeEngineMove();
}

function doFlip() {
    settings.flipped = !settings.flipped;
    saveSettings();
    renderAll();
}

function doSwitchSide() {
    settings.humanSide = 1 - settings.humanSide;
    saveSettings();
    newGame();
}

// A short analysis that shows the best move without playing it.
function doHint() {
    if (busy || gameOver() || !atLiveEnd()) return;
    if (!has('qr_analyze')) { toast('Hints need qr_analyze -- not available', 'warn'); return; }
    busy = true;
    renderNavButtons();
    var myGen = gameGen;
    var finish = function (lines) {
        busy = false;
        if (myGen !== gameGen) return;
        if (!lines || !lines.length || !lines[0].pv.length) { toast('No hint available', 'warn'); renderAll(); return; }
        hintOverlay = lines[0].pv[0];
        toast('Hint: ' + moveNotation(hintOverlay), 'ok');
        renderAll();
    };
    // Worker first so even a 400ms hint never freezes the board.
    var dispatched = ewAnalyzeAt(Q.qr_get_qfen(), clamp(settings.searchDepth, 2, 12), 400, 1,
                                 function (res) { finish(res ? res.lines : null); }, 400);
    if (dispatched) return;
    setTimeout(function () {
        finish(runAnalyzeSlice(clamp(settings.searchDepth, 2, 12), 400, 1));
    }, 10);
}

// =====================================================================
// SECTION 13 -- clocks (mode "game")
// =====================================================================

var clockMs = [0, 0];
var clockInc = 0;
var clockTimer = null;
var clockRunningFor = -1;

function fmtClock(ms) {
    if (ms < 0) ms = 0;
    var total = Math.ceil(ms / 1000);
    var m = Math.floor(total / 60), s = total % 60;
    return m + ':' + (s < 10 ? '0' : '') + s;
}

function renderClock(which, player) {
    var el = $('pl-clock-' + which);
    if (!el) return;
    if (settings.searchMode !== 'game') { el.style.display = 'none'; return; }
    el.style.display = '';
    el.textContent = fmtClock(clockMs[player]);
    el.classList.toggle('active-clock', clockRunningFor === player);
    el.classList.toggle('low-time', clockMs[player] > 0 && clockMs[player] < 30000);
}

function initClocks() {
    var parts = String(settings.timeControl || '300000|3000').split('|');
    var base = parseInt(parts[0], 10);
    clockInc = parseInt(parts[1], 10) || 0;
    if (!isFinite(base) || base <= 0) base = 300000;
    clockMs = [base, base];
    stopClock();
    startClock(Q.qr_turn());
}

function startClock(player) {
    stopClock();
    if (settings.searchMode !== 'game' || gameOver()) return;
    clockRunningFor = player;
    var t0 = performance.now();
    var start = clockMs[player];
    clockTimer = setInterval(function () {
        var remaining = start - (performance.now() - t0);
        clockMs[player] = Math.max(0, remaining);
        renderClock('top', 1 - bottomPlayer());
        renderClock('bot', bottomPlayer());
        if (remaining <= 0) {
            stopClock();
            gameGen++;    // stop any pending engine move for this game
            busy = false;
            // Real terminal state, not just a toast: gameOver() consults
            // this, so moves/analysis/hints are withdrawn immediately.
            flagFallWinner = 1 - player;
            renderAll();
            toast('Flag fall -- player ' + (1 - player) + ' wins on time', 'bad');
            sndEnd();
        }
    }, 100);
    renderClock('top', 1 - bottomPlayer());
    renderClock('bot', bottomPlayer());
}

function stopClock() {
    if (clockTimer) { clearInterval(clockTimer); clockTimer = null; }
    clockRunningFor = -1;
}

function addIncrement(player) {
    if (clockInc > 0) clockMs[player] += clockInc;
}

// =====================================================================
// SECTION 13b -- engine worker (background search)
//
// A dedicated Web Worker runs its own WASM instance and does every
// blocking search there, so long time-budget analyses never freeze the
// UI. The worker owns no game: the main thread sends a QFEN, the worker
// sets it as root (qr_set_qfen) and answers single-shot `go` requests.
// Repeated `go` at the same QFEN reuses the MCTS/TT state inside the
// worker, so continuous analysis still converges like an infinite search.
//
// Everything degrades silently to main-thread slicing when workers are
// unavailable (some file:// sandboxes block blob workers).
// =====================================================================

// Worker-side glue. Kept as a real function so it can be stringified
// into the blob source verbatim -- it never runs on the main thread.
function __ewGlue() {
    var MOD = null, booted = false, queue = [], curQfen = null;
    var NNUE_PATH = '/data/nnue/nnue_weights_int8.bin';
    function clamp(v, lo, hi) { v = v | 0; return v < lo ? lo : (v > hi ? hi : v); }
    function handle(m) {
        if (m.cmd === 'config') {
            try {
                if (m.nnue) MOD.ccall('qr_load_nnue_weights', 'number', ['string'], [NNUE_PATH]);
                else MOD._qr_set_eval_heuristic();
                MOD._qr_set_mcab_enabled((m.mcab && m.nnue) ? 1 : 0);
            } catch (e) { /* keep whatever mode the module booted in */ }
            return;
        }
        if (m.cmd === 'position') {
            var qfen = String(m.qfen || '');
            if (qfen !== curQfen || m.force) {
                try { MOD.ccall('qr_set_qfen', 'number', ['string'], [qfen]); } catch (e) {}
                curQfen = qfen;
            }
            return;
        }
        if (m.cmd === 'go') {
            if (!MOD) { postMessage({ type: 'info', token: m.token, lines: [] }); return; }
            // Interactive analysis is capped at 250ms per slice; engine-move
            // and hint requests pass their own (larger) cap so named
            // strength levels actually take effect.
            var cap = Math.max(20, Math.min((m.cap | 0) || 250, 20000));
            var lines = [];
            try {
                var count = MOD._qr_analyze(clamp(m.depth, 1, 40),
                                            clamp(Math.min(m.timeMs | 0, cap), 20, cap),
                                            clamp(m.multipv, 1, 5));
                if (!count || count < 0) count = MOD._qr_an_line_count();
                for (var i = 0; i < count; i++) {
                    var len = MOD._qr_an_line_len(i); if (len < 0) len = 0;
                    if (len > 12) len = 12;
                    var pv = [];
                    for (var j = 0; j < len; j++) {
                        pv.push({
                            isWall: MOD._qr_an_line_move_is_wall(i, j) === 1,
                            a: MOD._qr_an_line_move_a(i, j),
                            b: MOD._qr_an_line_move_b(i, j),
                            c: MOD._qr_an_line_move_c(i, j)
                        });
                    }
                    lines.push({ score: MOD._qr_an_line_score(i), visits: MOD._qr_an_line_visits(i), pv: pv });
                }
            } catch (e) { lines = []; }
            postMessage({
                type: 'info', token: m.token, lines: lines,
                nodes: MOD._qr_an_nodes(), depth: MOD._qr_an_depth(),
                mcab: MOD._qr_an_is_mcab()
            });
            return;
        }
    }
    self.onmessage = function (ev) {
        var m = ev.data;
        if (!m || !m.cmd) return;
        if (m.cmd === 'init') {
            var args = {};
            if (m.wasm) args.wasmBinary = new Uint8Array(m.wasm);
            if (m.data) args.getPreloadedPackage = function () { return m.data; };
            if (m.engineDir) args.locateFile = function (p) { return m.engineDir + p; };
            ZquoridorModule(args).then(function (M) {
                MOD = M;
                var nnue = 0;
                try { nnue = M.ccall('qr_load_nnue_weights', 'number', ['string'], [NNUE_PATH]); } catch (e) {}
                if (!nnue) {
                    try { M._qr_set_eval_heuristic(); M._qr_set_mcab_enabled(0); } catch (e2) {}
                }
                booted = true;
                postMessage({ type: 'boot', ok: 1, nnue: nnue === 1 ? 1 : 0 });
                while (queue.length) handle(queue.shift());
            }).catch(function (err) {
                postMessage({ type: 'boot', ok: 0, err: String(err) });
            });
            return;
        }
        if (!booted || !MOD) { queue.push(m); return; }
        handle(m);
    };
}

var Ew = { w: null, ready: false, failed: false, seq: 1, pending: {}, lastPos: null };

function ewAvailable() {
    return typeof Worker !== 'undefined' && typeof Blob !== 'undefined' &&
           typeof URL !== 'undefined' && typeof URL.createObjectURL === 'function';
}

function ewStart() {
    if (!ewAvailable() || Ew.w || Ew.failed) return;
    var src;
    try {
        // The emscripten modularized factory closes over `_scriptName`
        // (declared in its IIFE wrapper), so re-declare it here before
        // stringifying the factory into the blob.
        src = 'var _scriptName;\n' +
              'var ZquoridorModule = (' + window.ZquoridorModule.toString() + ');\n' +
              '(' + __ewGlue.toString() + ')();';
        var blob = new Blob([src], { type: 'application/javascript' });
        Ew.w = new Worker(URL.createObjectURL(blob));
    } catch (e) {
        try { console.warn('[zq] engine worker unavailable -- staying on the main thread', e); } catch (e2) {}
        Ew.failed = true; Ew.w = null;
        return;
    }
    Ew.w.onmessage = function (ev) {
        var m = ev.data;
        if (!m || !m.type) return;
        if (m.type === 'boot') {
            if (!m.ok) { ewFail(); return; }
            Ew.ready = true;
            Ew.lastPos = null;
            ewPushConfig();
            return;
        }
        if (m.type === 'info') {
            var cb = Ew.pending[m.token];
            delete Ew.pending[m.token];
            if (cb) cb(m);
        }
    };
    Ew.w.onerror = function () { ewFail(); };

    // In the standalone build the wasm/data bytes already sit in globals --
    // hand copies to the worker so it needs no fetch (file:// safe). In dev
    // mode they don't exist and the worker fetches zquoridor.wasm itself,
    // located relative to this page.
    var wasmBuf = null, dataBuf = null, transfer = [];
    try {
        if (typeof __QR_WASM_BYTES__ !== 'undefined' && __QR_WASM_BYTES__) {
            wasmBuf = __QR_WASM_BYTES__.slice().buffer;
            transfer.push(wasmBuf);
        }
        if (typeof __QR_DATA_BYTES__ !== 'undefined' && __QR_DATA_BYTES__) {
            dataBuf = __QR_DATA_BYTES__.slice(0);
            transfer.push(dataBuf);
        }
    } catch (e3) { /* dev mode: nothing embedded */ }
    var engineDir = '';
    try { engineDir = new URL('.', location.href).href; } catch (e4) {}
    Ew.w.postMessage({ cmd: 'init', wasm: wasmBuf, data: dataBuf, engineDir: engineDir }, transfer);
}

// Worker died or failed to boot: unblock every waiter with an empty
// answer; callers fall back to main-thread slicing from now on.
function ewFail() {
    var had = !!Ew.w;
    Ew.ready = false;
    Ew.failed = true;
    if (Ew.w) { try { Ew.w.terminate(); } catch (e) {} }
    Ew.w = null;
    Ew.pending = {};
    if (had) {
        try { console.warn('[zq] engine worker lost -- falling back to main thread'); } catch (e) {}
    }
}

function ewPushConfig() {
    if (!Ew.ready || !Ew.w) return;
    Ew.w.postMessage({
        cmd: 'config',
        nnue: (nnueLoaded && settings.nnue) ? 1 : 0,
        mcab: (nnueLoaded && settings.nnue && settings.mcab) ? 1 : 0
    });
}

// Position sync with coalescing: consecutive requests at the same cursor
// reuse the worker's tree instead of resetting it per slice.
function ewAnalyzeAt(qfen, depth, timeMs, multipv, cb, capMs) {
    if (!Ew.ready || !Ew.w) return false;
    qfen = String(qfen || '');
    if (Ew.lastPos !== qfen) {
        Ew.lastPos = qfen;
        Ew.w.postMessage({ cmd: 'position', qfen: qfen });
    }
    var token = Ew.seq++;
    var done = false;
    // Watchdog: never leave the UI blocked on a wedged worker.
    var guard = setTimeout(function () {
        if (done) return;
        done = true;
        delete Ew.pending[token];
        cb(null);
    }, Math.max(2000, (timeMs | 0) + 5000));
    Ew.pending[token] = function (m) {
        if (done) return;
        done = true;
        clearTimeout(guard);
        cb(m);
    };
    Ew.w.postMessage({
        cmd: 'go', token: token, depth: depth,
        timeMs: timeMs, multipv: multipv,
        cap: capMs || 250
    });
    return true;
}

// =====================================================================
// SECTION 14 -- analysis mode
// =====================================================================

var analysisOn = false;
var analysisTimer = null;
var analysisGen = 0;
var analysisLines = [];
var ANALYSIS_SLICE_MS = 250;   // hard cap from the plan: never block longer

// One blocking call to the engine, decoded into JS objects. Kept small so
// every caller (loop, hint, blunder check) shares one decode path.
function runAnalyzeSlice(depth, timeMs, multipv) {
    if (!has('qr_analyze')) return [];
    var count = Q.qr_analyze(depth, clamp(timeMs, 20, ANALYSIS_SLICE_MS), clamp(multipv, 1, PV_MAX));
    if (!count || count <= 0) count = Q.qr_an_line_count();
    var out = [];
    for (var i = 0; i < count; i++) {
        var len = clamp(Q.qr_an_line_len(i), 0, PV_PLIES);
        var pv = [];
        for (var j = 0; j < len; j++) {
            pv.push({
                isWall: Q.qr_an_line_move_is_wall(i, j) === 1,
                a: Q.qr_an_line_move_a(i, j),
                b: Q.qr_an_line_move_b(i, j),
                c: Q.qr_an_line_move_c(i, j)
            });
        }
        out.push({ score: Q.qr_an_line_score(i), visits: Q.qr_an_line_visits(i), pv: pv });
    }
    return out;
}

function analysisParams() {
    return {
        depth: clamp(settings.anDepth, 1, 20),
        timeMs: clamp(Math.min(settings.anTimeMs, ANALYSIS_SLICE_MS), 20, ANALYSIS_SLICE_MS),
        multipv: clamp(settings.anMultiPv, 1, PV_MAX)
    };
}

function startAnalysis() {
    if (!has('qr_analyze')) { toast('Analysis needs qr_analyze -- not available', 'warn'); return; }
    analysisOn = true;
    var btn = $('an-eng-btn');
    if (btn) { btn.classList.add('on'); btn.setAttribute('aria-pressed', 'true'); }
    restartAnalysis();
}

function stopAnalysis() {
    analysisOn = false;
    var btn = $('an-eng-btn');
    if (btn) { btn.classList.remove('on'); btn.setAttribute('aria-pressed', 'false'); }
    stopAnalysisSearchOnly();
}

// Kills the pending slice but leaves the toggle state alone -- used when
// navigating, so the loop resumes at the new position.
function stopAnalysisSearchOnly() {
    analysisGen++;
    if (analysisTimer) { clearTimeout(analysisTimer); analysisTimer = null; }
}

function restartAnalysis() {
    stopAnalysisSearchOnly();
    if (!analysisOn || gameOver() || bcRunning) { renderAnalysisLines([]); return; }
    var myGen = analysisGen;
    analysisTimer = setTimeout(function () { analysisSlice(myGen); }, 15);
}

function analysisSlice(myGen) {
    if (myGen !== analysisGen || !analysisOn) return;
    var p = analysisParams();
    var done = function (res) {
        if (myGen !== analysisGen || !analysisOn) return;
        var lines = res ? (res.lines || []) : null;
        if (lines === null) {
            // Worker answer lost -- redo this slice on the main thread.
            lines = runAnalyzeSlice(p.depth, p.timeMs, p.multipv);
            if (myGen !== analysisGen || !analysisOn) return;
        }
        renderAnalysisResult(lines);
        // Re-arm. The loop keeps re-searching the same position so the
        // numbers converge over time exactly like an infinite search would.
        if (analysisOn && myGen === analysisGen) {
            analysisTimer = setTimeout(function () { analysisSlice(myGen); }, 30);
        }
    };
    // Worker first (non-blocking); sync slicing only as a fallback.
    var dispatched = ewAnalyzeAt(Q.qr_get_qfen(), p.depth, p.timeMs, p.multipv, function (res) { done(res); });
    if (!dispatched) done(null);
}

// Shared tail of every continuous-analysis slice: lines -> UI. Used by
// both the worker path and the main-thread fallback.
function renderAnalysisResult(lines) {
    analysisLines = lines;
    renderAnalysisLines(lines);
    if (!lines.length) return;
    var pct = evalToWhitePercent(toP0Score(lines[0].score, Q.qr_turn()));
    renderEvalBars(pct);
    anArrows = lines[0].pv.length ? arrowsForMove(lines[0].pv[0], '#c8a84b') : [];
    if (boards.an) {
        boardCall(boards.an, 'setOverlays', [{
            lastMove: settings.highlight ? lastMoveOfCursor() : null,
            hint: null,
            arrows: settings.anArrows ? anArrows : [],
            ghostWall: null,
            paths: settings.paths ? [readPath(0), readPath(1)] : null
        }]);
        boardCall(boards.an, 'render', []);
    }
    drawEvalGraph();
}

function arrowsForMove(mv, color) {
    if (!mv) return [];
    if (mv.isWall) return [{ wall: { o: mv.a, r: mv.b, c: mv.c }, color: color }];
    return [{ from: Q.qr_pawn(Q.qr_turn()), to: mv.a, color: color, width: 3 }];
}

function pvText(pv, startTurn) {
    var parts = [];
    for (var i = 0; i < pv.length && i < PV_PLIES; i++) parts.push(moveNotation(pv[i]));
    return parts.join(' ');
}

function renderAnalysisLines(lines) {
    var multipv = clamp(settings.anMultiPv, 1, PV_MAX);
    var mover = Q.qr_turn();
    var isMcab = has('qr_an_is_mcab') ? Q.qr_an_is_mcab() === 1 : false;
    for (var i = 0; i < PV_MAX; i++) {
        var row = $('an-line' + (i + 1));
        if (!row) continue;
        if (i >= multipv) { row.style.display = 'none'; continue; }
        row.style.display = '';
        var e = lines && lines[i];
        var rank = row.querySelector('.line-rank');
        if (rank) rank.textContent = String(i + 1);
        if (!e) {
            setText('an-ls' + (i + 1), '');
            setText('an-lm' + (i + 1), '—');
            setText('an-lv' + (i + 1), '');
            row.removeAttribute('data-mv');
            continue;
        }
        var p0 = toP0Score(e.score, mover);
        setText('an-ls' + (i + 1), formatScore(p0) + '  ' + formatPercent(evalToWhitePercent(p0)));
        setText('an-lm' + (i + 1), pvText(e.pv, mover) || '—');
        setText('an-lv' + (i + 1), (isMcab && e.visits) ? (e.visits + ' v') : '');
        row.setAttribute('data-mv', e.pv.length ? JSON.stringify(e.pv[0]) : '');
    }
    var st = [];
    if (has('qr_an_depth')) st.push('d' + Q.qr_an_depth());
    if (has('qr_an_nodes')) st.push(Q.qr_an_nodes() + ' nodes');
    st.push(isMcab ? 'hybrid MCTS' : 'alpha-beta');
    st.push((mover === 0 ? 'player 0' : 'player 1') + ' to move');
    if (!bcRunning) setText('an-status', st.join('  ·  '));
}

function wireAnalysisLineRows() {
    for (var i = 0; i < PV_MAX; i++) {
        (function (idx) {
            var row = $('an-line' + (idx + 1));
            if (!row) return;
            row.addEventListener('mouseenter', function () {
                var raw = row.getAttribute('data-mv');
                if (!raw) return;
                var mv = null;
                try { mv = JSON.parse(raw); } catch (e) { return; }
                if (!mv || !boards.an) return;
                boardCall(boards.an, 'setOverlays', [{
                    lastMove: settings.highlight ? lastMoveOfCursor() : null,
                    hint: mv,
                    arrows: arrowsForMove(mv, '#4a7cc0'),
                    ghostWall: null,
                    paths: settings.paths ? [readPath(0), readPath(1)] : null
                }]);
                boardCall(boards.an, 'render', []);
            });
            row.addEventListener('mouseleave', function () { renderAll(); });
            row.addEventListener('click', function () {
                var raw = row.getAttribute('data-mv');
                if (!raw) return;
                var mv = null;
                try { mv = JSON.parse(raw); } catch (e) { return; }
                if (mv) analysisPlay(mv);
            });
        })(i);
    }
}

function setMultiPv(v) {
    settings.anMultiPv = clamp(v, 1, PV_MAX);
    setText('an-pv-count', String(settings.anMultiPv));
    saveSettings();
    renderAnalysisLines(analysisLines);
    if (analysisOn) restartAnalysis();
}

// =====================================================================
// SECTION 15 -- blunder check ("Analyze game")
//
// Classification is by WIN-PROBABILITY DROP in percentage points, from
// the mover's own perspective:
//   < 2   best / good     (and `!` when it matched the engine's top move)
//   2-5   inaccuracy  ?!
//   5-12  mistake     ?
//   > 12  blunder     ??
// =====================================================================

var bcRunning = false;
var bcGen = 0;

function classifyDrop(drop, matchedBest) {
    if (matchedBest) return '!';
    if (drop > 12) return '??';
    if (drop > 5) return '?';
    if (drop > 2) return '?!';
    return '';
}

function cancelBlunderCheck() {
    bcGen++;
    if (!bcRunning) return;
    bcRunning = false;
    var btn = $('an-btn-blunder');
    if (btn) btn.classList.remove('on');
}

function toggleBlunderCheck() {
    if (bcRunning) {
        cancelBlunderCheck();
        setText('an-status', 'Game analysis cancelled');
        renderAll();
        return;
    }
    if (!has('qr_analyze')) { toast('Game analysis needs qr_analyze', 'warn'); return; }
    var total = Q.qr_history_len();
    if (!total) { setText('an-status', 'No moves to analyse'); return; }

    stopAnalysis();
    bcRunning = true;
    bcGen++;
    var myGen = bcGen;
    var btn = $('an-btn-blunder');
    if (btn) btn.classList.add('on');

    var savedCursor = Q.qr_history_cursor();
    plyInfo = [];
    posEval = [];
    var p = analysisParams();
    var depth = p.depth;
    var sliceMs = p.timeMs;
    var i = 0;

    // One position evaluation: worker first, main-thread slicing as the
    // fallback (the caller must have qr_goto_ply'd to `ply` already in
    // that case -- the worker path is cursor-independent).
    var evalAt = function (ply, cb) {
        var dispatched = ewAnalyzeAt(Q.qr_get_qfen(), depth, sliceMs, 1, function (res) {
            if (myGen !== bcGen || !bcRunning) return;
            if (res && res.lines) { cb(res.lines); return; }
            Q.qr_goto_ply(ply);
            cb(runAnalyzeSlice(depth, sliceMs, 1));
        });
        if (!dispatched) {
            setTimeout(function () {
                if (myGen !== bcGen || !bcRunning) return;
                cb(runAnalyzeSlice(depth, sliceMs, 1));
            }, 0);
        }
    };

    var step = function () {
        if (myGen !== bcGen || !bcRunning) return;
        if (i >= total) {
            bcRunning = false;
            if (btn) btn.classList.remove('on');
            Q.qr_goto_ply(savedCursor);
            renderSummary(total);
            renderAll();
            setText('an-status', 'Game analysis done — ' + total + ' plies');
            return;
        }
        setText('an-status', 'Analysing ply ' + (i + 1) + '/' + total + '… (click again to cancel)');
        Q.qr_goto_ply(i);

        evalAt(i, function (before) {
            if (myGen !== bcGen || !bcRunning) return;
            // Position BEFORE the move: engine's best, from the mover's view.
            var mover = Q.qr_turn();
            var bestPct = before.length ? moverWinPercent(before[0].score) : 50;
            var bestMv = (before.length && before[0].pv.length) ? before[0].pv[0] : null;
            recordPosEval(i, evalToWhitePercent(toP0Score(before.length ? before[0].score : 0, mover)));
            var played = histMove(i);

            // Position AFTER the move: the score is now the OPPONENT's, so
            // the mover's own win probability is its complement.
            Q.qr_goto_ply(i + 1);
            evalAt(i + 1, function (after) {
                if (myGen !== bcGen || !bcRunning) return;
                var afterScore = after.length ? after[0].score : 0;
                var playedPct = 100 - moverWinPercent(afterScore);
                recordPosEval(i + 1, evalToWhitePercent(toP0Score(afterScore, Q.qr_turn())));

                var drop = Math.max(0, bestPct - playedPct);
                var matched = !!bestMv && bestMv.isWall === played.isWall &&
                              bestMv.a === played.a &&
                              (!played.isWall || (bestMv.b === played.b && bestMv.c === played.c));
                plyInfo[i] = { annot: classifyDrop(drop, matched), drop: drop, mover: mover };

                i++;
                renderMoveLog('an-movelog');
                drawEvalGraph();
                setTimeout(step, 0);
            });
        });
    };
    setTimeout(step, 0);
}

// Accuracy is a presentation heuristic, not an engine metric: each move
// scores 100 minus 2.5x its win-probability drop, floored at 0, averaged
// per player. Documented here so nobody mistakes it for a calibrated
// number from the engine.
function renderSummary(total) {
    var el = $('an-summary');
    if (!el) return;
    var acc = [[], []];
    var counts = [{ '!': 0, '': 0, '?!': 0, '?': 0, '??': 0 }, { '!': 0, '': 0, '?!': 0, '?': 0, '??': 0 }];
    for (var i = 0; i < total; i++) {
        var info = plyInfo[i];
        if (!info) continue;
        var m = info.mover;
        acc[m].push(clamp(100 - 2.5 * info.drop, 0, 100));
        counts[m][info.annot] = (counts[m][info.annot] || 0) + 1;
    }
    var html = '';
    for (var p = 0; p < 2; p++) {
        var a = acc[p].length ? (acc[p].reduce(function (x, y) { return x + y; }, 0) / acc[p].length) : null;
        html += '<div class="sum-row p' + p + '">' +
            '<span class="sum-name">Player ' + p + '</span>' +
            '<span class="sum-acc">' + (a === null ? '--' : a.toFixed(1) + '%') + '</span>' +
            '<span class="g-best">' + counts[p]['!'] + ' !</span>' +
            '<span class="g-inacc">' + counts[p]['?!'] + ' ?!</span>' +
            '<span class="g-mistake">' + counts[p]['?'] + ' ?</span>' +
            '<span class="g-blunder">' + counts[p]['??'] + ' ??</span>' +
            '</div>';
    }
    el.innerHTML = html;
}

// =====================================================================
// SECTION 16 -- eval graph
// =====================================================================

function graphColors() {
    var cs = null;
    try { cs = getComputedStyle(document.documentElement); } catch (e) { cs = null; }
    var pick = function (name, fb) {
        if (!cs) return fb;
        var v = cs.getPropertyValue(name);
        return (v && v.trim()) ? v.trim() : fb;
    };
    return {
        bg: pick('--surf2', '#1c1c28'),
        grid: pick('--bor', '#252538'),
        gold: pick('--gold', '#c8a84b'),
        gold2: pick('--gold2', '#e6c96e'),
        muted: pick('--muted', '#565672'),
        red: pick('--red', '#c0394a')
    };
}

function drawEvalGraph() {
    var cv = $('an-graph');
    if (!cv || !cv.getContext) return;
    // The resize observer can fire before the WASM module finishes booting;
    // Q bindings are not safe to touch until then.
    if (!has('qr_history_len')) return;
    var wrap = $('an-graph-wrap');
    var W = Math.max(80, (wrap ? wrap.clientWidth : 0) || cv.clientWidth || 240);
    var H = Math.max(50, cv.clientHeight || 90);
    var dpr = window.devicePixelRatio || 1;
    cv.width = Math.round(W * dpr);
    cv.height = Math.round(H * dpr);
    cv.style.width = W + 'px';
    var cx = cv.getContext('2d');
    cx.setTransform(dpr, 0, 0, dpr, 0, 0);

    var col = graphColors();
    cx.fillStyle = col.bg;
    cx.fillRect(0, 0, W, H);

    var len = Q.qr_history_len();
    var pts = [];
    for (var k = 0; k <= len; k++) pts.push(posEval[k] === undefined ? null : posEval[k]);
    var known = pts.filter(function (v) { return v !== null; }).length;

    var midY = H / 2;
    cx.strokeStyle = col.grid;
    cx.lineWidth = 1;
    cx.beginPath(); cx.moveTo(0, midY + 0.5); cx.lineTo(W, midY + 0.5); cx.stroke();

    if (known < 2) {
        cx.fillStyle = col.muted;
        cx.font = '10px JetBrains Mono, monospace';
        cx.textAlign = 'center'; cx.textBaseline = 'middle';
        cx.fillText('Run "Analyze game" for the eval graph', W / 2, H / 2);
        return;
    }

    // Player 0 above the midline: 100% -> y = 2, 0% -> y = H-2.
    var yOf = function (pct) { return H - 2 - (clamp(pct, 0, 100) / 100) * (H - 4); };
    var stepX = W / Math.max(1, pts.length - 1);
    var xOf = function (i) { return i * stepX; };

    // Fill under the curve.
    cx.beginPath();
    cx.moveTo(0, midY);
    var last = 50;
    for (var i = 0; i < pts.length; i++) {
        if (pts[i] !== null) last = pts[i];
        cx.lineTo(xOf(i), yOf(last));
    }
    cx.lineTo(W, midY);
    cx.closePath();
    cx.fillStyle = 'rgba(200,168,75,.20)';
    cx.fill();

    // The curve itself.
    cx.beginPath();
    last = 50;
    for (var j = 0; j < pts.length; j++) {
        if (pts[j] !== null) last = pts[j];
        var x = xOf(j), y = yOf(last);
        if (j === 0) cx.moveTo(x, y); else cx.lineTo(x, y);
    }
    cx.strokeStyle = col.gold;
    cx.lineWidth = 1.5;
    cx.lineJoin = 'round';
    cx.stroke();

    // Blunder markers, at the position AFTER the offending ply.
    for (var b = 0; b < plyInfo.length; b++) {
        var info = plyInfo[b];
        if (!info || (info.annot !== '??' && info.annot !== '?')) continue;
        var v = pts[b + 1];
        if (v === null || v === undefined) continue;
        cx.beginPath();
        cx.arc(xOf(b + 1), yOf(v), info.annot === '??' ? 3.5 : 2.5, 0, Math.PI * 2);
        cx.fillStyle = col.red;
        cx.fill();
    }

    // Current ply marker.
    var cur = Q.qr_history_cursor();
    if (cur >= 0 && cur < pts.length) {
        var cv0 = pts[cur] === null || pts[cur] === undefined ? 50 : pts[cur];
        cx.beginPath();
        cx.arc(xOf(cur), yOf(cv0), 3.5, 0, Math.PI * 2);
        cx.fillStyle = col.gold2;
        cx.fill();
    }
}

function onGraphClick(ev) {
    var cv = $('an-graph');
    if (!cv) return;
    var len = Q.qr_history_len();
    if (!len) return;
    var r = cv.getBoundingClientRect();
    var frac = (ev.clientX - r.left) / Math.max(1, r.width);
    gotoPly(clamp(Math.round(frac * len), 0, len));
}

// =====================================================================
// SECTION 17 -- position editor
// =====================================================================

var edTool = 'p0';

function editorEnter() {
    if (!has('qr_edit_begin')) { toast('Editor needs the qr_edit_* exports', 'warn'); return; }
    Q.qr_edit_begin();
    edRefresh();
}

function edReadPosition() {
    var wallsH = new Uint8Array(WS * WS);
    var wallsV = new Uint8Array(WS * WS);
    var ownH = new Int8Array(WS * WS);
    var ownV = new Int8Array(WS * WS);
    for (var i = 0; i < WS * WS; i++) {
        wallsH[i] = Q.qr_edit_wall_h_bit(i) === 1 ? 1 : 0;
        wallsV[i] = Q.qr_edit_wall_v_bit(i) === 1 ? 1 : 0;
        ownH[i] = -1; ownV[i] = -1;
    }
    return {
        pawns: [Q.qr_edit_pawn(0), Q.qr_edit_pawn(1)],
        wallsH: wallsH,
        wallsV: wallsV,
        wallOwner: { h: ownH, v: ownV },
        turn: Q.qr_edit_turn(),
        wallsLeft: [Q.qr_edit_walls_left(0), Q.qr_edit_walls_left(1)],
        winner: -1
    };
}

var EDIT_ERRORS = [
    [1, 'player 0 has no path to its goal'],
    [2, 'player 1 has no path to its goal'],
    [4, 'both pawns are on the same cell'],
    [8, 'more than 20 walls are placed'],
    [16, 'a pawn already stands on its goal row']
];

function edRefresh(skipQfen) {
    if (!boards.ed && !$('ed-validity')) return;
    var pos = edReadPosition();
    if (boards.ed) {
        boardCall(boards.ed, 'setFlipped', [settings.flipped]);
        boardCall(boards.ed, 'setEditTool', [edTool]);
        boardCall(boards.ed, 'setPosition', [pos]);
        boardCall(boards.ed, 'setInteractive', [true]);
        boardCall(boards.ed, 'setOverlays', [{ lastMove: null, hint: null, arrows: [], ghostWall: null, paths: null }]);
        boardCall(boards.ed, 'render', []);
    }
    setText('ed-walls0-val', String(pos.wallsLeft[0]));
    setText('ed-walls1-val', String(pos.wallsLeft[1]));
    var r0 = $('ed-turn-p0'), r1 = $('ed-turn-p1');
    if (r0) { r0.checked = pos.turn === 0; r0.classList.toggle('active', pos.turn === 0); }
    if (r1) { r1.checked = pos.turn === 1; r1.classList.toggle('active', pos.turn === 1); }

    var mask = Q.qr_edit_validate();
    var vEl = $('ed-validity');
    if (vEl) {
        vEl.classList.remove('ok', 'bad');
        if (!mask) {
            vEl.classList.add('ok');
            vEl.textContent = 'Position is legal.';
        } else {
            vEl.classList.add('bad');
            var msgs = EDIT_ERRORS.filter(function (e) { return (mask & e[0]) !== 0; })
                                  .map(function (e) { return e[1]; });
            vEl.textContent = 'Illegal position: ' + msgs.join('; ') + '.';
        }
    }
    setDisabled('ed-btn-play', !!mask);
    setDisabled('ed-btn-analyze', !!mask);

    if (!skipQfen) {
        var ta = $('ed-qfen');
        if (ta && document.activeElement !== ta) ta.value = Q.qr_edit_get_qfen();
    }
}

function edSetTool(tool) {
    edTool = tool;
    ['p0', 'p1', 'wallh', 'wallv', 'erase'].forEach(function (t) {
        var el = $('ed-tool-' + t);
        if (el) el.classList.toggle('active', t === tool);
    });
    boardCall(boards.ed, 'setEditTool', [tool]);
    boardCall(boards.ed, 'render', []);
}

function edCommit(target) {
    if (Q.qr_edit_validate() !== 0) { toast('Cannot commit an illegal position', 'warn'); return; }
    if (Q.qr_edit_commit() !== 1) { toast('The engine refused the position', 'bad'); return; }
    gameGen++;
    busy = false;
    resetAnnotations();
    hintOverlay = null;
    anArrows = [];
    toast('Position loaded', 'ok');
    switchTab(target);
    renderAll();
    if (target === 'play') maybeEngineMove();
    else if (analysisOn) restartAnalysis();
}

// =====================================================================
// SECTION 18 -- tabs
// =====================================================================

function switchTab(name) {
    activeTab = name;
    ['play', 'analysis', 'editor'].forEach(function (t) {
        var panel = $(t + '-panel');
        if (panel) panel.classList.toggle('active', t === name);
        var tab = $('tab-' + t);
        if (tab) {
            tab.classList.toggle('active', t === name);
            tab.setAttribute('aria-selected', t === name ? 'true' : 'false');
        }
    });
    if (name !== 'analysis') stopAnalysisSearchOnly();
    if (name === 'editor') editorEnter();
    resizeAll();
    if (name === 'analysis') {
        renderAll();
        if (analysisOn) restartAnalysis();
    } else {
        renderAll();
    }
}

function resizeAll() {
    ['play', 'an', 'ed'].forEach(function (k) { boardCall(boards[k], 'resize', []); });
    drawEvalGraph();
}

// =====================================================================
// SECTION 19 -- modals
// =====================================================================

function showModal(id, show) {
    var el = $(id);
    if (!el) return;
    el.classList.toggle('show', show);
    el.removeAttribute('hidden');
    // Inline display wins over whatever the stylesheet does, so a modal is
    // never invisible because of a class-name mismatch with the shell.
    el.style.display = show ? 'flex' : 'none';
    el.setAttribute('aria-hidden', show ? 'false' : 'true');
}

function anyModalOpen() {
    return ['settings-modal', 'about-modal', 'text-modal'].some(function (id) {
        var el = $(id);
        return el && el.style.display !== 'none' && el.classList.contains('show');
    });
}

function closeAllModals() {
    ['settings-modal', 'about-modal', 'text-modal'].forEach(function (id) { showModal(id, false); });
}

// ---- text modal (QFEN / game text) -----------------------------------
var textModalLoader = null;

function openTextModal(title, content, loader) {
    setText('text-modal-title', title);
    var ta = $('text-modal-area');
    if (ta) ta.value = content;
    textModalLoader = loader || null;
    setDisabled('text-modal-load', !loader);
    showModal('text-modal', true);
    if (ta) { ta.focus(); ta.select(); }
}

function openQfenModal() {
    openTextModal('QFEN', Q.qr_get_qfen(), function (txt) {
        if (Q.qr_set_qfen(txt.trim()) !== 1) { toast('Invalid QFEN', 'bad'); return false; }
        gameGen++; busy = false;
        resetAnnotations();
        toast('Position loaded from QFEN', 'ok');
        return true;
    });
}

function openGameTextModal() {
    openTextModal('Game text', Q.qr_get_game_text(), function (txt) {
        if (Q.qr_set_game_text(txt) !== 1) { toast('Could not parse the game text', 'bad'); return false; }
        gameGen++; busy = false;
        resetAnnotations();
        toast('Game loaded', 'ok');
        return true;
    });
}

// ---- settings modal --------------------------------------------------
function syncSettingsUi() {
    document.querySelectorAll('#cfg-board-themes .cfg-opt').forEach(function (el) {
        el.classList.toggle('active', el.getAttribute('data-theme') === settings.boardTheme);
    });
    document.querySelectorAll('#cfg-pawn-styles .cfg-opt').forEach(function (el) {
        el.classList.toggle('active', el.getAttribute('data-style') === settings.pawnStyle);
    });
    document.querySelectorAll('#cfg-ui-theme .cfg-opt').forEach(function (el) {
        el.classList.toggle('active', el.getAttribute('data-ui') === settings.uiTheme);
    });
    var checks = {
        'cfg-highlight': 'highlight', 'cfg-paths': 'paths', 'cfg-coords': 'coords',
        'cfg-dots': 'dots', 'cfg-anim': 'anim', 'cfg-sound': 'sound',
        'cfg-haptics': 'haptics',
        'cfg-evalbar': 'evalbar', 'cfg-movelog-eval': 'movelogEval',
        'cfg-nnue': 'nnue', 'cfg-mcab': 'mcab'
    };
    Object.keys(checks).forEach(function (id) {
        var el = $(id);
        if (el) el.checked = !!settings[checks[id]];
    });
    var vol = $('cfg-sound-vol');
    if (vol) vol.value = String(settings.soundVol);
}

function applyUiTheme() {
    document.documentElement.setAttribute('data-ui-theme', settings.uiTheme);
    document.documentElement.setAttribute('data-board-theme', settings.boardTheme);
}

// Applies engine-affecting settings (NNUE / hybrid MCTS) to the module.
function applyEngineSettings() {
    if (nnueLoaded) {
        if (settings.nnue) {
            // Reloading the weights is the only documented way back into
            // NNUE mode after qr_set_eval_heuristic().
            Q.qr_load_nnue_weights(NNUE_PATH);
        } else {
            Q.qr_set_eval_heuristic();
        }
        Q.qr_set_mcab_enabled(settings.mcab && settings.nnue ? 1 : 0);
    } else {
        Q.qr_set_eval_heuristic();
        Q.qr_set_mcab_enabled(0);
    }
    ewPushConfig();
    updateEngineInfo();
}

function updateEngineInfo() {
    var isNnue = has('qr_eval_mode_is_nnue') ? Q.qr_eval_mode_is_nnue() === 1 : false;
    var isMcab = has('qr_mcab_active') ? Q.qr_mcab_active() === 1 : false;
    engineInfoText = (isNnue ? 'NNUE' : 'Heuristic') + (isMcab ? ' + hybrid MCTS' : ' + alpha-beta');
    setText('pl-engine-info', engineInfoText);
}

// =====================================================================
// SECTION 20 -- wiring
// =====================================================================

function wireTabs() {
    on('tab-play', 'click', function () { switchTab('play'); });
    on('tab-analysis', 'click', function () { switchTab('analysis'); });
    on('tab-editor', 'click', function () { switchTab('editor'); });
    on('btn-settings', 'click', function () { syncSettingsUi(); showModal('settings-modal', true); });
    on('btn-about', 'click', function () { showModal('about-modal', true); });
    on('about-close', 'click', function () { showModal('about-modal', false); });
}

function wirePlay() {
    on('pl-btn-new', 'click', function () { audioUnlock(); newGame(); });
    on('pl-btn-flip', 'click', doFlip);
    on('pl-btn-side', 'click', doSwitchSide);
    on('pl-btn-undo', 'click', doUndo);
    on('pl-btn-redo', 'click', doRedo);
    on('pl-btn-hint', 'click', doHint);
    on('pl-btn-takeback', 'click', doTakeback);

    on('pl-wall-h', 'click', function () { setWallOrientation(0); });
    on('pl-wall-v', 'click', function () { setWallOrientation(1); });

    var lvl = $('pl-sel-level');
    if (lvl) {
        lvl.value = String(settings.strength);
        lvl.addEventListener('change', function () {
            settings.strength = lvl.value === 'custom' ? 'custom' : String(lvl.value);
            saveSettings();
            applySearchModeUi();
        });
    }

    var mode = $('pl-sel-mode');
    if (mode) {
        mode.value = settings.searchMode;
        mode.addEventListener('change', function () {
            settings.searchMode = mode.value;
            saveSettings();
            applySearchModeUi();
            if (settings.searchMode === 'game') initClocks(); else stopClock();
            renderAll();
        });
    }
    var d = $('pl-sel-depth');
    if (d) {
        d.value = String(settings.searchDepth);
        d.addEventListener('change', function () {
            settings.searchDepth = clamp(parseInt(d.value, 10) || 6, 2, 12);
            saveSettings();
        });
    }
    var t = $('pl-sel-time');
    if (t) {
        t.value = String(settings.searchTimeMs);
        t.addEventListener('change', function () {
            settings.searchTimeMs = clamp(parseInt(t.value, 10) || 500, 50, 20000);
            saveSettings();
        });
    }
    var tc = $('pl-sel-tc');
    if (tc) {
        tc.value = settings.timeControl;
        tc.addEventListener('change', function () {
            settings.timeControl = tc.value;
            saveSettings();
            if (settings.searchMode === 'game') initClocks();
        });
    }
    applySearchModeUi();
}

function applySearchModeUi() {
    // Named strength levels hide the raw search knobs entirely.
    var custom = settings.strength === 'custom';
    setShown('pl-adv-group', custom);
    if (!custom) return;
    setShown('pl-sel-depth', settings.searchMode === 'depth');
    setShown('pl-sel-time', settings.searchMode === 'time');
    setShown('pl-sel-tc', settings.searchMode === 'game');
}

function setWallOrientation(o) {
    wallOrientation = o === 1 ? 1 : 0;
    var h = $('pl-wall-h'), v = $('pl-wall-v');
    if (h) {
        h.classList.toggle('selected', wallOrientation === 0);
        h.setAttribute('aria-pressed', wallOrientation === 0 ? 'true' : 'false');
    }
    if (v) {
        v.classList.toggle('selected', wallOrientation === 1);
        v.setAttribute('aria-pressed', wallOrientation === 1 ? 'true' : 'false');
    }
    boardCall(boards.play, 'setWallOrientation', [wallOrientation]);
    boardCall(boards.play, 'render', []);
}

// The wall toolbar's remaining count tracks the human side.
function renderWallCount() {
    var el = $('pl-wall-count');
    if (!el) return;
    var n = Q.qr_walls_left(settings.humanSide);
    el.textContent = n + ' left';
    el.style.color = n === 0 ? 'var(--red)' : '';
}

function wireAnalysis() {
    on('an-eng-btn', 'click', function () { analysisOn ? stopAnalysis() : startAnalysis(); });
    on('an-first', 'click', function () { gotoPly(0); });
    on('an-prev', 'click', function () { gotoPly(Q.qr_history_cursor() - 1); });
    on('an-next', 'click', function () { gotoPly(Q.qr_history_cursor() + 1); });
    on('an-last', 'click', function () { gotoPly(Q.qr_history_len()); });
    on('an-flip', 'click', doFlip);
    on('an-pv-plus', 'click', function () { setMultiPv(settings.anMultiPv + 1); });
    on('an-pv-minus', 'click', function () { setMultiPv(settings.anMultiPv - 1); });
    on('an-btn-qfen', 'click', openQfenModal);
    on('an-btn-game', 'click', openGameTextModal);
    on('an-btn-blunder', 'click', toggleBlunderCheck);
    on('an-btn-arrows', 'click', function () {
        settings.anArrows = !settings.anArrows;
        var b = $('an-btn-arrows');
        if (b) { b.classList.toggle('on', settings.anArrows); b.setAttribute('aria-pressed', String(settings.anArrows)); }
        saveSettings();
        renderAll();
    });
    var ad = $('an-sel-depth');
    if (ad) {
        ad.value = String(settings.anDepth);
        ad.addEventListener('change', function () {
            settings.anDepth = clamp(parseInt(ad.value, 10) || 8, 1, 20);
            saveSettings();
            if (analysisOn) restartAnalysis();
        });
    }
    var at = $('an-sel-time');
    if (at) {
        at.value = String(settings.anTimeMs);
        at.addEventListener('change', function () {
            settings.anTimeMs = clamp(parseInt(at.value, 10) || 500, 50, 20000);
            saveSettings();
            if (analysisOn) restartAnalysis();
        });
    }
    setText('an-pv-count', String(settings.anMultiPv));
    var arr = $('an-btn-arrows');
    if (arr) arr.classList.toggle('on', settings.anArrows);
    wireAnalysisLineRows();
    on('an-graph', 'click', onGraphClick);
}

function wireEditor() {
    [['ed-tool-p0', 'p0'], ['ed-tool-p1', 'p1'], ['ed-tool-wallh', 'wallh'],
     ['ed-tool-wallv', 'wallv'], ['ed-tool-erase', 'erase']].forEach(function (pair) {
        on(pair[0], 'click', function () { edSetTool(pair[1]); });
    });
    on('ed-btn-clear', 'click', function () { Q.qr_edit_clear(); edRefresh(); });
    on('ed-btn-reset', 'click', function () { Q.qr_edit_begin(); edRefresh(); });
    on('ed-btn-flip', 'click', function () { settings.flipped = !settings.flipped; saveSettings(); edRefresh(); });
    on('ed-turn-p0', 'change', function () { Q.qr_edit_set_turn(0); edRefresh(); });
    on('ed-turn-p1', 'change', function () { Q.qr_edit_set_turn(1); edRefresh(); });
    on('ed-turn-p0', 'click', function () { Q.qr_edit_set_turn(0); edRefresh(); });
    on('ed-turn-p1', 'click', function () { Q.qr_edit_set_turn(1); edRefresh(); });
    [0, 1].forEach(function (p) {
        on('ed-walls' + p + '-dec', 'click', function () {
            Q.qr_edit_set_walls_left(p, clamp(Q.qr_edit_walls_left(p) - 1, 0, MAX_WALLS));
            edRefresh();
        });
        on('ed-walls' + p + '-inc', 'click', function () {
            Q.qr_edit_set_walls_left(p, clamp(Q.qr_edit_walls_left(p) + 1, 0, MAX_WALLS));
            edRefresh();
        });
    });
    on('ed-btn-qfen-load', 'click', function () {
        var ta = $('ed-qfen');
        if (!ta) return;
        if (Q.qr_edit_set_qfen(ta.value.trim()) !== 1) { toast('Invalid QFEN', 'bad'); return; }
        edRefresh();
        toast('QFEN loaded into the editor', 'ok');
    });
    on('ed-btn-qfen-copy', 'click', function () { copyText(Q.qr_edit_get_qfen()); });
    on('ed-btn-play', 'click', function () { edCommit('play'); });
    on('ed-btn-analyze', 'click', function () { edCommit('analysis'); });
}

function wireSettings() {
    document.querySelectorAll('#cfg-board-themes .cfg-opt').forEach(function (el) {
        el.addEventListener('click', function () {
            settings.boardTheme = el.getAttribute('data-theme') || 'wood';
            saveSettings(); applyUiTheme(); syncSettingsUi(); applyBoardOptionsAll();
        });
    });
    document.querySelectorAll('#cfg-pawn-styles .cfg-opt').forEach(function (el) {
        el.addEventListener('click', function () {
            settings.pawnStyle = el.getAttribute('data-style') || 'disc';
            saveSettings(); syncSettingsUi(); applyBoardOptionsAll();
        });
    });
    document.querySelectorAll('#cfg-ui-theme .cfg-opt').forEach(function (el) {
        el.addEventListener('click', function () {
            settings.uiTheme = el.getAttribute('data-ui') === 'light' ? 'light' : 'dark';
            saveSettings(); applyUiTheme(); syncSettingsUi(); drawEvalGraph();
        });
    });
    var checks = {
        'cfg-highlight': 'highlight', 'cfg-paths': 'paths', 'cfg-coords': 'coords',
        'cfg-dots': 'dots', 'cfg-anim': 'anim',
        'cfg-haptics': 'haptics',
        'cfg-evalbar': 'evalbar', 'cfg-movelog-eval': 'movelogEval'
    };
    Object.keys(checks).forEach(function (id) {
        on(id, 'change', function (ev) {
            settings[checks[id]] = !!ev.target.checked;
            saveSettings();
            applyBoardOptionsAll();
        });
    });
    on('cfg-sound', 'change', function (ev) {
        settings.sound = !!ev.target.checked;
        saveSettings();
        if (settings.sound) { audioUnlock(); sndPawn(); }
    });
    on('cfg-sound-vol', 'change', function (ev) {
        settings.soundVol = clamp(parseInt(ev.target.value, 10) || 0, 0, 5);
        saveSettings();
        applySoundVolume();
        audioUnlock();
        sndPawn();
    });
    on('cfg-nnue', 'change', function (ev) {
        settings.nnue = !!ev.target.checked;
        saveSettings();
        applyEngineSettings();
        renderAll();
    });
    on('cfg-mcab', 'change', function (ev) {
        settings.mcab = !!ev.target.checked;
        saveSettings();
        applyEngineSettings();
    });
    on('cfg-done', 'click', function () { showModal('settings-modal', false); });
    on('settings-modal', 'click', function (ev) {
        if (ev.target && ev.target.id === 'settings-modal') showModal('settings-modal', false);
    });
    on('about-modal', 'click', function (ev) {
        if (ev.target && ev.target.id === 'about-modal') showModal('about-modal', false);
    });
}

function wireTextModal() {
    on('text-modal-close', 'click', function () { showModal('text-modal', false); });
    on('text-modal-copy', 'click', function () {
        var ta = $('text-modal-area');
        if (ta) copyText(ta.value);
    });
    on('text-modal-load', 'click', function () {
        var ta = $('text-modal-area');
        if (!ta || !textModalLoader) return;
        if (textModalLoader(ta.value)) {
            showModal('text-modal', false);
            renderAll();
            if (analysisOn) restartAnalysis();
        }
    });
    on('text-modal', 'click', function (ev) {
        if (ev.target && ev.target.id === 'text-modal') showModal('text-modal', false);
    });
}

function wireBoards() {
    if (boards.play) {
        boards.play.onPawnMove = function (cell) { humanPawnMove(cell); };
        boards.play.onWallPlace = function (o, r, c) { humanWallMove(o, r, c); };
        boards.play.onWallOrientationChange = function (o) { setWallOrientation(o); };
        boards.play.onHoverWall = function () { /* the renderer draws its own ghost */ };
    }
    if (boards.an) {
        boards.an.onPawnMove = function (cell) { analysisPlay({ isWall: false, a: cell, b: 0, c: 0 }); };
        boards.an.onWallPlace = function (o, r, c) { analysisPlay({ isWall: true, a: o, b: r, c: c }); };
        boards.an.onWallOrientationChange = function () { /* analysis has no armed toggle */ };
    }
    if (boards.ed) {
        boards.ed.onPawnPlace = function (player, cell) {
            if (Q.qr_edit_set_pawn(player, cell) !== 1) { toast('That cell is taken', 'warn'); }
            edRefresh();
        };
        boards.ed.onWallPlace = function (o, r, c) {
            if (Q.qr_edit_toggle_wall(o, r, c) < 0) toast('Wall rejected (overlap or crossing)', 'warn');
            edRefresh();
        };
        boards.ed.onWallRemove = function (o, r, c) { Q.qr_edit_toggle_wall(o, r, c); edRefresh(); };
        boards.ed.onPawnMove = function (cell) {
            if (edTool !== 'p0' && edTool !== 'p1') return;   // wall/erase tools: no-op
            var p = edTool === 'p1' ? 1 : 0;
            Q.qr_edit_set_pawn(p, cell);
            edRefresh();
        };
    }
}

// =====================================================================
// SECTION 21 -- keyboard shortcuts
// =====================================================================

var SHORTCUT_HELP = [
    '← / →   previous / next ply',
    'Home / End   first / last ply',
    'f   flip the board',
    'n   new game',
    'h   hint',
    's   cycle engine strength level',
    'a   toggle the analysis engine',
    '1 / 2 / 3   play / analysis / editor',
    'e   toggle the editor',
    'Esc   close a modal or cancel wall placement',
    '?   show this list'
];

// Cycle the named strength levels (skips 'custom'); wraps around and
// mirrors the change into the select so the UI stays in sync.
function cycleStrength() {
    var cur = parseInt(settings.strength, 10);
    if (!(cur >= 0) || cur >= STRENGTH_LEVELS.length) cur = 0;
    var next = (cur + 1) % STRENGTH_LEVELS.length;
    settings.strength = String(next);
    saveSettings();
    var lvl = $('pl-sel-level');
    if (lvl) lvl.value = settings.strength;
    applySearchModeUi();
    toast('Strength: ' + STRENGTH_LEVELS[next].name + ' · ' + (STRENGTH_LEVELS[next].ms >= 1000
        ? (STRENGTH_LEVELS[next].ms / 1000) + 's' : STRENGTH_LEVELS[next].ms + 'ms'), 'ok');
}

function showShortcuts() {
    var box = $('about-box');
    if (box) {
        var pre = box.querySelector('.shortcut-list');
        if (!pre) {
            pre = document.createElement('pre');
            pre.className = 'shortcut-list';
            box.appendChild(pre);
        }
        pre.textContent = SHORTCUT_HELP.join('\n');
    }
    showModal('about-modal', true);
}

function wireKeyboard() {
    document.addEventListener('keydown', function (e) {
        var tag = e.target && e.target.tagName;
        if (tag === 'INPUT' || tag === 'SELECT' || tag === 'TEXTAREA') {
            if (e.key === 'Escape') e.target.blur();
            return;
        }
        if (e.key === 'Escape') {
            if (anyModalOpen()) { closeAllModals(); e.preventDefault(); return; }
            // Not in a modal: cancel any armed wall placement.
            boardCall(boards.play, 'setWallOrientation', [wallOrientation]);
            hintOverlay = null;
            renderAll();
            return;
        }
        if (anyModalOpen()) return;

        switch (e.key) {
            case 'ArrowLeft': e.preventDefault(); gotoPly(Q.qr_history_cursor() - 1); break;
            case 'ArrowRight': e.preventDefault(); gotoPly(Q.qr_history_cursor() + 1); break;
            case 'Home': e.preventDefault(); gotoPly(0); break;
            case 'End': e.preventDefault(); gotoPly(Q.qr_history_len()); break;
            case 'f': case 'F': e.preventDefault(); doFlip(); break;
            case 'n': case 'N': e.preventDefault(); newGame(); break;
            case 'h': case 'H': e.preventDefault(); doHint(); break;
            case 's': case 'S': e.preventDefault(); cycleStrength(); break;
            case 'a': case 'A':
                e.preventDefault();
                if (activeTab !== 'analysis') switchTab('analysis');
                analysisOn ? stopAnalysis() : startAnalysis();
                break;
            case '1': e.preventDefault(); switchTab('play'); break;
            case '2': e.preventDefault(); switchTab('analysis'); break;
            case '3': e.preventDefault(); switchTab('editor'); break;
            case 'e': case 'E':
                e.preventDefault();
                switchTab(activeTab === 'editor' ? 'play' : 'editor');
                break;
            case '?': e.preventDefault(); showShortcuts(); break;
            default: break;
        }
    });
}

// =====================================================================
// SECTION 22 -- boot
//
// The literal `ZquoridorModule().then((Module) => {` shape below is
// rewritten by build_standalone.py -- do not reformat it.
// =====================================================================

var NNUE_PATH = '/data/nnue/nnue_weights_int8.bin';

function showLoading(show, text) {
    var ov = $('loading-overlay');
    if (text) setText('loading-text', text);
    if (!ov) return;
    ov.style.display = show ? 'flex' : 'none';
    ov.classList.toggle('show', show);
}

function bootUi() {
    loadSettings();
    applySoundVolume();
    applyUiTheme();

    boards.play = makeBoard('pl-board', 'play');
    boards.an = makeBoard('an-board', 'analysis');
    boards.ed = makeBoard('ed-board', 'edit');
    boardsOk = !!(boards.play || boards.an || boards.ed);
    if (!boardsOk) toast('Board renderer (board.js / QBoard) not found -- boards will not draw', 'bad');
    ['play', 'an', 'ed'].forEach(function (k) { applyBoardOptions(boards[k]); });

    wireTabs();
    wirePlay();
    wireAnalysis();
    wireEditor();
    wireSettings();
    wireTextModal();
    wireBoards();
    wireKeyboard();
    syncSettingsUi();
    setWallOrientation(wallOrientation);
    edSetTool(edTool);

    window.addEventListener('resize', resizeAll);
    window.addEventListener('orientationchange', resizeAll);
    if (window.ResizeObserver) {
        try {
            var ro = new ResizeObserver(resizeAll);
            ['pl-board-wrap', 'an-board-col', 'ed-board-col', 'an-graph-wrap'].forEach(function (id) {
                var el = $(id);
                if (el) ro.observe(el);
            });
        } catch (e) { /* older engines: the window resize handler is enough */ }
    }
    // Any pointer interaction is a valid gesture to unlock WebAudio.
    document.addEventListener('pointerdown', audioUnlock, { once: true });
}

function bootEngine(mod) {
    Module = mod;
    var missing = setupWasmBindings(mod);

    // 1. NNUE weights.
    var loaded = has('qr_load_nnue_weights') ? Q.qr_load_nnue_weights(NNUE_PATH) : 0;
    nnueLoaded = loaded === 1;
    if (!nnueLoaded) {
        settings.nnue = false;
        settings.mcab = false;
        ['cfg-nnue', 'cfg-mcab'].forEach(function (id) {
            var el = $(id);
            if (el) {
                el.checked = false;
                el.disabled = true;
                el.setAttribute('title', 'NNUE weights could not be loaded (' + NNUE_PATH + '); the engine is running the heuristic evaluation, and the hybrid MCTS requires NNUE.');
            }
        });
        Q.qr_set_eval_heuristic();
        Q.qr_set_mcab_enabled(0);
    } else {
        applyEngineSettings();
    }
    updateEngineInfo();

    if (missing.length) {
        toast(missing.length + ' engine export(s) missing -- see the console for the list', 'warn');
    }

    newGame();
    switchTab('play');
    resizeAll();
    showLoading(false);

    // Background search: start the engine worker after the main module is
    // up. Purely additive -- every path still works without it.
    ewStart();
}

function bootFailed(err) {
    try { console.error('[zq] WASM boot failed', err); } catch (e) { /* ignore */ }
    showLoading(true, 'Engine failed to load: ' + err);
    toast('Engine failed to load: ' + err, 'bad');
}

function boot() {
    showLoading(true, 'Loading engine…');
    try {
        bootUi();
    } catch (e) {
        bootFailed(e);
        return;
    }
    if (typeof window.ZquoridorModule !== 'function') {
        bootFailed('ZquoridorModule is not defined');
        return;
    }
    ZquoridorModule().then((Module) => {
        try { bootEngine(Module); }
        catch (e) { bootFailed(e); }
    }).catch((err) => {
        bootFailed(err);
    });
}

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', boot);
} else {
    boot();
}
