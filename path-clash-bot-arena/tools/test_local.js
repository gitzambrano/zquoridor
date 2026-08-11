#!/usr/bin/env node
// test_local.js -- roda uma bateria de checagens de sanidade contra
// dist/bot.js: schema exato da spec, os dois player_id, mapeamento de
// paredes, orcamento de tempo, e uma mini-simulacao de varios turnos.
// Uso: node tools/test_local.js  (rode ./build.sh antes)
'use strict';
const path = require('path');

const botPath = path.join(__dirname, '..', 'dist', 'bot.js');
const bot = require(botPath);

let failures = 0;
function check(name, cond) {
    console.log((cond ? 'OK  ' : 'FAIL') + '  ' + name);
    if (!cond) failures++;
}

function baseState(overrides) {
    return Object.assign({
        player_id: 0, num_players: 2, phase: 'turn', actor: 0,
        legal_actions: ['MOVE_UP', 'MOVE_DOWN', 'MOVE_LEFT', 'MOVE_RIGHT'],
        board_size: 9, positions: [[8, 4], [0, 4]], goal_rows: [0, 8],
        walls_remaining: [10, 10], walls: [], history: [], turn: 0,
        decision_timeout: 1.5
    }, overrides || {});
}

// 0) diagnostico: a NNUE embutida realmente carregou (nao caiu pro
// fallback heuristico por dessincronia entre o .bin e o loader manual
// em engine_bridge.cpp -- ver comentario em loadNnueFromMemory)
(function () {
    const diag = bot.qrPcaDiagnostics ? bot.qrPcaDiagnostics() : null;
    check('NNUE embutida carregou (nao caiu pro heuristico)', diag && diag.nnueLoaded === true);
})();

// 1) schema exato do exemplo da spec (Classe A)
(function () {
    const state = {
        player_id: 0, num_players: 2, phase: 'turn', actor: 0,
        legal_actions: ['MOVE_UP', 'WALL_H_3_4'],
        board_size: 9, positions: [[8, 4], [0, 4]], goal_rows: [0, 8],
        walls_remaining: [10, 10], walls: [{ dir: 'H', row: 3, col: 4 }],
        history: [], turn: 0, decision_timeout: 5
    };
    const t0 = Date.now();
    const a = chooseAction(state);
    const elapsed = Date.now() - t0;
    check('exemplo exato da spec -> retorna acao legal', state.legal_actions.includes(a));
    check('exemplo exato da spec -> respeita 5s (< 4900ms)', elapsed < 4900);
})();

// 2) player_id 0 (white/interno1) avanca em direcao ao goal_row correto
(function () {
    const state = baseState({ decision_timeout: 1.2 });
    const a = chooseAction(state);
    check('player_id 0: acao em legal_actions', state.legal_actions.includes(a));
})();

// 3) player_id 1 (black/interno0) avanca em direcao ao goal_row correto
(function () {
    const state = baseState({ player_id: 1, actor: 1, decision_timeout: 1.2 });
    const a = chooseAction(state);
    check('player_id 1: acao em legal_actions', state.legal_actions.includes(a));
})();

// 4) parede em state.walls restringe o motor sem quebrar
(function () {
    const state = baseState({
        positions: [[4, 4], [0, 4]],
        legal_actions: ['MOVE_DOWN', 'MOVE_LEFT', 'MOVE_RIGHT'], // UP ja bloqueado
        walls: [{ dir: 'H', row: 3, col: 4 }],
        decision_timeout: 1.2
    });
    const a = chooseAction(state);
    check('parede bloqueando UP: acao legal retornada', state.legal_actions.includes(a));
})();

// 5) so 1 legal_action -> devolve direto, sem rodar busca (deve ser quase instantaneo)
(function () {
    const state = baseState({ legal_actions: ['MOVE_RIGHT'], decision_timeout: 5 });
    const t0 = Date.now();
    const a = chooseAction(state);
    const elapsed = Date.now() - t0;
    check('acao unica: devolve sem buscar (< 50ms)', elapsed < 50);
    check('acao unica: valor correto', a === 'MOVE_RIGHT');
})();

// 6) config fora do suportado (board_size != 9) nao trava, cai no fallback
(function () {
    const state = baseState({ board_size: 5, decision_timeout: 1 });
    const a = chooseAction(state);
    check('board_size != 9: fallback seguro', state.legal_actions.includes(a));
})();

// 7) mini-simulacao de 8 plies alternando os dois lados, sem paredes
(function () {
    let pos = [[8, 4], [0, 4]];
    const DIR = { UP: [-1, 0], DOWN: [1, 0], LEFT: [0, -1], RIGHT: [0, 1] };
    let ok = true;
    for (let ply = 0; ply < 8 && ok; ply++) {
        const actor = ply % 2;
        const state = baseState({
            player_id: actor, actor: actor, positions: pos, decision_timeout: 1
        });
        const a = chooseAction(state);
        if (!state.legal_actions.includes(a)) { ok = false; break; }
        const d = DIR[a.replace('MOVE_', '')];
        if (!d) { ok = false; break; }
        const cur = pos[actor];
        pos[actor] = [cur[0] + d[0], cur[1] + d[1]];
    }
    check('mini-simulacao 8 plies: todas as acoes legais', ok);
})();

// 8) entrada malformada (fora dos limites) nao trava o processo --
// regressao do bug corrigido: sem clamp, um row/col fora de [0,N-1]
// (peao) ou [0,WS-1] (muro) causava indice fora de faixa em
// cellIdx/slotIdx, podendo corromper memoria ou dar "1ull << 64" (UB)
// no WASM. Um crash aqui seria derrota automatica na arena.
(function () {
    const state = baseState({
        positions: [[99, -5], [0, 4]],
        walls: [{ dir: 'H', row: 50, col: -1 }],
        decision_timeout: 1
    });
    let crashed = false;
    let a = null;
    try { a = chooseAction(state); } catch (e) { crashed = true; }
    check('posicao/muro fora dos limites: nao trava', !crashed);
    check('posicao/muro fora dos limites: acao ainda legal', a && state.legal_actions.includes(a));
})();

console.log('');
if (failures > 0) {
    console.log(failures + ' checagem(ns) falharam.');
    process.exit(1);
} else {
    console.log('Todas as checagens passaram.');
}
