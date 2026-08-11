// qr_pca_wrapper.js -- CONCATENADO ao final de dist/bot.js por
// build.sh/build.bat (nao via --post-js do emcc, ver comentario em
// build.sh sobre por que). Define chooseAction(state), a unica funcao
// exigida pela Classe A do quoridor-arena.
//
// De proposito NAO envolvemos isso num IIFE: `chooseAction` precisa
// existir como uma FUNCTION DECLARATION de topo de arquivo, nao so
// `module.exports.chooseAction` -- nao sabemos se o harness da arena faz
// `require(arquivo).chooseAction` (CommonJS) ou avalia o arquivo inteiro
// num escopo onde so a declaracao de funcao de topo fica visivel (ex.:
// vm.Script/eval nao-estrito). Cobrimos os dois: a funcao propriamente
// dita aqui embaixo + module.exports/globalThis no fim do arquivo.
//
// Mapeamento de coordenadas / jogador (ver README.md secao
// "Mapeamento"):
//   - positions[]/walls[] do state usam (row,col) identicos a convencao
//     interna do motor -- SEM inversao de eixo.
//   - so o INDICE do jogador troca: player_id externo 0 == jogador
//     interno 1; player_id externo 1 == jogador interno 0.
//
// Nomes de lances de peao (MOVE_UP/DOWN/LEFT/RIGHT e variantes de pulo)
// nao sao 100% especificados pela spec do site -- em particular, NAO
// sabemos com certeza se "UP" significa linha decrescente ou crescente
// (o mesmo vale pra LEFT/RIGHT). Em vez de fixar um palpite, AUTO-
// DETECTAMOS a convencao a cada chamada: o motor (WASM) sabe, sem
// ambiguidade nenhuma, quais celulas de destino sao geometricamente
// legais pro peao do ator (qr_pca_legal_pawn_moves/dest); testamos as 4
// combinacoes possiveis de sinal pra UP/DOWN e LEFT/RIGHT e ficamos com
// a que faz o conjunto DECODIFICADO das strings *_UP/_DOWN/_LEFT/_RIGHT*
// de legal_actions bater exatamente com o conjunto geometrico do motor.
// Cada token separado por "_" que bata com um rotulo de direcao vira um
// vetor unitario, somamos os vetores -- cobre "MOVE_UP", "MOVE_UP_UP"
// (pulo reto), "MOVE_UP_LEFT" (pulo diagonal), "JUMP_UP" etc. sem
// precisar adivinhar o vocabulario exato, so a uniao de tokens
// direcionais reconhecidos.

var QR_PCA_DIR_VARIANTS = [
    { UP: [-1, 0], DOWN: [1, 0], LEFT: [0, -1], RIGHT: [0, 1] }, // palpite padrao (row0=topo)
    { UP: [1, 0], DOWN: [-1, 0], LEFT: [0, -1], RIGHT: [0, 1] }, // linha invertida
    { UP: [-1, 0], DOWN: [1, 0], LEFT: [0, 1], RIGHT: [0, -1] }, // coluna invertida
    { UP: [1, 0], DOWN: [-1, 0], LEFT: [0, 1], RIGHT: [0, -1] }  // ambas invertidas
];
var qrPcaWasmModule = null;

function qrPcaDecodeWithVariant(action, curRow, curCol, variant) {
    var toks = action.split('_');
    var dr = 0, dc = 0, any = false;
    for (var i = 0; i < toks.length; i++) {
        var d = variant[toks[i]];
        if (d) { dr += d[0]; dc += d[1]; any = true; }
    }
    if (!any) return null;
    return [curRow + dr, curCol + dc];
}

// Devolve o indice da variante em QR_PCA_DIR_VARIANTS cujo conjunto de
// destinos decodificados bate exatamente com engineDestSet (Set de
// "row,col"). Se nenhuma bater perfeitamente, devolve 0 (palpite
// padrao) -- melhor um palpite razoavel do que travar.
function qrPcaDetectVariant(pawnActions, curRow, curCol, engineDestSet) {
    for (var v = 0; v < QR_PCA_DIR_VARIANTS.length; v++) {
        var variant = QR_PCA_DIR_VARIANTS[v];
        var decodedSet = {};
        var count = 0;
        var ok = true;
        for (var i = 0; i < pawnActions.length; i++) {
            var dest = qrPcaDecodeWithVariant(pawnActions[i], curRow, curCol, variant);
            if (!dest) { ok = false; break; }
            var key = dest[0] + ',' + dest[1];
            if (!engineDestSet.has(key)) { ok = false; break; }
            if (!decodedSet[key]) { decodedSet[key] = true; count++; }
        }
        if (ok && count === engineDestSet.size) return v;
    }
    return 0;
}

function qrPcaEnsureModule() {
    if (!qrPcaWasmModule) {
        qrPcaWasmModule = PathClashBotModule();
        qrPcaWasmModule.ccall('qr_pca_init', null, [], []);
        // diagnostico silencioso: se os pesos NNUE embutidos nao
        // baterem com o layout que engine_bridge.cpp espera (ex.:
        // nnue.hpp mudou e o .h de pesos nao foi regenerado/o loader
        // nao foi atualizado), o motor cai pra avaliacao heuristica
        // sozinho (ainda joga, so mais fraco) -- nao lancamos erro
        // aqui de proposito, pra nao arriscar perder por timeout/crash
        // na arena por causa de um problema so de forca de jogo.
    }
    return qrPcaWasmModule;
}

function qrPcaSafeFallback(state) {
    return (state.legal_actions && state.legal_actions.length) ? state.legal_actions[0] : null;
}

function chooseAction(state) {
    try {
        if (!state || !state.legal_actions || state.legal_actions.length === 0) return null;
        if (state.legal_actions.length === 1) return state.legal_actions[0];
        if (state.board_size !== 9 || state.num_players !== 2) return qrPcaSafeFallback(state);

        var m = qrPcaEnsureModule();
        var actorExt = state.actor;
        var internalActor = 1 - actorExt;

        var posExt0 = state.positions[0];
        var posExt1 = state.positions[1];
        // jogador interno 0 == player_id externo 1 ; interno 1 == externo 0
        var p0r = posExt1[0], p0c = posExt1[1];
        var p1r = posExt0[0], p1c = posExt0[1];
        var wallsLeft0 = state.walls_remaining[1];
        var wallsLeft1 = state.walls_remaining[0];

        m.ccall('qr_pca_reset_turn', null,
            ['number', 'number', 'number', 'number', 'number', 'number', 'number'],
            [internalActor, p0r, p0c, p1r, p1c, wallsLeft0, wallsLeft1]);

        var walls = state.walls || [];
        for (var i = 0; i < walls.length; i++) {
            var w = walls[i];
            var orientation = (w.dir === 'H' || w.dir === 'h') ? 0 : 1;
            m.ccall('qr_pca_add_wall', null, ['number', 'number', 'number'],
                [orientation, w.row, w.col]);
        }

        var timeoutSec = state.decision_timeout || 5;
        var budgetMs = Math.max(300, Math.floor(timeoutSec * 1000) - 700);

        var encoded = m.ccall('qr_pca_choose', 'number', ['number'], [budgetMs]);

        var legal = state.legal_actions;
        if (encoded >= 1000) {
            var v = encoded - 1000;
            var orientation2 = Math.floor(v / 100);
            var rem = v % 100;
            var row = Math.floor(rem / 10);
            var col = rem % 10;
            var candidate = 'WALL_' + (orientation2 === 0 ? 'H' : 'V') + '_' + row + '_' + col;
            for (var j = 0; j < legal.length; j++) if (legal[j] === candidate) return candidate;
            for (var j2 = 0; j2 < legal.length; j2++) if (legal[j2].indexOf('WALL_') === 0) return legal[j2];
            return qrPcaSafeFallback(state);
        } else {
            var destRow = Math.floor(encoded / 9);
            var destCol = encoded % 9;
            var curRow = state.positions[actorExt][0];
            var curCol = state.positions[actorExt][1];

            var pawnActions = [];
            for (var p = 0; p < legal.length; p++) {
                if (legal[p].indexOf('WALL_') !== 0) pawnActions.push(legal[p]);
            }

            // conjunto geometrico de destinos legais, direto do motor
            // (sem ambiguidade de rotulo UP/DOWN/LEFT/RIGHT)
            var pawnCount = m.ccall('qr_pca_legal_pawn_moves', 'number', [], []);
            var engineDestSet = new Set();
            for (var q = 0; q < pawnCount; q++) {
                var cell = m.ccall('qr_pca_legal_pawn_dest', 'number', ['number'], [q]);
                engineDestSet.add(Math.floor(cell / 9) + ',' + (cell % 9));
            }

            var variantIdx = qrPcaDetectVariant(pawnActions, curRow, curCol, engineDestSet);
            var variant = QR_PCA_DIR_VARIANTS[variantIdx];

            var best = null, bestDist = Infinity;
            for (var k = 0; k < pawnActions.length; k++) {
                var a = pawnActions[k];
                var dest = qrPcaDecodeWithVariant(a, curRow, curCol, variant);
                if (!dest) continue;
                if (dest[0] === destRow && dest[1] === destCol) return a;
                var dist = Math.abs(dest[0] - destRow) + Math.abs(dest[1] - destCol);
                if (dist < bestDist) { bestDist = dist; best = a; }
            }
            if (best) return best;
            return qrPcaSafeFallback(state);
        }
    } catch (e) {
        return qrPcaSafeFallback(state);
    }
}

if (typeof module !== 'undefined' && module.exports) {
    module.exports.chooseAction = chooseAction;
    // Diagnostico pra testes (tools/test_local.js) -- nao faz parte do
    // contrato exigido pela arena, so ajuda a pegar automaticamente uma
    // dessincronia entre o .bin de pesos e o loader em engine_bridge.cpp
    // (ex.: nnue.hpp mudou de layout e o bot silenciosamente caiu pra
    // avaliacao heuristica em vez de usar a NNUE).
    module.exports.qrPcaDiagnostics = function () {
        var m = qrPcaEnsureModule();
        return { nnueLoaded: m.ccall('qr_pca_nnue_loaded', 'number', [], []) === 1 };
    };
}
if (typeof globalThis !== 'undefined') {
    globalThis.chooseAction = chooseAction;
}
