// engine_bridge.cpp -- ponte WASM sincrona do zquoridor para o
// "path-clash" (Classe A, quoridor-arena). NAO reaproveita nem modifica
// nada de gui_web/ (aquele build continua intacto para o jogo normal);
// este e' um binario WASM separado, com uma interface minima pensada
// para ser chamada de dentro de uma unica funcao JS sincrona
// (chooseAction(state)), sem stdin/stdout/rede/disco em runtime.
//
// Mapeamento de coordenadas (deduzido da spec do site, ver README.md):
//   - positions[]/walls[] do state da arena usam (row,col) 0-indexados
//     que batem 1:1 com a convencao interna do motor (cellIdx/slotIdx),
//     SEM inversao de eixo.
//   - so o INDICE do jogador e' invertido: player_id externo 0 == nosso
//     jogador interno 1; player_id externo 1 == nosso jogador interno 0
//     (deduzido cruzando positions/goal_rows do exemplo da spec com
//     GOAL_ROW[] em rules.hpp -- ver README.md secao "Mapeamento").
//
// Os pesos NNUE quantizados vao embutidos em nnue_weights_data.h
// (gerado por tools/gen_weights_header.py), carregados da MEMORIA em vez
// de arquivo -- o container da arena e' read-only e sem rede, entao nao
// da pra abrir data/nnue/*.bin em runtime.
#include <cstdint>
#include <cstring>
#include <emscripten.h>
#include "../../src/rules.hpp"
#include "../../src/search.hpp"
#include "../../src/nnue.hpp"
#include "nnue_weights_data.h"

using namespace qr;

namespace {

// Espelha campo-a-campo NNUEWeightsQuant::loadFromFile (nnue.hpp), mas
// lendo de um buffer em memoria em vez de FILE*. A ORDEM dos campos tem
// que bater exatamente com loadFromFile -- ver comentario la se o layout
// do .bin mudar (ex.: retrain com NUM_FEATURES diferente).
//
// ATUALIZADO 2026-08 pra acompanhar a remocao da cabeca AUXILIAR de
// imitacao da heuristica (wv1_aux/bv1_aux/wv2_aux/bv2_aux) em nnue.hpp:
// o .bin quantizado NAO tem mais esse bloco no meio do arquivo. Se
// nnue.hpp mudar de novo (outra cabeca adicionada/removida, mudanca de
// NUM_FEATURES/HIDDEN/POLICY_OUT), este loader precisa ser atualizado
// junto -- ver README.md secao "Mantendo isso sincronizado com o motor".
bool loadNnueFromMemory(const unsigned char* data, size_t len) {
    NNUEWeightsQuant& w = weightsQuant();
    size_t off = 0;
    auto rd = [&](void* dst, size_t bytes) -> bool {
        if (off + bytes > len) return false;
        std::memcpy(dst, data + off, bytes);
        off += bytes;
        return true;
    };
    bool ok = true;
    ok = ok && rd(&w.QA, sizeof(int32_t));
    ok = ok && rd(&w.QB, sizeof(int32_t));

    w.w1.assign(NUM_FEATURES, {});
    for (auto& row : w.w1) ok = ok && rd(row.data(), sizeof(int16_t) * HIDDEN);
    ok = ok && rd(w.b1.data(), sizeof(int16_t) * HIDDEN);

    for (auto& row : w.wv1_wl) ok = ok && rd(row.data(), 32);
    ok = ok && rd(w.bv1_wl.data(), sizeof(int32_t) * 32);
    ok = ok && rd(w.wv2_wl.data(), 32);
    ok = ok && rd(&w.bv2_wl, sizeof(int32_t));

    // (cabeca auxiliar removida 2026-08 -- nada a ler aqui)

    w.wp.assign(POLICY_OUT, {});
    for (auto& row : w.wp) ok = ok && rd(row.data(), sizeof(int8_t) * HIDDEN);
    w.bp.assign(POLICY_OUT, 0);
    ok = ok && rd(w.bp.data(), sizeof(int32_t) * POLICY_OUT);

    // Se sobrou ou faltou byte, o layout do .bin nao bate mais com esta
    // funcao (ex.: NUM_FEATURES/HIDDEN mudou, ou uma cabeca foi
    // adicionada/removida de novo em nnue.hpp) -- silenciosamente
    // continuar leria pesos deslocados/lixo em vez de falhar alto.
    if (ok && off != len) ok = false;

    w.loaded = ok;
    return ok;
}

Negamax g_engine;
State g_state = initialState();
bool g_nnueLoadOk = false;

// Recalcula o hash Zobrist do zero a partir do estado corrente -- mesma
// formula usada por initialState()/applyMove() (pawnKey ^ pawnKey ^
// wallHKey*... ^ wallVKey*... ^ (turno==1 ? turnKey : 0)), entao o
// resultado e' identico ao que uma sequencia real de lances teria
// produzido, mesmo reconstruindo o tabuleiro direto do snapshot da
// arena (sem repetir o historico de jogadas).
void recomputeHash(State& s) {
    Zobrist& z = zobrist();
    uint64_t h = z.pawnKey[0][s.pawn[0]] ^ z.pawnKey[1][s.pawn[1]];
    for (int slot = 0; slot < WS * WS; slot++) {
        if ((s.wallsH >> slot) & 1ull) h ^= z.wallHKey[slot];
        if ((s.wallsV >> slot) & 1ull) h ^= z.wallVKey[slot];
    }
    if (s.turn == 1) h ^= z.turnKey;
    s.hash = h;
}

} // namespace

extern "C" {

EMSCRIPTEN_KEEPALIVE
void qr_pca_init() {
    bool ok = loadNnueFromMemory(kNnueWeightsData, kNnueWeightsLen);
    g_engine.setEvalMode(ok ? Negamax::EvalMode::NNUE : Negamax::EvalMode::Heuristic);
    g_nnueLoadOk = ok;
}

// Diagnostico: 1 se os pesos NNUE embutidos carregaram e bateram
// exatamente com o tamanho esperado pelo layout atual de nnue.hpp, 0 se
// caiu no fallback heuristico (ver loadNnueFromMemory -- geralmente
// sinal de que o .bin foi trocado sem atualizar este arquivo, ou
// vice-versa). tools/test_local.js confere isso.
EMSCRIPTEN_KEEPALIVE
int qr_pca_nnue_loaded() { return g_nnueLoadOk ? 1 : 0; }

// Comeca um novo turno: zera o tabuleiro (sem muros) e posiciona os dois
// peoes + muros restantes. Chamar isso, depois qr_pca_add_wall() para
// cada muro do state.walls[], depois qr_pca_choose().
//
// Faz clamp defensivo de tudo que vem de fora (JSON da arena): um
// row/col fora de [0,N-1] chegando em cellIdx() geraria um indice fora
// da faixa de pawn[]/pawnKey[][], corrompendo memoria no WASM -- pior
// que um lance ruim, um crash e' derrota automatica na arena.
EMSCRIPTEN_KEEPALIVE
void qr_pca_reset_turn(int actorInternal, int p0Row, int p0Col, int p1Row, int p1Col,
                        int wallsLeft0, int wallsLeft1) {
    auto clampCoord = [](int v) { return v < 0 ? 0 : (v >= N ? N - 1 : v); };
    auto clampWalls = [](int v) { return v < 0 ? 0 : (v > WALLS_PER_PLAYER ? WALLS_PER_PLAYER : v); };
    g_state = State{};
    g_state.pawn[0] = cellIdx(clampCoord(p0Row), clampCoord(p0Col));
    g_state.pawn[1] = cellIdx(clampCoord(p1Row), clampCoord(p1Col));
    g_state.wallsH = 0;
    g_state.wallsV = 0;
    g_state.wallsLeft[0] = (int8_t)clampWalls(wallsLeft0);
    g_state.wallsLeft[1] = (int8_t)clampWalls(wallsLeft1);
    g_state.turn = (int8_t)(actorInternal == 1 ? 1 : 0);
}

EMSCRIPTEN_KEEPALIVE
void qr_pca_add_wall(int orientation, int row, int col) {
    if (row < 0 || row >= WS || col < 0 || col >= WS) return; // fora da faixa -- ignora em vez de UB
    int slot = slotIdx(row, col);
    if (orientation == 0) g_state.wallsH |= (1ull << slot);
    else g_state.wallsV |= (1ull << slot);
}

// Enumera os destinos de PEAO legais na posicao/turno correntes (a
// mesma reconstruida por qr_pca_reset_turn+qr_pca_add_wall). Existe pra
// que o lado JS possa AUTO-DETECTAR a convencao UP/DOWN/LEFT/RIGHT da
// arena comparando este conjunto (fonte da verdade, geometrico, sem
// ambiguidade de rotulo) contra o que cada candidato de nomenclatura
// prediz para as strings em legal_actions -- ver qr_pca_wrapper.js.
// Nao precisamos disso pra escolher o lance em si (isso e' o que
// qr_pca_choose ja faz via busca completa), so pra mapear de volta pro
// vocabulario textual da arena com seguranca.
namespace { int g_legalPawnDest[16]; int g_legalPawnCount = 0; }

EMSCRIPTEN_KEEPALIVE
int qr_pca_legal_pawn_moves() {
    MoveList moves = legalMoves(g_state);
    g_legalPawnCount = 0;
    for (size_t i = 0; i < moves.size() && g_legalPawnCount < 16; i++) {
        if (!moves[i].isWall) g_legalPawnDest[g_legalPawnCount++] = moves[i].a;
    }
    return g_legalPawnCount;
}

EMSCRIPTEN_KEEPALIVE
int qr_pca_legal_pawn_dest(int idx) {
    if (idx < 0 || idx >= g_legalPawnCount) return -1;
    return g_legalPawnDest[idx];
}

// Roda a busca e devolve o lance escolhido como um unico inteiro:
//   - lance de peao: valor = destCell (0..80)
//   - lance de muro: valor = 1000 + orientation*100 + row*10 + col
//     (orientation 0=H,1=V; row,col 0..7 -- cabe em row*10+col <= 77)
EMSCRIPTEN_KEEPALIVE
int qr_pca_choose(int timeBudgetMs) {
    recomputeHash(g_state);
    SearchStats stats;
    int cap = 40;
    Move m = g_engine.chooseMove(g_state, cap, timeBudgetMs, stats);
    if (m.isWall) {
        return 1000 + m.a * 100 + m.b * 10 + m.c;
    }
    return m.a;
}

} // extern "C"
