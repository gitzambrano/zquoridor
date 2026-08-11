// engine_wasm.cpp -- casca fina em C exportando o core (rules.hpp +
// search.hpp) pro WASM, no mesmo espírito do binding usado no Zchezz:
// funções extern "C" simples com tipos primitivos, sem embind (mais leve
// pra compilar e pra carregar no navegador; JS só chama via ccall/cwrap).
//
// Mantém UM jogo global no módulo (não é reentrante/thread-safe, mas a
// GUI é single-thread e chama tudo sequencialmente). Todo estado
// intermediário (lista de lances legais, último lance do motor) fica em
// variáveis estáticas consultadas pelos getters abaixo.
#include <vector>
#include <cstdint>
#include "../src/rules.hpp"
#include "../src/search.hpp"
#include "../src/nnue.hpp"

#ifdef __EMSCRIPTEN__
#include <emscripten/emscripten.h>
#else
#define EMSCRIPTEN_KEEPALIVE
#endif

using namespace qr;

namespace {
State g_state;
Negamax g_engine;   // reaproveita a TT entre lances do motor (padrão comum em GUIs de xadrez/damas)
std::vector<Move> g_moves;
Move g_lastEngineMove = Move::pawn(0);
int g_lastEngineScore = 0;   // avaliação (SearchStats::score) do último qr_engine_move
RepetitionTable g_reptbl;

void regenMoves() { g_moves = legalMoves(g_state).toVector(); }
} // namespace

extern "C" {

EMSCRIPTEN_KEEPALIVE
void qr_new_game() {
    g_state = initialState();
    // Reap roveit a o modo de avaliação configurado (NNUE ou heurístico)
    // entre partidas -- não reseta o evalMode ao recriar a engine.
    Negamax::EvalMode prevMode = g_engine.getEvalMode();
    // Idem pra ordenação assistida por política -- reaproveitada entre
    // partidas junto com o evalMode acima (ver qr_load_nnue_weights: liga
    // por default quando NNUE é carregado; sem isso, Negamax() zerava de
    // volta pro default da classe (desligada) a cada partida nova).
    bool prevPolicyOrdering = g_engine.isPolicyOrderingEnabled();
    int prevPolicyOrderingMinDepth = g_engine.getPolicyOrderingMinDepth();
    g_engine = Negamax();  // zera a TT: partida nova não deve herdar lixo da anterior
    g_engine.setEvalMode(prevMode);
    g_engine.setPolicyOrderingEnabled(prevPolicyOrdering);
    g_engine.setPolicyOrderingMinDepth(prevPolicyOrderingMinDepth);
    g_reptbl.clear();
    g_reptbl.push(g_state.hash);
    regenMoves();
}

EMSCRIPTEN_KEEPALIVE int qr_turn() { return g_state.turn; }
EMSCRIPTEN_KEEPALIVE int qr_winner() { return winner(g_state); }
EMSCRIPTEN_KEEPALIVE int qr_pawn(int player) { return g_state.pawn[player]; }
EMSCRIPTEN_KEEPALIVE int qr_walls_left(int player) { return g_state.wallsLeft[player]; }
EMSCRIPTEN_KEEPALIVE int qr_wall_h_bit(int slot) { return (int)((g_state.wallsH >> slot) & 1ull); }
EMSCRIPTEN_KEEPALIVE int qr_wall_v_bit(int slot) { return (int)((g_state.wallsV >> slot) & 1ull); }

// distância BFS até a meta -- usada só pra exibir um indicador na GUI, não afeta a busca
EMSCRIPTEN_KEEPALIVE int qr_dist_to_goal(int player) {
    return shortestPathLen(g_state.wallsH, g_state.wallsV, g_state.pawn[player], player);
}

EMSCRIPTEN_KEEPALIVE int qr_legal_moves_count() { return (int)g_moves.size(); }
EMSCRIPTEN_KEEPALIVE int qr_legal_move_is_wall(int i) { return g_moves[(size_t)i].isWall ? 1 : 0; }
EMSCRIPTEN_KEEPALIVE int qr_legal_move_a(int i) { return g_moves[(size_t)i].a; }
EMSCRIPTEN_KEEPALIVE int qr_legal_move_b(int i) { return g_moves[(size_t)i].b; }
EMSCRIPTEN_KEEPALIVE int qr_legal_move_c(int i) { return g_moves[(size_t)i].c; }

// devolve 1 em sucesso, 0 se o lance não está na lista de legais (JS deve
// sempre ter oferecido só lances vindos de qr_legal_move_*, então um 0
// aqui indica um bug no front-end, não uma jogada "quase legal")
EMSCRIPTEN_KEEPALIVE
int qr_apply_pawn_move(int destCell) {
    for (size_t i = 0; i < g_moves.size(); i++) {
        if (!g_moves[i].isWall && g_moves[i].a == destCell) {
            g_state = applyMove(g_state, g_moves[i]);
            g_reptbl.push(g_state.hash);
            regenMoves();
            return 1;
        }
    }
    return 0;
}

EMSCRIPTEN_KEEPALIVE
int qr_apply_wall_move(int orientation, int r, int c) {
    for (size_t i = 0; i < g_moves.size(); i++) {
        const Move& m = g_moves[i];
        if (m.isWall && m.a == orientation && m.b == r && m.c == c) {
            g_state = applyMove(g_state, m);
            g_reptbl.push(g_state.hash);
            regenMoves();
            return 1;
        }
    }
    return 0;
}

// roda a busca e já aplica o lance escolhido -- síncrono (bloqueia a
// thread principal do navegador pela duração de timeMs); aceitável pro
// orçamento de tempo por lance usado aqui (centenas de ms). Se precisar de
// UI responsiva durante o pensamento do motor, mover pra um Web Worker é o
// próximo passo natural (módulo já compila igual dentro de um worker).
EMSCRIPTEN_KEEPALIVE
int qr_engine_move(int maxDepth, int timeMs) {
    if (winner(g_state) != -1 || g_reptbl.count(g_state.hash) >= 3) return 0;
    SearchStats st;
    Move m = g_engine.chooseMove(g_state, maxDepth, timeMs, st, g_reptbl);
    g_lastEngineMove = m;
    g_lastEngineScore = st.score;
    g_state = applyMove(g_state, m);
    g_reptbl.push(g_state.hash);
    regenMoves();
    return 1;
}

EMSCRIPTEN_KEEPALIVE int qr_last_move_is_wall() { return g_lastEngineMove.isWall ? 1 : 0; }
EMSCRIPTEN_KEEPALIVE int qr_last_move_a() { return g_lastEngineMove.a; }
EMSCRIPTEN_KEEPALIVE int qr_last_move_b() { return g_lastEngineMove.b; }
EMSCRIPTEN_KEEPALIVE int qr_last_move_c() { return g_lastEngineMove.c; }

// avaliação (unidades de evalSimple) do lance acima, do ponto de vista de
// quem jogou -- positivo é bom pro motor, negativo é ruim. Só reflete a
// profundidade que a busca alcançou dentro do orçamento de tempo/lance.
EMSCRIPTEN_KEEPALIVE int qr_last_move_eval() { return g_lastEngineScore; }

// Carrega pesos NNUE quantizados de `path` e ativa NNUE no engine WASM.
// Retorna 1 em sucesso, 0 se o arquivo não pôde ser aberto ou está corrompido.
// Para REVERTER ao heurístico: chame qr_set_eval_heuristic() ou simplesmente
// não chame esta função -- o engine nasce em modo heurístico por default.
//
// Também liga a ordenação assistida por política (Negamax::setPolicyOrderingEnabled,
// search.hpp) por default aqui -- medido mais forte em teste (tools/arena/run_arena.py)
// que o move ordering puramente heurístico/CAT, então o engine "de
// produção" (este binário WASM, o único chooseMove() fora de benchmark/
// selfplay) passa a jogar com ela ativa sempre que NNUE está carregado.
// min-depth fica no default da classe (3, ver comentário em search.hpp).
// Chamada explícita aqui hoje é redundante com o default de search.hpp
// (2026-08: policyOrderingEnabled passou a nascer true na classe, e
// selfplay.hpp/tools/arena/arena.cpp seguiram o mesmo default) -- mantida
// mesmo assim, por clareza e para não depender silenciosamente do
// default da classe caso ele mude de novo no futuro.
EMSCRIPTEN_KEEPALIVE
int qr_load_nnue_weights(const char* path) {
    if (!qr::loadWeightsQuant(path)) return 0;
    g_engine.setEvalMode(qr::Negamax::EvalMode::NNUE);
    g_engine.setPolicyOrderingEnabled(true);
    return 1;
}

// Volta o engine ao modo heurístico (evalSimple), sem descarregar os pesos
// da memória -- permite alternar sem precisar recarregar o arquivo.
EMSCRIPTEN_KEEPALIVE
void qr_set_eval_heuristic() {
    g_engine.setEvalMode(qr::Negamax::EvalMode::Heuristic);
}

// Retorna 1 se o engine está em modo NNUE, 0 se heurístico.
EMSCRIPTEN_KEEPALIVE
int qr_eval_mode_is_nnue() {
    return g_engine.getEvalMode() == qr::Negamax::EvalMode::NNUE ? 1 : 0;
}

EMSCRIPTEN_KEEPALIVE
int qr_is_draw() {
    return g_reptbl.count(g_state.hash) >= 3 ? 1 : 0;
}

} // extern "C"
