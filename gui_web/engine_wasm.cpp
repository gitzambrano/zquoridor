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
#include <cstring>
#include <sstream>
#include <algorithm>
#include "../src/rules.hpp"
#include "../src/search.hpp"
#include "../src/nnue.hpp"
#include "../src/mcab.hpp"

#ifdef __EMSCRIPTEN__
#include <emscripten/emscripten.h>
#else
#define EMSCRIPTEN_KEEPALIVE
#endif

using namespace qr;

namespace {
State g_state;
Negamax g_engine;   // reaproveita a TT entre lances do motor (padrão comum em GUIs de xadrez/damas)

// MCTS híbrido com alpha-beta -- a busca de produção (src/mcab.hpp).
// Uma instância viva pela partida inteira: o reuso de subárvore entre
// lances depende de o mesmo runner ver todos os lances consecutivos.
using McabRunnerT = mcab::McabRunner<qr::Negamax, qr::State, qr::Move, qr::MoveList,
                                      qr::AccPair, qr::RepetitionTable, qr::SearchStats>;
McabRunnerT g_mcab;
// Espelha McabParams::enabled, mas só entra em vigor quando a NNUE está
// carregada -- os priors do PUCT vêm da cabeça de política, então em modo
// heurístico o híbrido não tem como funcionar e cai para AB puro.
bool g_mcabWanted = mcab::McabParams{}.enabled;

// RESSALVA DE FAIXA (importante justamente aqui): os +46.9 ±23.5 Elo do
// híbrido foram medidos a 200ms/lance. Ele roda a ~1/9 dos nós/s do
// alpha-beta puro, e a GUI costuma pedir buscas bem mais curtas que isso
// -- abaixo de algum ponto (não medido) a troca inverte e o AB puro volta
// a ser mais forte. Por isso `qr_set_mcab_enabled` existe e é exposto ao
// JS: dá para desligar por partida sem recompilar. Ver status.md.
std::vector<Move> g_moves;
Move g_lastEngineMove = Move::pawn(0);
int g_lastEngineScore = 0;   // avaliação (SearchStats::score) do último qr_engine_move
RepetitionTable g_reptbl;

// ---------------------------------------------------------------------------
// P6 (gui-premium.md): full game history plus a navigation cursor.
//
// Invariant: g_histStates has one entry more than g_histMoves, entry k of
// g_histStates is the position after k plies, and g_state always equals
// g_histStates[g_cursor]. The live game plays at cursor == end; the GUI can
// move the cursor back for review and forward again without copying anything
// through the JS side.
// ---------------------------------------------------------------------------
std::vector<State> g_histStates;
std::vector<Move> g_histMoves;
int g_cursor = 0;

// Scratch position shared by the analysis surface, the blunder check and the
// editor. It never touches g_state, so analysis can never mutate the live
// game (plan section 5.6).
State g_scratch;
std::vector<Move> g_scratchMoves;

// Dedicated search instance for analysis. A separate transposition table
// keeps the live game's TT warm while the user analyses other positions.
Negamax g_anEngine;

struct AnLine {
    int score = 0;              // mover-relative, side to move of the scratch
    std::vector<Move> pv;       // pv[0] is the root move itself
};
std::vector<AnLine> g_anLines;
long long g_anNodes = 0;
int g_anDepth = 0;

char g_lastErr[160] = {0};

void regenMoves() { g_moves = legalMoves(g_state).toVector(); }
void scratchRegen() { g_scratchMoves = legalMoves(g_scratch).toVector(); }

// Common tail of every live-game mutation: record the move, advance the
// cursor, refresh the repetition table and regenerate the legal move list.
// Playing while the cursor reviews an earlier ply truncates the future first,
// so the invariant above always holds.
void recordLiveMove(const Move& m) {
    if (g_cursor < (int)g_histMoves.size()) {
        g_histMoves.resize((size_t)g_cursor);
        g_histStates.resize((size_t)g_cursor + 1);
    }
    g_histMoves.push_back(m);
    g_histStates.push_back(g_state);
    g_cursor = (int)g_histStates.size() - 1;
    g_reptbl.push(g_state.hash);
    regenMoves();
}

// Starts a fresh game from an arbitrary (already validated) position: clears
// the TT, the repetition history, the MCTS tree and the move history.
void startFromState(const State& s) {
    g_state = s;
    // Reuse the configured eval mode and policy-ordering switches across
    // games -- see the comment in the original qr_new_game.
    Negamax::EvalMode prevMode = g_engine.getEvalMode();
    bool prevPolicyOrdering = g_engine.isPolicyOrderingEnabled();
    int prevPolicyOrderingMinDepth = g_engine.getPolicyOrderingMinDepth();
    g_engine = Negamax();  // zero the TT: a new game inherits nothing
    g_engine.setEvalMode(prevMode);
    g_engine.setPolicyOrderingEnabled(prevPolicyOrdering);
    g_engine.setPolicyOrderingMinDepth(prevPolicyOrderingMinDepth);
    g_reptbl.clear();
    g_reptbl.push(g_state.hash);
    g_mcab.resetTree();
    g_histStates.clear();
    g_histStates.push_back(g_state);
    g_histMoves.clear();
    g_cursor = 0;
    regenMoves();
    // The scratch starts from the same position (analysis root / editor
    // seed) so it is never left in a half-initialized zero state.
    g_scratch = g_state;
    scratchRegen();
}

// Packs a move into one int for the JS boundary:
// bit 24 = isWall, bits 16..23 = a, bits 8..15 = b, bits 0..7 = c.
int packMove(const Move& m) {
    return (m.isWall ? (1 << 24) : 0) | ((int)m.a << 16) | ((int)m.b << 8) | (int)m.c;
}

// Applies a legal move to the scratch position (legality against the current
// scratch move list). Used by PV preview and by the blunder check replay.
int scratchApply(const Move& m) {
    for (size_t i = 0; i < g_scratchMoves.size(); i++) {
        if (g_scratchMoves[i] == m) {
            g_scratch = applyMove(g_scratch, m);
            scratchRegen();
            return 1;
        }
    }
    return 0;
}

// ---------------------------------------------------------------------------
// QFEN serialization (plan section 16.1).
//
// Format: <pawn0> <pawn1> <walls0> <walls1> <wallList> <turn>[ <plyCount>].
// A wall token is <file><rank><h|v>, anchored at the south-west cell under
// the wall: slot (r,c) exports as file(c) + rank(r+1) + orientation. The
// parser validates before applying and reports the exact failing token, so a
// bad QFEN never corrupts the target position.
// ---------------------------------------------------------------------------

const char QF_FILES[N] = {'a','b','c','d','e','f','g','h','i'};

void qerr(const std::string& msg) {
    std::strncpy(g_lastErr, msg.c_str(), sizeof(g_lastErr) - 1);
    g_lastErr[sizeof(g_lastErr) - 1] = '\0';
}

bool algToCell(const std::string& t, int& cell, std::string& why) {
    if (t.size() != 2) { why = "a pawn cell is two characters (file a-i, rank 1-9)"; return false; }
    int f = t[0] - 'a', r = t[1] - '1';
    if (f < 0 || f >= N || r < 0 || r >= N) { why = "cell outside a-i / 1-9"; return false; }
    cell = r * N + f;
    return true;
}

bool algToWall(const std::string& t, int& o, int& r, int& c, std::string& why) {
    if (t.size() != 3) { why = "a wall token is three characters (file, rank, h/v)"; return false; }
    int f = t[0] - 'a', d = t[1] - '1';
    char orient = (char)tolower((unsigned char)t[2]);
    if (f < 0 || f >= N || d < 0 || d >= WS) { why = "wall anchor outside the board"; return false; }
    if (orient != 'h' && orient != 'v') { why = "wall orientation must be h or v"; return false; }
    o = (orient == 'h') ? 0 : 1;
    r = d; c = f;
    return true;
}

std::string writeQfen(const State& s, int plyCount) {
    std::string out;
    auto pushPawn = [&](int p) {
        out += QF_FILES[s.pawn[p] % N];
        out += (char)('1' + s.pawn[p] / N);
    };
    pushPawn(0); out += ' ';
    pushPawn(1); out += ' ';
    out += std::to_string((int)s.wallsLeft[0]); out += ' ';
    out += std::to_string((int)s.wallsLeft[1]); out += ' ';
    std::vector<std::string> toks;
    for (int o = 0; o < 2; o++) {
        uint64_t bb = (o == 0) ? s.wallsH : s.wallsV;
        for (int sl = 0; sl < WS * WS; sl++) {
            if ((bb >> sl) & 1ull) {
                std::string t;
                // Slot row r exports as rank r+1 (south-west cell under the
                // wall); slot column c is the file.
                t += QF_FILES[sl % WS];
                t += (char)('1' + sl / WS);
                t += (o == 0 ? 'h' : 'v');
                toks.push_back(t);
            }
        }
    }
    if (toks.empty()) out += '-';
    else for (size_t i = 0; i < toks.size(); i++) { if (i) out += ' '; out += toks[i]; }
    out += s.turn == 0 ? " 0" : " 1";
    if (plyCount > 0) { out += ' '; out += std::to_string(plyCount); }
    return out;
}

// Loads a QFEN string into the scratch position. Returns 0 on success,
// 1 on a parse error and 2 on a validation error; g_lastErr holds a human
// readable reason in both failure cases.
int qfenImportScratch(const std::string& qfenRaw) {
    g_lastErr[0] = '\0';
    std::vector<std::string> tok;
    std::istringstream in(qfenRaw);
    std::string w;
    while (in >> w) tok.push_back(w);
    if (tok.size() < 6) { qerr("a QFEN has 6 or 7 space-separated fields"); return 1; }

    State s;
    std::string why;
    int c0 = 0, c1 = 0;
    if (!algToCell(tok[0], c0, why)) { qerr("field 1 ('" + tok[0] + "'): " + why); return 1; }
    if (!algToCell(tok[1], c1, why)) { qerr("field 2 ('" + tok[1] + "'): " + why); return 1; }
    if (c0 == c1) { qerr("both pawns stand on " + tok[0]); return 2; }
    s.pawn[0] = (uint8_t)c0;
    s.pawn[1] = (uint8_t)c1;

    for (int f = 0; f < 2; f++) {
        const std::string& t = tok[2 + f];
        long n = std::strtol(t.c_str(), nullptr, 10);
        if (n < 0 || n > WALLS_PER_PLAYER || t.find_first_not_of("0123456789") != std::string::npos) {
            qerr("field " + std::to_string(3 + f) + " ('" + t + "'): wall count must be 0-10");
            return 1;
        }
        s.wallsLeft[f] = (int8_t)n;
    }

    // Wall list: '-' when empty, otherwise wall tokens separated by spaces,
    // slashes or nothing at all (tokens have a fixed width of 3). The turn
    // field is the lone character '0' or '1'.
    struct ParsedWall { int o, r, c; std::string tok; };
    std::vector<ParsedWall> walls;
    int turnFieldIdx = -1;
    for (size_t i = 4; i < tok.size(); i++) {
        const std::string& t = tok[i];
        if (t == "-") continue;
        if (t.size() == 1 && (t[0] == '0' || t[0] == '1')) { turnFieldIdx = (int)i; break; }
        // strip slashes, then chunk every 3 characters
        std::string clean;
        for (char ch : t) if (ch != '/' && ch != ',') clean += ch;
        if (clean.empty()) continue;
        if (clean.size() % 3 != 0) { qerr("field " + std::to_string(i + 1) + " ('" + t + "'): not a wall token"); return 1; }
        for (size_t k = 0; k < clean.size(); k += 3) {
            int o, r, c;
            std::string single = clean.substr(k, 3);
            if (!algToWall(single, o, r, c, why)) {
                qerr("field " + std::to_string(i + 1) + " ('" + single + "'): " + why);
                return 1;
            }
            walls.push_back({o, r, c, single});
        }
    }
    if (turnFieldIdx < 0) { qerr("the turn field ('0' or '1') is missing"); return 1; }
    s.turn = tok[(size_t)turnFieldIdx][0] - '0';

    // Optional ply count after the turn: accepted and ignored here (the GUI
    // tracks the ply cursor itself).
    if (tok.size() > (size_t)turnFieldIdx + 1) {
        const std::string& t = tok[(size_t)turnFieldIdx + 1];
        if (t.find_first_not_of("0123456789") != std::string::npos) {
            qerr("field " + std::to_string(turnFieldIdx + 2) + " ('" + t + "'): ply count must be a number");
            return 1;
        }
    }

    // Validate before applying (plan 16.1): place each wall through the
    // engine's own availability check, then budgets and paths.
    for (const ParsedWall& pw : walls) {
        if (!wallSlotAvailable(s.wallsH, s.wallsV, pw.o, pw.r, pw.c)) {
            qerr("wall '" + pw.tok + "' overlaps or crosses another wall");
            return 2;
        }
        if (pw.o == 0) s.wallsH |= 1ull << slotIdx(pw.r, pw.c);
        else s.wallsV |= 1ull << slotIdx(pw.r, pw.c);
    }
    // Wall-count sanity: board + both hands cannot exceed the 20 walls that
    // exist. Fewer is accepted on purpose -- the editor may drop walls.
    int placed = __builtin_popcountll(s.wallsH) + __builtin_popcountll(s.wallsV);
    if (placed + (int)s.wallsLeft[0] + (int)s.wallsLeft[1] > 2 * WALLS_PER_PLAYER) {
        qerr("wall budget exceeded: " + std::to_string(placed) + " placed plus " +
             std::to_string((int)s.wallsLeft[0] + (int)s.wallsLeft[1]) + " in hand");
        return 2;
    }
    if (!hasPathToGoal(s.wallsH, s.wallsV, s.pawn[0], 0)) { qerr("player 0 has no path to goal"); return 2; }
    if (!hasPathToGoal(s.wallsH, s.wallsV, s.pawn[1], 1)) { qerr("player 1 has no path to goal"); return 2; }

    // Hash convention: the turn key is included exactly when turn == 1.
    Zobrist& z = zobrist();
    uint64_t h = z.pawnKey[0][s.pawn[0]] ^ z.pawnKey[1][s.pawn[1]];
    for (int sl = 0; sl < WS * WS; sl++) {
        if ((s.wallsH >> sl) & 1ull) h ^= z.wallHKey[sl];
        if ((s.wallsV >> sl) & 1ull) h ^= z.wallVKey[sl];
    }
    if (s.turn == 1) h ^= z.turnKey;
    s.hash = h;

    g_scratch = s;
    scratchRegen();
    return 0;
}
} // namespace

extern "C" {

EMSCRIPTEN_KEEPALIVE
void qr_new_game() {
    startFromState(initialState());
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
            recordLiveMove(g_moves[i]);
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
            recordLiveMove(m);
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
    // Híbrido só quando a NNUE está ativa; senão, alpha-beta puro.
    mcab::McabParams p = g_mcab.params();
    p.enabled = g_mcabWanted && (g_engine.getEvalMode() == qr::Negamax::EvalMode::NNUE);
    g_mcab.setParams(p);
    Move m = g_mcab.choose(g_engine, g_state, maxDepth, timeMs, st, g_reptbl);
    g_lastEngineMove = m;
    g_lastEngineScore = st.score;
    g_state = applyMove(g_state, m);
    recordLiveMove(m);
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
    // The analysis engine follows the same eval mode as the game engine.
    g_anEngine.setEvalMode(qr::Negamax::EvalMode::NNUE);
    g_anEngine.setPolicyOrderingEnabled(true);
    return 1;
}

// Volta o engine ao modo heurístico (evalSimple), sem descarregar os pesos
// da memória -- permite alternar sem precisar recarregar o arquivo.
EMSCRIPTEN_KEEPALIVE
void qr_set_eval_heuristic() {
    g_engine.setEvalMode(qr::Negamax::EvalMode::Heuristic);
    g_anEngine.setEvalMode(qr::Negamax::EvalMode::Heuristic);
}

// Retorna 1 se o engine está em modo NNUE, 0 se heurístico.
EMSCRIPTEN_KEEPALIVE
int qr_eval_mode_is_nnue() {
    return g_engine.getEvalMode() == qr::Negamax::EvalMode::NNUE ? 1 : 0;
}

// Liga/desliga o MCTS híbrido em tempo de execução (1 = híbrido, 0 =
// alpha-beta puro). Ligado por default. Em modo heurístico é ignorado --
// o híbrido exige NNUE.
EMSCRIPTEN_KEEPALIVE
void qr_set_mcab_enabled(int on) {
    g_mcabWanted = (on != 0);
    g_mcab.resetTree();
}

// 1 se a busca do próximo lance vai de fato usar o híbrido (ou seja:
// pedido E com NNUE carregada).
EMSCRIPTEN_KEEPALIVE
int qr_mcab_active() {
    return (g_mcabWanted && g_engine.getEvalMode() == qr::Negamax::EvalMode::NNUE) ? 1 : 0;
}

EMSCRIPTEN_KEEPALIVE
int qr_is_draw() {
    return g_reptbl.count(g_state.hash) >= 3 ? 1 : 0;
}

// =========================================================================
// P6 (gui-premium.md) -- history navigation, scratch analysis, editor and
// QFEN serialization. English comments per AGENTS.md; the legacy surface
// above keeps its original comments.
// =========================================================================

// ---- history navigation (live game) ------------------------------------

EMSCRIPTEN_KEEPALIVE int qr_ply_count() { return (int)g_histMoves.size(); }
EMSCRIPTEN_KEEPALIVE int qr_cursor() { return g_cursor; }

EMSCRIPTEN_KEEPALIVE int qr_ply_is_wall(int i) { return g_histMoves[(size_t)i].isWall ? 1 : 0; }
EMSCRIPTEN_KEEPALIVE int qr_ply_a(int i) { return g_histMoves[(size_t)i].a; }
EMSCRIPTEN_KEEPALIVE int qr_ply_b(int i) { return g_histMoves[(size_t)i].b; }
EMSCRIPTEN_KEEPALIVE int qr_ply_c(int i) { return g_histMoves[(size_t)i].c; }

// Moves the review cursor to `ply` (clamped to the game length). Rebuilds
// the repetition table from the real position history so that 3-fold
// detection stays correct after a jump. Returns the cursor actually set.
EMSCRIPTEN_KEEPALIVE int qr_goto_ply(int ply) {
    if (ply < 0) ply = 0;
    if (ply > (int)g_histMoves.size()) ply = (int)g_histMoves.size();
    g_cursor = ply;
    g_state = g_histStates[(size_t)ply];
    g_reptbl.clear();
    for (int i = 0; i <= ply; i++) g_reptbl.push(g_histStates[(size_t)i].hash);
    regenMoves();
    // The MCTS tree reuse assumes one continuous line; a navigation jump
    // breaks that assumption, so drop the tree instead of risking a stale
    // subtree.
    g_mcab.resetTree();
    return g_cursor;
}

// Drops every ply after `ply` (clamped). This is the destructive variant of
// goto: takeback and "play a new move while reviewing" use it. Returns the
// new game length.
EMSCRIPTEN_KEEPALIVE int qr_truncate_history(int ply) {
    qr_goto_ply(ply);
    g_histMoves.resize((size_t)g_cursor);
    g_histStates.resize((size_t)g_cursor + 1);
    return (int)g_histMoves.size();
}

// ---- scratch position ---------------------------------------------------

EMSCRIPTEN_KEEPALIVE void qr_scratch_reset() {
    g_scratch = initialState();
    scratchRegen();
}
EMSCRIPTEN_KEEPALIVE void qr_scratch_from_live() {
    g_scratch = g_state;
    scratchRegen();
}
EMSCRIPTEN_KEEPALIVE void qr_scratch_from_ply(int ply) {
    if (ply < 0) ply = 0;
    if (ply > (int)g_histStates.size() - 1) ply = (int)g_histStates.size() - 1;
    g_scratch = g_histStates[(size_t)ply];
    scratchRegen();
}
EMSCRIPTEN_KEEPALIVE int qr_scr_apply_pawn(int destCell) {
    Move m = Move::pawn((uint8_t)destCell);
    return scratchApply(m);
}
EMSCRIPTEN_KEEPALIVE int qr_scr_apply_wall(int o, int r, int c) {
    return scratchApply(Move::wall(o, r, c));
}
EMSCRIPTEN_KEEPALIVE int qr_scr_turn() { return g_scratch.turn; }
EMSCRIPTEN_KEEPALIVE int qr_scr_pawn(int player) { return g_scratch.pawn[player]; }
EMSCRIPTEN_KEEPALIVE int qr_scr_walls_left(int player) { return g_scratch.wallsLeft[player]; }
EMSCRIPTEN_KEEPALIVE int qr_scr_wall_h_bit(int slot) { return (int)((g_scratch.wallsH >> slot) & 1ull); }
EMSCRIPTEN_KEEPALIVE int qr_scr_wall_v_bit(int slot) { return (int)((g_scratch.wallsV >> slot) & 1ull); }
EMSCRIPTEN_KEEPALIVE int qr_scr_dist(int player) {
    return shortestPathLen(g_scratch.wallsH, g_scratch.wallsV, g_scratch.pawn[player], player);
}

// ---- multi-line analysis (plan section 5.6) ------------------------------
//
// Runs on the scratch position and never touches the live game. Line 1 comes
// from a full iterative search at the scratch root, so it is the same move a
// real game turn would play. Lines 2..N come from independent searches of
// the child position after each candidate root move: this is honest MultiPV
// in the sense that each line is a real search of a different move, but only
// line 1 is proven best. Scores are mover-relative (positive is good for the
// side to move at the scratch).

EMSCRIPTEN_KEEPALIVE int qr_analyze(int maxDepth, int timeMs, int lines) {
    g_anLines.clear();
    g_anNodes = 0;
    g_anDepth = 0;
    if (winner(g_scratch) != -1) return 0;
    if (lines < 1) lines = 1;
    if (lines > 5) lines = 5;
    if (maxDepth < 1) maxDepth = 1;
    if (maxDepth > 24) maxDepth = 24;
    if (timeMs < 50) timeMs = 50;

    Move pvBuf[16];

    // Line 1: full search at the root.
    SearchStats st;
    Move best1 = g_anEngine.chooseMove(g_scratch, maxDepth,
                                       lines > 1 ? timeMs / 2 : timeMs, st);
    g_anNodes += st.nodes;
    g_anDepth = st.reachedDepth;
    AnLine l1;
    l1.score = st.score;
    l1.pv.push_back(best1);
    State child1 = applyMove(g_scratch, best1);
    int n1 = g_anEngine.extractPv(child1, 15, pvBuf);
    for (int i = 0; i < n1; i++) l1.pv.push_back(pvBuf[i]);
    g_anLines.push_back(l1);
    if (lines == 1) return 1;

    // Ordering pass over the remaining root moves: cheap fixed-depth score.
    // Terminal children skip the search entirely (exact score).
    struct Cand { Move m; int q; };
    std::vector<Cand> cands;
    MoveList rootMoves = legalMoves(g_scratch);
    for (size_t i = 0; i < rootMoves.size(); i++) {
        if (rootMoves[i] == best1) continue;
        State ch = applyMove(g_scratch, rootMoves[i]);
        int q;
        int w = winner(ch);
        if (w != -1) {
            q = (w == ch.turn) ? (SCORE_INF - 1) : -(SCORE_INF - 1);
        } else {
            SearchStats qs;
            q = -g_anEngine.searchShallow(ch, 2, qs);   // negamax flip
            g_anNodes += qs.nodes;
        }
        cands.push_back({rootMoves[i], q});
    }
    std::sort(cands.begin(), cands.end(),
              [](const Cand& a, const Cand& b) { return a.q > b.q; });

    int sliceMs = (timeMs - timeMs / 2) / (lines - 1);
    if (sliceMs < 40) sliceMs = 40;
    for (int k = 0; k < lines - 1 && k < (int)cands.size(); k++) {
        const Move m = cands[(size_t)k].m;
        State ch = applyMove(g_scratch, m);
        AnLine ln;
        ln.pv.push_back(m);
        int w = winner(ch);
        if (w != -1) {
            ln.score = (w == ch.turn) ? (SCORE_INF - 1) : -(SCORE_INF - 1);
        } else {
            SearchStats cst;
            Move cm = g_anEngine.chooseMove(ch, maxDepth, sliceMs, cst);
            ln.score = -cst.score;   // child value is from the opponent's view
            g_anNodes += cst.nodes;
            if (cst.reachedDepth > g_anDepth) g_anDepth = cst.reachedDepth;
            // chooseMove always returns a legal move of ch (the empty-hands
            // endgame shortcut too), so the PV continues even at nodes == 0.
            ln.pv.push_back(cm);
            State gc = applyMove(ch, cm);
            int n2 = g_anEngine.extractPv(gc, 14, pvBuf);
            for (int i = 0; i < n2; i++) ln.pv.push_back(pvBuf[i]);
        }
        g_anLines.push_back(ln);
    }
    std::sort(g_anLines.begin(), g_anLines.end(),
              [](const AnLine& a, const AnLine& b) { return a.score > b.score; });
    return (int)g_anLines.size();
}

// Analysis result getters. Line scores are mover-relative at the scratch
// position. qr_an_line_move packs one move (see packMove).
EMSCRIPTEN_KEEPALIVE int qr_an_line_count() { return (int)g_anLines.size(); }
EMSCRIPTEN_KEEPALIVE int qr_an_line_score(int i) { return g_anLines[(size_t)i].score; }
EMSCRIPTEN_KEEPALIVE int qr_an_line_len(int i) { return (int)g_anLines[(size_t)i].pv.size(); }
EMSCRIPTEN_KEEPALIVE int qr_an_line_move(int i, int j) {
    return packMove(g_anLines[(size_t)i].pv[(size_t)j]);
}
EMSCRIPTEN_KEEPALIVE int qr_an_nodes() {
    return g_anNodes > 2147483647LL ? 2147483647 : (int)g_anNodes;
}
EMSCRIPTEN_KEEPALIVE int qr_an_depth() { return g_anDepth; }

// ---- position editor (plan section 5.9) ----------------------------------
//
// The editor works on the same scratch position as the analysis. Wall
// placement is geometric only here: the validity bitmask reports path and
// budget problems instead of rejecting the edit, so the user can build a
// position freely and read what is wrong with it.

// Places or removes a wall. Placement fails (-1) only on physical
// conflicts: an occupied slot or a crossing wall. Path legality is NOT
// checked here on purpose.
EMSCRIPTEN_KEEPALIVE int qr_edit_set_wall(int o, int r, int c, int on) {
    if (r < 0 || r >= WS || c < 0 || c >= WS) return -1;
    uint64_t* bb = (o == 0) ? &g_scratch.wallsH : &g_scratch.wallsV;
    uint64_t* cross = (o == 0) ? &g_scratch.wallsV : &g_scratch.wallsH;
    int slot = slotIdx(r, c);
    if (on) {
        if ((*bb >> slot) & 1ull) return -1;                 // same slot
        if ((*cross >> slot) & 1ull) return -1;              // crossing wall
        // colinear overlap: an adjacent H (or V) slot sharing a segment
        if (o == 0) {
            if (c > 0 && ((*bb >> slotIdx(r, c - 1)) & 1ull)) return -1;
            if (c + 1 < WS && ((*bb >> slotIdx(r, c + 1)) & 1ull)) return -1;
        } else {
            if (r > 0 && ((*bb >> slotIdx(r - 1, c)) & 1ull)) return -1;
            if (r + 1 < WS && ((*bb >> slotIdx(r + 1, c)) & 1ull)) return -1;
        }
        *bb |= 1ull << slot;
    } else {
        *bb &= ~(1ull << slot);
    }
    scratchRegen();
    return on ? 1 : 0;
}

EMSCRIPTEN_KEEPALIVE int qr_edit_set_pawn(int player, int cell) {
    if (player < 0 || player > 1) return -1;
    if (cell < 0 || cell >= N * N) return -2;
    if (g_scratch.pawn[1 - player] == cell) return -3;
    g_scratch.pawn[player] = (uint8_t)cell;
    scratchRegen();
    return 0;
}

EMSCRIPTEN_KEEPALIVE void qr_edit_set_walls_left(int player, int n) {
    if (player < 0 || player > 1) return;
    if (n < 0) n = 0;
    if (n > WALLS_PER_PLAYER) n = WALLS_PER_PLAYER;
    g_scratch.wallsLeft[player] = (int8_t)n;
}

EMSCRIPTEN_KEEPALIVE void qr_edit_set_turn(int t) {
    if (t != 0 && t != 1) return;
    g_scratch.turn = (int8_t)t;
    // Recompute the hash from the edited fields. Convention (see
    // initialState/applyMove in rules.hpp): the turn key is part of the
    // hash exactly when turn == 1.
    Zobrist& z = zobrist();
    uint64_t h = z.pawnKey[0][g_scratch.pawn[0]] ^ z.pawnKey[1][g_scratch.pawn[1]];
    for (int s = 0; s < WS * WS; s++) {
        if ((g_scratch.wallsH >> s) & 1ull) h ^= z.wallHKey[s];
        if ((g_scratch.wallsV >> s) & 1ull) h ^= z.wallVKey[s];
    }
    if (t == 1) h ^= z.turnKey;
    g_scratch.hash = h;
    scratchRegen();
}

// Validity bitmask for the editor strip (plan section 5.9). A set bit marks
// one concrete problem; 0 means the position can be applied:
//   bit 0  both pawns share one cell
//   bit 1  player 0 has no path to goal
//   bit 2  player 1 has no path to goal
//   bit 3  a pawn already stands on its goal row (game would be over)
//   bit 4  board walls plus walls in hand exceed the 20 walls that exist
EMSCRIPTEN_KEEPALIVE int qr_edit_validity() {
    int v = 0;
    if (g_scratch.pawn[0] == g_scratch.pawn[1]) v |= 1;
    if (!hasPathToGoal(g_scratch.wallsH, g_scratch.wallsV, g_scratch.pawn[0], 0)) v |= 2;
    if (!hasPathToGoal(g_scratch.wallsH, g_scratch.wallsV, g_scratch.pawn[1], 1)) v |= 4;
    if (rowOf(g_scratch.pawn[0]) == GOAL_ROW[0] || rowOf(g_scratch.pawn[1]) == GOAL_ROW[1]) v |= 8;
    int placed = __builtin_popcountll(g_scratch.wallsH) + __builtin_popcountll(g_scratch.wallsV);
    if (placed + (int)g_scratch.wallsLeft[0] + (int)g_scratch.wallsLeft[1] > 2 * WALLS_PER_PLAYER)
        v |= 16;
    return v;
}

// Commits the scratch position as the new live game. Fails (0) while the
// validity bitmask is non-zero; the GUI keeps Apply disabled then anyway.
EMSCRIPTEN_KEEPALIVE int qr_edit_apply() {
    if (qr_edit_validity() != 0) return 0;
    startFromState(g_scratch);
    return 1;
}

// ---- QFEN import/export (plan section 16.1) ------------------------------

// Writes the QFEN of the live game at the cursor (ply count = cursor) into
// `out`. Returns the string length, or -1 when the buffer is too small.
EMSCRIPTEN_KEEPALIVE int qr_qfen_export(char* out, int cap) {
    std::string s = writeQfen(g_state, g_cursor);
    if ((int)s.size() + 1 > cap) return -1;
    std::memcpy(out, s.c_str(), s.size() + 1);
    return (int)s.size();
}

// Same, for the scratch position (editor / analysis root). Ply count 0.
EMSCRIPTEN_KEEPALIVE int qr_qfen_export_scratch(char* out, int cap) {
    std::string s = writeQfen(g_scratch, 0);
    if ((int)s.size() + 1 > cap) return -1;
    std::memcpy(out, s.c_str(), s.size() + 1);
    return (int)s.size();
}

// Parses and validates a QFEN into the SCRATCH position (never the live
// game; qr_edit_apply commits it). Returns 0 on success, 1 parse error,
// 2 validation error. qr_last_error carries the reason.
EMSCRIPTEN_KEEPALIVE int qr_qfen_import_scratch(const char* qfen) {
    if (!qfen) { qerr("empty input"); return 1; }
    return qfenImportScratch(std::string(qfen));
}

// Copies the last error text of the QFEN parser into `out`.
EMSCRIPTEN_KEEPALIVE void qr_last_error(char* out, int cap) {
    std::strncpy(out, g_lastErr, (size_t)cap - 1);
    out[cap - 1] = '\0';
}




} // extern "C"
