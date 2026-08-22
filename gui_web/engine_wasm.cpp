// engine_wasm.cpp -- thin C shim exporting the engine core (rules.hpp +
// search.hpp + nnue.hpp + mcab.hpp) to WASM, in the same spirit as the
// binding used by Zchezz: plain extern "C" functions over primitive types,
// no embind (cheaper to compile and to load in the browser; JS only ever
// calls through ccall/cwrap).
//
// The module owns ONE game object. It is not reentrant or thread safe, but
// the GUI is single threaded and calls everything sequentially.
//
// The game is modelled as a root position plus a move history plus a cursor:
//
//   rootState  -- start position of the current game, normally initialState()
//   history    -- every ply played from rootState
//   cursor     -- 0..history.size(); the DISPLAYED position is rootState with
//                 the first `cursor` moves applied
//   cur        -- cached State at the cursor, with curMoves (its legal moves)
//                 and repTable (repetition history up to the cursor)
//
// Every position query below reads the position at the cursor, so navigation,
// undo/redo, analysis without playing and the position editor all work off the
// same model. Applying a move while the cursor is behind the end of the
// history truncates the future first, the standard GUI behaviour.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "../src/rules.hpp"
#include "../src/search.hpp"
#include "../src/nnue.hpp"
#include "../src/mcab.hpp"
#include "qfen.hpp"

#ifdef __EMSCRIPTEN__
#include <emscripten/emscripten.h>
#else
#define EMSCRIPTEN_KEEPALIVE
#endif

using namespace qr;

namespace {

// =====================================================================
// Engine instances
// =====================================================================

Negamax g_engine;  // TT reused across engine moves (usual chess/draughts GUI pattern)

// Hybrid MCTS + alpha-beta -- the production search (src/mcab.hpp). A single
// instance stays alive for the whole game: subtree reuse between moves depends
// on the same runner observing every consecutive move.
using McabRunnerT = mcab::McabRunner<qr::Negamax, qr::State, qr::Move, qr::MoveList,
                                     qr::AccPair, qr::RepetitionTable, qr::SearchStats>;
using McabNodeT = McabRunnerT::NodeT;
McabRunnerT g_mcab;

// Mirrors McabParams::enabled, but only takes effect once the NNUE weights are
// loaded -- the PUCT priors come from the policy head, so in heuristic mode the
// hybrid cannot run at all and we fall back to pure alpha-beta.
bool g_mcabWanted = mcab::McabParams{}.enabled;

// RANGE CAVEAT (relevant precisely here): the +46.9 +/-23.5 Elo of the hybrid
// was measured at 200ms/move. It runs at roughly 1/9 the nodes/s of pure
// alpha-beta, and the GUI often asks for much shorter searches -- below some
// (unmeasured) point the trade inverts and pure AB is stronger again. That is
// why qr_set_mcab_enabled exists and is exposed to JS: it can be switched off
// per game without recompiling. See status.md.

bool g_nnueLoaded = false;

// =====================================================================
// Game object: root state + history + cursor
// =====================================================================

struct Game {
    State rootState;
    std::vector<Move> history;
    int cursor = 0;

    State cur;
    std::vector<Move> curMoves;
    RepetitionTable repTable;

    // Who placed each wall standing at the cursor, derived by replaying the
    // history. -1 for walls that came in with the root position (a QFEN or the
    // position editor), which have no author.
    int8_t ownerH[WS * WS];
    int8_t ownerV[WS * WS];

    // Per-position caches. qr_path_cell and qr_static_eval are called many
    // times per rendered frame, so both are computed at most once per cursor
    // position and dropped by rebuild().
    bool evalCached = false;
    int evalValue = 0;
    bool pathCached[2] = {false, false};
    std::vector<int> pathCells[2];

    void invalidateCaches() {
        evalCached = false;
        pathCached[0] = pathCached[1] = false;
        pathCells[0].clear();
        pathCells[1].clear();
    }

    // Recomputes cur/curMoves/repTable/ownership from rootState + history +
    // cursor. Every mutation of the model funnels through here so there is a
    // single place where the derived state can go stale.
    void rebuild() {
        if (cursor < 0) cursor = 0;
        if (cursor > (int)history.size()) cursor = (int)history.size();

        cur = rootState;
        repTable.clear();
        repTable.push(cur.hash);
        std::memset(ownerH, -1, sizeof(ownerH));
        std::memset(ownerV, -1, sizeof(ownerV));

        for (int i = 0; i < cursor; i++) {
            const Move& m = history[(size_t)i];
            int mover = cur.turn;
            if (m.isWall) {
                int slot = slotIdx(m.b, m.c);
                if (slot >= 0 && slot < WS * WS) {
                    if (m.a == 0) ownerH[slot] = (int8_t)mover;
                    else ownerV[slot] = (int8_t)mover;
                }
            }
            cur = applyMove(cur, m);
            repTable.push(cur.hash);
        }
        curMoves = legalMoves(cur).toVector();
        invalidateCaches();
    }
};

Game g_game;

// Last move played by the engine, and its search score. Kept separate from the
// history so the GUI can highlight it even after the user navigates away.
Move g_lastEngineMove = Move::pawn(0);
int g_lastEngineScore = 0;

void resetEngines() {
    // Reuse the configured eval mode (NNUE or heuristic) across games -- the
    // eval mode must not be reset when the engine is recreated. Same for the
    // policy-assisted move ordering.
    Negamax::EvalMode prevMode = g_engine.getEvalMode();
    bool prevPolicyOrdering = g_engine.isPolicyOrderingEnabled();
    int prevPolicyOrderingMinDepth = g_engine.getPolicyOrderingMinDepth();
    g_engine = Negamax();  // clears the TT: a new game must not inherit stale entries
    g_engine.setEvalMode(prevMode);
    g_engine.setPolicyOrderingEnabled(prevPolicyOrdering);
    g_engine.setPolicyOrderingMinDepth(prevPolicyOrderingMinDepth);
    // Same reason we clear the TT: the reused subtree belongs to the previous
    // game and is worthless for the new one.
    g_mcab.resetTree();
}

// True when the hybrid will actually run for the next search.
bool mcabActive() {
    return g_mcabWanted && g_engine.getEvalMode() == Negamax::EvalMode::NNUE;
}

int staticEvalOf(const State& s) {
    if (g_engine.getEvalMode() == Negamax::EvalMode::NNUE && g_nnueLoaded) {
        AccPair ap = buildAccPairRoot(s, g_engine.pathCache());
        return nnueEvalInt(ap, s.turn);
    }
    return evalSimple(s, s.turn);
}

// Builds one shortest path for `player` from the pawn cell to the goal row, as
// a list of cells starting at the pawn. Empty when the goal is unreachable.
void buildPath(const State& s, int player, std::vector<int>& out) {
    out.clear();
    PlayerPathCache pc;
    computeDistFull(s.wallsH, s.wallsV, s.pawn[player], player, pc);
    if (!pc.valid || pc.goalCell < 0 || pc.distToGoal < 0) return;
    int cell = pc.goalCell;
    while (cell != -1) {
        out.push_back(cell);
        cell = pc.parent[(size_t)cell];
    }
    std::reverse(out.begin(), out.end());
}

// =====================================================================
// Analysis results
// =====================================================================

struct AnalysisLine {
    int score = 0;    // mover-relative, evalSimple/NNUE_EVAL_SCALE units
    int visits = 0;   // MCTS visits; 0 when the analysis ran pure alpha-beta
    std::vector<Move> pv;
};

std::vector<AnalysisLine> g_anLines;
long long g_anNodes = 0;
int g_anDepth = 0;
int g_anIsMcab = 0;

constexpr int AN_MAX_PV = 12;
constexpr int AN_WIN_SCORE = 900000;

// Inverse of mcab::scoreToQ: q = 1/(1+exp(-score/scale)) => score = scale*logit(q).
int qToScore(double q, double scale) {
    const double eps = 1e-6;
    if (q < eps) q = eps;
    if (q > 1.0 - eps) q = 1.0 - eps;
    double v = scale * std::log(q / (1.0 - q));
    if (v > (double)AN_WIN_SCORE) v = (double)AN_WIN_SCORE;
    if (v < -(double)AN_WIN_SCORE) v = -(double)AN_WIN_SCORE;
    return (int)std::lround(v);
}

double edgeQOf(const McabNodeT& n, size_t e) {
    if (g_mcab.params().backupMode == mcab::BackupMode::AvgBlend) {
        if (n.N[e] <= 0.f) return 0.5;
        return (double)n.W[e] / (double)n.N[e];
    }
    return (double)n.W[e];
}

// Walks the max-visit child chain below root edge `edge`, appending moves to
// `pv` (which already contains the root move) until AN_MAX_PV plies or a node
// with no expanded child.
void walkMcabPV(const McabNodeT& root, size_t edge, std::vector<Move>& pv) {
    const McabNodeT* node = g_mcab.nodeAtForInspection(root.child[edge]);
    while (node && (int)pv.size() < AN_MAX_PV) {
        if (!node->expanded || node->terminal) break;
        size_t nm = (size_t)std::min(node->activeMoves, (int)node->moves.size());
        if (nm == 0) break;
        size_t best = nm;
        float bestN = 0.f;
        for (size_t i = 0; i < nm; i++) {
            if (node->N[i] > bestN) { bestN = node->N[i]; best = i; }
        }
        if (best == nm) break;  // nothing visited below here
        pv.push_back(node->moves[best]);
        node = g_mcab.nodeAtForInspection(node->child[best]);
    }
}

// Reads the multi-PV lines out of the MCTS tree left standing by the last
// choose()/chooseMoveMCAB(). Returns false when the tree is missing or belongs
// to a different position (the "empty hands" endgame shortcut and the
// non-NNUE fallback inside chooseMoveMCAB both leave the pool untouched).
bool captureMcabLines(const State& s, int multipv) {
    const McabNodeT* root = g_mcab.rootNodeForInspection();
    if (!root) return false;
    if (root->state.hash != s.hash) return false;
    if (root->terminal) return false;
    size_t nm = (size_t)std::min(root->activeMoves, (int)root->moves.size());
    if (nm == 0) return false;

    std::vector<size_t> order;
    for (size_t i = 0; i < nm; i++) {
        if (root->N[i] > 0.f) order.push_back(i);
    }
    if (order.empty()) return false;
    std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) {
        if (root->N[a] != root->N[b]) return root->N[a] > root->N[b];
        return edgeQOf(*root, a) > edgeQOf(*root, b);
    });

    double scale = g_mcab.params().scoreScale;
    g_anLines.clear();
    for (size_t k = 0; k < order.size() && (int)k < multipv; k++) {
        size_t e = order[k];
        AnalysisLine line;
        line.visits = (int)root->N[e];
        line.score = qToScore(edgeQOf(*root, e), scale);
        line.pv.push_back(root->moves[e]);
        walkMcabPV(*root, e, line.pv);
        g_anLines.push_back(std::move(line));
    }
    return !g_anLines.empty();
}

// Greedy principal-variation extension for the alpha-beta path.
//
// DEVIATION from GUI_PLAN.md, deliberate: the plan suggests walking the
// transposition table, but Negamax::probe() is private and search.hpp is not a
// W1-owned file, so there is no way to read the TT from here. Instead of
// settling for the 1-ply PV the plan allows as a fallback, the continuation is
// built greedily by picking, at each node, the move with the best static
// evaluation of the resulting position. It is cheap (no search) and gives the
// GUI a plausible line to display; it is NOT a searched PV, and the app should
// not treat plies beyond the first as engine-endorsed.
void extendGreedyPV(State s, std::vector<Move>& pv) {
    while ((int)pv.size() < AN_MAX_PV) {
        if (winner(s) != -1) break;
        MoveList ms = legalMoves(s);
        if (ms.empty()) break;
        int bestIdx = -1;
        int bestScore = 0;
        for (size_t i = 0; i < ms.size(); i++) {
            State child = applyMove(s, ms[i]);
            int sc = (winner(child) == s.turn) ? AN_WIN_SCORE : -staticEvalOf(child);
            if (bestIdx < 0 || sc > bestScore) { bestIdx = (int)i; bestScore = sc; }
        }
        if (bestIdx < 0) break;
        pv.push_back(ms[(size_t)bestIdx]);
        s = applyMove(s, ms[(size_t)bestIdx]);
    }
}

// Root-split alpha-beta analysis with iterative deepening. Every completed
// depth replaces the previous result, so a hard time cut always leaves the
// deepest fully finished iteration in place.
void analyzeAlphaBeta(const State& s, const std::vector<Move>& moves,
                      int maxDepth, int timeMs, int multipv) {
    using clock = std::chrono::steady_clock;
    auto t0 = clock::now();
    auto deadline = t0 + std::chrono::milliseconds(std::max(1, timeMs));

    RepetitionTable rt = g_game.repTable;
    rt.markRoot();
    g_engine.resetOrderingState();

    SearchStats stats;
    std::vector<std::pair<int, size_t>> best;  // (score, move index)
    int completedDepth = 0;

    for (int depth = 0; depth <= std::max(0, maxDepth - 1); depth++) {
        std::vector<std::pair<int, size_t>> scored;
        bool aborted = false;
        for (size_t i = 0; i < moves.size(); i++) {
            auto now = clock::now();
            if (now >= deadline) { aborted = true; break; }
            long long remainMs =
                std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now).count();
            State child = applyMove(s, moves[i]);
            int sc;
            if (winner(child) != -1) {
                // Only the side that just moved can have won.
                sc = AN_WIN_SCORE;
            } else {
                sc = -g_engine.searchLeaf(child, depth, stats, rt, nullptr,
                                          (int)std::max(1ll, remainMs));
                if (g_engine.searchWasStopped()) { aborted = true; break; }
            }
            scored.push_back({sc, i});
        }
        if (aborted) break;
        std::stable_sort(scored.begin(), scored.end(),
                         [](const std::pair<int, size_t>& a, const std::pair<int, size_t>& b) {
                             return a.first > b.first;
                         });
        best.swap(scored);
        completedDepth = depth + 1;
        if (clock::now() >= deadline) break;
    }

    if (best.empty()) {
        // Not even depth 0 finished: fall back to a static ranking so the GUI
        // still gets a legal best move instead of nothing.
        for (size_t i = 0; i < moves.size(); i++) {
            State child = applyMove(s, moves[i]);
            int sc = (winner(child) != -1) ? AN_WIN_SCORE : -staticEvalOf(child);
            best.push_back({sc, i});
        }
        std::stable_sort(best.begin(), best.end(),
                         [](const std::pair<int, size_t>& a, const std::pair<int, size_t>& b) {
                             return a.first > b.first;
                         });
    }

    g_anLines.clear();
    for (size_t k = 0; k < best.size() && (int)k < multipv; k++) {
        AnalysisLine line;
        line.score = best[k].first;
        line.visits = 0;
        line.pv.push_back(moves[best[k].second]);
        extendGreedyPV(applyMove(s, moves[best[k].second]), line.pv);
        g_anLines.push_back(std::move(line));
    }
    g_anNodes = (long long)stats.nodes;
    g_anDepth = completedDepth;
    g_anIsMcab = 0;
}

// Shared string buffer for the const char* returning exports. One buffer per
// export would be tidier, but the contract is only "valid until the next call
// to the SAME function", so a per-function static is what we use.
std::string& scratch(int slot) {
    static std::string bufs[6];
    return bufs[slot];
}

// =====================================================================
// Position editor buffer
// =====================================================================

State g_edit = initialState();

}  // namespace

extern "C" {

// =====================================================================
// Lifecycle / engine config
// =====================================================================

EMSCRIPTEN_KEEPALIVE
void qr_new_game() {
    g_game.rootState = initialState();
    g_game.history.clear();
    g_game.cursor = 0;
    g_game.rebuild();
    g_lastEngineMove = Move::pawn(0);
    g_lastEngineScore = 0;
    g_anLines.clear();
    g_anNodes = 0;
    g_anDepth = 0;
    g_anIsMcab = 0;
    resetEngines();
}

// Loads quantized NNUE weights from `path` and switches the engine to NNUE.
// Returns 1 on success, 0 when the file could not be opened or is corrupt.
// To revert to the heuristic: call qr_set_eval_heuristic(), or simply never
// call this -- the engine starts in heuristic mode.
//
// Policy-assisted move ordering (Negamax::setPolicyOrderingEnabled, search.hpp)
// is also switched on here: it measured stronger in arena testing than purely
// heuristic/CAT ordering, so the production engine (this WASM binary) always
// plays with it on whenever NNUE is loaded. min-depth stays at the class
// default (3). The explicit call is redundant with the search.hpp default
// today, but is kept so this does not silently depend on that default.
EMSCRIPTEN_KEEPALIVE
int qr_load_nnue_weights(const char* path) {
    if (!path) return 0;
    if (!qr::loadWeightsQuant(path)) return 0;
    g_nnueLoaded = true;
    g_engine.setEvalMode(qr::Negamax::EvalMode::NNUE);
    g_engine.setPolicyOrderingEnabled(true);
    g_game.invalidateCaches();
    return 1;
}

// Back to the heuristic eval (evalSimple) without unloading the weights, so
// the two modes can be toggled without re-reading the file.
EMSCRIPTEN_KEEPALIVE
void qr_set_eval_heuristic() {
    g_engine.setEvalMode(qr::Negamax::EvalMode::Heuristic);
    g_game.invalidateCaches();
}

EMSCRIPTEN_KEEPALIVE
int qr_eval_mode_is_nnue() {
    return g_engine.getEvalMode() == qr::Negamax::EvalMode::NNUE ? 1 : 0;
}

// Switches the hybrid MCTS on/off at runtime (1 = hybrid, 0 = pure
// alpha-beta). On by default. Ignored in heuristic mode -- the hybrid needs
// NNUE.
EMSCRIPTEN_KEEPALIVE
void qr_set_mcab_enabled(int on) {
    g_mcabWanted = (on != 0);
    g_mcab.resetTree();
}

// 1 when the next search will actually use the hybrid (requested AND NNUE).
EMSCRIPTEN_KEEPALIVE
int qr_mcab_active() { return mcabActive() ? 1 : 0; }

// =====================================================================
// Position query (cursor-relative)
// =====================================================================

EMSCRIPTEN_KEEPALIVE int qr_turn() { return g_game.cur.turn; }
EMSCRIPTEN_KEEPALIVE int qr_winner() { return winner(g_game.cur); }
EMSCRIPTEN_KEEPALIVE int qr_pawn(int player) {
    if (player < 0 || player > 1) return -1;
    return g_game.cur.pawn[player];
}
EMSCRIPTEN_KEEPALIVE int qr_walls_left(int player) {
    if (player < 0 || player > 1) return 0;
    return g_game.cur.wallsLeft[player];
}
EMSCRIPTEN_KEEPALIVE int qr_wall_h_bit(int slot) {
    if (slot < 0 || slot >= WS * WS) return 0;
    return (int)((g_game.cur.wallsH >> slot) & 1ull);
}
EMSCRIPTEN_KEEPALIVE int qr_wall_v_bit(int slot) {
    if (slot < 0 || slot >= WS * WS) return 0;
    return (int)((g_game.cur.wallsV >> slot) & 1ull);
}

// BFS distance to the goal -- a display indicator, it does not affect search.
EMSCRIPTEN_KEEPALIVE int qr_dist_to_goal(int player) {
    if (player < 0 || player > 1) return -1;
    return shortestPathLen(g_game.cur.wallsH, g_game.cur.wallsV, g_game.cur.pawn[player], player);
}

EMSCRIPTEN_KEEPALIVE int qr_path_len(int player) { return qr_dist_to_goal(player); }

// i-th cell of one shortest path for `player` (i == 0 is the pawn cell), or -1
// past the end / when the goal is unreachable. Cached per cursor position, so
// calling this a few dozen times per frame costs one BFS.
EMSCRIPTEN_KEEPALIVE int qr_path_cell(int player, int i) {
    if (player < 0 || player > 1 || i < 0) return -1;
    if (!g_game.pathCached[player]) {
        buildPath(g_game.cur, player, g_game.pathCells[player]);
        g_game.pathCached[player] = true;
    }
    const std::vector<int>& p = g_game.pathCells[player];
    if ((size_t)i >= p.size()) return -1;
    return p[(size_t)i];
}

// Mover-relative static evaluation of the position at the cursor: NNUE when
// loaded and selected, evalSimple otherwise. Cached per cursor position.
EMSCRIPTEN_KEEPALIVE int qr_static_eval() {
    if (!g_game.evalCached) {
        g_game.evalValue = staticEvalOf(g_game.cur);
        g_game.evalCached = true;
    }
    return g_game.evalValue;
}

EMSCRIPTEN_KEEPALIVE int qr_is_wall_legal(int orientation, int r, int c) {
    if (orientation < 0 || orientation > 1) return 0;
    if (r < 0 || r >= WS || c < 0 || c >= WS) return 0;
    return isWallMoveLegal(g_game.cur, g_game.cur.turn, orientation, r, c) ? 1 : 0;
}

// 0/1 = the player who placed this wall, -1 when the wall is absent or its
// author is unknown (it came from the root position: a QFEN or the editor).
EMSCRIPTEN_KEEPALIVE int qr_wall_owner(int orientation, int r, int c) {
    if (orientation < 0 || orientation > 1) return -1;
    if (r < 0 || r >= WS || c < 0 || c >= WS) return -1;
    int slot = slotIdx(r, c);
    uint64_t bb = (orientation == 0) ? g_game.cur.wallsH : g_game.cur.wallsV;
    if (!((bb >> slot) & 1ull)) return -1;
    return (orientation == 0) ? g_game.ownerH[slot] : g_game.ownerV[slot];
}

EMSCRIPTEN_KEEPALIVE
int qr_is_draw() { return g_game.repTable.count(g_game.cur.hash) >= 3 ? 1 : 0; }

// =====================================================================
// Legal moves at the cursor
// =====================================================================

EMSCRIPTEN_KEEPALIVE int qr_legal_moves_count() { return (int)g_game.curMoves.size(); }
EMSCRIPTEN_KEEPALIVE int qr_legal_move_is_wall(int i) {
    if (i < 0 || (size_t)i >= g_game.curMoves.size()) return 0;
    return g_game.curMoves[(size_t)i].isWall ? 1 : 0;
}
EMSCRIPTEN_KEEPALIVE int qr_legal_move_a(int i) {
    if (i < 0 || (size_t)i >= g_game.curMoves.size()) return -1;
    return g_game.curMoves[(size_t)i].a;
}
EMSCRIPTEN_KEEPALIVE int qr_legal_move_b(int i) {
    if (i < 0 || (size_t)i >= g_game.curMoves.size()) return -1;
    return g_game.curMoves[(size_t)i].b;
}
EMSCRIPTEN_KEEPALIVE int qr_legal_move_c(int i) {
    if (i < 0 || (size_t)i >= g_game.curMoves.size()) return -1;
    return g_game.curMoves[(size_t)i].c;
}

}  // extern "C"

namespace {
// Appends `m` at the cursor, truncating any future first, and advances the
// cursor onto the new ply.
void pushMove(const Move& m) {
    if (g_game.cursor < (int)g_game.history.size())
        g_game.history.resize((size_t)g_game.cursor);
    g_game.history.push_back(m);
    g_game.cursor = (int)g_game.history.size();
    g_game.rebuild();
}
}  // namespace

extern "C" {

// =====================================================================
// Playing
// =====================================================================

// 1 on success, 0 when the move is not in the legal list. JS should only ever
// offer moves that came out of qr_legal_move_*, so a 0 here means a front-end
// bug, not an "almost legal" move.
EMSCRIPTEN_KEEPALIVE
int qr_apply_pawn_move(int destCell) {
    for (size_t i = 0; i < g_game.curMoves.size(); i++) {
        if (!g_game.curMoves[i].isWall && g_game.curMoves[i].a == destCell) {
            pushMove(g_game.curMoves[i]);
            return 1;
        }
    }
    return 0;
}

EMSCRIPTEN_KEEPALIVE
int qr_apply_wall_move(int orientation, int r, int c) {
    for (size_t i = 0; i < g_game.curMoves.size(); i++) {
        const Move& m = g_game.curMoves[i];
        if (m.isWall && m.a == orientation && m.b == r && m.c == c) {
            pushMove(m);
            return 1;
        }
    }
    return 0;
}

// Runs the search and immediately plays the chosen move. Synchronous: it blocks
// the browser main thread for roughly timeMs. Acceptable for the per-move
// budgets used here (hundreds of ms). Moving it into a Web Worker is the
// natural next step if the UI must stay responsive while the engine thinks --
// the module compiles unchanged inside a worker.
//
// Also records the search result into the qr_an_* accessors, so the GUI can
// show what the engine was thinking without running a second search.
EMSCRIPTEN_KEEPALIVE
int qr_engine_move(int maxDepth, int timeMs) {
    if (winner(g_game.cur) != -1) return 0;
    if (g_game.repTable.count(g_game.cur.hash) >= 3) return 0;
    if (g_game.curMoves.empty()) return 0;

    State searched = g_game.cur;
    SearchStats st;
    // Hybrid only when NNUE is active; pure alpha-beta otherwise.
    mcab::McabParams p = g_mcab.params();
    p.enabled = mcabActive();
    g_mcab.setParams(p);
    Move m = g_mcab.choose(g_engine, searched, maxDepth, timeMs, st, g_game.repTable);
    g_lastEngineMove = m;
    g_lastEngineScore = st.score;

    g_anNodes = (long long)st.nodes;
    g_anIsMcab = p.enabled ? 1 : 0;
    if (!(p.enabled && captureMcabLines(searched, 5))) {
        g_anIsMcab = 0;
        g_anLines.clear();
        AnalysisLine line;
        line.score = st.score;
        line.visits = 0;
        line.pv.push_back(m);
        extendGreedyPV(applyMove(searched, m), line.pv);
        g_anLines.push_back(std::move(line));
    }
    g_anDepth = st.reachedDepth;
    if (g_anDepth <= 0) {
        for (const AnalysisLine& l : g_anLines)
            g_anDepth = std::max(g_anDepth, (int)l.pv.size());
    }

    pushMove(m);
    return 1;
}

EMSCRIPTEN_KEEPALIVE int qr_last_move_is_wall() { return g_lastEngineMove.isWall ? 1 : 0; }
EMSCRIPTEN_KEEPALIVE int qr_last_move_a() { return g_lastEngineMove.a; }
EMSCRIPTEN_KEEPALIVE int qr_last_move_b() { return g_lastEngineMove.b; }
EMSCRIPTEN_KEEPALIVE int qr_last_move_c() { return g_lastEngineMove.c; }

// Evaluation (evalSimple units) of the move above, from the point of view of
// the side that played it: positive is good for the engine. It only reflects
// the depth the search reached inside its time budget.
EMSCRIPTEN_KEEPALIVE int qr_last_move_eval() { return g_lastEngineScore; }

// =====================================================================
// History / navigation
// =====================================================================

EMSCRIPTEN_KEEPALIVE int qr_history_len() { return (int)g_game.history.size(); }
EMSCRIPTEN_KEEPALIVE int qr_history_cursor() { return g_game.cursor; }

EMSCRIPTEN_KEEPALIVE
int qr_goto_ply(int ply) {
    int target = ply;
    if (target < 0) target = 0;
    if (target > (int)g_game.history.size()) target = (int)g_game.history.size();
    if (target != g_game.cursor) {
        g_game.cursor = target;
        g_game.rebuild();
    }
    return g_game.cursor;
}

EMSCRIPTEN_KEEPALIVE int qr_undo() { return qr_goto_ply(g_game.cursor - 1); }
EMSCRIPTEN_KEEPALIVE int qr_redo() { return qr_goto_ply(g_game.cursor + 1); }

EMSCRIPTEN_KEEPALIVE
void qr_truncate_here() {
    if (g_game.cursor < (int)g_game.history.size()) {
        g_game.history.resize((size_t)g_game.cursor);
        g_game.rebuild();
    }
}

// The move leading from position `ply` to position `ply+1`.
EMSCRIPTEN_KEEPALIVE int qr_hist_move_is_wall(int ply) {
    if (ply < 0 || (size_t)ply >= g_game.history.size()) return 0;
    return g_game.history[(size_t)ply].isWall ? 1 : 0;
}
EMSCRIPTEN_KEEPALIVE int qr_hist_move_a(int ply) {
    if (ply < 0 || (size_t)ply >= g_game.history.size()) return -1;
    return g_game.history[(size_t)ply].a;
}
EMSCRIPTEN_KEEPALIVE int qr_hist_move_b(int ply) {
    if (ply < 0 || (size_t)ply >= g_game.history.size()) return -1;
    return g_game.history[(size_t)ply].b;
}
EMSCRIPTEN_KEEPALIVE int qr_hist_move_c(int ply) {
    if (ply < 0 || (size_t)ply >= g_game.history.size()) return -1;
    return g_game.history[(size_t)ply].c;
}

// Which player made the move at `ply`. Derived from the root turn and the ply
// parity, so it stays correct for games started from an edited position where
// player 1 moves first.
EMSCRIPTEN_KEEPALIVE int qr_hist_mover(int ply) {
    if (ply < 0 || (size_t)ply >= g_game.history.size()) return -1;
    return (g_game.rootState.turn + ply) & 1;
}

// =====================================================================
// Analysis
// =====================================================================

// Searches the position at the cursor WITHOUT applying anything. Returns the
// number of lines produced (0 when the position is already decided or has no
// legal moves). `multipv` is clamped to [1,5]; `timeMs` is a hard budget,
// because this blocks the main thread and the app calls it in small slices.
EMSCRIPTEN_KEEPALIVE
int qr_analyze(int maxDepth, int timeMs, int multipv) {
    if (multipv < 1) multipv = 1;
    if (multipv > 5) multipv = 5;
    if (maxDepth < 1) maxDepth = 1;
    if (timeMs < 1) timeMs = 1;

    g_anLines.clear();
    g_anNodes = 0;
    g_anDepth = 0;
    g_anIsMcab = 0;

    const State s = g_game.cur;
    if (winner(s) != -1 || g_game.curMoves.empty()) return 0;

    if (mcabActive()) {
        SearchStats st;
        mcab::McabParams p = g_mcab.params();
        p.enabled = true;
        g_mcab.setParams(p);
        mcab::McabStats ms;
        g_mcab.choose(g_engine, s, maxDepth, timeMs, st, g_game.repTable, &ms);
        g_anNodes = (long long)st.nodes;
        if (captureMcabLines(s, multipv)) {
            g_anIsMcab = 1;
            g_anDepth = 0;
            for (const AnalysisLine& l : g_anLines)
                g_anDepth = std::max(g_anDepth, (int)l.pv.size());
            return (int)g_anLines.size();
        }
        // The hybrid delegated to pure alpha-beta (empty-hands endgame
        // shortcut, or a terminal root) -- fall through and produce the lines
        // the alpha-beta way so the GUI always gets a usable answer.
    }

    analyzeAlphaBeta(s, g_game.curMoves, maxDepth, timeMs, multipv);
    return (int)g_anLines.size();
}

EMSCRIPTEN_KEEPALIVE int qr_an_line_count() { return (int)g_anLines.size(); }
EMSCRIPTEN_KEEPALIVE int qr_an_line_score(int i) {
    if (i < 0 || (size_t)i >= g_anLines.size()) return 0;
    return g_anLines[(size_t)i].score;
}
EMSCRIPTEN_KEEPALIVE int qr_an_line_visits(int i) {
    if (i < 0 || (size_t)i >= g_anLines.size()) return 0;
    return g_anLines[(size_t)i].visits;
}
EMSCRIPTEN_KEEPALIVE int qr_an_line_len(int i) {
    if (i < 0 || (size_t)i >= g_anLines.size()) return 0;
    return (int)g_anLines[(size_t)i].pv.size();
}

namespace {
const Move* anMove(int i, int j) {
    if (i < 0 || (size_t)i >= g_anLines.size()) return nullptr;
    const std::vector<Move>& pv = g_anLines[(size_t)i].pv;
    if (j < 0 || (size_t)j >= pv.size()) return nullptr;
    return &pv[(size_t)j];
}
}  // namespace

EMSCRIPTEN_KEEPALIVE int qr_an_line_move_is_wall(int i, int j) {
    const Move* m = anMove(i, j);
    return (m && m->isWall) ? 1 : 0;
}
EMSCRIPTEN_KEEPALIVE int qr_an_line_move_a(int i, int j) {
    const Move* m = anMove(i, j);
    return m ? (int)m->a : -1;
}
EMSCRIPTEN_KEEPALIVE int qr_an_line_move_b(int i, int j) {
    const Move* m = anMove(i, j);
    return m ? (int)m->b : -1;
}
EMSCRIPTEN_KEEPALIVE int qr_an_line_move_c(int i, int j) {
    const Move* m = anMove(i, j);
    return m ? (int)m->c : -1;
}
EMSCRIPTEN_KEEPALIVE int qr_an_nodes() {
    // Clamped: JS reads this as a 32-bit int.
    return (int)std::min<long long>(g_anNodes, 2147483647ll);
}
EMSCRIPTEN_KEEPALIVE int qr_an_depth() { return g_anDepth; }
EMSCRIPTEN_KEEPALIVE int qr_an_is_mcab() { return g_anIsMcab; }

// =====================================================================
// Position editor
// =====================================================================

EMSCRIPTEN_KEEPALIVE void qr_edit_begin() { g_edit = g_game.cur; }

EMSCRIPTEN_KEEPALIVE void qr_edit_clear() { g_edit = initialState(); }

EMSCRIPTEN_KEEPALIVE
int qr_edit_set_pawn(int player, int cell) {
    if (player < 0 || player > 1) return 0;
    if (cell < 0 || cell >= N * N) return 0;
    if (g_edit.pawn[1 - player] == (uint8_t)cell) return 0;
    g_edit.pawn[player] = (uint8_t)cell;
    qgui::recomputeHash(g_edit);
    return 1;
}

// 1 = placed, 0 = removed, -1 = rejected (the slot overlaps or crosses an
// existing wall). Placements that block a player are deliberately allowed:
// qr_edit_validate reports them so the UI can show the problem, and
// qr_edit_commit refuses them.
EMSCRIPTEN_KEEPALIVE
int qr_edit_toggle_wall(int orientation, int r, int c) {
    if (orientation < 0 || orientation > 1) return -1;
    if (r < 0 || r >= WS || c < 0 || c >= WS) return -1;
    int slot = slotIdx(r, c);
    uint64_t bit = 1ull << slot;
    uint64_t& target = (orientation == 0) ? g_edit.wallsH : g_edit.wallsV;
    if (target & bit) {
        target &= ~bit;
        qgui::recomputeHash(g_edit);
        return 0;
    }
    if (!wallSlotAvailable(g_edit.wallsH, g_edit.wallsV, orientation, r, c)) return -1;
    target |= bit;
    qgui::recomputeHash(g_edit);
    return 1;
}

EMSCRIPTEN_KEEPALIVE
void qr_edit_set_walls_left(int player, int n) {
    if (player < 0 || player > 1) return;
    if (n < 0) n = 0;
    if (n > WALLS_PER_PLAYER) n = WALLS_PER_PLAYER;
    g_edit.wallsLeft[player] = (int8_t)n;
}

EMSCRIPTEN_KEEPALIVE
void qr_edit_set_turn(int t) {
    int nt = (t != 0) ? 1 : 0;
    if (nt == g_edit.turn) return;
    g_edit.turn = nt;
    qgui::recomputeHash(g_edit);
}

EMSCRIPTEN_KEEPALIVE int qr_edit_pawn(int player) {
    if (player < 0 || player > 1) return -1;
    return g_edit.pawn[player];
}
EMSCRIPTEN_KEEPALIVE int qr_edit_walls_left(int player) {
    if (player < 0 || player > 1) return 0;
    return g_edit.wallsLeft[player];
}
EMSCRIPTEN_KEEPALIVE int qr_edit_turn() { return g_edit.turn; }
EMSCRIPTEN_KEEPALIVE int qr_edit_wall_h_bit(int slot) {
    if (slot < 0 || slot >= WS * WS) return 0;
    return (int)((g_edit.wallsH >> slot) & 1ull);
}
EMSCRIPTEN_KEEPALIVE int qr_edit_wall_v_bit(int slot) {
    if (slot < 0 || slot >= WS * WS) return 0;
    return (int)((g_edit.wallsV >> slot) & 1ull);
}

// 0 = legal. Otherwise a bitmask: 1 = player 0 has no path to goal,
// 2 = player 1 has no path to goal, 4 = pawns on the same cell,
// 8 = more than 20 walls placed, 16 = a pawn already stands on its goal row.
EMSCRIPTEN_KEEPALIVE int qr_edit_validate() { return qgui::validatePosition(g_edit); }

EMSCRIPTEN_KEEPALIVE
int qr_edit_commit() {
    if (qgui::validatePosition(g_edit) != 0) return 0;
    qgui::recomputeHash(g_edit);
    g_game.rootState = g_edit;
    g_game.history.clear();
    g_game.cursor = 0;
    g_game.rebuild();
    g_lastEngineMove = Move::pawn(0);
    g_lastEngineScore = 0;
    g_anLines.clear();
    g_anNodes = 0;
    g_anDepth = 0;
    g_anIsMcab = 0;
    resetEngines();
    return 1;
}

// =====================================================================
// Serialization
// =====================================================================
//
// The const char* returned by each of these points into a static std::string
// owned by the module (JS reads it with UTF8ToString). The pointer stays valid
// until the next call to the SAME function.

EMSCRIPTEN_KEEPALIVE
const char* qr_get_qfen() {
    std::string& b = scratch(0);
    b = qgui::formatQFEN(g_game.cur);
    return b.c_str();
}

// 1 ok, 0 on a parse error or an illegal position. On success the parsed
// position becomes the new root and the history is cleared.
EMSCRIPTEN_KEEPALIVE
int qr_set_qfen(const char* text) {
    if (!text) return 0;
    State s;
    if (!qgui::parseQFEN(std::string(text), s)) return 0;
    if (qgui::validatePosition(s) != 0) return 0;
    g_game.rootState = s;
    g_game.history.clear();
    g_game.cursor = 0;
    g_game.rebuild();
    g_lastEngineMove = Move::pawn(0);
    g_lastEngineScore = 0;
    g_anLines.clear();
    resetEngines();
    return 1;
}

// Numbered move list of the whole game (the full history, not just up to the
// cursor), prefixed with [QFEN "..."] when the root is not the initial
// position.
EMSCRIPTEN_KEEPALIVE
const char* qr_get_game_text() {
    std::string& b = scratch(1);
    b.clear();
    State init = initialState();
    if (g_game.rootState.hash != init.hash) {
        b += "[QFEN \"";
        b += qgui::formatQFEN(g_game.rootState);
        b += "\"]\n";
    }
    for (size_t i = 0; i < g_game.history.size(); i++) {
        if ((i % 2) == 0) {
            if (i) b += ' ';
            b += std::to_string(i / 2 + 1);
            b += ". ";
        } else {
            b += ' ';
        }
        b += qgui::moveNotation(g_game.history[i]);
    }
    return b.c_str();
}

EMSCRIPTEN_KEEPALIVE
int qr_set_game_text(const char* text) {
    if (!text) return 0;
    std::string src(text);

    State root = initialState();
    std::string body;
    body.reserve(src.size());

    // Header lines: [Key "value"], one per line, accepted and ignored except
    // for [QFEN "..."], which sets the start position.
    size_t pos = 0;
    while (pos < src.size()) {
        size_t lineEnd = src.find('\n', pos);
        if (lineEnd == std::string::npos) lineEnd = src.size();
        std::string line = src.substr(pos, lineEnd - pos);
        size_t a = line.find_first_not_of(" \t\r");
        if (a != std::string::npos && line[a] == '[') {
            size_t q1 = line.find('"', a);
            size_t q2 = (q1 == std::string::npos) ? std::string::npos : line.find('"', q1 + 1);
            size_t close = line.find(']', a);
            if (q1 == std::string::npos || q2 == std::string::npos || close == std::string::npos)
                return 0;
            std::string key = line.substr(a + 1, q1 - a - 1);
            while (!key.empty() && (key.back() == ' ' || key.back() == '\t')) key.pop_back();
            if (key == "QFEN") {
                State s;
                if (!qgui::parseQFEN(line.substr(q1 + 1, q2 - q1 - 1), s)) return 0;
                if (qgui::validatePosition(s) != 0) return 0;
                root = s;
            }
        } else {
            body += line;
            body += ' ';
        }
        pos = lineEnd + 1;
    }

    State s = root;
    std::vector<Move> hist;
    std::string tok;
    body += ' ';  // sentinel so the last token is flushed
    for (char ch : body) {
        if (ch != ' ' && ch != '\t' && ch != '\r' && ch != '\n') { tok += ch; continue; }
        if (tok.empty()) continue;
        std::string t = tok;
        tok.clear();

        // Strip annotation glyphs the app may have appended (!, ?, !?, ??).
        while (!t.empty() && (t.back() == '!' || t.back() == '?')) t.pop_back();
        if (t.empty()) continue;
        // Move numbers ("12." or "12") and result markers are skipped.
        if (t == "*" || t == "1-0" || t == "0-1" || t == "1/2-1/2") continue;
        bool numeric = true;
        for (char d : t) {
            if (d == '.') continue;
            if (d < '0' || d > '9') { numeric = false; break; }
        }
        if (numeric) continue;

        Move m;
        if (!qgui::parseMoveToken(t, m)) return 0;
        // Legality is checked against the running position: an illegal token
        // makes the whole import fail rather than producing a broken game.
        MoveList legal = legalMoves(s);
        bool ok = false;
        for (size_t i = 0; i < legal.size(); i++) {
            if (legal[i] == m) { ok = true; break; }
        }
        if (!ok) return 0;
        hist.push_back(m);
        s = applyMove(s, m);
    }

    g_game.rootState = root;
    g_game.history = hist;
    g_game.cursor = (int)hist.size();
    g_game.rebuild();
    g_lastEngineMove = Move::pawn(0);
    g_lastEngineScore = 0;
    g_anLines.clear();
    resetEngines();
    return 1;
}

EMSCRIPTEN_KEEPALIVE
const char* qr_edit_get_qfen() {
    std::string& b = scratch(2);
    b = qgui::formatQFEN(g_edit);
    return b.c_str();
}

// Loads a QFEN into the edit buffer. Structural parse only -- the buffer is
// allowed to hold an illegal position; qr_edit_validate reports it.
EMSCRIPTEN_KEEPALIVE
int qr_edit_set_qfen(const char* text) {
    if (!text) return 0;
    State s;
    if (!qgui::parseQFEN(std::string(text), s)) return 0;
    g_edit = s;
    return 1;
}

EMSCRIPTEN_KEEPALIVE
const char* qr_move_notation(int isWall, int a, int b, int c) {
    std::string& buf = scratch(3);
    buf = qgui::moveNotation(isWall != 0, a, b, c);
    return buf.c_str();
}

}  // extern "C"

#ifndef __EMSCRIPTEN__
// Native builds of this translation unit exist only as a compile sanity check
// (g++ -fsyntax-only, and the QFEN round-trip harness), so there is no main().
#endif
