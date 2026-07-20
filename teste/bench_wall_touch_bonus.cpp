// bench_wall_touch_bonus.cpp -- benchmark ad-hoc (não faz parte da suíte
// persistida, mesmo espírito do "benchmark controlado" da Seção 4.2.3 do
// plano): mede se o WALL_TOUCH_BONUS (search.hpp atual) realmente ajuda,
// de duas formas independentes:
//   1. nós/s e profundidade média em posições fixas, orçamento de tempo
//      fixo -- proxy de eficiência de busca (mesma metodologia já usada
//      nas Seções 4.2.1-4.2.3).
//   2. partidas diretas engine-vs-engine (COM bônus vs SEM bônus,
//      cores alternadas) -- resposta mais direta à pergunta "joga melhor".
//
// qr::Negamax (search.hpp) = versão ATUAL, com WALL_TOUCH_BONUS.
// qr::old::Negamax abaixo = cópia fiel de orderWallMoves de ANTES desta
// sessão (só wallByBFS + killer + history, sem o bônus), pra comparação
// isolada -- mesmo padrão de qr::ref::NegamaxReference em
// test_search_staging.cpp.
#include <cstdio>
#include <chrono>
#include <random>
#include "rules.hpp"
#include "search.hpp"

using namespace qr;
using clockT = std::chrono::steady_clock;
static double msSince(clockT::time_point t0) {
    return std::chrono::duration<double, std::milli>(clockT::now() - t0).count();
}

namespace qr { namespace old {

using qr::TT_BITS; using qr::TT_SIZE; using qr::SCORE_INF; using qr::MAX_PLY;
using qr::WALL_BFS_ORDER_MAX_PLY; using qr::TTFlag; using qr::EXACT; using qr::LOWER; using qr::UPPER;
using qr::TTEntry; using qr::SearchStats;

class Negamax {
public:
    Negamax() : tt(TT_SIZE) {}
    Move chooseMove(const State& root, int maxDepthCap, int timeBudgetMs, SearchStats& stats) {
        auto t0 = clockT::now();
        deadline = t0 + std::chrono::milliseconds(timeBudgetMs);
        stats = SearchStats{}; stopped = false;
        std::memset(killerValid, 0, sizeof(killerValid));
        std::memset(history, 0, sizeof(history));
        Move bestMove = legalMoves(root)[0];
        int prevScore = 0;
        for (int depth = 1; depth <= maxDepthCap && !stopped; depth++) {
            rootDepth = depth;
            int score;
            if (depth <= 2) score = negamax(root, depth, -SCORE_INF, SCORE_INF, stats);
            else {
                int alpha = prevScore - 50, beta = prevScore + 50;
                score = negamax(root, depth, alpha, beta, stats);
                if (!stopped && (score <= alpha || score >= beta))
                    score = negamax(root, depth, -SCORE_INF, SCORE_INF, stats);
            }
            if (stopped) break;
            prevScore = score;
            TTEntry& e = probe(root.hash);
            if (e.valid && e.key == root.hash) bestMove = e.best;
            stats.reachedDepth = depth; stats.score = prevScore;
            if (clockT::now() >= deadline) break;
        }
        return bestMove;
    }
private:
    std::vector<TTEntry> tt;
    clockT::time_point deadline;
    bool stopped = false;
    int rootDepth = 0;
    Move killers[MAX_PLY][2];
    bool killerValid[MAX_PLY][2] = {};
    int history[2][NUM_MOVE_INDICES] = {};
    TTEntry& probe(uint64_t hash) { return tt[hash & (TT_SIZE - 1)]; }
    void store(uint64_t hash, int depth, int score, TTFlag flag, const Move& best) {
        TTEntry& e = probe(hash);
        if (!e.valid || depth >= e.depth) e = {hash, (int16_t)score, (int8_t)depth, flag, best, true};
    }
    void recordCutoff(const Move& m, int ply, int side, int depth) {
        if (ply >= 0 && ply < MAX_PLY) {
            if (!(killerValid[ply][0] && killers[ply][0] == m)) {
                killers[ply][1] = killers[ply][0]; killerValid[ply][1] = killerValid[ply][0];
                killers[ply][0] = m; killerValid[ply][0] = true;
            }
        }
        history[side][moveToPolicyIndex(m)] += depth * depth;
    }
    static constexpr size_t CAP = 256;
    void orderPawnMoves(MoveList& moves, int ply, int side) {
        bool ply0 = ply >= 0 && ply < MAX_PLY;
        size_t n = moves.size();
        static thread_local std::pair<long long, Move> buf[CAP];
        for (size_t i = 0; i < n; i++) {
            const Move& m = moves[i];
            long long sc = history[side][moveToPolicyIndex(m)];
            if (ply0) {
                if (killerValid[ply][0] && m == killers[ply][0]) sc += 1'500'000;
                else if (killerValid[ply][1] && m == killers[ply][1]) sc += 1'400'000;
            }
            buf[i] = {sc, m};
        }
        std::sort(buf, buf + n, [](const auto& x, const auto& y) { return x.first > y.first; });
        for (size_t i = 0; i < n; i++) moves[i] = buf[i].second;
    }
    // ESTA é a versão "antiga" -- sem WALL_TOUCH_BONUS, sem
    // shortestPathTouchSlots, só wallByBFS restrito a ply<=2 + killer/history.
    void orderWallMoves(MoveList& moves, int ply, int side, const State& s) {
        bool ply0 = ply >= 0 && ply < MAX_PLY;
        bool wallByBFS = ply <= WALL_BFS_ORDER_MAX_PLY;
        int opp = 1 - side;
        size_t n = moves.size();
        static thread_local std::pair<long long, Move> buf[CAP];
        for (size_t i = 0; i < n; i++) {
            const Move& m = moves[i];
            long long sc = 0;
            if (wallByBFS) {
                State ns = applyMove(s, m);
                int before = shortestPathLen(s.wallsH, s.wallsV, s.pawn[opp], opp);
                int after = shortestPathLen(ns.wallsH, ns.wallsV, ns.pawn[opp], opp);
                sc = (long long)(after - before) * 1000;
            }
            sc += history[side][moveToPolicyIndex(m)];
            if (ply0) {
                if (killerValid[ply][0] && m == killers[ply][0]) sc += 1'500'000;
                else if (killerValid[ply][1] && m == killers[ply][1]) sc += 1'400'000;
            }
            buf[i] = {sc, m};
        }
        std::sort(buf, buf + n, [](const auto& x, const auto& y) { return x.first > y.first; });
        for (size_t i = 0; i < n; i++) moves[i] = buf[i].second;
    }
    int negamax(const State& s, int depth, int alpha, int beta, SearchStats& stats) {
        stats.nodes++;
        if ((stats.nodes & 0x3FF) == 0 && clockT::now() >= deadline) { stopped = true; return 0; }
        int w = winner(s);
        if (w != -1) return (w == s.turn) ? SCORE_INF - 1 : -(SCORE_INF - 1);
        int alphaOrig = alpha;
        TTEntry& e = probe(s.hash);
        bool hasTTMove = false; Move ttMoveVal = Move::pawn(0);
        if (e.valid && e.key == s.hash) {
            hasTTMove = true; ttMoveVal = e.best;
            if (e.depth >= depth) {
                if (e.flag == EXACT) return e.score;
                if (e.flag == LOWER) alpha = std::max(alpha, (int)e.score);
                else if (e.flag == UPPER) beta = std::min(beta, (int)e.score);
                if (alpha >= beta) return e.score;
            }
        }
        if (depth == 0) return evalSimple(s, s.turn);
        int ply = rootDepth - depth;
        int side = s.turn;
        MoveList pawnMoves;
        pawnStepMoves(s, side, pawnMoves);
        orderPawnMoves(pawnMoves, ply, side);
        bool ttTried = false;
        if (hasTTMove && ttMoveVal.isWall) {
            if (isWallMoveLegal(s, side, ttMoveVal.a, ttMoveVal.b, ttMoveVal.c)) ttTried = true;
        } else if (hasTTMove) {
            for (size_t i = 0; i < pawnMoves.size(); i++) if (pawnMoves[i] == ttMoveVal) { ttTried = true; break; }
        }
        int best = -SCORE_INF;
        Move bestMove = pawnMoves.empty() ? ttMoveVal : pawnMoves[0];
        bool haveMove = false, cutoff = false;
        auto tryMove = [&](const Move& m) -> bool {
            State ns = applyMove(s, m);
            int score = -negamax(ns, depth - 1, -beta, -alpha, stats);
            if (stopped) return true;
            if (!haveMove || score > best) { best = score; bestMove = m; haveMove = true; }
            alpha = std::max(alpha, score);
            if (alpha >= beta) { recordCutoff(m, ply, side, depth); return true; }
            return false;
        };
        if (ttTried) { cutoff = tryMove(ttMoveVal); if (stopped) return 0; }
        if (!cutoff) for (size_t i = 0; i < pawnMoves.size() && !cutoff; i++) {
            const Move& m = pawnMoves[i];
            if (ttTried && m == ttMoveVal) continue;
            cutoff = tryMove(m); if (stopped) return 0;
        }
        if (!cutoff) {
            MoveList wallMoves;
            legalWallMoves(s, side, wallMoves);
            orderWallMoves(wallMoves, ply, side, s);
            for (size_t i = 0; i < wallMoves.size() && !cutoff; i++) {
                const Move& m = wallMoves[i];
                if (ttTried && m == ttMoveVal) continue;
                cutoff = tryMove(m); if (stopped) return 0;
            }
        }
        TTFlag flag = (best <= alphaOrig) ? UPPER : (best >= beta ? LOWER : EXACT);
        store(s.hash, depth, best, flag, bestMove);
        return best;
    }
};

}} // namespace qr::old

// gera N posições fixas (partidas aleatórias curtas, seed fixa -- mesmo
// espírito de test_search_staging.cpp) pra comparação de nós/s.
static std::vector<State> fixedPositions(int n, int plyMin, int plyMax) {
    std::vector<State> out;
    std::mt19937 rng(12345);
    while ((int)out.size() < n) {
        State s = initialState();
        int targetPly = plyMin + (int)(rng() % (plyMax - plyMin + 1));
        for (int p = 0; p < targetPly; p++) {
            if (winner(s) != -1) break;
            MoveList moves = legalMoves(s);
            if (moves.empty()) break;
            s = applyMove(s, moves[rng() % moves.size()]);
        }
        if (winner(s) == -1) out.push_back(s);
    }
    return out;
}

static void benchNodeEfficiency() {
    printf("=== nos/s e profundidade -- 40 posicoes fixas, 200ms/lance ===\n");
    auto positions = fixedPositions(40, 5, 40);
    {
        Negamax engine; // ATUAL (com WALL_TOUCH_BONUS)
        uint64_t totalNodes = 0; int depthSum = 0;
        for (auto& s : positions) {
            SearchStats st;
            engine.chooseMove(s, 40, 200, st);
            totalNodes += st.nodes; depthSum += st.reachedDepth;
        }
        printf("ATUAL (com WALL_TOUCH_BONUS): nos totais=%llu, profundidade media=%.2f\n",
               (unsigned long long)totalNodes, depthSum / (double)positions.size());
    }
    {
        qr::old::Negamax engine; // ANTIGA (sem o bonus)
        uint64_t totalNodes = 0; int depthSum = 0;
        for (auto& s : positions) {
            SearchStats st;
            engine.chooseMove(s, 40, 200, st);
            totalNodes += st.nodes; depthSum += st.reachedDepth;
        }
        printf("ANTIGA (sem bonus):           nos totais=%llu, profundidade media=%.2f\n",
               (unsigned long long)totalNodes, depthSum / (double)positions.size());
    }
}

// partida completa entre as duas versoes; engineForTurn0 joga com o
// jogador 0, engineForTurn1 com o jogador 1.
template <typename E0, typename E1>
static int playGame(E0& e0, E1& e1, int movetimeMs, int maxPlies) {
    State s = initialState();
    for (int ply = 0; ply < maxPlies; ply++) {
        int w = winner(s);
        if (w != -1) return w;
        SearchStats st;
        Move m = (s.turn == 0) ? e0.chooseMove(s, 40, movetimeMs, st)
                                : e1.chooseMove(s, 40, movetimeMs, st);
        s = applyMove(s, m);
    }
    return -1; // nao terminou
}

static void benchHeadToHead() {
    printf("\n=== partidas diretas: ATUAL (com bonus) vs ANTIGA (sem bonus), 150ms/lance ===\n");
    int atualWins = 0, antigaWins = 0, indecisas = 0;
    const int NUM_GAMES = 6;
    for (int g = 0; g < NUM_GAMES; g++) {
        bool atualIsPlayer0 = (g % 2 == 0);
        auto t0 = clockT::now();
        int w;
        if (atualIsPlayer0) {
            Negamax atual; qr::old::Negamax antiga;
            w = playGame(atual, antiga, 150, 200);
        } else {
            qr::old::Negamax antiga; Negamax atual;
            w = playGame(antiga, atual, 150, 200);
        }
        double secs = msSince(t0) / 1000.0;
        int vencedorAtual = atualIsPlayer0 ? 0 : 1;
        if (w == -1) { indecisas++; printf("jogo %d: indecisa (%.1fs)\n", g, secs); }
        else if (w == vencedorAtual) { atualWins++; printf("jogo %d: ATUAL venceu (%.1fs)\n", g, secs); }
        else { antigaWins++; printf("jogo %d: ANTIGA venceu (%.1fs)\n", g, secs); }
    }
    printf("placar: ATUAL %d - %d ANTIGA (%d indecisas/%d jogos)\n", atualWins, antigaWins, indecisas, NUM_GAMES);
}

int main() {
    benchNodeEfficiency();
    benchHeadToHead();
    return 0;
}
