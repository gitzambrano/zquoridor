// test_search_staging.cpp -- validação de regressão da Fase 4.2.3
// (staged/lazy move generation, search.hpp): compara a busca nova
// (estagiada: TT -> peão -> muro sob demanda) contra uma cópia fiel da
// implementação MONOLÍTICA anterior (gera peão+muro juntos em todo nó,
// um único orderMoves combinado) em muitas posições aleatórias.
//
// Alpha-beta com janela cheia é invariante à ordem de geração/exploração
// dos lances -- o valor minimax retornado numa profundidade fixa não deve
// mudar só porque a árvore foi cortada mais cedo ou mais tarde por causa
// de estagiamento. Se a versão estagiada gerar o mesmo CONJUNTO de
// lances legais que a antiga (o que já é garantido por
// test_rules_sanity/legalMoves, inalterado nesta rodada) e a lógica de
// alpha-beta estiver correta, os dois devem concordar exatamente no score
// de raiz. Esse é o teste: não "parece mais rápido", e sim "responde
// exatamente igual".
#define QR_ENABLE_TEST_HOOKS  // habilita testFixedDepthFullWindow (search.hpp)
#include <cstdio>
#include <chrono>
#include <cstring>
#include <algorithm>
#include <random>
#include "rules.hpp"
#include "search.hpp"  // versão nova (estagiada) -- qr::Negamax; também traz
                        // as constantes/structs (TT_SIZE, SearchStats, ...)
                        // reaproveitadas pela cópia de referência abaixo.

namespace qr { namespace ref {

// --- cópia fiel do search.hpp de ANTES da Fase 4.2.3 -----------------
// (mesmas constantes/estrutura de TTEntry/SearchStats do search.hpp
// atual, só o corpo de negamax/orderMoves é a versão monolítica antiga)
using qr::TT_BITS;
using qr::TT_SIZE;
using qr::SCORE_INF;
using qr::MAX_PLY;
using qr::WALL_BFS_ORDER_MAX_PLY;
using qr::TTFlag;
using qr::EXACT;
using qr::LOWER;
using qr::UPPER;
using qr::TTEntry;
using qr::SearchStats;

class NegamaxReference {
public:
    NegamaxReference() : tt(TT_SIZE) {}

    int searchRoot(const State& root, int depth, SearchStats& stats) {
        stats = SearchStats{};
        stopped = false;
        rootDepth = depth;
        std::memset(killerValid, 0, sizeof(killerValid));
        std::memset(history, 0, sizeof(history));
        deadline = std::chrono::steady_clock::now() + std::chrono::hours(1); // sem limite de tempo neste teste
        return negamax(root, depth, -SCORE_INF, SCORE_INF, stats);
    }

private:
    std::vector<TTEntry> tt;
    std::chrono::steady_clock::time_point deadline;
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

    // Cópia fiel de Negamax::quiescence (search.hpp, Fase 4.2.10 item 3).
    // Necessária aqui porque este teste compara scores exatos entre a
    // busca estagiada e a monolítica -- sem essa cópia, a introdução da
    // quiescência em search.hpp faria os dois lados divergerem por causa
    // do item 3 (comportamento novo e esperado), não por causa de uma
    // regressão real do staging (Fase 4.2.3, o que este teste de fato
    // valida). Mantém o teste isolando só a variável que importa aqui.
    int quiescence(const State& s, int alpha, int beta, int qply, SearchStats& stats) {
        stats.nodes++;
        int w = winner(s);
        if (w != -1) return (w == s.turn) ? SCORE_INF - 1 : -(SCORE_INF - 1);

        int standPat = evalSimple(s, s.turn);
        if (standPat >= beta) return standPat;
        int localAlpha = alpha > standPat ? alpha : standPat;
        int best = standPat;

        if (qply >= QS_MAX_EXTRA_PLIES) return best;

        int side = s.turn, opp = 1 - side;
        if (s.wallsLeft[side] <= 0) return best;

        MoveList wallMoves;
        uint64_t touchH0, touchV0, touchH1, touchV1;
        legalWallMoves(s, side, wallMoves, &touchH0, &touchV0, &touchH1, &touchV1);
        uint64_t touchHOpp = (side == 0) ? touchH1 : touchH0;
        uint64_t touchVOpp = (side == 0) ? touchV1 : touchV0;

        int oppDistBefore = shortestPathLen(s.wallsH, s.wallsV, s.pawn[opp], opp);
        int oppRobustBefore = pathRobustness(s.wallsH, s.wallsV, s.pawn[opp], opp);

        for (size_t i = 0; i < wallMoves.size(); i++) {
            const Move& m = wallMoves[i];
            int slot = slotIdx(m.b, m.c);
            bool touches = (m.a == 0) ? ((touchHOpp >> slot) & 1ull) : ((touchVOpp >> slot) & 1ull);
            if (!touches) continue;

            State ns = applyMove(s, m);
            int oppDistAfter = shortestPathLen(ns.wallsH, ns.wallsV, ns.pawn[opp], opp);
            bool critical = (oppDistAfter - oppDistBefore) >= QS_CRITICAL_BFS_DELTA;
            if (!critical && oppRobustBefore > QS_CRITICAL_ROBUSTNESS_DROP_TO) {
                int oppRobustAfter = pathRobustness(ns.wallsH, ns.wallsV, ns.pawn[opp], opp);
                critical = (oppRobustAfter <= QS_CRITICAL_ROBUSTNESS_DROP_TO);
            }
            if (!critical) continue;

            int score = -quiescence(ns, -beta, -localAlpha, qply + 1, stats);
            if (score > best) best = score;
            if (score > localAlpha) localAlpha = score;
            if (localAlpha >= beta) break;
        }
        return best;
    }

    void recordCutoff(const Move& m, int ply, int side, int depth) {
        if (ply >= 0 && ply < MAX_PLY) {
            if (!(killerValid[ply][0] && killers[ply][0] == m)) {
                killers[ply][1] = killers[ply][0];
                killerValid[ply][1] = killerValid[ply][0];
                killers[ply][0] = m;
                killerValid[ply][0] = true;
            }
        }
        history[side][moveToPolicyIndex(m)] += depth * depth;
    }

    static constexpr size_t ORDER_BUF_CAP = 256;

    void orderMoves(MoveList& moves, const Move* ttMove, int ply, int side, const State& s) {
        bool ply0 = ply >= 0 && ply < MAX_PLY;
        bool wallByBFS = ply <= WALL_BFS_ORDER_MAX_PLY;
        int opp = 1 - side;
        size_t n = moves.size();
        static thread_local std::pair<long long, Move> buf[ORDER_BUF_CAP];
        for (size_t i = 0; i < n; i++) {
            const Move& m = moves[i];
            long long sc = 0;
            if (!m.isWall) {
                sc = 3'000'000;
            } else if (wallByBFS) {
                State ns = applyMove(s, m);
                int before = shortestPathLen(s.wallsH, s.wallsV, s.pawn[opp], opp);
                int after = shortestPathLen(ns.wallsH, ns.wallsV, ns.pawn[opp], opp);
                sc = 2'000'000 + (long long)(after - before) * 1000;
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
        if (ttMove) {
            auto it = std::find(moves.begin(), moves.end(), *ttMove);
            if (it != moves.end() && it != moves.begin()) std::iter_swap(moves.begin(), it);
        }
    }

    int negamax(const State& s, int depth, int alpha, int beta, SearchStats& stats) {
        stats.nodes++;
        int w = winner(s);
        if (w != -1) return (w == s.turn) ? SCORE_INF - 1 : -(SCORE_INF - 1);

        int alphaOrig = alpha;
        TTEntry& e = probe(s.hash);
        Move* ttMove = nullptr;
        if (e.valid && e.key == s.hash) {
            ttMove = &e.best;
            if (e.depth >= depth) {
                if (e.flag == EXACT) return e.score;
                if (e.flag == LOWER) alpha = std::max(alpha, (int)e.score);
                else if (e.flag == UPPER) beta = std::min(beta, (int)e.score);
                if (alpha >= beta) return e.score;
            }
        }

        if (depth == 0) return quiescence(s, alpha, beta, 0, stats);

        int ply = rootDepth - depth;

        auto moves = legalMoves(s);
        orderMoves(moves, ttMove, ply, s.turn, s);

        int best = -SCORE_INF;
        Move bestMove = moves[0];
        for (auto& m : moves) {
            State ns = applyMove(s, m);
            int score = -negamax(ns, depth - 1, -beta, -alpha, stats);
            if (score > best) { best = score; bestMove = m; }
            alpha = std::max(alpha, score);
            if (alpha >= beta) {
                recordCutoff(m, ply, s.turn, depth);
                break;
            }
        }

        TTFlag flag = (best <= alphaOrig) ? UPPER : (best >= beta ? LOWER : EXACT);
        store(s.hash, depth, best, flag, bestMove);
        return best;
    }
};

}} // namespace qr::ref

using namespace qr;

int main() {
    std::mt19937 rng(999);
    const int NUM_POSITIONS = 150;
    const int DEPTH = 4;  // full-window, sem iterative deepening -- score deve ser idêntico

    int checked = 0;
    long long nodesNew = 0, nodesRef = 0;

    for (int game = 0; game < NUM_POSITIONS; game++) {
        // percorre uma partida aleatória e testa uma posição por partida,
        // em ply variável (0..~40) -- cobre early/mid/endgame, densidade
        // de muro crescente conforme o plano (mesmo espírito de
        // testWallDsuCornerPocketRegression em test_rules_sanity.cpp).
        State s = initialState();
        int stopPly = 5 + (int)(rng() % 35);
        bool reachedEnd = false;
        for (int ply = 0; ply < stopPly; ply++) {
            auto moves = legalMoves(s);
            if (winner(s) != -1 || moves.empty()) { reachedEnd = true; break; }
            std::uniform_int_distribution<size_t> dist(0, moves.size() - 1);
            s = applyMove(s, moves[dist(rng)]);
        }
        if (reachedEnd || winner(s) != -1) continue;

        // Comparação via busca DIRETA de profundidade fixa / janela cheia
        // (testFixedDepthFullWindow, search.hpp), não via chooseMove.
        //
        // Motivo da troca (achado desta rodada, Fase 4.2.10): chooseMove
        // roda iterative deepening com aspiration windows e um TT que
        // PERSISTE entre as iterações de profundidade 1..DEPTH. Com
        // evalSimple puro (função sem estado, independente de janela
        // alpha-beta) isso nunca importava para o VALOR final -- um nó
        // folha sempre devolvia o mesmo número não importa a janela com
        // que fosse alcançado, então a reconvergência do aspiration
        // sempre batia com uma busca de janela cheia. A partir de agora
        // (item 3, quiescência de muro) a folha deixou de ser uma função
        // pura -- quiescence(s,alpha,beta) é ela mesma uma busca
        // alpha-beta fail-soft, e o Estágio de TT-entre-iterações do
        // chooseMove passou a produzir, em casos raros, um score de
        // depth=DEPTH que diverge por 1 ou poucos pontos do valor exato
        // de uma busca isolada -- confirmado isolando quiescence() sozinha
        // (0 divergências em 300 posições, janela cheia e estreita) e
        // comparando staging vs monolítico via busca direta de
        // profundidade fixa (0 divergências em 150 posições, ver
        // sessão de depuração). Ou seja: staging (4.2.3) e quiescência
        // (4.2.10 item 3) provadamente CONTINUAM corretos; o que não é
        // mais válido é usar chooseMove como oráculo de "valor exato" pra
        // este teste específico -- isso é uma fragilidade pré-existente
        // do aspiration+TT-entre-iterações (não deste teste, nem de
        // staging/quiescência) que fica registrada como pendência de
        // investigação separada (ver readme, Seção 4.2.10).
        Negamax fresh;
        SearchStats statsNew;
        int scoreNew = fresh.testFixedDepthFullWindow(s, DEPTH, statsNew);

        ref::NegamaxReference refEngine;
        SearchStats statsRef;
        int scoreRef = refEngine.searchRoot(s, DEPTH, statsRef);

        checked++;
        nodesNew += statsNew.nodes;
        nodesRef += statsRef.nodes;

        if (scoreNew != scoreRef) {
            printf("FALHOU: divergencia de score na posicao %d (stopPly=%d): "
                   "estagiado=%d referencia=%d\n", game, stopPly, scoreNew, scoreRef);
            return 1;
        }

        // smoke check separado (nao usado para comparacao de score): o
        // lance escolhido por chooseMove (com aspiration/iterative
        // deepening de verdade) tem que ser sempre um lance legal, mesmo
        // que o *score* que ele reporta tenha a fragilidade descrita
        // acima -- guarda contra bug de staging que jogasse um lance
        // "fantasma".
        SearchStats statsChoose;
        Move mv = fresh.chooseMove(s, DEPTH, 200, statsChoose);
        auto legal = legalMoves(s);
        bool mvLegal = std::find(legal.begin(), legal.end(), mv) != legal.end();
        if (!mvLegal) {
            printf("FALHOU: chooseMove (estagiado) devolveu lance ilegal na posicao %d\n", game);
            return 1;
        }
    }

    printf("staged vs referencia monolitica: %d posicoes comparadas, 0 divergencias de score\n", checked);
    printf("nos totais -- estagiado: %lld, referencia: %lld (razao %.3fx)\n",
           nodesNew, nodesRef, nodesRef > 0 ? (double)nodesNew / nodesRef : 0.0);

    // sanity extra: garante que winner()/negamax lidam bem com posicoes
    // proximas da meta (staging nao deveria mudar deteccao de vitoria)
    {
        State s = initialState();
        s.pawn[0] = cellIdx(N - 2, N / 2);
        Negamax eng;
        SearchStats st;
        Move mv = eng.chooseMove(s, 3, 500, st);
        (void)mv;
        printf("posicao perto da meta: busca estagiada roda sem crash, profundidade alcancada=%d\n", st.reachedDepth);
    }

    printf("TODOS OS TESTES DE STAGING PASSARAM\n");
    return 0;
}
