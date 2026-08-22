// bench_nodecap -- mede a distribuicao de nos MCTS por lance na busca de
// producao (MCAB+NNUE, 150ms/lance) para verificar se o nodeBudget
// (default 20000) esta truncando posicoes complexas.
#include <cstdio>
#include <algorithm>
#include <chrono>
#include <random>
#include <vector>
#include "rules.hpp"
#include "search.hpp"
#include "../src/mcab.hpp"

using namespace qr;

int main() {
    loadWeightsQuant("data/nnue/nnue_weights_int8.bin");
    std::mt19937 rng(4242);
    std::vector<long long> counts;
    for (int g = 0; g < 12; g++) {
        State s = initialState();
        RepetitionTable hist;
        hist.markRoot();
        for (int p = 0; p < 10 + 6 * g; p++) {
            if (winner(s) != -1) break;
            auto ms_ = legalMoves(s);
            MoveList candPawn;
            for (size_t i = 0; i < ms_.size(); i++) if (!ms_[i].isWall) candPawn.push_back(ms_[i]);
            const MoveList& src = !candPawn.empty() ? candPawn : ms_;
            s = applyMove(s, src[rng() % src.size()]);
            hist.push(s.hash, false);
        }
        // joga ~14 lances reais de producao nesta partida
        Negamax eng;
        eng.setEvalMode(Negamax::EvalMode::NNUE);
        eng.setPolicyOrderingEnabled(true);
        mcab::McabRunner<Negamax, State, Move, MoveList, AccPair, RepetitionTable, SearchStats> runner;
        for (int mv = 0; mv < 14; mv++) {
            if (winner(s) != -1) break;
            SearchStats st;
            mcab::McabStats mst;
            Move m = runner.choose(eng, s, 40, 150, st, hist, &mst);
            counts.push_back(mst.nodesExpanded);
            hist.push(s.hash, false);
            s = applyMove(s, m);
        }
    }
    std::sort(counts.begin(), counts.end());
    auto pct = [&](double p) { return counts[std::min(counts.size() - 1, (size_t)(p * counts.size()))]; };
    long long capped = 0;
    for (long long c : counts) if (c >= 19900) capped++;
    printf("lances=%d media=%.0f mediana=%lld p90=%lld p99=%lld MAX=%lld\n",
           (int)counts.size(), [&]{ double t = 0; for (long long c : counts) t += c; return t / counts.size(); }(),
           pct(0.5), pct(0.9), pct(0.99), counts.back());
    printf("lances no cap (>=19900): %lld/%d (%.1f%%)\n", capped, (int)counts.size(),
           100.0 * capped / counts.size());
    return 0;
}
