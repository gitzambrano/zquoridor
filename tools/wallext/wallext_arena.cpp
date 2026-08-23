// wallext_arena.cpp -- single-process pairwise strength matches between
// two configurations of the SAME binary (inv/qsendgame-ext, 2026-08).
//
// Engine A (baseline) always runs the production defaults. Engine B (the
// candidate) runs the same search with the wall-quiescence extension
// knobs overridden on the command line. Both play pure alpha-beta
// chooseMove, because the feature under test lives in the quiescence of
// the plain AB path (MCAB leaves bypass it at leafDepth 0).
//
// Games START from corpus positions (low-wall endgames) with colors
// swapped between the two games of a pair, so every position contributes
// once per side. This concentrates the sample where the rule fires.
//
// Build (from the repo root):
//   g++ -O3 -std=c++17 -march=native -mavx2 -mfma -Isrc
//       -o bin\wallext_arena.exe tools\wallext\wallext_arena.cpp
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <random>
#include <vector>
#include "rules.hpp"
#include "search.hpp"
#include "corpus.hpp"
#include "elo.hpp"

using namespace qr;
using namespace wallext;
using std::vector;

static int g_bBonus = -1;
static int g_bThreshold = -1;
static int g_bMaxExtra = -1;

struct SideStats { uint64_t nodes = 0; double ms = 0.0; };

// One game. Returns 1 if B wins, -1 if A wins, 0 on draw.
static int playGame(const State& start, bool bIsWhite, int depthCap, int timeMs,
                    bool nnueMode, uint64_t& pliesOut, SideStats& stA, SideStats& stB) {
    Negamax engA;
    Negamax engB;
    if (nnueMode) {
        engA.setEvalMode(Negamax::EvalMode::NNUE);
        engB.setEvalMode(Negamax::EvalMode::NNUE);
    }
    if (g_bMaxExtra != -1) engB.setQsMaxExtraPlies(g_bMaxExtra);
    if (g_bBonus != -1) engB.setQsLowWallsBonus(g_bBonus);
    if (g_bThreshold != -1) engB.setQsLowWallsThreshold(g_bThreshold);

    State s = start;
    RepetitionTable histA, histB, realHistory;

    int result = 0;
    const int MAX_PLIES = 300;
    for (int ply = 0; ply < MAX_PLIES; ply++) {
        int w = winner(s);
        if (w != -1) {
            result = ((w == 0) == bIsWhite) ? 1 : -1;
            break;
        }
        if (realHistory.count(s.hash) >= 2) { result = 0; break; }

        bool bToMove = (s.turn == 0) ? bIsWhite : !bIsWhite;
        SearchStats st;
        auto t0 = std::chrono::steady_clock::now();
        Move m = bToMove ? engB.chooseMove(s, depthCap, timeMs, st, histB)
                         : engA.chooseMove(s, depthCap, timeMs, st, histA);
        double ms = std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - t0).count();
        if (bToMove) { stB.nodes += st.nodes; stB.ms += ms; }
        else { stA.nodes += st.nodes; stA.ms += ms; }
        pliesOut = (uint64_t)(ply + 1);

        // Same bookkeeping order as tools/arena/arena.cpp: record the hash
        // of the position BEFORE the move, then advance.
        realHistory.push(s.hash, m.isWall);
        histA.push(s.hash, m.isWall);
        histB.push(s.hash, m.isWall);
        s = applyMove(s, m);
        if (ply == MAX_PLIES - 1) result = 0;  // cap hit without a decision
    }
    return result;
}

int main(int argc, char** argv) {
    int games = 200;
    int depthCap = 8;
    int timeMs = 600000;
    bool timeMode = false;
    bool nnueMode = false;
    const char* weightsPath = "data/nnue/nnue_weights_int8.bin";
    const char* outPath = "";
    int startOffset = 0;
    int reportEvery = 25;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--games") && i + 1 < argc) games = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--depth") && i + 1 < argc) { depthCap = atoi(argv[++i]); timeMode = false; }
        else if (!strcmp(argv[i], "--time-ms") && i + 1 < argc) { timeMs = atoi(argv[++i]); timeMode = true; }
        else if (!strcmp(argv[i], "--mode") && i + 1 < argc) nnueMode = (strcmp(argv[++i], "nnue") == 0);
        else if (!strcmp(argv[i], "--weights") && i + 1 < argc) weightsPath = argv[++i];
        else if (!strcmp(argv[i], "--out") && i + 1 < argc) outPath = argv[++i];
        else if (!strcmp(argv[i], "--start") && i + 1 < argc) startOffset = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--report-every") && i + 1 < argc) reportEvery = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--bonus") && i + 1 < argc) g_bBonus = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--threshold") && i + 1 < argc) g_bThreshold = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--max-extra") && i + 1 < argc) g_bMaxExtra = atoi(argv[++i]);
        else { printf("argumento desconhecido: %s\n", argv[i]); return 1; }
    }

    printf("wallext_arena: A=baseline vs B(bonus=%d thr=%d maxExtra=%d) jogos=%d %s=%d modo=%s start=%d\n",
           g_bBonus, g_bThreshold, g_bMaxExtra, games,
           timeMode ? "timeMs" : "depth", timeMode ? timeMs : depthCap,
           nnueMode ? "nnue" : "heuristic", startOffset);

    if (nnueMode && !loadWeightsQuant(weightsPath)) {
        printf("FALHOU: nao carregou pesos NNUE de %s\n", weightsPath);
        return 1;
    }

    std::vector<CorpusEntry> corpus = buildCorpus();
    printf("corpus: %zu posicoes de inicio\n", corpus.size());

    int winsB = 0, winsA = 0, draws = 0;
    double eloDiff = 0.0, margin = 0.0;
    FILE* out = outPath[0] ? fopen(outPath, "a") : nullptr;
    uint64_t totalPlies = 0;
    SideStats stA, stB;

    for (int g = 0; g < games; g++) {
        // The corpus index advances once per PAIR of games: both colors
        // see the same starting position before the next one comes up.
        size_t idx = (size_t)((startOffset + g / 2) % corpus.size());
        bool bIsWhite = (g % 2 == 0);
        uint64_t plies = 0;
        int r = playGame(corpus[idx].s, bIsWhite, depthCap, timeMs, nnueMode, plies, stA, stB);
        totalPlies += plies;
        if (r > 0) winsB++;
        else if (r < 0) winsA++;
        else draws++;

        if ((g + 1) % reportEvery == 0 || g + 1 == games) {
            wallextEloWithMargin(winsB, winsA, draws, eloDiff, margin);
            printf("[%4d/%4d] B:%d A:%d E:%d | Elo(B-A) %+7.1f +- %.1f | plies/jogo %.1f\n",
                   g + 1, games, winsB, winsA, draws, eloDiff, margin,
                   (double)totalPlies / (g + 1));
            fflush(stdout);
        }
    }

    wallextEloWithMargin(winsB, winsA, draws, eloDiff, margin);
    double npsA = stA.ms > 0 ? stA.nodes / (stA.ms / 1000.0) : 0.0;
    double npsB = stB.ms > 0 ? stB.nodes / (stB.ms / 1000.0) : 0.0;
    printf("\n=== resultado final ===\n");
    printf("vitorias B(variante)=%d A(baseline)=%d empates=%d em %d jogos\n",
           winsB, winsA, draws, winsA + winsB + draws);
    printf("Elo(B-A) = %+.1f (margem 95%% +- %.1f)\n", eloDiff, margin);
    printf("nos/s medios: A=%.0f B=%.0f | nos totais A=%lld B=%lld\n",
           npsA, npsB, (long long)stA.nodes, (long long)stB.nodes);

    if (out) {
        fprintf(out, "bonus=%d threshold=%d maxExtra=%d games=%d depth=%d timeMs=%d mode=%s "
                     "winsB=%d winsA=%d draws=%d elo=%+.1f margin=%.1f\n",
                g_bBonus, g_bThreshold, g_bMaxExtra, games, depthCap, timeMs,
                nnueMode ? "nnue" : "heuristic", winsB, winsA, draws, eloDiff, margin);
        fclose(out);
    }
    return 0;
}
