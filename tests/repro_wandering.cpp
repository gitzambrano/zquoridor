// repro_wandering.cpp -- reproduces the reported wall-poor wandering position.
#include <cstdio>
#include <string>
#include <array>
#include <functional>
#include <cmath>
#include "../src/rules.hpp"
#include "../src/search.hpp"
#include "../src/nnue.hpp"
#include "../src/mcab.hpp"
#include "../src/endgame_race.hpp"

using namespace qr;

static State buildPosition() {
    State s;
    s.pawn[0] = cellIdx(4, 6);
    s.pawn[1] = cellIdx(8, 0);
    auto addH = [&](int r, int c) { s.wallsH |= 1ull << slotIdx(r, c); };
    auto addV = [&](int r, int c) { s.wallsV |= 1ull << slotIdx(r, c); };
    addH(0, 0); addH(0, 2); addH(0, 4);
    addH(1, 1); addH(1, 3); addH(1, 5); addH(1, 7);
    addH(6, 6);
    addH(7, 1); addH(7, 3); addH(7, 5);
    addV(2, 0); addV(4, 0); addV(6, 0); addV(7, 7);
    s.wallsLeft[0] = 0;
    s.wallsLeft[1] = 5;
    s.turn = 0;
    Zobrist& z = zobrist();
    s.hash = 0;
    s.hash ^= z.pawnKey[0][s.pawn[0]] ^ z.pawnKey[1][s.pawn[1]];
    for (int i = 0; i < WS * WS; i++) {
        if ((s.wallsH >> i) & 1ull) s.hash ^= z.wallHKey[i];
        if ((s.wallsV >> i) & 1ull) s.hash ^= z.wallVKey[i];
    }
    return s;
}

static std::string moveStr(const Move& m) {
    char buf[32];
    if (!m.isWall) snprintf(buf, sizeof buf, "pawn(r%d,c%d)", rowOf(m.a), colOf(m.a));
    else snprintf(buf, sizeof buf, "wall(%s,r%d,c%d)", m.a == 0 ? "H" : "V", m.b, m.c);
    return buf;
}

static void simulate(const char* label, int mode) {
    State s = buildPosition();
    Negamax eng;
    RepetitionTable reptbl;
    reptbl.push(s.hash);
    if (mode >= 1) eng.setEvalMode(Negamax::EvalMode::NNUE);
    using McabRunnerT = mcab::McabRunner<qr::Negamax, qr::State, qr::Move, qr::MoveList,
                                         qr::AccPair, qr::RepetitionTable, qr::SearchStats>;
    McabRunnerT mcabRunner;

    printf("=== %s ===\n", label);
    for (int engineMoveIdx = 0; engineMoveIdx < 8; engineMoveIdx++) {
        if (winner(s) != -1) { printf("  winner=%d antes do lance %d\n", winner(s), engineMoveIdx); break; }
        SearchStats st;
        Move m;
        if (mode == 2) m = mcabRunner.choose(eng, s, 40, 500, st, reptbl);
        else m = eng.chooseMove(s, 40, 500, st, reptbl);
        int d0 = shortestPathLen(s.wallsH, s.wallsV, s.pawn[0], 0);
        int d1 = shortestPathLen(s.wallsH, s.wallsV, s.pawn[1], 1);
        printf("  mv%d: %-16s score=%6d depth=%2d nodes=%9lld | dist0=%d dist1=%d\n",
               engineMoveIdx, moveStr(m).c_str(), st.score, st.reachedDepth,
               (long long)st.nodes, d0, d1);
        s = applyMove(s, m);
        reptbl.push(s.hash);
        if (winner(s) != -1) { printf("  engine chegou! winner=%d\n", winner(s)); break; }
        MoveList lm = legalMoves(s);
        Move best = lm[0];
        int bestD = 99;
        for (size_t i = 0; i < lm.size(); i++) {
            if (lm[i].isWall) continue;
            State ns = applyMove(s, lm[i]);
            int d = shortestPathLen(ns.wallsH, ns.wallsV, ns.pawn[1], 1);
            if (d < bestD) { bestD = d; best = lm[i]; }
        }
        s = applyMove(s, best);
        reptbl.push(s.hash);
    }
    printf("\n");
}

static void rootMoveScores(Negamax::EvalMode mode, const char* label) {
    State s = buildPosition();
    Negamax eng;
    eng.setEvalMode(mode);
    printf("--- root move scores (%s), searchShallow depth 8 per child ---\n", label);
    MoveList lm = legalMoves(s);
    for (size_t i = 0; i < lm.size(); i++) {
        State ns = applyMove(s, lm[i]);
        SearchStats st;
        int sc = eng.searchShallow(ns, 7, st);
        int d0 = shortestPathLen(ns.wallsH, ns.wallsV, ns.pawn[0], 0);
        printf("  %-16s childScore=%6d | dist0=%d\n", moveStr(lm[i]).c_str(), sc, d0);
    }
    printf("\n");
}

static void dumpPriorsAndStaticEval() {
    State s = buildPosition();
    AccPair acc = buildAccPairRoot(s, nullptr);
    std::array<float, 209> pol{};
    forwardPolicyQuant(acc.acc[s.turn], pol);

    MoveList lm = legalMoves(s);
    float logits[209], maxLogit = -1e30f;
    for (size_t i = 0; i < lm.size(); i++) {
        uint16_t idx = moveToPolicyIndex(lm[i]);
        logits[i] = pol[idx];
        if (logits[i] > maxLogit) maxLogit = logits[i];
    }
    float sumExp = 0.f;
    for (size_t i = 0; i < lm.size(); i++) {
        logits[i] = std::exp(logits[i] - maxLogit);
        sumExp += logits[i];
    }

    printf("--- root policy prior and static NNUE eval ---\n");
    for (size_t i = 0; i < lm.size(); i++) {
        State ns = applyMove(s, lm[i]);
        AccPair childAcc = buildAccPairRoot(ns, nullptr);
        int ev = nnueEvalInt(childAcc, ns.turn);
        double q = 1.0 / (1.0 + std::exp(-(double)ev / 200.0));
        int d0 = shortestPathLen(ns.wallsH, ns.wallsV, ns.pawn[0], 0);
        printf("  %-16s P=%.3f evalStatic(op)=%6d Q=%.4f dist0=%d\n",
               moveStr(lm[i]).c_str(), logits[i] / sumExp, ev, q, d0);
    }
    printf("\n");
}

static void simulateVariant(const char* label, std::function<void(mcab::McabParams&)> tweak) {
    State s = buildPosition();
    Negamax eng;
    RepetitionTable reptbl;
    reptbl.push(s.hash);
    eng.setEvalMode(Negamax::EvalMode::NNUE);
    using McabRunnerT = mcab::McabRunner<qr::Negamax, qr::State, qr::Move, qr::MoveList,
                                         qr::AccPair, qr::RepetitionTable, qr::SearchStats>;
    McabRunnerT runner;
    mcab::McabParams p{};
    tweak(p);
    runner.setParams(p);

    printf("=== %s ===\n", label);
    for (int i = 0; i < 8; i++) {
        if (winner(s) != -1) break;
        SearchStats st;
        mcab::McabStats mst;
        Move m = runner.choose(eng, s, 40, 500, st, reptbl, &mst);
        int d0 = shortestPathLen(s.wallsH, s.wallsV, s.pawn[0], 0);
        printf("  mv%d: %-16s dist0=%d sims=%lld\n", i, moveStr(m).c_str(), d0, mst.simulations);
        s = applyMove(s, m);
        reptbl.push(s.hash);
        if (winner(s) != -1) { printf("  engine chegou! winner=0\n"); break; }
        MoveList lm = legalMoves(s);
        Move best = lm[0]; int bestD = 99;
        for (size_t k = 0; k < lm.size(); k++) {
            if (lm[k].isWall) continue;
            State ns = applyMove(s, lm[k]);
            int d = shortestPathLen(ns.wallsH, ns.wallsV, ns.pawn[1], 1);
            if (d < bestD) { bestD = d; best = lm[k]; }
        }
        s = applyMove(s, best);
        reptbl.push(s.hash);
    }
    printf("\n");
}

int main(int argc, char** argv) {
    State s = buildPosition();
    int d0 = shortestPathLen(s.wallsH, s.wallsV, s.pawn[0], 0);
    int d1 = shortestPathLen(s.wallsH, s.wallsV, s.pawn[1], 1);
    printf("position: dist0=%d dist1=%d wallsLeft=(%d,%d)\n\n", d0, d1, s.wallsLeft[0], s.wallsLeft[1]);

    bool nnue = argc > 1 && loadWeightsQuant(argv[1]);
    printf("nnue loaded: %s\n\n", nnue ? "yes" : "NO");

    rootMoveScores(Negamax::EvalMode::Heuristic, "heuristic");
    if (nnue) {
        rootMoveScores(Negamax::EvalMode::NNUE, "nnue");
        dumpPriorsAndStaticEval();
    }

    simulate("AB heuristic, 500ms/move", 0);
    if (!nnue) return 0;
    simulate("AB NNUE, 500ms/move", 1);
    simulate("production MCAB NNUE, 500ms/move", 2);
    simulateVariant("MCAB MaxQ", [](mcab::McabParams& p) { p.rootSelectMode = mcab::RootSelectMode::MaxQ; });
    simulateVariant("MCAB leafDepth=2", [](mcab::McabParams& p) { p.leafDepth = 2; });
    simulateVariant("MCAB AB prefilter depth=6 topK=4", [](mcab::McabParams& p) { p.abPrefilterDepth = 6; p.abPrefilterTopK = 4; });
    simulateVariant("MCAB fpuReduction=0.25", [](mcab::McabParams& p) { p.fpuReduction = 0.25; });
    return 0;
}
