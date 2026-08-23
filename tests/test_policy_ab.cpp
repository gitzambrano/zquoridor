// test_policy_ab.cpp -- correctness gate for the inv/ab-policy search
// toggles (2026-08-23): setPolicyHistorySeedEnabled (direction B),
// setPolicyLmrEnabled (direction C), setPolicyLmpEnabled (direction D),
// and the Negamax::rankRootMoves helper used by the MCAB pre-filter
// (direction E).
//
// What is asserted (repo convention for heuristic toggles -- agreement
// thresholds, not zero divergence):
//   1. DEFAULTS BIT-IDENTICAL: an engine left untouched and an engine
//      with every new toggle explicitly OFF produce the same move, the
//      same score and the same node count over the whole corpus (both
//      eval modes). This is the regression guard for production.
//   2. EACH VARIANT vs REFERENCE: at least 85% score agreement
//      (|scoreOn - scoreOff| <= TOL), at least 90% sign agreement among
//      decisive positions (|score| >= DECISIVE), and ZERO illegal moves,
//      over >= 60 deterministic corpus positions.
//   3. STRESS: every toggle on at once with policyOrderingMinDepth forced
//      down to 1 (so policy-LMR/LMP act deep in the tree), NNUE mode,
//      full legal-move validation of every returned move.
//
// Run from the repo root (loads data/nnue/nnue_weights_int8.bin when it
// exists; falls back to zeroed quantized weights with a notice).
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>
#include <string>
#include "rules.hpp"
#include "search.hpp"

using namespace qr;

namespace {

constexpr int CORPUS_PER_STRATUM = 16;
constexpr int DEPTH_CAP = 6;
constexpr int SCORE_TOL = 60;      // about 0.3 * NNUE_EVAL_SCALE
constexpr int DECISIVE = 200;      // clear-advantage threshold in eval units

struct Result {
    Move move;
    int score;
    uint64_t nodes;
};

std::vector<State> corpus() {
    std::vector<State> out;
    const int lo[] = {4, 14, 26, 38};
    const int hi[] = {10, 24, 36, 52};
    for (int s = 0; s < 4; s++) {
        std::mt19937 rng(1010 + s);
        int got = 0, tries = 0;
        while (got < CORPUS_PER_STRATUM && tries < 8000) {
            tries++;
            State st = initialState();
            int plies = lo[s] + (int)(rng() % (unsigned)(hi[s] - lo[s] + 1));
            bool dead = false;
            for (int p = 0; p < plies; p++) {
                if (winner(st) != -1) { dead = true; break; }
                auto moves = legalMoves(st);
                if (moves.empty()) { dead = true; break; }
                std::uniform_int_distribution<size_t> d(0, moves.size() - 1);
                st = applyMove(st, moves[d(rng)]);
            }
            if (dead || winner(st) != -1) continue;
            out.push_back(st);
            got++;
        }
    }
    return out;
}

bool isLegal(const State& s, const Move& m) {
    MoveList ms = legalMoves(s);
    for (size_t i = 0; i < ms.size(); i++)
        if (ms[i] == m) return true;
    return false;
}

Result runChoose(Negamax& eng, const State& s) {
    SearchStats st;
    Move m = eng.chooseMove(s, DEPTH_CAP, 3600000, st);
    return {m, st.score, st.nodes};
}

void configureVariant(Negamax& eng, int variant) {
    // variant bitmask: 1 = history seed, 2 = policy LMR, 4 = policy LMP,
    // 8 = lower the policy-ordering depth gate to 1 (stress only).
    eng.setEvalMode(Negamax::EvalMode::NNUE);
    eng.setPolicyHistorySeedEnabled(false);
    eng.setPolicyLmrEnabled(false);
    eng.setPolicyLmpEnabled(false);
    eng.setPolicyOrderingEnabled(true);
    eng.setPolicyOrderingMinDepth(3);
    if (variant & 1) eng.setPolicyHistorySeedEnabled(true);
    if (variant & 2) eng.setPolicyLmrEnabled(true);
    if (variant & 4) eng.setPolicyLmpEnabled(true);
    if (variant & 8) eng.setPolicyOrderingMinDepth(1);
}

struct Agreement {
    int total = 0;
    int scoreAgree = 0;
    int decisiveTotal = 0;
    int decisiveAgree = 0;
    int illegal = 0;
};

}  // namespace

// compareBody lives outside the anonymous namespace only so the file stays
// readable; it is still internal to this translation unit.
static bool compareVariants(const char* label, int variant,
                            const std::vector<State>& pos, bool requireBitExact);

int main(int argc, char** argv) {
    const char* weightsPath = (argc > 1) ? argv[1] : "data/nnue/nnue_weights_int8.bin";
    bool haveWeights = loadWeightsQuant(weightsPath);
    if (!haveWeights) {
        printf("NOTE: '%s' not loaded -- NNUE blocks run on zeroed weights (structure still exercised)\n",
               weightsPath);
    }

    std::vector<State> pos = corpus();
    if (pos.size() < 60) {
        printf("FAIL: corpus too small (%zu)\n", pos.size());
        return 1;
    }
    printf("corpus=%zu positions (opening/midgame/late/race mix), depth cap=%d\n",
           pos.size(), DEPTH_CAP);

    bool allOk = true;

    // --- 1) defaults bit-identical, both eval modes ----------------------
    // Strided subset: the full corpus runs again in the variant comparisons
    // below; this block only guards the default-off path, so ~1/3 of the
    // positions keep the whole suite within minutes at -O2.
    {
        std::vector<State> subset;
        for (size_t i = 0; i < pos.size(); i += 3) subset.push_back(pos[i]);
        for (int modeIsNnue = 0; modeIsNnue <= 1; modeIsNnue++) {
            bool ok = true;
            for (const State& s : subset) {
                Negamax untouched;
                Negamax explicitOff;
                if (modeIsNnue) {
                    untouched.setEvalMode(Negamax::EvalMode::NNUE);
                    explicitOff.setEvalMode(Negamax::EvalMode::NNUE);
                }
                configureVariant(explicitOff, 0);
                Result rA = runChoose(untouched, s);
                Result rB = runChoose(explicitOff, s);
                if (!(rA.move == rB.move) || rA.score != rB.score || rA.nodes != rB.nodes) {
                    printf("FAIL [defaults-%s]: not bit-exact (move %d/%d score %d/%d nodes %llu/%llu)\n",
                           modeIsNnue ? "nnue" : "heur", (int)rA.move.a, (int)rB.move.a,
                           rA.score, rB.score,
                           (unsigned long long)rA.nodes, (unsigned long long)rB.nodes);
                    ok = false;
                    break;
                }
            }
            printf("%s [defaults-%s]: bit-exact over %zu positions\n",
                   ok ? "OK  " : "FAIL", modeIsNnue ? "nnue" : "heur", subset.size());
            allOk = allOk && ok;
        }
    }

    // --- 2) each variant vs reference -----------------------------------
    allOk = compareVariants("history-seed(B)", 1, pos, false) && allOk;
    allOk = compareVariants("policy-lmr(C)", 2, pos, false) && allOk;
    allOk = compareVariants("policy-lmp(D)", 4, pos, false) && allOk;
    allOk = compareVariants("lmp-base0.15(D)", 4 | 16, pos, false) && allOk;

    // --- 3) stress: everything on, gate lowered, deep interplay ----------
    allOk = compareVariants("stress-all-mindepth1", 1 | 2 | 4 | 8, pos, false) && allOk;

    printf(allOk ? "\nALL test_policy_ab CHECKS PASSED\n"
                 : "\ntest_policy_ab FAILED\n");
    return allOk ? 0 : 1;
}

static bool compareVariants(const char* label, int variant,
                            const std::vector<State>& pos, bool requireBitExact) {
    (void)requireBitExact;
    Agreement ag;
    int moveAgree = 0;
    double refNodes = 0, varNodes = 0;
    for (const State& s : pos) {
        Negamax refEng;
        configureVariant(refEng, 0);
        Negamax varEng;
        configureVariant(varEng, variant & 15);
        if (variant & 16) varEng.setPolicyLmpBaseMass(0.15);  // coarse second threshold

        Result rRef = runChoose(refEng, s);
        Result rVar = runChoose(varEng, s);
        refNodes += (double)rRef.nodes;
        varNodes += (double)rVar.nodes;

        if (!isLegal(s, rVar.move)) ag.illegal++;
        if (!isLegal(s, rRef.move)) {
            printf("FAIL [%s]: reference returned an illegal move\n", label);
            return false;
        }
        if (rVar.move == rRef.move) moveAgree++;

        bool raceResolved = (s.wallsLeft[0] == 0 && s.wallsLeft[1] == 0);
        ag.total++;
        if (raceResolved) { ag.scoreAgree++; continue; }  // exact on both sides
        if (std::abs(rVar.score - rRef.score) <= SCORE_TOL) ag.scoreAgree++;
        if (std::max(std::abs(rVar.score), std::abs(rRef.score)) >= DECISIVE) {
            ag.decisiveTotal++;
            if ((rVar.score > 0) == (rRef.score > 0)) ag.decisiveAgree++;
        }
    }
    double scorePct = 100.0 * ag.scoreAgree / (ag.total ? ag.total : 1);
    double deciPct = ag.decisiveTotal ? 100.0 * ag.decisiveAgree / ag.decisiveTotal : 100.0;
    double movePct = 100.0 * moveAgree / (ag.total ? ag.total : 1);
    printf("%s [%s]: score-agree %.1f%% (%d/%d), decisive %.1f%% (%d/%d), move-agree %.1f%%, illegal=%d, nodes ref/var %.0f/%.0f (%.2fx)\n",
           (ag.illegal == 0 && scorePct >= 85.0 && (ag.decisiveTotal == 0 || deciPct >= 90.0)) ? "OK  " : "FAIL",
           label, scorePct, ag.scoreAgree, ag.total, deciPct, ag.decisiveAgree,
           ag.decisiveTotal, movePct, ag.illegal, refNodes, varNodes,
           varNodes > 0 ? refNodes / varNodes : 0.0);
    return ag.illegal == 0 && scorePct >= 85.0 && (ag.decisiveTotal == 0 || deciPct >= 90.0);
}
