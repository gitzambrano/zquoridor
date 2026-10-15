// Controlled TT poisoning test: a hypothetical descendant repetition cannot
// be stored as an exact or bound result at an ancestor state.
#define QR_ENABLE_TEST_HOOKS
#include <cstdio>
#include "search.hpp"
using namespace qr;
int main() {
#if !ZQ_EXP_TT_REPETITION_GUARD
    std::fprintf(stderr, "test requires ZQ_EXP_TT_REPETITION_GUARD=1\n");
    return 3;
#endif
    State root = initialState();
    const MoveList moves = legalMoves(root);
    if (moves.empty()) return 4;
    const State repeatingChild = applyMove(root, moves[0]);

    // The ancestor is fresh, but a child already appeared twice in the
    // hypothetical history. Descending into it yields a repetition draw.
    RepetitionTable history;
    history.push(repeatingChild.hash);
    history.push(repeatingChild.hash);
    history.markRoot();
    if (history.isRepetitionDraw(root.hash) ||
        !history.isRepetitionDraw(repeatingChild.hash)) return 5;

    Negamax engine;
    engine.setQuiescenceEnabled(false);
    SearchStats stats;
    engine.searchLeaf(root, 1, stats, history);
    if (engine.testHasTTEntry(root.hash)) {
        std::fprintf(stderr, "FAILED: path-dependent root stored in TT\n");
        return 1;
    }

    RepetitionTable clean;
    clean.markRoot();
    const int reused = engine.searchLeaf(root, 1, stats, clean);
    if (!engine.testHasTTEntry(root.hash)) {
        std::fprintf(stderr, "FAILED: history-independent result not cached\n");
        return 2;
    }

    Negamax fresh;
    fresh.setQuiescenceEnabled(false);
    const int reference = fresh.searchLeaf(root, 1, stats, clean);
    if (reused != reference) {
        std::fprintf(stderr, "FAILED: clean search differs after tainted search: %d != %d\n",reused,reference);
        return 6;
    }
    std::printf("TT repetition guard PASS: tainted TT suppressed, clean TT admitted, scores match\n");
    return 0;
}
