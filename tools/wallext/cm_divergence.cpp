// cm_divergence.cpp -- how often does chooseMove itself diverge between
// default caps and the low-walls rule? The arena pilot showed identical
// node totals over 4 games; this finds out whether that is structural
// (aspiration + TT absorb leaf-value changes at shallow caps) or a bug.
#define QR_ENABLE_TEST_HOOKS
#include <cstdio>
#include <vector>
#include "rules.hpp"
#include "search.hpp"
#include "../tools/wallext/corpus.hpp"

using namespace qr;
using namespace wallext;

int main(int argc, char** argv) {
    int cap = argc > 1 ? atoi(argv[1]) : 8;
    std::vector<CorpusEntry> corpus = buildCorpus();
    int tried = 0, moveDiff = 0, nodeDiff = 0, scoreDiff = 0;
    long long nodesA = 0, nodesB = 0;
    for (size_t i = 0; i < corpus.size(); i += 11) {
        const State& s = corpus[i].s;
        Negamax a;
        SearchStats sa;
        Move ma = a.chooseMove(s, cap, 600000, sa);
        Negamax b;
        b.setQsLowWallsBonus(2);
        b.setQsLowWallsThreshold(6);
        SearchStats sb;
        Move mb = b.chooseMove(s, cap, 600000, sb);
        tried++;
        nodesA += sa.nodes;
        nodesB += sb.nodes;
        if (!(ma == mb)) {
            moveDiff++;
            if (moveDiff <= 8) {
                printf("pos %zu (total=%d): lance difere A(%d,%d,%d,%d) B(%d,%d,%d,%d) | nos %lld vs %lld\n",
                       i, s.wallsLeft[0] + s.wallsLeft[1],
                       ma.isWall, ma.a, ma.b, ma.c, mb.isWall, mb.a, mb.b, mb.c,
                       (long long)sa.nodes, (long long)sb.nodes);
            }
        }
        if (sb.nodes != sa.nodes) nodeDiff++;
        if (sb.score != sa.score) scoreDiff++;
    }
    printf("cap=%d: posicoes=%d lance_difere=%d nos_difere=%d score_difere=%d | nos totais A=%lld B=%lld\n",
           cap, tried, moveDiff, nodeDiff, scoreDiff, nodesA, nodesB);
    return 0;
}
