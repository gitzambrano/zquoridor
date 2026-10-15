// Differential oracle: original queue BFS vs distance-only bit-parallel flood.
#include <cstdio>
#include <cstdint>
#include <random>
#include "rules.hpp"
using namespace qr;
int main() {
    std::mt19937 rng(20261014u);
    long long checks = 0, games = 0, walls = 0;
    for (int game = 0; game < 38; ++game) {
        State s = initialState();
        for (int ply = 0; ply < 48 && winner(s) == -1; ++ply) {
            ++games;
            for (int player = 0; player < 2; ++player) {
                detail::runBFS(s.wallsH, s.wallsV, s.pawn[player], player);
                const int expected = detail::engine().distToGoal;
                const int result = shortestPathLenBitboard(s.wallsH, s.wallsV, s.pawn[player], player);
                if (result != expected ||
                    shortestPathLen(s.wallsH, s.wallsV, s.pawn[player], player) != expected) {
                    std::fprintf(stderr,"Bitboard distance mismatch game=%d ply=%d player=%d expected=%d got=%d\n",game,ply,player,expected,result);
                    return 1;
                }
                ++checks;
            }
            MoveList lm = legalMoves(s);
            // Exercise real after-wall states, including dense walls.
            int tested = 0;
            for (const Move& m : lm) {
                if (!m.isWall || ((int)slotIdx(m.b,m.c) % 11 != ply % 11)) continue;
                State ns = applyMove(s,m);
                for (int player = 0; player < 2; ++player) {
                    detail::runBFS(ns.wallsH,ns.wallsV,ns.pawn[player],player);
                    int expected = detail::engine().distToGoal;
                    int got = shortestPathLenBitboard(ns.wallsH,ns.wallsV,ns.pawn[player],player);
                    if (expected != got) {
                        std::fprintf(stderr,"Post-wall mismatch game=%d ply=%d player=%d expected=%d got=%d\n",game,ply,player,expected,got);
                        return 2;
                    }
                    ++checks;
                }
                ++walls;
                if (++tested >= 12) break;
            }
            if (lm.empty()) break;
            int idx = (int)(rng() % lm.size());
            if (ply % 4 == 0) {
                for (size_t i=0;i<lm.size();++i) if (lm[i].isWall) {idx=(int)i;break;}
            }
            s = applyMove(s, lm[(size_t)idx]);
        }
    }
    std::printf("Bitboard distance oracle PASS: positions=%lld, wall candidates=%lld, player checks=%lld, mismatches=0\n",games,walls,checks);
    return checks > 1000 && walls > 100 ? 0 : 3;
}
