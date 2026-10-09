// race_oracle.hpp -- independent brute-force reference solver for the
// empty-handed pawn race (test support header, not part of the engine).
//
// Purpose: cross-check src/endgame_race.hpp against ground truth written
// from scratch. Deliberate independence choices:
//   1. Successor generation is re-derived here from the pawn-step rules,
//      sharing only the primitive predicates of rules.hpp (edgeBlocked,
//      inBounds, cellIdx). No call into pawnStepMoves or any solver code.
//   2. Win sets come from a naive full-sweep fixpoint (Knaster-Tarski
//      least fixed point), not from the production CSR + predecessor-
//      counting retrograde BFS.
//   3. DTM comes from a separate Gauss-Seidel relaxation of the min/max
//      equations restricted to the win region, not from the production
//      BFS layers.
// A state is terminal when either pawn stands on its goal row; the game
// ends there regardless of the turn (same convention as rules.hpp
// winner() and as the production solver seeds). A state is a draw for the
// trichotomy when it belongs to neither win region: neither side can
// force its own win.
#pragma once
#include <array>
#include <cstdint>
#include <cassert>
#include "rules.hpp"

namespace race_oracle {

constexpr int NS = qr::N * qr::N;          // 81 cells
constexpr int NST = NS * NS * 2;           // 13122 states

inline int stateIdx(int p0, int p1, int t) { return (p0 * NS + p1) * 2 + t; }

// One successor of the state (p0, p1, t): the mover's new cell plus the
// flipped turn. Written from the movement rules, not from rules.hpp
// generation code: orthogonal step into a free neighbor; straight jump
// over the opponent when the landing edge is open; otherwise the two
// diagonal detours around the opponent, each requiring its own edge open.
struct SuccGen {
    std::array<int, 6> p0v, p1v, tv;
    int n = 0;

    SuccGen(uint64_t wallsH, uint64_t wallsV, int p0, int p1, int t) {
        static const int dr[4] = {-1, 1, 0, 0};
        static const int dc[4] = {0, 0, -1, 1};
        int me = (t == 0) ? p0 : p1;
        int opp = (t == 0) ? p1 : p0;
        int mr = qr::rowOf(me), mc = qr::colOf(me);
        for (int d = 0; d < 4; d++) {
            int r1 = mr + dr[d], c1 = mc + dc[d];
            if (!qr::inBounds(r1, c1)) continue;
            if (qr::edgeBlocked(wallsH, wallsV, mr, mc, r1, c1)) continue;
            int s1 = qr::cellIdx(r1, c1);
            if (s1 != opp) {
                push((t == 0) ? s1 : p0, (t == 0) ? p1 : s1, 1 - t);
                continue;
            }
            int r2 = r1 + dr[d], c2 = c1 + dc[d];
            if (qr::inBounds(r2, c2) && !qr::edgeBlocked(wallsH, wallsV, r1, c1, r2, c2)) {
                int dest = qr::cellIdx(r2, c2);
                push((t == 0) ? dest : p0, (t == 0) ? p1 : dest, 1 - t);
                continue;
            }
            int pdA = (d < 2) ? 2 : 0;
            int pdB = (d < 2) ? 3 : 1;
            const int pds[2] = {pdA, pdB};
            for (int pi = 0; pi < 2; pi++) {
                int pd = pds[pi];
                int rd = r1 + dr[pd], cd = c1 + dc[pd];
                if (!qr::inBounds(rd, cd)) continue;
                if (qr::edgeBlocked(wallsH, wallsV, r1, c1, rd, cd)) continue;
                int dest = qr::cellIdx(rd, cd);
                push((t == 0) ? dest : p0, (t == 0) ? p1 : dest, 1 - t);
            }
        }
    }

private:
    void push(int a, int b, int tt) {
        p0v[n] = a;
        p1v[n] = b;
        tv[n] = tt;
        n++;
    }
};

inline bool isTerminal(int p0, int p1) {
    return qr::rowOf(p0) == qr::GOAL_ROW[0] || qr::rowOf(p1) == qr::GOAL_ROW[1];
}

struct Table {
    uint64_t wallsH = 0, wallsV = 0;
    bool built = false;

    std::array<uint8_t, NST> win0, win1;
    std::array<int, NST> dtm0, dtm1;

    // Naive fixpoint for the win regions, then relaxation for the DTMs.
    void build(uint64_t wh, uint64_t wv) {
        wallsH = wh;
        wallsV = wv;
        win0.fill(0);
        win1.fill(0);

        // Seeds: a pawn on its goal row ends the game in its own favor,
        // whatever the turn (double-goal states cannot arise in play; if
        // synthesized, both sides claim it and query() prefers player 0,
        // mirroring the production lookup order).
        for (int a = 0; a < NS; a++) {
            for (int b = 0; b < NS; b++) {
                if (a == b) continue;
                for (int t = 0; t < 2; t++) {
                    int s = stateIdx(a, b, t);
                    if (qr::rowOf(a) == qr::GOAL_ROW[0]) { win0[s] = 1; dtm0[s] = 0; }
                    if (qr::rowOf(b) == qr::GOAL_ROW[1]) { win1[s] = 1; dtm1[s] = 0; }
                }
            }
        }

        // Full-sweep fixpoint. A state enters the win region of X when the
        // mover is X and one successor already entered (existential), or
        // the mover is the opponent and every successor entered
        // (universal). Terminal states and invalid states never enter by
        // sweeping; they only exist as seeds.
        constexpr int SWEEP_CAP = 40000;
        for (int sweep = 0; sweep < SWEEP_CAP; sweep++) {
            bool changed = false;
            for (int a = 0; a < NS; a++) {
                for (int b = 0; b < NS; b++) {
                    if (a == b || isTerminal(a, b)) continue;
                    for (int t = 0; t < 2; t++) {
                        int s = stateIdx(a, b, t);
                        for (int X = 0; X < 2; X++) {
                            std::array<uint8_t, NST>& win = (X == 0) ? win0 : win1;
                            if (win[s]) continue;
                            if (t != X) {
                                // universal: every successor must already win
                                SuccGen g(wallsH, wallsV, a, b, t);
                                if (g.n == 0) continue;
                                bool all = true;
                                for (int i = 0; i < g.n && all; i++) {
                                    int ss = stateIdx(g.p0v[i], g.p1v[i], g.tv[i]);
                                    if (!win[ss]) all = false;
                                }
                                if (all) { win[s] = 1; changed = true; }
                            } else {
                                // existential: one winning successor suffices
                                SuccGen g(wallsH, wallsV, a, b, t);
                                bool any = false;
                                for (int i = 0; i < g.n && !any; i++) {
                                    int ss = stateIdx(g.p0v[i], g.p1v[i], g.tv[i]);
                                    if (win[ss]) any = true;
                                }
                                if (any) { win[s] = 1; changed = true; }
                            }
                        }
                    }
                }
            }
            if (!changed) break;
            assert(sweep + 1 < SWEEP_CAP && "oracle fixpoint did not converge");
        }

        relaxDtm(0);
        relaxDtm(1);
        built = true;
    }

    // Gauss-Seidel relaxation of dtm(s) = 1 + min/max over winning
    // successors, starting from BIG everywhere except the terminal seeds.
    // The system restricted to the win region has a unique fixed point
    // (every chain of optimal play grounds at a terminal), so monotone
    // decreasing updates converge to the true distances.
    void relaxDtm(int X) {
        std::array<uint8_t, NST>& win = (X == 0) ? win0 : win1;
        std::array<int, NST>& dtm = (X == 0) ? dtm0 : dtm1;
        constexpr int BIG = 1 << 28;
        for (int s = 0; s < NST; s++) dtm[s] = win[s] ? BIG : -1;
        for (int a = 0; a < NS; a++) {
            for (int b = 0; b < NS; b++) {
                if (a == b) continue;
                int goalRow = (X == 0) ? qr::GOAL_ROW[0] : qr::GOAL_ROW[1];
                int px = (X == 0) ? a : b;
                if (qr::rowOf(px) == goalRow) {
                    for (int t = 0; t < 2; t++) dtm[stateIdx(a, b, t)] = 0;
                }
            }
        }
        constexpr int RELAX_CAP = 40000;
        for (int sweep = 0; sweep < RELAX_CAP; sweep++) {
            bool changed = false;
            for (int a = 0; a < NS; a++) {
                for (int b = 0; b < NS; b++) {
                    if (a == b || isTerminal(a, b)) continue;
                    for (int t = 0; t < 2; t++) {
                        int s = stateIdx(a, b, t);
                        if (!win[s]) continue;
                        SuccGen g(wallsH, wallsV, a, b, t);
                        if (g.n == 0) continue;  // sealed pocket: not a real win line
                        if (t == X) {
                            int best = BIG;
                            for (int i = 0; i < g.n; i++) {
                                int ss = stateIdx(g.p0v[i], g.p1v[i], g.tv[i]);
                                if (!win[ss]) continue;
                                int cand = dtm[ss] + 1;
                                if (cand < best) best = cand;
                            }
                            if (best < dtm[s] && best < BIG) { dtm[s] = best; changed = true; }
                        } else {
                            // every successor must be winning here; take the
                            // most resistant one
                            int worst = -1;
                            bool all = true;
                            for (int i = 0; i < g.n; i++) {
                                int ss = stateIdx(g.p0v[i], g.p1v[i], g.tv[i]);
                                if (!win[ss]) { all = false; break; }
                                int cand = dtm[ss] + 1;
                                if (cand > worst) worst = cand;
                            }
                            if (all && worst >= 0 && worst < dtm[s]) { dtm[s] = worst; changed = true; }
                        }
                    }
                }
            }
            if (!changed) break;
            assert(sweep + 1 < RELAX_CAP && "oracle dtm relaxation did not converge");
        }
#ifndef NDEBUG
        for (int s = 0; s < NST; s++) {
            if (win[s]) assert(dtm[s] < (1 << 27) && "oracle dtm left at BIG inside win region");
        }
#endif
    }

    qr::RaceOutcome query(int p0, int p1, int t) const {
        assert(built);
        int s = stateIdx(p0, p1, t);
        if (win0[s]) return {0, dtm0[s]};
        if (win1[s]) return {1, dtm1[s]};
        return {-1, 0};
    }
};

} // namespace race_oracle
