// test_contempt_repetition -- correctness checks for the contempt /
// repetition / endgame-draw handling investigated in August 2026
// ("engine wanders in near-endgames with multiple paths").
//
// Profile: -O2 -std=c++17, no AVX2 (reproducibility convention of the
// other correctness tests).
//
// Build:
//   g++ -O2 -std=c++17 -Isrc -o bin/test_contempt_repetition.exe ^
//       tests/test_contempt_repetition.cpp
//
// What is pinned here:
//   T1  Default draw-sign convention (frozen): a pure-draw empty-handed
//       root reports stats.score == +30; a won root reports race scale.
//   T2  setParityAnchoredRaceDraw(true) flips the pure-draw root report
//       to -30 (same anchoring as repetition draws).
//   T3  setEndgameProgressTiebreak(true) may change the move only between
//       EXACTLY EQUAL child values; game value never changes.
//   T4  Agreement thresholds: contempt -30 vs 0 agree on the best move in
//       >=85% of multi-path near-endgame positions; never illegal.
//   T5  RepetitionTable occurrence semantics: 2-in-search vs 3-with-
//       pre-root, and the markRoot interplay (H4).
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>
#include "rules.hpp"
#include "search.hpp"

using namespace qr;

static int g_failures = 0;

static void check(bool ok, const char* what) {
    if (!ok) {
        printf("  FAIL: %s\n", what);
        g_failures++;
    }
}

// Known infinite-pursuit topology from tests/test_endgame_race.cpp.
static const uint64_t kDrawWallsH = 0x48000008000000ull;
static const uint64_t kDrawWallsV = 0x8014020000022000ull;
static const int kDrawPawn0 = 57;  // cellIdx(6,3)
static const int kDrawPawn1 = 49;  // cellIdx(5,4)

static State makeState(uint64_t wh, uint64_t wv, int p0, int p1, int turn) {
    State s;
    s.wallsH = wh;
    s.wallsV = wv;
    s.pawn[0] = (uint8_t)p0;
    s.pawn[1] = (uint8_t)p1;
    s.wallsLeft[0] = 0;
    s.wallsLeft[1] = 0;
    s.turn = (int8_t)turn;
    s.hash = zobrist().pawnKey[0][p0] ^ zobrist().pawnKey[1][p1];
    for (int i = 0; i < 64; i++) {
        if ((wh >> i) & 1ull) s.hash ^= zobrist().wallHKey[i];
        if ((wv >> i) & 1ull) s.hash ^= zobrist().wallVKey[i];
    }
    s.hash ^= zobrist().turnKey;
    return s;
}

// Exact child value from `mover` perspective at an empty-handed position:
// +RACE_SCORE_BASE-dtm win, -(RACE_SCORE_BASE-dtm) loss, contempt draw.
static int childValue(const State& ns, int mover) {
    int w = winner(ns);
    if (w != -1) return (w == mover) ? SCORE_INF - 1 : -(SCORE_INF - 1);
    RaceOutcome ro = resolveEmptyHandedEndgame(ns.wallsH, ns.wallsV,
                                                ns.pawn[0], ns.pawn[1], ns.turn);
    if (ro.winner == -1) return CONTEMPT;
    int raw = RACE_SCORE_BASE - ro.dtm;
    return (ro.winner == mover) ? raw : -raw;
}

// T1 + T2: sign conventions on the known pursuit topology.
static void testDrawSignConventions() {
    printf("T1/T2: draw-sign conventions (empty-handed root branch)\n");
    State drawRoot = makeState(kDrawWallsH, kDrawWallsV, kDrawPawn0, kDrawPawn1, 1);
    RaceOutcome ro = resolveEmptyHandedEndgame(drawRoot.wallsH, drawRoot.wallsV,
                                                drawRoot.pawn[0], drawRoot.pawn[1], 1);
    check(ro.winner == -1, "setup: known topology must be a theoretical draw with turn=1");

    Negamax eng;
    SearchStats st;
    Move mDef = eng.chooseMove(drawRoot, 6, 60000, st);
    // Default: every drawing child is worth +30 to the root.
    check(st.score == -CONTEMPT,
          "default: pure-draw root should report exactly -CONTEMPT (+30)");
    bool mDefLegal = false;
    {
        MoveList lm = legalMoves(drawRoot);
        for (size_t i = 0; i < lm.size(); i++) if (lm[i] == mDef) { mDefLegal = true; break; }
    }
    check(mDefLegal, "default: chosen move must be legal");
    {
        State ns = applyMove(drawRoot, mDef);
        RaceOutcome after = resolveEmptyHandedEndgame(ns.wallsH, ns.wallsV,
                                                       ns.pawn[0], ns.pawn[1], ns.turn);
        check(after.winner != 0,
              "default: chosen move must not hand player 0 a forced win");
    }

    // Parity-anchored toggle flips the report to contempt (-30).
    eng.setParityAnchoredRaceDraw(true);
    SearchStats st2;
    Move mPar = eng.chooseMove(drawRoot, 6, 60000, st2);
    check(st2.score == CONTEMPT,
          "parityAnchoredRaceDraw: pure-draw root should report exactly CONTEMPT (-30)");
    check(mPar == mDef,
          "parity toggle must NOT change which move is picked (draw ties with draws)");
    eng.setParityAnchoredRaceDraw(false);

    // Won root (same topology, turn=0): forced win in 9 plies at the root;
    // a dtm-optimal move reaches a child solved at dtm 8, so the reported
    // root score is exactly RACE_SCORE_BASE - 8.
    State winRoot = makeState(kDrawWallsH, kDrawWallsV, kDrawPawn0, kDrawPawn1, 0);
    {
        RaceOutcome rw = resolveEmptyHandedEndgame(winRoot.wallsH, winRoot.wallsV,
                                                    winRoot.pawn[0], winRoot.pawn[1], 0);
        check(rw.winner == 0 && rw.dtm == 9,
              "setup: turn=0 must be a forced win in 9 plies on this topology");
    }
    SearchStats st3;
    eng.chooseMove(winRoot, 6, 60000, st3);
    check(st3.score == RACE_SCORE_BASE - 8,
          "won empty-handed root should report exactly RACE_SCORE_BASE - 8");
}

// T3: progress tie-break may only swap EXACTLY EQUAL children.
static void testProgressTiebreakInvariant() {
    printf("T3: endgame progress tie-break invariant\n");
    std::mt19937_64 rng(20260822);
    int differing = 0, checked = 0;
    for (int g = 0; g < 400 && checked < 80; g++) {
        // Walk a pawn-biased playout until both wall stocks are empty.
        State s = initialState();
        for (int ply = 0; ply < 220; ply++) {
            if (winner(s) != -1 || (s.wallsLeft[0] == 0 && s.wallsLeft[1] == 0)) break;
            MoveList all = legalMoves(s);
            if (all.empty()) break;
            MoveList pm, wm;
            for (size_t i = 0; i < all.size(); i++)
                (all[i].isWall ? wm : pm).push_back(all[i]);
            bool usePawn = !pm.empty() && ((int)(rng() % 100) < 80 || wm.empty());
            MoveList& src = usePawn ? pm : wm;
            s = applyMove(s, src[rng() % src.size()]);
        }
        if (s.wallsLeft[0] != 0 || s.wallsLeft[1] != 0 || winner(s) != -1) continue;

        Negamax eng;
        SearchStats stA, stB;
        // Default is ON since 2026-08-23; compare explicitly OFF (legacy
        // generation order) against explicitly ON.
        eng.setEndgameProgressTiebreak(false);
        Move mA = eng.chooseMove(s, 6, 60000, stA);
        eng.setEndgameProgressTiebreak(true);
        Move mB = eng.chooseMove(s, 6, 60000, stB);
        checked++;

        check(stA.score == stB.score,
              "tie-break must not change the reported game value");
        if (!(mA == mB)) {
            differing++;
            int vA = childValue(applyMove(s, mA), s.turn);
            int vB = childValue(applyMove(s, mB), s.turn);
            check(vA == vB, "tie-break may only swap EXACTLY EQUAL children");
        }
    }
    printf("  info: tie-break changed the move on %d/%d empty-handed roots\n",
           differing, checked);
    check(differing > 0, "expected the tie-break to fire on at least one root");
}

// T4: agreement thresholds across contempt settings (heuristic mode).
static void testContemptAgreement() {
    printf("T4: contempt -30 vs 0 agreement on multi-path near-endgames\n");
    std::mt19937_64 rng(777001);
    int agree = 0, total = 0, illegal = 0;
    for (int g = 0; g < 400 && total < 60; g++) {
        State s = initialState();
        for (int ply = 0; ply < 220; ply++) {
            if (winner(s) != -1) break;
            int tw = s.wallsLeft[0] + s.wallsLeft[1];
            if (tw >= 5 && tw <= 12 &&
                pathRobustness(s.wallsH, s.wallsV, s.pawn[0], 0) >= 2 &&
                pathRobustness(s.wallsH, s.wallsV, s.pawn[1], 1) >= 2) break;
            MoveList all = legalMoves(s);
            if (all.empty()) break;
            MoveList pm, wm;
            for (size_t i = 0; i < all.size(); i++)
                (all[i].isWall ? wm : pm).push_back(all[i]);
            bool usePawn = !pm.empty() && ((int)(rng() % 100) < 80 || wm.empty());
            MoveList& src = usePawn ? pm : wm;
            s = applyMove(s, src[rng() % src.size()]);
        }
        int tw = s.wallsLeft[0] + s.wallsLeft[1];
        if (winner(s) != -1 || tw < 5 || tw > 12) continue;

        Negamax e1, e2;
        e1.setContempt(-30);
        e2.setContempt(0);
        SearchStats s1, s2;
        Move m1 = e1.chooseMove(s, 6, 30000, s1);
        Move m2 = e2.chooseMove(s, 6, 30000, s2);
        bool ok1 = false, ok2 = false;
        {
            MoveList lm = legalMoves(s);
            for (size_t i = 0; i < lm.size(); i++) {
                if (lm[i] == m1) ok1 = true;
                if (lm[i] == m2) ok2 = true;
            }
        }
        if (!ok1 || !ok2) illegal++;
        if (m1 == m2) agree++;
        total++;
    }
    printf("  agreement: %d/%d (%.1f%%), illegal=%d\n", agree, total,
           total ? 100.0 * agree / total : 0.0, illegal);
    check(illegal == 0, "no illegal moves may be returned");
    check(total >= 40, "sample size too small");
    check(agree * 100 >= 85 * total, "contempt -30 vs 0 must agree on >=85% of moves");
}

// T5: occurrence semantics of RepetitionTable (H4).
static void testRepetitionOccurrenceSemantics() {
    printf("T5: repetition occurrence semantics / markRoot interplay\n");
    // A 4-ply mutual shuffle returns to the same hash (both pawns home,
    // same side to move). Start from the initial state.
    State s0 = initialState();
    State sA = applyMove(s0, Move::pawn(cellIdx(1, N / 2)));  // p0 down
    State sB = applyMove(sA, Move::pawn(cellIdx(N - 2, N / 2)));  // p1 up
    State sC = applyMove(sB, Move::pawn(cellIdx(0, N / 2)));  // p0 back up
    State sD = applyMove(sC, Move::pawn(cellIdx(N - 1, N / 2)));  // p1 back down
    check(sD.hash == s0.hash, "setup: 4-ply mutual shuffle must recreate the start hash");

    // Case 1: one REAL occurrence before the root (markRoot). The cycle
    // completes once inside the search line -> total=2, but one occurrence
    // is pre-root -> needs 3 -> NOT a draw yet.
    {
        RepetitionTable rt;
        rt.push(s0.hash);      // real game played through s0 once
        rt.push(sA.hash);
        rt.markRoot();         // search starts here (sB to move)
        rt.push(sB.hash);
        rt.push(sC.hash);
        rt.push(sD.hash);      // == s0 again (in-search occurrence #1)
        check(!rt.isRepetitionDraw(sD.hash),
              "pre-root occurrence: 2 total occurrences must NOT be a draw (needs 3)");
        rt.push(sA.hash);      // cycle continues: back to A (2nd time)
        rt.push(sB.hash);
        rt.push(sC.hash);
        rt.push(sD.hash);      // == s0 third occurrence overall
        check(rt.isRepetitionDraw(sD.hash),
              "pre-root occurrence: 3 total occurrences must be a draw");
    }
    // Case 2: no pre-root history. Two occurrences INSIDE the hypothetical
    // line are already scored as a draw (search optimization); covered by
    // testRepetitionInSearchTwoOccurrences below.
}

static void testRepetitionInSearchTwoOccurrences() {
    State s0 = initialState();
    State sA = applyMove(s0, Move::pawn(cellIdx(1, N / 2)));
    State sB = applyMove(sA, Move::pawn(cellIdx(N - 2, N / 2)));
    State sC = applyMove(sB, Move::pawn(cellIdx(0, N / 2)));
    State sD = applyMove(sC, Move::pawn(cellIdx(N - 1, N / 2)));
    State sE = applyMove(sD, Move::pawn(cellIdx(1, N / 2)));
    State sF = applyMove(sE, Move::pawn(cellIdx(N - 2, N / 2)));
    State sG = applyMove(sF, Move::pawn(cellIdx(0, N / 2)));
    State sH = applyMove(sG, Move::pawn(cellIdx(N - 1, N / 2)));
    check(sH.hash == s0.hash, "setup: second cycle recreates the start hash");
    RepetitionTable rt;
    rt.markRoot();
    rt.push(sA.hash); rt.push(sB.hash); rt.push(sC.hash); rt.push(sD.hash);
    rt.push(sE.hash); rt.push(sF.hash); rt.push(sG.hash); rt.push(sH.hash);
    check(rt.isRepetitionDraw(sH.hash),
          "no pre-root history: SECOND in-search recurrence must be a draw");
}

int main() {
    testDrawSignConventions();
    testProgressTiebreakInvariant();
    testContemptAgreement();
    testRepetitionOccurrenceSemantics();
    testRepetitionInSearchTwoOccurrences();
    if (g_failures == 0) {
        printf("OK -- all contempt/repetition tests passed\n");
        return 0;
    }
    printf("%d FAILURE(S)\n", g_failures);
    return 1;
}
