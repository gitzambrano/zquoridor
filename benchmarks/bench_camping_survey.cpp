// bench_camping_survey.cpp -- camping survey for the endgame pawn race
// (inv/race-fuzz, priority-zero user symptom). Not wired into the build
// scripts; compile standalone:
//
//   g++ -O3 -std=c++17 -march=native -mavx2 -mfma -Isrc -Itests -o
//       bin/bench_camping_survey.exe benchmarks/bench_camping_survey.cpp
//
// Symptom under test: "sometimes the engine in endgame seems not to
// understand it has to go to the end of the board and STAYS ON THE FIRST
// ROW". A chosen move counts as CAMPING when it keeps the root-side pawn
// inside its own back two rows while at least one legal move advances that
// pawn one row toward its goal.
//
// Method, empty-handed section: hands-empty positions come from wall-biased
// random playouts, plus a short random pawn walk from each endpoint. The
// frozen topology of each family gets ONE oracle build. For every state the
// survey runs Negamax::chooseMove twice (endgameProgressTiebreak ON and
// OFF) and classifies every camping instance with the independent oracle:
//   - outcome for the root side: WON / LOST / DRAWN;
//   - whether the chosen child value equals the oracle-best child value.
// Camping while LOST with an optimal maximum-delay value is correct play.
// Camping while WON or DRAWN with an optimal value is legal geometry (for
// example a sideways step along the only shortest path). Any value mismatch
// is a bug. The MCAB shell is covered by construction here: src/mcab.hpp
// delegates an empty-handed root straight to Negamax::chooseMove, so both
// eval modes below exercise that same path. The NNUE pass loads the
// production weights read-only from C:/Zquoridor/data/nnue when present.
//
// Quasi-endgame section: positions where at least one side holds 2 or fewer
// walls but not both hold zero. No solver shortcut runs there. The survey
// measures how often the side that leads by 2 or more raw plies camps under
// heuristic search. This reports heuristic behavior only; the race solver
// is not involved on these roots.
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>
#include "rules.hpp"
#include "endgame_race.hpp"
#include "search.hpp"
#include "race_oracle.hpp"
using namespace qr;

static constexpr int ROOT_DEPTH_CAP = 10;
static constexpr int RACE_WALK_PLIES = 5;
static constexpr int FAMILIES = 340;
static constexpr int QUASI_TARGET = 220;
static constexpr int QUASI_BUDGET_MS = 120;

struct SurveyState {
    uint64_t wh = 0, wv = 0;
    int p0 = 0, p1 = 0, turn = 0;
    int family = 0, idxInFamily = 0;
};

static bool playoutToEmptyHands(std::mt19937_64& rng, State& out) {
    State s = initialState();
    for (int ply = 0; ply < 400; ply++) {
        if (winner(s) != -1) return false;
        if (s.wallsLeft[0] == 0 && s.wallsLeft[1] == 0) { out = s; return true; }
        MoveList moves = legalMoves(s);
        if (moves.empty()) return false;
        size_t nWall = 0;
        for (size_t i = 0; i < moves.size(); i++) if (moves[i].isWall) nWall++;
        const Move* picked = nullptr;
        if (nWall > 0 && (rng() % 4) != 0) {
            size_t k = rng() % nWall;
            for (size_t i = 0; i < moves.size(); i++) {
                if (!moves[i].isWall) continue;
                if (k-- == 0) { picked = &moves[i]; break; }
            }
        } else {
            size_t nPawn = moves.size() - nWall;
            if (nPawn == 0) return false;
            size_t k = rng() % nPawn;
            for (size_t i = 0; i < moves.size(); i++) {
                if (moves[i].isWall) continue;
                if (k-- == 0) { picked = &moves[i]; break; }
            }
        }
        s = applyMove(s, *picked);
    }
    return false;
}

// Collect hands-empty states: one wall-biased playout per family endpoint,
// then a short random pawn walk over the frozen topology. Each visited
// state becomes one survey root.
static void collectEmptyHanded(std::mt19937_64& rng, std::vector<SurveyState>& out,
                               std::vector<std::pair<uint64_t, uint64_t>>& famTopo) {
    for (int f = 0; f < FAMILIES; f++) {
        State s;
        if (!playoutToEmptyHands(rng, s)) continue;
        famTopo.push_back({s.wallsH, s.wallsV});
        for (int w = 0; w <= RACE_WALK_PLIES; w++) {
            if (winner(s) != -1) break;
            SurveyState ss{s.wallsH, s.wallsV, s.pawn[0], s.pawn[1], s.turn, f, w};
            out.push_back(ss);
            MoveList moves = legalMoves(s);
            if (moves.empty()) break;
            s = applyMove(s, moves[rng() % moves.size()]);
        }
    }
}

// Collect quasi-endgame states: not both hands empty, at least one side
// holds 2 or fewer walls. First matching ply of each playout wins. Full
// states are kept so wallsLeft stays exact for move legality.
static void collectQuasi(std::mt19937_64& rng, std::vector<State>& out) {
    for (int f = 0; f < QUASI_TARGET * 3 && (int)out.size() < QUASI_TARGET; f++) {
        State s = initialState();
        bool got = false;
        for (int ply = 0; ply < 400 && !got; ply++) {
            if (winner(s) != -1) break;
            bool q = !(s.wallsLeft[0] == 0 && s.wallsLeft[1] == 0) &&
                     (s.wallsLeft[0] <= 2 || s.wallsLeft[1] <= 2);
            if (q) { out.push_back(s); got = true; break; }
            MoveList moves = legalMoves(s);
            if (moves.empty()) break;
            s = applyMove(s, moves[rng() % moves.size()]);
        }
    }
}

static bool inBackTwoRows(int side, int cell) {
    int r = rowOf(cell);
    return (side == 0) ? (r <= 1) : (r >= N - 2);
}

// A forward move exists when some legal pawn move of `side` advances that
// pawn one or more rows toward its goal (straight steps and forward jumps,
// including diagonal detours around the opponent).
static bool forwardExists(uint64_t wh, uint64_t wv, int meCell, int oppCell, int side) {
    static const int dr[4] = {-1, 1, 0, 0};
    static const int dc[4] = {0, 0, -1, 1};
    int mr = rowOf(meCell), mc = colOf(meCell);
    int goalRow = GOAL_ROW[side];
    for (int d = 0; d < 4; d++) {
        int r1 = mr + dr[d], c1 = mc + dc[d];
        if (!inBounds(r1, c1)) continue;
        if (edgeBlocked(wh, wv, mr, mc, r1, c1)) continue;
        int s1 = cellIdx(r1, c1);
        if (s1 != oppCell) {
            if (rowOf(s1) == goalRow || ((side == 0) ? dr[d] > 0 : dr[d] < 0)) return true;
            continue;
        }
        int r2 = r1 + dr[d], c2 = c1 + dc[d];
        if (inBounds(r2, c2) && !edgeBlocked(wh, wv, r1, c1, r2, c2)) {
            // straight jump over the opponent
            if ((side == 0) ? dr[d] > 0 : dr[d] < 0) return true;
        } else {
            int pdA = (d < 2) ? 2 : 0;
            int pdB = (d < 2) ? 3 : 1;
            const int pds[2] = {pdA, pdB};
            for (int pi = 0; pi < 2; pi++) {
                int pd = pds[pi];
                int rd = r1 + dr[pd], cd = c1 + dc[pd];
                if (!inBounds(rd, cd)) continue;
                if (edgeBlocked(wh, wv, r1, c1, rd, cd)) continue;
                // diagonal detour: row delta toward the goal is dr[d]
                if ((side == 0) ? dr[d] > 0 : dr[d] < 0) return true;
            }
        }
    }
    return false;
}

// Child score exactly the way the empty-handed root branch of chooseMove
// computes it (parityAnchoredRaceDraw OFF in both survey passes), but with
// ORACLE outcomes as ground truth.
static int expectedChildScoreForRoot(const race_oracle::Table& orc,
                                     const State& root, const Move& m, int contempt) {
    State ns = applyMove(root, m);
    int w = winner(ns);
    if (w != -1) return (w == ns.turn) ? -(SCORE_INF - 1) : (SCORE_INF - 1);
    RaceOutcome oc = orc.query(ns.pawn[0], ns.pawn[1], ns.turn);
    if (oc.winner == -1) return contempt;
    int raw = RACE_SCORE_BASE - oc.dtm;
    int childScore = (oc.winner == ns.turn) ? raw : -raw;
    return -childScore;
}

struct Counters {
    long roots = 0;
    long campTotal = 0;
    long campLostOptimal = 0;   // correct maximum-delay defense
    long campWonDrawnOptimal = 0;  // legal geometry, value already best
    long campValueBug = 0;      // camping AND chosen value below oracle best
    long valueMismatchAny = 0;  // over all roots, camping or not
    long campByOutcome[3] = {0, 0, 0};  // indexed WON=0, LOST=1, DRAWN=2
};

int main() {
    std::mt19937_64 rng(20260823ULL);

    std::vector<SurveyState> emptyRoots;
    std::vector<std::pair<uint64_t, uint64_t>> famTopo;
    collectEmptyHanded(rng, emptyRoots, famTopo);
    std::printf("(info) empty-handed roots: %zu from %zu families\n",
                emptyRoots.size(), famTopo.size());
    std::fflush(stdout);

    FILE* csv = std::fopen("data\\camping_survey.csv", "w");
    if (csv) std::fprintf(csv, "section,family,idx,wallsH,wallsV,p0,p1,turn,tb,outcome,dtm,"
                               "camping,fwdExists,valueOptimal,score,chosenRowAfter\n");

    Negamax engOn;   // production defaults: tiebreak ON
    Negamax engOff;  // tiebreak OFF
    engOff.setEndgameProgressTiebreak(false);
    Negamax engNnue;
    bool haveWeights = false;
    {
        const char* candidates[] = {
            "data/nnue/nnue_weights_int8.bin",
            "C:/Zquoridor/data/nnue/nnue_weights_int8.bin",
        };
        for (const char* path : candidates) {
            FILE* f = std::fopen(path, "rb");
            if (f) { std::fclose(f); haveWeights = loadWeightsQuant(path); break; }
        }
        if (haveWeights) engNnue.setEvalMode(Negamax::EvalMode::NNUE);
        else std::printf("(info) NNUE weights not found -- NNUE pass skipped\n");
    }

    enum Mode { MODE_ON = 0, MODE_OFF = 1 };
    Counters ct[2];
    long nnueAgree = 0, nnueChecked = 0;

    for (Mode mode : {MODE_ON, MODE_OFF}) {
        Negamax& eng = (mode == MODE_ON) ? engOn : engOff;
        Counters& C = ct[mode];
        // Consecutive walk states share one frozen topology, so keep the
        // last oracle alive and rebuild only when the topology changes.
        race_oracle::Table orc;
        uint64_t lastWh = 0, lastWv = 0;
        bool haveOrc = false;
        for (size_t i = 0; i < emptyRoots.size(); i++) {
            const SurveyState& ss = emptyRoots[i];
            if (!haveOrc || ss.wh != lastWh || ss.wv != lastWv) {
                orc.build(ss.wh, ss.wv);
                lastWh = ss.wh;
                lastWv = ss.wv;
                haveOrc = true;
            }
            State root;
            root.pawn[0] = (uint8_t)ss.p0;
            root.pawn[1] = (uint8_t)ss.p1;
            root.wallsH = ss.wh;
            root.wallsV = ss.wv;
            root.wallsLeft[0] = 0;
            root.wallsLeft[1] = 0;
            root.turn = ss.turn;
            uint64_t h = zobrist().pawnKey[0][ss.p0] ^ zobrist().pawnKey[1][ss.p1];
            for (int slot = 0; slot < WS * WS; slot++) {
                if ((ss.wh >> slot) & 1ull) h ^= zobrist().wallHKey[slot];
                if ((ss.wv >> slot) & 1ull) h ^= zobrist().wallVKey[slot];
            }
            if (ss.turn == 1) h ^= zobrist().turnKey;
            root.hash = h;

            RaceOutcome oc = orc.query(ss.p0, ss.p1, ss.turn);
            int relWinner = (oc.winner == -1) ? -1 : (oc.winner == ss.turn ? 0 : 1);  // 0=WON 1=LOST

            MoveList ms = legalMoves(root);
            int myCell = root.pawn[root.turn];
            int oppCell = root.pawn[1 - root.turn];
            bool fwd = forwardExists(ss.wh, ss.wv, myCell, oppCell, root.turn);

            SearchStats st;
            Move chosen = eng.chooseMove(root, ROOT_DEPTH_CAP, 100, st);
            C.roots++;

            bool legal = false;
            for (size_t k = 0; k < ms.size(); k++) if (ms[k] == chosen) { legal = true; break; }
            if (!legal) {
                std::printf("ILEGAL fam=%d idx=%d mode=%d\n", ss.family, ss.idxInFamily, (int)mode);
                continue;
            }

            int newRow = rowOf((int)chosen.a);
            bool camps = fwd && inBackTwoRows(root.turn, (int)chosen.a);

            int chosenVal = INT32_MIN, bestVal = INT32_MIN;
            for (size_t k = 0; k < ms.size(); k++) {
                int v = expectedChildScoreForRoot(orc, root, ms[k], eng.getContempt());
                if (ms[k] == chosen) chosenVal = v;
                if (k == 0 || v > bestVal) bestVal = v;
            }
            bool valOk = (chosenVal == bestVal);
            if (!valOk) {
                C.valueMismatchAny++;
                if (C.valueMismatchAny <= 10)
                    std::printf("VALUE-MISMATCH mode=%d wallsH=0x%llx wallsV=0x%llx p0=%d p1=%d t=%d "
                                "chosen=%d best=%d\n", (int)mode,
                                (unsigned long long)ss.wh, (unsigned long long)ss.wv,
                                ss.p0, ss.p1, ss.turn, chosenVal, bestVal);
            }
            if (camps) {
                C.campTotal++;
                int oidx = (relWinner == -1) ? 2 : relWinner;
                C.campByOutcome[oidx]++;
                if (relWinner == 1) {
                    if (valOk) C.campLostOptimal++;
                    else C.campValueBug++;
                } else {
                    if (valOk) C.campWonDrawnOptimal++;
                    else C.campValueBug++;
                }
            }
            if (csv)
                std::fprintf(csv, "empty,%d,%d,%llx,%llx,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d\n",
                             ss.family, ss.idxInFamily,
                             (unsigned long long)ss.wh, (unsigned long long)ss.wv,
                             ss.p0, ss.p1, ss.turn, (int)mode, relWinner, oc.dtm,
                             (int)camps, (int)fwd, (int)valOk, st.score, newRow);

            if (mode == MODE_ON) {
                if (haveWeights) {
                    SearchStats stN;
                    Move chosenN = engNnue.chooseMove(root, ROOT_DEPTH_CAP, 100, stN);
                    nnueChecked++;
                    if (chosenN == chosen) nnueAgree++;
                    else std::printf("NNUE-DIFF wallsH=0x%llx wallsV=0x%llx p0=%d p1=%d t=%d\n",
                                     (unsigned long long)ss.wh, (unsigned long long)ss.wv,
                                     ss.p0, ss.p1, ss.turn);
                }
            }
            if ((i % 500) == 499) {
                std::printf("(tick) mode=%d %zu/%zu\n", (int)mode, i + 1, emptyRoots.size());
                std::fflush(stdout);
            }
        }
    }

    // ---- quasi-endgame: heuristic regime, no solver at the root --------
    // Full states are kept so wallsLeft stays exact for move legality.
    std::vector<State> quasiStates;
    collectQuasi(rng, quasiStates);
    long quasiAheadRoots = 0, quasiAheadCamp = 0, quasiCampAnySide = 0;
    for (size_t i = 0; i < quasiStates.size(); i++) {
        const State& root = quasiStates[i];
        SearchStats st;
        Move chosen = engOn.chooseMove(root, ROOT_DEPTH_CAP, QUASI_BUDGET_MS, st);
        MoveList ms = legalMoves(root);
        bool legal = false;
        for (size_t k = 0; k < ms.size(); k++) if (ms[k] == chosen) { legal = true; break; }
        if (!legal) { std::printf("QUASI-ILEGAL idx=%zu\n", i); continue; }
        int side = root.turn;
        int dMe = shortestPathLen(root.wallsH, root.wallsV, root.pawn[side], side);
        int dOp = shortestPathLen(root.wallsH, root.wallsV, root.pawn[1 - side], 1 - side);
        bool aheadBy2 = (dMe >= 0 && dOp >= 0 && dMe + 2 <= dOp);
        bool camps = false, fwd = false;
        if (chosen.a != root.pawn[side]) {
            fwd = forwardExists(root.wallsH, root.wallsV, root.pawn[side],
                                root.pawn[1 - side], side);
            camps = fwd && inBackTwoRows(side, (int)chosen.a);
        }
        if (camps) quasiCampAnySide++;
        if (aheadBy2) {
            quasiAheadRoots++;
            if (camps) quasiAheadCamp++;
        }
        if (csv)
            std::fprintf(csv, "quasi,%zu,0,%llx,%llx,%d,%d,%d,2,-9,-9,%d,%d,-9,-9,%d,%d,%d,%d,%d,%d\n",
                         i, (unsigned long long)root.wallsH, (unsigned long long)root.wallsV,
                         root.pawn[0], root.pawn[1], root.turn, (int)camps, (int)fwd,
                         rowOf((int)chosen.a), dMe, dOp, st.score,
                         root.wallsLeft[0], root.wallsLeft[1]);
        if ((i % 50) == 49) { std::printf("(tick) quasi %zu/%zu\n", i + 1, quasiStates.size()); std::fflush(stdout); }
    }

    if (csv) std::fclose(csv);

    for (int m = 0; m < 2; m++) {
        const Counters& C = ct[m];
        double rate = C.roots ? 100.0 * C.campTotal / C.roots : 0.0;
        std::printf("SUMMARY mode=%s roots=%ld camping=%ld (%.2f%%) [won=%ld lost=%ld drawn=%ld] "
                    "lost-optimal=%ld wondrawn-optimal=%ld value-bugs=%ld mismatches-any=%ld\n",
                    m == 0 ? "TB-on" : "TB-off", C.roots, C.campTotal, rate,
                    C.campByOutcome[0], C.campByOutcome[1], C.campByOutcome[2],
                    C.campLostOptimal, C.campWonDrawnOptimal, C.campValueBug,
                    C.valueMismatchAny);
    }
    if (haveWeights)
        std::printf("SUMMARY nnue agreement %ld/%ld (empty-handed root branch)\n",
                    nnueAgree, nnueChecked);
    std::printf("SUMMARY quasi states=%zu ahead-by-2-at-root=%ld camping-ahead=%ld "
                "camping-any-side=%ld\n",
                quasiStates.size(), quasiAheadRoots, quasiAheadCamp, quasiCampAnySide);
    std::printf("OK -- bench_camping_survey finished (CSV at data/camping_survey.csv)\n");
    return 0;
}
