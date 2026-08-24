// test_endgame_race_fuzz.cpp -- adversarial cross-check of the empty-handed
// race solver (src/endgame_race.hpp) against an independent brute-force
// oracle (tests/race_oracle.hpp), plus optimality checks of the
// empty-handed root branch of Negamax::chooseMove.
//
// Phases:
//   A. Value agreement: the production pipeline (gate + Service B) AND bare
//      raceExactDTM must match the oracle exactly -- 100%, zero tolerance,
//      because both sides are exact computations -- over thousands of
//      frozen topologies (playout-born, synthetic, adversarial pawn
//      configs). Comparing the bare DP too splits a gate bug from a DP bug.
//   B. Root-move optimality: for won/lost/drawn empty-handed roots, the
//      move returned by chooseMove must achieve the oracle-best child
//      value, under all four toggle combos of endgameProgressTiebreak x
//      parityAnchoredRaceDraw. This pins the class of "correct score,
//      wrong move" bugs documented in the solver history.
//   C. Budget-exhaustion fallback: with the global used-budget forced past
//      the cap, the search must fall back cleanly (no crash, finite
//      scores); a tiny-time-budget chooseMove must always return a legal
//      move.
//   D. Cache soundness: interleaved production queries across hundreds of
//      topologies with eviction pressure beyond the 1024-slot table must
//      stay equal to remembered oracle results.
//   E. Degenerate-input probes: a pawn already on its goal row, and pawns
//      sealed in pockets whose region lacks their goal row. Real play can
//      never produce these states (wall legality keeps both paths alive),
//      but resolveEmptyHandedEndgame is a public inline utility and must
//      stay sound for direct callers.
//   F. Long-game differential (compact, deterministic): engine vs engine
//      from hands-empty starts through the exact-solver root branch. The
//      realized winner and the realized game length must equal the oracle
//      prediction exactly (both sides play value-optimal moves there), and
//      drawn pursuits must never finish. benchmarks/bench_race_differential
//      runs the same check at larger scale on multiple threads.
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>
#include <algorithm>
#include "rules.hpp"
#include "endgame_race.hpp"
#include "search.hpp"
#include "race_oracle.hpp"
using namespace qr;

static int failures = 0;
#define CHECK(cond, msg) do { \
    if (!(cond)) { std::printf("FALHOU: %s (linha %d)\n", msg, __LINE__); failures++; } \
} while (0)

static void reportMismatch(const char* where, uint64_t wh, uint64_t wv,
                           int p0, int p1, int t,
                           RaceOutcome prod, RaceOutcome dp, RaceOutcome orc) {
    std::printf("  DIVERGENCIA [%s] wallsH=0x%llx wallsV=0x%llx p0=%d p1=%d turn=%d "
                "prod=(%d,%d) dp=(%d,%d) oracle=(%d,%d)\n",
                where, (unsigned long long)wh, (unsigned long long)wv, p0, p1, t,
                prod.winner, prod.dtm, dp.winner, dp.dtm, orc.winner, orc.dtm);
}

// ---------------------------------------------------------------------
// Topology generators
// ---------------------------------------------------------------------

// Random legal walk until both wall reserves hit zero. Biased toward wall
// moves so the walk exhausts the reserves quickly and lands on rich frozen
// topologies. Returns false when the walk ended the game or timed out
// before emptying both hands.
static bool playoutToEmptyHands(std::mt19937_64& rng, int maxPlies, State& out) {
    State s = initialState();
    for (int ply = 0; ply < maxPlies; ply++) {
        if (winner(s) != -1) return false;
        if (s.wallsLeft[0] == 0 && s.wallsLeft[1] == 0) { out = s; return true; }
        MoveList moves = legalMoves(s);
        if (moves.empty()) return false;
        size_t nWall = 0;
        for (size_t i = 0; i < moves.size(); i++) if (moves[i].isWall) nWall++;
        bool preferWall = (nWall > 0) && ((rng() % 4) != 0);
        const Move* picked = nullptr;
        if (preferWall) {
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

// Synthetic frozen topology: K overlapping-legal walls placed uniformly.
static void synthWalls(std::mt19937_64& rng, int k, uint64_t& wh, uint64_t& wv) {
    wh = 0;
    wv = 0;
    int placed = 0;
    while (placed < k) {
        int orient = (int)(rng() % 2);
        int r = (int)(rng() % WS);
        int c = (int)(rng() % WS);
        if (!wallSlotAvailable(wh, wv, orient, r, c)) continue;
        int slot = slotIdx(r, c);
        if (orient == 0) wh |= (1ull << slot);
        else wv |= (1ull << slot);
        placed++;
    }
}

static int openEdgeCount(uint64_t wh, uint64_t wv, int cell) {
    static const int dr[4] = {-1, 1, 0, 0};
    static const int dc[4] = {0, 0, -1, 1};
    int r = rowOf(cell), c = colOf(cell);
    int n = 0;
    for (int d = 0; d < 4; d++) {
        int nr = r + dr[d], nc = c + dc[d];
        if (!inBounds(nr, nc)) continue;
        if (!edgeBlocked(wh, wv, r, c, nr, nc)) n++;
    }
    return n;
}

struct PawnConfig {
    int p0, p1;
    const char* kind;
};

// Pawn configurations for one frozen topology, restricted to the real-play
// invariant (both pawns keep a path to their own goal; nobody stands on a
// goal row). Mixes random pairs, head-on adjacencies, dead-end cells and
// same-column pairs.
static void collectPawnConfigs(std::mt19937_64& rng, uint64_t wh, uint64_t wv,
                               int wantRandom, std::vector<PawnConfig>& out) {
    auto usable = [&](int cell, int player) {
        if (rowOf(cell) == GOAL_ROW[player]) return false;
        return hasPathToGoal(wh, wv, cell, player);
    };
    for (int i = 0; i < wantRandom * 4 && (int)out.size() < wantRandom; i++) {
        int a = (int)(rng() % (N * N));
        int b = (int)(rng() % (N * N));
        if (a == b) continue;
        if (!usable(a, 0) || !usable(b, 1)) continue;
        out.push_back({a, b, "rand"});
    }
    // head-on adjacencies along open edges (jump/blocking territory)
    static const int dr[4] = {-1, 1, 0, 0};
    static const int dc[4] = {0, 0, -1, 1};
    int added = 0;
    for (int cell = 0; cell < N * N && added < 2; cell++) {
        int r = rowOf(cell), c = colOf(cell);
        for (int d = 0; d < 4 && added < 2; d++) {
            int nr = r + dr[d], nc = c + dc[d];
            if (!inBounds(nr, nc) || nr * N + nc <= cell) continue;
            if (edgeBlocked(wh, wv, r, c, nr, nc)) continue;
            int a = cell, b = nr * N + nc;
            if (!usable(a, 0) || !usable(b, 1)) continue;
            out.push_back({a, b, "headon"});
            added++;
        }
    }
    // dead-end cells (exactly one open edge)
    int deadAdded = 0;
    for (int cell = 0; cell < N * N && deadAdded < 2; cell++) {
        if (openEdgeCount(wh, wv, cell) != 1) continue;
        if (!usable(cell, 0)) continue;
        for (int tries = 0; tries < 16; tries++) {
            int b = (int)(rng() % (N * N));
            if (b == cell || !usable(b, 1)) continue;
            out.push_back({cell, b, "deadend"});
            deadAdded++;
            break;
        }
    }
    // same-column pairs
    for (int tries = 0; tries < 8; tries++) {
        int col = (int)(rng() % N);
        int r0 = (int)(rng() % N), r1 = (int)(rng() % N);
        if (r0 == r1) continue;
        int a = r0 * N + col, b = r1 * N + col;
        if (!usable(a, 0) || !usable(b, 1)) continue;
        out.push_back({a, b, "samecol"});
        break;
    }
}

struct TopoCase {
    uint64_t wh = 0, wv = 0;
    std::vector<PawnConfig> configs;
    // remembered oracle answers (capped) for the cache-stress replay
    struct Remembered { int p0, p1, t, win, dtm; };
    std::vector<Remembered> remembered;
};

static long g_comparisons = 0, g_gateDecided = 0, g_gateRefused = 0;

static void buildTopoCase(std::mt19937_64& rng, TopoCase& tc, bool synthetic,
                          std::mt19937_64& synthRng) {
    if (synthetic) {
        int k = 4 + (int)(synthRng() % 31);  // 4..34 walls
        synthWalls(synthRng, k, tc.wh, tc.wv);
    }
    collectPawnConfigs(rng, tc.wh, tc.wv, 3, tc.configs);
}

// Full agreement check of one topology against the oracle; remembers a cap
// of oracle answers for the later cache-stress replay.
static void checkTopology(const char* where, TopoCase& tc, race_oracle::Table& orc) {
    for (const PawnConfig& pc : tc.configs) {
        for (int turn = 0; turn < 2; turn++) {
            RaceOutcome prod = resolveEmptyHandedEndgame(tc.wh, tc.wv, pc.p0, pc.p1, turn);
            RaceOutcome dp = raceExactDTM(tc.wh, tc.wv, pc.p0, pc.p1, turn);
            RaceOutcome oc = orc.query(pc.p0, pc.p1, turn);
            g_comparisons++;
            bool prodOk = (prod.winner == oc.winner && prod.dtm == oc.dtm);
            bool dpOk = (dp.winner == oc.winner && dp.dtm == oc.dtm);
            if (!prodOk || !dpOk) reportMismatch(where, tc.wh, tc.wv, pc.p0, pc.p1, turn, prod, dp, oc);
            CHECK(prodOk, "production pipeline must equal the independent oracle");
            CHECK(dpOk, "bare raceExactDTM must equal the independent oracle");
            int gw, gd;
            int rd0 = shortestPathLen(tc.wh, tc.wv, pc.p0, 0);
            int rd1 = shortestPathLen(tc.wh, tc.wv, pc.p1, 1);
            if (raceDisjointGate(tc.wh, tc.wv, pc.p0, pc.p1, turn, rd0, rd1, gw, gd)) {
                g_gateDecided++;
                bool gateOk = (gw == oc.winner && gd == oc.dtm);
                if (!gateOk) reportMismatch("gate", tc.wh, tc.wv, pc.p0, pc.p1, turn, prod, dp, oc);
                CHECK(gateOk, "disjoint-region gate decision must equal the oracle");
            } else {
                g_gateRefused++;
            }
            if (tc.remembered.size() < 8) {
                tc.remembered.push_back({pc.p0, pc.p1, turn, oc.winner, oc.dtm});
            }
        }
    }
}

// ---------------------------------------------------------------------
// Empty-handed State construction with a correct Zobrist hash
// ---------------------------------------------------------------------

static State makeEmptyHanded(uint64_t wh, uint64_t wv, int p0, int p1, int turn) {
    State s;
    s.pawn[0] = (uint8_t)p0;
    s.pawn[1] = (uint8_t)p1;
    s.wallsH = wh;
    s.wallsV = wv;
    s.wallsLeft[0] = 0;
    s.wallsLeft[1] = 0;
    s.turn = turn;
    uint64_t h = zobrist().pawnKey[0][p0] ^ zobrist().pawnKey[1][p1];
    for (int slot = 0; slot < WS * WS; slot++) {
        if ((wh >> slot) & 1ull) h ^= zobrist().wallHKey[slot];
        if ((wv >> slot) & 1ull) h ^= zobrist().wallVKey[slot];
    }
    if (turn == 1) h ^= zobrist().turnKey;
    s.hash = h;
    return s;
}

// Score a root child exactly the way the empty-handed root branch of
// chooseMove does, using ORACLE outcomes as ground truth.
static int expectedChildScoreForRoot(const race_oracle::Table& orc,
                                     const State& root, const Move& m,
                                     bool parityAnchored, int contempt) {
    State ns = applyMove(root, m);
    int w = winner(ns);
    if (w != -1) return (w == ns.turn) ? -(SCORE_INF - 1) : (SCORE_INF - 1);
    RaceOutcome oc = orc.query(ns.pawn[0], ns.pawn[1], ns.turn);
    if (oc.winner == -1) return parityAnchored ? contempt : -contempt;
    int raw = RACE_SCORE_BASE - oc.dtm;
    int childScore = (oc.winner == ns.turn) ? raw : -raw;
    return -childScore;
}

// ---------------------------------------------------------------------
// Phases
// ---------------------------------------------------------------------

static constexpr int N_PLAYOUT_CASES = 110;
static constexpr int N_SYNTH_CASES = 470;
static constexpr int N_CONSTRUCTED_CASES = 24;

int main() {
    std::mt19937_64 rng(20260823ULL);
    std::mt19937_64 synthRng(0x5AFECAFEULL);
    std::vector<TopoCase> cases;
    std::vector<TopoCase> drawCases;

    // ---- build the corpus --------------------------------------------
    for (int i = 0; i < N_PLAYOUT_CASES; i++) {
        State s;
        if (!playoutToEmptyHands(rng, 400, s)) continue;
        TopoCase tc;
        tc.wh = s.wallsH;
        tc.wv = s.wallsV;
        collectPawnConfigs(rng, tc.wh, tc.wv, 3, tc.configs);
        if (!tc.configs.empty()) cases.push_back(tc);
    }
    size_t playoutCases = cases.size();

    for (int i = 0; i < N_SYNTH_CASES; i++) {
        TopoCase tc;
        buildTopoCase(rng, tc, /*synthetic=*/true, synthRng);
        if (!tc.configs.empty()) cases.push_back(tc);
    }

    // Constructed families: a full vertical/horizontal partition wall, and
    // the same wall with exactly one gap (regions touch through the gap ->
    // exercises the DP path right at the gate boundary).
    for (int orient = 0; orient < 2; orient++) {
        for (int line = 1; line < WS - 1 && cases.size() < (size_t)(playoutCases + N_SYNTH_CASES + N_CONSTRUCTED_CASES); line += 3) {
            for (int gapMode = 0; gapMode < 2; gapMode++) {
                uint64_t wh = 0, wv = 0;
                for (int k = 0; k < WS; k++) {
                    if (gapMode == 1 && k == 4) continue;
                    int slot = (orient == 0) ? slotIdx(k, line) : slotIdx(line, k);
                    if (orient == 0) wv |= (1ull << slot);
                    else wh |= (1ull << slot);
                }
                TopoCase tc;
                tc.wh = wh;
                tc.wv = wv;
                collectPawnConfigs(rng, wh, wv, 2, tc.configs);
                if (!tc.configs.empty()) cases.push_back(tc);
            }
        }
    }

    std::printf("(info) corpus: %zu topologies (%zu playout-born)\n",
                cases.size(), playoutCases);
    std::fflush(stdout);

    // ---- draw-root corpus --------------------------------------------
    // Oracle-level draws (perpetual pursuit) never appeared in the random
    // corpora above. Two sources instead: an exhaustive oracle scan of the
    // known synthetic draw topology, then single-wall mutations of it probed
    // with the SAME seed pairing (measured pass rate of the production
    // pre-filter there: about half of the candidates stay drawn, versus
    // roughly zero for random pawn placements). The oracle confirms every
    // candidate; production can only nominate, never decide.
    {
        const uint64_t baseH = 0x48000008000000ull;
        const uint64_t baseV = 0x8014020000022000ull;
        long oracleBuilds = 0;
        auto pawnOk = [&](uint64_t wh, uint64_t wv, int p0, int p1) {
            if (p0 == p1) return false;
            if (rowOf(p0) == GOAL_ROW[0] || rowOf(p1) == GOAL_ROW[1]) return false;
            return hasPathToGoal(wh, wv, p0, 0) && hasPathToGoal(wh, wv, p1, 1);
        };
        auto addCase = [&](uint64_t wh, uint64_t wv, int p0, int p1) {
            for (const TopoCase& tc : drawCases)
                if (tc.wh == wh && tc.wv == wv && tc.configs[0].p0 == p0 && tc.configs[0].p1 == p1) return;
            TopoCase tc;
            tc.wh = wh;
            tc.wv = wv;
            tc.configs.push_back({p0, p1, "draw"});
            drawCases.push_back(tc);
        };
        const int seedPairs[6][2] = {{57, 49}, {57, 48}, {57, 58}, {58, 48}, {58, 49}, {58, 57}};
        while (drawCases.size() < 32 && oracleBuilds < 140) {
            uint64_t wh = baseH, wv = baseV;
            int muts = 1 + (int)(synthRng() % 3);
            for (int m = 0; m < muts; m++) {
                if ((synthRng() % 2) == 0) {
                    uint64_t bits[2] = {wh, wv};
                    int total = __builtin_popcountll(bits[0]) + __builtin_popcountll(bits[1]);
                    if (total == 0) continue;
                    int k = (int)(synthRng() % (unsigned)total);
                    bool done = false;
                    for (int ori = 0; ori < 2 && !done; ori++) {
                        for (int slot = 0; slot < WS * WS; slot++) {
                            if (!((bits[ori] >> slot) & 1ull)) continue;
                            if (k-- == 0) {
                                if (ori == 0) wh &= ~(1ull << slot);
                                else wv &= ~(1ull << slot);
                                done = true;
                                break;
                            }
                        }
                    }
                } else {
                    int ori = (int)(synthRng() % 2);
                    int r = (int)(synthRng() % WS), c = (int)(synthRng() % WS);
                    if (wallSlotAvailable(wh, wv, ori, r, c)) {
                        int slot = slotIdx(r, c);
                        if (ori == 0) wh |= (1ull << slot);
                        else wv |= (1ull << slot);
                    }
                }
            }
            const int* pair = seedPairs[synthRng() % 6];
            if (!pawnOk(wh, wv, pair[0], pair[1])) continue;
            // cheap production nomination (either turn looking drawn suffices)
            if (resolveEmptyHandedEndgame(wh, wv, pair[0], pair[1], 0).winner != -1 &&
                resolveEmptyHandedEndgame(wh, wv, pair[0], pair[1], 1).winner != -1) continue;
            race_oracle::Table o;
            o.build(wh, wv);
            oracleBuilds++;
            if (o.query(pair[0], pair[1], 0).winner != -1 &&
                o.query(pair[0], pair[1], 1).winner != -1) continue;
            addCase(wh, wv, pair[0], pair[1]);
            // free extra draw configs on an accepted topology
            for (int extra = 0; extra < 40 && drawCases.size() < 36; extra++) {
                int q0 = (int)(synthRng() % (N * N)), q1 = (int)(synthRng() % (N * N));
                if (!pawnOk(wh, wv, q0, q1)) continue;
                if (o.query(q0, q1, 0).winner == -1 || o.query(q0, q1, 1).winner == -1) {
                    addCase(wh, wv, q0, q1);
                }
            }
        }
        std::printf("(info) draw corpus: %zu drawn-root cases (%ld oracle builds)\n",
                    drawCases.size(), oracleBuilds);
        std::fflush(stdout);
        CHECK(drawCases.size() >= 8, "draw hunting must find at least eight drawn-root cases");
        cases.insert(cases.end(), drawCases.begin(), drawCases.end());
    }

    // ---- phases A + B: agreement, then root-move optimality ----------
    Negamax engDefault;  // production defaults
    struct Combo { const char* name; bool tb; bool parity; };
    Combo combos[4] = {
        {"TB=on parity=off (production)", true, false},
        {"TB=on parity=on", true, true},
        {"TB=off parity=off", false, false},
        {"TB=off parity=on", false, true},
    };
    Negamax engCombos[4];
    for (int i = 0; i < 4; i++) {
        engCombos[i].setEndgameProgressTiebreak(combos[i].tb);
        engCombos[i].setParityAnchoredRaceDraw(combos[i].parity);
    }
    Negamax engNnue;  // NNUE-mode smoke, defaults

    long rootsWon = 0, rootsLost = 0, rootsDrawn = 0, rootChecks = 0;
    long valueOptimal = 0, scoreConventionHits = 0;
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
        else std::printf("(info) NNUE weights not found -- NNUE smoke skipped\n");
    }
    long nnueChecks = 0, nnueAgree = 0;

    for (size_t ci = 0; ci < cases.size(); ci++) {
        TopoCase& tc = cases[ci];
        race_oracle::Table orc;
        orc.build(tc.wh, tc.wv);

        // Phase A
        checkTopology("A", tc, orc);

        // Phase B: every (config, turn) becomes an empty-handed root.
        for (const PawnConfig& pc : tc.configs) {
            for (int turn = 0; turn < 2; turn++) {
                RaceOutcome rootOc = orc.query(pc.p0, pc.p1, turn);
                if (rootOc.winner == 0) rootsWon++;
                else if (rootOc.winner == 1) rootsLost++;
                else rootsDrawn++;

                State root = makeEmptyHanded(tc.wh, tc.wv, pc.p0, pc.p1, turn);
                MoveList rootMoves = legalMoves(root);
                CHECK(!rootMoves.empty(), "empty-handed root must have at least one pawn move");

                for (int cb = 0; cb < 4; cb++) {
                    SearchStats st;
                    Move chosen = engCombos[cb].chooseMove(root, 8, 200, st);
                    bool legal = false;
                    for (size_t i = 0; i < rootMoves.size(); i++)
                        if (rootMoves[i] == chosen) { legal = true; break; }
                    if (!legal) {
                        std::printf("  ILEGAL root(wallsH=0x%llx,wallsV=0x%llx,p0=%d,p1=%d,t=%d) combo=%s\n",
                                    (unsigned long long)tc.wh, (unsigned long long)tc.wv,
                                    pc.p0, pc.p1, turn, combos[cb].name);
                    }
                    CHECK(legal, "chooseMove must return a legal move at an empty-handed root");
                    rootChecks++;

                    int chosenVal = INT32_MIN;
                    int bestVal = INT32_MIN;
                    for (size_t i = 0; i < rootMoves.size(); i++) {
                        int v = expectedChildScoreForRoot(orc, root, rootMoves[i],
                                                          combos[cb].parity,
                                                          engCombos[cb].getContempt());
                        if (rootMoves[i] == chosen) chosenVal = v;
                        if (i == 0 || v > bestVal) bestVal = v;
                    }
                    if (chosenVal == bestVal) valueOptimal++;
                    else {
                        std::printf("  SUBOTIMO root(wallsH=0x%llx,wallsV=0x%llx,p0=%d,p1=%d,t=%d) combo=%s chosen=%d best=%d\n",
                                    (unsigned long long)tc.wh, (unsigned long long)tc.wv,
                                    pc.p0, pc.p1, turn, combos[cb].name, chosenVal, bestVal);
                    }
                    CHECK(chosenVal == bestVal,
                          "chosen move must achieve the oracle-best child value");
                    CHECK(st.score == bestVal,
                          "reported root score must equal the oracle-best child value");
                    if (st.score == bestVal) scoreConventionHits++;
                }

                if (haveWeights) {
                    SearchStats st;
                    Move chosen = engNnue.chooseMove(root, 8, 200, st);
                    Move chosenHeur = engDefault.chooseMove(root, 8, 200, st);
                    nnueChecks++;
                    if (chosen == chosenHeur) nnueAgree++;
                    else CHECK(false, "NNUE mode must pick the same empty-handed root move (solver regime)");
                }
            }
        }
    }

    std::printf("(info) A: %ld solver comparisons, gate decided %ld / refused %ld\n",
                g_comparisons, g_gateDecided, g_gateRefused);
    CHECK(g_comparisons > 3000, "phase A coverage: expected thousands of comparisons");
    CHECK(g_gateDecided > 50, "phase A coverage: the cheap gate must be exercised");
    std::printf("(info) B: %ld roots (win %ld / lose %ld / draw %ld), %ld optimality checks, "
                "%ld value-optimal, %ld score-convention hits, NNUE agree %ld/%ld\n",
                rootsWon + rootsLost + rootsDrawn, rootsWon, rootsLost, rootsDrawn,
                rootChecks, valueOptimal, scoreConventionHits, nnueAgree, nnueChecks);
    CHECK(rootsDrawn >= 25, "phase B coverage: drawn roots must appear in the corpus");
    CHECK(rootChecks >= 4 * 200, "phase B coverage: expected hundreds of roots x 4 combos");

    // ---- phase C: budget-exhaustion fallback --------------------------
    {
        std::mt19937_64 crng(777ULL);
        double savedUsed = g_raceExactUsedUs;
        long deltaCount = 0, tried = 0;
        for (size_t ci = 0; ci < cases.size() && tried < 24; ci += 17) {
            TopoCase& tc = cases[ci];
            const PawnConfig& pc = tc.configs[0];
            State root = makeEmptyHanded(tc.wh, tc.wv, pc.p0, pc.p1, 0);

            Negamax eng;
            g_raceExactUsedUs = 2e18;  // above any budget any entry point sets
            eng.clearTT();
            SearchStats st1;
            int scHeur = eng.searchShallow(root, 6, st1);
            CHECK(scHeur > -SCORE_INF && scHeur < SCORE_INF,
                  "budget-exhausted search must return a finite sane score");

            g_raceExactUsedUs = 0.0;
            eng.clearTT();
            SearchStats st2;
            int scExact = eng.searchShallow(root, 6, st2);
            CHECK(scExact > -SCORE_INF && scExact < SCORE_INF,
                  "unlimited-budget search must return a finite sane score");
            if (scHeur != scExact) deltaCount++;
            tried++;
        }
        g_raceExactUsedUs = savedUsed;
        std::printf("(info) C: forced-fallback searches tried=%ld, score deltas vs exact=%ld "
                    "(deltas are allowed: heuristic vs exact regimes)\n", tried, deltaCount);

        // statistical tiny-budget chooseMove: legality only
        long tinyTried = 0;
        std::mt19937_64 trng(555ULL);
        for (size_t ci = 0; ci < cases.size(); ci++) {
            TopoCase& tc = cases[ci];
            const PawnConfig& pc = tc.configs[(trng() % tc.configs.size())];
            int turn = (int)(trng() % 2);
            State root = makeEmptyHanded(tc.wh, tc.wv, pc.p0, pc.p1, turn);
            Negamax eng;
            SearchStats st;
            Move m = eng.chooseMove(root, 6, 1 + (int)(trng() % 4), st);
            MoveList ms = legalMoves(root);
            bool legal = false;
            for (size_t i = 0; i < ms.size(); i++) if (ms[i] == m) { legal = true; break; }
            if (!legal) CHECK(false, "tiny-budget chooseMove returned an illegal move");
            tinyTried++;
            if (tinyTried >= 60) break;
        }
        std::printf("(info) C: tiny-budget chooseMove legality checks=%ld\n", tinyTried);
    }

    // ---- phase D: cache soundness under eviction pressure -------------
    {
        std::mt19937_64 drng(31337ULL);
        const size_t K = std::min<size_t>(cases.size(), 320);
        uint64_t hitsBefore = g_raceCacheHits, missesBefore = g_raceCacheMisses;
        long checked = 0, bad = 0;
        // eviction pool of throwaway topologies, larger than the 1024-slot table
        std::vector<std::pair<uint64_t, uint64_t>> pool;
        pool.reserve(1600);
        while (pool.size() < 1600) {
            uint64_t wh, wv;
            synthWalls(drng, 6 + (int)(drng() % 20), wh, wv);
            pool.push_back({wh, wv});
        }
        for (int round = 0; round < 3; round++) {
            for (size_t k = 0; k < K; k++) {
                TopoCase& tc = cases[k];
                if (tc.remembered.empty()) continue;
                const auto& rem = tc.remembered[(round + k) % tc.remembered.size()];
                RaceOutcome prod = resolveEmptyHandedEndgame(tc.wh, tc.wv, rem.p0, rem.p1, rem.t);
                checked++;
                if (!(prod.winner == rem.win && prod.dtm == rem.dtm)) {
                    bad++;
                    std::printf("  CACHE-DIVERGENCIA wallsH=0x%llx wallsV=0x%llx p0=%d p1=%d t=%d "
                                "prod=(%d,%d) oracle=(%d,%d)\n",
                                (unsigned long long)tc.wh, (unsigned long long)tc.wv,
                                rem.p0, rem.p1, rem.t, prod.winner, prod.dtm, rem.win, rem.dtm);
                }
                // two throwaway resolves per check: constant eviction pressure
                auto& pp = pool[(round * K + 2 * k) % pool.size()];
                auto& qq = pool[(round * K + 2 * k + 1) % pool.size()];
                (void)resolveEmptyHandedEndgame(pp.first, pp.second, 0, 80, 0);
                (void)resolveEmptyHandedEndgame(qq.first, qq.second, 40, 40 + 1, 1);
            }
        }
        CHECK(bad == 0, "production cache must be transparent under eviction pressure");
        std::printf("(info) D: %ld interleaved cache checks (bad=%ld), race cache hits %llu->%llu, misses %llu->%llu\n",
                    checked, bad,
                    (unsigned long long)hitsBefore, (unsigned long long)g_raceCacheHits,
                    (unsigned long long)missesBefore, (unsigned long long)g_raceCacheMisses);
        CHECK(g_raceCacheMisses > missesBefore + 1000, "cache stress must actually miss (evictions happening)");
    }

    // ---- phase E: degenerate inputs outside the real-play invariant ---
    {
        // E1: pawn already on its own goal row -> immediate win, dtm 0.
        for (int turn = 0; turn < 2; turn++) {
            uint64_t wh = 0, wv = 0;
            int p0 = cellIdx(N - 1, 4);
            int p1 = cellIdx(4, 8);
            race_oracle::Table orc;
            orc.build(wh, wv);
            RaceOutcome oc = orc.query(p0, p1, turn);
            RaceOutcome prod = resolveEmptyHandedEndgame(wh, wv, p0, p1, turn);
            RaceOutcome dp = raceExactDTM(wh, wv, p0, p1, turn);
            bool ok = prod.winner == oc.winner && prod.dtm == oc.dtm &&
                      dp.winner == oc.winner && dp.dtm == oc.dtm &&
                      oc.winner == 0 && oc.dtm == 0;
            if (!ok) reportMismatch("E1", wh, wv, p0, p1, turn, prod, dp, oc);
            CHECK(ok, "pawn on its goal row must be an immediate win with dtm 0 in every path");
        }
        // E1b: pawn on its goal row BEHIND A PARTITION WALL, so the two
        // regions are disjoint and the cheap gate actually fires (the open
        // board variant of E1 never reaches the gate). Truth: immediate
        // win, dtm 0. rawDist0 == 0 breaks the gate's ply arithmetic.
        for (int turn = 0; turn < 2; turn++) {
            uint64_t wh = 0, wv = 0;
            for (int r = 0; r < WS; r++) wv |= (1ull << slotIdx(r, 4));
            int p0 = cellIdx(N - 1, 1);  // on its own goal row, left region
            int p1 = cellIdx(3, 6);      // right region
            race_oracle::Table orc;
            orc.build(wh, wv);
            RaceOutcome oc = orc.query(p0, p1, turn);
            RaceOutcome prod = resolveEmptyHandedEndgame(wh, wv, p0, p1, turn);
            bool ok = prod.winner == oc.winner && prod.dtm == oc.dtm &&
                      oc.winner == 0 && oc.dtm == 0;
            if (!ok) reportMismatch("E1b", wh, wv, p0, p1, turn, prod, prod, oc);
            CHECK(ok, "partitioned pawn-on-goal must stay an immediate dtm-0 win through the gate path");
            int gw, gd;
            int rd0 = shortestPathLen(wh, wv, p0, 0);
            CHECK(rd0 == 0, "E1b setup: pawn on goal row must have raw distance 0");
            if (raceDisjointGate(wh, wv, p0, p1, turn, rd0,
                                 shortestPathLen(wh, wv, p1, 1), gw, gd)) {
                std::printf("  NOTA E1b: gate decided on degenerate rawDist before fix (w=%d dtm=%d)\n", gw, gd);
            }
        }
        // E2: one pawn sealed in a minimal 1-cell pocket (4 wall slots), the
        // other free -> the free side wins. rawDist of the sealed side is
        // -1: the gate must refuse and let the DP answer.
        {
            uint64_t wh = 0, wv = 0;
            // pocket around (4,4): H(3,4)+H(4,3) block north/south,
            // V(4,3)+V(3,4) block west/east -- four slots, nothing else.
            wh |= (1ull << slotIdx(3, 4));
            wh |= (1ull << slotIdx(4, 3));
            wv |= (1ull << slotIdx(3, 4));
            wv |= (1ull << slotIdx(4, 3));
            int p0 = cellIdx(4, 4);  // sealed
            int p1 = cellIdx(2, 7);  // free
            race_oracle::Table orc;
            orc.build(wh, wv);
            for (int turn = 0; turn < 2; turn++) {
                RaceOutcome oc = orc.query(p0, p1, turn);
                RaceOutcome prod = resolveEmptyHandedEndgame(wh, wv, p0, p1, turn);
                RaceOutcome dp = raceExactDTM(wh, wv, p0, p1, turn);
                // The sealed side has zero legal moves, so no side can force
                // a finish: every exact path must agree on the draw. (The
                // pre-fix gate fabricated a win for the sealed side here.)
                bool ok = prod.winner == oc.winner && prod.dtm == oc.dtm &&
                          dp.winner == oc.winner && dp.dtm == oc.dtm &&
                          oc.winner == -1;
                if (!ok) reportMismatch("E2", wh, wv, p0, p1, turn, prod, dp, oc);
                CHECK(ok, "sealed losing pawn must not flip the result (gate must refuse on unreachable goal)");
            }
        }
        // E3: both pawns sealed in disjoint minimal pockets -> neither ever
        // finishes -> draw. The gate must not fabricate a winner from the
        // negative ply arithmetic of two rawDist == -1 sides.
        {
            uint64_t wh = 0, wv = 0;
            auto sealPocket = [](uint64_t& wh, uint64_t& wv, int R, int C) {
                wh |= (1ull << slotIdx(R - 1, C)); wh |= (1ull << slotIdx(R, C - 1));
                wv |= (1ull << slotIdx(R, C - 1)); wv |= (1ull << slotIdx(R - 1, C));
            };
            sealPocket(wh, wv, 2, 2);
            sealPocket(wh, wv, 6, 6);
            int p0 = cellIdx(2, 2), p1 = cellIdx(6, 6);
            race_oracle::Table orc;
            orc.build(wh, wv);
            for (int turn = 0; turn < 2; turn++) {
                RaceOutcome oc = orc.query(p0, p1, turn);
                RaceOutcome prod = resolveEmptyHandedEndgame(wh, wv, p0, p1, turn);
                RaceOutcome dp = raceExactDTM(wh, wv, p0, p1, turn);
                bool ok = prod.winner == -1 && dp.winner == -1 && oc.winner == -1;
                if (!ok) reportMismatch("E3", wh, wv, p0, p1, turn, prod, dp, oc);
                CHECK(ok, "two sealed pockets must be a draw in every path");
            }
        }
        // E4: randomized degenerate sweep over real corpus topologies. For
        // each sampled topology, probe (a) a pawn parked on its own goal
        // row (rawDist == 0) against a live opponent, and (b) a pawn sealed
        // in a fresh 4-wall pocket carved out of the topology (rawDist ==
        // -1). Every answer must come from Service B and match the oracle;
        // the gate must refuse to decide.
        {
            std::mt19937_64 erng(987654321ULL);
            long degChecks = 0, degGateRefused = 0;
            size_t stride = std::max<size_t>(1, cases.size() / 60);
            for (size_t ci = 0; ci < cases.size(); ci += stride) {
                TopoCase& tc = cases[ci];
                race_oracle::Table orc;
                orc.build(tc.wh, tc.wv);
                // (a) root-side pawn already on its goal row: rawDist0 == 0.
                // Truth is always an immediate dtm-0 win for player 0.
                int goalCell0 = cellIdx(GOAL_ROW[0], (int)(erng() % N));
                for (const PawnConfig& pc : tc.configs) {
                    if (pc.p1 == goalCell0 || degChecks >= 400) continue;
                    for (int turn = 0; turn < 2 && degChecks < 400; turn++) {
                        RaceOutcome oc = orc.query(goalCell0, pc.p1, turn);
                        RaceOutcome prod = resolveEmptyHandedEndgame(tc.wh, tc.wv, goalCell0, pc.p1, turn);
                        bool ok = prod.winner == oc.winner && prod.dtm == oc.dtm &&
                                  oc.winner == 0 && oc.dtm == 0;
                        if (!ok) reportMismatch("E4a", tc.wh, tc.wv, goalCell0, pc.p1, turn, prod, prod, oc);
                        CHECK(ok, "E4a: pawn on its goal row must stay a dtm-0 win");
                        int gw, gd;
                        bool decided = raceDisjointGate(tc.wh, tc.wv, goalCell0, pc.p1, turn,
                                                        0, shortestPathLen(tc.wh, tc.wv, pc.p1, 1), gw, gd);
                        if (!decided) degGateRefused++;
                        else CHECK(false, "E4a: gate must refuse on rawDist 0");
                        degChecks++;
                    }
                }
                // (b) fresh sealed pocket around a random interior cell:
                // rawDist(sealed) == -1. Truth comes from the oracle.
                int R = 2 + (int)(erng() % (N - 3));
                int C = 2 + (int)(erng() % (N - 3));
                uint64_t needH = (1ull << slotIdx(R - 1, C)) | (1ull << slotIdx(R, C - 1));
                uint64_t needV = (1ull << slotIdx(R, C - 1)) | (1ull << slotIdx(R - 1, C));
                if (((tc.wh & needH) | (tc.wv & needV)) == 0) {
                    uint64_t wh2 = tc.wh | needH;
                    uint64_t wv2 = tc.wv | needV;
                    int sealed = cellIdx(R, C);
                    if (!hasPathToGoal(wh2, wv2, sealed, 0)) {
                        race_oracle::Table orc2;
                        orc2.build(wh2, wv2);
                        for (const PawnConfig& pc : tc.configs) {
                            if (pc.p1 == sealed || degChecks >= 600) continue;
                            for (int turn = 0; turn < 2 && degChecks < 600; turn++) {
                                RaceOutcome oc = orc2.query(sealed, pc.p1, turn);
                                RaceOutcome prod = resolveEmptyHandedEndgame(wh2, wv2, sealed, pc.p1, turn);
                                RaceOutcome dp = raceExactDTM(wh2, wv2, sealed, pc.p1, turn);
                                bool ok = prod.winner == oc.winner && prod.dtm == oc.dtm &&
                                          dp.winner == oc.winner && dp.dtm == oc.dtm;
                                if (!ok) reportMismatch("E4b", wh2, wv2, sealed, pc.p1, turn, prod, dp, oc);
                                CHECK(ok, "E4b: sealed-pocket state must match the oracle");
                                degChecks++;
                            }
                        }
                    }
                }
            }
            std::printf("(info) E4: %ld randomized degenerate checks, gate refusals %ld\n",
                        degChecks, degGateRefused);
            CHECK(degChecks >= 300, "phase E4 coverage: expected hundreds of degenerate probes");
        }
    }

    // ---- phase F: long-game differential (compact, deterministic) -----
    {
        std::mt19937_64 frng(424242ULL);
        const int F_GAMES = 14;
        long fDecisive = 0, fDrawn = 0, fBad = 0;
        Negamax eng;
        for (int g = 0; g < F_GAMES; ) {
            State start;
            if (!playoutToEmptyHands(frng, 400, start)) continue;
            race_oracle::Table orc;
            orc.build(start.wallsH, start.wallsV);
            RaceOutcome pred = orc.query(start.pawn[0], start.pawn[1], start.turn);
            State cur = start;
            int played = 0;
            bool illegal = false;
            while (winner(cur) == -1 && played < 240) {
                SearchStats st;
                Move m = eng.chooseMove(cur, 12, 20, st);
                MoveList ms = legalMoves(cur);
                bool legal = false;
                for (size_t i = 0; i < ms.size(); i++) if (ms[i] == m) { legal = true; break; }
                if (!legal) { illegal = true; break; }
                cur = applyMove(cur, m);
                played++;
                if (pred.winner == -1 && played >= 160) break;
            }
            g++;
            if (illegal) { fBad++; continue; }
            if (pred.winner == -1) {
                fDrawn++;
                if (winner(cur) != -1) fBad++;
            } else {
                fDecisive++;
                if (winner(cur) != pred.winner || played != pred.dtm) {
                    fBad++;
                    std::printf("  DIFERENCIAL wallsH=0x%llx wallsV=0x%llx p0=%d p1=%d t=%d "
                                "pred=(%d,%d) actual=(%d,%d)\n",
                                (unsigned long long)start.wallsH, (unsigned long long)start.wallsV,
                                start.pawn[0], start.pawn[1], start.turn,
                                pred.winner, pred.dtm, winner(cur), played);
                }
            }
        }
        std::printf("(info) F: %ld decisive + %ld drawn differential games, contradictions %ld\n",
                    fDecisive, fDrawn, fBad);
        CHECK(fDecisive >= 8, "phase F coverage: expected mostly decisive starts");
        CHECK(fBad == 0, "played-out race must match the oracle winner and length exactly");

        // the known synthetic pursuit topology must never finish
        {
            const uint64_t dH = 0x48000008000000ull;
            const uint64_t dV = 0x8014020000022000ull;
            race_oracle::Table orc;
            orc.build(dH, dV);
            const int pairs[3][2] = {{57, 49}, {58, 48}, {58, 49}};
            for (const auto& pr : pairs) {
                for (int t = 0; t < 2; t++) {
                    if (orc.query(pr[0], pr[1], t).winner != -1) continue;
                    State cur = makeEmptyHanded(dH, dV, pr[0], pr[1], t);
                    for (int ply = 0; ply < 160 && winner(cur) == -1; ply++) {
                        SearchStats st;
                        Move m = eng.chooseMove(cur, 12, 20, st);
                        cur = applyMove(cur, m);
                    }
                    bool ok = winner(cur) == -1;
                    if (!ok)
                        std::printf("  PERSEGUICAO pair=(%d,%d) t=%d winner=%d\n",
                                    pr[0], pr[1], t, winner(cur));
                    CHECK(ok, "drawn pursuit must never finish under optimal play");
                }
            }
        }
    }

    if (failures == 0) {
        std::printf("OK -- test_endgame_race_fuzz passou (oracle agreement 100%%, raiz otima em todos os combos)\n");
        return 0;
    }
    std::printf("%d FALHA(S)\n", failures);
    return 1;
}
