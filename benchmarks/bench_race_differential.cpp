// bench_race_differential.cpp -- differential long-game check of the
// empty-handed regime (inv/race-fuzz). Not wired into the build scripts
// (same status as tools/arena/arena.cpp); compile standalone:
//
//   g++ -O3 -std=c++17 -march=native -mavx2 -mfma -Isrc -Itests \
//       -o bin/bench_race_differential.exe benchmarks/bench_race_differential.cpp
//
// Method: generate hands-empty positions from random playouts, ask the
// independent oracle (tests/race_oracle.hpp) for ground truth, then play
// engine-vs-engine with Negamax::chooseMove from that position. Inside the
// empty-handed regime the root branch of chooseMove maximizes EXACT solver
// values, so both sides play game-theoretically optimal pawn races:
//   - a position the oracle scores as a win for W in D plies must end with
//     winner W in EXACTLY D plies (optimal attacker shortens, optimal
//     defender lengthens, and any tie among exactly-equal children keeps
//     the same length);
//   - a position scored as a draw must never produce a winner;
//   - every emitted move must be legal.
// Any contradiction is a real bug (value bug, move-choice bug, or cache
// corruption). Four workers stay within the investigation thread cap.
#include <cstdio>
#include <atomic>
#include <mutex>
#include <random>
#include <thread>
#include <vector>
#include "rules.hpp"
#include "endgame_race.hpp"
#include "search.hpp"
#include "race_oracle.hpp"
using namespace qr;

static constexpr int GAMES_PER_WORKER = 90;
static constexpr int WORKERS = 4;
static constexpr int MOVE_BUDGET_MS = 20;
static constexpr int DRAW_PLY_CAP = 160;

// Build a hands-empty State with a correct Zobrist hash (rules.hpp has no
// public constructor for synthetic states).
static State makeState(uint64_t wh, uint64_t wv, int p0, int p1, int turn) {
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

struct Result {
    long games = 0, decisive = 0, drawn = 0, skipped = 0;
    long contradictions = 0, illegalMoves = 0;
    long lenExact = 0, lenMismatch = 0;
    long maxDecisiveLen = 0, maxDrawLen = 0;
};

int main() {
    std::atomic<long> totalContradictions{0};
    std::vector<Result> results(WORKERS);
    std::mutex printMutex;

    auto worker = [&](int w) {
        Result& R = results[w];
        std::mt19937_64 rng(0xBEEF0000ull + 7919ull * (unsigned)w);
        Negamax eng;  // heuristic mode: the empty-handed branch never evaluates
        for (int g = 0; g < GAMES_PER_WORKER; g++) {
            State start;
            int attempts = 0;
            while (!playoutToEmptyHands(rng, start)) {
                if (++attempts > 50) break;
            }
            if (attempts > 50) { R.skipped++; continue; }

            race_oracle::Table orc;
            orc.build(start.wallsH, start.wallsV);
            RaceOutcome pred = orc.query(start.pawn[0], start.pawn[1], start.turn);

            // play it out, both sides through the exact-solver root branch
            State cur = start;
            int played = 0;
            bool contradiction = false;
            while (winner(cur) == -1 && played < DRAW_PLY_CAP + 80) {
                SearchStats st;
                Move m = eng.chooseMove(cur, 12, MOVE_BUDGET_MS, st);
                MoveList ms = legalMoves(cur);
                bool legal = false;
                for (size_t i = 0; i < ms.size(); i++) if (ms[i] == m) { legal = true; break; }
                if (!legal) {
                    R.illegalMoves++;
                    std::lock_guard<std::mutex> lk(printMutex);
                    std::printf("ILLEGAL w=%d g=%d wallsH=0x%llx wallsV=0x%llx p0=%d p1=%d t=%d\n",
                                w, g, (unsigned long long)cur.wallsH, (unsigned long long)cur.wallsV,
                                cur.pawn[0], cur.pawn[1], cur.turn);
                    contradiction = true;
                    break;
                }
                cur = applyMove(cur, m);
                played++;
                if (pred.winner == -1 && played >= DRAW_PLY_CAP) break;
            }
            if (contradiction) { R.contradictions++; continue; }

            int actualWinner = winner(cur);
            R.games++;
            if (pred.winner == -1) {
                R.drawn++;
                if (actualWinner != -1) {
                    R.contradictions++;
                    std::lock_guard<std::mutex> lk(printMutex);
                    std::printf("DRAW-CONTRADICTION w=%d g=%d predicted draw but winner=%d after %d plies "
                                "wallsH=0x%llx wallsV=0x%llx p0=%d p1=%d t=%d\n",
                                w, g, actualWinner, played,
                                (unsigned long long)start.wallsH, (unsigned long long)start.wallsV,
                                start.pawn[0], start.pawn[1], start.turn);
                } else if (played > R.maxDrawLen) R.maxDrawLen = played;
            } else {
                R.decisive++;
                if (played > R.maxDecisiveLen) R.maxDecisiveLen = played;
                bool ok = actualWinner == pred.winner && played == pred.dtm;
                if (ok) R.lenExact++;
                else {
                    R.lenMismatch++;
                    R.contradictions++;
                    std::lock_guard<std::mutex> lk(printMutex);
                    std::printf("DECISIVE-MISMATCH w=%d g=%d pred=(%d,%d) actual=(%d,%d) "
                                "wallsH=0x%llx wallsV=0x%llx p0=%d p1=%d t=%d\n",
                                w, g, pred.winner, pred.dtm, actualWinner, played,
                                (unsigned long long)start.wallsH, (unsigned long long)start.wallsV,
                                start.pawn[0], start.pawn[1], start.turn);
                }
            }
        }
        totalContradictions += R.contradictions;
    };

    std::vector<std::thread> pool;
    for (int t = 0; t < WORKERS; t++) pool.emplace_back(worker, t);
    for (auto& th : pool) th.join();

    Result T;
    for (const Result& r : results) {
        T.games += r.games; T.decisive += r.decisive; T.drawn += r.drawn;
        T.skipped += r.skipped; T.contradictions += r.contradictions;
        T.illegalMoves += r.illegalMoves; T.lenExact += r.lenExact;
        T.lenMismatch += r.lenMismatch;
        if (r.maxDecisiveLen > T.maxDecisiveLen) T.maxDecisiveLen = r.maxDecisiveLen;
        if (r.maxDrawLen > T.maxDrawLen) T.maxDrawLen = r.maxDrawLen;
    }

    // Draw stage: random playouts never land on oracle-drawn states, so
    // play out the known synthetic pursuit topology for all six drawn
    // pawn pairs and both turns. Optimal play cannot force a finish here,
    // so the only acceptable outcome is "no winner ever".
    {
        const uint64_t dH = 0x48000008000000ull;
        const uint64_t dV = 0x8014020000022000ull;
        const int pairs[6][2] = {{57, 49}, {57, 48}, {57, 58}, {58, 48}, {58, 49}, {58, 57}};
        Negamax eng;
        long drawGames = 0, drawBad = 0;
        race_oracle::Table orc;
        orc.build(dH, dV);
        for (const auto& pr : pairs) {
            for (int t = 0; t < 2; t++) {
                if (!hasPathToGoal(dH, dV, pr[0], 0) || !hasPathToGoal(dH, dV, pr[1], 1)) continue;
                RaceOutcome pred = orc.query(pr[0], pr[1], t);
                if (pred.winner != -1) continue;  // only true draws
                State cur = makeState(dH, dV, pr[0], pr[1], t);
                bool bad = false;
                for (int ply = 0; ply < DRAW_PLY_CAP && winner(cur) == -1; ply++) {
                    SearchStats st;
                    Move m = eng.chooseMove(cur, 12, MOVE_BUDGET_MS, st);
                    MoveList ms = legalMoves(cur);
                    bool legal = false;
                    for (size_t i = 0; i < ms.size(); i++) if (ms[i] == m) { legal = true; break; }
                    if (!legal) { bad = true; break; }
                    cur = applyMove(cur, m);
                }
                drawGames++;
                if (bad || winner(cur) != -1) {
                    drawBad++;
                    std::printf("DRAW-STAGE CONTRADICTION pair=(%d,%d) t=%d winner=%d\n",
                                pr[0], pr[1], t, winner(cur));
                }
            }
        }
        T.games += drawGames;
        T.drawn += drawGames;
        T.contradictions += drawBad;
        std::printf("draw stage: %ld drawn-race games, contradictions %ld\n", drawGames, drawBad);
    }

    std::printf("differential: %ld games (%ld decisive, %ld drawn, %ld setup-skips)\n",
                T.games, T.decisive, T.drawn, T.skipped);
    std::printf("  decisive length == oracle dtm: %ld/%ld ; mismatches %ld\n",
                T.lenExact, T.decisive, T.lenMismatch);
    std::printf("  illegal moves %ld ; contradictions %ld ; longest decisive %d plies, longest draw-capped %d plies\n",
                T.illegalMoves, T.contradictions, T.maxDecisiveLen, T.maxDrawLen);
    bool ok = (T.contradictions == 0 && T.illegalMoves == 0 &&
               T.lenMismatch == 0 && T.games >= 300 && T.decisive >= 100);
    if (ok) {
        std::printf("OK -- bench_race_differential: play matches oracle exactly\n");
        return 0;
    }
    std::printf("FALHOU -- differential contradictions found\n");
    return 1;
}
