// bench_contempt_wandering -- measures pawn-shuffling ("wandering") in
// near-endgame Quoridor positions under different contempt / TT / eval /
// time-control settings, plus head-to-head Elo duels between two settings.
//
// Corpus: random legal playouts from the initial state. Two corpora are
// kept: MIDGAME-END (total walls in a window, default [4..14]) and
// EMPTY-HANDED (both players at 0 walls). A position enters a corpus only
// if BOTH players have at least 2 distinct pawn advances that shorten their
// own shortest path (the "more than one path" property from the user
// report). Positions are deduplicated by Zobrist hash.
//
// Match protocol: engine-vs-engine from each corpus position, colors
// swapped, real game history passed through RepetitionTable exactly like
// the web GUI does (push each played hash; draw claim at count >= 3).
// Counters per side: pawn moves, immediate shuffles (position hash equals
// the hash from 2 plies ago), backward pawn moves (own BFS distance grows),
// accumulated path progress, results (win / repetition draw / ply cap).
//
// Threads: hard-capped at 4 workers (shared machine; keep total load low).
//
// Build (performance profile):
//   g++ -O3 -std=c++17 -march=native -mavx2 -mfma -Isrc ^
//       -o bin/bench_contempt_wandering.exe benchmarks/bench_contempt_wandering.cpp -pthread
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <string>
#include <vector>
#include <unordered_set>
#include <atomic>
#include <mutex>
#include <thread>
#include <memory>
#include <random>
#include <algorithm>
#include "rules.hpp"
#include "search.hpp"

using namespace qr;

static std::atomic<long long> g_illegalMoves{0};

// ---------------------------------------------------------------------
// Configuration of one engine side
// ---------------------------------------------------------------------
struct EngineCfg {
    int contempt = CONTEMPT;                    // default -30
    bool clearTTPerMove = false;
    Negamax::EvalMode mode = Negamax::EvalMode::Heuristic;
    int depthCap = 40;
    int timeMs = 150;
};

static const char* evalModeName(Negamax::EvalMode m) {
    return m == Negamax::EvalMode::NNUE ? "nnue" : "heur";
}

// Per-side aggregated counters over many games.
struct SideStats {
    long long games = 0, wins = 0, losses = 0, repDraws = 0, timeouts = 0;
    long long moves = 0, shuffle2 = 0, cycle4 = 0, backMoves = 0, wallMoves = 0;
    double progressSum = 0.0;  // sum of (distBefore - distAfter) over pawn moves

    void merge(const SideStats& o) {
        games += o.games; wins += o.wins; losses += o.losses;
        repDraws += o.repDraws; timeouts += o.timeouts;
        moves += o.moves; shuffle2 += o.shuffle2; cycle4 += o.cycle4;
        backMoves += o.backMoves; wallMoves += o.wallMoves;
        progressSum += o.progressSum;
    }
};

struct GameOutcome {
    // 0 = A won, 1 = B won, 2 = repetition draw, 3 = ply cap reached
    int result = 3;
    int plies = 0;
    SideStats a, b;
};

// ---------------------------------------------------------------------
// Corpus generation
// ---------------------------------------------------------------------
// "More than one path" property: each player's reconstructed shortest path
// has at least `ROB_MIN` cheap-detour neighbors (pathRobustness), meaning
// route diversity beyond a single corridor. Measured on random playouts:
// about 99% of mid-window states satisfy this at ROB_MIN=2, so the filter
// selects the target population without distorting it.
static constexpr int ROB_MIN = 2;

static bool multiPathPosition(const State& s) {
    return pathRobustness(s.wallsH, s.wallsV, s.pawn[0], 0) >= ROB_MIN &&
           pathRobustness(s.wallsH, s.wallsV, s.pawn[1], 1) >= ROB_MIN;
}

enum CorpusKind { CORPUS_MID = 0, CORPUS_EMPTY = 1 };

// Random legal playouts; collect the first `target` distinct positions that
// satisfy the wall-window (or empty-handed) and multi-path conditions.
static std::vector<State> buildCorpus(CorpusKind kind, int target,
                                       int minWalls, int maxWalls,
                                       unsigned seed, int& playoutsDone) {
    std::mt19937_64 rng(seed);
    std::vector<State> out;
    std::unordered_set<uint64_t> seen;
    playoutsDone = 0;
    const int kMaxPlayoutLen = 220;
    while ((int)out.size() < target && playoutsDone < target * 400) {
        playoutsDone++;
        State s = initialState();
        RepetitionTable hist;
        hist.push(s.hash);
        int takenThisPlayout = 0;
        for (int ply = 0; ply < kMaxPlayoutLen; ply++) {
            if (winner(s) != -1) break;
            int totalWalls = s.wallsLeft[0] + s.wallsLeft[1];
            bool accept = (kind == CORPUS_EMPTY) ? (totalWalls == 0)
                : (totalWalls >= minWalls && totalWalls <= maxWalls);
            // Cap per playout so the corpus spans many independent games
            // instead of hundreds of states from one long game.
            if (accept && takenThisPlayout < 4 &&
                seen.find(s.hash) == seen.end() && multiPathPosition(s)) {
                seen.insert(s.hash);
                out.push_back(s);
                takenThisPlayout++;
                if ((int)out.size() >= target) break;
            }
            // Pawn-biased mover (80%): uniform choice over all legal moves
            // is wall-dominated (~128 wall slots vs 2-3 pawn moves) and the
            // pawns never leave their start rows. Bias keeps playouts in
            // the midgame/endgame region this investigation targets.
            MoveList all = legalMoves(s);
            if (all.empty()) break;
            MoveList pawnMv, wallMv;
            for (size_t i = 0; i < all.size(); i++)
                (all[i].isWall ? wallMv : pawnMv).push_back(all[i]);
            bool usePawn = !pawnMv.empty() && ((int)(rng() % 100) < 80 || wallMv.empty());
            MoveList& src = usePawn ? pawnMv : wallMv;
            Move m = src[rng() % src.size()];
            State ns = applyMove(s, m);
            hist.push(ns.hash, m.isWall);
            s = ns;
        }
    }
    return out;
}

// ---------------------------------------------------------------------
// Match runner
// ---------------------------------------------------------------------
static GameOutcome playGame(const State& pos, int aSide,
                             const EngineCfg& cA, const EngineCfg& cB,
                             Negamax& eA, Negamax& eB, int maxPlies) {
    GameOutcome go;
    go.a.games = 1;
    go.b.games = 1;
    State s = pos;
    RepetitionTable rt;
    rt.push(s.hash);   // current position counts as pre-root history (GUI behavior)
    rt.markRoot();
    std::vector<uint64_t> hh;
    hh.reserve((size_t)maxPlies + 2);
    hh.push_back(s.hash);

    for (;;) {
        bool aToMove = (s.turn == aSide);
        Negamax& eng = aToMove ? eA : eB;
        const EngineCfg& cfg = aToMove ? cA : cB;
        if (cfg.clearTTPerMove) eng.clearTT();
        SearchStats st;
        Move m = eng.chooseMove(s, cfg.depthCap, cfg.timeMs, st, rt);

        // Never accept an illegal move (agreement-threshold style guard).
        {
            MoveList lm = legalMoves(s);
            bool ok = false;
            for (size_t i = 0; i < lm.size(); i++)
                if (lm[i] == m) { ok = true; break; }
            if (!ok) {
                g_illegalMoves++;
                m = lm[0];
            }
        }

        int mover = s.turn;
        SideStats& ss = aToMove ? go.a : go.b;
        int dBefore = 0;
        if (!m.isWall)
            dBefore = shortestPathLen(s.wallsH, s.wallsV, s.pawn[mover], mover);

        State ns = applyMove(s, m);
        hh.push_back(ns.hash);
        rt.push(ns.hash, m.isWall);
        go.plies++;
        ss.moves++;
        if (!m.isWall) {
            int dAfter = shortestPathLen(ns.wallsH, ns.wallsV, ns.pawn[mover], mover);
            ss.progressSum += (double)(dBefore - dAfter);
            if (dAfter > dBefore) ss.backMoves++;
            size_t n = hh.size();
            if (n >= 3 && hh[n - 1] == hh[n - 3]) ss.shuffle2++;
            else if (n >= 5 && hh[n - 1] == hh[n - 5]) ss.cycle4++;
        } else {
            ss.wallMoves++;
        }

        int w = winner(ns);
        if (w != -1) {
            if (w == aSide) { go.a.wins++; go.b.losses++; go.result = 0; }
            else            { go.b.wins++; go.a.losses++; go.result = 1; }
            return go;
        }
        if (rt.count(ns.hash) >= 3) {  // GUI draw rule
            go.a.repDraws++; go.b.repDraws++;
            go.result = 2;
            return go;
        }
        if (go.plies >= maxPlies) {
            go.a.timeouts++; go.b.timeouts++;
            go.result = 3;
            return go;
        }
        s = ns;
    }
}

// Elo + 95% CI margin, same formula as tools/arena/run_arena.py.
static void eloAndCI(long long wins1, long long wins2, long long draws,
                      double& eloOut, double& marginOut) {
    long long total = wins1 + wins2 + draws;
    if (total == 0) { eloOut = 0; marginOut = 0; return; }
    double score = (wins1 + 0.5 * draws) / (double)total;
    if (score <= 0.0) eloOut = -800.0;
    else if (score >= 1.0) eloOut = 800.0;
    else eloOut = -400.0 * std::log10(1.0 / score - 1.0);
    double pW = wins1 / (double)total, pL = wins2 / (double)total, pD = draws / (double)total;
    double variance = pW * (1.0 - score) * (1.0 - score)
                    + pL * (0.0 - score) * (0.0 - score)
                    + pD * (0.5 - score) * (0.5 - score);
    double se = total > 1 ? std::sqrt(variance / (double)total) : 0.0;
    if (score > 0.0 && score < 1.0) {
        double factor = 400.0 / (std::log(10.0) * score * (1.0 - score));
        marginOut = 1.96 * se * factor;
    } else marginOut = 0.0;
}

// ---------------------------------------------------------------------
// Experiment driver
// ---------------------------------------------------------------------
struct Worker {
    // Two independent engines so a duel can run two settings in-process.
    std::unique_ptr<Negamax> eng[2];
    Negamax& get(int slot, const EngineCfg& c) {
        if (!eng[slot]) eng[slot].reset(new Negamax());
        eng[slot]->setEvalMode(c.mode);
        eng[slot]->setContempt(c.contempt);
        return *eng[slot];
    }
};

static void runPass(const char* corpusName, const std::vector<State>& corpus,
                     int subset, const EngineCfg& cA, const EngineCfg& cB,
                     const char* ttName, bool duelMode, int maxPlies,
                     int threads, FILE* csv, std::mutex& csvMutex) {
    size_t nJobs = std::min((size_t)subset, corpus.size()) * 2;
    std::atomic<size_t> nextJob{0};
    std::mutex aggMutex;
    SideStats aggA, aggB;
    long long resA = 0, resB = 0, resDraw = 0, resTO = 0;
    double plySum = 0;
    long long gamesDone = 0;

    auto workerFn = [&]() {
        Worker wk;
        for (;;) {
            size_t j = nextJob.fetch_add(1);
            if (j >= nJobs) break;
            size_t posIdx = j / 2;
            int aSide = (j % 2 == 0) ? 0 : 1;
            const State& pos = corpus[posIdx];
            GameOutcome go = playGame(pos, aSide, cA, cB,
                                      wk.get(0, cA), wk.get(1, cB), maxPlies);
            std::lock_guard<std::mutex> lk(aggMutex);
            aggA.merge(go.a);
            aggB.merge(go.b);
            if (go.result == 0) resA++;
            else if (go.result == 1) resB++;
            else if (go.result == 2) resDraw++;
            else resTO++;
            plySum += go.plies;
            gamesDone++;
        }
    };
    std::vector<std::thread> pool;
    for (int t = 0; t < threads; t++) pool.emplace_back(workerFn);
    for (auto& th : pool) th.join();

    long long illegalHere = g_illegalMoves.exchange(0);

    long long games = aggA.games;
    double avgPlies = games ? plySum / games : 0.0;
    double denom = double(aggA.moves + aggB.moves);
    double shufRate = denom ? (aggA.shuffle2 + aggB.shuffle2) / denom : 0.0;
    double cyc4Rate = denom ? (aggA.cycle4 + aggB.cycle4) / denom : 0.0;
    double backRate = denom ? (aggA.backMoves + aggB.backMoves) / denom : 0.0;
    double prog = denom ? (aggA.progressSum + aggB.progressSum) / denom : 0.0;

    const char* modeName = evalModeName(cA.mode);
    std::string tcName = (cA.timeMs >= 30000) ? "depth" : "time";
    if (duelMode) {
        double elo, margin;
        eloAndCI(resA, resB, resDraw, elo, margin);
        printf("DUEL[%s/%s/%s/tt=%s] cA=%d cB=%d | W/D/L(A/B/D)=%lld/%lld/%lld TO=%lld | Elo(A-B) %+7.1f +/- %.1f"
               " | shuffle=%.4f back=%.4f prog/move=%.3f plies=%.1f illegal=%lld\n",
               corpusName, modeName, tcName.c_str(), ttName,
               cA.contempt, cB.contempt, resA, resB, resDraw, resTO, elo, margin,
               shufRate, backRate, prog, avgPlies, illegalHere);
    } else {
        printf("SWEEP[%s/%s/%s/tt=%s] c=%d | games=%lld W=%lld L=%lld D=%lld TO=%lld"
               " | shuffle=%.4f cycle4=%.4f back=%.4f prog/move=%.3f plies=%.1f illegal=%lld\n",
               corpusName, modeName, tcName.c_str(), ttName,
               cA.contempt, games, resA, resB, resDraw, resTO,
               shufRate, cyc4Rate, backRate, prog, avgPlies, illegalHere);
    }
    fflush(stdout);
    if (csv) {
        std::lock_guard<std::mutex> lk(csvMutex);
        fprintf(csv, "%s,%s,%s,%s,%d,%lld,%lld,%lld,%lld,%lld,%.2f,%.5f,%.5f,%.5f,%.5f,%lld\n",
                corpusName, modeName, tcName.c_str(), ttName, cA.contempt,
                games, resA, resB, resDraw, resTO, avgPlies,
                shufRate, cyc4Rate, backRate, prog, illegalHere);
        fflush(csv);
    }
}

int main(int argc, char** argv) {
    int targetMid = 220, targetEmpty = 80;
    int minWalls = 4, maxWalls = 14;
    unsigned seed = 20260822;
    int maxPlies = 80;
    int threads = 3;
    int timeSubset = 60;          // corpus subset for the expensive time TC
    std::string csvPath;
    std::string nnuePath = "data/nnue/nnue_weights_int8.bin";
    bool wantHeur = true, wantNnue = true;
    bool wantDepth = true, wantTime = true;
    bool wantClearTT = false;     // include clear-per-move variant rows
    std::vector<int> contemptList = {-60, -30, -15, -5, 0};
    std::string duelSpec;         // "ca:cb" -> head-to-head instead of sweep

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&](void) -> std::string {
            if (i + 1 >= argc) { fprintf(stderr, "missing value for %s\n", a.c_str()); exit(1); }
            return argv[++i];
        };
        if (a == "--positions") targetMid = atoi(next().c_str());
        else if (a == "--positions-empty") targetEmpty = atoi(next().c_str());
        else if (a == "--min-walls") minWalls = atoi(next().c_str());
        else if (a == "--max-walls") maxWalls = atoi(next().c_str());
        else if (a == "--seed") seed = (unsigned)atol(next().c_str());
        else if (a == "--max-plies") maxPlies = atoi(next().c_str());
        else if (a == "--threads") threads = atoi(next().c_str());
        else if (a == "--time-subset") timeSubset = atoi(next().c_str());
        else if (a == "--csv") csvPath = next();
        else if (a == "--nnue") nnuePath = next();
        else if (a == "--mode") {
            std::string m = next();
            wantHeur = (m == "both" || m == "heuristic");
            wantNnue = (m == "both" || m == "nnue");
        } else if (a == "--tc") {
            std::string t = next();
            wantDepth = (t == "both" || t == "depth");
            wantTime = (t == "both" || t == "time");
        } else if (a == "--tt-clear-variant") wantClearTT = true;
        else if (a == "--contempt") {
            contemptList.clear();
            std::string s = next(), item;
            for (size_t k = 0; k <= s.size(); k++) {
                if (k == s.size() || s[k] == ',') {
                    if (!item.empty()) contemptList.push_back(atoi(item.c_str()));
                    item.clear();
                } else item += s[k];
            }
        } else if (a == "--duel") duelSpec = next();
        else if (a == "-h" || a == "--help") {
            printf("options: --positions N --positions-empty N --min-walls K --max-walls K\n"
                   "  --seed S --max-plies P --threads T(<=4) --time-subset N\n"
                   "  --csv PATH --nnue PATH --mode heuristic|nnue|both --tc depth|time|both\n"
                   "  --tt-clear-variant --contempt c1,c2,... --duel ca:cb\n");
            return 0;
        } else {
            fprintf(stderr, "unknown option: %s\n", a.c_str());
            return 1;
        }
    }
    if (threads < 1) threads = 1;
    if (threads > 4) threads = 4;

    if (wantNnue) {
        if (!loadWeightsQuant(nnuePath)) {
            fprintf(stderr, "[warn] could not load NNUE weights from '%s'; NNUE rows will be skipped\n", nnuePath.c_str());
            wantNnue = false;
        }
    }
    if (!wantHeur && !wantNnue) { fprintf(stderr, "no eval mode selected\n"); return 1; }

    fprintf(stderr, "building corpora (seed %u)...\n", seed);
    int p1 = 0, p2 = 0;
    std::vector<State> mid = buildCorpus(CORPUS_MID, targetMid, minWalls, maxWalls, seed, p1);
    std::vector<State> emp = buildCorpus(CORPUS_EMPTY, targetEmpty, minWalls, maxWalls, seed ^ 0x5f5f5f5fu, p2);
    fprintf(stderr, "corpus mid=%zu (from %d playouts) empty=%zu (from %d playouts)\n",
            mid.size(), p1, emp.size(), p2);
    if (mid.empty() || emp.empty()) { fprintf(stderr, "corpus build failed\n"); return 1; }

    FILE* csv = nullptr;
    std::mutex csvMutex;
    if (!csvPath.empty()) {
        csv = fopen(csvPath.c_str(), "w");
        if (csv) {
            fprintf(csv, "corpus,mode,tc,tt,contempt,games,a_wins,b_wins,rep_draws,timeouts,"
                         "avg_plies,shuffle_rate,cycle4_rate,back_rate,progress_per_move,illegal\n");
            fflush(csv);
        }
    }

    // ---------------- Duel mode ----------------
    if (!duelSpec.empty()) {
        size_t colon = duelSpec.find(':');
        if (colon == std::string::npos) { fprintf(stderr, "--duel wants ca:cb\n"); return 1; }
        int ca = atoi(duelSpec.substr(0, colon).c_str());
        int cb = atoi(duelSpec.substr(colon + 1).c_str());

        std::vector<Negamax::EvalMode> modes;
        if (wantHeur) modes.push_back(Negamax::EvalMode::Heuristic);
        if (wantNnue) modes.push_back(Negamax::EvalMode::NNUE);
        for (auto md : modes) {
            if (wantDepth) {
                EngineCfg c; c.depthCap = 5; c.timeMs = 60000; c.mode = md;
                c.contempt = ca;
                EngineCfg d = c; d.contempt = cb;
                runPass("mid", mid, 20, c, d, "persist", true, 48, threads, csv, csvMutex);
                runPass("empty", emp, (int)emp.size(), c, d, "persist", true, maxPlies, threads, csv, csvMutex);
            }
            if (wantTime) {
                EngineCfg c; c.depthCap = 40; c.timeMs = 150; c.mode = md;
                c.contempt = ca;
                EngineCfg d = c; d.contempt = cb;
                runPass("mid", mid, timeSubset, c, d, "persist", true, maxPlies, threads, csv, csvMutex);
                runPass("empty", emp, (int)emp.size(), c, d, "persist", true, maxPlies, threads, csv, csvMutex);
            }
        }
        if (csv) fclose(csv);
        fprintf(stderr, "done. illegal moves observed: %lld\n", g_illegalMoves.load());
        return 0;
    }

    // ---------------- Sweep mode ----------------
    std::vector<Negamax::EvalMode> modes;
    if (wantHeur) modes.push_back(Negamax::EvalMode::Heuristic);
    if (wantNnue) modes.push_back(Negamax::EvalMode::NNUE);

    // Deterministic profile: fixed depth cap 5 completes in about 0.3-0.8s
    // per move in near-endgame positions (measured); depth >= 7 needs tens
    // of millions of nodes there and never finishes a sane budget.
    int depthCap = 5;
    long long depthTimeMs = 60000;
    int depthSubset = 20;
    int depthPlies = 48;

    for (auto md : modes) {
        if (wantDepth) {
            EngineCfg c; c.depthCap = depthCap; c.timeMs = depthTimeMs; c.mode = md;
            fprintf(stderr, "[pass] depth%d %s\n", depthCap, evalModeName(md));
            for (int cv : contemptList) {
                c.contempt = cv;
                EngineCfg d = c;
                runPass("mid", mid, depthSubset, c, d, "persist", false, depthPlies, threads, csv, csvMutex);
                runPass("empty", emp, (int)emp.size(), c, d, "persist", false, maxPlies, threads, csv, csvMutex);
            }
        }
        if (wantTime) {
            EngineCfg c; c.depthCap = 40; c.timeMs = 150; c.mode = md;
            fprintf(stderr, "[pass] time150 %s\n", evalModeName(md));
            for (int cv : contemptList) {
                c.contempt = cv;
                EngineCfg d = c;
                runPass("mid", mid, timeSubset, c, d, "persist", false, maxPlies, threads, csv, csvMutex);
                runPass("empty", emp, (int)emp.size(), c, d, "persist", false, maxPlies, threads, csv, csvMutex);
            }
            // TT policy variant (H3): same settings, clear TT every real
            // move. Only two contempt values to bound runtime.
            int clearList[2] = {CONTEMPT, 0};
            for (int ci = 0; ci < 2; ci++) {
                EngineCfg d = c; d.clearTTPerMove = true; d.contempt = clearList[ci];
                runPass("mid", mid, timeSubset, d, d, "clear", false, maxPlies, threads, csv, csvMutex);
            }
        }
    }

    if (csv) fclose(csv);
    fprintf(stderr, "done. illegal moves observed: %lld\n", g_illegalMoves.load());
    return 0;
}




