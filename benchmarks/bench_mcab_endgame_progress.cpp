// bench_mcab_endgame_progress -- compares the two MCAB backup modes for pawn
// progress on a corpus of near-endgame positions.
//
// RESULT (2026-08-24, 40 positions, 150ms per move). AvgBlend beat
// MinimaxHard on every metric: mean progress 6.625 against 5.650, 32 wins
// against 29, 0 wandering positions against 5. Therefore the wandering
// reported in benchmarks/repro_wander.cpp is NOT a general property of the
// AvgBlend backup. MinimaxHard also loses the arena badly (0 wins in 60
// games, approximately -585 Elo at 200ms). Read status.md before you change
// McabParams::backupMode.
//
// Corpus: random legal playouts from initialState() with a seeded RNG,
// sampled at every ply (not just once per playout). A position enters the
// corpus only when ALL of the following hold:
//   - the side to move has wallsLeft == 0
//   - the opponent has wallsLeft >= 4 (still has ammunition to keep blocking)
//   - the side to move leads the pawn race by >= 3 shortestPathLen steps
//   - the side to move's own shortestPathLen >= 6 (not already at the door)
//   - the game is not already decided
//   - AMBIGUITY: the single pawn step straight toward the goal is either
//     illegal (blocked by a wall directly ahead) or does not reduce the
//     side-to-move's shortestPathLen. This forces at least one sideways or
//     backward step before any forward progress is possible -- the exact
//     shape of the hand-built position in benchmarks/repro_wander.cpp.
// Positions are deduplicated by Zobrist hash.
//
// Protocol: for each corpus position, play 12 engine moves. The engine (the
// side to move at the corpus position) uses MCAB with a fixed backup mode;
// the opponent always takes the legal pawn move that most shortens its own
// shortestPathLen (never places walls), so the two arms face an identical,
// deterministic opponent. Each position is run twice, once per backup mode,
// with a fresh runner.resetTree() and McabParams{} defaults otherwise.
//
// Build:
//   g++ -O3 -std=c++17 -march=native [-mavx2 -mfma] -Isrc \
//       -o bin/bench_mcab_endgame_progress benchmarks/bench_mcab_endgame_progress.cpp
// Run (from repo root):
//   bin/bench_mcab_endgame_progress [timeMs] [targetCorpus]
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <unordered_set>
#include <random>
#include "rules.hpp"
#include "search.hpp"
#include "../src/mcab.hpp"

using namespace qr;
using McabRunnerT = mcab::McabRunner<Negamax, State, Move, MoveList, AccPair,
                                      RepetitionTable, SearchStats>;

static int dist(const State& s, int p) {
    return shortestPathLen(s.wallsH, s.wallsV, s.pawn[p], p);
}

// Opponent policy: always take the legal pawn move that most shortens its
// own shortest path. Never places a wall.
static Move greedyOpponent(const State& s) {
    MoveList ms = legalMoves(s);
    int best = -1, bestD = 1 << 30;
    for (size_t i = 0; i < ms.size(); i++) {
        if (ms[i].isWall) continue;
        State ns = applyMove(s, ms[i]);
        int d = shortestPathLen(ns.wallsH, ns.wallsV, ns.pawn[s.turn], s.turn);
        if (d < bestD) { bestD = d; best = (int)i; }
    }
    return ms[best];
}

// Cell one step straight toward `player`'s goal row, from its current pawn
// cell. Always in-bounds here because the caller only calls this once the
// player's own shortestPathLen is already >= 6 (far from the goal edge).
static int forwardCell(const State& s, int player) {
    int r = rowOf(s.pawn[player]), c = colOf(s.pawn[player]);
    int dir = (GOAL_ROW[player] > r) ? 1 : -1;
    return cellIdx(r + dir, c);
}

// True when the single pawn step straight toward the goal is either illegal
// (no legal pawn move lands on that cell) or legal but does not shorten the
// mover's own shortestPathLen -- i.e. real progress requires a sideways or
// backward step first.
static bool forwardStepIsAmbiguous(const State& s, int mover, int dMover) {
    int fwd = forwardCell(s, mover);
    MoveList ms = legalMoves(s);
    for (size_t i = 0; i < ms.size(); i++) {
        if (ms[i].isWall) continue;
        if (ms[i].a != fwd) continue;
        State ns = applyMove(s, ms[i]);
        int nd = shortestPathLen(ns.wallsH, ns.wallsV, ns.pawn[mover], mover);
        return !(nd < dMover);
    }
    return true;  // no legal move onto the straight-ahead cell -> blocked
}

struct CorpusStats {
    long long candidatesSeen = 0;   // ply-samples that passed the numeric filters
    long long accepted = 0;         // candidates that also passed the ambiguity test
    long long playouts = 0;
};

// Picks a random legal move, weighted toward pawn steps. Plain uniform
// selection over legalMoves() almost never reaches the target shape: legal
// wall slots vastly outnumber legal pawn steps in the move list, so a
// uniform random walk burns down both players' wall budgets in lockstep and
// essentially never leaves one player at wallsLeft==0 while the opponent
// still holds >=4 (empirically ~5 in 2.4M qualifying ply-samples). Biasing
// the walk toward pawn moves (kPawnBias chance of a pawn step when one is
// legal) still produces fully legal random games -- it only changes which
// legal move is sampled -- and makes the asymmetric-wall-budget, blocked-
// lane shape the failure needs common enough to corpus-build in seconds.
static constexpr double kPawnBias = 0.85;

static Move pickBiasedRandomMove(const State& s, std::mt19937& rng) {
    MoveList ms = legalMoves(s);
    std::vector<size_t> pawnIdx, wallIdx;
    for (size_t i = 0; i < ms.size(); i++) (ms[i].isWall ? wallIdx : pawnIdx).push_back(i);
    std::uniform_real_distribution<double> coin(0.0, 1.0);
    if (!pawnIdx.empty() && (wallIdx.empty() || coin(rng) < kPawnBias)) {
        std::uniform_int_distribution<size_t> pick(0, pawnIdx.size() - 1);
        return ms[pawnIdx[pick(rng)]];
    }
    if (!wallIdx.empty()) {
        std::uniform_int_distribution<size_t> pick(0, wallIdx.size() - 1);
        return ms[wallIdx[pick(rng)]];
    }
    std::uniform_int_distribution<size_t> pick(0, pawnIdx.size() - 1);
    return ms[pawnIdx[pick(rng)]];
}

// Builds a corpus of near-endgame "must step sideways" positions by random
// legal playout from initialState() (pawn-biased random walk, see above).
// Every ply of every playout is checked (not just one sample per playout),
// so a single random game can contribute several corpus positions.
// Deduplicated by Zobrist hash.
static std::vector<State> buildCorpus(int target, unsigned seed, int maxGames,
                                       CorpusStats& stats) {
    std::vector<State> corpus;
    std::unordered_set<uint64_t> seen;
    std::mt19937 rng(seed);
    while ((int)corpus.size() < target && stats.playouts < maxGames) {
        stats.playouts++;
        State s = initialState();
        for (int ply = 0; ply < 200; ply++) {
            if (winner(s) != -1) break;
            MoveList ms = legalMoves(s);
            if (ms.size() == 0) break;
            s = applyMove(s, pickBiasedRandomMove(s, rng));

            if (winner(s) != -1) continue;
            int mover = s.turn, opp = 1 - mover;
            if (s.wallsLeft[mover] != 0) continue;
            if (s.wallsLeft[opp] < 4) continue;
            int dMover = dist(s, mover);
            int dOpp = dist(s, opp);
            if (dMover < 6) continue;
            if (dMover + 3 > dOpp) continue;
            stats.candidatesSeen++;
            if (!forwardStepIsAmbiguous(s, mover, dMover)) continue;
            stats.accepted++;
            if (seen.count(s.hash)) continue;
            seen.insert(s.hash);
            corpus.push_back(s);
            if ((int)corpus.size() >= target) break;
        }
    }
    return corpus;
}

struct ArmResult {
    long long totalProgress = 0;
    int wins = 0;
    int wanderers = 0;        // progress <= 0
    int shuffles = 0;         // immediate 2-ply shuffles
    double totalDistinctRatio = 0.0;  // sum of (distinct cells visited / engine moves)
    int n = 0;
};

static constexpr int kEngineMoves = 12;

// Plays kEngineMoves engine moves from `start` with the engine using MCAB
// under `params`, alternating against the greedy opponent. Tracks progress,
// wins, wandering, immediate 2-ply shuffles, and cell-revisit ratio.
static void playArm(const State& start, const mcab::McabParams& params,
                     int timeMs, ArmResult& acc) {
    Negamax eng;
    eng.setEvalMode(Negamax::EvalMode::NNUE);
    McabRunnerT runner;
    runner.setParams(params);
    runner.resetTree();

    State s = start;
    int engineSide = s.turn;
    RepetitionTable hist;
    hist.push(s.hash);

    int startD = dist(s, engineSide);
    int prevEngineCell = -1;      // engine cell 2 engine-moves ago
    int prevPrevEngineCell = -1;  // updated as we go
    bool shuffled = false;
    int engineMoves = 0;
    bool won = false;
    std::unordered_set<int> visited;
    visited.insert(s.pawn[engineSide]);

    for (int i = 0; i < kEngineMoves && winner(s) == -1; i++) {
        SearchStats st;
        mcab::McabStats ms{};
        Move m = runner.choose(eng, s, 40, timeMs, st, hist, &ms);
        s = applyMove(s, m);
        hist.push(s.hash);
        engineMoves++;
        int cellNow = s.pawn[engineSide];
        visited.insert(cellNow);
        if (engineMoves >= 3 && cellNow == prevPrevEngineCell) shuffled = true;
        prevPrevEngineCell = prevEngineCell;
        prevEngineCell = cellNow;

        if (winner(s) == engineSide) { won = true; break; }
        if (winner(s) != -1) break;

        Move om = greedyOpponent(s);
        s = applyMove(s, om);
        hist.push(s.hash);
        if (winner(s) != -1) break;
    }

    int endD = dist(s, engineSide);
    if (winner(s) == engineSide) endD = 0;
    int progress = startD - endD;

    acc.n++;
    acc.totalProgress += progress;
    if (won) acc.wins++;
    if (progress <= 0) acc.wanderers++;
    if (shuffled) acc.shuffles++;
    if (engineMoves > 0)
        acc.totalDistinctRatio += (double)visited.size() / (double)engineMoves;
}

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    int timeMs = argc > 1 ? atoi(argv[1]) : 200;
    int targetCorpus = argc > 2 ? atoi(argv[2]) : 40;

    if (!loadWeightsQuant("data/nnue/nnue_weights_int8.bin")) {
        printf("FATAL: could not load data/nnue/nnue_weights_int8.bin "
               "(run from repo root)\n");
        return 1;
    }

    printf("Building corpus (target=%d)...\n", targetCorpus);
    CorpusStats cstats;
    std::vector<State> corpus = buildCorpus(targetCorpus, /*seed=*/12345,
                                             /*maxGames=*/400000, cstats);
    double acceptRate = cstats.candidatesSeen > 0
                             ? 100.0 * (double)cstats.accepted / (double)cstats.candidatesSeen
                             : 0.0;
    printf("Corpus built: %d positions (target %d)\n", (int)corpus.size(), targetCorpus);
    printf("playouts=%lld  numeric-filter candidates=%lld  ambiguity-accepted=%lld "
           "(%.2f%% of candidates)  unique-after-dedup=%d\n\n",
           cstats.playouts, cstats.candidatesSeen, cstats.accepted, acceptRate,
           (int)corpus.size());
    if (corpus.empty()) { printf("no corpus positions found\n"); return 1; }
    if ((int)corpus.size() < 15) {
        printf("WARNING: corpus below the 15-position floor requested; "
               "reporting results anyway.\n\n");
    } else if ((int)corpus.size() < targetCorpus) {
        printf("NOTE: could not fill the full target corpus; reporting the %d "
               "positions found (>= 15 floor met).\n\n", (int)corpus.size());
    }

    mcab::McabParams pAvg;
    pAvg.backupMode = mcab::BackupMode::AvgBlend;
    mcab::McabParams pHard;
    pHard.backupMode = mcab::BackupMode::MinimaxHard;

    ArmResult avgRes, hardRes;
    for (size_t i = 0; i < corpus.size(); i++) {
        printf("position %2d/%2d ... ", (int)i + 1, (int)corpus.size());
        playArm(corpus[i], pAvg, timeMs, avgRes);
        playArm(corpus[i], pHard, timeMs, hardRes);
        printf("done\n");
    }

    double avgMean = avgRes.n ? (double)avgRes.totalProgress / avgRes.n : 0.0;
    double hardMean = hardRes.n ? (double)hardRes.totalProgress / hardRes.n : 0.0;
    double avgDistinct = avgRes.n ? avgRes.totalDistinctRatio / avgRes.n : 0.0;
    double hardDistinct = hardRes.n ? hardRes.totalDistinctRatio / hardRes.n : 0.0;

    printf("\n=== Summary over %d corpus positions (timeMs=%d, %d engine moves each) ===\n",
           (int)corpus.size(), timeMs, kEngineMoves);
    printf("%-30s %12s %12s\n", "backup mode", "AvgBlend", "MinimaxHard");
    printf("%-30s %12.3f %12.3f\n", "mean progress", avgMean, hardMean);
    printf("%-30s %12d %12d\n", "wins", avgRes.wins, hardRes.wins);
    printf("%-30s %12d %12d\n", "wandering (prog<=0)", avgRes.wanderers, hardRes.wanderers);
    printf("%-30s %12d %12d\n", "immediate shuffles", avgRes.shuffles, hardRes.shuffles);
    printf("%-30s %12.3f %12.3f\n", "distinct cells / moves", avgDistinct, hardDistinct);

    return 0;
}
