// wall_reanalysis.cpp -- targeted MCAB reanalysis for wall-critical V3 games.
//
// Reads complete 64-byte TrainingSample games, reconstructs legal State
// objects from mover-canonical records, and re-runs a much more expensive
// root search only at wall-critical positions. The output remains a sequence
// of COMPLETE games, so ply_index()/plies_remaining() remain valid. Policy
// visit targets are replaced only at reanalysed positions; non-critical
// positions have policyTopProb zeroed in the replay so they contribute value
// supervision but no duplicated policy supervision.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "../../tools/selfplay/selfplay.hpp"

using namespace qr;

static uint64_t rebuildHash(const State& s) {
    Zobrist& z = zobrist();
    uint64_t h = z.pawnKey[0][s.pawn[0]] ^ z.pawnKey[1][s.pawn[1]];
    uint64_t wh = s.wallsH, wv = s.wallsV;
    for (int i = 0; i < WS * WS; ++i) {
        if ((wh >> i) & 1ull) h ^= z.wallHKey[i];
        if ((wv >> i) & 1ull) h ^= z.wallVKey[i];
    }
    if (s.turn == 1) h ^= z.turnKey;
    return h;
}

static State stateFromRecord(const TrainingSample& r) {
    State s{};
    int mover = (int)r.mover;
    int opp = 1 - mover;
    // The mirror is an involution: applying it again restores raw board coords.
    s.pawn[mover] = (uint8_t)mirroredPawnCell((int)r.ownPawn, mover);
    s.pawn[opp] = (uint8_t)mirroredPawnCell((int)r.oppPawn, mover);
    s.wallsH = mirrorWallBitboard(r.wallsH, mover);
    s.wallsV = mirrorWallBitboard(r.wallsV, mover);
    s.wallsLeft[mover] = r.wallsLeftOwn;
    s.wallsLeft[opp] = r.wallsLeftOpp;
    s.turn = mover;
    s.hash = rebuildHash(s);
    return s;
}

static double wallVisitMass(const TrainingSample& r) {
    double sum = 0.0;
    for (int j = 0; j < 8; ++j) {
        if (r.policyTopProb[j] == 0) continue;
        if (r.policyTopIdx[j] >= N * N && r.policyTopIdx[j] < NUM_MOVE_INDICES)
            sum += (double)r.policyTopProb[j] / 65535.0;
    }
    return sum;
}

static bool critical(const TrainingSample& r) {
    const double wm = wallVisitMass(r);
    const bool ambiguous = wm >= 0.10 && wm <= 0.90;
    const bool lowResource = r.wallsLeftOwn <= 3 && wm >= 0.05;
    const bool depletionChoice = r.wallsLeftOwn <= 2;
    const bool wallPoorRace = r.wallsLeftOwn <= 2 &&
        ((int)r.wallsLeftOpp - (int)r.wallsLeftOwn) >= 4 &&
        ((int)r.oppDist - (int)r.ownDist) >= 2;
    return ambiguous || lowResource || depletionChoice || wallPoorRace;
}

static void clearPolicy(TrainingSample& r) {
    std::fill(std::begin(r.policyTopIdx), std::end(r.policyTopIdx), (uint16_t)0);
    std::fill(std::begin(r.policyTopProb), std::end(r.policyTopProb), (uint16_t)0);
}

static bool fillRootVisits(const State& s, int timeMs, int nodeBudget,
                           uint32_t noiseSeed, Negamax& engine,
                           TrainingSample& out, uint64_t& nodes) {
    mcab::McabParams p;
    p.enabled = true;
    p.rootNoiseEnabled = false;
    p.nodeBudget = nodeBudget;
    p.cPuct = 1.20;
    McabRunnerT runner;
    runner.setParams(p);
    runner.seedNoise(noiseSeed);
    runner.resetTree();

    RepetitionTable rep;
    rep.push(s.hash);
    SearchStats st;
    (void)runner.choose(engine, s, 60, timeMs, st, rep);
    nodes += st.nodes;

    const auto* root = runner.search.rootNodeForInspection();
    if (!root || !root->expanded || root->state.hash != s.hash) return false;

    struct Edge { size_t i; float n; };
    std::vector<Edge> ranked;
    size_t nm = std::min(root->moves.size(), root->N.size());
    ranked.reserve(nm);
    for (size_t i = 0; i < nm; ++i)
        if (root->N[i] > 0.f) ranked.push_back({i, root->N[i]});
    if (ranked.empty()) return false;
    std::stable_sort(ranked.begin(), ranked.end(),
        [](const Edge& a, const Edge& b) { return a.n > b.n; });
    if (ranked.size() > 8) ranked.resize(8);

    clearPolicy(out);
    double total = 0.0;
    for (const auto& e : ranked) total += (double)e.n;
    uint32_t assigned = 0;
    int mover = s.turn;
    for (size_t j = 0; j < ranked.size(); ++j) {
        const Move& m = root->moves[ranked[j].i];
        out.policyTopIdx[j] = moveToPolicyIndex(mirrorMoveForPerspective(m, mover));
        uint16_t q = (uint16_t)std::floor(65535.0 * (double)ranked[j].n / total);
        out.policyTopProb[j] = q;
        assigned += q;
    }
    out.policyTopProb[0] = (uint16_t)(out.policyTopProb[0] + (65535u - assigned));
    // Keep policyTarget aligned with the highest-visit action for diagnostics.
    out.policyTarget = out.policyTopIdx[0];
    return true;
}

static bool readAll(const std::string& path, std::vector<TrainingSample>& v) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    std::fseek(f, 0, SEEK_END);
    long nbytes = std::ftell(f);
    std::rewind(f);
    if (nbytes <= 0 || nbytes % (long)sizeof(TrainingSample) != 0) {
        std::fclose(f); return false;
    }
    v.resize((size_t)nbytes / sizeof(TrainingSample));
    bool ok = std::fread(v.data(), sizeof(TrainingSample), v.size(), f) == v.size();
    std::fclose(f);
    return ok;
}

static bool isGameStart(const TrainingSample& r) {
    return r.wallsH == 0 && r.wallsV == 0 &&
           r.wallsLeftOwn == 10 && r.wallsLeftOpp == 10 &&
           r.ownDist == 8 && r.oppDist == 8;
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::fprintf(stderr, "usage: %s <in.bin> <out.bin> <weights-int8> [max_positions=250] [time_ms=250] [node_budget=100000] [seed=20260906]\n", argv[0]);
        return 2;
    }
    const std::string in = argv[1], outPath = argv[2], weights = argv[3];
    int maxPositions = argc > 4 ? std::atoi(argv[4]) : 250;
    int timeMs = argc > 5 ? std::atoi(argv[5]) : 250;
    int nodeBudget = argc > 6 ? std::atoi(argv[6]) : 100000;
    uint32_t seed = argc > 7 ? (uint32_t)std::strtoul(argv[7], nullptr, 10) : 20260906u;
    if (maxPositions <= 0 || timeMs <= 0 || nodeBudget <= 0) return 2;
    if (!loadWeightsQuant(weights)) {
        std::fprintf(stderr, "failed to load NNUE weights: %s\n", weights.c_str());
        return 2;
    }

    std::vector<TrainingSample> src;
    if (!readAll(in, src) || src.empty() || !isGameStart(src[0])) {
        std::fprintf(stderr, "invalid V3 input: %s\n", in.c_str());
        return 2;
    }

    std::vector<size_t> starts;
    for (size_t i = 0; i < src.size(); ++i) if (isGameStart(src[i])) starts.push_back(i);
    if (starts.empty()) return 2;
    starts.push_back(src.size());

    struct GameScore { size_t g; int criticalCount; };
    std::vector<GameScore> games;
    for (size_t g = 0; g + 1 < starts.size(); ++g) {
        int n = 0;
        for (size_t i = starts[g]; i < starts[g + 1]; ++i) if (critical(src[i])) ++n;
        if (n > 0) games.push_back({g, n});
    }
    std::stable_sort(games.begin(), games.end(), [](const GameScore& a, const GameScore& b) {
        return a.criticalCount > b.criticalCount;
    });

    FILE* fout = std::fopen(outPath.c_str(), "wb");
    if (!fout) { std::perror("fopen output"); return 2; }
    Negamax engine;
    engine.setEvalMode(Negamax::EvalMode::NNUE);
    engine.setPolicyOrderingEnabled(true);
    engine.setPolicyOrderingMinDepth(3);
    std::mt19937 rng(seed);

    int done = 0, failed = 0, gamesWritten = 0;
    uint64_t totalNodes = 0;
    for (const auto& gs : games) {
        if (done >= maxPositions) break;
        size_t a = starts[gs.g], b = starts[gs.g + 1];
        std::vector<TrainingSample> game(src.begin() + (ptrdiff_t)a, src.begin() + (ptrdiff_t)b);
        bool touched = false;
        for (size_t j = 0; j < game.size(); ++j) {
            if (!critical(game[j])) {
                clearPolicy(game[j]);
                continue;
            }
            if (done >= maxPositions) { clearPolicy(game[j]); continue; }
            State s = stateFromRecord(game[j]);
            if (winner(s) != -1 || legalMoves(s).empty()) { clearPolicy(game[j]); ++failed; continue; }
            TrainingSample repl = game[j];
            if (fillRootVisits(s, timeMs, nodeBudget, rng(), engine, repl, totalNodes)) {
                game[j] = repl;
                ++done;
                touched = true;
            } else {
                clearPolicy(game[j]);
                ++failed;
            }
        }
        if (touched) {
            std::fwrite(game.data(), sizeof(TrainingSample), game.size(), fout);
            ++gamesWritten;
        }
    }
    std::fclose(fout);
    std::printf("wall reanalysis: input_games=%zu selected_games=%d positions=%d failed=%d time_ms=%d node_budget=%d nodes=%llu output=%s\n",
        starts.size() - 1, gamesWritten, done, failed, timeMs, nodeBudget,
        (unsigned long long)totalNodes, outPath.c_str());
    return done > 0 ? 0 : 1;
}
