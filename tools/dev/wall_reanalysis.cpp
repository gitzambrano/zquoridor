// wall_reanalysis.cpp -- targeted MCAB reanalysis for wall-critical V3 games.
//
// Reads complete 64-byte TrainingSample games, reconstructs legal State
// objects from mover-canonical records, and re-runs a much more expensive
// root search only at selected wall-resource decisions. Output is still a
// sequence of COMPLETE games, so ply_index()/plies_remaining() remain valid.
// Non-selected positions have visit probabilities zeroed in this replay: they
// still provide value supervision but do not duplicate the old policy target.
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
    for (int i = 0; i < WS * WS; ++i) {
        if ((s.wallsH >> i) & 1ull) h ^= z.wallHKey[i];
        if ((s.wallsV >> i) & 1ull) h ^= z.wallVKey[i];
    }
    if (s.turn == 1) h ^= z.turnKey;
    return h;
}

static State stateFromRecord(const TrainingSample& r) {
    State s{};
    int mover = (int)r.mover;
    int opp = 1 - mover;
    // Mirroring is an involution, so applying the same transformation restores
    // the raw board coordinates written before canonicalisation.
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

static bool stateMatchesRecord(const State& s, const TrainingSample& r) {
    if (r.mover > 1 || s.pawn[0] == s.pawn[1]) return false;
    if (s.wallsLeft[0] < 0 || s.wallsLeft[0] > WALLS_PER_PLAYER ||
        s.wallsLeft[1] < 0 || s.wallsLeft[1] > WALLS_PER_PLAYER) return false;
    int placed = __builtin_popcountll(s.wallsH) + __builtin_popcountll(s.wallsV);
    if (placed + (int)s.wallsLeft[0] + (int)s.wallsLeft[1] != 2 * WALLS_PER_PLAYER) return false;
    int mover = s.turn, opp = 1 - mover;
    int ownD = shortestPathLen(s.wallsH, s.wallsV, s.pawn[mover], mover);
    int oppD = shortestPathLen(s.wallsH, s.wallsV, s.pawn[opp], opp);
    return ownD == (int)r.ownDist && oppD == (int)r.oppDist && winner(s) == -1;
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

// Priority is only a sampling policy. It never becomes a training label.
// We spend expensive search on real, legal states where conservation of the
// last few walls matters most, especially when the old root was undecided.
static int reanalysisPriority(const TrainingSample& r) {
    int own = (int)r.wallsLeftOwn;
    if (own < 1 || own > 4) return 0;
    double wm = wallVisitMass(r);
    bool playedWall = r.policyTarget >= N * N;
    bool ambiguous = wm >= 0.05 && wm <= 0.95;
    bool wallPoorRace = own <= 2 &&
        ((int)r.wallsLeftOpp - own) >= 4 &&
        ((int)r.oppDist - (int)r.ownDist) >= 2;

    int score = own == 1 ? 140 : own == 2 ? 100 : own == 3 ? 55 : 20;
    if (playedWall) score += 45;
    if (wallPoorRace) score += 100;
    if (((int)r.wallsLeftOpp - own) >= 4) score += 20;
    if (((int)r.oppDist - (int)r.ownDist) >= 2) score += 15;
    if (ambiguous) {
        double centered = 1.0 - std::abs(2.0 * wm - 1.0); // 1 at 50/50
        score += 30 + (int)std::lround(50.0 * centered);
    }
    return score;
}

static void clearPolicy(TrainingSample& r) {
    std::fill(std::begin(r.policyTopIdx), std::end(r.policyTopIdx), (uint16_t)0);
    std::fill(std::begin(r.policyTopProb), std::end(r.policyTopProb), (uint16_t)0);
}

static bool fillRootVisits(const State& s, int timeMs, int nodeBudget,
                           uint32_t noiseSeed, Negamax& engine,
                           TrainingSample& out, uint64_t& abNodes,
                           uint64_t& rootVisits) {
    mcab::McabParams p;
    p.enabled = true;
    p.rootNoiseEnabled = false;
    p.nodeBudget = nodeBudget;
    p.cPuct = 1.20;
    McabRunnerT runner;
    runner.setParams(p);
    runner.seedNoise(noiseSeed);
    runner.resetTree();

    // Match selfplay semantics: the table contains PRIOR positions, not the
    // current root itself. We do not have earlier repetition history in an
    // isolated record, so an empty table is the least-assumptive reconstruction.
    RepetitionTable rep;
    SearchStats st;
    (void)runner.choose(engine, s, 60, timeMs, st, rep);
    abNodes += st.nodes;

    const auto* root = runner.search.rootNodeForInspection();
    if (!root || !root->expanded || root->state.hash != s.hash) return false;

    struct Edge { size_t i; float n; };
    std::vector<Edge> ranked;
    size_t nm = std::min(root->moves.size(), root->N.size());
    ranked.reserve(nm);
    double allVisits = 0.0;
    for (size_t i = 0; i < nm; ++i) {
        allVisits += (double)root->N[i];
        if (root->N[i] > 0.f) ranked.push_back({i, root->N[i]});
    }
    if (ranked.empty() || allVisits <= 0.0) return false;
    rootVisits += (uint64_t)std::llround(allVisits);
    std::stable_sort(ranked.begin(), ranked.end(),
        [](const Edge& a, const Edge& b) { return a.n > b.n; });
    if (ranked.size() > 8) ranked.resize(8);

    clearPolicy(out);
    double topVisits = 0.0;
    for (const auto& e : ranked) topVisits += (double)e.n;
    uint32_t assigned = 0;
    int mover = s.turn;
    for (size_t j = 0; j < ranked.size(); ++j) {
        const Move& m = root->moves[ranked[j].i];
        out.policyTopIdx[j] = moveToPolicyIndex(mirrorMoveForPerspective(m, mover));
        uint16_t q = (uint16_t)std::floor(65535.0 * (double)ranked[j].n / topVisits);
        out.policyTopProb[j] = q;
        assigned += q;
    }
    out.policyTopProb[0] = (uint16_t)(out.policyTopProb[0] + (65535u - assigned));
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

    struct Candidate { size_t index; size_t game; int priority; };
    std::vector<Candidate> candidates;
    int reconstructionErrors = 0;
    for (size_t g = 0; g + 1 < starts.size(); ++g) {
        for (size_t i = starts[g]; i < starts[g + 1]; ++i) {
            int prio = reanalysisPriority(src[i]);
            if (prio <= 0) continue;
            State s = stateFromRecord(src[i]);
            if (!stateMatchesRecord(s, src[i])) { ++reconstructionErrors; continue; }
            candidates.push_back({i, g, prio});
        }
    }
    std::stable_sort(candidates.begin(), candidates.end(), [](const Candidate& a, const Candidate& b) {
        if (a.priority != b.priority) return a.priority > b.priority;
        return a.index < b.index;
    });
    if ((int)candidates.size() > maxPositions) candidates.resize((size_t)maxPositions);

    std::vector<uint8_t> selected(src.size(), 0);
    for (const auto& c : candidates) selected[c.index] = 1;

    FILE* fout = std::fopen(outPath.c_str(), "wb");
    if (!fout) { std::perror("fopen output"); return 2; }
    Negamax engine;
    engine.setEvalMode(Negamax::EvalMode::NNUE);
    engine.setPolicyOrderingEnabled(true);
    engine.setPolicyOrderingMinDepth(3);
    std::mt19937 rng(seed);

    int done = 0, failed = 0, gamesWritten = 0;
    uint64_t totalAbNodes = 0, totalRootVisits = 0;
    for (size_t g = 0; g + 1 < starts.size(); ++g) {
        size_t a = starts[g], b = starts[g + 1];
        bool wanted = false;
        for (size_t i = a; i < b; ++i) if (selected[i]) { wanted = true; break; }
        if (!wanted) continue;

        std::vector<TrainingSample> game(src.begin() + (ptrdiff_t)a, src.begin() + (ptrdiff_t)b);
        bool touched = false;
        for (size_t j = 0; j < game.size(); ++j) {
            size_t global = a + j;
            if (!selected[global]) { clearPolicy(game[j]); continue; }
            State s = stateFromRecord(game[j]);
            if (!stateMatchesRecord(s, game[j]) || legalMoves(s).empty()) {
                clearPolicy(game[j]); ++failed; continue;
            }
            TrainingSample repl = game[j];
            if (fillRootVisits(s, timeMs, nodeBudget, rng(), engine, repl,
                               totalAbNodes, totalRootVisits)) {
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
    std::printf("wall reanalysis: input_games=%zu candidates=%zu selected_games=%d positions=%d failed=%d reconstruction_errors=%d time_ms=%d node_budget=%d ab_nodes=%llu root_visits=%llu output=%s\n",
        starts.size() - 1, candidates.size(), gamesWritten, done, failed, reconstructionErrors,
        timeMs, nodeBudget, (unsigned long long)totalAbNodes,
        (unsigned long long)totalRootVisits, outPath.c_str());
    return done > 0 && reconstructionErrors == 0 ? 0 : 1;
}
