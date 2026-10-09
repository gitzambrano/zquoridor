// Validate temperature sampling from MCAB root visits.
#include <cassert>
#include <cstdint>
#include <random>
#include <vector>

#include "../tools/selfplay/selfplay.hpp"

namespace {

struct RootVisits {
    qr::MoveList moves;
    std::vector<float> N;
};

void testLowTemperatureSelectsMostVisitedMove() {
    RootVisits root;
    root.moves.push_back(qr::Move::pawn(10));
    root.moves.push_back(qr::Move::pawn(11));
    root.N = {9.0f, 1.0f};

    std::mt19937_64 rng(7);
    for (int sample = 0; sample < 200; ++sample) {
        const qr::Move move = qr::sampleMoveByVisitTemperature(root, 0.05, rng);
        assert(move == root.moves[0]);
    }
}

void testUnitTemperatureExploresVisitedAlternatives() {
    RootVisits root;
    root.moves.push_back(qr::Move::pawn(10));
    root.moves.push_back(qr::Move::pawn(11));
    root.N = {9.0f, 1.0f};

    std::mt19937_64 rng(19);
    int alternativeCount = 0;
    for (int sample = 0; sample < 1000; ++sample) {
        const qr::Move move = qr::sampleMoveByVisitTemperature(root, 1.0, rng);
        assert(move == root.moves[0] || move == root.moves[1]);
        alternativeCount += move == root.moves[1];
    }
    assert(alternativeCount > 40);
    assert(alternativeCount < 160);
}

void testNoVisitsUsesFirstLegalMove() {
    RootVisits root;
    root.moves.push_back(qr::Move::pawn(10));
    root.moves.push_back(qr::Move::pawn(11));
    root.N = {0.0f, 0.0f};

    std::mt19937_64 rng(23);
    assert(qr::sampleMoveByVisitTemperature(root, 1.0, rng) == root.moves[0]);
}

void testZeroVisitEdgesHaveNoSamplingMass() {
    RootVisits root;
    root.moves.push_back(qr::Move::pawn(10));
    root.moves.push_back(qr::Move::pawn(11));
    root.moves.push_back(qr::Move::pawn(12));
    root.N = {0.0f, 4.0f, 0.0f};

    std::mt19937_64 rng(29);
    for (int sample = 0; sample < 200; ++sample)
        assert(qr::sampleMoveByVisitTemperature(root, 1.0, rng) == root.moves[1]);
}

qr::SelfPlayConfig temperaturePlayoutConfig(double fullSearchProb) {
    qr::SelfPlayConfig cfg;
    cfg.maxDepth = 2;
    cfg.timeBudgetMs = 1;
    cfg.cheapTimeBudgetMs = 1;
    cfg.maxPlies = 200;
    cfg.playoutCapEnabled = true;
    cfg.fullSearchProb = fullSearchProb;
    cfg.mcMode = true;
    // Every ply exercises the temperature branch, including terminal games.
    cfg.mcObviousPlies = cfg.maxPlies;
    cfg.mcTempDecayPlies = 0;
    cfg.mcTemperatureObvious = 1.0;
    cfg.epsilonMidgame = 0.0;
    cfg.mcabParams.enabled = false;
    return cfg;
}

void testTemperaturePliesHonorPlayoutCap() {
    qr::Negamax engine0;
    qr::Negamax engine1;
    std::mt19937_64 rng(31);
    uint64_t nodes = 0;
    qr::SelfPlayStats fullStats;
    auto fullSamples = qr::playOneGame(engine0, engine1, rng,
                                       temperaturePlayoutConfig(1.0), nodes, fullStats);
    assert(fullStats.fullSearchPlies > 0);
    assert(!fullSamples.empty());

    rng.seed(31);
    nodes = 0;
    qr::SelfPlayStats cheapStats;
    auto cheapSamples = qr::playOneGame(engine0, engine1, rng,
                                        temperaturePlayoutConfig(0.0), nodes, cheapStats);
    assert(cheapStats.fullSearchPlies == 0);
    assert(cheapSamples.empty());
    assert(cheapStats.samplesSkipped > 0);
}

}  // namespace

int main() {
    testLowTemperatureSelectsMostVisitedMove();
    testUnitTemperatureExploresVisitedAlternatives();
    testNoVisitsUsesFirstLegalMove();
    testZeroVisitEdgesHaveNoSamplingMass();
    testTemperaturePliesHonorPlayoutCap();
    return 0;
}
