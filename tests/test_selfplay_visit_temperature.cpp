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

}  // namespace

int main() {
    testLowTemperatureSelectsMostVisitedMove();
    testUnitTemperatureExploresVisitedAlternatives();
    testNoVisitsUsesFirstLegalMove();
    return 0;
}
