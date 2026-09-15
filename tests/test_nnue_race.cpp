#include "nnue.hpp"
#include <cassert>
#include <iostream>
#include <random>

int main() {
    static_assert(qr::NUM_FEATURES == 456, "build this test with race features");
    auto indices = qr::raceFeatureIndices(3, 5, 0, 4);
    assert(indices[0] == 368 && indices[1] == 393 && indices[2] == 411);
    std::mt19937 rng(42);
    // Nonzero random weights ensure an omitted incremental update is visible.
    for (auto& row : qr::weightsQuant().w1)
        for (auto& value : row) value = int(rng() % 21) - 10;
    for (int game = 0; game < 8; ++game) {
        auto s = qr::initialState();
        auto a = qr::buildAccumulatorQuant(s, 0);
        auto b = qr::buildAccumulatorQuant(s, 1);
        for (int ply = 0; ply < 80 && qr::winner(s) < 0; ++ply) {
            auto moves = qr::legalMoves(s);
            auto m = moves[rng() % moves.size()];
            qr::updateAccumulatorForMoveQuant(a, s.turn == 0, s, m);
            qr::updateAccumulatorForMoveQuant(b, s.turn == 1, s, m);
            s = qr::applyMove(s, m);
            assert(a.v == qr::buildAccumulatorQuant(s, 0).v);
            assert(b.v == qr::buildAccumulatorQuant(s, 1).v);
        }
    }
    std::cout << "Race feature incremental checks passed\n";
}
