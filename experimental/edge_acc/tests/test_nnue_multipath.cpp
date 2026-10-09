#define ZQ_NNUE_MULTIPATH_FEATURES 1
#include "nnue.hpp"
#include <cassert>
#include <iostream>
#include <random>

int main() {
    static_assert(qr::NUM_FEATURES == 480, "build this test with multipath features (480)");
    
    // Check initial state
    auto s0 = qr::initialState();
    auto mp0 = qr::getMultipathFeatures(s0, 0);
    // At start, both pawns are in row 0/8, col 4:
    // own pawn has 3 unblocked exits (Forward/South, Left/West, Right/East); Backward/North is off-board
    // exit count = 3
    assert(mp0.count > 0);
    
    std::mt19937 rng(42);
    // Nonzero random weights ensure an omitted incremental update is immediately visible
    for (auto& row : qr::weightsQuant().w1)
        for (auto& value : row) value = int(rng() % 21) - 10;
        
    int totalPliesChecked = 0;
    for (int game = 0; game < 16; ++game) {
        auto s = qr::initialState();
        auto a = qr::buildAccumulatorQuant(s, 0);
        auto b = qr::buildAccumulatorQuant(s, 1);
        for (int ply = 0; ply < 100 && qr::winner(s) < 0; ++ply) {
            auto moves = qr::legalMoves(s);
            auto m = moves[rng() % moves.size()];
            
            qr::updateAccumulatorForMoveQuant(a, s.turn == 0, s, m);
            qr::updateAccumulatorForMoveQuant(b, s.turn == 1, s, m);
            s = qr::applyMove(s, m);
            
            auto a_rebuilt = qr::buildAccumulatorQuant(s, 0);
            auto b_rebuilt = qr::buildAccumulatorQuant(s, 1);
            
            assert(a.v == a_rebuilt.v);
            assert(b.v == b_rebuilt.v);
            totalPliesChecked++;
        }
    }
    std::cout << "Multipath feature incremental checks passed (" << totalPliesChecked << " plies verified)\n";
    return 0;
}
