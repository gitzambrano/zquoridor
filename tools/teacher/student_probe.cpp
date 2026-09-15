// Emit canonical states and deployed integer-network outputs for parity checks.
#include "nnue.hpp"
#include <iomanip>
#include <iostream>
#include <random>

int main(int argc, char** argv) {
    if (argc != 2 || !qr::loadWeightsQuant(argv[1])) return 2;
    std::mt19937 random(20260914);
    auto state = qr::initialState();
    std::cout << std::setprecision(9);
    for (int ply = 0; ply < 96; ++ply) {
        if (qr::winner(state) != -1) state = qr::initialState();
        for (int side = 0; side < 2; ++side) {
            auto acc = qr::buildAccumulatorQuant(state, side);
            std::array<float, qr::POLICY_OUT> policy{};
            qr::forwardPolicyQuant(acc, policy);
            std::cout << "{\"own_pawn\":" << qr::mirroredPawnCell(state.pawn[side], side)
                << ",\"opp_pawn\":" << qr::mirroredPawnCell(state.pawn[1-side], side)
                << ",\"walls_h\":" << qr::mirrorWallBitboard(state.wallsH, side)
                << ",\"walls_v\":" << qr::mirrorWallBitboard(state.wallsV, side)
                << ",\"own_dist\":" << acc.ownDistBucket
                << ",\"opp_dist\":" << acc.oppDistBucket
                << ",\"walls_left_own\":" << acc.ownWallsLeftBucket
                << ",\"walls_left_opp\":" << acc.oppWallsLeftBucket
                << ",\"value\":" << qr::forwardValueWLQuant(acc) << ",\"policy\":[";
            for (int i = 0; i < qr::POLICY_OUT; ++i) {
                if (i) std::cout << ',';
                std::cout << policy[i];
            }
            std::cout << "]}\n";
        }
        auto moves = qr::legalMoves(state);
        state = qr::applyMove(state, moves[random() % moves.size()]);
    }
}
