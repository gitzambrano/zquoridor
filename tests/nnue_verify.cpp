// nnue_verify.cpp -- parity check for the NNUE float and quantized paths.
#include <cstdio>
#include <cstdlib>
#include "rules.hpp"
#include "nnue.hpp"
using namespace qr;

int main(int argc, char** argv) {
    if (argc != 2 && argc != 3) {
        std::fprintf(stderr, "usage: %s <float_weights.bin> [quant_weights.bin]\n", argv[0]);
        return 1;
    }
    if (!loadWeights(argv[1])) {
        std::fprintf(stderr, "cannot load %s\n", argv[1]);
        return 1;
    }
    bool haveQuant = false;
    if (argc == 3) {
        haveQuant = loadWeightsQuant(argv[2]);
        if (!haveQuant) {
            std::fprintf(stderr, "cannot load %s\n", argv[2]);
            return 1;
        }
    }

    State s = initialState();
    s.wallsH |= (1ull << slotIdx(3, 4));
    s.wallsV |= (1ull << slotIdx(5, 2));

    Accumulator accF[2] = {buildAccumulator(s, 0), buildAccumulator(s, 1)};
    AccumulatorQuant accQ[2] = {buildAccumulatorQuant(s, 0), buildAccumulatorQuant(s, 1)};

    for (int perspective = 0; perspective < 2; perspective++) {
        int opponent = 1 - perspective;
        float valueWL = forwardValueWL(accF[perspective], accF[opponent]);
        std::array<float, POLICY_OUT> policy;
        forwardPolicy(accF[perspective], accF[opponent], policy);

        int bestIdx = 0;
        for (int i = 1; i < POLICY_OUT; i++) {
            if (policy[i] > policy[bestIdx]) bestIdx = i;
        }

        std::printf("perspective=%d value_wl=%.6f argmax_policy=%d policy[argmax]=%.6f (float32)\n",
                    perspective, valueWL, bestIdx, policy[bestIdx]);
        std::printf("  policy[0..4] = %.6f %.6f %.6f %.6f %.6f\n",
                    policy[0], policy[1], policy[2], policy[3], policy[4]);

        if (haveQuant) {
            float valueWLQ = forwardValueWLQuant(accQ[perspective], accQ[opponent]);
            std::array<float, POLICY_OUT> policyQ;
            forwardPolicyQuant(accQ[perspective], accQ[opponent], policyQ);

            int bestIdxQ = 0;
            for (int i = 1; i < POLICY_OUT; i++) {
                if (policyQ[i] > policyQ[bestIdxQ]) bestIdxQ = i;
            }

            std::printf("perspective=%d value_wl=%.6f argmax_policy=%d policy[argmax]=%.6f "
                        "(int8, value_error=%.6f, argmax_match=%s)\n",
                        perspective, valueWLQ, bestIdxQ, policyQ[bestIdxQ],
                        valueWL - valueWLQ, bestIdx == bestIdxQ ? "yes" : "no");
            std::printf("  policy[0..4] = %.6f %.6f %.6f %.6f %.6f\n",
                        policyQ[0], policyQ[1], policyQ[2], policyQ[3], policyQ[4]);
        }
    }
    return 0;
}
