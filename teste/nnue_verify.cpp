// nnue_verify.cpp -- checagem de paridade: carrega pesos treinados (Fase 5,
// train_nnue.py) e imprime value/policy pra uma posição de teste fixa,
// codificada também em training/parity_check.py com os MESMOS índices de
// feature. Se os dois lados baterem numericamente, o pipeline completo
// (export PyTorch -> load C++ -> forward C++) está correto.
//
// Uso: ./nnue_verify data/nnue_weights.bin [data/nnue_weights_int8.bin]
// Se o segundo argumento (pesos quantizados, Seção 7.8) for passado,
// também roda o forward inteiro (int8/int16, ver NNUEWeightsQuant em
// nnue.hpp) na mesma posição e imprime lado a lado com o float32, pra
// checar visualmente o erro de quantização e a preservação do argmax da
// política -- a checagem numérica exata (comparação com o lado Python
// independente) fica em training/parity_check.py.
#include <cstdio>
#include <cstdlib>
#include "rules.hpp"
#include "nnue.hpp"
using namespace qr;

int main(int argc, char** argv) {
    if (argc != 2 && argc != 3) {
        std::fprintf(stderr, "uso: %s <pesos_float32.bin> [pesos_int8.bin]\n", argv[0]);
        return 1;
    }
    if (!loadWeights(argv[1])) { std::fprintf(stderr, "erro ao carregar %s\n", argv[1]); return 1; }
    bool haveQuant = false;
    if (argc == 3) {
        haveQuant = loadWeightsQuant(argv[2]);
        if (!haveQuant) { std::fprintf(stderr, "erro ao carregar %s\n", argv[2]); return 1; }
    }

    // posição de teste fixa: estado inicial + um muro H em (3,4) + um muro V em (5,2)
    // (mesma posição hardcoded em training/parity_check.py)
    State s = initialState();
    s.wallsH |= (1ull << slotIdx(3, 4));
    s.wallsV |= (1ull << slotIdx(5, 2));

    for (int perspective = 0; perspective < 2; perspective++) {
        Accumulator acc = buildAccumulator(s, perspective);
        float valueWL = forwardValueWL(acc);
        float valueAux = forwardValueAux(acc);
        std::array<float, POLICY_OUT> policy;
        forwardPolicy(acc, policy);

        int bestIdx = 0;
        for (int i = 1; i < POLICY_OUT; i++) if (policy[i] > policy[bestIdx]) bestIdx = i;

        std::printf("perspectiva=%d  value_wl=%.6f  value_aux=%.6f  argmax_policy=%d  policy[argmax]=%.6f  (float32)\n",
                    perspective, valueWL, valueAux, bestIdx, policy[bestIdx]);
        std::printf("  policy[0..4] = %.6f %.6f %.6f %.6f %.6f\n",
                    policy[0], policy[1], policy[2], policy[3], policy[4]);

        if (haveQuant) {
            AccumulatorQuant accQ = buildAccumulatorQuant(s, perspective);
            float valueWLQ = forwardValueWLQuant(accQ);
            float valueAuxQ = forwardValueAuxQuant(accQ);
            std::array<float, POLICY_OUT> policyQ;
            forwardPolicyQuant(accQ, policyQ);

            int bestIdxQ = 0;
            for (int i = 1; i < POLICY_OUT; i++) if (policyQ[i] > policyQ[bestIdxQ]) bestIdxQ = i;

            std::printf("perspectiva=%d  value_wl=%.6f  value_aux=%.6f  argmax_policy=%d  policy[argmax]=%.6f  (int8, "
                        "erro_wl=%.6f, erro_aux=%.6f, argmax_bate=%s)\n",
                        perspective, valueWLQ, valueAuxQ, bestIdxQ, policyQ[bestIdxQ],
                        valueWL - valueWLQ, valueAux - valueAuxQ, bestIdx == bestIdxQ ? "sim" : "NAO");
            std::printf("  policy[0..4] = %.6f %.6f %.6f %.6f %.6f\n",
                        policyQ[0], policyQ[1], policyQ[2], policyQ[3], policyQ[4]);
        }
    }
    return 0;
}
