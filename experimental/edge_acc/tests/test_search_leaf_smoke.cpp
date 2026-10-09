// Smoke test isolado (não faz parte do plano formal, só validação manual da
// Fase 0 antes de seguir para a Fase 1): confirma que resetOrderingState()
// público e searchLeaf() compilam e funcionam, em modo Heuristic e NNUE
// (com e sem seedAcc), e que os resultados batem com chooseMove()/negamax
// já existentes na mesma profundidade a partir da raiz.
#include <cstdio>
#include <cassert>
#include "search.hpp"

using namespace qr;

int main() {
    State root = initialState();

    // Sem pesos carregados o modo NNUE avalia tudo como 0 e os asserts
    // abaixo passariam por empate trivial (0 == 0), sem provar nada. Melhor
    // esforço: se os pesos publicados estiverem no repo, carrega.
    // Convenção do projeto: binários rodam a partir da RAIZ do repo.
    bool haveWeights = loadWeightsQuant("data/nnue/nnue_weights_int8.bin");
    printf("[setup] pesos NNUE %s\n",
           haveWeights ? "carregados" : "NAO carregados (modo NNUE avaliara 0 em tudo)");

    // Modo Heuristic: searchLeaf deve bater com searchShallow na mesma posicao/profundidade.
    {
        Negamax eng;
        eng.setEvalMode(Negamax::EvalMode::Heuristic);
        SearchStats stats1, stats2;
        int viaShallow = eng.searchShallow(root, 3, stats1);

        eng.resetOrderingState();
        eng.clearTT();
        RepetitionTable reptbl;
        int viaLeaf = eng.searchLeaf(root, 3, stats2, reptbl);
        printf("[Heuristic] searchShallow=%d searchLeaf=%d\n", viaShallow, viaLeaf);
        assert(viaShallow == viaLeaf);
    }

    // Modo NNUE, sem seedAcc (deve cair no rebuild via buildAccPairRoot).
    {
        Negamax eng;
        eng.setEvalMode(Negamax::EvalMode::NNUE);
        SearchStats stats1, stats2;
        int viaShallow = eng.searchShallow(root, 3, stats1);

        eng.resetOrderingState();
        eng.clearTT();
        RepetitionTable reptbl;
        int viaLeaf = eng.searchLeaf(root, 3, stats2, reptbl, nullptr);
        printf("[NNUE, sem seedAcc] searchShallow=%d searchLeaf=%d\n", viaShallow, viaLeaf);
        assert(viaShallow == viaLeaf);
    }

    // Modo NNUE, com seedAcc pre-construido via buildAccPairRoot (simula o
    // uso real do MCABSearch, que constroi o acumulador incrementalmente).
    {
        Negamax eng;
        eng.setEvalMode(Negamax::EvalMode::NNUE);
        SearchStats stats1, stats2;
        int viaShallow = eng.searchShallow(root, 3, stats1);

        PlayerPathCacheTable localCache;
        AccPair seed = buildAccPairRoot(root, &localCache);

        eng.resetOrderingState();
        eng.clearTT();
        RepetitionTable reptbl;
        int viaLeaf = eng.searchLeaf(root, 3, stats2, reptbl, &seed);
        printf("[NNUE, com seedAcc] searchShallow=%d searchLeaf=%d\n", viaShallow, viaLeaf);
        assert(viaShallow == viaLeaf);
    }

    // REGRESSAO: searchLeaf precisa reinicializar `stopped`/`deadline` a cada
    // chamada, igual searchShallow/chooseMove. Sem isso, (a) uma chooseMove
    // anterior que estourou o tempo deixa `stopped=true` grudado e toda
    // searchLeaf seguinte devolve 0 na hora, e (b) numa engine recem-criada
    // `deadline` é o epoch, o que aborta a busca ja na primeira checagem de
    // tempo. Nos dois casos o hibrido MCAB veria Q=0.5 em toda folha --
    // silencioso, porque nada crasha.
    {
        Negamax eng;
        eng.setEvalMode(Negamax::EvalMode::Heuristic);
        SearchStats stRef;
        int esperado = eng.searchShallow(root, 3, stRef);

        // Forca `stopped=true`: busca profunda com orcamento de 1ms.
        SearchStats stBurn;
        RepetitionTable histBurn;
        eng.chooseMove(root, 40, 1, stBurn, histBurn);

        // clearTT() antes de medir: sem isso a TT ainda quente da chooseMove
        // acima devolve a raiz num unico no, e `nos > 0` passaria mesmo com
        // `stopped` grudado (a checagem de TT vem antes da checagem de tempo).
        // Com a TT limpa, uma busca de profundidade 3 a partir da posicao
        // inicial tem que visitar centenas de nos.
        eng.clearTT();
        SearchStats stAfter;
        RepetitionTable reptbl;
        int depois = eng.searchLeaf(root, 3, stAfter, reptbl);
        printf("[regressao stopped/deadline] esperado=%d depois-de-timeout=%d nos=%llu\n",
               esperado, depois, (unsigned long long)stAfter.nodes);
        assert(stAfter.nodes > 100 && "searchLeaf mal buscou -- flag `stopped`/`deadline` ficou grudada");
        assert(depois == esperado && "searchLeaf devolveu valor diferente apos um timeout anterior");
    }

    printf("TODOS OS SMOKE TESTS DE searchLeaf/resetOrderingState PASSARAM\n");
    return 0;
}
