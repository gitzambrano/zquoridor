// bench_quiescence_toggle.cpp -- compara com/sem a extensão de
// quiescência de muro (Fase 4.2.10, item 3) usando a flag runtime
// setQuiescenceEnabled (search.hpp).
//
// Ruído do orçamento de tempo (200ms/lance, o `bench` de main.cpp) já
// documentado como grande o bastante pra mascarar sinais de poucos %:
// profundidade varia de rodada pra rodada mesmo sem tocar em nada.
// Metodologia aqui é a mesma usada pra medir o ganho do contador de
// geração: profundidade FIXA (maxDepthCap == cap real, timeBudgetMs
// gigante pra nunca cortar por tempo), mesma sequência FIXA e
// determinística de posições nos dois lados da comparação (gerada uma
// vez com quiescência LIGADA -- não é "self-play com o motor
// enfraquecido jogando contra si mesmo", é só uma trilha de posições
// plausíveis reaproveitada igual nos dois braços do A/B) -- assim o
// número de nós/tempo medido é 100% atribuível à flag, não a variação
// de posição ou de profundidade alcançada.
#include <cstdio>
#include <chrono>
#include <vector>
#include "rules.hpp"
#include "search.hpp"
using namespace qr;
using clockT = std::chrono::steady_clock;

static double msSince(clockT::time_point t0) {
    return std::chrono::duration<double, std::milli>(clockT::now() - t0).count();
}

// gera uma trilha fixa de N posições jogando com quiescência LIGADA,
// profundidade fixa (não tempo) -- serve só pra ter posições variadas
// e reprodutíveis, não faz parte do que é medido.
static std::vector<State> buildFixedTrajectory(int numPositions, int genDepth) {
    std::vector<State> traj;
    Negamax engine;  // quiescenceEnabled = true por default
    State s = initialState();
    traj.push_back(s);
    for (int i = 1; i < numPositions; i++) {
        if (winner(s) != -1) break;
        SearchStats st;
        Move m = engine.chooseMove(s, /*maxDepthCap=*/genDepth, /*timeBudgetMs=*/600000, st);
        s = applyMove(s, m);
        traj.push_back(s);
    }
    return traj;
}

struct RunResult {
    uint64_t totalNodes = 0;
    double totalMs = 0.0;
};

// roda a MESMA trilha fixa de posições, profundidade fixa, com a flag
// de quiescência no valor pedido -- 1 instância de Negamax nova por
// posição (evita reuso de TT entre posições da trilha contaminar a
// comparação; queremos custo "frio" e comparável posição-a-posição).
static RunResult runFixedTrajectory(const std::vector<State>& traj, int searchDepth, bool quiescence) {
    RunResult r;
    for (const State& s : traj) {
        if (winner(s) != -1) continue;
        Negamax engine;
        engine.setQuiescenceEnabled(quiescence);
        SearchStats st;
        auto t0 = clockT::now();
        engine.chooseMove(s, /*maxDepthCap=*/searchDepth, /*timeBudgetMs=*/600000, st);
        r.totalMs += msSince(t0);
        r.totalNodes += st.nodes;
    }
    return r;
}

int main() {
    const int NUM_POSITIONS = 24;
    const int GEN_DEPTH = 3;     // profundidade rasa só pra gerar a trilha rápido
    const int SEARCH_DEPTH = 5;  // profundidade da medição em si

    printf("=== gerando trilha fixa de %d posicoes (quiescencia ligada, profundidade %d) ===\n",
           NUM_POSITIONS, GEN_DEPTH);
    std::vector<State> traj = buildFixedTrajectory(NUM_POSITIONS, GEN_DEPTH);
    printf("trilha gerada: %zu posicoes\n\n", traj.size());

    printf("=== medindo profundidade fixa=%d, SEM quiescencia ===\n", SEARCH_DEPTH);
    RunResult off = runFixedTrajectory(traj, SEARCH_DEPTH, false);
    printf("nos totais: %llu, tempo: %.1f ms -> %.0f nos/seg\n",
           (unsigned long long)off.totalNodes, off.totalMs, off.totalNodes / (off.totalMs / 1000.0));

    printf("\n=== medindo profundidade fixa=%d, COM quiescencia ===\n", SEARCH_DEPTH);
    RunResult on = runFixedTrajectory(traj, SEARCH_DEPTH, true);
    printf("nos totais: %llu, tempo: %.1f ms -> %.0f nos/seg\n",
           (unsigned long long)on.totalNodes, on.totalMs, on.totalNodes / (on.totalMs / 1000.0));

    double nodesRatio = (double)on.totalNodes / (double)off.totalNodes;
    double nodesPerSecOff = off.totalNodes / (off.totalMs / 1000.0);
    double nodesPerSecOn = on.totalNodes / (on.totalMs / 1000.0);
    double throughputRatio = nodesPerSecOn / nodesPerSecOff;

    printf("\n=== resumo ===\n");
    printf("nos com quiescencia / nos sem quiescencia: %.3fx (>1 = quiescencia explora mais nos, esperado -- extensao)\n", nodesRatio);
    printf("nos/seg com / nos/seg sem: %.3fx (mede custo de throughput por no da propria extensao, isolado do total de nos)\n", throughputRatio);
    printf("tempo total com / sem: %.3fx\n", on.totalMs / off.totalMs);
    return 0;
}
