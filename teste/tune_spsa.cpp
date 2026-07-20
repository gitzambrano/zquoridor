// tune_spsa.cpp -- SPSA (Simultaneous Perturbation Stochastic Approximation)
// para os 6 pesos de evalSimple (Fase 4.2.10). Recriado do zero nesta
// rodada -- o tune_spsa.cpp original citado nos comentários de rules.hpp
// não fazia parte do repositório entregue (usuário confirmou que já não
// era mais necessário depois do SPSA anterior, mas agora precisamos de um
// novo por causa do 6º parâmetro, robustnessWeight, que não existia antes).
//
// Design: cada iteração testa uma direção aleatória de perturbação nos 6
// pesos simultaneamente (SPSA clássico, Spall 1998) via UM PAR de partidas
// cabeça-a-cabeça (theta+c*delta vs theta-c*delta, depois com as cores
// trocadas -- antitético, remove o viés de quem começa). O resultado
// agregado das 2 partidas estima o gradiente em UMA chamada de "função
// objetivo" por iteração (2 partidas), não em 6 (uma por peso) -- é
// exatamente a vantagem do SPSA sobre um coordinate-descent ingênuo
// quando a "avaliação" é cara (aqui, jogar uma partida real).
//
// Escala por parâmetro: os 6 pesos têm magnitudes muito diferentes
// (distWeight~11 vs urgencyScale~0.64), então tanto o passo de
// perturbação (c_k) quanto o tamanho do passo de atualização (a_k) são
// escalados proporcionalmente ao valor inicial de CADA parâmetro (prática
// padrão de SPSA multi-escala), não um escalar único pra todos.
#include <cstdio>
#include <cmath>
#include <random>
#include <array>
#include <chrono>
#include <sstream>
#include "rules.hpp"
#include "search.hpp"
using namespace qr;

constexpr int NPARAM = 6;
using Vec6 = std::array<double, NPARAM>;

const char* PARAM_NAMES[NPARAM] = {
    "distWeight", "urgencyScale", "wallWeightClose", "wallWeightFar", "mobWeight", "robustnessWeight"
};

Vec6 toVec(const EvalWeights& w) {
    return {w.distWeight, w.urgencyScale, w.wallWeightClose, w.wallWeightFar, w.mobWeight, w.robustnessWeight};
}
EvalWeights toWeights(const Vec6& v) {
    EvalWeights w;
    w.distWeight = v[0]; w.urgencyScale = v[1]; w.wallWeightClose = v[2];
    w.wallWeightFar = v[3]; w.mobWeight = v[4]; w.robustnessWeight = v[5];
    return w;
}

// --- config da partida de auto-jogo (rápida de propósito -- SPSA precisa
// de MUITAS partidas, não de partidas profundas) ---------------------
constexpr int SEARCH_DEPTH = 3;
constexpr int TIME_BUDGET_MS = 120;
constexpr int MAX_PLIES = 70;
constexpr int OPENING_RANDOM_PLIES = 2;  // diversidade de abertura

// joga 1 partida: engineByPlayer[0] controla o peão 0, [1] controla o 1.
// devolve 0 ou 1 (vencedor) ou -1 (empate/estourou MAX_PLIES).
int playGame(Negamax& eng0, Negamax& eng1, std::mt19937& rng) {
    State s = initialState();
    for (int i = 0; i < OPENING_RANDOM_PLIES; i++) {
        auto moves = legalMoves(s);
        if (winner(s) != -1) return winner(s);
        std::uniform_int_distribution<size_t> dist(0, moves.size() - 1);
        s = applyMove(s, moves[dist(rng)]);
    }
    for (int ply = 0; ply < MAX_PLIES; ply++) {
        int w = winner(s);
        if (w != -1) return w;
        Negamax& eng = (s.turn == 0) ? eng0 : eng1;
        SearchStats st;
        Move m = eng.chooseMove(s, SEARCH_DEPTH, TIME_BUDGET_MS, st);
        s = applyMove(s, m);
    }
    return -1;  // partida longa demais -- não decide (tratado como 0 no gradiente)
}

// partida antitética: plus controla peão0 na 1a partida / peão1 na 2a
// (troca de cor), pra cancelar a vantagem estrutural de quem começa.
// devolve score em [-1,+1]: +1 = plus venceu as 2, -1 = minus venceu as 2.
double antitheticMatch(const EvalWeights& wPlus, const EvalWeights& wMinus, std::mt19937& rng) {
    Negamax ePlus(wPlus), eMinus(wMinus);
    double score = 0.0;
    int r1 = playGame(ePlus, eMinus, rng);   // plus=jogador0, minus=jogador1
    if (r1 == 0) score += 1.0; else if (r1 == 1) score -= 1.0;
    int r2 = playGame(eMinus, ePlus, rng);   // cores trocadas
    if (r2 == 1) score += 1.0; else if (r2 == 0) score -= 1.0;
    return score / 2.0;
}

int main(int argc, char** argv) {
    int totalIterations = (argc > 1) ? atoi(argv[1]) : 40;
    unsigned seed = (argc > 2) ? (unsigned)atoi(argv[2]) : 20260719u;
    double timeBudgetSec = (argc > 3) ? atof(argv[3]) : 1e18;
    const char* ckptPath = "spsa_checkpoint.txt";

    std::mt19937 rng(seed);
    Vec6 theta0 = toVec(evalWeights());
    Vec6 theta = theta0;
    Vec6 thetaAvg = theta0;
    int avgCount = 0;
    int kStart = 0;

    // retoma de um checkpoint em disco, se existir (rodar em lotes --
    // processos em background não sobrevivem entre chamadas da ferramenta
    // de shell neste ambiente, então persistimos em arquivo em vez disso).
    FILE* ckptIn = fopen(ckptPath, "r");
    if (ckptIn) {
        fscanf(ckptIn, "%d %d", &kStart, &avgCount);
        for (int i = 0; i < NPARAM; i++) fscanf(ckptIn, "%lf", &theta[i]);
        for (int i = 0; i < NPARAM; i++) fscanf(ckptIn, "%lf", &thetaAvg[i]);
        std::stringstream rngState;
        char buf[8192]; size_t n;
        while ((n = fread(buf, 1, sizeof(buf), ckptIn)) > 0) rngState.write(buf, n);
        rngState >> rng;
        fclose(ckptIn);
        printf("retomando checkpoint: iteracao %d, avgCount=%d\n", kStart, avgCount);
    }

    // bounds por parâmetro: [0.1x, 4x] do valor inicial -- evita que o
    // ruído de partidas curtas/rasas mande um peso pra região absurda
    // (ex.: negativo, o que inverteria o sinal do termo).
    Vec6 lo, hi;
    for (int i = 0; i < NPARAM; i++) { lo[i] = 0.1 * theta0[i]; hi[i] = 4.0 * theta0[i]; }

    // Spall (1998), valores-padrão recomendados de alpha/gamma.
    const double alpha = 0.602, gamma = 0.101;
    const double bigA = 0.1 * totalIterations;
    // c0/a0 por parâmetro, proporcional à magnitude inicial (15%/8%).
    Vec6 c0, a0;
    for (int i = 0; i < NPARAM; i++) { c0[i] = 0.15 * theta0[i]; a0[i] = 0.08 * theta0[i]; }

    if (kStart == 0) {
        printf("SPSA -- %d iteracoes totais, profundidade=%d, %dms/lance, %d plies aleatorios de abertura\n",
               totalIterations, SEARCH_DEPTH, TIME_BUDGET_MS, OPENING_RANDOM_PLIES);
        printf("theta0: ");
        for (int i = 0; i < NPARAM; i++) printf("%s=%.3f ", PARAM_NAMES[i], theta0[i]);
        printf("\n\n");
    }

    auto t0 = std::chrono::steady_clock::now();
    const int avgStart = totalIterations / 2;  // só faz média da 2a metade (fase mais estável)
    int k = kStart;

    for (; k < totalIterations; k++) {
        double elapsedSoFar = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        if (elapsedSoFar > timeBudgetSec) break;

        double ck_scale = 1.0 / std::pow(k + 1, gamma);
        double ak_scale = 1.0 / std::pow(k + 1 + bigA, alpha);

        std::bernoulli_distribution coin(0.5);
        std::array<int, NPARAM> delta;
        for (int i = 0; i < NPARAM; i++) delta[i] = coin(rng) ? 1 : -1;

        Vec6 thetaPlus, thetaMinus;
        for (int i = 0; i < NPARAM; i++) {
            double ck = c0[i] * ck_scale;
            thetaPlus[i]  = std::clamp(theta[i] + ck * delta[i], lo[i], hi[i]);
            thetaMinus[i] = std::clamp(theta[i] - ck * delta[i], lo[i], hi[i]);
        }

        double score = antitheticMatch(toWeights(thetaPlus), toWeights(thetaMinus), rng);

        for (int i = 0; i < NPARAM; i++) {
            double ck = c0[i] * ck_scale;
            double ak = a0[i] * ak_scale;
            double ghat = score * delta[i] / (2.0 * ck);
            theta[i] = std::clamp(theta[i] + ak * ghat, lo[i], hi[i]);
        }

        if (k >= avgStart) {
            avgCount++;
            for (int i = 0; i < NPARAM; i++)
                thetaAvg[i] += (theta[i] - thetaAvg[i]) / avgCount;
        }

        auto now = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(now - t0).count();
        printf("iter %3d/%d score=%+.2f elapsed=%.0fs theta: ", k + 1, totalIterations, score, elapsed);
        for (int i = 0; i < NPARAM; i++) printf("%.3f ", theta[i]);
        printf("\n");
        fflush(stdout);
    }

    // salva checkpoint (se ainda não terminou tudo) ou resultado final
    FILE* ckptOut = fopen(ckptPath, "w");
    if (ckptOut) {
        fprintf(ckptOut, "%d %d\n", k, avgCount);
        for (int i = 0; i < NPARAM; i++) fprintf(ckptOut, "%.10f ", theta[i]);
        fprintf(ckptOut, "\n");
        for (int i = 0; i < NPARAM; i++) fprintf(ckptOut, "%.10f ", thetaAvg[i]);
        fprintf(ckptOut, "\n");
        std::stringstream rngState;
        rngState << rng;
        fprintf(ckptOut, "%s", rngState.str().c_str());
        fclose(ckptOut);
    }

    if (k >= totalIterations) {
        printf("\n=== CONCLUIDO -- resultado (media de Polyak da 2a metade das iteracoes) ===\n");
        for (int i = 0; i < NPARAM; i++)
            printf("%s: %.4f (era %.4f)\n", PARAM_NAMES[i], thetaAvg[i], theta0[i]);

        FILE* f = fopen("spsa_result.txt", "w");
        if (f) {
            for (int i = 0; i < NPARAM; i++) fprintf(f, "%.6f\n", thetaAvg[i]);
            fclose(f);
        }
    } else {
        printf("\n=== lote parcial salvo em checkpoint (%d/%d iteracoes feitas) -- rode de novo pra continuar ===\n", k, totalIterations);
    }
    return 0;
}
