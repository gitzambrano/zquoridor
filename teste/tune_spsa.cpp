// tune_spsa.cpp -- SPSA (Simultaneous Perturbation Stochastic Approximation)
// para os parâmetros de busca/ordenação que interagem com a NNUE
// (Fase 4.2.10+). Reescrito nesta rodada: a versão anterior tunava os 6
// pesos de evalSimple (EvalWeights) -- ficou obsoleta porque o motor
// virou NNUE-first (selfplay/arena/wasm já ligam EvalMode::NNUE +
// policy ordering por default) e o pedido explícito desta rodada é NÃO
// tunar mais a heurística, só os parâmetros que afetam avaliação/
// ordenação em modo NNUE:
//
//   contempt          -- Negamax::setContempt/getContempt (era CONTEMPT)
//   policyOrderScale  -- Negamax::setPolicyOrderScale/getPolicyOrderScale
//                         (era POLICY_ORDER_SCALE) -- escala do logit cru
//                         da cabeça de política somado na ordenação.
//   catScoreScale     -- Negamax::setCatScoreScale/getCatScoreScale (era
//                         CAT_SCORE_SCALE) -- peso do calor CAT vs. o
//                         termo de política no MESMO score de
//                         orderWallMoves; é o parâmetro que decide o
//                         trade-off policy-vs-CAT.
//   policyOrderingMinDepth -- discreto (piso de profundidade em que o
//                         forward pass extra da política é pago), tratado
//                         à parte de SPSA (ver "SWEEP" abaixo).
//
// search.hpp expõe os três primeiros como membros de instância com
// getters/setters -- é a mudança que viabiliza rodar 2 engines com
// valores DIFERENTES no mesmo processo (essencial para self-play de
// tuning), igual ao que já era feito com EvalWeights.
//
// Outros candidatos considerados e DESCARTADOS desta rodada (ver
// resposta que acompanha esta entrega para a justificativa completa):
// CAT_HOT_CM/CAT_COLD_CM/LMR_DIVISOR (afetam redução de LMR, não são
// específicos de NNUE -- mudam a busca igual em modo heurístico) e
// NNUE_EVAL_SCALE (constante de CALIBRAÇÃO compartilhada com o treino
// via VALUE_SCALE em train_nnue.py -- tunar só do lado da busca
// descalibraria a rede sem re-treinar).
//
// DESIGN -- 2 modos, configuráveis via CLI (o driver Python
// teste/run_spsa.py expõe as mesmas opções como variáveis no topo do
// arquivo, sem precisar editar C++ nem recompilar por mudança de config):
//
//   --mode spsa   (default): SPSA clássico (Spall 1998) sobre o
//     subconjunto de {contempt, policyOrderScale, catScoreScale} marcado
//     via --tune-contempt/--tune-policy-scale/--tune-cat-scale (todos
//     ligados por default). Cada iteração testa UMA direção aleatória de
//     perturbação simultânea em TODOS os parâmetros ativos via um par de
//     partidas cabeça-a-cabeça antitético (theta+c*delta vs theta-c*delta,
//     depois cores trocadas) -- 1 "avaliação" de gradiente por iteração,
//     independente de quantos parâmetros estão ativos (a vantagem clássica
//     do SPSA sobre coordinate descent quando avaliar é caro).
//
//   --mode sweep-mindepth: policyOrderingMinDepth é discreto e de faixa
//     pequena (tipicamente 0-6) -- não se presta bem a SPSA contínuo
//     (perturbação +-c em torno de um inteiro pequeno quase sempre cai
//     fora do domínio útil). Em vez disso, roda um mini-torneio round-
//     robin: cada valor candidato (--mindepth-candidates, lista separada
//     por vírgula) joga --mindepth-games partidas antitéticas contra cada
//     outro candidato, com contempt/policyOrderScale/catScoreScale FIXOS
//     no valor atual (--contempt/--policy-scale/--cat-scale, ou o default
//     de search.hpp se omitidos). Imprime uma tabela de win-rate agregado
//     por candidato -- a escolha final é do usuário (não há um "melhor"
//     único sem também pesar nós/s, que este programa não mede).
//
// NNUE + policy ordering: LIGADOS por default em ambos os modos (o
// pedido desta rodada é que o tuner reflita o mesmo default já usado por
// selfplay/arena/wasm) -- requer --nnue-weights apontando para um
// arquivo válido (default: data/nnue/nnue_weights_int8.bin, mesmo
// default de defaultNnueWeightsPath() em nnue.hpp). --heuristic desliga
// tudo isso e cai para evalSimple puro, útil só para comparação/depuração
// (SPSA sobre contempt ainda funciona em modo heurístico, mas
// catScoreScale/policyOrderScale não têm efeito nenhum sem NNUE).
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <random>
#include <vector>
#include <array>
#include <string>
#include <chrono>
#include <sstream>
#include <algorithm>
#include "rules.hpp"
#include "search.hpp"
#include "nnue.hpp"
using namespace qr;

// =============================================================================
// PARÂMETROS TUNÁVEIS -- lista genérica (em vez de um struct Vec6 fixo
// como na versão anterior) para não precisar editar este arquivo toda vez
// que um parâmetro entra/sai do conjunto tunado; teste/run_spsa.py
// controla o subconjunto ativo via flags, este arquivo só precisa saber
// como LER/ESCREVER cada um num Negamax (getter/setter) e default.
// =============================================================================
struct TunableParam {
    const char* name;
    double value;          // valor corrente (double por uniformidade; contempt/
                            // catScoreScale são inteiros na engine, arredondados
                            // ao aplicar -- ver applyParams)
    double lo, hi;          // bounds absolutos (não relativos ao inicial -- ver
                             // nota em main() sobre contempt poder ser negativo)
    bool active;             // participa do SPSA nesta rodada?
};

enum ParamId { P_CONTEMPT = 0, P_POLICY_SCALE = 1, P_CAT_SCALE = 2, NPARAM_MAX = 3 };

void applyParams(Negamax& eng, const std::vector<TunableParam>& params) {
    for (size_t i = 0; i < params.size(); i++) {
        const auto& p = params[i];
        if (i == P_CONTEMPT) eng.setContempt((int)std::lround(p.value));
        else if (i == P_POLICY_SCALE) eng.setPolicyOrderScale((long long)std::lround(p.value));
        else if (i == P_CAT_SCALE) eng.setCatScoreScale((long long)std::lround(p.value));
    }
}

// --- config da partida de auto-jogo (rápida de propósito -- SPSA precisa
// de MUITAS partidas, não de partidas profundas) ---------------------
int g_searchDepth = 4;
int g_timeBudgetMs = 120;
int g_maxPlies = 70;
int g_openingRandomPlies = 2;
int g_policyOrderMinDepth = 3;
bool g_useNNUE = true;
bool g_policyOrdering = true;

void configureEngine(Negamax& eng) {
    if (g_useNNUE) {
        eng.setEvalMode(Negamax::EvalMode::NNUE);
        if (g_policyOrdering) {
            eng.setPolicyOrderingEnabled(true);
            eng.setPolicyOrderingMinDepth(g_policyOrderMinDepth);
        }
    }
}

// joga 1 partida: engineByPlayer[0] controla o peão 0, [1] controla o 1.
// devolve 0 ou 1 (vencedor) ou -1 (empate/estourou MAX_PLIES).
int playGame(Negamax& eng0, Negamax& eng1, std::mt19937& rng) {
    State s = initialState();
    for (int i = 0; i < g_openingRandomPlies; i++) {
        auto moves = legalMoves(s);
        if (winner(s) != -1) return winner(s);
        std::uniform_int_distribution<size_t> dist(0, moves.size() - 1);
        s = applyMove(s, moves[dist(rng)]);
    }
    for (int ply = 0; ply < g_maxPlies; ply++) {
        int w = winner(s);
        if (w != -1) return w;
        Negamax& eng = (s.turn == 0) ? eng0 : eng1;
        SearchStats st;
        Move m = eng.chooseMove(s, g_searchDepth, g_timeBudgetMs, st);
        s = applyMove(s, m);
    }
    return -1;  // partida longa demais -- não decide (tratado como 0 no gradiente)
}

// partida antitética: plus controla peão0 na 1a partida / peão1 na 2a
// (troca de cor), pra cancelar a vantagem estrutural de quem começa.
// devolve score em [-1,+1]: +1 = plus venceu as 2, -1 = minus venceu as 2.
double antitheticMatch(const std::vector<TunableParam>& paramsPlus,
                        const std::vector<TunableParam>& paramsMinus,
                        std::mt19937& rng) {
    Negamax ePlus, eMinus;
    configureEngine(ePlus);
    configureEngine(eMinus);
    applyParams(ePlus, paramsPlus);
    applyParams(eMinus, paramsMinus);
    double score = 0.0;
    int r1 = playGame(ePlus, eMinus, rng);   // plus=jogador0, minus=jogador1
    if (r1 == 0) score += 1.0; else if (r1 == 1) score -= 1.0;
    int r2 = playGame(eMinus, ePlus, rng);   // cores trocadas
    if (r2 == 1) score += 1.0; else if (r2 == 0) score -= 1.0;
    return score / 2.0;
}

// --- SWEEP: mini-torneio round-robin sobre valores discretos de
// policyOrderingMinDepth, com os parâmetros contínuos fixos. ------------
void runSweepMinDepth(const std::vector<int>& candidates, int gamesPerPair,
                       const std::vector<TunableParam>& fixedParams,
                       unsigned seed) {
    int n = (int)candidates.size();
    std::vector<double> scoreSum(n, 0.0);
    std::vector<int> gamesPlayed(n, 0);
    std::mt19937 rng(seed);

    printf("=== SWEEP policyOrderingMinDepth -- candidatos: ");
    for (int d : candidates) printf("%d ", d);
    printf("(%d partidas antiteticas por par) ===\n\n", gamesPerPair);

    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            double pairScore = 0.0;
            for (int g = 0; g < gamesPerPair; g++) {
                Negamax eA, eB;
                configureEngine(eA); configureEngine(eB);
                eA.setPolicyOrderingMinDepth(candidates[i]);
                eB.setPolicyOrderingMinDepth(candidates[j]);
                applyParams(eA, fixedParams);
                applyParams(eB, fixedParams);
                double s = 0.0;
                int r1 = playGame(eA, eB, rng);
                if (r1 == 0) s += 1.0; else if (r1 == 1) s -= 1.0;
                int r2 = playGame(eB, eA, rng);
                if (r2 == 1) s += 1.0; else if (r2 == 0) s -= 1.0;
                s /= 2.0;  // [-1,+1] do ponto de vista de i
                pairScore += s;
            }
            pairScore /= gamesPerPair;
            scoreSum[i] += pairScore;      gamesPlayed[i]++;
            scoreSum[j] -= pairScore;      gamesPlayed[j]++;
            printf("  minDepth=%d vs minDepth=%d: score(%d)=%+.3f\n",
                   candidates[i], candidates[j], candidates[i], pairScore);
            fflush(stdout);
        }
    }

    printf("\n=== resultado agregado (score medio, positivo = melhor) ===\n");
    for (int i = 0; i < n; i++) {
        double avg = gamesPlayed[i] ? scoreSum[i] / gamesPlayed[i] : 0.0;
        printf("  minDepth=%d: score_medio=%+.3f (%d confrontos)\n", candidates[i], avg, gamesPlayed[i]);
    }
    printf("\nEscolha manual: use o setPolicyOrderingMinDepth() correspondente ao\n"
           "melhor score_medio acima (ou o melhor trade-off com nos/s, nao medido\n"
           "aqui -- ver benchNegamaxNNUE/run_arena.py para isso).\n");
}

// --- SPSA continuo -------------------------------------------------------
void runSpsa(std::vector<TunableParam>& params, int totalIterations, unsigned seed,
             double timeBudgetSec, const char* ckptPath) {
    std::vector<int> activeIdx;
    for (size_t i = 0; i < params.size(); i++) if (params[i].active) activeIdx.push_back((int)i);
    int nActive = (int)activeIdx.size();
    if (nActive == 0) {
        printf("nenhum parametro ativo (--tune-contempt/--tune-policy-scale/--tune-cat-scale) -- nada a fazer.\n");
        return;
    }

    std::mt19937 rng(seed);
    std::vector<double> theta0(nActive), theta(nActive), thetaAvg(nActive);
    for (int k = 0; k < nActive; k++) theta0[k] = theta[k] = thetaAvg[k] = params[activeIdx[k]].value;
    int avgCount = 0;
    int kStart = 0;

    // retoma de um checkpoint em disco, se existir (rodar em lotes --
    // processos em background não sobrevivem entre chamadas da ferramenta
    // de shell neste ambiente, então persistimos em arquivo em vez disso).
    FILE* ckptIn = fopen(ckptPath, "r");
    if (ckptIn) {
        int nSaved = 0;
        fscanf(ckptIn, "%d %d %d", &kStart, &avgCount, &nSaved);
        if (nSaved == nActive) {
            for (int i = 0; i < nActive; i++) fscanf(ckptIn, "%lf", &theta[i]);
            for (int i = 0; i < nActive; i++) fscanf(ckptIn, "%lf", &thetaAvg[i]);
            std::stringstream rngState;
            char buf[8192]; size_t n;
            while ((n = fread(buf, 1, sizeof(buf), ckptIn)) > 0) rngState.write(buf, n);
            rngState >> rng;
            printf("retomando checkpoint: iteracao %d, avgCount=%d\n", kStart, avgCount);
        } else {
            printf("checkpoint incompativel com o conjunto de parametros ativo (esperava %d, achou %d) -- ignorando e comecando do zero.\n", nActive, nSaved);
            kStart = 0; avgCount = 0;
        }
        fclose(ckptIn);
    }

    // bounds: usa os bounds absolutos configurados por parâmetro (ver
    // main()) -- ao contrário da versão anterior (0.1x/4x do valor
    // inicial), que quebrava para parâmetros que podem ser negativos
    // (contempt) porque multiplicar por um fator > 1 INVERTE qual bound é
    // o "lo" quando o valor inicial é negativo.
    std::vector<double> lo(nActive), hi(nActive);
    for (int i = 0; i < nActive; i++) { lo[i] = params[activeIdx[i]].lo; hi[i] = params[activeIdx[i]].hi; }

    // Spall (1998), valores-padrão recomendados de alpha/gamma.
    const double alpha = 0.602, gamma = 0.101;
    const double bigA = 0.1 * totalIterations;
    // c0/a0 por parâmetro, proporcional à AMPLITUDE do bound (hi-lo), não
    // ao valor inicial -- generaliza melhor para parâmetros que podem
    // começar perto de zero (theta0 perto de 0 faria c0/a0 colapsarem a 0
    // com a fórmula antiga, "travando" o parâmetro).
    std::vector<double> c0(nActive), a0(nActive);
    for (int i = 0; i < nActive; i++) {
        double range = hi[i] - lo[i];
        c0[i] = 0.075 * range;
        a0[i] = 0.04 * range;
    }

    if (kStart == 0) {
        printf("SPSA -- %d iteracoes totais, profundidade=%d, %dms/lance, %d plies aleatorios de abertura, NNUE=%s policyOrdering=%s(minDepth=%d)\n",
               totalIterations, g_searchDepth, g_timeBudgetMs, g_openingRandomPlies,
               g_useNNUE ? "on" : "off", g_policyOrdering ? "on" : "off", g_policyOrderMinDepth);
        printf("theta0: ");
        for (int i = 0; i < nActive; i++) printf("%s=%.3f ", params[activeIdx[i]].name, theta0[i]);
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
        std::vector<int> delta(nActive);
        for (int i = 0; i < nActive; i++) delta[i] = coin(rng) ? 1 : -1;

        std::vector<TunableParam> paramsPlus = params, paramsMinus = params;
        for (int i = 0; i < nActive; i++) {
            double ck = c0[i] * ck_scale;
            double vp = std::clamp(theta[i] + ck * delta[i], lo[i], hi[i]);
            double vm = std::clamp(theta[i] - ck * delta[i], lo[i], hi[i]);
            paramsPlus[activeIdx[i]].value = vp;
            paramsMinus[activeIdx[i]].value = vm;
        }

        double score = antitheticMatch(paramsPlus, paramsMinus, rng);

        for (int i = 0; i < nActive; i++) {
            double ck = c0[i] * ck_scale;
            double ak = a0[i] * ak_scale;
            double ghat = score * delta[i] / (2.0 * ck);
            theta[i] = std::clamp(theta[i] + ak * ghat, lo[i], hi[i]);
        }

        if (k >= avgStart) {
            avgCount++;
            for (int i = 0; i < nActive; i++)
                thetaAvg[i] += (theta[i] - thetaAvg[i]) / avgCount;
        }

        auto now = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(now - t0).count();
        printf("iter %3d/%d score=%+.2f elapsed=%.0fs theta: ", k + 1, totalIterations, score, elapsed);
        for (int i = 0; i < nActive; i++) printf("%s=%.3f ", params[activeIdx[i]].name, theta[i]);
        printf("\n");
        fflush(stdout);
    }

    // salva checkpoint (se ainda não terminou tudo) ou resultado final
    FILE* ckptOut = fopen(ckptPath, "w");
    if (ckptOut) {
        fprintf(ckptOut, "%d %d %d\n", k, avgCount, nActive);
        for (int i = 0; i < nActive; i++) fprintf(ckptOut, "%.10f ", theta[i]);
        fprintf(ckptOut, "\n");
        for (int i = 0; i < nActive; i++) fprintf(ckptOut, "%.10f ", thetaAvg[i]);
        fprintf(ckptOut, "\n");
        std::stringstream rngState;
        rngState << rng;
        fprintf(ckptOut, "%s", rngState.str().c_str());
        fclose(ckptOut);
    }

    if (k >= totalIterations) {
        printf("\n=== CONCLUIDO -- resultado (media de Polyak da 2a metade das iteracoes) ===\n");
        FILE* f = fopen("spsa_result.txt", "w");
        for (int i = 0; i < nActive; i++) {
            printf("%s: %.4f (era %.4f)\n", params[activeIdx[i]].name, thetaAvg[i], theta0[i]);
            if (f) fprintf(f, "%s %.6f\n", params[activeIdx[i]].name, thetaAvg[i]);
        }
        if (f) fclose(f);
    } else {
        printf("\n=== lote parcial salvo em checkpoint (%d/%d iteracoes feitas) -- rode de novo pra continuar ===\n", k, totalIterations);
    }
}

// =============================================================================
// main -- toda a config vem de CLI (o driver Python teste/run_spsa.py
// monta os flags a partir de um bloco de variaveis no topo do arquivo,
// para nao precisar editar/recompilar C++ a cada mudanca de config).
// =============================================================================
void printUsage(const char* prog) {
    printf("Uso: %s [opcoes]\n", prog);
    printf("  --mode spsa|sweep-mindepth       (default: spsa)\n");
    printf("  --iterations N                   (default: 40, modo spsa)\n");
    printf("  --seed N                         (default: 20260719)\n");
    printf("  --time-budget-sec S              (default: sem limite)\n");
    printf("  --checkpoint PATH                (default: spsa_checkpoint.txt)\n");
    printf("  --depth N                        (default: 4)\n");
    printf("  --time-ms N                      (default: 120)\n");
    printf("  --max-plies N                    (default: 70)\n");
    printf("  --opening-plies N                (default: 2)\n");
    printf("  --nnue-weights PATH              (default: data/nnue/nnue_weights_int8.bin)\n");
    printf("  --heuristic                      desliga NNUE (eval heuristica evalSimple)\n");
    printf("  --no-policy-order                desliga a ordenacao assistida por politica\n");
    printf("  --policy-order-min-depth N       (default: 3)\n");
    printf("  --contempt V                     valor inicial (default: search.hpp, -30)\n");
    printf("  --policy-scale V                 valor inicial (default: search.hpp, 400)\n");
    printf("  --cat-scale V                    valor inicial (default: search.hpp, 2)\n");
    printf("  --contempt-bounds LO,HI          (default: -150,0)\n");
    printf("  --policy-scale-bounds LO,HI      (default: 0,2000)\n");
    printf("  --cat-scale-bounds LO,HI         (default: 0,20)\n");
    printf("  --no-tune-contempt / --no-tune-policy-scale / --no-tune-cat-scale\n");
    printf("                                   remove o parametro do SPSA (fica fixo no valor inicial)\n");
    printf("  --mindepth-candidates D1,D2,...  (default: 1,2,3,4,5 -- modo sweep-mindepth)\n");
    printf("  --mindepth-games N               partidas antiteticas por par (default: 20, modo sweep-mindepth)\n");
}

std::vector<int> parseIntList(const std::string& s) {
    std::vector<int> out;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) if (!item.empty()) out.push_back(std::atoi(item.c_str()));
    return out;
}

bool parseBounds(const std::string& s, double& lo, double& hi) {
    auto comma = s.find(',');
    if (comma == std::string::npos) return false;
    lo = std::atof(s.substr(0, comma).c_str());
    hi = std::atof(s.substr(comma + 1).c_str());
    return true;
}

int main(int argc, char** argv) {
    std::string mode = "spsa";
    int totalIterations = 40;
    unsigned seed = 20260719u;
    double timeBudgetSec = 1e18;
    std::string ckptPath = "spsa_checkpoint.txt";
    std::string nnueWeightsPath = defaultNnueWeightsPath();

    std::vector<TunableParam> params(NPARAM_MAX);
    params[P_CONTEMPT]      = {"contempt",         (double)CONTEMPT,           -150.0, 0.0,    true};
    params[P_POLICY_SCALE]  = {"policyOrderScale", (double)POLICY_ORDER_SCALE,    0.0, 2000.0, true};
    params[P_CAT_SCALE]     = {"catScoreScale",     2.0,                          0.0,   20.0, true};

    std::vector<int> mindepthCandidates = {1, 2, 3, 4, 5};
    int mindepthGames = 20;

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() -> std::string { return (i + 1 < argc) ? argv[++i] : ""; };
        if (a == "--help" || a == "-h") { printUsage(argv[0]); return 0; }
        else if (a == "--mode") mode = next();
        else if (a == "--iterations") totalIterations = std::atoi(next().c_str());
        else if (a == "--seed") seed = (unsigned)std::atoi(next().c_str());
        else if (a == "--time-budget-sec") timeBudgetSec = std::atof(next().c_str());
        else if (a == "--checkpoint") ckptPath = next();
        else if (a == "--depth") g_searchDepth = std::atoi(next().c_str());
        else if (a == "--time-ms") g_timeBudgetMs = std::atoi(next().c_str());
        else if (a == "--max-plies") g_maxPlies = std::atoi(next().c_str());
        else if (a == "--opening-plies") g_openingRandomPlies = std::atoi(next().c_str());
        else if (a == "--nnue-weights") nnueWeightsPath = next();
        else if (a == "--heuristic") g_useNNUE = false;
        else if (a == "--no-policy-order") g_policyOrdering = false;
        else if (a == "--policy-order-min-depth") g_policyOrderMinDepth = std::atoi(next().c_str());
        else if (a == "--contempt") params[P_CONTEMPT].value = std::atof(next().c_str());
        else if (a == "--policy-scale") params[P_POLICY_SCALE].value = std::atof(next().c_str());
        else if (a == "--cat-scale") params[P_CAT_SCALE].value = std::atof(next().c_str());
        else if (a == "--contempt-bounds") parseBounds(next(), params[P_CONTEMPT].lo, params[P_CONTEMPT].hi);
        else if (a == "--policy-scale-bounds") parseBounds(next(), params[P_POLICY_SCALE].lo, params[P_POLICY_SCALE].hi);
        else if (a == "--cat-scale-bounds") parseBounds(next(), params[P_CAT_SCALE].lo, params[P_CAT_SCALE].hi);
        else if (a == "--no-tune-contempt") params[P_CONTEMPT].active = false;
        else if (a == "--no-tune-policy-scale") params[P_POLICY_SCALE].active = false;
        else if (a == "--no-tune-cat-scale") params[P_CAT_SCALE].active = false;
        else if (a == "--mindepth-candidates") mindepthCandidates = parseIntList(next());
        else if (a == "--mindepth-games") mindepthGames = std::atoi(next().c_str());
        else { fprintf(stderr, "flag desconhecida: %s (use --help)\n", a.c_str()); return 1; }
    }

    if (g_useNNUE) {
        if (!loadWeightsQuant(nnueWeightsPath)) {
            fprintf(stderr, "erro: nao consegui carregar pesos NNUE de '%s' -- passe --nnue-weights ou --heuristic\n", nnueWeightsPath.c_str());
            return 1;
        }
        printf("NNUE carregada de %s\n", nnueWeightsPath.c_str());
    } else {
        printf("modo heuristico (evalSimple) -- catScoreScale/policyOrderScale nao tem efeito\n");
    }

    if (mode == "spsa") {
        runSpsa(params, totalIterations, seed, timeBudgetSec, ckptPath.c_str());
    } else if (mode == "sweep-mindepth") {
        runSweepMinDepth(mindepthCandidates, mindepthGames, params, seed);
    } else {
        fprintf(stderr, "--mode desconhecido: '%s' (use spsa ou sweep-mindepth)\n", mode.c_str());
        return 1;
    }
    return 0;
}
