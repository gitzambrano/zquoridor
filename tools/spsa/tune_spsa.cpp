// tune_spsa.cpp -- tuner evolutivo/genético robusto para os parâmetros de busca.
//
// O nome do arquivo é mantido por compatibilidade com os scripts existentes,
// mas o algoritmo agora é um Genetic Algorithm (GA), não SPSA.
//
// Motivo: a função objetivo de Quoridor é discreta e ruidosa (resultado de
// partidas), portanto estimar gradiente local é particularmente frágil.
// O GA mantém uma população, usa crossover, mutação, elitismo e imigrantes
// aleatórios, permitindo explorar regiões distantes e escapar de ótimos locais.
//
// Parâmetros tunados:
//   contempt, policyOrderScale, catScoreScale,
//   lmrMinDepth, lmrMinMoveIndex, lmrDivisor,
//   catHotCm, catColdCm, wallBfsOrderMaxPly, qsCriticalBfsDelta,
//   policyOrderingMinDepth, policyOrderingEnabled, quiescenceEnabled,
//   lmrPvsEnabled.
//
// O registro de parâmetros é genérico: para adicionar um parâmetro basta
// adicionar uma entrada à tabela PARAM_DEFS e o respectivo setter no engine.

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <random>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <thread>
#include <atomic>
#include <chrono>
#include <numeric>
#include <limits>
#include "rules.hpp"
#include "search.hpp"
#include "nnue.hpp"

using namespace qr;

enum ParamId {
    P_CONTEMPT,
    P_POLICY_SCALE,
    P_CAT_SCALE,
    P_LMR_MIN_DEPTH,
    P_LMR_MIN_MOVE_INDEX,
    P_LMR_DIVISOR,
    P_CAT_HOT,
    P_CAT_COLD,
    P_WALL_BFS_MAX_PLY,
    P_QS_CRITICAL_BFS_DELTA,
    P_POLICY_MIN_DEPTH,
    P_POLICY_ENABLED,
    P_QUIESCENCE,
    P_LMR_PVS,
    NPARAM
};

enum class ParamKind { Integer, Real, Boolean };

struct ParamDef {
    const char* name;
    ParamKind kind;
    double lo, hi, init;
    double mutationSigma; // fração da faixa
};

static const ParamDef PARAM_DEFS[NPARAM] = {
    {"contempt",              ParamKind::Integer, -150, 0,     CONTEMPT,          0.12},
    {"policyOrderScale",      ParamKind::Integer,    0, 2000,  POLICY_ORDER_SCALE, 0.15},
    {"catScoreScale",         ParamKind::Integer,    0, 20,    2,                 0.18},
    {"lmrMinDepth",           ParamKind::Integer,    1, 8,     LMR_MIN_DEPTH,     0.18},
    {"lmrMinMoveIndex",       ParamKind::Integer,    1, 8,     LMR_MIN_MOVE_INDEX, 0.18},
    {"lmrDivisor",            ParamKind::Real,       1.0, 4.5, LMR_DIVISOR,       0.12},
    {"catHotCm",              ParamKind::Integer,    50, 220,  CAT_HOT_CM,        0.12},
    {"catColdCm",             ParamKind::Integer,    0, 100,   CAT_COLD_CM,       0.15},
    {"wallBfsOrderMaxPly",    ParamKind::Integer,    0, 5,     WALL_BFS_ORDER_MAX_PLY, 0.20},
    {"qsCriticalBfsDelta",    ParamKind::Integer,    1, 5,     QS_CRITICAL_BFS_DELTA, 0.20},
    {"policyOrderingMinDepth",ParamKind::Integer,    0, 6,     3,                  0.20},
    {"policyOrderingEnabled", ParamKind::Boolean,     0, 1,     1,                  0.30},
    {"quiescenceEnabled",     ParamKind::Boolean,     0, 1,     1,                  0.30},
    {"lmrPvsEnabled",         ParamKind::Boolean,     0, 1,     1,                  0.30},
};

struct Individual {
    std::array<double, NPARAM> x{};
    double fitness = -1e9;
    double games = 0;
};

static double clampValue(double x, int i) {
    return std::max(PARAM_DEFS[i].lo, std::min(PARAM_DEFS[i].hi, x));
}

static void applyParams(Negamax& e, const Individual& ind) {
    e.setContempt((int)std::lround(ind.x[P_CONTEMPT]));
    e.setPolicyOrderScale((long long)std::lround(ind.x[P_POLICY_SCALE]));
    e.setCatScoreScale((long long)std::lround(ind.x[P_CAT_SCALE]));
    e.setLmrMinDepth((int)std::lround(ind.x[P_LMR_MIN_DEPTH]));
    e.setLmrMinMoveIndex((int)std::lround(ind.x[P_LMR_MIN_MOVE_INDEX]));
    e.setLmrDivisor(ind.x[P_LMR_DIVISOR]);
    e.setCatHotCm((int)std::lround(ind.x[P_CAT_HOT]));
    e.setCatColdCm((int)std::lround(ind.x[P_CAT_COLD]));
    e.setWallBfsOrderMaxPly((int)std::lround(ind.x[P_WALL_BFS_MAX_PLY]));
    e.setQsCriticalBfsDelta((int)std::lround(ind.x[P_QS_CRITICAL_BFS_DELTA]));
    e.setPolicyOrderingEnabled(ind.x[P_POLICY_ENABLED] >= 0.5);
    e.setPolicyOrderingMinDepth((int)std::lround(ind.x[P_POLICY_MIN_DEPTH]));
    e.setQuiescenceEnabled(ind.x[P_QUIESCENCE] >= 0.5);
    e.setLmrPvsEnabled(ind.x[P_LMR_PVS] >= 0.5);
}

static void sanitize(Individual& a) {
    for (int i = 0; i < NPARAM; ++i) {
        a.x[i] = clampValue(a.x[i], i);
        if (PARAM_DEFS[i].kind == ParamKind::Integer)
            a.x[i] = std::lround(a.x[i]);
        else if (PARAM_DEFS[i].kind == ParamKind::Boolean)
            a.x[i] = a.x[i] >= 0.5 ? 1.0 : 0.0;
    }
    // Não permita que o limiar frio fique acima do limiar quente.
    if (a.x[P_CAT_COLD] >= a.x[P_CAT_HOT])
        a.x[P_CAT_COLD] = std::max(PARAM_DEFS[P_CAT_COLD].lo, a.x[P_CAT_HOT] - 1);

    // Canonicalização de genes condicionais (epistáticos): alguns genes só
    // afetam o fenótipo quando um "gate" booleano correspondente está
    // ligado -- ver os pontos exatos em search.hpp onde cada um é lido:
    //   - policyOrderScale/policyOrderingMinDepth: só lidos dentro do `if
    //     (policyOrderingEnabled && ...)` que monta o ponteiro de policy;
    //     com o gate off, esse ponteiro nunca é setado e o valor nunca é
    //     usado (orderMoves só soma policyOrderScale quando `policy`
    //     não é nulo).
    //   - qsCriticalBfsDelta: só lido dentro de quiescence(), que só é
    //     chamada quando quiescenceEnabled é true.
    //   - lmrMinDepth/lmrMinMoveIndex/lmrDivisor/catHotCm/catColdCm: só
    //     lidos dentro do branch de tryMove() que só executa quando
    //     lmrPvsEnabled é true (com o gate off, cai sempre no ramo de
    //     janela cheia/profundidade cheia, sem calcular redução alguma).
    // Sem canonicalizar, dois indivíduos com o mesmo gate OFF mas valores
    // diferentes nesses genes jogam EXATAMENTE igual, mas o GA os trata
    // como genótipos distintos -- crossover/mutação gastam diversidade da
    // população distinguindo dimensões fantasma. Fixamos esses genes no
    // valor de init (o mesmo do baseline) sempre que o gate estiver OFF.
    if (a.x[P_POLICY_ENABLED] < 0.5) {
        a.x[P_POLICY_SCALE] = PARAM_DEFS[P_POLICY_SCALE].init;
        a.x[P_POLICY_MIN_DEPTH] = PARAM_DEFS[P_POLICY_MIN_DEPTH].init;
    }
    if (a.x[P_QUIESCENCE] < 0.5) {
        a.x[P_QS_CRITICAL_BFS_DELTA] = PARAM_DEFS[P_QS_CRITICAL_BFS_DELTA].init;
    }
    if (a.x[P_LMR_PVS] < 0.5) {
        a.x[P_LMR_MIN_DEPTH] = PARAM_DEFS[P_LMR_MIN_DEPTH].init;
        a.x[P_LMR_MIN_MOVE_INDEX] = PARAM_DEFS[P_LMR_MIN_MOVE_INDEX].init;
        a.x[P_LMR_DIVISOR] = PARAM_DEFS[P_LMR_DIVISOR].init;
        a.x[P_CAT_HOT] = PARAM_DEFS[P_CAT_HOT].init;
        a.x[P_CAT_COLD] = PARAM_DEFS[P_CAT_COLD].init;
    }
}

static Individual defaultIndividual() {
    Individual a;
    for (int i = 0; i < NPARAM; ++i) a.x[i] = PARAM_DEFS[i].init;
    sanitize(a);
    return a;
}

static Individual randomIndividual(std::mt19937& rng) {
    Individual a;
    for (int i = 0; i < NPARAM; ++i) {
        if (PARAM_DEFS[i].kind == ParamKind::Boolean)
            a.x[i] = (rng() & 1u) ? 1.0 : 0.0;
        else {
            std::uniform_real_distribution<double> d(PARAM_DEFS[i].lo, PARAM_DEFS[i].hi);
            a.x[i] = d(rng);
        }
    }
    sanitize(a);
    return a;
}

static Individual mutate(const Individual& src, std::mt19937& rng, double rate) {
    Individual a = src;
    std::uniform_real_distribution<double> u(0.0, 1.0);
    for (int i = 0; i < NPARAM; ++i) {
        if (u(rng) < rate) {
            if (PARAM_DEFS[i].kind == ParamKind::Boolean)
                a.x[i] = 1.0 - a.x[i];
            else {
                double range = PARAM_DEFS[i].hi - PARAM_DEFS[i].lo;
                std::normal_distribution<double> n(0.0, PARAM_DEFS[i].mutationSigma * range);
                a.x[i] += n(rng);
            }
        }
    }
    // Garante exploração: pelo menos um gene sofre mutação.
    if (rate > 0 && u(rng) < 0.35) {
        int i = (int)(rng() % NPARAM);
        double range = PARAM_DEFS[i].hi - PARAM_DEFS[i].lo;
        if (PARAM_DEFS[i].kind == ParamKind::Boolean)
            a.x[i] = 1.0 - a.x[i];
        else {
            std::normal_distribution<double> n(0.0, PARAM_DEFS[i].mutationSigma * range);
            a.x[i] += n(rng);
        }
    }
    sanitize(a);
    a.fitness = -1e9;
    return a;
}

static Individual crossover(const Individual& a, const Individual& b, std::mt19937& rng) {
    Individual c;
    std::uniform_real_distribution<double> u(0.0, 1.0);
    for (int i = 0; i < NPARAM; ++i) {
        // BLX-alpha: não apenas escolhe A/B; permite explorar entre e ligeiramente
        // além deles, aumentando diversidade.
        double lo = std::min(a.x[i], b.x[i]);
        double hi = std::max(a.x[i], b.x[i]);
        double span = hi - lo;
        double alpha = 0.20;
        std::uniform_real_distribution<double> d(lo - alpha*span, hi + alpha*span);
        c.x[i] = (span > 0.0) ? d(rng) : (u(rng) < 0.5 ? a.x[i] : b.x[i]);
    }
    sanitize(c);
    return c;
}

// -----------------------------------------------------------------------------
// Self-play
// -----------------------------------------------------------------------------
static int g_depth = 12;
static int g_timeMs = 80;
static int g_maxPlies = 140;
static int g_openingPlies = 4;
static bool g_useNNUE = true;
static bool g_policyOrder = true;

static void configureEngine(Negamax& e) {
    if (g_useNNUE) e.setEvalMode(Negamax::EvalMode::NNUE);
}

// Override explícito de --no-policy-order, aplicado DEPOIS de applyParams()
// para cada engine de uma partida. Sem isso, applyParams() (que roda no
// mesmo lugar pra setar todos os genes do indivíduo, incluindo
// policyOrderingEnabled) sobrescrevia silenciosamente a flag externa --
// --no-policy-order na prática não tinha efeito nenhum. policyOrderingMinDepth
// não tem um override equivalente porque não faz sentido conceitual: ele já
// é inteiramente controlado pelo gene do GA (policyOrderingMinDepth), não
// existe uma "flag externa de profundidade mínima" que não conflite com
// isso -- por isso removemos --policy-order-min-depth em vez de tentar
// consertar o override dela.
static void applyExternalOverrides(Negamax& e) {
    if (!g_policyOrder) e.setPolicyOrderingEnabled(false);
}

static int playGame(Negamax& a, Negamax& b, std::mt19937& rng) {
    State s = initialState();

    for (int i = 0; i < g_openingPlies; ++i) {
        if (winner(s) != -1) return winner(s);
        auto moves = legalMoves(s);
        if (moves.empty()) return -1;
        s = applyMove(s, moves[rng() % moves.size()]);
    }

    for (int ply = 0; ply < g_maxPlies; ++ply) {
        int w = winner(s);
        if (w != -1) return w;
        Negamax& e = (s.turn == 0) ? a : b;
        SearchStats st;
        Move m = e.chooseMove(s, g_depth, g_timeMs, st);
        s = applyMove(s, m);
    }
    return -1;
}

// Score do candidato A contra B, sempre com as duas cores.
// Empates/partidas truncadas contam 0,5, evitando transformar um empate em
// vitória/derrota artificial.
static double antithetic(const Individual& A, const Individual& B, unsigned seed) {
    std::mt19937 rng(seed);
    Negamax a1, b1;
    configureEngine(a1); configureEngine(b1);
    applyParams(a1, A); applyParams(b1, B);
    applyExternalOverrides(a1); applyExternalOverrides(b1);

    int r1 = playGame(a1, b1, rng);
    double score = (r1 == 0) ? 1.0 : (r1 == 1 ? 0.0 : 0.5);

    Negamax a2, b2;
    configureEngine(a2); configureEngine(b2);
    applyParams(a2, A); applyParams(b2, B);
    applyExternalOverrides(a2); applyExternalOverrides(b2);
    int r2 = playGame(b2, a2, rng); // B é jogador 0, A jogador 1
    double score2 = (r2 == 1) ? 1.0 : (r2 == 0 ? 0.0 : 0.5);

    return 0.5 * (score + score2);
}

static double evaluatePair(const Individual& A, const Individual& B,
                           unsigned seed, int repeats) {
    repeats = std::max(1, repeats);
    std::vector<unsigned> seeds(repeats);
    std::mt19937 master(seed);
    for (int i = 0; i < repeats; ++i) seeds[i] = master();

    // Sequencial: o paralelismo do GA acontece no nível de população (ver
    // main()), não aqui dentro. Rodar um pool de threads aninhado por par
    // causaria oversubscription (threads^2) sem ganho real de throughput.
    double sum = 0.0;
    for (int i = 0; i < repeats; ++i) sum += antithetic(A, B, seeds[i]);
    return sum / repeats;
}

// Cada indivíduo enfrenta:
//   1) baseline atual;
//   2) elite;
//   3) um adversário escolhido deterministicamente da população.
// Isso reduz a chance de o GA aprender apenas a explorar um único baseline.
//
// `refPop` é uma referência CONGELADA (a população já avaliada e ordenada
// da geração anterior -- ou, na primeira geração, a população inicial ainda
// não avaliada). Ela nunca é mutada durante a avaliação da geração atual.
// Isso é essencial: se em vez disso escaneássemos a população que está
// sendo avaliada agora (como na versão anterior), o "elite" escolhido para
// um indivíduo dependeria de quais outros indivíduos já tinham sido
// avaliados antes dele nesta mesma geração -- um vazamento de ordem que
// tornava o fitness não-determinístico em relação à ordem de avaliação
// (mesmo sendo single-threaded, e pior ainda ao paralelizar). Com `refPop`
// fixo, todo indivíduo da geração é avaliado contra exatamente a mesma
// referência, então o resultado independe de ordem e é seguro paralelizar.
static double evaluateIndividual(const Individual& ind,
                                 const std::vector<Individual>& refPop,
                                 int idx, unsigned seed, int repeats) {
    Individual base = defaultIndividual();
    double total = 0.0;
    int n = 0;

    total += evaluatePair(ind, base, seed ^ 0xA341316Cu, repeats); n++;

    int elite = 0;
    for (int i = 1; i < (int)refPop.size(); ++i)
        if (refPop[i].fitness > refPop[elite].fitness) elite = i;
    if (elite != idx) {
        total += evaluatePair(ind, refPop[elite], seed ^ 0xC8013EA4u, repeats); n++;
    }

    // Adversários adicionais. A seleção muda por indivíduo e por geração.
    int p1 = (idx * 7 + 3) % (int)refPop.size();
    int p2 = (idx * 11 + 5) % (int)refPop.size();
    if (p1 != idx) { total += evaluatePair(ind, refPop[p1], seed ^ 0xAD90777Du, repeats); n++; }
    if (p2 != idx && p2 != p1) { total += evaluatePair(ind, refPop[p2], seed ^ 0x7E95761Eu, repeats); n++; }

    return n ? total / n : 0.5;
}

static bool timeExpired(std::chrono::steady_clock::time_point start, double limitSec) {
    if (limitSec <= 0 || limitSec >= 1e17) return false;
    double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    return sec >= limitSec;
}

static void printIndividual(const Individual& a, const char* prefix = "") {
    std::printf("%sfitness=%.4f", prefix, a.fitness);
    for (int i = 0; i < NPARAM; ++i)
        std::printf(" %s=%.4g", PARAM_DEFS[i].name, a.x[i]);
    std::printf("\n");
}

static void saveCheckpoint(const std::string& path, int generation, const Individual& best) {
    std::ofstream f(path);
    if (!f) return;
    f << generation << "\n";
    for (int i = 0; i < NPARAM; ++i) f << PARAM_DEFS[i].name << " " << best.x[i] << "\n";
}

static bool loadCheckpoint(const std::string& path, int& generation, Individual& best) {
    std::ifstream f(path);
    if (!f) return false;
    if (!(f >> generation)) return false;
    std::string name;
    for (int i = 0; i < NPARAM; ++i) {
        double v;
        if (!(f >> name >> v)) return false;
        best.x[i] = v;
    }
    sanitize(best);
    best.fitness = -1e9;
    return true;
}

static void writeHistoryHeader(const std::string& path) {
    std::ifstream f(path);
    if (f.good()) return;
    std::ofstream out(path);
    out << "generation,bestFitness,meanFitness,diversity";
    for (int i = 0; i < NPARAM; ++i) out << "," << PARAM_DEFS[i].name;
    out << "\n";
}

static void writeHistory(const std::string& path, int gen,
                         const std::vector<Individual>& pop, const Individual& best) {
    double mean = 0.0;
    for (const auto& x : pop) mean += x.fitness;
    mean /= std::max<size_t>(1, pop.size());

    double diversity = 0.0;
    for (int i = 0; i < NPARAM; ++i) {
        double m = 0.0;
        for (const auto& x : pop) m += x.x[i];
        m /= pop.size();
        double v = 0.0;
        for (const auto& x : pop) { double d = x.x[i] - m; v += d*d; }
        diversity += std::sqrt(v / pop.size()) /
                     std::max(1.0, PARAM_DEFS[i].hi - PARAM_DEFS[i].lo);
    }
    diversity /= NPARAM;

    std::ofstream out(path, std::ios::app);
    out << gen << "," << best.fitness << "," << mean << "," << diversity;
    for (int i = 0; i < NPARAM; ++i) out << "," << best.x[i];
    out << "\n";
}

static void printUsage(const char* p) {
    std::printf("GA tuner (arquivo mantido como tune_spsa.cpp por compatibilidade)\n");
    std::printf("Uso: %s [opcoes]\n", p);
    std::printf("  --generations N        (default 30)\n");
    std::printf("  --population N         (default 24)\n");
    std::printf("  --games-per-match N    partidas antiteticas por confronto (default 3)\n");
    std::printf("  --threads N            threads paralelas do GA -- avalia N individuos\n");
    std::printf("                         da populacao simultaneamente, cada um em sua\n");
    std::printf("                         propria thread com engines independentes\n");
    std::printf("                         (default 14)\n");
    std::printf("  --elite-fraction X     (default 0.20)\n");
    std::printf("  --mutation-rate X      (default 0.25)\n");
    std::printf("  --immigrants X         fração de indivíduos aleatórios (default 0.10)\n");
    std::printf("  --seed N               (default 20260809)\n");
    std::printf("  --depth N              (default 12)\n");
    std::printf("  --time-ms N            (default 80)\n");
    std::printf("  --max-plies N          (default 140)\n");
    std::printf("  --opening-plies N      (default 4)\n");
    std::printf("  --nnue-weights PATH    (default data/nnue/nnue_weights_int8.bin)\n");
    std::printf("  --heuristic            usa evalSimple em vez de NNUE\n");
    std::printf("  --no-policy-order      override explicito: forca policy ordering\n");
    std::printf("                         desligado em TODA a rodada, por cima do gene\n");
    std::printf("                         policyOrderingEnabled de cada individuo\n");
    std::printf("  --checkpoint PATH      (default ga_checkpoint.txt)\n");
    std::printf("  --history PATH         (default ga_history.csv)\n");
    std::printf("  --result PATH          (default ga_result.txt)\n");
    std::printf("\nParâmetros explorados:\n");
    for (int i = 0; i < NPARAM; ++i)
        std::printf("  %-24s [%g, %g] init=%g\n", PARAM_DEFS[i].name,
                    PARAM_DEFS[i].lo, PARAM_DEFS[i].hi, PARAM_DEFS[i].init);
}

int main(int argc, char** argv) {
    int generations = 30;
    int populationSize = 24;
    int gamesPerMatch = 3;
    int threads = 14;
    double eliteFraction = 0.20;
    double mutationRate = 0.25;
    double immigrantFraction = 0.10;
    unsigned seed = 20260809u;
    double timeLimitSec = 1e18;
    std::string checkpoint = "ga_checkpoint.txt";
    std::string history = "ga_history.csv";
    std::string result = "ga_result.txt";
    std::string nnuePath = defaultNnueWeightsPath();

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string { return i + 1 < argc ? argv[++i] : ""; };
        if (a == "--help" || a == "-h") { printUsage(argv[0]); return 0; }
        else if (a == "--generations") generations = std::atoi(next().c_str());
        else if (a == "--population") populationSize = std::atoi(next().c_str());
        else if (a == "--games-per-match") gamesPerMatch = std::atoi(next().c_str());
        else if (a == "--threads") threads = std::atoi(next().c_str());
        else if (a == "--elite-fraction") eliteFraction = std::atof(next().c_str());
        else if (a == "--mutation-rate") mutationRate = std::atof(next().c_str());
        else if (a == "--immigrants") immigrantFraction = std::atof(next().c_str());
        else if (a == "--seed") seed = (unsigned)std::strtoul(next().c_str(), nullptr, 10);
        else if (a == "--time-budget-sec") timeLimitSec = std::atof(next().c_str());
        else if (a == "--checkpoint") checkpoint = next();
        else if (a == "--history") history = next();
        else if (a == "--result") result = next();
        else if (a == "--depth") g_depth = std::atoi(next().c_str());
        else if (a == "--time-ms") g_timeMs = std::atoi(next().c_str());
        else if (a == "--max-plies") g_maxPlies = std::atoi(next().c_str());
        else if (a == "--opening-plies") g_openingPlies = std::atoi(next().c_str());
        else if (a == "--nnue-weights") nnuePath = next();
        else if (a == "--heuristic") g_useNNUE = false;
        else if (a == "--no-policy-order") g_policyOrder = false;
        else { std::fprintf(stderr, "flag desconhecida: %s\n", a.c_str()); return 1; }
    }

    populationSize = std::max(8, populationSize);
    gamesPerMatch = std::max(1, gamesPerMatch);
    threads = std::max(1, threads);
    eliteFraction = std::clamp(eliteFraction, 0.05, 0.5);
    mutationRate = std::clamp(mutationRate, 0.01, 1.0);
    immigrantFraction = std::clamp(immigrantFraction, 0.0, 0.5);

    if (g_useNNUE) {
        if (!loadWeightsQuant(nnuePath)) {
            std::fprintf(stderr, "ERRO: nao consegui carregar NNUE: %s\n", nnuePath.c_str());
            return 1;
        }
        std::printf("NNUE carregada: %s\n", nnuePath.c_str());
    } else {
        std::printf("Modo heuristico.\n");
    }

    std::mt19937 rng(seed);
    std::vector<Individual> pop;
    pop.reserve(populationSize);
    pop.push_back(defaultIndividual());
    while ((int)pop.size() < populationSize) pop.push_back(randomIndividual(rng));

    Individual checkpointBest;
    int startGeneration = 0;
    if (loadCheckpoint(checkpoint, startGeneration, checkpointBest)) {
        std::printf("Checkpoint encontrado: geração %d; campeão será preservado.\n", startGeneration);
        pop[0] = checkpointBest;
    }

    // Referência congelada usada por evaluateIndividual() para escolher
    // elite/oponentes. Começa igual à população inicial (na 1a geração
    // ninguém tem fitness real ainda -- tanto faz, os adversários são só
    // pra dar diversidade) e é substituída pela população já avaliada e
    // ordenada ao final de cada geração, antes do crossover/mutação. Nunca
    // é mutada durante a avaliação da geração corrente.
    std::vector<Individual> refPop = pop;

    writeHistoryHeader(history);
    auto start = std::chrono::steady_clock::now();
    Individual globalBest = pop[0];

    for (int gen = startGeneration; gen < generations; ++gen) {
        // Avalia todos os indivíduos da população em paralelo. Cada
        // avaliação é independente: cria suas próprias engines locais e só
        // lê refPop (nunca escreve nela), então não há necessidade de
        // sincronização além do contador atômico de trabalho. `threads`
        // controla quantos indivíduos são avaliados simultaneamente.
        std::atomic<int> next{0};
        auto worker = [&]() {
            for (;;) {
                int i = next.fetch_add(1);
                if (i >= (int)pop.size()) break;
                pop[i].fitness = evaluateIndividual(pop[i], refPop, i,
                                                    seed + 1000003u*(gen+1) + 7919u*i,
                                                    gamesPerMatch);
            }
        };

        int nWorkers = std::max(1, std::min(threads, (int)pop.size()));
        std::vector<std::thread> pool;
        pool.reserve(nWorkers);
        for (int t = 0; t < nWorkers; ++t) pool.emplace_back(worker);
        for (auto& t : pool) t.join();

        std::sort(pop.begin(), pop.end(), [](const Individual& a, const Individual& b) {
            return a.fitness > b.fitness;
        });

        if (pop.front().fitness > globalBest.fitness) {
            globalBest = pop.front();
            saveCheckpoint(checkpoint, gen + 1, globalBest);
        }

        writeHistory(history, gen + 1, pop, globalBest);
        std::printf("GEN %3d/%d  ", gen + 1, generations);
        printIndividual(pop.front());
        std::printf("           global: "); printIndividual(globalBest);

        // A população recém-avaliada e ordenada desta geração vira a
        // referência congelada para a avaliação da PRÓXIMA geração.
        refPop = pop;

        if (timeExpired(start, timeLimitSec)) {
            std::printf("Limite de tempo atingido.\n");
            break;
        }

        int eliteCount = std::max(2, (int)std::lround(populationSize * eliteFraction));
        int immigrantCount = (int)std::lround(populationSize * immigrantFraction);

        std::vector<Individual> nextPop;
        nextPop.reserve(populationSize);

        // Elitismo: campeões nunca são perdidos.
        for (int i = 0; i < eliteCount; ++i) nextPop.push_back(pop[i]);

        // Imigrantes aleatórios: defesa explícita contra convergência prematura.
        while ((int)nextPop.size() < eliteCount + immigrantCount &&
               (int)nextPop.size() < populationSize)
            nextPop.push_back(randomIndividual(rng));

        std::uniform_int_distribution<int> parentDist(0, eliteCount - 1);
        while ((int)nextPop.size() < populationSize) {
            const Individual& a = pop[parentDist(rng)];
            const Individual& b = pop[parentDist(rng)];
            Individual child = crossover(a, b, rng);
            child = mutate(child, rng, mutationRate);
            nextPop.push_back(child);
        }
        pop.swap(nextPop);
    }


    std::ofstream out(result);
    out << "algorithm=genetic\n";
    out << "generations=" << generations << "\n";
    out << "population=" << populationSize << "\n";
    out << "games_per_match=" << gamesPerMatch << "\n";
    out << "fitness=" << globalBest.fitness << "\n";
    for (int i = 0; i < NPARAM; ++i)
        out << PARAM_DEFS[i].name << "=" << globalBest.x[i] << "\n";

    std::printf("\n=== RESULTADO FINAL ===\n");
    printIndividual(globalBest);
    std::printf("Salvo em %s\n", result.c_str());
    return 0;
}
