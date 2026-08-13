// selfplay_main.cpp -- CLI da Fase 4 (self-play).
//
// Uso basico (arquivo unico):
//   ./selfplay --games 2000 --depth 40 --time-ms 100 --threads 8 \
//              --out data/selfplay_shard001.bin
//
// Uso com multiplos chunks:
//   ./selfplay --games 20000 --chunk-games 2000 --depth 40 --time-ms 100 \
//              --threads 8 --out "data/selfplay_{shard:03d}.bin"
//
// --chunk-games N : partidas por arquivo .bin (default 2000).
//                   Se --games <= --chunk-games, grava um unico arquivo.
// --out TEMPLATE  : caminho de saida. Se contem o placeholder literal
//                   {shard:03d}, cada chunk recebe sufixo numerico (000,
//                   001, ...). Sem placeholder, comportamento antigo.
//
// Grava array de TrainingSample packed (ver selfplay.hpp), pronto para
// leitura com numpy.fromfile + dtype estruturado (training/read_selfplay.py).
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <string>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <thread>
#include "selfplay.hpp"
using namespace qr;

// =============================================================================
// CONFIG DEFAULTS -- MCab (hibrido MCTS+alpha-beta, plan-hybrid-mc-ab.md,
// Fase 6). Mesmo padrao/regra do bloco equivalente em tools/arena/arena.cpp
// (Secao 10.1 do plano): toda flag de argv abaixo tem uma constante
// `constexpr` nomeada aqui, uma variavel `static` inicializada a partir
// dela, e o parsing em main() so sobrescreve essa variavel quando a flag
// correspondente aparece. Rodar este binario sem NENHUMA flag --mcab-*
// produz exatamente o mesmo comportamento que passar todas nos valores
// abaixo -- e, com MCAB_ENABLED_DEFAULT=false, exatamente o mesmo
// comportamento de antes desta fase existir (AB puro, byte-compativel).
//
// Sem prefixo e1-/e2- (ao contrario de arena.cpp): self-play so tem UMA
// engine logica por partida.
//
// ATENCAO, unica diferenca de proposito frente aos defaults de arena.cpp:
// MCAB_ROOT_NOISE_ENABLED_DEFAULT = true aqui (arena.cpp usa false). Ruido
// de Dirichlet na raiz serve para diversidade de ABERTURA em dados de
// treino -- e o comportamento correto por default quando o hibrido estiver
// ligado aqui -- nao para forca de partida (arena mede forca, entao la o
// ruido so deve entrar via flag explicita). Isso so tem efeito quando
// MCAB_ENABLED_DEFAULT (ou --mcab) estiver de fato ligado; com o default
// MCAB_ENABLED_DEFAULT=false o binario continua 100% AB puro.
// =============================================================================
constexpr bool   MCAB_ENABLED_DEFAULT             = false;
constexpr int    MCAB_NODE_BUDGET_DEFAULT         = 20000;
constexpr int    MCAB_LEAF_DEPTH_DEFAULT          = 4;
constexpr int    MCAB_LEAF_DEPTH_MAX_DEFAULT      = 8;
constexpr bool   MCAB_ADAPTIVE_LEAF_DEPTH_DEFAULT = false;
constexpr double MCAB_CPUCT_DEFAULT               = 1.5;
constexpr double MCAB_FPU_REDUCTION_DEFAULT       = 0.1;
constexpr double MCAB_SCORE_SCALE_DEFAULT         = 200.0;  // = NNUE_EVAL_SCALE
constexpr bool   MCAB_TREE_REUSE_DEFAULT          = true;
constexpr bool   MCAB_ROOT_NOISE_ENABLED_DEFAULT  = true;   // diferente de arena.cpp -- ver nota acima
constexpr double MCAB_ROOT_NOISE_ALPHA_DEFAULT    = 0.3;
constexpr double MCAB_ROOT_NOISE_EPSILON_DEFAULT  = 0.25;
constexpr int    MCAB_MAX_TREE_DEPTH_DEFAULT      = 48;

static bool   g_mcabEnabled           = MCAB_ENABLED_DEFAULT;
static int    g_mcabNodeBudget        = MCAB_NODE_BUDGET_DEFAULT;
static int    g_mcabLeafDepth         = MCAB_LEAF_DEPTH_DEFAULT;
static int    g_mcabLeafDepthMax      = MCAB_LEAF_DEPTH_MAX_DEFAULT;
static bool   g_mcabAdaptiveLeafDepth = MCAB_ADAPTIVE_LEAF_DEPTH_DEFAULT;
static double g_mcabCPuct             = MCAB_CPUCT_DEFAULT;
static double g_mcabFpuReduction      = MCAB_FPU_REDUCTION_DEFAULT;
static double g_mcabScoreScale        = MCAB_SCORE_SCALE_DEFAULT;
static bool   g_mcabTreeReuse         = MCAB_TREE_REUSE_DEFAULT;
static bool   g_mcabRootNoiseEnabled  = MCAB_ROOT_NOISE_ENABLED_DEFAULT;
static double g_mcabRootNoiseAlpha    = MCAB_ROOT_NOISE_ALPHA_DEFAULT;
static double g_mcabRootNoiseEpsilon  = MCAB_ROOT_NOISE_EPSILON_DEFAULT;
static int    g_mcabMaxTreeDepth      = MCAB_MAX_TREE_DEPTH_DEFAULT;

// Substitui "{shard:03d}" no template pelo numero do shard com zero-padding.
static std::string formatShardPath(const std::string& tmpl, int shard) {
    const std::string marker = "{shard:03d}";
    size_t pos = tmpl.find(marker);
    if (pos == std::string::npos) return tmpl;
    std::ostringstream oss;
    oss << std::setw(3) << std::setfill('0') << shard;
    return tmpl.substr(0, pos) + oss.str() + tmpl.substr(pos + marker.size());
}

static void printUsage(const char* prog) {
    std::fprintf(stderr,
        "Uso: %s [opcoes]\n"
        "  --games N          total de partidas a gerar (default 2000)\n"
        "  --chunk-games N    partidas por arquivo .bin (default 2000)\n"
        "  --depth N           profundidade maxima da busca (default 40)\n"
        "  --time-ms N         orcamento de tempo por lance em ms (default 100)\n"
        "  --nnue-weights PATH caminho para pesos NNUE quantizados (default:\n"
        "                      data/nnue/nnue_weights_int8.bin). NNUE e o\n"
        "                      default de avaliacao de folha deste binario; se\n"
        "                      o arquivo (default ou passado aqui) nao existir,\n"
        "                      cai automaticamente para evalSimple (heuristico).\n"
        "  --heuristic         forca avaliacao heuristica (evalSimple), mesmo\n"
        "                      que pesos NNUE existam -- debug/historico/fallback.\n"
        "  --policy-order      liga (redundante -- ja e o default, ver --no-policy-order)\n"
        "                      a ordenacao de lances assistida pela cabeca de politica\n"
        "                      da NNUE (Negamax::setPolicyOrderingEnabled,\n"
        "                      prompt_policy_ordering.md). Só tem efeito quando NNUE\n"
        "                      está ativo (ignorado silenciosamente com --heuristic).\n"
        "                      Default: ligado.\n"
        "  --no-policy-order   desliga a ordenacao por politica (volta ao comportamento\n"
        "                      anterior a 2026-08 / reproduz shards gerados antes disso).\n"
        "  --policy-order-min-depth N  so aplica a ordenacao por politica em nos\n"
        "                      com depth restante >= N (default 3). forwardPolicyQuant\n"
        "                      custa ~5.8x mais que o eval de folha; sem este piso\n"
        "                      ele roda em todo no interno e derruba nos/s ~3x.\n"
        "  --opening-plies N   fase 1: lances 0..N-1 com epsilon1 (default 6)\n"
        "  --epsilon F         epsilon da fase 1 (default 0.05)\n"
        "  --opening-plies2 N  fase 2: lances N1..N2-1 com epsilon2 (default 10)\n"
        "  --epsilon-opening2 F epsilon da fase 2 (default 0.8)\n"
        "  --epsilon-midgame F  epsilon apos fase 2 (default 0.02; tambem usado\n"
        "                      como ruido residual do modo --mc-mode apos a\n"
        "                      janela de decaimento de temperatura)\n"
        "\n"
        "  --- modo Monte Carlo / temperatura (alternativa ao epsilon-greedy\n"
        "      acima -- ver nota completa em SelfPlayConfig::mcMode, selfplay.hpp) ---\n"
        "  --mc-mode           liga o modo Monte Carlo (default: desligado, usa\n"
        "                      o modo epsilon-greedy antigo controlado pelas\n"
        "                      flags --opening-plies*/--epsilon* acima)\n"
        "  --mc-obvious-plies N nº de lances iniciais (ply 0..N-1) com temperatura\n"
        "                      fixa e baixa (lances obvios do Quoridor, pouca\n"
        "                      variancia de proposito) (default 3)\n"
        "  --mc-temp-obvious F temperatura da janela de lances obvios acima (default 0.15)\n"
        "  --mc-temp-opening F temperatura no primeiro ply pos-obvios, estilo AlphaZero (default 1.35)\n"
        "  --mc-temp-end F     temperatura ao fim da janela de decaimento (default 0.12)\n"
        "  --mc-temp-decay-plies N  nº de lances (apos --mc-obvious-plies) sobre os\n"
        "                      quais a temperatura decai linearmente de\n"
        "                      --mc-temp-opening a --mc-temp-end (default 20);\n"
        "                      sem busca nenhuma nesses lances -- so forward da cabeca de politica\n"
        "\n"
        "  --- MCab: hibrido MCTS+alpha-beta (plan-hybrid-mc-ab.md, Fase 6) ---\n"
        "  --mcab              liga o hibrido MCab (default: desligado -- AB puro,\n"
        "                      igual a antes desta feature existir). REQUER NNUE\n"
        "                      ativa (erro se combinado com --heuristic ou se o\n"
        "                      load dos pesos NNUE falhar).\n"
        "  --mcab-nodes N      orcamento de nos expandidos por lance (default %d)\n"
        "  --mcab-leaf-depth N profundidade (plies) da busca AB rasa em cada folha\n"
        "                      MCTS (default %d)\n"
        "  --mcab-leaf-depth-max N teto de --mcab-leaf-depth quando\n"
        "                      --mcab-adaptive-leaf-depth estiver ligado (default %d)\n"
        "  --mcab-adaptive-leaf-depth  escala leafDepth com N do no pai (default: desligado)\n"
        "  --mcab-cpuct X      constante de exploracao do PUCT (default %.2f)\n"
        "  --mcab-fpu X        reducao de first-play-urgency p/ filhos nao visitados\n"
        "                      (Q(pai) - X), default %.2f\n"
        "  --mcab-score-scale X escala usada na conversao score->Q (default %.1f,\n"
        "                      = NNUE_EVAL_SCALE)\n"
        "  --mcab-no-tree-reuse desliga reuso de subarvore entre lances (default: ligado)\n"
        "  --mcab-no-root-noise desliga ruido de Dirichlet nos priors da raiz\n"
        "                      (default: LIGADO aqui -- diferente da arena --\n"
        "                      serve pra diversidade de abertura nos dados gerados)\n"
        "  --mcab-root-noise-alpha X   parametro alfa da Dirichlet (default %.2f)\n"
        "  --mcab-root-noise-epsilon X peso do ruido misturado ao prior (default %.2f)\n"
        "  --mcab-max-tree-depth N     dimensiona a pilha de acumuladores NNUE\n"
        "                      incremental da arvore MCab (default %d)\n"
        "  --separate-tt        cada cor usa sua propria TT/engine, isolada\n"
        "                       (default: TT compartilhada entre as 2 cores --\n"
        "                       mais rapido para gerar dados; use esta flag para\n"
        "                       comparar taxa de empate com a arena)\n"
        "  --max-plies N       corte de seguranca por partida (default 300)\n"
        "  --threads N         threads paralelas (default hardware_concurrency)\n"
        "  --seed N            semente do RNG (default 1)\n"
        "  --start-shard N     primeiro indice de shard a escrever (default 0);\n"
        "                      permite retomar sem sobrescrever shards existentes.\n"
        "  --out PATH          arquivo/template de saida (obrigatorio).\n"
        "                      Use {shard:03d} para chunks: data/selfplay_{shard:03d}.bin\n",
        prog,
        MCAB_NODE_BUDGET_DEFAULT, MCAB_LEAF_DEPTH_DEFAULT, MCAB_LEAF_DEPTH_MAX_DEFAULT,
        MCAB_CPUCT_DEFAULT, MCAB_FPU_REDUCTION_DEFAULT, MCAB_SCORE_SCALE_DEFAULT,
        MCAB_ROOT_NOISE_ALPHA_DEFAULT, MCAB_ROOT_NOISE_EPSILON_DEFAULT, MCAB_MAX_TREE_DEPTH_DEFAULT);
}

int main(int argc, char** argv) {
    SelfPlayConfig cfg;
    std::string outTemplate;
    int totalGames  = 2000;
    int chunkGames  = 2000;
    int startShard  = 0;

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&](const char* flag) -> std::string {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "faltou valor para %s\n", flag);
                std::exit(1);
            }
            return argv[++i];
        };
        if      (a == "--games")           totalGames                = std::atoi(next("--games").c_str());
        else if (a == "--chunk-games")     chunkGames                = std::atoi(next("--chunk-games").c_str());
        else if (a == "--depth")           cfg.maxDepth              = std::atoi(next("--depth").c_str());
        else if (a == "--time-ms")         cfg.timeBudgetMs          = std::atoi(next("--time-ms").c_str());
        else if (a == "--nnue-weights")  { cfg.nnueWeightsPath       = next("--nnue-weights");
                                            cfg.nnueWeightsExplicit   = true; }
        else if (a == "--heuristic")       cfg.forceHeuristic        = true;
        else if (a == "--policy-order")    cfg.policyOrderingEnabled = true;
        else if (a == "--no-policy-order") cfg.policyOrderingEnabled = false;
        else if (a == "--policy-order-min-depth") cfg.policyOrderingMinDepth = std::atoi(next("--policy-order-min-depth").c_str());
        else if (a == "--opening-plies")   cfg.openingRandomPlies    = std::atoi(next("--opening-plies").c_str());
        else if (a == "--epsilon")         cfg.epsilon               = std::atof(next("--epsilon").c_str());
        else if (a == "--opening-plies2")  cfg.openingRandomPlies2   = std::atoi(next("--opening-plies2").c_str());
        else if (a == "--epsilon-opening2") cfg.epsilon2             = std::atof(next("--epsilon-opening2").c_str());
        else if (a == "--epsilon-midgame") cfg.epsilonMidgame        = std::atof(next("--epsilon-midgame").c_str());
        else if (a == "--mc-mode")         cfg.mcMode                = true;
        else if (a == "--mc-obvious-plies") cfg.mcObviousPlies       = std::atoi(next("--mc-obvious-plies").c_str());
        else if (a == "--mc-temp-obvious") cfg.mcTemperatureObvious  = std::atof(next("--mc-temp-obvious").c_str());
        else if (a == "--mc-temp-opening") cfg.mcTemperatureOpening  = std::atof(next("--mc-temp-opening").c_str());
        else if (a == "--mc-temp-end")     cfg.mcTemperatureEnd      = std::atof(next("--mc-temp-end").c_str());
        else if (a == "--mc-temp-decay-plies") cfg.mcTempDecayPlies  = std::atoi(next("--mc-temp-decay-plies").c_str());
        else if (a == "--mcab")                     g_mcabEnabled           = true;
        else if (a == "--mcab-nodes")                g_mcabNodeBudget        = std::atoi(next("--mcab-nodes").c_str());
        else if (a == "--mcab-leaf-depth")           g_mcabLeafDepth         = std::atoi(next("--mcab-leaf-depth").c_str());
        else if (a == "--mcab-leaf-depth-max")       g_mcabLeafDepthMax      = std::atoi(next("--mcab-leaf-depth-max").c_str());
        else if (a == "--mcab-adaptive-leaf-depth")  g_mcabAdaptiveLeafDepth = true;
        else if (a == "--mcab-cpuct")                g_mcabCPuct             = std::atof(next("--mcab-cpuct").c_str());
        else if (a == "--mcab-fpu")                  g_mcabFpuReduction      = std::atof(next("--mcab-fpu").c_str());
        else if (a == "--mcab-score-scale")          g_mcabScoreScale        = std::atof(next("--mcab-score-scale").c_str());
        else if (a == "--mcab-no-tree-reuse")        g_mcabTreeReuse         = false;
        else if (a == "--mcab-no-root-noise")        g_mcabRootNoiseEnabled  = false;
        else if (a == "--mcab-root-noise-alpha")     g_mcabRootNoiseAlpha    = std::atof(next("--mcab-root-noise-alpha").c_str());
        else if (a == "--mcab-root-noise-epsilon")   g_mcabRootNoiseEpsilon  = std::atof(next("--mcab-root-noise-epsilon").c_str());
        else if (a == "--mcab-max-tree-depth")       g_mcabMaxTreeDepth      = std::atoi(next("--mcab-max-tree-depth").c_str());
        else if (a == "--separate-tt")     cfg.sharedTT              = false;
        else if (a == "--max-plies")       cfg.maxPlies              = std::atoi(next("--max-plies").c_str());
        else if (a == "--threads")         cfg.numThreads            = std::atoi(next("--threads").c_str());
        else if (a == "--seed")            cfg.seed                  = (unsigned)std::atol(next("--seed").c_str());
        else if (a == "--start-shard")     startShard                = std::atoi(next("--start-shard").c_str());
        else if (a == "--out")           outTemplate         = next("--out");
        else if (a == "-h" || a == "--help") { printUsage(argv[0]); return 0; }
        else {
            std::fprintf(stderr, "opcao desconhecida: %s\n", a.c_str());
            printUsage(argv[0]);
            return 1;
        }
    }

    if (outTemplate.empty()) {
        std::fprintf(stderr, "erro: --out e obrigatorio\n");
        printUsage(argv[0]);
        return 1;
    }
    if (chunkGames <= 0) {
        std::fprintf(stderr, "erro: --chunk-games deve ser > 0\n");
        return 1;
    }
    // MCab requer NNUE ativa (Secao 2 do plano) -- checagem estatica aqui
    // cobre o caso explicito (--mcab junto de --heuristic na propria linha
    // de comando); o caso dinamico (pesos NNUE nao carregam mesmo sem
    // --heuristic) e coberto separadamente dentro de runSelfPlay
    // (selfplay.hpp), que e onde o resultado do load so fica conhecido.
    if (g_mcabEnabled && cfg.forceHeuristic) {
        std::fprintf(stderr,
            "erro: --mcab requer avaliacao NNUE ativa (Secao 2 do plano MCab) e nao\n"
            "pode ser combinado com --heuristic. Remova uma das duas flags.\n");
        return 1;
    }
    cfg.mcabParams.enabled           = g_mcabEnabled;
    cfg.mcabParams.nodeBudget        = g_mcabNodeBudget;
    cfg.mcabParams.leafDepth         = g_mcabLeafDepth;
    cfg.mcabParams.leafDepthMax      = g_mcabLeafDepthMax;
    cfg.mcabParams.adaptiveLeafDepth = g_mcabAdaptiveLeafDepth;
    cfg.mcabParams.cPuct             = g_mcabCPuct;
    cfg.mcabParams.fpuReduction      = g_mcabFpuReduction;
    cfg.mcabParams.scoreScale        = g_mcabScoreScale;
    cfg.mcabParams.treeReuse         = g_mcabTreeReuse;
    cfg.mcabParams.rootNoiseEnabled  = g_mcabRootNoiseEnabled;
    cfg.mcabParams.rootNoiseAlpha    = g_mcabRootNoiseAlpha;
    cfg.mcabParams.rootNoiseEpsilon  = g_mcabRootNoiseEpsilon;
    cfg.mcabParams.maxTreeDepth      = g_mcabMaxTreeDepth;
    // rootNoiseSeed fica no default de McabParams (0x9E3779B9) aqui -- é
    // sobreposto por thread/partida em playOneGame via mcabRunner.seedNoise
    // (ver selfplay.hpp), então o valor global não importa na prática.

    bool multiChunk = (outTemplate.find("{shard:03d}") != std::string::npos);
    int nChunks = multiChunk ? (totalGames + chunkGames - 1) / chunkGames : 1;
    int nThreads = cfg.numThreads > 0
                   ? cfg.numThreads
                   : (int)std::max(1u, std::thread::hardware_concurrency());

    std::printf("=== self-play: %d partidas totais | %d por chunk | %d chunk(s) ===\n",
                totalGames, chunkGames, nChunks);
    std::printf("busca: profundidade<=%d, %dms/lance\n",
                cfg.maxDepth, cfg.timeBudgetMs);
    std::printf("avaliacao de folha: %s\n",
                cfg.forceHeuristic ? "heuristica (evalSimple) -- forcada via --heuristic"
                                   : ("NNUE quantizada (" + cfg.nnueWeightsPath +
                                      "), com fallback automatico para heuristica se o arquivo nao existir").c_str());
    std::printf("ordenacao por politica NNUE: %s\n",
                cfg.forceHeuristic ? "n/a (heuristica forcada)"
                                   : (cfg.policyOrderingEnabled ? ("LIGADA (default, min-depth=" + std::to_string(cfg.policyOrderingMinDepth) + ")").c_str() : "desligada (--no-policy-order)"));
    if (cfg.mcMode) {
        std::printf("modo: MONTE CARLO/temperatura | obvios=[0..%d) temp=%.3f | opening=[%d..%d) temp=[%.3f..%.3f) | pos-decaimento eps=%.3f (2o/3o melhor)\n",
                    cfg.mcObviousPlies, cfg.mcTemperatureObvious,
                    cfg.mcObviousPlies, cfg.mcObviousPlies + cfg.mcTempDecayPlies,
                    cfg.mcTemperatureOpening, cfg.mcTemperatureEnd,
                    cfg.epsilonMidgame);
    } else {
        std::printf("modo: epsilon-greedy (antigo) | fase1=[0..%d) eps=%.2f | fase2=[%d..%d) eps=%.2f | midgame eps=%.3f\n",
                    cfg.openingRandomPlies, cfg.epsilon,
                    cfg.openingRandomPlies, cfg.openingRandomPlies2, cfg.epsilon2,
                    cfg.epsilonMidgame);
    }
    if (cfg.mcabParams.enabled) {
        std::printf("MCab: LIGADO | nodes=%d leaf-depth=%d cpuct=%.2f fpu=%.2f score-scale=%.1f"
                    " | tree-reuse=%s root-noise=%s(alpha=%.2f eps=%.2f) max-tree-depth=%d\n",
                    cfg.mcabParams.nodeBudget, cfg.mcabParams.leafDepth, cfg.mcabParams.cPuct,
                    cfg.mcabParams.fpuReduction, cfg.mcabParams.scoreScale,
                    cfg.mcabParams.treeReuse ? "on" : "off",
                    cfg.mcabParams.rootNoiseEnabled ? "on" : "off",
                    cfg.mcabParams.rootNoiseAlpha, cfg.mcabParams.rootNoiseEpsilon,
                    cfg.mcabParams.maxTreeDepth);
    } else {
        std::printf("MCab: desligado (AB puro -- default; use --mcab para ligar)\n");
    }
    std::printf("threads: %d | corte de seguranca: %d lances/partida\n", nThreads, cfg.maxPlies);
    std::printf("TT: %s\n", cfg.sharedTT ? "compartilhada entre as 2 cores (default)" : "separada por cor (--separate-tt)");
    std::printf("registro: %zu bytes/posicao (packed)\n\n", sizeof(TrainingSample));

    auto wallT0 = std::chrono::steady_clock::now();
    uint64_t totalPositions = 0;
    unsigned baseSeed = cfg.seed;

    for (int chunk = 0; chunk < nChunks; chunk++) {
        int shardIdx = startShard + chunk;
        // Seed diferente por chunk para variedade.
        cfg.seed = baseSeed + (unsigned)shardIdx * 999983u;

        int gamesThisChunk  = std::min(chunkGames, totalGames - chunk * chunkGames);
        cfg.numGames        = gamesThisChunk;
        std::string outPath = multiChunk ? formatShardPath(outTemplate, shardIdx) : outTemplate;

        std::printf("--- chunk %d/%d | %d partidas -> %s ---\n",
                    chunk + 1, nChunks, gamesThisChunk, outPath.c_str());

        SelfPlayStats stats;
        auto t0 = std::chrono::steady_clock::now();

        // thread de progresso: imprime a cada 5s
        std::atomic<bool> chunkDone{false};
        std::thread progress([&]() {
            while (!chunkDone.load()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(5000));
                if (chunkDone.load()) break;
                std::printf("  progresso: %llu/%d partidas | %llu posicoes | %llu empates | %llu descartadas\n",
                            (unsigned long long)stats.gamesPlayed.load(), gamesThisChunk,
                            (unsigned long long)stats.positionsWritten.load(),
                            (unsigned long long)stats.gamesDrawn.load(),
                            (unsigned long long)stats.gamesDiscarded.load());
                std::fflush(stdout);
            }
        });

        runSelfPlay(cfg, outPath, stats);
        chunkDone = true;
        progress.join();

        double chunkS = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t0).count();
        totalPositions += stats.positionsWritten.load();

        double posPerGame = (stats.gamesPlayed > stats.gamesDiscarded)
            ? (double)stats.positionsWritten.load() /
              (stats.gamesPlayed.load() - stats.gamesDiscarded.load())
            : 0.0;

        std::printf("  ok: %.1f s | %llu partidas (%llu empates, %llu desc.) | %llu pos (%.1f/partida)"
                    " | %.0f nos/s | %.1f pos/s\n\n",
                    chunkS,
                    (unsigned long long)stats.gamesPlayed.load(),
                    (unsigned long long)stats.gamesDrawn.load(),
                    (unsigned long long)stats.gamesDiscarded.load(),
                    (unsigned long long)stats.positionsWritten.load(),
                    posPerGame,
                    stats.totalNodes.load() / chunkS,
                    stats.positionsWritten.load() / chunkS);
    }

    double totalS = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - wallT0).count();
    std::printf("=== total: %d chunk(s) | %llu posicoes | %.1f s (%.1f pos/s) ===\n",
                nChunks,
                (unsigned long long)totalPositions,
                totalS,
                totalPositions / totalS);
    return 0;
}
