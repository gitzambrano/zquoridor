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
// CONFIG DEFAULTS -- edite aqui para mudar o comportamento padrao sem
// precisar passar flag nenhuma. Cada constante abaixo tem uma flag de
// linha de comando equivalente (ver printUsage()); a flag, quando
// passada, tem prioridade sobre a constante. Rodar `./selfplay --out
// x.bin` sem mais nada usa TODOS os valores daqui.
//
// GERAR DADOS COM AVALIACAO HEURISTICA (sem NNUE), p.ex. pra montar uma
// nova bateria de testes: mude FORCE_HEURISTIC_DEFAULT para true abaixo,
// OU passe --heuristic na linha de comando toda vez -- as duas formas tem
// o mesmo efeito, a flag so existe pra nao precisar editar/recompilar.
// =============================================================================
constexpr int GAMES_DEFAULT = 2000;
constexpr int CHUNK_GAMES_DEFAULT = 2000;
constexpr int DEPTH_DEFAULT = 40;
constexpr int TIME_MS_DEFAULT = 100;
// NNUE e o default de avaliacao de folha (SelfPlayConfig::nnueWeightsPath
// ja nasce apontando pra defaultNnueWeightsPath(), nnue.hpp -- ver lá se
// quiser mudar o CAMINHO default). Esta constante aqui e' só o
// liga/desliga: true = ignora qualquer NNUE e usa evalSimple sempre
// (equivalente a --heuristic); false = tenta NNUE, cai pra heuristico
// sozinho se o arquivo de pesos nao existir.
constexpr bool FORCE_HEURISTIC_DEFAULT = false;
constexpr int OPENING_PLIES_DEFAULT = 6;
constexpr double EPSILON_DEFAULT = 0.05;
constexpr int OPENING_PLIES2_DEFAULT = 10;
constexpr double EPSILON_OPENING2_DEFAULT = 0.8;
constexpr double EPSILON_MIDGAME_DEFAULT = 0.02;
constexpr bool SEPARATE_TT_DEFAULT = false;  // false = TT compartilhada entre as 2 cores (mais rapido p/ gerar dados)
constexpr int MAX_PLIES_DEFAULT = 300;
constexpr int THREADS_DEFAULT = 0;           // 0 = usa hardware_concurrency()
constexpr unsigned SEED_DEFAULT = 1;
constexpr int START_SHARD_DEFAULT = 0;
constexpr const char* OUT_DEFAULT = "";      // vazio = --out continua obrigatorio

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
        "  --games N          total de partidas a gerar (default %d)\n"
        "  --chunk-games N    partidas por arquivo .bin (default %d)\n"
        "  --depth N           profundidade maxima da busca (default %d)\n"
        "  --time-ms N         orcamento de tempo por lance em ms (default %d)\n"
        "  --nnue-weights PATH caminho para pesos NNUE quantizados (default:\n"
        "                      data/nnue/nnue_weights_int8.bin). NNUE e o\n"
        "                      default de avaliacao de folha deste binario; se\n"
        "                      o arquivo (default ou passado aqui) nao existir,\n"
        "                      cai automaticamente para evalSimple (heuristico).\n"
        "  --heuristic         forca avaliacao heuristica (evalSimple), mesmo\n"
        "                      que pesos NNUE existam -- debug/historico/fallback/\n"
        "                      gerar bateria de testes sem NNUE. Equivalente a\n"
        "                      mudar FORCE_HEURISTIC_DEFAULT=true no topo do\n"
        "                      arquivo (default atual: %s).\n"
        "  --nnue              forca NNUE, sobrepondo FORCE_HEURISTIC_DEFAULT=true\n"
        "                      do arquivo sem precisar recompilar.\n"
        "  --opening-plies N   fase 1: lances 0..N-1 com epsilon1 (default %d)\n"
        "  --epsilon F         epsilon da fase 1 (default %.2f)\n"
        "  --opening-plies2 N  fase 2: lances N1..N2-1 com epsilon2 (default %d)\n"
        "  --epsilon-opening2 F epsilon da fase 2 (default %.2f)\n"
        "  --epsilon-midgame F  epsilon apos fase 2 (default %.3f)\n"
        "  --separate-tt        cada cor usa sua propria TT/engine, isolada\n"
        "                       (default: %s -- mais rapido para gerar dados;\n"
        "                       use esta flag para comparar taxa de empate com\n"
        "                       a arena)\n"
        "  --max-plies N       corte de seguranca por partida (default %d)\n"
        "  --threads N         threads paralelas (default %s)\n"
        "  --seed N            semente do RNG (default %u)\n"
        "  --start-shard N     primeiro indice de shard a escrever (default %d);\n"
        "                      permite retomar sem sobrescrever shards existentes.\n"
        "  --out PATH          arquivo/template de saida (obrigatorio).\n"
        "                      Use {shard:03d} para chunks: data/selfplay_{shard:03d}.bin\n",
        prog, GAMES_DEFAULT, CHUNK_GAMES_DEFAULT, DEPTH_DEFAULT, TIME_MS_DEFAULT,
        FORCE_HEURISTIC_DEFAULT ? "true, ja forcado" : "false",
        OPENING_PLIES_DEFAULT, EPSILON_DEFAULT, OPENING_PLIES2_DEFAULT, EPSILON_OPENING2_DEFAULT,
        EPSILON_MIDGAME_DEFAULT, SEPARATE_TT_DEFAULT ? "separada" : "compartilhada entre as 2 cores",
        MAX_PLIES_DEFAULT, THREADS_DEFAULT > 0 ? std::to_string(THREADS_DEFAULT).c_str() : "hardware_concurrency",
        SEED_DEFAULT, START_SHARD_DEFAULT);
}

int main(int argc, char** argv) {
    SelfPlayConfig cfg;
    cfg.maxDepth            = DEPTH_DEFAULT;
    cfg.timeBudgetMs        = TIME_MS_DEFAULT;
    cfg.forceHeuristic      = FORCE_HEURISTIC_DEFAULT;
    cfg.openingRandomPlies  = OPENING_PLIES_DEFAULT;
    cfg.epsilon             = EPSILON_DEFAULT;
    cfg.openingRandomPlies2 = OPENING_PLIES2_DEFAULT;
    cfg.epsilon2            = EPSILON_OPENING2_DEFAULT;
    cfg.epsilonMidgame      = EPSILON_MIDGAME_DEFAULT;
    cfg.sharedTT            = !SEPARATE_TT_DEFAULT;
    cfg.maxPlies            = MAX_PLIES_DEFAULT;
    cfg.numThreads          = THREADS_DEFAULT;
    cfg.seed                = SEED_DEFAULT;
    std::string outTemplate = OUT_DEFAULT;
    int totalGames  = GAMES_DEFAULT;
    int chunkGames  = CHUNK_GAMES_DEFAULT;
    int startShard  = START_SHARD_DEFAULT;

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
        else if (a == "--nnue")            cfg.forceHeuristic        = false;
        else if (a == "--opening-plies")   cfg.openingRandomPlies    = std::atoi(next("--opening-plies").c_str());
        else if (a == "--epsilon")         cfg.epsilon               = std::atof(next("--epsilon").c_str());
        else if (a == "--opening-plies2")  cfg.openingRandomPlies2   = std::atoi(next("--opening-plies2").c_str());
        else if (a == "--epsilon-opening2") cfg.epsilon2             = std::atof(next("--epsilon-opening2").c_str());
        else if (a == "--epsilon-midgame") cfg.epsilonMidgame        = std::atof(next("--epsilon-midgame").c_str());
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
    std::printf("abertura: fase1=[0..%d) eps=%.2f | fase2=[%d..%d) eps=%.2f | midgame eps=%.3f\n",
                cfg.openingRandomPlies, cfg.epsilon,
                cfg.openingRandomPlies, cfg.openingRandomPlies2, cfg.epsilon2,
                cfg.epsilonMidgame);
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
