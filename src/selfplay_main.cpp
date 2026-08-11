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
        prog);
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
