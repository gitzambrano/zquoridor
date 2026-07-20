// selfplay_main.cpp -- CLI da Fase 4 (self-play).
//
// Uso:
//   ./selfplay --games 2000 --depth 40 --time-ms 100 --threads 8 \
//              --opening-plies 6 --epsilon 0.25 --seed 1 --out data/selfplay_001.bin
//
// Grava um único arquivo binário de TrainingSample (ver selfplay.hpp) --
// sem cabeçalho, sem separadores: puro array C de structs packed, pronto
// pra ler com numpy.fromfile + dtype estruturado (training/read_selfplay.py).
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <string>
#include <chrono>
#include <thread>
#include "selfplay.hpp"
using namespace qr;

static void printUsage(const char* prog) {
    std::fprintf(stderr,
        "Uso: %s [opcoes]\n"
        "  --games N          numero de partidas (default 1000)\n"
        "  --depth N          profundidade maxima da busca (default 40)\n"
        "  --time-ms N        orcamento de tempo por lance em ms (default 100)\n"
        "  --opening-plies N  lances iniciais sujeitos a ruido (default 6)\n"
        "  --epsilon F        prob. de lance aleatorio na abertura (default 0.25)\n"
        "  --max-plies N      corte de seguranca por partida (default 300)\n"
        "  --threads N        threads paralelas (default hardware_concurrency)\n"
        "  --seed N           semente do RNG (default 1)\n"
        "  --out PATH         arquivo binario de saida (obrigatorio)\n",
        prog);
}

int main(int argc, char** argv) {
    SelfPlayConfig cfg;
    std::string outPath;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&](const char* flag) -> std::string {
            if (i + 1 >= argc) { std::fprintf(stderr, "faltou valor para %s\n", flag); std::exit(1); }
            return argv[++i];
        };
        if (a == "--games") cfg.numGames = std::atoi(next("--games").c_str());
        else if (a == "--depth") cfg.maxDepth = std::atoi(next("--depth").c_str());
        else if (a == "--time-ms") cfg.timeBudgetMs = std::atoi(next("--time-ms").c_str());
        else if (a == "--opening-plies") cfg.openingRandomPlies = std::atoi(next("--opening-plies").c_str());
        else if (a == "--epsilon") cfg.epsilon = std::atof(next("--epsilon").c_str());
        else if (a == "--max-plies") cfg.maxPlies = std::atoi(next("--max-plies").c_str());
        else if (a == "--threads") cfg.numThreads = std::atoi(next("--threads").c_str());
        else if (a == "--seed") cfg.seed = (unsigned)std::atol(next("--seed").c_str());
        else if (a == "--out") outPath = next("--out");
        else if (a == "-h" || a == "--help") { printUsage(argv[0]); return 0; }
        else { std::fprintf(stderr, "opcao desconhecida: %s\n", a.c_str()); printUsage(argv[0]); return 1; }
    }
    if (outPath.empty()) { std::fprintf(stderr, "erro: --out e obrigatorio\n"); printUsage(argv[0]); return 1; }

    int nThreads = cfg.numThreads > 0 ? cfg.numThreads : std::max(1u, std::thread::hardware_concurrency());
    std::printf("self-play: %d partidas, profundidade<=%d, %dms/lance, %d threads, out=%s\n",
                cfg.numGames, cfg.maxDepth, cfg.timeBudgetMs, nThreads, outPath.c_str());
    std::printf("abertura: %d lances com epsilon=%.2f | corte de seguranca: %d lances/partida\n",
                cfg.openingRandomPlies, cfg.epsilon, cfg.maxPlies);
    std::printf("registro de treino: %zu bytes/posicao (packed)\n\n", sizeof(TrainingSample));

    SelfPlayStats stats;
    auto t0 = std::chrono::steady_clock::now();

    // thread de progresso simples (imprime a cada ~2s)
    std::atomic<bool> done{false};
    std::thread progress([&]() {
        while (!done.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(2000));
            if (done.load()) break;
            std::printf("  progresso: %llu/%d partidas | %llu posicoes gravadas | %llu descartadas\n",
                        (unsigned long long)stats.gamesPlayed.load(), cfg.numGames,
                        (unsigned long long)stats.positionsWritten.load(),
                        (unsigned long long)stats.gamesDiscarded.load());
        }
    });

    runSelfPlay(cfg, outPath, stats);
    done = true;
    progress.join();

    double totalS = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::printf("\nconcluido em %.1f s\n", totalS);
    std::printf("partidas jogadas: %llu (descartadas por nao terminar: %llu)\n",
                (unsigned long long)stats.gamesPlayed.load(), (unsigned long long)stats.gamesDiscarded.load());
    std::printf("posicoes gravadas: %llu (%.1f/partida valida)\n",
                (unsigned long long)stats.positionsWritten.load(),
                stats.gamesPlayed.load() > stats.gamesDiscarded.load()
                    ? (double)stats.positionsWritten.load() / (stats.gamesPlayed.load() - stats.gamesDiscarded.load())
                    : 0.0);
    std::printf("nos de busca totais: %llu (%.0f nos/s)\n",
                (unsigned long long)stats.totalNodes.load(), stats.totalNodes.load() / totalS);
    std::printf("throughput: %.1f posicoes/s\n", stats.positionsWritten.load() / totalS);
    return 0;
}
