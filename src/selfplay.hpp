// selfplay.hpp -- Fase 4 do plano: geração de dados via self-play.
//
// Formato de saída: array de structs packed (TrainingSample), gravado cru
// em disco com fwrite. Não existe passo de pré-processamento no lado
// Python: o arquivo binário É o dataset. Basta abrir com
// numpy.fromfile(path, dtype=SAMPLE_DTYPE) (ver training/read_selfplay.py)
// e já se tem um array estruturado pronto pra virar tensores.
//
// Por que não usar as features esparsas de 332 bits (nnue.hpp) direto no
// arquivo? Porque isso desperdiçaria a maior parte de cada posição em
// zeros (332 bits = ~42 bytes, quase todos 0) e prenderia o formato de
// dados à arquitetura atual da rede. Guardamos o estado compacto (peões +
// bitboards de muro, 20 bytes) e derivamos as features esparsas em
// runtime no laço de treino (custo desprezível: é exatamente o que
// buildAccumulator já faz). Isso também deixa o dataset reutilizável se a
// arquitetura da rede mudar.
//
// Exceção: os campos ownDist/oppDist (distância BFS até a meta) SÃO
// gravados crus aqui, mesmo sendo "derivados" do estado -- não por
// eficiência, mas porque a informação necessária pra derivá-los de volta
// corretamente (o índice 0/1 do jogador mover, que decide qual GOAL_ROW
// usar) não sobrevive no registro; só a célula absoluta do peão é
// gravada. Calcular no C++, onde o índice do jogador ainda é conhecido
// com certeza, é mais robusto que tentar reconstruir isso no lado Python
// a partir da posição do peão (ver comentário completo no struct abaixo).
#pragma once
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
#include <random>
#include <thread>
#include <mutex>
#include <atomic>
#include <algorithm>
#include <string>
#include "rules.hpp"
#include "search.hpp"

namespace qr {

// --- formato do registro de treino -------------------------------------
#pragma pack(push, 1)
struct TrainingSample {
    uint8_t  ownPawn;       // célula (0..80) do peão de quem tem o lance ("mover")
    uint8_t  oppPawn;       // célula (0..80) do peão adversário
    uint64_t wallsH;        // bitboard de muros horizontais (mesma orientação p/ os 2 lados)
    uint64_t wallsV;        // bitboard de muros verticais
    int8_t   wallsLeftOwn;  // muros restantes do mover
    int8_t   wallsLeftOpp;  // muros restantes do adversário
    int16_t  searchScore;   // evalSimple(posição, mover) no momento do lance -- alvo auxiliar p/ bootstrapping
    int8_t   gameResult;    // +1 se o mover desta amostra venceu a partida, -1 se perdeu
    uint16_t policyTarget;  // índice do lance jogado, 0..208 (81 destino peão + 128 slot de muro)
    // Distância BFS (shortestPathLen, rules.hpp) até a linha de chegada,
    // já calculada aqui (não derivada em Python) por um motivo específico:
    // "de quem é o lance" (mover) é a única informação que dá o índice de
    // jogador (0/1) certo pra saber qual GOAL_ROW usar -- e essa
    // informação NÃO sobrevive no arquivo (o registro guarda só a célula
    // absoluta do peão, não o índice de jogador). Calcular aqui, onde
    // `mover`/`opp` ainda são conhecidos com certeza, evita ter que
    // reconstruir a identidade do jogador a partir de heurísticas de
    // posição no lado Python (frágil: um peão pode legalmente recuar).
    // uint8_t comporta até 255, bem acima de qualquer distância possível
    // no tabuleiro 9x9 (máximo teórico bem menor); nunca precisa de clamp
    // aqui (isso só acontece no lado do bucket, em nnue.hpp/nnue.py).
    uint8_t  ownDist;       // shortestPathLen do peão do mover até a meta dele
    uint8_t  oppDist;       // shortestPathLen do peão do oponente até a meta dele
};
#pragma pack(pop)
static_assert(sizeof(TrainingSample) == 27,
    "TrainingSample precisa ficar packed/sem padding -- o layout é lido direto por numpy no treino");

struct SelfPlayConfig {
    int numGames = 1000;
    int maxDepth = 40;
    int timeBudgetMs = 100;      // orçamento de tempo por lance na busca
    int openingRandomPlies = 6;  // primeiros N lances sujeitos a ruído epsilon-greedy
    double epsilon = 0.25;       // probabilidade de lance aleatório dentro da janela de abertura
    int maxPlies = 300;          // corte de segurança (partidas que não terminam são descartadas)
    unsigned seed = 1;
    int numThreads = 0;          // 0 = usar hardware_concurrency()
};

struct SelfPlayStats {
    std::atomic<uint64_t> gamesPlayed{0};
    std::atomic<uint64_t> gamesDiscarded{0};  // não terminaram dentro de maxPlies
    std::atomic<uint64_t> positionsWritten{0};
    std::atomic<uint64_t> totalNodes{0};
};

// Joga uma partida completa contra si mesmo e devolve as amostras já
// rotuladas com o resultado final. Descarta a partida (vetor vazio) se
// não terminar dentro de maxPlies -- evita rótulo de resultado incorreto.
inline std::vector<TrainingSample> playOneGame(Negamax& engine, std::mt19937_64& rng,
                                                const SelfPlayConfig& cfg, uint64_t& nodesOut) {
    State s = initialState();
    std::vector<TrainingSample> samples;
    samples.reserve(cfg.maxPlies);
    std::uniform_real_distribution<double> unif(0.0, 1.0);
    nodesOut = 0;
    int ply = 0;
    for (; ply < cfg.maxPlies; ply++) {
        if (winner(s) != -1) break;
        auto moves = legalMoves(s);
        Move chosen;
        int searchScore = evalSimple(s, s.turn);  // sempre calculado: é barato e vira o alvo auxiliar
        bool randomMove = (ply < cfg.openingRandomPlies) && (unif(rng) < cfg.epsilon);
        if (randomMove) {
            std::uniform_int_distribution<size_t> pick(0, moves.size() - 1);
            chosen = moves[pick(rng)];
        } else {
            SearchStats st;
            chosen = engine.chooseMove(s, cfg.maxDepth, cfg.timeBudgetMs, st);
            nodesOut += st.nodes;
        }

        TrainingSample rec;
        int mover = s.turn, opp = 1 - s.turn;
        rec.ownPawn = s.pawn[mover];
        rec.oppPawn = s.pawn[opp];
        rec.wallsH = s.wallsH;
        rec.wallsV = s.wallsV;
        rec.wallsLeftOwn = s.wallsLeft[mover];
        rec.wallsLeftOpp = s.wallsLeft[opp];
        rec.searchScore = (int16_t)std::max(-30000, std::min(30000, searchScore));
        rec.gameResult = 0;  // preenchido abaixo, depois que a partida terminar
        rec.policyTarget = moveToPolicyIndex(chosen);
        // mesma BFS que evalSimple já calcula por dentro -- reaproveitável
        // porque shortestPathLen não aloca (arrays thread_local fixos), mas
        // recalculada explicitamente aqui em vez de extraída de evalSimple
        // pra manter os dois campos (own/opp) sem depender de refatorar a
        // assinatura dessa função só por causa do dataset.
        rec.ownDist = (uint8_t)std::min(255, shortestPathLen(s.wallsH, s.wallsV, s.pawn[mover], mover));
        rec.oppDist = (uint8_t)std::min(255, shortestPathLen(s.wallsH, s.wallsV, s.pawn[opp], opp));
        samples.push_back(rec);

        s = applyMove(s, chosen);
    }

    int w = winner(s);
    if (w == -1) { samples.clear(); return samples; }  // não terminou -> descarta

    // s.turn alterna estritamente a cada lance e a partida sempre começa com
    // turn=0 (initialState), então o mover da amostra i é simplesmente i%2.
    for (size_t i = 0; i < samples.size(); i++) {
        int moverOfSample = (int)(i % 2);
        samples[i].gameResult = (w == moverOfSample) ? 1 : -1;
    }
    return samples;
}

// Gera cfg.numGames partidas em paralelo (cfg.numThreads threads) e grava
// tudo em outputPath como um único arquivo binário de TrainingSample
// consecutivos. Cada thread acumula suas amostras em memória e escreve seu
// bloco sob mutex só quando uma partida termina -- I/O não é o gargalo
// (busca domina o tempo), então o mutex não vira ponto de contenção.
inline void runSelfPlay(const SelfPlayConfig& cfg, const std::string& outputPath, SelfPlayStats& stats) {
    int nThreads = cfg.numThreads > 0 ? cfg.numThreads
                                       : std::max(1u, std::thread::hardware_concurrency());
    FILE* f = std::fopen(outputPath.c_str(), "wb");
    if (!f) { std::fprintf(stderr, "erro: nao foi possivel abrir '%s' para escrita\n", outputPath.c_str()); return; }
    std::mutex fileMutex;

    std::atomic<int> nextGame{0};
    int totalGames = cfg.numGames;

    auto worker = [&](int threadIdx) {
        Negamax engine;  // TT própria por thread (evita contenção e é o padrão usual em self-play multi-thread)
        std::mt19937_64 rng(cfg.seed + 1000003ull * (unsigned)threadIdx);
        for (;;) {
            int g = nextGame.fetch_add(1);
            if (g >= totalGames) break;
            uint64_t nodes = 0;
            auto samples = playOneGame(engine, rng, cfg, nodes);
            stats.totalNodes += nodes;
            if (samples.empty()) { stats.gamesDiscarded++; stats.gamesPlayed++; continue; }
            {
                std::lock_guard<std::mutex> lock(fileMutex);
                std::fwrite(samples.data(), sizeof(TrainingSample), samples.size(), f);
            }
            stats.positionsWritten += samples.size();
            stats.gamesPlayed++;
        }
    };

    std::vector<std::thread> pool;
    for (int t = 0; t < nThreads; t++) pool.emplace_back(worker, t);
    for (auto& th : pool) th.join();
    std::fclose(f);
}

} // namespace qr
