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
#include <memory>
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
    int timeBudgetMs = 100;       // orçamento de tempo por lance na busca
    int openingRandomPlies  = 6;  // fase 1: lances 0..N1-1 com epsilon1 (lances óbvios, pouco ruído)
    double epsilon          = 0.05; // epsilon da fase 1
    int openingRandomPlies2 = 10; // fase 2: lances N1..N2-1 com epsilon2 (exploração pesada)
    double epsilon2         = 0.8;  // epsilon da fase 2
    double epsilonMidgame   = 0.02; // probabilidade de lance aleatório após a fase 2
    int maxPlies = 300;           // corte de segurança (partidas que não terminam são descartadas)
    unsigned seed = 1;
    int numThreads = 0;           // 0 = usar hardware_concurrency()
    // true (default) = as duas cores dividem uma única engine/TT dentro da
    // mesma partida (mais rápido -- metade da memória de TT por thread, e
    // aproveita transposições encontradas pelo lado oposto; é o padrão
    // usual em geração de dados de self-play e o objetivo aqui é
    // throughput). false = cada cor usa sua própria engine/TT isolada,
    // igual à arena (teste/arena_dual.cpp) -- útil quando o objetivo é
    // comparar a taxa de empate/comportamento do selfplay com o da arena,
    // não gerar dados de treino.
    bool sharedTT = true;
};

struct SelfPlayStats {
    std::atomic<uint64_t> gamesPlayed{0};
    std::atomic<uint64_t> gamesDiscarded{0};  // não terminaram dentro de maxPlies
    std::atomic<uint64_t> gamesDrawn{0};      // empates por repetição
    std::atomic<uint64_t> positionsWritten{0};
    std::atomic<uint64_t> totalNodes{0};
};

// Joga uma partida completa contra si mesmo e devolve as amostras já
// rotuladas com o resultado final. Descarta a partida (vetor vazio) se
// não terminar dentro de maxPlies -- evita rótulo de resultado incorreto.
inline std::vector<TrainingSample> playOneGame(Negamax& engine0, Negamax& engine1, std::mt19937_64& rng,
                                                const SelfPlayConfig& cfg, uint64_t& nodesOut,
                                                SelfPlayStats& stats) {
    State s = initialState();
    std::vector<TrainingSample> samples;
    samples.reserve(cfg.maxPlies);
    std::uniform_real_distribution<double> unif(0.0, 1.0);
    nodesOut = 0;
    int ply = 0;
    RepetitionTable reptbl;
    bool isDraw = false;

    for (; ply < cfg.maxPlies; ply++) {
        if (winner(s) != -1) break;

        // Tripla repetição: se a posição já foi vista 2 vezes no histórico,
        // com a visita atual ela ocorre pela 3ª vez -> empate.
        if (reptbl.count(s.hash) >= 2) {
            isDraw = true;
            break;
        }

        // Engine da vez: cada cor tem sua própria TT, isolada -- assim
        // como na arena (arena_dual.cpp/arena.cpp), e ao contrário do
        // engine único compartilhado que existia aqui antes. Isso evita
        // que a busca de um lado "vaze" para o outro via transposições
        // encontradas poucos lances antes pelo lado oposto -- efeito que
        // não existe numa partida real (dois processos independentes) e
        // que enviesava a comparação selfplay vs arena.
        Negamax& engine = (s.turn == 0) ? engine0 : engine1;

        auto moves = legalMoves(s);
        Move chosen;
        int searchScore = evalSimple(s, s.turn);  // sempre calculado: é barato e vira o alvo auxiliar
        bool randomMove = false;
        double epsNow;
        if (ply < cfg.openingRandomPlies) {
            // Fase 1: lances iniciais óbvios, pouco ruído
            epsNow = cfg.epsilon;
        } else if (ply < cfg.openingRandomPlies2) {
            // Fase 2: janela de exploração pesada
            epsNow = cfg.epsilon2;
        } else {
            // Midgame: ruído mínimo para quebrar loops simétricos
            epsNow = cfg.epsilonMidgame;
        }
        randomMove = (unif(rng) < epsNow);

        if (randomMove) {
            if (ply < cfg.openingRandomPlies2) {
                // Abertura (fase 1 ou 2): totalmente aleatória para criar novos cenários
                std::uniform_int_distribution<size_t> pick(0, moves.size() - 1);
                chosen = moves[pick(rng)];
            } else {
                // Meio/fim do jogo: escolhe o 2º ou 3º melhor lance via busca rasa (depth=2)
                struct ScoredMove {
                    Move m;
                    int score;
                };
                std::vector<ScoredMove> scoredMoves;
                scoredMoves.reserve(moves.size());

                SearchStats dummyStats;
                for (size_t i = 0; i < moves.size(); i++) {
                    const auto& m = moves[i];
                    State ns = applyMove(s, m);
                    reptbl.push(ns.hash);
                    // Busca rasa do ponto de vista do oponente, então negamos o score
                    int score = -engine.searchShallow(ns, 2, dummyStats);
                    reptbl.pop();
                    scoredMoves.push_back({m, score});
                }

                // Ordena decrescente pelo score
                std::sort(scoredMoves.begin(), scoredMoves.end(), [](const ScoredMove& a, const ScoredMove& b) {
                    return a.score > b.score;
                });

                if (scoredMoves.size() <= 2) {
                    chosen = scoredMoves[0].m;
                } else {
                    // Escolhe entre o 2º (index 1) ou 3º (index 2) melhor lance
                    std::uniform_int_distribution<size_t> pick(1, std::min<size_t>(2, scoredMoves.size() - 1));
                    chosen = scoredMoves[pick(rng)].m;
                }
            }
        } else {
            SearchStats st;
            chosen = engine.chooseMove(s, cfg.maxDepth, cfg.timeBudgetMs, st, reptbl);
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
        rec.ownDist = (uint8_t)std::min(255, shortestPathLen(s.wallsH, s.wallsV, s.pawn[mover], mover));
        rec.oppDist = (uint8_t)std::min(255, shortestPathLen(s.wallsH, s.wallsV, s.pawn[opp], opp));
        samples.push_back(rec);

        reptbl.push(s.hash);
        s = applyMove(s, chosen);
    }

    if (isDraw) {
        // Empate por repetição: resultado = 0
        for (size_t i = 0; i < samples.size(); i++) {
            samples[i].gameResult = 0;
        }
        stats.gamesDrawn++;
        return samples;
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
        Negamax engine0;
        // Só aloca a 2ª TT (custosa: 2M entradas) quando de fato vamos
        // usá-la; no modo default (sharedTT=true) as duas cores usam
        // engine0 e a alocação extra seria desperdício de memória.
        std::unique_ptr<Negamax> engine1Storage;
        if (!cfg.sharedTT) engine1Storage = std::make_unique<Negamax>();
        Negamax& engine1 = cfg.sharedTT ? engine0 : *engine1Storage;

        std::mt19937_64 rng(cfg.seed + 1000003ull * (unsigned)threadIdx);
        for (;;) {
            int g = nextGame.fetch_add(1);
            if (g >= totalGames) break;
            uint64_t nodes = 0;
            // Limpa a TT antes de cada partida: scores de repetição são
            // path-dependent (dependem do histórico da partida atual), mas
            // a TT é indexada só pelo hash da posição. Se não limpar, uma
            // posição avaliada como empate em G1 contamina a busca de G2,
            // onde o mesmo hash é atingido sem repetição -- causando
            // aumento progressivo de empates conforme a TT se enche.
            engine0.clearTT();
            if (!cfg.sharedTT) engine1.clearTT();  // se compartilhada, já foi limpa acima (mesmo objeto)
            auto samples = playOneGame(engine0, engine1, rng, cfg, nodes, stats);
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
