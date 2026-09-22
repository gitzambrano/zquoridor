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
#include <cstdlib>
#include <cstring>
#include <vector>
#include <random>
#include <thread>
#include <mutex>
#include <atomic>
#include <algorithm>
#include <limits>
#include <string>
#include <memory>
#include <stdexcept>
#include "rules.hpp"
#include "search.hpp"
#include "seed_positions.hpp"
#include "selfplay_metadata.hpp"
// Modulo hibrido MCTS+alpha-beta (MCab, plan-hybrid-mc-ab.md, Fase 6) -- fica
// FORA da arvore versionada-por-ref (Secao 4.4 do plano): e sempre a versao
// do HEAD atual de tools/, igual a este proprio arquivo. Caminho relativo a
// partir de tools/selfplay/, mesmo padrao de tools/arena/arena.cpp.
#include "mcab.hpp"
// Overrides dos parametros de busca (contempt, LMR, CAT, quiescencia...) --
// mesma ideia de "vazio = valor de producao" do McabParams acima.
#include "search_tuning.hpp"

namespace qr {

// --- formato do registro de treino -------------------------------------
#pragma pack(push, 1)
// EV_SCALE: escala fixa de TrainingSample::evalNNUE -- ver nota completa
// no campo. Mesmo valor usado em selfplay.hpp/arena.cpp (gravação) e
// esperado por training/train_nnue.py (leitura); 65535 (uint16_t máximo)
// dá resolução de ~0,0000153 na probabilidade, bem além do que a NNUE
// consegue distinguir de verdade -- a escala em si não é sensível.
constexpr uint16_t EV_SCALE = 65535;
struct TrainingSample {
    // ATENÇÃO (mudança 2026-08, arquitetura de walls-restantes + correção
    // de perspectiva em nnue.hpp): ownPawn/oppPawn/wallsH/wallsV/
    // policyTarget abaixo são gravados JÁ ESPELHADOS pra perspectiva
    // canônica do mover (mesmo mirroredPawnCell/mirrorWallBitboard/
    // mirrorMoveForPerspective usados por buildAccumulator em nnue.hpp) --
    // NÃO são mais coordenada crua do tabuleiro. Datasets .bin gravados
    // ANTES desta mudança têm essas mesmas 4 colunas em coordenada crua
    // (sem espelho) e precisam ser regerados; não misturar os dois no
    // mesmo treino (sem marca de versão no arquivo pra detectar isso
    // automaticamente -- ver nota em train_nnue.py).
    uint8_t  ownPawn;       // célula (0..80) do peão de quem tem o lance ("mover"), já espelhada
    uint8_t  oppPawn;       // célula (0..80) do peão adversário, já espelhada
    uint64_t wallsH;        // bitboard de muros horizontais, já espelhado
    uint64_t wallsV;        // bitboard de muros verticais, já espelhado
    int8_t   wallsLeftOwn;  // muros restantes do mover (escalar -- não precisa de espelho)
    int8_t   wallsLeftOpp;  // muros restantes do adversário (escalar -- não precisa de espelho)
    // Avaliação da própria NNUE (2026-08, substitui o antigo searchScore
    // heurístico -- ver nota "Avaliação: o que cada estágio usa" em
    // docs/plan.md/CLAUDE.md). uint16_t in [0, EV_SCALE], with a fixed
    // EV_SCALE = 65535: 0 = vitória certa das PRETAS, EV_SCALE = vitória
    // certa das BRANCAS -- perspectiva ABSOLUTA de cor (não do mover, ao
    // contrário de gameResult abaixo), pra poder ser lida sem precisar de
    // `mover` sempre que só se quer "o placar do ponto de vista de quem
    // joga de brancas". train_nnue.py reconverte pra perspectiva do mover
    // (usando o campo `mover` abaixo) na hora de misturar com o resultado
    // real do jogo (WL_mod = WL*k + EV*(1-k), ver DATA_SOURCES_DEFAULT).
    // Calculado com a NNUE quantizada carregada nesta execução de
    // selfplay (buildAccumulatorQuant + forwardValueWLQuant, sigmoid do
    // logit -- ver nnueWinProbQuant em nnue.hpp); se NNUE não está
    // carregada (pesos zerados), o forward degrada sozinho pra logit 0 ->
    // sigmoid 0.5 -> EV_SCALE/2 (neutro), sem caso especial nenhum aqui.
    // .bin antigos (gravados antes desta mudança) têm um int16_t heurístico
    // neste mesmo offset/tamanho -- continuam lidos sem erro (mesmos 2
    // bytes), mas o valor não tem esse significado; treine com k=1.0 pra
    // essa fonte em DATA_SOURCES_DEFAULT, o que zera o termo EV e ignora
    // esse conteúdo por completo, seja lá o que ele signifique.
    uint16_t evalNNUE;
    int8_t   gameResult;    // +1 se o mover desta amostra venceu a partida, -1 se perdeu
    uint16_t policyTarget;  // índice do lance jogado (0..208), já espelhado -- ver mirrorMoveForPerspective
    // Distância BFS (shortestPathLen, rules.hpp) até a linha de chegada,
    // já calculada aqui (não derivada em Python) por um motivo específico:
    // "de quem é o lance" (mover) é a única informação que dá o índice de
    // jogador (0/1) certo pra saber qual GOAL_ROW usar -- e essa
    // informação NÃO sobrevive no arquivo (o registro guarda só a célula
    // absoluta -- espelhada -- do peão, não o índice de jogador). Calcular
    // aqui, onde `mover`/`opp` ainda são conhecidos com certeza, evita ter
    // que reconstruir a identidade do jogador a partir de heurísticas de
    // posição no lado Python (frágil: um peão pode legalmente recuar).
    // uint8_t comporta até 255, bem acima de qualquer distância possível
    // no tabuleiro 9x9 (máximo teórico bem menor); nunca precisa de clamp
    // aqui (isso só acontece no lado do bucket, em nnue.hpp/nnue.py).
    uint8_t  ownDist;       // shortestPathLen do peão do mover até a meta dele (escalar -- não precisa de espelho)
    uint8_t  oppDist;       // shortestPathLen do peão do oponente até a meta dele (escalar -- não precisa de espelho)
    // Campos novos (2026-08, mesma leva de mudança de formato do espelho
    // de perspectiva -- ver nota no topo do struct):
    uint8_t  mover;         // 0 ou 1 -- identidade FÍSICA de quem tinha o lance (não confundir com
                             // "own/opp": own/opp são sempre relativos ao mover; `mover` é o único
                             // campo que diz qual jogador REAL isso era, útil pra reconstruir o
                             // tabuleiro de verdade fora do referencial espelhado (debug/visualização/
                             // qualquer ferramenta externa) ou pra desfazer o espelho do policyTarget
                             // se um dia o policy head for consumido na busca (ver nota em
                             // mirrorMoveForPerspective, nnue.hpp).
    // CAT (Corridor Attention Table, cat.hpp) -- calor de corredor somado
    // sobre as 81 casas, por lado, na topologia de muro DESTA posição
    // (antes do lance ser jogado). NÃO é feature de entrada da NNUE ainda
    // -- gravado só como candidato pra uma rodada futura de features (ver
    // plano-additional.md): é um resumo compacto de "quanto do tabuleiro
    // está corredor-crítico" pra cada lado, complementar a ownDist/oppDist
    // (que só dizem a distância final, não quantas casas estão perto da
    // rota ótima). Calculado com computeCorridorHeat(wallsH, wallsV,
    // pawn[player], player) -- mesma função que orderWallMoves já usa pra
    // ordenar candidatos de muro na busca, chamada de novo aqui (custo
    // ~2 BFS, mesma ordem de grandeza de ownDist/oppDist, não persiste
    // entre a chamada da busca e a da gravação). int16_t comporta
    // folgadamente o máximo teórico (CAT_CORRIDOR_CM=200 * 81 casas =
    // 16200, bem abaixo de 32767) sem overflow nem precisar de clamp.
    int16_t  ownCatTotal;   // soma do calor de corredor do mover (perspectiva própria, já natural -- sem espelho a fazer: é escalar)
    int16_t  oppCatTotal;   // soma do calor de corredor do oponente
    uint16_t policyTopIdx[8];   // top-8 policy indices by MCAB root visits, mirrored
    uint16_t policyTopProb[8];  // top-8 visit probabilities, normalized to sum 65535
};
#pragma pack(pop)
static_assert(sizeof(TrainingSample) == 64,
    "TrainingSample must stay packed; Python reads the 64-byte layout directly");

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
    // Optional nonterminal positions used as game roots. The CLI loads these
    // from zquoridor.position.v1 JSONL through --positions. An empty vector
    // preserves standard self-play from the initial position.
    std::vector<SelfPlaySeed> startPositions;
    // true (default) = as duas cores dividem uma única engine/TT dentro da
    // mesma partida (mais rápido -- metade da memória de TT por thread, e
    // aproveita transposições encontradas pelo lado oposto; é o padrão
    // usual em geração de dados de self-play e o objetivo aqui é
    // throughput). false = cada cor usa sua própria engine/TT isolada,
    // igual à arena (teste/arena_dual.cpp) -- útil quando o objetivo é
    // comparar a taxa de empate/comportamento do selfplay com o da arena,
    // não gerar dados de treino.
    bool sharedTT = true;
    // Caminho para os pesos NNUE quantizados (.qbin/.bin gerado por
    // quantize_nnue.py). NNUE é o default de avaliação de folha deste
    // binário: nnueWeightsPath já nasce apontando para
    // defaultNnueWeightsPath() (data/nnue/nnue_weights_int8.bin) e
    // runSelfPlay tenta carregá-lo automaticamente antes de spawnar as
    // threads. Se esse arquivo não existir e o caminho não foi passado
    // explicitamente via --nnue-weights (ver nnueWeightsExplicit), o
    // fallback é silencioso: cai para evalSimple (heurístico) com um aviso
    // no stderr. Se o caminho FOI passado explicitamente e falha ao
    // carregar, é tratado como erro do usuário (aviso mais enfático, mas
    // ainda sem travar o processo -- ver runSelfPlay).
    std::string nnueWeightsPath = defaultNnueWeightsPath();
    // true se --nnue-weights foi passado explicitamente na linha de
    // comando (selfplay_main.cpp); false quando nnueWeightsPath ainda é
    // só o default automático. Só afeta a mensagem/severidade do log
    // quando o load falha, não o comportamento de fallback em si.
    bool nnueWeightsExplicit = false;
    // Força avaliação heurística (evalSimple) mesmo que pesos NNUE
    // existam/carreguem com sucesso. Uso: debug, comparação histórica com
    // versões pré-NNUE, ou fallback manual caso a NNUE se comporte mal.
    // Equivale a --heuristic no selfplay_main.
    bool forceHeuristic = false;
    // Liga a ordenacao assistida por politica (prompt_policy_ordering.md,
    // Negamax::setPolicyOrderingEnabled) em toda engine que usar NNUE
    // (useNNUE==true) -- só tem efeito quando NNUE está de fato ativo (ver
    // runSelfPlay); em heurística é ignorado sem aviso, já que a flag não
    // existe pra fazer sentido lá. Equivale a --policy-order no
    // selfplay_main. Default true desde 2026-08 (era false) -- mesmo
    // default de search.hpp/arena.cpp/wasm agora: NNUE default -> policy
    // head default, mesmo sem nenhum parametro passado. Shards gerados
    // antes dessa mudanca NAO sao reprodutiveis bit-a-bit com o default
    // atual; passe --no-policy-order pra voltar ao comportamento antigo.
    bool policyOrderingEnabled = true;
    // Piso de profundidade (search.hpp: Negamax::setPolicyOrderingMinDepth)
    // -- mesmo mecanismo/motivo do arena.cpp: forwardPolicyQuant é ~5.8x
    // mais caro que o eval de folha; sem piso ele roda em todo nó interno
    // e derruba nós/s ~3x (medido em produção). Default 3, sobreponivel
    // via --policy-order-min-depth.
    int policyOrderingMinDepth = 3;

    // =====================================================================
    // Modo Monte Carlo / temperatura (opção A MAIS, 2026-08) -- alternativa
    // ao modo epsilon-greedy acima (que continua existindo intacto: todos
    // os campos epsilon*/openingRandomPlies* acima seguem valendo quando
    // mcMode==false, que é o default -- nada do comportamento antigo muda).
    //
    // AlphaZero-style root-visit temperature. Each temperature ply first
    // runs the normal MCAB search, then samples its root visit distribution.
    // The stored policy target remains the raw, untempered root distribution.
    //
    //   1) "óbvios" (ply 0..mcObviousPlies-1): temperatura BAIXA e
    //      constante (mcTemperatureObvious) -- os 2-3 primeiros lances do
    //      Quoridor são essencialmente forçados/óbvios (não faz sentido
    //      sortear com a mesma variância que o resto da abertura; softmax
    //      quase-argmax aqui ainda dá alguma variedade sem jogar lances
    //      ruins de propósito).
    //   2) "opening" (mcObviousPlies..mcObviousPlies+mcTempDecayPlies-1):
    //      temperatura decai linearmente de mcTemperatureOpening (alta,
    //      quase uniforme -> muita variedade) até mcTemperatureEnd (baixa,
    //      quase argmax) ao longo de mcTempDecayPlies lances -- mesmo
    //      espírito da temperatura de AlphaZero (1.0 nos primeiros N
    //      lances, depois ~0).
    //   3) "meio/fim de jogo" (dali em diante): cai no MESMO fallback
    //      residual do modo antigo (epsilonMidgame -> 2º/3º melhor lance
    //      via busca rasa) ou busca completa (chooseMove) -- ou seja, o
    //      pós-abertura das duas partidas é gerado do mesmo jeito; só a
    //      ABERTURA muda de mecanismo entre os dois modos.
    //
    // Temperature changes only the selected move. Search budget, value
    // evaluation, and recorded visit counts remain the production settings.
    bool mcMode = false;              // false (default) = modo antigo, epsilon-greedy. true = modo Monte Carlo/temperatura.
    int    mcObviousPlies       = 3;    // fase 1: nº de lances iniciais (a partir do ply 0) com temperatura fixa baixa (lances "óbvios" do Quoridor)
    double mcTemperatureObvious = 0.15; // temperatura da fase 1 (baixa -> quase argmax, pouca variância nos lances óbvios)
    double mcTemperatureOpening = 1.35; // Phase-2 root-visit temperature; >1 adds exploration.
    double mcTemperatureEnd     = 0.12; // temperatura ao final da fase 2 (<1 afia a softmax -> quase argmax)
    int    mcTempDecayPlies     = 20;   // nº de lances da fase 2 (logo após mcObviousPlies) sobre os quais a temperatura decai linearmente de mcTemperatureOpening a mcTemperatureEnd
    // epsilonMidgame (campo antigo, acima) é reaproveitado como ruído
    // residual do modo MC após a fase 2 -- não duplicamos o conceito; dá
    // pra tunar os dois modos com os mesmos --epsilon-midgame.

    // =====================================================================
    // MCab (híbrido MCTS+alpha-beta, plan-hybrid-mc-ab.md, Fase 6) -- só
    // afeta o(s) lance(s) escolhido(s) pela busca "de verdade" (o ramo
    // engine.chooseMove(...) de cada modo acima, agora trocado por
    // mcabRunner.choose(...), ver playOneGame). Default = McabParams{}
    // default (enabled=false), ou seja, AB puro, byte-compatível com o
    // comportamento anterior a esta fase. NÃO afeta chooseShallowRunnerUp
    // nem sampleMoveByPolicyTemperature -- esses caminhos continuam
    // exatamente como estão (ver instruções da Fase 6).
    mcab::McabParams mcabParams;

    // KataGo/Claustrophobia-style playout-cap randomization for datagen.
    // Default OFF preserves the historical generator exactly: every ply is
    // recorded and every real search uses timeBudgetMs.
    //
    // When enabled, only a Bernoulli fraction fullSearchProb of plies that
    // reach the real search path use the full time budget and become training
    // samples. The remaining searched plies use cheapTimeBudgetMs only to steer
    // the trajectory and are NOT written. Temperature/random/shallow plies are
    // likewise trajectory-only while this mode is enabled. This spends teacher
    // compute on high-quality targets instead of paying full price everywhere.
    bool playoutCapEnabled = false;
    double fullSearchProb = 1.0;
    int cheapTimeBudgetMs = 20;

    // Optional aligned sidecar for TrainingSample records.
    std::string metadataOutputPath;
    uint8_t metadataSourceClass = 0;
    uint64_t metadataGameIdBase = 0;

    // Overrides dos parâmetros de busca de search.hpp (contempt, LMR, CAT,
    // quiescência, escala da política na ordenação...). Todo campo vazio por
    // default => applySearchTuning é um no-op e cada engine fica no valor de
    // produção. Preenchido pelo bloco de config/flags de selfplay_main.cpp.
    tuning::SearchTuning tuning;
};

struct SelfPlayStats {
    std::atomic<uint64_t> gamesPlayed{0};
    std::atomic<uint64_t> gamesDiscarded{0};  // não terminaram dentro de maxPlies
    std::atomic<uint64_t> gamesDrawn{0};      // empates por repetição
    std::atomic<uint64_t> positionsWritten{0};
    std::atomic<uint64_t> totalNodes{0};
    std::atomic<uint64_t> fullSearchPlies{0};
    std::atomic<uint64_t> cheapSearchPlies{0};
    std::atomic<uint64_t> samplesSkipped{0};
};

// Alias de instanciação do McabRunner (Fase 6 do plano) para a engine única
// deste binário (sem o truque de dual-namespace qr_e1/qr_e2 do arena.cpp --
// aqui só existe `qr::Negamax`). McabRunner::choose() cai sozinho em
// eng.chooseMove(...) quando params().enabled == false (o default), então
// os call sites em playOneGame podem chamá-lo incondicionalmente sem custo
// extra perceptível no caminho AB puro.
using McabRunnerT = mcab::McabRunner<Negamax, State, Move, MoveList, AccPair,
                                      RepetitionTable, SearchStats>;

// Amostra um lance de uma distribuição softmax(logit/temperatura) sobre os
// logits crus da cabeça de política da NNUE (policyOut, já calculado pelo
// chamador via forwardPolicyQuant -- não recomputa nada aqui). Estilo
// "move temperature" do AlphaZero: temperatura alta achata a distribuição
// (quase uniforme sobre os lances legais -- muita exploração); temperatura
// baixa afia (aproxima do argmax -- quase o lance que a política prefere).
// Não faz nenhuma busca: custo O(nº de lances legais), a mesma ordem de
// grandeza de gerar os lances legais em si, não da busca alfa-beta.
inline Move sampleMoveByPolicyTemperature(const std::array<float, POLICY_OUT>& policyOut,
                                           const MoveList& moves, int side,
                                           double temperature, std::mt19937_64& rng) {
    // Piso de temperatura: evita divisão por ~0 (softmax degenerando em
    // NaN/Inf quando mcTemperatureEnd for configurada muito perto de 0).
    // Abaixo deste piso o comportamento já é argmax para qualquer efeito
    // prático (exp((logit-max)/1e-3) satura em 0 ou 1).
    constexpr double MIN_TEMP = 1e-3;
    double t = std::max(temperature, MIN_TEMP);

    // Softmax numericamente estável: subtrai o logit máximo antes do exp
    // (evita overflow; não muda a distribuição resultante).
    // Buffer de tamanho fixo na stack: evita alocação na heap e thread_local
    // com destruidor dinâmico (que causa STATUS_HEAP_CORRUPTION no MinGW no exit das threads).
    std::array<double, POLICY_OUT> weights{};
    size_t nMoves = moves.size();
    if (nMoves == 0) return Move::pawn(0); // defensivo

    double maxLogit = -1e300;
    for (size_t i = 0; i < nMoves; i++) {
        double logit = (double)policyLogitForMove(policyOut, moves[i], side);
        weights[i] = logit;  // guarda o logit cru temporariamente
        maxLogit = std::max(maxLogit, logit);
    }
    double sum = 0.0;
    for (size_t i = 0; i < nMoves; i++) {
        weights[i] = std::exp((weights[i] - maxLogit) / t);
        sum += weights[i];
    }
    // sum > 0 sempre: pelo menos um termo vale exp(0)=1 (o do logit máximo).
    std::uniform_real_distribution<double> unif(0.0, sum);
    double r = unif(rng);
    double acc = 0.0;
    for (size_t i = 0; i < nMoves; i++) {
        acc += weights[i];
        if (r <= acc) return moves[i];
    }
    return moves[nMoves - 1];  // fallback de arredondamento de ponto flutuante (r ligeiramente > sum)
}

// Sample a root move from the MCAB visit distribution. The self-play policy
// target remains the untempered visit distribution stored below.
template <typename RootNode>
inline Move sampleMoveByVisitTemperature(const RootNode& root, double temperature,
                                         std::mt19937_64& rng) {
    constexpr double MIN_TEMP = 1e-3;
    const size_t nMoves = std::min(root.moves.size(), root.N.size());
    if (nMoves == 0) return Move::pawn(0);

    const double inverseTemperature = 1.0 / std::max(temperature, MIN_TEMP);
    std::array<double, POLICY_OUT> logWeights{};
    double maxLogWeight = -std::numeric_limits<double>::infinity();
    for (size_t i = 0; i < nMoves; ++i) {
        const double visits = static_cast<double>(root.N[i]);
        if (!(visits > 0.0) || !std::isfinite(visits)) continue;
        logWeights[i] = std::log(visits) * inverseTemperature;
        maxLogWeight = std::max(maxLogWeight, logWeights[i]);
    }
    if (!std::isfinite(maxLogWeight)) return root.moves[0];

    double sum = 0.0;
    for (size_t i = 0; i < nMoves; ++i) {
        const double visits = static_cast<double>(root.N[i]);
        if (!(visits > 0.0) || !std::isfinite(visits)) continue;
        logWeights[i] = std::exp(logWeights[i] - maxLogWeight);
        sum += logWeights[i];
    }
    std::uniform_real_distribution<double> pick(0.0, sum);
    const double target = pick(rng);
    double cumulative = 0.0;
    for (size_t i = 0; i < nMoves; ++i) {
        cumulative += logWeights[i];
        if (target <= cumulative) return root.moves[i];
    }
    return root.moves[nMoves - 1];
}

// Escolhe o 2º ou 3º melhor lance (empate: 50/50) via busca rasa depth=2
// sobre todos os lances legais -- mesmo mecanismo usado pelo ramo
// epsilonMidgame do modo antigo (fatorado aqui pra ser reaproveitado
// também pelo ruído residual pós-decaimento do modo Monte Carlo).
inline Move chooseShallowRunnerUp(Negamax& engine, const State& s, const MoveList& moves,
                                   RepetitionTable& reptbl, std::mt19937_64& rng) {
    struct ScoredMove { Move m; int score; };
    size_t nMoves = moves.size();
    if (nMoves == 0) return Move::pawn(0);

    std::array<ScoredMove, POLICY_OUT> scoredMoves{};

    SearchStats dummyStats;
    for (size_t i = 0; i < nMoves; i++) {
        const auto& m = moves[i];
        State ns = applyMove(s, m);
        reptbl.push(ns.hash, m.isWall);
        // Busca rasa do ponto de vista do oponente, então negamos o score
        int score = -engine.searchShallow(ns, 2, dummyStats);
        reptbl.pop();
        scoredMoves[i] = {m, score};
    }

    std::sort(scoredMoves.begin(), scoredMoves.begin() + nMoves, [](const ScoredMove& a, const ScoredMove& b) {
        return a.score > b.score;
    });

    if (nMoves <= 2) return scoredMoves[0].m;
    std::uniform_int_distribution<size_t> pick(1, std::min<size_t>(2, nMoves - 1));
    return scoredMoves[pick(rng)].m;
}

// Joga uma partida completa contra si mesmo e devolve as amostras já
// rotuladas com o resultado final. Descarta a partida (vetor vazio) se
// não terminar dentro de maxPlies -- evita rótulo de resultado incorreto.
inline std::vector<TrainingSample> playOneGame(Negamax& engine0, Negamax& engine1, std::mt19937_64& rng,
                                                const SelfPlayConfig& cfg, uint64_t& nodesOut,
                                                SelfPlayStats& stats,
                                                const SelfPlaySeed* start = nullptr,
                                                std::vector<TrainingMetaV1>* metadataOut = nullptr,
                                                uint64_t gameId = 0) {
    State s = start ? start->state : initialState();
    std::vector<TrainingSample> samples;
    samples.reserve(cfg.maxPlies);
    std::vector<TrainingMetaV1> metadata;
    std::vector<uint16_t> samplePlies;
    if (metadataOut) {
        metadata.reserve(cfg.maxPlies);
        samplePlies.reserve(cfg.maxPlies);
    }
    std::uniform_real_distribution<double> unif(0.0, 1.0);
    nodesOut = 0;
    int ply = 0;
    RepetitionTable reptbl = start ? start->repetition : RepetitionTable{};
    bool isDraw = false;

    // MCab (Fase 6 do plano): UMA instância de McabRunner por PARTIDA (não
    // por lance), viva pela função inteira -- necessário para o reuso de
    // subárvore entre lances (Seção 8 do plano) funcionar. Compartilhada
    // entre as duas cores (engine0/engine1 abaixo): a árvore rastreia a
    // sequência REAL de posições jogadas nesta partida (s -> s' -> s'' ...),
    // independente de qual objeto Negamax (TT) gera cada folha -- mesmo
    // princípio do runner único por engine em arena.cpp (lá há 2 runners
    // porque há 2 tipos de engine distintos por ref; aqui há só 1 tipo).
    McabRunnerT mcabRunner;
    mcabRunner.setParams(cfg.mcabParams);
    mcabRunner.resetTree();
    // Semente de ruído de raiz (rootNoiseSeed): precisa VARIAR por
    // thread/partida, senão todas as threads/partidas gerariam a MESMA
    // sequência de ruído de Dirichlet nos priors da raiz (McabParams
    // nasce com uma semente fixa, 0x9E3779B9) e a diversidade de abertura
    // que o ruído existe para dar desapareceria -- todas as partidas
    // convergiriam para as mesmas primeiras jogadas sempre que
    // mcabRootNoiseEnabled estiver ligado. Derivada de `rng` (já é o RNG
    // por-thread desta partida, semeado em runSelfPlay como
    // cfg.seed + 1000003*threadIdx -- ver worker()), então cada chamada de
    // playOneGame consome um valor diferente e imprevisível da mesma
    // sequência já usada para o resto das decisões aleatórias da partida.
    mcabRunner.seedNoise((uint32_t)rng());

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
        // Avaliação da própria NNUE nesta posição, ANTES do lance, na
        // perspectiva ABSOLUTA das brancas (ver nota completa em
        // TrainingSample::evalNNUE acima) -- sempre calculada (barato: um
        // forward da cabeça WL, mesmo custo de nnueEvalInt) e independente
        // do lance escolhido abaixo (aleatório, raso ou busca completa),
        // ao contrário do antigo searchScore que só existia pra virar alvo
        // auxiliar de treino.
        double evalWhiteProb;
        int mcTemperatureWindow = cfg.mcObviousPlies + cfg.mcTempDecayPlies;
        bool mcTemperaturePly = cfg.mcMode && ply < mcTemperatureWindow;
        {
            AccumulatorQuant accMover = buildAccumulatorQuant(s, s.turn);
            double probMoverWins = (double)nnueWinProbQuant(accMover);
            evalWhiteProb = (s.turn == 0) ? probMoverWins : (1.0 - probMoverWins);
        }

        // Playout-cap bookkeeping. In legacy mode every ply is still
        // recorded exactly as before. In playout-cap mode only a full real
        // search may set recordSample=true.
        bool recordSample = !cfg.playoutCapEnabled;

        auto searchedMove = [&](bool fullSearch) {
            SearchStats st;
            int budgetMs = fullSearch ? cfg.timeBudgetMs : cfg.cheapTimeBudgetMs;
            if (budgetMs <= 0) budgetMs = cfg.timeBudgetMs;
            Move m = mcabRunner.choose(engine, s, cfg.maxDepth, budgetMs, st, reptbl);
            nodesOut += st.nodes;
            if (cfg.playoutCapEnabled) {
                if (fullSearch) stats.fullSearchPlies++;
                else stats.cheapSearchPlies++;
            }
            return m;
        };

        if (cfg.mcMode) {
            // --- Modo Monte Carlo / temperatura --------------------------
            if (mcTemperaturePly) {
                double temperature;
                if (ply < cfg.mcObviousPlies) {
                    // Fase 1: lances iniciais óbvios do Quoridor -- temperatura
                    // baixa e constante (pouca variância de propósito).
                    temperature = cfg.mcTemperatureObvious;
                } else {
                    // Fase 2: decaimento linear de mcTemperatureOpening (no
                    // primeiro ply desta fase) até mcTemperatureEnd (no
                    // último), estilo AlphaZero. mcTempDecayPlies<=1 usa
                    // direto mcTemperatureOpening.
                    int decayPly = ply - cfg.mcObviousPlies;
                    double frac = (cfg.mcTempDecayPlies > 1)
                        ? (double)decayPly / (double)(cfg.mcTempDecayPlies - 1)
                        : 0.0;
                    temperature = cfg.mcTemperatureOpening +
                        frac * (cfg.mcTemperatureEnd - cfg.mcTemperatureOpening);
                }
                // Run the usual MCAB search. Then sample from the root
                // visits. This matches the AlphaZero self-play convention.
                chosen = searchedMove(true);
                const auto* rootNode = mcabRunner.search.rootNodeForInspection();
                if (rootNode && rootNode->expanded && rootNode->state.hash == s.hash) {
                    chosen = sampleMoveByVisitTemperature(*rootNode, temperature, rng);
                }
            } else if (unif(rng) < cfg.epsilonMidgame) {
                // Mesmo ruído residual do modo antigo, reaproveitado aqui
                // pra manter alguma variedade depois que a temperatura já
                // decaiu (quebra loops simétricos no meio/fim de jogo).
                chosen = chooseShallowRunnerUp(engine, s, moves, reptbl, rng);
            } else {
                bool fullSearch = !cfg.playoutCapEnabled || (unif(rng) < cfg.fullSearchProb);
                chosen = searchedMove(fullSearch);
                recordSample = !cfg.playoutCapEnabled || fullSearch;
            }
        } else {
            // --- Modo antigo (epsilon-greedy, inalterado) -----------------
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
            bool randomMove = (unif(rng) < epsNow);

            if (randomMove) {
                if (ply < cfg.openingRandomPlies2) {
                    // Abertura (fase 1 ou 2): totalmente aleatória para criar novos cenários
                    std::uniform_int_distribution<size_t> pick(0, moves.size() - 1);
                    chosen = moves[pick(rng)];
                } else {
                    // Meio/fim do jogo: escolhe o 2º ou 3º melhor lance via busca rasa (depth=2)
                    chosen = chooseShallowRunnerUp(engine, s, moves, reptbl, rng);
                }
            } else {
                bool fullSearch = !cfg.playoutCapEnabled || (unif(rng) < cfg.fullSearchProb);
                chosen = searchedMove(fullSearch);
                recordSample = !cfg.playoutCapEnabled || fullSearch;
            }
        }

        // Tree reuse and graph search must never be allowed to apply a move
        // that is not legal in the current root. Treat such a stale result as
        // a teacher miss and continue with a legal fallback instead of
        // corrupting the state (for example, decrementing walls below zero).
        bool chosenLegal = false;
        for (const Move& legal : moves) {
            if (legal == chosen) {
                chosenLegal = true;
                break;
            }
        }
        if (!chosenLegal) {
            chosen = moves[0];
            recordSample = false;
            if (cfg.playoutCapEnabled) stats.samplesSkipped++;
        }

        if (!recordSample && cfg.playoutCapEnabled) stats.samplesSkipped++;

        if (recordSample) {
        TrainingSample rec{};
        int mover = s.turn, opp = 1 - s.turn;
        // Gravar já espelhado (ver nota em TrainingSample acima e em
        // mirroredPawnCell/mirrorWallBitboard/mirrorMoveForPerspective,
        // nnue.hpp): ownDist/oppDist abaixo continuam usando s.wallsH/
        // s.wallsV CRUS (shortestPathLen opera em coordenada real de
        // tabuleiro; o "mover"/"opp" já bastam pra saber o GOAL_ROW certo,
        // não precisam do espelho) -- só os campos gravados em `rec` é que
        // vão espelhados.
        rec.ownPawn = (uint8_t)mirroredPawnCell(s.pawn[mover], mover);
        rec.oppPawn = (uint8_t)mirroredPawnCell(s.pawn[opp], mover);
        rec.wallsH = mirrorWallBitboard(s.wallsH, mover);
        rec.wallsV = mirrorWallBitboard(s.wallsV, mover);
        rec.wallsLeftOwn = s.wallsLeft[mover];
        rec.wallsLeftOpp = s.wallsLeft[opp];
        rec.evalNNUE = (uint16_t)std::lround(std::max(0.0, std::min(1.0, evalWhiteProb)) * (double)EV_SCALE);
        rec.gameResult = 0;  // preenchido abaixo, depois que a partida terminar
        rec.policyTarget = moveToPolicyIndex(mirrorMoveForPerspective(chosen, mover));
        // Preserve the information produced by the MCAB teacher instead of
        // collapsing ~20k simulations to one one-hot move label. A stale tree
        // is rejected by hash, which also masks epsilon/shallow and exact
        // no-wall fallback plies from the visit-policy loss.
        if (cfg.mcabParams.enabled) {
            const auto* rootNode = mcabRunner.search.rootNodeForInspection();
            if (rootNode && rootNode->expanded && rootNode->state.hash == s.hash) {
                struct VisitEdge { size_t edge; float visits; };
                std::vector<VisitEdge> ranked;
                size_t nm = std::min(rootNode->moves.size(), rootNode->N.size());
                ranked.reserve(nm);
                for (size_t i = 0; i < nm; ++i)
                    if (rootNode->N[i] > 0.f) ranked.push_back({i, rootNode->N[i]});
                std::stable_sort(ranked.begin(), ranked.end(),
                    [](const VisitEdge& a, const VisitEdge& b) { return a.visits > b.visits; });
                if (ranked.size() > 8) ranked.resize(8);
                double sumVisits = 0.0;
                for (const auto& e : ranked) sumVisits += (double)e.visits;
                uint32_t assigned = 0;
                for (size_t j = 0; j < ranked.size() && sumVisits > 0.0; ++j) {
                    const Move& vm = rootNode->moves[ranked[j].edge];
                    rec.policyTopIdx[j] = moveToPolicyIndex(mirrorMoveForPerspective(vm, mover));
                    uint16_t q = (uint16_t)std::floor(65535.0 * ranked[j].visits / sumVisits);
                    rec.policyTopProb[j] = q;
                    assigned += q;
                }
                if (!ranked.empty() && sumVisits > 0.0)
                    rec.policyTopProb[0] = (uint16_t)(rec.policyTopProb[0] + (65535u - assigned));
            }
        }
        rec.ownDist = (uint8_t)std::min(255, shortestPathLen(s.wallsH, s.wallsV, s.pawn[mover], mover));
        rec.oppDist = (uint8_t)std::min(255, shortestPathLen(s.wallsH, s.wallsV, s.pawn[opp], opp));
        rec.mover = (uint8_t)mover;
        {
            CorridorHeat ownHeat = computeCorridorHeat(s.wallsH, s.wallsV, s.pawn[mover], mover);
            CorridorHeat oppHeat = computeCorridorHeat(s.wallsH, s.wallsV, s.pawn[opp], opp);
            int ownSum = 0, oppSum = 0;
            for (int i = 0; i < N * N; i++) { ownSum += ownHeat.heat[i]; oppSum += oppHeat.heat[i]; }
            rec.ownCatTotal = (int16_t)ownSum;
            rec.oppCatTotal = (int16_t)oppSum;
        }
        samples.push_back(rec);
        if (metadataOut) {
            float rootValue = std::numeric_limits<float>::quiet_NaN();
            uint8_t qualityFlags = TRAINING_META_ROOT_VALUE_MISSING;
            if (cfg.mcabParams.enabled) {
                const auto* rootNode = mcabRunner.search.rootNodeForInspection();
                if (rootNode && rootNode->expanded && rootNode->state.hash == s.hash) {
                    double weighted = 0.0;
                    double totalVisits = 0.0;
                    for (size_t i = 0; i < rootNode->N.size(); ++i) {
                        if (rootNode->N[i] <= 0.0f) continue;
                        const float q = (cfg.mcabParams.backupMode == mcab::BackupMode::AvgBlend)
                            ? rootNode->W[i] / rootNode->N[i]
                            : rootNode->W[i];
                        if (!std::isfinite(q)) continue;
                        weighted += (double)rootNode->N[i] * (double)q;
                        totalVisits += (double)rootNode->N[i];
                    }
                    if (totalVisits > 0.0) {
                        rootValue = (float)std::max(0.0, std::min(1.0,
                            weighted / totalVisits));
                    }
                    if (std::isfinite(rootValue)) {
                        qualityFlags = TRAINING_META_ROOT_VALUE_VALID;
                    }
                }
            }
            metadata.push_back(makeTrainingMeta(gameId, rootValue,
                                                cfg.metadataSourceClass, qualityFlags));
            samplePlies.push_back((uint16_t)std::min(ply, (int)UINT16_MAX));
        }
        }

        reptbl.push(s.hash, chosen.isWall);
        s = applyMove(s, chosen);
    }

    const int gameStartPly = start ? start->repetition.size : 0;
    const int terminalPly = gameStartPly + ply;
    if (isDraw) {
        // Empate por repetição: resultado = 0
        for (size_t i = 0; i < samples.size(); i++) {
            samples[i].gameResult = 0;
        }
        stats.gamesDrawn++;
        if (metadataOut) {
            for (size_t i = 0; i < metadata.size(); ++i) {
                const int absoluteSamplePly = gameStartPly + (int)samplePlies[i];
                metadata[i].pliesToEnd = (uint16_t)std::min(
                    std::max(0, terminalPly - absoluteSamplePly), (int)UINT16_MAX);
                metadata[i].gameLength = (uint16_t)std::min(terminalPly, (int)UINT16_MAX);
            }
            *metadataOut = std::move(metadata);
        }
        return samples;
    }

    int w = winner(s);
    if (w == -1) { samples.clear(); return samples; }  // não terminou -> descarta

    // Samples may be sparse when playout-cap randomization is active, so
    // sample_index parity is NOT the game ply. The record already stores the
    // absolute mover; use it directly (also equivalent to i%2 in legacy mode).
    for (size_t i = 0; i < samples.size(); i++) {
        int moverOfSample = (int)samples[i].mover;
        samples[i].gameResult = (w == moverOfSample) ? 1 : -1;
    }
    if (metadataOut) {
        for (size_t i = 0; i < metadata.size(); ++i) {
            const int absoluteSamplePly = gameStartPly + (int)samplePlies[i];
            metadata[i].pliesToEnd = (uint16_t)std::min(
                std::max(0, terminalPly - absoluteSamplePly), (int)UINT16_MAX);
            metadata[i].gameLength = (uint16_t)std::min(terminalPly, (int)UINT16_MAX);
        }
        *metadataOut = std::move(metadata);
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

    // Carrega pesos NNUE uma única vez antes de spawnar os threads.
    // loadWeightsQuant escreve no singleton weightsQuant() (globals, não
    // por thread) -- seguro porque todos os threads só LEEM a partir daqui.
    // NNUE é o default: só pulamos o load se --heuristic foi pedido
    // explicitamente (cfg.forceHeuristic) ou se nnueWeightsPath ficou vazio
    // (defaultNnueWeightsPath() nunca é vazio, então isso só acontece se
    // alguém zerar o campo manualmente via API).
    bool useNNUE = !cfg.forceHeuristic && !cfg.nnueWeightsPath.empty();
    if (useNNUE) {
        if (!loadWeightsQuant(cfg.nnueWeightsPath)) {
            if (cfg.nnueWeightsExplicit) {
                std::fprintf(stderr, "[selfplay] ERRO: nao foi possivel carregar pesos NNUE de '%s'\n"
                                     "  Verifique o caminho passado em --nnue-weights.\n"
                                     "  Caindo para avaliacao heuristica (evalSimple) nesta execucao.\n",
                             cfg.nnueWeightsPath.c_str());
            } else {
                std::fprintf(stderr, "[selfplay] aviso: pesos NNUE default nao encontrados em '%s'\n"
                                     "  (rode training/quantize_nnue.py ou passe --nnue-weights <arquivo>)\n"
                                     "  Caindo para avaliacao heuristica (evalSimple) nesta execucao.\n",
                             cfg.nnueWeightsPath.c_str());
            }
            // Continua com heurístico em vez de travar o selfplay inteiro.
            // Seguro mesmo sem load: weightsQuant() nasce zerada (ver
            // NNUEWeightsQuant() em nnue.hpp), nunca com vetores vazios.
            useNNUE = false;
        } else {
            std::fprintf(stderr, "[selfplay] pesos NNUE carregados de '%s'\n", cfg.nnueWeightsPath.c_str());
        }
    } else if (cfg.forceHeuristic) {
        std::fprintf(stderr, "[selfplay] avaliacao heuristica forcada via --heuristic\n");
    }

    // MCab (Fase 6 do plano, Secao 2): o hibrido depende da cabeca de
    // politica da NNUE para os priors do PUCT -- em modo Heuristico nao ha
    // politica treinada nenhuma, e MCABSearch::chooseMoveMCAB tem um
    // `assert(engine.getEvalMode() == Eng::EvalMode::NNUE, ...)` interno
    // (mcab.hpp) que so dispara em build de debug (NDEBUG remove asserts em
    // release). Checagem explicita aqui cobre os DOIS jeitos de acabar em
    // heuristico: --heuristic explicito (cfg.forceHeuristic) OU fallback
    // automatico por falha ao carregar os pesos NNUE (useNNUE==false acima)
    // -- em qualquer um dos dois casos, se MCAB foi pedido, e um erro do
    // usuario, nao uma condicao silenciosa: erro claro + saida != 0 antes
    // de abrir o arquivo de saida (evita gerar um .bin vazio/parcial).
    if (cfg.mcabParams.enabled && !useNNUE) {
        std::fprintf(stderr,
            "[selfplay] ERRO: --mcab requer avaliacao NNUE ativa (Secao 2 do\n"
            "  plano MCab), mas esta execucao esta em modo heuristico (por\n"
            "  --heuristic ou por falha ao carregar os pesos NNUE, ver aviso\n"
            "  acima). Remova --mcab, remova --heuristic, ou corrija o\n"
            "  caminho de --nnue-weights.\n");
        std::exit(1);
    }

    FILE* metaFile = nullptr;
    if (!cfg.metadataOutputPath.empty()) {
        metaFile = std::fopen(cfg.metadataOutputPath.c_str(), "wb");
        if (!metaFile) {
            std::fprintf(stderr, "erro: nao foi possivel abrir '%s' para escrita\n",
                         cfg.metadataOutputPath.c_str());
            throw std::runtime_error("self-play metadata output open failed");
        }
    }
    FILE* f = std::fopen(outputPath.c_str(), "wb");
    if (!f) {
        std::fprintf(stderr, "erro: nao foi possivel abrir '%s' para escrita\n", outputPath.c_str());
        if (metaFile) {
            std::fclose(metaFile);
            std::remove(cfg.metadataOutputPath.c_str());
        }
        throw std::runtime_error("self-play V3 output open failed");
    }
    std::mutex fileMutex;
    std::atomic<bool> writeFailed{false};

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

        // Parâmetros de busca vindos do bloco de config/CLI (contempt, LMR,
        // CAT, quiescência...). Vazio = no-op, engine fica em produção.
        tuning::applySearchTuning(engine0, cfg.tuning);
        if (!cfg.sharedTT) tuning::applySearchTuning(engine1, cfg.tuning);

        // Ativa NNUE em todas as engines se pesos foram carregados.
        if (useNNUE) {
            engine0.setEvalMode(Negamax::EvalMode::NNUE);
            if (!cfg.sharedTT) engine1.setEvalMode(Negamax::EvalMode::NNUE);
            // Ordenacao assistida por politica (prompt_policy_ordering.md)
            // -- so faz sentido junto de NNUE (precisa do AccPair mantido
            // na pilha de busca pra tirar o forward pass, ver
            // policyOrderingEnabled em search.hpp), por isso fica dentro
            // do `if (useNNUE)`. Default cfg.policyOrderingEnabled=true
            // (2026-08): liga por default sempre que NNUE esta ativo;
            // use --no-policy-order pra reproduzir shards gerados antes
            // dessa mudanca.
            // Sempre chama o setter (não só quando ligado): search.hpp passou
            // a nascer com policyOrderingEnabled=true, então omitir a chamada
            // no caso "desligado" deixava --no-policy-order sem efeito nenhum.
            engine0.setPolicyOrderingEnabled(cfg.policyOrderingEnabled);
            if (!cfg.sharedTT) engine1.setPolicyOrderingEnabled(cfg.policyOrderingEnabled);
            if (cfg.policyOrderingEnabled) {
                engine0.setPolicyOrderingMinDepth(cfg.policyOrderingMinDepth);
                if (!cfg.sharedTT) engine1.setPolicyOrderingMinDepth(cfg.policyOrderingMinDepth);
            }
        }

        std::mt19937_64 rng(cfg.seed + 1000003ull * (unsigned)threadIdx);
        for (;;) {
            if (writeFailed.load()) break;
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
            const SelfPlaySeed* start = nullptr;
            if (!cfg.startPositions.empty()) {
                const uint64_t mixed = (uint64_t)cfg.seed
                    + (uint64_t)(unsigned)g * 0x9E3779B97F4A7C15ull;
                start = &cfg.startPositions[(size_t)(mixed % cfg.startPositions.size())];
            }
            std::vector<TrainingMetaV1> metadata;
            auto samples = playOneGame(engine0, engine1, rng, cfg, nodes, stats, start,
                                       metaFile ? &metadata : nullptr,
                                       cfg.metadataGameIdBase + (uint64_t)(unsigned)g);
            stats.totalNodes += nodes;
            if (samples.empty()) { stats.gamesDiscarded++; stats.gamesPlayed++; continue; }
            {
                std::lock_guard<std::mutex> lock(fileMutex);
                if (metaFile && metadata.size() != samples.size()) {
                    std::fprintf(stderr,
                                 "[selfplay] ERRO: metadata desalinhada (%zu de %zu amostras)\n",
                                 metadata.size(), samples.size());
                    writeFailed = true;
                    continue;
                }
                size_t written = std::fwrite(samples.data(), sizeof(TrainingSample), samples.size(), f);
                if (written != samples.size()) {
                    std::fprintf(stderr, "[selfplay] ERRO: fwrite escreveu %zu de %zu amostras\n",
                                 written, samples.size());
                    writeFailed = true;
                }
                if (metaFile && !writeFailed.load()) {
                    size_t metaWritten = std::fwrite(metadata.data(), sizeof(TrainingMetaV1),
                                                     metadata.size(), metaFile);
                    if (metaWritten != metadata.size()) {
                        std::fprintf(stderr,
                                     "[selfplay] ERRO: fwrite escreveu %zu de %zu metadados\n",
                                     metaWritten, metadata.size());
                        writeFailed = true;
                    }
                }
                // fflush garante que os dados chegam ao SO antes de continuar
                // -- em particular, o último jogo do chunk não fica em buffer
                // quando o processo encerra logo depois do ultimo fwrite.
                std::fflush(f);
                if (metaFile) std::fflush(metaFile);
            }
            if (writeFailed.load()) break;
            stats.positionsWritten += samples.size();
            stats.gamesPlayed++;
        }
    };

    std::vector<std::thread> pool;
    for (int t = 0; t < nThreads; t++) pool.emplace_back(worker, t);
    for (auto& th : pool) th.join();
    std::fclose(f);
    if (metaFile) std::fclose(metaFile);
    if (writeFailed.load()) {
        throw std::runtime_error("self-play output write failed");
    }
}

} // namespace qr
