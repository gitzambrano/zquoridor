// src/mcab.hpp -- MCTS híbrido com alpha-beta para zquoridor.
//
// Esta é a BUSCA DE PRODUÇÃO do engine (default ligada desde 2026-08-13,
// +46.9 ±23.5 Elo sobre alpha-beta puro a 200ms/lance). Alpha-beta puro
// continua íntegro e selecionável -- ver `McabParams::enabled`.
//
// Fase 1 do plano (ver plan-hybrid-mc-ab.md, Seções 4-5): núcleo de
// MCABSearch<...> -- MCABNode, PUCT, expansão, backup -- isolado do
// arena/selfplay/tuner. Testado por tests/test_mcab_core.cpp diretamente
// contra a `src/` local (sem dual-ref).
//
// NÃO é incluído por src/search.hpp em nenhuma direção (Seção 4.2):
// dependência unidirecional -- este arquivo consome tipos/funções de
// search.hpp/nnue.hpp/rules.hpp exclusivamente via parâmetros de template,
// resolvidos por ADL (argument-dependent lookup) no ponto de instanciação.
// Nenhum binário que só usa AB puro paga custo de compilação ou risco de
// regressão por causa deste arquivo, porque nunca o inclui.
//
// MORAVA em `tools/common/` até 2026-08-13, para ficar fora da árvore que
// `run_arena.py` versiona por ref (ele faz checkout de `src/` por
// --ref1/--ref2). Ao virar a busca de produção, precisa estar em `src/`
// junto do resto do core, para que TODO consumidor -- incluindo o build
// WASM/GUI, que só enxerga `src/` -- possa usá-lo.
//
// A compatibilidade com refs antigos continua garantida, por outro
// caminho: `tools/arena/arena.cpp` inclui este arquivo por caminho
// relativo a SI MESMO (`../../src/mcab.hpp`), não via `-Isrc`. Isso o
// prende ao HEAD atual, então o `src/` que o worktree do ref antigo
// materializou nunca é usado para este módulo -- e um ref anterior à
// feature, que nem tem `src/mcab.hpp`, ainda compila. O dispatch SFINAE
// de `hasMcabSupport` cuida do resto e faz aquele lado jogar AB puro.
//
// NOTA DE IMPLEMENTAÇÃO (desvio pontual do plano, documentado explicita-
// mente): a lista de parâmetros de template da Seção 4.1 tem 7 tipos
// (Eng, StateT, MoveT, MoveListT, AccPairT, RepTblT, SearchStatsT). Esta
// implementação adiciona um 8º parâmetro não-type, `PolicyDim`, com
// default 209 (= qr::POLICY_OUT). Motivo: `POLICY_OUT`/`SCORE_INF` são
// CONSTANTES (não funções), então não são encontráveis por ADL como
// `buildAccPairRoot`/`legalMoves`/etc. -- ADL só resolve chamadas de
// função a partir do tipo dos argumentos, não nomes de constantes soltos.
// Como este arquivo não inclui rules.hpp/nnue.hpp diretamente (Seção 4.2),
// `POLICY_OUT` não estaria visível para `std::array<float, POLICY_OUT>`
// dentro do template. `PolicyDim` resolve isso sem exigir include
// direto -- o valor é estável entre refs (fixado pelas dimensões do
// tabuleiro 9x9, não muda por tuning) e o default bate com
// `qr::POLICY_OUT` em todos os refs relevantes. Scores de vitória/derrota
// usados no backup de nós terminais (Seção 5, passo b) também usam uma
// constante local (`MCAB_WIN_SCORE`) em vez de `qr::SCORE_INF`, pelo
// mesmo motivo -- qualquer valor grande o suficiente satura `scoreToQ`
// perto de 0/1 de forma idêntica (comentário da Seção 4.3.1), então não
// precisa ser bit-a-bit igual à constante interna do AB.

#pragma once

#include <vector>
#include <array>
#include <cstdint>
#include <cstring>      // std::strcmp em resolveRootSelect
#include <cmath>
#include <cassert>
#include <algorithm>
#include <limits>
#include <chrono>
#include <random>
#include <type_traits>
#include <utility>

namespace mcab {

// =========================================================================
// 4.3.1 -- Conversão score -> Q
// =========================================================================
// Idêntico (matematicamente) a nnueWinProbQuant aplicado ao logit implícito
// no score -- mesma curva sigmoide já usada em treino/self-play, não uma
// calibração nova (ver Seção 4.3.1 do plano). `scale` é mcabScoreScale
// (Seção 9), default = NNUE_EVAL_SCALE (200.0).
inline double scoreToQ(int score, double scale) {
    return 1.0 / (1.0 + std::exp(-(double)score / scale));
}

// Placeholder de score para nós terminais (Seção 5 passo b) -- grande o
// bastante para saturar scoreToQ perto de 0/1 sem overflow (double aguenta
// exp(-5000) tranquilo, ver Seção 4.3.1). Não precisa coincidir com
// qr::SCORE_INF (ver nota de implementação no topo do arquivo).
constexpr int MCAB_WIN_SCORE = 1'000'000;

// =========================================================================
// Seção 9 -- Parâmetros ajustáveis (tabela completa), agrupados num struct
// só para poder ser usado tanto como membro de instância de MCABSearch
// quanto como parâmetro de chooseMoveAuto (Seção 4.4, Fase 2) sem duplicar
// a lista de campos.
// =========================================================================
enum class RootSelectMode { MaxVisits, MaxQ, MaxVisitsThenQ };
enum class BackupMode { MinimaxHard, AvgBlend };  // AvgBlend reservado (não implementado na Fase 1)

// -------------------------------------------------------------------------
// "Campo vazio" para os blocos de CONFIG no topo das ferramentas.
//
// Regra: um campo vazio usa o valor de produção definido em McabParams logo
// abaixo; preenchê-lo sobrescreve. Assim o default mora num lugar só, e o
// bloco de config de cada ferramenta é apenas uma lista de exceções -- não
// uma segunda cópia dos números, que se descola da primeira no dia em que
// alguém mudar só uma das duas.
//
// -1 como sentinela é seguro para todos os parâmetros do MCab: nenhum tem
// -1 como valor legítimo. Repare que NÃO é 0 -- `nodeBudget = 0` é um valor
// legítimo e significativo (modo equivalência, Seção 6).
// -------------------------------------------------------------------------
constexpr int    UNSET_INT  = -1;
constexpr double UNSET_REAL = -1.0;
enum class Tri { Unset = -1, Off = 0, On = 1 };   // booleano de três estados

inline int    resolve(int v, int prod)       { return v == UNSET_INT  ? prod : v; }
inline double resolve(double v, double prod) { return v == UNSET_REAL ? prod : v; }
inline bool   resolve(Tri v, bool prod)      { return v == Tri::Unset ? prod : (v == Tri::On); }

// String vazia = campo vazio. Um valor não reconhecido cai no de produção em
// vez de virar MaxVisits silenciosamente -- um typo em "visits-then-q" não
// deve parecer que funcionou.
inline RootSelectMode resolveRootSelect(const char* s, RootSelectMode prod) {
    if (s == nullptr || s[0] == '\0') return prod;
    if (std::strcmp(s, "visits") == 0)        return RootSelectMode::MaxVisits;
    if (std::strcmp(s, "q") == 0)             return RootSelectMode::MaxQ;
    if (std::strcmp(s, "visits-then-q") == 0) return RootSelectMode::MaxVisitsThenQ;
    return prod;
}

// Inverso de resolveRootSelect, para os banners das ferramentas: com o modo
// escolhível por flag, imprimir só nodes/cpuct/... deixava o log ambíguo
// sobre qual critério de raiz aquela execução usou de fato.
inline const char* rootSelectName(RootSelectMode m) {
    switch (m) {
        case RootSelectMode::MaxQ:            return "q";
        case RootSelectMode::MaxVisitsThenQ:  return "visits-then-q";
        default:                              return "visits";
    }
}

// Estes são os valores de PRODUÇÃO: é aqui que mora o default de verdade.
// Os blocos de CONFIG no topo de arena.cpp e selfplay_main.cpp são só listas
// de override -- campo vazio (mcab::UNSET_*/Tri::Unset) cai no valor daqui.
// Mudar um número neste struct muda o comportamento das duas ferramentas.
struct McabParams {
    // LIGADO por default desde 2026-08-13: com leafDepth=0/fpuReduction=0.0 o
    // híbrido mede +46.9 ±23.5 Elo sobre alpha-beta puro a 200ms/lance (800
    // partidas). Vale a ressalva de faixa: isso foi medido a 200ms; o híbrido
    // roda a ~1/9 dos nós/s do AB puro, e em controles de tempo bem mais
    // curtos a troca deve inverter. Ver a nota "Hybrid MCTS + alpha-beta" em
    // status.md antes de assumir que vale para o seu controle de tempo.
    bool enabled = true;
    int nodeBudget = 20000;              // 0/1 = modo equivalência, Seção 6
    // 0 = folha avaliada só por NNUE + quiescência de muro, sem alpha-beta
    // abaixo dela. Era 4 (valor do plano); a Fase 8 mediu 4 como catastrófico
    // a 200ms/lance e 0 como o único ponto que bate o AB puro (-26 Elo em 1000
    // partidas). Todo teste/benchmark que exercita folhas de AB de verdade
    // fixa `leafDepth` explicitamente, então não dependem deste default.
    int leafDepth = 0;
    int leafDepthMax = 8;                // teto p/ mcabAdaptiveLeafDepth (não usado na Fase 1)
    bool adaptiveLeafDepth = false;      // não implementado na Fase 1 (Seção 9 já documenta v1=false)
    double cPuct = 1.5;
    // FPU = Q(pai) - fpuReduction (Seção 5.1). 0.0, não o 0.1 do plano: medido
    // -24.4 ±22.9 Elo a favor de 0.0 em 800 partidas a 200ms, leafDepth=0.
    double fpuReduction = 0.0;
    double scoreScale = 200.0;           // = NNUE_EVAL_SCALE
    RootSelectMode rootSelectMode = RootSelectMode::MaxVisits;
    BackupMode backupMode = BackupMode::MinimaxHard;
    bool treeReuse = true;               // reuso de subárvore entre lances (Seção 8)
    bool clearTTPerMove = false;
    bool rootNoiseEnabled = false;       // ruído de Dirichlet nos priors da raiz (Seção 9) -- só self-play
    double rootNoiseAlpha = 0.3;
    double rootNoiseEpsilon = 0.25;
    uint32_t rootNoiseSeed = 0x9E3779B9u; // semente do gerador de Dirichlet; varie por thread em self-play
    int maxTreeDepth = 48;               // dimensiona mcabAccStack (Seção 4.3.3)
};

// Estatísticas agregadas de UMA chamada a chooseMoveMCAB (não confundir
// com SearchStatsT, que é por-chamada-de-searchLeaf/negamax).
struct McabStats {
    long long simulations = 0;
    long long nodesExpanded = 0;
    long long leafSearches = 0;      // chamadas reais a engine.searchLeaf
    long long leafDepthSum = 0;      // soma das profundidades usadas (média = /leafSearches)
    bool treeReused = false;         // esta chamada reaproveitou a subárvore do lance anterior (Seção 8)
    int reusedNodes = 0;             // tamanho do pool herdado após compactação
    long long leafTruncated = 0;     // folhas que estouraram o teto de tempo e foram descartadas
                                     // (ver evaluateLeaf). Muitas = leafDepth alto demais para o
                                     // controle de tempo em uso; a árvore fica cega nessas folhas.
};

// Chave de identidade de posição usada pelo reuso de subárvore (Seção 8):
// `state.hash` (Zobrist) quando o tipo de estado a expõe -- caso de
// qr::State --, senão 0. Detecção por SFINAE em vez de include direto de
// rules.hpp, mantendo a independência da Seção 4.2. Chave 0 é tratada como
// "não identificável": desliga o reuso em vez de arriscar casar posições
// diferentes (é o que acontece com os tipos de brinquedo de
// tests/test_mcab_dispatch.cpp).
template <typename S>
inline auto mcabStateKey(const S& s, int) -> decltype((uint64_t)s.hash) {
    return (uint64_t)s.hash;
}
template <typename S>
inline uint64_t mcabStateKey(const S&, ...) {
    return 0;
}

// =========================================================================
// 4.3.2 -- Node pool linear (índices, não ponteiros)
// =========================================================================
template <typename StateT, typename MoveListT>
struct MCABNode {
    StateT state;
    int side = 0;                 // state.turn, cacheado
    bool expanded = false;
    bool terminal = false;
    int terminalScore = 0;        // score (unidades NNUE_EVAL_SCALE); convertido via scoreToQ na hora de usar
    MoveListT moves;               // gerado 1x na expansão (legalMoves(state))
    std::vector<float> P;         // prior por lance, mesmo índice de `moves`
    std::vector<float> N;         // visitas por aresta
    std::vector<float> W;         // soma de valor por aresta (Q = W/N)
    std::vector<int32_t> child;   // índice no pool, -1 = não expandido
    int totalN = 0;
    bool noised = false;          // ruído de Dirichlet já aplicado a `P` (Seção 9) -- evita
                                  // recompor o ruído sobre si mesmo quando este nó vira raiz
                                  // reaproveitada de novo (Seção 8).
};

// =========================================================================
// Seção 4.1/4.3/5 -- MCABSearch<...>: seleção PUCT, expansão, backup.
// =========================================================================
template <typename Eng, typename StateT, typename MoveT, typename MoveListT,
          typename AccPairT, typename RepTblT, typename SearchStatsT,
          int PolicyDim = 209>
class MCABSearch {
public:
    using NodeT = MCABNode<StateT, MoveListT>;

    McabParams params;

    // Modo equivalência (Seção 6): mcabNodeBudget <= 1 vira um atalho sem
    // árvore/PUCT -- só para validação de corretude antes de arena real.
    bool equivMode() const { return params.nodeBudget <= 1; }

    // Acesso somente-leitura ao nó raiz da árvore atual, para inspeção em
    // teste/benchmark (priors, visitas por aresta). nullptr se a árvore
    // está vazia -- ex.: logo após o atalho de "mãos vazias", ou com
    // treeReuse desligado depois de chooseMoveMCAB retornar.
    const NodeT* rootNodeForInspection() const { return pool.empty() ? nullptr : &pool[0]; }

    size_t poolSize() const { return pool.size(); }
    size_t poolCapacity() const { return pool.capacity(); }

    // Memória aproximada da árvore (Seção 7): os nós em si mais os buffers
    // por-aresta que cada nó aloca fora de linha (moves/P/N/W/child). Não
    // conta a fragmentação do alocador -- é uma estimativa para o
    // benchmark de custo da Fase 4, não contabilidade exata.
    size_t approxTreeBytes() const {
        size_t bytes = pool.capacity() * sizeof(NodeT);
        for (const NodeT& n : pool) {
            bytes += n.P.capacity() * sizeof(float);
            bytes += n.N.capacity() * sizeof(float);
            bytes += n.W.capacity() * sizeof(float);
            bytes += n.child.capacity() * sizeof(int32_t);
        }
        return bytes;
    }

    // ---------------------------------------------------------------
    // Seção 5 -- chooseMoveMCAB
    // ---------------------------------------------------------------
    // Assinatura próxima da Seção 5 do plano, com `maxDepthCap` explícito
    // (necessário para o atalho de "mãos vazias", passo 1, que delega a
    // engine.chooseMove(...) -- mesma assinatura que esse método já
    // espera) e um McabStats* opcional para inspeção em teste/benchmark.
    MoveT chooseMoveMCAB(Eng& engine, const StateT& root, int maxDepthCap, int timeBudgetMs,
                          SearchStatsT& stats, const RepTblT& gameHistory,
                          McabStats* outStats = nullptr) {
        McabStats localStats;
        McabStats& mstats = outStats ? *outStats : localStats;
        mstats = McabStats{};

        // Passo 1 (Seção 5): atalho de final "mãos vazias" -- delega
        // direto para o solver exato já existente em chooseMove(). Não
        // há nada para o MCTS ganhar aqui (ver Seção 5, item 1).
        if (root.wallsLeft[0] == 0 && root.wallsLeft[1] == 0) {
            return engine.chooseMove(root, maxDepthCap, timeBudgetMs, stats, gameHistory);
        }

        // Seção 2: MCAB só roda em modo NNUE (depende da cabeça de
        // política para os priors do PUCT). `assert`/fallback defensivo
        // para build de release (NDEBUG) -- não lê AccPair de lixo.
        assert(engine.getEvalMode() == Eng::EvalMode::NNUE &&
               "MCABSearch requer evalMode == NNUE (Seção 2 do plano)");
        if (engine.getEvalMode() != Eng::EvalMode::NNUE) {
            return engine.chooseMove(root, maxDepthCap, timeBudgetMs, stats, gameHistory);
        }

        // Passo 2 (Seção 5): reset de estado de ordenação -- 1x por
        // chooseMoveMCAB(), não por nó (permite reuso de TT entre folhas
        // vizinhas, ver comentário de searchLeaf em search.hpp).
        engine.resetOrderingState();
        if (params.clearTTPerMove) engine.clearTT();

        localRepTbl = gameHistory;
        localRepTbl.markRoot();

        // Seção 6 -- modo equivalência: nodeBudget<=1 vira uma avaliação
        // direta dos filhos do root (sem árvore/PUCT), usada como teste
        // de sanidade contra Negamax::chooseMove pura na mesma leafDepth.
        if (equivMode()) {
            return equivalenceMove(engine, root, stats, mstats);
        }

        // Passo 3 (Seção 5): inicializa OU recupera a árvore. Reuso de
        // subárvore (Seção 8): se a nova raiz é filho/neto da raiz antiga
        // dentro da árvore que sobrou do lance anterior, esse nó vira a
        // nova raiz e o pool é compactado para conter só a subárvore dele
        // (descartando o resto, que nunca mais será alcançado) --
        // estatísticas N/W/P já acumuladas são reaproveitadas.
        if ((int)mcabAccStack.size() < params.maxTreeDepth + 2) {
            mcabAccStack.resize(params.maxTreeDepth + 2);
        }

        int budget = std::max(1, params.nodeBudget);
        bool reused = false;
        // Seção 8.2: não reusar quando clearTTPerMove está ligado -- a
        // árvore depende de valores computados com aquela TT.
        if (params.treeReuse && !params.clearTTPerMove && !pool.empty()) {
            int idx = findNodeForRoot(root);
            if (idx == 0) {
                reused = true;
            } else if (idx > 0 && !pool[idx].terminal) {
                reused = compactTo(idx, budget);
            }
        }
        if (reused) {
            pool[0].state = root;  // hash bate; normaliza o objeto por segurança
            pool[0].side = root.turn;
            mstats.treeReused = true;
            mstats.reusedNodes = (int)pool.size();
        } else {
            pool.clear();
            NodeT rootNode;
            rootNode.state = root;
            rootNode.side = root.turn;
            pool.push_back(std::move(rootNode));
        }
        pool.reserve(pool.size() + (size_t)budget + 1);

        mcabAccStack[0] = buildAccPairRoot(root, nullptr);

        // Passo 4 (Seção 5): expande a raiz se necessário.
        int wRoot = winner(root);
        if (wRoot != -1) {
            pool[0].terminal = true;
            pool[0].terminalScore = (wRoot == root.turn) ? MCAB_WIN_SCORE : -MCAB_WIN_SCORE;
        } else if (!pool[0].expanded) {
            expandNode(0, /*depthInTree=*/0, mstats);
        }

        // Seção 9 -- ruído de Dirichlet nos priors da raiz. Aplicado uma
        // única vez por nó (flag `noised`): quando este nó já foi raiz numa
        // chamada anterior e voltou a sê-lo, recompor o ruído sobre si
        // mesmo distorceria os priors progressivamente.
        if (params.rootNoiseEnabled && !pool[0].terminal && !pool[0].noised) {
            applyRootNoise(pool[0]);
        }

        // Passo 5 (Seção 5): loop de simulações até orçamento (nós OU
        // tempo, o que vier primeiro).
        //
        // O teto de tempo é checado ENTRE simulações, mas uma simulação não
        // é interrompível por si só: ela contém uma busca AB inteira de
        // `leafDepth` plies, que em posição de meio-jogo custa dezenas de
        // milissegundos. Sem repassar o tempo restante para dentro da folha,
        // um orçamento de 60ms virava ~110ms medidos (quase 2x) -- inaceitável
        // sob controle de tempo real no arena. `leafDeadline` propaga o teto
        // até engine.searchLeaf (ver evaluateLeaf).
        auto t0 = std::chrono::steady_clock::now();
        haveLeafDeadline = (timeBudgetMs > 0);
        if (haveLeafDeadline) leafDeadline = t0 + std::chrono::milliseconds(timeBudgetMs);
        while (mstats.nodesExpanded < budget) {
            if (timeBudgetMs > 0) {
                auto elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(
                                     std::chrono::steady_clock::now() - t0)
                                     .count();
                if (elapsedMs >= timeBudgetMs) break;
            }
            if (pool[0].terminal) break;  // raiz já resolvida (ex.: vitória em 0 lances -- não deveria ocorrer)
            runSimulation(engine, stats, mstats);
        }

        // Passo 6 (Seção 5): escolhe o lance final na raiz.
        MoveT best = pickRootMove(pool[0]);

        // Passo 7 (Seção 5): com reuso ligado, o pool inteiro fica de pé
        // para a próxima chamada (a compactação acontece lá, quando a nova
        // raiz é conhecida -- só aí dá para saber qual subárvore sobrevive).
        // Sem reuso, descarta agora para não segurar memória entre lances.
        if (!params.treeReuse) pool.clear();
        return best;
    }

    // Descarta a árvore acumulada (usado entre partidas, ou quando o
    // chamador sabe que o histórico mudou de forma incompatível).
    void resetTree() { pool.clear(); }

    // Semente do gerador de ruído de Dirichlet -- varie por thread em
    // self-play, senão todas as threads geram a mesma sequência de ruído e
    // a diversidade de aberturas some.
    void seedNoise(uint32_t seed) { rng.seed(seed); noiseSeeded = true; }

private:
    std::vector<NodeT> pool;
    std::vector<AccPairT> mcabAccStack;   // Seção 4.3.3 -- pilha por caminho de descida
    RepTblT localRepTbl;                  // cópia mutável de gameHistory, 1x por chooseMoveMCAB (ver Seção 5, negamax/searchLeaf)
    std::mt19937 rng;                     // ruído de Dirichlet (Seção 9) -- por instância, nunca compartilhado entre threads
    bool noiseSeeded = false;
    // Teto de tempo da chamada corrente, propagado para dentro de cada
    // engine.searchLeaf (ver evaluateLeaf). Só válido enquanto
    // haveLeafDeadline == true, o que só ocorre com timeBudgetMs > 0.
    std::chrono::steady_clock::time_point leafDeadline;
    bool haveLeafDeadline = false;

    // NOTA: buildAccPairRoot/makeChildAccPair recebem um `PlayerPathCacheTable*`
    // opcional (default nullptr) para cachear BFS de distância entre
    // chamadas -- mesmo cache que Negamax::xdistCache usa internamente na
    // busca AB. MCABSearch não nomeia esse tipo (ficaria acoplado a
    // rules.hpp, quebrando a independência da Seção 4.2) e passa nullptr
    // em todas as chamadas abaixo: cada BFS é recalculado do zero, custo
    // aceito nesta Fase 1 (correção > performance; cache de BFS fica como
    // otimização futura se o profiling da Fase 4 apontar necessidade).

    struct PathEdge {
        int nodeIdx;
        int edgeIdx;
    };

    // ---------------------------------------------------------------
    // Seção 8 -- reuso de subárvore entre lances
    // ---------------------------------------------------------------
    // Procura, na árvore do lance anterior, o nó cuja posição é a nova
    // raiz: normalmente um NETO da raiz antiga (nosso lance escolhido +
    // resposta do oponente), mas o caso de FILHO é aceito também (cobre
    // chamadas consecutivas sem lance intermediário do oponente, ex.:
    // re-análise da mesma posição). Devolve o índice no pool, ou -1 se a
    // posição não está na árvore (Seção 8.2: cai para árvore nova).
    int findNodeForRoot(const StateT& root) const {
        uint64_t key = mcabStateKey(root, 0);
        if (key == 0 || pool.empty()) return -1;  // tipo de estado sem hash: reuso desligado
        if (mcabStateKey(pool[0].state, 0) == key) return 0;
        for (size_t e = 0; e < pool[0].child.size(); e++) {
            int c = pool[0].child[e];
            if (c < 0) continue;
            if (mcabStateKey(pool[c].state, 0) == key) return c;
            for (size_t e2 = 0; e2 < pool[c].child.size(); e2++) {
                int g = pool[c].child[e2];
                if (g >= 0 && mcabStateKey(pool[g].state, 0) == key) return g;
            }
        }
        return -1;
    }

    // Move a subárvore enraizada em `rootIdx` para os índices [0, k) do
    // pool, remapeando os índices de `child`, e descarta todo o resto
    // (Seção 8.1: compactação obrigatória para não vazar memória do que
    // ficou fora do caminho jogado). Aborta (devolve false -> árvore nova)
    // se a subárvore herdada sozinha já passar do orçamento de nós: isso
    // limita o pool a no máximo ~2x nodeBudget mesmo após muitos lances
    // seguidos com reuso (Seção 12, risco de memória).
    bool compactTo(int rootIdx, int budget) {
        std::vector<int32_t> remap(pool.size(), -1);
        std::vector<int32_t> order;
        order.reserve((size_t)budget + 1);
        remap[rootIdx] = 0;
        order.push_back(rootIdx);
        for (size_t i = 0; i < order.size(); i++) {
            if ((int)order.size() > budget) return false;
            const NodeT& n = pool[order[i]];
            for (int32_t c : n.child) {
                if (c >= 0 && remap[c] < 0) {
                    remap[c] = (int32_t)order.size();
                    order.push_back(c);
                }
            }
        }
        std::vector<NodeT> compacted;
        compacted.reserve(order.size());
        for (int32_t oldIdx : order) compacted.push_back(std::move(pool[oldIdx]));
        for (NodeT& n : compacted) {
            for (int32_t& c : n.child) {
                if (c >= 0) c = remap[c];
            }
        }
        pool.swap(compacted);
        return true;
    }

    // ---------------------------------------------------------------
    // Seção 9 -- ruído de Dirichlet nos priors da raiz
    // ---------------------------------------------------------------
    // P[i] <- (1-eps) * P[i] + eps * Dir(alpha)[i]. Amostra Dirichlet pela
    // via padrão (k variáveis Gamma(alpha,1) normalizadas). Só faz sentido
    // em geração de dados de self-play -- em arena de força é ruído puro
    // (Seção 2), por isso o default é `false`.
    void applyRootNoise(NodeT& node) {
        size_t nm = node.P.size();
        if (nm == 0) return;
        if (!noiseSeeded) { rng.seed(params.rootNoiseSeed); noiseSeeded = true; }

        std::gamma_distribution<double> gamma(params.rootNoiseAlpha, 1.0);
        std::vector<double> noise(nm);
        double sum = 0.0;
        for (size_t i = 0; i < nm; i++) {
            noise[i] = gamma(rng);
            sum += noise[i];
        }
        if (sum <= 0.0) return;  // degenerado (alpha minúsculo): mantém os priors da rede

        double eps = params.rootNoiseEpsilon;
        for (size_t i = 0; i < nm; i++) {
            node.P[i] = (float)((1.0 - eps) * (double)node.P[i] + eps * (noise[i] / sum));
        }
        node.noised = true;
    }

    // ---------------------------------------------------------------
    // Seção 9 -- mcabAdaptiveLeafDepth
    // ---------------------------------------------------------------
    // Escala a profundidade da busca AB de folha com o nº de visitas do
    // ramo: +1 ply a cada 4x visitas (log_4), saturando em leafDepthMax.
    // Ramos que o PUCT insiste em revisitar ganham avaliação mais profunda
    // sem encarecer a cauda de ramos visitados 1x. FATOR NÃO CALIBRADO --
    // ver status.md; default de `adaptiveLeafDepth` é false.
    int effectiveLeafDepth(int branchVisits) const {
        if (!params.adaptiveLeafDepth) return params.leafDepth;
        int bonus = 0;
        long long threshold = 4;
        while (threshold <= (long long)branchVisits + 1 &&
               params.leafDepth + bonus < params.leafDepthMax) {
            bonus++;
            threshold *= 4;
        }
        return std::min(params.leafDepthMax, params.leafDepth + bonus);
    }

    // ---------------------------------------------------------------
    // Seção 6 -- modo equivalência (mcabNodeBudget <= 1)
    // ---------------------------------------------------------------
    // Avalia cada filho direto da raiz via searchLeaf (sem árvore, sem
    // PUCT) e escolhe o de maior Q -- serve para validar que a
    // integração está correta antes de gastar ciclos em arena real
    // (Seção 6: deve bater com Negamax::chooseMove pura na mesma
    // leafDepth, dentro do ruído esperado).
    MoveT equivalenceMove(Eng& engine, const StateT& root, SearchStatsT& stats, McabStats& mstats) {
        MoveListT moves = legalMoves(root);
        if (moves.empty()) return MoveT{};

        AccPairT rootAcc = buildAccPairRoot(root, nullptr);

        double bestQ = -std::numeric_limits<double>::infinity();
        int bestIdx = 0;
        for (size_t i = 0; i < moves.size(); i++) {
            StateT childState = applyMove(root, moves[i]);
            AccPairT parentCopy = rootAcc;  // makeChildAccPair pode resolver pending em `parent`
            AccPairT seedAcc{};
            makeChildAccPair(parentCopy, seedAcc, root, moves[i], nullptr);

            double q;  // do ponto de vista de quem joga em `root` (root.turn)
            int w = winner(childState);
            if (w != -1) {
                int sc = (w == childState.turn) ? MCAB_WIN_SCORE : -MCAB_WIN_SCORE;
                q = 1.0 - scoreToQ(sc, params.scoreScale);
            } else {
                // Isolamento por filho: TT e killers/history zerados antes de
                // cada busca. O caminho REAL (runSimulation/evaluateLeaf) faz o
                // contrário de propósito -- compartilha a TT entre folhas
                // vizinhas, que é o ganho da Seção 5 passo 2. Aqui não: o modo
                // equivalência existe só para confrontar com Negamax::chooseMove
                // no aferidor do bench, que avalia cada lance com engine fresca.
                // Com a TT compartilhada, os scores dos irmãos avaliados depois
                // herdam limites (fail-soft) das buscas anteriores e sobem
                // sistematicamente -- medido em até +105 num único lance. Isso é
                // instabilidade normal de alpha-beta, não erro de sinal, mas
                // basta para inverter a ordem de dois lances próximos e fazer o
                // bench acusar regressão onde não há.
                engine.clearTT();
                engine.resetOrderingState();
                int score = engine.searchLeaf(childState, params.leafDepth, stats, localRepTbl, &seedAcc);
                q = 1.0 - scoreToQ(score, params.scoreScale);
                mstats.leafSearches++;
                mstats.leafDepthSum += params.leafDepth;
            }
            mstats.nodesExpanded++;
            mstats.simulations++;
            if (q > bestQ) {
                bestQ = q;
                bestIdx = (int)i;
            }
        }
        return moves[bestIdx];
    }

    // ---------------------------------------------------------------
    // Seção 5.2 -- Expansão
    // ---------------------------------------------------------------
    // Gera `moves`, computa `P` via forwardPolicyQuant + moveToPolicyIndex
    // (softmax restrito aos lances legais -- mesmo padrão de ordenação
    // assistida por política já usado em search.hpp). NÃO avalia o nó em
    // si (isso é feito em runSimulation logo depois de expandir, Seção 5
    // passo 5b) -- expandNode só monta a estrutura (moves/P/N/W/child).
    void expandNode(int idx, int depthInTree, McabStats& mstats) {
        NodeT& node = pool[idx];
        node.moves = legalMoves(node.state);
        size_t nm = node.moves.size();
        node.P.assign(nm, 0.f);
        node.N.assign(nm, 0.f);
        node.W.assign(nm, 0.f);
        node.child.assign(nm, -1);

        if (nm > 0) {
            std::array<float, PolicyDim> policyOut{};
            forwardPolicyQuant(mcabAccStack[depthInTree].acc[node.side], policyOut);

            float maxLogit = -std::numeric_limits<float>::infinity();
            std::vector<float> logits(nm);
            for (size_t i = 0; i < nm; i++) {
                logits[i] = policyLogitForMove(policyOut, node.moves[i], node.side);
                maxLogit = std::max(maxLogit, logits[i]);
            }
            float sumExp = 0.f;
            for (size_t i = 0; i < nm; i++) {
                node.P[i] = std::exp(logits[i] - maxLogit);
                sumExp += node.P[i];
            }
            if (sumExp > 0.f) {
                for (size_t i = 0; i < nm; i++) node.P[i] /= sumExp;
            } else {
                for (size_t i = 0; i < nm; i++) node.P[i] = 1.f / (float)nm;
            }
        }

        node.expanded = true;
        mstats.nodesExpanded++;
    }

    // ---------------------------------------------------------------
    // Seção 5.1 -- Seleção PUCT
    // ---------------------------------------------------------------
    // score(a) = Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
    // Q(s,a) = W(s,a)/N(s,a) se N(s,a) > 0; senão FPU = Q(pai) - fpuReduction,
    // onde Q(pai) = soma(W)/totalN do próprio nó `s` (0.5 neutro se totalN==0).
    int selectChildPUCT(const NodeT& node) const {
        size_t nm = node.moves.size();
        if (nm == 0) return -1;

        double parentQ = 0.5;
        if (node.totalN > 0) {
            double sumW = 0.0;
            for (size_t i = 0; i < nm; i++) sumW += node.W[i];
            parentQ = sumW / (double)node.totalN;
        }
        double fpu = parentQ - params.fpuReduction;

        double sqrtN = std::sqrt((double)std::max(0, node.totalN));
        int best = -1;
        double bestScore = -std::numeric_limits<double>::infinity();
        for (size_t i = 0; i < nm; i++) {
            double q = node.N[i] > 0.0 ? (node.W[i] / node.N[i]) : fpu;
            double u = params.cPuct * (double)node.P[i] * sqrtN / (1.0 + (double)node.N[i]);
            double score = q + u;
            if (score > bestScore) {
                bestScore = score;
                best = (int)i;
            }
        }
        return best;
    }

    // ---------------------------------------------------------------
    // Seção 5, passos a-c -- uma simulação completa (seleção, expansão +
    // avaliação, backup).
    // ---------------------------------------------------------------
    void runSimulation(Eng& engine, SearchStatsT& stats, McabStats& mstats) {
        std::vector<PathEdge> path;
        path.reserve((size_t)params.maxTreeDepth + 2);

        int curIdx = 0;
        int depth = 0;  // ply desde a raiz; indexa mcabAccStack

        while (true) {
            NodeT& node = pool[curIdx];

            if (node.terminal) {
                double leafQ = scoreToQ(node.terminalScore, params.scoreScale);
                backup(path, leafQ, mstats);
                return;
            }

            // Guarda contra estouro de mcabAccStack/profundidade máxima
            // (Seção 9, mcabMaxTreeDepth) -- trata o nó atual como folha
            // forçada em vez de continuar descendo.
            bool depthCapped = (depth + 1 >= (int)mcabAccStack.size()) ||
                                (depth >= params.maxTreeDepth);

            if (!node.expanded || depthCapped) {
                if (!node.expanded) expandNode(curIdx, depth, mstats);
                // Visitas do RAMO que levou até aqui (Seção 9,
                // mcabAdaptiveLeafDepth): totalN do nó PAI, ou o totalN da
                // própria raiz quando estamos nela.
                //
                // Tem que ser o totalN do PAI, e não as visitas da ARESTA
                // pai->este nó: uma folha é avaliada exatamente uma vez, no
                // instante em que é criada, e nesse instante a aresta que
                // leva até ela ainda tem N==0 (o backup só a incrementa
                // depois). Chavear na aresta deixava effectiveLeafDepth
                // constante em `leafDepth` -- a feature inteira era inerte,
                // pego pelo testAdaptiveLeafDepth de tests/test_mcab_phase9.
                // O totalN do pai é também o que a Seção 9 do plano
                // descreve ("escala leafDepth com N do nó pai").
                int branchVisits = path.empty()
                                        ? pool[0].totalN
                                        : pool[path.back().nodeIdx].totalN;
                double leafQ = evaluateLeaf(engine, pool[curIdx], depth, branchVisits, stats, mstats);
                backup(path, leafQ, mstats);
                return;
            }

            int e = selectChildPUCT(node);
            if (e < 0) {
                // Nó expandido sem lances -- não deveria ocorrer em
                // Quoridor não-terminal, mas não crasha: trata como
                // neutro.
                backup(path, 0.5, mstats);
                return;
            }
            path.push_back({curIdx, e});

            StateT beforeState = node.state;  // cópia -- `node` pode ser invalidada por pool.push_back abaixo
            MoveT mv = node.moves[e];
            int parentSide = node.side;

            int childIdx = node.child[e];
            if (childIdx == -1) {
                StateT childState = applyMove(beforeState, mv);
                childIdx = createChild(childState);
                pool[curIdx].child[e] = childIdx;  // reindexado -- `node` pode ter sido invalidada
            }

            makeChildAccPair(mcabAccStack[depth], mcabAccStack[depth + 1], beforeState, mv, nullptr);
            (void)parentSide;

            curIdx = childIdx;
            depth++;
        }
    }

    // Cria o MCABNode do estado `s` (ainda não expandido) e o insere no
    // pool. Marca terminal=true imediatamente se `s` já é posição de fim
    // de jogo (Seção 5 passo b: "Se terminal, marca terminal=true e usa
    // o valor exato, sem chamar searchLeaf").
    int createChild(const StateT& s) {
        NodeT node;
        node.state = s;
        node.side = s.turn;
        int w = winner(s);
        if (w != -1) {
            node.terminal = true;
            node.terminalScore = (w == s.turn) ? MCAB_WIN_SCORE : -MCAB_WIN_SCORE;
        }
        pool.push_back(std::move(node));
        return (int)pool.size() - 1;
    }

    // Avalia um nó recém-expandido e não-terminal via searchLeaf (Seção
    // 5.2) -- devolve Q do ponto de vista do PRÓPRIO nó (node.side), ou
    // seja, de quem vai jogar a partir dele. O backup (Seção 5.3) é quem
    // se encarrega de inverter a perspectiva subindo a árvore.
    double evaluateLeaf(Eng& engine, NodeT& node, int depthInTree, int branchVisits,
                         SearchStatsT& stats, McabStats& mstats) {
        if (node.moves.empty()) return 0.5;  // sem lances e não-terminal: não deveria ocorrer; neutro defensivo
        int leafDepth = effectiveLeafDepth(branchVisits);

        // Teto de tempo herdado da chamada (ver comentário do loop de
        // simulações). 0 = sem teto, que é o caso de chooseMoveMCAB com
        // timeBudgetMs=0 -- e é o caso que os benchmarks de equivalência
        // usam, justamente para manter as folhas determinísticas.
        int leafBudgetMs = 0;
        if (haveLeafDeadline) {
            auto restanteMs = std::chrono::duration_cast<std::chrono::milliseconds>(
                                  leafDeadline - std::chrono::steady_clock::now())
                                  .count();
            // <=0 significa que o orçamento já acabou; 1ms para que
            // searchLeaf devolva imediatamente em vez de tratar 0 como
            // "sem limite prático".
            leafBudgetMs = (int)std::max<long long>(1, restanteMs);
        }

        int score = engine.searchLeaf(node.state, leafDepth, stats, localRepTbl,
                                       &mcabAccStack[depthInTree], leafBudgetMs);
        mstats.leafSearches++;
        mstats.leafDepthSum += leafDepth;

        // Folha truncada pelo tempo devolve um score PARCIAL (o que a
        // negamax tinha na mão quando `stopped` subiu), tipicamente 0.
        // Usá-lo como avaliação enviesaria o nó para Q=0.5 sem que nada
        // sinalize o problema. Devolvemos o Q do PAI-como-neutro (0.5) do
        // mesmo jeito, mas contabilizado: leafTruncated alto é o sintoma de
        // leafDepth grande demais para o controle de tempo em uso.
        if (haveLeafDeadline && engine.searchWasStopped()) {
            mstats.leafTruncated++;
            return 0.5;
        }
        return scoreToQ(score, params.scoreScale);
    }

    // ---------------------------------------------------------------
    // Seção 5.3 -- Backup (minimax hard)
    // ---------------------------------------------------------------
    // `leafQ` é do ponto de vista do nó recém-avaliado (quem vai jogar
    // ali). Subindo a árvore, cada nível corresponde a uma troca de
    // mover -- inverte o sinal a cada passo (Q_pai = 1 - Q_filho, análogo
    // a negamax) e atualiza N/W da aresta correspondente.
    void backup(const std::vector<PathEdge>& path, double leafQ, McabStats& mstats) {
        double v = leafQ;
        for (int i = (int)path.size() - 1; i >= 0; i--) {
            v = 1.0 - v;
            NodeT& node = pool[path[i].nodeIdx];
            int e = path[i].edgeIdx;
            node.N[e] += 1.f;
            node.W[e] += (float)v;
            node.totalN += 1;
        }
        mstats.simulations++;
    }

    // ---------------------------------------------------------------
    // Seção 5 passo 6 -- escolha do lance final na raiz.
    // ---------------------------------------------------------------
    MoveT pickRootMove(const NodeT& r) const {
        size_t nm = r.moves.size();
        assert(nm > 0 && "raiz não-terminal sem lances legais -- não deveria ocorrer em Quoridor");
        if (nm == 0) return MoveT{};

        auto qOf = [&](size_t i) { return r.N[i] > 0.f ? (double)(r.W[i] / r.N[i]) : -1.0; };

        size_t best = 0;
        for (size_t i = 1; i < nm; i++) {
            bool better;
            switch (params.rootSelectMode) {
                case RootSelectMode::MaxQ:
                    better = qOf(i) > qOf(best);
                    break;
                case RootSelectMode::MaxVisitsThenQ:
                    better = (r.N[i] != r.N[best]) ? (r.N[i] > r.N[best]) : (qOf(i) > qOf(best));
                    break;
                case RootSelectMode::MaxVisits:
                default:
                    better = r.N[i] > r.N[best];
                    break;
            }
            if (better) best = i;
        }
        return r.moves[best];
    }
};

// =========================================================================
// Seção 4.4 / Fase 2 -- camada de compatibilidade com refs antigos
// =========================================================================
// `arena.cpp` compila sempre o `tools/` do HEAD atual, mas o `src/search.hpp`
// de cada engine vem de um git worktree do ref pedido (`--ref1`/`--ref2`).
// Um ref anterior à Fase 0 deste plano não tem `searchLeaf`/
// `resetOrderingState` público -- instanciar MCABSearch contra ele
// quebraria a COMPILAÇÃO do arena inteiro, que é justamente a ferramenta
// usada para comparar a versão nova contra as antigas.
//
// A saída é a mesma já usada em arena.cpp para
// `trySetPolicyOrdering`/`hasPolicyOrdering`: detectar a capacidade por
// SFINAE e resolver a escolha em TEMPO DE COMPILAÇÃO. Passar `--e1-mcab`
// contra um ref antigo não é erro -- a flag é aceita, um aviso é impresso
// 1x pelo chamador, e aquele lado joga AB puro.

// Trait: `Eng` expõe searchLeaf(const StateT&, int, SearchStatsT&, RepTblT&),
// resetOrderingState() e searchWasStopped() públicos? Os três entram na
// mesma detecção porque MCABSearch usa os três; um ref que tenha só parte
// deles (build intermediário desta feature) cai no fallback de AB puro em
// vez de quebrar a compilação do arena.
template <typename Eng, typename StateT, typename SearchStatsT, typename RepTblT>
constexpr auto hasMcabSupport(int)
    -> decltype(std::declval<Eng&>().searchLeaf(std::declval<const StateT&>(), 1,
                                                 std::declval<SearchStatsT&>(),
                                                 std::declval<RepTblT&>()),
                std::declval<Eng&>().resetOrderingState(),
                std::declval<const Eng&>().searchWasStopped(),
                bool()) {
    return true;
}
template <typename Eng, typename StateT, typename SearchStatsT, typename RepTblT>
constexpr bool hasMcabSupport(...) {
    return false;
}

// Dispatch sem estado: usa MCAB se o tipo da engine suporta E `p.enabled`;
// senão delega a `eng.chooseMove(...)`. As duas sobrecargas têm condições
// de `enable_if` mutuamente exclusivas -- nunca são ambíguas, e a que não
// se aplica nem chega a instanciar `MCABSearch`.
//
// ATENÇÃO: esta forma cria uma MCABSearch local por chamada, ou seja, SEM
// reuso de subárvore entre lances (Seção 8). Use `McabRunner` abaixo (uma
// instância por engine, viva pela partida inteira) quando o reuso importa
// -- é o que arena/selfplay/tuner fazem.
template <typename Eng, typename StateT, typename MoveT, typename MoveListT,
          typename AccPairT, typename RepTblT, typename SearchStatsT, int PolicyDim = 209>
std::enable_if_t<hasMcabSupport<Eng, StateT, SearchStatsT, RepTblT>(0), MoveT>
chooseMoveAuto(Eng& eng, const McabParams& p, const StateT& root, int maxDepthCap,
                int timeBudgetMs, SearchStatsT& stats, const RepTblT& hist,
                McabStats* outStats = nullptr) {
    if (!p.enabled) return eng.chooseMove(root, maxDepthCap, timeBudgetMs, stats, hist);
    MCABSearch<Eng, StateT, MoveT, MoveListT, AccPairT, RepTblT, SearchStatsT, PolicyDim> search;
    search.params = p;
    return search.chooseMoveMCAB(eng, root, maxDepthCap, timeBudgetMs, stats, hist, outStats);
}

template <typename Eng, typename StateT, typename MoveT, typename MoveListT,
          typename AccPairT, typename RepTblT, typename SearchStatsT, int PolicyDim = 209>
std::enable_if_t<!hasMcabSupport<Eng, StateT, SearchStatsT, RepTblT>(0), MoveT>
chooseMoveAuto(Eng& eng, const McabParams& p, const StateT& root, int maxDepthCap,
                int timeBudgetMs, SearchStatsT& stats, const RepTblT& hist,
                McabStats* outStats = nullptr) {
    // Eng sem suporte a MCAB (ref anterior a este plano): ignora `p`
    // completamente e roda AB puro, sem erro de compilação nem aviso de
    // runtime intrusivo (o aviso 1x fica a cargo do chamador, que sabe
    // qual engine é qual).
    (void)p;
    (void)outStats;
    return eng.chooseMove(root, maxDepthCap, timeBudgetMs, stats, hist);
}

// Portador com estado: mantém UMA MCABSearch viva pela partida inteira
// (necessário para o reuso de subárvore da Seção 8) e expõe a mesma
// decisão de dispatch. Quando `Eng` não suporta MCAB, o membro vira um
// stub vazio -- `MCABSearch<Eng,...>` nunca é instanciada, então o arena
// continua compilando contra refs antigos (Seção 4.4).
template <typename Eng, typename StateT, typename MoveT, typename MoveListT,
          typename AccPairT, typename RepTblT, typename SearchStatsT, int PolicyDim = 209>
struct McabRunner {
    static constexpr bool supported = hasMcabSupport<Eng, StateT, SearchStatsT, RepTblT>(0);

    struct Unsupported {
        McabParams params;
        void resetTree() {}
        void seedNoise(uint32_t) {}
    };
    using SearchT = std::conditional_t<
        supported,
        MCABSearch<Eng, StateT, MoveT, MoveListT, AccPairT, RepTblT, SearchStatsT, PolicyDim>,
        Unsupported>;

    SearchT search;

    McabParams& params() { return search.params; }
    const McabParams& params() const { return search.params; }
    void setParams(const McabParams& p) { search.params = p; }

    // Verdadeiro só quando esta chamada de choose() vai de fato rodar o
    // híbrido -- é o que o chamador usa para imprimir o aviso 1x de
    // fallback por ref incompatível.
    bool activeForThisEngine() const { return supported && search.params.enabled; }

    void resetTree() { search.resetTree(); }
    void seedNoise(uint32_t seed) { search.seedNoise(seed); }

    MoveT choose(Eng& eng, const StateT& root, int maxDepthCap, int timeBudgetMs,
                  SearchStatsT& stats, const RepTblT& hist, McabStats* outStats = nullptr) {
        if constexpr (supported) {
            if (!search.params.enabled) {
                return eng.chooseMove(root, maxDepthCap, timeBudgetMs, stats, hist);
            }
            return search.chooseMoveMCAB(eng, root, maxDepthCap, timeBudgetMs, stats, hist, outStats);
        } else {
            (void)outStats;
            return eng.chooseMove(root, maxDepthCap, timeBudgetMs, stats, hist);
        }
    }
};

}  // namespace mcab
