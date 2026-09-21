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
#include <numeric>
#include <limits>
#include <chrono>
#include <random>
#include <type_traits>
#include <utility>
#include <memory>
#include <queue>
#include <unordered_map>

namespace mcab {

// =========================================================================
// 4.3.1 -- Conversão score -> Q
// =========================================================================

// perf/speed-elo-100: exp rápido pra caminhos de INFERÊNCIA do MCAB
// (conversão score->Q e softmax de política) -- `std::exp` double era o
// maior item do perfil de produção (~21%) depois da vetorização da NNUE.
// Implementação 2^(x*log2e) com arredondamento do expoente inteiro +
// polinômio grau-5 para a fração: erro relativo ~1e-6 em toda a faixa
// útil, saturação preservada fora dela. NÃO é usado por nada que grave
// dados de treino (nnueWinProbQuant em nnue.hpp continua exato) nem por
// caminho algum de paridade C++<->Python -- só por heurísticas de busca.
inline float mcabFastExp(float x) {
    if (x < -87.f) return 0.f;
    if (x > 87.f) return 3.4e38f;
    float t = x * 1.4426950408889634f;               // x / ln(2)
    int i = (int)(t >= 0.f ? t + 0.5f : t - 0.5f);   // round-to-nearest
    float f = t - (float)i;
    // 2^f, f em [-0.5, 0.5] -- série de Taylor truncada (coeficientes de ln2^k/k!)
    float p = 1.f
            + f * (0.69314718f
            + f * (0.24022651f
            + f * (0.05550411f
            + f * (0.00961813f
            + f * 0.00133336f))));
    uint32_t bits = (uint32_t)(i + 127) << 23;
    float scale2;
    std::memcpy(&scale2, &bits, sizeof(scale2));
    return p * scale2;
}

// Idêntico (matematicamente) a nnueWinProbQuant aplicado ao logit implícito
// no score -- mesma curva sigmoide já usada em treino/self-play, não uma
// calibração nova (ver Seção 4.3.1 do plano). `scale` é mcabScoreScale
// (Seção 9), default = NNUE_EVAL_SCALE (200.0).
//
// perf/speed-elo-100: usa mcabFastExp acima (erro ~1e-6 na curva,
// saturação em ±inf preservada -- MCAB_WIN_SCORE=1e6 => score/scale=5000
// satura exatamente igual). Chamado 1x POR SIMULAÇÃO; era o maior custo
// escalar remanescente do caminho de produção.
inline double scoreToQ(int score, double scale) {
    return 1.0 / (1.0 + (double)mcabFastExp(-(float)((double)score / scale)));
}

// Placeholder de score para nós terminais (Seção 5 passo b) -- grande o
// bastante para saturar scoreToQ perto de 0/1 sem overflow (double aguenta
// exp(-5000) tranquilo, ver Seção 4.3.1). Não precisa coincidir com
// qr::SCORE_INF (ver nota de implementação no topo do arquivo).
constexpr int MCAB_WIN_SCORE = 1'000'000;
constexpr int MCAB_WALL_GRID = 8;

template <typename StateT>
inline bool mcabWallSlotAvailable(const StateT& s, int orientation, int r, int c) {
    int slot = r * MCAB_WALL_GRID + c;
    if (orientation == 0) {
        if ((s.wallsH >> slot) & 1ull) return false;
        if (c > 0 && ((s.wallsH >> (slot - 1)) & 1ull)) return false;
        if (c + 1 < MCAB_WALL_GRID && ((s.wallsH >> (slot + 1)) & 1ull)) return false;
        if ((s.wallsV >> slot) & 1ull) return false;
    } else {
        if ((s.wallsV >> slot) & 1ull) return false;
        if (r > 0 && ((s.wallsV >> (slot - MCAB_WALL_GRID)) & 1ull)) return false;
        if (r + 1 < MCAB_WALL_GRID && ((s.wallsV >> (slot + MCAB_WALL_GRID)) & 1ull)) return false;
        if ((s.wallsH >> slot) & 1ull) return false;
    }
    return true;
}

// =========================================================================
// Seção 9 -- Parâmetros ajustáveis (tabela completa), agrupados num struct
// só para poder ser usado tanto como membro de instância de MCABSearch
// quanto como parâmetro de chooseMoveAuto (Seção 4.4, Fase 2) sem duplicar
// a lista de campos.
// =========================================================================
enum class RootSelectMode { MaxVisits, MaxQ, MaxVisitsThenQ };
enum class BackupMode { MinimaxHard, AvgBlend };

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

// String form used by the arena/self-play command lines and config blocks.
inline BackupMode resolveBackupMode(const char* s, BackupMode prod) {
    if (s == nullptr || s[0] == '\0') return prod;
    if (std::strcmp(s, "minimax") == 0 || std::strcmp(s, "hard") == 0)
        return BackupMode::MinimaxHard;
    if (std::strcmp(s, "avg") == 0 || std::strcmp(s, "mean") == 0)
        return BackupMode::AvgBlend;
    return prod;
}

inline const char* backupModeName(BackupMode m) {
    return m == BackupMode::AvgBlend ? "avg" : "minimax";
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
    // A fixed node budget is useful for deterministic benchmarks and
    // self-play. Real game front-ends enable autoNodeBudget so wall-clock
    // time, not a 200 ms-era node cap, becomes the primary limit.
    bool autoNodeBudget = false;
    int autoNodeBudgetPerMs = 32;
    int autoNodeBudgetCeiling = 640000;

    // Experimental real-clock time allocation. Front-ends set
    // adaptiveOptimumMs to TimeManager::optimum and pass maximumMs as the
    // hard timeBudgetMs. Fixed-movetime/self-play keep this disabled.
    bool adaptiveTime = false;
    int adaptiveOptimumMs = 0;
    double adaptiveMinFactor = 1.00;
    double adaptiveStableFactor = 1.00;
    double adaptiveUncertainFactor = 1.25;
    double adaptiveVolatileFactor = 1.60;
    int adaptiveMinSimulations = 1500;
    // 0 = folha avaliada só por nnueEvalInt no acumulador incremental, sem
    // searchLeaf e sem quiescência de muro. Era 4 (valor do plano); a Fase 8
    // mediu 4 como catastrófico a 200ms/lance e 0 como o único ponto que
    // bate o AB puro. Até 2026-08-14 o 0 ainda caía em searchLeaf→QS;
    // o atalho de rede é o default de produção. Todo teste/benchmark que
    // exercita folhas de AB de verdade fixa `leafDepth` explicitamente.
    int leafDepth = 0;
    int leafDepthMax = 8;                // teto p/ mcabAdaptiveLeafDepth (não usado na Fase 1)
    bool adaptiveLeafDepth = false;      // não implementado na Fase 1 (Seção 9 já documenta v1=false)
    double cPuct = 0.80;
    // FPU = Q(pai) - fpuReduction (Seção 5.1). 0.0, não o 0.1 do plano: medido
    // -24.4 ±22.9 Elo a favor de 0.0 em 800 partidas a 200ms, leafDepth=0.
    double fpuReduction = 0.0;
    double scoreScale = 160.0;           // = NNUE_EVAL_SCALE
    RootSelectMode rootSelectMode = RootSelectMode::MaxVisits;
    BackupMode backupMode = BackupMode::AvgBlend;
    bool progressiveWidening = false;   // production default: off until an Elo gate
    int wideningInitialMoves = 16;      // top policy moves available at N=0
    double wideningCoefficient = 2.0;   // added coefficient in c*N^alpha
    double wideningExponent = 0.5;      // exponent alpha in c*N^alpha
    bool treeReuse = true;               // reuso de subárvore entre lances (Seção 8)
    // Experimental DAG mode. Nodes at the same root-relative ply that
    // represent the same complete state share one MCAB node.
    bool transpositionGraph = true;
    // MCGS-style graph semantics: allow exact transpositions across plies and
    // back them up with node-own-frame Q-correction. Off keeps the promoted
    // same-ply DAG byte-for-byte in behavior.
    bool graphQCorrection = true;
    double graphLeafMix = 0.75;
    bool clearTTPerMove = false;
    // Separate bounded policy/value inference caches; experimental opt-in.
    bool evalCache = true;
    bool rootNoiseEnabled = false;       // ruído de Dirichlet nos priors da raiz (Seção 9) -- só self-play
    double rootNoiseAlpha = 0.3;
    double rootNoiseEpsilon = 0.25;
    uint32_t rootNoiseSeed = 0x9E3779B9u; // semente do gerador de Dirichlet; varie por thread em self-play
    int maxTreeDepth = 48;               // dimensiona mcabAccStack (Seção 4.3.3)

    // inv/ab-policy, direction E ("two-stage root"): before the tree loop,
    // score every root child with a full-window AB search of
    // abPrefilterDepth plies and keep only the top abPrefilterTopK children
    // active at the MCTS root (priors renormalized over them). Zero keeps
    // the plain behavior. The AB pass gets at most
    // abPrefilterTimeFrac of the move's time budget when timeBudgetMs > 0;
    // with timeBudgetMs == 0 the pass runs unbounded, so keep the depth
    // modest in that case. Off by default -- bit-identical production.
    int abPrefilterDepth = 0;
    int abPrefilterTopK = 0;
    double abPrefilterTimeFrac = 0.25;

    // inv/endgame-wander: leaf depth for a spent wall stock.
    //
    // The WL head is almost blind to the pawn race after a side spends its
    // walls. It reports a win probability of 0.13 to 0.28 for every pawn
    // distance from 5 down to 2, and it separates the moves only at
    // distance 1. A tree of static leaves therefore has no gradient toward
    // the goal. `AvgBlend` averages 20000 of those near-equal leaves, the
    // Q values of the root moves land within 0.02 of each other, and
    // `MaxVisits` picks whichever branch PUCT happened to visit more. The
    // engine then shuffles its pawn sideways in a won race. See
    // benchmarks/repro_wander.cpp for the position, and status.md.
    //
    // When the SIDE TO MOVE at the root holds at most
    // endgameMoverWallThreshold walls, every leaf gets a real alpha-beta
    // search of endgameLeafDepth plies instead of the static net value. The
    // alpha-beta search sees the goal row through winner(), so it supplies
    // the gradient the net does not.
    //
    // The gate counts the MOVER's own walls, not the combined stock, for
    // two reasons. The reported failure needs only the mover to be out of
    // walls: the opponent still held 8 walls in the repro position. A
    // combined-stock gate wide enough to cover that position fires through
    // most of the game and costs more than half the node rate. That variant
    // measured -88.7 +/- 86.7 Elo over 60 games at 200ms.
    //
    // Threshold 0 is the production default since 2026-08-24. It measured
    // +17.4 +/- 37.8 Elo over 300 games at 200ms. The difference is inside
    // the error margin, therefore the rule is neutral for strength. It is
    // on by default because it fixes the endgame wandering at no measured
    // cost. The interval still admits a small loss, so re-measure with more
    // games before you treat the rule as an Elo gain.
    //
    // A negative threshold turns the rule off. Do NOT raise the threshold
    // toward 10: that makes the rule global, which is the leafDepth >= 1
    // setting status.md already rejected at approximately -250 Elo.
    int endgameMoverWallThreshold = 0;
    int endgameLeafDepth = 2;
};

inline int effectiveNodeBudget(const McabParams& params, int timeBudgetMs) {
    int budget = std::max(1, params.nodeBudget);
    if (!params.autoNodeBudget || timeBudgetMs <= 0 || params.nodeBudget <= 1)
        return budget;

    const long long scaled =
        (long long)timeBudgetMs * (long long)std::max(1, params.autoNodeBudgetPerMs);
    const long long ceiling =
        std::max<long long>(params.nodeBudget, params.autoNodeBudgetCeiling);
    return (int)std::min<long long>(
        std::max<long long>(params.nodeBudget, scaled), ceiling);
}

// Stockfish-inspired, but MCAB-specific, conversion from root uncertainty to
// a multiplier of the TimeManager optimum. These values are intentionally
// exposed as a pure function so regression tests can pin the classification.
inline double adaptiveTimeFactor(double visitRatio, double qGap, double stableFraction,
                                 const McabParams& p) {
    if (stableFraction < 0.15 || visitRatio < 1.15 || qGap < -0.01)
        return p.adaptiveVolatileFactor;
    if (visitRatio < 1.50 || qGap < 0.005)
        return p.adaptiveUncertainFactor;
    if (stableFraction >= 0.35 && visitRatio >= 2.0 && qGap >= 0.015)
        return p.adaptiveStableFactor;
    return 1.0;
}

// Estatísticas agregadas de UMA chamada a chooseMoveMCAB (não confundir
// com SearchStatsT, que é por-chamada-de-searchLeaf/negamax).
struct McabStats {
    long long simulations = 0;
    long long nodesExpanded = 0;
    int effectiveNodeBudget = 0;
    long long leafSearches = 0;      // avaliações de folha (nnueEvalInt ou searchLeaf)
    long long leafDepthSum = 0;      // soma das profundidades usadas (média = /leafSearches)
    bool treeReused = false;         // esta chamada reaproveitou a subárvore do lance anterior (Seção 8)
    int reusedNodes = 0;             // tamanho do pool herdado após compactação
    long long leafTruncated = 0;     // folhas que estouraram o teto de tempo e foram descartadas
    long long evalCachePolicyHits = 0;
    long long evalCachePolicyMisses = 0;
    long long evalCacheValueHits = 0;
    long long evalCacheValueMisses = 0;
    long long evalCachePolicyEvictions = 0;
    long long evalCacheValueEvictions = 0;
    long long transpositionHits = 0;
    long long transpositionLookups = 0;
    long long graphCycleStops = 0;
    bool adaptiveTimeStop = false;
    int adaptiveTargetMs = 0;
    long long adaptiveLeaderChanges = 0;
    double adaptiveVisitRatio = 0.0;
    double adaptiveQGap = 0.0;
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

// The normal Zobrist key does not include wallsLeft. For a graph
// transposition, the wall stock is part of the game state and must be in
// the identity key.
template <typename S>
inline auto mcabGraphKey(const S& s, int)
    -> decltype((void)s.wallsLeft[0], (uint64_t)s.hash) {
    uint64_t k = (uint64_t)s.hash;
    k ^= (uint64_t)(uint8_t)s.wallsLeft[0] * 0x9E3779B185EBCA87ULL;
    k ^= (uint64_t)(uint8_t)s.wallsLeft[1] * 0xC2B2AE3D27D4EB4FULL;
    return k;
}
template <typename S>
inline uint64_t mcabGraphKey(const S& s, ...) {
    return mcabStateKey(s, 0);
}

template <typename S>
inline auto mcabGraphStateEqual(const S& a, const S& b, int)
    -> decltype((void)a.pawn[0], (void)a.wallsH, (void)a.wallsV,
                (void)a.wallsLeft[0], (void)a.turn, bool()) {
    return a.pawn[0] == b.pawn[0] &&
           a.pawn[1] == b.pawn[1] &&
           a.wallsH == b.wallsH &&
           a.wallsV == b.wallsV &&
           a.wallsLeft[0] == b.wallsLeft[0] &&
           a.wallsLeft[1] == b.wallsLeft[1] &&
           a.turn == b.turn;
}
template <typename S>
inline bool mcabGraphStateEqual(const S& a, const S& b, ...) {
    return mcabStateKey(a, 0) == mcabStateKey(b, 0);
}
// The board Zobrist key does not encode wall ownership, while NNUE consumes
// both remaining-wall counts. Include both stocks in inference-cache identity.
template <typename S>
inline auto mcabEvalStateKey(const S& s, int)
    -> decltype((void)s.wallsLeft[0], (uint64_t)s.hash) {
    uint64_t key = mcabStateKey(s, 0);
    if (key == 0) return 0;
    uint64_t stocks = (uint64_t)(uint8_t)s.wallsLeft[0]
                    | ((uint64_t)(uint8_t)s.wallsLeft[1] << 8);
    key ^= stocks + 0x9E3779B97F4A7C15ull + (key << 6) + (key >> 2);
    key ^= key >> 30;
    key *= 0xBF58476D1CE4E5B9ull;
    key ^= key >> 27;
    key *= 0x94D049BB133111EBull;
    key ^= key >> 31;
    return key;
}
template <typename S>
inline uint64_t mcabEvalStateKey(const S& s, ...) {
    return mcabStateKey(s, 0);
}

// Cache de BFS de distância (PlayerPathCacheTable em rules.hpp) usado ao
// construir acumuladores NNUE no caminho da árvore. MCABSearch não nomeia
// esse tipo -- Seção 4.2 -- então a detecção é por SFINAE, no mesmo
// padrão de mcabStateKey: Negamax expõe pathCache(); engines de brinquedo
// e refs antigos não, e nesses casos o ponteiro fica nullptr (rebuild de
// BFS a cada ply, o comportamento anterior). nullptr converte para
// qualquer T*, inclusive o `const void*` dos fakes em
// tests/test_mcab_dispatch.cpp.
template <typename E>
inline auto mcabPathCache(E& engine, int) -> decltype(engine.pathCache()) {
    return engine.pathCache();
}
template <typename E>
inline decltype(nullptr) mcabPathCache(E&, ...) {
    return nullptr;
}

// nnueEvalInt vive em qr:: e é encontrado por ADL a partir de AccPair.
// Engines de brinquedo não o têm: evaluateLeaf cai em searchLeaf mesmo
// com leafDepth==0, o que é o comportamento antigo e o único disponível.
template <typename Acc, typename = void>
struct hasNnueEvalInt : std::false_type {};
template <typename Acc>
struct hasNnueEvalInt<Acc, std::void_t<decltype(nnueEvalInt(std::declval<const Acc&>(), 0))>>
    : std::true_type {};

template <typename Acc, typename Cache>
inline auto mcabResolvePending(Acc& ap, int side, Cache cache, int)
    -> decltype(resolvePending(ap, side, cache), void()) {
    resolvePending(ap, side, cache);
}
template <typename Acc, typename Cache>
inline void mcabResolvePending(Acc&, int, Cache, ...) {}

// Lazy progressive-widening legalization. Current rules.hpp exposes the
// staged one-wall predicate; older refs may not, so the fallback checks the
// candidate against their complete legal-move list and preserves compatibility.
template <typename StateT, typename MoveT>
inline auto mcabSingleWallLegal(const StateT& s, int player, const MoveT& m, int)
    -> decltype(isWallMoveLegal(s, player, (int)m.a, (int)m.b, (int)m.c), bool()) {
    return isWallMoveLegal(s, player, (int)m.a, (int)m.b, (int)m.c);
}

template <typename MoveT>
inline auto mcabMovesEqual(const MoveT& a, const MoveT& b, int) -> decltype((bool)(a == b)) {
    return a == b;
}
template <typename MoveT>
inline bool mcabMovesEqual(const MoveT& a, const MoveT& b, ...) {
    return std::memcmp(&a, &b, sizeof(MoveT)) == 0;
}

template <typename StateT, typename MoveT>
inline bool mcabSingleWallLegal(const StateT& s, int player, const MoveT& m, ...) {
    auto legal = legalMoves(s);
    for (const auto& candidate : legal) {
        if (mcabMovesEqual(candidate, m, 0)) return true;
    }
    (void)player;
    return false;
}

template <typename MoveT>
inline auto mcabIsWall(const MoveT& m, int) -> decltype((bool)m.isWall) {
    return (bool)m.isWall;
}
template <typename MoveT>
inline bool mcabIsWall(const MoveT&, ...) {
    return false;
}

// inv/ab-policy hooks (directions B and E). Both are optional engine
// capabilities detected by SFINAE, same pattern as mcabPathCache above:
// refs older than these methods compile fine and simply skip the feature.
//
// mcabSeedPolicyHistory -> Negamax::seedPolicyHistoryFromRoot (direction B):
// one policy pass over the game root copied into the AB history table.
//
// mcabRankRootMoves -> Negamax::rankRootMoves (direction E): score every
// root child with a shallow full-window AB search; the hybrid then keeps
// only the top-k children active at its root ("two-stage root").
template <typename Eng, typename StateT>
inline auto mcabSeedPolicyHistory(Eng& e, const StateT& s, int)
    -> decltype(e.seedPolicyHistoryFromRoot(s), void()) {
    e.seedPolicyHistoryFromRoot(s);
}
template <typename Eng, typename StateT>
inline void mcabSeedPolicyHistory(Eng&, const StateT&, ...) {}

template <typename Eng, typename StateT, typename MoveT, typename StatsT>
inline auto mcabRankRootMoves(Eng& e, const StateT& s, int depth, int budgetMs,
                               StatsT& stats, std::vector<std::pair<int, MoveT>>& out, int)
    -> decltype(e.rankRootMoves(s, depth, budgetMs, stats, out), void()) {
    e.rankRootMoves(s, depth, budgetMs, stats, out);
}
template <typename Eng, typename StateT, typename MoveT, typename StatsT>
inline void mcabRankRootMoves(Eng&, const StateT&, int, int, StatsT&,
                               std::vector<std::pair<int, MoveT>>&, ...) {}

template <typename MoveT, typename StateT, typename MoveListT>
inline auto mcabEnumerateCandidates(const StateT& s, int side, MoveListT& out, int)
    -> decltype(pawnStepMoves(s, side, out),
                out.push_back(MoveT::wall(0, 0, 0)),
                (bool)s.wallsH,
                void()) {
    pawnStepMoves(s, side, out);
    if (s.wallsLeft[side] > 0) {
        for (int orientation = 0; orientation < 2; orientation++) {
            for (int r = 0; r < MCAB_WALL_GRID; r++) {
                for (int c = 0; c < MCAB_WALL_GRID; c++) {
                    if (mcabWallSlotAvailable(s, orientation, r, c))
                        out.push_back(MoveT::wall(orientation, r, c));
                }
            }
        }
    }
}

template <typename MoveT, typename StateT, typename MoveListT>
inline void mcabEnumerateCandidates(const StateT& s, int /*side*/, MoveListT& out, ...) {
    out = legalMoves(s);
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
    MoveListT moves;               // currently active, already-legal moves
    std::vector<float> P;         // prior por lance, mesmo índice de `moves`
    std::vector<float> N;         // visitas por aresta
    std::vector<float> W;         // soma (AvgBlend) or backed value (MinimaxHard)
    std::vector<int32_t> child;   // índice no pool, -1 = não expandido
    int activeMoves = 0;          // progressive-widening prefix currently unpruned
    // Progressive-widening-only storage. Production has PW disabled, so an
    // inline MoveList here made every node pay for a second 256-move buffer
    // that was never touched. Allocate it only on the PW path.
    std::unique_ptr<MoveListT> candidateMoves;
    std::vector<float> candidateP;
    std::vector<size_t> activeCandidateIndices;
    size_t nextCandidate = 0;
    int totalN = 0;
    int graphDepth = -1;           // root-relative ply for safe DAG sharing
    uint32_t graphN = 0;           // MCGS node visits, own side-to-move frame
    double graphW = 0.0;           // MCGS node value sum, own frame
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
    // por-aresta que cada nó aloca fora de linha (moves/P/N/W/child plus the
    // lazy candidate priors). Não
    // conta a fragmentação do alocador -- é uma estimativa para o
    // benchmark de custo da Fase 4, não contabilidade exata.
    size_t approxTreeBytes() const {
        size_t bytes = pool.capacity() * sizeof(NodeT);
        for (const NodeT& n : pool) {
            bytes += n.P.capacity() * sizeof(float);
            bytes += n.N.capacity() * sizeof(float);
            bytes += n.W.capacity() * sizeof(float);
            bytes += n.child.capacity() * sizeof(int32_t);
            bytes += n.candidateP.capacity() * sizeof(float);
            if (n.candidateMoves) bytes += sizeof(MoveListT);
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
        if (params.evalCache) {
            if (policyEvalCache.empty()) policyEvalCache.resize(kEvalCacheEntries);
            if (valueEvalCache.empty()) valueEvalCache.resize(kEvalCacheEntries);
            ++evalCacheGeneration;
            if (evalCacheGeneration == 0) {
                for (auto& entry : policyEvalCache) entry.generation = 0;
                for (auto& entry : valueEvalCache) entry.generation = 0;
                evalCacheGeneration = 1;
            }
        }

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

        // inv/ab-policy, direction E: optional AB pre-ranking of the root
        // children, before any tree work. Runs its own resetOrderingState()
        // internally; TT/BFS caches stay shared with the tree phase. The
        // ranking consumes part of THIS move's time budget: whatever it
        // actually spent (up to abPrefilterTimeFrac of timeBudgetMs) is
        // deducted from the tree phase below, so the whole move stays
        // inside timeBudgetMs.
        std::vector<std::pair<int, MoveT>> ranked;
        int treeBudgetMs = timeBudgetMs;
        if (params.abPrefilterDepth > 0 && params.abPrefilterTopK > 0) {
            SearchStatsT preStats{};
            int preMs = timeBudgetMs > 0
                            ? (int)((double)timeBudgetMs * params.abPrefilterTimeFrac)
                            : 0;
            auto preT0 = std::chrono::steady_clock::now();
            mcabRankRootMoves(engine, root, params.abPrefilterDepth, preMs, preStats, ranked, 0);
            if (timeBudgetMs > 0) {
                auto spent = std::chrono::duration_cast<std::chrono::milliseconds>(
                                 std::chrono::steady_clock::now() - preT0)
                                 .count();
                // Floor of 5ms: a starved tree phase would expand nothing
                // and pick by prior alone.
                long long left = (long long)timeBudgetMs - (long long)spent;
                treeBudgetMs = (int)std::max<long long>(5, left);
            }
        }

        // inv/ab-policy, direction B: seed the AB history table from the
        // game root (one policy pass per move) so searchLeaf-style leaves
        // inherit policy guidance. After the ranking pass on purpose: both
        // reset the ordering state.
        mcabSeedPolicyHistory(engine, root, 0);

        if (params.clearTTPerMove) engine.clearTT();

        // inv/endgame-wander: decide the leaf rule once per move, from the
        // wall stock of the side to move at the root, so that the whole tree
        // uses one leaf depth.
        endgameLeafActive =
            params.endgameMoverWallThreshold >= 0 &&
            (int)root.wallsLeft[root.turn] <= params.endgameMoverWallThreshold;

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

        int budget = effectiveNodeBudget(params, timeBudgetMs);
        mstats.effectiveNodeBudget = budget;
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
            normalizeGraphDepths();
            mstats.treeReused = true;
            mstats.reusedNodes = (int)pool.size();
        } else {
            pool.clear();
            NodeT rootNode;
            rootNode.state = root;
            rootNode.side = root.turn;
            rootNode.graphDepth = 0;
            pool.push_back(std::move(rootNode));
        }
        rebuildTranspositionIndex();
        // With adaptive clocks, timeBudgetMs is the hard maximum and can be
        // ~3x optimum. Reserve only the expected optimum working set up front;
        // vector growth remains safe because the tree stores indices, not
        // pointers.
        int reserveBudget = budget;
        if (params.adaptiveTime && params.adaptiveOptimumMs > 0)
            reserveBudget = std::min(budget, effectiveNodeBudget(params, params.adaptiveOptimumMs));
        pool.reserve(pool.size() + (size_t)reserveBudget + 1);

        mcabAccStack[0] = buildAccPairRoot(root, mcabPathCache(engine, 0));

        // Passo 4 (Seção 5): expande a raiz se necessário.
        int wRoot = winner(root);
        if (wRoot != -1) {
            pool[0].terminal = true;
            pool[0].terminalScore = (wRoot == root.turn) ? MCAB_WIN_SCORE : -MCAB_WIN_SCORE;
        } else if (!pool[0].expanded) {
            expandNode(0, /*depthInTree=*/0, mstats);
        }

        // inv/ab-policy, direction E: restrict the active root edges to the
        // top-k children of the AB pre-ranking. Applied to a fresh AND to a
        // reused root (each move re-ranks, so yesterday's filter never
        // persists). Edge stats (N/W/child links) of kept moves survive;
        // dropped subtrees stay in the pool until the next compaction,
        // bounded by the same 2x nodeBudget rule as Seção 8.1.
        if (!ranked.empty() && !pool[0].terminal && params.abPrefilterTopK > 0) {
            filterRootTopK(pool[0], ranked, params.abPrefilterTopK);
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
        haveLeafDeadline = (treeBudgetMs > 0);
        if (haveLeafDeadline) leafDeadline = t0 + std::chrono::milliseconds(treeBudgetMs);

        const int optimumMs =
            params.adaptiveTime && params.adaptiveOptimumMs > 0 && treeBudgetMs > 0
                ? std::clamp(params.adaptiveOptimumMs, 1, treeBudgetMs)
                : treeBudgetMs;
        const int earliestStopMs =
            params.adaptiveTime && optimumMs > 0
                ? std::clamp((int)std::lround(params.adaptiveMinFactor * optimumMs), 1, treeBudgetMs)
                : treeBudgetMs;
        int adaptiveLeader = -1;
        long long adaptiveLeaderChangedMs = 0;

        while (mstats.nodesExpanded < budget) {
            long long elapsedMs = 0;
            if (treeBudgetMs > 0) {
                elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(
                                std::chrono::steady_clock::now() - t0)
                                .count();
                if (elapsedMs >= treeBudgetMs) break;

                if (params.adaptiveTime && optimumMs > 0 &&
                    mstats.simulations >= params.adaptiveMinSimulations &&
                    elapsedMs >= earliestStopMs) {
                    RootTimeSignal sig = rootTimeSignal(pool[0]);
                    if (sig.valid) {
                        double stableFraction =
                            (double)std::max<long long>(0, elapsedMs - adaptiveLeaderChangedMs) /
                            (double)std::max(1, optimumMs);
                        double factor = adaptiveTimeFactor(
                            sig.visitRatio, sig.qGap, stableFraction, params);
                        int targetMs = std::clamp(
                            (int)std::lround((double)optimumMs * factor),
                            earliestStopMs, treeBudgetMs);
                        mstats.adaptiveTargetMs = targetMs;
                        mstats.adaptiveVisitRatio = sig.visitRatio;
                        mstats.adaptiveQGap = sig.qGap;
                        if (elapsedMs >= targetMs) {
                            mstats.adaptiveTimeStop = true;
                            break;
                        }
                    }
                }
            }
            if (pool[0].terminal) break;  // raiz já resolvida (ex.: vitória em 0 lances -- não deveria ocorrer)
            runSimulation(engine, stats, mstats);

            if (params.adaptiveTime && (mstats.simulations & 63LL) == 0) {
                int leader = rootVisitLeader(pool[0]);
                long long nowMs = treeBudgetMs > 0
                    ? std::chrono::duration_cast<std::chrono::milliseconds>(
                          std::chrono::steady_clock::now() - t0).count()
                    : 0;
                if (adaptiveLeader < 0) {
                    adaptiveLeader = leader;
                    adaptiveLeaderChangedMs = nowMs;
                } else if (leader >= 0 && leader != adaptiveLeader) {
                    adaptiveLeader = leader;
                    adaptiveLeaderChangedMs = nowMs;
                    ++mstats.adaptiveLeaderChanges;
                }
            }
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
    static constexpr size_t kEvalCacheWays = 4;
    static constexpr size_t kEvalCacheSets = 1024;
    static constexpr size_t kEvalCacheEntries = kEvalCacheWays * kEvalCacheSets;

    struct PolicyEvalCacheEntry {
        uint64_t key = 0;
        uint32_t generation = 0;
        std::array<float, PolicyDim> policy{};
    };
    struct ValueEvalCacheEntry {
        uint64_t key = 0;
        uint32_t generation = 0;
        int score = 0;
    };

    std::vector<PolicyEvalCacheEntry> policyEvalCache;
    std::vector<ValueEvalCacheEntry> valueEvalCache;
    uint32_t evalCacheGeneration = 0;

    size_t evalCacheSet(uint64_t key) const {
        uint64_t mixed = key ^ (key >> 33) ^ (key >> 17);
        return (size_t)mixed & (kEvalCacheSets - 1);
    }

    PolicyEvalCacheEntry* findPolicyEvalCache(uint64_t key) {
        if (!params.evalCache || key == 0 || policyEvalCache.empty()) return nullptr;
        size_t base = evalCacheSet(key) * kEvalCacheWays;
        for (size_t way = 0; way < kEvalCacheWays; ++way) {
            auto& entry = policyEvalCache[base + way];
            if (entry.generation == evalCacheGeneration && entry.key == key) return &entry;
        }
        return nullptr;
    }

    PolicyEvalCacheEntry& storePolicyEvalCache(uint64_t key, McabStats& mstats) {
        size_t base = evalCacheSet(key) * kEvalCacheWays;
        for (size_t way = 0; way < kEvalCacheWays; ++way) {
            auto& entry = policyEvalCache[base + way];
            if (entry.generation == evalCacheGeneration && entry.key == key) return entry;
            if (entry.generation != evalCacheGeneration) {
                entry.key = key;
                entry.generation = evalCacheGeneration;
                return entry;
            }
        }
        size_t victim = (size_t)((key >> 10) & (kEvalCacheWays - 1));
        auto& entry = policyEvalCache[base + victim];
        ++mstats.evalCachePolicyEvictions;
        entry.key = key;
        entry.generation = evalCacheGeneration;
        return entry;
    }

    ValueEvalCacheEntry* findValueEvalCache(uint64_t key) {
        if (!params.evalCache || key == 0 || valueEvalCache.empty()) return nullptr;
        size_t base = evalCacheSet(key) * kEvalCacheWays;
        for (size_t way = 0; way < kEvalCacheWays; ++way) {
            auto& entry = valueEvalCache[base + way];
            if (entry.generation == evalCacheGeneration && entry.key == key) return &entry;
        }
        return nullptr;
    }

    ValueEvalCacheEntry& storeValueEvalCache(uint64_t key, McabStats& mstats) {
        size_t base = evalCacheSet(key) * kEvalCacheWays;
        for (size_t way = 0; way < kEvalCacheWays; ++way) {
            auto& entry = valueEvalCache[base + way];
            if (entry.generation == evalCacheGeneration && entry.key == key) return entry;
            if (entry.generation != evalCacheGeneration) {
                entry.key = key;
                entry.generation = evalCacheGeneration;
                return entry;
            }
        }
        size_t victim = (size_t)((key >> 18) & (kEvalCacheWays - 1));
        auto& entry = valueEvalCache[base + victim];
        ++mstats.evalCacheValueEvictions;
        entry.key = key;
        entry.generation = evalCacheGeneration;
        return entry;
    }

    std::vector<NodeT> pool;
    std::unordered_multimap<uint64_t, int32_t> transpositionIndex;
    std::vector<AccPairT> mcabAccStack;   // Seção 4.3.3 -- pilha por caminho de descida
    RepTblT localRepTbl;                  // cópia mutável de gameHistory, 1x por chooseMoveMCAB (ver Seção 5, negamax/searchLeaf)
    std::mt19937 rng;                     // ruído de Dirichlet (Seção 9) -- por instância, nunca compartilhado entre threads
    bool noiseSeeded = false;
    // Teto de tempo da chamada corrente, propagado para dentro de cada
    // engine.searchLeaf (ver evaluateLeaf). Só válido enquanto
    // haveLeafDeadline == true, o que só ocorre com timeBudgetMs > 0.
    std::chrono::steady_clock::time_point leafDeadline;
    bool haveLeafDeadline = false;
    // Resolved once per chooseMoveMCAB from the root wall stock. True means
    // that leaves use params.endgameLeafDepth (see
    // McabParams::endgameMoverWallThreshold).
    bool endgameLeafActive = false;

    // NOTA: buildAccPairRoot/makeChildAccPair recebem um `PlayerPathCacheTable*`
    // opcional para cachear BFS de distância entre chamadas -- o mesmo
    // cache que Negamax::xdistCache usa internamente na busca AB.
    // MCABSearch não nomeia esse tipo (Seção 4.2); o ponteiro vem de
    // mcabPathCache(engine), que devolve engine.pathCache() quando o tipo
    // o expõe e nullptr caso contrário.

    struct PathEdge {
        int nodeIdx;
        int edgeIdx;
    };

    // Per-search descent storage. Avoid dynamic thread-local storage: MinGW
    // can report heap corruption while destroying a thread-local vector when
    // a self-play worker exits.
    std::vector<PathEdge> simulationPath;

    void rebuildTranspositionIndex() {
        transpositionIndex.clear();
        if (!params.transpositionGraph) return;
        transpositionIndex.reserve(pool.size() * 2 + 1);
        for (size_t i = 0; i < pool.size(); ++i) {
            uint64_t key = mcabGraphKey(pool[i].state, 0);
            if (key != 0) transpositionIndex.emplace(key, (int32_t)i);
        }
    }

    void normalizeGraphDepths() {
        if (pool.empty()) return;
        for (NodeT& n : pool) n.graphDepth = -1;
        std::queue<int32_t> q;
        pool[0].graphDepth = 0;
        q.push(0);
        while (!q.empty()) {
            int32_t idx = q.front();
            q.pop();
            int nextDepth = pool[(size_t)idx].graphDepth + 1;
            for (int32_t childIdx : pool[(size_t)idx].child) {
                if (childIdx < 0) continue;
                NodeT& child = pool[(size_t)childIdx];
                if (child.graphDepth < 0) {
                    child.graphDepth = nextDepth;
                    q.push(childIdx);
                }
            }
        }
    }

    bool graphQCorrectionActive() const {
        return params.transpositionGraph && params.graphQCorrection &&
               params.backupMode == BackupMode::AvgBlend &&
               params.leafDepth == 0 && !endgameLeafActive;
    }

    int findGraphTransposition(const StateT& state, int depth, McabStats& mstats) const {
        if (!params.transpositionGraph) return -1;
        uint64_t key = mcabGraphKey(state, 0);
        if (key == 0) return -1;
        ++mstats.transpositionLookups;
        auto range = transpositionIndex.equal_range(key);
        for (auto it = range.first; it != range.second; ++it) {
            int32_t idx = it->second;
            if (idx < 0 || idx >= (int32_t)pool.size()) continue;
            const NodeT& candidate = pool[(size_t)idx];
            // Production DAG shares only same-ply states. Q-corrected MCGS is
            // path-safe instead, so it may merge the same position at any ply.
            if (!graphQCorrectionActive() && candidate.graphDepth != depth) continue;
            if (!mcabGraphStateEqual(candidate.state, state, 0)) continue;
            ++mstats.transpositionHits;
            return idx;
        }
        return -1;
    }

    static bool pathContainsNode(const std::vector<PathEdge>& path, int nodeIdx) {
        for (const auto& pe : path)
            if (pe.nodeIdx == nodeIdx) return true;
        return false;
    }


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
    // ficou fora do caminho jogado). Se a subárvore herdada for maior que
    // o orçamento, preserva até `budget` nós priorizando os filhos mais
    // visitados, em vez de descartar toda a árvore reutilizável.
    bool compactTo(int rootIdx, int budget) {
        if (rootIdx < 0 || rootIdx >= (int)pool.size()) return false;
        budget = std::max(1, budget);

        std::vector<int32_t> remap(pool.size(), -1);
        std::vector<uint8_t> queued(pool.size(), 0);
        std::vector<int32_t> order;
        order.reserve(std::min(pool.size(), (size_t)budget));

        // Retain the hottest inherited nodes first. This avoids discarding
        // the entire reused tree just because the reachable subtree is larger
        // than the reuse budget. Ties are deterministic by old pool index.
        using FrontierItem = std::pair<int, int32_t>; // {visits, -oldIdx}
        std::priority_queue<FrontierItem> frontier;

        auto enqueueChildren = [&](int32_t oldIdx) {
            const NodeT& n = pool[(size_t)oldIdx];
            for (int32_t childIdx : n.child) {
                if (childIdx < 0 || queued[(size_t)childIdx]) continue;
                queued[(size_t)childIdx] = 1;
                frontier.emplace(pool[(size_t)childIdx].totalN, -childIdx);
            }
        };

        remap[(size_t)rootIdx] = 0;
        queued[(size_t)rootIdx] = 1;
        order.push_back(rootIdx);
        enqueueChildren(rootIdx);

        while ((int)order.size() < budget && !frontier.empty()) {
            int32_t oldIdx = -frontier.top().second;
            frontier.pop();
            if (remap[(size_t)oldIdx] >= 0) continue;
            remap[(size_t)oldIdx] = (int32_t)order.size();
            order.push_back(oldIdx);
            enqueueChildren(oldIdx);
        }

        std::vector<NodeT> compacted;
        compacted.reserve(order.size());
        for (int32_t oldIdx : order)
            compacted.push_back(std::move(pool[(size_t)oldIdx]));

        for (NodeT& n : compacted) {
            for (int32_t& childIdx : n.child) {
                if (childIdx < 0) continue;
                int32_t mapped = remap[(size_t)childIdx];
                childIdx = mapped >= 0 ? mapped : -1;
            }
        }
        pool.swap(compacted);
        return true;
    }

    // inv/ab-policy, direction E: keep only the moves of `ranked` that fit
    // in the first `topK` ranked positions (moves absent from the ranking
    // -- e.g. unsearched when the AB pass ran out of time -- are dropped,
    // which can leave fewer than topK edges; that is intended). Priors are
    // renormalized over the survivors so cPuct keeps its meaning.
    void filterRootTopK(NodeT& r, const std::vector<std::pair<int, MoveT>>& ranked, int topK) {
        if ((int)r.moves.size() <= topK || ranked.empty()) return;
        std::vector<MoveT> keep;
        for (size_t i = 0; i < ranked.size() && (int)keep.size() < topK; i++)
            keep.push_back(ranked[i].second);

        MoveListT oldMoves = r.moves;
        std::vector<float> oldP = std::move(r.P);
        std::vector<float> oldN = std::move(r.N);
        std::vector<float> oldW = std::move(r.W);
        std::vector<int32_t> oldChild = std::move(r.child);

        r.moves = MoveListT{};
        r.P.clear();
        r.N.clear();
        r.W.clear();
        r.child.clear();

        for (size_t e = 0; e < oldMoves.size(); e++) {
            bool inKeep = false;
            for (const MoveT& k : keep) {
                if (mcabMovesEqual(oldMoves[e], k, 0)) { inKeep = true; break; }
            }
            if (!inKeep) continue;
            r.moves.push_back(oldMoves[e]);
            r.P.push_back(oldP.empty() ? 0.f : oldP[e]);
            r.N.push_back(oldN.empty() ? 0.f : oldN[e]);
            r.W.push_back(oldW.empty() ? 0.f : oldW[e]);
            r.child.push_back(oldChild.empty() ? -1 : oldChild[e]);
        }
        r.activeMoves = (int)r.moves.size();

        float sum = 0.f;
        for (float p : r.P) sum += p;
        if (sum > 0.f) {
            for (float& p : r.P) p /= sum;
        }
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
        if (endgameLeafActive) return params.endgameLeafDepth;
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

        AccPairT rootAcc = buildAccPairRoot(root, mcabPathCache(engine, 0));

        double bestQ = -std::numeric_limits<double>::infinity();
        int bestIdx = 0;
        for (size_t i = 0; i < moves.size(); i++) {
            StateT childState = applyMove(root, moves[i]);
            AccPairT parentCopy = rootAcc;  // makeChildAccPair pode resolver pending em `parent`
            AccPairT seedAcc{};
            makeChildAccPair(parentCopy, seedAcc, root, moves[i], mcabPathCache(engine, 0));

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
    int wideningLimit(int moveCount, int visits) const {
        if (!params.progressiveWidening) return moveCount;
        double raw = (double)params.wideningInitialMoves +
                     params.wideningCoefficient *
                         std::pow((double)std::max(0, visits), params.wideningExponent);
        int limit = (int)std::ceil(raw);
        limit = std::max(limit, params.wideningInitialMoves);
        return std::min(moveCount, std::max(1, limit));
    }

    void activateWidening(NodeT& node, int desired) {
        size_t oldSize = node.moves.size();
        while ((int)node.moves.size() < desired &&
               node.candidateMoves &&
               node.nextCandidate < node.candidateMoves->size()) {
            size_t i = node.nextCandidate++;
            const MoveT& candidate = (*node.candidateMoves)[i];
            bool legal = !mcabIsWall(candidate, 0) ||
                         mcabSingleWallLegal(node.state, node.side, candidate, 0);
            if (!legal) continue;

            node.moves.push_back(candidate);
            node.P.push_back(node.candidateP[i]);
            node.N.push_back(0.f);
            node.W.push_back(0.f);
            node.child.push_back(-1);
            node.activeCandidateIndices.push_back(i);
        }

        // candidateP is the softmax over the complete cheap candidate list.
        // Recompute the active-prefix normalization from those raw priors;
        // renormalizing node.P in place would compound the previous scale
        // every time a new candidate is admitted.
        if (node.moves.size() > oldSize && !node.noised) {
            float activePriorSum = 0.f;
            for (size_t candidateIndex : node.activeCandidateIndices)
                activePriorSum += node.candidateP[candidateIndex];
            if (activePriorSum > 0.f) {
                for (size_t i = 0; i < node.P.size(); i++)
                    node.P[i] = node.candidateP[node.activeCandidateIndices[i]] / activePriorSum;
            }
        }
        node.activeMoves = (int)node.moves.size();
    }

    void updateWidening(NodeT& node) {
        if (!params.progressiveWidening) {
            node.activeMoves = (int)node.moves.size();
            return;
        }
        int desired = wideningLimit(node.candidateMoves ? (int)node.candidateMoves->size() : 0,
                                  node.totalN);
        activateWidening(node, desired);
    }

    void policyOutputForNode(const NodeT& node, int depthInTree,
                             std::array<float, PolicyDim>& out, McabStats& mstats) {
        uint64_t key = params.evalCache ? mcabEvalStateKey(node.state, 0) : 0;
        if (key != 0) {
            if (auto* entry = findPolicyEvalCache(key)) {
                out = entry->policy;
                ++mstats.evalCachePolicyHits;
                return;
            }
        }
        forwardPolicyQuant(mcabAccStack[depthInTree].acc[node.side], out);
        if (key != 0) {
            auto& entry = storePolicyEvalCache(key, mstats);
            entry.policy = out;
            ++mstats.evalCachePolicyMisses;
        }
    }

    void expandNode(int idx, int depthInTree, McabStats& mstats) {
        NodeT& node = pool[idx];

        if (!params.progressiveWidening) {
            node.moves = legalMoves(node.state);
            size_t nm = node.moves.size();
            node.P.assign(nm, 0.f);
            node.N.assign(nm, 0.f);
            node.W.assign(nm, 0.f);
            node.child.assign(nm, -1);

            if (nm > 0) {
                std::array<float, PolicyDim> policyOut{};
                policyOutputForNode(node, depthInTree, policyOut, mstats);
                // perf/speed-elo-100: era std::vector<float> alocado por
                // expansao; nm <= 131 < PolicyDim(=209), entao um buffer de
                // pilha elimina o malloc do caminho quente.
                float logits[PolicyDim];
                float maxLogit = -std::numeric_limits<float>::infinity();
                for (size_t i = 0; i < nm; i++) {
                    logits[i] = policyLogitForMove(policyOut, node.moves[i], node.side);
                    maxLogit = std::max(maxLogit, logits[i]);
                }
                float sumExp = 0.f;
                for (size_t i = 0; i < nm; i++) {
                    node.P[i] = mcabFastExp(logits[i] - maxLogit);   // perf: era std::exp
                    sumExp += node.P[i];
                }
                if (sumExp > 0.f) {
                    for (size_t i = 0; i < nm; i++) node.P[i] /= sumExp;
                } else {
                    for (size_t i = 0; i < nm; i++) node.P[i] = 1.f / (float)nm;
                }
            }
            node.activeMoves = (int)nm;
        } else {
            // Cheap candidate enumeration: pawn moves are already legal;
            // wall candidates only pay the local slot-overlap check here.
            // The path-preserving legality test is deferred until a
            // candidate enters the active prefix in activateWidening().
            node.candidateMoves = std::make_unique<MoveListT>();
            mcabEnumerateCandidates<MoveT>(node.state, node.side, *node.candidateMoves, 0);

            size_t nc = node.candidateMoves->size();
            node.candidateP.assign(nc, 0.f);
            if (nc > 0) {
                std::array<float, PolicyDim> policyOut{};
                policyOutputForNode(node, depthInTree, policyOut, mstats);
                std::vector<float> logits(nc);
                for (size_t i = 0; i < nc; i++)
                    logits[i] = policyLogitForMove(policyOut, (*node.candidateMoves)[i], node.side);

                std::vector<size_t> order(nc);
                std::iota(order.begin(), order.end(), (size_t)0);
                std::stable_sort(order.begin(), order.end(),
                                 [&](size_t a, size_t b) { return logits[a] > logits[b]; });
                auto orderedCandidates = std::make_unique<MoveListT>();
                std::vector<float> orderedLogits;
                for (size_t i : order) {
                    orderedCandidates->push_back((*node.candidateMoves)[i]);
                    orderedLogits.push_back(logits[i]);
                }
                node.candidateMoves = std::move(orderedCandidates);
                logits = std::move(orderedLogits);

                float maxLogit = *std::max_element(logits.begin(), logits.end());
                float sumExp = 0.f;
                // perf: exp computado UMA vez por elemento (era 2x: soma +
                // normalização) e via mcabFastExp
                for (float logit : logits) sumExp += mcabFastExp(logit - maxLogit);
                node.candidateP.resize(nc);
                for (size_t i = 0; i < nc; i++)
                    node.candidateP[i] = sumExp > 0.f ? mcabFastExp(logits[i] - maxLogit) / sumExp
                                                      : 1.f / (float)nc;
            }
            node.moves = MoveListT{};
            node.P.clear();
            node.N.clear();
            node.W.clear();
            node.child.clear();
            node.activeCandidateIndices.clear();
            node.nextCandidate = 0;
            node.activeMoves = 0;
        }

        node.expanded = true;
        updateWidening(node);
        mstats.nodesExpanded++;
    }

    // ---------------------------------------------------------------
    // Seção 5.1 -- Seleção PUCT
    // ---------------------------------------------------------------
    // score(a) = Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
    // Q(s,a) is W/N for AvgBlend and the hard-backed W value for MinimaxHard.
    // Unvisited edges use FPU = Q(pai) - fpuReduction.
    double edgeQ(const NodeT& node, size_t edge) const {
        if (params.backupMode == BackupMode::AvgBlend)
            return (double)node.W[edge] / (double)node.N[edge];
        return (double)node.W[edge];
    }

    double nodeQ(const NodeT& node) const {
        if (params.backupMode == BackupMode::AvgBlend) {
            if (node.totalN <= 0) return 0.5;
            double sumW = 0.0;
            for (float w : node.W) sumW += w;
            return sumW / (double)node.totalN;
        }
        double best = 0.5;
        bool visited = false;
        for (size_t i = 0; i < node.N.size(); i++) {
            if (node.N[i] <= 0.f) continue;
            best = std::max(best, edgeQ(node, i));
            visited = true;
        }
        return visited ? best : 0.5;
    }

    struct RootTimeSignal {
        bool valid = false;
        int leader = -1;
        int second = -1;
        double visitRatio = 0.0;
        double qGap = 0.0;
    };

    int rootVisitLeader(const NodeT& node) const {
        size_t nm = (size_t)std::min(node.activeMoves, (int)node.moves.size());
        if (nm == 0) return -1;
        int best = 0;
        for (size_t i = 1; i < nm; ++i)
            if (node.N[i] > node.N[(size_t)best]) best = (int)i;
        return best;
    }

    RootTimeSignal rootTimeSignal(const NodeT& node) const {
        RootTimeSignal s;
        size_t nm = (size_t)std::min(node.activeMoves, (int)node.moves.size());
        if (nm <= 1) return s;
        s.leader = rootVisitLeader(node);
        if (s.leader < 0 || node.N[(size_t)s.leader] <= 0.f) return s;
        for (size_t i = 0; i < nm; ++i) {
            if ((int)i == s.leader || node.N[i] <= 0.f) continue;
            if (s.second < 0 || node.N[i] > node.N[(size_t)s.second])
                s.second = (int)i;
        }
        if (s.second < 0 || node.N[(size_t)s.second] <= 0.f) return s;
        s.visitRatio = (double)node.N[(size_t)s.leader] /
                       (double)node.N[(size_t)s.second];
        s.qGap = edgeQ(node, (size_t)s.leader) -
                 edgeQ(node, (size_t)s.second);
        s.valid = true;
        return s;
    }

    int selectChildPUCT(const NodeT& node) const {
        size_t nm = (size_t)std::min(node.activeMoves, (int)node.moves.size());
        if (nm == 0) return -1;

        double parentQ = nodeQ(node);
        double fpu = parentQ - params.fpuReduction;

        double sqrtN = std::sqrt((double)std::max(0, node.totalN));
        int best = -1;
        double bestScore = -std::numeric_limits<double>::infinity();
        for (size_t i = 0; i < nm; i++) {
            double q = node.N[i] > 0.0 ? edgeQ(node, i) : fpu;
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
        // Search is sequential within one MCABSearch. Reuse the descent buffer
        // across simulations instead of paying allocator traffic every visit.
        std::vector<PathEdge>& path = simulationPath;
        path.clear();
        const size_t need = (size_t)params.maxTreeDepth + 2;
        if (path.capacity() < need) path.reserve(need);

        int curIdx = 0;
        int depth = 0;  // ply desde a raiz; indexa mcabAccStack

        while (true) {
            NodeT& node = pool[curIdx];

            if (node.expanded) updateWidening(node);

            if (node.terminal) {
                double leafQ = scoreToQ(node.terminalScore, params.scoreScale);
                if (graphQCorrectionActive()) backupGraph(path, curIdx, leafQ, mstats);
                else backup(path, leafQ, mstats);
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
                if (graphQCorrectionActive()) backupGraph(path, curIdx, leafQ, mstats);
                else backup(path, leafQ, mstats);
                return;
            }

            int e = selectChildPUCT(node);
            if (e < 0) {
                // Nó expandido sem lances -- não deveria ocorrer em
                // Quoridor não-terminal, mas não crasha: trata como
                // neutro.
                if (graphQCorrectionActive()) backupGraph(path, curIdx, 0.5, mstats);
                else backup(path, 0.5, mstats);
                return;
            }
            path.push_back({curIdx, e});

            StateT beforeState = node.state;  // cópia -- `node` pode ser invalidada por pool.push_back abaixo
            MoveT mv = node.moves[e];
            int parentSide = node.side;

            int childIdx = node.child[e];
            if (childIdx == -1) {
                StateT childState = applyMove(beforeState, mv);
                childIdx = findGraphTransposition(childState, depth + 1, mstats);
                if (childIdx < 0)
                    childIdx = createChild(childState, depth + 1);
                pool[curIdx].child[e] = childIdx;  // reindexado -- `node` pode ter sido invalidada
            }

            // Cross-ply MCGS permits global cycles, but never follows one on
            // the current descent. Score that refused edge as draw-ish and
            // back up without updating the repeated node twice.
            if (graphQCorrectionActive() && pathContainsNode(path, childIdx)) {
                ++mstats.graphCycleStops;
                backupGraphCycle(path, mstats);
                return;
            }

            makeChildAccPair(mcabAccStack[depth], mcabAccStack[depth + 1], beforeState, mv,
                             mcabPathCache(engine, 0));
            (void)parentSide;

            curIdx = childIdx;
            depth++;
        }
    }

    // Cria o MCABNode do estado `s` (ainda não expandido) e o insere no
    // pool. Marca terminal=true imediatamente se `s` já é posição de fim
    // de jogo (Seção 5 passo b: "Se terminal, marca terminal=true e usa
    // o valor exato, sem chamar searchLeaf").
    int createChild(const StateT& s, int graphDepth) {
        NodeT node;
        node.state = s;
        node.side = s.turn;
        node.graphDepth = graphDepth;
        int w = winner(s);
        if (w != -1) {
            node.terminal = true;
            node.terminalScore = (w == s.turn) ? MCAB_WIN_SCORE : -MCAB_WIN_SCORE;
        }
        pool.push_back(std::move(node));
        int idx = (int)pool.size() - 1;
        if (params.transpositionGraph) {
            uint64_t key = mcabGraphKey(pool[(size_t)idx].state, 0);
            if (key != 0) transpositionIndex.emplace(key, idx);
        }
        return idx;
    }

    // Avalia um nó recém-expandido e não-terminal. Com leafDepth==0 (o
    // default de produção) usa nnueEvalInt no acumulador já incremental
    // -- sem searchLeaf e sem quiescência de muro. leafDepth>0 continua
    // sendo uma busca AB real (Seção 5.2). Devolve Q do ponto de vista
    // do PRÓPRIO nó (node.side). O backup (Seção 5.3) inverte a
    // perspectiva subindo a árvore.
    double evaluateLeaf(Eng& engine, NodeT& node, int depthInTree, int branchVisits,
                         SearchStatsT& stats, McabStats& mstats) {
        if (node.moves.empty()) return 0.5;  // sem lances e não-terminal: não deveria ocorrer; neutro defensivo
        int leafDepth = effectiveLeafDepth(branchVisits);

        if (leafDepth <= 0) {
            if constexpr (hasNnueEvalInt<AccPairT>::value) {
                AccPairT& ap = mcabAccStack[depthInTree];
                // Always resolve pending accumulator work first: cache lookup must
                // not alter descendant accumulator state or search semantics.
                mcabResolvePending(ap, node.side, mcabPathCache(engine, 0), 0);
                uint64_t key = params.evalCache ? mcabEvalStateKey(node.state, 0) : 0;
                int score = 0;
                if (key != 0) {
                    if (auto* entry = findValueEvalCache(key)) {
                        score = entry->score;
                        ++mstats.evalCacheValueHits;
                    } else {
                        score = nnueEvalInt(ap, node.side);
                        auto& stored = storeValueEvalCache(key, mstats);
                        stored.score = score;
                        ++mstats.evalCacheValueMisses;
                    }
                } else {
                    score = nnueEvalInt(ap, node.side);
                }
                mstats.leafSearches++;
                mstats.leafDepthSum += 0;
                return scoreToQ(score, params.scoreScale);
            }
        }

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
    // Seção 5.3 -- Backup
    // ---------------------------------------------------------------
    // `leafQ` é do ponto de vista do nó recém-avaliado (quem vai jogar
    // ali). Subindo a árvore, cada nível corresponde a uma troca de
    // mover -- inverte o sinal a cada passo (Q_pai = 1 - Q_filho, análogo
    // a negamax) e atualiza N/W da aresta correspondente. MinimaxHard first
    // updates the edge value, then recomputes the parent node's max value;
    // that node value is what gets inverted for the next ancestor. This is
    // the actual minimax backup, rather than taking a max over unrelated
    // leaf samples on the same edge. AvgBlend keeps the standard Monte Carlo
    // sum, so Q=W/N is the mean of all visits.
    void backupGraph(const std::vector<PathEdge>& path, int leafIdx,
                     double leafQ, McabStats& mstats) {
        double v = leafQ;
        double raw = leafQ;
        int childIdx = leafIdx;
        for (int i = (int)path.size() - 1; i >= 0; --i) {
            NodeT& child = pool[(size_t)childIdx];
            child.graphN += 1;
            child.graphW += v;
            const double mean = child.graphW / (double)child.graphN;
            const double upChild = params.graphLeafMix * raw +
                                   (1.0 - params.graphLeafMix) * mean;
            NodeT& parent = pool[(size_t)path[(size_t)i].nodeIdx];
            const int e = path[(size_t)i].edgeIdx;
            const double parentV = 1.0 - upChild;
            parent.N[(size_t)e] += 1.f;
            parent.W[(size_t)e] += (float)parentV;
            parent.totalN += 1;
            v = parentV;
            raw = 1.0 - raw;
            childIdx = path[(size_t)i].nodeIdx;
        }
        NodeT& root = pool[(size_t)childIdx];
        root.graphN += 1;
        root.graphW += v;
        mstats.simulations++;
    }

    void backupGraphCycle(const std::vector<PathEdge>& path, McabStats& mstats) {
        assert(!path.empty());
        const PathEdge last = path.back();
        NodeT& current = pool[(size_t)last.nodeIdx];
        current.N[(size_t)last.edgeIdx] += 1.f;
        current.W[(size_t)last.edgeIdx] += 0.5f;
        current.totalN += 1;

        double v = 0.5, raw = 0.5;
        int childIdx = last.nodeIdx;
        for (int i = (int)path.size() - 2; i >= 0; --i) {
            NodeT& child = pool[(size_t)childIdx];
            child.graphN += 1;
            child.graphW += v;
            const double mean = child.graphW / (double)child.graphN;
            const double upChild = params.graphLeafMix * raw +
                                   (1.0 - params.graphLeafMix) * mean;
            NodeT& parent = pool[(size_t)path[(size_t)i].nodeIdx];
            const int e = path[(size_t)i].edgeIdx;
            const double parentV = 1.0 - upChild;
            parent.N[(size_t)e] += 1.f;
            parent.W[(size_t)e] += (float)parentV;
            parent.totalN += 1;
            v = parentV;
            raw = 1.0 - raw;
            childIdx = path[(size_t)i].nodeIdx;
        }
        NodeT& root = pool[(size_t)childIdx];
        root.graphN += 1;
        root.graphW += v;
        mstats.simulations++;
    }

    void backup(const std::vector<PathEdge>& path, double leafQ, McabStats& mstats) {
        double v = leafQ;
        for (int i = (int)path.size() - 1; i >= 0; i--) {
            v = 1.0 - v;
            NodeT& node = pool[path[i].nodeIdx];
            int e = path[i].edgeIdx;
            node.N[e] += 1.f;
            if (params.backupMode == BackupMode::AvgBlend) {
                node.W[e] += (float)v;
            } else {
                node.W[e] = (float)v;
            }
            node.totalN += 1;
            if (params.backupMode == BackupMode::MinimaxHard)
                v = nodeQ(node);
        }
        mstats.simulations++;
    }

    // ---------------------------------------------------------------
    // Seção 5 passo 6 -- escolha do lance final na raiz.
    // ---------------------------------------------------------------
    MoveT pickRootMove(const NodeT& r) const {
        size_t nm = (size_t)std::min(r.activeMoves, (int)r.moves.size());
        assert(nm > 0 && "raiz não-terminal sem lances legais -- não deveria ocorrer em Quoridor");
        if (nm == 0) return MoveT{};

        auto qOf = [&](size_t i) { return r.N[i] > 0.f ? edgeQ(r, i) : -1.0; };

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
