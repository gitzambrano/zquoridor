// arena.cpp -- Arena de verdade entre DUAS engines distintas.
//
// Por que este arquivo substituiu a versao antiga de arena.cpp (guardada
// em teste/arena_singleengine_old.cpp): a versao antiga compilava UM único binário (a partir só do ref1) e
// instanciava duas Negamax daquele MESMO binário -- ou seja, "Engine 2"
// no run_arena.py nunca era de fato o código do --ref2: o worktree do
// ref2 era criado e depois descartado sem nunca ser compilado. O
// confronto rodava, na prática, o engine do ref1 contra ele mesmo,
// disfarçado de A/B test.
//
// Este arquivo resolve isso incluindo os headers (rules.hpp/search.hpp)
// de CADA ref sob um namespace próprio (qr_e1 / qr_e2), via o truque de
// pré-processador `#define qr qr_eN` antes do #include -- como tudo em
// rules.hpp/search.hpp/cat.hpp/endgame_race.hpp/dsu.hpp vive dentro de
// `namespace qr { ... }`, a macro renomeia esse namespace no momento da
// expansão, e os dois conjuntos de símbolos (Negamax, State, Move, TT,
// etc.) convivem no mesmo processo sem colidir. Cada engine roda com o
// código-fonte REAL do ref correspondente.
//
// Ponte entre os dois "mundos": como State pode (em tese) ter layout
// diferente entre refs, nunca fazemos cast entre qr_e1::State e
// qr_e2::State. Mantemos DOIS estados paralelos (s1, s2), avançados pela
// MESMA sequência lógica de lances (Move é um POD trivial e estável:
// isWall/a/b/c), e usamos s1 como fonte-da-verdade para abertura
// aleatória, detecção de vencedor e detecção de empate por repetição do
// jogo real (exatamente o que arena.cpp fazia com seu único estado).
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <vector>
#include <random>
#include <string>
#include <cstdint>
#include <algorithm>
#include <utility>

#ifndef ENGINE1_SEARCH_HPP
#error "compile com -DENGINE1_SEARCH_HPP=\"\\\"/caminho/absoluto/search.hpp\\\"\""
#endif
#ifndef ENGINE2_SEARCH_HPP
#error "compile com -DENGINE2_SEARCH_HPP=\"\\\"/caminho/absoluto/search.hpp\\\"\""
#endif

#define qr qr_e1
#include ENGINE1_SEARCH_HPP
#undef qr

#define qr qr_e2
#include ENGINE2_SEARCH_HPP
#undef qr

// MCTS hibrido com alpha-beta. O caminho e RELATIVO A ESTE ARQUIVO
// (tools/arena/ -> ../../src/), nao via -Isrc: isso garante que o modulo
// venha sempre do HEAD atual, nunca do src/ que run_arena.py fez checkout
// para um --ref1/--ref2 antigo. Um ref anterior a esta feature nem tem
// src/mcab.hpp, e ainda assim o arena compila -- o dispatch SFINAE de
// hasMcabSupport resolve isso em tempo de compilacao e aquele lado joga
// alpha-beta puro.
#include "../../src/mcab.hpp"

// Overrides dos parametros de busca (contempt, LMR, CAT, quiescencia...).
// Mesmo motivo do caminho relativo acima: vem sempre do HEAD, e cada setter
// e aplicado por SFINAE -- um ref antigo que nao tenha um dos metodos
// simplesmente ignora aquele knob, sem quebrar a compilacao.
#include "../../src/search_tuning.hpp"

// Aliases de instanciacao do McabRunner para cada engine (Secao 4.1 do
// plano) -- um McabRunner por engine, vivo pela partida inteira (criado em
// playArenaGame junto de eng1/eng2) para permitir o reuso de subarvore
// entre lances (Secao 8). McabRunner::choose() cai sozinho em
// eng.chooseMove(...) quando o ref nao suporta MCAB (McabRunner::supported
// == false, resolvido em tempo de compilacao) OU quando params().enabled
// == false, entao os call sites podem chamar choose() incondicionalmente.
using McabRunner1 = mcab::McabRunner<qr_e1::Negamax, qr_e1::State, qr_e1::Move,
                                      qr_e1::MoveList, qr_e1::AccPair,
                                      qr_e1::RepetitionTable, qr_e1::SearchStats>;
using McabRunner2 = mcab::McabRunner<qr_e2::Negamax, qr_e2::State, qr_e2::Move,
                                      qr_e2::MoveList, qr_e2::AccPair,
                                      qr_e2::RepetitionTable, qr_e2::SearchStats>;

// =============================================================================
// CONFIG DEFAULTS -- edite aqui para mudar o comportamento padrao sem
// precisar passar flag nenhuma. Cada constante tem uma flag equivalente
// (ver o parsing em main() abaixo); a flag, quando passada, tem
// prioridade sobre a constante. Rodar `./arena.exe` sem argumento nenhum
// usa TODOS os valores daqui.
//
// E1_FORCE_HEURISTIC_DEFAULT / E2_FORCE_HEURISTIC_DEFAULT SEPARADOS (nao
// um unico liga/desliga global): permite configurar heuristica vs. NNUE
// no mesmo confronto so mudando estas duas linhas -- ex. E1=true,
// E2=false roda Engine 1 heuristica contra Engine 2 NNUE sem precisar de
// nenhuma flag de linha de comando. As flags --heuristic (as duas de
// uma vez), --e1-heuristic/--e2-heuristic (uma de cada vez) e
// --e1-nnue/--e2-nnue (caminho de pesos especifico) continuam disponiveis
// pra sobrepor isso por execucao, sem precisar recompilar.
// =============================================================================
constexpr int GAMES_DEFAULT = 40;
constexpr int TIME_MS_DEFAULT = 10;
constexpr int RANDOM_PLIES_DEFAULT = 4;
constexpr int REPORT_GAMES_DEFAULT = 0;
constexpr uint64_t SEED_DEFAULT = 42;
constexpr const char* BIN_FILE_DEFAULT = "";   // vazio = nao grava dataset .bin
constexpr bool INVERT_COLORS_DEFAULT = true;
// Gerar uma bateria de testes SEM NNUE (so heuristica, as duas engines):
// mude as duas constantes abaixo para true, OU passe --heuristic na
// linha de comando toda vez -- mesmo efeito, a flag so existe pra nao
// precisar editar/recompilar.
constexpr bool E1_FORCE_HEURISTIC_DEFAULT = false;
constexpr bool E2_FORCE_HEURISTIC_DEFAULT = false;

// =============================================================================
// MCTS HIBRIDO COM ALPHA-BETA -- bloco de OVERRIDE
//
// Regra deste bloco: campo VAZIO (mcab::UNSET_INT / mcab::UNSET_REAL /
// mcab::Tri::Unset / "") usa o valor de PRODUCAO, que mora em
// mcab::McabParams (src/mcab.hpp). O valor real de producao esta
// no comentario de cada campo. Preencher um campo aqui sobrescreve so ele.
//
// Todos tambem tem flag de linha de comando equivalente (--e1-mcab-*,
// --e2-mcab-*, ou a forma global --mcab-*), que tem prioridade sobre o que
// estiver escrito aqui. Ordem de precedencia, do mais forte pro mais fraco:
//   flag de CLI  >  override deste bloco  >  producao (mcab::McabParams)
// =============================================================================
constexpr mcab::Tri E1_MCAB_ENABLED_OVERRIDE = mcab::Tri::Unset;   // producao: LIGADO
constexpr mcab::Tri E2_MCAB_ENABLED_OVERRIDE = mcab::Tri::Unset;   // producao: LIGADO
constexpr int E1_MCAB_NODE_BUDGET_OVERRIDE = mcab::UNSET_INT;      // producao: 20000
constexpr int E2_MCAB_NODE_BUDGET_OVERRIDE = mcab::UNSET_INT;      // producao: 20000
constexpr int E1_MCAB_LEAF_DEPTH_OVERRIDE = mcab::UNSET_INT;       // producao: 0 (medido)
constexpr int E2_MCAB_LEAF_DEPTH_OVERRIDE = mcab::UNSET_INT;       // producao: 0 (medido)
constexpr int E1_MCAB_LEAF_DEPTH_MAX_OVERRIDE = mcab::UNSET_INT;   // producao: 8
constexpr int E2_MCAB_LEAF_DEPTH_MAX_OVERRIDE = mcab::UNSET_INT;   // producao: 8
constexpr mcab::Tri E1_MCAB_ADAPTIVE_LEAF_DEPTH_OVERRIDE = mcab::Tri::Unset;  // producao: desligado
constexpr mcab::Tri E2_MCAB_ADAPTIVE_LEAF_DEPTH_OVERRIDE = mcab::Tri::Unset;  // producao: desligado
constexpr double E1_MCAB_CPUCT_OVERRIDE = mcab::UNSET_REAL;        // producao: 1.5
constexpr double E2_MCAB_CPUCT_OVERRIDE = mcab::UNSET_REAL;        // producao: 1.5
constexpr double E1_MCAB_FPU_REDUCTION_OVERRIDE = mcab::UNSET_REAL;  // producao: 0.0 (medido)
constexpr double E2_MCAB_FPU_REDUCTION_OVERRIDE = mcab::UNSET_REAL;  // producao: 0.0 (medido)
constexpr double E1_MCAB_SCORE_SCALE_OVERRIDE = mcab::UNSET_REAL;  // producao: 200.0 (= NNUE_EVAL_SCALE)
constexpr double E2_MCAB_SCORE_SCALE_OVERRIDE = mcab::UNSET_REAL;  // producao: 200.0 (= NNUE_EVAL_SCALE)
// "" | "visits" | "q" | "visits-then-q"
constexpr const char* E1_MCAB_ROOT_SELECT_OVERRIDE = "";           // producao: "visits" (MaxVisits)
constexpr const char* E2_MCAB_ROOT_SELECT_OVERRIDE = "";           // producao: "visits" (MaxVisits)
constexpr mcab::Tri E1_MCAB_TREE_REUSE_OVERRIDE = mcab::Tri::Unset;  // producao: ligado
constexpr mcab::Tri E2_MCAB_TREE_REUSE_OVERRIDE = mcab::Tri::Unset;  // producao: ligado
constexpr mcab::Tri E1_MCAB_CLEAR_TT_PER_MOVE_OVERRIDE = mcab::Tri::Unset;  // producao: desligado
constexpr mcab::Tri E2_MCAB_CLEAR_TT_PER_MOVE_OVERRIDE = mcab::Tri::Unset;  // producao: desligado
constexpr int E1_MCAB_MAX_TREE_DEPTH_OVERRIDE = mcab::UNSET_INT;   // producao: 48
constexpr int E2_MCAB_MAX_TREE_DEPTH_OVERRIDE = mcab::UNSET_INT;   // producao: 48
// Ruido de Dirichlet na raiz: producao = DESLIGADO aqui de proposito. Serve
// para diversidade de abertura em dados de treino (selfplay liga), nunca
// para medir forca -- arena mede forca. Sem flag de CLI correspondente.
constexpr mcab::Tri E1_MCAB_ROOT_NOISE_OVERRIDE = mcab::Tri::Off;  // producao do modulo: desligado
constexpr mcab::Tri E2_MCAB_ROOT_NOISE_OVERRIDE = mcab::Tri::Off;  // producao do modulo: desligado
constexpr double E1_MCAB_ROOT_NOISE_ALPHA_OVERRIDE = mcab::UNSET_REAL;    // producao: 0.3
constexpr double E2_MCAB_ROOT_NOISE_ALPHA_OVERRIDE = mcab::UNSET_REAL;    // producao: 0.3
constexpr double E1_MCAB_ROOT_NOISE_EPSILON_OVERRIDE = mcab::UNSET_REAL;  // producao: 0.25
constexpr double E2_MCAB_ROOT_NOISE_EPSILON_OVERRIDE = mcab::UNSET_REAL;  // producao: 0.25
// Atalho da Secao 6: forca nodeBudget=0 (modo equivalencia) quando ligado.
constexpr bool E1_MCAB_EQUIV_MODE_DEFAULT = false;
constexpr bool E2_MCAB_EQUIV_MODE_DEFAULT = false;
// backupMode nao tem override: AvgBlend nao esta implementado (Secao 5).
constexpr mcab::BackupMode E1_MCAB_BACKUP_MODE_DEFAULT = mcab::BackupMode::MinimaxHard;
constexpr mcab::BackupMode E2_MCAB_BACKUP_MODE_DEFAULT = mcab::BackupMode::MinimaxHard;

// =============================================================================
// PARAMETROS DE BUSCA (search.hpp) -- bloco de OVERRIDE, por engine
// =============================================================================
// Mesma regra do bloco MCab acima: campo vazio (tuning::UNSET_*/Tri::Unset)
// = nao chama o setter e vale o default de producao de search.hpp, anotado no
// comentario de cada linha. Preencher sobrescreve so aquele knob, so naquela
// engine. As flags --<knob> (as duas engines) e --e1-<knob>/--e2-<knob> (um
// lado) sobrescrevem isto por execucao, sem recompilar.
//
// Precedencia: --e1-/--e2- > flag global > override daqui > producao.
//
// Estes sao os mesmos parametros que tune_spsa.cpp otimiza: o ponto de
// expo-los aqui e conseguir MEDIR em Elo, com run_arena.py, o candidato que
// o SPSA propos -- antes disso era preciso editar search.hpp e recompilar.
static const tuning::SearchTuning E1_TUNING_OVERRIDE = {
    /*quiescence*/          tuning::Tri::Unset,   // producao: ligada
    /*lmrPvs*/              tuning::Tri::Unset,   // producao: ligado
    /*contempt*/            tuning::UNSET_INT,    // producao: -30
    /*policyOrderScale*/    tuning::UNSET_I64,    // producao: 400
    /*catScoreScale*/       tuning::UNSET_I64,    // producao: 2
    /*lmrMinDepth*/         tuning::UNSET_INT,    // producao: 3
    /*lmrMinMoveIndex*/     tuning::UNSET_INT,    // producao: 3
    /*lmrDivisor*/          tuning::UNSET_REAL,   // producao: 2.25
    /*catHotCm*/            tuning::UNSET_INT,    // producao: 150
    /*catColdCm*/           tuning::UNSET_INT,    // producao: 30
    /*wallBfsOrderMaxPly*/  tuning::UNSET_INT,    // producao: 2
    /*qsCriticalBfsDelta*/  tuning::UNSET_INT,    // producao: 2
};
static const tuning::SearchTuning E2_TUNING_OVERRIDE = E1_TUNING_OVERRIDE;

// Config resolvida; as flags de CLI escrevem por cima em main().
static tuning::SearchTuning g_e1Tuning = E1_TUNING_OVERRIDE;
static tuning::SearchTuning g_e2Tuning = E2_TUNING_OVERRIDE;

// Valores de producao, resolvidos uma vez (McabParams default-construido).
static const mcab::McabParams MCAB_PROD;

// Config resolvida (override do bloco acima, senao producao). As flags de
// CLI sobrescrevem estes valores em main() antes do inicio das partidas.
static bool g_e1McabEnabled = mcab::resolve(E1_MCAB_ENABLED_OVERRIDE, MCAB_PROD.enabled);
static bool g_e2McabEnabled = mcab::resolve(E2_MCAB_ENABLED_OVERRIDE, MCAB_PROD.enabled);
static int g_e1McabNodeBudget = mcab::resolve(E1_MCAB_NODE_BUDGET_OVERRIDE, MCAB_PROD.nodeBudget);
static int g_e2McabNodeBudget = mcab::resolve(E2_MCAB_NODE_BUDGET_OVERRIDE, MCAB_PROD.nodeBudget);
static int g_e1McabLeafDepth = mcab::resolve(E1_MCAB_LEAF_DEPTH_OVERRIDE, MCAB_PROD.leafDepth);
static int g_e2McabLeafDepth = mcab::resolve(E2_MCAB_LEAF_DEPTH_OVERRIDE, MCAB_PROD.leafDepth);
static int g_e1McabLeafDepthMax = mcab::resolve(E1_MCAB_LEAF_DEPTH_MAX_OVERRIDE, MCAB_PROD.leafDepthMax);
static int g_e2McabLeafDepthMax = mcab::resolve(E2_MCAB_LEAF_DEPTH_MAX_OVERRIDE, MCAB_PROD.leafDepthMax);
static bool g_e1McabAdaptiveLeafDepth = mcab::resolve(E1_MCAB_ADAPTIVE_LEAF_DEPTH_OVERRIDE, MCAB_PROD.adaptiveLeafDepth);
static bool g_e2McabAdaptiveLeafDepth = mcab::resolve(E2_MCAB_ADAPTIVE_LEAF_DEPTH_OVERRIDE, MCAB_PROD.adaptiveLeafDepth);
static double g_e1McabCPuct = mcab::resolve(E1_MCAB_CPUCT_OVERRIDE, MCAB_PROD.cPuct);
static double g_e2McabCPuct = mcab::resolve(E2_MCAB_CPUCT_OVERRIDE, MCAB_PROD.cPuct);
static double g_e1McabFpuReduction = mcab::resolve(E1_MCAB_FPU_REDUCTION_OVERRIDE, MCAB_PROD.fpuReduction);
static double g_e2McabFpuReduction = mcab::resolve(E2_MCAB_FPU_REDUCTION_OVERRIDE, MCAB_PROD.fpuReduction);
static double g_e1McabScoreScale = mcab::resolve(E1_MCAB_SCORE_SCALE_OVERRIDE, MCAB_PROD.scoreScale);
static double g_e2McabScoreScale = mcab::resolve(E2_MCAB_SCORE_SCALE_OVERRIDE, MCAB_PROD.scoreScale);
static mcab::RootSelectMode g_e1McabRootSelectMode = mcab::resolveRootSelect(E1_MCAB_ROOT_SELECT_OVERRIDE, MCAB_PROD.rootSelectMode);
static mcab::RootSelectMode g_e2McabRootSelectMode = mcab::resolveRootSelect(E2_MCAB_ROOT_SELECT_OVERRIDE, MCAB_PROD.rootSelectMode);
static mcab::BackupMode g_e1McabBackupMode = E1_MCAB_BACKUP_MODE_DEFAULT;
static mcab::BackupMode g_e2McabBackupMode = E2_MCAB_BACKUP_MODE_DEFAULT;
static bool g_e1McabTreeReuse = mcab::resolve(E1_MCAB_TREE_REUSE_OVERRIDE, MCAB_PROD.treeReuse);
static bool g_e2McabTreeReuse = mcab::resolve(E2_MCAB_TREE_REUSE_OVERRIDE, MCAB_PROD.treeReuse);
static bool g_e1McabClearTTPerMove = mcab::resolve(E1_MCAB_CLEAR_TT_PER_MOVE_OVERRIDE, MCAB_PROD.clearTTPerMove);
static bool g_e2McabClearTTPerMove = mcab::resolve(E2_MCAB_CLEAR_TT_PER_MOVE_OVERRIDE, MCAB_PROD.clearTTPerMove);
static bool g_e1McabRootNoiseEnabled = mcab::resolve(E1_MCAB_ROOT_NOISE_OVERRIDE, MCAB_PROD.rootNoiseEnabled);
static bool g_e2McabRootNoiseEnabled = mcab::resolve(E2_MCAB_ROOT_NOISE_OVERRIDE, MCAB_PROD.rootNoiseEnabled);
static double g_e1McabRootNoiseAlpha = mcab::resolve(E1_MCAB_ROOT_NOISE_ALPHA_OVERRIDE, MCAB_PROD.rootNoiseAlpha);
static double g_e2McabRootNoiseAlpha = mcab::resolve(E2_MCAB_ROOT_NOISE_ALPHA_OVERRIDE, MCAB_PROD.rootNoiseAlpha);
static double g_e1McabRootNoiseEpsilon = mcab::resolve(E1_MCAB_ROOT_NOISE_EPSILON_OVERRIDE, MCAB_PROD.rootNoiseEpsilon);
static double g_e2McabRootNoiseEpsilon = mcab::resolve(E2_MCAB_ROOT_NOISE_EPSILON_OVERRIDE, MCAB_PROD.rootNoiseEpsilon);
static int g_e1McabMaxTreeDepth = mcab::resolve(E1_MCAB_MAX_TREE_DEPTH_OVERRIDE, MCAB_PROD.maxTreeDepth);
static int g_e2McabMaxTreeDepth = mcab::resolve(E2_MCAB_MAX_TREE_DEPTH_OVERRIDE, MCAB_PROD.maxTreeDepth);
static bool g_e1McabEquivMode = E1_MCAB_EQUIV_MODE_DEFAULT;
static bool g_e2McabEquivMode = E2_MCAB_EQUIV_MODE_DEFAULT;

// Struct de amostra de treino local (independente das duas engines) --
// layout idêntico ao TrainingSample de selfplay.hpp/arena.cpp original.
#pragma pack(push, 1)
struct TrainingSample {
    uint8_t  ownPawn;
    uint8_t  oppPawn;
    uint64_t wallsH;
    uint64_t wallsV;
    int8_t   wallsLeftOwn;
    int8_t   wallsLeftOpp;
    uint16_t evalNNUE;      // avaliação da própria NNUE (0..EV_SCALE, perspectiva brancas) -- ver nota completa em selfplay.hpp
    int8_t   gameResult;
    uint16_t policyTarget;
    uint8_t  ownDist;
    uint8_t  oppDist;
    uint8_t  mover;         // 0/1 -- ver nota completa em selfplay.hpp
    int16_t  ownCatTotal;   // soma do calor de corredor (cat.hpp) do mover -- ver nota em selfplay.hpp
    int16_t  oppCatTotal;
};
#pragma pack(pop)
static_assert(sizeof(TrainingSample) == 32, "TrainingSample precisa ficar packed");

static inline qr_e2::Move toE2(const qr_e1::Move& m) {
    qr_e2::Move r; r.isWall = m.isWall; r.a = m.a; r.b = m.b; r.c = m.c; return r;
}

// Mesma escala fixa de selfplay.hpp (TrainingSample::evalNNUE / EV_SCALE) --
// duplicada aqui porque arena.cpp mantém sua própria cópia independente da
// struct TrainingSample (ver comentário logo abaixo), sem incluir
// selfplay.hpp. Se um dos dois valores mudar, o outro precisa acompanhar.
static constexpr uint16_t ARENA_EV_SCALE = 65535;

// Helpers SFINAE para tentar calcular a avaliação da própria NNUE (cabeça
// WL) se qr_eN tiver os símbolos necessários (buildAccumulatorQuant,
// nnueWinProbQuant) -- refs Git antigos sem NNUE, ou execuções em que os
// pesos não carregaram, caem no overload "..." e devolvem 0.5 (neutro),
// igual ao que a NNUE zerada devolveria de qualquer forma. `mover` é a
// perspectiva do lado que jogou nesta posição (rec.moverTurn); devolve a
// probabilidade de vitória do MOVER (não das brancas -- a conversão pra
// perspectiva absoluta de cor acontece no site de chamada, igual
// selfplay.hpp).
template <typename Dummy = void>
auto tryNnueWinProbE1(const qr_e1::State& s, int mover, int)
    -> decltype(qr_e1::nnueWinProbQuant(qr_e1::buildAccumulatorQuant(s, mover))) {
    return qr_e1::nnueWinProbQuant(qr_e1::buildAccumulatorQuant(s, mover));
}
inline float tryNnueWinProbE1(const qr_e1::State&, int, ...) { return 0.5f; }

// Helpers SFINAE para tentar chamar setEvalMode / loadWeightsQuant se existirem
// no namespace qr_e1 / qr_e2 (permite compilar arena entre a versao atual e
// refs Git antigas que nao tinham NNUE).
template <typename Dummy = void>
auto tryLoadWeightsE1(const std::string& path, int) -> decltype(qr_e1::loadWeightsQuant(path)) {
    return qr_e1::loadWeightsQuant(path);
}
inline bool tryLoadWeightsE1(const std::string&, ...) { return false; }

template <typename Dummy = void>
auto tryLoadWeightsE2(const std::string& path, int) -> decltype(qr_e2::loadWeightsQuant(path)) {
    return qr_e2::loadWeightsQuant(path);
}
inline bool tryLoadWeightsE2(const std::string&, ...) { return false; }

// Mesmo truque SFINAE para defaultNnueWeightsPath(): refs antigos (antes
// de NNUE virar default) não têm essa função em nnue.hpp, então o overload
// "..." devolve string vazia (== "sem default disponível para este ref",
// tratado abaixo como "essa engine fica heurística por não ter pesos").
template <typename Dummy = void>
auto tryDefaultPathE1(int) -> decltype(qr_e1::defaultNnueWeightsPath()) {
    return qr_e1::defaultNnueWeightsPath();
}
inline std::string tryDefaultPathE1(...) { return std::string(); }

template <typename Dummy = void>
auto tryDefaultPathE2(int) -> decltype(qr_e2::defaultNnueWeightsPath()) {
    return qr_e2::defaultNnueWeightsPath();
}
inline std::string tryDefaultPathE2(...) { return std::string(); }

template <typename Eng>
auto trySetEvalModeNnue(Eng& eng, int) -> decltype(eng.setEvalMode(Eng::EvalMode::NNUE), void()) {
    eng.setEvalMode(Eng::EvalMode::NNUE);
}
template <typename Eng>
void trySetEvalModeNnue(Eng&, ...) {}

// Mesmo truque SFINAE do trySetEvalModeNnue acima -- setPolicyOrderingEnabled
// e um metodo NOVO (prompt_policy_ordering.md), entao um ref antigo de
// qr_e1/qr_e2 (git ref anterior a esta mudanca) nao tem esse metodo. O
// overload "..." (nao-template, prioridade mais baixa na resolucao de
// overload) e escolhido automaticamente quando o metodo nao existe --
// vira um no-op silencioso, SEM erro de compilacao, exatamente como ja
// acontece hoje pra engines antigas sem NNUE nenhum. Ou seja: pode deixar
// g_e1UsePolicyOrdering/g_e2UsePolicyOrdering = true apontando pra um ref
// antigo sem a flag que o arena continua compilando e rodando normal --
// só não terá efeito nenhum naquele engine (equivalente a nunca ter ligado).
template <typename Eng>
auto trySetPolicyOrdering(Eng& eng, bool enabled, int) -> decltype(eng.setPolicyOrderingEnabled(enabled), void()) {
    eng.setPolicyOrderingEnabled(enabled);
}
template <typename Eng>
void trySetPolicyOrdering(Eng&, bool, ...) {}

// Mesmo padrao para o piso de profundidade (setPolicyOrderingMinDepth,
// search.hpp) -- CORRECAO DE PERFORMANCE: sem este piso, o forward pass
// de politica (~5.8x mais caro que o eval de folha) rodava em todo no
// interno, derrubando nos/s em ~3x (medido em producao via run_arena.py).
// O piso restringe isso aos poucos nos de cima da arvore. Ref antigo sem
// o metodo -> no-op silencioso, mesma logica SFINAE de cima.
template <typename Eng>
auto trySetPolicyOrderingMinDepth(Eng& eng, int d, int) -> decltype(eng.setPolicyOrderingMinDepth(d), void()) {
    eng.setPolicyOrderingMinDepth(d);
}
template <typename Eng>
void trySetPolicyOrderingMinDepth(Eng&, int, ...) {}

// Detector SFINAE puramente informativo (nao afeta comportamento -- so
// evita logar "ligada" quando o ref daquela engine nem tem o metodo).
template <typename Eng>
constexpr auto hasPolicyOrdering(int) -> decltype(std::declval<Eng&>().setPolicyOrderingEnabled(true), bool()) {
    return true;
}
template <typename Eng>
constexpr bool hasPolicyOrdering(...) { return false; }

static bool g_e1UseNnue = false;
static bool g_e2UseNnue = false;
// Liga/desliga a ordenacao assistida por politica (prompt_policy_ordering.md)
// em cada engine -- só tem efeito quando a engine correspondente estiver
// em NNUE (g_eXUseNnue==true); em heuristica é inofensivo (nao é sequer
// chamado, ver playArenaGame abaixo). Default LIGADA desde 2026-08 (era
// desligada) -- mesmo default de search.hpp/selfplay/wasm agora: "NNUE
// default -> policy head default", mesmo sem nenhum parametro passado.
// Mude aqui OU use as flags de linha de comando --e1-policy-order/
// --e2-policy-order/--policy-order/--e1-no-policy-order/--e2-no-policy-order
// (ver main()).
constexpr bool E1_POLICY_ORDERING_DEFAULT = true;
constexpr bool E2_POLICY_ORDERING_DEFAULT = true;
static bool g_e1UsePolicyOrdering = E1_POLICY_ORDERING_DEFAULT;
static bool g_e2UsePolicyOrdering = E2_POLICY_ORDERING_DEFAULT;
// Piso de profundidade -- mesmo default de search.hpp (3), sobreponivel
// via --policy-order-min-depth/--e1-.../--e2-... na linha de comando.
constexpr int E1_POLICY_ORDER_MIN_DEPTH_DEFAULT = 3;
constexpr int E2_POLICY_ORDER_MIN_DEPTH_DEFAULT = 3;
static int g_e1PolicyOrderMinDepth = E1_POLICY_ORDER_MIN_DEPTH_DEFAULT;
static int g_e2PolicyOrderMinDepth = E2_POLICY_ORDER_MIN_DEPTH_DEFAULT;

// Um jogo completo. engine1PlayerIdx define quem (0=brancas,1=pretas) é
// o engine do ref1 nesta partida. Estado é mantido em paralelo nos dois
// namespaces; s1 é a referência canônica para regras de fim de jogo.
int playArenaGame(int engine1PlayerIdx, int timeMs, int randomPlies, std::mt19937_64& rng,
                   uint64_t& eng1NodesOut, uint64_t& eng2NodesOut,
                   double& eng1TimeOut, double& eng2TimeOut,
                   std::vector<TrainingSample>* samplesOut) {
    qr_e1::Negamax eng1;
    qr_e2::Negamax eng2;
    if (g_e1UseNnue) trySetEvalModeNnue(eng1, 0);
    if (g_e2UseNnue) trySetEvalModeNnue(eng2, 0);

    // Runners MCab (Secao 5/8 do plano) -- vivem pela partida inteira (mesmo
    // escopo que eng1/eng2) para o reuso de subarvore entre lances
    // funcionar; resetTree() no inicio de cada partida nova para nao
    // vazar a arvore do jogo anterior para este.
    McabRunner1 mcabRunner1;
    McabRunner2 mcabRunner2;
    {
        mcab::McabParams p1;
        p1.enabled = g_e1McabEnabled;
        p1.nodeBudget = g_e1McabEquivMode ? 0 : g_e1McabNodeBudget;
        p1.leafDepth = g_e1McabLeafDepth;
        p1.leafDepthMax = g_e1McabLeafDepthMax;
        p1.adaptiveLeafDepth = g_e1McabAdaptiveLeafDepth;
        p1.cPuct = g_e1McabCPuct;
        p1.fpuReduction = g_e1McabFpuReduction;
        p1.scoreScale = g_e1McabScoreScale;
        p1.rootSelectMode = g_e1McabRootSelectMode;
        p1.backupMode = g_e1McabBackupMode;
        p1.treeReuse = g_e1McabTreeReuse;
        p1.clearTTPerMove = g_e1McabClearTTPerMove;
        p1.rootNoiseEnabled = g_e1McabRootNoiseEnabled;
        p1.rootNoiseAlpha = g_e1McabRootNoiseAlpha;
        p1.rootNoiseEpsilon = g_e1McabRootNoiseEpsilon;
        p1.maxTreeDepth = g_e1McabMaxTreeDepth;
        mcabRunner1.setParams(p1);

        mcab::McabParams p2;
        p2.enabled = g_e2McabEnabled;
        p2.nodeBudget = g_e2McabEquivMode ? 0 : g_e2McabNodeBudget;
        p2.leafDepth = g_e2McabLeafDepth;
        p2.leafDepthMax = g_e2McabLeafDepthMax;
        p2.adaptiveLeafDepth = g_e2McabAdaptiveLeafDepth;
        p2.cPuct = g_e2McabCPuct;
        p2.fpuReduction = g_e2McabFpuReduction;
        p2.scoreScale = g_e2McabScoreScale;
        p2.rootSelectMode = g_e2McabRootSelectMode;
        p2.backupMode = g_e2McabBackupMode;
        p2.treeReuse = g_e2McabTreeReuse;
        p2.clearTTPerMove = g_e2McabClearTTPerMove;
        p2.rootNoiseEnabled = g_e2McabRootNoiseEnabled;
        p2.rootNoiseAlpha = g_e2McabRootNoiseAlpha;
        p2.rootNoiseEpsilon = g_e2McabRootNoiseEpsilon;
        p2.maxTreeDepth = g_e2McabMaxTreeDepth;
        mcabRunner2.setParams(p2);
    }
    mcabRunner1.resetTree();
    mcabRunner2.resetTree();
    // Só faz sentido (e só tem efeito real dentro de Negamax, ver
    // policyOrderingEnabled em search.hpp) quando a engine correspondente
    // está em NNUE -- gate aqui evita uma chamada supérflua em heurística.
    // Sempre chama o setter (nao so quando ligado): search.hpp passou a
    // nascer com policyOrderingEnabled=true, entao nao chamar nada no caso
    // "desligado" deixava --e1-no-policy-order/--no-policy-order sem efeito
    // nenhum -- a engine seguia com a ordenacao por politica ligada.
    if (g_e1UseNnue) {
        trySetPolicyOrdering(eng1, g_e1UsePolicyOrdering, 0);
        if (g_e1UsePolicyOrdering) trySetPolicyOrderingMinDepth(eng1, g_e1PolicyOrderMinDepth, 0);
    }
    if (g_e2UseNnue) {
        trySetPolicyOrdering(eng2, g_e2UsePolicyOrdering, 0);
        if (g_e2UsePolicyOrdering) trySetPolicyOrderingMinDepth(eng2, g_e2PolicyOrderMinDepth, 0);
    }
    // Parametros de busca do bloco de config/CLI. Vazio = no-op completo.
    tuning::applySearchTuning(eng1, g_e1Tuning);
    tuning::applySearchTuning(eng2, g_e2Tuning);
    eng1.clearTT();
    eng2.clearTT();

    qr_e1::State s1 = qr_e1::initialState();
    qr_e2::State s2 = qr_e2::initialState();

    // Abertura aleatória: gerada a partir do ref1 (fonte-da-verdade das
    // regras) e replicada lance-a-lance no estado do ref2.
    for (int ply = 0; ply < randomPlies; ply++) {
        if (qr_e1::winner(s1) != -1) break;
        auto moves = qr_e1::legalMoves(s1);
        if (moves.empty()) break;
        std::uniform_int_distribution<size_t> pick(0, moves.size() - 1);
        qr_e1::Move m = moves[pick(rng)];
        s1 = qr_e1::applyMove(s1, m);
        s2 = qr_e2::applyMove(s2, toE2(m));
    }

    qr_e1::RepetitionTable realHistory;

    struct MoveRecord {
        qr_e1::State state;
        qr_e1::Move move;
        int moverTurn;
        int searchScore;
    };
    std::vector<MoveRecord> gameRecords;

    int winnerPlayer = -1;
    int maxPlies = 300;
    qr_e1::RepetitionTable hist1;
    qr_e2::RepetitionTable hist2;

    for (int ply = 0; ply < maxPlies; ply++) {
        int w = qr_e1::winner(s1);
        if (w != -1) { winnerPlayer = w; break; }

        if (realHistory.count(s1.hash) >= 2) { winnerPlayer = -1; break; }

        int currentTurn = s1.turn;
        qr_e1::Move mChosen;

        if (currentTurn == engine1PlayerIdx) {
            qr_e1::SearchStats st;
            auto t0 = std::chrono::high_resolution_clock::now();
            mChosen = mcabRunner1.choose(eng1, s1, 40, timeMs, st, hist1);
            auto t1 = std::chrono::high_resolution_clock::now();
            eng1TimeOut += std::chrono::duration<double>(t1 - t0).count();
            eng1NodesOut += st.nodes;
            if (samplesOut) gameRecords.push_back({s1, mChosen, currentTurn, st.score});
        } else {
            qr_e2::SearchStats st;
            auto t0 = std::chrono::high_resolution_clock::now();
            qr_e2::Move m2 = mcabRunner2.choose(eng2, s2, 40, timeMs, st, hist2);
            auto t1 = std::chrono::high_resolution_clock::now();
            eng2TimeOut += std::chrono::duration<double>(t1 - t0).count();
            eng2NodesOut += st.nodes;
            mChosen = qr_e1::Move{m2.isWall, m2.a, m2.b, m2.c};
            if (samplesOut) gameRecords.push_back({s1, mChosen, currentTurn, st.score});
        }

        realHistory.push(s1.hash);
        hist1.push(s1.hash);
        hist2.push(s2.hash);
        s1 = qr_e1::applyMove(s1, mChosen);
        s2 = qr_e2::applyMove(s2, toE2(mChosen));
    }

    if (winnerPlayer == -1 && qr_e1::winner(s1) != -1) winnerPlayer = qr_e1::winner(s1);

    if (samplesOut && winnerPlayer != -1) {
        for (const auto& rec : gameRecords) {
            int mover = rec.moverTurn, opp = 1 - rec.moverTurn;
            // Mesmo espelho de perspectiva de selfplay.hpp (nnue.hpp,
            // 2026-08) -- sem isso este --bin-file gerava dados em
            // coordenada CRUA, incompatível com o que selfplay.hpp grava
            // hoje (silenciosamente: mesmo dtype/tamanho de arquivo,
            // corrompe o treino se misturado).
            TrainingSample ts{};
            ts.ownPawn = (uint8_t)qr_e1::mirroredPawnCell(rec.state.pawn[mover], mover);
            ts.oppPawn = (uint8_t)qr_e1::mirroredPawnCell(rec.state.pawn[opp], mover);
            ts.wallsH = qr_e1::mirrorWallBitboard(rec.state.wallsH, mover);
            ts.wallsV = qr_e1::mirrorWallBitboard(rec.state.wallsV, mover);
            ts.wallsLeftOwn = rec.state.wallsLeft[mover];
            ts.wallsLeftOpp = rec.state.wallsLeft[opp];
            {
                float probMoverWins = tryNnueWinProbE1(rec.state, mover, 0);
                float evalWhite = (mover == 0) ? probMoverWins : (1.0f - probMoverWins);
                evalWhite = std::max(0.0f, std::min(1.0f, evalWhite));
                ts.evalNNUE = (uint16_t)std::lround(evalWhite * (float)ARENA_EV_SCALE);
            }
            ts.gameResult = (mover == winnerPlayer) ? (int8_t)1 : (int8_t)-1;
            ts.policyTarget = qr_e1::moveToPolicyIndex(qr_e1::mirrorMoveForPerspective(rec.move, mover));
            ts.ownDist = (uint8_t)std::min(255, qr_e1::shortestPathLen(rec.state.wallsH, rec.state.wallsV, rec.state.pawn[mover], mover));
            ts.oppDist = (uint8_t)std::min(255, qr_e1::shortestPathLen(rec.state.wallsH, rec.state.wallsV, rec.state.pawn[opp], opp));
            ts.mover = (uint8_t)mover;
            {
                qr_e1::CorridorHeat ownHeat = qr_e1::computeCorridorHeat(rec.state.wallsH, rec.state.wallsV, rec.state.pawn[mover], mover);
                qr_e1::CorridorHeat oppHeat = qr_e1::computeCorridorHeat(rec.state.wallsH, rec.state.wallsV, rec.state.pawn[opp], opp);
                int ownSum = 0, oppSum = 0;
                for (int i = 0; i < qr_e1::N * qr_e1::N; i++) { ownSum += ownHeat.heat[i]; oppSum += oppHeat.heat[i]; }
                ts.ownCatTotal = (int16_t)ownSum;
                ts.oppCatTotal = (int16_t)oppSum;
            }
            samplesOut->push_back(ts);
        }
    }

    return winnerPlayer;
}

int main(int argc, char* argv[]) {
    int totalGames = GAMES_DEFAULT;
    int timeMs = TIME_MS_DEFAULT;
    int randomPlies = RANDOM_PLIES_DEFAULT;
    int reportGames = REPORT_GAMES_DEFAULT;
    uint64_t seed = SEED_DEFAULT;
    std::string binFilePath = BIN_FILE_DEFAULT;
    bool invertColors = INVERT_COLORS_DEFAULT;
    std::string e1NnuePath = "";
    std::string e2NnuePath = "";
    bool e1Explicit = false, e2Explicit = false;
    bool forceHeuristic = false;
    bool e1ForceHeuristic = E1_FORCE_HEURISTIC_DEFAULT, e2ForceHeuristic = E2_FORCE_HEURISTIC_DEFAULT;
    bool e1PolicyOrder = E1_POLICY_ORDERING_DEFAULT, e2PolicyOrder = E2_POLICY_ORDERING_DEFAULT;
    int e1PolicyMinDepth = E1_POLICY_ORDER_MIN_DEPTH_DEFAULT, e2PolicyMinDepth = E2_POLICY_ORDER_MIN_DEPTH_DEFAULT;

    // Mesma resolucao das globais acima (override do bloco de CONFIG, senao
    // producao); o parsing de argv logo abaixo sobrescreve o que vier por flag.
    bool e1McabEnabled = mcab::resolve(E1_MCAB_ENABLED_OVERRIDE, MCAB_PROD.enabled);
    bool e2McabEnabled = mcab::resolve(E2_MCAB_ENABLED_OVERRIDE, MCAB_PROD.enabled);
    int e1McabNodeBudget = mcab::resolve(E1_MCAB_NODE_BUDGET_OVERRIDE, MCAB_PROD.nodeBudget);
    int e2McabNodeBudget = mcab::resolve(E2_MCAB_NODE_BUDGET_OVERRIDE, MCAB_PROD.nodeBudget);
    int e1McabLeafDepth = mcab::resolve(E1_MCAB_LEAF_DEPTH_OVERRIDE, MCAB_PROD.leafDepth);
    int e2McabLeafDepth = mcab::resolve(E2_MCAB_LEAF_DEPTH_OVERRIDE, MCAB_PROD.leafDepth);
    int e1McabLeafDepthMax = mcab::resolve(E1_MCAB_LEAF_DEPTH_MAX_OVERRIDE, MCAB_PROD.leafDepthMax);
    int e2McabLeafDepthMax = mcab::resolve(E2_MCAB_LEAF_DEPTH_MAX_OVERRIDE, MCAB_PROD.leafDepthMax);
    bool e1McabAdaptiveLeafDepth = mcab::resolve(E1_MCAB_ADAPTIVE_LEAF_DEPTH_OVERRIDE, MCAB_PROD.adaptiveLeafDepth);
    bool e2McabAdaptiveLeafDepth = mcab::resolve(E2_MCAB_ADAPTIVE_LEAF_DEPTH_OVERRIDE, MCAB_PROD.adaptiveLeafDepth);
    double e1McabCPuct = mcab::resolve(E1_MCAB_CPUCT_OVERRIDE, MCAB_PROD.cPuct);
    double e2McabCPuct = mcab::resolve(E2_MCAB_CPUCT_OVERRIDE, MCAB_PROD.cPuct);
    double e1McabFpu = mcab::resolve(E1_MCAB_FPU_REDUCTION_OVERRIDE, MCAB_PROD.fpuReduction);
    double e2McabFpu = mcab::resolve(E2_MCAB_FPU_REDUCTION_OVERRIDE, MCAB_PROD.fpuReduction);
    double e1McabScoreScale = mcab::resolve(E1_MCAB_SCORE_SCALE_OVERRIDE, MCAB_PROD.scoreScale);
    double e2McabScoreScale = mcab::resolve(E2_MCAB_SCORE_SCALE_OVERRIDE, MCAB_PROD.scoreScale);
    mcab::RootSelectMode e1McabRootSelect = mcab::resolveRootSelect(E1_MCAB_ROOT_SELECT_OVERRIDE, MCAB_PROD.rootSelectMode);
    mcab::RootSelectMode e2McabRootSelect = mcab::resolveRootSelect(E2_MCAB_ROOT_SELECT_OVERRIDE, MCAB_PROD.rootSelectMode);
    bool e1McabTreeReuse = mcab::resolve(E1_MCAB_TREE_REUSE_OVERRIDE, MCAB_PROD.treeReuse);
    bool e2McabTreeReuse = mcab::resolve(E2_MCAB_TREE_REUSE_OVERRIDE, MCAB_PROD.treeReuse);
    bool e1McabClearTTPerMove = mcab::resolve(E1_MCAB_CLEAR_TT_PER_MOVE_OVERRIDE, MCAB_PROD.clearTTPerMove);
    bool e2McabClearTTPerMove = mcab::resolve(E2_MCAB_CLEAR_TT_PER_MOVE_OVERRIDE, MCAB_PROD.clearTTPerMove);
    int e1McabMaxTreeDepth = mcab::resolve(E1_MCAB_MAX_TREE_DEPTH_OVERRIDE, MCAB_PROD.maxTreeDepth);
    int e2McabMaxTreeDepth = mcab::resolve(E2_MCAB_MAX_TREE_DEPTH_OVERRIDE, MCAB_PROD.maxTreeDepth);
    bool e1McabEquivMode = E1_MCAB_EQUIV_MODE_DEFAULT, e2McabEquivMode = E2_MCAB_EQUIV_MODE_DEFAULT;

    // Valor nao reconhecido cai no de producao (nao vira MaxVisits calado):
    // um typo em "visits-then-q" nao deve parecer que funcionou.
    auto parseRootSelectMode = [](const char* s) -> mcab::RootSelectMode {
        return mcab::resolveRootSelect(s, MCAB_PROD.rootSelectMode);
    };

    for (int i = 1; i < argc; i++) {
        if (std::strcmp(argv[i], "--games") == 0 && i + 1 < argc) totalGames = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--time") == 0 && i + 1 < argc) timeMs = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--random-plies") == 0 && i + 1 < argc) randomPlies = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--seed") == 0 && i + 1 < argc) seed = std::stoull(argv[++i]);
        else if (std::strcmp(argv[i], "--report-games") == 0 && i + 1 < argc) reportGames = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--bin-file") == 0 && i + 1 < argc) binFilePath = argv[++i];
        else if (std::strcmp(argv[i], "--e1-nnue") == 0 && i + 1 < argc) { e1NnuePath = argv[++i]; e1Explicit = true; e1ForceHeuristic = false; }
        else if (std::strcmp(argv[i], "--e2-nnue") == 0 && i + 1 < argc) { e2NnuePath = argv[++i]; e2Explicit = true; e2ForceHeuristic = false; }
        else if (std::strcmp(argv[i], "--heuristic") == 0) forceHeuristic = true;
        else if (std::strcmp(argv[i], "--e1-heuristic") == 0) e1ForceHeuristic = true;
        else if (std::strcmp(argv[i], "--e2-heuristic") == 0) e2ForceHeuristic = true;
        else if (std::strcmp(argv[i], "--e1-nnue-default") == 0) e1ForceHeuristic = false;
        else if (std::strcmp(argv[i], "--e2-nnue-default") == 0) e2ForceHeuristic = false;
        else if (std::strcmp(argv[i], "--policy-order") == 0) { e1PolicyOrder = true; e2PolicyOrder = true; }
        else if (std::strcmp(argv[i], "--e1-policy-order") == 0) e1PolicyOrder = true;
        else if (std::strcmp(argv[i], "--e2-policy-order") == 0) e2PolicyOrder = true;
        else if (std::strcmp(argv[i], "--e1-no-policy-order") == 0) e1PolicyOrder = false;
        else if (std::strcmp(argv[i], "--e2-no-policy-order") == 0) e2PolicyOrder = false;
        else if (std::strcmp(argv[i], "--policy-order-min-depth") == 0 && i + 1 < argc) {
            int d = std::atoi(argv[++i]); e1PolicyMinDepth = d; e2PolicyMinDepth = d;
        }
        else if (std::strcmp(argv[i], "--e1-policy-order-min-depth") == 0 && i + 1 < argc) e1PolicyMinDepth = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--e2-policy-order-min-depth") == 0 && i + 1 < argc) e2PolicyMinDepth = std::atoi(argv[++i]);
        // Modulo hibrido MCTS+alpha-beta (MCab, plan-hybrid-mc-ab.md, Secao 10).
        else if (std::strcmp(argv[i], "--mcab") == 0) { e1McabEnabled = true; e2McabEnabled = true; }
        else if (std::strcmp(argv[i], "--e1-mcab") == 0) e1McabEnabled = true;
        else if (std::strcmp(argv[i], "--e2-mcab") == 0) e2McabEnabled = true;
        // O hibrido passou a ser o default (producao), entao o lado
        // "desligar" virou necessario: e assim que se mede alpha-beta puro,
        // e e o que run_arena.py manda quando o usuario pede --no-mcab.
        else if (std::strcmp(argv[i], "--no-mcab") == 0) { e1McabEnabled = false; e2McabEnabled = false; }
        else if (std::strcmp(argv[i], "--e1-no-mcab") == 0) e1McabEnabled = false;
        else if (std::strcmp(argv[i], "--e2-no-mcab") == 0) e2McabEnabled = false;
        else if (std::strcmp(argv[i], "--mcab-nodes") == 0 && i + 1 < argc) {
            int n = std::atoi(argv[++i]); e1McabNodeBudget = n; e2McabNodeBudget = n;
        }
        else if (std::strcmp(argv[i], "--e1-mcab-nodes") == 0 && i + 1 < argc) e1McabNodeBudget = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--e2-mcab-nodes") == 0 && i + 1 < argc) e2McabNodeBudget = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--mcab-leaf-depth") == 0 && i + 1 < argc) {
            int d = std::atoi(argv[++i]); e1McabLeafDepth = d; e2McabLeafDepth = d;
        }
        else if (std::strcmp(argv[i], "--e1-mcab-leaf-depth") == 0 && i + 1 < argc) e1McabLeafDepth = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--e2-mcab-leaf-depth") == 0 && i + 1 < argc) e2McabLeafDepth = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--e1-mcab-leaf-depth-max") == 0 && i + 1 < argc) e1McabLeafDepthMax = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--e2-mcab-leaf-depth-max") == 0 && i + 1 < argc) e2McabLeafDepthMax = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--e1-mcab-adaptive-leaf-depth") == 0) e1McabAdaptiveLeafDepth = true;
        else if (std::strcmp(argv[i], "--e2-mcab-adaptive-leaf-depth") == 0) e2McabAdaptiveLeafDepth = true;
        else if (std::strcmp(argv[i], "--mcab-cpuct") == 0 && i + 1 < argc) {
            double v = std::atof(argv[++i]); e1McabCPuct = v; e2McabCPuct = v;
        }
        else if (std::strcmp(argv[i], "--e1-mcab-cpuct") == 0 && i + 1 < argc) e1McabCPuct = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--e2-mcab-cpuct") == 0 && i + 1 < argc) e2McabCPuct = std::atof(argv[++i]);
        // Aliases globais (as duas engines de uma vez), mesmo padrao de
        // --mcab-nodes/--mcab-leaf-depth/--mcab-cpuct. Existem para que
        // run_arena.py possa varrer estes parametros sem precisar saber de
        // que lado o MCab esta ligado -- o lado com MCab desligado ignora.
        else if (std::strcmp(argv[i], "--mcab-fpu") == 0 && i + 1 < argc) {
            double v = std::atof(argv[++i]); e1McabFpu = v; e2McabFpu = v;
        }
        else if (std::strcmp(argv[i], "--mcab-score-scale") == 0 && i + 1 < argc) {
            double v = std::atof(argv[++i]); e1McabScoreScale = v; e2McabScoreScale = v;
        }
        else if (std::strcmp(argv[i], "--mcab-root-select") == 0 && i + 1 < argc) {
            mcab::RootSelectMode m = parseRootSelectMode(argv[++i]);
            e1McabRootSelect = m; e2McabRootSelect = m;
        }
        else if (std::strcmp(argv[i], "--e1-mcab-fpu") == 0 && i + 1 < argc) e1McabFpu = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--e2-mcab-fpu") == 0 && i + 1 < argc) e2McabFpu = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--e1-mcab-score-scale") == 0 && i + 1 < argc) e1McabScoreScale = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--e2-mcab-score-scale") == 0 && i + 1 < argc) e2McabScoreScale = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--e1-mcab-root-select") == 0 && i + 1 < argc) e1McabRootSelect = parseRootSelectMode(argv[++i]);
        else if (std::strcmp(argv[i], "--e2-mcab-root-select") == 0 && i + 1 < argc) e2McabRootSelect = parseRootSelectMode(argv[++i]);
        else if (std::strcmp(argv[i], "--e1-mcab-no-tree-reuse") == 0) e1McabTreeReuse = false;
        else if (std::strcmp(argv[i], "--e2-mcab-no-tree-reuse") == 0) e2McabTreeReuse = false;
        else if (std::strcmp(argv[i], "--e1-mcab-clear-tt-per-move") == 0) e1McabClearTTPerMove = true;
        else if (std::strcmp(argv[i], "--e2-mcab-clear-tt-per-move") == 0) e2McabClearTTPerMove = true;
        else if (std::strcmp(argv[i], "--e1-mcab-max-tree-depth") == 0 && i + 1 < argc) e1McabMaxTreeDepth = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--e2-mcab-max-tree-depth") == 0 && i + 1 < argc) e2McabMaxTreeDepth = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--e1-mcab-equiv-mode") == 0) e1McabEquivMode = true;
        else if (std::strcmp(argv[i], "--e2-mcab-equiv-mode") == 0) e2McabEquivMode = true;
        else if (std::strcmp(argv[i], "--no-invert") == 0) invertColors = false;
        // Parametros de busca (search.hpp): --e1-<knob>/--e2-<knob> valem so
        // aquele lado, --<knob> escreve nos dois. O helper casa o nome
        // inteiro, entao "--e1-contempt" nunca e confundido com "--contempt".
        else {
            int j = i;   // indice da flag; cada tentativa avanca sua propria copia
            if (tuning::parseSearchTuningArg(argv[i], argc, argv, j, "e1-", g_e1Tuning)) { i = j; continue; }
            j = i;
            if (tuning::parseSearchTuningArg(argv[i], argc, argv, j, "e2-", g_e2Tuning)) { i = j; continue; }
            j = i;
            if (tuning::parseSearchTuningArg(argv[i], argc, argv, j, "", g_e1Tuning)) {
                int k = i;   // mesmo argumento, agora para a Engine 2
                tuning::parseSearchTuningArg(argv[i], argc, argv, k, "", g_e2Tuning);
                i = j;
                continue;
            }
        }
    }

    g_e1McabEnabled = e1McabEnabled;
    g_e2McabEnabled = e2McabEnabled;
    g_e1McabNodeBudget = e1McabNodeBudget;
    g_e2McabNodeBudget = e2McabNodeBudget;
    g_e1McabLeafDepth = e1McabLeafDepth;
    g_e2McabLeafDepth = e2McabLeafDepth;
    g_e1McabLeafDepthMax = e1McabLeafDepthMax;
    g_e2McabLeafDepthMax = e2McabLeafDepthMax;
    g_e1McabAdaptiveLeafDepth = e1McabAdaptiveLeafDepth;
    g_e2McabAdaptiveLeafDepth = e2McabAdaptiveLeafDepth;
    g_e1McabCPuct = e1McabCPuct;
    g_e2McabCPuct = e2McabCPuct;
    g_e1McabFpuReduction = e1McabFpu;
    g_e2McabFpuReduction = e2McabFpu;
    g_e1McabScoreScale = e1McabScoreScale;
    g_e2McabScoreScale = e2McabScoreScale;
    g_e1McabRootSelectMode = e1McabRootSelect;
    g_e2McabRootSelectMode = e2McabRootSelect;
    g_e1McabTreeReuse = e1McabTreeReuse;
    g_e2McabTreeReuse = e2McabTreeReuse;
    g_e1McabClearTTPerMove = e1McabClearTTPerMove;
    g_e2McabClearTTPerMove = e2McabClearTTPerMove;
    g_e1McabMaxTreeDepth = e1McabMaxTreeDepth;
    g_e2McabMaxTreeDepth = e2McabMaxTreeDepth;
    g_e1McabEquivMode = e1McabEquivMode;
    g_e2McabEquivMode = e2McabEquivMode;

    g_e1UsePolicyOrdering = e1PolicyOrder;
    g_e2UsePolicyOrdering = e2PolicyOrder;
    g_e1PolicyOrderMinDepth = e1PolicyMinDepth;
    g_e2PolicyOrderMinDepth = e2PolicyMinDepth;

    // NNUE é o default das duas engines. Se --e1-nnue/--e2-nnue não foram
    // passados explicitamente, tenta o caminho default de cada ref (via
    // SFINAE -- refs antigos sem NNUE devolvem string vazia e ficam
    // heurísticos, sem erro). --heuristic desliga isso por completo nas
    // duas (debug/histórico/comparação com a heurística antiga).
    // --e1-heuristic/--e2-heuristic desligam só uma engine de cada vez --
    // é o que permite medir a força de jogo NNUE vs heurística num mesmo
    // confronto (--e1-heuristic sozinho: Engine 1 heurística, Engine 2
    // NNUE default).
    if (!forceHeuristic) {
        if (!e1ForceHeuristic && e1NnuePath.empty()) e1NnuePath = tryDefaultPathE1(0);
        if (!e2ForceHeuristic && e2NnuePath.empty()) e2NnuePath = tryDefaultPathE2(0);
    }
    if (forceHeuristic || e1ForceHeuristic) e1NnuePath.clear();
    if (forceHeuristic || e2ForceHeuristic) e2NnuePath.clear();

    if (!e1NnuePath.empty()) {
        if (!tryLoadWeightsE1(e1NnuePath, 0)) {
            if (e1Explicit) {
                std::fprintf(stderr, "[arena] ERRO: falha ao carregar pesos NNUE para Engine 1 de '%s'\n", e1NnuePath.c_str());
            } else {
                std::fprintf(stderr, "[arena] aviso: pesos NNUE default nao encontrados para Engine 1 ('%s') -- usando heuristica\n", e1NnuePath.c_str());
            }
        } else {
            g_e1UseNnue = true;
            std::fprintf(stderr, "[arena] Engine 1 usando NNUE ('%s')\n", e1NnuePath.c_str());
        }
    } else if (forceHeuristic || e1ForceHeuristic) {
        std::fprintf(stderr, "[arena] Engine 1: heuristica (via --heuristic/--e1-heuristic, ou E1_FORCE_HEURISTIC_DEFAULT no topo do arquivo)\n");
    }
    if (!e2NnuePath.empty()) {
        if (!tryLoadWeightsE2(e2NnuePath, 0)) {
            if (e2Explicit) {
                std::fprintf(stderr, "[arena] ERRO: falha ao carregar pesos NNUE para Engine 2 de '%s'\n", e2NnuePath.c_str());
            } else {
                std::fprintf(stderr, "[arena] aviso: pesos NNUE default nao encontrados para Engine 2 ('%s') -- usando heuristica\n", e2NnuePath.c_str());
            }
        } else {
            g_e2UseNnue = true;
            std::fprintf(stderr, "[arena] Engine 2 usando NNUE ('%s')\n", e2NnuePath.c_str());
        }
    } else if (forceHeuristic || e2ForceHeuristic) {
        std::fprintf(stderr, "[arena] Engine 2: heuristica (via --heuristic/--e2-heuristic, ou E2_FORCE_HEURISTIC_DEFAULT no topo do arquivo)\n");
    }

    // Log informativo do estado da ordenacao por politica -- so tem efeito
    // de verdade quando g_eXUseNnue==true (o gate real esta em
    // playArenaGame). Se a flag foi pedida (--eX-policy-order ou
    // EX_POLICY_ORDERING_DEFAULT) mas o ref daquela engine nao tem o
    // metodo (compilado contra um search.hpp anterior a esta mudanca),
    // avisa que foi ignorada em vez de fingir que ligou.
    if (g_e1UsePolicyOrdering) {
        if (!g_e1UseNnue) {
            std::fprintf(stderr, "[arena] aviso: --e1-policy-order pedido mas Engine 1 nao esta em NNUE -- ignorado\n");
        } else if (!hasPolicyOrdering<qr_e1::Negamax>(0)) {
            std::fprintf(stderr, "[arena] aviso: --e1-policy-order pedido mas o ref da Engine 1 nao tem setPolicyOrderingEnabled (versao antiga) -- ignorado\n");
        } else {
            std::fprintf(stderr, "[arena] Engine 1: ordenacao por politica LIGADA (min-depth=%d)\n", g_e1PolicyOrderMinDepth);
        }
    }
    if (g_e2UsePolicyOrdering) {
        if (!g_e2UseNnue) {
            std::fprintf(stderr, "[arena] aviso: --e2-policy-order pedido mas Engine 2 nao esta em NNUE -- ignorado\n");
        } else if (!hasPolicyOrdering<qr_e2::Negamax>(0)) {
            std::fprintf(stderr, "[arena] aviso: --e2-policy-order pedido mas o ref da Engine 2 nao tem setPolicyOrderingEnabled (versao antiga) -- ignorado\n");
        } else {
            std::fprintf(stderr, "[arena] Engine 2: ordenacao por politica LIGADA (min-depth=%d)\n", g_e2PolicyOrderMinDepth);
        }
    }

    // MCab (plan-hybrid-mc-ab.md, Secao 2) exige evalMode == NNUE -- depende
    // da cabeca de politica para os priors do PUCT. Se o usuario ligou MCAB
    // para um lado que ficou heuristico (sem pesos NNUE, ref antigo sem
    // NNUE, ou --eX-heuristic explicito), desliga MCAB para aquele lado em
    // vez de deixar cair no assert() de McabSearch::chooseMoveMCAB.
    if (g_e1McabEnabled && !g_e1UseNnue) {
        std::fprintf(stderr, "[arena] aviso: MCTS hibrido ativo para Engine 1 mas ela nao esta em NNUE -- o hibrido exige NNUE (os priors do PUCT vem da cabeca de politica), caindo para alpha-beta puro\n");
        g_e1McabEnabled = false;
    }
    if (g_e2McabEnabled && !g_e2UseNnue) {
        std::fprintf(stderr, "[arena] aviso: MCTS hibrido ativo para Engine 2 mas ela nao esta em NNUE -- o hibrido exige NNUE (os priors do PUCT vem da cabeca de politica), caindo para alpha-beta puro\n");
        g_e2McabEnabled = false;
    }

    // Aviso 1x quando o ref daquela engine nao suporta MCAB (compilado antes
    // de searchLeaf/resetOrderingState virarem publicos, Secao 4.4) -- mesmo
    // padrao do aviso de policy-ordering acima: a flag e aceita, mas nao tem
    // efeito, e a partida roda normalmente com AB puro para aquele lado.
    if (g_e1McabEnabled) {
        if (!McabRunner1::supported) {
            std::fprintf(stderr, "[arena] Engine 1: ref nao suporta o MCTS hibrido (compilado antes da feature), rodando alpha-beta puro\n");
        } else {
            std::fprintf(stderr, "[arena] Engine 1: MCTS HIBRIDO (nodes=%d, leaf-depth=%d, cpuct=%.2f, fpu=%.2f, scale=%.0f, root-select=%s, tree-reuse=%s%s)\n",
                          g_e1McabEquivMode ? 0 : g_e1McabNodeBudget, g_e1McabLeafDepth, g_e1McabCPuct,
                          g_e1McabFpuReduction, g_e1McabScoreScale,
                          mcab::rootSelectName(g_e1McabRootSelectMode),
                          g_e1McabTreeReuse ? "on" : "off",
                          g_e1McabEquivMode ? ", modo equivalencia" : "");
        }
    } else {
        std::fprintf(stderr, "[arena] Engine 1: ALPHA-BETA PURO (MCTS hibrido desligado)\n");
    }
    if (g_e2McabEnabled) {
        if (!McabRunner2::supported) {
            std::fprintf(stderr, "[arena] Engine 2: ref nao suporta o MCTS hibrido (compilado antes da feature), rodando alpha-beta puro\n");
        } else {
            std::fprintf(stderr, "[arena] Engine 2: MCTS HIBRIDO (nodes=%d, leaf-depth=%d, cpuct=%.2f, fpu=%.2f, scale=%.0f, root-select=%s, tree-reuse=%s%s)\n",
                          g_e2McabEquivMode ? 0 : g_e2McabNodeBudget, g_e2McabLeafDepth, g_e2McabCPuct,
                          g_e2McabFpuReduction, g_e2McabScoreScale,
                          mcab::rootSelectName(g_e2McabRootSelectMode),
                          g_e2McabTreeReuse ? "on" : "off",
                          g_e2McabEquivMode ? ", modo equivalencia" : "");
        }
    } else {
        std::fprintf(stderr, "[arena] Engine 2: ALPHA-BETA PURO (MCTS hibrido desligado)\n");
    }

    // Parametros de busca fora do valor de producao. Silencio = nada foi
    // configurado (as duas engines em producao) -- e o caso normal; imprimir
    // so o que difere evita que o banner vire ruido.
    {
        const std::string t1 = tuning::describeSearchTuning(g_e1Tuning);
        const std::string t2 = tuning::describeSearchTuning(g_e2Tuning);
        if (!t1.empty()) std::fprintf(stderr, "[arena] Engine 1: busca fora do default -> %s\n", t1.c_str());
        if (!t2.empty()) std::fprintf(stderr, "[arena] Engine 2: busca fora do default -> %s\n", t2.c_str());
    }

    if (totalGames % 2 != 0) totalGames++;
    int totalPairs = totalGames / 2;
    std::mt19937_64 rng(seed);

    int eng1Wins = 0, baseWins = 0, draws = 0;
    uint64_t eng1Nodes = 0, baseNodes = 0;
    double eng1TimeSec = 0.0, baseTimeSec = 0.0;
    int totalEng1Wins = 0, totalBaseWins = 0, totalDraws = 0;
    uint64_t totalEng1Nodes = 0, totalBaseNodes = 0;
    double totalEng1Time = 0.0, totalBaseTime = 0.0;
    int lastReportedGames = 0;

    std::vector<TrainingSample> allSamples;
    std::vector<TrainingSample>* samplesPtr = binFilePath.empty() ? nullptr : &allSamples;

    auto reportIfNeeded = [&](int gamesPlayed) {
        if (reportGames > 0 && (gamesPlayed - lastReportedGames >= reportGames)) {
            double eng1Nps = eng1TimeSec > 0 ? eng1Nodes / eng1TimeSec : 0.0;
            double baseNps = baseTimeSec > 0 ? baseNodes / baseTimeSec : 0.0;
            std::printf("PROGRESS_JSON:{\"games\":%d,\"candWins\":%d,\"baseWins\":%d,\"draws\":%d,\"candNodes\":%llu,\"baseNodes\":%llu,\"candTimeSec\":%.4f,\"baseTimeSec\":%.4f,\"candNps\":%.0f,\"baseNps\":%.0f}\n",
                        gamesPlayed - lastReportedGames, eng1Wins, baseWins, draws,
                        (unsigned long long)eng1Nodes, (unsigned long long)baseNodes,
                        eng1TimeSec, baseTimeSec, eng1Nps, baseNps);
            std::fflush(stdout);
            totalEng1Nodes += eng1Nodes; totalBaseNodes += baseNodes;
            totalEng1Time += eng1TimeSec; totalBaseTime += baseTimeSec;
            eng1Wins = 0; baseWins = 0; draws = 0;
            eng1Nodes = 0; baseNodes = 0; eng1TimeSec = 0.0; baseTimeSec = 0.0;
            lastReportedGames = gamesPlayed;
        }
    };

    if (invertColors) {
        for (int p = 0; p < totalPairs; p++) {
            std::mt19937_64 openingRng(rng());
            std::mt19937_64 openingRngCopy = openingRng;

            int resA = playArenaGame(0, timeMs, randomPlies, openingRng, eng1Nodes, baseNodes, eng1TimeSec, baseTimeSec, samplesPtr);
            if (resA == -1) { draws++; totalDraws++; }
            else if (resA == 0) { eng1Wins++; totalEng1Wins++; }
            else { baseWins++; totalBaseWins++; }

            int resB = playArenaGame(1, timeMs, randomPlies, openingRngCopy, eng1Nodes, baseNodes, eng1TimeSec, baseTimeSec, samplesPtr);
            if (resB == -1) { draws++; totalDraws++; }
            else if (resB == 1) { eng1Wins++; totalEng1Wins++; }
            else { baseWins++; totalBaseWins++; }

            reportIfNeeded((p + 1) * 2);
        }
    } else {
        for (int g = 0; g < totalGames; g++) {
            std::mt19937_64 openingRng(rng());
            int eng1Player = g % 2;
            int res = playArenaGame(eng1Player, timeMs, randomPlies, openingRng, eng1Nodes, baseNodes, eng1TimeSec, baseTimeSec, samplesPtr);
            if (res == -1) { draws++; totalDraws++; }
            else if (res == eng1Player) { eng1Wins++; totalEng1Wins++; }
            else { baseWins++; totalBaseWins++; }
            reportIfNeeded(g + 1);
        }
    }

    totalEng1Nodes += eng1Nodes; totalBaseNodes += baseNodes;
    totalEng1Time += eng1TimeSec; totalBaseTime += baseTimeSec;

    if (!binFilePath.empty() && !allSamples.empty()) {
        FILE* fp = std::fopen(binFilePath.c_str(), "wb");
        if (fp) { std::fwrite(allSamples.data(), sizeof(TrainingSample), allSamples.size(), fp); std::fclose(fp); }
    }

    double eng1Nps = totalEng1Time > 0 ? totalEng1Nodes / totalEng1Time : 0.0;
    double baseNps = totalBaseTime > 0 ? totalBaseNodes / totalBaseTime : 0.0;
    std::printf("RESULT_JSON:{\"candWins\":%d,\"baseWins\":%d,\"draws\":%d,\"candNodes\":%llu,\"baseNodes\":%llu,\"candTimeSec\":%.4f,\"baseTimeSec\":%.4f,\"candNps\":%.0f,\"baseNps\":%.0f}\n",
                totalEng1Wins, totalBaseWins, totalDraws,
                (unsigned long long)totalEng1Nodes, (unsigned long long)totalBaseNodes,
                totalEng1Time, totalBaseTime, eng1Nps, baseNps);
    std::fflush(stdout);
    return 0;
}
