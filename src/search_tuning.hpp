#pragma once
// search_tuning.hpp -- superfície ÚNICA de configuração dos parâmetros de
// busca de Negamax (search.hpp) para os binários de laboratório (arena,
// self-play e quem mais precisar).
//
// Motivação: todo setter runtime de search.hpp (setContempt, setLmrDivisor,
// setCatHotCm, ...) existia só para o SPSA e não era alcançável de fora do
// C++ -- para rodar uma arena com contempt diferente era preciso editar e
// recompilar. Este header dá a cada binário um struct de overrides
// "vazio = valor de produção", que a CLI e os blocos de config em Python
// preenchem, e um applySearchTuning() que só toca no que foi preenchido.
//
// Convenção de "vazio" (a mesma ideia de mcab::UNSET_*, mas com sentinelas
// próprias porque -1 é valor LEGAL em vários destes campos -- contempt=-30
// é o default de produção):
//     int/long long  -> UNSET_INT / UNSET_I64  (INT_MIN / INT64_MIN)
//     double         -> UNSET_REAL             (-1e300)
//     bool           -> Tri::Unset
//
// Por que SFINAE em cada setter: o arena compila os headers de DOIS refs Git
// no mesmo binário (qr_e1/qr_e2, ver arena.cpp). Um ref anterior a qualquer
// um destes setters simplesmente não tem o método -- o overload "..."
// (não-template, prioridade mais baixa) vira no-op silencioso, sem erro de
// compilação, exatamente como já se faz com setPolicyOrderingEnabled. Este
// header vive em src/ mas é incluído por CAMINHO RELATIVO no arena, para vir
// sempre do HEAD e nunca do src/ que run_arena.py fez checkout de um ref.
#include <cstdint>
#include <climits>
#include <cstdlib>
#include <string>

namespace tuning {

constexpr int      UNSET_INT  = INT_MIN;
constexpr long long UNSET_I64 = INT64_MIN;
constexpr double   UNSET_REAL = -1e300;
enum class Tri { Unset = -1, Off = 0, On = 1 };   // booleano de três estados

inline bool isSet(int v)       { return v != UNSET_INT; }
inline bool isSet(long long v) { return v != UNSET_I64; }
inline bool isSet(double v)    { return v != UNSET_REAL; }
inline bool isSet(Tri v)       { return v != Tri::Unset; }

// Overrides dos parâmetros de busca. Todo campo vazio => o binário não toca
// no setter correspondente e vale o default de search.hpp (anotado ao lado).
struct SearchTuning {
    Tri  quiescence       = Tri::Unset;  // produção: ligado  (setQuiescenceEnabled)
    Tri  lmrPvs           = Tri::Unset;  // produção: ligado  (setLmrPvsEnabled)
    int  contempt         = UNSET_INT;   // produção: -30     (CONTEMPT)
    long long policyOrderScale = UNSET_I64;  // produção: 400 (POLICY_ORDER_SCALE)
    long long catScoreScale    = UNSET_I64;  // produção: 2
    int  lmrMinDepth      = UNSET_INT;   // produção: 3       (LMR_MIN_DEPTH)
    int  lmrMinMoveIndex  = UNSET_INT;   // produção: 3       (LMR_MIN_MOVE_INDEX)
    double lmrDivisor     = UNSET_REAL;  // produção: 2.25    (LMR_DIVISOR)
    int  catHotCm         = UNSET_INT;   // produção: 150     (CAT_HOT_CM)
    int  catColdCm        = UNSET_INT;   // produção: 30      (CAT_COLD_CM)
    int  wallBfsOrderMaxPly  = UNSET_INT;  // produção: 2     (WALL_BFS_ORDER_MAX_PLY)
    int  qsCriticalBfsDelta  = UNSET_INT;  // produção: 2     (QS_CRITICAL_BFS_DELTA)
    int  qsMaxExtraPlies     = UNSET_INT;  // produção: 2     (QS_MAX_EXTRA_PLIES)
    int  qsLowWallsBonus     = UNSET_INT;  // produção: 0     (desligado)
    int  qsLowWallsThreshold = UNSET_INT;  // produção: 0     (nunca dispara)
    // inv/ab-policy (2026-08-23) -- todos default DESLIGADOS/ vazios; o
    // engine de produção não muda enquanto ninguém preencher estes campos.
    Tri  policyHistory       = Tri::Unset;  // produção: off   (setPolicyHistorySeedEnabled)
    long long policyHistoryScale = UNSET_I64;  // produção: 400  (setPolicyHistorySeedScale)
    Tri  policyLmr           = Tri::Unset;  // produção: off   (setPolicyLmrEnabled)
    double policyLmrHot      = UNSET_REAL;  // produção: 2.5   (setPolicyLmrHotDelta)
    double policyLmrCold     = UNSET_REAL;  // produção: 5.0   (setPolicyLmrColdDelta)
    Tri  policyLmp           = Tri::Unset;  // produção: off   (setPolicyLmpEnabled)
    double policyLmpBase     = UNSET_REAL;  // produção: 0.05  (setPolicyLmpBaseMass)
    int  policyLmpMinCount   = UNSET_INT;   // produção: 8     (setPolicyLmpMinCount)
};

// Gera um par de helpers "trySet<NOME>(engine, valor, 0)": o template casa
// quando o método existe no ref daquela engine; o overload variádico é o
// fallback no-op para refs antigos que não o têm.
#define QR_TUNING_TRY_SETTER(HELPER, METHOD, TYPE)                                 \
    template <typename Eng>                                                        \
    auto HELPER(Eng& e, TYPE v, int) -> decltype(e.METHOD(v), void()) { e.METHOD(v); } \
    template <typename Eng>                                                        \
    void HELPER(Eng&, TYPE, ...) {}

QR_TUNING_TRY_SETTER(trySetQuiescence,        setQuiescenceEnabled,  bool)
QR_TUNING_TRY_SETTER(trySetLmrPvs,            setLmrPvsEnabled,      bool)
QR_TUNING_TRY_SETTER(trySetContempt,          setContempt,           int)
QR_TUNING_TRY_SETTER(trySetPolicyOrderScale,  setPolicyOrderScale,   long long)
QR_TUNING_TRY_SETTER(trySetCatScoreScale,     setCatScoreScale,      long long)
QR_TUNING_TRY_SETTER(trySetLmrMinDepth,       setLmrMinDepth,        int)
QR_TUNING_TRY_SETTER(trySetLmrMinMoveIndex,   setLmrMinMoveIndex,    int)
QR_TUNING_TRY_SETTER(trySetLmrDivisor,        setLmrDivisor,         double)
QR_TUNING_TRY_SETTER(trySetCatHotCm,          setCatHotCm,           int)
QR_TUNING_TRY_SETTER(trySetCatColdCm,         setCatColdCm,          int)
QR_TUNING_TRY_SETTER(trySetWallBfsOrderMaxPly, setWallBfsOrderMaxPly, int)
QR_TUNING_TRY_SETTER(trySetQsCriticalBfsDelta, setQsCriticalBfsDelta, int)
QR_TUNING_TRY_SETTER(trySetQsMaxExtraPlies,    setQsMaxExtraPlies,    int)
QR_TUNING_TRY_SETTER(trySetQsLowWallsBonus,    setQsLowWallsBonus,    int)
QR_TUNING_TRY_SETTER(trySetQsLowWallsThreshold, setQsLowWallsThreshold, int)
QR_TUNING_TRY_SETTER(trySetPolicyHistorySeedEnabled, setPolicyHistorySeedEnabled, bool)
QR_TUNING_TRY_SETTER(trySetPolicyHistorySeedScale,  setPolicyHistorySeedScale,  long long)
QR_TUNING_TRY_SETTER(trySetPolicyLmrEnabled,        setPolicyLmrEnabled,        bool)
QR_TUNING_TRY_SETTER(trySetPolicyLmrHotDelta,       setPolicyLmrHotDelta,       float)
QR_TUNING_TRY_SETTER(trySetPolicyLmrColdDelta,      setPolicyLmrColdDelta,      float)
QR_TUNING_TRY_SETTER(trySetPolicyLmpEnabled,        setPolicyLmpEnabled,        bool)
QR_TUNING_TRY_SETTER(trySetPolicyLmpBaseMass,       setPolicyLmpBaseMass,       double)
QR_TUNING_TRY_SETTER(trySetPolicyLmpMinCount,       setPolicyLmpMinCount,       int)

#undef QR_TUNING_TRY_SETTER

// Aplica SÓ os campos preenchidos. Chamar com um SearchTuning{} vazio é um
// no-op completo -- é o que garante que ligar este mecanismo não muda o
// comportamento de nenhuma execução que não configurou nada.
template <typename Eng>
inline void applySearchTuning(Eng& e, const SearchTuning& t) {
    if (isSet(t.quiescence))       trySetQuiescence(e, t.quiescence == Tri::On, 0);
    if (isSet(t.lmrPvs))           trySetLmrPvs(e, t.lmrPvs == Tri::On, 0);
    if (isSet(t.contempt))         trySetContempt(e, t.contempt, 0);
    if (isSet(t.policyOrderScale)) trySetPolicyOrderScale(e, t.policyOrderScale, 0);
    if (isSet(t.catScoreScale))    trySetCatScoreScale(e, t.catScoreScale, 0);
    if (isSet(t.lmrMinDepth))      trySetLmrMinDepth(e, t.lmrMinDepth, 0);
    if (isSet(t.lmrMinMoveIndex))  trySetLmrMinMoveIndex(e, t.lmrMinMoveIndex, 0);
    if (isSet(t.lmrDivisor))       trySetLmrDivisor(e, t.lmrDivisor, 0);
    if (isSet(t.catHotCm))         trySetCatHotCm(e, t.catHotCm, 0);
    if (isSet(t.catColdCm))        trySetCatColdCm(e, t.catColdCm, 0);
    if (isSet(t.wallBfsOrderMaxPly)) trySetWallBfsOrderMaxPly(e, t.wallBfsOrderMaxPly, 0);
    if (isSet(t.qsCriticalBfsDelta)) trySetQsCriticalBfsDelta(e, t.qsCriticalBfsDelta, 0);
    if (isSet(t.qsMaxExtraPlies))    trySetQsMaxExtraPlies(e, t.qsMaxExtraPlies, 0);
    if (isSet(t.qsLowWallsBonus))    trySetQsLowWallsBonus(e, t.qsLowWallsBonus, 0);
    if (isSet(t.qsLowWallsThreshold)) trySetQsLowWallsThreshold(e, t.qsLowWallsThreshold, 0);
    // inv/ab-policy knobs: hot/cold/base/min-count are applied only when
    // their feature flag was set, so a bare --policy-lmr keeps the default
    // deltas and never silently turns LMP on.
    if (isSet(t.policyHistory))      trySetPolicyHistorySeedEnabled(e, t.policyHistory == Tri::On, 0);
    if (isSet(t.policyHistoryScale)) trySetPolicyHistorySeedScale(e, t.policyHistoryScale, 0);
    if (isSet(t.policyLmr))          trySetPolicyLmrEnabled(e, t.policyLmr == Tri::On, 0);
    if (t.policyLmr == Tri::On) {
        if (isSet(t.policyLmrHot))  trySetPolicyLmrHotDelta(e, (float)t.policyLmrHot, 0);
        if (isSet(t.policyLmrCold)) trySetPolicyLmrColdDelta(e, (float)t.policyLmrCold, 0);
    }
    if (isSet(t.policyLmp))          trySetPolicyLmpEnabled(e, t.policyLmp == Tri::On, 0);
    if (t.policyLmp == Tri::On) {
        if (isSet(t.policyLmpBase))     trySetPolicyLmpBaseMass(e, t.policyLmpBase, 0);
        if (isSet(t.policyLmpMinCount)) trySetPolicyLmpMinCount(e, t.policyLmpMinCount, 0);
    }
}

// Parsing compartilhado pelas CLIs (arena.cpp / selfplay_main.cpp): consome
// `--<prefixo><knob> [valor]` e devolve true se `arg` era uma flag deste
// conjunto (nesse caso `i` já avançou sobre o valor). `prefixo` é "" (global)
// ou "e1-"/"e2-" no arena, para o mesmo knob existir nas duas formas.
//
// Uma flag desconhecida devolve false e o chamador segue seu próprio parsing
// -- este helper nunca consome o que não é dele.
inline bool parseSearchTuningArg(const char* arg, int argc, char** argv, int& i,
                                 const char* prefixo, SearchTuning& t) {
    const std::string base = std::string("--") + prefixo;
    auto flag = [&](const char* nome) { return base + nome == arg; };
    // Só chamado pelos knobs com valor; se o valor faltar, mantém o campo
    // vazio em vez de ler lixo (o binário fica no valor de produção).
    auto temValor = [&]() { return i + 1 < argc; };

    if (flag("quiescence"))    { t.quiescence = Tri::On;  return true; }
    if (flag("no-quiescence")) { t.quiescence = Tri::Off; return true; }
    if (flag("lmr-pvs"))       { t.lmrPvs = Tri::On;  return true; }
    if (flag("no-lmr-pvs"))    { t.lmrPvs = Tri::Off; return true; }
    if (flag("contempt"))            { if (temValor()) t.contempt = std::atoi(argv[++i]); return true; }
    if (flag("policy-order-scale"))  { if (temValor()) t.policyOrderScale = std::atoll(argv[++i]); return true; }
    if (flag("cat-score-scale"))     { if (temValor()) t.catScoreScale = std::atoll(argv[++i]); return true; }
    if (flag("lmr-min-depth"))       { if (temValor()) t.lmrMinDepth = std::atoi(argv[++i]); return true; }
    if (flag("lmr-min-move-index"))  { if (temValor()) t.lmrMinMoveIndex = std::atoi(argv[++i]); return true; }
    if (flag("lmr-divisor"))         { if (temValor()) t.lmrDivisor = std::atof(argv[++i]); return true; }
    if (flag("cat-hot-cm"))          { if (temValor()) t.catHotCm = std::atoi(argv[++i]); return true; }
    if (flag("cat-cold-cm"))         { if (temValor()) t.catColdCm = std::atoi(argv[++i]); return true; }
    if (flag("wall-bfs-order-max-ply")) { if (temValor()) t.wallBfsOrderMaxPly = std::atoi(argv[++i]); return true; }
    if (flag("qs-critical-bfs-delta"))  { if (temValor()) t.qsCriticalBfsDelta = std::atoi(argv[++i]); return true; }
    if (flag("qs-max-extra-plies"))     { if (temValor()) t.qsMaxExtraPlies = std::atoi(argv[++i]); return true; }
    if (flag("qs-low-walls-bonus"))     { if (temValor()) t.qsLowWallsBonus = std::atoi(argv[++i]); return true; }
    if (flag("qs-low-walls-threshold")) { if (temValor()) t.qsLowWallsThreshold = std::atoi(argv[++i]); return true; }
    // inv/ab-policy knobs
    if (flag("policy-history"))        { t.policyHistory = Tri::On;  return true; }
    if (flag("no-policy-history"))     { t.policyHistory = Tri::Off; return true; }
    if (flag("policy-history-scale"))  { if (temValor()) t.policyHistoryScale = std::atoll(argv[++i]); return true; }
    if (flag("policy-lmr"))            { t.policyLmr = Tri::On;  return true; }
    if (flag("no-policy-lmr"))         { t.policyLmr = Tri::Off; return true; }
    if (flag("policy-lmr-hot"))        { if (temValor()) t.policyLmrHot = std::atof(argv[++i]); return true; }
    if (flag("policy-lmr-cold"))       { if (temValor()) t.policyLmrCold = std::atof(argv[++i]); return true; }
    if (flag("policy-lmp"))            { t.policyLmp = Tri::On;  return true; }
    if (flag("no-policy-lmp"))         { t.policyLmp = Tri::Off; return true; }
    if (flag("policy-lmp-base"))       { if (temValor()) t.policyLmpBase = std::atof(argv[++i]); return true; }
    if (flag("policy-lmp-min-count"))  { if (temValor()) t.policyLmpMinCount = std::atoi(argv[++i]); return true; }
    return false;
}

// Resumo dos knobs PREENCHIDOS, para o banner dos binários. Devolve "" quando
// nada foi configurado -- é o sinal de "tudo em produção", e o chamador
// simplesmente não imprime linha nenhuma nesse caso.
inline std::string describeSearchTuning(const SearchTuning& t) {
    std::string s;
    auto add = [&](const std::string& kv) { if (!s.empty()) s += ", "; s += kv; };
    if (isSet(t.quiescence))       add(std::string("quiescence=") + (t.quiescence == Tri::On ? "on" : "off"));
    if (isSet(t.lmrPvs))           add(std::string("lmr-pvs=") + (t.lmrPvs == Tri::On ? "on" : "off"));
    if (isSet(t.contempt))         add("contempt=" + std::to_string(t.contempt));
    if (isSet(t.policyOrderScale)) add("policy-order-scale=" + std::to_string(t.policyOrderScale));
    if (isSet(t.catScoreScale))    add("cat-score-scale=" + std::to_string(t.catScoreScale));
    if (isSet(t.lmrMinDepth))      add("lmr-min-depth=" + std::to_string(t.lmrMinDepth));
    if (isSet(t.lmrMinMoveIndex))  add("lmr-min-move-index=" + std::to_string(t.lmrMinMoveIndex));
    if (isSet(t.lmrDivisor))       add("lmr-divisor=" + std::to_string(t.lmrDivisor));
    if (isSet(t.catHotCm))         add("cat-hot-cm=" + std::to_string(t.catHotCm));
    if (isSet(t.catColdCm))        add("cat-cold-cm=" + std::to_string(t.catColdCm));
    if (isSet(t.wallBfsOrderMaxPly)) add("wall-bfs-order-max-ply=" + std::to_string(t.wallBfsOrderMaxPly));
    if (isSet(t.qsCriticalBfsDelta)) add("qs-critical-bfs-delta=" + std::to_string(t.qsCriticalBfsDelta));
    if (isSet(t.qsMaxExtraPlies))    add("qs-max-extra-plies=" + std::to_string(t.qsMaxExtraPlies));
    if (isSet(t.qsLowWallsBonus))    add("qs-low-walls-bonus=" + std::to_string(t.qsLowWallsBonus));
    if (isSet(t.qsLowWallsThreshold)) add("qs-low-walls-threshold=" + std::to_string(t.qsLowWallsThreshold));
    if (isSet(t.policyHistory))       add(std::string("policy-history=") + (t.policyHistory == Tri::On ? "on" : "off"));
    if (isSet(t.policyHistoryScale))  add("policy-history-scale=" + std::to_string(t.policyHistoryScale));
    if (isSet(t.policyLmr))           add(std::string("policy-lmr=") + (t.policyLmr == Tri::On ? "on" : "off"));
    if (isSet(t.policyLmrHot))        add("policy-lmr-hot=" + std::to_string(t.policyLmrHot));
    if (isSet(t.policyLmrCold))       add("policy-lmr-cold=" + std::to_string(t.policyLmrCold));
    if (isSet(t.policyLmp))           add(std::string("policy-lmp=") + (t.policyLmp == Tri::On ? "on" : "off"));
    if (isSet(t.policyLmpBase))       add("policy-lmp-base=" + std::to_string(t.policyLmpBase));
    if (isSet(t.policyLmpMinCount))   add("policy-lmp-min-count=" + std::to_string(t.policyLmpMinCount));
    return s;
}

// Texto de ajuda dos knobs acima, compartilhado pelos --help dos binários
// (o prefixo entra formatado pelo chamador).
inline const char* searchTuningUsage() {
    return
        "  --contempt N              score de empate em cp (producao: -30)\n"
        "  --policy-order-scale N    escala do logit da politica na ordenacao (producao: 400)\n"
        "  --cat-score-scale N       peso do calor CAT vs. politica em orderWallMoves (producao: 2)\n"
        "  --lmr-min-depth N         profundidade minima para LMR reduzir (producao: 3)\n"
        "  --lmr-min-move-index N    indice 1-based do 1o lance reduzido por LMR (producao: 3)\n"
        "  --lmr-divisor F           divisor da formula de reducao do LMR (producao: 2.25)\n"
        "  --cat-hot-cm N            calor CAT que marca o lance como tatico, pula LMR (producao: 150)\n"
        "  --cat-cold-cm N           calor CAT abaixo do qual LMR reduz +1 (producao: 30)\n"
        "  --wall-bfs-order-max-ply N  ply maximo com ordenacao de muro por BFS (producao: 2)\n"
        "  --qs-critical-bfs-delta N   delta de BFS que torna o muro critico na quiescencia (producao: 2)\n"
        "  --qs-max-extra-plies N    extensao maxima de quiescencia de muro (producao: 2)\n"
        "  --qs-low-walls-bonus N    plies extras de quiescencia com poucos muros no total (producao: 0)\n"
        "  --qs-low-walls-threshold N  total de muros no tabuleiro que ativa o bonus (producao: 0)\n"
        "  --quiescence/--no-quiescence  quiescencia de muro (producao: ligada)\n"
        "  --lmr-pvs/--no-lmr-pvs        LMR+PVS (producao: ligado)\n"
        "  --policy-history/--no-policy-history  semeia history com a politica da raiz (producao: off)\n"
        "  --policy-history-scale N      escala do logit semeado na history (producao: 400)\n"
        "  --policy-lmr/--no-policy-lmr  LMR modulado pelo logit da politica (producao: off)\n"
        "  --policy-lmr-hot F            janela do maximo do estagio que impede reducao (default 2.5)\n"
        "  --policy-lmr-cold F           distancia do maximo que adiciona +1 de reducao (default 5.0)\n"
        "  --policy-lmp/--no-policy-lmp  poda tardia por massa softmax da politica (producao: off)\n"
        "  --policy-lmp-base F           massa sufixo tolerada no horizonte (default 0.05)\n"
        "  --policy-lmp-min-count N      candidatos iniciais nunca podados (default 8)\n";
}

}  // namespace tuning
